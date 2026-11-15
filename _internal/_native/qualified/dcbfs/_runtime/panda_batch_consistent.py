"""Native collision batches and vectorized unchanged fixed-grid certification."""
from contextlib import contextmanager
from unittest.mock import patch
from time import perf_counter
import threading
import os
import numpy as np
import numba

_kernel=None


def clearance_kernel():
    global _kernel
    if _kernel is not None:return _kernel
    from .collision_scratch import batch
    _kernel=batch;return batch


def measurements(source,X,audit):
    began=perf_counter()
    controls,T=source.controls(X);n=len(X)
    B,D,gram,derivatives=source._basis
    positions=B@controls;velocity=D@controls
    values=clearance_kernel()(np.ascontiguousarray(positions),*source._collision)
    collided=perf_counter();audit['collision_and_positions_seconds']+=collided-began
    grid=np.linspace(np.float32(0),np.float32(1),source.samples,dtype=np.float32)
    width=np.diff(grid)
    acceleration=np.max(np.abs(derivatives[1]@controls),axis=1)
    bound=np.maximum(np.abs(velocity[:,:-1]),np.abs(velocity[:,1:]))+width[None,:,None]*acceleration[:,None,:]/np.float32(2)
    lower=np.minimum(values[:,:-1],values[:,1:])-(bound@source._motion_radii.T)*width[None,:,None]/np.float32(2)
    unresolved=np.any(lower<source.clearance,axis=(1,2))
    early_collision=values.min(axis=(1,2))<source.clearance
    slow=unresolved&~early_collision&(source.samples<source.max_collision_samples)
    M=np.empty((n,len(source.metric_names)),np.float32)
    M[:,0]=T
    # Preserve NumPy's original per-control reduction order for jerk. This
    # small postprocessing loop is separate from native collision evaluation.
    duration_cubes=np.empty(n,np.float32)
    for row in range(n):
        duration_cubes[row]=np.float32(T[row]**3)
        control=controls[row]
        energy=max(np.float32(0),np.float32(np.sum(control*(gram@control))))
        M[row,1]=energy/np.float32(T[row]**5)
    speed=np.linalg.norm(velocity,axis=2)
    M[:,2]=np.trapezoid(speed,dx=np.float32(1)/np.float32(source.samples-1),axis=1)
    density=np.float32(1)/(np.float32(1)+np.maximum(values[:,:,0],np.float32(0))/np.float32(.03))**2
    M[:,3]=np.trapezoid(density,grid,axis=1)
    M[:,4:6]=(source.clearance-lower.min(axis=1))/np.float32(.03)
    for order,(matrix,limit)in enumerate(zip(derivatives,[source.velocity,source.acceleration,source.jerk_limit]),1):
        powers=duration_cubes if order==3 else T**order
        M[:,6+7*(order-1):6+7*order]=np.max(np.abs(matrix@controls),axis=1)/(powers[:,None]*limit)-np.float32(1)
    M[:,-1]=source.samples
    # Preserve the existing adaptive refinement exactly for those rows.
    for row in np.flatnonzero(slow):M[row]=source._measurement((controls[row],T[row]))
    audit['batches']+=1;audit['rows']+=n;audit['adaptive_original_rows']+=int(slow.sum())
    audit['postprocess_seconds']+=perf_counter()-collided
    return M


@contextmanager
def context(workers, *, batch_stage=None):
    from dcbfs.problems.panda import PandaTrajectory
    from dcbfs.algorithms.runtime import ProblemView,Evaluation
    from threadpoolctl import threadpool_limits
    original_view=ProblemView.evaluate;previous=numba.get_num_threads()
    original_evaluate=PandaTrajectory._evaluate
    workers=min(int(workers),numba.config.NUMBA_NUM_THREADS)
    audit=dict(batches=0,rows=0,adaptive_original_rows=0,workers=workers,maximum_workers=workers,granted_widths={},collision_and_positions_seconds=0.,postprocess_seconds=0.)
    def evaluate(source,X,out,*args,**kwargs):
        if threading.current_thread()is not threading.main_thread():return original_evaluate(source,X,out,*args,**kwargs)
        def compute():
            active=workers if batch_stage is None else min(int(os.environ['REMOTE_COMPUTE_CPUS']),numba.config.NUMBA_NUM_THREADS)
            before=numba.get_num_threads();numba.set_num_threads(active)
            audit['maximum_workers']=max(audit['maximum_workers'],active)
            key=str(active);audit['granted_widths'][key]=audit['granted_widths'].get(key,0)+1
            try:return measurements(source,X,audit)
            finally:numba.set_num_threads(before)
        M=compute() if batch_stage is None else batch_stage(compute,configurations=len(X)*source.samples)
        out.update(M=M,F=source.objectives_from_metrics(M),G=source.constraints_from_metrics(M))
    def view_evaluate(view,X,return_as_dictionary=False):
        if not isinstance(view.source,PandaTrajectory):return original_view(view,X,return_as_dictionary=return_as_dictionary)
        raw=view.source.evaluate(X,return_as_dictionary=True)
        return raw if return_as_dictionary else Evaluation(raw['F'],raw['G'],{'M':raw['M']})
    numba.set_num_threads(1)
    try:
        with threadpool_limits(limits=1),patch.object(PandaTrajectory,'_evaluate',evaluate),patch.object(ProblemView,'evaluate',view_evaluate):yield audit
    finally:numba.set_num_threads(previous)
