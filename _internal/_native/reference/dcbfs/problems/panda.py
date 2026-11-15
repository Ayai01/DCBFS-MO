"""Panda trajectory design on frozen VAMP/MotionBenchMaker scenes.

The public model uses 59 conservative collision spheres. Its Apache-2.0
license, upstream revision and source hashes accompany data/panda.json.
Objectives describe kinematic trajectories, not motor energy or torque.
"""
from functools import lru_cache
import json

import numpy as np
from numba import njit
from pymoo.core.problem import Problem
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation

from .. import PROBLEM_DATA
from ..utils.design import validate_X
from ..utils.parallel import parallel_map


@lru_cache(maxsize=1)
def robot_model():
    return json.loads((PROBLEM_DATA/'panda.json').read_text(encoding='utf8'))


def _robot_arrays(model):
    joints=model['joints']
    origin=np.array([Rotation.from_euler('xyz',j['rpy']).as_matrix() for j in joints])
    return (np.array([j['parent'] for j in joints]),np.array([j['child'] for j in joints]),
        np.array([j['coordinate'] for j in joints]),origin,np.array([j['xyz'] for j in joints]),
        np.array([s['link'] for s in model['spheres']]),np.array([s['center'] for s in model['spheres']]))


@lru_cache(maxsize=1)
def _motion_radii():
    model=robot_model();parents={j['child']:j for j in model['joints']}
    radii=np.zeros((len(model['spheres']),7));active=np.zeros_like(radii,dtype=bool)
    for s,sphere in enumerate(model['spheres']):
        length=float(np.linalg.norm(sphere['center']));link=sphere['link']
        while link in parents:
            joint=parents[link];axis=joint['coordinate']
            if axis>=0:radii[s,axis]=length;active[s,axis]=True
            length+=np.linalg.norm(joint['xyz']);link=joint['parent']
    relative=np.zeros(7)
    for a,b in model['self_pairs']:
        # A common ancestral rotation preserves a pair's mutual distance.
        relative=np.maximum(relative,np.where(active[a]^active[b],radii[a]+radii[b],0.))
    return np.array([radii.max(axis=0),relative])


@njit(cache=(not bool(__import__('os').environ.get('DCBFS_NO_FILES'))),nogil=True)
def _sphere_centers(q,parent,child,coordinate,origin,translation,links,local):
    rotation=np.zeros((13,3,3));position=np.zeros((13,3))
    for i in range(3):rotation[0,i,i]=1.
    for j in range(len(parent)):
        p,c=parent[j],child[j]
        for a in range(3):
            position[c,a]=position[p,a]
            for b in range(3):position[c,a]+=rotation[p,a,b]*translation[j,b]
        relative=origin[j].copy()
        if coordinate[j]>=0:
            cosine,sine=np.cos(q[coordinate[j]]),np.sin(q[coordinate[j]])
            for a in range(3):
                x,y=origin[j,a,0],origin[j,a,1]
                relative[a,0]=cosine*x+sine*y
                relative[a,1]=-sine*x+cosine*y
        for a in range(3):
            for b in range(3):
                for k in range(3):rotation[c,a,b]+=rotation[p,a,k]*relative[k,b]
    result=np.empty((len(links),3))
    for s in range(len(links)):
        link=links[s]
        for a in range(3):
            result[s,a]=position[link,a]
            for b in range(3):result[s,a]+=rotation[link,a,b]*local[s,b]
    return result


@njit(cache=(not bool(__import__('os').environ.get('DCBFS_NO_FILES'))),nogil=True)
def _configuration_clearance(q,parent,child,coordinate,origin,translation,links,local,radii,pairs,
                             types,centers,rotations,dimensions):
    spheres=_sphere_centers(q,parent,child,coordinate,origin,translation,links,local)
    world,self_distance=1e6,1e6
    for s in range(len(spheres)):
        for obj in range(len(types)):
            x,y,z=spheres[s,0]-centers[obj,0],spheres[s,1]-centers[obj,1],spheres[s,2]-centers[obj,2]
            px=rotations[obj,0,0]*x+rotations[obj,1,0]*y+rotations[obj,2,0]*z
            py=rotations[obj,0,1]*x+rotations[obj,1,1]*y+rotations[obj,2,1]*z
            pz=rotations[obj,0,2]*x+rotations[obj,1,2]*y+rotations[obj,2,2]*z
            if types[obj]==0:
                dx,dy,dz=abs(px)-dimensions[obj,0],abs(py)-dimensions[obj,1],abs(pz)-dimensions[obj,2]
                distance=(max(dx,0.)**2+max(dy,0.)**2+max(dz,0.)**2)**.5+min(max(dx,dy,dz),0.)
            elif types[obj]==1:
                radial=(px**2+py**2)**.5-dimensions[obj,1]
                axial=abs(pz)-dimensions[obj,0]
                distance=(max(radial,0.)**2+max(axial,0.)**2)**.5+min(max(radial,axial),0.)
            else:
                distance=(px**2+py**2+pz**2)**.5-dimensions[obj,0]
            world=min(world,distance-radii[s])
    for i in range(len(pairs)):
        a,b=pairs[i,0],pairs[i,1]
        dx,dy,dz=spheres[a,0]-spheres[b,0],spheres[a,1]-spheres[b,1],spheres[a,2]-spheres[b,2]
        self_distance=min(self_distance,(dx**2+dy**2+dz**2)**.5-radii[a]-radii[b])
    return world,self_distance


@njit(cache=(not bool(__import__('os').environ.get('DCBFS_NO_FILES'))),nogil=True)
def _clearance_samples(q,parent,child,coordinate,origin,translation,links,local,radii,pairs,
                    types,centers,rotations,dimensions):
    result=np.empty((len(q),2))
    for i in range(len(q)):
        result[i]=_configuration_clearance(q[i],parent,child,coordinate,origin,translation,links,local,
            radii,pairs,types,centers,rotations,dimensions)
    return result


@lru_cache(maxsize=12)
def _spline_basis(internal_points,samples):
    count=internal_points+6
    knots=np.r_[np.zeros(6),np.arange(1,count-5)/(count-5),np.ones(6)]
    spline=BSpline(knots,np.eye(count),5)
    grid=np.linspace(0.,1.,samples)
    # Squared third derivatives have degree four on each knot span. Three
    # Gauss points per span integrate the jerk Gram matrix exactly.
    nodes,weights=np.polynomial.legendre.leggauss(3)
    unique=np.unique(knots);left,right=unique[:-1],unique[1:]
    quadrature=((left[:,None]+right[:,None])/2+(right-left)[:,None]/2*nodes).ravel()
    weighted=((right-left)[:,None]/2*weights).ravel()
    third=spline.derivative(3)(quadrature)
    gram=third.T@(weighted[:,None]*third)
    derivatives=tuple(spline.derivative(order).c[:count-order] for order in (1,2,3))
    return spline(grid),spline.derivative(1)(grid),gram,derivatives


class PandaTrajectory(Problem):
    """Ten free seven-joint controls plus duration by default: 71D / four goals."""

    def __init__(self, *, query='cage-0001', internal_points=10, samples=129,
                 duration_bounds=(.5,15.), clearance=.002, max_collision_samples=4097):
        if isinstance(internal_points,bool) or int(internal_points)!=internal_points or internal_points<4:
            raise ValueError('internal_points must be an integer >=4')
        if isinstance(samples,bool) or int(samples)!=samples or samples<33:
            raise ValueError('samples must be an integer >=33')
        duration_bounds=np.asarray(duration_bounds,float)
        if duration_bounds.shape!=(2,) or not np.isfinite(duration_bounds).all() or not 0.<duration_bounds[0]<duration_bounds[1]:
            raise ValueError('duration_bounds must be two increasing positive seconds')
        if not np.isfinite(clearance) or clearance<0.:
            raise ValueError('clearance must be nonnegative metres')
        if isinstance(max_collision_samples,bool) or int(max_collision_samples)!=max_collision_samples or max_collision_samples<samples:
            raise ValueError('max_collision_samples must be an integer >=samples')
        model=robot_model()
        selected=next((item for item in model['queries'] if item['id']==query),None)
        if selected is None:raise ValueError('Unknown frozen Panda query: '+str(query))
        self.name='panda';self.query=query
        self.internal_points,self.samples=int(internal_points),int(samples)
        self.max_collision_samples=int(max_collision_samples)
        self.duration_bounds=duration_bounds;self.clearance=float(clearance)
        self.joint_lower,self.joint_upper=np.array(model['lower']),np.array(model['upper'])
        self.start,self.goal=np.array(selected['start']),np.array(selected['goal'])
        self.path_length_scale=2.*float(np.linalg.norm(self.goal-self.start))
        if np.any(np.r_[self.start-self.joint_lower,self.goal-self.joint_lower,self.joint_upper-self.start,self.joint_upper-self.goal]<-1e-9):
            raise ValueError('Published endpoints exceed the physical Panda joint limits')
        self._robot=_robot_arrays(model)
        radii=np.array([s['radius'] for s in model['spheres']]);pairs=np.array(model['self_pairs'],dtype=np.int64)
        obstacles=selected['boxes'];types=np.array([{'box':0,'cylinder':1,'sphere':2}[o['type']] for o in obstacles])
        centers=np.array([o['center'] for o in obstacles]);rotations=Rotation.from_quat([o['quaternion_xyzw'] for o in obstacles]).as_matrix()
        dimensions=np.zeros((len(obstacles),3))
        for i,o in enumerate(obstacles):
            d=np.array(o['dimensions'],float);dimensions[i,:len(d)]=d
            if types[i]==0:dimensions[i]*=.5
            elif types[i]==1:dimensions[i,0]*=.5
        self._collision=(*self._robot,radii,pairs,types,centers,rotations,dimensions)
        self._basis=_spline_basis(self.internal_points,self.samples)
        count=self.internal_points+6
        self._spline=BSpline(np.r_[np.zeros(6),np.arange(1,count-5)/(count-5),np.ones(6)],np.eye(count),5)
        self._motion_radii=_motion_radii()
        self.velocity=np.array(model['velocity']);self.acceleration=np.array(model['acceleration']);self.jerk_limit=np.array(model['jerk'])
        self.objective_names=['duration','integrated_squared_jerk','joint_path_length','clearance_exposure']
        self.variable_names=[f'control_{i+1}_joint_{j+1}' for i in range(self.internal_points) for j in range(7)]+['duration']
        self.constraint_names=['world_clearance','self_clearance']+[f'joint_{j+1}_{kind}' for kind in ['velocity','acceleration','jerk'] for j in range(7)]
        super().__init__(n_var=7*self.internal_points+1,n_obj=4,n_ieq_constr=len(self.constraint_names),xl=0.,xu=1.)
        self.lower,self.upper=np.zeros(self.n_var),np.ones(self.n_var)
        self.physical_lower=np.r_[np.tile(self.joint_lower,self.internal_points),duration_bounds[0]]
        self.physical_upper=np.r_[np.tile(self.joint_upper,self.internal_points),duration_bounds[1]]
        self.metric_names=self.objective_names+self.constraint_names+['collision_configurations']
        self.metric_metadata=[dict(name=name,label=name,direction='minimize',unit=unit) for name,unit in zip(self.metric_names,
            ['s','rad^2/s^5','rad','dimensionless']+['normalized violation']*self.n_ieq_constr+['configurations'])]
        self.normalization={name:dict(metric=name,direction='minimize',worst=scale,target=0.,method='log1p' if i==1 else 'linear',scale=1.)
            for i,(name,scale) in enumerate(zip(self.objective_names,[15.,1e5,self.path_length_scale,1.]))}
        self.objective_metadata=[dict(self.metric_metadata[i],**self.normalization[name],spec='仅排序目标')
            for i,name in enumerate(self.objective_names)]
        self.normalization_info=dict(display_space='utility',coordinate_space='utility',internal_objective_formula='F=-U',
            utility_formula='U=1-raw/scale; jerk uses log1p(raw)/log1p(100000)',utility_direction='higher is better',
            utility_clipping=False,anchors=self.normalization,constraint_formula='published kinematic limits and world/self clearance; no objective target gates')
        self.hv_reference_point=0.
        self.constraints_depend_on_objectives=False
        self.reference_metric_indices=list(range(4));self.diagnostic_constraint_count=self.n_ieq_constr
        self.profile=dict(robot='Panda',query=query,internal_points=self.internal_points,samples=self.samples,
            duration_bounds=duration_bounds.tolist(),clearance=self.clearance,max_collision_samples=self.max_collision_samples,objectives=self.objective_names,
            upstream=model['provenance'],kinematic_limits='FER/Panda official limits; fixed open fingers')
        self.model_files=[PROBLEM_DATA/'panda.json']
        t=np.linspace(0.,1.,self.internal_points+2)[1:-1]
        initial=np.array(selected.get('seed_controls',[]))
        if initial.shape!=(self.internal_points,7):
            initial=(1.-t[:,None])*self.start+t[:,None]*self.goal
        self.nominal=self.encode(np.r_[initial.ravel(),selected.get('seed_duration',8.)])[0]

    def decode(self,X):
        return self.physical_lower+validate_X(X,self.n_var,self.lower,self.upper)*(self.physical_upper-self.physical_lower)

    def encode(self,physical):
        physical=validate_X(physical,self.n_var,self.physical_lower,self.physical_upper)
        return (physical-self.physical_lower)/(self.physical_upper-self.physical_lower)

    def controls(self,X):
        physical=self.decode(X)
        return np.concatenate((np.broadcast_to(self.start,(len(physical),3,7)),physical[:,:-1].reshape(-1,self.internal_points,7),
            np.broadcast_to(self.goal,(len(physical),3,7))),axis=1),physical[:,-1]

    def clearance_at(self,joints):
        return _clearance_samples(np.atleast_2d(joints),*self._collision)

    def trajectory(self,X,samples=1025):
        controls,duration=self.controls(X)
        basis=_spline_basis(self.internal_points,int(samples))[0]
        return np.einsum('sc,ncj->nsj',basis,controls),duration

    def objectives_from_metrics(self,M):
        raw=np.atleast_2d(M)[:,:4].copy()
        raw[:,1]=np.log1p(raw[:,1])/np.log1p(1e5)
        raw[:,[0,2,3]]/=np.array([15.,self.path_length_scale,1.])
        return raw-1.

    def constraints_from_metrics(self,M):
        return np.atleast_2d(M)[:,4:4+self.n_ieq_constr].copy()

    def certified_clearance(self,control):
        """Bound whole intervals, refining only unresolved collision gaps.

        Sphere/obstacle signed distance is 1-Lipschitz in sphere position.
        Serial-chain motion radii and continuous acceleration bounds enclose
        every intervening configuration. An exhausted refinement is infeasible.
        """
        B,D,_,derivatives=self._basis
        grid=np.linspace(0.,1.,self.samples)
        values=self.clearance_at(B@control);velocity=D@control
        exposure=float(np.trapezoid(1./(1.+np.maximum(values[:,0],0.)/.03)**2,grid))
        acceleration=np.max(np.abs(derivatives[1]@control),axis=0)
        while True:
            width=np.diff(grid)
            bound=np.maximum(np.abs(velocity[:-1]),np.abs(velocity[1:]))+width[:,None]*acceleration/2.
            lower=np.minimum(values[:-1],values[1:])-(bound@self._motion_radii.T)*width[:,None]/2.
            unresolved=np.flatnonzero(np.any(lower<self.clearance,axis=1))
            if not len(unresolved) or values.min()<self.clearance or len(grid)>=self.max_collision_samples:
                return lower.min(axis=0),exposure,len(grid)
            room=self.max_collision_samples-len(grid)
            chosen=unresolved[np.argsort(lower[unresolved].min(axis=1),kind='stable')[:room]]
            mid=(grid[chosen]+grid[chosen+1])/2.
            positions=self._spline(mid)@control;new_velocity=self._spline.derivative(1)(mid)@control
            new_values=self.clearance_at(positions)
            order=np.argsort(np.r_[grid,mid],kind='stable')
            grid=np.r_[grid,mid][order]
            values=np.vstack((values,new_values))[order];velocity=np.vstack((velocity,new_velocity))[order]

    def _measurement(self,item):
        control,T=item;B,D,gram,derivatives=self._basis
        lower,exposure,configurations=self.certified_clearance(control)
        jerk=max(0.,float(np.sum(control*(gram@control))))/T**5
        speed=np.linalg.norm(D@control,axis=1)
        length=float(np.trapezoid(speed,dx=1./(self.samples-1)))
        constraints=list((self.clearance-lower)/.03)
        for order,(matrix,limit) in enumerate(zip(derivatives,[self.velocity,self.acceleration,self.jerk_limit]),1):
            # Derivative B-spline convex hull bounds the whole continuous
            # trajectory, including extrema between collision samples.
            constraints.extend(np.max(np.abs(matrix@control),axis=0)/(T**order*limit)-1.)
        return np.r_[T,jerk,length,exposure,constraints,configurations]

    def _evaluate(self,X,out,*args,**kwargs):
        X=validate_X(X,self.n_var,self.lower,self.upper)
        from numba import get_num_threads
        controls,duration=self.controls(X)
        M=np.asarray(parallel_map(self._measurement,zip(controls,duration),workers=get_num_threads())).reshape(-1,len(self.metric_names))
        out['M']=M;out['F']=self.objectives_from_metrics(M);out['G']=self.constraints_from_metrics(M)
