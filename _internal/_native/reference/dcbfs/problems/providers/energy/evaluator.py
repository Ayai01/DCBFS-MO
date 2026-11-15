"""Frozen robust storage dispatch and CORT phantom fluence optimization."""
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix


class ApplicationProblem:
    def __init__(self,case,data):
        self.case=self.name=case
        with np.load(Path(data),allow_pickle=False)as z:self.data={k:z[k].copy()for k in z.files}
        d=self.data;self.kind=str(d['kind']);self.n_var=int(d['n_var'])
        self.lower=np.zeros(self.n_var);self.upper=np.ones(self.n_var);self.nominal=np.full(self.n_var,.5)
        self.constraints_depend_on_objectives=False;self.compact_cache_keys=True;self.fidelity_levels=('highest',)
        if self.kind=='storage':
            self.n_obj,self.n_ieq_constr=4,4
            self.metric_names=['grid_energy_MWh','peak_grid_MW','grid_mean_abs_ramp_MW','battery_throughput_MWh',
                'minimum_SOC_MWh','maximum_SOC_MWh','terminal_SOC_error_MWh','curtailed_energy_MWh']
            self.objective_names=['grid_energy','grid_peak','grid_ramping','battery_wear']
        else:
            self.n_obj,self.n_ieq_constr=3,3
            self.metric_names=['mean_Core_Gy','mean_BODY_Gy','mean_adjacent_fluence_difference_MU','minimum_target_Gy','maximum_target_Gy','maximum_Core_Gy']
            self.objective_names=['Core_mean_dose','BODY_mean_dose','fluence_modulation']
            for name in ('target','core'):
                setattr(self,name,csr_matrix((d[name+'_data'],d[name+'_indices'],d[name+'_indptr']),shape=tuple(d[name+'_shape'])))

    def canonicalize(self,X):return np.clip(np.asarray(X,float),0,1).copy()

    def compute(self,X):
        d=self.data
        if self.kind=='storage':
            power=float(d['battery_power_MW'])*(2*X-1)
            discharge=np.maximum(power,0);charge=np.maximum(-power,0)
            soc=float(d['initial_SOC_MWh'])+np.cumsum(.95*charge-discharge/.95,axis=1)
            balance=d['net_load_MW'][None,:,:]-power[:,None,:]
            grid=np.maximum(balance,0);curtailed=np.maximum(-balance,0)
            return np.column_stack((grid.sum(axis=2).mean(axis=1),grid.max(axis=(1,2)),
                np.abs(np.diff(grid,axis=2)).mean(axis=(1,2)),np.abs(power).sum(axis=1),
                np.minimum(soc.min(axis=1),float(d['initial_SOC_MWh'])),
                np.maximum(soc.max(axis=1),float(d['initial_SOC_MWh'])),
                np.abs(soc[:,-1]-float(d['initial_SOC_MWh'])),curtailed.sum(axis=2).mean(axis=1)))
        fluence=d['fluence_upper_MU']*X
        target=self.target@fluence.T;core=self.core@fluence.T
        modulation=np.abs(fluence[:,d['adjacent_i']]-fluence[:,d['adjacent_j']]).mean(axis=1)
        return np.column_stack((fluence@d['Core_mean_coeff'],fluence@d['BODY_mean_coeff'],modulation,
            target.min(axis=0),target.max(axis=0),core.max(axis=0)))

    def objectives_from_metrics(self,M):
        cols=4 if self.kind=='storage'else 3
        return -np.clip(1-M[:,:cols]/self.data['objective_scales'],0,1)

    def constraints_from_metrics(self,M):
        d=self.data
        if self.kind=='storage':
            return np.column_stack((.1*float(d['battery_energy_MWh'])-M[:,4],
                M[:,5]-.9*float(d['battery_energy_MWh']),M[:,6]-.05*float(d['battery_energy_MWh']),
                M[:,1]-float(d['grid_capacity_MW'])))/d['constraint_scales']
        return np.column_stack((1-M[:,3],M[:,4]-1.2,M[:,5]-1.2))

    def evaluate(self,X,return_as_dictionary=False):
        M=self.compute(self.canonicalize(X));out=dict(M=M,F=self.objectives_from_metrics(M),G=self.constraints_from_metrics(M))
        return out if return_as_dictionary else out['F']

    def qualification_metrics(self,X):
        s=np.sin(6.3*X[:,:4]);n=len(X)
        if self.kind=='storage':
            return np.column_stack((40+20*s[:,0],1+.3*s[:,1],.2+.1*s[:,2],20+10*s[:,3],
                .8+.4*s[:,0],7.2+.4*s[:,1],.4+.3*s[:,2],np.full(n,5.)))
        return np.column_stack((.3+.1*s[:,0],.15+.05*s[:,1],10+5*s[:,2],
            1+.2*s[:,0],1.2+.2*s[:,1],.6+.3*s[:,3]))
