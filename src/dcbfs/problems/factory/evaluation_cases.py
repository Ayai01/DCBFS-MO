import numpy as np
import pickle
from pathlib import Path
import os
STATIC=Path(os.environ.get('DCBFS_STATIC_ROOT') or Path(__file__).parent)
from pymoo.core.problem import Problem
from pymoo.problems import get_problem
from dcbfs.problems import make_problem as registered_problem


def ctp_paper_constraint(theta,a,b,c,d,e,f1,f2):
    # Original continuous paper form (also documented in pymoo's source).
    first=(f2-e)*np.cos(theta)-f1*np.sin(theta)
    phase=(f2-e)*np.sin(theta)+f1*np.cos(theta)
    second=a*np.abs(np.sin(b*np.pi*phase**c))**d
    return second-first


class NativeStandard(Problem):
    """Independent published CMOPs: original constraints, fixed affine F only."""
    constraints_depend_on_objectives = False

    def __init__(self,name,**options):
        self.name=name
        if name in ('dascmop7','dascmop8','dascmop9') and 'difficulty' in options:
            options['difficulty_factors']=options.pop('difficulty')
        self.base=get_problem(name,**options)
        if name.startswith('ctp') and name!='ctp1':
            self.base.calc_constraint=ctp_paper_constraint
        key=f'native-{name}'
        if name.startswith('dascmop'):key+=f'-difficulty{self.base.difficulty}'
        elif 'n_obj' in options and self.base.n_obj>3:key+=f'-m{self.base.n_obj}'
        reference=pickle.loads((STATIC/'native-references.pkl').read_bytes())[key]
        self.worst,self.denom=reference['worst'],reference['denom']
        super().__init__(n_var=self.base.n_var,n_obj=self.base.n_obj,
            n_ieq_constr=self.base.n_ieq_constr,xl=0.,xu=1.)
        self.metric_names=[f'raw_f{i+1}' for i in range(self.n_obj)]
        self.objective_names=[f'f{i+1}' for i in range(self.n_obj)]
        self.variable_names=[f'x{i+1}' for i in range(self.n_var)]
        self.reference_definition={'generator':'independent-native-cmop'}
        self.hv_reference=reference['reference_point']
        self.hv_axis_scale=reference['axis_scale']

    def decode(self,X):
        return self.base.xl+np.asarray(X)*(self.base.xu-self.base.xl)

    def raw_values(self,X):
        physical=self.decode(X)
        if self.name not in ('mw6','mw11'):
            return self.base.evaluate(physical,return_as_dictionary=True)
        # Same closed-domain formulas; (r-x)(r+x) avoids f0/g cancellation.
        radius=1.1 if self.name=='mw6' else np.sqrt(2.)
        g=self.base.g2(physical) if self.name=='mw6' else self.base.g3(physical)
        f0=g*physical[:,0]
        f1=g*np.sqrt((radius-physical[:,0])*(radius+physical[:,0]))
        if self.name=='mw6':
            angle=np.arctan2(f1,f0)
            g0=f0**2/(1.+self.base.LA3(.15,6.,4.,10.,angle))**2+f1**2/(1.+self.base.LA3(.75,6.,4.,10.,angle))**2-1.
            G=g0[:,None]
        else:
            G=np.column_stack((-(3.-f0*f0-f1)*(3.-2.*f0*f0-f1),
                (3.-.625*f0*f0-f1)*(3.-7.*f0*f0-f1),
                -(1.62-.18*f0*f0-f1)*(1.125-.125*f0*f0-f1),
                (2.07-.23*f0*f0-f1)*(.63-.07*f0*f0-f1)))
        return dict(F=np.column_stack((f0,f1)),G=G)

    def _evaluate(self,X,out,*args,**kwargs):
        raw=self.raw_values(X)
        out['F']=(raw['F']-self.worst)/self.denom
        out['G']=raw['G']
        out['M']=raw['F']

    def metrics(self,X):
        raw=self.raw_values(X)['F']
        return dict(zip(self.metric_names,raw.T))


def make_problem(name,**options):
    return NativeStandard(name[7:],**options) if name.startswith('native-') else registered_problem(name,**options)
