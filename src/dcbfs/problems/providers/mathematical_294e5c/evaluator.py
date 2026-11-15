"""Published continuous benchmark formulas with frozen front-based scales."""
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from pymoo.core.problem import Problem
from pymoo.problems import get_problem


class ApplicationProblem(Problem):
    def __init__(self,case,data_path):
        self.case=case;self.name=case
        with np.load(data_path,allow_pickle=False)as z:self.data={k:z[k]for k in z}
        spec=json.loads(str(self.data['definition']))
        self.specification=spec
        self.base=get_problem(spec['problem'],**spec['options'])
        module=Path(inspect.getfile(type(self.base)))
        if hashlib.sha256(module.read_bytes()).hexdigest()!=spec['implementation_sha256']:
            raise ValueError('Published problem implementation differs from the frozen source')
        self.ideal=self.data['ideal'];self.nadir=self.data['nadir']
        self.span=self.nadir-self.ideal
        assert np.all(self.span>0)
        self.metric_names=[f'published_f{i}'for i in range(self.base.n_obj)]+[f'published_g{i}'for i in range(self.base.n_ieq_constr)]
        self.objective_names=self.metric_names[:self.base.n_obj]
        self.constraint_names=self.metric_names[self.base.n_obj:]
        self.variable_names=[f'x{i}'for i in range(self.base.n_var)]
        self.constraints_depend_on_objectives=False
        self.compact_cache_keys=False
        super().__init__(n_var=self.base.n_var,n_obj=self.base.n_obj,n_ieq_constr=self.base.n_ieq_constr,xl=np.zeros(self.base.n_var),xu=np.ones(self.base.n_var))

    def canonicalize(self,X):return np.clip(np.asarray(X,float),0,1)

    def compute(self,X):
        X=np.atleast_2d(X);physical=self.base.xl+X*(self.base.xu-self.base.xl)
        result=self.base.evaluate(physical,return_as_dictionary=True)
        G=result.get('G',np.empty((len(X),0)))
        return np.column_stack((result['F'],G))

    def objectives_from_metrics(self,M):
        return (np.asarray(M)[:,:self.n_obj]-self.ideal)/self.span-1.1

    def constraints_from_metrics(self,M):return np.asarray(M)[:,self.n_obj:]

    def qualification_metrics(self,X):
        X=np.atleast_2d(X)
        raw=np.column_stack([self.ideal[i]+self.span[i]*(.15+.7*X[:,i%self.n_var])for i in range(self.n_obj)])
        G=np.column_stack([X[:,i%self.n_var]-.55 for i in range(self.n_ieq_constr)])if self.n_ieq_constr else np.empty((len(X),0))
        return np.column_stack((raw,G))

    def _evaluate(self,X,out,*args,**kwargs):
        M=self.compute(X);out.update(F=self.objectives_from_metrics(M),G=self.constraints_from_metrics(M),M=M)
