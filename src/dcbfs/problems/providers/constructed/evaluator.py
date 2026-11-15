"""Original ENV48/logistics/lake models and frozen affine N-1 adapter."""
from pathlib import Path
import importlib.util
import numpy as np
class ApplicationProblem:
    constraints_depend_on_objectives=False
    compact_cache_keys=True
    fidelity_levels=('highest',)
    def __init__(self,case,data_path):
        name='models_root' if case=='ENV48' else 'models_domains'
        source=importlib.util.spec_from_file_location('extra3_original_'+name,Path(__file__).with_name(name+'.py'))
        module=importlib.util.module_from_spec(source);source.loader.exec_module(module)
        assets=Path(data_path).parent;module.HERE=assets;module.DATA=assets/'domain-data'
        self.model=module.build(case);self.domain_affine=name=='models_domains'
        self.case=self.name=case;self.n_var=self.model.n_var;self.n_obj=self.model.n_obj;self.n_ieq_constr=self.model.n_ieq_constr
        self.n_constr=self.n_ieq_constr;self.n_eq_constr=0
        self.lower=np.zeros(self.n_var,np.float32);self.upper=np.ones(self.n_var,np.float32)
        self.metric_names=self.model.metric_names;self.objective_names=self.model.objective_names
    def canonicalize(self,X):return self.model.canonicalize(X)
    def compute(self,X):return self.model.compute(X)
    def objectives_from_metrics(self,M):
        N=self.model.objectives_from_metrics(M)
        if self.domain_affine:N=self.model.affine_objectives(N)
        return np.subtract(np.asarray(N,np.float32),np.float32(1),dtype=np.float32)
    def constraints_from_metrics(self,M):return self.model.constraints_from_metrics(M)
    def evaluate(self,X,return_as_dictionary=False):
        M=self.compute(self.canonicalize(X));r=dict(M=M,F=self.objectives_from_metrics(M),G=self.constraints_from_metrics(M))
        return r if return_as_dictionary else r['F']
