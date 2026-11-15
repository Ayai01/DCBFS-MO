import numpy as np

from pymoo.core.problem import Problem

from .evaluation_cases import make_problem as ordinary_problem

class PublishedUnconstrained(Problem):
    constraints_depend_on_objectives=True
    def __init__(self,name,**options):
        self.original=ordinary_problem(name,**options);q=self.original
        super().__init__(n_var=q.n_var,n_obj=q.n_obj,n_ieq_constr=q.base.n_ieq_constr,xl=0.,xu=1.)
        for k in ('metric_names','objective_names','variable_names','normalization','normalization_info','reference_definition'):
            setattr(self,k,getattr(q,k))
    def _evaluate(self,X,out,*args,**kw):
        values=self.original._native(X)
        out['F']=-self.original.utilities(values['F'])
        out['G']=values.get('G',np.empty((len(X),0)));out['M']=values['F']
    def metrics(self,X):return self.original.metrics(X)
    def decode(self,X):return self.original.decode(X)

def make_problem(name,**options):
    if name.startswith('unconstrained-'):return PublishedUnconstrained(name[len('unconstrained-'):],**options)
    return ordinary_problem(name,**options)

