"""The four accepted application models, with their original physical arithmetic.

F is float32(N)-1; G and diagnostic M retain the archived model definitions.
The original modules and fixed inputs are separately hashed in the catalog.
"""
from pathlib import Path
import importlib.util
import numpy as np


class ApplicationProblem:
    constraints_depend_on_objectives = False
    compact_cache_keys = True
    fidelity_levels = ("highest",)

    def __init__(self, case, data_path):
        assets = Path(data_path).parent
        if case == "AGR48-N":
            module_name = "models_root"
        elif case in ("DOM_RESERVOIR_BASE", "DOM_RIVER_DO_BASE"):
            module_name = "models_domains"
        elif case == "GAA27":
            module_name = "models_gaa"
        else:
            raise KeyError(case)
        path = Path(__file__).with_name(module_name + ".py")
        spec = importlib.util.spec_from_file_location("selected4_" + module_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.HERE = assets
        module.DATA = assets / "domain-data"
        module.SOURCE_DIR = assets / "gaa-source"
        self.model = module.build(case)
        self.case = self.name = case
        self.n_var, self.n_obj = self.model.n_var, self.model.n_obj
        self.n_ieq_constr = self.model.n_ieq_constr
        self.n_constr, self.n_eq_constr = self.n_ieq_constr, 0
        self.lower = np.zeros(self.n_var, dtype=np.float32)
        self.upper = np.ones(self.n_var, dtype=np.float32)
        self.metric_names = self.model.metric_names
        self.objective_names = self.model.objective_names
        self.domain_affine = module_name == "models_domains"

    def canonicalize(self, X):
        return self.model.canonicalize(X)

    def compute(self, X):
        return self.model.compute(X)

    def objectives_from_metrics(self, M):
        values = self.model.objectives_from_metrics(M)
        if self.domain_affine:
            values = self.model.affine_objectives(values)
        return np.subtract(np.asarray(values, dtype=np.float32), np.float32(1), dtype=np.float32)

    def constraints_from_metrics(self, M):
        return self.model.constraints_from_metrics(M)

    def evaluate(self, X, return_as_dictionary=False):
        M = self.compute(self.canonicalize(X))
        result = dict(M=M, F=self.objectives_from_metrics(M), G=self.constraints_from_metrics(M))
        return result if return_as_dictionary else result["F"]
