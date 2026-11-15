"""Fast, auditable landscapes that isolate multiobjective search mechanisms.

These are diagnostic constructions, not renamed official benchmark instances.
Every instance has a deterministic parameterized global Pareto front. Optimizer
coordinates are dimensionless unit coordinates; F=-U and all G must be <=0.
"""
from __future__ import annotations

from operator import index

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from pymoo.core.problem import Problem
from ..utils.design import validate_X


LANDSCAPE_NAMES = (
    "rotated-ellipsoid", "multimodal-rastrigin", "disconnected-corridor",
    "narrow-corridor", "cutoff-plateau", "degenerate-redundant", "mixed-scale",
    "biased-dtlz4",
)

FORMULA_SOURCES = (
    {"title": "Deb, Thiele, Laumanns and Zitzler: Scalable Multi-Objective Optimization Test Problems",
     "url": "https://www.research-collection.ethz.ch/server/api/core/bitstreams/4d08c69c-9f97-424d-bbf6-7a647bcb3d13/content",
     "used_for": "DTLZ position-distance construction, spherical front, multimodal distance and angular bias"},
    {"title": "COCO BBOB f10: rotated ellipsoid",
     "url": "https://coco-platform.org/testsuites/bbob/functions/f10.html",
     "used_for": "Rotated quadratic anisotropy; our multiobjective construction shares one positive definite Hessian"},
)

MECHANISMS = {
    "rotated-ellipsoid": "Nonseparable quadratic with controlled Hessian condition number; two-dimensional triangular Pareto design manifold",
    "multimodal-rastrigin": "Repeated distance minima around a known spherical global front; stagnation and escape without domain failures",
    "disconnected-corridor": "Three disconnected feasible Pareto segments and a curved design manifold; island coverage and constraint navigation",
    "narrow-corridor": "Two narrow curved feasibility corridors around a continuous front; constraint progress versus objective progress",
    "cutoff-plateau": "Coupled square-law bias cutoff, unavailable performance objectives, finite cost objective and finite violation signal",
    "degenerate-redundant": "Six objective axes, three independent component objectives, three redundant combinations, intrinsic front dimension two",
    "mixed-scale": "Twelve orders of raw physical scale and monotone nonlinear objective curvature; utility normalization removes units",
    "biased-dtlz4": "DTLZ4-style angular power bias with a spherical front; direction collapse and uneven search-space accessibility",
}


def _count(value, name, minimum=1):
    try:
        result = index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    if isinstance(value, (bool, np.bool_)) or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _sobol(n, dimension):
    if not n:
        return np.empty((0, dimension))
    # random_base2 avoids Sobol balance warnings and is independent of n's
    # power-of-two status; the first n points always share the same prefix.
    return qmc.Sobol(dimension, scramble=True, seed=4117).random_base2(
        max(0, (n-1).bit_length()))[:n]


def _sphere_samples(n):
    anchors = np.eye(3)[:min(n, 3)]
    if n <= 3:
        return anchors
    P = np.clip(_sobol(n-3, 3), 1e-12, 1.-1e-12)
    # Independent half-normal variables yield uniform surface measure over
    # the positive orthant, instead of concentrating angular samples at poles.
    Z = ndtri(.5+.5*P)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    return np.vstack((anchors, Z))


class LandscapeProblem(Problem):
    """Batch problem with exact global-front diagnostics and fixed U anchors."""

    def __init__(self, name, *, n_var=None, normalization=None,
                 condition_number=1000., bias_exponent=20., corridor_width=None):
        name = str(name).strip().lower()
        if name not in LANDSCAPE_NAMES:
            raise ValueError(f"Unknown landscape: {name}")
        self.name = name
        default_n = {"rotated-ellipsoid": 8, "disconnected-corridor": 8,
                     "narrow-corridor": 10, "cutoff-plateau": 10}.get(name, 12)
        self.n_var = _count(default_n if n_var is None else n_var, "n_var", 4)
        self.n_obj = (2 if name in {"disconnected-corridor", "narrow-corridor"}
                      else 6 if name == "degenerate-redundant" else 3)
        self.lower, self.upper = np.zeros(self.n_var), np.ones(self.n_var)
        self.variable_names = [f"x{i}" for i in range(self.n_var)]
        self.objectives = self.objective_names = [f"f{i}" for i in range(self.n_obj)]
        self.condition_number, self.bias_exponent = float(condition_number), float(bias_exponent)
        if not np.isfinite(self.condition_number) or not 1. <= self.condition_number <= 1e8:
            raise ValueError("condition_number must be finite in [1, 1e8]")
        if not np.isfinite(self.bias_exponent) or not 1. <= self.bias_exponent <= 100.:
            raise ValueError("bias_exponent must be finite in [1, 100]")
        width = (.025 if name == "narrow-corridor" else .10) if corridor_width is None else float(corridor_width)
        if not np.isfinite(width) or not .005 <= width <= .25:
            raise ValueError("corridor_width must be finite in [.005, .25]")
        self.corridor_width = width
        self._regions = ((0., .18), (.34, .56), (.78, 1.))
        self._redundancy = np.vstack((np.eye(3), [.6, .4, 0.], [0., .5, .5], [.3, .4, .3]))
        if name == "rotated-ellipsoid":
            j, k = np.arange(self.n_var)[None, :], np.arange(self.n_var)[:, None]
            self._rotation = np.sqrt(2./self.n_var)*np.cos(np.pi/self.n_var*(j+.5)*k)
            self._rotation[0] = 1./np.sqrt(self.n_var)
            self._weights = np.r_[1., 1., np.geomspace(1., self.condition_number, self.n_var-2)]
            self._centers = np.zeros((3, self.n_var))
            angle = 2.*np.pi*np.arange(3)/3.
            self._centers[:, :2] = .45*np.column_stack((np.cos(angle), np.sin(angle)))
            target = np.array([np.max(np.sum((self._centers-center)**2*self._weights, axis=1))
                               for center in self._centers])
            best = np.zeros(3)
        elif name in {"disconnected-corridor", "narrow-corridor"}:
            target, best = np.ones(2), np.zeros(2)
        elif name == "cutoff-plateau":
            target, best = np.array([1., 1.4, 1.]), np.zeros(3)
        elif name == "degenerate-redundant":
            target = np.linalg.norm(self._redundancy, axis=1)
            best = self._redundancy.min(axis=1)
        elif name == "mixed-scale":
            target = self._mixed_raw(np.ones((1, 3)))[0]
            best = np.zeros(3)
        else:
            target, best = np.ones(3), np.zeros(3)
        # A few upward ulps make the mathematically closed front remain closed
        # after trigonometric reconstruction; no clipping changes utilities.
        target += 16.*np.finfo(float).eps*np.maximum(np.abs(target), np.finfo(float).tiny)
        default_worst = target+.1*(target-best)
        self.normalization = {}
        for i, objective in enumerate(self.objectives):
            anchors = dict((normalization or {}).get(objective, {}))
            if set(anchors)-{"worst", "target"}:
                raise ValueError("Landscape normalization accepts worst and target")
            poor, goal = float(anchors.get("worst", default_worst[i])), float(anchors.get("target", target[i]))
            if not np.isfinite([poor, goal]).all() or poor <= goal:
                raise ValueError("Landscape minimization goals require finite worst > target")
            self.normalization[objective] = {"metric": f"raw_{objective}", "worst": poor,
                "target": goal, "direction": "minimize", "method": "linear", "scale": None, "unit": "dimensionless"}
        if set(normalization or {})-set(self.objectives):
            raise ValueError("Unknown landscape objective normalization")
        self._worst = np.array([self.normalization[k]["worst"] for k in self.objectives])
        self._target = np.array([self.normalization[k]["target"] for k in self.objectives])
        original_constraints = ({"disconnected-corridor": ["feasible_island", "curved_corridor"],
                                 "narrow-corridor": ["corridor_1", "corridor_2"],
                                 "cutoff-plateau": ["operating_headroom", "operating_upper_bias"]}.get(name, []))
        self.constraint_names = original_constraints+[f"target_{key}" for key in self.objectives]
        super().__init__(n_var=self.n_var, n_obj=self.n_obj, n_ieq_constr=len(self.constraint_names), xl=0., xu=1.)
        self.metric_names = [f"raw_f{i}" for i in range(self.n_obj)] + list(self._raw(np.full((1,self.n_var), .5))[2])
        self.metric_metadata = [{"name":metric, "label":goal, "spec":goal,
            "spec_value":self.normalization[goal]["target"], "strict":False, "comparison":"<=",
            **self.normalization[goal]} for metric,goal in zip(self.metric_names,self.objective_names)]
        self.metric_metadata += [{"name":name, "label":name, "unit":"", "direction":"minimize"}
                                 for name in self.metric_names[self.n_obj:]]
        self.objective_metadata = [{**item, "name":name} for name,item in zip(self.objective_names,self.metric_metadata)]
        self.normalization_info = {"display_space":"utility", "coordinate_space":"utility",
            "utility_formula":"U=(raw_F-worst)/(target-worst)", "internal_objective_formula":"F=-U",
            "utility_target":1., "utility_direction":"higher is better", "utility_clipping":False,
            "constraint_formula":"native physical G and target G=1-U", "anchors":self.normalization}
        self.reference_definition = {"generator":"landscape-analytic", "problem":self.name,
            "n_var":self.n_var, "n_obj":self.n_obj, "sample_count":1000, "normalization":self.normalization,
            "options":{"condition_number":self.condition_number,"bias_exponent":self.bias_exponent,
                       "corridor_width":self.corridor_width}}
        if name=="rotated-ellipsoid":
            formula={"kind":"triangular-quadratic","centers":self._centers[:,:2].tolist()}
        elif name in ("disconnected-corridor","narrow-corridor"):
            formula={"kind":"line","regions":[list(r) for r in self._regions] if name=="disconnected-corridor" else [[0.,1.]]}
        elif name=="cutoff-plateau":
            formula={"kind":"cutoff-plane","objectives":["t","1-t+0.4*r","1-r"],"parameter_bounds":[[0.,1.],[0.,1.]]}
        else:
            formula={"kind":"positive-unit-sphere","raw_transform":"mixed-scale" if name=="mixed-scale" else "identity"}
            if name=="degenerate-redundant":formula["redundancy"]=self._redundancy.tolist()
        self.reference_definition["front_formula"]=formula

    def encode(self, X):
        return validate_X(X, self.n_var, self.lower, self.upper).copy()

    def decode(self, Z):
        return validate_X(Z, self.n_var, self.lower, self.upper).copy()

    def utilities(self, raw_F):
        return (np.asarray(raw_F)-self._worst)/(self._target-self._worst)

    @staticmethod
    def _mixed_raw(F):
        return np.column_stack((1e-6*F[:, 0], 1e3*np.expm1(2.*F[:, 1]),
                                1e6*np.log1p(9.*F[:, 2])))

    def _curved_center(self, t):
        center = np.full((len(t), self.n_var-1), .5)
        center[:, 0] += .18*np.sin(2.*np.pi*t)
        center[:, 1] += .14*np.cos(4.*np.pi*t)
        return center

    def _raw(self, X):
        N = len(X)
        G = np.empty((N, 0))
        extras = {}
        if self.name == "rotated-ellipsoid":
            Y = (2.*X-1.)@self._rotation.T
            weighted = Y*self._weights
            F = np.sum(Y*weighted, axis=1)[:, None]-2.*weighted@self._centers.T
            F += np.sum(self._centers**2*self._weights, axis=1)
            return np.maximum(F, 0.), G, extras
        if self.name in {"disconnected-corridor", "narrow-corridor"}:
            t = X[:, 0]
            residual = X[:, 1:]-self._curved_center(t)
            g = 4.*np.mean(residual**2, axis=1)
            F = np.column_stack((t+g, 1.-t+g))
            if self.name == "disconnected-corridor":
                island = np.minimum.reduce([np.maximum(a-t, t-b) for a, b in self._regions])
                G = np.column_stack((island, np.abs(residual[:, 0])-self.corridor_width))
            else:
                G = np.abs(residual[:, :2])-self.corridor_width
            extras = {"distance_to_front_manifold": np.linalg.norm(residual, axis=1)}
            return F, G, extras
        if self.name == "cutoff-plateau":
            t = X[:, 0]
            b = (np.sqrt((.04+.96*X[:, 1])/(.8+.4*X[:, 2]))-.2)/.8
            g = 4.*np.mean((X[:, 2:]-.5)**2, axis=1)
            r = (b-.2)/.8
            F = np.column_stack((t+g, 1.-t+.4*r+g, 1.-r+g))
            F[b <= .15, 1:] = np.nan
            G = np.column_stack((.2-b, b-1.))
            extras = {"operating_bias": b, "cutoff_margin": b-.15,
                      "available_objectives": np.isfinite(F).sum(axis=1).astype(float)}
            return F, G, extras
        A = X[:, :2].copy()
        if self.name == "biased-dtlz4":
            A **= self.bias_exponent
        A *= np.pi/2.
        direction = np.column_stack((np.cos(A[:, 0])*np.cos(A[:, 1]),
                                     np.cos(A[:, 0])*np.sin(A[:, 1]), np.sin(A[:, 0])))
        residual = X[:, 2:]-.5
        if self.name == "multimodal-rastrigin":
            g = 3.*np.mean(residual**2+1.-np.cos(8.*np.pi*residual), axis=1)
        else:
            g = 4.*np.mean(residual**2, axis=1)
        F = (1.+g)[:, None]*direction
        if self.name == "degenerate-redundant":
            F = F@self._redundancy.T
        elif self.name == "mixed-scale":
            F = self._mixed_raw(F)
        return F, G, {"distance_penalty": g}

    def _evaluate(self, X, out, *args, **kwargs):
        X = validate_X(X, self.n_var, self.lower, self.upper)
        raw_F, physical_G, metrics = self._raw(X)
        U = self.utilities(raw_F)
        # Unavailable values remain NaN objectives. Algorithms already use -10
        # for missing U; the accompanying violation is finite and informative.
        target_G = 1.-np.nan_to_num(U, nan=-10.)
        out["F"], out["G"] = -U, np.column_stack((physical_G, target_G))
        metrics |= {f"raw_f{i}": raw_F[:, i] for i in range(self.n_obj)}
        out["M"] = np.column_stack([metrics[name] for name in self.metric_names])

    def reference_designs(self, n=1000):
        n = _count(n, "n")
        X = np.full((n, self.n_var), .5)
        if self.name == "rotated-ellipsoid":
            if n == 1:
                bary = np.full((1, 3), 1./3.)
            else:
                anchors = np.eye(3)[:min(n, 3)]
                S = _sobol(max(0, n-3), 2)
                root = np.sqrt(S[:, 0])
                bary = np.vstack((anchors, np.column_stack((1.-root, root*(1.-S[:, 1]), root*S[:, 1]))))
            X = .5+.5*(bary@self._centers@self._rotation)
        elif self.name in {"disconnected-corridor", "narrow-corridor"}:
            if self.name == "disconnected-corridor":
                counts = np.full(3, n//3)
                counts[:n % 3] += 1
                t = np.concatenate([np.linspace(a, b, k) if k > 1 else np.array([.5*(a+b)])
                                    for (a, b), k in zip(self._regions, counts) if k])
            else:
                t = np.array([.5]) if n == 1 else np.linspace(0., 1., n)
            X[:, 0], X[:, 1:] = t, self._curved_center(t)
        elif self.name == "cutoff-plateau":
            P = np.array([[.5, .5]]) if n == 1 else _sobol(n, 2)
            if n >= 4:
                P[:4] = [[0., 0.], [1., 0.], [0., 1.], [1., 1.]]
            b = .2+.8*P[:, 1]
            b[b == .2] += 1e-12  # inward sample of the floating-point headroom boundary
            X[:, 0] = P[:, 0]
            X[:, 1] = ((.2+.8*b)**2-.04)/.96
        else:
            S = _sphere_samples(n)
            X[:, 0] = 2./np.pi*np.arcsin(S[:, 2])
            X[:, 1] = 2./np.pi*np.arctan2(S[:, 1], S[:, 0])
            if self.name == "biased-dtlz4":
                X[:, :2] **= 1./self.bias_exponent
        return np.clip(X, 0., 1.)

    def reference_front(self, n=1000):
        """Deterministic global Pareto samples in the public minimization F=-U."""
        return -self.utilities(self._raw(self.reference_designs(n))[0])

    def reference_hypervolume(self):
        from .pareto_volume import reference_hypervolume
        return reference_hypervolume(self)

    def metadata(self):
        options = {}
        if self.name == "rotated-ellipsoid":
            options["condition_number"] = self.condition_number
        if self.name == "biased-dtlz4":
            options["bias_exponent"] = self.bias_exponent
        if self.name in {"disconnected-corridor", "narrow-corridor"}:
            options["corridor_width"] = self.corridor_width
        return {"name": self.name, "n_var": self.n_var, "n_obj": self.n_obj,
                "lower": self.lower.tolist(), "upper": self.upper.tolist(),
                "variable_names": self.variable_names, "objectives": self.objectives,
                "normalization": self.normalization, "constraint_names": self.constraint_names,
                "options": options, "mechanism": MECHANISMS[self.name],
                "formula_sources": FORMULA_SOURCES,
                "objective_direction": "maximize U; algorithm tensor F=-U",
                "default_target_rule": "exact coordinate nadir of the known global front, with 16-ulp reconstruction margin",
                "reference_front": "deterministic parameterized global front; Sobol seed 4117, sphere sampled uniformly in surface measure",
                "diagnostic_construction": True, "initial_design_supplied": False}


def analytic_reference(definition):
    anchors = {key:{"worst":value["worst"],"target":value["target"]}
               for key,value in definition["normalization"].items()}
    problem = LandscapeProblem(definition["problem"],n_var=definition["n_var"],
                               normalization=anchors,**definition["options"])
    return problem.reference_front(definition["sample_count"])


__all__ = ["LANDSCAPE_NAMES", "MECHANISMS", "LandscapeProblem", "analytic_reference"]
