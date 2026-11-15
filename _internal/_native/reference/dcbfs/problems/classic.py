"""ZDT, DTLZ, WFG and constrained DTLZ with fixed utility anchors."""
from itertools import product
from copy import copy

import numpy as np
from pymoo.core.problem import Problem
from pymoo.problems import get_problem
from scipy.optimize import brentq
from scipy.stats import qmc
from moocore import is_nondominated

CLASSIC = ("zdt1", "zdt2", "zdt3", "zdt4", "zdt6", *(f"dtlz{i}" for i in range(1, 8)),
           *(f"wfg{i}" for i in range(1, 10)), "c1dtlz1", "c1dtlz3", "c2dtlz2", "c3dtlz1", "c3dtlz4")


class Classic(Problem):
    constraints_depend_on_objectives = True

    def __init__(self, name, normalization=None, **options):
        self.name = name
        if name.startswith("wfg"):
            options.setdefault("n_obj", 3)
            options.setdefault("k", max(4, 2*(options["n_obj"]-1)))
            options.setdefault("n_var", options["k"]+20)
        if name in ("dtlz2", "dtlz3", "dtlz7") and "n_var" not in options:
            options["n_var"] = options.get("n_obj", 3) - 1 + (20 if name == "dtlz7" else 10)
        self.base = get_problem(name, **options)
        # WFG starts by dividing physical variables by their upper bounds.
        # Evaluate its unchanged transforms directly in unit coordinates:
        # multiplying/dividing 0.35 otherwise creates tiny distance residuals
        # that WFG1's exponent 0.02 amplifies into a substantial objective error.
        self._unit_base = copy(self.base) if name.startswith("wfg") else None
        if self._unit_base is not None:
            self._unit_base.xu = np.ones(self.base.n_var)
        super().__init__(n_var=self.base.n_var, n_obj=self.base.n_obj,
                         n_ieq_constr=self.base.n_obj+self.base.n_ieq_constr, xl=0., xu=1.)
        self.front_parameters = {"r": float(self.base.r)} if hasattr(self.base, "r") else {}
        self.objective_names = [f"f{i}" for i in range(self.n_obj)]
        self.variable_names = [f"x{i}" for i in range(self.n_var)]
        self.metric_names = [f"raw_{name}" for name in self.objective_names]
        supplied = normalization or {}
        if set(supplied) - set(self.objective_names):
            raise ValueError("Unknown mathematical objective normalization")
        reference = self.raw_reference_front(2048)
        best, target = reference.min(axis=0), reference.max(axis=0)
        worst = target + .1 * np.maximum(target - best, .1)
        self.normalization = {}
        for i, name in enumerate(self.objective_names):
            custom = supplied.get(name, {})
            poor, goal = float(custom.get("worst", worst[i])), float(custom.get("target", target[i]))
            if not np.isfinite([poor, goal]).all() or poor <= goal:
                raise ValueError("Mathematical minimization objectives require finite worst > target")
            self.normalization[name] = {"metric": self.metric_names[i], "worst": poor,
                "target": goal, "method": "linear", "scale": None,
                "direction": "minimize", "unit": "dimensionless"}
        self._worst = np.array([self.normalization[name]["worst"] for name in self.objective_names])
        self._target = np.array([self.normalization[name]["target"] for name in self.objective_names])
        self.constraint_names = [f"target_{name}" for name in self.objective_names] + [
            f"native_g{i}" for i in range(self.base.n_ieq_constr)]
        self.metric_metadata = [{"name": metric, "label": name, "spec": name,
            "spec_value": self.normalization[name]["target"], "strict": False, "comparison": "<=",
            **self.normalization[name]} for name, metric in zip(self.objective_names, self.metric_names)]
        self.objective_metadata = [{**metadata, "name": name}
            for name, metadata in zip(self.objective_names, self.metric_metadata)]
        self.normalization_info = {
            "version": 2, "display_space": "utility", "coordinate_space": "utility",
            "utility_formula": "U=(native_F-worst)/(target-worst)",
            "internal_objective_formula": "F=-U", "utility_clipping": False,
            "utility_direction": "higher is better", "utility_target": 1.,
            "constraint_formula": "G=1-U; published native constraints appended" if self.base.n_ieq_constr else "G=1-U",
            "default_target_rule": "maxima of the 2048-point analytic Pareto front",
            "default_worst_rule": "worst=target+0.1*max(target-best,0.1)",
            "anchors": self.normalization,
        }
        self.reference_definition = {"generator": "classic-analytic", "problem": self.name,
                                     "n_obj": self.n_obj, "sample_count": 1000,
                                     "normalization": self.normalization}
        if self.front_parameters:
            self.reference_definition["parameters"] = self.front_parameters

    def decode(self, X):
        return self.base.xl + np.asarray(X) * (self.base.xu - self.base.xl)

    def _evaluate(self, X, out, *args, **kwargs):
        native = self._native(X)
        raw = native["F"]
        utility = self.utilities(raw)
        out["F"], out["M"] = -utility, raw
        out["G"] = np.column_stack((1.-utility, native.get("G", np.empty((len(X), 0)))))

    def metrics(self, X):
        raw = self._native(X)["F"]
        return {name: raw[:, i] for i, name in enumerate(self.metric_names)}

    def _native(self, X):
        base = self._unit_base if self._unit_base is not None else self.base
        return base.evaluate(X if self._unit_base is not None else self.decode(X), return_as_dictionary=True)

    def utilities(self, raw_F):
        return (np.asarray(raw_F) - self._worst) / (self._target - self._worst)

    def reference_front(self, n=1000):
        return -self.utilities(self.raw_reference_front(n))

    def reference_hypervolume(self):
        from .pareto_volume import reference_hypervolume
        return reference_hypervolume(self)

    def raw_reference_front(self, n=1000):
        """Native analytic objectives, sampled exactly as the original platform."""
        return analytic_front(self.name, self.n_obj, n, self.front_parameters)


def analytic_reference(definition):
    """Generate a scoring reference from its saved formula and fixed anchors."""
    raw = analytic_front(definition["problem"], definition["n_obj"], definition["sample_count"],
                         definition.get("parameters"))
    anchors = list(definition["normalization"].values())
    worst = np.array([anchor["worst"] for anchor in anchors])
    target = np.array([anchor["target"] for anchor in anchors])
    return -(raw - worst) / (target - worst)


def analytic_front(name, n_obj, n=1000, parameters=None):
    """Sample the known front without constructing or evaluating a problem."""
    if name.startswith("wfg"):
        return _wfg_front(name, n_obj, n)
    if name.startswith(("c1dtlz", "c2dtlz", "c3dtlz")):
        return _constrained_front(name, n_obj, n, parameters or {})
    if name.startswith("zdt"):
        if name == "zdt3":
            regions = ((0., .0830015349269116), (.182228728029398, .257762363387830),
                       (.409313674808657, .453882104088830), (.618396794416603, .652511703804663),
                       (.823331798326633, .851832865436414))
            counts = np.full(5, n // 5, dtype=int)
            counts[:n % 5] += 1
            first = np.concatenate([np.linspace(left + (1e-10 if i else 0.), right, count)
                for i, ((left, right), count) in enumerate(zip(regions, counts)) if count])
            second = 1. - np.sqrt(first) - first * np.sin(10. * np.pi * first)
        elif name == "zdt6":
            xmin = np.arctan(9. * np.pi) / (6. * np.pi)
            minimum = 1. - np.exp(-4. * xmin) * np.sin(6. * np.pi * xmin) ** 6
            first = np.linspace(minimum, 1., n)
            second = 1. - first ** 2
        else:
            first = np.linspace(0., 1., n)
            second = 1. - (first ** 2 if name == "zdt2" else np.sqrt(first))
        return np.column_stack((first, second))
    m = n_obj
    if name in ("dtlz5", "dtlz6"):
        theta = np.full((n, m-1), np.pi / 4)
        theta[:, 0] = np.linspace(0, np.pi / 2, n)
        F = np.ones((len(theta), m))
        for i in range(m):
            F[:, i] = np.prod(np.cos(theta[:, :m-1-i]), axis=1)
            if i:
                F[:, i] *= np.sin(theta[:, m-1-i])
        return F
    if name == "dtlz7":
        # Two monotone, nondominated intervals per free objective.
        q = lambda t: t * (1 + np.sin(3*np.pi*t))
        dq = lambda t: 1 + np.sin(3*np.pi*t) + 3*np.pi*t*np.cos(3*np.pi*t)
        a = brentq(dq, 1/6, 1/3)
        b = brentq(dq, 5/6, 1)
        c = brentq(lambda t: q(t)-q(a), .5, 5/6) + 1e-12
        sampler = qmc.Halton(m-1, scramble=False)
        sampler.fast_forward(1)
        u = sampler.random(n)
        first = np.where(u < .5, 2*u*a, c+(2*u-1)*(b-c))
        if m-1 < 20 and 2**(m-1) <= n:
            corners = np.asarray(list(product((0., c), repeat=m-1)))
            first[:len(corners)] = corners
        return np.column_stack((first, 2*m-np.sum(q(first), axis=1)))
    rng = np.random.default_rng(0)
    if name == "dtlz1":
        directions = rng.exponential(size=(n, m))
        F = .5 * directions / directions.sum(axis=1, keepdims=True)
    else:
        directions = np.abs(rng.normal(size=(n, m)))
        F = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    count = min(n, m)
    F[:count] = np.eye(m)[:count] * (.5 if name == "dtlz1" else 1.)
    return F


def _wfg_front(name, m, n):
    """Published WFG shapes; positive rays resolve convex/mixed front sections."""
    scales = 2*np.arange(1, m+1)
    if name == "wfg3":
        x = np.full((n, m-1), .5)
        x[:, 0] = np.linspace(0., 1., n)
        return np.column_stack([np.prod(x[:, :m-1-i], axis=1)*
            (1.-x[:, m-1-i] if i else 1.) for i in range(m)])*scales
    if name not in ("wfg1", "wfg2"):
        return analytic_front("dtlz2", m, n)*scales
    rng = np.random.default_rng(0)
    rays = rng.exponential(size=(n, m))
    c = np.ones((n, m))
    for j in range(2, m+1):
        i = m-j
        ratio = rays[:, j-1]/rays[:, 0]*np.prod(1.-c[:, i+1:m-1], axis=1)
        c[:, i] = (ratio**2-ratio+np.sqrt(2.*ratio))/(ratio**2+1.)
    x = np.arccos(np.clip(c, -1., 1.))*2./np.pi
    ratio = (1.-np.sin(np.pi/2.*x[:, 1]))*rays[:, -1]/rays[:, -2]
    if name == "wfg1":
        for row in range(n):
            x[row, 0] = brentq(lambda t: ratio[row]*(1.-np.cos(np.pi/2.*t))-1.+t-
                              np.sin(10.*np.pi*t)/(10.*np.pi), 0., 1.)
    else:
        grid = np.linspace(0., 1., 4097)
        shape = 1.-grid*np.cos(5.*np.pi*grid)**2
        for row in range(n):
            residual = ratio[row]*(1.-np.cos(np.pi/2.*grid))-shape
            first = np.flatnonzero((residual[:-1] <= 0.) & (residual[1:] >= 0.))[0]
            x[row, 0] = brentq(lambda t: ratio[row]*(1.-np.cos(np.pi/2.*t))-1.+
                              t*np.cos(5.*np.pi*t)**2, grid[first], grid[first+1])
    front = np.column_stack([np.prod(1.-np.cos(np.pi/2.*x[:, :m-1-i]), axis=1)*
        (1.-np.sin(np.pi/2.*x[:, m-1-i]) if i else 1.) for i in range(m)])
    t = x[:, 0]
    front[:, -1] = (1.-t+np.sin(10.*np.pi*t)/(10.*np.pi) if name == "wfg1"
                    else 1.-t*np.cos(5.*np.pi*t)**2)
    if name == "wfg2":
        # Earlier t lowers every convex objective. A point belongs to the
        # disconnected front only while its last objective improves on every
        # earlier local minimum, including minima absent from the ray sample.
        keep = np.ones(len(t), dtype=bool)
        for lobe in range(5):
            minimum = brentq(lambda u: -np.cos(5.*np.pi*u)**2+5.*np.pi*u*np.sin(10.*np.pi*u),
                             lobe/5., (lobe+.25)/5.)
            value = 1.-minimum*np.cos(5.*np.pi*minimum)**2
            keep &= (t <= minimum) | (front[:, -1] < value)
        front = front[keep]
    front = np.vstack((np.eye(m), front))*scales
    front = front[is_nondominated(front, keep_weakly=False)]
    return front[np.linspace(0, len(front)-1, min(n, len(front))).astype(int)]


def _constrained_front(name, m, n, parameters):
    from pymoo.problems.many.cdtlz import constraint_c2

    if name == "c1dtlz1":
        return analytic_front("dtlz1", m, n)
    if name == "c1dtlz3":
        # DTLZ3 has radius >= 1; when |r| < 1 its inner feasible shell is
        # unreachable, so the first attainable feasible shell has radius 4.
        radius = 4. if abs(parameters.get("r", 1.)) < 1. else 1.
        return radius*analytic_front("dtlz3", m, n)
    if name == "c3dtlz1":
        rays = analytic_front("dtlz1", m, n)*2.
        return rays/(1.+rays.min(axis=1, keepdims=True))
    if name == "c3dtlz4":
        rays = analytic_front("dtlz2", m, n)
        return rays/np.sqrt(1.-.75*(rays**2).max(axis=1, keepdims=True))
    radius = parameters.get("r", .2 if m == 2 else .4 if m == 3 else .5)
    centers = np.vstack((np.eye(m), np.full((1, m), 1./np.sqrt(m))))
    rng = np.random.default_rng(0)
    parts = [centers]
    angle_limit = min(np.pi/2., 2.*np.arcsin(min(abs(radius)/2., 1.)))
    for center in centers:
        tangent = rng.normal(size=(max(2, 2*n//(m+1)+1), m))
        tangent -= (tangent @ center)[:, None]*center
        if np.count_nonzero(center) == 1:
            tangent = np.abs(tangent)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        angle = rng.random((len(tangent), 1))*angle_limit
        cap = np.cos(angle)*center+np.sin(angle)*tangent
        parts.append(cap[(cap >= 0.).all(axis=1)])
    front = np.unique(np.vstack(parts), axis=0)
    front = front[constraint_c2(front, radius) <= 1e-14]
    other = front[~np.any(np.all(front[:, None, :] == centers[None, :, :], axis=2), axis=1)]
    count = min(max(0, n-len(centers)), len(other))
    sample = other[np.linspace(0, len(other)-1, count).astype(int)]
    return np.vstack((centers[:n], sample))
