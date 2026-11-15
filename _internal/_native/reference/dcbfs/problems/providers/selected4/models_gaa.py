"""Native GAA27 (27 continuous variables, 10 objectives, ONE aggregate CV).

Equations are extracted only from the frozen upstream MIT C header, not from
DESDEO or the LGPL Python subtree. See gaa-source/SOURCE.json and GAA.h.
Upstream copyright 2004, 2010-2012 Timothy W. Simpson and others; MIT license
is reproduced in gaa-source/LICENSE-MIT.txt and the unmodified source header.

Public X is unit-box float32. Physical mapping, coefficient operands, polynomial
products/accumulation, PFPF, signed constraints and F/G/M all use float32. The
nonconstant-first polynomial order passes the frozen 32eps physical-scale gate.
Only static source parsing/interval metadata and the independent verifier use64.
M contains 27 physical responses, PFPF, then 18 normalized SIGNED inequalities.
G preserves the original one-column sum(max(signed_g, 0)), NOT an 18-G variant.
Raw objectives are the original ten all-minimized objectives. F uses a FIXED
positive affine normalization from C coefficient interval bounds, never clipping.
M and raw_objectives_from_metrics preserve physical source values. No copied
objective, synthetic objective, variable rounding or empirical/adaptive scaling.
"""
from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
import re

import numpy as np

SOURCE_DIR = Path(__file__).resolve().parent / "gaa-source"
SOURCE_COMMIT = "6bbb4ed2f4be48c94bb9f325c6c04b39fff93acd"
VARIABLE_BASES = ("CSPD", "AR", "SWEEP", "DPROP", "WINGLD", "AF", "SEATW", "ELODT", "TAPER")
RESPONSE_BASES = ("NOISE", "WEMP", "DOC", "ROUGH", "WFUEL", "PURCH", "RANGE", "LDMAX", "VCMAX")
AIRCRAFT = (2, 4, 6)
RESPONSE_NAMES = tuple(f"{base}{a}" for a in AIRCRAFT for base in RESPONSE_BASES)
RAW_CONSTRAINT_NAMES = tuple(f"{base}{a}_CV_signed" for a in AIRCRAFT
                             for base in ("NOISE", "WEMP", "DOC", "ROUGH", "WFUEL", "RANGE"))
CROSS_I, CROSS_J = np.triu_indices(9, 1)
MONOMIALS = ((),) + tuple((i,) for i in range(9)) + tuple(zip(CROSS_I, CROSS_J)) + tuple((i, i) for i in range(9))


def _add_poly(left, right, sign=1.0):
    result = dict(left)
    for key, value in right.items():
        result[key] = result.get(key, 0.0) + sign * value
    return result


def _polynomial(node, variables):
    """Read a restricted quadratic expression as exact source float constants.

    No eval, source execution, fitted coefficients or data-dependent regression.
    The AST rejects expressions outside sums/products of constants and variables,
    and pow(variable,2); a mismatch in upstream syntax fails closed.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return {(): float(node.value)}
    if isinstance(node, ast.Name) and node.id in variables:
        return {(variables.index(node.id),): 1.0}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _polynomial(node.operand, variables)
        return {k: (-v if isinstance(node.op, ast.USub) else v) for k, v in value.items()}
    if isinstance(node, ast.BinOp):
        left = _polynomial(node.left, variables)
        right = _polynomial(node.right, variables)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return _add_poly(left, right, -1.0 if isinstance(node.op, ast.Sub) else 1.0)
        if isinstance(node.op, ast.Mult):
            result = {}
            for k, a in left.items():
                for m, b in right.items():
                    key = tuple(sorted(k + m))
                    if len(key) > 2:
                        raise ValueError("GAA source polynomial exceeds degree two")
                    result[key] = result.get(key, 0.0) + a * b
            return result
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "pow" and len(node.args) == 2
            and isinstance(node.args[0], ast.Name) and node.args[0].id in variables
            and isinstance(node.args[1], ast.Constant) and node.args[1].value == 2
            and not node.keywords):
        i = variables.index(node.args[0].id)
        return {(i, i): 1.0}
    raise ValueError(f"Unsupported GAA expression: {ast.dump(node)}")


@lru_cache(maxsize=1)
def source_coefficients():
    """Return (3,9,55) immutable coefficients extracted from MIT C source."""
    text = (SOURCE_DIR / "GAA.h").read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    coeff = np.zeros((3, 9, 55), dtype=np.float64)
    for a_index, a in enumerate(AIRCRAFT):
        variables = tuple(f"{base}{a}" for base in VARIABLE_BASES)
        for r_index, base in enumerate(RESPONSE_BASES):
            matches = re.findall(r"\bdouble\s+" + base + str(a) + r"\s*=([^;]+);", text)
            if len(matches) != 1:
                raise ValueError(f"Expected one MIT C expression for {base}{a}")
            expression = " ".join(matches[0].split())
            polynomial = _polynomial(ast.parse(expression, mode="eval").body, variables)
            if not set(polynomial).issubset(MONOMIALS):
                raise ValueError("Unknown GAA monomial")
            coeff[a_index, r_index] = [polynomial.get(k, 0.0) for k in MONOMIALS]
    coeff.setflags(write=False)
    return coeff


class GAAProblem:
    """Duck-typed private application provider; no optimizer dependency."""

    def __init__(self, case="GAA27"):
        if case != "GAA27":
            raise ValueError("Only the unmodified native GAA27 case is provided")
        self.case = self.name = case
        self.n_var, self.n_obj, self.n_ieq_constr = 27, 10, 1
        self.lower = np.zeros(27, dtype=np.float32)
        self.upper = np.ones(27, dtype=np.float32)
        self.nominal = np.full(27, 0.5, dtype=np.float32)
        self.physical_lower = np.tile([.24, 7, 0, 5.5, 19, 85, 14, 3, .46], 3).astype(np.float32)
        self.physical_upper = np.tile([.48, 11, 6, 5.968, 25, 110, 20, 3.75, 1], 3).astype(np.float32)
        self._center = np.array([.36, 9, 3, 5.734, 22, 97.5, 17, 3.375, .73], dtype=np.float32)
        self._scale = np.array([.12, 2, 3, .234, 3, 12.5, 3, .375, .27], dtype=np.float32)
        self._coefficients = source_coefficients().astype(np.float32)
        self._coefficients.setflags(write=False)
        # Every nonconstant monomial is in [-1,1]. This intentionally
        # conservative bound also uses abs(square coefficients), per frozen
        # owner protocol. No observed points, PF or method outputs enter it.
        # Float64 is used only for source parsing / static bound metadata, NOT
        # for any physical evaluation. Frozen FP32 affine operands are below.
        c = source_coefficients()
        radius = np.abs(c[:, :, 1:]).sum(axis=2)
        self.response_lower = c[:, :, 0] - radius
        self.response_upper = c[:, :, 0] + radius
        self.objective_lower = np.r_[self.response_lower[:, :6].max(axis=0),
                                     -self.response_upper[:, 6:].min(axis=0), 0.]
        self.objective_upper = np.r_[self.response_upper[:, :6].max(axis=0),
                                     -self.response_lower[:, 6:].min(axis=0), np.sqrt(24.)]
        self.objective_scale = self.objective_upper - self.objective_lower
        if not np.all(self.objective_scale > 0):
            raise ValueError("GAA fixed affine normalization has nonpositive span")
        self._objective_lower32 = self.objective_lower.astype(np.float32)
        self._objective_scale32 = self.objective_scale.astype(np.float32)
        self.constraint_scale = np.ones(1, dtype=np.float32)
        self.objective_names = ["worst_NOISE", "worst_WEMP", "worst_DOC", "worst_ROUGH",
                                "worst_WFUEL", "worst_PURCH", "negative_worst_RANGE",
                                "negative_worst_LDMAX", "negative_worst_VCMAX", "PFPF"]
        self.metric_names = list(RESPONSE_NAMES) + ["PFPF"] + list(RAW_CONSTRAINT_NAMES)
        self.constraint_names = ["native_sum_positive_normalized_CV"]
        self.raw_constraint_names = list(RAW_CONSTRAINT_NAMES)
        self.constraints_depend_on_objectives = False
        self.compact_cache_keys = True
        self.fidelity_levels = ("highest",)
        self.dtype_profile = {"canonical_X": "float32", "physical_and_internal": "float32",
                              "coefficients_and_polynomial_accumulator": "float32",
                              "M": "float32", "F": "float32", "G": "float32",
                              "static_source_parse_and_bound_metadata_only": "float64"}

    def canonicalize(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        if X.ndim != 2 or X.shape[1] != self.n_var:
            raise ValueError(f"Expected (n,{self.n_var}) unit-box decisions, got {X.shape}")
        result = np.clip(X, np.float32(0), np.float32(1)).copy()
        if not np.isfinite(result).all():
            raise ValueError("GAA decisions contain NaN")
        return result

    def physical(self, X):
        X = self.canonicalize(X).astype(np.float32)
        return self.physical_lower + X * (self.physical_upper - self.physical_lower)

    def _compute32(self, canonical_X):
        """Physical evaluation is entirely FP32; add the large intercept ONCE."""
        X = np.asarray(canonical_X, dtype=np.float32)
        physical = self.physical_lower + X * (self.physical_upper - self.physical_lower)
        z = (physical.reshape(-1, 3, 9) - self._center) / self._scale
        phi = np.concatenate((z, z[:, :, CROSS_I] * z[:, :, CROSS_J], z * z), axis=2)
        # Summing a large intercept first loses small increments in FP32.
        # The equivalent nonconstant-first order passes the fixed 32eps gate.
        response = np.einsum("nat,art->nar", phi,
                             self._coefficients[:, :, 1:], optimize=False)
        response += self._coefficients[:, :, 0]
        delta = z - z.mean(axis=1, keepdims=True)
        pfpf = np.sqrt(np.sum(delta * delta, axis=(1, 2)))
        limit = np.array([[75, 2200, 80, 2, 450, 2000],
                          [75, 2200, 80, 2, 475, 2000],
                          [75, 2200, 80, 2, 500, 2000]], dtype=np.float32)
        signed = (response[:, :, [0, 1, 2, 3, 4, 6]] - limit) / limit
        signed[:, :, 5] *= -1.0
        return np.column_stack((response.reshape(-1, 27), pfpf, signed.reshape(-1, 18)))

    def compute(self, X):
        return self._compute32(self.canonicalize(X))

    def _metrics(self, M):
        M = np.atleast_2d(np.asarray(M))
        if M.ndim != 2 or M.shape[1] != len(self.metric_names):
            raise ValueError("Expected 46 GAA metrics (27 responses + PFPF + 18 signed g)")
        return M

    def raw_objectives_from_metrics(self, M):
        """Original ten minimized objectives, retaining the input metric dtype."""
        M = self._metrics(M)
        response = M[:, :27].reshape(-1, 3, 9)
        return np.column_stack((response[:, :, :6].max(axis=1),
                                -response[:, :, 6:].min(axis=1), M[:, 27]))

    def objectives_from_metrics(self, M):
        raw = self.raw_objectives_from_metrics(M).astype(np.float32)
        # Positive affine transformation only; do NOT clip to the nominal box.
        return (raw - self._objective_lower32) / self._objective_scale32

    def constraints_from_metrics(self, M):
        M = self._metrics(M)
        # This nonnegative sum is the author's native one-column constraint.
        return np.maximum(M[:, 28:].astype(np.float32), np.float32(0)).sum(axis=1, keepdims=True, dtype=np.float32)

    def raw_constraints(self, X):
        """18 normalized SIGNED physical inequalities, diagnostic only."""
        return self.compute(X)[:, 28:].copy()

    def evaluate(self, X, return_as_dictionary=False):
        M = self.compute(X)
        output = {"F": self.objectives_from_metrics(M), "G": self.constraints_from_metrics(M), "M": M}
        return output if return_as_dictionary else output["F"]


def build(case="GAA27"):
    """Build one native case (no fabricated multi-instance family)."""
    return GAAProblem(case)
