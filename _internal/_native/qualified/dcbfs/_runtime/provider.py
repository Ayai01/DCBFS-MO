"""Equivalent Panda evaluation leaves for the accepted portable FP32 profile."""
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
from numba import njit


@njit(nogil=True, fastmath=False, cache=False)
def basis32(flat, t, degree, count):
    work = np.empty((len(flat), len(t)-1), np.float32)
    for row in range(len(flat)):
        for j in range(len(t)-1):
            work[row, j] = np.float32(flat[row] >= t[j] and flat[row] < t[j+1])
        for order in range(1, degree+1):
            for j in range(len(t)-order-1):
                left = t[order+j]-t[j]
                right = t[order+1+j]-t[j+1]
                a, b = np.float32(0), np.float32(0)
                if left != np.float32(0):
                    a = (flat[row]-t[j])/left
                if right != np.float32(0):
                    b = (t[order+1+j]-flat[row])/right
                work[row, j] = a*work[row, j]+b*work[row, j+1]
        if flat[row] == t[-degree-1]:
            for j in range(count):
                work[row, j] = np.float32(j == count-1)
    return np.ascontiguousarray(work[:, :count])


@contextmanager
def context(stable):
    from dcbfs.problems.panda import PandaTrajectory
    from . import portable_collision_scratch as collision
    audit = dict(spline_calls=0, adaptive_calls=0, adaptive_configurations=0,
                 precision='existing portable FP32 profile; no new casts or arithmetic boundary')
    original = PandaTrajectory.clearance_at

    def spline_call(spline, x, nu=0):
        if nu:
            return spline.derivative(nu)(x)
        x = np.asarray(x, np.float32)
        audit['spline_calls'] += 1
        basis = basis32(x.ravel(), spline.t, spline.k, len(spline.c))
        return (basis@spline.c).reshape(x.shape+spline.c.shape[1:])

    def clearance(source, joints):
        joints = np.atleast_2d(joints)
        if joints.dtype != np.float32:
            return original(source, joints)
        audit['adaptive_calls'] += 1
        audit['adaptive_configurations'] += len(joints)
        return collision.batch(np.ascontiguousarray(joints[None]), *source._collision)[0]

    with patch.object(stable.precision.Spline32, '__call__', spline_call), \
         patch.object(PandaTrajectory, 'clearance_at', clearance):
        yield audit
