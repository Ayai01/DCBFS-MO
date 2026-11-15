"""Continuous Pareto HV from closed forms and deterministic 1D sections."""
import math
import hashlib
import json
from pathlib import Path
from uuid import uuid4
from functools import lru_cache
from itertools import combinations
import numpy as np
from numba import njit
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import erf, betainc
from scipy.stats import qmc

from ..utils.hypervolume import SOBOL_SEEDS


@njit(cache=(not bool(__import__('os').environ.get('DCBFS_NO_FILES'))), nogil=True)
def _c2_covered(queries, cap):
    covered = 0
    m = queries.shape[1]
    for row in queries:
        norm2,maximum,minimum = 0.,0.,np.inf
        for j in range(m):
            norm2 += row[j]*row[j]
            maximum = max(maximum,row[j])
            minimum = min(minimum,row[j])
        if norm2 < 1.:
            continue
        if maximum >= cap or minimum >= 1./math.sqrt(m):
            covered += 1
            continue
        ordered = np.sort(row)
        squared, total = 0., 0.
        for j in range(m):
            level = math.sqrt(max(0., (1.-squared)/(m-j)))
            if level <= ordered[j]:
                if total+(m-j)*level >= cap*math.sqrt(m):
                    covered += 1
                break
            squared += ordered[j]**2
            total += ordered[j]
    return covered


@lru_cache(maxsize=128)
def _c2_volume(m, target, radius):
    r = np.asarray(target)
    cap = 1.-radius*radius/2.
    if cap <= 1./math.sqrt(m):
        return float(np.prod(r))-clipped_ball(r,1.)
    if m == 2:
        angle = math.acos(np.clip(cap, -1., 1.))
        pieces = sorted([(0., min(angle, math.pi/2)),
                         (max(0., math.pi/2-angle), math.pi/2),
                         (max(0., math.pi/4-angle), min(math.pi/2, math.pi/4+angle))])
        intervals = []
        for a, b in pieces:
            if intervals and a <= intervals[-1][1]:
                intervals[-1][1] = max(intervals[-1][1], b)
            else:
                intervals.append([a, b])
        total, previous_x, previous_y = 0., None, None
        for a, b in reversed(intervals):
            left, right = math.cos(b), math.cos(a)
            if previous_x is not None:
                total += max(0., min(left, r[0])-previous_x)*max(0., r[1]-previous_y)
            end = min(right, r[0])
            if end > left:
                total += integral(lambda x: max(0., r[1]-math.sqrt(max(0., 1.-x*x))), left, end)
            previous_x, previous_y = right, math.sin(a)
        return total+max(0., r[0]-previous_x)*max(0., r[1]-previous_y)
    if m == 3:
        from scipy.special import roots_legendre
        nodes,weights = roots_legendre(1024)
        return float(np.prod(r)-np.prod(np.minimum(r,1.))+
                     _c2_three_volume(nodes,weights,np.minimum(r,1.),cap))
    # A query outside the unit cube is dominated by a feasible coordinate pole.
    # The remaining cube is integrated with the exact continuous-cap predicate.
    upper = np.minimum(r, 1.)
    inside = float(np.prod(upper))
    estimates = []
    for seed in SOBOL_SEEDS:
        sampler = qmc.Sobol(m, scramble=True, seed=seed)
        count = 0
        for _ in range(256):
            count += _c2_covered(sampler.random(65536)*upper, cap)
        estimates.append(inside*count/2**24)
    return float(np.prod(r))-inside+float(np.mean(estimates))


@njit(cache=(not bool(__import__('os').environ.get('DCBFS_NO_FILES'))), nogil=True)
def _c2_minimum_z(x,y,cap):
    """First feasible sphere height in a two-coordinate query rectangle."""
    low = math.sqrt(max(0.,1.-x*x-y*y))
    best = low if max(x,y) >= cap else max(cap,low)
    small,large = min(x,y),max(x,y)
    total = cap*math.sqrt(3.)
    radius = math.sqrt(max(0.,1.-low*low))
    if small >= radius/math.sqrt(2.):
        at_low = math.sqrt(2.)*radius+low
    else:
        at_low = small+math.sqrt(max(0.,radius*radius-small*small))+low
    if at_low >= total:
        return min(best,low)
    z = (total-math.sqrt(max(0.,6.-2.*total*total)))/3.
    if z >= low and z >= 0. and math.sqrt(max(0.,(1.-z*z)/2.)) <= small:
        best = min(best,z)
    c = total-small
    discriminant = 2.*(1.-small*small)-c*c
    if discriminant >= 0.:
        z = (c-math.sqrt(discriminant))/2.
        if z >= low and z >= 0. and z*z+small*small <= 1.:
            other = math.sqrt(max(0.,1.-small*small-z*z))
            if other <= large:
                best = min(best,z)
    return best


@njit(cache=(not bool(__import__('os').environ.get('DCBFS_NO_FILES'))), nogil=True)
def _c2_three_volume(nodes,weights,target,cap):
    volume = 0.
    minimum = (cap-math.sqrt(max(0.,2.*(1.-cap*cap))))/math.sqrt(3.)
    balanced = math.sqrt(max(0.,(1.-max(0.,minimum)**2)/2.))
    xs = np.sort(np.array([0.,min(target[0],max(0.,minimum)),
                          min(target[0],balanced),min(target[0],cap),target[0]]))
    for k in range(len(xs)-1):
        dx = .5*(xs[k+1]-xs[k])
        if dx <= 0.:
            continue
        for i in range(len(nodes)):
            x = xs[k]+dx*(1.+nodes[i])
            if x < minimum:
                central = target[1]
            elif x >= balanced:
                central = max(0.,minimum)
            else:
                c = cap*math.sqrt(3.)-x
                central = max(0.,(c-math.sqrt(max(0.,2.*(1.-x*x)-c*c)))/2.)
            ys = np.sort(np.array([0.,min(target[1],central),
                                  min(target[1],cap),min(target[1],math.sqrt(max(0.,1.-x*x))),target[1]]))
            for l in range(len(ys)-1):
                dy = .5*(ys[l+1]-ys[l])
                if dy <= 0.:
                    continue
                for j in range(len(nodes)):
                    y = ys[l]+dy*(1.+nodes[j])
                    volume += dx*dy*weights[i]*weights[j]*max(0.,target[2]-_c2_minimum_z(x,y,cap))
    return volume

def ball(m):
    return math.pi ** (m / 2) / (2 ** m * math.gamma(m / 2 + 1))


def clipped_ball(bounds, radius):
    """Positive orthant of a ball intersected with a finite positive box."""
    r = np.maximum(np.asarray(bounds),0.)
    if not np.all(r > 0.):
        return 0.
    m = len(r)
    cuts = r[r < radius-1e-12*max(1.,radius)]
    if not len(cuts):
        return radius**m*ball(m)
    if len(cuts) == 1:
        return radius**m*ball(m)*betainc(.5,(m+1)/2.,(cuts[0]/radius)**2)
    if np.sum(r*r) <= radius*radius:
        return float(np.prod(r))
    if m == 2:
        stop = min(r[0],radius)
        flat = min(stop,math.sqrt(max(0.,radius*radius-r[1]*r[1])))
        def primitive(x):
            return .5*(x*math.sqrt(max(0.,radius*radius-x*x))+radius*radius*math.asin(x/radius))
        return r[1]*flat+primitive(stop)-primitive(flat)
    if m == 3:
        return integral(lambda x:clipped_ball(r[1:],math.sqrt(max(0.,radius*radius-x*x))),0.,min(radius,r[0]))
    raise NotImplementedError('multiply clipped high-dimensional ball')
def integral(fn, a, b):
    return quad(fn, a, b, epsabs=1e-12, epsrel=1e-12, limit=200)[0]
def convex_complement(m):
    value = 1.0
    for k in range(2, m + 1):
        value *= integral(lambda t: (1 - math.cos(t)) ** (k - 1) * math.cos(t), 0.0, math.pi / 2)
    return value

def native_hv(name, m, target, params=None):
    """Native-coordinate PF volume, with exact simplex, ball and curve sections."""
    r = np.asarray(target)
    params = params or {}
    if name != 'zdt3' and np.any(r <= 0.0):
        return 0.0
    if name == 'c2dtlz2':
        return _c2_volume(m, tuple(r), abs(params.get('r', 0.4 if m == 3 else 0.5)))
    if name in ('dtlz1', 'c1dtlz1'):
        slack = float(r.sum()) - 0.5
        if slack <= 0.0:
            return 0.0
        level = min(0.5, slack)
        sums, signs = (np.zeros(1), np.ones(1))
        for bound in r[r < level]:
            sums, signs = (np.r_[sums, sums + bound], np.r_[signs, -signs])
        section = float(signs @ np.maximum(level - sums, 0.0) ** m) / math.factorial(m)
        return section if slack <= 0.5 else float(np.prod(r)) - section
    if name in ('dtlz2', 'dtlz3', 'dtlz4', 'c1dtlz3'):
        radius = 4.0 if name == 'c1dtlz3' and abs(params.get('r', 1.0)) < 1 else 1.0
        return float(np.prod(r)) - clipped_ball(r, radius)
    if name in ('dtlz5', 'dtlz6'):
        a = np.ones(m - 1)
        for i in range(m - 1):
            a[i] = 2 ** (-(m - 1 - i) / 2) if i else 2 ** (-(m - 2) / 2)
        return integral(lambda s: float(np.prod(np.maximum(r[:-1] - a * math.sqrt(max(0.0, 1 - s * s)), 0.0))), 0.0, min(1.0, r[-1])) + max(0.0, r[-1] - 1) * float(np.prod(r[:-1]))
    if name == 'dtlz7':
        if r[-1] < 2 * m - 1e-12:
            raise NotImplementedError('clipped DTLZ7 final objective')
        q = lambda x: x * (1 + math.sin(3 * math.pi * x))
        dq = lambda x: 1 + math.sin(3 * math.pi * x) + 3 * math.pi * x * math.cos(3 * math.pi * x)
        a = brentq(dq, 1 / 6, 1 / 3)
        b = brentq(dq, 5 / 6, 1)
        c = brentq(lambda x: q(x) - q(a), 0.5, 5 / 6)
        Q = lambda x: x * x / 2 - x * math.cos(3 * math.pi * x) / (3 * math.pi) + math.sin(3 * math.pi * x) / (9 * math.pi ** 2)

        def area(t):
            return Q(min(t, a)) + max(0.0, min(t, c) - a) * q(a) + (Q(min(t, b)) - Q(c) if t > c else 0.0) + max(0.0, t - b) * q(b)
        return float(np.prod(r[:-1])) * (r[-1] - 2 * m + sum((area(t) / t for t in r[:-1])))
    if name.startswith('wfg'):
        s = 2 * np.arange(1, m + 1)
        unit = r / s
        if name == 'wfg3':
            a = s[:-1] * np.r_[2.0 ** (-(m - 2)), 2.0 ** (-np.arange(m - 2, 0, -1))]
            return integral(lambda y: float(np.prod(np.maximum(r[:-1] - a * max(0.0, 1 - y / s[-1]), 0.0))), 0.0, r[-1])
        if name not in ('wfg1', 'wfg2'):
            excluded = clipped_ball(unit, 1.0)
        else:
            if np.any(unit < 1.0 - 1e-12):
                raise NotImplementedError('clipped WFG convex sections')
            weight = lambda t: (1 - math.cos(math.pi * t / 2)) ** (m - 1)
            if name == 'wfg1':
                area = integral(lambda t: weight(t) * (1 - math.cos(10 * math.pi * t)), 0.0, 1.0)
            else:
                h = lambda t: 1 - t * math.cos(5 * math.pi * t) ** 2
                derivative = lambda t: -math.cos(5 * math.pi * t) ** 2 + 5 * math.pi * t * math.sin(10 * math.pi * t)
                area = 0.0
                previous = 1.0
                for lobe in range(6):
                    stop = brentq(derivative, lobe / 5, (lobe + 0.25) / 5) if lobe < 5 else 1.0
                    start = 0.0 if lobe == 0 else brentq(lambda t: h(t) - previous, (lobe - 0.5) / 5, stop)
                    area += integral(lambda t: -weight(t) * derivative(t), start, stop)
                    previous = h(stop)
            excluded = convex_complement(m - 1) * area
        return float(np.prod(s)) * (float(np.prod(unit)) - excluded)
    if name == 'c3dtlz1':
        if np.any(r < 1.0 - 1e-12):
            raise NotImplementedError('clipped C3DTLZ1')
        return float(np.prod(r)) - 1 / ((m + 1) * math.factorial(m - 1))
    if name == 'c3dtlz4':
        if np.any(r < 2.0 - 1e-12):
            raise NotImplementedError('clipped C3DTLZ4')
        sector = integral(lambda z: math.sqrt(2 / math.pi) * math.exp(-z * z / 2) * erf(math.sqrt(2) * z) ** (m - 1), 0.0, np.inf)
        return float(np.prod(r)) - 2 * m * ball(m) * sector
    if name.startswith('zdt'):
        if name == 'zdt6':
            x = math.atan(9 * math.pi) / (6 * math.pi)
            low = 1 - math.exp(-4 * x) * math.sin(6 * math.pi * x) ** 6
        else:
            low = 0.0
        if name != 'zdt3':
            curve = (lambda x: 1 - x * x) if name in ('zdt2', 'zdt6') else lambda x: 1 - math.sqrt(x)
            if r[0] <= low:
                return 0.
            return (integral(lambda x: max(0.0, r[1] - curve(x)), low, min(1.,r[0])) +
                    max(0.,r[0]-1.)*r[1])
        intervals = ((0.0, 0.0830015349269116), (0.182228728029398, 0.25776236338783), (0.409313674808657, 0.45388210408883), (0.618396794416603, 0.652511703804663), (0.823331798326633, 0.851832865436414))
        curve = lambda x: 1 - math.sqrt(x) - x * math.sin(10 * math.pi * x)
        total = 0.0
        for i, (a, b) in enumerate(intervals):
            if r[0] <= a:
                break
            total += integral(lambda x: max(0.0, r[1] - curve(x)), a, min(b, r[0]))
            end = intervals[i + 1][0] if i + 1 < len(intervals) else r[0]
            total += max(0.0, min(end, r[0]) - b) * max(0.0, r[1] - curve(b))
        return total
    raise NotImplementedError(name)

_reference_cache = {}

def reference_hypervolume(problem):
    """Restore fixed-utility volume and cache expensive continuous definitions."""
    definition = {'formula_version': 2, **problem.reference_definition}
    key = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
    expensive = problem.name in ('c2dtlz2', 'degenerate-redundant', 'mixed-scale', 'rotated-ellipsoid')
    if hasattr(problem,'frozen_reference_hv'):return float(problem.frozen_reference_hv)
    if key in _reference_cache:
        value,dense_count=_reference_cache[key]
        problem.reference_hv_cache_hit=True;problem.reference_hv_dense_count=dense_count
        return value
    problem.reference_hv_dense_count = 0
    try:
        if hasattr(problem, 'front_parameters'):
            volume = native_hv(problem.name, problem.n_obj, problem._target, problem.front_parameters)
        else:
            volume = _landscape_volume(problem.name, tuple(problem._target))
        value = volume / float(np.prod(problem._worst - problem._target))
    except NotImplementedError:
        from ..utils.hypervolume import hypervolume
        problem.reference_hv_dense_count = 2 ** 20
        value = hypervolume(problem.reference_front(2 ** 20), 2 ** 28)
        expensive = True
    problem.reference_hv_cache_hit = False
    _reference_cache[key]=(value,problem.reference_hv_dense_count)
    return value


@njit(cache=(not bool(__import__('os').environ.get('DCBFS_NO_FILES'))), nogil=True)
def _third_disk_distance(q0, q1, centers):
    x0,y0 = centers[0,0],centers[0,1]
    x1,y1 = centers[1,0],centers[1,1]
    xt,yt = centers[2,0],centers[2,1]
    dt0 = (xt-x0)**2+(yt-y0)**2
    dt1 = (xt-x1)**2+(yt-y1)**2
    if dt0 <= q0 and dt1 <= q1:
        return 0.
    best = np.inf
    for i in range(2):
        x,y = (x0,y0) if i == 0 else (x1,y1)
        xo,yo = (x1,y1) if i == 0 else (x0,y0)
        radius2 = q0 if i == 0 else q1
        other_radius2 = q1 if i == 0 else q0
        length2 = dt0 if i == 0 else dt1
        fraction = min(1.,np.sqrt(radius2/length2))
        cx,cy = x+(xt-x)*fraction,y+(yt-y)*fraction
        if (cx-xo)**2+(cy-yo)**2 <= other_radius2:
            best = min(best,length2*(1.-fraction)**2)
    dx,dy = x1-x0,y1-y0
    distance = np.sqrt(dx*dx+dy*dy)
    offset = (q0-q1+distance*distance)/(2.*distance)
    height2 = q0-offset*offset
    if height2 >= 0.:
        mx,my = x0+offset*dx/distance-xt,y0+offset*dy/distance-yt
        projection = abs((-mx*dy+my*dx)/distance)
        best = min(best,mx*mx+my*my+height2-2.*np.sqrt(height2)*projection)
    return best


@njit(cache=(not bool(__import__('os').environ.get('DCBFS_NO_FILES'))), nogil=True)
def _disk_volume(nodes, weights, target, centers):
    value = 0.
    for i in range(len(nodes)):
        x = .5*target[0]*(1.+nodes[i])
        for j in range(len(nodes)):
            y = .5*target[1]*(1.+nodes[j])
            value += weights[i]*weights[j]*max(0.,target[2]-_third_disk_distance(x,y,centers))
    return .25*target[0]*target[1]*value


@njit(cache=(not bool(__import__('os').environ.get('DCBFS_NO_FILES'))), nogil=True)
def _redundant_covered(queries, matrix, active, inverses):
    covered = 0
    for q in queries:
        if q[0]*q[0]+q[1]*q[1]+q[2]*q[2] < 1.:
            continue
        for rows, inverse in zip(active, inverses):
            a = q[rows[0]] if rows[0] < 6 else 0.
            b = q[rows[1]] if rows[1] < 6 else 0.
            c = q[rows[2]] if rows[2] < 6 else 0.
            x = inverse[0,0]*a+inverse[0,1]*b+inverse[0,2]*c
            y = inverse[1,0]*a+inverse[1,1]*b+inverse[1,2]*c
            z = inverse[2,0]*a+inverse[2,1]*b+inverse[2,2]*c
            if min(x,y,z) < -1e-12 or x*x+y*y+z*z < 1.:
                continue
            feasible = True
            for i in range(6):
                if matrix[i,0]*x+matrix[i,1]*y+matrix[i,2]*z > q[i]+1e-12:
                    feasible = False
                    break
            if feasible:
                covered += 1
                break
    return covered


@lru_cache(maxsize=128)
def _landscape_volume(name, target):
    """Continuous diagnostic-front geometry, independent of discrete plotting samples."""
    r = np.asarray(target)
    if name in ('multimodal-rastrigin', 'biased-dtlz4'):
        return native_hv('dtlz2', 3, r)
    if name in ('narrow-corridor', 'disconnected-corridor'):
        regions = ((0.0, 0.18), (0.34, 0.56), (0.78, 1.0)) if name == 'disconnected-corridor' else ((0.0, 1.0),)
        total, previous = (0.0, None)
        for a, b in regions:
            if previous is not None:
                total += max(0.0, min(a, r[0]) - previous) * max(0.0, r[1] - 1.0 + previous)
            if min(b, r[0]) > a:
                total += integral(lambda x: max(0.0, r[1] - 1.0 + x), a, min(b, r[0]))
            previous = b
        return total + max(0.0, r[0] - previous) * max(0.0, r[1] - 1.0 + previous)
    if name == 'cutoff-plateau':

        def section(x):
            c = r[1] - 1.4 + min(x, 1.0)
            stop = min(1.0, r[2])
            return (max(0.0, c + 0.4 * stop) ** 2 - max(0.0, c) ** 2) / 0.8 + max(0.0, r[2] - 1.0) * max(0.0, c + 0.4)
        return integral(section, 0.0, r[0])
    if name == 'mixed-scale':
        xupper, yupper, zupper = (r[0] / 1e-06, np.log1p(r[1] / 1000.0) / 2.0, np.expm1(r[2] / 1000000.0) / 9.0)
        height = np.log1p(9.0 * zupper)

        def section(x):
            low = np.sqrt(max(0.0, 1.0 - x * x - zupper * zupper))
            circle = np.sqrt(max(0.0, 1.0 - x * x))
            stop = min(yupper, circle)
            total = integral(lambda y: 2.0 * np.exp(2.0 * y) * (height - np.log1p(9.0 * np.sqrt(max(0.0, 1.0 - x * x - y * y)))), low, stop) if stop > low else 0.0
            return total + height * max(0.0, np.exp(2.0 * yupper) - np.exp(2.0 * circle))
        return 1000.0 * integral(section, 0.0, xupper)
    if name == 'rotated-ellipsoid':
        angle = 2.0 * np.pi * np.arange(3) / 3.0
        centers = 0.45 * np.column_stack((np.cos(angle), np.sin(angle)))
        from scipy.special import roots_legendre
        nodes, weights = roots_legendre(4096)
        return float(_disk_volume(nodes, weights, r, centers))
    if name == 'degenerate-redundant':
        matrix = np.vstack((np.eye(3), [0.6, 0.4, 0.0], [0.0, 0.5, 0.5], [0.3, 0.4, 0.3]))
        constraints = np.vstack((matrix, -np.eye(3)))
        active, inverses = ([], [])
        for rows in combinations(range(9), 3):
            selected = constraints[list(rows)]
            if abs(np.linalg.det(selected)) > 1e-12:
                active.append(rows)
                inverses.append(np.linalg.inv(selected))
        active, inverses = (np.asarray(active), np.asarray(inverses))
        estimates = []
        for seed in SOBOL_SEEDS:
            sampler = qmc.Sobol(6, scramble=True, seed=seed)
            count = 0
            for _ in range(256):
                count += _redundant_covered(sampler.random(65536) * r, matrix, active, inverses)
            estimates.append(float(np.prod(r)) * count / 2 ** 24)
        return float(np.mean(estimates))
    raise NotImplementedError(name)

