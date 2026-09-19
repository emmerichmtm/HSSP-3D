"""Efficient polynomial-time approximation scheme for 3-D hypervolume subset selection.

Implements Section 5 of

    K. Bringmann, S. Cabello, M. T. M. Emmerich,
    "Maximum Volume Subset Selection for Anchored Boxes" (arXiv:1803.00849)

for d = 3.  With  beta = (1-eps)^(-1/d),  tau = the smallest integer > d/eps  and
lambda = beta^g  the smallest power of beta larger than d/eps:

  (1) iterate over all grid offsets  l in {0..tau-1}^d                 (shifting technique);
  (2) delete the points in the thick grid boundaries, i.e. with region index
      x_i = floor(log_lambda p_i)  congruent to l_i (mod tau) for some i;
  (3) partition the remaining points into grid cells  y_i = floor((x_i - l_i)/tau);
  (4) round all coordinates down to powers of beta and remove duplicates;
  (5) solve every cell exactly for all budgets k' (exhaustive enumeration);
  (6) distribute the budget k over the cells by dynamic programming,
      V(l) = max_{sum k_i <= k} sum_i VolSel(cell_i, k_i);
  (7) return the best offset.

Guarantees (Lemmas 5.2-5.5), with V = max_l V(l) and S the returned set:

      mu(S) >= (1-eps) V,          V >= (1-eps)^2 VolSel(P,k),

hence  mu(S) >= (1-eps)^3 VolSel(P,k),  and  V / (1-eps)^2  is a *certified upper
bound* on the optimum, which is returned together with the solution.

Two exact simplifications are used in step (5): rounded points that are dominated by
another rounded point of the same cell are dropped (they never increase a union), and
the results of a cell are cached, because the same group of points forms a cell for
many offsets.
"""
from __future__ import annotations

import math
import time
from itertools import product

import numpy as np
from numba import njit

from . import _kernels as K
from .solver import Result, nondominated, to_anchored

D = 3


def grid_constants(eps):
    """(beta, tau, g) with lambda = beta**g."""
    if not 0.0 < eps <= 0.5:
        raise ValueError("eps must be in (0, 1/2]")
    beta = (1.0 - eps) ** (-1.0 / D)
    tau = math.floor(D / eps + 1e-12) + 1
    g = math.floor(math.log(D / eps) / math.log(beta) + 1e-12) + 1
    return beta, tau, g


def rounded_exponents(P, beta):
    """e with beta**e <= p < beta**(e+1), robust against rounding of the logarithm."""
    E = np.floor(np.log(P) / math.log(beta)).astype(np.int64)
    for _ in range(3):
        E = np.where(beta ** (E + 1.0) <= P, E + 1, E)
        E = np.where(beta ** E.astype(float) > P, E - 1, E)
    return E


@njit(cache=True)
def _cell_volsel(X, Y, Z, m, kmax):
    """VolSel(cell, k') and an optimal subset (bit mask) for k' = 0..kmax."""
    H = np.zeros(kmax + 1, np.float64)
    masks = np.zeros(kmax + 1, np.int64)
    comb = np.empty(max(kmax, 1), np.int64)
    for size in range(1, kmax + 1):
        for a in range(size):
            comb[a] = a
        while True:
            v = K.hv3d(X, Y, Z, comb, size)
            if v > H[size]:
                H[size] = v
                mk = 0
                for a in range(size):
                    mk |= (1 << comb[a])
                masks[size] = mk
            if not K._next_combination(comb, size, m):
                break
    return H, masks


@njit(cache=True)
def _distribute(H, sizes, k):
    """T[i,k'] = max_{kappa} H[i,kappa] + T[i-1,k'-kappa]   (step 6).  Returns the
    optimal value and the budget of every cell."""
    m = H.shape[0]
    T = np.zeros((m + 1, k + 1), np.float64)
    choice = np.zeros((m + 1, k + 1), np.int64)
    for i in range(1, m + 1):
        for kk in range(k + 1):
            best = T[i - 1, kk]
            bc = 0
            for kappa in range(1, min(kk, sizes[i - 1]) + 1):
                v = H[i - 1, kappa] + T[i - 1, kk - kappa]
                if v > best:
                    best = v
                    bc = kappa
            T[i, kk] = best
            choice[i, kk] = bc
    budgets = np.zeros(m, np.int64)
    kk = k
    for i in range(m, 0, -1):
        budgets[i - 1] = choice[i, kk]
        kk -= choice[i, kk]
    return T[m, k], budgets


class CellTooLarge(RuntimeError):
    """A grid cell holds too many rounded points for exhaustive enumeration."""


class _Eptas:
    def __init__(self, P, k, eps, max_cell_subsets):
        self.P = P
        self.k = k
        self.eps = eps
        self.beta, self.tau, self.g = grid_constants(eps)
        self.E = rounded_exponents(P, self.beta)            # rounded point = beta**E
        self.region = np.floor_divide(self.E, self.g)       # floor(log_lambda p)
        self.vol = P.prod(axis=1)
        self.max_cell_subsets = max_cell_subsets
        self.cell_cache = {}
        self.stats = dict(offsets=0, distinct_partitions=0, cells_solved=0, cell_subsets=0,
                          max_cell_points=0, max_cells=0)

    def solve_cell(self, idx):
        """idx: sorted tuple of point indices forming a cell.  Returns
        (H[0..kmax], for every k' the tuple of original point indices)."""
        hit = self.cell_cache.get(idx)
        if hit is not None:
            return hit
        # (4) rounding + duplicates: keep one original per rounded point
        rep = {}
        for p in idx:
            key = tuple(self.E[p])
            if key not in rep or self.vol[p] > self.vol[rep[key]]:
                rep[key] = p
        keys = np.array(list(rep.keys()), np.int64)
        orig = np.array(list(rep.values()), np.int64)
        keep = nondominated(keys.astype(float))              # exact: dominated rounded points are useless
        keys, orig = keys[keep], orig[keep]
        m = len(orig)
        kmax = min(self.k, m)
        nsub = sum(math.comb(m, j) for j in range(1, kmax + 1))
        if m > 62 or nsub > self.max_cell_subsets:
            raise CellTooLarge(
                f"a grid cell contains {m} distinct rounded points ({nsub:.3g} subsets to enumerate); "
                f"this is the 2^O((eps^-2 log 1/eps)^d) term of the EPTAS - use a larger eps, a smaller k "
                f"or raise max_cell_subsets")
        R = self.beta ** keys.astype(float)
        H, masks = _cell_volsel(np.ascontiguousarray(R[:, 0]), np.ascontiguousarray(R[:, 1]),
                                np.ascontiguousarray(R[:, 2]), m, kmax)
        subsets = [tuple(int(orig[a]) for a in range(m) if (int(masks[j]) >> a) & 1)
                   for j in range(kmax + 1)]
        self.stats["cells_solved"] += 1
        self.stats["cell_subsets"] += nsub
        self.stats["max_cell_points"] = max(self.stats["max_cell_points"], m)
        res = (H, subsets)
        self.cell_cache[idx] = res
        return res

    def run(self):
        n, k, tau = len(self.P), self.k, self.tau
        best_V, best_S = -1.0, ()
        seen = {}
        for off in product(range(tau), repeat=D):
            self.stats["offsets"] += 1
            rel = self.region - np.array(off, np.int64)
            alive = np.all(np.mod(rel, tau) != 0, axis=1)            # (2)
            cell = np.floor_divide(rel, tau)                          # (3)
            groups = {}
            for p in np.flatnonzero(alive):
                groups.setdefault(tuple(cell[p]), []).append(int(p))
            signature = frozenset(tuple(v) for v in groups.values())
            if signature in seen:
                continue
            seen[signature] = True
            cells = sorted(signature)
            if not cells:
                continue
            self.stats["max_cells"] = max(self.stats["max_cells"], len(cells))
            solved = [self.solve_cell(c) for c in cells]              # (4), (5)
            H = np.zeros((len(cells), k + 1))
            sizes = np.zeros(len(cells), np.int64)
            for i, (h, _) in enumerate(solved):
                H[i, :len(h)] = h
                sizes[i] = len(h) - 1
            V, budgets = _distribute(H, sizes, k)                     # (6)
            if V > best_V:                                            # (7)
                best_V = V
                best_S = tuple(p for (h, subs), b in zip(solved, budgets) for p in subs[int(b)])
        self.stats["distinct_partitions"] = len(seen)
        return best_V, best_S


def solve_eptas(points, k, eps, reference=None, max_cell_subsets=5e7, pad=True):
    """(1-eps)^3-approximation of 3-D hypervolume subset selection.

    points, reference : as in :func:`hssp3d.solve`.
    eps               : the parameter of the paper, 0 < eps <= 1/2.
    pad               : complete the set greedily if the scheme selects fewer than k points
                        (can only increase the hypervolume).

    Returns a Result; ``stats`` contains ``V`` (the value of step 7), the certified
    ``upper_bound`` = V/(1-eps)^2 on the optimum, ``certified_ratio`` = hypervolume/upper_bound
    and ``guaranteed_ratio`` = (1-eps)^3.
    """
    P = np.asarray(points, float)
    if reference is not None:
        P = to_anchored(P, reference)
    if P.ndim != 2 or P.shape[1] != D:
        raise ValueError("points must be an (n,3) array")
    valid = np.flatnonzero(np.all(P > 0, axis=1))
    nd = valid[nondominated(P[valid])]
    Q = P[nd]
    t0 = time.perf_counter()
    k_eff = min(k, len(nd))
    algo = _Eptas(Q, k_eff, eps, max_cell_subsets)
    V, S = algo.run()
    chosen = list(S)
    X, Y, Z = (np.ascontiguousarray(Q[:, c]) for c in range(D))
    padded = 0
    if pad:
        while len(chosen) < k_eff:
            rest = [p for p in range(len(nd)) if p not in chosen]
            gains = [K.hv3d(X, Y, Z, np.array(chosen + [p], np.int64), len(chosen) + 1) for p in rest]
            chosen.append(rest[int(np.argmax(gains))])
            padded += 1
    hv = K.hv3d(X, Y, Z, np.array(chosen, np.int64), len(chosen))
    hv_unpadded = K.hv3d(X, Y, Z, np.array(S, np.int64), len(S))
    stats = dict(algo.stats)
    stats.update(eps=eps, beta=algo.beta, tau=algo.tau, g=algo.g, V=V,
                 upper_bound=V / (1.0 - eps) ** 2, guaranteed_ratio=(1.0 - eps) ** 3,
                 hv_unpadded=hv_unpadded, greedy_padded=padded, time=time.perf_counter() - t0)
    stats["certified_ratio"] = hv / stats["upper_bound"]
    return Result(np.sort(nd[chosen]), hv, V, stats)
