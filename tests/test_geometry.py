"""Invariants of the canonical triangulation T(Q) of Section 4."""
import itertools

import numpy as np
import pytest

from hssp3d import FRONTS, _kernels as K
from hssp3d.solver import _Solver, SeparatorParams


@pytest.mark.parametrize("front", sorted(FRONTS))
@pytest.mark.parametrize("seed", range(4))
def test_triangulation_invariants(front, seed):
    gen, ref = FRONTS[front]
    n = 8
    s = _Solver(ref - gen(n, seed), 3, SeparatorParams())
    W = s.W
    side = s.gx[-1] - s.gx[0]
    for m in range(n + 1):
        for sel in itertools.combinations(range(n), m):
            sel = np.array(sel, np.int64)
            tri, owner, F = s._build(sel)
            V = len(np.unique(tri[:F]))
            assert F == 2 * V - 6                       # triangulated square (Euler)
            areas = np.array([K.tri_area(tri, t, s.gx, s.gy, W) for t in range(F)])
            assert areas.min() > 0                      # ccw, non-degenerate
            assert abs(areas.sum() - side * side) < 1e-9
            directed = {(tri[t, e], tri[t, (e + 1) % 3]) for t in range(F) for e in range(3)}
            assert len(directed) == 3 * F               # conforming: every interior edge
            assert sum((b, a) not in directed for a, b in directed) == 4   # is shared by 2
            vol = sum(areas[t] * s.Z[owner[t]] for t in range(F) if owner[t] >= 0)
            assert abs(vol - K.hv3d(s.X, s.Y, s.Z, sel, m)) < 1e-10


def test_hv3d_against_inclusion_exclusion():
    rng = np.random.default_rng(1)
    P = rng.random((6, 3)) + 0.05
    X, Y, Z = (np.ascontiguousarray(P[:, c]) for c in range(3))
    for m in range(1, 7):
        for sel in itertools.combinations(range(6), m):
            ie = 0.0
            for r in range(1, m + 1):
                for sub in itertools.combinations(sel, r):
                    ie += (-1) ** (r + 1) * np.prod(P[list(sub)].min(axis=0))
            assert abs(ie - K.hv3d(X, Y, Z, np.array(sel, np.int64), m)) < 1e-12
