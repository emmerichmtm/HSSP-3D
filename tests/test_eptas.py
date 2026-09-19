"""EPTAS of Section 5: grid constants, rounding, and the proven guarantees."""
import math

import numpy as np
import pytest

from hssp3d import (FRONTS, CellTooLarge, grid_constants, log_simplex_points, solve_brute_force,
                    solve_eptas)
from hssp3d.eptas import rounded_exponents


@pytest.mark.parametrize("eps", [0.5, 0.3, 0.25, 0.1])
def test_grid_constants(eps):
    beta, tau, g = grid_constants(eps)
    assert beta == pytest.approx((1 - eps) ** (-1 / 3))
    assert tau > 3 / eps >= tau - 1                      # smallest integer larger than d/eps
    assert beta ** g > 3 / eps >= beta ** (g - 1) * (1 - 1e-12)   # smallest power larger than d/eps


def test_invalid_eps():
    with pytest.raises(ValueError):
        grid_constants(0.7)


def test_rounded_exponents():
    beta = grid_constants(0.25)[0]
    rng = np.random.default_rng(0)
    P = 10.0 ** rng.uniform(-30, 30, size=(500, 3))
    P[:50] = beta ** rng.integers(-200, 200, size=(50, 3)).astype(float)    # exact powers
    E = rounded_exponents(P, beta)
    assert np.all(beta ** E.astype(float) <= P)
    assert np.all(beta ** (E + 1.0) > P)


def check_guarantees(F, k, eps, ref=None):
    bf = solve_brute_force(F, k, ref)
    r = solve_eptas(F, k, eps, ref)
    s = r.stats
    assert len(r.subset) == k
    assert r.hypervolume <= bf.hypervolume * (1 + 1e-12)
    assert r.hypervolume >= (1 - eps) ** 3 * bf.hypervolume           # Theorem 5.1
    assert s["upper_bound"] >= bf.hypervolume * (1 - 1e-12)           # Lemmas 5.2, 5.3, 5.5
    assert s["hv_unpadded"] >= (1 - eps) * s["V"] * (1 - 1e-12)       # Lemma 5.4
    return r


@pytest.mark.parametrize("front", sorted(FRONTS))
@pytest.mark.parametrize("eps", [0.5, 0.25])
@pytest.mark.parametrize("k", [3, 5])
def test_guarantee_on_dtlz(front, eps, k):
    gen, ref = FRONTS[front]
    check_guarantees(gen(12, seed=k), k, eps, ref)


@pytest.mark.parametrize("eps", [0.5, 0.25])
@pytest.mark.parametrize("seed", range(3))
def test_guarantee_with_many_cells(eps, seed):
    P = log_simplex_points(14, decades=25, seed=seed)
    r = check_guarantees(P, 4, eps)
    assert r.stats["max_cells"] > 1


def test_large_instance_is_fast_and_certified():
    P = log_simplex_points(150, decades=40, seed=1)
    r = solve_eptas(P, 15, 0.5)
    s = r.stats
    assert math.comb(150, 15) > 1e19                                   # far beyond enumeration
    assert s["time"] < 60
    assert s["hv_unpadded"] >= 0.5 * s["V"] * (1 - 1e-12)
    assert s["certified_ratio"] >= s["guaranteed_ratio"] * (1 - 1e-12)


def test_cell_too_large():
    gen, ref = FRONTS["DTLZ2"]
    with pytest.raises(CellTooLarge):
        solve_eptas(gen(40, seed=0), 20, 0.5, ref, max_cell_subsets=1e5)
