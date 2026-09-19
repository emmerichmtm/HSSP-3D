"""The separator DP must reproduce the brute-force optimum."""
import numpy as np
import pytest

from hssp3d import FRONTS, SeparatorParams, solve, solve_brute_force, solve_greedy

MODES = {
    "full": SeparatorParams(),
    "separators_only": SeparatorParams(allow_trivial=False),
    "separators_only_noprune": SeparatorParams(allow_trivial=False, prune=False),
    "deep_recursion": SeparatorParams(c_base=0.5, allow_trivial=False),
}


@pytest.mark.parametrize("front", sorted(FRONTS))
@pytest.mark.parametrize("mode", ["full", "separators_only", "separators_only_noprune"])
@pytest.mark.parametrize("k", [2, 3, 4])
def test_matches_brute_force(front, mode, k):
    gen, ref = FRONTS[front]
    F = gen(9, seed=k)
    bf = solve_brute_force(F, k, ref)
    dp = solve(F, k, ref, MODES[mode])
    assert len(dp.subset) == k
    assert dp.hypervolume == pytest.approx(bf.hypervolume, rel=1e-10)
    assert dp.dp_value == pytest.approx(bf.hypervolume, rel=1e-9)
    assert dp.hypervolume >= solve_greedy(F, k, ref).hypervolume - 1e-12


def test_deep_recursion():
    gen, ref = FRONTS["DTLZ2"]
    F = gen(8, seed=5)
    dp = solve(F, 4, ref, MODES["deep_recursion"])
    assert dp.stats["max_depth"] >= 2
    assert dp.hypervolume == pytest.approx(solve_brute_force(F, 4, ref).hypervolume, rel=1e-10)


def test_dominated_and_large_k():
    gen, ref = FRONTS["DTLZ1"]
    F = np.vstack([gen(6, seed=0), [[0.5, 0.5, 0.5]]])        # last point is dominated
    r = solve(F, 10, ref)
    assert list(r.subset) == [0, 1, 2, 3, 4, 5]
    r = solve(F, 3, ref)
    assert 6 not in r.subset
    assert r.hypervolume == pytest.approx(solve_brute_force(F, 3, ref).hypervolume)


def test_proven_constants():
    """With the constants of Miller's theorem optimality is guaranteed."""
    gen, ref = FRONTS["DTLZ1"]
    F = gen(8, seed=1)
    dp = solve(F, 3, ref, SeparatorParams(theory=True))
    assert dp.hypervolume == pytest.approx(solve_brute_force(F, 3, ref).hypervolume, rel=1e-10)
