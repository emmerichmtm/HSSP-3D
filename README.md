# hssp3d — 3-D hypervolume subset selection: the n^O(√k) algorithm and the EPTAS

Python/Numba implementation of the exact algorithm (Section 4) and of the efficient
polynomial-time approximation scheme (Section 5, for d = 3) of

> K. Bringmann, S. Cabello, M. T. M. Emmerich.
> *Maximum Volume Subset Selection for Anchored Boxes.*
> SoCG 2017, [arXiv:1803.00849](https://arxiv.org/abs/1803.00849).

**Problem (HSSP / Volume Selection).** Given *n* points in R³ and *k* ≤ *n*, select *k*
points that maximise the volume of the union of the boxes anchored at the origin
(equivalently: the hypervolume indicator w.r.t. a reference point). The problem is NP-hard
for *d* = 3; all earlier exact algorithms enumerate Ω(C(n,k)) subsets. The paper breaks this
bound with a dynamic programme over planar separators of the *xy*-projection of the optimal
solution, running in n^O(√k) time, and gives an EPTAS based on an exponential grid and the
shifting technique. This repository implements both.

A short technical report is in [`report/report.pdf`](report/report.pdf).

## Install / use

```bash
pip install -e .[test]
pytest
```

```python
import numpy as np
from hssp3d import solve, solve_brute_force, dtlz2_front

F = dtlz2_front(15, seed=0)                  # minimisation objective vectors
r = solve(F, k=5, reference=[1.1, 1.1, 1.1])
print(r.subset, r.hypervolume, r.stats)
assert abs(r.hypervolume - solve_brute_force(F, 5, [1.1, 1.1, 1.1]).hypervolume) < 1e-12
```

`solve(points, k)` without `reference` expects maximisation points spanning boxes anchored at
the origin, as in the paper.

## What is implemented

| Paper (Section 4)                                   | Code                                          |
|-----------------------------------------------------|-----------------------------------------------|
| projection graph G(Q), canonical triangulation T(Q) | `_kernels.build_T`                            |
| Q-compliant domain D, vol(U(S∪Q) ∩ D×R)             | `_kernels.eval_in_domain`                     |
| Φ_comp(S, D, ℓ) by enumeration (Lemma 4.2)          | `_kernels.base_case`                          |
| valid partitions via cycles γ with P_γ∖S = S₀       | `_kernels.enumerate_partitions`               |
| recursion Ψ_comp(S, D, ℓ), memoised over valid tuples | `solver._Solver.psi`                        |
| reference algorithms (brute force, greedy), exact 3-D HV | `_kernels.brute_force`, `greedy`, `hv3d` |

All geometric kernels are `@njit`-compiled; the memoised recursion over the tuples
(S, D, ℓ) is driven from Python.

### The constants in the O(·)

The running time n^O(√k) rests on three constants, exposed as `SeparatorParams`:

* `c_base`: tuples with ℓ ≤ ⌈c_base·√k⌉ are solved by enumeration;
* `c_cycle`: separator cycles have at most ⌈c_cycle·√(|S|+ℓ)⌉ vertices;
* `c_s0`: |S₀| ≤ ⌈c_s0·√(|S|+ℓ)⌉.

For **any** fixed constants the number of tuples and the work per tuple are n^O(√k), and
every value computed is the volume of a feasible selection (a lower bound, Lemma 4.3). The
result is **provably** optimal once the constants reach those of Miller's cycle-separator
theorem (`SeparatorParams(theory=True)`: |γ| ≤ 4·√|V(T)|, |V(T)| ≤ 5(|S|+ℓ)+5) — but these are
so large that for every practical *k* the algorithm then degenerates into enumeration.
The defaults (`c_base=1, c_cycle=2.5, c_s0=1`) are small enough for the separator recursion
to be exercised on 15 points, and reproduced the brute-force optimum in all our experiments.

Two further switches: `allow_trivial=False` restricts the valid partitions to those induced by
a cycle that really cuts the domain (used to test the separator machinery in isolation), and
`prune` turns a branch-and-bound over S₀ on/off (it never changes the optimum found).

**This is a galactic algorithm.** For n = 15 brute force needs milliseconds, the separator DP
seconds to minutes. The purpose of the code is to make the construction of the paper concrete
and testable, not to be a practical HSSP solver.

### Is there an instance where it beats brute force?

Not one that can be run. Even on a favourable structured family ("chain" instances with a
two-point balanced separator at every level) and with the smallest constants that work, n = 16,
k = 6 was stopped unfinished after ten minutes, while brute force needs milliseconds. Because
C(n,k) ≤ 2ⁿ, enumeration is simply not expensive enough for small n; the √k in the exponent
pays off only for n ≫ k ≫ 1. `experiments/crossover_model.py` compares a worst-case operation
bound of the implementation with the exact cost C(n,k)·k² of enumeration (default constants):

| n    | k    | separator DP ≤ | brute force | fewer operations |
|------|------|----------------|-------------|------------------|
| ≤10³ | any  |                |             | brute force      |
| 10⁶  | 50   | 10³¹⁵          | 10²³⁹       | brute force      |
| 10⁶  | 79   | 10³⁶⁰          | 10³⁶¹       | DP (cross-over)  |
| 10⁶  | 200  | 10⁶⁵⁶          | 10⁸³⁰       | DP, factor 10¹⁷⁴ |
| 10⁶  | 1000 | 10¹⁴³³         | 10³⁴³⁸      | DP, factor 10²⁰⁰⁵|

With the proven constants no cross-over occurs for n ≤ 10⁹, k ≤ 3000. The advantage is real
but purely asymptotic.

## The EPTAS (Section 5, d = 3)

```python
from hssp3d import solve_eptas, log_simplex_points

P = log_simplex_points(200, decades=40, seed=0)   # coordinates over 80 orders of magnitude
r = solve_eptas(P, k=20, eps=0.25)                # C(200,20) ~ 1e27 subsets; runs in seconds
print(r.hypervolume, r.stats["upper_bound"], r.stats["certified_ratio"])
```

`solve_eptas` follows the paper step by step: all τ³ grid offsets, deletion of the points in the
thick grid boundaries, partition into cells, rounding down to powers of β = (1−ε)^(−1/3), exact
solution of every cell for all budgets, and a dynamic programme that distributes k over the cells.
With V the value of the best offset and S the returned set,

    μ(S) ≥ (1−ε)·V,    V ≥ (1−ε)²·OPT,    hence    μ(S) ≥ (1−ε)³·OPT,

so `V/(1−ε)²` is a **certified upper bound on the optimum**; it is returned in `stats` together
with the certified ratio μ(S)/UB (it certifies any other solution, e.g. greedy's, as well).

* Against brute force (n = 15, k = 5, DTLZ1/DTLZ2/log-simplex, 45 runs) the guarantee and the
  upper bound hold everywhere; observed ratios are ≥ 0.927 (ε = 0.5), ≥ 0.979 (ε = 0.25) and
  ≥ 0.994 (ε = 0.1).
* On wide-range instances with n = 100…400, k = 10…20 (up to 10³³ subsets) it needs 0.1 s to a
  few minutes — a measured advantage over enumeration. Greedy is faster still and 0.2–3.6 %
  better on these (easy) instances; what the EPTAS adds is the certificate (≈ 0.65 for ε = 0.25).
* The 2^O((ε⁻² log 1/ε)^d) term is the enumeration inside a cell. A cell spans 5.6 / 14 / 46
  orders of magnitude per coordinate for ε = 0.5 / 0.25 / 0.1, so the scheme is practical for
  small n or for inputs spanning many orders of magnitude; otherwise it raises `CellTooLarge`.

## Experiments

```bash
python experiments/run_comparison.py main        # n = 15, DTLZ1/DTLZ2, k = 3,4,5
python experiments/run_comparison.py k6          # n = 15, k = 6: two recursion levels; not finished after 90 min on a laptop
python experiments/run_comparison.py deep        # forced deep recursion (base threshold 1)
python experiments/run_comparison.py constants   # how small may the constants be?
python experiments/crossover_model.py            # operation-count cross-over with brute force
python experiments/run_eptas.py                  # EPTAS vs optimum (n = 15) and on n = 100..400
```

Results (JSON) are written to `experiments/`; `python experiments/make_tables.py` turns them into the LaTeX tables of the report.

## License

MIT
