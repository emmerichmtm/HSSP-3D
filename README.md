# hssp3d — 3-D hypervolume subset selection in n^O(√k) time

Python/Numba implementation of the exact algorithm of Section 4 of

> K. Bringmann, S. Cabello, M. T. M. Emmerich.
> *Maximum Volume Subset Selection for Anchored Boxes.*
> SoCG 2017, [arXiv:1803.00849](https://arxiv.org/abs/1803.00849).

**Problem (HSSP / Volume Selection).** Given *n* points in R³ and *k* ≤ *n*, select *k*
points that maximise the volume of the union of the boxes anchored at the origin
(equivalently: the hypervolume indicator w.r.t. a reference point). The problem is NP-hard
for *d* = 3; all earlier exact algorithms enumerate Ω(C(n,k)) subsets. The paper breaks this
bound with a dynamic programme over planar separators of the *xy*-projection of the optimal
solution, running in n^O(√k) time. This repository implements that dynamic programme (the
EPTAS of Section 5 is not included).

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

## Experiments

```bash
python experiments/run_comparison.py main        # n = 15, DTLZ1/DTLZ2, k = 3,4,5
python experiments/run_comparison.py k6          # n = 15, k = 6: two recursion levels; not finished after 90 min on a laptop
python experiments/run_comparison.py deep        # forced deep recursion (base threshold 1)
python experiments/run_comparison.py constants   # how small may the constants be?
```

Results (JSON) are written to `experiments/`; `python experiments/make_tables.py` turns them into the LaTeX tables of the report.

## License

MIT
