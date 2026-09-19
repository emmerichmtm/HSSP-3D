"""Experiments with the EPTAS (Section 5 of the paper, d = 3).

    python experiments/run_eptas.py

 small : n = 15, k = 5, comparison with the brute-force optimum (DTLZ1, DTLZ2 and a
         log-simplex instance whose coordinates span 20 orders of magnitude).
 large : log-simplex instances with up to 400 points, k = 20, far beyond enumeration;
         comparison with greedy and with the certified upper bound.

Writes experiments/results_eptas.json (tables: experiments/make_tables.py).
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hssp3d import (FRONTS, CellTooLarge, log_simplex_points, nondominated, solve_brute_force,  # noqa: E402
                    solve_eptas, solve_greedy)

ROOT = os.path.join(os.path.dirname(__file__), "..")
KEEP = ["V", "upper_bound", "certified_ratio", "guaranteed_ratio", "hv_unpadded", "greedy_padded",
        "offsets", "distinct_partitions", "cells_solved", "cell_subsets", "max_cell_points",
        "max_cells", "tau", "g", "time"]


def instance(family, n, seed):
    if family in FRONTS:
        gen, ref = FRONTS[family]
        return ref - gen(n, seed)
    decades = float(family.split("-")[1])
    return log_simplex_points(n, decades, seed)


def small():
    rows = []
    solve_eptas(instance("DTLZ2", 8, 0), 3, 0.5)                     # JIT warm-up
    for family in ["DTLZ1", "DTLZ2", "logsimplex-10"]:
        for seed in range(5):
            P = instance(family, 15, seed)
            bf = solve_brute_force(P, 5)
            gr = solve_greedy(P, 5)
            for eps in [0.5, 0.25, 0.1]:
                r = solve_eptas(P, 5, eps)
                row = dict(exp="small", family=family, n=15, k=5, seed=seed, eps=eps,
                           hv=r.hypervolume, hv_opt=bf.hypervolume, hv_greedy=gr.hypervolume,
                           t_bf=bf.stats["time"])
                row.update({a: r.stats[a] for a in KEEP})
                rows.append(row)
                print(json.dumps(row), flush=True)
    return rows


def large():
    rows = []
    for n, k in [(100, 10), (200, 20), (400, 20)]:
        for seed in range(3):
            P = instance("logsimplex-40", n, seed)
            nd = len(nondominated(P))
            gr = solve_greedy(P, k)
            for eps in [0.5, 0.25]:
                row = dict(exp="large", family="logsimplex-40", n=n, n_nd=nd, k=k, seed=seed, eps=eps,
                           hv_greedy=gr.hypervolume, t_greedy=gr.stats["time"],
                           log10_subsets=math.log10(math.comb(nd, k)))
                try:
                    r = solve_eptas(P, k, eps)
                    row.update(hv=r.hypervolume, failed=False)
                    row.update({a: r.stats[a] for a in KEEP})
                except CellTooLarge as err:
                    row.update(failed=True, error=str(err))
                rows.append(row)
                print(json.dumps(row), flush=True)
    return rows


if __name__ == "__main__":
    rows = small() + large()
    with open(os.path.join(ROOT, "experiments", "results_eptas.json"), "w") as f:
        json.dump(rows, f, indent=1)
    ok = [r for r in rows if r["exp"] == "small"]
    print("small: guarantee holds on all:",
          all(r["hv"] >= r["guaranteed_ratio"] * r["hv_opt"] for r in ok),
          " min ratio: %.4f" % min(r["hv"] / r["hv_opt"] for r in ok),
          " upper bound valid on all:", all(r["upper_bound"] >= r["hv_opt"] * (1 - 1e-12) for r in ok))
