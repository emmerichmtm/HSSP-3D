"""Compare the separator DP with brute force on random DTLZ fronts.

    python experiments/run_comparison.py main      # n = 15, k = 3, 4, 5
    python experiments/run_comparison.py k6        # n = 15, k = 6 (two recursion levels; did not finish within 90 min)
    python experiments/run_comparison.py deep      # forced deep recursion, n = 10
    python experiments/run_comparison.py constants # effect of the separator constant

Each experiment writes  experiments/results_<name>.json ;  afterwards
experiments/make_tables.py  turns the results into the LaTeX tables of the report.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hssp3d import FRONTS, SeparatorParams, solve, solve_brute_force, solve_greedy  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def run(front, n, k, seed, params):
    gen, ref = FRONTS[front]
    F = gen(n, seed)
    bf = solve_brute_force(F, k, ref)
    gr = solve_greedy(F, k, ref)
    dp = solve(F, k, ref, params)
    rec = dict(front=front, n=n, k=k, seed=seed, hv_bf=bf.hypervolume, hv_dp=dp.hypervolume,
               hv_greedy=gr.hypervolume, dp_value=dp.dp_value,
               match=bool(abs(bf.hypervolume - dp.hypervolume) <= 1e-10 * bf.hypervolume),
               same_subset=bool(np.array_equal(bf.subset, dp.subset)),
               subsets_bf=bf.stats["subsets"], t_bf=bf.stats["time"], t_dp=dp.stats["time"],
               subset=[int(i) for i in dp.subset])
    rec.update({a: b for a, b in dp.stats.items() if a != "time"})
    return rec


def warmup():
    gen, ref = FRONTS["DTLZ2"]
    F = gen(7, 0)
    solve_brute_force(F, 3, ref)
    solve_greedy(F, 3, ref)
    solve(F, 3, ref, SeparatorParams(c_base=0.5))
    solve(F, 3, ref, SeparatorParams(c_base=0.5, allow_trivial=False, prune=False))


def experiment_main(ks=(3, 4, 5)):
    modes = [("full", SeparatorParams()),
             ("sep", SeparatorParams(allow_trivial=False))]
    rows = []
    for front in ["DTLZ1", "DTLZ2"]:
        for k in ks:
            for seed in ([0, 1, 2] if k < 6 else [0]):
                for mname, par in modes:
                    r = run(front, 15, k, seed, par)
                    r["mode"] = mname
                    rows.append(r)
                    print(json.dumps(r), flush=True)
    return rows


def experiment_deep():
    rows = []
    par = SeparatorParams(c_base=0.5, allow_trivial=False, prune=False)
    for front in ["DTLZ1", "DTLZ2"]:
        for k in [3, 4]:
            for seed in [0, 1, 2]:
                r = run(front, 10, k, seed, par)
                r["mode"] = "sep/noprune"
                rows.append(r)
                print(json.dumps(r), flush=True)
    return rows


def experiment_constants():
    """Separators only, no pruning: how large must the constants be for exactness?"""
    rows = []
    for c_s0, c_cycle in [(0.4, 1.0), (0.4, 2.0), (0.4, 3.0), (0.8, 1.0), (0.8, 2.0), (0.8, 3.0),
                          (1.0, 1.0), (1.0, 2.5)]:
        par = SeparatorParams(c_cycle=c_cycle, c_s0=c_s0, allow_trivial=False, prune=False)
        for front in ["DTLZ1", "DTLZ2"]:
            for seed in range(3):
                r = run(front, 12, 5, seed, par)
                r["mode"] = "sep/noprune"
                r["c_cycle"] = c_cycle
                r["c_s0"] = c_s0
                r["L"] = par.cycle_len(5)
                r["s0max"] = par.s0_max(5)
                rows.append(r)
                print(json.dumps(r), flush=True)
    return rows


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "main"
    warmup()
    t0 = time.time()
    if name == "k6":
        rows = experiment_main(ks=(6,))
    else:
        rows = {"main": experiment_main, "deep": experiment_deep,
                "constants": experiment_constants}[name]()
    with open(os.path.join(ROOT, "experiments", f"results_{name}.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(f"done in {time.time() - t0:.0f}s; all match: {all(r['match'] for r in rows)}")
