"""Turn experiments/results_*.json into the LaTeX table rows used by the report."""
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load(name):
    path = os.path.join(ROOT, "experiments", f"results_{name}.json")
    return json.load(open(path)) if os.path.exists(path) else []


def num(x):
    return f"{int(round(x)):,}".replace(",", "\\,")


def write(name, lines):
    with open(os.path.join(ROOT, "report", f"table_{name}.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def gap(r):
    return (r["hv_bf"] - r["hv_dp"]) / r["hv_bf"]


def values_table(rows):
    """One line per instance: optimal value, DP values, greedy value."""
    lines = []
    keys = sorted({(r["front"], r["k"], r["seed"]) for r in rows})
    for front, k, seed in keys:
        rs = {r["mode"]: r for r in rows if (r["front"], r["k"], r["seed"]) == (front, k, seed)}
        any_r = next(iter(rs.values()))
        cells = [front, str(k), str(seed), f"{any_r['hv_bf']:.8f}"]
        for mode in ["full", "sep"]:
            r = rs.get(mode)
            cells.append("--" if r is None else f"{r['hv_dp']:.8f}")
        ok = all(r["match"] for r in rs.values())
        cells.append(f"{any_r['hv_greedy']:.8f}")
        cells.append("yes" if ok else "\\textbf{no}")
        lines.append(" & ".join(cells) + " \\\\")
    return lines


def effort_table(rows):
    """Aggregated over the seeds: matches and work counters."""
    lines = []
    keys = sorted({(r["front"], r["k"], r["mode"]) for r in rows})
    for front, k, mode in keys:
        rs = [r for r in rows if (r["front"], r["k"], r["mode"]) == (front, k, mode)]
        lines.append(" & ".join([
            front, str(k), mode, f"{sum(r['match'] for r in rs)}/{len(rs)}",
            str(max(r["max_depth"] for r in rs)), str(max(r["max_S"] for r in rs)),
            num(np.mean([r["tuples"] for r in rs])),
            num(np.mean([r["partitions"] for r in rs])),
            num(np.mean([r["base_subsets"] for r in rs])),
            num(rs[0]["subsets_bf"]),
            f"{1000 * np.mean([r['t_bf'] for r in rs]):.1f}",
            f"{np.mean([r['t_dp'] for r in rs]):.1f}"]) + " \\\\")
    return lines


def constants_table(rows):
    lines = []
    for cfg in sorted({(r["c_s0"], r["c_cycle"]) for r in rows}):
        rs = [r for r in rows if (r["c_s0"], r["c_cycle"]) == cfg]
        lines.append(" & ".join([
            f"{cfg[0]:.1f}", f"{cfg[1]:.1f}", str(rs[0]["s0max"]), str(rs[0]["L"]),
            f"{sum(r['match'] for r in rs)}/{len(rs)}",
            f"{100 * max(gap(r) for r in rs):.2f}",
            str(max(r.get("greedy_padded", 0) for r in rs)),
            num(np.mean([r["partitions"] for r in rs])),
            f"{np.mean([r['t_dp'] for r in rs]):.1f}"]) + " \\\\")
    return lines


def eptas_small_table(rows):
    lines = []
    rows = [r for r in rows if r["exp"] == "small"]
    for family in ["DTLZ1", "DTLZ2", "logsimplex-10"]:
        for eps in sorted({r["eps"] for r in rows}, reverse=True):
            rs = [r for r in rows if r["family"] == family and r["eps"] == eps]
            ratio = [r["hv"] / r["hv_opt"] for r in rs]
            lines.append(" & ".join([
                family.replace("logsimplex-10", "log-simplex"), f"{eps:g}", f"{rs[0]['guaranteed_ratio']:.3f}",
                f"{min(ratio):.4f}", f"{np.mean(ratio):.4f}",
                f"{min(r['hv_greedy'] / r['hv_opt'] for r in rs):.4f}",
                f"{min(r['upper_bound'] / r['hv_opt'] for r in rs):.3f}",
                f"{np.mean([r['certified_ratio'] for r in rs]):.3f}",
                num(rs[0]["offsets"]), num(np.mean([r["distinct_partitions"] for r in rs])),
                str(max(r["max_cell_points"] for r in rs)),
                f"{np.mean([r['time'] for r in rs]):.2f}"]) + " \\\\")
    return lines


def eptas_large_table(rows):
    lines = []
    rows = [r for r in rows if r["exp"] == "large"]
    for n, k, eps in sorted({(r["n"], r["k"], -r["eps"]) for r in rows}):
        eps = -eps
        rs = [r for r in rows if (r["n"], r["k"], r["eps"]) == (n, k, eps)]
        ok = [r for r in rs if not r["failed"]]
        cells = [str(n), str(k), f"$10^{{{np.mean([r['log10_subsets'] for r in rs]):.0f}}}$", f"{eps:g}",
                 f"{len(ok)}/{len(rs)}"]
        if ok:
            cells += [num(np.mean([r["max_cells"] for r in ok])), str(max(r["max_cell_points"] for r in ok)),
                      f"{np.mean([r['hv'] / r['hv_greedy'] for r in ok]):.4f}",
                      f"{rs[0]['guaranteed_ratio']:.3f}" if "guaranteed_ratio" in rs[0] else "",
                      f"{np.mean([r['certified_ratio'] for r in ok]):.3f}",
                      f"{np.mean([r['time'] for r in ok]):.1f}",
                      f"{1000 * np.mean([r['t_greedy'] for r in ok]):.0f}"]
        else:
            cells += ["--"] * 7
        lines.append(" & ".join(cells) + " \\\\")
    return lines


if __name__ == "__main__":
    main = load("main") + load("k6")
    write("values", values_table(main))
    write("effort", effort_table(main))
    write("deep", effort_table(load("deep")))
    write("constants", constants_table(load("constants")))
    inst = {(r["front"], r["k"], r["seed"]): r for r in main}
    n_greedy_sub = sum(r["hv_greedy"] < r["hv_bf"] * (1 - 1e-10) for r in inst.values())
    n_same = sum(r["same_subset"] for r in main)
    ks = ",".join(str(k) for k in sorted({r["k"] for r in main}))
    verdict = "equals" if all(r["match"] for r in main) else "DOES NOT ALWAYS EQUAL"
    summary = (f"In all {len(main)} runs ({len(inst)} instances with $k\\in\\{{{ks}\\}}$, two modes each) "
               f"the dynamic programme returned a subset whose hypervolume {verdict} the "
               f"brute-force optimum; in {n_same} of the runs it is the identical subset.  "
               f"The plain greedy algorithm is sub-optimal on {n_greedy_sub} of the {len(inst)} "
               f"instances.")
    with open(os.path.join(ROOT, "report", "summary_main.tex"), "w") as f:
        f.write(summary + "\n")
    eptas = load("eptas")
    if eptas:
        write("eptas_small", eptas_small_table(eptas))
        write("eptas_large", eptas_large_table(eptas))
    allrows = main + load("deep")
    print("instances:", len(allrows), "all match:", all(r["match"] for r in allrows))
