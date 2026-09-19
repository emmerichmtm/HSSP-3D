"""Where does the separator DP overtake brute force?  An operation-count model.

No instance on which the n^{O(sqrt k)} algorithm beats enumeration can actually be
run (see the report), so this script answers the question analytically.  It evaluates a
worst-case *upper bound* on the number of elementary operations of the implementation,

    sum over recursion levels i of   (#tuples at level i) x (work per tuple),

and compares it with the *exact* cost  C(n,k) * k^2  of brute force.  Along the
worst-case chain of the recursion

    l_0 = k, |S_0| = 0,   s_i = s_max(|S_i| + l_i),   L_i = cycle_len(|S_i| + l_i),
    |S_{i+1}| = |S_i| + s_i,   l_{i+1} = floor(2 l_i / 3),   until l_i <= base,

the model uses, with V_i = 5(|S_i|+s_i)+5 vertices and F_i = 2V_i-6 triangles of T(S+S_0):

    tuples_0 = 1,
    tuples_{i+1} <= tuples_i * C(n, <=s_i) * V_i^{L_i} * F_i * k
                    (choice of S_0, of the cycle, of the component, of the budget),
                    capped by  C(n,|S_{i+1}|) * 2^{F} * k  (all S, all domains, all budgets),
    work of a recursive tuple <= C(n, <=s_i) * V_i^{L_i} * F_i * k^2,
    work of a base tuple      <= C(n, <=l_i) * (|S_i| + l_i)^2.

All quantities are handled as log10.  Usage:  python experiments/crossover_model.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hssp3d import SeparatorParams  # noqa: E402

LOG10 = math.log(10.0)
KMAX = 3000          # largest k examined


def lbinom(n, k):
    if k < 0 or k > n:
        return -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / LOG10


def ladd(a, b):
    if a < b:
        a, b = b, a
    return a if b == -math.inf else a + math.log10(1.0 + 10.0 ** (b - a))


def lbinom_upto(n, k):
    out = -math.inf
    for j in range(0, min(n, k) + 1):
        out = ladd(out, lbinom(n, j))
    return out


def log_brute_force(n, k):
    return lbinom(n, k) + 2 * math.log10(k)


def log_separator_dp(n, k, par):
    base = par.base(k)
    ell, m = k, 0
    ltuples = 0.0
    total = -math.inf
    while ell > base:
        s = min(ell, par.s0_max(m + ell))
        L = par.cycle_len(m + ell)
        V = 5 * (m + s) + 5
        F = 2 * V - 6
        branch = lbinom_upto(n, s) + L * math.log10(V)
        total = ladd(total, ltuples + branch + math.log10(F) + 2 * math.log10(k))
        m2 = m + s
        reach = ltuples + branch + math.log10(F) + math.log10(k)
        every = lbinom(n, m2) + (10 * m2 + 4) * math.log10(2.0) + math.log10(k)
        ltuples = min(reach, every)
        m, ell = m2, (2 * ell) // 3
    total = ladd(total, ltuples + lbinom_upto(n, ell) + 2 * math.log10(m + ell))
    return total


def crossover(n, par):
    for k in range(2, min(n // 2, KMAX) + 1):
        if log_separator_dp(n, k, par) < log_brute_force(n, k):
            return k
    return None


if __name__ == "__main__":
    modes = [("default", SeparatorParams()), ("proven", SeparatorParams(theory=True))]
    lines = []
    for n in [15, 100, 10 ** 3, 10 ** 4, 10 ** 6, 10 ** 9]:
        row = [f"$10^{{{round(math.log10(n))}}}$" if n >= 100 else str(n)]
        for name, par in modes:
            k = crossover(n, par)
            if k is None:
                row += ["--", "--", "--"]
                print(f"n={n:>10} {name:8s}: brute force is never beaten for k <= min(n/2, KMAX)")
            else:
                dp, bf = log_separator_dp(n, k, par), log_brute_force(n, k)
                row += [str(k), f"$10^{{{dp:.0f}}}$", f"$10^{{{bf:.0f}}}$"]
                print(f"n={n:>10} {name:8s}: first k with DP bound < brute force: k={k:>4}  "
                      f"DP <= 1e{dp:.0f} ops, BF = 1e{bf:.0f} ops")
        lines.append(" & ".join(row) + " \\\\")
    with open(os.path.join(os.path.dirname(__file__), "..", "report", "table_crossover.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    lines = []
    par = SeparatorParams()
    for n in [10 ** 6, 10 ** 9]:
        for k in [50, 100, 200, 500, 1000]:
            dp, bf = log_separator_dp(n, k, par), log_brute_force(n, k)
            lines.append(f"$10^{{{round(math.log10(n))}}}$ & {k} & $10^{{{dp:.0f}}}$ & $10^{{{bf:.0f}}}$ & "
                         f"{'DP' if dp < bf else 'brute force'} & $10^{{{abs(bf - dp):.0f}}}$ \\\\")
            print(f"n=1e{round(math.log10(n))} k={k:>5}: DP <= 1e{dp:.0f}  BF = 1e{bf:.0f}")
    with open(os.path.join(os.path.dirname(__file__), "..", "report", "table_advantage.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("n=15, k=5:  DP bound 1e%.1f   brute force 1e%.1f" %
          (log_separator_dp(15, 5, SeparatorParams()), log_brute_force(15, 5)))
