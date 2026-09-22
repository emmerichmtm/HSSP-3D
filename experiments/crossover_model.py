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
KMAX_LINEAR = 3000       # default constants: every k up to here is examined
KMAX_PROVEN = 600_000    # proven constants: geometric scan up to here


def lbinom(n, k):
    """log10 C(n, k).

    For large n the difference lgamma(n+1) - lgamma(n-k+1) of two numbers of size n ln n
    cancels catastrophically (at n = 1e18 the absolute error is in the thousands), so for
    n >= 1e13 the expansion  ln C(n,k) = k ln n - ln k! - k(k-1)/(2n) + O(k^3/n^2)  is used;
    for k <= 1e6 its truncation error is below 1e-7."""
    if k < 0 or k > n:
        return -math.inf
    k = min(k, n - k)
    if n < 1e13:
        return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / LOG10
    return (k * math.log(n) - math.lgamma(k + 1) - k * (k - 1) / (2.0 * n)) / LOG10


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


def wins(n, k, par):
    return log_separator_dp(n, k, par) < log_brute_force(n, k)


def crossover(n, par):
    """Smallest k <= n/2 at which the DP bound drops below brute force, or None.

    The model is not monotone in k (the ceilings in the separator bounds make it jump), so
    for the default constants every k is tried.  For the proven constants the crossing lies
    at k in the tens of thousands, where that is too slow; a geometric scan in steps of 4 %
    followed by bisection is used, which places the crossing to within about 4 %."""
    if not par.theory:
        for k in range(2, min(n // 2, KMAX_LINEAR) + 1):
            if wins(n, k, par):
                return k
        return None
    prev, k = 1, 2
    while k <= min(n // 2, KMAX_PROVEN):
        if wins(n, k, par):
            lo, hi = prev, k
            while hi - lo > 1:
                mid = (lo + hi) // 2
                lo, hi = (lo, mid) if wins(n, mid, par) else (mid, hi)
            return hi
        prev, k = k, max(k + 1, int(k * 1.04))
    return None


def implied_c(n, k, dp):
    """c such that the DP bound equals n^(c sqrt k) at the crossing."""
    return dp / (math.sqrt(k) * math.log10(n))


def power(x):
    return f"$10^{{{x:,.0f}}}$".replace(",", r"\,")


if __name__ == "__main__":
    modes = [("default", SeparatorParams()), ("proven", SeparatorParams(theory=True))]
    lines = []
    for n in [15, 100, 10 ** 3, 10 ** 4, 10 ** 6, 10 ** 7, 10 ** 9, 10 ** 12, 10 ** 18, 10 ** 30]:
        row = [f"$10^{{{round(math.log10(n))}}}$" if n >= 100 else str(n)]
        for name, par in modes:
            k = crossover(n, par)
            if k is None:
                row += ["--"] * 4
                print(f"n={n:>8.0e} {name:8s}: no crossover for k <= n/2")
            else:
                dp, bf = log_separator_dp(n, k, par), log_brute_force(n, k)
                c = implied_c(n, k, dp)
                row += [f"{k:,}".replace(",", r"\,"), power(dp), power(bf), f"{c:.1f}"]
                print(f"n={n:>8.0e} {name:8s}: k={k:>7,}  DP <= 1e{dp:,.0f}  BF = 1e{bf:,.0f}"
                      f"  c = {c:.1f}")
            sys.stdout.flush()
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
