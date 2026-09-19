"""Draw the canonical triangulation T(Q*) of an optimal subset (figure of the report)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hssp3d import FRONTS, solve_brute_force, to_anchored  # noqa: E402
from hssp3d.solver import SeparatorParams, _Solver, nondominated  # noqa: E402

front, n, k, seed = "DTLZ2", 15, 5, 0
gen, ref = FRONTS[front]
P = to_anchored(gen(n, seed), ref)
opt = solve_brute_force(gen(n, seed), k, ref).subset
s = _Solver(P[nondominated(P)], k, SeparatorParams())
tri, owner, F = s._build(np.asarray(opt, np.int64))
W = s.W

fig, ax = plt.subplots(figsize=(5.2, 5.2))
zs = s.Z[opt]
for t in range(F):
    xy = np.array([(s.gx[v // W], s.gy[v % W]) for v in tri[t]])
    if owner[t] >= 0:
        shade = 0.95 - 0.5 * (s.Z[owner[t]] - zs.min()) / (zs.max() - zs.min() + 1e-12)
        ax.fill(xy[:, 0], xy[:, 1], color=str(shade), lw=0)
    for e in range(3):
        a, b = xy[e], xy[(e + 1) % 3]
        axis_parallel = a[0] == b[0] or a[1] == b[1]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="k" if axis_parallel else "tab:red",
                lw=1.1 if axis_parallel else 0.45, zorder=2 if axis_parallel else 1)
rest = [p for p in range(s.n) if p not in set(opt)]
ax.plot(s.X[rest], s.Y[rest], "o", ms=3.5, mfc="white", mec="tab:blue", zorder=3,
        label="unselected points")
ax.plot(s.X[opt], s.Y[opt], "o", ms=4.5, color="tab:blue", zorder=4, label=r"$v_q,\ q\in Q^*$")
ax.set_aspect("equal")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__), "..", "report", "fig_triangulation.pdf"))
print("triangles", F, "subset", opt)
