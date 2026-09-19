"""Random samples of the 3-objective DTLZ1 / DTLZ2 Pareto fronts (minimisation)."""
import numpy as np


def dtlz2_front(n, seed=0):
    """n points uniformly distributed on the positive octant of the unit sphere."""
    rng = np.random.default_rng(seed)
    v = np.abs(rng.normal(size=(n, 3)))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def dtlz1_front(n, seed=0):
    """n points uniformly distributed on the simplex f1 + f2 + f3 = 0.5."""
    rng = np.random.default_rng(seed)
    return 0.5 * rng.dirichlet(np.ones(3), size=n)


def convex_dtlz2_front(n, seed=0):
    """Convex variant (DTLZ2 front with every objective raised to the 4th power)."""
    return dtlz2_front(n, seed) ** 4


FRONTS = {
    "DTLZ1": (dtlz1_front, np.array([0.55, 0.55, 0.55])),
    "DTLZ2": (dtlz2_front, np.array([1.1, 1.1, 1.1])),
    "cDTLZ2": (convex_dtlz2_front, np.array([1.1, 1.1, 1.1])),
}


def log_simplex_points(n, decades=40.0, seed=0, spread=1.0):
    """Anchored-box points (maximisation) 10^u with u uniform on the part of the plane
    u1 + u2 + u3 = c inside [-decades, decades]^3, with c uniform in [-spread, spread].

    The box volumes 10^c differ by up to 2*spread orders of magnitude, and the coordinates
    span many orders of magnitude - the regime in which the exponential grid of the EPTAS
    splits the input into many small cells.  (For spread > 0 some points may be dominated.)"""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        u = rng.uniform(-decades, decades, size=2)
        w = -u.sum() + rng.uniform(-spread, spread)
        if abs(w) <= decades:
            out.append([u[0], u[1], w])
    return 10.0 ** np.array(out)
