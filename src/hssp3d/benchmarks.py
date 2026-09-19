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
