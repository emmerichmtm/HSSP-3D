"""3-D hypervolume subset selection in n^{O(sqrt k)} time (Bringmann, Cabello, Emmerich)."""
from .solver import (Result, SeparatorParams, solve, solve_brute_force, solve_greedy,
                     nondominated, to_anchored)
from .benchmarks import FRONTS, dtlz1_front, dtlz2_front, convex_dtlz2_front

__all__ = ["Result", "SeparatorParams", "solve", "solve_brute_force", "solve_greedy",
           "nondominated", "to_anchored", "FRONTS", "dtlz1_front", "dtlz2_front",
           "convex_dtlz2_front"]
__version__ = "0.1.0"
