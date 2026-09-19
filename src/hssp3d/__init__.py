"""3-D hypervolume subset selection in n^{O(sqrt k)} time (Bringmann, Cabello, Emmerich)."""
from .solver import (Result, SeparatorParams, solve, solve_brute_force, solve_greedy,
                     nondominated, to_anchored)
from .eptas import CellTooLarge, grid_constants, solve_eptas
from .benchmarks import (FRONTS, dtlz1_front, dtlz2_front, convex_dtlz2_front,
                         log_simplex_points)

__all__ = ["Result", "SeparatorParams", "solve", "solve_brute_force", "solve_greedy",
           "nondominated", "to_anchored", "FRONTS", "dtlz1_front", "dtlz2_front",
           "convex_dtlz2_front", "log_simplex_points", "solve_eptas", "grid_constants",
           "CellTooLarge"]
__version__ = "0.2.0"
