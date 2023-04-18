from typing import Dict
import numpy as np

from boundary.boundary import Boundary
from boundary.boundaries_base import Boundaries
from boundary.boundary_conditions_2d import Sides, BoundaryConditions2D


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class Boundaries2D(Boundaries):
    def __init__(self, boundaries: dict):
        super().__init__(boundaries)

    @classmethod
    def create_boundaries(cls, solution: np.ndarray, h: float) -> Dict:
        l, r = [Boundary(solution[1:-1, out_ind], solution[1:-1, in_ind], h) for out_ind, in_ind in cls.out_in_indices]
        t, b = [Boundary(solution[out_ind, 1:-1], solution[in_ind, 1:-1], h) for out_ind, in_ind in cls.out_in_indices]
        return {Sides.LEFT: l, Sides.RIGHT: r, Sides.TOP: t, Sides.BOTTOM: b}

    def get_boundary_conditions(self) -> BoundaryConditions2D:
        return BoundaryConditions2D(self.boundary_conditions)

