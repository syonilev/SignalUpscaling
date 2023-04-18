from typing import Dict
import numpy as np

from boundary.boundary import Boundary
from boundary.boundaries_base import Boundaries
from boundary.boundary_conditions import Sides, BoundaryConditions1D


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class Boundaries1D(Boundaries):
    def __init__(self, boundaries: dict):
        super().__init__(boundaries)

    @classmethod
    def create_boundaries(cls, solution: np.ndarray, h: float) -> Dict:
        l = Boundary(solution[:1], solution[1:2], h)
        r = Boundary(solution[-1:], solution[-1:-2], h)
        return {Sides.LEFT: l, Sides.RIGHT: r}

    def get_boundary_conditions(self) -> BoundaryConditions1D:
        return BoundaryConditions1D(self.boundary_conditions)

