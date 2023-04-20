from typing import Dict
import numpy as np
from abc import ABC, abstractmethod

from boundary.boundary_conditions import BoundaryConditions


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class Boundaries(ABC):
    out_in_indices = [(0, 1), (-1, -2)]

    def __init__(self, boundaries: dict):
        self.boundaries = boundaries

    @classmethod
    def init_from_solution(cls, solution: np.ndarray, h: float) -> 'Boundaries':
        boundaries = cls.create_boundaries(solution, h)
        return cls(boundaries)

    def set_boundary_conditions(self, boundary_conditions: BoundaryConditions):
        for side, boundary in self.boundaries.items():
            boundary.set_derivative(boundary_conditions.get_side(side))

    @property
    def boundary_conditions(self) -> Dict:
        return {side: boundary.get_derivative() for side, boundary in self.boundaries.items()}

    @classmethod
    @abstractmethod
    def create_boundaries(cls, solution: np.ndarray, h: float) -> Dict:
        pass

    @abstractmethod
    def get_boundary_conditions(self) -> BoundaryConditions:
        pass

