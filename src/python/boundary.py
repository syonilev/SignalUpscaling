import cv2
from typing import Optional, Dict
import numpy as np
from scipy.signal import convolve2d
from abc import ABC, abstractmethod

from boundary_conditions import Sides, BoundaryConditions, BoundaryConditions2D



# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class Boundary:
    def __init__(self, inside: np.ndarray, ghost: np.ndarray, h: float):
        self.inside = inside
        self.ghost = ghost
        self.h = h

    def get_derivative(self):
        return (self.ghost - self.inside) / self.h

    def set_derivative(self, boundary_conditions: np.ndarray):
        self.ghost[:] = boundary_conditions * self.h + self.inside


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class Boundaries(ABC):
    out_in_indices = [(0, 1), (-1, -2)]

    def __init__(self, boundaries: dict):
        self.boundaries = boundaries

    @classmethod
    def init_from_solution(cls, solution: np.ndarray, h: float) -> 'Boundaries':
        boundaries = cls.create_boundaries(solution, h)
        return cls(boundaries)


    @classmethod
    def create_boundaries(cls, solution: np.ndarray, h: float) -> Dict:
        pass

    # @abstractmethod
    # def set_derivative(self, boundary_conditions: BoundaryConditions):
    #     pass

    @abstractmethod
    def get_boundary_conditions(self) -> BoundaryConditions:
        pass


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class Boundaries2D(Boundaries):
    def __init__(self, boundaries: dict):
        super().__init__(boundaries)

    @classmethod
    def create_boundaries(cls, solution: np.ndarray, h: float) -> Dict:
        l, r = [Boundary(solution[1:-1, out_ind], solution[1:-1, in_ind], h) for out_ind, in_ind in cls.out_in_indices]
        t, b = [Boundary(solution[out_ind, 1:-1], solution[in_ind, 1:-1], h) for out_ind, in_ind in cls.out_in_indices]
        return {Sides.LEFT: l, Sides.RIGHT: r, Sides.TOP: t, Sides.BOTTOM: b}

    def set_boundary_condirions(self, boundary_conditions: BoundaryConditions):
        for side, boundary in self.boundaries.items():
            boundary.set_derivative(boundary_conditions.get_side(side))

    def get_boundary_conditions(self) -> BoundaryConditions2D:
        boundary_conditions = {side: boundary.get_derivative() for side, boundary in self.boundaries.items()}
        return BoundaryConditions2D(boundary_conditions)

