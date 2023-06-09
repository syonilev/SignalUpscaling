from typing import Optional
import numpy as np
from grid.minimization_params import MinimizationParams
from common import NP_FLOAT32
from abc import ABC, abstractmethod


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def damping_factor_formula(dim: int):
    two_dim = 2 * dim
    damping_factor = two_dim / (two_dim + 1)
    return damping_factor


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class Grid(ABC):
    dim = 0
    kernel = None

    def __init__(self, shape: tuple, factor: int):
        self.damping_factor = damping_factor_formula(self.dim)
        self.factor = factor
        self.h = 1 / factor
        self.h_squared = self.h ** 2
        self.coarse_grid = None

        self.set_shapes(shape, factor)
        self.solution = np.zeros(self.solution_shape, dtype=NP_FLOAT32)
        self.boundaries = self.create_boundaries()

    def set_shapes(self, shape: tuple, factor: int):
        self.orig_shape = shape
        self.grid_valid_shape = tuple([n * factor for n in shape])
        self.solution_shape = tuple([n + 2 for n in self.grid_valid_shape])

    def set_initial_guess(self, initial_guess: Optional[np.ndarray] = None):
        self.solution_valid = 0 if initial_guess is None else initial_guess

    def smooth(self, f: np.ndarray):
        w = self.damping_factor

        sol_new = (self.neighbors_sum - self.h_squared * f) / (2 * self.dim)
        self.solution_valid = (1 - w) * self.solution_valid + w * sol_new

    @property
    def integrals(self):
        return self.resize_area(self.solution_valid, self.orig_shape)

    def resize_integrals_to_grid_valid(self, integrals: np.ndarray):
        return self.resize_area(integrals, self.grid_valid_shape)

    def enforce_integral_constraints(self, integral_constraints):
        integrals_diff = integral_constraints - self.integrals
        self.solution_valid += self.resize_integrals_to_grid_valid(integrals_diff)

    def relaxation(self, params: MinimizationParams):
        self.boundaries.set_boundary_conditions(params.boundary_conditions)

        if self.coarse_grid is not None:  # if not at lowest reslution
            self.smooth(params.f)
            self.enforce_integral_constraints(params.integral_constraints)

    @property
    def laplacian(self):
        return (self.neighbors_sum - 2 * self.dim * self.solution_valid) / self.h_squared

    def get_error_params(self, params: MinimizationParams):
        error_integrals = params.integral_constraints - self.integrals
        error_f = params.f - self.laplacian
        solution_bc = self.boundaries.get_boundary_conditions()
        error_bc = params.boundary_conditions - solution_bc
        return MinimizationParams(error_f, error_integrals, error_bc)

    @property
    @abstractmethod
    def solution_valid(self):
        pass

    @solution_valid.setter
    @abstractmethod
    def solution_valid(self, value: np.ndarray):
        pass

    @classmethod
    @abstractmethod
    def get_valid(cls, arr: np.ndarray):
        pass

    @abstractmethod
    def create_boundaries(self):
        pass

    @property
    @abstractmethod
    def neighbors_sum(self):
        pass

    @classmethod
    @abstractmethod
    def resize_area(cls, arr: np.ndarray, shape: tuple):
        pass

    @classmethod
    @abstractmethod
    def downscale(cls, mat: np.ndarray):
        pass

    @property
    @abstractmethod
    def interpolated_solution(self):
        pass

