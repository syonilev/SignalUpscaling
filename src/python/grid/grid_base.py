from typing import Optional
import numpy as np
from grid.upscaling_params import UpscalingParams
from common import NP_FLOAT32
from abc import ABC, abstractmethod


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class Grid(ABC):
    dim = 0
    two_dim = 2 * dim
    damping_factor = two_dim / (two_dim + 1)

    kernel = None

    def __init__(self, shape: tuple, factor: int):
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

    @abstractmethod
    def create_boundaries(self):
        pass

    @property
    @abstractmethod
    def solution_valid(self):
        pass

    @solution_valid.setter
    @abstractmethod
    def solution_valid(self, value: np.ndarray):
        pass

    def set_initial_guess(self, initial_guess: Optional[np.ndarray] = None):
        self.solution_valid = 0 if initial_guess is None else initial_guess

    @property
    @abstractmethod
    def neighbors_sum(self):
        pass

    @abstractmethod
    def smooth(self, f: np.ndarray):
        pass

    @property
    @abstractmethod
    def integrals(self):
        pass

    @abstractmethod
    def enforce_integral_constraints(self, integral_constraints):
        pass

    def relaxation(self, params: UpscalingParams):
        self.boundaries.set_boundary_condirions(params.boundary_conditions)

        if self.coarse_grid is not None: # if not at lowest reslution
            self.smooth(params.f)
            self.enforce_integral_constraints(params.integral_constraints)

    @property
    def laplacian(self):
        return (self.neighbors_sum - self.two_dim * self.solution_valid) / self.h_squared

    def get_error_params(self, params: UpscalingParams):
        error_integrals = params.integral_constraints - self.integrals
        error_f = params.f - self.laplacian
        solution_bc = self.boundaries.get_boundary_conditions()
        error_bc = params.boundary_conditions - solution_bc
        return UpscalingParams(error_f, error_integrals, error_bc)

