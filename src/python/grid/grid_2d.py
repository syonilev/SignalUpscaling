import cv2
from typing import Optional
import numpy as np
from scipy.signal import convolve2d
from boundary import Boundaries2D
from grid.upscaling_params import UpscalingParams
from common import NP_FLOAT32


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class Grid2D:
    dim = 2
    two_dim = 2 * dim
    damping_factor = two_dim / (two_dim + 1)

    # kernel
    kernel = np.zeros((3, 3), dtype=NP_FLOAT32)
    kernel[1, :] = 1
    kernel[:, 1] = 1
    kernel[1, 1] = 0

    def __init__(self, wh_orig: tuple, factor: int):
        self.factor = factor
        self.h = 1 / factor
        self.h_squared = self.h ** 2
        self.coarse_grid = None

        width, height = wh_orig
        self.wh_orig = wh_orig
        self.wh_grid_valid = (width * factor, height * factor)
        self.solution = np.zeros((height * factor + 2, width * factor + 2), dtype=NP_FLOAT32)
        self.boundaries = Boundaries2D.init_from_solution(self.solution, self.h)

    def set_initial_guess(self, initial_guess: Optional[np.ndarray] = None):
        self.solution_valid = 0 if initial_guess is None else initial_guess

    @property
    def solution_valid(self):
        return self.solution[1:-1, 1:-1]

    @solution_valid.setter
    def solution_valid(self, value: np.ndarray):
        self.solution[1:-1, 1:-1] = value

    @property
    def neighbors_sum(self):
        return convolve2d(self.solution, self.kernel, mode="valid")

    def smooth(self, f: np.ndarray):
        w = self.damping_factor

        sol_new = (self.neighbors_sum - self.h_squared * f) / self.two_dim
        self.solution_valid = (1 - w) * self.solution_valid + w * sol_new

    @property
    def integrals(self):
        return self.resize(self.solution_valid, self.wh_orig)

    def enforce_integral_constraints(self, integral_constraints):
        integrals_diff = integral_constraints - self.integrals
        self.solution_valid += self.resize(integrals_diff, self.wh_grid_valid)

    def relaxation(self, params: UpscalingParams):
        self.boundaries.set_boundary_condirions(params.boundary_conditions)

        if self.coarse_grid is not None: # if not at lowest reslution
            self.smooth(params.f)
            self.enforce_integral_constraints(params.integral_constraints)

    @classmethod
    def resize(cls, arr: np.ndarray, wh: tuple):
        return cv2.resize(arr, wh, interpolation=cv2.INTER_AREA)

    @classmethod
    def downscale(cls, mat: np.ndarray):
        h, w = mat.shape
        return cls.resize(mat, (w // 2, h // 2))

    @property
    def laplacian(self):
        return (self.neighbors_sum - self.two_dim * self.solution_valid) / self.h_squared

    def get_error_params(self, params: UpscalingParams):
        error_integrals = params.integral_constraints - self.integrals
        error_f = params.f - self.laplacian
        solution_bc = self.boundaries.get_boundary_conditions()
        error_bc = params.boundary_conditions - solution_bc
        return UpscalingParams(error_f, error_integrals, error_bc)

