import cv2
import numpy as np
from scipy.signal import convolve2d
from boundary.boundary import Boundaries2D
from common import NP_FLOAT32
from grid.grid_base import Grid


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def kernel_2d():
    kernel = np.zeros((3, 3), dtype=NP_FLOAT32)
    kernel[1, :] = 1
    kernel[:, 1] = 1
    kernel[1, 1] = 0
    return kernel


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class Grid2D(Grid):
    dim = 2
    two_dim = 2 * dim
    damping_factor = two_dim / (two_dim + 1)

    kernel = kernel_2d()

    def __init__(self, shape: tuple, factor: int):
        super().__init__(shape, factor)

    def set_shapes(self, shape: tuple, factor: int):
        super().set_shapes(shape, factor)
        self.orig_wh = tuple(reversed(self.orig_shape))
        self.grid_valid_wh = tuple(reversed(self.grid_valid_shape))

    def create_boundaries(self):
        return Boundaries2D.init_from_solution(self.solution, self.h)

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
        return self.resize(self.solution_valid, self.orig_wh)

    def enforce_integral_constraints(self, integral_constraints):
        integrals_diff = integral_constraints - self.integrals
        self.solution_valid += self.resize(integrals_diff, self.grid_valid_wh)

    @classmethod
    def resize(cls, arr: np.ndarray, wh: tuple):
        return cv2.resize(arr, wh, interpolation=cv2.INTER_AREA)

    @classmethod
    def downscale(cls, mat: np.ndarray):
        h, w = mat.shape
        return cls.resize(mat, (w // 2, h // 2))



