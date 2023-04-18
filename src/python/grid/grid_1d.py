import numpy as np
from scipy.signal import convolve
from boundary.boundary import Boundaries1D
from common import NP_FLOAT32
from grid.grid_base import Grid


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def kernel_1d():
    kernel = np.ones(3, dtype=NP_FLOAT32)
    kernel[1] = 0
    return kernel


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class Grid1D(Grid):
    dim = 1
    two_dim = 2 * dim
    damping_factor = two_dim / (two_dim + 1)

    kernel = kernel_1d()

    def __init__(self, shape: tuple, factor: int):
        super().__init__(shape, factor)

    def set_shapes(self, shape: tuple, factor: int):
        super().set_shapes(shape, factor)
        self.orig_len = shape[0]
        self.grid_valid_len = self.grid_valid_shape[0]

    def create_boundaries(self):
        return Boundaries1D.init_from_solution(self.solution, self.h)

    @property
    def solution_valid(self):
        return self.solution[1:-1]

    @solution_valid.setter
    def solution_valid(self, value: np.ndarray):
        self.solution[1:-1] = value

    @property
    def neighbors_sum(self):
        return convolve(self.solution, self.kernel, mode="valid")

    @property
    def integrals(self):
        return np.average(self.solution_valid.reshape(-1, self.factor), axis=1)

    def resize_integrals_to_grid_valid(self, integrals: np.ndarray):
        return np.repeat(integrals, self.factor)



