import numpy as np
from scipy.signal import convolve
from boundary.boundaries_1d import Boundaries1D
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
    kernel = kernel_1d()

    def __init__(self, shape: tuple, factor: int):
        super().__init__(shape, factor)
        solution_len = self.solution_shape[0]
        x_first, x_last = 0, solution_len - 1
        self.x_solution = np.linspace(x_first, x_last , solution_len)
        self.x_to_evaluate = np.linspace(x_first + 0.25, x_last - 0.25, 2 * solution_len - 2)

    @property
    def solution_valid(self):
        return self.solution[1:-1]

    @solution_valid.setter
    def solution_valid(self, value: np.ndarray):
        self.solution[1:-1] = value

    @classmethod
    def get_valid(cls, arr: np.ndarray):
        return arr[1:-1]

    def create_boundaries(self):
        return Boundaries1D.init_from_solution(self.solution, self.h)

    @property
    def neighbors_sum(self):
        return convolve(self.solution, self.kernel, mode="valid")

    @classmethod
    def resize_area(self, arr: np.ndarray, shape: tuple):
        desired_len = shape[0]
        arr_len = len(arr)

        if arr_len < desired_len:
            factor = desired_len // arr_len
            res = np.repeat(arr, factor)
        else:
            factor = arr_len // desired_len
            res = np.average(arr.reshape(-1, factor), axis=1)
        return res

    @classmethod
    def downscale(cls, mat: np.ndarray):
        desired_len = mat.shape[0] // 2
        return cls.resize_area(mat, (desired_len,))

    @property
    def interpolated_solution(self):
        return np.interp(self.x_to_evaluate, self.x_solution, self.solution)



