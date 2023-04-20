import cv2
import numpy as np
from scipy.signal import convolve2d
from boundary.boundaries_2d import Boundaries2D
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
    kernel = kernel_2d()

    def __init__(self, shape: tuple, factor: int):
        super().__init__(shape, factor)

    @property
    def solution_valid(self):
        return self.solution[1:-1, 1:-1]

    @solution_valid.setter
    def solution_valid(self, value: np.ndarray):
        self.solution[1:-1, 1:-1] = value

    @classmethod
    def get_valid(cls, arr: np.ndarray):
        return arr[1:-1, 1:-1]

    def create_boundaries(self):
        return Boundaries2D.init_from_solution(self.solution, self.h)

    @property
    def neighbors_sum(self):
        return convolve2d(self.solution, self.kernel, mode="valid")

    @classmethod
    def resize_area(cls, arr: np.ndarray, shape: tuple):
        h, w = shape
        return cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)

    @classmethod
    def downscale(cls, mat: np.ndarray):
        h, w = mat.shape
        return cls.resize_area(mat, (h // 2, w // 2))

    @property
    def interpolated_solution(self):
        h, w = self.solution.shape

        self.solution[0, 0] = (self.solution[0, 1] + self.solution[1, 0]) / 2
        self.solution[h - 1, 0] = (self.solution[h - 1, 1] + self.solution[h - 2, 0]) / 2
        self.solution[0, w - 1] = (self.solution[0, w - 2] + self.solution[1, w - 1]) / 2
        self.solution[h - 1, w - 1] = (self.solution[h - 1, w - 2] + self.solution[h - 2, w - 1]) / 2

        return cv2.resize(self.solution, (2 * w - 2, 2 * h - 2), interpolation=cv2.INTER_LINEAR)
