import cv2
import numpy as np
from boundary.boundaries_2d import Boundaries2D
from grid.grid_2d import Grid2D
from grid.grid_base import UpscalingParams
from multigrid.multigrid_base import MultigridSolver


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class MultigridSolver2D(MultigridSolver):
    def __init__(self, shape: tuple, factor: int,
                 pre_relaxation_iters: int = 2,
                 post_relaxation_iters: int = 2):

        super().__init__(shape, factor, pre_relaxation_iters, post_relaxation_iters)
        self.construct_grids()
        self.finsest_params = self.init_zero_params()

    def init_zero_params(self):
        grid = self.finset_grid
        zeros_mat = np.zeros_like(grid.solution)
        boundaries_tmp = Boundaries2D.init_from_solution(zeros_mat, 1)
        f = zeros_mat[1:-1, 1:-1]
        boundary_conditions = boundaries_tmp.get_boundary_conditions()
        return UpscalingParams(f, None, boundary_conditions)

    def construct_grid(self, factor) -> Grid2D:
        return Grid2D(self.shape, factor)

    def construct_grids(self):
        self.grids = []

        factor = 1
        for i in range(self.grids_num):
            grid = self.construct_grid(factor)
            self.grids.append(grid)
            factor *= 2

        for i in range(self.grids_num - 1):
            self.grids[i + 1].coarse_grid = self.grids[i]

    def set_initial_guess(self, integral_constraints: np.ndarray):
        for i in range(self.grids_num - 1):
            self.grids[i].set_initial_guess()

        grid = self.finset_grid
        initial_guess = Grid2D.resize(integral_constraints, grid.grid_valid_wh)
        grid.set_initial_guess(initial_guess)

    def restrict(self, params_h: UpscalingParams):
        f_2h = Grid2D.downscale(params_h.f)
        integral_constraints_2h = params_h.integral_constraints
        boundary_conditions_2h = params_h.boundary_conditions.restrict()
        return UpscalingParams(f_2h, integral_constraints_2h, boundary_conditions_2h)

    def interpolate(self, solution: np.ndarray):
        h, w = solution.shape
        wh_fine = (2 * w - 2, 2 * h - 2)
        return cv2.resize(solution, wh_fine, interpolation=cv2.INTER_LINEAR)

    def v_cycle(self, grid_h: Grid2D, params: UpscalingParams):
        for i in range(self.pre_relaxation_iters):
            grid_h.relaxation(params)

        if grid_h.coarse_grid is None:  # lowest resolution
            return

        error_params_h = grid_h.get_error_params(params)
        error_params_2h = self.restrict(error_params_h)
        self.v_cycle(grid_h.coarse_grid, error_params_2h)
        error_solution_h = self.interpolate(grid_h.coarse_grid.solution)
        grid_h.solution += error_solution_h

        for i in range(self.post_relaxation_iters):
            grid_h.relaxation(params)

    def process(self, img: np.ndarray, cycles_num: int = 4):
        self.set_initial_guess(img)
        self.finsest_params.integral_constraints = img
        for i in range(cycles_num):
            self.v_cycle(self.finset_grid, self.finsest_params)
        return self.finset_grid.solution_valid.copy()

