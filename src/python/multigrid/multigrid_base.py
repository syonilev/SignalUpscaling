from typing import Optional
import numpy as np
from abc import ABC
from grid.grid_base import Grid
from grid.minimization_params import MinimizationParams
from multigrid.multigrid_params import MultigridParams


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class MultigridSolver(ABC):
    grid_class = None
    boundaries_class = None

    def __init__(self, shape: tuple, mg_params: MultigridParams):
        factor = mg_params.factor

        self.grids_num = int(np.log2(factor))
        if 2 ** self.grids_num != factor:
            raise ValueError(f'factor ({factor}) must be an integer power of 2')
        self.grids_num += 1

        self.shape = shape
        self.mg_params = mg_params
        self.grids: Optional[list] = None

        self.construct_grids()
        self.finsest_params = self.init_zero_params()

    @property
    def finset_grid(self) -> Grid:
        return self.grids[-1]

    def construct_grids(self):
        self.grids = []

        factor = 1
        for i in range(self.grids_num):
            grid = self.grid_class(self.shape, factor)
            self.grids.append(grid)
            factor *= 2

        for i in range(self.grids_num - 1):
            self.grids[i + 1].coarse_grid = self.grids[i]

    def set_initial_guess(self, integral_constraints: np.ndarray):
        for i in range(self.grids_num - 1):
            self.grids[i].set_initial_guess()

        grid = self.finset_grid
        initial_guess = grid.resize_area(integral_constraints, grid.grid_valid_shape)
        grid.set_initial_guess(initial_guess)

    @classmethod
    def restrict(cls, params_h: MinimizationParams):
        f_2h = cls.grid_class.downscale(params_h.f)
        integral_constraints_2h = params_h.integral_constraints
        boundary_conditions_2h = params_h.boundary_conditions.restrict()
        return MinimizationParams(f_2h, integral_constraints_2h, boundary_conditions_2h)

    def init_zero_params(self):
        grid = self.finset_grid
        zeros_mat = np.zeros_like(grid.solution)
        boundaries_tmp = self.boundaries_class.init_from_solution(zeros_mat, 1)
        f = grid.get_valid(zeros_mat)
        boundary_conditions = boundaries_tmp.get_boundary_conditions()
        return MinimizationParams(f, None, boundary_conditions)

    def relaxation(self, grid: Grid, params: MinimizationParams, iters: int):
        for i in range(iters):
            grid.relaxation(params)

    def v_cycle(self, grid_h: Grid, params: MinimizationParams):
        self.relaxation(grid_h, params, self.mg_params.pre_relaxation_iters)

        grid_2h = grid_h.coarse_grid
        if grid_2h is None:  # lowest resolution
            return

        grid_2h.solution_valid = 0
        error_params_h = grid_h.get_error_params(params)
        error_params_2h = self.restrict(error_params_h)
        
        self.v_cycle(grid_2h, error_params_2h)

        error_solution_h = grid_2h.interpolated_solution
        grid_h.solution += error_solution_h

        self.relaxation(grid_h, params, self.mg_params.post_relaxation_iters)

    def set_signal(self, signal: np.ndarray):
        self.set_initial_guess(signal)
        self.finsest_params.integral_constraints = signal

    def apply_cycles(self, iters: int):
        for i in range(iters):
            self.v_cycle(self.finset_grid, self.finsest_params)

    def get_solution(self):
        return self.finset_grid.solution_valid.copy()

    def process(self, signal: np.ndarray):
        self.set_signal(signal)
        self.apply_cycles(self.mg_params.cycles_num)
        return self.get_solution()


