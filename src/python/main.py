import cv2
from typing import Optional
import numpy as np
from scipy.signal import convolve2d
from boundary import Boundaries2D, BoundaryConditions


NP_FLOAT32 = np.float32




# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class UpscalingParams:
    def __init__(self, f: np.ndarray, integral_constraints: np.ndarray, boundary_conditions: BoundaryConditions):
        self.f = f
        self.integral_constraints = integral_constraints
        self.boundary_conditions = boundary_conditions



# # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# class Boundary:
#     def __init__(self, inside: np.ndarry, ghost: np.ndarry, h: int):
#         self.inside = inside
#         self.ghost = ghost
#         self.h = h
#
#     def get_derivative(self):
#         return (self.ghost - self.inside) / self.h
#
#     def set_derivative(self, boundary_conditions: np.ndarry):
#         self.ghost[:] = boundary_conditions * self.h + self.inside

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

    # def create_boundaries(self):
    #     sol, h = self.solution, self.h
    #     out_in_indices = [(0, 1), (-1, -2)]
    #
    #     l, r = [Boundary(sol[:, out_ind], sol[:, in_ind], h) for out_ind, in_ind in out_in_indices]
    #     t, b = [Boundary(sol[out_ind, :], sol[in_ind, :], h) for out_ind, in_ind in out_in_indices]
    #     return {LEFT: l, RIGHT: r, TOP: t, BOTTOM: b}

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

    # def set_boundary_conditions(self, boundary_conditions: dict):
    #     for name, boundary in self.boundaries.items():
    #         boundary.set_derivative(boundary_conditions[name])

    # def get_boundary_conditions(self):
    #     return {name: boundary.get_derivative() for name, boundary in self.boundaries.items()}

    @property
    def integrals(self):
        return self.resize(self.solution_valid, self.wh_orig)

    def enforce_integral_constraints(self, integral_constraints):
        integrals_diff = integral_constraints - self.integrals
        self.solution_valid += self.resize(integrals_diff, self.wh_grid_valid)

    def relaxation(self, params: UpscalingParams):
        self.boundaries.set_boundary_condirions(params.boundary_conditions)

        if self.factor > 1: # if not at lowest reslution
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


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class MultigridSolver:
    def __init__(self, shape: tuple, factor: int,
                 pre_relaxation_iters: int = 2, post_relaxation_iters: int = 2):

        self.grids_num = int(np.log2(factor))
        if self.grids_num ** 2 != factor:
            raise ValueError(f'factor ({factor}) must be an integer power of 2')

        self.shape = shape
        self.factor = factor
        self.pre_relaxation_iters = pre_relaxation_iters
        self.post_relaxation_iters = post_relaxation_iters
        self.grids: Optional[list] = None

    @property
    def finset_grid(self) -> Grid2D:
        return self.grids[-1]


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class MultigridSolver2D(MultigridSolver):
    def __init__(self, wh: tuple, factor: int,
                 pre_relaxation_iters: int = 2, post_relaxation_iters: int = 2):

        super().__init__(wh, factor, pre_relaxation_iters, post_relaxation_iters)
        self.construct_grids()
        grid = self.finset_grid

        zeros_mat = np.zeros_like(grid.solution)
        boundaries_tmp = Boundaries2D.init_from_solution(zeros_mat, 1)
        f = zeros_mat[1:-1, 1:-1]
        boundary_conditions = boundaries_tmp.get_boundary_conditions()
        self.finsest_params = UpscalingParams(f, None, boundary_conditions)

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
        initial_guess = Grid2D.resize(integral_constraints, grid.wh_grid_valid)
        grid.set_initial_guess(initial_guess)


    def restrict(self, params_h: UpscalingParams):
        f_2h = Grid2D.downscale(params_h.f)
        integral_constraints_2h = Grid2D.downscale(params_h.integral_constraints)
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
        return self.finset_grid.solution_valid


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class SignalUpscaling2D:
    def __init__(self, cycles_num: int = 4, pre_relaxation_iters: int = 2, post_relaxation_iters: int = 2):
        self.cycles_num = cycles_num
        self.pre_relaxation_iters = pre_relaxation_iters
        self.post_relaxation_iters = post_relaxation_iters

    @staticmethod
    def get_hwc(img: np.ndarray):
        if len(img.shape) == 2:
            h, w = img.shape
            c = 0
        else:
            h, w, c = img.shape
        return h, w, c

    def init_solver(self, wh, factor):
        return MultigridSolver2D(wh, factor,
                                 pre_relaxation_iters=self.pre_relaxation_iters,
                                 post_relaxation_iters=self.post_relaxation_iters)

    def apply_solver(self, solver: MultigridSolver2D, img: np.ndarray, c: int):
        if c == 0:
            upscaled_img = solver.process(img, cycles_num=self.cycles_num)
        else:
            upscaled_images = [solver.process(img[:, :, i], cycles_num=self.cycles_num) for i in range(c)]
            upscaled_img = np.stack(upscaled_images, axis=2)
        return upscaled_img

    def process(self, img: np.ndarray, factor: int):
        img_dtype = img.dtype
        if img_dtype != NP_FLOAT32:
            img = img.astype(NP_FLOAT32)

        h, w, c = self.get_hwc(img)
        solver = self.init_solver((w, h), factor)
        upscaled_img = self.apply_solver(solver, img, c)

        if img_dtype != NP_FLOAT32:
            upscaled_img = upscaled_img.astype(img_dtype)
        return upscaled_img


def load_image(image_path):
    image = cv2.imread(image_path)
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# if __name__ == "__main__":
#     imp="/home/cg5/Yoni/Personal/Search/Bosch/aerial.jpg"
#     img=load_image(imp)[:100, :200]
#
#     # import matplotlib
#     # import matplotlib.pyplot as plt
#     #
#     #
#     # backend = 'TkAgg'
#     # matplotlib.use(backend)
#     # plt.figure()
#     # plt.imshow(img)
#     # plt.show(block=False)
#
#     su2d = SignalUpscaling2D(cycles_num = 4, pre_relaxation_iters = 2, post_relaxation_iters = 2)
#     img_upscaled = su2d.process(img, factor=4)
#
#     aaa=1

