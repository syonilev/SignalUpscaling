from typing import Optional
import numpy as np
from grid.grid_base import Grid


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class MultigridSolver:
    def __init__(self, shape: tuple, factor: int,
                 pre_relaxation_iters: int = 2,
                 post_relaxation_iters: int = 2):

        self.grids_num = int(np.log2(factor))
        if self.grids_num ** 2 != factor:
            raise ValueError(f'factor ({factor}) must be an integer power of 2')
        self.grids_num += 1

        self.shape = shape
        self.factor = factor
        self.pre_relaxation_iters = pre_relaxation_iters
        self.post_relaxation_iters = post_relaxation_iters
        self.grids: Optional[list] = None

    @property
    def finset_grid(self) -> Grid:
        return self.grids[-1]

