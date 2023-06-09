from boundary.boundaries_2d import Boundaries2D
from grid.grid_2d import Grid2D
from multigrid.multigrid_base import MultigridSolver, MultigridParams


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class MultigridSolver2D(MultigridSolver):
    grid_class = Grid2D
    boundaries_class = Boundaries2D

    def __init__(self, shape: tuple, mg_params: MultigridParams):
        super().__init__(shape, mg_params)


