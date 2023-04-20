from boundary.boundaries_1d import Boundaries1D
from grid.grid_1d import Grid1D
from multigrid.multigrid_base import MultigridSolver, MultigridParams


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class MultigridSolver1D(MultigridSolver):
    grid_class = Grid1D
    boundaries_class = Boundaries1D

    def __init__(self, shape: tuple, mg_params: MultigridParams):
        super().__init__(shape, mg_params)

