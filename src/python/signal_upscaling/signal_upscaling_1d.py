from multigrid.multigrid_1d import MultigridSolver1D
from signal_upscaling.signal_upscaling_base import SignalUpscaling, MultigridParams

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class SignalUpscaling1D(SignalUpscaling):
    multigrid_class = MultigridSolver1D

    def __init__(self, mg_params: MultigridParams):
        super().__init__(mg_params)

