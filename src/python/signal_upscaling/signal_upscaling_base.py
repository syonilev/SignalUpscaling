import numpy as np
from common import NP_FLOAT32, NP_UINT8
from multigrid.multigrid_base import MultigridSolver, MultigridParams
from abc import ABC


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class SignalUpscaling(ABC):
    multigrid_class = None

    def __init__(self, mg_params: MultigridParams):
        self.mg_params = mg_params

    @staticmethod
    def get_shape(signal: np.ndarray):
        return signal.shape

    def init_solver(self, shape):
        return self.multigrid_class(shape, self.mg_params)

    def apply_solver(self, solver: MultigridSolver, signal: np.ndarray):
        return  solver.process(signal)

    def process(self, signal: np.ndarray, factor: int):
        self.mg_params.factor = factor

        signal_dtype = signal.dtype
        if signal_dtype != NP_FLOAT32:
            signal = signal.astype(NP_FLOAT32)

        solver = self.init_solver(self.get_shape(signal))
        upscaled_signal = self.apply_solver(solver, signal)

        if signal_dtype != NP_FLOAT32:
            if signal_dtype == NP_UINT8:
                upscaled_signal = np.clip(upscaled_signal, 0, 255)
            upscaled_signal = upscaled_signal.astype(signal_dtype)
        return upscaled_signal



