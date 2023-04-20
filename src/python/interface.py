import numpy as np

from multigrid.multigrid_base import MultigridParams
from signal_upscaling.signal_upscaling_base import SignalUpscaling
from signal_upscaling.signal_upscaling_1d import SignalUpscaling1D
from signal_upscaling.signal_upscaling_2d import SignalUpscaling2D


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def get_upscaler(dim: int, mg_params: MultigridParams) -> SignalUpscaling:
    if dim == 1:
        upscaler = SignalUpscaling1D(mg_params)
    elif dim == 2:
        upscaler = SignalUpscaling2D(mg_params)
    else:
        raise ValueError(f'Supporting 1D upscaling, and 2D upscaling only.\ndim value must be either 1 or 2 (dim={dim} was provided)')
    return upscaler


def upscale_1d(y: np.ndarray, mg_params: MultigridParams) -> np.ndarray:
    upscaler = SignalUpscaling1D(mg_params)
    return upscaler.process(y, factor=mg_params.factor)


def upscale_2d(img: np.ndarray, mg_params: MultigridParams) -> np.ndarray:
    upscaler = SignalUpscaling2D(mg_params)
    return upscaler.process(img, factor=mg_params.factor)

