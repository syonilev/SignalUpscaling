import numpy as np
from multigrid.multigrid_2d import MultigridSolver2D
from signal_upscaling.signal_upscaling_base import SignalUpscaling, MultigridSolver, MultigridParams


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class SignalUpscaling2D(SignalUpscaling):
    multigrid_class = MultigridSolver2D

    def __init__(self, mg_params: MultigridParams):
        super().__init__(mg_params)

    @staticmethod
    def get_shape(signal: np.ndarray):
        return signal.shape[:2]

    def apply_solver(self, solver: MultigridSolver, img: np.ndarray):
        if len(img.shape) == 2: # gray scale image
            upscaled_img = solver.process(img)
        else:
            c = img.shape[2]
            upscaled_images = [solver.process(img[:, :, i]) for i in range(c)]
            upscaled_img = np.stack(upscaled_images, axis=2)
        return upscaled_img

