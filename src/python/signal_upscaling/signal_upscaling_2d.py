import cv2
import numpy as np
from common import NP_FLOAT32
from multigrid.multigrid_2d import MultigridSolver2D


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class SignalUpscaling2D:
    def __init__(self, cycles_num: int = 4, pre_relaxation_iters: int = 2, post_relaxation_iters: int = 2):
        self.cycles_num = cycles_num
        self.pre_relaxation_iters = pre_relaxation_iters
        self.post_relaxation_iters = post_relaxation_iters

    @staticmethod
    def get_hwc(img: np.ndarray):
        if len(img.shape) == 2: # gray sclae image
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
            if img_dtype == np.uint8:
                upscaled_img = np.clip(upscaled_img, 0, 255)
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
