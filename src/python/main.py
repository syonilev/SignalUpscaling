import cv2

from signal_upscaling.signal_upscaling_2d import SignalUpscaling2D

import matplotlib
import matplotlib.pyplot as plt


backend = 'TkAgg'
matplotlib.use(backend)



def load_image(image_path):
    image = cv2.imread(image_path)
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def show(img, title=None):
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.show(block=False)
    plt.title(title)


if __name__ == "__main__":
    imp="/home/cg5/Yoni/Personal/Search/Bosch/aerial.jpg"
    img=load_image(imp)[:100, :200]
    # img = img[:,:,0]



    su2d = SignalUpscaling2D(cycles_num = 4, pre_relaxation_iters = 2, post_relaxation_iters = 2)
    img_upscaled = su2d.process(img, factor=4)

    show(img, title="orig")
    show(img_upscaled, title="upscaled")

    aaa=1