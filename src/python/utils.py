import os
import cv2

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IMG_EXTS = ('.bmp', '.png', '.tif', '.tiff', '.gif', '.jpg', '.jpeg')


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def load_image(image_path):
    image = cv2.imread(image_path)
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_image(image_path, image) -> bool:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(image_path, image)


def get_image_names(dir_path: str, exts=IMG_EXTS):
    exts = set(exts)
    fns = os.listdir(dir_path)
    fns = [fn for fn in fns if os.path.splitext(fn.lower())[1] in exts]
    return fns


def get_image_paths(dir_path: str, exts=IMG_EXTS):
    file_names = get_image_names(dir_path, exts=exts)
    return [os.path.join(dir_path, file_name) for file_name in file_names]