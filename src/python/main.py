import os
import argparse
from typing import Tuple, List
from tqdm import tqdm

from utils import get_image_names, load_image, save_image
from multigrid.multigrid_base import MultigridParams
from interface import get_upscaler


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def _init_multigrid_params(args: argparse.Namespace):
    return MultigridParams(factor=args.factor, cycles_num=args.cycles_num,
                           pre_relaxation_iters=args.pre_relaxation_iters,
                           post_relaxation_iters=args.post_relaxation_iters)


def _get_image_paths(src_path, dst_path) -> Tuple[List[str], List[str]]:
    if os.path.isdir(src_path):
        if os.path.isfile(dst_path):
            raise ValueError('if input-path is a directory, then output-path cannot be a path to file')
        file_names = get_image_names(src_path)
        src_paths = [os.path.join(src_path, file_name) for file_name in file_names]
        dst_paths = [os.path.join(dst_path, file_name) for file_name in file_names]
    elif os.path.isfile(src_path):
        src_paths = [src_path]
        dst_paths = [os.path.join(dst_path, os.path.basename(src_path))] if os.path.isdir(dst_path) else [dst_path]
    else:
        raise ValueError(f'No such file or directory: {args.input_path}')

    if len(dst_paths) > 0:
        dst_dir = os.path.dirname(dst_paths[0])
        os.makedirs(dst_dir, exist_ok=True)

    return src_paths, dst_paths


def _main(args: argparse.Namespace):
    mg_params = _init_multigrid_params(args)
    upscaler = get_upscaler(2, mg_params)

    src_paths, dst_paths = _get_image_paths(args.input_path, args.output_path)
    for src_path, dst_path in tqdm(zip(src_paths, dst_paths), total=len(src_paths)):
        img = load_image(src_path)
        img_upscaled = upscaler.process(img, factor=mg_params.factor)
        save_image(dst_path, img_upscaled)


def _get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, required=True)
    parser.add_argument('-o', '--output_path', type=str, required=True)
    parser.add_argument('-f', '--factor', type=int, required=True, help="factor need to be an integer power of 2, and >= 1")
    parser.add_argument('--cycles_num', type=int, default=4)
    parser.add_argument('--pre_relaxation_iters', type=int, default=3)
    parser.add_argument('--post_relaxation_iters', type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = _get_parameters()
    _main(args)