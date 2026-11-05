import argparse
import glob
import os.path as osp

import numpy as np
from PIL import Image


IMAGE_DIR_MAP = {
    'irrg': 'img_dir',
    'rgb': 'rgb_dir',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Check Goji image/label/index patch alignment.')
    parser.add_argument(
        '--dataset_path',
        default='/root/shared-nvme/mydataset',
        help='Goji dataset root that contains 26/ and 30/.')
    parser.add_argument('--region', required=True, choices=['26', '30'])
    parser.add_argument(
        '--image_type', default='irrg', choices=['irrg', 'rgb'])
    parser.add_argument(
        '--index_names',
        nargs='+',
        default=['ndvi', 'ndre', 'cire'],
        help='Vegetation index names to check.')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument(
        '--split',
        default='train',
        choices=['train', 'val', 'test'],
        help='Converted split to check.')
    return parser.parse_args()


def read_tif(path):
    return np.array(Image.open(path))


def resolve_dir(region_dir, converted_name, split, raw_name=None):
    converted_dir = osp.join(region_dir, converted_name, split)
    if osp.isdir(converted_dir):
        return converted_dir
    raw_dir = osp.join(region_dir, raw_name or converted_name)
    if osp.isdir(raw_dir):
        return raw_dir
    raise FileNotFoundError(f'Directory not found: {converted_dir}')


def stats(index):
    finite = np.nan_to_num(index.astype(np.float32), nan=0.0, posinf=0.0,
                           neginf=0.0)
    return (
        float(finite.min()),
        float(finite.max()),
        float(finite.mean()),
        float(finite.std()),
        int(np.isnan(index).sum()),
        int(np.isinf(index).sum()),
    )


def main():
    args = parse_args()
    region_dir = osp.join(args.dataset_path, args.region)
    image_root = IMAGE_DIR_MAP[args.image_type]
    image_dir = resolve_dir(region_dir, image_root, args.split,
                            args.image_type)
    label_dir = resolve_dir(region_dir, 'ann_dir', args.split, 'label')
    index_dirs = {
        name: resolve_dir(region_dir, f'{name}_dir', args.split, name)
        for name in args.index_names
    }

    image_paths = sorted(glob.glob(osp.join(image_dir, '*.tif')))
    if not image_paths:
        raise FileNotFoundError(f'No tif files found in {image_dir}')

    for image_path in image_paths[:args.num_samples]:
        filename = osp.basename(image_path)
        image = read_tif(image_path)
        label_path = osp.join(label_dir, filename)
        if not osp.exists(label_path):
            raise FileNotFoundError(f'Label file missing: {label_path}')
        label = read_tif(label_path)
        if label.ndim == 3:
            label = label[..., 0]

        if image.shape[:2] != label.shape[:2]:
            raise ValueError(
                f'[ERROR] shape mismatch: image H/W != label H/W for '
                f'{filename}: image={image.shape}, label={label.shape}')

        index_shapes = {}
        index_stats = {}
        for name, index_dir in index_dirs.items():
            index_path = osp.join(index_dir, filename)
            if not osp.exists(index_path):
                raise FileNotFoundError(f'{name} file missing: {index_path}')
            index = read_tif(index_path)
            if index.ndim == 3:
                index = index[..., 0]
            if image.shape[:2] != index.shape[:2]:
                raise ValueError(
                    f'[ERROR] shape mismatch: image H/W != {name} H/W for '
                    f'{filename}: image={image.shape}, {name}={index.shape}')
            index_shapes[name] = index.shape
            index_stats[name] = stats(index)

        index_shape_text = ', '.join(
            f'{name}={shape}' for name, shape in index_shapes.items())
        print(f'[OK] {filename} image={image.shape}, label={label.shape}, '
              f'{index_shape_text}')

        for name, values in index_stats.items():
            min_val, max_val, mean_val, std_val, nan_count, inf_count = values
            print(f'[STAT] {name} min={min_val:.6f}, max={max_val:.6f}, '
                  f'mean={mean_val:.6f}, std={std_val:.6f}, '
                  f'nan={nan_count}, inf={inf_count}')


if __name__ == '__main__':
    main()
