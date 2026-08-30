import argparse
import os
import os.path as osp
import sys

import cv2
import numpy as np
from tqdm import tqdm

GOJI_DEFAULT_ROOT = 'shared-nvme/mydataset/26'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Calculate per-channel mean and std for segmentation '
        'training images.')
    parser.add_argument(
        '--data-root',
        default=GOJI_DEFAULT_ROOT,
        help='Dataset root or image directory. For goji, this is usually '
        'shared-nvme/mydataset/26 or shared-nvme/mydataset/30.')
    parser.add_argument(
        '--img-dir',
        default='img_dir/train',
        help='Relative or absolute image directory. Default: img_dir/train')
    parser.add_argument(
        '--suffixes',
        nargs='+',
        default=['.tif', '.tiff', '.png'],
        help='Image suffixes to include. Default: .tif .tiff .png')
    parser.add_argument(
        '--num-channels',
        type=int,
        default=3,
        help='Expected number of channels. Default: 3')
    parser.add_argument(
        '--convert-to-rgb',
        action='store_true',
        help='Apply BGR->RGB conversion after cv2.imread. Leave disabled for '
        'goji IRRG if you want to preserve the stored channel order.')
    parser.add_argument(
        '--dataset',
        choices=['auto', 'generic', 'goji'],
        default='auto',
        help='Dataset preset. "goji" keeps the stored 3-channel order by '
        'default. "auto" infers from the path.')
    return parser.parse_args()


def infer_dataset_name(data_root):
    lowered = osp.abspath(data_root).lower()
    if 'mydataset' in lowered or lowered.endswith(f'{os.sep}26') or \
            lowered.endswith(f'{os.sep}30'):
        return 'goji'
    return 'generic'


def resolve_img_dir(data_root, img_dir):
    if osp.isabs(img_dir):
        return img_dir
    if osp.basename(osp.abspath(data_root)) == 'train':
        return osp.abspath(data_root)
    if osp.isdir(osp.join(data_root, img_dir)):
        return osp.join(data_root, img_dir)
    return osp.abspath(data_root)


def collect_image_paths(img_dir, suffixes):
    img_list = []
    suffixes = tuple(s.lower() for s in suffixes)
    for filename in sorted(os.listdir(img_dir)):
        path = osp.join(img_dir, filename)
        if osp.isfile(path) and filename.lower().endswith(suffixes):
            img_list.append(path)
    return img_list


def calculate_stats(img_dir, num_channels, convert_to_rgb):
    img_list = collect_image_paths(img_dir, args.suffixes)
    if not img_list:
        raise FileNotFoundError(f'No images found in {img_dir}')

    print(f'Found {len(img_list)} images. Starting calculation...')

    channels_sum = np.zeros(num_channels, dtype=np.float64)
    channels_squared_sum = np.zeros(num_channels, dtype=np.float64)
    pixels_count = 0

    for img_path in tqdm(img_list):
        try:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f'Error loading image: {img_path}')
                continue
        except Exception as e:
            print(f'Exception loading image: {img_path}, error: {e}')
            continue

        if img.ndim != 3:
            print(f'Warning: Image {img_path} is not 3D. Skipping.')
            continue

        h, w, c = img.shape
        if c != num_channels:
            print(f'Warning: Image {img_path} has {c} channels, expected '
                  f'{num_channels}. Skipping.')
            continue

        if convert_to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float64)
        channels_sum += img.reshape(-1, c).sum(axis=0)
        channels_squared_sum += (img.reshape(-1, c)**2).sum(axis=0)
        pixels_count += h * w

    if pixels_count == 0:
        raise RuntimeError('No valid pixels were accumulated.')

    final_mean = channels_sum / pixels_count
    final_var = (channels_squared_sum / pixels_count) - (final_mean**2)
    final_std = np.sqrt(final_var)

    print('\n' + '=' * 40)
    print(f'Dataset Stats for: {img_dir}')
    print('-' * 40)
    print(f'Final Mean (convert_to_rgb={convert_to_rgb}):')
    print([f'{x:.3f}' for x in final_mean])
    print('-' * 40)
    print(f'Final Std (convert_to_rgb={convert_to_rgb}):')
    print([f'{x:.3f}' for x in final_std])
    print('=' * 40)
    print('Now update this in your dataset config file!')


if __name__ == '__main__':
    args = parse_args()
    dataset_name = infer_dataset_name(
        args.data_root) if args.dataset == 'auto' else args.dataset
    convert_to_rgb = args.convert_to_rgb
    if dataset_name == 'goji' and '--convert-to-rgb' not in sys.argv:
        convert_to_rgb = False
    img_dir = resolve_img_dir(args.data_root, args.img_dir)
    calculate_stats(img_dir, args.num_channels, convert_to_rgb)
