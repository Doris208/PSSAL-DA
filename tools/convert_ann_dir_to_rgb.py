import argparse
import os.path as osp
import sys
from typing import Dict, List, Tuple

import mmcv
import numpy as np
from PIL import Image

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mmseg.datasets.goji_meta import GOJI_RAW_ID_TO_COLOR, \
    GOJI_REDUCED_ID_TO_COLOR


DATASET_SPECS: Dict[str, Dict[str, object]] = {
    'isprs': {
        # Original RGB labels used before converting to single-channel ids.
        'raw_id_to_color': [
            [0, 0, 0],
            [255, 255, 255],
            [255, 0, 0],
            [255, 255, 0],
            [0, 255, 0],
            [0, 255, 255],
            [0, 0, 255],
        ],
        # Reduced labels used after reduce_zero_label in the training pipeline.
        'reduced_id_to_color': [
            [255, 255, 255],
            [255, 0, 0],
            [255, 255, 0],
            [0, 255, 0],
            [0, 255, 255],
            [0, 0, 255],
        ],
        'aliases': ['potsdam', 'vaihingen'],
        'raw_ignore_ids': [],
        'reduced_ignore_ids': [255],
    },
    'cityscapes': {
        'raw_id_to_color': [
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [70, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
        ],
        'reduced_id_to_color': [
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [70, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
        ],
        'aliases': ['gta', 'synthia', 'acdc', 'darkzurich', 'dark_zurich'],
        'raw_ignore_ids': [255],
        'reduced_ignore_ids': [255],
    },
    'loveda': {
        # LoveDA raw annotations use 0 as ignore and 1..7 as class ids.
        'raw_id_to_color': [
            [0, 0, 0],
            [255, 255, 255],
            [255, 0, 0],
            [255, 255, 0],
            [0, 0, 255],
            [159, 129, 183],
            [0, 255, 0],
            [255, 195, 128],
        ],
        'reduced_id_to_color': [
            [255, 255, 255],
            [255, 0, 0],
            [255, 255, 0],
            [0, 0, 255],
            [159, 129, 183],
            [0, 255, 0],
            [255, 195, 128],
        ],
        'aliases': [],
        'raw_ignore_ids': [0, 255],
        'reduced_ignore_ids': [255],
    },
    'goji': {
        # Goji raw annotations use 0 as ignore and 1..5 as class ids.
        'raw_id_to_color': GOJI_RAW_ID_TO_COLOR,
        'reduced_id_to_color': GOJI_REDUCED_ID_TO_COLOR,
        'aliases': ['goji26', 'goji30'],
        'raw_ignore_ids': [0, 255],
        'reduced_ignore_ids': [255],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert single-channel ann_dir masks to RGB masks.')
    parser.add_argument(
        'data_root',
        help='Dataset root containing ann_dir/, or the ann_dir path itself.')
    parser.add_argument(
        '--dataset',
        default='auto',
        help='Dataset name. Supports auto, isprs, potsdam, vaihingen, '
        'cityscapes, gta, synthia, acdc, darkzurich, loveda, goji.')
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val'],
        help='Splits to process. Default: train val')
    parser.add_argument(
        '--output-dir-name',
        default='ann_dir_RGB_GT',
        help='Name of the generated RGB annotation directory.')
    parser.add_argument(
        '--label-mode',
        choices=['auto', 'raw', 'reduced'],
        default='auto',
        help='How to interpret single-channel ids. '
        '"raw" means original ids before reduce_zero_label, '
        '"reduced" means ids after reduce_zero_label, '
        '"auto" infers from value range.')
    parser.add_argument(
        '--ignore-color',
        nargs=3,
        type=int,
        default=[0, 0, 0],
        help='RGB color used for ignore label 255. Default: 0 0 0')
    return parser.parse_args()


def normalize_dataset_name(dataset_name: str) -> str:
    dataset_name = dataset_name.lower()
    if dataset_name in DATASET_SPECS:
        return dataset_name
    for canonical_name, spec in DATASET_SPECS.items():
        aliases = spec.get('aliases', [])
        if dataset_name in aliases:
            return canonical_name
    raise ValueError(f'Unsupported dataset "{dataset_name}".')


def infer_dataset_name(path: str) -> str:
    lowered = path.lower()
    for canonical_name, spec in DATASET_SPECS.items():
        if canonical_name in lowered:
            return canonical_name
        for alias in spec.get('aliases', []):
            if alias in lowered:
                return canonical_name
    raise ValueError(
        'Failed to infer dataset type from the path. '
        'Please pass --dataset explicitly.')


def resolve_dirs(
        data_root: str,
        output_dir_name: str) -> Tuple[str, str, str]:
    norm_root = osp.abspath(data_root)
    if osp.basename(norm_root) == 'ann_dir':
        ann_dir = norm_root
        dataset_root = osp.dirname(norm_root)
    else:
        dataset_root = norm_root
        ann_dir = osp.join(dataset_root, 'ann_dir')

    if not osp.isdir(ann_dir):
        raise FileNotFoundError(f'ann_dir not found: {ann_dir}')

    out_dir = osp.join(dataset_root, output_dir_name)
    return dataset_root, ann_dir, out_dir


def build_color_map(
        spec: Dict[str, object],
        ann: np.ndarray,
        label_mode: str) -> np.ndarray:
    raw_colors = np.array(spec['raw_id_to_color'], dtype=np.uint8)
    reduced_colors = np.array(spec['reduced_id_to_color'], dtype=np.uint8)
    if label_mode == 'raw':
        return raw_colors
    if label_mode == 'reduced':
        return reduced_colors

    ignore_ids = set(spec.get('raw_ignore_ids', [])) | set(
        spec.get('reduced_ignore_ids', []))
    valid_mask = np.ones_like(ann, dtype=bool)
    for ignore_id in ignore_ids:
        valid_mask &= ann != ignore_id
    valid_ids = ann[valid_mask]

    if valid_ids.size == 0:
        return reduced_colors

    min_id = int(valid_ids.min())
    max_id = int(valid_ids.max())

    if max_id == len(raw_colors) - 1:
        return raw_colors
    if min_id >= 1 and max_id <= len(raw_colors) - 1:
        return raw_colors
    if min_id == 0 and max_id <= len(reduced_colors) - 1:
        return reduced_colors

    raise ValueError(
        f'Label ids out of supported range. min={min_id}, max={max_id}')


def mask_to_rgb(
        ann: np.ndarray,
        color_map: np.ndarray,
        ignore_color: List[int],
        ignore_ids: List[int]) -> np.ndarray:
    rgb = np.zeros((ann.shape[0], ann.shape[1], 3), dtype=np.uint8)
    ignore_color = np.array(ignore_color, dtype=np.uint8)
    valid_mask = np.ones_like(ann, dtype=bool)
    for ignore_id in ignore_ids:
        ignore_mask = ann == ignore_id
        rgb[ignore_mask] = ignore_color
        valid_mask &= ~ignore_mask

    valid_ids = ann[valid_mask]
    if valid_ids.size == 0:
        return rgb

    min_valid_id = int(valid_ids.min())
    max_valid_id = int(valid_ids.max())

    if min_valid_id >= 1 and max_valid_id <= len(color_map) - 1:
        for label_id in range(1, len(color_map)):
            rgb[ann == label_id] = color_map[label_id]
        return rgb

    if max_valid_id <= len(color_map) - 1:
        for label_id in range(len(color_map)):
            rgb[ann == label_id] = color_map[label_id]
        return rgb

    raise ValueError(
        f'Found label id {max_valid_id}, but color map only supports up to '
        f'{len(color_map) - 1}.')


def convert_split(
        split_dir: str,
        out_split_dir: str,
        spec: Dict[str, object],
        ignore_color: List[int],
        label_mode: str) -> int:
    mmcv.mkdir_or_exist(out_split_dir)
    filenames = sorted(
        mmcv.scandir(split_dir, suffix=('.png', '.tif', '.tiff'),
                     recursive=False))
    for filename in filenames:
        src_path = osp.join(split_dir, filename)
        dst_path = osp.join(out_split_dir, filename)

        ann = np.array(Image.open(src_path))
        if ann.ndim == 3:
            ann = ann[..., 0]
        ann = ann.astype(np.uint8)

        color_map = build_color_map(spec, ann, label_mode)
        ignore_ids = spec['raw_ignore_ids'] if label_mode == 'raw' \
            else spec['reduced_ignore_ids']
        if label_mode == 'auto':
            valid_without_raw_ignore = ann[
                ~np.isin(ann, spec.get('raw_ignore_ids', []))]
            if valid_without_raw_ignore.size > 0 and \
                    int(valid_without_raw_ignore.min()) >= 1:
                ignore_ids = spec['raw_ignore_ids']
            else:
                ignore_ids = spec['reduced_ignore_ids']
        rgb = mask_to_rgb(ann, color_map, ignore_color, ignore_ids)
        Image.fromarray(rgb, mode='RGB').save(dst_path)
    return len(filenames)


def main():
    args = parse_args()
    dataset_root, ann_dir, out_dir = resolve_dirs(
        args.data_root, args.output_dir_name)

    dataset_name = infer_dataset_name(dataset_root) \
        if args.dataset == 'auto' else normalize_dataset_name(args.dataset)
    spec = DATASET_SPECS[dataset_name]

    print(f'Dataset: {dataset_name}')
    print(f'Input ann_dir: {ann_dir}')
    print(f'Output ann_dir: {out_dir}')

    total = 0
    for split in args.splits:
        split_dir = osp.join(ann_dir, split)
        if not osp.isdir(split_dir):
            print(f'Skip missing split: {split_dir}')
            continue
        out_split_dir = osp.join(out_dir, split)
        count = convert_split(
            split_dir,
            out_split_dir,
            spec,
            args.ignore_color,
            args.label_mode)
        total += count
        print(f'Converted {count} files from {split}')

    print(f'Done. Total converted files: {total}')


if __name__ == '__main__':
    main()
