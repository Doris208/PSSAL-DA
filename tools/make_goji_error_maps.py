#!/usr/bin/env python
"""Create binary Goji error maps from prediction masks and ground truth."""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


PALETTE = np.array(
    [[220, 20, 60], [34, 139, 34], [70, 70, 70], [0, 0, 255],
     [244, 164, 96]],
    dtype=np.uint8)
IGNORE_INDEX = 255
CORRECT_COLOR = np.array([218, 218, 218], dtype=np.uint8)
ERROR_COLOR = np.array([210, 35, 35], dtype=np.uint8)
IGNORE_COLOR = np.array([255, 255, 255], dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate binary error maps for Goji predictions.')
    parser.add_argument(
        '--gt-dir',
        default=r'D:\Local_HRDA\shared-nvme\mydataset\30\ann_dir\test',
        help='Ground-truth label directory.')
    parser.add_argument(
        '--out-dir',
        default=(r'D:\Local_HRDA\shared-nvme\goji26RGB2goji30RGB'
                 r'\baseline_MPARC_Full\error_maps'),
        help='Output root for error maps and CSV summary.')
    parser.add_argument(
        '--method',
        action='append',
        nargs=2,
        metavar=('NAME', 'PRED_DIR'),
        help='Method name and prediction directory. Can be repeated.')
    parser.add_argument(
        '--raw-gt-mode',
        choices=['auto', 'zero-based', 'one-based'],
        default='auto',
        help='How to interpret single-channel GT labels.')
    parser.add_argument(
        '--raw-pred-mode',
        choices=['auto', 'zero-based', 'one-based'],
        default='auto',
        help='How to interpret single-channel prediction labels.')
    return parser.parse_args()


def default_methods():
    root = Path(r'D:\Local_HRDA\shared-nvme\goji26RGB2goji30RGB')
    return [
        ('baseline',
         root / 'goji26RGB2goji30RGB_hrda_3vi_phy_gate_True' / 'preds' /
         'target_prediction'),
        ('mparc',
         root / 'goji26RGB2goji30RGB_hrda_3vi_phy_struct_gate' / 'preds' /
         'target_prediction'),
        ('full',
         root / 'goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge' / 'preds' /
         'target_prediction'),
    ]


def rgb_to_ids(arr):
    rgb = arr[..., :3].astype(np.int16)
    out = np.full(rgb.shape[:2], IGNORE_INDEX, dtype=np.uint8)
    for cls_id, color in enumerate(PALETTE.astype(np.int16)):
        out[np.all(rgb == color, axis=-1)] = cls_id
    black = np.all(rgb == np.array([0, 0, 0], dtype=np.int16), axis=-1)
    out[black] = IGNORE_INDEX
    return out


def raw_to_ids(arr, mode):
    raw = np.asarray(arr)
    if raw.ndim == 3:
        raw = raw[..., 0]
    raw = raw.astype(np.int64)
    out = np.full(raw.shape, IGNORE_INDEX, dtype=np.uint8)

    valid = raw != IGNORE_INDEX
    if mode == 'one-based':
        valid &= (raw >= 1) & (raw <= len(PALETTE))
        out[valid] = (raw[valid] - 1).astype(np.uint8)
        return out
    if mode == 'zero-based':
        valid &= (raw >= 0) & (raw < len(PALETTE))
        out[valid] = raw[valid].astype(np.uint8)
        return out

    unique = set(np.unique(raw).astype(int).tolist())
    has_one_based = any(v in unique for v in range(1, len(PALETTE) + 1))
    has_zero_based = any(v in unique for v in range(0, len(PALETTE)))
    if 0 in unique and has_one_based and len(PALETTE) in unique:
        valid &= (raw >= 1) & (raw <= len(PALETTE))
        out[valid] = (raw[valid] - 1).astype(np.uint8)
    elif has_zero_based:
        valid &= (raw >= 0) & (raw < len(PALETTE))
        out[valid] = raw[valid].astype(np.uint8)
    else:
        raise ValueError(f'Cannot infer raw label mode from values: {unique}')
    return out


def mask_to_ids(path, raw_mode):
    arr = np.array(Image.open(path))
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        return rgb_to_ids(arr)
    return raw_to_ids(arr, raw_mode)


def save_error_map(gt, pred, out_file):
    valid = (gt != IGNORE_INDEX) & (pred != IGNORE_INDEX)
    correct = valid & (gt == pred)
    wrong = valid & (gt != pred)
    canvas = np.zeros((*gt.shape, 3), dtype=np.uint8)
    canvas[:] = IGNORE_COLOR
    canvas[correct] = CORRECT_COLOR
    canvas[wrong] = ERROR_COLOR
    Image.fromarray(canvas, mode='RGB').save(out_file)
    return int(wrong.sum()), int(valid.sum())


def main():
    args = parse_args()
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    methods = [(name, Path(path)) for name, path in args.method] \
        if args.method else default_methods()

    if not gt_dir.is_dir():
        raise FileNotFoundError(f'GT directory not found: {gt_dir}')
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    gt_paths = sorted(p for p in gt_dir.iterdir() if p.is_file())
    for method_name, pred_dir in methods:
        if not pred_dir.is_dir():
            raise FileNotFoundError(f'Prediction directory not found: '
                                    f'{pred_dir}')
        method_out = out_dir / method_name
        method_out.mkdir(parents=True, exist_ok=True)
        for gt_path in gt_paths:
            pred_path = pred_dir / gt_path.name
            if not pred_path.is_file():
                continue
            gt = mask_to_ids(gt_path, args.raw_gt_mode)
            pred = mask_to_ids(pred_path, args.raw_pred_mode)
            if gt.shape != pred.shape:
                raise ValueError(f'Shape mismatch for {gt_path.name}: '
                                 f'gt={gt.shape}, pred={pred.shape}')
            wrong, valid = save_error_map(gt, pred, method_out / gt_path.name)
            rows.append({
                'method': method_name,
                'filename': gt_path.name,
                'wrong_pixels': wrong,
                'valid_pixels': valid,
                'error_rate': wrong / valid if valid else 0.0,
            })

    csv_path = out_dir / 'error_map_summary.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'method', 'filename', 'wrong_pixels', 'valid_pixels',
                'error_rate'
            ])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} error maps/rows to {out_dir}')
    print(f'Summary: {csv_path}')


if __name__ == '__main__':
    main()
