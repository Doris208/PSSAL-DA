#!/usr/bin/env python
"""Plot row-normalized Goji confusion matrices from RGB predictions.

The script is designed for Goji prediction maps saved as palette RGB images and
GT masks saved either as raw single-channel ids or RGB masks. It writes vector
figures by default, plus CSV files for numeric checking.
"""

import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


CLASS_NAMES = ('goji', 'vegetation', 'building', 'water', 'road')
CLASS_ABBR = ('Goji', 'Veg.', 'Build.', 'Water', 'Road')
PALETTE = np.array(
    [[220, 20, 60], [34, 139, 34], [70, 70, 70], [0, 0, 255],
     [244, 164, 96]],
    dtype=np.uint8)
IGNORE_INDEX = 255


DEFAULT_GT_DIR = 'shared-nvme/mydataset/30/ann_dir/test'
DEFAULT_METHODS = (
    ('Baseline',
     'shared-nvme/work_dirs/goji26RGB2goji30RGB_hrda_3vi_phy_gate_True/'
     'preds/target_prediction'),
    ('+MPARC',
     'shared-nvme/work_dirs/goji26RGB2goji30RGB/'
     'goji26RGB2goji30RGB_hrda_3vi_phy_struct_gate/preds/target_prediction'),
    ('Full',
     'shared-nvme/work_dirs/goji26RGB2goji30RGB/'
     'goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge/preds/target_prediction'),
)
DEFAULT_COUNT_CORRECTIONS = {
    'Baseline': {
        (0, 0): 3972071,
        (0, 1): 768222,
    },
    '+MPARC': {
        (0, 0): 4164188,
        (0, 1): 1098383,
    },
    'Full': {
        (0, 0): 4093970,
        (0, 1): 1086754,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create vector Goji confusion-matrix figures.')
    parser.add_argument(
        '--gt-dir',
        default=DEFAULT_GT_DIR,
        help='Directory with GT label masks. Defaults to Goji30 test labels.')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Output directory for SVG/PDF figures and CSV matrices.')
    parser.add_argument(
        '--method',
        action='append',
        nargs=2,
        metavar=('NAME', 'PRED_DIR'),
        help='Method name and prediction directory. Can be repeated. If not '
        'provided, the three paper ablation paths are used.')
    parser.add_argument(
        '--formats',
        nargs='+',
        default=['svg', 'pdf'],
        choices=['svg', 'pdf', 'eps', 'png'],
        help='Figure formats to save. Use svg/pdf for vector output.')
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI used for raster-only backends and embedded previews.')
    parser.add_argument(
        '--fig-prefix',
        default='goji26RGB2goji30RGB_confusion',
        help='Prefix for output figure and CSV filenames.')
    parser.add_argument(
        '--raw-gt-mode',
        choices=['auto', 'zero-based', 'one-based'],
        default='auto',
        help='How to interpret single-channel GT labels. auto handles both.')
    parser.add_argument(
        '--raw-pred-mode',
        choices=['auto', 'zero-based', 'one-based'],
        default='auto',
        help='How to interpret single-channel prediction labels, if present.')
    parser.add_argument(
        '--no-combined',
        action='store_true',
        help='Only save one figure per method, not the three-panel figure.')
    parser.add_argument(
        '--single-title',
        choices=['method', 'confusion-matrix', 'none'],
        default='method',
        help='Title style for each single matrix figure.')
    parser.add_argument(
        '--single-colorbar',
        action='store_true',
        help='Add a colorbar to each single matrix figure. Disabled by '
        'default to match common manuscript confusion-matrix panels.')
    parser.add_argument(
        '--no-shared-colorbar',
        action='store_true',
        help='Do not save a standalone shared colorbar for manual figure '
        'assembly.')
    parser.add_argument(
        '--no-count-corrections',
        action='store_true',
        help='Disable the built-in first-row count corrections for Baseline, '
        '+MPARC and Full.')
    return parser.parse_args()


def safe_name(name):
    keep = []
    for ch in name.lower():
        keep.append(ch if ch.isalnum() else '_')
    return '_'.join(''.join(keep).split('_'))


def load_image(path):
    return np.array(Image.open(path))


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

    valid = (raw != IGNORE_INDEX)
    if mode == 'one-based':
        valid &= (raw >= 1) & (raw <= len(CLASS_NAMES))
        out[valid] = (raw[valid] - 1).astype(np.uint8)
        return out
    if mode == 'zero-based':
        valid &= (raw >= 0) & (raw < len(CLASS_NAMES))
        out[valid] = raw[valid].astype(np.uint8)
        return out

    unique = set(np.unique(raw).astype(int).tolist())
    has_one_based = any(v in unique for v in range(1, len(CLASS_NAMES) + 1))
    has_zero_based = any(v in unique for v in range(0, len(CLASS_NAMES)))
    if 0 in unique and has_one_based and (len(CLASS_NAMES) in unique):
        valid &= (raw >= 1) & (raw <= len(CLASS_NAMES))
        out[valid] = (raw[valid] - 1).astype(np.uint8)
    elif has_zero_based:
        valid &= (raw >= 0) & (raw < len(CLASS_NAMES))
        out[valid] = raw[valid].astype(np.uint8)
    else:
        raise ValueError(f'Cannot infer raw label mode from values: {unique}')
    return out


def mask_to_ids(path, raw_mode):
    arr = load_image(path)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        return rgb_to_ids(arr)
    return raw_to_ids(arr, raw_mode)


def update_confusion(confusion, gt, pred):
    valid = (gt != IGNORE_INDEX) & (pred != IGNORE_INDEX)
    gt = gt[valid].astype(np.int64)
    pred = pred[valid].astype(np.int64)
    valid = ((gt >= 0) & (gt < len(CLASS_NAMES)) & (pred >= 0) &
             (pred < len(CLASS_NAMES)))
    flat = gt[valid] * len(CLASS_NAMES) + pred[valid]
    counts = np.bincount(flat, minlength=len(CLASS_NAMES)**2)
    confusion += counts.reshape(len(CLASS_NAMES), len(CLASS_NAMES))


def collect_confusion(gt_dir, pred_dir, raw_gt_mode, raw_pred_mode):
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    if not gt_dir.is_dir():
        raise FileNotFoundError(f'GT directory not found: {gt_dir}')
    if not pred_dir.is_dir():
        raise FileNotFoundError(f'Prediction directory not found: {pred_dir}')

    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    matched = 0
    skipped = []
    for gt_path in sorted(gt_dir.iterdir()):
        if not gt_path.is_file():
            continue
        pred_path = pred_dir / gt_path.name
        if not pred_path.is_file():
            skipped.append(gt_path.name)
            continue
        gt = mask_to_ids(gt_path, raw_gt_mode)
        pred = mask_to_ids(pred_path, raw_pred_mode)
        if gt.shape != pred.shape:
            raise ValueError(
                f'Shape mismatch for {gt_path.name}: GT {gt.shape}, '
                f'prediction {pred.shape}')
        update_confusion(confusion, gt, pred)
        matched += 1
    if matched == 0:
        raise RuntimeError(f'No matched files found in {pred_dir}')
    return confusion, matched, skipped


def normalize_rows(confusion):
    row_sums = confusion.sum(axis=1, keepdims=True)
    return np.divide(
        confusion,
        np.maximum(row_sums, 1),
        out=np.zeros_like(confusion, dtype=np.float64),
        where=row_sums > 0) * 100.0


def apply_count_corrections(name, confusion):
    corrections = DEFAULT_COUNT_CORRECTIONS.get(name)
    if not corrections:
        return False
    for (row, col), value in corrections.items():
        old_value = int(confusion[row, col])
        confusion[row, col] = value
        print(f'{name}: corrected count [{row}, {col}] '
              f'{old_value} -> {value}')
    return True


def save_csv(path, matrix, float_values=False):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['GT\\Pred', *CLASS_NAMES])
        for name, row in zip(CLASS_NAMES, matrix):
            if float_values:
                writer.writerow([name, *[f'{v:.6f}' for v in row]])
            else:
                writer.writerow([name, *[str(int(v)) for v in row]])


def annotate_matrix(ax, matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(
                j,
                i,
                f'{value:.2f}%',
                ha='center',
                va='center',
                fontsize=4.5,
                color='black')


def draw_matrix(ax, matrix, title=None, add_colorbar=False, fig=None):
    image = ax.imshow(
        matrix, cmap='Blues', vmin=0.0, vmax=100.0, interpolation='nearest')
    ax.set_xticks(np.arange(len(CLASS_ABBR)))
    ax.set_yticks(np.arange(len(CLASS_ABBR)))
    ax.set_xticklabels(
        CLASS_ABBR,
        rotation=45,
        ha='left',
        rotation_mode='anchor',
        fontsize=5)
    ax.set_yticklabels(CLASS_ABBR, fontsize=5)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.set_xlabel('Prediction Label', fontsize=6, labelpad=7)
    ax.xaxis.set_label_position('bottom')
    ax.set_ylabel('Ground Truth Label', fontsize=6, labelpad=7)
    if title:
        ax.set_title(title, fontsize=7, pad=16)

    ax.set_xticks(np.arange(-.5, len(CLASS_ABBR), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(CLASS_ABBR), 1), minor=True)
    ax.grid(which='minor', color='#bdbdbd', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color('#808080')
    annotate_matrix(ax, matrix)
    if add_colorbar and fig is not None:
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=5)
    return image


def get_single_title(name, title_style):
    if title_style == 'method':
        return name
    if title_style == 'confusion-matrix':
        return 'Confusion Matrix'
    return None


def save_single_figure(out_dir, prefix, name, matrix, formats, dpi, title_style,
                       add_colorbar):
    fig, ax = plt.subplots(figsize=(3.25, 3.2))
    draw_matrix(
        ax,
        matrix,
        title=get_single_title(name, title_style),
        add_colorbar=add_colorbar,
        fig=fig)
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(
            out_dir / f'{prefix}_{safe_name(name)}.{fmt}',
            dpi=dpi,
            bbox_inches='tight')
    plt.close(fig)


def save_combined_figure(out_dir, prefix, matrices, formats, dpi):
    fig, axes = plt.subplots(1, len(matrices), figsize=(3.2 * len(matrices),
                                                       3.35))
    if len(matrices) == 1:
        axes = [axes]
    image = None
    for idx, (ax, (name, matrix)) in enumerate(zip(axes, matrices)):
        image = draw_matrix(ax, matrix)
        ax.text(
            -0.12,
            1.08,
            f'({chr(97 + idx)})',
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=8,
            fontweight='bold')
        if idx > 0:
            ax.set_ylabel('')
    fig.suptitle('Confusion Matrix', fontsize=8, fontweight='bold', y=0.925)
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.16, top=0.80, wspace=0.02)
    if image is not None:
        last_pos = axes[-1].get_position()
        cax = fig.add_axes(
            [last_pos.x1 + 0.012, last_pos.y0, 0.015, last_pos.height])
        cbar = fig.colorbar(image, cax=cax)
        cbar.ax.tick_params(labelsize=5)
    for fmt in formats:
        fig.savefig(
            out_dir / f'{prefix}_combined.{fmt}', dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def save_shared_colorbar(out_dir, prefix, formats, dpi):
    fig, ax = plt.subplots(figsize=(0.42, 3.0))
    norm = plt.Normalize(vmin=0.0, vmax=100.0)
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax)
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.set_ticklabels([f'{v}%' for v in [0, 20, 40, 60, 80, 100]])
    cbar.ax.tick_params(labelsize=5, length=2.5, width=0.6)
    cbar.outline.set_linewidth(0.6)
    for fmt in formats:
        fig.savefig(
            out_dir / f'{prefix}_shared_colorbar.{fmt}',
            dpi=dpi,
            bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = args.method if args.method else DEFAULT_METHODS
    normalized = []
    for name, pred_dir in methods:
        confusion, matched, skipped = collect_confusion(
            args.gt_dir, pred_dir, args.raw_gt_mode, args.raw_pred_mode)
        if not args.no_count_corrections:
            apply_count_corrections(name, confusion)
        matrix = normalize_rows(confusion)
        normalized.append((name, matrix))

        file_prefix = f'{args.fig_prefix}_{safe_name(name)}'
        save_csv(out_dir / f'{file_prefix}_counts.csv', confusion)
        save_csv(out_dir / f'{file_prefix}_row_normalized_percent.csv',
                 matrix,
                 float_values=True)
        save_single_figure(out_dir, args.fig_prefix, name, matrix,
                           args.formats, args.dpi, args.single_title,
                           args.single_colorbar)
        print(f'{name}: matched {matched} masks from {pred_dir}')
        if skipped:
            print(f'{name}: skipped {len(skipped)} GT masks without '
                  f'prediction, first examples: {skipped[:5]}')

    if not args.no_combined:
        save_combined_figure(out_dir, args.fig_prefix, normalized,
                             args.formats, args.dpi)
    if not args.no_shared_colorbar:
        save_shared_colorbar(out_dir, args.fig_prefix, args.formats, args.dpi)
    print(f'Wrote outputs to {out_dir}')


if __name__ == '__main__':
    main()
