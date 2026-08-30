import argparse
import json
import os
import os.path as osp
from collections import defaultdict

import mmcv
import numpy as np


DATASET_PRESETS = {
    'generic': {
        'ann_dir': 'ann_dir/train',
        'suffix': '.png',
        'reduce_zero_label': False,
        'num_classes': None,
    },
    'goji': {
        'ann_dir': 'ann_dir/train',
        'suffix': '.tif',
        'reduce_zero_label': True,
        'num_classes': 5,
    },
}

GOJI_AREA_NAMES = ['26', '30']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate RCS stats files for segmentation labels')
    parser.add_argument(
        '--data-root',
        required=True,
        help='Dataset root, where output json files will be written.')
    parser.add_argument(
        '--dataset',
        default='auto',
        choices=['auto', 'generic', 'goji'],
        help='Dataset preset. "goji" adapts defaults to the custom goji '
        'dataset structure. "auto" infers from the path.')
    parser.add_argument(
        '--ann-dir',
        default=None,
        help='Relative or absolute annotation directory. If omitted, uses the '
        'dataset preset default.')
    parser.add_argument(
        '--suffix',
        default=None,
        help='Annotation file suffix. If omitted, uses the dataset preset '
        'default.')
    parser.add_argument(
        '--reduce-zero-label',
        action='store_true',
        help='Apply reduce_zero_label transform before counting.')
    parser.add_argument(
        '--no-reduce-zero-label',
        action='store_true',
        help='Disable reduce_zero_label even if enabled by dataset preset.')
    parser.add_argument(
        '--num-classes',
        type=int,
        default=None,
        help='Optional class count limit after preprocessing (0..num_classes-1).'
    )
    parser.add_argument(
        '--areas',
        nargs='+',
        default=None,
        help='Optional sub-area names to process under data-root. Useful for '
        'the goji root that contains 26/ and 30/.')
    return parser.parse_args()


def infer_dataset_name(data_root):
    lowered = osp.abspath(data_root).lower()
    if 'mydataset' in lowered or any(
            f'{os.sep}{area}{os.sep}' in lowered or lowered.endswith(
                f'{os.sep}{area}') for area in GOJI_AREA_NAMES):
        return 'goji'
    return 'generic'


def resolve_preset_args(args):
    dataset_name = infer_dataset_name(
        args.data_root) if args.dataset == 'auto' else args.dataset
    preset = DATASET_PRESETS[dataset_name]

    ann_dir = args.ann_dir if args.ann_dir is not None else preset['ann_dir']
    suffix = args.suffix if args.suffix is not None else preset['suffix']
    reduce_zero_label = preset['reduce_zero_label']
    if args.reduce_zero_label:
        reduce_zero_label = True
    if args.no_reduce_zero_label:
        reduce_zero_label = False
    num_classes = args.num_classes if args.num_classes is not None else \
        preset['num_classes']

    return dataset_name, ann_dir, suffix, reduce_zero_label, num_classes


def preprocess_label(label, reduce_zero_label=False):
    label = label.astype(np.int32)
    if reduce_zero_label:
        # Same convention as MMSeg LoadAnnotations(reduce_zero_label=True):
        # 0 -> 255(ignore), 1..K -> 0..K-1
        zero_mask = label == 0
        label = label - 1
        label[zero_mask] = 255
    return label


def collect_stats(ann_dir, suffix, reduce_zero_label=False, num_classes=None):
    ann_files = sorted(
        f for f in os.listdir(ann_dir)
        if osp.isfile(osp.join(ann_dir, f)) and f.endswith(suffix))
    if not ann_files:
        raise FileNotFoundError(f'No annotation files found in: {ann_dir}')

    sample_class_stats = []
    samples_with_class = defaultdict(list)

    for fname in ann_files:
        path = osp.join(ann_dir, fname)
        label = mmcv.imread(path, flag='unchanged')
        if label is None:
            continue
        if label.ndim == 3:
            # Safety fallback: if image is RGB, use first channel.
            label = label[..., 0]
        label = preprocess_label(label, reduce_zero_label=reduce_zero_label)

        classes, counts = np.unique(label, return_counts=True)
        file_stats = {'file': fname}
        for c, n in zip(classes.tolist(), counts.tolist()):
            if c == 255:
                continue
            if num_classes is not None and (c < 0 or c >= num_classes):
                continue
            if c < 0:
                continue
            file_stats[int(c)] = int(n)
            samples_with_class[int(c)].append((fname, int(n)))
        sample_class_stats.append(file_stats)

    return sample_class_stats, samples_with_class


def dump_stats(out_root, sample_class_stats, samples_with_class):
    out_stats = osp.join(out_root, 'sample_class_stats.json')
    out_samples = osp.join(out_root, 'samples_with_class.json')

    with open(out_stats, 'w', encoding='utf-8') as f:
        json.dump(sample_class_stats, f, indent=2)

    with open(out_samples, 'w', encoding='utf-8') as f:
        json.dump(samples_with_class, f, indent=2)

    print(f'Processed: {len(sample_class_stats)} files')
    print(f'Wrote: {out_stats}')
    print(f'Wrote: {out_samples}')


def analyze_single_root(data_root, ann_dir, suffix, reduce_zero_label,
                        num_classes):
    resolved_ann_dir = ann_dir
    if not osp.isabs(resolved_ann_dir):
        resolved_ann_dir = osp.join(data_root, resolved_ann_dir)

    sample_class_stats, samples_with_class = collect_stats(
        ann_dir=resolved_ann_dir,
        suffix=suffix,
        reduce_zero_label=reduce_zero_label,
        num_classes=num_classes)
    dump_stats(data_root, sample_class_stats, samples_with_class)


def main():
    args = parse_args()
    dataset_name, ann_dir, suffix, reduce_zero_label, num_classes = \
        resolve_preset_args(args)

    data_root = osp.abspath(args.data_root)
    areas = args.areas
    if areas is None and dataset_name == 'goji' and \
            not osp.isdir(osp.join(data_root, ann_dir)):
        candidate_areas = [
            area for area in GOJI_AREA_NAMES if osp.isdir(osp.join(data_root, area))
        ]
        if candidate_areas:
            areas = candidate_areas

    if areas:
        for area in areas:
            area_root = osp.join(data_root, area)
            if not osp.isdir(area_root):
                raise FileNotFoundError(f'Area root not found: {area_root}')
            print(f'Analyzing area: {area}')
            analyze_single_root(area_root, ann_dir, suffix, reduce_zero_label,
                                num_classes)
    else:
        analyze_single_root(data_root, ann_dir, suffix, reduce_zero_label,
                            num_classes)


if __name__ == '__main__':
    main()
