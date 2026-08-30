# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import csv
import glob
import math
import os
import os.path as osp

import mmcv
import numpy as np
from PIL import Image


AREA_SPLITS = {
    '26': {
        'train': ['1', '2', '5', '8', '9'],
        'val': ['3', '7'],
        'test': ['3', '7'],
    },
    '30': {
        'train': ['2', '4', '11'],
        'val': ['3', '9'],
        'test': ['3', '9'],
    },
}

MODALITY_DIRS = {
    'irrg': 'img_dir',
    'label': 'ann_dir',
    'rgb': 'rgb_dir',
    'ndvi': 'ndvi_dir',
}

EXTRA_INDEX_DIRS = {
    'ndre': 'ndre_dir',
    'cire': 'cire_dir',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert goji dataset to mmsegmentation format')
    parser.add_argument(
        '--dataset_path',
        default='shared-nvme/mydataset',
        help='path to the prepared goji dataset root')
    parser.add_argument(
        '-o',
        '--out_dir',
        default=None,
        help='optional output root; area folders will be created under it')
    parser.add_argument(
        '--clip_size',
        type=int,
        help='clipped size of image after preparation',
        default=256)
    parser.add_argument(
        '--stride_size',
        type=int,
        help='stride of clipping original images',
        default=128)
    parser.add_argument(
        '--only_extra_indices',
        action='store_true',
        help='only clip supplemental vegetation indices: ndre and cire')
    args = parser.parse_args()
    return args


def load_tif(image_path):
    image = np.array(Image.open(image_path))
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    return image


def save_tif(image, save_path):
    Image.fromarray(image).save(save_path)


def get_start_positions(length, clip_size, stride_size):
    if length <= clip_size:
        return np.array([0], dtype=np.int32)

    num_steps = math.ceil((length - clip_size) / stride_size) + 1
    starts = np.arange(0, num_steps * stride_size, stride_size, dtype=np.int32)
    starts = np.minimum(starts, length - clip_size)
    return np.unique(starts)


def extract_image_id(file_path):
    stem = osp.splitext(osp.basename(file_path))[0]
    parts = stem.split('_')
    if len(parts) < 4:
        raise ValueError(
            f'Unexpected filename format for image id extraction: {file_path}')
    image_id = parts[3]
    if not image_id.isdigit():
        raise ValueError(
            f'Image id extracted from {file_path} is not numeric: {image_id}')
    return image_id


def build_patch_filename(image_id, x1, y1, x2, y2):
    return f'{image_id}_{x1}_{y1}_{x2}_{y2}.tif'


def remap_label(label):
    remapped = np.full(label.shape, 255, dtype=np.uint8)
    valid_mask = (label >= 1) & (label <= 5)
    remapped[valid_mask] = label[valid_mask] - 1
    return remapped


def ensure_output_dirs(area_out_dir):
    for root_name in ['img_dir', 'ann_dir', 'rgb_dir', 'ndvi_dir']:
        for split in ['train', 'val', 'test']:
            mmcv.mkdir_or_exist(osp.join(area_out_dir, root_name, split))


def ensure_extra_index_output_dirs(area_out_dir):
    for root_name in EXTRA_INDEX_DIRS.values():
        for split in ['train', 'val', 'test']:
            mmcv.mkdir_or_exist(osp.join(area_out_dir, root_name, split))


def clip_big_image(image_path,
                   clip_save_dir,
                   image_id,
                   modality,
                   record_meta=False):
    image = load_tif(image_path)
    if modality == 'label':
        if image.ndim == 3:
            image = image[..., 0]
        image = remap_label(image.astype(np.uint8))

    h, w = image.shape[:2]
    cs = args.clip_size
    ss = args.stride_size

    start_x = get_start_positions(w, cs, ss)
    start_y = get_start_positions(h, cs, ss)
    num_cols = len(start_x)
    num_rows = len(start_y)

    meta_list = [] if record_meta else None

    for r in range(num_rows):
        for c in range(num_cols):
            start_x_val = int(start_x[c])
            start_y_val = int(start_y[r])
            end_x_val = int(start_x_val + cs)
            end_y_val = int(start_y_val + cs)

            if image.ndim == 2:
                clipped_image = image[start_y_val:end_y_val, start_x_val:end_x_val]
            else:
                clipped_image = image[start_y_val:end_y_val, start_x_val:end_x_val, :]

            patch_filename = build_patch_filename(image_id, start_x_val,
                                                  start_y_val, end_x_val,
                                                  end_y_val)
            save_tif(clipped_image, osp.join(clip_save_dir, patch_filename))

            if record_meta:
                position_type = 'middle'
                is_row_boundary = (r == 0 or r == num_rows - 1)
                is_col_boundary = (c == 0 or c == num_cols - 1)

                if is_row_boundary and is_col_boundary:
                    position_type = 'corner'
                elif is_row_boundary or is_col_boundary:
                    position_type = 'edge'

                meta_list.append({
                    'image_id': image_id,
                    'row': r,
                    'col': c,
                    'x1': start_x_val,
                    'y1': start_y_val,
                    'x2': end_x_val,
                    'y2': end_y_val,
                    'position_type': position_type,
                    'filename': patch_filename
                })

    return meta_list


def collect_area_files(area_dir, modality):
    src_dir = osp.join(area_dir, modality)
    src_files = sorted(glob.glob(osp.join(src_dir, '*.tif')))
    if not src_files:
        raise FileNotFoundError(f'No tif files found in {src_dir}')
    return {extract_image_id(path): path for path in src_files}


def write_split_file(area_out_dir, split, filenames):
    split_path = osp.join(area_out_dir, f'{split}.txt')
    with open(split_path, 'w', encoding='utf-8') as f:
        for filename in filenames:
            f.write(f'{filename}\n')


def process_area(area_name, area_in_dir, area_out_dir):
    if area_name not in AREA_SPLITS:
        raise KeyError(f'Unsupported area: {area_name}')

    ensure_output_dirs(area_out_dir)

    modality_files = {
        modality: collect_area_files(area_in_dir, modality)
        for modality in MODALITY_DIRS
    }

    irrg_ids = sorted(modality_files['irrg'].keys(), key=lambda item: int(item))
    for modality, file_dict in modality_files.items():
        missing_ids = sorted(set(irrg_ids) - set(file_dict.keys()),
                             key=lambda item: int(item))
        extra_ids = sorted(set(file_dict.keys()) - set(irrg_ids),
                           key=lambda item: int(item))
        if missing_ids or extra_ids:
            raise ValueError(
                f'Area {area_name} has mismatched ids in {modality}. '
                f'Missing: {missing_ids}, extra: {extra_ids}')

    metadata_by_split = []
    split_basenames = {split: [] for split in ['train', 'val', 'test']}

    for split, split_ids in AREA_SPLITS[area_name].items():
        prog_bar = mmcv.ProgressBar(len(split_ids))
        for image_id in split_ids:
            if image_id not in modality_files['irrg']:
                raise KeyError(
                    f'Image id {image_id} required by area {area_name} '
                    'split definition is missing from irrg/')

            for modality, output_dir_name in MODALITY_DIRS.items():
                dst_dir = osp.join(area_out_dir, output_dir_name, split)
                src_path = modality_files[modality][image_id]
                meta_list = clip_big_image(
                    src_path,
                    dst_dir,
                    image_id=image_id,
                    modality=modality,
                    record_meta=(modality == 'irrg'))
                if meta_list:
                    for meta in meta_list:
                        split_basenames[split].append(
                            osp.splitext(meta['filename'])[0])
                        meta['split'] = split
                        metadata_by_split.append(meta)
            prog_bar.update()

    for split in ['train', 'val', 'test']:
        unique_names = sorted(set(split_basenames[split]))
        write_split_file(area_out_dir, split, unique_names)

    csv_path = osp.join(area_out_dir, 'patch_metadata.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'split', 'image_id', 'row', 'col', 'x1', 'y1', 'x2', 'y2',
            'position_type', 'filename'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_by_split)


def process_extra_indices_only(area_name, area_in_dir, area_out_dir):
    if area_name not in AREA_SPLITS:
        raise KeyError(f'Unsupported area: {area_name}')

    ensure_extra_index_output_dirs(area_out_dir)

    index_files = {
        modality: collect_area_files(area_in_dir, modality)
        for modality in EXTRA_INDEX_DIRS
    }

    for split, split_ids in AREA_SPLITS[area_name].items():
        prog_bar = mmcv.ProgressBar(len(split_ids))
        for image_id in split_ids:
            for modality, file_dict in index_files.items():
                if image_id not in file_dict:
                    raise KeyError(
                        f'Image id {image_id} required by area {area_name} '
                        f'split definition is missing from {modality}/')

                dst_dir = osp.join(area_out_dir, EXTRA_INDEX_DIRS[modality],
                                   split)
                clip_big_image(
                    file_dict[image_id],
                    dst_dir,
                    image_id=image_id,
                    modality=modality)
            prog_bar.update()


def main():
    dataset_path = args.dataset_path
    base_out_dir = args.out_dir if args.out_dir is not None else dataset_path

    for area_name in AREA_SPLITS:
        area_in_dir = osp.join(dataset_path, area_name)
        area_out_dir = osp.join(base_out_dir, area_name)
        if not osp.isdir(area_in_dir):
            raise FileNotFoundError(f'Area directory does not exist: {area_in_dir}')
        if args.only_extra_indices:
            process_extra_indices_only(area_name, area_in_dir, area_out_dir)
        else:
            process_area(area_name, area_in_dir, area_out_dir)

    print('Done!')


if __name__ == '__main__':
    args = parse_args()
    main()
