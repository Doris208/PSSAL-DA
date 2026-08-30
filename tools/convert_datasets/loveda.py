# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import csv
import math
import os
import os.path as osp
import tempfile
import zipfile

import mmcv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert LoveDA dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='LoveDA folder path')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    parser.add_argument(
        '--clip_size',
        type=int,
        default=512,
        help='clipped size of image after preparation')
    parser.add_argument(
        '--stride_size',
        type=int,
        default=256,
        help='stride of clipping original images')
    return parser.parse_args()


def get_start_positions(length, clip_size, stride_size):
    if length <= clip_size:
        return np.array([0], dtype=np.int32)

    num_steps = math.ceil((length - clip_size) / stride_size) + 1
    starts = np.arange(0, num_steps * stride_size, stride_size, dtype=np.int32)
    starts = np.minimum(starts, length - clip_size)
    return np.unique(starts)


def clip_big_image(image_path,
                   clip_save_dir,
                   image_id,
                   to_label=False,
                   record_meta=False):
    if to_label:
        image = mmcv.imread(image_path, flag='unchanged')
        if image.ndim == 3:
            image = image[..., 0]
    else:
        image = mmcv.imread(image_path)

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
            end_x_val = start_x_val + cs
            end_y_val = start_y_val + cs

            if to_label:
                clipped_image = image[start_y_val:end_y_val,
                                      start_x_val:end_x_val]
            else:
                clipped_image = image[start_y_val:end_y_val,
                                      start_x_val:end_x_val, :]

            patch_filename = (
                f'{image_id}_{start_x_val}_{start_y_val}_{end_x_val}_{end_y_val}.png'
            )
            mmcv.imwrite(clipped_image.astype(np.uint8),
                         osp.join(clip_save_dir, patch_filename))

            if record_meta:
                position_type = 'middle'
                is_row_boundary = r == 0 or r == num_rows - 1
                is_col_boundary = c == 0 or c == num_cols - 1

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


def process_split(split_name, split_tmp_dir, out_dir, all_metadata):
    for domain_name in ['Urban', 'Rural']:
        domain = domain_name.lower()
        img_src_dir = osp.join(split_tmp_dir, split_name, domain_name,
                               'images_png')
        ann_src_dir = osp.join(split_tmp_dir, split_name, domain_name,
                               'masks_png')

        img_dst_dir = osp.join(out_dir, 'img_dir',
                               f'{split_name.lower()}_{domain}')
        ann_dst_dir = osp.join(out_dir, 'ann_dir',
                               f'{split_name.lower()}_{domain}')

        mmcv.mkdir_or_exist(img_dst_dir)
        if split_name != 'Test':
            mmcv.mkdir_or_exist(ann_dst_dir)

        img_files = sorted(
            f for f in os.listdir(img_src_dir)
            if osp.isfile(osp.join(img_src_dir, f)))

        prog_bar = mmcv.ProgressBar(len(img_files))
        for img_file in img_files:
            stem = osp.splitext(img_file)[0]
            image_id = f'{split_name.lower()}_{domain}_{stem}'
            img_path = osp.join(img_src_dir, img_file)

            meta_list = clip_big_image(
                img_path,
                img_dst_dir,
                image_id=image_id,
                to_label=False,
                record_meta=True)
            if meta_list:
                all_metadata.extend(meta_list)

            if split_name != 'Test':
                ann_path = osp.join(ann_src_dir, f'{stem}.png')
                clip_big_image(
                    ann_path,
                    ann_dst_dir,
                    image_id=image_id,
                    to_label=True,
                    record_meta=False)

            prog_bar.update()


def main():
    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = osp.join('data', 'loveda')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mmcv.mkdir_or_exist(out_dir)

    required_files = ['Train.zip', 'Val.zip', 'Test.zip']
    for file_name in required_files:
        if file_name not in os.listdir(dataset_path):
            raise FileNotFoundError(f'{file_name} is not in {dataset_path}')

    all_metadata = []

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        for split_name in ['Train', 'Val', 'Test']:
            zip_path = osp.join(dataset_path, f'{split_name}.zip')
            split_tmp_dir = osp.join(tmp_dir, split_name)
            mmcv.mkdir_or_exist(split_tmp_dir)

            print(f'Processing {zip_path}...')
            with zipfile.ZipFile(zip_path) as zip_file:
                zip_file.extractall(split_tmp_dir)

            process_split(split_name, split_tmp_dir, out_dir, all_metadata)

        print('Removing the temporary files...')

    if all_metadata:
        csv_path = osp.join(out_dir, 'patch_metadata.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'image_id', 'row', 'col', 'x1', 'y1', 'x2', 'y2',
                'position_type', 'filename'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metadata)
        print(f'Patch metadata saved to {csv_path}')
    else:
        print('No metadata recorded.')

    print('Done!')


if __name__ == '__main__':
    args = parse_args()
    main()
