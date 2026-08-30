# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import csv
import glob
import math
import os
import os.path as osp
import tempfile
import zipfile

import mmcv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert vaihingen dataset to mmsegmentation format')
    parser.add_argument('image_zip', help='path to the image zip file')
    parser.add_argument('label_zip', help='path to the label zip file')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    parser.add_argument(
        '--clip_size',
        type=int,
        help='clipped size of image after preparation',
        default=512)
    parser.add_argument(
        '--stride_size',
        type=int,
        help='stride of clipping original images',
        default=256)
    args = parser.parse_args()
    return args


def clip_big_image(image_path, clip_save_dir, to_label=False, record_meta=False):
    image = mmcv.imread(image_path)
    h, w, c = image.shape
    cs = args.clip_size
    ss = args.stride_size

    # 计算行数和列数（图块数量）
    num_rows = math.ceil((h - cs) / ss) + 1
    num_cols = math.ceil((w - cs) / ss) + 1

    # 生成所有起始坐标
    start_x = np.arange(0, num_cols * ss, ss)
    start_y = np.arange(0, num_rows * ss, ss)
    # 调整最后一个起始点，使其不超过图像边界
    start_x = np.minimum(start_x, w - cs)
    start_y = np.minimum(start_y, h - cs)
    # 去除可能因调整产生的重复值（当步长不能整除时，最后几个点可能相同）
    start_x = np.unique(start_x)
    start_y = np.unique(start_y)
    num_cols = len(start_x)
    num_rows = len(start_y)

    # 生成网格
    # xv, yv = np.meshgrid(start_x, start_y)
    # xv = xv.ravel()
    # yv = yv.ravel()
    # boxes = np.stack([xv, yv, xv + cs, yv + cs], axis=1)

    # 如果 to_label=True，将 RGB 标签转换为单通道类别索引
    if to_label:
        color_map = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0],
                              [255, 255, 0], [0, 255, 0], [0, 255, 255],
                              [0, 0, 255]])
        flatten_v = np.matmul(
            image.reshape(-1, c),
            np.array([2, 3, 4]).reshape(3, 1))
        out = np.zeros_like(flatten_v)
        for idx, class_color in enumerate(color_map):
            value_idx = np.matmul(class_color,
                                  np.array([2, 3, 4]).reshape(3, 1))
            out[flatten_v == value_idx] = idx
        image = out.reshape(h, w)

    image_id = osp.basename(image_path).split('_')[3].strip('.tif')
    meta_list = [] if record_meta else None

    # 使用双重循环直接生成行列索引
    for r in range(num_rows):
        for c in range(num_cols):
            # 获取当前 Patch 的左上角和右下角坐标
            start_x_val = int(start_x[c])
            start_y_val = int(start_y[r])
            end_x_val = int(start_x_val + cs)
            end_y_val = int(start_y_val + cs)

            # 裁剪图像
            if to_label:
                clipped_image = image[start_y_val:end_y_val, start_x_val:end_x_val]
            else:
                clipped_image = image[start_y_val:end_y_val, start_x_val:end_x_val, :]

            # 生成文件名
            patch_filename = f'{image_id}_{start_x_val}_{start_y_val}_{end_x_val}_{end_y_val}.png'
            mmcv.imwrite(clipped_image.astype(np.uint8),
                         osp.join(clip_save_dir, patch_filename))

            if record_meta:
                # 判定位置类型
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
def main():
    splits = {
        'train': [
            'area1', 'area11', 'area13', 'area15', 'area17', 'area21',
            'area23', 'area26', 'area28', 'area3', 'area30', 'area32',
            'area34', 'area37', 'area5', 'area7'
        ],
        'val': [
            'area6', 'area24', 'area35', 'area16', 'area14', 'area22',
            'area10', 'area4', 'area2', 'area20', 'area8', 'area31', 'area33',
            'area27', 'area38', 'area12', 'area29'
        ],
    }

    if args.out_dir is None:
        out_dir = osp.join('data', 'vaihingen')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))

    zip_files = [args.image_zip, args.label_zip]
    for zf in zip_files:
        if not osp.isfile(zf):
            print(f'Error: {zf} does not exist.')
            return

    # 用于收集所有 patch 的元数据（仅影像）
    all_metadata = []

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        for zipp in zip_files:
            is_label_zip = 'ground_truth' in zipp and 'COMPLETE' in zipp
            print(f'Processing {zipp} ({"label" if is_label_zip else "image"})...')

            zip_file = zipfile.ZipFile(zipp)
            zip_file.extractall(tmp_dir)

            if is_label_zip:
                src_path_list = glob.glob(os.path.join(tmp_dir, '*.tif'))
                src_path_list = [p for p in src_path_list if 'area9' not in p]
            else:
                src_path_list = glob.glob(os.path.join(tmp_dir, 'top', '*.tif'))
                if not src_path_list:
                    src_path_list = glob.glob(os.path.join(tmp_dir, '*.tif'))
                    print('Warning: "top" subdirectory not found, using files in root.')

            prog_bar = mmcv.ProgressBar(len(src_path_list))
            for src_path in src_path_list:
                area_idx = osp.basename(src_path).split('_')[3].strip('.tif')
                data_type = 'train' if area_idx in splits['train'] else 'val'

                if is_label_zip:
                    dst_dir = osp.join(out_dir, 'ann_dir', data_type)
                    # 标签文件：不记录元数据（record_meta=False）
                    clip_big_image(src_path, dst_dir, to_label=True, record_meta=False)
                else:
                    dst_dir = osp.join(out_dir, 'img_dir', data_type)
                    # 影像文件：记录元数据
                    meta_list = clip_big_image(src_path, dst_dir, to_label=False, record_meta=True)
                    if meta_list:
                        all_metadata.extend(meta_list)

                prog_bar.update()

        print('Removing the temporary files...')

    # 将所有元数据写入 CSV
    if all_metadata:
        csv_path = osp.join(out_dir, 'patch_metadata.csv')
        with open(csv_path, 'w', newline='') as f:
            # 按要求的字段顺序写入
            fieldnames = ['image_id', 'row', 'col', 'x1', 'y1', 'x2', 'y2', 'position_type', 'filename']
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