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
        description='Convert potsdam dataset to mmsegmentation format')
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
    """
    Clip a large image into patches of size clip_size x clip_size with given stride.
    If to_label=True, convert RGB label image to single-channel class indices.
    If record_meta=True, return a list of metadata for each patch.
    """
    image = mmcv.imread(image_path)
    h, w, c = image.shape
    cs = args.clip_size
    ss = args.stride_size

    # 计算行数和列数（图块数量）——与 vaihingen.py 完全一致
    num_rows = math.ceil((h - cs) / ss) + 1
    num_cols = math.ceil((w - cs) / ss) + 1

    # 生成所有起始坐标
    start_x = np.arange(0, num_cols * ss, ss)
    start_y = np.arange(0, num_rows * ss, ss)
    # 调整最后一个起始点，使其不超过图像边界
    start_x = np.minimum(start_x, w - cs)
    start_y = np.minimum(start_y, h - cs)
    # 去除可能因调整产生的重复值（当步长能整除时，最后几个点可能相同）
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
        # Potsdam 的颜色映射（与 Vaihingen 相同）
        color_map = np.array([[0, 0, 0],          # 忽略类
                               [255, 255, 255],    # 不透水面
                               [255, 0, 0],        # 建筑
                               [255, 255, 0],      # 低植被
                               [0, 255, 0],        # 树木
                               [0, 255, 255],      # 汽车
                               [0, 0, 255]])       # 背景/杂波
        flatten_v = np.matmul(
            image.reshape(-1, c),
            np.array([2, 3, 4]).reshape(3, 1))
        out = np.zeros_like(flatten_v)
        for idx, class_color in enumerate(color_map):
            value_idx = np.matmul(class_color,
                                  np.array([2, 3, 4]).reshape(3, 1))
            out[flatten_v == value_idx] = idx
        image = out.reshape(h, w)

    # 从文件名提取区域标识符（Potsdam 格式：top_potsdam_2_10_IRRG.tif → 取第 3、4 部分）
    parts = osp.basename(image_path).split('_')
    image_id = f"{parts[2]}_{parts[3]}"

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
    # Potsdam 训练/验证集划分（与原 potsdam.py 保持一致）
    splits = {
        'train': [
            '2_10', '2_11', '2_12', '3_10', '3_11', '3_12', '4_10', '4_11',
            '4_12', '5_10', '5_11', '5_12', '6_10', '6_11', '6_12', '6_7',
            '6_8', '6_9', '7_10', '7_11', '7_12', '7_7', '7_8', '7_9'
        ],
        'val': [
            '5_15', '6_15', '6_13', '3_13', '4_14', '6_14', '5_14', '2_13',
            '4_15', '2_14', '5_13', '4_13', '3_14', '7_13'
        ]
    }

    if args.out_dir is None:
        out_dir = osp.join('data', 'potsdam')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))

    # 检查两个 ZIP 文件是否存在
    zip_files = [args.image_zip, args.label_zip]
    for zf in zip_files:
        if not osp.isfile(zf):
            print(f'Error: {zf} does not exist.')
            return

    # 用于收集所有 patch 的元数据（仅影像）
    all_metadata = []

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        for zipp in zip_files:
            # 根据文件名判断是否为标签压缩包（与 vaihingen.py 类似，使用关键词）
            is_label_zip = 'label' in osp.basename(zipp).lower()
            print(f'Processing {zipp} ({"label" if is_label_zip else "image"})...')

            zip_file = zipfile.ZipFile(zipp)
            zip_file.extractall(tmp_dir)

            # 搜索所有 .tif 文件
            src_path_list = glob.glob(os.path.join(tmp_dir, '*.tif'))
            if not src_path_list:
                # 如果根目录没有，进入第一个子目录查找
                sub_tmp_dir = os.path.join(tmp_dir, os.listdir(tmp_dir)[0])
                src_path_list = glob.glob(os.path.join(sub_tmp_dir, '*.tif'))

            prog_bar = mmcv.ProgressBar(len(src_path_list))
            for src_path in src_path_list:
                # 提取区域标识符（如 "2_10"）
                parts = osp.basename(src_path).split('_')
                idx_i, idx_j = parts[2], parts[3]
                area_id = f'{idx_i}_{idx_j}'
                data_type = 'train' if area_id in splits['train'] else 'val'

                if is_label_zip:
                    dst_dir = osp.join(out_dir, 'ann_dir', data_type)
                    # 标签文件：不记录元数据
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