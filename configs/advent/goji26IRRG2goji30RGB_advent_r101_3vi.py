_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/uda_goji26IRRG_to_goji30RGB_256x256_3vi.py',
    './_base_/advent_r101_3vi.py',
]

work_dir = '/root/shared-nvme/work_dirs/advent_r101/goji26IRRG2goji30RGB_advent_r101_3vi'
name = 'goji26IRRG2goji30RGB_advent_r101_3vi'
exp = 'goji_uda_indices'
name_dataset = 'goji26IRRG2goji30RGB_256x256_3vi'
