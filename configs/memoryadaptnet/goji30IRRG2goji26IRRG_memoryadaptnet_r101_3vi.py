_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/uda_goji30IRRG_to_goji26IRRG_256x256_3vi.py',
    './_base_/memoryadaptnet_r101_3vi.py',
]

work_dir = '/root/shared-nvme/work_dirs/memoryadaptnet_r101/goji30IRRG2goji26IRRG_memoryadaptnet_r101_3vi'
name = 'goji30IRRG2goji26IRRG_memoryadaptnet_r101_3vi'
exp = 'goji_uda_indices'
name_dataset = 'goji30IRRG2goji26IRRG_256x256_3vi'
