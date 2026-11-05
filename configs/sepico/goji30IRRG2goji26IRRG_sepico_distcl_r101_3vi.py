_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/uda_goji30IRRG_to_goji26IRRG_256x256_3vi.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py',
    './sepico_distcl_r101_3vi_base.py',
]

seed = 0
crop_size = (256, 256)

data = dict(
    samples_per_gpu=2,
    train=dict(target=dict(crop_pseudo_margins=[30, 30, 30, 30])),
    workers_per_gpu=4)

optimizer_config = None
optimizer = dict(lr=6e-5)

work_dir = '/root/shared-nvme/work_dirs/sepico_distcl_r101/goji30IRRG2goji26IRRG_sepico_distcl_r101_3vi'

name = 'goji30IRRG2goji26IRRG_sepico_distcl_r101_3vi'
name_dataset = 'goji30IRRG2goji26IRRG_256x256_3vi'
