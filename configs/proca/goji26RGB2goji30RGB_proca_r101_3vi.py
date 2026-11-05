_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/uda_goji26RGB_to_goji30RGB_256x256_3vi.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py',
    './_base_/proca_r101_3vi.py',
]

seed = 0
crop_size = (256, 256)

data = dict(
    samples_per_gpu=2,
    train=dict(target=dict(crop_pseudo_margins=[30, 30, 30, 30])),
    workers_per_gpu=4)

optimizer_config = None
optimizer = dict(
    lr=6e-5,
    paramwise_cfg=dict(custom_keys=dict(decode_head=dict(lr_mult=10.0))))

work_dir = '/root/shared-nvme/work_dirs/proca_r101/goji26RGB2goji30RGB_proca_r101_3vi'

name = 'goji26RGB2goji30RGB_proca_r101_3vi'
name_dataset = 'goji26RGB2goji30RGB_256x256_3vi'
