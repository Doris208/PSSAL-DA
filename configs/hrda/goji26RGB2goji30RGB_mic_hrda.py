_base_ = ['./goji26RGB2goji30RGB_hrda_3vi.py']

model = dict(
    backbone=dict(
        pretrained='/root/HRDA-master/pretrained/resnet101-5d3b4d8f.pth'))

uda = dict(
    mask_mode='separatetrgaug',
    mask_alpha='same',
    mask_pseudo_threshold='same',
    mask_lambda=1.0,
    mask_generator=dict(
        type='block',
        mask_ratio=0.5,
        mask_block_size=32,
        _delete_=True))

work_dir = '/root/shared-nvme/work_dirs/mic_hrda/goji26RGB2goji30RGB_mic_hrda'
name = 'goji26RGB2goji30RGB_mic_hrda'
name_uda = 'dacs_mic_hrda'
runner = dict(type='IterBasedRunner', max_iters=15000)
name_opt = 'adamw_6e-05_poly10warm_15k'
