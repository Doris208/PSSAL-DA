_base_ = ['../hrda/goji30IRRG2goji26IRRG_hrda_3vi.py']

model = dict(
    backbone=dict(
        _delete_=True,
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        pretrained='/root/HRDA-master/pretrained/resnet101-5d3b4d8f.pth',
        in_channels=6),
    decode_head=dict(
        in_channels=[256, 512, 1024, 2048]))

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

data = dict(
    train=dict(target=dict(crop_pseudo_margins=[0, 0, 0, 0])))

work_dir = '/root/shared-nvme/work_dirs/mic_hrda_r101_new/goji30IRRG2goji26IRRG_mic_hrda'
name = 'goji30IRRG2goji26IRRG_mic_hrda'
name_architecture = 'hrda1-128-0.1_daformer_sepaspp_r101'
name_encoder = 'r101'
name_uda = 'dacs_mic_hrda'
runner = dict(type='IterBasedRunner', max_iters=15000)
name_opt = 'adamw_6e-05_poly10warm_15k'
