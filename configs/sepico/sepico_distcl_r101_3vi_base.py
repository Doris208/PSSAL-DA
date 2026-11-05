norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='EncoderDecoderProjector',
    pretrained=None,
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        pretrained='/root/HRDA-master/pretrained/resnet101-5d3b4d8f.pth',
        in_channels=6),
    decode_head=dict(
        type='DLV2Head',
        in_channels=2048,
        in_index=3,
        dilations=(6, 12, 18, 24),
        num_classes=5,
        align_corners=False,
        init_cfg=dict(
            type='Normal', std=0.01, override=dict(name='aspp_modules')),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='ProjHead',
        in_channels=2048,
        in_index=3,
        channels=256,
        num_classes=5,
        num_convs=2,
        kernel_size=1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='ContrastiveLoss',
            use_dist=True,
            use_bank=False,
            use_reg=True,
            use_avg_pool=True,
            scale_min_ratio=0.75,
            num_classes=5,
            contrast_temp=100.0,
            loss_weight=1.0,
            reg_relative_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', stride=(128, 128), crop_size=(256, 256)))

uda = dict(
    type='SePiCo',
    alpha=0.999,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    debug_img_interval=1000,
    print_grad_magnitude=False,
    enable_self_training=True,
    enable_strong_aug=True,
    start_distribution_iter=3000,
    pseudo_random_crop=False,
    regen_pseudo=False,
    memory_length=0)

runner = dict(type='IterBasedRunner', max_iters=15000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=5)
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')

method_name = 'sepico'
exp = 'goji_uda_indices'
name_architecture = 'sepico_distcl_r101_3vi'
name_encoder = 'r101'
name_decoder = 'dlv2_proj'
name_uda = 'sepico_distcl'
name_opt = 'adamw_6e-05_poly10warm_15k'
use_uda = True
use_ddp_wrapper = True
