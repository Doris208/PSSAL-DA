custom_imports = dict(
    imports=[
        'mmseg.models.uda.advent',
        'mmseg.models.discriminators.fc_discriminator',
        'mmseg.models.utils.advent_entropy',
        'mmseg.core.optimizer.advent_optimizer_hook',
    ],
    allow_failed_imports=False)

seed = 0

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
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
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=5,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(
        mode='slide',
        stride=(128, 128),
        crop_size=(256, 256)))

uda = dict(
    type='AdvEnt',
    num_classes=5,
    ignore_index=255,
    multi_level=True,
    lambda_seg_main=1.0,
    lambda_seg_aux=0.1,
    lambda_adv_main=0.001,
    lambda_adv_aux=0.0002,
    source_domain_label=0.0,
    target_domain_label=1.0,
    entropy_eps=1e-30,
    normalize_entropy=True,
    discriminator=dict(
        type='FCDiscriminator',
        in_channels=5,
        base_channels=64,
        num_layers=5,
        negative_slope=0.2),
    discriminator_optimizer=dict(
        type='Adam',
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0.0))

optimizer_config = None
optimizer = dict(
    type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01)
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

runner = dict(type='IterBasedRunner', max_iters=15000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=5)
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')
custom_hooks = [dict(type='AdvEntOptimizerStateHook', interval=1000)]

method_name = 'advent'
name_encoder = 'r101'
name_architecture = 'advent_r101_3vi'
name_decoder = 'dlv2'
name_uda = 'advent'
name_opt = 'adamw_6e-05_poly10warm_15k'
