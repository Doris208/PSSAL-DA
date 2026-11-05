custom_imports = dict(
    imports=[
        'mmseg.models.uda.proca',
        'mmseg.models.utils.proca_prototype_estimator',
        'mmseg.models.losses.proca_prototype_contrastive_loss',
    ],
    allow_failed_imports=False)

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
    train_cfg=dict(),
    test_cfg=dict(mode='slide', stride=(128, 128), crop_size=(256, 256)))

uda = dict(
    type='ProCA',
    num_classes=5,
    ignore_index=255,
    feature_index=3,
    feature_dim=2048,
    multi_level=True,
    target_pseudo_threshold=0.9,
    temperature=50.0,
    lambda_feat=1.0,
    lambda_out=1.0,
    lambda_ssl=1.0,
    use_momentum=False,
    prototype_momentum=0.9,
    init_feature_prototype=None,
    init_output_prototype=None,
    proca_stage='adapt',
    ssl_pseudo_label_root=None)

proca_pipeline = dict(
    threshold_percent=0.6,
    threshold_upper_bound=0.9,
    pseudo_label_ignore_index=255,
    training_iters=15000,
    baseline_total_iters=15000)

runner = dict(type='IterBasedRunner', max_iters=15000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=5)
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')

method_name = 'proca'
exp = 'goji_uda_indices'
name_architecture = 'proca_r101_3vi'
name_encoder = 'r101'
name_decoder = 'dlv2'
name_uda = 'proca'
name_opt = 'adamw_6e-05_poly10warm_15k'
use_uda = True
use_ddp_wrapper = True
find_unused_parameters = True
