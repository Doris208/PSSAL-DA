custom_imports = dict(
    imports=[
        'mmseg.models.segmentors.memory_adapt_encoder_decoder',
        'mmseg.models.uda.memory_adaptnet',
        'mmseg.models.utils.domain_prototype_memory',
        'mmseg.models.utils.memory_attention',
        'mmseg.models.discriminators.fc_discriminator',
        'mmseg.core.optimizer.memoryadaptnet_optimizer_hook',
    ],
    allow_failed_imports=False)

seed = 0

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='MemoryAdaptEncoderDecoder',
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
    stage1_head=dict(
        in_channels=2048,
        channels=256,
        num_classes=5,
        dilations=(6, 12, 18, 24)),
    memory_bottleneck=dict(
        in_channels=2048,
        out_channels=1024,
        kernel_size=3,
        stride=1,
        padding=1),
    memory_module=dict(
        type='DomainPrototypeMemory',
        num_classes=5,
        memory_dim=1024,
        transform_channels=512,
        num_prototypes_per_class=1,
        output_channels=2048,
        use_hard_aggregate=False,
        update_strategy='cosine_similarity',
        base_momentum=0.9,
        base_lr=0.01,
        adjust_by_learning_rate=True),
    stage2_head=dict(
        in_channels=2048,
        num_classes=5),
    train_cfg=dict(),
    test_cfg=dict(
        mode='slide',
        stride=(128, 128),
        crop_size=(256, 256)))

uda = dict(
    type='MemoryAdaptNet',
    num_classes=5,
    ignore_index=255,
    memory_start_iter=3000,
    target_entropy_threshold=0.3,
    entropy_eps=1e-30,
    lambda_stage1=1.0,
    lambda_stage2=1.0,
    lambda_adv=0.001,
    source_domain_label=0.0,
    target_domain_label=1.0,
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

method_name = 'memoryadaptnet'
name_encoder = 'r101'
name_architecture = 'memoryadaptnet_r101_3vi'
name_decoder = 'memory_deeplab'
name_uda = 'memoryadaptnet'
name_opt = 'adamw_6e-05_poly10warm_15k'
