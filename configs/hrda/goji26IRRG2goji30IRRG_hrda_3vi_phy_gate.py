_base_ = ['./goji26IRRG2goji30IRRG_hrda_3vi.py']

crop_size = (256, 256)
vi_names = ['ndvi', 'ndre', 'cire']
vi_norm_cfg = dict(
    ndvi=dict(type='range', min=-1.0, max=1.0),
    ndre=dict(type='range', min=-1.0, max=1.0),
    cire=dict(type='range', min=0.0, max=5.0))

source_img_norm_cfg = dict(
    mean=[111.366, 122.724, 135.261, 0.0, 0.0, 0.0],
    std=[61.302, 62.369, 55.989, 1.0, 1.0, 1.0],
    to_rgb=False,
    channel_order=[2, 1, 0, 3, 4, 5],
    vis_channel_order=[0, 1, 2])
target_img_norm_cfg = dict(
    mean=[140.700, 144.074, 151.917, 0.0, 0.0, 0.0],
    std=[65.998, 63.564, 53.397, 1.0, 1.0, 1.0],
    to_rgb=False,
    channel_order=[2, 1, 0, 3, 4, 5],
    vis_channel_order=[0, 1, 2])

source_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadGojiIndices'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(256, 256), ratio_range=(0.8, 1.2)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='Normalize', **source_img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'vi_img']),
]
target_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadGojiIndices'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(256, 256), ratio_range=(0.8, 1.2)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='Normalize', **target_img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'vi_img']),
]

uda = dict(
    phy_cfg=dict(
        enable=True,
        vi_names=vi_names,
        phy_positive_classes=[0, 1],
        lambda_phy_src=0.1,
        lambda_phy_tgt=0.01,
        use_target_gate=True,
        phy_min_weight=0.85,
        phy_warmup_iters=6000,
        apply_to_classes=[0, 1],
        detach_target_gate=True,
        save_phy_debug=True,
        phy_debug_interval=1000,
        phy_head=dict(in_channels=3, mid_channels=32, num_layers=2)))

data = dict(
    train=dict(
        source=dict(
            use_vi=True,
            vi_names=vi_names,
            vi_as_input=True,
            vi_norm_for_input=False,
            vi_norm_cfg=vi_norm_cfg,
            pipeline=source_train_pipeline),
        target=dict(
            use_vi=True,
            vi_names=vi_names,
            vi_as_input=True,
            vi_norm_for_input=False,
            vi_norm_cfg=vi_norm_cfg,
            crop_pseudo_margins=[30, 30, 30, 30],
            pipeline=target_train_pipeline)))

runner = dict(type='IterBasedRunner', max_iters=15000)
work_dir = '/root/shared-nvme/work_dirs/goji26IRRG2goji30IRRG_hrda_3vi_phy_gate_v2'
name = 'goji26IRRG2goji30IRRG_hrda_3vi_phy_gate_v2'
exp = 'goji_uda_indices_phy'
name_dataset = 'goji26IRRG2goji30IRRG_256x256_3vi_phy_gate'
name_uda = 'dacs_3vi_phy_gate_v2'
name_opt = 'adamw_6e-05_poly10warm_15k'
