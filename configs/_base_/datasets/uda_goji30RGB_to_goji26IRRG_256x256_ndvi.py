dataset_type = 'GojiDataset'
source_data_root = '/root/shared-nvme/mydataset/30'
target_data_root = '/root/shared-nvme/mydataset/26'
index_names = ['ndvi']

source_img_norm_cfg = dict(
    mean=[144.074, 140.700, 133.597, 0.0],
    std=[63.564, 65.998, 65.842, 1.0],
    to_rgb=False,
    channel_order=[2, 1, 0, 3],
    vis_channel_order=[0, 1, 2])

target_img_norm_cfg = dict(
    mean=[111.366, 122.724, 135.261, 0.0],
    std=[61.302, 62.369, 55.989, 1.0],
    to_rgb=False,
    channel_order=[2, 1, 0, 3],
    vis_channel_order=[0, 1, 2])

crop_size = (256, 256)

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
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
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
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadGojiIndices'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **target_img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type=dataset_type,
            data_root=source_data_root,
            img_dir='rgb_dir/train',
            ann_dir='ann_dir/train',
            use_indices=True,
            index_names=index_names,
            pipeline=source_train_pipeline),
        target=dict(
            type=dataset_type,
            data_root=target_data_root,
            img_dir='img_dir/train',
            ann_dir='ann_dir/train',
            use_indices=True,
            index_names=index_names,
            pipeline=target_train_pipeline),
    ),
    val=dict(
        type=dataset_type,
        data_root=target_data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        use_indices=True,
        index_names=index_names,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=target_data_root,
        img_dir='img_dir/test',
        ann_dir='ann_dir/test',
        use_indices=True,
        index_names=index_names,
        pipeline=test_pipeline))
