dataset_type = 'LoveDADataset'
source_data_root = '/root/shared-nvme/LoveDA/'
target_data_root = '/root/shared-nvme/LoveDA/'

source_img_norm_cfg = dict(
    mean=[75.274, 79.875, 78.403],
    std=[40.727, 37.510, 36.436],
    to_rgb=True)

target_img_norm_cfg = dict(
    mean=[73.908, 80.458, 75.002],
    std=[41.229, 35.252, 33.359],
    to_rgb=True)

crop_size = (512, 512)

source_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.8, 1.2)),
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
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.8, 1.2)),
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
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
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
            img_dir='img_dir/train_urban',
            ann_dir='ann_dir/train_urban',
            pipeline=source_train_pipeline),
        target=dict(
            type=dataset_type,
            data_root=target_data_root,
            img_dir='img_dir/train_rural',
            ann_dir='ann_dir/train_rural',
            pipeline=target_train_pipeline),
    ),
    val=dict(
        type=dataset_type,
        data_root=target_data_root,
        img_dir='img_dir/val_rural',
        ann_dir='ann_dir/val_rural',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=target_data_root,
        img_dir='img_dir/val_rural',
        ann_dir='ann_dir/val_rural',
        pipeline=test_pipeline)
)
