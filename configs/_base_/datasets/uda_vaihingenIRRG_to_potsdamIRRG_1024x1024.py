dataset_type = 'ISPRSDataset'
source_data_root = '/root/shared-nvme/ISPRS_dataset/Vaihingen/'
target_data_root = '/root/shared-nvme/ISPRS_dataset/Potsdam/IRRG/'

# Source domain (Vaihingen IRRG)
source_img_norm_cfg = dict(
    mean=[80.593, 81.922, 120.426],
    std=[37.444, 39.566, 54.791],
    to_rgb=False,
    # cv2 loads 3-channel images in BGR; remap to IRRG-like order [IR, R, G]
    channel_order=[2, 1, 0],
    # Keep debug visualization in IRRG color space (vegetation appears red)
    vis_channel_order=[0, 1, 2]
)

# Target domain (Potsdam IRRG)
target_img_norm_cfg = dict(
    mean=[84.907, 91.694, 97.018],
    std=[36.544, 35.182, 37.126],
    to_rgb=False,
    channel_order=[2, 1, 0],
    vis_channel_order=[0, 1, 2]
)

crop_size = (1024, 1024)

source_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(1024, 1024), ratio_range=(0.8, 1.2)),
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
    dict(type='Resize', img_scale=(1024, 1024), ratio_range=(0.8, 1.2)),
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
        img_scale=(1024, 1024),
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
            img_dir='img_dir/train',
            ann_dir='ann_dir/train',
            pipeline=source_train_pipeline),
        target=dict(
            type=dataset_type,
            data_root=target_data_root,
            img_dir='img_dir/train',
            ann_dir='ann_dir/train',
            pipeline=target_train_pipeline),
    ),
    val=dict(
        type=dataset_type,
        data_root=target_data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=target_data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=test_pipeline)
)
