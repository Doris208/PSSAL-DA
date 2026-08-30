use_uda = True

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/daformer_sepaspp_mitb5.py',
    '../_base_/uda/dacs.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py',
]

seed = 0
crop_size = (256, 256)
dataset_type = 'GojiDataset'
source_data_root = '/root/shared-nvme/mydataset/26'
target_data_root = '/root/shared-nvme/mydataset/30'
index_names = ['ndvi', 'ndre', 'cire']
vi_names = index_names
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
    mean=[144.074, 140.700, 133.597, 0.0, 0.0, 0.0],
    std=[63.564, 65.998, 65.842, 1.0, 1.0, 1.0],
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

model = dict(
    type='HRDAEncoderDecoder',
    pretrained=None,
    backbone=dict(
        pretrained='/root/HRDA-master/pretrained/mit_b5.pth',
        in_chans=6),
    decode_head=dict(
        type='HRDAHead',
        single_scale_head='DAFormerHead',
        attention_classwise=True,
        hr_loss_weight=0.1,
        num_classes=5),
    scales=[1, 0.5],
    hr_crop_size=[128, 128],
    feature_scale=0.5,
    crop_coord_divisible=8,
    hr_slide_inference=True,
    test_cfg=dict(
        mode='slide',
        batched_slide=True,
        stride=[128, 128],
        crop_size=[256, 256]))

uda = dict(
    type='DACS',
    use_mixing=True,
    alpha=0.999,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    imnet_feature_dist_lambda=0.0,
    imnet_feature_dist_classes=None,
    imnet_feature_dist_scale_min_ratio=None,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    debug_img_interval=1000,
    print_grad_magnitude=False,
    input_channel_order=['B', 'G', 'R', 'NDVI', 'NDRE', 'CIRE'],
    class_names=['goji', 'vegetation', 'building', 'water', 'road'],
    reduced_label_note='raw labels 1..5 are reduced to 0..4; ignore=255',
    phy_cfg=dict(enable=False),
    struct_cfg=dict(enable=False),
    proto_cfg=dict(enable=False),
    struct_edge_cfg=dict(enable=False),
    joint_gate_cfg=dict(enable=False))

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
            use_indices=True,
            index_names=index_names,
            use_vi=True,
            vi_names=vi_names,
            vi_as_input=True,
            vi_norm_for_input=False,
            vi_norm_cfg=vi_norm_cfg,
            pipeline=source_train_pipeline),
        target=dict(
            type=dataset_type,
            data_root=target_data_root,
            img_dir='rgb_dir/train',
            ann_dir='ann_dir/train',
            use_indices=True,
            index_names=index_names,
            use_vi=True,
            vi_names=vi_names,
            vi_as_input=True,
            vi_norm_for_input=False,
            vi_norm_cfg=vi_norm_cfg,
            crop_pseudo_margins=[30, 30, 30, 30],
            pipeline=target_train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=target_data_root,
        img_dir='rgb_dir/val',
        ann_dir='ann_dir/val',
        use_indices=True,
        index_names=index_names,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=target_data_root,
        img_dir='rgb_dir/test',
        ann_dir='ann_dir/test',
        use_indices=True,
        index_names=index_names,
        pipeline=test_pipeline))

optimizer_config = None
fp16 = dict()
optimizer = dict(
    lr=6e-5,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))

runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=5)
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')

name_dataset = 'goji26IRRG2goji30RGB_256x256_3vi'
name_architecture = 'hrda1-128-0.1_daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'hrda1-128-0.1_daformer_sepaspp'
name_opt = 'adamw_6e-05_poly10warm_20k'
