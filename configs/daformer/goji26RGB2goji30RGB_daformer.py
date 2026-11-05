"""
DAFormer on Goji 26 RGB -> Goji 30 RGB.
"""

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/daformer_sepaspp_mitb5.py',
    '../_base_/datasets/uda_goji26RGB_to_goji30RGB_256x256.py',
    '../_base_/uda/dacs.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py'
]

seed = 0
crop_size = (256, 256)

model = dict(
    pretrained=None,
    backbone=dict(pretrained='/root/HRDA-master/pretrained/mit_b5.pth'),
    decode_head=dict(num_classes=5),
    test_cfg=dict(
        mode='slide',
        batched_slide=True,
        stride=[128, 128],
        crop_size=[256, 256]))

uda = dict(
    type='DACS',
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
)

data = dict(
    samples_per_gpu=2,
    train=dict(target=dict(crop_pseudo_margins=[30, 30, 30, 30])),
    workers_per_gpu=4)

optimizer_config = None
optimizer = dict(
    lr=6e-5,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
        )))

runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=5)
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')
work_dir = '/root/shared-nvme/work_dirs/goji26RGB2goji30RGB_daformer'

name = 'goji26RGB2goji30RGB_daformer'
exp = 'goji_uda'
name_dataset = 'goji26RGB2goji30RGB_256x256'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'dacs'
name_opt = 'adamw_6e-05_poly10warm_10k'
