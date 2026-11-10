# DORIS comparison: MemoryAdaptNet + MiT-B5 + 3VI, 26IRRG -> 30RGB.
_base_ = ['./goji26IRRG2goji30RGB_memoryadaptnet_r101_3vi.py']

model = dict(
    backbone=dict(
        _delete_=True,
        type='mit_b5', in_chans=6,
        pretrained='/root/shared-nvme/pretrained/mit_b5.pth'),
    stage1_head=dict(in_channels=512),
    memory_bottleneck=dict(in_channels=512),
    # MiT stage 4 is stride 32; retain the original stride-8 memory/head grid
    # and a 32x32 probability input for the five-layer discriminator.
    feature_upsample_scale=4)
uda = dict(log_train_accuracy=True)

optimizer = dict(paramwise_cfg=dict(custom_keys=dict(
    # The discriminator owns a separate Adam optimizer. Do not step it twice.
    discriminator=dict(lr_mult=0.0, decay_mult=0.0))))
custom_hooks = [dict(type='MemoryAdaptNetOptimizerStateHook', interval=1000)]
runner = dict(type='IterBasedRunner', max_iters=20000)
log_config = dict(interval=50, hooks=[dict(type='PrecisionLoggerHook', by_epoch=False)])
archive_source_code = False

work_dir = '/root/shared-nvme/work_dirs/doris_stage1/goji26IRRG2goji30RGB_memoryadaptnet_3vi'
name = 'doris_goji26IRRG2goji30RGB_memoryadaptnet_3vi'
exp = 'doris_stage1_memoryadaptnet_mitb5_comparison'
name_encoder = 'mitb5'
name_architecture = 'memoryadaptnet_mitb5_3vi'
name_opt = 'adamw_6e-05_poly10warm_20k'
