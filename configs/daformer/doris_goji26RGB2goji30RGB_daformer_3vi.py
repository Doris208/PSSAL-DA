# DORIS comparison: DAFormer + MiT-B5 + 3VI, 26RGB -> 30RGB.
_base_ = ['./goji26RGB2goji30RGB_daformer_3vi.py']

model = dict(
    backbone=dict(
        _delete_=True,
        type='mit_b5', in_chans=6,
        pretrained='/root/shared-nvme/pretrained/mit_b5.pth'),
    decode_head=dict(in_channels=[64, 128, 320, 512]))
uda = dict(
    output_mode='minimal', paper_process_logging=False,
    save_class_mix_debug=False, save_teacher_pseudo_debug=False,
    phy_cfg=dict(enable=False, save_phy_debug=False),
    struct_cfg=dict(enable=False, save_struct_debug=False),
    proto_cfg=dict(enable=False, save_proto_debug=False, log_class_stats=False),
    struct_edge_cfg=dict(enable=False, use_edge_loss=False),
    joint_gate_cfg=dict(enable=False, save_joint_debug=False))
runner = dict(type='IterBasedRunner', max_iters=15000)
log_config = dict(interval=50, hooks=[dict(type='PrecisionLoggerHook', by_epoch=False)])
archive_source_code = False

work_dir = '/root/shared-nvme/work_dirs/doris_stage1/goji26RGB2goji30RGB_daformer_3vi'
name = 'doris_goji26RGB2goji30RGB_daformer_3vi'
exp = 'doris_stage1_daformer_mitb5_comparison'
name_encoder = 'mitb5'
name_architecture = 'daformer_sepaspp_mitb5'
name_opt = 'adamw_6e-05_poly10warm_15k'
