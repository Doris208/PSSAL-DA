# Inherit the complete Full protocol. Only tau_proto and observation I/O vary.
_base_ = ['./doris_goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge.py']

sensitivity_analysis = dict(enabled=True, tau_proto=0.25, csv_path=None)
work_dir = '/root/shared-nvme/work_dirs/sensitivity_tau_proto/goji26IRRG2goji30RGB/tau_0p25'
checkpoint_config = None
evaluation = dict(
    interval=1000, metric='mIoU', save_best=None,
    save_best_goji=False, doris_best_only=False)
log_config = dict(
    interval=1000,
    hooks=[dict(type='PrecisionLoggerHook', by_epoch=False,
                log_memory=False, paper_names=False, output_mode='minimal',
                paper_process_logging=False, log_diagnostics=False)])
uda = dict(
    mechanism_logging=False, mechanism_online_plot=False,
    mechanism_save_raw=False, mechanism_make_figures=False,
    fixed_probe_enabled=False, mechanism_detailed_output=False,
    mechanism_strict=False, doris_process_logging=False,
    paper_process_logging=False, save_class_mix_debug=False,
    save_teacher_pseudo_debug=False, debug_img_interval=0,
    output_mode='minimal', print_grad_magnitude=False,
    phy_cfg=dict(save_phy_debug=False),
    struct_cfg=dict(save_struct_debug=False),
    proto_cfg=dict(save_proto_debug=False, log_class_stats=False),
    joint_gate_cfg=dict(save_joint_debug=False),
    struct_edge_cfg=dict(debug=False))
