_base_ = ['./goji26IRRG2goji30IRRG_hrda_3vi_phy_gate.py']

uda = dict(
    phy_cfg=dict(
        enable=True,
        vi_names=['ndvi', 'ndre', 'cire'],
        phy_positive_classes=[0, 1],
        apply_to_classes=[0, 1],
        lambda_phy_src=0.08,
        lambda_phy_tgt=0.008,
        use_target_gate=True,
        phy_min_weight=0.82,
        phy_filter_threshold=0.30,
        phy_filter_mode='soft',
        phy_warmup_iters=6000,
        phy_rampup_iters=4000,
        detach_target_gate=True,
        save_phy_debug=True,
        phy_debug_interval=1000,
        phy_debug_max_samples=50,
        phy_head=dict(in_channels=3, mid_channels=32, num_layers=2)),
    struct_cfg=dict(
        enable=True,
        struct_positive_classes=[2, 3, 4],
        apply_to_classes=[2, 3, 4],
        lambda_struct_tgt=0.05,
        use_target_gate=True,
        struct_min_weight=0.65,
        struct_filter_threshold=0.32,
        struct_filter_mode='soft',
        struct_warmup_iters=6000,
        struct_rampup_iters=4000,
        source_adj_momentum=0.99,
        semantic_proto_momentum=0.99,
        use_source_adjacency_ema=True,
        use_semantic_proto=True,
        kernel_size=3,
        confidence_power=1.0,
        continuity_power=1.0,
        adjacency_power=0.8,
        semantic_power=0.5,
        detach_gate=True,
        save_struct_debug=True,
        struct_debug_interval=1000,
        struct_debug_max_samples=50,
        struct_debug_dir='struct_gate_debug'),
    joint_gate_cfg=dict(
        enable=True,
        fusion_mode='classwise_disjoint',
        phy_classes=[0, 1],
        struct_classes=[2, 3, 4],
        use_original_weight_for_others=True,
        normalize_after_fusion=False,
        clamp_min=0.0,
        clamp_max=1.0,
        warmup_iters=6000,
        rampup_iters=4000,
        log_classwise_retain=True,
        save_joint_debug=True,
        joint_debug_interval=1000,
        joint_debug_max_samples=50,
        joint_debug_dir='phy_struct_gate_debug'),
    proto_cfg=dict(
        enable=True,
        num_classes=5,
        ignore_index=255,
        momentum=0.99,
        min_pixels=8,
        source_update=True,
        target_update=True,
        target_update_start=3000,
        target_conf_thr=0.88,
        target_weight_thr=0.35,
        proto_min_weight=0.88,
        proto_temp=0.10,
        proto_margin_thr=0.02,
        proto_power=0.70,
        apply_to_classes=[0, 1, 2, 3, 4],
        weight_start=6000,
        min_valid_classes_for_weight=3,
        lambda_proto_tgt=0.001,
        lambda_proto_src=0.0,
        loss_start=6000,
        min_valid_classes_for_loss=3,
        save_proto_debug=True,
        proto_debug_interval=1000,
        log_class_stats=True,
        detach=True))

runner = dict(type='IterBasedRunner', max_iters=15000)
log_config = dict(
    interval=50,
    hooks=[
        dict(
            type='PrecisionLoggerHook',
            by_epoch=False,
            console_keys=[
                'loss', 'decode.loss_seg', 'decode.acc_seg',
                'mix.decode.loss_seg', 'mix.decode.acc_seg',
                'loss_phy_src', 'loss_phy_tgt', 'loss_proto_tgt',
                'teacher_target_confidence_mean',
                'teacher_pseudo_label_retain_ratio',
                'final_pseudo_weight_mean', 'phy_gate_mean',
                'struct_gate_mean', 'proto_weight_mean',
                'proto_margin_mean', 'proto_valid_classes',
                'proto_src_update_ratio', 'proto_tgt_update_ratio'
            ])
    ])
work_dir = 'shared-nvme/work_dirs/goji26IRRG2goji30IRRG_hrda_3vi_ps_proto'
name = 'goji26IRRG2goji30IRRG_hrda_3vi_ps_proto'
exp = 'goji_uda_indices_ps_proto'
name_dataset = 'goji26IRRG2goji30IRRG_256x256_3vi_ps_proto'
name_uda = 'dacs_3vi_ps_proto'
name_opt = 'adamw_6e-05_poly10warm_15k'
