_base_ = ['./goji30RGB2goji26RGB_hrda_3vi.py']

uda = dict(
    phy_cfg=dict(enable=False),
    struct_cfg=dict(
        enable=True,
        struct_positive_classes=[2, 3, 4],
        lambda_struct_tgt=0.08,
        use_target_gate=True,
        struct_min_weight=0.60,
        struct_filter_threshold=0.35,
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
        adjacency_power=1.0,
        semantic_power=0.5,
        detach_gate=True,
        debug=True,
        save_struct_debug=True,
        struct_debug_interval=1000,
        struct_debug_max_samples=50,
        struct_debug_dir='struct_gate_debug'))

runner = dict(type='IterBasedRunner', max_iters=15000)
work_dir = 'shared-nvme/work_dirs/goji30RGB2goji26RGB_hrda_3vi_struct_gate'
name = 'goji30RGB2goji26RGB_hrda_3vi_struct_gate'
exp = 'goji_uda_indices_struct'
name_dataset = 'goji30RGB2goji26RGB_256x256_3vi_struct_gate'
name_uda = 'dacs_3vi_struct_gate'
name_opt = 'adamw_6e-05_poly10warm_15k'
