def make_sg_bde_proto_edge(mode):
    mode_cfgs = dict(
        plain=dict(
            use_struct_gate=False,
            use_spec_gate=False,
            use_conf_weight=False,
            use_proto_gate=False,
            edge_weight_mode='plain',
            struct_gate_weight=0.0,
            spec_gate_weight=0.0),
        struct=dict(
            use_struct_gate=True,
            use_spec_gate=False,
            use_conf_weight=False,
            use_proto_gate=False,
            use_boundary_pair_prior=True,
            use_spatial_layout_prior=True,
            use_neighbor_patch_prior=True,
            edge_weight_mode='struct',
            struct_gate_weight=1.0,
            spec_gate_weight=0.0,
            loss_weight_src=0.02,
            loss_weight_tgt=0.015,
            warmup_iters=5000),
        spec=dict(
            use_struct_gate=False,
            use_spec_gate=True,
            use_conf_weight=True,
            use_proto_gate=True,
            edge_weight_mode='spec',
            struct_gate_weight=0.0,
            spec_gate_weight=1.0),
        full=dict(
            use_struct_gate=True,
            use_spec_gate=True,
            use_conf_weight=True,
            use_proto_gate=True,
            use_boundary_pair_prior=True,
            use_spatial_layout_prior=True,
            use_neighbor_patch_prior=True,
            edge_weight_mode='full',
            struct_gate_weight=0.7,
            spec_gate_weight=0.3))
    if mode not in mode_cfgs:
        raise ValueError(f'Unsupported SG-BDE mode: {mode}')

    struct_edge_cfg = dict(
        enable=True,
        use_edge_loss=True,
        edge_radius=1,
        gate_min=0.25,
        gate_temp=2.0,
        gate_dilate=3,
        vi_names=['NDVI', 'NDRE', 'CIRE'],
        vi_channels=3,
        spec_fuse='max',
        spec_gate_temp=2.0,
        spec_gate_dilate=3,
        strict_spec=False,
        use_boundary_pair_prior=False,
        use_spatial_layout_prior=False,
        use_neighbor_patch_prior=False,
        num_classes=5,
        boundary_pair_momentum=0.99,
        boundary_pair_min_weight=0.60,
        boundary_pair_power=0.5,
        boundary_pair_warmup_iters=3000,
        spatial_middle_weight=1.00,
        spatial_edge_weight=0.90,
        spatial_corner_weight=0.80,
        spatial_border_width=16,
        spatial_border_min_weight=0.60,
        spatial_stride=128,
        neighbor_patch_min_weight=0.70,
        neighbor_patch_power=0.5,
        neighbor_patch_momentum=0.90,
        neighbor_patch_warmup_iters=3000,
        neighbor_patch_max_items=4096,
        loss_weight_src=0.01,
        loss_weight_tgt=0.02,
        warmup_iters=6000,
        detach_gate=True,
        debug=False)
    struct_edge_cfg.update(mode_cfgs[mode])

    model = dict(
        decode_head=dict(
            loss_decode=[
                dict(type='CrossEntropyLoss', use_sigmoid=False,
                     loss_weight=1.0),
                dict(
                    type='StructEdgeLoss',
                    edge_radius=1,
                    pred_edge='sobel_prob',
                    bce_weight=1.0,
                    ce_weight=0.1,
                    dice_weight=0.0,
                    loss_weight=1.0,
                    ignore_index=255,
                    loss_name='loss_struct_edge')
            ]))
    uda = dict(struct_edge_cfg=struct_edge_cfg)
    log_config = dict(
        interval=50,
        hooks=[
            dict(
                type='PrecisionLoggerHook',
                by_epoch=False,
                console_keys=[
                    'loss', 'decode.loss_seg', 'decode.loss_struct_edge',
                    'decode.acc_seg', 'mix.decode.loss_seg',
                    'mix.decode.loss_struct_edge', 'mix.decode.acc_seg',
                    'loss_proto_tgt',
                    'teacher_pseudo_label_retain_ratio',
                    'final_pseudo_weight_mean', 'proto_weight_mean',
                    'struct_edge_w_mix_mean',
                    'boundary_pair_gate_tgt_mean',
                    'spatial_layout_gate_tgt_mean',
                    'neighbor_patch_gate_tgt_mean',
                    'neighbor_patch_memory_size_tgt'
                ])
        ])
    return model, uda, log_config
