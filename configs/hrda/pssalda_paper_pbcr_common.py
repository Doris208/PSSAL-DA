def make_feature_interaction():
    model = dict(
        decode_head=dict(
            struct_prior_interaction=dict(
                enable=True,
                gamma_struct=0.05,
                apply_to='hr')))
    uda = dict(
        struct_cfg=dict(
            use_feature_interaction=True,
            feature_interaction_warmup_iters=6000,
            feature_interaction_rampup_iters=4000))
    log_config = dict(
        interval=50,
        hooks=[
            dict(
                type='PrecisionLoggerHook',
                by_epoch=False,
                console_keys=[
                    'loss', 'decode.loss_seg', 'decode.acc_seg',
                    'decode.hr.loss_seg', 'decode.hr.acc_seg',
                    'mix.decode.loss_seg', 'mix.decode.acc_seg',
                    'mix.decode.hr.loss_seg', 'mix.decode.hr.acc_seg',
                    'teacher_target_confidence_mean',
                    'teacher_pseudo_label_retain_ratio',
                    'struct_feature_ratio',
                    'struct_gate_mean', 'struct_gate_struct_class_mean',
                    'struct_gate_building_mean', 'struct_gate_water_mean',
                    'struct_gate_road_mean', 'struct_filter_ratio',
                    'struct_continuity_mean', 'struct_adj_mean',
                    'struct_sem_mean'
                ])
        ])
    return model, uda, log_config


def make_paper_pbcr_edge(enable_proto, enable_struct_prior=True):
    struct_edge_cfg = dict(
        enable=True,
        use_edge_loss=True,
        edge_radius=1,
        edge_weight_mode='paper_pbcr',
        use_struct_gate=False,
        use_spec_gate=False,
        use_conf_weight=False,
        use_proto_gate=False,
        use_boundary_pair_prior=True,
        use_spatial_layout_prior=False,
        use_neighbor_patch_prior=False,
        num_classes=5,
        boundary_pair_momentum=0.99,
        boundary_pair_min_weight=0.60,
        boundary_pair_power=0.5,
        boundary_pair_warmup_iters=3000,
        prob_edge_weight=0.4,
        img_edge_weight=0.4,
        pair_edge_weight=0.2,
        loss_weight_src=0.02,
        loss_weight_tgt=0.015,
        warmup_iters=5000,
        detach_gate=True,
        debug=False)
    decode_head = dict(
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
            ])
    if enable_struct_prior:
        decode_head['struct_prior_interaction'] = dict(
            enable=True,
            gamma_struct=0.05,
            apply_to='hr')
    model = dict(decode_head=decode_head)
    uda = dict(proto_cfg=dict(enable=enable_proto),
               struct_edge_cfg=struct_edge_cfg)
    if enable_struct_prior:
        uda['struct_cfg'] = dict(
            use_feature_interaction=True,
            feature_interaction_warmup_iters=6000,
            feature_interaction_rampup_iters=4000)
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
                    'struct_edge_w_mix_mean', 'prob_edge_tgt_mean',
                    'img_edge_tgt_mean', 'boundary_pair_response_tgt_mean',
                    'boundary_pair_prior_ready_tgt'
                ])
        ])
    return model, uda, log_config
