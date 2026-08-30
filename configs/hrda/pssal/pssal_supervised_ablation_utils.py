VI_NAMES = ['ndvi', 'ndre', 'cire']


def make_phy_cfg(enable):
    if not enable:
        return dict(enable=False)
    return dict(
        enable=True,
        vi_names=VI_NAMES,
        phy_positive_classes=[0, 1],
        lambda_phy_src=0.08,
        phy_head=dict(in_channels=3, mid_channels=32, num_layers=2))


def make_struct_cfg(enable):
    if not enable:
        return dict(enable=False)
    return dict(
        enable=True,
        use_feature_interaction=True,
        feature_interaction_warmup_iters=6000,
        feature_interaction_rampup_iters=4000,
        struct_positive_classes=[2, 3, 4],
        apply_to_classes=[2, 3, 4],
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
        detach_gate=True)


def make_proto_cfg(enable):
    if not enable:
        return dict(enable=False)
    return dict(
        enable=True,
        num_classes=5,
        ignore_index=255,
        momentum=0.99,
        min_pixels=8,
        source_update=True,
        target_update=False,
        proto_min_weight=0.88,
        proto_temp=0.10,
        proto_margin_thr=0.02,
        proto_power=0.70,
        apply_to_classes=[0, 1, 2, 3, 4],
        weight_start=6000,
        min_valid_classes_for_weight=3,
        lambda_proto_src=0.001,
        lambda_proto_tgt=0.0,
        loss_start=6000,
        min_valid_classes_for_loss=3,
        save_proto_debug=False,
        log_class_stats=True,
        detach=True)


def make_edge_cfg(enable):
    if not enable:
        return dict(enable=False, use_edge_loss=False)
    return dict(
        enable=True,
        use_edge_loss=True,
        edge_radius=1,
        num_classes=5,
        boundary_pair_min_weight=0.60,
        prob_edge_weight=0.4,
        img_edge_weight=0.4,
        pair_edge_weight=0.2,
        loss_weight_src=0.02,
        warmup_iters=5000,
        ignore_index=255)


def make_model_override(structural=False, edge=False):
    decode_head = {}
    if structural:
        decode_head['struct_prior_interaction'] = dict(
            enable=True, gamma_struct=0.05, apply_to='hr')
    if edge:
        decode_head['loss_decode'] = [
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
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
        ]
    return dict(decode_head=decode_head) if decode_head else dict()


def make_uda_override(physiology=False,
                      structural=False,
                      prototype=False,
                      edge=False):
    return dict(
        _delete_=True,
        type='PSSALSupervised',
        phy_cfg=make_phy_cfg(physiology),
        struct_cfg=make_struct_cfg(structural),
        proto_cfg=make_proto_cfg(prototype),
        struct_edge_cfg=make_edge_cfg(edge))


def make_log_config(prototype=False, edge=False):
    keys = [
        'loss', 'decode.loss_seg', 'decode.acc_seg',
        'loss_phy_src', 'R_phy_mean',
        'R_struct_mean',
        'loss_proto_src', 'proto_src_update_ratio',
        'prob_edge_src_mean', 'img_edge_src_mean',
        'boundary_pair_response_src_mean', 'struct_edge_w_src_mean',
        'decode.loss_struct_edge'
    ]
    if not prototype:
        keys = [k for k in keys if not k.startswith('proto')
                and k != 'loss_proto_src']
    if not edge:
        keys = [k for k in keys if 'edge' not in k]
    return dict(
        interval=50,
        hooks=[dict(
            type='PrecisionLoggerHook',
            by_epoch=False,
            console_keys=keys)])
