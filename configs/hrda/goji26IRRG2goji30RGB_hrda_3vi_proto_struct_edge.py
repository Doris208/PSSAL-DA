# PSSAL-DA aligned 20k ablation override.
_base_ = ['./pssalda_26irrg2_30rgb_ablation_common.py']

from configs.hrda.pssalda_26rgb2_30rgb_ablation_utils import (
    make_log_config, make_model_override, make_uda_override)

model = make_model_override(True, True)
uda = make_uda_override(True, True, True, True)

# This direction is fragile: the best known 20k run is the MPARC
# phy+struct branch. Keep proto/edge as weak late auxiliaries so they do not
# dominate the main segmentation gradients.
model['decode_head']['loss_decode'][1]['loss_weight'] = 0.1
uda['proto_cfg'].update(
    target_update_start=8000,
    target_conf_thr=0.88,
    target_weight_thr=0.30,
    proto_min_weight=0.98,
    proto_margin_thr=0.02,
    proto_power=0.35,
    weight_start=14000,
    lambda_proto_tgt=0.0002,
    loss_start=14000)
uda['struct_edge_cfg'].update(
    edge_weight_mode='paper_pbcr',
    use_struct_gate=False,
    use_boundary_pair_prior=True,
    use_spatial_layout_prior=False,
    use_neighbor_patch_prior=False,
    boundary_pair_momentum=0.99,
    boundary_pair_min_weight=0.80,
    boundary_pair_power=0.5,
    boundary_pair_warmup_iters=3000,
    loss_weight_src=0.004,
    loss_weight_tgt=0.002,
    warmup_iters=5000)
log_config = make_log_config(True, True)

runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=5)
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')
work_dir = '/root/shared-nvme/work_dirs/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge_safe'
name = 'goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge_safe'
exp = 'goji_uda_indices_ablation20k_mparc_pbcr_safe'
name_dataset = 'goji26IRRG2goji30RGB_256x256_3vi_proto_struct_edge_safe'
name_uda = 'dacs_3vi_mparc_pbcr_safe'
name_opt = 'adamw_6e-05_poly10warm_20k'

del make_log_config, make_model_override, make_uda_override
