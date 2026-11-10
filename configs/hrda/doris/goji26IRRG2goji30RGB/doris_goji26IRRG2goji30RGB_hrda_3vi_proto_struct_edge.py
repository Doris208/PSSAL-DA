# DORIS stage 1: Full candidate (not a claimed optimum).
_base_ = ['../../pssalda_26irrg2_30rgb_ablation_common.py']

from configs.hrda.pssalda_26rgb2_30rgb_ablation_utils import (
    make_log_config, make_model_override, make_uda_override)

model = make_model_override(True, True)
uda = make_uda_override(True, True, True, True)
log_config = make_log_config(True, True)

uda['phy_cfg'].update(
    phy_target_supervision_mode='balanced_high_conf',
    phy_pos_weight=0.5, phy_neg_weight=0.5, tau_conf_phy=0.968,
    lambda_phy_src=0.08, lambda_phy_tgt=0.008)
uda.update(physiology_collection=True)

model['decode_head']['loss_decode'][1].update(
    boundary_loss_mode='doris_balanced', loss_weight=0.1,
    bce_weight=1.0, ce_weight=0.1, dice_weight=0.0,
    positive_weight=0.75, negative_weight=0.25, negative_ring_radius=3)
uda['struct_cfg'].update(
    fusion_mode='doris_confidence_guided',
    continuity_weight=0.40, adjacency_weight=0.30, semantic_weight=0.30)
uda['struct_edge_cfg'].update(
    edge_fusion_mode='doris_reliable', pair_response_mode='valid_boundary_only',
    confidence_floor=0.60, confidence_range=0.35,
    prob_edge_weight=0.45, img_edge_weight=0.35, pair_edge_weight=0.20,
    loss_weight_src=0.004, loss_weight_tgt=0.002)
uda['proto_cfg'].update(
    target_weight_thr=None, target_conf_thr=None, tau_proto=0.25, tau_conf_proto=0.88,
    target_update_start=4000, weight_start=8000, loss_start=8000,
    lambda_proto_tgt=0.0002, proto_min_weight=0.98, proto_power=0.35)
# Retain the direction-specific legacy pair gate parameter (unused by the new mode).
uda['struct_edge_cfg']['boundary_pair_min_weight'] = 0.80

# Existing scalar debug logs remain off; mechanism CSV/probes are sampled separately.
uda.update(doris_process_logging=False, doris_log_interval=1000)
log_config['hooks'][0].update(log_memory=True, paper_names=False)
uda.update(
    source_input_channel_order=['NIR', 'R', 'G', 'NDVI', 'NDRE', 'CIRE'],
    target_input_channel_order=['R', 'G', 'B', 'NDVI', 'NDRE', 'CIRE'])

uda.update(
    mechanism_logging=True, mechanism_sample_interval=100,
    mechanism_output_mode='branches', mechanism_detailed_output=False,
    mechanism_online_plot=True, mechanism_strict=True,
    mechanism_summary_interval=500, fixed_probe_enabled=True,
    fixed_probe_interval=500, fixed_probe_max_samples=1, fixed_probe_index=0,
    mechanism_save_raw=True, mechanism_make_figures=True)

work_dir = '/root/shared-nvme/work_dirs/doris_stage1/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge'
name = 'doris_goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge'
exp = 'doris_stage1_full_candidate'
name_uda = 'doris_mparc_pbcr'
name_opt = 'adamw_6e-05_poly10warm_20k'

del make_log_config, make_model_override, make_uda_override
