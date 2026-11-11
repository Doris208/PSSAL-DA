# DORIS stage 1: Edge ablation, aligned with the direction-specific Full.
# Adapted from goji26RGB2goji30RGB_hrda_3vi_struct_edge.py.
_base_ = ['../../pssalda_26rgb2_30rgb_ablation_common.py']

from configs.hrda.pssalda_26rgb2_30rgb_ablation_utils import (
    make_log_config, make_model_override, make_uda_override)

model = make_model_override(False, True)
uda = make_uda_override(False, False, False, True)
log_config = make_log_config(False, True)

model['decode_head']['loss_decode'][1].update(
    boundary_loss_mode='doris_balanced', loss_weight=0.1,
    bce_weight=1.0, ce_weight=0.1, dice_weight=0.0,
    positive_weight=0.75, negative_weight=0.25, negative_ring_radius=3)
uda['struct_edge_cfg'].update(
    edge_fusion_mode='doris_reliable', pair_response_mode='valid_boundary_only',
    confidence_floor=0.60, confidence_range=0.35,
    prob_edge_weight=0.45, img_edge_weight=0.35, pair_edge_weight=0.20,
    loss_weight_src=0.01, loss_weight_tgt=0.006)

# Match Full's basic logging; module-specific losses are retained.
uda.update(doris_process_logging=False, doris_log_interval=1000)
log_config['hooks'][0].update(log_memory=True, paper_names=False)
uda.update(
    source_input_channel_order=['R', 'G', 'B', 'NDVI', 'NDRE', 'CIRE'],
    target_input_channel_order=['R', 'G', 'B', 'NDVI', 'NDRE', 'CIRE'])

# Collect only enabled module evidence; disabled modules are never probed.
uda.update(
    mechanism_logging=True, mechanism_sample_interval=100,
    mechanism_output_mode='branches', mechanism_detailed_output=False,
    mechanism_online_plot=True, mechanism_strict=True,
    mechanism_summary_interval=500, fixed_probe_enabled=True,
    fixed_probe_interval=500, fixed_probe_max_samples=1, fixed_probe_index=0,
    mechanism_save_raw=True, mechanism_make_figures=True)

work_dir = '/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_struct_edge'
name = 'doris_goji26RGB2goji30RGB_hrda_3vi_struct_edge'
exp = 'doris_stage1_ablation'
name_uda = 'doris_edge'
runner = dict(type='IterBasedRunner', max_iters=15000)
name_opt = 'adamw_6e-05_poly10warm_15k'

del make_log_config, make_model_override, make_uda_override
