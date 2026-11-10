# DORIS stage 1: Structural ablation, aligned with the direction-specific Full.
# Adapted from goji26RGB2goji30RGB_hrda_3vi_struct_gate.py.
_base_ = ['../../pssalda_26rgb2_30rgb_ablation_common.py']

from configs.hrda.pssalda_26rgb2_30rgb_ablation_utils import (
    make_log_config, make_model_override, make_uda_override)

model = make_model_override(True, False)
uda = make_uda_override(False, True, False, False)
log_config = make_log_config(False, False)

uda['struct_cfg'].update(
    fusion_mode='doris_confidence_guided',
    continuity_weight=0.40, adjacency_weight=0.30, semantic_weight=0.30)

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

work_dir = '/root/shared-nvme/work_dirs/doris_stage1/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_struct_gate'
name = 'doris_goji26RGB2goji30RGB_hrda_3vi_struct_gate'
exp = 'doris_stage1_ablation'
name_uda = 'doris_structural'
runner = dict(type='IterBasedRunner', max_iters=15000)
name_opt = 'adamw_6e-05_poly10warm_15k'

del make_log_config, make_model_override, make_uda_override
