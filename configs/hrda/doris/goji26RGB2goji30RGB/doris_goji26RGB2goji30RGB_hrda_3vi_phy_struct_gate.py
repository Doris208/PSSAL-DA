# DORIS stage 1: MPARC ablation, aligned with the direction-specific Full.
# Adapted from goji26RGB2goji30RGB_hrda_3vi_phy_struct_gate.py.
_base_ = ['../../pssalda_26rgb2_30rgb_ablation_common.py']

from configs.hrda.pssalda_26rgb2_30rgb_ablation_utils import (
    make_log_config, make_model_override, make_uda_override)

model = make_model_override(True, False)
uda = make_uda_override(True, True, False, False)
log_config = make_log_config(False, False)

uda['phy_cfg'].update(
    mode='hierarchical_gv', beta_gv=1.0, lambda_phy_tgt_gv=0.0,
    phy_gate_schedule='early_nonzero_linear',
    phy_gate_initial_strength=0.10, phy_gate_full_iter=4000,
    phy_min_weight=0.75, phy_filter_threshold=None,
    phy_warmup_iters=0, phy_rampup_iters=4000, detach_target_gate=True,
    phy_target_supervision_mode='balanced_high_conf',
    phy_pos_weight=0.5, phy_neg_weight=0.5, tau_conf_phy=0.968,
    lambda_phy_src=0.08, lambda_phy_tgt=0.008)
uda.update(physiology_collection=True)

uda['struct_cfg'].update(
    fusion_mode='struct_distinct_v1',
    adjacency_mode='cross_class_conditional_v1',
    semantic_score_mode='probability_margin_v1',
    confidence_weight=0.15, continuity_weight=0.30,
    adjacency_weight=0.30, semantic_weight=0.25,
    adj_cross_uniform_smoothing=0.10, adj_min_cross_mass=0.01,
    semantic_temperature=0.15,
    struct_gate_schedule='early_nonzero_linear',
    struct_gate_initial_strength=0.10, struct_gate_full_iter=4000)

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
    fixed_probe_interval=500, fixed_probe_max_samples=6, fixed_probe_index=0,
    fixed_probe_filenames=[
        '3_1024_1280_1280_1536.tif',
        '3_256_1152_512_1408.tif',
        '3_1536_512_1792_768.tif',
        '3_1536_1024_1792_1280.tif',
        '3_1792_1280_2048_1536.tif',
        '3_1536_640_1792_896.tif',
    ],
    mechanism_save_raw=True, mechanism_make_figures=True)

work_dir = '/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_phy_struct_gate'
name = 'doris_goji26RGB2goji30RGB_hrda_3vi_phy_struct_gate_hphy_early4k'
exp = 'doris_stage1_ablation'
name_uda = 'doris_mparc'
runner = dict(type='IterBasedRunner', max_iters=15000)
name_opt = 'adamw_6e-05_poly10warm_15k'

del make_log_config, make_model_override, make_uda_override

# H-Phy V1 provenance. Runtime Git SHA is also recorded by tools/train.py.
hphy_version = 'hierarchical_gv_early4k_v2'
hphy_base_commit = 'ef089d4dc036b4b1ac3069d5b4e6964734be3da5'
hphy_vi_order = ['NDVI', 'NDRE', 'CIRE']
hphy_label_mapping = {0: 'Goji', 1: 'Vegetation', 2: 'Building', 3: 'Water', 4: 'Road', 255: 'Ignore'}
