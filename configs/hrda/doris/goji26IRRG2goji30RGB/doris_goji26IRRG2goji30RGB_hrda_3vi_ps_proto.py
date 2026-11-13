# DORIS stage 1: Prototype ablation, aligned with the direction-specific Full.
# Adapted from goji26IRRG2goji30RGB_hrda_3vi_ps_proto.py.
_base_ = ['../../pssalda_26irrg2_30rgb_ablation_common.py']

from configs.hrda.pssalda_26rgb2_30rgb_ablation_utils import (
    make_log_config, make_model_override, make_uda_override)

model = make_model_override(False, False)
uda = make_uda_override(False, False, True, False)
log_config = make_log_config(True, False)

uda['proto_cfg'].update(
    target_weight_thr=None, target_conf_thr=None, tau_proto=0.25, tau_conf_proto=0.88,
    target_update_start=3000, weight_start=6000, loss_start=6000,
    lambda_proto_tgt=0.001, proto_min_weight=0.88, proto_power=0.7)

# Match Full's basic logging; module-specific losses are retained.
uda.update(doris_process_logging=False, doris_log_interval=1000)
log_config['hooks'][0].update(log_memory=True, paper_names=False)
uda.update(
    source_input_channel_order=['NIR', 'R', 'G', 'NDVI', 'NDRE', 'CIRE'],
    target_input_channel_order=['R', 'G', 'B', 'NDVI', 'NDRE', 'CIRE'])

# Collect only enabled module evidence; disabled modules are never probed.
uda.update(
    mechanism_logging=True, mechanism_sample_interval=100,
    mechanism_output_mode='branches', mechanism_detailed_output=False,
    mechanism_online_plot=True, mechanism_strict=True,
    mechanism_summary_interval=500, fixed_probe_enabled=True,
    fixed_probe_interval=500, fixed_probe_max_samples=1, fixed_probe_index=0,
    mechanism_save_raw=True, mechanism_make_figures=True)

work_dir = '/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26IRRG2goji30RGB/goji26IRRG2goji30RGB_hrda_3vi_ps_proto'
name = 'doris_goji26IRRG2goji30RGB_hrda_3vi_ps_proto'
exp = 'doris_stage1_ablation'
name_uda = 'doris_prototype'
runner = dict(type='IterBasedRunner', max_iters=15000)
name_opt = 'adamw_6e-05_poly10warm_15k'

del make_log_config, make_model_override, make_uda_override
