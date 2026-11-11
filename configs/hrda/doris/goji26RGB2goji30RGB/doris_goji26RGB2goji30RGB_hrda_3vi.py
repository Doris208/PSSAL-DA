# DORIS stage 1: unchanged HRDA baseline.
_base_ = ['../../pssalda_26rgb2_30rgb_ablation_common.py']

from configs.hrda.pssalda_26rgb2_30rgb_ablation_utils import (
    make_log_config, make_model_override, make_uda_override)

model = make_model_override(False, False)
uda = make_uda_override(False, False, False, False)
log_config = make_log_config(False, False)
log_config['hooks'][0].update(log_memory=True, paper_names=False)
uda.update(mechanism_logging=False, fixed_probe_enabled=False)

work_dir = '/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi'
name = 'doris_goji26RGB2goji30RGB_hrda_3vi'
exp = 'doris_stage1_hrda_baseline'
name_uda = 'dacs_3vi_baseline'
runner = dict(type='IterBasedRunner', max_iters=15000)
name_opt = 'adamw_6e-05_poly10warm_15k'

del make_log_config, make_model_override, make_uda_override
