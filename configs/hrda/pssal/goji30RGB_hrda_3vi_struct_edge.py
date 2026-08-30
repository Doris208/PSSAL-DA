_base_ = ['./goji30RGB_supervised_common.py']

from configs.hrda.pssal.pssal_supervised_ablation_utils import (
    make_log_config, make_model_override, make_uda_override)

model = make_model_override(False, True)
uda = make_uda_override(False, False, False, True)
log_config = make_log_config(False, True)
work_dir = 'shared-nvme/work_dirs/goji30RGB/goji30RGB_hrda_3vi_struct_edge'
name = 'goji30RGB_hrda_3vi_struct_edge'
exp = 'goji_supervised_ablation20k_edge'
name_uda = 'pssal_supervised_3vi_edge'
del make_log_config, make_model_override, make_uda_override
