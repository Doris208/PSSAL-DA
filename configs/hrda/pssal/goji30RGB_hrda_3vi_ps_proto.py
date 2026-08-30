_base_ = ['./goji30RGB_supervised_common.py']

from configs.hrda.pssal.pssal_supervised_ablation_utils import (
    make_log_config, make_model_override, make_uda_override)

model = make_model_override(False, False)
uda = make_uda_override(False, False, True, False)
log_config = make_log_config(True, False)
work_dir = 'shared-nvme/work_dirs/goji30RGB/goji30RGB_hrda_3vi_ps_proto'
name = 'goji30RGB_hrda_3vi_ps_proto'
exp = 'goji_supervised_ablation20k_prototype'
name_uda = 'pssal_supervised_3vi_prototype'
del make_log_config, make_model_override, make_uda_override
