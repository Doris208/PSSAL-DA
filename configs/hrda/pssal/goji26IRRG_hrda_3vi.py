_base_ = ['./goji26IRRG_supervised_common.py']

from configs.hrda.pssal.pssal_supervised_ablation_utils import (
    make_log_config, make_model_override, make_uda_override)

model = make_model_override(False, False)
uda = make_uda_override(False, False, False, False)
log_config = make_log_config(False, False)
work_dir = 'shared-nvme/work_dirs/goji26IRRG/goji26IRRG_hrda_3vi'
name = 'goji26IRRG_hrda_3vi'
exp = 'goji_supervised_ablation20k_baseline'
name_uda = 'pssal_supervised_3vi_baseline'
del make_log_config, make_model_override, make_uda_override
