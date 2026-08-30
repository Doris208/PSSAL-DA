_base_ = ['./goji26IRRG_supervised_common.py']

from configs.hrda.pssal.pssal_supervised_ablation_utils import (
    make_log_config, make_model_override, make_uda_override)

model = make_model_override(False, True)
uda = make_uda_override(False, False, True, True)
log_config = make_log_config(True, True)
work_dir = 'shared-nvme/work_dirs/goji26IRRG/goji26IRRG_hrda_3vi_pbcr'
name = 'goji26IRRG_hrda_3vi_pbcr'
exp = 'goji_supervised_ablation20k_pbcr'
name_uda = 'pssal_supervised_3vi_pbcr'
del make_log_config, make_model_override, make_uda_override
