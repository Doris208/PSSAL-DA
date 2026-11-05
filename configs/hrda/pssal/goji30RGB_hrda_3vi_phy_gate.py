_base_ = ['./goji30RGB_supervised_common.py']

from configs.hrda.pssal.pssal_supervised_ablation_utils import (
    make_log_config, make_model_override, make_uda_override)

model = make_model_override(False, False)
uda = make_uda_override(True, False, False, False)
log_config = make_log_config(False, False)
work_dir = '/root/shared-nvme/work_dirs/goji30RGB/goji30RGB_hrda_3vi_phy_gate'
name = 'goji30RGB_hrda_3vi_phy_gate'
exp = 'goji_supervised_ablation20k_physiology'
name_uda = 'pssal_supervised_3vi_physiology'
del make_log_config, make_model_override, make_uda_override
