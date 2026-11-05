_base_ = ['./goji30RGB_supervised_common.py']

from configs.hrda.pssal.pssal_supervised_ablation_utils import (
    make_log_config, make_model_override, make_uda_override)

model = make_model_override(True, True)
uda = make_uda_override(True, True, True, True)
log_config = make_log_config(True, True)
work_dir = '/root/shared-nvme/work_dirs/goji30RGB/goji30RGB_hrda_3vi_proto_struct_edge'
name = 'goji30RGB_hrda_3vi_proto_struct_edge'
exp = 'goji_supervised_ablation20k_mparc_pbcr'
name_uda = 'pssal_supervised_3vi_mparc_pbcr'
del make_log_config, make_model_override, make_uda_override
