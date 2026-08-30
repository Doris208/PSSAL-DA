_base_ = ['./goji30IRRG2goji26IRRG_hrda_3vi_ps_proto.py']

from configs.hrda.sg_bde_proto_edge_common import make_sg_bde_proto_edge

model, uda, log_config = make_sg_bde_proto_edge('struct')
del make_sg_bde_proto_edge

work_dir = 'shared-nvme/work_dirs/goji30IRRG2goji26IRRG_hrda_3vi_proto_struct_edge'
name = 'goji30IRRG2goji26IRRG_hrda_3vi_proto_struct_edge'
exp = 'goji_uda_indices_proto_struct_edge'
name_dataset = 'goji30IRRG2goji26IRRG_256x256_3vi_proto_struct_edge'
name_uda = 'dacs_3vi_proto_struct_edge'
name_opt = 'adamw_6e-05_poly10warm_20k'
# PSSAL-DA aligned 20k ablation override.
_base_ = ['./pssalda_30irrg2_26irrg_ablation_common.py']
from configs.hrda.pssalda_26rgb2_30rgb_ablation_utils import (
    make_log_config, make_model_override, make_uda_override)

model = make_model_override(True, True)
uda = make_uda_override(True, True, True, True)
log_config = make_log_config(True, True)
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=5)
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')
work_dir = 'shared-nvme/work_dirs/goji30IRRG2goji26IRRG/goji30IRRG2goji26IRRG_hrda_3vi_proto_struct_edge'
name = 'goji30IRRG2goji26IRRG_hrda_3vi_proto_struct_edge'
exp = 'goji_uda_indices_ablation20k_mparc_pbcr'
name_uda = 'dacs_3vi_mparc_pbcr'
name_opt = 'adamw_6e-05_poly10warm_20k'
del make_log_config, make_model_override, make_uda_override