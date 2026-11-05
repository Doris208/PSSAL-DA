_base_ = ['./goji26RGB2goji30RGB_hrda_3vi.py']

from configs.hrda.pssalda_paper_pbcr_common import make_paper_pbcr_edge

model, uda, log_config = make_paper_pbcr_edge(
    enable_proto=False, enable_struct_prior=False)
del make_paper_pbcr_edge

runner = dict(type='IterBasedRunner', max_iters=20000)
work_dir = '/root/shared-nvme/work_dirs/goji26RGB2goji30RGB_hrda_3vi_struct_edge'
name = 'goji26RGB2goji30RGB_hrda_3vi_struct_edge'
exp = 'goji_uda_indices_struct_edge'
name_dataset = 'goji26RGB2goji30RGB_256x256_3vi_struct_edge'
name_uda = 'dacs_3vi_struct_edge'
name_opt = 'adamw_6e-05_poly10warm_20k'

# PSSAL-DA 26RGB->30RGB aligned 20k ablation override.
_base_ = ['./pssalda_26rgb2_30rgb_ablation_common.py']
from configs.hrda.pssalda_26rgb2_30rgb_ablation_utils import (
    make_log_config, make_model_override, make_uda_override)

model = make_model_override(False, True)
uda = make_uda_override(False, False, False, True)
log_config = make_log_config(False, True)
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=5)
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')
work_dir = '/root/shared-nvme/work_dirs/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_struct_edge'
name = 'goji26RGB2goji30RGB_hrda_3vi_struct_edge'
exp = 'goji_uda_indices_ablation20k_edge'
name_uda = 'dacs_3vi_edge'
del make_log_config, make_model_override, make_uda_override
