_base_ = ['./pssalda_30irrg2_26irrg_ablation_common.py']

from configs.hrda.pssalda_26rgb2_30rgb_ablation_utils import (
    make_log_config, make_model_override, make_uda_override)

model = make_model_override(False, True)
uda = make_uda_override(False, False, True, True)
log_config = make_log_config(True, True)
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=5)
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')
work_dir = '/root/shared-nvme/work_dirs/goji30IRRG2goji26IRRG/goji30IRRG2goji26IRRG_hrda_3vi_pbcr'
name = 'goji30IRRG2goji26IRRG_hrda_3vi_pbcr'
exp = 'goji_uda_indices_ablation20k_pbcr'
name_uda = 'dacs_3vi_pbcr'
name_opt = 'adamw_6e-05_poly10warm_20k'
del make_log_config, make_model_override, make_uda_override
