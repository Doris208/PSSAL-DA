_base_ = ['./goji26RGB2goji30RGB_hrda_3vi.py']

from configs.hrda.pssalda_paper_pbcr_common import make_paper_pbcr_edge

model, uda, log_config = make_paper_pbcr_edge(
    enable_proto=True, enable_struct_prior=False)
del make_paper_pbcr_edge

uda['proto_cfg'].update(
    num_classes=5,
    ignore_index=255,
    momentum=0.99,
    min_pixels=8,
    source_update=True,
    target_update=True,
    target_update_start=3000,
    target_conf_thr=0.88,
    target_weight_thr=0.35,
    proto_min_weight=0.88,
    proto_temp=0.10,
    proto_margin_thr=0.02,
    proto_power=0.70,
    apply_to_classes=[0, 1, 2, 3, 4],
    weight_start=6000,
    min_valid_classes_for_weight=3,
    lambda_proto_tgt=0.001,
    lambda_proto_src=0.0,
    loss_start=6000,
    min_valid_classes_for_loss=3,
    save_proto_debug=True,
    proto_debug_interval=1000,
    log_class_stats=True,
    detach=True)

runner = dict(type='IterBasedRunner', max_iters=20000)
work_dir = 'shared-nvme/work_dirs/goji26RGB2goji30RGB_hrda_3vi_pbcr'
name = 'goji26RGB2goji30RGB_hrda_3vi_pbcr'
exp = 'goji_uda_indices_pbcr'
name_dataset = 'goji26RGB2goji30RGB_256x256_3vi_pbcr'
name_uda = 'dacs_3vi_pbcr'
name_opt = 'adamw_6e-05_poly10warm_20k'

# PSSAL-DA 26RGB->30RGB aligned 20k ablation override.
_base_ = ['./pssalda_26rgb2_30rgb_ablation_common.py']
from configs.hrda.pssalda_26rgb2_30rgb_ablation_utils import (
    make_log_config, make_model_override, make_uda_override)

model = make_model_override(False, True)
uda = make_uda_override(False, False, True, True)
log_config = make_log_config(True, True)
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=5)
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')
work_dir = 'shared-nvme/work_dirs/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_pbcr'
name = 'goji26RGB2goji30RGB_hrda_3vi_pbcr'
exp = 'goji_uda_indices_ablation20k_pbcr'
name_uda = 'dacs_3vi_pbcr'
del make_log_config, make_model_override, make_uda_override
