# Table II: retain MemoryAdaptNet; reference runs used 20k (user confirmed).
_base_ = ['./goji26IRRG2goji30RGB_memoryadaptnet_r101_3vi.py']

from copy import deepcopy
from configs._base_.datasets.uda_goji26IRRG_to_goji30RGB_256x256 import (
    data as no3vi_data, source_img_norm_cfg, target_img_norm_cfg,
    source_train_pipeline, target_train_pipeline, test_pipeline)

data = deepcopy(no3vi_data)
data['_delete_'] = True
index_names = []
vi_names = []
model = dict(backbone=dict(
    in_channels=3, pretrained='open-mmlab://resnet101_v1c'))
# Match the DORIS comparison: only the dedicated Adam updates discriminator.
optimizer = dict(paramwise_cfg=dict(custom_keys=dict(
    discriminator=dict(lr_mult=0.0, decay_mult=0.0))))
archive_source_code = False
runner = dict(type='IterBasedRunner', max_iters=15000)
log_config = dict(interval=50, hooks=[dict(
    type='PrecisionLoggerHook', by_epoch=False, output_mode='diagnostic')])

work_dir = '/root/shared-nvme/work_dirs/doris_no3vi_r101/goji26IRRG2goji30RGB_memoryadaptnet_r101_no3vi'
name = 'doris_goji26IRRG2goji30RGB_memoryadaptnet_r101_no3vi'
name_dataset = 'goji26IRRG2goji30RGB_256x256_no3vi'
name_architecture = 'memoryadaptnet_r101_no3vi'
name_opt = 'adamw_6e-05_poly10warm_15k'

del deepcopy, no3vi_data
