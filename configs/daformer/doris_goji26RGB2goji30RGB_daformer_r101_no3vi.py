# Table III: single-scale DAFormer; reference runs used 20k (user confirmed).
_base_ = ['./goji26RGB2goji30RGB_daformer_3vi.py']

from copy import deepcopy
from configs._base_.datasets.uda_goji26RGB_to_goji30RGB_256x256 import (
    data as no3vi_data, source_img_norm_cfg, target_img_norm_cfg,
    source_train_pipeline, target_train_pipeline, test_pipeline)

data = deepcopy(no3vi_data)
data['_delete_'] = True
data['train']['target']['crop_pseudo_margins'] = [30, 30, 30, 30]
index_names = []
vi_names = []
model = dict(backbone=dict(
    in_channels=3, pretrained='open-mmlab://resnet101_v1c'))
uda = dict(input_channel_order=['R', 'G', 'B'])
archive_source_code = False
runner = dict(type='IterBasedRunner', max_iters=20000)
log_config = dict(interval=50, hooks=[dict(
    type='PrecisionLoggerHook', by_epoch=False, output_mode='diagnostic')])

work_dir = '/root/shared-nvme/work_dirs/doris_no3vi_r101/goji26RGB2goji30RGB_daformer_r101_no3vi'
name = 'doris_goji26RGB2goji30RGB_daformer_r101_no3vi'
name_dataset = 'goji26RGB2goji30RGB_256x256_no3vi'
name_opt = 'adamw_6e-05_poly10warm_20k'

del deepcopy, no3vi_data
