# Table I: R101 without VI; reference runs used 20k (confirmed by the user).
_base_ = ['./goji30IRRG2goji26IRRG_baseline_resnet101.py']

from copy import deepcopy
from configs._base_.datasets.uda_goji30IRRG_to_goji26IRRG_256x256 import (
    data as no3vi_data, source_img_norm_cfg, target_img_norm_cfg,
    source_train_pipeline, target_train_pipeline, test_pipeline)

# Replace the entire effective dataset config, including inherited VI flags.
data = deepcopy(no3vi_data)
data['_delete_'] = True
data['train']['target']['crop_pseudo_margins'] = [30, 30, 30, 30]
index_names = []
vi_names = []
model = dict(backbone=dict(
    in_channels=3, pretrained='open-mmlab://resnet101_v1c'))
uda = dict(input_channel_order=['NIR', 'R', 'G'])
archive_source_code = False
runner = dict(type='IterBasedRunner', max_iters=15000)

work_dir = '/root/shared-nvme/work_dirs/doris_no3vi_r101/goji30IRRG2goji26IRRG_hrda_r101_no3vi'
name = 'doris_goji30IRRG2goji26IRRG_hrda_r101_no3vi'
name_dataset = 'goji30IRRG2goji26IRRG_256x256_no3vi'
name_opt = 'adamw_6e-05_poly10warm_15k'

del deepcopy, no3vi_data
