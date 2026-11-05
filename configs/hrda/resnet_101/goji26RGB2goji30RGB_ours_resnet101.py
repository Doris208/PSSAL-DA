_base_ = ['../goji26RGB2goji30RGB_hrda_3vi_proto_struct_edge.py']

model = dict(
    backbone=dict(
        _delete_=True,
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        pretrained='/root/HRDA-master/pretrained/resnet101-5d3b4d8f.pth',
        in_channels=6),
    decode_head=dict(in_channels=[256, 512, 1024, 2048]))

uda = dict(
    proto_cfg=dict(target_weight_thr=0.20))

data = dict(
    train=dict(target=dict(crop_pseudo_margins=[0, 0, 0, 0])))

work_dir = '/root/shared-nvme/work_dirs/resnet_101_new/goji26RGB2goji30RGB_ours_resnet101'
name = 'goji26RGB2goji30RGB_ours_resnet101'
name_architecture = 'hrda1-128-0.1_daformer_sepaspp_r101'
name_encoder = 'r101'
