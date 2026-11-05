# Adapted from SePiCo and mmsegmentation v0.16.0.
# Supports representation-learning heads whose losses need extra kwargs.

from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from ..builder import build_loss


class BaseDecodeHeadDecorator(BaseModule, metaclass=ABCMeta):

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 decoder_params=None,
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 init_cfg=None):
        super(BaseDecodeHeadDecorator, self).__init__(init_cfg)
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        self.sampler = build_pixel_sampler(sampler, context=self) \
            if sampler is not None else None
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        self.fp16_enabled = False

    def _init_inputs(self, in_channels, in_index, input_transform):
        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            self.in_channels = sum(in_channels) \
                if input_transform == 'resize_concat' else in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]
        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        pass

    def forward_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      seg_weight=None,
                      **kwargs):
        seg_logits = self.forward(inputs)
        return self.losses(seg_logits, gt_semantic_seg, seg_weight, **kwargs)

    def forward_test(self, inputs, img_metas, test_cfg):
        return self.forward(inputs)

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label, seg_weight=None, **kwargs):
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        return dict(
            loss_cl=self.loss_decode(
                seg_logit,
                seg_label,
                weight=seg_weight,
                ignore_index=self.ignore_index,
                **kwargs))
