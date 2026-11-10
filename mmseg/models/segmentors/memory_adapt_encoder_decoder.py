import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16

from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from ..utils.domain_prototype_memory import DomainPrototypeMemory
from .base import BaseSegmentor


class ASPPBranch(nn.Sequential):

    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPBranch, self).__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))


class MemoryAdaptStage1Head(nn.Module):

    def __init__(self,
                 in_channels=2048,
                 channels=256,
                 num_classes=5,
                 dilations=(6, 12, 18, 24)):
        super(MemoryAdaptStage1Head, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.aspp = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels), nn.ReLU(True))
        ] + [ASPPBranch(in_channels, channels, d) for d in dilations])
        self.project = nn.Sequential(
            nn.Conv2d(channels * (len(dilations) + 1), channels, 1,
                      bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(True), nn.Dropout(0.1),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(True))
        self.classifier = nn.Conv2d(channels, num_classes, 1)

    def forward(self, x):
        feats = torch.cat([branch(x) for branch in self.aspp], dim=1)
        feats = self.project(feats)
        return self.classifier(feats)


@SEGMENTORS.register_module()
class MemoryAdaptEncoderDecoder(BaseSegmentor):
    """MemoryAdaptNet deployment segmentor with stage-2 memory prediction."""

    def __init__(self,
                 backbone,
                 stage1_head,
                 memory_bottleneck,
                 memory_module,
                 stage2_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 feature_upsample_scale=1):
        super(MemoryAdaptEncoderDecoder, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        self.stage1_head = MemoryAdaptStage1Head(**stage1_head)
        self.memory_bottleneck = nn.Sequential(
            nn.Conv2d(
                memory_bottleneck['in_channels'],
                memory_bottleneck['out_channels'],
                memory_bottleneck.get('kernel_size', 3),
                stride=memory_bottleneck.get('stride', 1),
                padding=memory_bottleneck.get('padding', 1),
                bias=False),
            nn.BatchNorm2d(memory_bottleneck['out_channels']),
            nn.ReLU(inplace=True))
        memory_cfg = dict(memory_module)
        memory_cfg.pop('type', None)
        self.memory_module = DomainPrototypeMemory(**memory_cfg)
        self.stage2_head = nn.Conv2d(stage2_head['in_channels'],
                                     stage2_head['num_classes'], 1)
        self.num_classes = stage2_head['num_classes']
        self.align_corners = False
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if feature_upsample_scale < 1:
            raise ValueError('feature_upsample_scale must be >= 1')
        self.feature_upsample_scale = feature_upsample_scale

    def extract_feat(self, img):
        return self.backbone(img)

    def _stage_features(self, img):
        feats = self.extract_feat(img)
        stage4 = feats[-1]
        if self.feature_upsample_scale != 1:
            stage4 = F.interpolate(stage4, scale_factor=self.feature_upsample_scale,
                                   mode='bilinear', align_corners=False)
        return feats, stage4

    def forward_memory(self, img, use_memory=True):
        feats, stage4 = self._stage_features(img)
        stage1_logits = self.stage1_head(stage4)
        memory_feats = self.memory_bottleneck(stage4)
        output = dict(
            features=feats,
            stage4=stage4,
            memory_features=memory_feats,
            stage1_logits=stage1_logits)
        if use_memory:
            _, memory_context = self.memory_module(memory_feats,
                                                   stage1_logits)
            stage2_logits = self.stage2_head(memory_context)
            output['stage2_logits'] = stage2_logits
        return output

    def encode_decode(self, img, img_metas):
        outputs = self.forward_memory(img, use_memory=True)
        seg_logits = resize(
            outputs['stage2_logits'],
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return seg_logits

    def forward_train(self, img, img_metas, gt_semantic_seg, **kwargs):
        outputs = self.forward_memory(img, use_memory=True)
        return outputs

    def simple_test(self, img, img_meta, rescale=True):
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            return seg_pred.unsqueeze(0)
        return list(seg_pred.cpu().numpy())

    def aug_test(self, imgs, img_metas, rescale=True):
        assert rescale
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            seg_logit += self.inference(imgs[i], img_metas[i], rescale)
        seg_logit /= len(imgs)
        return list(seg_logit.argmax(dim=1).cpu().numpy())

    def inference(self, img, img_meta, rescale):
        assert self.test_cfg.mode in ['slide', 'whole']
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        return F.softmax(seg_logit, dim=1)

    def whole_inference(self, img, img_meta, rescale):
        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            seg_logit = resize(
                seg_logit,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return seg_logit

    def slide_inference(self, img, img_meta, rescale):
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, self.num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (x1, preds.shape[3] - x2, y1,
                                preds.shape[2] - y2))
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        return self.forward_test(img, img_metas, **kwargs)
