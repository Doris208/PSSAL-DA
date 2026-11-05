# Adapted from BIT-DA/SePiCo.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weight_reduce_loss


def downscale_label_ratio(gt,
                          scale_factor,
                          min_ratio,
                          n_classes,
                          ignore_index=255):
    assert scale_factor >= 1
    if scale_factor == 1:
        return gt.clone()
    bs, orig_c, orig_h, orig_w = gt.shape
    assert orig_c == 1
    trg_h, trg_w = orig_h // scale_factor, orig_w // scale_factor
    ignore_substitute = n_classes

    out = gt.clone()
    out[out == ignore_index] = ignore_substitute
    out = F.one_hot(
        out.squeeze(1), num_classes=n_classes + 1).permute(0, 3, 1, 2)
    out = F.avg_pool2d(out.float(), kernel_size=scale_factor)
    gt_ratio, out = torch.max(out, dim=1, keepdim=True)
    out[out == ignore_substitute] = ignore_index
    out[gt_ratio < min_ratio] = ignore_index
    assert list(out.shape) == [bs, 1, trg_h, trg_w], out.shape
    return out


def contrast_preparations(feat,
                          mask,
                          use_avg_pool,
                          scale_min_ratio,
                          num_classes,
                          ignore_index):
    if use_avg_pool:
        scale_factor = mask.shape[-1] // feat.shape[-1]
        mask = downscale_label_ratio(mask, scale_factor, scale_min_ratio,
                                     num_classes, ignore_index).long().detach()
    else:
        mask = F.interpolate(
            mask.float(), size=feat.shape[-2:], mode='nearest').long()
    channels = feat.size(1)
    feat = feat.permute(0, 2, 3, 1).contiguous().view(-1, channels)
    mask = mask.contiguous().view(-1)

    valid = mask != ignore_index
    return feat[valid], mask[valid]


def proto_reg(feat, mean=None, contrast_temp=100., contrast_norm=None, **kwargs):
    assert mean is not None
    assert contrast_norm is not None
    if feat.size(0) == 0:
        return feat.new_tensor(0., requires_grad=True)
    mean_feat = torch.mean(feat, 0, keepdim=True)
    proto_sim = mean_feat.mm(mean.permute(1, 0).contiguous()) / contrast_temp
    return torch.sum(torch.softmax(proto_sim, dim=1).log()) / contrast_norm


def proto_contrastive(feat,
                      mask,
                      mean=None,
                      index=-1,
                      contrast_temp=100.,
                      use_avg_pool=True,
                      scale_min_ratio=0.75,
                      num_classes=19,
                      weight=None,
                      class_weight=None,
                      reduction='mean',
                      avg_factor=None,
                      reg_weight=0,
                      ignore_index=255,
                      **kwargs):
    if index >= 0:
        feat = feat[index]
        mean = mean[index]
    feat, mask = contrast_preparations(feat, mask, use_avg_pool,
                                       scale_min_ratio, num_classes,
                                       ignore_index)
    if feat.size(0) == 0:
        return feat.new_tensor(0., requires_grad=True)
    logits = feat.mm(mean.permute(1, 0).contiguous()) / contrast_temp
    loss = F.cross_entropy(
        logits,
        mask,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)
    loss = weight_reduce_loss(
        loss, weight=weight.float() if weight is not None else None,
        reduction=reduction, avg_factor=avg_factor)
    if reg_weight > 0.:
        loss += reg_weight * proto_reg(
            feat,
            mean,
            contrast_temp,
            contrast_norm=num_classes * np.log(num_classes))
    return loss


def dist_contrastive(feat,
                     mask,
                     mean=None,
                     covariance=None,
                     ratio=1.0,
                     index=-1,
                     contrast_temp=100.,
                     use_avg_pool=True,
                     scale_min_ratio=0.75,
                     num_classes=19,
                     weight=None,
                     class_weight=None,
                     reduction='mean',
                     avg_factor=None,
                     reg_weight=0,
                     ignore_index=255,
                     **kwargs):
    if index >= 0:
        feat = feat[index]
        mean = mean[index]
        covariance = covariance[index]
    feat, mask = contrast_preparations(feat, mask, use_avg_pool,
                                       scale_min_ratio, num_classes,
                                       ignore_index)
    if feat.size(0) == 0:
        return feat.new_tensor(0., requires_grad=True)
    temp1 = feat.mm(mean.permute(1, 0).contiguous())
    covariance = covariance * ratio / contrast_temp
    temp2 = 0.5 * feat.pow(2).mm(covariance.permute(1, 0).contiguous())
    logits = (temp1 + temp2) / contrast_temp
    ce_loss = F.cross_entropy(
        logits,
        mask,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)
    jcl_loss = 0.5 * torch.sum(
        feat.pow(2).mul(covariance[mask]), dim=1) / contrast_temp
    loss = weight_reduce_loss(
        ce_loss + jcl_loss,
        weight=weight.float() if weight is not None else None,
        reduction=reduction,
        avg_factor=avg_factor)
    if reg_weight > 0.:
        loss += reg_weight * proto_reg(
            feat,
            mean,
            contrast_temp,
            contrast_norm=num_classes * np.log(num_classes))
    return loss


@LOSSES.register_module()
class ContrastiveLoss(nn.Module):

    def __init__(self,
                 use_dist=False,
                 use_bank=False,
                 use_reg=False,
                 use_avg_pool=True,
                 scale_min_ratio=0.75,
                 num_classes=None,
                 contrast_temp=100.,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 reg_relative_weight=1.0):
        super(ContrastiveLoss, self).__init__()
        assert not (use_dist and use_bank)
        assert num_classes is not None
        self.use_dist = use_dist
        self.use_bank = use_bank
        self.use_reg = use_reg
        self.use_avg_pool = use_avg_pool
        self.scale_min_ratio = scale_min_ratio
        self.contrast_temp = contrast_temp
        self.num_classes = num_classes
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.reg_weight = reg_relative_weight
        self.class_weight = get_class_weight(class_weight)
        if use_bank:
            raise NotImplementedError('BankCL is not enabled for this baseline.')
        self.contrast_criterion = dist_contrastive if use_dist else proto_contrastive

    def forward(self,
                feat,
                mask,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        class_weight = feat.new_tensor(self.class_weight) \
            if self.class_weight is not None and not isinstance(feat, list) \
            else None
        if isinstance(feat, list):
            loss_weight = self.loss_weight if isinstance(self.loss_weight, list) \
                else [self.loss_weight for _ in feat]
            loss = [
                loss_weight[i] * self.contrast_criterion(
                    feat,
                    mask,
                    weight=weight,
                    index=i,
                    contrast_temp=self.contrast_temp,
                    use_avg_pool=self.use_avg_pool,
                    scale_min_ratio=self.scale_min_ratio,
                    num_classes=self.num_classes,
                    class_weight=class_weight,
                    reduction=reduction,
                    avg_factor=avg_factor,
                    reg_weight=self.reg_weight if self.use_reg else 0,
                    **kwargs) for i in range(len(feat))
            ]
            return sum(loss)
        return self.loss_weight * self.contrast_criterion(
            feat,
            mask,
            weight=weight,
            contrast_temp=self.contrast_temp,
            use_avg_pool=self.use_avg_pool,
            scale_min_ratio=self.scale_min_ratio,
            num_classes=self.num_classes,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            reg_weight=self.reg_weight if self.use_reg else 0,
            **kwargs)
