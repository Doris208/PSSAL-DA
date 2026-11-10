import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule


class PhyReliabilityHead(nn.Module):
    """Lightweight physiology reliability head for VI maps."""

    def __init__(self,
                 in_channels=3,
                 mid_channels=32,
                 num_layers=2,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 out_act='sigmoid'):
        super().__init__()
        layers = []
        channels = in_channels
        hidden = max(mid_channels // 2, 1)
        for i in range(num_layers):
            out_channels = hidden if i == 0 else mid_channels
            layers.append(
                ConvModule(
                    channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            channels = out_channels
        layers.append(nn.Conv2d(channels, 1, kernel_size=1))
        self.layers = nn.Sequential(*layers)
        self.out_act = out_act

    def forward(self, x):
        x = self.layers(x)
        if self.out_act == 'sigmoid':
            x = torch.sigmoid(x)
        return x


def masked_bce_loss(pred, target, valid_mask):
    loss = F.binary_cross_entropy(
        pred, target.float(), reduction='none')
    loss = loss * valid_mask.float()
    denom = valid_mask.float().sum().clamp_min(1.0)
    return loss.sum() / denom


def target_phy_loss(pred, pseudo_label, confidence, valid_mask=None,
                    mode='legacy', threshold=0.968, positive_classes=(0, 1),
                    num_classes=5, pos_weight=0.5, neg_weight=0.5):
    """Target physiology supervision; diagnostics never enter MMSeg loss sums."""
    if mode not in ('legacy', 'balanced_high_conf'):
        raise ValueError('Unknown phy_target_supervision_mode: ' + mode)
    label = pseudo_label.detach()
    if label.dim() == 4:
        label = label[:, 0]
    conf = confidence.detach()
    if conf.dim() == 4:
        conf = conf[:, 0]
    vegetation = torch.zeros_like(label, dtype=torch.bool)
    for cls in positive_classes:
        vegetation |= label.eq(cls)
    target = vegetation.unsqueeze(1)
    high = conf.ge(threshold)
    if mode == 'legacy':
        # Exactly the previous positive-only behavior, including its mask scope.
        return masked_bce_loss(pred, target, (vegetation & high).unsqueeze(1)), {}
    if pos_weight <= 0 or neg_weight <= 0:
        raise ValueError('Balanced physiology group weights must be positive')
    valid = label.ne(255) & label.ge(0) & label.lt(num_classes)
    if valid_mask is not None:
        spatial = valid_mask[:, 0] if valid_mask.dim() == 4 else valid_mask
        valid &= spatial.detach().gt(0)
    positive = valid & high & vegetation
    negative = valid & high & ~vegetation
    # BCE remains float32 under mixed precision; only the existing head forward is used.
    with torch.cuda.amp.autocast(enabled=False):
        bce = F.binary_cross_entropy(pred.float(), target.float(), reduction='none')[:, 0]
    pos_count, neg_count = positive.sum(), negative.sum()
    loss_pos = (bce * positive).sum() / pos_count.clamp_min(1)
    loss_neg = (bce * negative).sum() / neg_count.clamp_min(1)
    wp = pos_weight * pos_count.gt(0).to(bce)
    wn = neg_weight * neg_count.gt(0).to(bce)
    loss = (wp * loss_pos + wn * loss_neg) / (wp + wn).clamp_min(1e-12)
    return loss, dict(pos_bce=loss_pos.detach(), neg_bce=loss_neg.detach(),
                      pos_count=pos_count.detach(), neg_count=neg_count.detach())
