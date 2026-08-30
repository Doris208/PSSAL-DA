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
