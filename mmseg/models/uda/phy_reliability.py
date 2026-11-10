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
    hierarchical = isinstance(pred, dict)
    if hierarchical and mode != 'balanced_high_conf':
        raise ValueError('H-Phy requires balanced_high_conf target supervision')
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
        if hierarchical:
            bce = F.binary_cross_entropy_with_logits(
                pred['logit_veg'].float(), target.float(), reduction='none')[:, 0]
        else:
            bce = F.binary_cross_entropy(pred.float(), target.float(), reduction='none')[:, 0]
    pos_count, neg_count = positive.sum(), negative.sum()
    loss_pos = (bce * positive).sum() / pos_count.clamp_min(1)
    loss_neg = (bce * negative).sum() / neg_count.clamp_min(1)
    wp = pos_weight * pos_count.gt(0).to(bce)
    wn = neg_weight * neg_count.gt(0).to(bce)
    loss = (wp * loss_pos + wn * loss_neg) / (wp + wn).clamp_min(1e-12)
    return loss, dict(pos_bce=loss_pos.detach(), neg_bce=loss_neg.detach(),
                      pos_count=pos_count.detach(), neg_count=neg_count.detach())


class HierarchicalPhyReliabilityHead(nn.Module):
    """VI-only vegetation response and conditional Goji response."""

    def __init__(self, in_channels=3, mid_channels=32, num_layers=2,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        if in_channels != 3:
            raise ValueError('H-Phy requires NDVI, NDRE, CIRE in three channels')
        # Reuse the legacy encoder construction without changing its state keys.
        binary = PhyReliabilityHead(in_channels, mid_channels, num_layers,
                                    norm_cfg, act_cfg)
        self.encoder = nn.Sequential(*list(binary.layers.children())[:-1])
        channels = binary.layers[-1].in_channels
        self.veg_head = binary.layers[-1]
        self.gv_head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, vi):
        features = self.encoder(vi)
        z_veg, z_gv = self.veg_head(features), self.gv_head(features)
        r_veg, q_goji = z_veg.sigmoid(), z_gv.sigmoid()
        return dict(logit_veg=z_veg, logit_gv=z_gv, r_veg=r_veg,
                    q_goji_given_veg=q_goji, p_goji=r_veg * q_goji,
                    p_other_veg=r_veg * (1 - q_goji), p_nonveg=1 - r_veg)


def balanced_logit_bce(logits, target, positive, negative):
    """Equal present-group means, including graph-connected empty batches."""
    bce = F.binary_cross_entropy_with_logits(
        logits.float(), target.float(), reduction='none')
    pos_count, neg_count = positive.sum(), negative.sum()
    pos_loss = (bce * positive).sum() / pos_count.clamp_min(1)
    neg_loss = (bce * negative).sum() / neg_count.clamp_min(1)
    groups = pos_count.gt(0).to(bce) + neg_count.gt(0).to(bce)
    return (pos_loss + neg_loss) / groups.clamp_min(1)


def hierarchical_source_loss(response, gt, beta_gv=1.0):
    if gt.dim() == 3:
        gt = gt.unsqueeze(1)
    valid = gt.ge(0) & gt.lt(5)
    goji, other = gt.eq(0), gt.eq(1)
    vegetation = goji | other
    with torch.cuda.amp.autocast(enabled=False):
        bce = F.binary_cross_entropy_with_logits(
            response['logit_veg'].float(), vegetation.float(), reduction='none')
        loss_veg = (bce * valid).sum() / valid.sum().clamp_min(1)
        loss_gv = balanced_logit_bce(response['logit_gv'], goji, goji, other)
    combined = (loss_veg + beta_gv * loss_gv) / (1 + beta_gv)
    return combined, dict(veg=loss_veg, gv=loss_gv,
                         valid_count=valid.sum(), goji_count=goji.sum(),
                         other_veg_count=other.sum())


def phy_gate_strength(iteration, schedule='legacy_warmup_linear',
                      initial_strength=0.10, full_iter=4000, warmup=6000, rampup=4000):
    """One schedule definition shared by training, probes and checkpoint replay."""
    if schedule == 'early_nonzero_linear':
        if not 0 < initial_strength <= 1 or not 0 < full_iter < float('inf'):
            raise ValueError('H-Phy initial strength must be in (0,1] and full_iter must be positive and finite')
        progress = min(1.0, max(0.0, iteration / float(full_iter)))
        return initial_strength + (1 - initial_strength) * progress
    if schedule == 'legacy_warmup_linear':
        return min(1.0, max(0.0, (iteration - warmup) / max(float(rampup), 1.0)))
    raise ValueError('Unknown H-Phy gate schedule: ' + str(schedule))


def phy_gate_parameters(config):
    """Canonical effective settings, with historical defaults for unmarked heads."""
    schedule = config.get('phy_gate_schedule', 'legacy_warmup_linear')
    if config.get('mode', 'legacy_binary') != 'hierarchical_gv':
        schedule = 'legacy_warmup_linear'
    parameters = dict(schedule=schedule)
    if schedule == 'early_nonzero_linear':
        parameters.update(initial_strength=config.get('phy_gate_initial_strength', .10),
                          full_iter=config.get('phy_gate_full_iter', 4000))
    else:
        parameters.update(warmup=config.get('phy_warmup_iters', 1000),
                          rampup=config.get('phy_rampup_iters', 0))
    phy_gate_strength(0, **parameters)
    return parameters


def restore_phy_gate_config(requested, saved):
    """Replay restores the saved schedule while retaining other requested checks."""
    keys = ('phy_gate_schedule', 'phy_gate_initial_strength', 'phy_gate_full_iter',
            'phy_warmup_iters', 'phy_rampup_iters')
    restored = type(requested)(requested)
    for key in keys:
        restored.pop(key, None)
        if key in saved:
            restored[key] = saved[key]
    phy_gate_parameters(restored)
    return restored


def hierarchical_phy_gate(response, label, weight, iteration, warmup=6000,
                          rampup=4000, min_weight=0.75, enabled=True, valid_mask=None,
                          schedule='legacy_warmup_linear', initial_strength=.10,
                          full_iter=4000, return_details=False):
    """Detached, continuous class-conditioned gate on supported vegetation pixels."""
    goji = response['p_goji'][:, 0].detach().float()
    other = response['p_other_veg'][:, 0].detach().float()
    agreement = torch.where(label.eq(0), goji,
                            torch.where(label.eq(1), other, torch.ones_like(goji)))
    if not 0 < min_weight <= 1:
        raise ValueError('H-Phy min_weight must be in (0,1]')
    ramp = phy_gate_strength(iteration, schedule, initial_strength, full_iter, warmup, rampup)
    if not enabled:
        ramp = 0.0
    valid = weight.gt(0) if valid_mask is None else valid_mask.reshape_as(label).gt(0)
    active = (label.eq(0) | label.eq(1)) & valid
    gate = torch.where(active, 1 - ramp * (1 - min_weight) * (1 - agreement),
                       torch.ones_like(agreement)).to(weight)
    after = weight * gate
    if return_details:
        return dict(rho=ramp, S_phy=agreement, G_phy=gate, A_phy=1-gate,
                    W_before=weight, W_after=after)
    return after, gate, agreement


def validate_phy_checkpoint(mode, state_dict, prefix='phy_head.'):
    keys = [key[len(prefix):] for key in state_dict if key.startswith(prefix)]
    saved_mode = ('hierarchical_gv' if any(key.startswith(('encoder.', 'veg_head.', 'gv_head.'))
                                         for key in keys) else 'legacy_binary')
    if not keys or saved_mode != mode:
        raise RuntimeError('Physiology checkpoint mode mismatch: saved={}, requested={}. '
                           'Use the historical config for legacy replay. A new H-Phy '
                           'experiment requires standard initialization.'.format(saved_mode, mode))
