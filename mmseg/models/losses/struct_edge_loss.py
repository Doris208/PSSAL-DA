import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


def _as_label(label):
    if label.dim() == 4:
        label = label[:, 0]
    return label.long()


def label_to_edge(label, ignore_index=255, edge_radius=2):
    label = _as_label(label)
    valid = label.ne(ignore_index)
    edge = torch.zeros_like(label, dtype=torch.bool)
    shifts = ((1, 0), (-1, 0), (0, 1), (0, -1),
              (1, 1), (1, -1), (-1, 1), (-1, -1))
    for dy, dx in shifts:
        shifted = torch.roll(label, shifts=(dy, dx), dims=(1, 2))
        shifted_valid = torch.roll(valid, shifts=(dy, dx), dims=(1, 2))
        if dy > 0:
            shifted_valid[:, :dy, :] = False
        elif dy < 0:
            shifted_valid[:, dy:, :] = False
        if dx > 0:
            shifted_valid[:, :, :dx] = False
        elif dx < 0:
            shifted_valid[:, :, dx:] = False
        edge |= valid & shifted_valid & label.ne(shifted)

    edge = edge.float().unsqueeze(1)
    if edge_radius > 0:
        kernel_size = edge_radius * 2 + 1
        edge = F.max_pool2d(
            edge, kernel_size=kernel_size, stride=1, padding=edge_radius)
    return edge


def resize_weight(weight, size):
    if weight is None:
        return None
    if weight.dim() == 3:
        weight = weight.unsqueeze(1)
    if weight.shape[-2:] != size:
        weight = F.interpolate(weight.float(), size=size, mode='nearest')
    return weight.float()


def prob_to_edge(logits, eps=1e-6):
    prob = torch.softmax(logits, dim=1)
    sobel_x = logits.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = logits.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(prob.shape[1], 1, 1, 1)
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(prob.shape[1], 1, 1, 1)
    gx = F.conv2d(prob, sobel_x, padding=1, groups=prob.shape[1])
    gy = F.conv2d(prob, sobel_y, padding=1, groups=prob.shape[1])
    edge = torch.sqrt(gx.pow(2) + gy.pow(2) + eps).amax(dim=1, keepdim=True)
    norm = edge.amax(dim=(2, 3), keepdim=True).clamp_min(eps)
    return (edge / norm).clamp(0.0, 1.0)


@LOSSES.register_module()
class StructEdgeLoss(nn.Module):
    """Structure-gated boundary detail auxiliary loss."""

    def __init__(self,
                 edge_radius=2,
                 pred_edge='sobel_prob',
                 loss_weight=1.0,
                 bce_weight=1.0,
                 ce_weight=0.3,
                 dice_weight=0.0,
                 ignore_index=255,
                 loss_name='loss_struct_edge',
                 eps=1e-6):
        super(StructEdgeLoss, self).__init__()
        if pred_edge != 'sobel_prob':
            raise ValueError('StructEdgeLoss only supports sobel_prob.')
        self.edge_radius = edge_radius
        self.pred_edge = pred_edge
        self.loss_weight = loss_weight
        self.bce_weight = bce_weight
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name
        self.eps = eps

    @property
    def loss_name(self):
        return self._loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                struct_edge_weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        del weight, avg_factor, reduction_override, kwargs
        target = _as_label(target)
        target_edge = label_to_edge(target, self.ignore_index,
                                    self.edge_radius).to(pred.dtype)
        pred_edge = prob_to_edge(pred, self.eps)
        if target_edge.shape[-2:] != pred_edge.shape[-2:]:
            target_edge = F.interpolate(
                target_edge, size=pred_edge.shape[-2:], mode='nearest')
            target = F.interpolate(
                target.unsqueeze(1).float(),
                size=pred_edge.shape[-2:],
                mode='nearest').squeeze(1).long()

        valid = target.ne(self.ignore_index).unsqueeze(1)
        base_edge_weight = target_edge * valid.float()
        edge_weight = base_edge_weight
        struct_edge_weight = resize_weight(struct_edge_weight,
                                           pred_edge.shape[-2:])
        if struct_edge_weight is not None:
            edge_weight = edge_weight * struct_edge_weight.to(edge_weight)
        denom = base_edge_weight.sum().clamp_min(1.0)

        bce = F.binary_cross_entropy(
            pred_edge.clamp(self.eps, 1.0 - self.eps),
            target_edge,
            reduction='none')
        loss_bce = (bce * edge_weight).sum() / denom

        loss_ce = pred.sum() * 0.0
        if self.ce_weight > 0:
            ce = F.cross_entropy(
                pred, target, ignore_index=self.ignore_index,
                reduction='none')
            ce_weight_map = edge_weight.squeeze(1)
            ce_denom = base_edge_weight.squeeze(1).sum().clamp_min(1.0)
            loss_ce = (ce * ce_weight_map).sum() / ce_denom

        loss_dice = pred.sum() * 0.0
        if self.dice_weight > 0:
            inter = (pred_edge * target_edge * edge_weight).sum()
            union = ((pred_edge + target_edge) * edge_weight).sum()
            loss_dice = 1.0 - (2.0 * inter + self.eps) / (union + self.eps)

        return self.loss_weight * (
            self.bce_weight * loss_bce + self.ce_weight * loss_ce +
            self.dice_weight * loss_dice)
