# Adapted from ProCA for the local MMSegmentation-based HRDA project.

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class ProCAPrototypeEstimator(nn.Module):

    def __init__(self,
                 num_classes,
                 feature_dim,
                 ignore_index=255,
                 use_momentum=False,
                 momentum=0.9):
        super(ProCAPrototypeEstimator, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.ignore_index = ignore_index
        self.use_momentum = use_momentum
        self.momentum = momentum
        self.register_buffer('prototype',
                             torch.zeros(num_classes, feature_dim))
        self.register_buffer('amount', torch.zeros(num_classes))

    def _resize_label(self, labels, size):
        if labels.dim() == 3:
            labels = labels.unsqueeze(1)
        return F.interpolate(labels.float(), size=size,
                             mode='nearest').long().squeeze(1)

    def _class_sum_count(self, feats, labels):
        if feats.dim() != 4:
            raise ValueError('Expected features with shape [N, C, H, W].')
        labels = self._resize_label(labels, feats.shape[-2:])
        feats = feats.permute(0, 2, 3, 1).reshape(-1, self.feature_dim)
        labels = labels.reshape(-1)
        valid = (labels != self.ignore_index) & (labels >= 0) & \
            (labels < self.num_classes)
        feats = feats[valid]
        labels = labels[valid]

        sums = feats.new_zeros(self.num_classes, self.feature_dim)
        counts = feats.new_zeros(self.num_classes)
        if feats.numel() == 0:
            return sums, counts
        sums.index_add_(0, labels, feats)
        counts.index_add_(0, labels, feats.new_ones(labels.shape[0]))
        return sums, counts

    def _sync(self, sums, counts):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        return sums, counts

    @torch.no_grad()
    def update(self, feats, labels):
        sums, counts = self._class_sum_count(feats.detach(), labels.detach())
        sums, counts = self._sync(sums, counts)
        present = counts > 0
        if not torch.any(present):
            return counts

        means = sums[present] / counts[present].unsqueeze(1).clamp_min(1.0)
        if self.use_momentum:
            old_has_data = self.amount[present] > 0
            old = self.prototype[present]
            updated = means
            if torch.any(old_has_data):
                updated[old_has_data] = \
                    self.momentum * old[old_has_data] + \
                    (1.0 - self.momentum) * means[old_has_data]
            self.prototype[present] = updated
        else:
            old_count = self.amount[present].unsqueeze(1)
            new_count = counts[present].unsqueeze(1)
            total = (old_count + new_count).clamp_min(1.0)
            self.prototype[present] = \
                (self.prototype[present] * old_count + sums[present]) / total
        self.amount[present] += counts[present]
        return counts

    def load_distribution(self, path):
        data = torch.load(path, map_location='cpu')
        proto = data.get('prototype', data.get('prototypes', None))
        amount = data.get('amount', data.get('prototype_amount', None))
        if proto is None or amount is None:
            raise KeyError(
                'Prototype checkpoint must contain prototype and amount.')
        if tuple(proto.shape) != (self.num_classes, self.feature_dim):
            raise ValueError(
                f'Prototype shape {tuple(proto.shape)} does not match '
                f'[{self.num_classes}, {self.feature_dim}].')
        if tuple(amount.shape) != (self.num_classes, ):
            raise ValueError(
                f'Amount shape {tuple(amount.shape)} does not match '
                f'[{self.num_classes}].')
        self.prototype.copy_(proto.to(self.prototype.device,
                                      dtype=self.prototype.dtype))
        self.amount.copy_(amount.to(self.amount.device,
                                    dtype=self.amount.dtype))

    def state_for_save(self):
        return {
            'prototype': self.prototype.detach().cpu(),
            'amount': self.amount.detach().cpu(),
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim,
            'ignore_index': self.ignore_index,
        }
