# Adapted from MemoryAdaptNet:
# https://github.com/ChenjieCSU/MemoryAdaptNet-master
# Code is provided by the original authors for non-commercial research use.

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .memory_attention import MemorySelfAttentionBlock


class DomainPrototypeMemory(nn.Module):
    """Domain-level class prototype memory used during train and inference."""

    def __init__(self,
                 num_classes,
                 memory_dim,
                 transform_channels,
                 num_prototypes_per_class=1,
                 output_channels=2048,
                 use_hard_aggregate=False,
                 update_strategy='cosine_similarity',
                 base_momentum=0.9,
                 base_lr=0.01,
                 adjust_by_learning_rate=True):
        super(DomainPrototypeMemory, self).__init__()
        assert num_prototypes_per_class == 1
        assert update_strategy in ['mean', 'cosine_similarity']
        self.num_classes = num_classes
        self.memory_dim = memory_dim
        self.transform_channels = transform_channels
        self.num_prototypes_per_class = num_prototypes_per_class
        self.output_channels = output_channels
        self.use_hard_aggregate = use_hard_aggregate
        self.update_strategy = update_strategy
        self.base_momentum = base_momentum
        self.base_lr = base_lr
        self.adjust_by_learning_rate = adjust_by_learning_rate

        self.register_buffer(
            'memory',
            torch.zeros(num_classes, num_prototypes_per_class, memory_dim))
        self.self_attention = MemorySelfAttentionBlock(
            key_in_channels=memory_dim,
            query_in_channels=memory_dim,
            transform_channels=transform_channels,
            out_channels=memory_dim,
            matmul_norm=True)
        self.fusion = nn.Sequential(
            nn.Conv2d(memory_dim * 2, output_channels, 3, padding=1,
                      bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True))

    def forward(self, feats, preds=None):
        batch_size = feats.size(0)
        selected_memory = self.memory.detach().clone()[:, 0, :].t().contiguous()
        selected_memory = selected_memory.view(1, self.memory_dim,
                                               self.num_classes, 1)
        selected_memory = selected_memory.expand(batch_size, -1, -1, -1)
        memory_context = self.self_attention(feats, selected_memory)
        return self.memory, self.fusion(torch.cat([feats, memory_context], 1))

    def _momentum(self, learning_rate=None):
        momentum = self.base_momentum
        if self.adjust_by_learning_rate and learning_rate is not None:
            momentum = self.base_momentum / self.base_lr * learning_rate
        return max(0.0, min(float(momentum), 1.0))

    @torch.no_grad()
    def update(self,
               features,
               segmentation,
               ignore_index=255,
               learning_rate=None,
               allow_uninitialized=True):
        """Update class memory from labels or filtered pseudo labels."""
        if segmentation.dim() == 4:
            segmentation = segmentation.squeeze(1)
        segmentation = F.interpolate(
            segmentation.unsqueeze(1).float(),
            size=features.shape[2:],
            mode='nearest').squeeze(1).long()

        features = features.detach()
        flat_feats = features.permute(0, 2, 3, 1).reshape(-1,
                                                          self.memory_dim)
        flat_seg = segmentation.reshape(-1)
        sums = features.new_zeros(self.num_classes, self.memory_dim)
        counts = features.new_zeros(self.num_classes)

        for cls_id in range(self.num_classes):
            mask = flat_seg == cls_id
            if mask.any():
                cls_feats = flat_feats[mask]
                if self.update_strategy == 'cosine_similarity' and \
                        not self._is_empty(cls_id):
                    memory = self.memory[cls_id, 0].expand_as(cls_feats)
                    similarity = F.cosine_similarity(cls_feats, memory, dim=1)
                    weights = 1 - similarity
                    denom = weights.sum().clamp_min(1e-6)
                    cls_sum = (cls_feats * (weights / denom).unsqueeze(1)).sum(0)
                    cls_count = features.new_tensor(1.0)
                else:
                    cls_sum = cls_feats.sum(0)
                    cls_count = features.new_tensor(float(cls_feats.size(0)))
                sums[cls_id] = cls_sum
                counts[cls_id] = cls_count

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sums)
            dist.all_reduce(counts)

        momentum = self._momentum(learning_rate)
        for cls_id in range(self.num_classes):
            if counts[cls_id] <= 0:
                continue
            if self._is_empty(cls_id):
                if not allow_uninitialized:
                    continue
                new_value = sums[cls_id] / counts[cls_id].clamp_min(1.0)
            elif self.update_strategy == 'cosine_similarity':
                new_value = sums[cls_id] / counts[cls_id].clamp_min(1.0)
            else:
                new_value = sums[cls_id] / counts[cls_id].clamp_min(1.0)
            updated = (1 - momentum) * self.memory[cls_id, 0] + \
                momentum * new_value
            self.memory[cls_id, 0].copy_(updated)

    def _is_empty(self, cls_id):
        return bool(torch.count_nonzero(self.memory[cls_id, 0]) == 0)

    def has_valid_memory(self):
        return bool(torch.count_nonzero(self.memory) > 0)
