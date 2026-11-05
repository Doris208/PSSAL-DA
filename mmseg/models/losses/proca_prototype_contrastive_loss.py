# Adapted from ProCA for feature-level and output-level prototype contrast.

import torch
import torch.nn.functional as F


def proca_prototype_contrastive_loss(feats,
                                     labels,
                                     prototypes,
                                     temperature=50.0,
                                     ignore_index=255):
    if labels.dim() == 3:
        labels = labels.unsqueeze(1)
    labels = F.interpolate(
        labels.float(), size=feats.shape[-2:], mode='nearest').long()
    labels = labels.squeeze(1)
    channels = feats.shape[1]
    feats = feats.permute(0, 2, 3, 1).reshape(-1, channels)
    labels = labels.reshape(-1)
    valid = (labels != ignore_index) & (labels >= 0) & \
        (labels < prototypes.shape[0])
    feats = feats[valid]
    labels = labels[valid]
    if feats.numel() == 0:
        return prototypes.sum() * 0.0

    # ProCA updates prototype buffers during the same forward pass. Clone the
    # detached snapshot so later buffer updates do not invalidate autograd's
    # version check for this loss.
    proto_snapshot = prototypes.detach().clone()
    logits = feats.mm(proto_snapshot.t()) / temperature
    return F.cross_entropy(logits, labels, ignore_index=ignore_index)
