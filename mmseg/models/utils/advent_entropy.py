# Adapted from ADVENT:
# https://github.com/valeoai/ADVENT
# Licensed under the Apache License, Version 2.0.

import math

import torch
import torch.nn.functional as F


def prob_2_entropy(logits, num_classes, eps=1e-30, normalize=True):
    """Convert logits to ADVENT weighted self-information maps."""
    prob = F.softmax(logits, dim=1)
    entropy = -(prob * torch.log2(prob + eps))
    if normalize:
        entropy = entropy / math.log2(num_classes)
    return entropy
