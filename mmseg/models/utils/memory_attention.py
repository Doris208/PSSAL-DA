# Adapted from MemoryAdaptNet:
# https://github.com/ChenjieCSU/MemoryAdaptNet-master
# Code is provided by the original authors for non-commercial research use.

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemorySelfAttentionBlock(nn.Module):
    """Self-attention block for querying class prototype memory."""

    def __init__(self,
                 key_in_channels,
                 query_in_channels,
                 transform_channels,
                 out_channels,
                 matmul_norm=True):
        super(MemorySelfAttentionBlock, self).__init__()
        self.query_project = nn.Sequential(
            nn.Conv2d(query_in_channels, transform_channels, 1, bias=False),
            nn.BatchNorm2d(transform_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(transform_channels, transform_channels, 1, bias=False),
            nn.BatchNorm2d(transform_channels),
            nn.ReLU(inplace=True))
        self.key_project = nn.Sequential(
            nn.Conv2d(key_in_channels, transform_channels, 1, bias=False),
            nn.BatchNorm2d(transform_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(transform_channels, transform_channels, 1, bias=False),
            nn.BatchNorm2d(transform_channels),
            nn.ReLU(inplace=True))
        self.value_project = nn.Sequential(
            nn.Conv2d(key_in_channels, transform_channels, 1, bias=False),
            nn.BatchNorm2d(transform_channels),
            nn.ReLU(inplace=True))
        self.out_project = nn.Sequential(
            nn.Conv2d(transform_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.transform_channels = transform_channels
        self.matmul_norm = matmul_norm

    def forward(self, query_feats, key_feats):
        batch_size = query_feats.size(0)
        query = self.query_project(query_feats).flatten(2).transpose(1, 2)
        key = self.key_project(key_feats).flatten(2)
        value = self.value_project(key_feats).flatten(2).transpose(1, 2)
        sim_map = torch.matmul(query, key)
        if self.matmul_norm:
            sim_map = (self.transform_channels**-0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value).transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, *query_feats.shape[2:])
        return self.out_project(context)
