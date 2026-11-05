# Adapted from BIT-DA/SePiCo.

from collections import deque

import torch


class ProtoEstimator:

    def __init__(self, dim, class_num, memory_length=0, resume=''):
        self.dim = dim
        self.class_num = class_num
        self.memory_length = memory_length
        self.resume = resume
        self._initialized = False

    def _init_tensors(self, device):
        if self.resume:
            checkpoint = torch.load(self.resume, map_location='cpu')
            self.CoVariance = checkpoint['CoVariance'].to(device)
            self.Ave = checkpoint['Ave'].to(device)
            self.Amount = checkpoint['Amount'].to(device)
        else:
            self.CoVariance = torch.zeros(self.class_num, self.dim, device=device)
            self.Ave = torch.zeros(self.class_num, self.dim, device=device)
            self.Amount = torch.zeros(self.class_num, device=device)
        length = max(int(self.memory_length), 1)
        self.MemoryBank = [
            deque([self.Ave[cls].unsqueeze(0).detach()], maxlen=length)
            for cls in range(self.class_num)
        ]
        self._initialized = True

    def update_proto(self, features, labels):
        if not self._initialized:
            self._init_tensors(features.device)
        if features.numel() == 0 or labels.numel() == 0:
            return

        labels = labels.long()
        valid = (labels >= 0) & (labels < self.class_num)
        features = features[valid]
        labels = labels[valid]
        if features.numel() == 0:
            return

        n, dim = features.size()
        num_classes = self.class_num
        features_exp = features.view(n, 1, dim).expand(n, num_classes, dim)
        onehot = features.new_zeros(n, num_classes)
        onehot.scatter_(1, labels.view(-1, 1), 1)
        onehot_exp = onehot.view(n, num_classes, 1).expand(n, num_classes, dim)
        features_by_class = features_exp.mul(onehot_exp)

        amount = onehot_exp.sum(0)
        amount[amount == 0] = 1
        ave = features_by_class.sum(0) / amount

        for cls in torch.unique(labels):
            self.MemoryBank[int(cls.item())].append(
                ave[int(cls.item())].unsqueeze(0).detach())

        var = features_by_class - ave.expand(n, num_classes, dim).mul(onehot_exp)
        var = var.pow(2).sum(0).div(amount)

        weight_sum = onehot.sum(0).view(num_classes, 1).expand(num_classes, dim)
        weight = weight_sum.div(
            weight_sum + self.Amount.view(num_classes, 1).expand(num_classes, dim))
        weight[weight != weight] = 0
        additional = weight.mul(1 - weight).mul((self.Ave - ave).pow(2))

        self.CoVariance = (
            self.CoVariance.mul(1 - weight) + var.mul(weight)).detach() \
            + additional.detach()
        self.Ave = (self.Ave.mul(1 - weight) + ave.mul(weight)).detach()
        self.Amount = self.Amount + onehot.sum(0)

    def save_proto(self, path):
        torch.save(
            {
                'CoVariance': self.CoVariance.cpu(),
                'Ave': self.Ave.cpu(),
                'Amount': self.Amount.cpu()
            }, path)
