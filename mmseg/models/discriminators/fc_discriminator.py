# Adapted from ADVENT:
# https://github.com/valeoai/ADVENT
# Licensed under the Apache License, Version 2.0.

from torch import nn

from mmseg.models.builder import MODELS


@MODELS.register_module()
class FCDiscriminator(nn.Module):
    """Fully convolutional output-space discriminator used by AdvEnt."""

    def __init__(self,
                 in_channels,
                 base_channels=64,
                 num_layers=5,
                 negative_slope=0.2):
        super(FCDiscriminator, self).__init__()
        assert num_layers == 5
        conv_args = dict(kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(in_channels, base_channels, **conv_args)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, **conv_args)
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4,
                               **conv_args)
        self.conv4 = nn.Conv2d(base_channels * 4, base_channels * 8,
                               **conv_args)
        self.leaky_relu = nn.LeakyReLU(
            negative_slope=negative_slope, inplace=True)
        self.classifier = nn.Conv2d(base_channels * 8, 1, **conv_args)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        return self.classifier(x)
