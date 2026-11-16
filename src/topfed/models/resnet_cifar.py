"""CIFAR-style ResNet-18 with GroupNorm.

The CIFAR variant of ResNet-18: a 3x3 stem convolution with 64 filters and no
max-pooling, four stages of two basic blocks each at 64/128/256/512 filters
(stride 2 at the start of stages two to four), global average pooling and a
linear classifier. With 10 classes the network has 11,173,962 parameters.

Every normalisation layer is GroupNorm rather than BatchNorm. GroupNorm
normalises over channel groups within each sample, so it carries no running
statistics: the state dict holds learnable weights and biases only and no
buffers, and the model behaves identically in training and evaluation mode.
That makes it suitable for federated training, where clients hold small,
non-IID batches and the server averages whole state dicts; ``groups`` sets
the number of channel groups per layer (two is the usual choice in
federated-learning work).

Spatial reduction on a 32x32 input: 32 -> 32 -> 16 -> 8 -> 4, after which the
adaptive average pool reduces any remaining spatial extent to 1x1, so the
classifier's input size does not depend on ``input_size``.
"""

from __future__ import annotations

import torch
from torch import nn

CIFAR10_NUM_CLASSES = 10
STAGE_WIDTHS = (64, 128, 256, 512)
BLOCKS_PER_STAGE = 2
DEFAULT_GROUPS = 2
# Four stages, three of them stride 2: the smallest input that still leaves a
# 1x1 map at the last stage.
MIN_INPUT_SIZE = 8


def _group_norm(channels: int, groups: int) -> nn.GroupNorm:
    if groups < 1 or channels % groups != 0:
        raise ValueError(
            f"groups must be a positive divisor of the channel count; "
            f"got groups={groups} for {channels} channels"
        )
    return nn.GroupNorm(groups, channels)


class BasicBlock(nn.Module):
    """Two 3x3 convolutions with a residual connection.

    The shortcut is the identity when the shape is unchanged and a 1x1
    strided convolution (with its own normalisation) otherwise.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int, groups: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm1 = _group_norm(out_channels, groups)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm2 = _group_norm(out_channels, groups)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                _group_norm(out_channels, groups),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNet18GN(nn.Module):
    """ResNet-18 for 32x32 colour images with GroupNorm throughout.

    ``input_size`` is the (square) spatial size of the input; it is validated
    but does not change the parameter count, because the classifier sits
    behind a global average pool. ``groups`` is the number of GroupNorm
    channel groups in every normalisation layer.
    """

    def __init__(
        self,
        num_classes: int = CIFAR10_NUM_CLASSES,
        input_size: int = 32,
        groups: int = DEFAULT_GROUPS,
    ) -> None:
        super().__init__()
        if input_size < MIN_INPUT_SIZE:
            raise ValueError(
                f"input_size must be at least {MIN_INPUT_SIZE} for four stages with "
                f"three stride-2 reductions; got {input_size}"
            )
        self.groups = groups
        self.conv1 = nn.Conv2d(3, STAGE_WIDTHS[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = _group_norm(STAGE_WIDTHS[0], groups)
        self.relu = nn.ReLU(inplace=True)
        in_channels = STAGE_WIDTHS[0]
        stages: list[nn.Module] = []
        for i, width in enumerate(STAGE_WIDTHS):
            stride = 1 if i == 0 else 2
            blocks: list[nn.Module] = []
            for b in range(BLOCKS_PER_STAGE):
                blocks.append(BasicBlock(in_channels, width, stride if b == 0 else 1, groups))
                in_channels = width
            stages.append(nn.Sequential(*blocks))
        self.layer1, self.layer2, self.layer3, self.layer4 = stages
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(STAGE_WIDTHS[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)
