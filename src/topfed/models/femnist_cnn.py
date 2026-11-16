"""FEMNIST CNN backbone.

Two 5x5 conv layers (32 then 64 filters) with ReLU and 2x2 max pooling, then
FC(512) -> FC(62), mirroring the CIFAR backbone but for single-channel 28x28
inputs and the 62-class EMNIST/FEMNIST label space.

Spatial reduction on a 28x28 input: 28 -> 24 -> 12 -> 8 -> 4, giving a flatten
dimension of 64 * 4 * 4 = 1024.
"""

from __future__ import annotations

import torch
from torch import nn

FEMNIST_NUM_CLASSES = 62


def _flatten_dim(input_size: int) -> int:
    """Flattened feature size after two (conv5, pool2) stages."""
    if input_size < 16:
        raise ValueError(
            f"input_size must be at least 16 for two conv5/pool2 stages; "
            f"got {input_size}"
        )
    side = ((input_size - 4) // 2 - 4) // 2
    return 64 * side * side


class FemnistCNN(nn.Module):
    """``input_size`` is the (square) spatial size; the default matches the
    unpadded dataset and reproduces the reference flatten dimension."""

    def __init__(
        self, num_classes: int = FEMNIST_NUM_CLASSES, input_size: int = 28
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(_flatten_dim(input_size), 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
