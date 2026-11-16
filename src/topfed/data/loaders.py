"""Per-client DataLoader construction.

Builds one ``DataLoader`` per client over a partition of the global
training tensors. Optional train-time augmentation (random crop with
4-pixel padding + horizontal flip) is applied via a per-batch transform
on the input tensor — same convention as the original FedAvg / Bagdasaryan
pipelines.

Images may carry a constant border added by :func:`pad_images`. Such a
border is a region that natural content never varies; augmentation
leaves it in place (only the interior is cropped and flipped) so it stays
constant on every training sample.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch.utils.data import DataLoader, Dataset

from topfed.core.types import ClientId


class _AugmentedTensorDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """TensorDataset variant that runs an optional transform on ``x``.

    The transform receives a single ``(C, H, W)`` tensor and returns a
    same-shaped tensor.
    """

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        transform: Callable[[torch.Tensor], torch.Tensor] | None,
    ) -> None:
        if x.size(0) != y.size(0):
            raise ValueError(
                f"x and y disagree on length: {x.size(0)} vs {y.size(0)}"
            )
        self._x = x
        self._y = y
        self._transform = transform

    def __len__(self) -> int:
        return self._x.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self._x[idx]
        if self._transform is not None:
            sample = self._transform(sample)
        return sample, self._y[idx]


def pad_images(x: torch.Tensor, pad_border: int, value: float = 0.0) -> torch.Tensor:
    """Surround every image in ``x`` (``(N, C, H, W)``) with a constant border.

    A zero border in normalised pixel space is the per-channel mean colour, so
    it adds no signal to the classification task; it exists to give a trigger
    somewhere to live that honest data never touches. ``pad_border=0``
    returns the input unchanged.
    """
    if pad_border < 0:
        raise ValueError(f"pad_border must be >= 0; got {pad_border}")
    if pad_border == 0:
        return x
    if x.dim() != 4:
        raise ValueError(f"expected (N, C, H, W); got {tuple(x.shape)}")
    p = pad_border
    return torch.nn.functional.pad(x, (p, p, p, p), mode="constant", value=value)


def _random_crop_and_flip(x: torch.Tensor) -> torch.Tensor:
    """Pad-4 random crop back to the input size plus 50% horizontal flip."""
    h, w = x.shape[-2:]
    padded = torch.nn.functional.pad(x, (4, 4, 4, 4), mode="reflect")
    h_off = int(torch.randint(0, 9, (1,)).item())
    w_off = int(torch.randint(0, 9, (1,)).item())
    cropped = padded[:, h_off : h_off + h, w_off : w_off + w]
    if torch.rand(()) < 0.5:
        cropped = torch.flip(cropped, dims=(-1,))
    return cropped


def _make_augmentation(
    pad_border: int,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Crop-and-flip that keeps a constant ``pad_border`` ring untouched."""
    if pad_border == 0:
        return _random_crop_and_flip

    def transform(x: torch.Tensor) -> torch.Tensor:
        p = pad_border
        interior = _random_crop_and_flip(x[:, p:-p, p:-p])
        out = x.clone()
        out[:, p:-p, p:-p] = interior
        return out

    return transform


def build_client_loaders(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    partition: list[torch.Tensor] | list,
    *,
    batch_size: int,
    augment: bool = True,
    shuffle: bool = True,
    drop_last: bool = False,
    pad_border: int = 0,
) -> dict[ClientId, DataLoader]:
    """Build one ``DataLoader`` per client from a partition of indices.

    ``partition[i]`` holds the row indices into ``train_x`` / ``train_y``
    assigned to client ``c{i}``. Empty shards are still returned (with an
    empty loader) so the server's ``num_clients`` matches reality; the
    backdoor poisoner short-circuits cleanly on empty batches.

    ``pad_border`` tells the augmentation how wide a constant border the
    images already carry (see :func:`pad_images`) so cropping never moves
    natural content into it.
    """
    if pad_border < 0:
        raise ValueError(f"pad_border must be >= 0; got {pad_border}")
    transform = _make_augmentation(pad_border) if augment else None
    loaders: dict[ClientId, DataLoader] = {}
    for i, idx in enumerate(partition):
        idx_t = torch.as_tensor(idx, dtype=torch.long)
        ds = _AugmentedTensorDataset(
            train_x[idx_t], train_y[idx_t], transform=transform
        )
        loaders[f"c{i}"] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle and len(ds) > 0,
            drop_last=drop_last,
        )
    return loaders
