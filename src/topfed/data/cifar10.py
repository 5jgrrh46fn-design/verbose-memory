"""CIFAR-10 loading utilities.

The training set is loaded with optional crop/flip augmentation; the test
set is split into a small server-side validation slice (``val_per_class``
samples per class, default 100) and the remaining clean test pool. The
ASR helper builds a triggered evaluation set from the clean test pool,
excluding samples whose original label is already the target class so
ASR measures induced flips, not free hits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from topfed.attacks.data.backdoor import apply_trigger

CIFAR10_NORMALIZE_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_NORMALIZE_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_NUM_CLASSES = 10


@dataclass(frozen=True, slots=True)
class CifarBundle:
    """Materialised CIFAR-10 split.

    ``train_x`` is augmented at training time on the fly via the
    ``train_transform`` callable closed over by the per-client DataLoader
    builder; the tensors stored here are pre-normalised and un-augmented.
    """

    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor


def _to_tensor_dataset(ds: datasets.CIFAR10) -> tuple[torch.Tensor, torch.Tensor]:
    # ``ds.data`` is uint8 (N, 32, 32, 3); convert to float (N, 3, 32, 32)
    # and normalise. Doing this once up front (rather than per-batch via a
    # Dataset transform) keeps the per-client DataLoader hot path tight.
    x = torch.from_numpy(ds.data).float().div_(255.0).permute(0, 3, 1, 2).contiguous()
    mean = torch.tensor(CIFAR10_NORMALIZE_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_NORMALIZE_STD).view(1, 3, 1, 1)
    x = (x - mean) / std
    y = torch.tensor(ds.targets, dtype=torch.long)
    return x, y


def load_cifar10(
    root: str | Path,
    *,
    val_per_class: int = 100,
    seed: int = 0,
    download: bool = True,
) -> CifarBundle:
    """Load CIFAR-10 and carve a server-side validation slice from the test set.

    The validation slice contains ``val_per_class`` samples per class,
    chosen by a seeded shuffle of each class's test indices. Remaining
    test samples are exposed as ``test_*`` for ASR / clean-test evaluation.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    train_ds = datasets.CIFAR10(
        root=str(root), train=True, download=download, transform=None
    )
    test_ds = datasets.CIFAR10(
        root=str(root), train=False, download=download, transform=None
    )

    train_x, train_y = _to_tensor_dataset(train_ds)
    test_x_full, test_y_full = _to_tensor_dataset(test_ds)

    rng = np.random.default_rng(seed)
    val_mask = np.zeros(len(test_y_full), dtype=bool)
    for c in range(CIFAR10_NUM_CLASSES):
        class_idx = np.flatnonzero(test_y_full.numpy() == c)
        chosen = rng.choice(class_idx, size=val_per_class, replace=False)
        val_mask[chosen] = True

    val_x = test_x_full[torch.from_numpy(val_mask)]
    val_y = test_y_full[torch.from_numpy(val_mask)]
    test_x = test_x_full[torch.from_numpy(~val_mask)]
    test_y = test_y_full[torch.from_numpy(~val_mask)]

    return CifarBundle(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
    )


def build_asr_loader(
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    target_class: int,
    trigger_size: int,
    trigger_amplitude: float,
    trigger_position: str = "bottom_right",
    batch_size: int = 256,
) -> DataLoader:
    """Construct the ASR evaluation loader.

    Samples whose original label is ``target_class`` are excluded so the
    reported ASR captures induced flips rather than samples that would
    have been classified correctly even without the trigger. The trigger
    is applied in normalised pixel space — ``trigger_amplitude`` is the
    value written into the patch (in normalised units) at
    ``trigger_position``, matching what the poisoner does at training time.
    """
    if test_x.dim() != 4:
        raise ValueError(f"test_x must be (N, C, H, W); got {tuple(test_x.shape)}")

    keep = test_y != target_class
    eval_x = test_x[keep]
    if eval_x.numel() == 0:
        raise ValueError(
            "no samples remain after excluding the target class; "
            "check the test set / target_class"
        )
    triggered = apply_trigger(
        eval_x,
        trigger_size=trigger_size,
        trigger_amplitude=trigger_amplitude,
        trigger_position=trigger_position,
    )
    target_y = torch.full((triggered.size(0),), target_class, dtype=torch.long)
    return DataLoader(
        TensorDataset(triggered, target_y), batch_size=batch_size, shuffle=False
    )
