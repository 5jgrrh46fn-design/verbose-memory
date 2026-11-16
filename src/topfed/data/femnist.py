"""FEMNIST loading via Flower Datasets, with natural per-writer partitions.

FEMNIST (the federated EMNIST variant) is partitioned by the identity of the
writer who produced each character, giving organic, uncontrolled non-IID
heterogeneity rather than the synthetic Dirichlet split used for CIFAR-10. We
draw the data from the ``flwrlabs/femnist`` Hugging Face dataset through Flower
Datasets' ``NaturalIdPartitioner`` (keyed on ``writer_id``).

A deterministic, seeded shuffle selects ``num_clients`` writers as the client
population and a disjoint set of held-out writers to form the server's clean
validation slice and the global test pool. Images are single-channel 28x28;
pixel values are standardised using statistics computed from the loaded
training pool, so the loader is agnostic to the dataset's ink/background
convention.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

FEMNIST_NUM_CLASSES = 62


@dataclass(frozen=True, slots=True)
class FemnistBundle:
    """Materialised FEMNIST split with the per-sample writer identity.

    ``train_writers`` is parallel to ``train_x`` / ``train_y`` and carries the
    integer writer id behind each training sample, consumed by the natural
    partitioner to form one client per writer.
    """

    train_x: torch.Tensor
    train_y: torch.Tensor
    train_writers: np.ndarray
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    # The standardisation applied to every image above (training-set pixel
    # mean and standard deviation), so other image sources can be brought
    # into the same input space.
    norm_mean: float = 0.0
    norm_std: float = 1.0


def _partition_to_arrays(part: Any) -> tuple[np.ndarray, np.ndarray]:
    images = np.stack([np.asarray(im, dtype=np.uint8) for im in part["image"]])
    labels = np.asarray(part["character"], dtype=np.int64)
    return images, labels


def load_femnist(
    root: str | Path,
    *,
    num_clients: int,
    val_per_class: int = 30,
    held_out_writers: int = 60,
    test_cap: int = 8000,
    seed: int = 0,
    download: bool = True,  # noqa: ARG001 — kept for loader-API symmetry
) -> FemnistBundle:
    """Load FEMNIST and carve client / validation / test splits by writer.

    ``num_clients`` distinct writers form the client population; a disjoint
    ``held_out_writers`` writers are pooled to draw the server validation slice
    (up to ``val_per_class`` per class) and the clean test set (capped at
    ``test_cap``). Selection is deterministic in ``seed``.
    """
    # Route the Hugging Face cache under the project data root so it lands on
    # the durable data volume alongside the other datasets.
    os.environ.setdefault("HF_HOME", str(Path(root) / "hf"))
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import NaturalIdPartitioner

    fds = FederatedDataset(
        dataset="flwrlabs/femnist",
        partitioners={"train": NaturalIdPartitioner(partition_by="writer_id")},
    )
    partitioner = fds.partitioners["train"]
    num_writers = partitioner.num_partitions
    if num_clients + held_out_writers > num_writers:
        raise ValueError(
            f"need {num_clients + held_out_writers} writers but FEMNIST has "
            f"{num_writers}"
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(num_writers)
    client_writers = order[:num_clients]
    pool_writers = order[num_clients : num_clients + held_out_writers]

    train_imgs: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    train_writers: list[np.ndarray] = []
    for wi in client_writers:
        imgs, labels = _partition_to_arrays(fds.load_partition(int(wi), "train"))
        train_imgs.append(imgs)
        train_labels.append(labels)
        train_writers.append(np.full(labels.shape[0], int(wi), dtype=np.int64))

    pool_imgs: list[np.ndarray] = []
    pool_labels: list[np.ndarray] = []
    for wi in pool_writers:
        imgs, labels = _partition_to_arrays(fds.load_partition(int(wi), "train"))
        pool_imgs.append(imgs)
        pool_labels.append(labels)

    train_arr = np.concatenate(train_imgs)
    train_y = np.concatenate(train_labels)
    writers = np.concatenate(train_writers)
    pool_arr = np.concatenate(pool_imgs)
    pool_y = np.concatenate(pool_labels)

    # Validation slice: up to val_per_class held-out samples per class.
    val_mask = np.zeros(pool_y.shape[0], dtype=bool)
    for c in range(FEMNIST_NUM_CLASSES):
        class_idx = np.flatnonzero(pool_y == c)
        if class_idx.size == 0:
            continue
        k = min(val_per_class, class_idx.size)
        val_mask[rng.choice(class_idx, size=k, replace=False)] = True
    test_idx = np.flatnonzero(~val_mask)
    if test_idx.size > test_cap:
        test_idx = rng.choice(test_idx, size=test_cap, replace=False)

    # Standardise on training statistics; convention-agnostic.
    train_t = torch.from_numpy(train_arr).float().div_(255.0)
    mean = float(train_t.mean())
    std = max(float(train_t.std()), 1e-6)

    def _norm(arr: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(arr).float().div_(255.0).unsqueeze(1)
        return (t - mean) / std

    return FemnistBundle(
        train_x=_norm(train_arr),
        train_y=torch.from_numpy(train_y),
        train_writers=writers,
        val_x=_norm(pool_arr[val_mask]),
        val_y=torch.from_numpy(pool_y[val_mask]),
        test_x=_norm(pool_arr[test_idx]),
        test_y=torch.from_numpy(pool_y[test_idx]),
        norm_mean=mean,
        norm_std=std,
    )
