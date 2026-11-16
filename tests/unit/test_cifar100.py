"""CIFAR-100 loader: shapes, class counts, the superclass mapping, the split.

A stand-in for torchvision's dataset class supplies deterministic uint8 images
with a known number of samples per fine class, so the loader's tensor
conversion, normalisation and per-class validation split are checked without
the download. The superclass table is checked structurally against the fine
class list and, when the real archive is present under the repository's
``data/`` directory, against the ``coarse_labels`` the dataset ships.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from topfed.data import cifar100
from topfed.data.cifar100 import (
    CIFAR100_FINE_CLASSES,
    CIFAR100_NORMALIZE_MEAN,
    CIFAR100_NORMALIZE_STD,
    CIFAR100_NUM_CLASSES,
    CIFAR100_NUM_COARSE_CLASSES,
    CIFAR100_SUPERCLASSES,
    FINE_TO_COARSE,
    fine_to_coarse,
    load_cifar100,
)

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = REPO / "data" / "cifar-100-python"
TRAIN_PER_CLASS = 2
TEST_PER_CLASS = 3


class _FakeCifar100:
    """``datasets.CIFAR100`` stand-in: ``k`` images per fine class, labels interleaved."""

    def __init__(self, root: str, train: bool, download: bool, transform=None) -> None:  # noqa: ANN001
        k = TRAIN_PER_CLASS if train else TEST_PER_CLASS
        self.targets = [c for _ in range(k) for c in range(CIFAR100_NUM_CLASSES)]
        n = len(self.targets)
        # Pixel value = fine label, so an image's label can be read back from
        # its (normalised) pixels after the split.
        self.data = np.repeat(
            np.asarray(self.targets, dtype=np.uint8)[:, None, None, None], 32, axis=1
        ).repeat(32, axis=2).repeat(3, axis=3)
        assert self.data.shape == (n, 32, 32, 3)


@pytest.fixture
def fake_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cifar100.datasets, "CIFAR100", _FakeCifar100)


def _labels_from_pixels(x: torch.Tensor) -> torch.Tensor:
    """Undo the normalisation of channel 0 to recover the fine label."""
    raw = x[:, 0, 0, 0] * CIFAR100_NORMALIZE_STD[0] + CIFAR100_NORMALIZE_MEAN[0]
    return (raw * 255.0).round().long()


# ----------------------------------------------------------------------------
# Superclass mapping
# ----------------------------------------------------------------------------


def test_fine_classes_are_alphabetical_and_complete() -> None:
    assert len(CIFAR100_FINE_CLASSES) == CIFAR100_NUM_CLASSES
    assert len(set(CIFAR100_FINE_CLASSES)) == CIFAR100_NUM_CLASSES
    assert list(CIFAR100_FINE_CLASSES) == sorted(CIFAR100_FINE_CLASSES)


def test_superclasses_partition_the_fine_classes() -> None:
    assert len(CIFAR100_SUPERCLASSES) == CIFAR100_NUM_COARSE_CLASSES
    names = [name for name, _ in CIFAR100_SUPERCLASSES]
    assert names == sorted(names)
    members = [m for _, group in CIFAR100_SUPERCLASSES for m in group]
    assert all(len(group) == 5 for _, group in CIFAR100_SUPERCLASSES)
    assert sorted(members) == list(CIFAR100_FINE_CLASSES)


def test_fine_to_coarse_table_matches_the_named_groups() -> None:
    assert len(FINE_TO_COARSE) == CIFAR100_NUM_CLASSES
    assert sorted(set(FINE_TO_COARSE)) == list(range(CIFAR100_NUM_COARSE_CLASSES))
    assert all(FINE_TO_COARSE.count(c) == 5 for c in range(CIFAR100_NUM_COARSE_CLASSES))
    for coarse, (_, group) in enumerate(CIFAR100_SUPERCLASSES):
        for name in group:
            assert FINE_TO_COARSE[CIFAR100_FINE_CLASSES.index(name)] == coarse
    # Spot checks against the published table.
    fine = CIFAR100_FINE_CLASSES.index
    assert FINE_TO_COARSE[fine("whale")] == 0  # aquatic mammals
    assert FINE_TO_COARSE[fine("woman")] == 14  # people
    assert FINE_TO_COARSE[fine("tractor")] == 19  # vehicles 2
    assert FINE_TO_COARSE[fine("apple")] == 4  # fruit and vegetables


def test_fine_to_coarse_maps_tensors_and_rejects_bad_labels() -> None:
    labels = torch.arange(CIFAR100_NUM_CLASSES)
    coarse = fine_to_coarse(labels)
    assert coarse.dtype == torch.long and coarse.shape == labels.shape
    assert coarse.tolist() == list(FINE_TO_COARSE)
    assert fine_to_coarse(torch.tensor([], dtype=torch.long)).numel() == 0
    with pytest.raises(ValueError):
        fine_to_coarse(torch.tensor([100]))
    with pytest.raises(ValueError):
        fine_to_coarse(torch.tensor([-1]))


@pytest.mark.skipif(not (ARCHIVE / "test").exists(), reason="CIFAR-100 archive not downloaded")
def test_fine_to_coarse_matches_the_dataset_archive() -> None:
    with (ARCHIVE / "meta").open("rb") as fh:
        meta = pickle.load(fh, encoding="latin1")
    assert tuple(meta["fine_label_names"]) == CIFAR100_FINE_CLASSES
    assert tuple(meta["coarse_label_names"]) == tuple(n for n, _ in CIFAR100_SUPERCLASSES)
    with (ARCHIVE / "test").open("rb") as fh:
        batch = pickle.load(fh, encoding="latin1")
    fine = np.asarray(batch["fine_labels"])
    coarse = np.asarray(batch["coarse_labels"])
    assert set(fine.tolist()) == set(range(CIFAR100_NUM_CLASSES))
    np.testing.assert_array_equal(np.asarray(FINE_TO_COARSE)[fine], coarse)


# ----------------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------------


def test_fine_loader_shapes_normalisation_and_split(fake_dataset, tmp_path: Path) -> None:
    bundle = load_cifar100(tmp_path, val_per_class=1, seed=0)
    n_train = TRAIN_PER_CLASS * CIFAR100_NUM_CLASSES
    n_test = TEST_PER_CLASS * CIFAR100_NUM_CLASSES
    assert bundle.train_x.shape == (n_train, 3, 32, 32)
    assert bundle.train_y.shape == (n_train,) and bundle.train_y.dtype == torch.long
    assert bundle.val_x.shape == (CIFAR100_NUM_CLASSES, 3, 32, 32)
    assert bundle.test_x.shape == (n_test - CIFAR100_NUM_CLASSES, 3, 32, 32)
    # Every class appears exactly val_per_class times in the validation slice
    # and the rest of the test set is the complement.
    assert torch.bincount(bundle.val_y, minlength=100).tolist() == [1] * 100
    assert torch.bincount(bundle.test_y, minlength=100).tolist() == [TEST_PER_CLASS - 1] * 100
    assert bundle.train_y.min() == 0 and bundle.train_y.max() == CIFAR100_NUM_CLASSES - 1
    # Pixels are normalised with CIFAR-100's own statistics, per channel.
    zero = bundle.train_x[bundle.train_y == 0][0]
    for ch in range(3):
        expected = (0.0 - CIFAR100_NORMALIZE_MEAN[ch]) / CIFAR100_NORMALIZE_STD[ch]
        assert zero[ch].min().item() == pytest.approx(expected, abs=1e-6)
        assert zero[ch].max().item() == pytest.approx(expected, abs=1e-6)
    # Labels stayed attached to their images through the split.
    assert torch.equal(_labels_from_pixels(bundle.val_x), bundle.val_y)
    assert torch.equal(_labels_from_pixels(bundle.test_x), bundle.test_y)
    assert torch.equal(_labels_from_pixels(bundle.train_x), bundle.train_y)


def test_coarse_loader_relabels_and_splits_per_superclass(fake_dataset, tmp_path: Path) -> None:
    bundle = load_cifar100(tmp_path, val_per_class=5, coarse=True, seed=0)
    n_train = TRAIN_PER_CLASS * CIFAR100_NUM_CLASSES
    assert bundle.train_x.shape == (n_train, 3, 32, 32)
    assert bundle.train_y.min() == 0 and bundle.train_y.max() == CIFAR100_NUM_COARSE_CLASSES - 1
    counts = torch.bincount(bundle.train_y, minlength=CIFAR100_NUM_COARSE_CLASSES)
    assert counts.tolist() == [5 * TRAIN_PER_CLASS] * CIFAR100_NUM_COARSE_CLASSES
    assert bundle.val_x.shape[0] == 5 * CIFAR100_NUM_COARSE_CLASSES
    assert torch.bincount(bundle.val_y, minlength=20).tolist() == [5] * 20
    assert torch.bincount(bundle.test_y, minlength=20).tolist() == [5 * TEST_PER_CLASS - 5] * 20
    # Every coarse label is the superclass of the image's fine label.
    for x, y in ((bundle.train_x, bundle.train_y), (bundle.val_x, bundle.val_y),
                 (bundle.test_x, bundle.test_y)):
        assert torch.equal(fine_to_coarse(_labels_from_pixels(x)), y)


def test_split_is_seeded_and_rejects_oversized_slices(fake_dataset, tmp_path: Path) -> None:
    a = load_cifar100(tmp_path, val_per_class=1, seed=3)
    b = load_cifar100(tmp_path, val_per_class=1, seed=3)
    c = load_cifar100(tmp_path, val_per_class=1, seed=4)
    assert torch.equal(a.val_x, b.val_x) and torch.equal(a.test_y, b.test_y)
    assert not torch.equal(a.val_x, c.val_x)
    with pytest.raises(ValueError, match="cannot hold out"):
        load_cifar100(tmp_path, val_per_class=TEST_PER_CLASS + 1)
