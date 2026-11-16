"""CIFAR-100 loading utilities.

The pipeline mirrors CIFAR-10's: the training set is pre-normalised once with
CIFAR-100's own channel statistics, the test set is split into a server-side
validation slice (``val_per_class`` samples per class) and the remaining
clean test pool, and the result is a :class:`~topfed.data.cifar10.CifarBundle`
so every consumer of the CIFAR-10 tensors works unchanged. The ASR loader is
CIFAR-10's :func:`~topfed.data.cifar10.build_asr_loader`; the trigger lives in
normalised pixel space, so it carries over verbatim.

``coarse=True`` replaces the 100 fine labels by the dataset's 20 superclasses.
The superclass membership is written out explicitly below (it is the table
published with the dataset) and keyed by class name, so it does not depend on
the label numbering of any particular loader; the integer mapping is derived
from torchvision's class order, which is alphabetical by fine-class name, and
the superclass index is alphabetical by superclass name, the numbering the
dataset's own ``coarse_labels`` use.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision import datasets

from topfed.data.cifar10 import CifarBundle

# Channel statistics of the CIFAR-100 training set in [0, 1] pixel units.
CIFAR100_NORMALIZE_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_NORMALIZE_STD = (0.2673, 0.2564, 0.2762)
CIFAR100_NUM_CLASSES = 100
CIFAR100_NUM_COARSE_CLASSES = 20

# Fine classes in torchvision's label order (alphabetical).
CIFAR100_FINE_CLASSES: tuple[str, ...] = (
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
    "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
)

# The 20 superclasses in the dataset's coarse-label order (alphabetical by
# superclass name), each with its five fine classes.
CIFAR100_SUPERCLASSES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("aquatic_mammals", ("beaver", "dolphin", "otter", "seal", "whale")),
    ("fish", ("aquarium_fish", "flatfish", "ray", "shark", "trout")),
    ("flowers", ("orchid", "poppy", "rose", "sunflower", "tulip")),
    ("food_containers", ("bottle", "bowl", "can", "cup", "plate")),
    ("fruit_and_vegetables", ("apple", "mushroom", "orange", "pear", "sweet_pepper")),
    ("household_electrical_devices", ("clock", "keyboard", "lamp", "telephone", "television")),
    ("household_furniture", ("bed", "chair", "couch", "table", "wardrobe")),
    ("insects", ("bee", "beetle", "butterfly", "caterpillar", "cockroach")),
    ("large_carnivores", ("bear", "leopard", "lion", "tiger", "wolf")),
    ("large_man-made_outdoor_things", ("bridge", "castle", "house", "road", "skyscraper")),
    ("large_natural_outdoor_scenes", ("cloud", "forest", "mountain", "plain", "sea")),
    ("large_omnivores_and_herbivores", ("camel", "cattle", "chimpanzee", "elephant", "kangaroo")),
    ("medium_mammals", ("fox", "porcupine", "possum", "raccoon", "skunk")),
    ("non-insect_invertebrates", ("crab", "lobster", "snail", "spider", "worm")),
    ("people", ("baby", "boy", "girl", "man", "woman")),
    ("reptiles", ("crocodile", "dinosaur", "lizard", "snake", "turtle")),
    ("small_mammals", ("hamster", "mouse", "rabbit", "shrew", "squirrel")),
    ("trees", ("maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree")),
    ("vehicles_1", ("bicycle", "bus", "motorcycle", "pickup_truck", "train")),
    ("vehicles_2", ("lawn_mower", "rocket", "streetcar", "tank", "tractor")),
)


def _fine_to_coarse_table() -> tuple[int, ...]:
    index = {name: i for i, name in enumerate(CIFAR100_FINE_CLASSES)}
    table = [-1] * len(CIFAR100_FINE_CLASSES)
    for coarse, (_, members) in enumerate(CIFAR100_SUPERCLASSES):
        for name in members:
            table[index[name]] = coarse
    if any(c < 0 for c in table):
        raise RuntimeError("superclass table does not cover every fine class")
    return tuple(table)


# ``FINE_TO_COARSE[fine_label]`` is the superclass of that fine label.
FINE_TO_COARSE: tuple[int, ...] = _fine_to_coarse_table()


def fine_to_coarse(labels: torch.Tensor) -> torch.Tensor:
    """Map a tensor of fine labels (0..99) to superclass labels (0..19)."""
    if labels.numel() and (labels.min() < 0 or labels.max() >= CIFAR100_NUM_CLASSES):
        raise ValueError("fine labels must lie in [0, 100)")
    table = torch.tensor(FINE_TO_COARSE, dtype=torch.long)
    return table[labels.long()]


def _to_tensor_dataset(
    ds: datasets.CIFAR100, *, coarse: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    # ``ds.data`` is uint8 (N, 32, 32, 3); convert to float (N, 3, 32, 32)
    # and normalise once up front, as the CIFAR-10 loader does.
    x = torch.from_numpy(ds.data).float().div_(255.0).permute(0, 3, 1, 2).contiguous()
    mean = torch.tensor(CIFAR100_NORMALIZE_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR100_NORMALIZE_STD).view(1, 3, 1, 1)
    x = (x - mean) / std
    y = torch.tensor(ds.targets, dtype=torch.long)
    if coarse:
        y = fine_to_coarse(y)
    return x, y


def load_cifar100(
    root: str | Path,
    *,
    val_per_class: int = 10,
    coarse: bool = False,
    seed: int = 0,
    download: bool = True,
) -> CifarBundle:
    """Load CIFAR-100 and carve a server-side validation slice from the test set.

    Labels are the 100 fine classes, or the 20 superclasses when ``coarse``
    is set; the validation slice then holds ``val_per_class`` samples of
    each label in use, chosen by a seeded shuffle of that label's test
    indices. Remaining test samples are exposed as ``test_*`` for ASR /
    clean-test evaluation.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    train_ds = datasets.CIFAR100(
        root=str(root), train=True, download=download, transform=None
    )
    test_ds = datasets.CIFAR100(
        root=str(root), train=False, download=download, transform=None
    )

    train_x, train_y = _to_tensor_dataset(train_ds, coarse=coarse)
    test_x_full, test_y_full = _to_tensor_dataset(test_ds, coarse=coarse)

    num_classes = CIFAR100_NUM_COARSE_CLASSES if coarse else CIFAR100_NUM_CLASSES
    rng = np.random.default_rng(seed)
    val_mask = np.zeros(len(test_y_full), dtype=bool)
    for c in range(num_classes):
        class_idx = np.flatnonzero(test_y_full.numpy() == c)
        if len(class_idx) < val_per_class:
            raise ValueError(
                f"class {c} has {len(class_idx)} test samples; "
                f"cannot hold out {val_per_class} per class"
            )
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
