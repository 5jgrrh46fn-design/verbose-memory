"""Client partitioning for federated splits.

Three partitioners share one output shape, a list of index arrays with one
entry per client:

* :func:`dirichlet_partition` — for each class, draw a Dirichlet vector with
  concentration ``alpha`` over ``num_clients`` and split that class's indices
  proportionally. Lower ``alpha`` produces more skewed partitions (in the
  limit, single-class clients); higher ``alpha`` approaches uniform IID.
* :func:`iid_partition` — a seeded shuffle split into equal-sized shards.
* :func:`natural_partition` — one client per data-provider identity (for
  example, the writer of a handwritten character), which keeps whatever
  heterogeneity the data source has.

:func:`build_partition` dispatches on a ``kind`` string so a run can swap
the heterogeneity model of any dataset by config alone: pooling a naturally
partitioned dataset and re-splitting it by Dirichlet keeps the model, the
data, and the evaluation sets fixed and changes only how heterogeneity is
generated.

Empty shards are possible at small ``alpha`` and are returned as zero-length
arrays — callers should handle the empty case explicitly.
"""

from __future__ import annotations

import numpy as np

PARTITION_KINDS: tuple[str, ...] = ("dirichlet", "iid", "natural")


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
) -> list[np.ndarray]:
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1-D; got shape {labels.shape}")
    if num_clients <= 0:
        raise ValueError(f"num_clients must be positive; got {num_clients}")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive; got {alpha}")

    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    shards: list[list[int]] = [[] for _ in range(num_clients)]

    for c in classes:
        class_idx = np.flatnonzero(labels == c)
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        # Cumulative split points; rounded so every index is assigned exactly
        # once. ``cuts`` includes neither 0 nor the final length, matching
        # ``np.split`` semantics.
        cuts = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        for client_id, sub in enumerate(np.split(class_idx, cuts)):
            shards[client_id].extend(sub.tolist())

    return [np.asarray(sorted(s), dtype=np.int64) for s in shards]


def iid_partition(
    num_samples: int, num_clients: int, seed: int
) -> list[np.ndarray]:
    """Shuffle all indices and deal them out in equal-sized shards."""
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive; got {num_samples}")
    if num_clients <= 0:
        raise ValueError(f"num_clients must be positive; got {num_clients}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(num_samples)
    return [
        np.asarray(sorted(s.tolist()), dtype=np.int64)
        for s in np.array_split(order, num_clients)
    ]


def build_partition(
    kind: str,
    *,
    labels: np.ndarray,
    num_clients: int,
    seed: int,
    alpha: float | None = None,
    writer_ids: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Dispatch to a partitioner by name.

    ``dirichlet`` needs ``alpha``; ``natural`` needs ``writer_ids`` parallel
    to ``labels``; ``iid`` needs neither.
    """
    if kind == "dirichlet":
        if alpha is None:
            raise ValueError("dirichlet partition requires alpha")
        return dirichlet_partition(
            labels=labels, num_clients=num_clients, alpha=alpha, seed=seed
        )
    if kind == "iid":
        return iid_partition(
            num_samples=int(labels.shape[0]), num_clients=num_clients, seed=seed
        )
    if kind == "natural":
        if writer_ids is None:
            raise ValueError("natural partition requires writer_ids")
        return natural_partition(writer_ids, seed=seed, num_clients=num_clients)
    raise ValueError(
        f"partition kind must be one of {PARTITION_KINDS}; got {kind!r}"
    )


def natural_partition(
    writer_ids: np.ndarray,
    seed: int,
    num_clients: int | None = None,
) -> list[np.ndarray]:
    """Group sample indices by their writer id — one client per writer.

    ``writer_ids[i]`` is the writer behind sample ``i``. Returns a list whose
    ``j``-th entry holds the indices of the ``j``-th client's writer; client
    ordering is a seeded shuffle of the unique writers so it is reproducible
    but not tied to load order. ``num_clients``, if given, caps the number of
    writers used (it must not exceed the number present).
    """
    if writer_ids.ndim != 1:
        raise ValueError(f"writer_ids must be 1-D; got shape {writer_ids.shape}")
    unique = np.unique(writer_ids)
    rng = np.random.default_rng(seed)
    order = unique.copy()
    rng.shuffle(order)
    if num_clients is not None:
        if num_clients > order.size:
            raise ValueError(
                f"requested {num_clients} clients but only {order.size} writers"
            )
        order = order[:num_clients]
    return [np.flatnonzero(writer_ids == w).astype(np.int64) for w in order]
