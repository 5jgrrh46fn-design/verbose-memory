"""Deterministic selection of the malicious client subset.

The malicious set is fixed once for a run, derived from ``(seed, fraction)``
so the same clients are adversarial every round they are sampled.
"""

from __future__ import annotations

import hashlib
import random

from topfed.core.types import ClientId


def deterministic_malicious_set(
    num_clients: int, fraction: float, seed: int
) -> frozenset[ClientId]:
    """Return ``ceil(num_clients * fraction)`` malicious client ids.

    The selection is keyed by ``seed`` via a stable hash, so a given
    ``(num_clients, fraction, seed)`` triple always produces the same set
    regardless of Python hash randomization.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1]; got {fraction}")
    if num_clients <= 0:
        raise ValueError(f"num_clients must be positive; got {num_clients}")

    num_malicious = int(-(-num_clients * fraction // 1))  # ceil
    key = hashlib.blake2b(
        f"{seed}:{num_clients}:{fraction}".encode(), digest_size=16
    ).digest()
    rng = random.Random(int.from_bytes(key, "big"))
    all_ids = [f"c{i}" for i in range(num_clients)]
    rng.shuffle(all_ids)
    return frozenset(all_ids[:num_malicious])
