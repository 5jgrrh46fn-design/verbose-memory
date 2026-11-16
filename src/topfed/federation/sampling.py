"""Per-round client sampling.

Two samplers share one contract: given the round's RNG they return the
ordered tuple of client ids that participate this round.

* :func:`uniform_sample` draws ``k`` clients uniformly without replacement.
* :func:`forced_co_selection_sample` fixes the number of malicious clients in
  the cohort to exactly ``m`` and fills the remaining ``k - m`` slots uniformly
  from the benign population. This removes the sampling variance in how many
  attackers meet in a round, so an experiment can dial the co-selected
  attacker count directly instead of waiting for the hypergeometric tail to
  produce it. It is a diagnostic instrument, not a threat model: it does not
  change who is malicious, only who is sampled.

Both are deterministic in the RNG passed in, so a run seed reproduces the same
cohort sequence.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

from topfed.core.types import ClientId


def uniform_sample(
    rng: random.Random, client_ids: Iterable[ClientId], k: int
) -> tuple[ClientId, ...]:
    """Draw ``k`` distinct clients uniformly at random."""
    return tuple(rng.sample(list(client_ids), k))


def forced_co_selection_sample(
    rng: random.Random,
    client_ids: Iterable[ClientId],
    malicious_clients: frozenset[ClientId],
    k: int,
    m: int,
) -> tuple[ClientId, ...]:
    """Draw ``k`` clients with exactly ``m`` of them malicious.

    The malicious members are drawn uniformly from ``malicious_clients`` and
    the benign members uniformly from the rest; the cohort order is then
    shuffled so position carries no information about a client's role.
    """
    ids = list(client_ids)
    malicious = sorted(c for c in ids if c in malicious_clients)
    benign = sorted(c for c in ids if c not in malicious_clients)
    if m < 0 or m > k:
        raise ValueError(f"forced malicious count {m} must lie in [0, {k}]")
    if m > len(malicious):
        raise ValueError(
            f"cannot force {m} malicious clients per round: only "
            f"{len(malicious)} exist"
        )
    if k - m > len(benign):
        raise ValueError(
            f"cannot fill {k - m} benign slots per round: only "
            f"{len(benign)} benign clients exist"
        )
    cohort = rng.sample(malicious, m) + rng.sample(benign, k - m)
    rng.shuffle(cohort)
    return tuple(cohort)
