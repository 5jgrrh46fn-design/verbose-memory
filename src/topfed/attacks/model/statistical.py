"""Update manipulators built on benign-update statistics.

The attacks here craft the submitted update from statistics of benign
updates (a per-coordinate mean and standard deviation, or the pairwise
distances between them), so that it sits where a robust aggregator (a
median or a trimmed mean) still accepts it. Two are untargeted and drag the
aggregate away from the benign direction:

* :class:`LittleIsEnoughManipulator`: "A Little Is Enough" (Baruch, Baruch
  and Goldberg, NeurIPS 2019). The update is the benign mean shifted by
  ``num_std`` standard deviations in every coordinate.
* :class:`FangTrimmedMeanManipulator`: the partial-knowledge attack on the
  median and the trimmed mean of Fang, Cao, Jia and Gong ("Local Model
  Poisoning Attacks to Byzantine-Robust Federated Learning", USENIX Security
  2020, Section 4.2). Every coordinate is drawn three to four standard
  deviations away from the benign mean, against the benign direction.

Two carry a backdoor inside the benign range instead:

* :class:`InRangeBackdoorManipulator`: the backdoor variant of "A Little Is
  Enough". The poisoned update is clipped, coordinate by coordinate, into
  ``num_std`` standard deviations around the benign mean.
* :class:`MinMaxBackdoorManipulator`: the Min-Max attack of Shejwalkar and
  Houmansadr ("Manipulating the Byzantine", NDSS 2021) with the backdoor as
  its perturbation direction. The update moves from the benign mean along
  the backdoor direction as far as it can while staying no farther from any
  benign update than the benign updates are from each other.

The papers estimate the statistics from the updates of the clients the
adversary controls. Here an attacker controls one client and never sees
another client's update, so it estimates them from its own clean training:
the client trains one clean update on each of ``reference_shards`` disjoint
shards of its local data from the round's global model (the "virtual
clients" it can simulate) and hands their deltas to
:meth:`set_reference_deltas`. With fewer than two shard updates there is no
spread to estimate and the update is passed through unchanged.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence

import numpy as np

from scipy import stats

from topfed.core.types import ClientUpdate, NDArrays, RoundContext


def _is_float(a: np.ndarray) -> bool:
    return bool(np.issubdtype(np.asarray(a).dtype, np.floating))


class _ShardStatistics:
    """Per-coordinate mean and spread of the attacker's own clean shard updates."""

    def __init__(self, reference_shards: int, ddof: int) -> None:
        if reference_shards < 2:
            raise ValueError(
                f"reference_shards must be at least 2; got {reference_shards}"
            )
        self.reference_shards = int(reference_shards)
        self._ddof = ddof
        self._reference_deltas: list[NDArrays] = []

    def set_reference_deltas(self, reference_deltas: Sequence[NDArrays]) -> None:
        """Record this round's clean shard deltas (``local - global`` each)."""
        self._reference_deltas = [
            [np.asarray(a, dtype=np.float64) for a in delta] for delta in reference_deltas
        ]

    def _statistics(self) -> tuple[NDArrays, NDArrays] | None:
        """``(mean, std)`` per tensor, or ``None`` without at least two deltas."""
        deltas = self._reference_deltas
        if len(deltas) < 2:
            return None
        stacked = [np.stack(layer) for layer in zip(*deltas, strict=True)]
        mean = [s.mean(axis=0) for s in stacked]
        std = [s.std(axis=0, ddof=self._ddof) for s in stacked]
        return mean, std

    @staticmethod
    def _replace(
        update: ClientUpdate, global_parameters: NDArrays, crafted: NDArrays
    ) -> ClientUpdate:
        """The update with its floating-point tensors set to ``global + crafted``."""
        parameters = [
            (np.asarray(g) + c).astype(np.asarray(w).dtype) if _is_float(w) else np.asarray(w)
            for w, g, c in zip(update.parameters, global_parameters, crafted, strict=True)
        ]
        return ClientUpdate(
            client_id=update.client_id,
            parameters=parameters,
            num_examples=update.num_examples,
            train_loss=update.train_loss,
            train_accuracy=update.train_accuracy,
        )


class LittleIsEnoughManipulator(_ShardStatistics):
    """Submit ``mean - num_std * std`` of the estimated benign updates.

    Follows ``DriftAttack`` in the authors' code
    (https://github.com/moranant/attacking_distributed_learning, commit
    7c4f5eba8e4f6b26c03dac5104c4d4c805b1897a, ``malicious.py``; the
    repository carries no licence file, and nothing is copied from it): the
    mean and the population standard deviation of the controlled clients'
    benign updates, and the same shifted vector for every coordinate sign.

    ``num_std`` is the attack's ``z``. The paper derives the largest ``z``
    that keeps the crafted values inside the majority from the cohort size
    ``n`` and the number of attackers ``m``; with ten clients a round and
    one or two attackers that bound is not positive, so the default is the
    reference code's own default, 1.5.
    """

    def __init__(self, num_std: float = 1.5, reference_shards: int = 4) -> None:
        super().__init__(reference_shards, ddof=0)
        if num_std <= 0:
            raise ValueError(f"num_std must be positive; got {num_std}")
        self.num_std = float(num_std)

    def manipulate(
        self,
        update: ClientUpdate,
        global_parameters: NDArrays,
        ctx: RoundContext,  # noqa: ARG002
    ) -> ClientUpdate:
        stats = self._statistics()
        if stats is None:
            return update
        mean, std = stats
        crafted = [m - self.num_std * s for m, s in zip(mean, std, strict=True)]
        return self._replace(update, global_parameters, crafted)


class FangTrimmedMeanManipulator(_ShardStatistics):
    """Draw every coordinate ``low`` to ``high`` deviations against the benign direction.

    With ``mu`` and ``sigma`` the estimated benign mean and sample standard
    deviation of a coordinate's update, and the sign of ``mu`` the estimated
    direction the global parameter moves without an attack, the submitted
    value is uniform in ``[mu - high * sigma, mu - low * sigma]`` where that
    direction is positive and in ``[mu + low * sigma, mu + high * sigma]``
    where it is negative or zero (``low = 3``, ``high = 4`` in the paper).

    The structure follows ``fang_attack_trmean_partial`` in
    https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning (commit
    f2879cd5eade97f02ebbaa795c63067a53de7979,
    ``cifar10/release-fedsgd-alexnet-trmean-unknown-benign-gradients.ipynb``;
    the repository carries no licence file, and nothing is copied from it).
    That function selects both ranges with the same ``deviation > 0`` mask;
    this class assigns the ranges by direction as the paper states them.

    The draw is seeded by ``seed``, the round's own seed and number and the
    client id, so a run is reproducible and two attackers of a round draw
    independently.
    """

    def __init__(
        self,
        reference_shards: int = 4,
        low: float = 3.0,
        high: float = 4.0,
        seed: int = 0,
    ) -> None:
        super().__init__(reference_shards, ddof=1)
        if not 0.0 <= low <= high:
            raise ValueError(f"need 0 <= low <= high; got low={low}, high={high}")
        self.low = float(low)
        self.high = float(high)
        self.seed = int(seed)

    def manipulate(
        self,
        update: ClientUpdate,
        global_parameters: NDArrays,
        ctx: RoundContext,
    ) -> ClientUpdate:
        stats = self._statistics()
        if stats is None:
            return update
        mean, std = stats
        rng = np.random.default_rng(
            [
                self.seed,
                int(ctx.seed) % 2**32,
                int(ctx.round_num),
                zlib.crc32(str(update.client_id).encode()),
            ]
        )
        crafted = []
        for m, s in zip(mean, std, strict=True):
            distance = (self.low + (self.high - self.low) * rng.uniform(size=m.shape)) * s
            crafted.append(np.where(m > 0, m - distance, m + distance))
        return self._replace(update, global_parameters, crafted)


def alie_z_max(num_clients: int, num_attackers: int) -> float:
    """The largest shift, in standard deviations, "A Little Is Enough" derives.

    With ``n`` clients and ``m`` attackers the attackers need ``s = floor(n / 2
    + 1) - m`` benign clients on their side for a majority, and ``z_max`` is
    the standard-normal quantile of ``(n - m - s) / (n - m)``: a benign value
    lies farther out than ``z_max`` deviations with enough probability to
    supply those ``s`` clients. It is not positive when the attackers are too
    few for the count at hand (ten clients and one or two attackers, say).
    """
    n, m = int(num_clients), int(num_attackers)
    if not 0 < m < n:
        raise ValueError(f"need 0 < num_attackers < num_clients; got {m}, {n}")
    s = n // 2 + 1 - m
    return float(stats.norm.ppf((n - m - s) / (n - m)))


class InRangeBackdoorManipulator(_ShardStatistics):
    """Clip a poisoned update into ``mean +- num_std * std`` per coordinate.

    The backdoor attack of "A Little Is Enough": the client trains its
    backdoored model (its poisoner and loss term stay in charge of that),
    and every coordinate of the finished update is clipped into ``num_std``
    population standard deviations around the estimated benign mean, so the
    update never leaves the range a benign update is expected in. Follows
    ``BackdoorAttack._attack_grads`` in the authors' code
    (https://github.com/moranant/attacking_distributed_learning, commit
    7c4f5eba8e4f6b26c03dac5104c4d4c805b1897a, ``backdoor.py``; no licence
    file, nothing copied). ``num_std`` is the attack's ``z``;
    :func:`alie_z_max` gives the paper's bound.
    """

    def __init__(self, num_std: float = 1.0, reference_shards: int = 4) -> None:
        super().__init__(reference_shards, ddof=0)
        if num_std < 0:
            raise ValueError(f"num_std must not be negative; got {num_std}")
        self.num_std = float(num_std)

    def manipulate(
        self,
        update: ClientUpdate,
        global_parameters: NDArrays,
        ctx: RoundContext,  # noqa: ARG002
    ) -> ClientUpdate:
        statistics = self._statistics()
        if statistics is None:
            return update
        mean, std = statistics
        crafted = [
            np.clip(
                np.asarray(w, dtype=np.float64) - np.asarray(g, dtype=np.float64),
                m - self.num_std * sd,
                m + self.num_std * sd,
            )
            if _is_float(w)
            else m
            for w, g, m, sd in zip(update.parameters, global_parameters, mean, std, strict=True)
        ]
        return self._replace(update, global_parameters, crafted)


class MinMaxBackdoorManipulator(_ShardStatistics):
    """Move from the benign mean along the backdoor direction, Min-Max bounded.

    With ``mu`` the mean of the estimated benign updates ``b_i`` and ``p``
    the unit vector from ``mu`` to the client's poisoned update, the
    submitted update is ``mu + gamma * p`` for the largest ``gamma`` such
    that ``max_i ||mu + gamma * p - b_i|| <= max_{i,j} ||b_i - b_j||``: it is
    no farther from any benign update than two benign updates are from each
    other. The bound has a closed form (the smallest positive root over
    ``i`` of a quadratic in ``gamma``), which is what this class evaluates.

    Follows ``our_attack_dist`` in
    https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning (commit
    f2879cd5eade97f02ebbaa795c63067a53de7979,
    ``cifar10/release-fedsgd-alexnet-trmean-unknown-benign-gradients.ipynb``;
    no licence file, nothing copied), which reaches the same ``gamma`` by a
    halving search and perturbs along the inverse benign direction; here the
    perturbation is the backdoor.
    """

    def __init__(self, reference_shards: int = 4, eps: float = 1e-12) -> None:
        super().__init__(reference_shards, ddof=0)
        self.eps = float(eps)

    def manipulate(
        self,
        update: ClientUpdate,
        global_parameters: NDArrays,
        ctx: RoundContext,  # noqa: ARG002
    ) -> ClientUpdate:
        statistics = self._statistics()
        if statistics is None:
            return update
        mean, _ = statistics
        floats = [_is_float(w) for w in update.parameters]

        def flat(arrays: Sequence[np.ndarray]) -> np.ndarray:
            return np.concatenate(
                [np.asarray(a, dtype=np.float64).ravel() for a, f in zip(arrays, floats, strict=True) if f]
            )

        mu = flat(mean)
        poisoned = flat(
            [
                np.asarray(w, dtype=np.float64) - np.asarray(g, dtype=np.float64)
                for w, g in zip(update.parameters, global_parameters, strict=True)
            ]
        )
        direction = poisoned - mu
        norm = float(np.linalg.norm(direction))
        if norm <= self.eps:
            return update
        p = direction / norm
        benign = np.stack([flat(d) for d in self._reference_deltas])
        gaps = benign[:, None, :] - benign[None, :, :]
        max_sq = float(np.max(np.sum(gaps * gaps, axis=2)))
        # ||(mu - b_i) + gamma p||^2 <= max_sq  <=>  gamma <= -a.p + sqrt((a.p)^2 - |a|^2 + max_sq)
        a = mu[None, :] - benign
        ap = a @ p
        gamma = float(np.min(-ap + np.sqrt(np.maximum(ap * ap - np.sum(a * a, axis=1) + max_sq, 0.0))))
        crafted_flat = mu + max(gamma, 0.0) * p
        crafted: NDArrays = []
        offset = 0
        for m, f in zip(mean, floats, strict=True):
            if f:
                crafted.append(crafted_flat[offset : offset + m.size].reshape(m.shape))
                offset += m.size
            else:
                crafted.append(m)
        return self._replace(update, global_parameters, crafted)
