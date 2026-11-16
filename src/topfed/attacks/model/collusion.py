"""Colluding attackers: a round-level coordinator and four update strategies.

Independent attackers each decide their own update. Colluders instead share
information within a round: every co-selected malicious client knows who else
was selected (white-box), and they agree on what each of them submits. The
:class:`CollusionCoordinator` is that shared channel. Clients train one after
another in cohort order, so the first colluder in the cohort acts as the
round's *leader*: it trains its poisoned update as usual, posts the resulting
payload, and every later colluder queries it.

Each strategy is an update manipulator that wraps the attack's own base
manipulator (round-aware model replacement, or a norm bound) and then applies
the coordination rule:

* :class:`CloneManipulator` (S1) -- every colluder submits the leader's
  base-attack update. Identical updates are the Sybil upper bound on
  cross-client correlation and put the whole bloc's weight on one value.
* :class:`BudgetSplitManipulator` (S2) -- every colluder submits the update a
  lone attacker would have submitted, divided by the bloc size ``m``, so the
  bloc's total magnitude equals one lone attacker's while each member is ``m``
  times smaller. The round-aware replacement factor already divides the lone
  budget by ``m``, so for that base attack S2 and S1 submit the same update;
  the two differ only under a fixed-norm base.
* :class:`ReferenceShiftManipulator` (S3) -- the leader is the *carrier* and
  submits a lone attacker's update; the other colluders are *decoys* that
  submit the carrier's direction at the norm of their own clean update. The
  decoys exist to pull the cohort mean toward the carrier and away from the
  honest clients, which is what peer-relative trust metrics are scored
  against.
* :class:`CoordinateCaptureManipulator` (S4) -- colluders agree on identical
  values on the top ``top_fraction`` of coordinates of the backdoor direction
  (the leader's poisoned delta minus its clean delta) and submit their own
  clean update everywhere else, so the bloc wins the per-coordinate median
  exactly where the backdoor lives while looking benign on the rest.

With a single colluder in the round every strategy submits exactly the base
attack's update and requests no clean reference pass, so ``m = 1`` is the
independent-attacker case for all four.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from topfed.attacks.protocols import UpdateManipulator
from topfed.core.types import ClientId, ClientUpdate, NDArrays, RoundContext


def _deltas(parameters: NDArrays, global_parameters: NDArrays) -> NDArrays:
    return [
        np.asarray(p) - np.asarray(g)
        for p, g in zip(parameters, global_parameters, strict=True)
    ]


def _norm(arrays: NDArrays) -> float:
    return float(np.sqrt(sum(float(np.sum(np.square(a))) for a in arrays)))


def _submit(
    update: ClientUpdate, global_parameters: NDArrays, delta: NDArrays
) -> ClientUpdate:
    """``update`` with its parameters replaced by ``global + delta``."""
    return ClientUpdate(
        client_id=update.client_id,
        parameters=[
            np.asarray(g + d) for g, d in zip(global_parameters, delta, strict=True)
        ],
        num_examples=update.num_examples,
        train_loss=update.train_loss,
        train_accuracy=update.train_accuracy,
    )


@dataclass(frozen=True, slots=True)
class CollusionPayload:
    """What the round's leader shares with the other colluders.

    ``submitted_delta`` is the leader's update after the base attack in the
    actual cohort; ``lone_delta`` is what the base attack would have produced
    had the leader been the only attacker selected; ``poisoned_delta`` is the
    leader's raw poisoned training delta before the base attack; and
    ``clean_delta`` is the leader's clean update when the strategy asked for a
    clean reference pass (``None`` otherwise). ``shared`` carries
    strategy-specific agreements (e.g. the captured coordinates).
    """

    round_num: int
    leader: ClientId
    submitted_delta: NDArrays
    lone_delta: NDArrays
    poisoned_delta: NDArrays
    clean_delta: NDArrays | None
    shared: dict[str, Any]


class CollusionCoordinator:
    """Round-level shared channel between co-selected malicious clients.

    White-box: the coordinator knows the malicious set and reads the round's
    cohort from the :class:`RoundContext`, so every colluder knows how many
    others were selected, in what order, and which of them leads. The leader
    (first colluder in cohort order) posts one :class:`CollusionPayload` per
    round; the others read it. State is keyed by round and replaced when a new
    round starts.

    ``records`` holds one row per round in which at least one colluder was
    selected (round, leader, colluders, strategy), for offline analysis.
    """

    def __init__(self, malicious_clients: frozenset[ClientId]) -> None:
        self.malicious_clients = frozenset(malicious_clients)
        self._payload: CollusionPayload | None = None
        self.records: list[dict[str, Any]] = []

    def colluders(self, ctx: RoundContext) -> tuple[ClientId, ...]:
        """The round's malicious clients, in cohort (training) order."""
        return tuple(c for c in ctx.selected_clients if c in self.malicious_clients)

    def leader(self, ctx: RoundContext) -> ClientId:
        bloc = self.colluders(ctx)
        if not bloc:
            raise ValueError(f"round {ctx.round_num} has no colluders")
        return bloc[0]

    def is_leader(self, client_id: ClientId, ctx: RoundContext) -> bool:
        return self.leader(ctx) == client_id

    def lone_context(self, client_id: ClientId, ctx: RoundContext) -> RoundContext:
        """``ctx`` as it would look had ``client_id`` been the only attacker selected."""
        return replace(
            ctx,
            selected_clients=tuple(
                c
                for c in ctx.selected_clients
                if c == client_id or c not in self.malicious_clients
            ),
        )

    def post(self, payload: CollusionPayload, ctx: RoundContext, strategy: str) -> None:
        if payload.leader != self.leader(ctx):
            raise ValueError(
                f"{payload.leader} is not the leader of round {ctx.round_num}"
            )
        self._payload = payload
        self.records.append(
            {
                "round": ctx.round_num,
                "leader": payload.leader,
                "colluders": list(self.colluders(ctx)),
                "strategy": strategy,
            }
        )

    def payload(self, ctx: RoundContext) -> CollusionPayload:
        """The leader's payload for this round; raises if it has not posted yet."""
        found = self._payload
        if found is None or found.round_num != ctx.round_num:
            raise RuntimeError(
                f"no collusion payload for round {ctx.round_num}: the leader "
                f"{self.leader(ctx)} has not submitted yet"
            )
        return found


class _CollusionManipulator:
    """Shared machinery: base attack, leader payload, optional clean reference.

    Subclasses set ``strategy`` and implement :meth:`_submission`. A client
    calls :meth:`wants_reference_delta` before training; when it returns true
    the client retrains once on its local data with the attack hooks off and
    hands the clean delta over through :meth:`set_reference_delta`.
    """

    strategy = ""

    def __init__(
        self,
        coordinator: CollusionCoordinator,
        base: UpdateManipulator | None = None,
    ) -> None:
        if base is not None and callable(getattr(base, "set_reference_delta", None)):
            raise ValueError(
                "a collusion strategy cannot wrap a base manipulator that needs "
                "its own clean reference"
            )
        self.coordinator = coordinator
        self.base = base
        self._pending_reference: NDArrays | None = None

    # -- clean reference -------------------------------------------------

    def wants_reference_delta(self, client_id: ClientId, ctx: RoundContext) -> bool:
        """Whether this client needs its clean update this round (default: no)."""
        return False

    def set_reference_delta(self, reference_delta: NDArrays) -> None:
        """Record the calling client's clean delta; claimed in :meth:`manipulate`."""
        self._pending_reference = [np.asarray(r) for r in reference_delta]

    def _clean_delta(self, client_id: ClientId, ctx: RoundContext) -> NDArrays:
        pending = self._pending_reference
        if pending is None:
            raise RuntimeError(
                f"{self.strategy}: {client_id} needs its clean delta in round "
                f"{ctx.round_num} but none was set"
            )
        self._pending_reference = None
        return pending

    # -- base attack -----------------------------------------------------

    def _base(
        self, update: ClientUpdate, global_parameters: NDArrays, ctx: RoundContext
    ) -> ClientUpdate:
        if self.base is None:
            return update
        return self.base.manipulate(update, global_parameters, ctx)

    # -- entry point -----------------------------------------------------

    def manipulate(
        self,
        update: ClientUpdate,
        global_parameters: NDArrays,
        ctx: RoundContext,
    ) -> ClientUpdate:
        cid = update.client_id
        bloc = self.coordinator.colluders(ctx)
        if cid not in bloc:
            raise ValueError(f"{cid} is not a selected colluder in round {ctx.round_num}")
        wants_clean = self.wants_reference_delta(cid, ctx)
        clean = self._clean_delta(cid, ctx) if wants_clean else None
        if len(bloc) == 1:
            return self._base(update, global_parameters, ctx)
        if self.coordinator.is_leader(cid, ctx):
            submitted = self._base(update, global_parameters, ctx)
            lone = self._base(
                update, global_parameters, self.coordinator.lone_context(cid, ctx)
            )
            poisoned = _deltas(update.parameters, global_parameters)
            payload = CollusionPayload(
                round_num=ctx.round_num,
                leader=cid,
                submitted_delta=_deltas(submitted.parameters, global_parameters),
                lone_delta=_deltas(lone.parameters, global_parameters),
                poisoned_delta=poisoned,
                clean_delta=clean,
                shared=self._agree(poisoned, clean),
            )
            self.coordinator.post(payload, ctx, self.strategy)
        payload = self.coordinator.payload(ctx)
        delta = self._submission(cid, payload, clean, len(bloc))
        return _submit(update, global_parameters, delta)

    def _agree(self, poisoned: NDArrays, clean: NDArrays | None) -> dict[str, Any]:
        """Strategy-specific agreements the leader fixes for the round."""
        return {}

    def _submission(
        self,
        client_id: ClientId,
        payload: CollusionPayload,
        clean: NDArrays | None,
        m: int,
    ) -> NDArrays:
        raise NotImplementedError


class CloneManipulator(_CollusionManipulator):
    """S1: every colluder submits the leader's base-attack update."""

    strategy = "clone"

    def _submission(
        self,
        client_id: ClientId,
        payload: CollusionPayload,
        clean: NDArrays | None,
        m: int,
    ) -> NDArrays:
        return [np.array(d) for d in payload.submitted_delta]


class BudgetSplitManipulator(_CollusionManipulator):
    """S2: every colluder submits a lone attacker's update divided by ``m``."""

    strategy = "budget_split"

    def _submission(
        self,
        client_id: ClientId,
        payload: CollusionPayload,
        clean: NDArrays | None,
        m: int,
    ) -> NDArrays:
        return [d / m for d in payload.lone_delta]


class ReferenceShiftManipulator(_CollusionManipulator):
    """S3: one carrier at a lone attacker's update, decoys along its direction.

    The leader carries the payload. Each decoy submits the carrier's delta
    direction scaled to the L2 norm of its own clean update, so it looks
    benign on magnitude while pulling the cohort mean toward the carrier.
    """

    strategy = "reference_shift"

    def __init__(
        self,
        coordinator: CollusionCoordinator,
        base: UpdateManipulator | None = None,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(coordinator, base)
        self.eps = float(eps)

    def wants_reference_delta(self, client_id: ClientId, ctx: RoundContext) -> bool:
        bloc = self.coordinator.colluders(ctx)
        return len(bloc) > 1 and bloc[0] != client_id

    def _submission(
        self,
        client_id: ClientId,
        payload: CollusionPayload,
        clean: NDArrays | None,
        m: int,
    ) -> NDArrays:
        carrier = payload.lone_delta
        if client_id == payload.leader:
            return [np.array(d) for d in carrier]
        assert clean is not None
        carrier_norm = _norm(carrier)
        if carrier_norm <= self.eps:
            return [np.array(d) for d in carrier]
        scale = _norm(clean) / carrier_norm
        return [scale * d for d in carrier]


class CoordinateCaptureManipulator(_CollusionManipulator):
    """S4: identical values on the backdoor's top coordinates, clean elsewhere.

    The leader fixes the captured set: the ``top_fraction`` of all coordinates
    with the largest ``|poisoned - clean|`` in its own updates (at least one).
    On those coordinates every colluder submits the leader's base-attack value;
    everywhere else it submits its own clean delta. The result is not
    renormalised.
    """

    strategy = "coordinate_capture"

    def __init__(
        self,
        coordinator: CollusionCoordinator,
        base: UpdateManipulator | None = None,
        top_fraction: float = 0.05,
    ) -> None:
        if not 0.0 < top_fraction <= 1.0:
            raise ValueError(f"top_fraction must lie in (0, 1]; got {top_fraction}")
        super().__init__(coordinator, base)
        self.top_fraction = float(top_fraction)

    def wants_reference_delta(self, client_id: ClientId, ctx: RoundContext) -> bool:
        return len(self.coordinator.colluders(ctx)) > 1

    def _agree(self, poisoned: NDArrays, clean: NDArrays | None) -> dict[str, Any]:
        assert clean is not None
        direction = np.concatenate(
            [np.ravel(p - c) for p, c in zip(poisoned, clean, strict=True)]
        )
        k = max(1, math.ceil(self.top_fraction * direction.size))
        top = np.argpartition(np.abs(direction), direction.size - k)[-k:]
        return {"captured": np.sort(top)}

    def _submission(
        self,
        client_id: ClientId,
        payload: CollusionPayload,
        clean: NDArrays | None,
        m: int,
    ) -> NDArrays:
        assert clean is not None
        captured = payload.shared["captured"]
        flat = np.concatenate([np.ravel(c) for c in clean]).astype(np.float64)
        leader = np.concatenate([np.ravel(d) for d in payload.submitted_delta])
        flat[captured] = leader[captured]
        out: NDArrays = []
        offset = 0
        for c in clean:
            out.append(flat[offset : offset + c.size].reshape(c.shape).astype(c.dtype))
            offset += c.size
        return out


STRATEGIES: dict[str, type[_CollusionManipulator]] = {
    cls.strategy: cls
    for cls in (
        CloneManipulator,
        BudgetSplitManipulator,
        ReferenceShiftManipulator,
        CoordinateCaptureManipulator,
    )
}


def build_collusion_manipulator(
    strategy: str,
    coordinator: CollusionCoordinator,
    base: UpdateManipulator | None,
    **kwargs: Any,
) -> _CollusionManipulator:
    """Construct the named strategy around ``base``; ``kwargs`` are its knobs."""
    try:
        cls = STRATEGIES[strategy]
    except KeyError:
        raise ValueError(
            f"unknown collusion strategy {strategy!r}; expected one of {sorted(STRATEGIES)}"
        ) from None
    return cls(coordinator, base, **kwargs)
