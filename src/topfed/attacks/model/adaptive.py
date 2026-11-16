"""Adaptive (defense-aware) update manipulator.

A defense-aware attacker knows the server flags updates that are unusually
large or that deviate sharply from the cohort, so instead of amplifying its
update without limit (as model replacement does), it projects its delta onto a
fixed L2-norm budget chosen to sit in the benign range. This keeps the
submitted update's magnitude — and, because direction is preserved, its
per-layer deviation profile — indistinguishable from an honest update on
scale-based signals, while still steering the global model toward the
attacker's objective as hard as the budget permits.

It is the update-side counterpart to the constrain-and-scale loss term: rather
than penalising drift during training, it caps the magnitude of the finished
update at submission time. Targets the norm-ratio and layer-deviation metrics;
it does not, on its own, evade the server-anchored validation-loss signal.

The module also holds the direction-aligned manipulator, which evades a trust
signal measured as cosine to a clean reference, and a composition that applies
manipulators in sequence so an attacker can meet both constraints at once.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from topfed.attacks.protocols import UpdateManipulator
from topfed.core.types import ClientUpdate, NDArrays, RoundContext


@dataclass(frozen=True, slots=True)
class NormBoundedManipulator:
    """Rescale the full-update delta to a fixed L2-norm budget.

    ``target_norm`` is the attacker's estimate of a benign update's norm. The
    delta ``local - global`` is scaled (direction preserved) so its total L2
    norm equals ``target_norm``; the update is left unchanged if its delta is
    numerically zero.
    """

    target_norm: float
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if self.target_norm <= 0:
            raise ValueError(f"target_norm must be positive; got {self.target_norm}")

    def manipulate(
        self,
        update: ClientUpdate,
        global_parameters: NDArrays,
        ctx: RoundContext,
    ) -> ClientUpdate:
        deltas = [
            w - g
            for w, g in zip(update.parameters, global_parameters, strict=True)
        ]
        total_norm = float(np.sqrt(sum(float(np.sum(d**2)) for d in deltas)))
        if total_norm <= self.eps:
            return update
        scale = self.target_norm / total_norm
        projected = [
            g + scale * d
            for g, d in zip(global_parameters, deltas, strict=True)
        ]
        return ClientUpdate(
            client_id=update.client_id,
            parameters=[np.asarray(p) for p in projected],
            num_examples=update.num_examples,
            train_loss=update.train_loss,
            train_accuracy=update.train_accuracy,
        )


def _flat_dot(a: NDArrays, b: NDArrays) -> float:
    return float(sum(float(np.sum(ai * bi)) for ai, bi in zip(a, b, strict=True)))


class DirectionAlignedManipulator:
    """Align the submitted delta with a benign reference direction.

    A trust filter that scores each client by the cosine between its update and
    the server's own clean-data update — and normalizes every client's
    magnitude to the server's before averaging — cannot be evaded by shrinking
    the update: magnitude is discarded, and only the *direction* is judged. The
    counter is to point the submitted delta almost exactly along a benign
    reference direction (earning near-maximal trust) while retaining the
    component of the poisoned delta that is *orthogonal* to that reference. That
    orthogonal component is the part of the attack that benign learning does not
    explain — it carries the backdoor — and it survives the cosine gate because
    it is invisible to a similarity measured against the reference.

    The reference is the attacker's own clean update on its local data (set each
    round via :meth:`set_reference_delta`), a white-box proxy for the server's
    clean-data direction. ``cosine_target`` fixes the submitted update's cosine
    to that reference: closer to 1 buys more trust but injects less poison per
    round; lower values inject more poison at the cost of trust. It is the
    direction-domain analogue of a norm budget — the knob the attacker tunes
    against this defense.
    """

    def __init__(self, cosine_target: float = 0.9, eps: float = 1e-12) -> None:
        if not 0.0 < cosine_target < 1.0:
            raise ValueError(
                f"cosine_target must lie in (0, 1); got {cosine_target}"
            )
        self.cosine_target = float(cosine_target)
        self.eps = float(eps)
        self._reference_delta: NDArrays | None = None

    def set_reference_delta(self, reference_delta: NDArrays) -> None:
        """Record the benign reference direction for the current round."""
        self._reference_delta = [np.asarray(r, dtype=np.float64) for r in reference_delta]

    def manipulate(
        self,
        update: ClientUpdate,
        global_parameters: NDArrays,
        ctx: RoundContext,  # noqa: ARG002
    ) -> ClientUpdate:
        reference = self._reference_delta
        if reference is None:
            return update  # no reference set -> pass through unchanged

        poisoned = [
            w - g for w, g in zip(update.parameters, global_parameters, strict=True)
        ]
        # Isolate the poison as the part of the poisoned delta that the benign
        # update does not account for, then strip its reference-aligned share so
        # only the orthogonal (unexplained-by-benign) backdoor remains.
        poison = [p - r for p, r in zip(poisoned, reference, strict=True)]
        ref_sq = _flat_dot(reference, reference)
        if ref_sq <= self.eps:
            return update
        proj = _flat_dot(poison, reference) / ref_sq
        poison_orth = [pp - proj * r for pp, r in zip(poison, reference, strict=True)]

        norm_ref = float(np.sqrt(ref_sq))
        norm_orth = float(np.sqrt(_flat_dot(poison_orth, poison_orth)))
        if norm_orth <= self.eps:
            return update  # poison lies entirely along benign; nothing to hide

        c = self.cosine_target
        along = c / norm_ref
        across = float(np.sqrt(1.0 - c * c)) / norm_orth
        # u = norm_ref * (c * r_hat + sqrt(1-c^2) * orth_hat): cosine to the
        # reference is exactly c, magnitude equals the benign reference norm.
        aligned = [
            norm_ref * (along * r + across * o)
            for r, o in zip(reference, poison_orth, strict=True)
        ]
        projected = [
            g + a for g, a in zip(global_parameters, aligned, strict=True)
        ]
        return ClientUpdate(
            client_id=update.client_id,
            parameters=[np.asarray(p) for p in projected],
            num_examples=update.num_examples,
            train_loss=update.train_loss,
            train_accuracy=update.train_accuracy,
        )


class ComposedManipulator:
    """Apply several update manipulators in sequence, first to last.

    Each stage sees the previous stage's output. The intended composition is a
    direction-aligning stage followed by a norm-bounding stage: the aligned
    delta keeps its cosine to the benign reference exactly, and the norm bound
    only rescales it, so the submitted update satisfies both constraints at
    once. That is the attacker's response to a defense that checks both the
    direction to a clean reference and the magnitude.

    A stage that needs a benign reference delta (see
    :meth:`DirectionAlignedManipulator.set_reference_delta`) receives it through
    :meth:`set_reference_delta`, which forwards to every stage that accepts it.
    """

    def __init__(self, manipulators: Sequence[UpdateManipulator]) -> None:
        stages = tuple(manipulators)
        if not stages:
            raise ValueError("ComposedManipulator needs at least one stage")
        self.manipulators = stages

    def set_reference_delta(self, reference_delta: NDArrays) -> None:
        """Forward the round's benign reference delta to the stages that use it."""
        for stage in self.manipulators:
            set_ref = getattr(stage, "set_reference_delta", None)
            if callable(set_ref):
                set_ref(reference_delta)

    def manipulate(
        self,
        update: ClientUpdate,
        global_parameters: NDArrays,
        ctx: RoundContext,
    ) -> ClientUpdate:
        for stage in self.manipulators:
            update = stage.manipulate(update, global_parameters, ctx)
        return update
