"""Gradient-inversion probe: what a participant can reconstruct of its peers' data.

A federated round leaks information about the clients' training images
through the parameter updates it produces. Two observers are modelled:

* the **server**, which receives every client's update verbatim and can
  invert it directly (the classic single-client gradient-inversion setting);
* an honest-but-curious **peer**, which only sees the broadcast global model
  before and after the round, so its target is the aggregated update. The
  peer knows its own update and removes it analytically from a mean; for a
  median-type rule it masks out the coordinates where the aggregate equals
  its own value, since those were taken from itself.

For one captured round the script computes the aggregate under several
rules (sample-weighted mean, the capped trust-weighted coordinate median,
the plain coordinate median, the trimmed mean, Krum, and two means driven
by the same trust scores: one weighted by the score, one equal-weighted
over the clients scoring at least one half), runs a
cosine-matching inversion (Geiping et al., 2020: optimise dummy images so a
single gradient step on them aligns with the observed update, with total
variation as the image prior; by default labels are given to the attacker,
an upper bound on its power, and ``--labels recovered`` makes it guess them
from the output-bias delta instead) against each observer's target, and scores the
reconstructions against the true images by PSNR and SSIM after an optimal
one-to-one matching.

Several client protocols are captured so the comparison spans the range
from the most attacker-favourable (one image, one SGD step) to the paper's
protocol (the client's whole Dirichlet shard, five epochs, batch 64), and
from a freshly initialised global model to one warmed up by a few FedAvg
rounds. Everything is written under ``--out``: a JSON ledger of all numbers
and one figure per captured round.

Usage::

    uv run python experiments/gradient_inversion.py --dataset cifar100 \\
        --data-root data --out /tmp/recon
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from topfed.aggregation import (
    coordinate_median_aggregate,
    krum_aggregate,
    trimmed_mean_aggregate,
)
from topfed.core.seeding import derive_round_seed, set_all_seeds
from topfed.core.types import ClientUpdate, FederationConfig, NDArrays, RoundContext
from topfed.data import build_client_loaders, build_partition, load_cifar10, load_cifar100
from topfed.data.cifar10 import CIFAR10_NORMALIZE_MEAN, CIFAR10_NORMALIZE_STD
from topfed.data.cifar100 import CIFAR100_NORMALIZE_MEAN, CIFAR100_NORMALIZE_STD
from topfed.federation.client import BenignTorchClient, get_parameters, set_parameters
from topfed.federation.sampling import uniform_sample
from topfed.federation.scoring_bridge import MetricScorer
from topfed.federation.server import (
    fedavg_aggregate,
    run_federation,
    weighted_aggregate,
    weighted_coordinate_median,
)
from topfed.integration.mechanisms import CappedScoreWeight, HardFilter, ScoreWeight
from topfed.metrics import MetricsExtractor
from topfed.models.cifar_cnn import CifarCNN
from topfed.scoring import WEIGHTING_STRATEGIES, TopsisScorer

# ----------------------------------------------------------------------------
# Protocols captured. ``images`` is the number of training images each
# selected client keeps from its shard (``None`` = the whole shard).
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Protocol:
    name: str
    images: int | None
    local_epochs: int
    batch_size: int
    description: str


PROTOCOLS: tuple[Protocol, ...] = (
    Protocol("1step_b1", 1, 1, 1, "one image, one SGD step (most favourable)"),
    Protocol("1step_b2", 2, 1, 2, "two images, one SGD step"),
    Protocol("1step_b4", 4, 1, 4, "four images, one SGD step"),
    Protocol("1step_b8", 8, 1, 8, "eight images, one SGD step"),
    Protocol("1step_b16", 16, 1, 16, "sixteen images, one SGD step"),
    Protocol("5ep_b1", 1, 5, 1, "one image, five epochs (five steps)"),
    Protocol("paper", None, 5, 64, "whole shard, five epochs, batch 64 (paper)"),
)
DEFAULT_PROTOCOLS = ("1step_b1", "1step_b2", "5ep_b1", "paper")
_AD_HOC_PROTOCOL = re.compile(r"i(\d+)_e(\d+)_b(\d+)")


def _resolve_protocols(names: list[str]) -> list[Protocol]:
    """The named protocols first (in table order), then the ad hoc ones as given.

    An ad hoc name ``i<images>_e<epochs>_b<batch>`` describes a client that
    keeps ``images`` images and trains ``epochs`` epochs at batch size
    ``batch``.
    """
    known = {p.name for p in PROTOCOLS}
    out = [p for p in PROTOCOLS if p.name in names]
    for name in names:
        if name in known:
            continue
        m = _AD_HOC_PROTOCOL.fullmatch(name)
        if m is None:
            raise ValueError(f"unknown protocol {name!r}")
        images, epochs, batch = (int(v) for v in m.groups())
        if min(images, epochs, batch) < 1:
            raise ValueError(f"protocol {name!r} needs positive images, epochs and batch size")
        out.append(
            Protocol(name, images, epochs, batch, f"{images} images, {epochs} epochs, batch {batch}")
        )
    return out


def _local_steps(protocol: Protocol) -> int | None:
    """SGD steps a client takes under the protocol (``None`` for a whole shard)."""
    if protocol.images is None:
        return None
    return protocol.local_epochs * math.ceil(protocol.images / protocol.batch_size)

AGGREGATORS: tuple[str, ...] = (
    "mean",
    "topfed",
    "median",
    "trimmed_mean",
    "krum",
    "scoreweight",
    "hardfilter",
)
# Rules that average the clients under weights derived from the trust scores
# (``scoreweight``: proportional to the score; ``hardfilter``: an equal share
# for every client scoring at least ``HARD_FILTER_THRESHOLD``, zero below).
TRUST_MEAN_RULES: tuple[str, ...] = ("scoreweight", "hardfilter")
HARD_FILTER_THRESHOLD = 0.5
OBSERVERS: tuple[str, ...] = (
    "server",
    "peer",
    "peer_aware",
    "peer_estimated",
    "peer_oracle",
    "calibration",
)
# The weight-estimating attacker scores its own simulation of the round on a
# validation set of its own: up to this many images of its shard that it
# does not train on in the captured round.
PROXY_VAL_MAX = 256
REWEIGHT_EVERY = 100


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _flat(arrs: NDArrays) -> np.ndarray:
    return np.concatenate([np.asarray(a).ravel() for a in arrs])


def _denorm(x: torch.Tensor, mean: tuple[float, ...], std: tuple[float, ...]) -> torch.Tensor:
    m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x * s + m).clamp(0.0, 1.0)


def _recover_labels(
    bias_delta: np.ndarray, count: int, device: str | torch.device
) -> torch.Tensor:
    """Guess ``count`` labels from the output-bias delta of an attacked update.

    Slots go to classes in proportion to their positive bias delta (largest
    remainder); a class with no positive delta gets none unless the positive
    classes cannot fill the count, in which case the largest deltas fill it.
    """
    pos = np.clip(np.asarray(bias_delta, dtype=np.float64), 0.0, None)
    if pos.sum() <= 0:
        order = np.argsort(-bias_delta)
        return torch.as_tensor(np.resize(order, count), dtype=torch.long, device=device)
    share = pos / pos.sum() * count
    alloc = np.floor(share).astype(int)
    rem = count - int(alloc.sum())
    if rem > 0:
        for c in np.argsort(-(share - alloc))[:rem]:
            alloc[c] += 1
    labels = np.repeat(np.arange(len(pos)), alloc)
    return torch.as_tensor(labels, dtype=torch.long, device=device)


def _psnr_matrix(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """PSNR between every image of ``a`` (M) and every image of ``b`` (P), in [0,1] space."""
    out = np.empty((a.shape[0], b.shape[0]), dtype=np.float64)
    bf = b.flatten(1)
    for i in range(a.shape[0]):
        mse = ((a[i].flatten()[None, :] - bf) ** 2).mean(dim=1).clamp_min(1e-10)
        out[i] = (10.0 * torch.log10(1.0 / mse)).cpu().numpy()
    return out


def _gaussian_window(
    size: int = 11, sigma: float = 1.5, device: torch.device | None = None
) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(3, 1, size, size).contiguous()


def _ssim(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """Per-image SSIM (Wang et al. 2004) for aligned batches in [0,1] space."""
    w = _gaussian_window(device=a.device)
    c1, c2 = 0.01**2, 0.03**2
    mu_a = F.conv2d(a, w, padding=5, groups=3)
    mu_b = F.conv2d(b, w, padding=5, groups=3)
    s_aa = F.conv2d(a * a, w, padding=5, groups=3) - mu_a**2
    s_bb = F.conv2d(b * b, w, padding=5, groups=3) - mu_b**2
    s_ab = F.conv2d(a * b, w, padding=5, groups=3) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * s_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (s_aa + s_bb + c2)
    return (num / den).flatten(1).mean(dim=1).cpu().numpy()


def _total_variation(x: torch.Tensor) -> torch.Tensor:
    dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    return dx + dy


# ----------------------------------------------------------------------------
# The inversion
# ----------------------------------------------------------------------------


@dataclass
class InversionResult:
    images: torch.Tensor  # (M, 3, H, W) normalised space
    rec_loss: float
    masked_fraction: float
    # The simulator's per-client weights when the attack ended, peers first
    # and the attacker last; only set when the attacker re-estimates them.
    weights: list[float] | None = None


@dataclass(frozen=True)
class AwareSimulator:
    """Lets the attacker simulate a median-type rule instead of a plain gradient.

    ``rule`` is ``median`` (mean of the two middle order statistics, as
    ``numpy.median`` for an even cohort), ``lower_median`` (the value at which
    the cumulative equal weight first reaches one half, i.e. the capped
    trust-weighted median with every client at its fair share),
    ``weighted_median`` or ``weighted_mean`` (the rule under known per-client
    weights), or ``trimmed_mean`` (drop ``trim`` clients at each end). The
    attacker knows
    the cohort size and the client step size, groups its dummy images by
    peer, turns each group's gradient into that peer's delta, stacks its own
    true delta on top and applies the rule coordinate-wise.
    """

    rule: str
    own_delta: torch.Tensor  # (P,) the attacker's own true delta
    groups: tuple[tuple[int, int], ...]  # [start, stop) of each peer's dummy images
    step_size: float
    trim: int = 2
    # ``weighted_median`` and ``weighted_mean`` only: the rule's true
    # per-client weights, peers first (in group order) and the attacker last.
    # An oracle the real peer does not have; it measures how much of the
    # protection is the secrecy of the weights rather than the rule itself.
    # For ``weighted_mean`` the attacker's entry is zero when its own term
    # has already been subtracted from the target.
    weights: torch.Tensor | None = None

    def aggregate(self, peer_grads: list[torch.Tensor]) -> torch.Tensor:
        deltas = torch.stack([-self.step_size * g for g in peer_grads] + [self.own_delta])
        n = deltas.shape[0]
        sorted_v, order = torch.sort(deltas, dim=0)
        if self.rule == "median":
            return 0.5 * (sorted_v[n // 2 - 1] + sorted_v[n // 2]) if n % 2 == 0 else sorted_v[n // 2]
        if self.rule == "lower_median":
            idx = int(np.argmax(np.cumsum(np.full(n, 1.0 / n)) >= 0.5 - 1e-12))
            return sorted_v[idx]
        if self.rule == "weighted_median":
            assert self.weights is not None
            w = self.weights.to(deltas.device)[order]  # weight of each sorted entry
            cumw = torch.cumsum(w, dim=0)
            half = w.sum(dim=0, keepdim=True) / 2.0
            pick = torch.argmax((cumw >= half - 1e-12).to(torch.int8), dim=0, keepdim=True)
            return torch.gather(sorted_v, 0, pick)[0]
        if self.rule == "weighted_mean":
            assert self.weights is not None
            wm = self.weights.to(deltas.device)
            return (wm[:, None] * deltas).sum(dim=0) / wm.sum()
        if self.rule == "trimmed_mean":
            return sorted_v[self.trim : n - self.trim].mean(dim=0)
        raise ValueError(f"unknown rule {self.rule!r}")


def invert(
    model: nn.Module,
    target_delta: np.ndarray,
    mask: np.ndarray | None,
    labels: torch.Tensor,
    *,
    bounds: tuple[torch.Tensor, torch.Tensor],
    iterations: int,
    restarts: int,
    tv_weight: float,
    lr: float,
    seed: int,
    device: torch.device,
    simulator: AwareSimulator | None = None,
    reweight: Callable[[np.ndarray], np.ndarray] | None = None,
    reweight_every: int = REWEIGHT_EVERY,
) -> InversionResult:
    """Optimise ``len(labels)`` dummy images whose gradient aligns with the target.

    The target is an observed parameter *delta* (``local - global``); a
    gradient step moves against the gradient, so the dummy gradient is matched
    to ``-delta``. The loss is the cosine distance over the unmasked
    coordinates (scale-free, so the client's step size and step count need
    not be known) plus ``tv_weight`` times the total variation. Signed
    Adam with a three-stage step decay, as in Geiping et al.; the best of
    ``restarts`` random initialisations is returned.

    With a ``simulator`` the dummy images are grouped by peer and the
    simulated rule's output, rather than one batch gradient, is matched to
    the observed delta itself (a median-aware attacker).

    With ``reweight`` the attacker does not know the rule's per-client
    weights and estimates them as it goes: every ``reweight_every``
    iterations the peers' deltas simulated from the current dummy images
    (one row per peer) are passed to ``reweight``, and the weights it returns
    (peers first, the attacker last) replace the simulator's. Each restart
    begins from the weights the simulator was built with.
    """
    params = list(model.parameters())
    sign = 1.0 if simulator is not None else -1.0
    target = torch.from_numpy(sign * target_delta.astype(np.float32)).to(device)
    if mask is None:
        m = torch.ones_like(target)
    else:
        m = torch.from_numpy(mask.astype(np.float32)).to(device)
    target_m = target * m
    target_norm = target_m.norm().clamp_min(1e-12)
    lo, hi = bounds
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def peer_grads(x: torch.Tensor, groups: tuple[tuple[int, int], ...], graph: bool) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        for start, stop in groups:
            loss = F.cross_entropy(model(x[start:stop]), labels[start:stop])
            grads = torch.autograd.grad(loss, params, create_graph=graph)
            out.append(torch.cat([gi.reshape(-1) for gi in grads]))
        return out

    def simulated(x: torch.Tensor, sim: AwareSimulator | None) -> torch.Tensor:
        if sim is None:
            loss = F.cross_entropy(model(x), labels)
            grads = torch.autograd.grad(loss, params, create_graph=True)
            return torch.cat([gi.reshape(-1) for gi in grads])
        return sim.aggregate(peer_grads(x, sim.groups, True))

    best: InversionResult | None = None
    for _ in range(restarts):
        x = (
            torch.randn(len(labels), 3, 32, 32, generator=gen)
            .to(device)
            .requires_grad_(True)
        )
        opt = torch.optim.Adam([x], lr=lr)
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt,
            milestones=[
                int(iterations / 2.667),
                int(iterations / 1.6),
                int(iterations / 1.142),
            ],
            gamma=0.1,
        )
        last = math.inf
        sim = simulator
        for it in range(iterations):
            if reweight is not None and sim is not None and it > 0 and it % reweight_every == 0:
                sim_deltas = torch.stack(
                    [-sim.step_size * g for g in peer_grads(x, sim.groups, False)]
                )
                estimate = np.asarray(reweight(sim_deltas.detach().cpu().numpy()), dtype=np.float32)
                sim = replace(sim, weights=torch.from_numpy(estimate))
            opt.zero_grad()
            model.zero_grad()
            g = simulated(x, sim) * m
            rec = 1.0 - (g * target_m).sum() / (g.norm().clamp_min(1e-12) * target_norm)
            total = rec + tv_weight * _total_variation(x)
            total.backward()
            assert x.grad is not None
            x.grad.sign_()
            opt.step()
            sched.step()
            with torch.no_grad():
                x.data = torch.max(torch.min(x, hi), lo)
            last = float(rec.item())
        cand = InversionResult(
            images=x.detach().clone(),
            rec_loss=last,
            masked_fraction=float(1.0 - m.mean().item()),
            weights=(
                [float(v) for v in sim.weights]
                if reweight is not None and sim is not None and sim.weights is not None
                else None
            ),
        )
        if best is None or cand.rec_loss < best.rec_loss:
            best = cand
    assert best is not None
    return best


# ----------------------------------------------------------------------------
# Capture
# ----------------------------------------------------------------------------


def _subset_partition(partition: list[np.ndarray], images: int | None) -> list[np.ndarray]:
    if images is None:
        return partition
    return [np.asarray(shard[:images], dtype=np.int64) for shard in partition]


def _client_updates(
    *,
    bundle: Any,
    partition: list[np.ndarray],
    selected: tuple[str, ...],
    global_parameters: NDArrays,
    model_factory: Any,
    protocol: Protocol,
    round_num: int,
    seed: int,
    device: str,
) -> list[ClientUpdate]:
    cfg = FederationConfig(
        num_clients=len(partition),
        clients_per_round=len(selected),
        num_rounds=1,
        local_epochs=protocol.local_epochs,
        local_batch_size=protocol.batch_size,
        local_learning_rate=0.01,
        local_momentum=0.9,
        seed=seed,
    )
    loaders = build_client_loaders(
        bundle.train_x,
        bundle.train_y,
        _subset_partition(partition, protocol.images),
        batch_size=protocol.batch_size,
        augment=False,
    )
    ctx = RoundContext(
        round_num=round_num,
        total_rounds=200,
        global_parameters=global_parameters,
        selected_clients=selected,
        seed=derive_round_seed(seed, round_num),
    )
    set_all_seeds(ctx.seed)
    return [
        BenignTorchClient(cid, model_factory, loaders[cid], device).local_train(ctx, cfg)
        for cid in selected
    ]


def _trust_scores(
    updates: list[ClientUpdate],
    global_parameters: NDArrays,
    *,
    round_num: int,
    val_loader: DataLoader,
    model_factory: Any,
    device: str,
) -> tuple[tuple[str, ...], np.ndarray]:
    """TOPSIS/CRITIC trust scores of one round, from a fresh metrics extractor."""
    extractor = MetricsExtractor(val_loader=val_loader, model_factory=model_factory, device=device)
    scorer = MetricScorer(extractor, TopsisScorer(weighting=WEIGHTING_STRATEGIES["critic"]))
    return scorer.score_round(round_num, updates, global_parameters)


def _aggregates(
    updates: list[ClientUpdate],
    global_parameters: NDArrays,
    *,
    round_num: int,
    **scoring_kwargs: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Aggregated delta under every rule, and the weights the scored rules applied.

    The three score-driven rules share one set of trust scores: capped at a
    fair share and weighting a coordinate median (``topfed``), weighting a
    mean (``scoreweight``), or selecting the clients of an equal-weight mean
    (``hardfilter``).
    """
    g = _flat(global_parameters)
    out: dict[str, np.ndarray] = {}
    info: dict[str, Any] = {}
    out["mean"] = _flat(fedavg_aggregate(updates)) - g
    ids, scores = _trust_scores(updates, global_parameters, round_num=round_num, **scoring_kwargs)
    info["trust_scores"] = [float(v) for v in scores]
    w = CappedScoreWeight(cap=1.0, alpha=1.0).integrate(round_num, ids, scores).weights
    out["topfed"] = _flat(weighted_coordinate_median(updates, w)) - g
    info["topfed_weights"] = [float(v) for v in w]
    for rule, integrator in (
        ("scoreweight", ScoreWeight()),
        ("hardfilter", HardFilter(threshold=HARD_FILTER_THRESHOLD)),
    ):
        w = integrator.integrate(round_num, ids, scores).weights
        out[rule] = _flat(weighted_aggregate(updates, w)) - g
        info[f"{rule}_weights"] = [float(v) for v in w]
    out["median"] = _flat(coordinate_median_aggregate(updates)) - g
    out["trimmed_mean"] = _flat(trimmed_mean_aggregate(updates, trim_ratio=0.2)) - g
    out["krum"] = _flat(krum_aggregate(updates, num_byzantine=2, num_selected=1)) - g
    return out, info


def _estimate_topfed_weights(
    peer_deltas: np.ndarray,
    own_update: ClientUpdate,
    global_parameters: NDArrays,
    images_per_peer: int,
    *,
    round_num: int,
    **scoring_kwargs: Any,
) -> np.ndarray:
    """The capped trust weights as a participant can compute them for itself.

    The attacker rebuilds the round from what it has: the peers' deltas it
    simulated from its dummy images (one row each) and its own true update.
    It scores that cohort exactly as the server would, except that the
    validation loss is measured on the attacker's own data (``val_loader``
    in ``scoring_kwargs``), and caps the scores at a fair share. Returns
    the weights with the peers first (in row order) and the attacker last.
    """
    flat_global = _flat(global_parameters)
    shapes = [np.asarray(a).shape for a in global_parameters]
    bounds = np.cumsum([0] + [int(np.prod(shape)) for shape in shapes])
    simulated = [
        ClientUpdate(
            f"peer{i}",
            [
                (flat_global + delta)[bounds[t] : bounds[t + 1]].reshape(shape)
                for t, shape in enumerate(shapes)
            ],
            images_per_peer,
            0.0,
            0.0,
        )
        for i, delta in enumerate(np.asarray(peer_deltas, dtype=flat_global.dtype))
    ]
    ids, scores = _trust_scores(
        [*simulated, own_update], global_parameters, round_num=round_num, **scoring_kwargs
    )
    weights: np.ndarray = (
        CappedScoreWeight(cap=1.0, alpha=1.0).integrate(round_num, ids, scores).weights
    )
    return weights


def _peer_target(
    rule: str,
    agg_delta: np.ndarray,
    updates: list[ClientUpdate],
    deltas: list[np.ndarray],
    attacker: int,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """What the peer attacks, and the coordinates it knows carry only itself.

    ``weights`` applies to the trust-weighted means only: the per-client
    weights the attacker believes the rule used when it removes its own
    term. ``None`` is the standard attacker, which assumes an equal share
    for every client; the rule's true weights make it the oracle.
    """
    own = deltas[attacker]
    if rule == "mean":
        n = np.array([u.num_examples for u in updates], dtype=np.float64)
        total = n.sum()
        residual = (agg_delta * total - n[attacker] * own) / (total - n[attacker])
        return residual, None, {"removed_own": True}
    if rule in TRUST_MEAN_RULES:
        w = np.ones(len(updates)) if weights is None else np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        tinfo: dict[str, Any] = {
            "removed_own": True,
            "assumed_weights": "equal" if weights is None else "true",
        }
        rest = 1.0 - w[attacker]
        if rest <= 1e-12:
            # Every peer had zero weight: the aggregate is the attacker's own update.
            return agg_delta, np.zeros_like(agg_delta, dtype=bool), tinfo
        return (agg_delta - w[attacker] * own) / rest, None, tinfo
    same = agg_delta == own
    if rule == "krum":
        # Krum returns one client's parameters verbatim. If it chose the
        # attacker, the aggregate carries nothing about anyone else.
        if same.all():
            return agg_delta, np.zeros_like(agg_delta, dtype=bool), {"krum_selected_attacker": True}
        return agg_delta, None, {"krum_selected_attacker": False}
    # A coordinate equal to the attacker's own non-zero value was selected
    # from the attacker and says nothing about the peers. A shared zero (a
    # unit no client activated) is left in: it is consistent with the peers
    # too, and dropping it would only hide information from the attacker.
    own_nonzero = same & (own != 0.0)
    info: dict[str, Any] = {"own_coordinate_fraction": float(own_nonzero.mean())}
    return agg_delta, ~own_nonzero, info


# ----------------------------------------------------------------------------
# Scoring and figures
# ----------------------------------------------------------------------------


def _score(
    recon01: torch.Tensor, pool01: torch.Tensor, one_to_one: bool
) -> tuple[dict[str, float], np.ndarray]:
    """Match reconstructions to true images and summarise.

    One-to-one (Hungarian on PSNR) when the attacker reconstructs exactly the
    pool; otherwise each reconstruction takes its nearest true image.
    Returns the summary and, per reconstruction, the index of its match.
    """
    pm = _psnr_matrix(recon01, pool01)
    if one_to_one and recon01.shape[0] == pool01.shape[0]:
        rows, cols = linear_sum_assignment(-pm)
        match = np.empty(recon01.shape[0], dtype=np.int64)
        match[rows] = cols
    else:
        match = pm.argmax(axis=1)
    psnr = pm[np.arange(recon01.shape[0]), match]
    ssim = _ssim(recon01, pool01[torch.from_numpy(match).to(pool01.device)])
    return (
        {
            "psnr_mean": float(psnr.mean()),
            "psnr_max": float(psnr.max()),
            "ssim_mean": float(ssim.mean()),
            "ssim_max": float(ssim.max()),
        },
        match,
    )


def _save_figure(
    path: Path,
    title: str,
    truth01: torch.Tensor,
    rows: list[tuple[str, torch.Tensor | None, np.ndarray | None, str]],
    max_cols: int,
) -> None:
    """Grid: first row the true images, then one row per attacked target.

    Each reconstruction is placed under the true image it was matched to;
    unmatched columns stay blank.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = min(truth01.shape[0], max_cols)
    n_rows = 1 + len(rows)
    fig, axes = plt.subplots(
        n_rows, cols, figsize=(1.25 * cols + 1.8, 1.3 * n_rows + 0.6), squeeze=False
    )
    for c in range(cols):
        axes[0, c].imshow(truth01[c].permute(1, 2, 0).cpu().numpy())
    axes[0, 0].set_ylabel("true", rotation=0, ha="right", va="center", fontsize=9)
    for r, (label, imgs, match, note) in enumerate(rows, start=1):
        if imgs is not None and match is not None:
            for i in range(imgs.shape[0]):
                c = int(match[i])
                if c < cols:
                    axes[r, c].imshow(imgs[i].permute(1, 2, 0).cpu().numpy())
        axes[r, 0].set_ylabel(
            f"{label}\n{note}", rotation=0, ha="right", va="center", fontsize=8
        )
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar100")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--warmup-rounds", type=int, default=10, help="FedAvg rounds before the 'warm' capture"
    )
    ap.add_argument("--states", default="init,warm", help="comma list from {init,warm}")
    ap.add_argument(
        "--protocols",
        default=",".join(DEFAULT_PROTOCOLS),
        help=f"comma list from {{{', '.join(p.name for p in PROTOCOLS)}}}, or ad hoc "
        "names i<images>_e<epochs>_b<batch>",
    )
    ap.add_argument(
        "--surrogate-step",
        action="store_true",
        help="also run the rule-simulating attackers on multi-step captures, treating each "
        "peer's update as one full-batch step whose size the attacker calibrates on its own update",
    )
    ap.add_argument(
        "--rules",
        default=",".join(AGGREGATORS),
        help="comma list of the aggregation rules attacked in the peer views",
    )
    ap.add_argument(
        "--observers",
        default=",".join(OBSERVERS),
        help="comma list of the observers (attacker variants and calibration rows) to run",
    )
    ap.add_argument("--iterations", type=int, default=3000)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--tv", type=float, default=1e-2)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument(
        "--paper-images",
        type=int,
        default=8,
        help="images the attacker reconstructs from a whole-shard update",
    )
    ap.add_argument(
        "--labels",
        choices=("given", "recovered"),
        default="given",
        help="whether the attacker is given the true labels (upper bound) or "
        "recovers them from the output-bias delta of the update it attacks",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    rules = tuple(args.rules.split(","))
    observers = tuple(args.observers.split(","))
    if unknown := sorted(set(rules) - set(AGGREGATORS)):
        ap.error(f"unknown --rules {unknown}; choose from {', '.join(AGGREGATORS)}")
    if unknown := sorted(set(observers) - set(OBSERVERS)):
        ap.error(f"unknown --observers {unknown}; choose from {', '.join(OBSERVERS)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    set_all_seeds(args.seed)

    if args.dataset == "cifar10":
        bundle = load_cifar10(args.data_root, val_per_class=100, seed=0, download=False)
        num_classes, mean, std = 10, CIFAR10_NORMALIZE_MEAN, CIFAR10_NORMALIZE_STD
    else:
        bundle = load_cifar100(
            args.data_root, val_per_class=10, coarse=False, seed=0, download=False
        )
        num_classes, mean, std = 100, CIFAR100_NORMALIZE_MEAN, CIFAR100_NORMALIZE_STD
    model_factory = partial(CifarCNN, num_classes=num_classes, input_size=32)
    partition = build_partition(
        "dirichlet", labels=bundle.train_y.numpy(), num_clients=100, seed=0, alpha=0.5
    )
    val_loader = DataLoader(
        TensorDataset(bundle.val_x, bundle.val_y), batch_size=256, shuffle=False
    )
    lo = ((0.0 - torch.tensor(mean)) / torch.tensor(std)).view(1, 3, 1, 1).to(device)
    hi = ((1.0 - torch.tensor(mean)) / torch.tensor(std)).view(1, 3, 1, 1).to(device)

    # The model whose gradients the attacker simulates; parameter order must
    # match the flattened state_dict order the federation uses.
    probe = model_factory().to(device)
    assert list(probe.state_dict()) == [n for n, _ in probe.named_parameters()]

    init_params = get_parameters(model_factory())
    states: dict[str, tuple[NDArrays, int]] = {}
    wanted = args.states.split(",")
    if "init" in wanted:
        states["init"] = (init_params, 1)
    if "warm" in wanted:
        print(
            f"[warmup] {args.warmup_rounds} FedAvg rounds at the paper protocol",
            flush=True,
        )
        fed = FederationConfig(
            num_clients=100,
            clients_per_round=10,
            num_rounds=args.warmup_rounds,
            local_epochs=5,
            local_batch_size=64,
            local_learning_rate=0.01,
            local_momentum=0.9,
            seed=args.seed,
        )
        loaders = build_client_loaders(
            bundle.train_x, bundle.train_y, partition, batch_size=64, augment=True
        )
        last: list[NDArrays] = []

        def keep_last(outcome: Any) -> None:
            last[:] = [outcome.aggregated_parameters]

        run_federation(
            fed,
            lambda cid: BenignTorchClient(cid, model_factory, loaders[cid], args.device),
            init_params,
            verbose=True,
            retain_parameters=False,
            on_round=keep_last,
        )
        states["warm"] = (last[0], args.warmup_rounds + 1)

    try:
        protocols = _resolve_protocols(args.protocols.split(","))
    except ValueError as err:
        ap.error(str(err))
    sampler = random.Random(args.seed)
    all_ids = tuple(f"c{i}" for i in range(100))
    selected = uniform_sample(sampler, all_ids, 10)
    attacker = 0
    peers = [i for i in range(len(selected)) if i != attacker]
    print(f"[capture] cohort {selected}; attacker is {selected[attacker]}", flush=True)

    ledger: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for state_name, (global_parameters, round_num) in states.items():
        flat_global = _flat(global_parameters)
        set_parameters(probe, global_parameters)
        probe.train()
        for proto in protocols:
            print(f"\n[{state_name}/{proto.name}] {proto.description}", flush=True)
            updates = _client_updates(
                bundle=bundle,
                partition=partition,
                selected=selected,
                global_parameters=global_parameters,
                model_factory=model_factory,
                protocol=proto,
                round_num=round_num,
                seed=args.seed,
                device=args.device,
            )
            deltas = [_flat(u.parameters) - flat_global for u in updates]
            aggs, info = _aggregates(
                updates,
                global_parameters,
                round_num=round_num,
                val_loader=val_loader,
                model_factory=model_factory,
                device=args.device,
            )
            print(
                f"  topfed weights: {np.round(info['topfed_weights'], 3).tolist()}",
                flush=True,
            )
            for rule in TRUST_MEAN_RULES:
                if rule in rules:
                    print(
                        f"  {rule} weights: {np.round(info[f'{rule}_weights'], 3).tolist()}",
                        flush=True,
                    )

            sub = _subset_partition(partition, proto.images)
            idx_of = {cid: sub[int(cid[1:])] for cid in selected}
            if proto.images is not None and min(len(idx_of[cid]) for cid in selected) < proto.images:
                raise SystemExit(
                    f"protocol {proto.name}: a selected client holds fewer than {proto.images} images"
                )
            images_of = {cid: bundle.train_x[torch.as_tensor(idx_of[cid])] for cid in selected}
            labels_of = {cid: bundle.train_y[torch.as_tensor(idx_of[cid])] for cid in selected}
            peer_pool = torch.cat([images_of[selected[i]] for i in peers]).to(device)
            peer_labels = torch.cat([labels_of[selected[i]] for i in peers]).to(device)
            whole_shard = proto.images is None

            def attack_labels(
                target: np.ndarray, truth_labels: torch.Tensor, tinfo: dict[str, Any] | None = None
            ) -> torch.Tensor:
                if not whole_shard and args.labels == "given":
                    return truth_labels  # known labels: the attacker's upper bound
                if not whole_shard:
                    # Label-free attacker: the number of images is known, the
                    # labels are not. Cross-entropy pushes the output bias of
                    # a present class up by an amount growing with its count,
                    # so allocate the slots to classes in proportion to the
                    # positive output-bias delta (largest remainder), which
                    # also handles repeated labels, unlike plain iDLG.
                    guess = _recover_labels(target[-num_classes:], len(truth_labels), device)
                    if tinfo is not None:  # multiset overlap with the truth
                        g = np.bincount(guess.cpu().numpy(), minlength=num_classes)
                        t = np.bincount(truth_labels.cpu().numpy(), minlength=num_classes)
                        tinfo["label_overlap"] = float(np.minimum(g, t).sum() / max(1, len(truth_labels)))
                    return guess
                # Whole-shard target: the attacker picks the classes whose
                # output-bias delta is most positive (cross-entropy pushes
                # the bias of a present class up, so the delta is positive
                # there) and reconstructs one image per class.
                bias = target[-num_classes:]
                top = np.argsort(-bias)[: args.paper_images]
                return torch.as_tensor(top, dtype=torch.long, device=device)

            Target = tuple[
                str,
                str,
                np.ndarray,
                np.ndarray | None,
                torch.Tensor,
                torch.Tensor,
                dict[str, Any],
                AwareSimulator | None,
                Callable[[np.ndarray], np.ndarray] | None,
            ]
            targets: list[Target] = []
            # Server view: one peer's own update, inverted directly. The victim
            # is the first peer, so its images lead the peer pool.
            victim = selected[peers[0]]
            sinfo: dict[str, Any] = {"victim": victim}
            tl = attack_labels(deltas[peers[0]], labels_of[victim].to(device), sinfo)
            targets.append(
                ("server", "client", deltas[peers[0]], None, images_of[victim].to(device), tl, sinfo, None, None)
            )
            # Peer view: the aggregate under each rule, inverted as if it were
            # one gradient (the standard attack).
            for rule in AGGREGATORS:
                tgt, mask, tinfo = _peer_target(rule, aggs[rule], updates, deltas, attacker)
                tl = attack_labels(tgt, peer_labels, tinfo)
                targets.append(("peer", rule, tgt, mask, peer_pool, tl, tinfo, None, None))
            # Peer view, median-aware: the attacker simulates the rule itself.
            # Exact when one gradient step is the whole update and the peers'
            # image counts are known (not the whole-shard protocol). For a
            # multi-step update it is run only on request, as a surrogate:
            # each peer's update is simulated as one full-batch step, whose
            # size the attacker reads off its own update (the projection of
            # its true delta on its own full-batch gradient at the global
            # model).
            steps = _local_steps(proto)
            if steps == 1 or (steps is not None and args.surrogate_step):
                b = int(proto.images or 0)
                groups = tuple((i * b, (i + 1) * b) for i in range(len(peers)))
                own = torch.from_numpy(deltas[attacker].astype(np.float32)).to(device)
                step_size = 0.01
                if steps != 1:
                    own_loss = F.cross_entropy(
                        probe(images_of[selected[attacker]].to(device)),
                        labels_of[selected[attacker]].to(device),
                    )
                    own_grad = torch.cat(
                        [g.reshape(-1) for g in torch.autograd.grad(own_loss, list(probe.parameters()))]
                    )
                    step_size = float(-(own @ own_grad) / (own_grad @ own_grad))
                    print(f"  surrogate step size: {step_size:.4f} ({steps} local steps)", flush=True)
                for rule, sim_rule in (
                    ("topfed", "lower_median"),
                    ("median", "median"),
                    ("trimmed_mean", "trimmed_mean"),
                ):
                    tgt, mask, tinfo = _peer_target(rule, aggs[rule], updates, deltas, attacker)
                    sim = AwareSimulator(sim_rule, own, groups, step_size=step_size)
                    targets.append(
                        ("peer_aware", rule, tgt, None, peer_pool, attack_labels(tgt, peer_labels, tinfo), tinfo, sim, None)
                    )
                # Oracle: the rule-aware attacker is also handed the rule's true
                # capped weights, which a real peer never sees.
                tgt, mask, tinfo = _peer_target("topfed", aggs["topfed"], updates, deltas, attacker)
                true_w = np.asarray(info["topfed_weights"], dtype=np.float32)
                oracle_w = torch.from_numpy(np.concatenate([true_w[peers], true_w[[attacker]]]))
                sim = AwareSimulator("weighted_median", own, groups, step_size=step_size, weights=oracle_w)
                targets.append(
                    ("peer_oracle", "topfed", tgt, None, peer_pool, attack_labels(tgt, peer_labels, tinfo), tinfo, sim, None)
                )
                # Estimated weights: the rule-aware attacker is not told the
                # capped weights but computes them itself. It starts from
                # equal weights and, every ``REWEIGHT_EVERY`` iterations,
                # scores the round as it currently simulates it (its dummy
                # peers plus its own update), measuring the validation loss
                # on images of its own shard that it does not train on in
                # this round, never on the server's validation set.
                own_shard = partition[int(selected[attacker][1:])]
                proxy_idx = torch.as_tensor(
                    np.asarray(own_shard[b : b + PROXY_VAL_MAX], dtype=np.int64)
                )
                if "peer_estimated" in observers and len(proxy_idx) > 0:
                    proxy_loader = DataLoader(
                        TensorDataset(bundle.train_x[proxy_idx], bundle.train_y[proxy_idx]),
                        batch_size=256,
                        shuffle=False,
                    )
                    tgt, mask, tinfo = _peer_target("topfed", aggs["topfed"], updates, deltas, attacker)
                    tinfo["proxy_val_size"] = int(len(proxy_idx))
                    sim = AwareSimulator(
                        "weighted_median", own, groups, step_size=step_size, weights=torch.ones(len(selected))
                    )
                    estimator = partial(
                        _estimate_topfed_weights,
                        own_update=updates[attacker],
                        global_parameters=global_parameters,
                        images_per_peer=b,
                        round_num=round_num,
                        val_loader=proxy_loader,
                        model_factory=model_factory,
                        device=args.device,
                    )
                    targets.append(
                        ("peer_estimated", "topfed", tgt, None, peer_pool, attack_labels(tgt, peer_labels, tinfo), tinfo, sim, estimator)
                    )
                # Oracle for the trust-weighted means: the attacker removes its
                # own term under the true weights and weights each peer's
                # simulated delta by that peer's true weight. A peer the rule
                # gave zero weight is absent from the target, so its dummy
                # images receive no signal.
                for rule in TRUST_MEAN_RULES:
                    true_w = np.asarray(info[f"{rule}_weights"], dtype=np.float32)
                    tgt, mask, tinfo = _peer_target(
                        rule, aggs[rule], updates, deltas, attacker, weights=true_w
                    )
                    peer_w = torch.from_numpy(np.concatenate([true_w[peers], np.zeros(1, np.float32)]))
                    sim = AwareSimulator("weighted_mean", own, groups, step_size=step_size, weights=peer_w)
                    targets.append(
                        ("peer_oracle", rule, tgt, mask, peer_pool, attack_labels(tgt, peer_labels, tinfo), tinfo, sim, None)
                    )
                # Krum-aware: the aggregate is one peer's update verbatim, so
                # the attacker inverts it as a single client of ``b`` images,
                # recovering the labels from the output-bias delta (iDLG).
                tgt, mask, tinfo = _peer_target("krum", aggs["krum"], updates, deltas, attacker)
                if mask is None:
                    bias = tgt[-num_classes:]
                    lbl = torch.as_tensor(np.argsort(-bias)[:b], dtype=torch.long, device=device)
                    targets.append(("peer_aware", "krum", tgt, None, peer_pool, lbl, tinfo, None, None))

            targets = [
                t for t in targets if t[0] in observers and (t[0] == "server" or t[1] in rules)
            ]

            fig_rows: list[tuple[str, torch.Tensor | None, np.ndarray | None, str]] = []
            pool01_all = _denorm(peer_pool, mean, std)
            for observer, rule, tgt, mask, pool, lbl, tinfo, simulator, reweight in targets:
                row: dict[str, Any] = {
                    "dataset": args.dataset,
                    "seed": args.seed,
                    "state": state_name,
                    "protocol": proto.name,
                    "observer": observer,
                    "rule": rule,
                    "labels": args.labels,
                    "n_recon": int(len(lbl)),
                    "n_pool": int(pool.shape[0]),
                    **tinfo,
                }
                if simulator is not None and steps != 1:
                    row["surrogate_step_size"] = simulator.step_size
                    row["local_steps"] = steps
                included: np.ndarray | None = None
                if rule in TRUST_MEAN_RULES:
                    # Which peers the rule kept, so a reader can separate what
                    # leaked about them from the peers that never entered the
                    # aggregate.
                    rule_w = np.asarray(info[f"{rule}_weights"])
                    included = rule_w[peers] > 0.0
                    row["rule_weights"] = [float(v) for v in rule_w]
                    row["n_included_peers"] = int(included.sum())
                if mask is not None and not mask.any():
                    row["skipped"] = "aggregate carries only the attacker's own update"
                    fig_rows.append((rule, None, None, "attacker selected"))
                    ledger.append(row)
                    print(
                        f"  {observer:10s} {rule:13s} skipped (only the attacker is in the aggregate)",
                        flush=True,
                    )
                    continue
                t1 = time.perf_counter()
                res = invert(
                    probe,
                    tgt,
                    mask,
                    lbl,
                    bounds=(lo, hi),
                    iterations=args.iterations,
                    restarts=args.restarts,
                    tv_weight=args.tv,
                    lr=args.lr,
                    seed=args.seed,
                    device=device,
                    simulator=simulator,
                    reweight=reweight,
                )
                if res.weights is not None:
                    # Cohort order, like the true weights next to them.
                    est = np.empty(len(selected))
                    est[peers], est[attacker] = res.weights[:-1], res.weights[-1]
                    est = est / est.sum()
                    true = np.asarray(info["topfed_weights"])
                    row["estimated_weights"] = [float(v) for v in est]
                    row["true_weights"] = [float(v) for v in true]
                    row["weight_l1_error"] = float(np.abs(est - true).sum())
                recon01 = _denorm(res.images, mean, std)
                pool01 = _denorm(pool, mean, std)
                summary, match = _score(recon01, pool01, one_to_one=not whole_shard)
                row.update(summary)
                if included is not None and not whole_shard and included.any():
                    # SSIM over the true images of the included peers only
                    # (the pool lists each peer's images in cohort order).
                    of_included = np.repeat(included, int(proto.images or 0))[match]
                    per_image = _ssim(recon01, pool01[torch.from_numpy(match).to(pool01.device)])
                    row["ssim_included_mean"] = float(per_image[of_included].mean())
                row.update(
                    {
                        "rec_loss": res.rec_loss,
                        "masked_fraction": res.masked_fraction,
                        "seconds": time.perf_counter() - t1,
                    }
                )
                ledger.append(row)
                print(
                    f"  {observer:10s} {rule:13s} psnr {summary['psnr_mean']:5.2f}"
                    f" (max {summary['psnr_max']:5.2f})  ssim {summary['ssim_mean']:.3f}"
                    f"  rec_loss {res.rec_loss:.4f}  masked {res.masked_fraction:.2f}"
                    f"  {row['seconds']:.0f}s",
                    flush=True,
                )
                if observer == "server":
                    label = "server view\n(one client)"
                elif observer == "peer_aware":
                    label = f"{rule}\n(median-aware)"
                elif observer == "peer_estimated":
                    label = f"{rule}\n(aware + estimated weights)"
                elif observer == "peer_oracle":
                    label = f"{rule}\n(aware + true weights)"
                else:
                    label = rule
                fig_rows.append((label, recon01, match, f"psnr {summary['psnr_mean']:.1f}"))

            if "calibration" in observers:
                # Calibration rows: what a non-reconstruction scores against this pool.
                gray01 = _denorm(torch.zeros_like(peer_pool[:1]), mean, std)
                g_summary, _ = _score(gray01, pool01_all, one_to_one=False)
                rng = np.random.default_rng(args.seed)
                in_pool = {int(i) for cid in selected for i in idx_of[cid]}
                candidates = np.array([i for i in range(len(bundle.train_y)) if i not in in_pool])
                others = rng.choice(candidates, size=8, replace=False)
                rand01 = _denorm(bundle.train_x[torch.as_tensor(others)].to(device), mean, std)
                r_summary, _ = _score(rand01, pool01_all, one_to_one=False)
                calib = {"dataset": args.dataset, "seed": args.seed, "state": state_name, "protocol": proto.name, "observer": "calibration", "labels": args.labels}
                ledger.append({**calib, "rule": "flat_gray", **g_summary})
                ledger.append({**calib, "rule": "random_other_images", **r_summary})
                print(
                    f"  calib  flat gray     psnr {g_summary['psnr_mean']:5.2f}"
                    f"   random other images psnr {r_summary['psnr_mean']:5.2f} (nearest in pool)",
                    flush=True,
                )

            _save_figure(
                out / f"recon_{state_name}_{proto.name}.png",
                f"{args.dataset}, {state_name} model, {proto.description}",
                pool01_all,
                fig_rows,
                max_cols=18,
            )
            (out / "ledger.json").write_text(json.dumps(ledger, indent=1))
    print(f"\n[done] {len(ledger)} rows in {time.perf_counter() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
