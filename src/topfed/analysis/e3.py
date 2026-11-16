"""E3 systematic FLTrust comparison: attack success, clean accuracy, and cost.

Reads the per-run ledgers under ``results/`` (``manifest.json`` and
``metrics.json``). Only runs in the reference setting are in scope (uniform
sampling, K=10, N=100, the dataset's native partition, corner trigger, no
attack window); a run's cell is keyed by its resolved config, so frozen cells
and tagged new cells reached by any override spelling land in the same table.

Sections of the report:

* **Attack by defense** -- one panel per (dataset, f), for CIFAR-10, FEMNIST
  and, once they have runs, the two CIFAR-100 label variants (100 fine
  classes; 20 superclasses): rows are attacks,
  columns are {FLTrust, capped-wmed, median, undefended, ScoreWeight} x
  {ASR or delta-accuracy, clean accuracy}. Delta-accuracy is clean accuracy
  minus the benign (no attack, no defense) accuracy. The last row is the
  server overhead from the timing runs. For the two attackers with a tuning
  knob (the norm-bounded adaptive backdoor's budget and the direction-aligned
  backdoor's cosine target) each defense faces its own best response: the
  knob value with the highest mean ASR among values with at least
  ``MIN_SEEDS`` seeds, provided it beats the attack config's value on the
  shared seeds (BCa interval above zero). A dagger marks a cell where another
  value won.
* **SOTA baselines** -- the same panel for the published defenses added
  for the comparison (FLAME, MESAS, BackdoorIndicator, FoolsGold, Krum,
  trimmed mean) next to capped-wmed and FLTrust.
* **Detection** -- per cell, from the defense ledger: the share of
  malicious client-rounds given zero aggregation weight (TPR), the share of
  benign client-rounds given zero weight (FPR), and the mean aggregation
  weight share held by malicious clients in rounds that had any.
* **Best-response knob** -- mean ASR per knob value behind that choice.
* **Paired comparisons** -- seed-matched differences on the primary metric
  (ASR for targeted attacks, clean accuracy for untargeted ones): two-sided
  Wilcoxon signed-rank, rank-biserial effect size, BCa 95% interval of the
  mean difference, Holm correction within each (dataset, f, attack) row.
* **Root-set sensitivity** -- FLTrust and capped-wmed at 1, 10 and 100
  validation samples per class (the set is FLTrust's root set and the
  validation-loss metric's reference), with paired differences against 100.
* **Server overhead** -- per-round client training, scoring, aggregation and
  server training seconds from the 50-round timing runs (rounds after a
  warm-up), peak server memory from the profiling runs, and communication
  per round computed from the model's state-dict size, identical for every
  scheme.

Runs made with another backbone (an optional ``model`` block in the config,
such as the ResNet-18 with GroupNorm) or another local learning rate form
their own cells: they never merge with the small-CNN cells above, and the
report tables show the small CNN at the default rate only. ``--model NAME
[--lr RATE]`` renders the appendix table for such a block instead: ASR and
clean accuracy per defense for the vanilla and norm-bounded adaptive
backdoors at f = 0.2, each defense at its best-response budget.

Read-only over ``results/``; every output path is rejected inside the results
tree.

Usage::

    uv run python -m topfed.analysis.e3 --out tables/e3-tables.md
    uv run python -m topfed.analysis.e3 --json e3-runs.json
    uv run python -m topfed.analysis.e3 --model resnet18_gn --lr 0.05
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar

import numpy as np

from topfed.analysis.e5 import (
    REFERENCE,
    REFERENCE_COMMON,
    _defense_label,
    _fmt,
    _mean_sd,
    _partition_label,
    _table,
    _window,
)
from topfed.analysis.stats import bca_ci, holm_bonferroni, wilcoxon_vs_baseline

NUM_CLIENTS = 100
CLIENTS_PER_ROUND = 10
NUM_ROUNDS = 200
OVERHEAD_ROUNDS = 50
CONVERGENCE_WINDOW = 50
# Round 1 carries one-off device initialisation; the first rounds are dropped
# from the per-round timing means.
WARMUP_ROUNDS = 5
MIN_SEEDS = 5
MIN_SHARED_SEEDS = 3
FRACTIONS = (0.10, 0.20)
# Datasets in report order, their display names and class counts, and the
# reference size of the server validation slice (per class). CIFAR-100 comes
# in two label variants that are separate datasets to the tables: the 100
# fine classes and the 20 superclasses.
DATASETS = ("cifar10", "femnist", "cifar100", "cifar100_coarse")
# Rendered whether or not they have runs; the others only when they do.
CORE_DATASETS = ("cifar10", "femnist")
DATASET_NAMES = {
    "cifar10": "CIFAR-10",
    "femnist": "FEMNIST",
    "cifar100": "CIFAR-100 (100 fine classes)",
    "cifar100_coarse": "CIFAR-100 (20 superclasses)",
}
DATASET_NUM_CLASSES = {"cifar10": 10, "femnist": 62, "cifar100": 100, "cifar100_coarse": 20}
REFERENCE_VAL = {"cifar10": 100, "femnist": 30, "cifar100": 10, "cifar100_coarse": 50}
ROOT_VALS = (1, 10, 100)
ROOT_ATTACKS = ("backdoor", "adaptive")
ROOT_DEFENSES = ("fltrust", "capped-wmed")
ROOT_FRACTION = 0.10
TABLE_DEFENSES = ("fltrust", "capped-wmed", "median", "undefended", "ScoreWeight")
# The label of a run without a ``model`` block, and the default local rate.
DEFAULT_MODEL = "cnn"
DEFAULT_LOCAL_LR = 0.01
DEFAULT_GROUPS = 2
MODEL_TABLE_DEFENSES = ("undefended", "capped-wmed", "fltrust-median", "flame", "median")
MODEL_TABLE_ATTACKS = ("backdoor", "adaptive")
MODEL_TABLE_FRACTION = 0.20
SOTA_DEFENSES = ("flame", "mesas", "indicator", "foolsgold", "krum", "trimmed_mean", "clipped-mean")
SOTA_PANEL = ("capped-wmed", "fltrust") + SOTA_DEFENSES
# The two instantiations of the bounded-weight median next to their parents.
ROOT_PANEL = ("capped-wmed", "fltrust", "fltrust-median", "fltrust-median-capped")
DETECTION_DEFENSES = ("capped-wmed", "fltrust", "median") + SOTA_DEFENSES
OVERHEAD_DEFENSES = (
    "undefended", "ScoreWeight", "capped-wmed", "median", "trimmed_mean", "krum", "fltrust",
)
TARGETED = ("backdoor", "adaptive", "direction_aligned", "clean_label")
UNTARGETED = ("gradient_scaling", "sign_flip", "dirty_label")
ATTACKS = {
    "cifar10": ("backdoor", "adaptive", "direction_aligned", "clean_label",
                "gradient_scaling", "sign_flip", "dirty_label"),
    "femnist": ("backdoor", "adaptive", "gradient_scaling", "dirty_label"),
    "cifar100": ("backdoor", "adaptive"),
    "cifar100_coarse": ("backdoor", "adaptive"),
}
# The attack config's value of each attacker's strength parameter.
REFERENCE_STRENGTH: dict[str, float | None] = {
    "backdoor": None,
    "adaptive": 0.6,
    "direction_aligned": 0.9,
    "clean_label": None,
    "gradient_scaling": 5.0,
    "sign_flip": 1.0,
    "dirty_label": 1.0,
}
KNOB_ATTACKS = {"adaptive": "budget", "direction_aligned": "cosine"}
# Sweep spellings of the knob attackers and the defenses, for the lines the
# ``--knob-bumps`` listing prints (consumed by experiments/sweep_baselines.sh).
KNOB_EXPERIMENTS = {"adaptive": "adaptive_backdoor", "direction_aligned": "fltrust_backdoor"}
KNOB_OVERRIDES = {
    "adaptive": ("attack.update_manipulator.target_norm", "g"),
    "direction_aligned": ("attack.update_manipulator.cosine_target", "c"),
}
DEFENSE_OVERRIDES = {
    "undefended": "",
    "ScoreWeight": "+defense=topsis_score_weight",
    "wmed": "+defense=topsis_weighted_median",
    "capped-wmed": "+defense=topsis_capped_weighted_median",
    "median": "+baseline=coordinate_median",
    "fltrust": "+baseline=fltrust",
    "fltrust-median": "+baseline=fltrust_median",
    "fltrust-median-capped": "+baseline=fltrust_median_capped",
    "flame": "+baseline=flame",
    "mesas": "+baseline=mesas",
    "indicator": "+baseline=backdoor_indicator",
    "foolsgold": "+baseline=foolsgold",
    "krum": "+baseline=krum",
    "trimmed_mean": "+baseline=trimmed_mean",
    "clipped-mean": "+baseline=clipped_mean",
}
ATTACK_NAMES = {
    "backdoor": "vanilla backdoor",
    "adaptive": "norm-bounded adaptive",
    "direction_aligned": "direction-aligned",
    "clean_label": "clean-label backdoor",
    "gradient_scaling": "gradient scaling",
    "sign_flip": "sign flip",
    "dirty_label": "dirty label",
}
COMPARISONS = (
    ("capped-wmed", "fltrust"),
    ("capped-wmed", "median"),
    ("fltrust", "median"),
    ("capped-wmed", "undefended"),
    ("fltrust", "undefended"),
) + tuple(("capped-wmed", d) for d in SOTA_DEFENSES) + (
    ("fltrust-median", "capped-wmed"),
    ("fltrust-median-capped", "fltrust-median"),
    ("fltrust-median-capped", "capped-wmed"),
)
TIMING_KEYS = ("client_train_s", "scoring_s", "aggregation_s", "server_train_s", "round_s")
SERVER_KEYS = ("scoring_s", "aggregation_s", "server_train_s")
MEMORY_KEYS = ("server_peak_host_mb", "server_peak_device_mb")
# With S seed pairs all of one sign, the exact two-sided signed-rank p-value
# is 2 / 2**S, which only drops below 0.05 from S = 6.
MIN_PAIRS_FOR_SIGNIFICANCE = 6


@dataclass(frozen=True)
class CellKey:
    """One E3 cell: dataset, attacker and its strength, f, defense, and setting.

    ``strength`` is the attacker's tuning parameter (norm budget, cosine
    target, scaling factor, flip magnitude, poisoning rate); ``None`` for the
    scaled and clean-label backdoors and the benign run. ``model`` is the
    backbone label (``cnn`` without a ``model`` block) and ``local_lr`` the
    clients' learning rate, so runs with another backbone or rate never
    share a cell with the reference ones.
    """

    dataset: str
    attack: str
    strength: float | None
    fraction: float
    defense: str
    val_per_class: int
    num_rounds: int
    profile_memory: bool
    model: str = DEFAULT_MODEL
    local_lr: float = DEFAULT_LOCAL_LR


@dataclass
class RunSummary:
    """Per-seed quantities the E3 tables are built from."""

    seed: int
    asr_final: float | None
    acc_final: float
    timing_rounds: dict[str, list[float]] = field(default_factory=dict)
    memory_peak: dict[str, float] = field(default_factory=dict)
    # From the defense ledger, when the run wrote one: share of malicious
    # (benign) client-rounds that received zero aggregation weight, and the
    # mean weight share held by malicious clients over rounds with any.
    drop_tpr: float | None = None
    drop_fpr: float | None = None
    mal_share: float | None = None
    source: str = ""


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------


def _target_name(block: dict[str, Any] | None) -> str | None:
    if not block:
        return None
    return str(block.get("_target_", "")).rsplit(".", 1)[-1]


def _model_label(cfg: dict[str, Any]) -> str:
    """Backbone label: ``cnn`` without a ``model`` block, else its name.

    A non-default group count is appended, so ``resnet18_gn(groups=4)`` is
    its own cell.
    """
    block = cfg.get("model")
    if not block:
        return DEFAULT_MODEL
    name = str(block.get("name", "unknown"))
    groups = block.get("groups")
    if groups is not None and int(groups) != DEFAULT_GROUPS:
        return f"{name}(groups={int(groups)})"
    return name


def _is_reference_setting(cfg: dict[str, Any]) -> bool:
    """True when sampling, partition and schedule are canonical."""
    fed = cfg["federation"]
    dataset = str(cfg["dataset"]["name"])
    if dataset not in REFERENCE:
        return False
    return (
        int(fed["num_clients"]) == NUM_CLIENTS
        and int(fed["clients_per_round"]) == CLIENTS_PER_ROUND
        and fed.get("force_malicious_per_round") is None
        and _partition_label(cfg) == REFERENCE[dataset]["partition"]
        and int(cfg["dataset"].get("pad_border", 0) or 0) == REFERENCE_COMMON["pad_border"]
        and _window(cfg) == REFERENCE_COMMON["window"]
    )


def _reference_trigger(cfg: dict[str, Any], poisoner: dict[str, Any]) -> bool:
    ref = REFERENCE[str(cfg["dataset"]["name"])]
    return (
        str(poisoner.get("trigger_position", "bottom_right")) == REFERENCE_COMMON["trigger_position"]
        and int(poisoner["trigger_size"]) == REFERENCE_COMMON["trigger_size"]
        and float(poisoner["trigger_amplitude"]) == ref["trigger_amplitude"]
    )


def _attack(cfg: dict[str, Any]) -> tuple[str, float | None] | None:
    """Name the attacker and its strength; ``None`` for a diagnostic variant."""
    attack = cfg["attack"]
    kind = attack.get("kind")
    poisoner = attack.get("poisoner") or {}
    manipulator = attack.get("update_manipulator") or {}
    target = _target_name(manipulator)
    if kind == "none":
        return ("benign", None)
    if kind == "backdoor":
        if not poisoner or not _reference_trigger(cfg, poisoner):
            return None
        if str(poisoner.get("batch_mode", "replace")) != REFERENCE_COMMON["batch_mode"]:
            return None
        if target == "ModelReplacementManipulator" and manipulator.get("gamma") is None:
            return ("backdoor", None)
        if target == "NormBoundedManipulator":
            return ("adaptive", float(manipulator["target_norm"]))
        if target == "DirectionAlignedManipulator":
            return ("direction_aligned", float(manipulator["cosine_target"]))
        return None
    if kind == "clean_label":
        if not poisoner or not _reference_trigger(cfg, poisoner):
            return None
        return ("clean_label", None)
    if kind == "gradient_scaling" and target == "GradientScalingManipulator":
        return ("gradient_scaling", float(manipulator["factor"]))
    if kind == "sign_flip" and target == "SignFlippingManipulator":
        return ("sign_flip", float(manipulator["eta"]))
    if kind == "dirty_label" and poisoner:
        return ("dirty_label", float(poisoner["poisoning_rate"]))
    return None


def cell_key(manifest: dict[str, Any]) -> CellKey | None:
    """Build the E3 cell key from a manifest; ``None`` for out-of-scope runs."""
    cfg = manifest["config"]
    if not _is_reference_setting(cfg):
        return None
    found = _attack(cfg)
    if found is None:
        return None
    name, strength = found
    fed = cfg["federation"]
    return CellKey(
        dataset=str(cfg["dataset"]["name"]),
        attack=name,
        strength=strength,
        fraction=round(float(cfg["attack"]["assignment"]["fraction"]), 4),
        defense=_defense_label(cfg),
        val_per_class=int(cfg["dataset"]["val_per_class"]),
        num_rounds=int(fed["num_rounds"]),
        profile_memory=bool(fed.get("profile_memory", False)),
        model=_model_label(cfg),
        local_lr=float(fed.get("local_learning_rate", DEFAULT_LOCAL_LR)),
    )


def detection_rates(
    ledger: list[dict[str, Any]] | None,
) -> tuple[float | None, float | None, float | None]:
    """``(TPR, FPR, malicious weight share)`` from a defense ledger.

    A client-round is "dropped" when its ``weight`` is zero. TPR is dropped
    malicious over all malicious client-rounds, FPR dropped benign over all
    benign client-rounds; each is ``None`` when its denominator is empty. The
    share is the malicious fraction of the round's total weight, averaged
    over rounds with at least one malicious client (``None`` when there are
    none or when no round carried weight). Ledgers without a ``weight``
    column give ``(None, None, None)``.
    """
    if not ledger or "weight" not in ledger[0]:
        return None, None, None
    mal = [r for r in ledger if r["is_malicious"]]
    ben = [r for r in ledger if not r["is_malicious"]]
    tpr = sum(float(r["weight"]) == 0.0 for r in mal) / len(mal) if mal else None
    fpr = sum(float(r["weight"]) == 0.0 for r in ben) / len(ben) if ben else None
    num: dict[int, float] = defaultdict(float)
    den: dict[int, float] = defaultdict(float)
    has_mal: set[int] = set()
    for r in ledger:
        rnd = int(r["round"])
        den[rnd] += float(r["weight"])
        if r["is_malicious"]:
            num[rnd] += float(r["weight"])
            has_mal.add(rnd)
    shares = [num[r] / den[r] for r in sorted(has_mal) if den[r] > 0]
    share = float(np.mean(shares)) if shares else None
    return tpr, fpr, share


def summarise_run(
    seed: int, metrics: list[dict[str, Any]], ledger: list[dict[str, Any]] | None = None
) -> RunSummary:
    acc = np.array([float(m.get("test_accuracy", math.nan)) for m in metrics])
    has_asr = bool(metrics) and "asr_accuracy" in metrics[-1]
    asr = np.array([float(m.get("asr_accuracy", math.nan)) for m in metrics])
    timed = [m for m in metrics if int(m["round"]) > WARMUP_ROUNDS] or metrics
    timing_rounds = {
        key: [float(m[key]) for m in timed]
        for key in TIMING_KEYS
        if timed and all(key in m for m in timed)
    }
    memory_peak = {
        key: max(float(m[key]) for m in metrics)
        for key in MEMORY_KEYS
        if metrics and all(key in m for m in metrics)
    }
    tpr, fpr, share = detection_rates(ledger)
    return RunSummary(
        seed=seed,
        asr_final=float(np.nanmean(asr[-CONVERGENCE_WINDOW:])) if has_asr else None,
        acc_final=float(np.nanmean(acc[-CONVERGENCE_WINDOW:])),
        timing_rounds=timing_rounds,
        memory_peak=memory_peak,
        drop_tpr=tpr,
        drop_fpr=fpr,
        mal_share=share,
    )


class _Keyed(Protocol):
    @property
    def num_rounds(self) -> int: ...


K = TypeVar("K", bound=_Keyed)


def load_cells(results_root: Path) -> dict[CellKey, list[RunSummary]]:
    """Walk ``results/`` and group completed in-scope runs by E3 cell.

    See :func:`load_cells_by` for how duplicates and incomplete runs are
    handled.
    """
    return load_cells_by(results_root, cell_key)


def load_cells_by(
    results_root: Path, key_fn: Callable[[dict[str, Any]], K | None]
) -> dict[K, list[RunSummary]]:
    """Walk ``results/`` and group completed runs by ``key_fn(manifest)``.

    Runs for which ``key_fn`` returns ``None`` are out of scope. The same cell
    can sit under more than one ``<label>/<hash>`` directory (an identical
    config under another experiment name or tag). Each seed is then taken
    from the directory with the most completed seeds, ties by path, and the
    others are reported on stderr. A run is complete when ``metrics.json``
    holds every configured round.
    """
    candidates: dict[tuple[K, int], list[Path]] = defaultdict(list)
    completed: dict[Path, int] = defaultdict(int)
    for manifest_path in sorted(results_root.glob("*/*/seed_*/manifest.json")):
        run_dir = manifest_path.parent
        if not (run_dir / "metrics.json").exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        key = key_fn(manifest)
        if key is None:
            continue
        completed[run_dir.parent] += 1
        candidates[(key, int(manifest["seed"]))].append(run_dir)

    cells: dict[K, list[RunSummary]] = defaultdict(list)
    ignored: dict[tuple[str, str], int] = defaultdict(int)
    for (key, seed), run_dirs in candidates.items():
        ranked = sorted(run_dirs, key=lambda d: (-completed[d.parent], str(d)))
        for loser in ranked[1:]:
            ignored[
                (str(ranked[0].parent.relative_to(results_root)),
                 str(loser.parent.relative_to(results_root)))
            ] += 1
        run_dir = ranked[0]
        metrics = json.loads((run_dir / "metrics.json").read_text())
        if len(metrics) != key.num_rounds:
            continue
        ledger_path = run_dir / "defense_ledger.json"
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else None
        summary = summarise_run(seed, metrics, ledger)
        summary.source = str(run_dir.relative_to(results_root))
        cells[key].append(summary)
    for (kept, dropped), n in sorted(ignored.items()):
        print(f"note: same config in {dropped}; {n} seed(s) taken from {kept}", file=sys.stderr)
    return {k: sorted(v, key=lambda r: r.seed) for k, v in cells.items()}


# ----------------------------------------------------------------------------
# Cell selection
# ----------------------------------------------------------------------------


def primary_metric(attack: str) -> str:
    """ASR for a targeted attack, clean accuracy for an untargeted one."""
    return "asr_final" if attack in TARGETED else "acc_final"


def _values(runs: list[RunSummary], attr: str) -> list[float]:
    out = []
    for r in runs:
        v = getattr(r, attr)
        if v is not None and not math.isnan(v):
            out.append(float(v))
    return out


def _mean(runs: list[RunSummary], attr: str) -> float | None:
    xs = _values(runs, attr)
    return float(np.mean(xs)) if xs else None


@dataclass(frozen=True)
class Selection:
    """The runs standing for one (dataset, attack, f, defense) table cell.

    ``by_strength`` lists every knob value found with its mean ASR and seed
    count, so the best-response choice can be shown.
    """

    strength: float | None
    runs: list[RunSummary]
    best_response: bool
    by_strength: dict[float | None, tuple[float | None, int]]


def _candidates(
    cells: dict[CellKey, list[RunSummary]],
    dataset: str,
    attack: str,
    fraction: float,
    defense: str,
    val_per_class: int,
    model: str = DEFAULT_MODEL,
    local_lr: float = DEFAULT_LOCAL_LR,
) -> dict[float | None, list[RunSummary]]:
    return {
        k.strength: runs
        for k, runs in cells.items()
        if k.dataset == dataset
        and k.attack == attack
        and abs(k.fraction - fraction) < 1e-9
        and k.defense == defense
        and k.val_per_class == val_per_class
        and k.num_rounds == NUM_ROUNDS
        and not k.profile_memory
        and k.model == model
        and abs(k.local_lr - local_lr) < 1e-12
    }


def _raises_asr(candidate: list[RunSummary], reference: list[RunSummary]) -> bool:
    """True when ``candidate`` raises ASR over ``reference`` on shared seeds.

    Needs at least ``MIN_SHARED_SEEDS`` seed-matched pairs and a BCa 95%
    interval of the mean difference that lies above zero.
    """
    ref = {r.seed: r.asr_final for r in reference if r.asr_final is not None}
    diffs = [
        float(r.asr_final) - float(ref[r.seed])
        for r in candidate
        if r.asr_final is not None and r.seed in ref
    ]
    if len(diffs) < MIN_SHARED_SEEDS:
        return False
    return bca_ci(diffs)[0] > 0.0


def select_cell(
    cells: dict[CellKey, list[RunSummary]],
    dataset: str,
    attack: str,
    fraction: float,
    defense: str,
    *,
    val_per_class: int | None = None,
    best_response: bool = True,
    model: str = DEFAULT_MODEL,
    local_lr: float = DEFAULT_LOCAL_LR,
) -> Selection | None:
    """Pick the runs for one table cell.

    Attackers without a knob use the attack config's strength. For a knob
    attacker with ``best_response``, another knob value replaces the attack
    config's value only when it has at least ``MIN_SEEDS`` seeds and at
    least as many seeds as the attack config's cell, and raises ASR on the
    shared seeds with a BCa 95% interval above zero; among several such
    values the highest mean ASR wins. A table entry therefore never rests on
    fewer seeds than the reference cell it displaces; a smaller rival that
    would win is listed by :func:`knob_bumps` for a seed upgrade. When the
    attack config's value has fewer than ``MIN_SEEDS`` runs (no runs, or only
    a probe seed), it is treated as absent and the highest mean ASR among the
    values with ``MIN_SEEDS`` seeds is used. Without ``best_response`` the attack
    config's value is used. ``model`` and ``local_lr`` select the backbone
    and client learning rate; the defaults are the small CNN at its rate.
    """
    val = REFERENCE_VAL[dataset] if val_per_class is None else val_per_class
    cands = _candidates(cells, dataset, attack, fraction, defense, val, model, local_lr)
    ref = REFERENCE_STRENGTH.get(attack)
    by_strength = {s: (_mean(r, "asr_final"), len(r)) for s, r in cands.items()}
    if attack not in KNOB_ATTACKS or not best_response:
        runs = cands.get(ref)
        return Selection(ref, runs, False, by_strength) if runs else None
    if not cands:
        return None
    eligible = {s: r for s, r in cands.items() if len(r) >= MIN_SEEDS} or cands

    def mean_asr(s: float | None) -> float:
        m = _mean(eligible[s], "asr_final")
        return -math.inf if m is None else m

    if ref in cands and len(cands[ref]) >= MIN_SEEDS:
        rivals = [
            s for s in eligible
            if s != ref
            and len(eligible[s]) >= len(cands[ref])
            and _raises_asr(eligible[s], cands[ref])
        ]
        if not rivals:
            return Selection(ref, cands[ref], False, by_strength)
        choice = max(rivals, key=mean_asr)
    else:
        choice = max(eligible, key=mean_asr)
    return Selection(choice, eligible[choice], choice != ref, by_strength)


@dataclass(frozen=True)
class KnobBump:
    """A knob value that would displace the reference once it has its seeds."""

    dataset: str
    attack: str
    fraction: float
    defense: str
    strength: float
    missing_seeds: tuple[int, ...]

    def sweep_line(self) -> str:
        """``tag<TAB>experiment<TAB>seeds<TAB>override[<TAB>override]`` for the sweep."""
        f = int(round(self.fraction * 100))
        exp = f"{KNOB_EXPERIMENTS[self.attack]}_{self.dataset}_f{f:02d}"
        key, letter = KNOB_OVERRIDES[self.attack]
        value = str(float(self.strength))  # the sweeps spell budgets as 1.0, 1.5, 0.5
        tag = f"sp_{exp}_{letter}{value}"
        parts = [tag, exp, ",".join(str(x) for x in self.missing_seeds), f"{key}={value}"]
        defense = DEFENSE_OVERRIDES[self.defense]
        if defense:
            parts.append(defense)
        return "\t".join(parts)


def knob_bumps(
    cells: dict[CellKey, list[RunSummary]],
    dataset: str,
    defenses: tuple[str, ...],
    fractions: tuple[float, ...] = FRACTIONS,
) -> list[KnobBump]:
    """Knob values barred from a table only by their seed count.

    For each (attack, f, defense) with a reference-strength cell, every
    other value that raises ASR on the shared seeds but has fewer seeds than
    the reference is reported with the reference's seeds it lacks, so a
    sweep can bring it to the same count and let :func:`select_cell` judge
    it on equal footing.
    """
    out = []
    for attack in KNOB_ATTACKS:
        ref = REFERENCE_STRENGTH[attack]
        for f in fractions:
            for defense in defenses:
                cands = _candidates(cells, dataset, attack, f, defense, REFERENCE_VAL[dataset])
                reference = cands.get(ref)
                if not reference:
                    continue
                ref_seeds = {r.seed for r in reference}
                for strength, runs in cands.items():
                    if strength is None or strength == ref or len(runs) >= len(reference):
                        continue
                    if not _raises_asr(runs, reference):
                        continue
                    missing = tuple(sorted(ref_seeds - {r.seed for r in runs}))
                    if missing:
                        out.append(KnobBump(dataset, attack, f, defense, float(strength), missing))
    return out


def benign_accuracy(
    cells: dict[CellKey, list[RunSummary]],
    dataset: str,
    model: str = DEFAULT_MODEL,
    local_lr: float = DEFAULT_LOCAL_LR,
) -> tuple[float | None, int]:
    """Mean clean accuracy of the benign, undefended runs and their count."""
    runs = cells.get(
        CellKey(dataset, "benign", None, 0.0, "undefended", REFERENCE_VAL[dataset], NUM_ROUNDS, False,
                model, local_lr),
        [],
    )
    return _mean(runs, "acc_final"), len(runs)


# ----------------------------------------------------------------------------
# Paired statistics
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Comparison:
    """One seed-matched difference ``a - b`` on a cell's primary metric."""

    family: str
    label: str
    metric: str
    n: int
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    rank_biserial: float
    holm_p: float = math.nan
    reject: bool = False


def paired(family: str, label: str, metric: str,
           a: list[RunSummary], b: list[RunSummary]) -> Comparison | None:
    """Seed-matched comparison; ``None`` below two shared seeds."""
    a_by = {r.seed: getattr(r, metric) for r in a}
    b_by = {r.seed: getattr(r, metric) for r in b}
    seeds = sorted(s for s in set(a_by) & set(b_by) if a_by[s] is not None and b_by[s] is not None)
    if len(seeds) < 2:
        return None
    xa = [float(a_by[s]) for s in seeds]
    xb = [float(b_by[s]) for s in seeds]
    diff = np.array(xa) - np.array(xb)
    test = wilcoxon_vs_baseline(xa, xb)
    lo, hi = bca_ci(diff)
    return Comparison(family, label, metric, len(seeds), float(diff.mean()), lo, hi,
                      test.p_value, test.rank_biserial)


def holm(comparisons: list[Comparison]) -> list[Comparison]:
    """Holm-adjust p-values within each family."""
    out: list[Comparison] = []
    families: dict[str, list[Comparison]] = defaultdict(list)
    for c in comparisons:
        families[c.family].append(c)
    for members in families.values():
        adjusted, reject = holm_bonferroni([c.p_value for c in members])
        out.extend(
            Comparison(**{**asdict(c), "holm_p": p, "reject": r})
            for c, p, r in zip(members, adjusted, reject)
        )
    return out


def table_comparisons(cells: dict[CellKey, list[RunSummary]], dataset: str) -> list[Comparison]:
    comps = []
    for f in FRACTIONS:
        for attack in ATTACKS[dataset]:
            metric = primary_metric(attack)
            for a, b in COMPARISONS:
                sa = select_cell(cells, dataset, attack, f, a)
                sb = select_cell(cells, dataset, attack, f, b)
                if sa is None or sb is None:
                    continue
                c = paired(f"{dataset} f={f:g} {attack}", f"f={f:g} {ATTACK_NAMES[attack]}: {a} − {b}",
                           metric, sa.runs, sb.runs)
                if c is not None:
                    comps.append(c)
    return holm(comps)


def root_comparisons(cells: dict[CellKey, list[RunSummary]]) -> list[Comparison]:
    comps = []
    ref = REFERENCE_VAL["cifar10"]

    def runs(attack: str, defense: str, v: int) -> list[RunSummary]:
        s = select_cell(cells, "cifar10", attack, ROOT_FRACTION, defense,
                        val_per_class=v, best_response=False)
        return s.runs if s is not None else []

    for attack in ROOT_ATTACKS:
        for v in ROOT_VALS:
            if v == ref:
                continue
            for defense in ROOT_DEFENSES:
                c = paired(f"root {attack}", f"{ATTACK_NAMES[attack]} {defense}: {v}/class − {ref}/class",
                           "asr_final", runs(attack, defense, v), runs(attack, defense, ref))
                if c is not None:
                    comps.append(c)
        for v in ROOT_VALS:
            c = paired(f"root {attack}", f"{ATTACK_NAMES[attack]} {v}/class: capped-wmed − fltrust",
                       "asr_final", runs(attack, "capped-wmed", v), runs(attack, "fltrust", v))
            if c is not None:
                comps.append(c)
    return holm(comps)


# ----------------------------------------------------------------------------
# Overhead
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class OverheadRow:
    dataset: str
    defense: str
    rounds: int
    mean_s: dict[str, float]
    sd_s: dict[str, float]
    server_s: float | None
    memory_mb: dict[str, float]


def _overhead_runs(
    cells: dict[CellKey, list[RunSummary]], dataset: str, defense: str, profile_memory: bool
) -> list[RunSummary]:
    """Timing runs (``OVERHEAD_ROUNDS`` long) or memory-profiling runs of a cell."""
    out: list[RunSummary] = []
    for k, runs in cells.items():
        if (k.dataset == dataset and k.defense == defense and k.profile_memory == profile_memory
                and (profile_memory or k.num_rounds == OVERHEAD_ROUNDS)
                and k.val_per_class == REFERENCE_VAL[dataset]
                and k.model == DEFAULT_MODEL and k.local_lr == DEFAULT_LOCAL_LR):
            out.extend(runs)
    return out


def overhead_row(cells: dict[CellKey, list[RunSummary]], dataset: str, defense: str) -> OverheadRow:
    """Per-round timing from the timing runs, peak memory from the profiling runs.

    Rounds after the warm-up are pooled across every timing run of the cell.
    Server seconds are scoring plus aggregation plus server-side training.
    """
    timed = [r for r in _overhead_runs(cells, dataset, defense, False) if r.timing_rounds]
    pooled: dict[str, list[float]] = defaultdict(list)
    for r in timed:
        for key, vals in r.timing_rounds.items():
            pooled[key].extend(vals)
    mean_s = {k: float(np.mean(v)) for k, v in pooled.items() if v}
    sd_s = {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in pooled.items() if v}
    server = (
        float(sum(mean_s.get(k, 0.0) for k in SERVER_KEYS))
        if all(k in mean_s for k in SERVER_KEYS)
        else None
    )
    memory: dict[str, float] = {}
    for r in _overhead_runs(cells, dataset, defense, True):
        for key, v in r.memory_peak.items():
            memory[key] = max(memory.get(key, 0.0), v)
    return OverheadRow(dataset, defense, len(pooled.get("round_s", [])), mean_s, sd_s, server, memory)


def model_state_bytes(dataset: str) -> tuple[int, int]:
    """``(numel, bytes)`` of the model state dict a client uploads or downloads."""
    from topfed.models.cifar_cnn import CifarCNN
    from topfed.models.femnist_cnn import FemnistCNN

    model: Any = (
        FemnistCNN() if dataset == "femnist"
        else CifarCNN(num_classes=DATASET_NUM_CLASSES[dataset])
    )
    state = model.state_dict().values()
    return (sum(t.numel() for t in state),
            sum(t.numel() * t.element_size() for t in state))


def communication_per_round(dataset: str, k: int = CLIENTS_PER_ROUND) -> dict[str, float]:
    """Bytes moved per round: K state-dict uploads and K downloads.

    Every scheme here moves exactly this; FLTrust's root set stays at the
    server and adds no traffic.
    """
    numel, nbytes = model_state_bytes(dataset)
    return {
        "numel": float(numel),
        "state_bytes": float(nbytes),
        "upload_bytes": float(k * nbytes),
        "download_bytes": float(k * nbytes),
        "total_bytes": float(2 * k * nbytes),
    }


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------


def _cell_text(values: list[float], nd: int = 3, signed: bool = False) -> str:
    if not values:
        return "–"
    a = np.asarray(values, dtype=float)
    sd = float(a.std(ddof=1)) if a.size > 1 else 0.0
    mean = f"{a.mean():+.{nd}f}" if signed else f"{a.mean():.{nd}f}"
    return f"{mean} ± {sd:.{nd}f} ({a.size})"


def _knob_mark(sel: Selection) -> str:
    if not sel.best_response or sel.strength is None:
        return ""
    return f" †{sel.strength:g}"


def _mib(nbytes: float) -> str:
    return f"{nbytes / 2**20:.2f}"


def _server_cell(row: OverheadRow) -> str:
    if row.server_s is None:
        return "–"
    return f"{row.server_s:.3f} s"


def _memory_cell(row: OverheadRow) -> str:
    if not row.memory_mb:
        return "–"
    return " + ".join(f"{row.memory_mb[k]:.1f}" for k in MEMORY_KEYS if k in row.memory_mb) + " MiB"


def table_panel(
    cells: dict[CellKey, list[RunSummary]],
    dataset: str,
    fraction: float,
    defenses: tuple[str, ...] = TABLE_DEFENSES,
) -> str:
    benign, _ = benign_accuracy(cells, dataset)
    rows = []
    for attack in ATTACKS[dataset]:
        targeted = attack in TARGETED
        row = [ATTACK_NAMES[attack], "ASR" if targeted else "Δacc"]
        for defense in defenses:
            sel = select_cell(cells, dataset, attack, fraction, defense)
            if sel is None:
                row += ["–", "–"]
                continue
            acc = _values(sel.runs, "acc_final")
            if targeted:
                first = _cell_text(_values(sel.runs, "asr_final")) + _knob_mark(sel)
            elif benign is not None:
                first = _cell_text([a - benign for a in acc], signed=True)
            else:
                first = "–"
            row += [first, _cell_text(acc)]
        rows.append(row)
    overhead = [overhead_row(cells, dataset, d) for d in defenses]
    rows.append(["server overhead", "s/round, peak"]
                + [x for o in overhead for x in (_server_cell(o), _memory_cell(o))])
    header = ["attack", "metric"] + [
        x for d in defenses for x in (f"{d} ASR/Δacc", f"{d} clean acc")
    ]
    return _table(header, rows)


def table_model(
    cells: dict[CellKey, list[RunSummary]],
    model: str,
    local_lr: float = DEFAULT_LOCAL_LR,
    dataset: str = "cifar10",
    fraction: float = MODEL_TABLE_FRACTION,
    attacks: tuple[str, ...] = MODEL_TABLE_ATTACKS,
    defenses: tuple[str, ...] = MODEL_TABLE_DEFENSES,
) -> str:
    """ASR and clean accuracy per defense for one backbone's cells.

    Rows are the targeted ``attacks``; a last row gives the benign (no
    attack, no defense) clean accuracy of that backbone. Knob attackers take
    each defense's best response among the backbone's own cells, so a budget
    chosen by a probe is marked with a dagger when it differs from the attack
    config's value.
    """
    rows = []
    for attack in attacks:
        row = [ATTACK_NAMES[attack], "ASR"]
        for defense in defenses:
            sel = select_cell(cells, dataset, attack, fraction, defense,
                              model=model, local_lr=local_lr)
            if sel is None:
                row += ["–", "–"]
                continue
            row += [_cell_text(_values(sel.runs, "asr_final")) + _knob_mark(sel),
                    _cell_text(_values(sel.runs, "acc_final"))]
        rows.append(row)
    benign, n_benign = benign_accuracy(cells, dataset, model, local_lr)
    rows.append(["benign (no attack)", "clean acc", f"{_fmt(benign)} (n={n_benign})"]
                + ["–"] * (2 * len(defenses) - 1))
    header = ["attack", "metric"] + [
        x for d in defenses for x in (f"{d} ASR", f"{d} clean acc")
    ]
    return _table(header, rows)


def build_model_report(
    cells: dict[CellKey, list[RunSummary]], model: str, local_lr: float = DEFAULT_LOCAL_LR
) -> str:
    """The appendix section for one backbone: its table and how to read it."""
    in_scope = {k: v for k, v in cells.items()
                if k.model == model and abs(k.local_lr - local_lr) < 1e-12}
    n_runs = sum(len(v) for v in in_scope.values())
    # One panel per compromise level the backbone's attacked cells cover.
    fractions = sorted({k.fraction for k in in_scope if k.attack in MODEL_TABLE_ATTACKS})
    lines = [f"# {model} (local lr {local_lr:g}): attack by defense\n"]
    lines.append(
        f"Converged values average the last {CONVERGENCE_WINDOW} of {NUM_ROUNDS} rounds; "
        "cells read 'mean ± sd (n)'. Only this backbone's reference-setting runs are "
        "included; † marks a knob attacker whose best response against that defense "
        "is not the attack config's value (the number after † is the value used).\n"
    )
    lines.append(f"Runs found: {n_runs} in {len(in_scope)} cells.\n")
    for fraction in fractions or [MODEL_TABLE_FRACTION]:
        lines.append(f"## f = {fraction:g}\n")
        lines.append(table_model(cells, model, local_lr, fraction=fraction) + "\n")
    return "\n".join(lines)


def _rate_text(runs: list[RunSummary], attr: str) -> str:
    values = [getattr(r, attr) for r in runs if getattr(r, attr) is not None]
    return _cell_text(values) if values else "–"


def table_detection(
    cells: dict[CellKey, list[RunSummary]],
    dataset: str,
    fraction: float,
    defenses: tuple[str, ...] = DETECTION_DEFENSES,
) -> str:
    """Zero-weight rates and malicious weight share per (attack, defense)."""
    rows = []
    for attack in ATTACKS[dataset]:
        row = [ATTACK_NAMES[attack]]
        for defense in defenses:
            sel = select_cell(cells, dataset, attack, fraction, defense)
            if sel is None:
                row += ["–", "–", "–"]
                continue
            row += [
                _rate_text(sel.runs, "drop_tpr"),
                _rate_text(sel.runs, "drop_fpr"),
                _rate_text(sel.runs, "mal_share"),
            ]
        rows.append(row)
    header = ["attack"] + [
        x for d in defenses for x in (f"{d} TPR", f"{d} FPR", f"{d} mal. share")
    ]
    return _table(header, rows)


def table_knobs(cells: dict[CellKey, list[RunSummary]]) -> str:
    rows = []
    for attack, knob in KNOB_ATTACKS.items():
        for f in FRACTIONS:
            for defense in TABLE_DEFENSES:
                sel = select_cell(cells, "cifar10", attack, f, defense)
                if sel is None:
                    continue
                parts = [
                    f"{s:g}: {_fmt(a)} ({n})"
                    for s, (a, n) in sorted(sel.by_strength.items(), key=lambda kv: -1.0 if kv[0] is None else kv[0])
                ]
                rows.append([ATTACK_NAMES[attack], knob, f"{f:g}", defense, "; ".join(parts),
                             "–" if sel.strength is None else f"{sel.strength:g}"])
    return _table(["attack", "knob", "f", "defense", "mean ASR by knob (n)", "used"], rows)


def table_stats(comparisons: list[Comparison]) -> str:
    rows = []
    for c in comparisons:
        note = "" if c.n >= MIN_PAIRS_FOR_SIGNIFICANCE else f"n<{MIN_PAIRS_FOR_SIGNIFICANCE}: p floor {2 / 2 ** c.n:.3f}"
        rows.append([
            c.label,
            "ASR" if c.metric == "asr_final" else "clean acc",
            str(c.n),
            f"{c.mean_diff:+.3f}",
            f"[{c.ci_low:+.3f}, {c.ci_high:+.3f}]",
            f"{c.p_value:.4f}",
            f"{c.holm_p:.4f}",
            "yes" if c.reject else "no",
            f"{c.rank_biserial:+.2f}",
            note,
        ])
    return _table(["comparison (a − b)", "metric", "pairs", "mean Δ", "BCa 95%", "p", "Holm p",
                   "reject", "r", "note"], rows)


def table_root(cells: dict[CellKey, list[RunSummary]]) -> str:
    rows = []
    classes = 10
    for attack in ROOT_ATTACKS:
        undefended = select_cell(cells, "cifar10", attack, ROOT_FRACTION, "undefended", best_response=False)
        for defense in ROOT_DEFENSES:
            row = [ATTACK_NAMES[attack], defense]
            for v in ROOT_VALS:
                sel = select_cell(cells, "cifar10", attack, ROOT_FRACTION, defense,
                                  val_per_class=v, best_response=False)
                runs = sel.runs if sel is not None else []
                row.append(_cell_text(_values(runs, "asr_final")))
                row.append(_cell_text(_values(runs, "acc_final")))
            row.append(_cell_text(_values(undefended.runs, "asr_final")) if undefended else "–")
            rows.append(row)
    header = ["attack", "defense"]
    for v in ROOT_VALS:
        header += [f"ASR @ {v}/class ({v * classes} samples)", f"clean acc @ {v}/class"]
    header.append("undefended ASR")
    return _table(header, rows)


def table_overhead(cells: dict[CellKey, list[RunSummary]], dataset: str) -> str:
    rows = []
    base = overhead_row(cells, dataset, "undefended").server_s
    for defense in OVERHEAD_DEFENSES:
        o = overhead_row(cells, dataset, defense)

        def cell(key: str) -> str:
            if key not in o.mean_s:
                return "–"
            return f"{o.mean_s[key]:.3f} ± {o.sd_s[key]:.3f}"

        added = "–" if o.server_s is None or base is None else f"{o.server_s - base:+.3f}"
        share = (
            f"{100 * o.server_s / o.mean_s['round_s']:.1f}%"
            if o.server_s is not None and o.mean_s.get("round_s") else "–"
        )
        rows.append([
            defense, str(o.rounds), cell("client_train_s"), cell("scoring_s"),
            cell("aggregation_s"), cell("server_train_s"),
            "–" if o.server_s is None else f"{o.server_s:.3f}", added, cell("round_s"), share,
            _fmt(o.memory_mb.get("server_peak_host_mb"), 1),
            _fmt(o.memory_mb.get("server_peak_device_mb"), 1),
        ])
    return _table(
        ["defense", "rounds", "client train s", "scoring s", "aggregation s", "server train s",
         "server s", "vs undefended", "round s", "server share", "peak host MiB",
         "peak device MiB"],
        rows,
    )


def table_communication() -> str:
    rows = []
    for dataset in DATASETS:
        c = communication_per_round(dataset)
        rows.append([
            dataset, f"{int(c['numel']):,}", _mib(c["state_bytes"]), _mib(c["upload_bytes"]),
            _mib(c["download_bytes"]), _mib(c["total_bytes"]),
            f"{c['total_bytes'] * NUM_ROUNDS / 2**30:.2f}",
        ])
    return _table(["dataset", "state-dict values", "model MiB", f"upload MiB/round (K={CLIENTS_PER_ROUND})",
                   "download MiB/round", "total MiB/round", f"total GiB over {NUM_ROUNDS} rounds"], rows)


def report_datasets(cells: dict[CellKey, list[RunSummary]]) -> tuple[str, ...]:
    """Datasets the report renders: the core two, plus any other with runs."""
    present = {k.dataset for k in cells}
    return tuple(d for d in DATASETS if d in CORE_DATASETS or d in present)


def build_report(cells: dict[CellKey, list[RunSummary]]) -> str:
    lines = ["# E3 systematic FLTrust comparison\n"]
    lines.append(
        f"Converged values average the last {CONVERGENCE_WINDOW} of {NUM_ROUNDS} rounds; "
        "cells read 'mean ± sd (n)'. Only reference-setting runs (N=100, K=10, native "
        "partition, corner trigger, no attack window) are included.\n"
    )
    n_runs = sum(len(v) for v in cells.values())
    lines.append(f"Runs found: {n_runs} in {len(cells)} cells.\n")
    datasets = report_datasets(cells)

    for dataset in datasets:
        benign, n_benign = benign_accuracy(cells, dataset)
        lines.append(f"## {dataset} ({DATASET_NAMES[dataset]}): attack by defense\n")
        lines.append(
            "ASR for targeted attacks; Δacc (clean accuracy minus benign FedAvg "
            f"accuracy {_fmt(benign)}, n={n_benign}) for untargeted ones. † marks a "
            "knob attacker whose best response against that defense is not the "
            "attack config's value (the number after † is the value used). The "
            "overhead row gives server seconds per round and peak server memory "
            "(host + device) from the overhead runs.\n"
        )
        for f in FRACTIONS:
            lines.append(f"### f = {f:g}\n")
            lines.append(table_panel(cells, dataset, f) + "\n")
            lines.append(f"#### SOTA baselines, f = {f:g}\n")
            lines.append(table_panel(cells, dataset, f, SOTA_PANEL) + "\n")
            lines.append(f"#### Bounded-weight median instantiations, f = {f:g}\n")
            lines.append(table_panel(cells, dataset, f, ROOT_PANEL) + "\n")
            lines.append(f"#### Detection, f = {f:g}\n")
            lines.append(
                "From each run's defense ledger: TPR is the share of malicious "
                "client-rounds given zero aggregation weight, FPR the same for "
                "benign client-rounds, and mal. share the mean weight share held "
                "by malicious clients in rounds that had any. Soft-weighting rules "
                "rarely zero a client, so their TPR/FPR are near 0 and the share "
                "carries the result; undefended has no ledger.\n"
            )
            lines.append(table_detection(cells, dataset, f) + "\n")

    lines.append("## Best-response knob (CIFAR-10)\n")
    lines.append(
        "Another value replaces the attack config's only when it raises ASR on the "
        f"shared seeds (at least {MIN_SHARED_SEEDS}) with a BCa 95% interval above zero. "
        f"Values with fewer than {MIN_SEEDS} seeds are listed but not eligible unless "
        "no value has that many.\n"
    )
    lines.append(table_knobs(cells) + "\n")

    lines.append("## Paired comparisons\n")
    lines.append(
        "Seed-matched a − b on each row's primary metric (lower ASR is better, higher "
        "clean accuracy is better), each defense at its own best response. Two-sided "
        "Wilcoxon signed-rank; r is the rank-biserial correlation; BCa is the 95% "
        "bootstrap interval of the mean difference; Holm within each (dataset, f, "
        "attack) row, the defenses compared on one attack. "
        f"Below {MIN_PAIRS_FOR_SIGNIFICANCE} pairs the exact p-value cannot fall "
        "below 0.05, so r and the interval carry the result.\n"
    )
    for dataset in datasets:
        comps = table_comparisons(cells, dataset)
        lines.append(f"### {dataset}\n")
        lines.append((table_stats(comps) if comps else "_no paired cells yet_") + "\n")

    lines.append("## Root-set sensitivity (CIFAR-10, f = 0.1)\n")
    lines.append(
        "The server validation set is FLTrust's root set and the validation-loss "
        "metric's reference, so both defenses move with it. The attackers keep "
        "their attack-config strength (the adaptive budget was calibrated at "
        "100/class). The validation slice is carved from the test set, so the "
        "test pool grows slightly as it shrinks.\n"
    )
    lines.append(table_root(cells) + "\n")
    root = root_comparisons(cells)
    lines.append((table_stats(root) if root else "_no paired root-set cells yet_") + "\n")

    lines.append("## Server overhead\n")
    lines.append(
        f"Seconds per round over rounds {WARMUP_ROUNDS + 1}–{OVERHEAD_ROUNDS} of the "
        "timing runs (mean ± sd over rounds). Server s = scoring + aggregation + "
        "server training; client training and evaluation are common to every "
        "scheme. Peak memory is the server-side allocation above the round's "
        "starting level, from the separate profiling runs: host is traced NumPy and "
        "Python memory, device is the CUDA allocator's peak.\n"
    )
    for dataset in datasets:
        lines.append(f"### {dataset}\n")
        lines.append(table_overhead(cells, dataset) + "\n")
    lines.append("### Communication\n")
    lines.append(
        "Identical for every scheme: each round the K sampled clients download the "
        "global state dict and upload theirs. FLTrust's extra requirement is data "
        "held at the server (the root set), not bandwidth.\n"
    )
    lines.append(table_communication() + "\n")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _json_dump(cells: dict[CellKey, list[RunSummary]]) -> list[dict[str, Any]]:
    out = []
    for key, runs in cells.items():
        for r in runs:
            record = {**asdict(key), **asdict(r)}
            record["timing_mean"] = {k: float(np.mean(v)) for k, v in r.timing_rounds.items() if v}
            del record["timing_rounds"]
            out.append(record)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None, help="per-run dump")
    parser.add_argument(
        "--knob-bumps",
        action="store_true",
        help="print sweep lines (tag, experiment, seeds, overrides) for knob values "
        "that would win a table cell once they match the reference seed count",
    )
    parser.add_argument(
        "--defenses",
        default=",".join(TABLE_DEFENSES + SOTA_DEFENSES),
        help="comma-separated defense labels for --knob-bumps",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="render the appendix table for this backbone label (e.g. resnet18_gn) "
        "instead of the full report",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LOCAL_LR,
        help="client learning rate of the --model cells",
    )
    args = parser.parse_args(argv)

    results_root: Path = args.results_root.resolve()
    if not results_root.is_dir():
        raise SystemExit(f"results root not found: {results_root}")
    for target in (args.out, args.json):
        if target is not None:
            resolved = target.resolve()
            if results_root == resolved or results_root in resolved.parents:
                raise SystemExit(f"refusing to write inside the results tree: {resolved}")

    cells = load_cells(results_root)
    if args.knob_bumps:
        defenses = tuple(d for d in args.defenses.split(",") if d)
        unknown = [d for d in defenses if d not in DEFENSE_OVERRIDES]
        if unknown:
            raise SystemExit(f"no sweep override known for defense(s): {', '.join(unknown)}")
        for bump in knob_bumps(cells, "cifar10", defenses):
            print(bump.sweep_line())
        return 0
    report = (
        build_model_report(cells, args.model, args.lr) if args.model else build_report(cells)
    )
    if args.out is not None:
        args.out.write_text(report)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(report)
    if args.json is not None:
        args.json.write_text(json.dumps(_json_dump(cells), indent=2))
        print(f"wrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
