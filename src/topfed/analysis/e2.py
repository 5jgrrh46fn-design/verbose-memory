"""E2 compromise-rate sensitivity: where each defense breaks as f rises, and why.

Reads the per-run ledgers under ``results/`` (``manifest.json``,
``metrics.json``, ``client_metrics.json``, and ``defense_ledger.json`` when a
defense wrote one). Only reference-setting runs are in scope (uniform
sampling, K=10, N=100, 200 rounds, the dataset's native partition, corner
trigger, no attack window); diagnostic variants belong to other analyses. A
run's cell is keyed by its resolved config, so the frozen f=0.10 / 0.20 cells
and the tagged new levels land in the same curves.

Sections of the report:

* **Sampling tail** -- per f, the expected co-selected attackers K*f, the
  hypergeometric probability of a round in which attackers are at least half
  the cohort (the weighted median's breakdown point), the probability of at
  least one such round in a run, and the observed per-round malicious-count
  histogram next to the hypergeometric pmf.
* **ASR and clean accuracy vs f** per attack and defense, with the breaking
  f (first f whose mean converged ASR reaches 0.5).
* **Adaptive budget rescan** -- the max-over-budget rule at the rescanned
  levels: seed-0 ASR per budget, the maximising budget, and whether the main
  grid (budget 0.6) understates the attacker there.
* **Detection vs aggregation** -- oracle weights on the weighted-median base
  against capped-wmed at each f, with a per-f verdict.
* **Per-metric AUC vs f** -- malicious-vs-benign AUC of each trust metric in an
  early (rounds 1-25) and a late (51-200) window.
* **FEMNIST cliff** and the optional **untargeted** f-sweep.

``--fig-dir`` also writes the curves as PNGs, with the hypergeometric tail
overlaid on the ASR panel. ``--breaking-f`` prints, one per line, the
``<experiment> <override>`` pairs whose breaking f is a new level and still
needs the extra seeds, for ``experiments/sweep_e2.sh``.

Read-only over ``results/``; every output path is rejected inside the
results tree.

Usage::

    uv run python -m topfed.analysis.e2 --out tables/e2-tables.md
    uv run python -m topfed.analysis.e2 --fig-dir figures --json e2-runs.json
    uv run python -m topfed.analysis.e2 --breaking-f
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from topfed.analysis.e5 import (
    ASR_THRESHOLD,
    CONVERGENCE_WINDOW,
    FIRST_VALID_ROUND,
    REFERENCE,
    REFERENCE_COMMON,
    _defense_label,
    _fmt,
    _malicious_share,
    _mean_sd,
    _partition_label,
    _table,
    _window,
    hypergeom_tail,
    sustained_implant,
)
from topfed.analysis.stats import auc_separability

NUM_CLIENTS = 100
CLIENTS_PER_ROUND = 10
NUM_ROUNDS = 200
# Server validation (root) set size per class; smaller root sets are a
# separate experiment and must not enter the compromise-rate curves.
REFERENCE_VAL_PER_CLASS = {"cifar10": 100, "femnist": 30}
CIFAR_FRACTIONS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
FEMNIST_FRACTIONS = (0.05, 0.10, 0.15, 0.20)
FROZEN_FRACTIONS = (0.10, 0.20)
CIFAR_DEFENSES = ("undefended", "ScoreWeight", "capped-wmed", "median", "fltrust")
FEMNIST_DEFENSES = ("undefended", "wmed", "capped-wmed", "median", "ScoreWeight")
UNTARGETED_DEFENSES = ("undefended", "capped-wmed")
REFERENCE_BUDGET = 0.6
RESCAN_FRACTIONS = (0.30, 0.40)
RESCAN_BUDGETS = (0.6, 1.0, 1.5)
RESCAN_DEFENSES = ("capped-wmed", "ScoreWeight")
RESCAN_SEED = 0
# Defenses whose breaking f gets extra seeds, and the override that runs them.
BUMP_OVERRIDES = {
    "capped-wmed": "+defense=topsis_capped_weighted_median",
    "fltrust": "+baseline=fltrust",
}
BUMP_TARGET_SEEDS = 10
MIN_SEEDS = 5
METRICS = (
    "m1_cosine_to_mean",
    "m2_norm_ratio",
    "m3_temporal_cv",
    "m4_val_loss_delta",
    "m5_max_layer_deviation",
)
AUC_WINDOWS: dict[str, tuple[int, int | None]] = {
    "early": (1, 25),
    "late": (51, None),
}
ATTACK_NAMES = {
    "backdoor": "vanilla backdoor",
    "adaptive": "adaptive backdoor",
    "gradient_scaling": "gradient scaling",
}


@dataclass(frozen=True)
class CellKey:
    """One point of an E2 curve: dataset, attacker, compromise level, defense.

    ``strength`` is the adaptive attacker's norm budget or the gradient-scaling
    factor; ``None`` for the scaled (model-replacement) backdoor.
    """

    dataset: str
    attack: str
    strength: float | None
    fraction: float
    defense: str


@dataclass
class RunSummary:
    """Per-seed quantities the E2 tables are built from."""

    seed: int
    num_malicious: int
    asr_final: float | None
    acc_final: float
    implant_round: int | None
    first_majority_round: int | None
    max_malicious_share: float
    malicious_count_hist: list[int]
    # Sum of the malicious aggregation-weight share over the rounds with
    # exactly m attackers (index m); empty when the run wrote no ledger.
    share_sum_by_m: list[float] = field(default_factory=list)
    auc: dict[str, dict[str, float]] = field(default_factory=dict)
    source: str = ""


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------


def _target_name(block: dict[str, Any] | None) -> str | None:
    if not block:
        return None
    return str(block.get("_target_", "")).rsplit(".", 1)[-1]


def _is_reference_setting(cfg: dict[str, Any]) -> bool:
    """True when every knob other than f, the attacker and the defense is canonical."""
    fed = cfg["federation"]
    dataset = str(cfg["dataset"]["name"])
    if dataset not in REFERENCE:
        return False
    return (
        int(fed["num_clients"]) == NUM_CLIENTS
        and int(fed["clients_per_round"]) == CLIENTS_PER_ROUND
        and int(fed["num_rounds"]) == NUM_ROUNDS
        and fed.get("force_malicious_per_round") is None
        and _partition_label(cfg) == REFERENCE[dataset]["partition"]
        and int(cfg["dataset"].get("val_per_class", REFERENCE_VAL_PER_CLASS[dataset]))
        == REFERENCE_VAL_PER_CLASS[dataset]
        and int(cfg["dataset"].get("pad_border", 0) or 0) == REFERENCE_COMMON["pad_border"]
        and _window(cfg) == REFERENCE_COMMON["window"]
    )


def _backdoor_attack(cfg: dict[str, Any]) -> tuple[str, float | None] | None:
    attack = cfg["attack"]
    poisoner = attack.get("poisoner") or {}
    if not poisoner:
        return None
    ref = REFERENCE[str(cfg["dataset"]["name"])]
    if (
        str(poisoner.get("trigger_position", "bottom_right"))
        != REFERENCE_COMMON["trigger_position"]
        or int(poisoner["trigger_size"]) != REFERENCE_COMMON["trigger_size"]
        or float(poisoner["trigger_amplitude"]) != ref["trigger_amplitude"]
        or str(poisoner.get("batch_mode", "replace")) != REFERENCE_COMMON["batch_mode"]
    ):
        return None
    manipulator = attack.get("update_manipulator") or {}
    target = _target_name(manipulator)
    if target == "ModelReplacementManipulator" and manipulator.get("gamma") is None:
        return ("backdoor", None)
    if target == "NormBoundedManipulator":
        return ("adaptive", float(manipulator["target_norm"]))
    return None


def cell_key(manifest: dict[str, Any]) -> CellKey | None:
    """Build the E2 cell key from a manifest; ``None`` for out-of-scope runs."""
    cfg = manifest["config"]
    if not _is_reference_setting(cfg):
        return None
    attack = cfg["attack"]
    kind = attack.get("kind")
    if kind == "backdoor":
        found = _backdoor_attack(cfg)
        if found is None:
            return None
        name, strength = found
    elif kind == "gradient_scaling":
        manipulator = attack.get("update_manipulator") or {}
        if _target_name(manipulator) != "GradientScalingManipulator":
            return None
        name, strength = "gradient_scaling", float(manipulator["factor"])
    else:
        return None
    return CellKey(
        dataset=str(cfg["dataset"]["name"]),
        attack=name,
        strength=strength,
        fraction=round(float(attack["assignment"]["fraction"]), 4),
        defense=_defense_label(cfg),
    )


def _window_auc(
    client_rows: list[dict[str, Any]], lo: int, hi: int | None
) -> dict[str, float]:
    out: dict[str, float] = {}
    rows = [r for r in client_rows if lo <= int(r["round"]) and (hi is None or int(r["round"]) <= hi)]
    for metric in METRICS:
        mal = [float(r[metric]) for r in rows if r["is_malicious"] and r.get(metric) is not None]
        ben = [float(r[metric]) for r in rows if not r["is_malicious"] and r.get(metric) is not None]
        out[metric] = auc_separability(mal, ben)
    return out


def summarise_run(
    seed: int,
    num_malicious: int,
    metrics: list[dict[str, Any]],
    client_rows: list[dict[str, Any]],
    ledger: list[dict[str, Any]] | None,
    k: int = CLIENTS_PER_ROUND,
) -> RunSummary:
    rounds = np.array([int(m["round"]) for m in metrics])
    acc = np.array([float(m.get("test_accuracy", math.nan)) for m in metrics])
    has_asr = bool(metrics) and "asr_accuracy" in metrics[-1]
    asr = np.array([float(m.get("asr_accuracy", math.nan)) for m in metrics])

    per_round_mal: dict[int, int] = defaultdict(int)
    for row in client_rows:
        per_round_mal[int(row["round"])] += int(bool(row["is_malicious"]))
    mal_counts = np.array([per_round_mal[int(r)] for r in rounds], dtype=int)
    share = _malicious_share(rounds, ledger, mal_counts, k)
    tail = share[FIRST_VALID_ROUND - 1 :] if share.size >= FIRST_VALID_ROUND else share
    majority = np.flatnonzero(tail >= 0.5) + min(FIRST_VALID_ROUND, share.size)
    hist = np.bincount(mal_counts, minlength=k + 1)[: k + 1] if mal_counts.size else np.zeros(k + 1, dtype=int)
    has_ledger = bool(ledger) and "weight" in ledger[0]  # type: ignore[index]
    share_sum = (
        np.bincount(mal_counts, weights=share, minlength=k + 1)[: k + 1]
        if has_ledger and mal_counts.size
        else np.zeros(0)
    )

    return RunSummary(
        seed=seed,
        num_malicious=num_malicious,
        asr_final=float(np.nanmean(asr[-CONVERGENCE_WINDOW:])) if has_asr else None,
        acc_final=float(np.nanmean(acc[-CONVERGENCE_WINDOW:])),
        implant_round=sustained_implant(asr) if has_asr else None,
        first_majority_round=int(majority[0]) if majority.size else None,
        max_malicious_share=float(tail.max()) if tail.size else 0.0,
        malicious_count_hist=[int(x) for x in hist],
        share_sum_by_m=[float(x) for x in share_sum],
        auc={
            name: _window_auc(client_rows, lo, hi)
            for name, (lo, hi) in AUC_WINDOWS.items()
        },
    )


def load_cells(results_root: Path) -> dict[CellKey, list[RunSummary]]:
    """Walk ``results/`` and group completed in-scope runs by cell.

    The same cell can sit under more than one ``<label>/<hash>`` directory
    (an identical config under another experiment name or tag). Each seed is
    then taken from the directory with the most completed seeds, ties by
    path, and the others are reported on stderr.
    """
    candidates: dict[tuple[CellKey, int], list[Path]] = defaultdict(list)
    manifests: dict[Path, dict[str, Any]] = {}
    completed: dict[Path, int] = defaultdict(int)
    for manifest_path in sorted(results_root.glob("*/*/seed_*/manifest.json")):
        run_dir = manifest_path.parent
        if not (run_dir / "metrics.json").exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        key = cell_key(manifest)
        if key is None:
            continue
        manifests[run_dir] = manifest
        completed[run_dir.parent] += 1
        candidates[(key, int(manifest["seed"]))].append(run_dir)

    cells: dict[CellKey, list[RunSummary]] = defaultdict(list)
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
        if len(metrics) != NUM_ROUNDS:
            continue
        client_path = run_dir / "client_metrics.json"
        client_rows = json.loads(client_path.read_text()) if client_path.exists() else []
        ledger_path = run_dir / "defense_ledger.json"
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else None
        num_malicious = int(
            manifests[run_dir].get("num_malicious")
            or math.ceil(NUM_CLIENTS * key.fraction - 1e-9)
        )
        summary = summarise_run(seed, num_malicious, metrics, client_rows, ledger)
        summary.source = str(run_dir.relative_to(results_root))
        cells[key].append(summary)
    for (kept, dropped), n in sorted(ignored.items()):
        print(f"note: same config in {dropped}; {n} seed(s) taken from {kept}", file=sys.stderr)
    return {k: sorted(v, key=lambda r: r.seed) for k, v in cells.items()}


# ----------------------------------------------------------------------------
# Derived quantities
# ----------------------------------------------------------------------------


def _find(
    cells: dict[CellKey, list[RunSummary]],
    dataset: str,
    attack: str,
    strength: float | None,
    fraction: float,
    defense: str,
) -> list[RunSummary]:
    return cells.get(CellKey(dataset, attack, strength, round(fraction, 4), defense), [])


def _mean(values: list[float | None]) -> float | None:
    xs = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.mean(xs)) if xs else None


def mean_asr(runs: list[RunSummary]) -> float | None:
    return _mean([r.asr_final for r in runs])


def breaking_fraction(
    cells: dict[CellKey, list[RunSummary]],
    dataset: str,
    attack: str,
    strength: float | None,
    defense: str,
    fractions: tuple[float, ...],
) -> tuple[float | None, bool]:
    """First f (ascending) whose mean converged ASR reaches the threshold.

    Returns ``(f, complete)``. ``complete`` is False when a level at or below
    the returned f (or any level, when none breaks) has no runs, so the
    first crossing is not established.
    """
    complete = True
    for f in fractions:
        runs = _find(cells, dataset, attack, strength, f, defense)
        if not runs:
            complete = False
            continue
        asr = mean_asr(runs)
        if asr is not None and asr >= ASR_THRESHOLD:
            return f, complete
    return None, complete


def experiment_name(dataset: str, attack: str, fraction: float) -> str:
    """The experiment config that runs this (dataset, attacker, f) cell."""
    level = f"f{round(fraction * 100):02d}"
    if dataset == "femnist":
        return f"femnist_backdoor_{level}"
    prefix = {"backdoor": "backdoor", "adaptive": "adaptive_backdoor",
              "gradient_scaling": "gradient_scaling"}[attack]
    return f"{prefix}_{dataset}_{level}"


def bump_targets(cells: dict[CellKey, list[RunSummary]]) -> list[tuple[str, str]]:
    """``(experiment, override)`` cells whose breaking f needs more seeds.

    Only a new level qualifies (the frozen levels already have n=10), only
    when every level up to it has runs, and only while the cell has fewer
    than the target seed count.
    """
    out: list[tuple[str, str]] = []
    for attack, strength in (("backdoor", None), ("adaptive", REFERENCE_BUDGET)):
        for defense, override in BUMP_OVERRIDES.items():
            f, complete = breaking_fraction(
                cells, "cifar10", attack, strength, defense, CIFAR_FRACTIONS
            )
            label = f"{attack}/{defense}"
            if f is None:
                print(f"{label}: never breaks{'' if complete else ' (grid incomplete)'}", file=sys.stderr)
                continue
            if not complete:
                print(f"{label}: first crossing at f={f:g} not established (grid incomplete)", file=sys.stderr)
                continue
            if any(abs(f - x) < 1e-9 for x in FROZEN_FRACTIONS):
                print(f"{label}: breaks at frozen level f={f:g}, already n=10", file=sys.stderr)
                continue
            n = len(_find(cells, "cifar10", attack, strength, f, defense))
            if n >= BUMP_TARGET_SEEDS:
                print(f"{label}: breaks at f={f:g}, already n={n}", file=sys.stderr)
                continue
            print(f"{label}: breaks at f={f:g}, n={n}", file=sys.stderr)
            out.append((experiment_name("cifar10", attack, f), override))
    return out


@dataclass(frozen=True)
class BudgetRescan:
    """Max-over-budget rule for one (f, defense) rescan cell."""

    fraction: float
    defense: str
    asr_by_budget: dict[float, float | None]
    best_budget: float | None
    best_asr: float | None

    @property
    def understated(self) -> bool:
        """The main-grid budget is beaten by another budget at this cell."""
        ref = self.asr_by_budget.get(REFERENCE_BUDGET)
        return (
            self.best_budget is not None
            and self.best_budget != REFERENCE_BUDGET
            and self.best_asr is not None
            and (ref is None or self.best_asr > ref)
        )


def budget_rescan(
    cells: dict[CellKey, list[RunSummary]],
    fraction: float,
    defense: str,
    budgets: tuple[float, ...] = RESCAN_BUDGETS,
    seed: int = RESCAN_SEED,
) -> BudgetRescan:
    """Seed-``seed`` ASR per adaptive budget and the maximising budget.

    Ties go to the reference budget, so a flat response never flags the main
    grid as understated.
    """
    asr_by_budget: dict[float, float | None] = {}
    for g in budgets:
        runs = [r for r in _find(cells, "cifar10", "adaptive", g, fraction, defense) if r.seed == seed]
        asr_by_budget[g] = runs[0].asr_final if runs else None
    scored = [(g, a) for g, a in asr_by_budget.items() if a is not None]
    if not scored:
        return BudgetRescan(fraction, defense, asr_by_budget, None, None)
    best_asr = max(a for _, a in scored)
    winners = [g for g, a in scored if a == best_asr]
    best = REFERENCE_BUDGET if REFERENCE_BUDGET in winners else winners[0]
    return BudgetRescan(fraction, defense, asr_by_budget, best, best_asr)


def oracle_verdict(oracle_asr: float | None, capped_asr: float | None) -> str:
    """Which side limits the defense at one f."""
    if oracle_asr is None:
        return "–"
    if oracle_asr >= ASR_THRESHOLD:
        return "aggregation-limited"
    if capped_asr is None:
        return "–"
    if capped_asr >= ASR_THRESHOLD:
        return "detection-limited"
    return "holds"


def hypergeom_pmf(N: int, M: int, K: int, m: int) -> float:
    if m < 0 or m > K or m > M or K - m > N - M:
        return 0.0
    return math.comb(M, m) * math.comb(N - M, K - m) / math.comb(N, K)


@dataclass(frozen=True)
class TailRow:
    dataset: str
    fraction: float
    num_malicious: int
    expected_per_round: float
    breakdown_m: int
    p_round: float
    p_any: float
    observed_round_frac: float | None
    observed_seeds_with_majority: tuple[int, int]
    max_m_observed: int | None
    observed_pmf: list[float]
    expected_pmf: list[float]


def sampling_tail(
    cells: dict[CellKey, list[RunSummary]], dataset: str, fractions: tuple[float, ...]
) -> list[TailRow]:
    """Hypergeometric tail vs the observed per-round malicious counts.

    Cohort sampling depends only on the seed, so each seed is counted once
    per (dataset, f), from the first cell that has it.
    """
    k = CLIENTS_PER_ROUND
    breakdown = math.ceil(k / 2)
    rows = []
    for f in fractions:
        by_seed: dict[int, RunSummary] = {}
        for key, runs in sorted(cells.items(), key=lambda kv: (kv[0].defense != "undefended", str(kv[0]))):
            if key.dataset != dataset or abs(key.fraction - f) > 1e-9:
                continue
            for r in runs:
                by_seed.setdefault(r.seed, r)
        num_malicious = (
            next(iter(by_seed.values())).num_malicious
            if by_seed
            else math.ceil(NUM_CLIENTS * f - 1e-9)
        )
        p_round = hypergeom_tail(NUM_CLIENTS, num_malicious, k, breakdown)
        expected_pmf = [hypergeom_pmf(NUM_CLIENTS, num_malicious, k, m) for m in range(k + 1)]
        if by_seed:
            total = np.sum([r.malicious_count_hist for r in by_seed.values()], axis=0)
            rounds = float(total.sum())
            observed_pmf = [float(x) / rounds for x in total]
            observed_frac: float | None = float(total[breakdown:].sum()) / rounds
            with_majority = sum(1 for r in by_seed.values() if sum(r.malicious_count_hist[breakdown:]) > 0)
            nonzero = np.flatnonzero(total)
            max_m: int | None = int(nonzero[-1]) if nonzero.size else 0
        else:
            observed_pmf, observed_frac, with_majority, max_m = [], None, 0, None
        rows.append(
            TailRow(
                dataset=dataset,
                fraction=f,
                num_malicious=num_malicious,
                expected_per_round=k * num_malicious / NUM_CLIENTS,
                breakdown_m=breakdown,
                p_round=p_round,
                p_any=1.0 - (1.0 - p_round) ** NUM_ROUNDS,
                observed_round_frac=observed_frac,
                observed_seeds_with_majority=(with_majority, len(by_seed)),
                max_m_observed=max_m,
                observed_pmf=observed_pmf,
                expected_pmf=expected_pmf,
            )
        )
    return rows


def suppression_by_m(
    runs: list[RunSummary], k: int = CLIENTS_PER_ROUND, min_rounds: int = 5
) -> dict[int, float]:
    """Malicious weight share over its equal share m/K, by attackers in the round.

    Pooled over the runs' ledger rounds; 0 means the attackers were fully
    down-weighted, 1 means the defense was neutral, above 1 means honest
    clients were down-weighted instead. Counts below ``min_rounds`` are
    left out.
    """
    with_ledger = [r for r in runs if r.share_sum_by_m]
    if not with_ledger:
        return {}
    share = np.sum([r.share_sum_by_m for r in with_ledger], axis=0)
    rounds = np.sum([r.malicious_count_hist for r in with_ledger], axis=0)
    return {
        m: float(share[m] / rounds[m] / (m / k))
        for m in range(1, k + 1)
        if rounds[m] >= min_rounds
    }


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------


def _f_header(fractions: tuple[float, ...]) -> list[str]:
    return [f"f={f:g} (Kf={CLIENTS_PER_ROUND * f:g})" for f in fractions]


def _cell(runs: list[RunSummary], attr: str) -> str:
    if not runs:
        return "–"
    return f"{_mean_sd([getattr(r, attr) for r in runs])} (n={len(runs)})"


def table_tail(cells: dict[CellKey, list[RunSummary]], dataset: str, fractions: tuple[float, ...]) -> str:
    rows = []
    for t in sampling_tail(cells, dataset, fractions):
        with_m, seeds = t.observed_seeds_with_majority
        rows.append([
            f"{t.fraction:g}",
            str(t.num_malicious),
            f"{t.expected_per_round:g}",
            f"{t.p_round:.4f}",
            f"{1 / t.p_round:.0f}" if t.p_round > 0 else "–",
            f"{t.p_any:.3f}",
            _fmt(t.observed_round_frac, 4),
            f"{with_m}/{seeds}" if seeds else "–",
            "–" if t.max_m_observed is None else str(t.max_m_observed),
        ])
    return _table(
        ["f", "attackers", "E[m] = Kf", f"P(m>={math.ceil(CLIENTS_PER_ROUND / 2)}) per round",
         "E[first such round]", f"P(any in {NUM_ROUNDS})", "observed rounds with m>=5",
         "seeds with such a round", "max m seen"],
        rows,
    )


def table_count_hist(cells: dict[CellKey, list[RunSummary]], dataset: str, fractions: tuple[float, ...]) -> str:
    """Observed vs hypergeometric per-round malicious-count distribution."""
    top = 7
    rows = []
    for t in sampling_tail(cells, dataset, fractions):
        obs = t.observed_pmf
        exp = t.expected_pmf
        cols = []
        for m in range(top):
            o = _fmt(obs[m], 3) if obs else "–"
            cols.append(f"{o} / {exp[m]:.3f}")
        o_tail = _fmt(sum(obs[top:]), 3) if obs else "–"
        cols.append(f"{o_tail} / {sum(exp[top:]):.3f}")
        rows.append([f"{t.fraction:g}"] + cols)
    return _table(["f"] + [f"m={m}" for m in range(top)] + [f"m>={top}"], rows)


def table_vs_f(
    cells: dict[CellKey, list[RunSummary]],
    dataset: str,
    attack: str,
    strength: float | None,
    defenses: tuple[str, ...],
    fractions: tuple[float, ...],
    attr: str,
    with_break: bool,
) -> str:
    rescans = (
        {(b.fraction, b.defense): b for f in RESCAN_FRACTIONS for d in RESCAN_DEFENSES
         for b in [budget_rescan(cells, f, d)]}
        if attack == "adaptive" and dataset == "cifar10" and strength == REFERENCE_BUDGET
        else {}
    )
    rows = []
    for defense in defenses:
        row = [defense]
        for f in fractions:
            text = _cell(_find(cells, dataset, attack, strength, f, defense), attr)
            rescan = rescans.get((round(f, 4), defense))
            if attr == "asr_final" and rescan is not None and rescan.understated:
                text += f" †{rescan.best_budget:g}"
            row.append(text)
        if with_break:
            f_break, complete = breaking_fraction(cells, dataset, attack, strength, defense, fractions)
            if f_break is None:
                row.append("none" if complete else "none yet")
            else:
                row.append(f"{f_break:g}" + ("" if complete else " (gaps below)"))
        rows.append(row)
    header = ["defense"] + _f_header(fractions) + (["breaking f"] if with_break else [])
    return _table(header, rows)


def table_rescan(cells: dict[CellKey, list[RunSummary]]) -> str:
    rows = []
    for f in RESCAN_FRACTIONS:
        for defense in RESCAN_DEFENSES:
            b = budget_rescan(cells, f, defense)
            main = mean_asr(_find(cells, "cifar10", "adaptive", REFERENCE_BUDGET, f, defense))
            rows.append(
                [f"{f:g}", defense]
                + [_fmt(b.asr_by_budget[g]) for g in RESCAN_BUDGETS]
                + [
                    "–" if b.best_budget is None else f"{b.best_budget:g}",
                    _fmt(b.best_asr),
                    _fmt(main),
                    "yes" if b.understated else ("no" if b.best_budget is not None else "–"),
                ]
            )
    return _table(
        ["f", "defense"]
        + [f"ASR seed {RESCAN_SEED} @ budget {g:g}" for g in RESCAN_BUDGETS]
        + ["max-ASR budget", "max over budgets", f"main grid ASR @ {REFERENCE_BUDGET:g} (n=5)",
           "main grid understated"],
        rows,
    )


def table_oracle(cells: dict[CellKey, list[RunSummary]]) -> str:
    rows = []
    for attack, strength in (("backdoor", None), ("adaptive", REFERENCE_BUDGET)):
        for f in CIFAR_FRACTIONS:
            oracle = _find(cells, "cifar10", attack, strength, f, "oracle-wmed")
            capped = _find(cells, "cifar10", attack, strength, f, "capped-wmed")
            if not oracle and not capped:
                continue
            rows.append([
                ATTACK_NAMES[attack],
                f"{f:g}",
                f"{CLIENTS_PER_ROUND * f:g}",
                _cell(oracle, "asr_final"),
                _cell(capped, "asr_final"),
                _cell(oracle, "acc_final"),
                _cell(capped, "acc_final"),
                oracle_verdict(mean_asr(oracle), mean_asr(capped)),
            ])
    return _table(
        ["attack", "f", "Kf", "oracle-wmed ASR", "capped-wmed ASR", "oracle-wmed acc",
         "capped-wmed acc", "verdict"],
        rows,
    )


SUPPRESSION_DEFENSES = ("capped-wmed", "wmed")
SUPPRESSION_MAX_M = 8


def table_suppression(cells: dict[CellKey, list[RunSummary]]) -> str:
    rows = []
    for dataset, attack, strength, fractions in (
        ("cifar10", "backdoor", None, CIFAR_FRACTIONS),
        ("cifar10", "adaptive", REFERENCE_BUDGET, CIFAR_FRACTIONS),
        ("femnist", "backdoor", None, FEMNIST_FRACTIONS),
    ):
        for defense in SUPPRESSION_DEFENSES:
            for f in fractions:
                ratios = suppression_by_m(_find(cells, dataset, attack, strength, f, defense))
                if not ratios:
                    continue
                rows.append(
                    [dataset, ATTACK_NAMES[attack], defense, f"{f:g}"]
                    + [_fmt(ratios.get(m), 2) for m in range(1, SUPPRESSION_MAX_M + 1)]
                )
    return _table(
        ["dataset", "attack", "defense", "f"] + [f"m={m}" for m in range(1, SUPPRESSION_MAX_M + 1)],
        rows,
    )


def _short_metric(metric: str) -> str:
    return metric.split("_", 1)[0].upper()


def table_auc(
    cells: dict[CellKey, list[RunSummary]],
    dataset: str,
    attack: str,
    strength: float | None,
    defenses: tuple[str, ...],
    fractions: tuple[float, ...],
) -> str:
    rows = []
    for defense in defenses:
        for metric in METRICS:
            row = [defense, _short_metric(metric)]
            for f in fractions:
                runs = _find(cells, dataset, attack, strength, f, defense)
                parts = [
                    _fmt(_mean([r.auc.get(w, {}).get(metric) for r in runs]))
                    for w in AUC_WINDOWS
                ]
                row.append(" / ".join(parts) if runs else "–")
            rows.append(row)
    return _table(["defense", "metric"] + _f_header(fractions), rows)


def table_femnist(cells: dict[CellKey, list[RunSummary]]) -> str:
    tail = {t.fraction: t for t in sampling_tail(cells, "femnist", FEMNIST_FRACTIONS)}
    rows = []
    for defense in FEMNIST_DEFENSES:
        for f in FEMNIST_FRACTIONS:
            runs = _find(cells, "femnist", "backdoor", None, f, defense)
            if not runs:
                rows.append([defense, f"{f:g}", "0"] + ["–"] * 6)
                continue
            captured = sum(1 for r in runs if r.implant_round is not None)
            rows.append([
                defense,
                f"{f:g}",
                str(len(runs)),
                _mean_sd([r.asr_final for r in runs]),
                _mean_sd([r.acc_final for r in runs]),
                ",".join("never" if r.implant_round is None else str(r.implant_round) for r in runs),
                ",".join("never" if r.first_majority_round is None else str(r.first_majority_round) for r in runs),
                f"{captured}/{len(runs)}",
                f"{tail[f].p_any:.2f}",
            ])
    return _table(
        ["defense", "f", "n", "ASR final", "clean acc", "implant round (per seed)",
         "first share>=0.5 (per seed)", "seeds implanted", f"P(any m>=5 in {NUM_ROUNDS})"],
        rows,
    )


def _untargeted_strength(cells: dict[CellKey, list[RunSummary]]) -> float | None:
    factors = sorted({k.strength for k in cells if k.attack == "gradient_scaling" and k.strength is not None})
    return 5.0 if 5.0 in factors else (factors[0] if factors else None)


def build_report(cells: dict[CellKey, list[RunSummary]]) -> str:
    lines = ["# E2 compromise-rate sensitivity\n"]
    lines.append(
        f"Converged values average the last {CONVERGENCE_WINDOW} rounds; cells "
        f"read 'mean ± sd (n)'. 'Breaking f' is the first f whose mean ASR "
        f"reaches {ASR_THRESHOLD}. Kf is the expected number of attackers among "
        f"the K={CLIENTS_PER_ROUND} sampled clients. Only reference-setting runs "
        f"(N={NUM_CLIENTS}, K={CLIENTS_PER_ROUND}, {NUM_ROUNDS} rounds, native "
        "partition, corner trigger) are included.\n"
    )
    n_runs = sum(len(v) for v in cells.values())
    lines.append(f"Runs found: {n_runs} in {len(cells)} cells.\n")

    lines.append("## Sampling tail\n")
    lines.append(
        f"A round with m >= {math.ceil(CLIENTS_PER_ROUND / 2)} attackers can hold "
        "half the weight of an equal-weight cohort, the weighted median's "
        "breakdown point. Observed counts come from the client ledgers, one "
        "run per seed.\n"
    )
    for dataset, fractions in (("cifar10", CIFAR_FRACTIONS), ("femnist", FEMNIST_FRACTIONS)):
        lines.append(f"### {dataset}\n")
        lines.append(table_tail(cells, dataset, fractions) + "\n")
        lines.append("Per-round malicious count, observed / hypergeometric:\n")
        lines.append(table_count_hist(cells, dataset, fractions) + "\n")

    for attack, strength in (("backdoor", None), ("adaptive", REFERENCE_BUDGET)):
        suffix = f" (budget {strength:g})" if strength is not None else ""
        lines.append(f"## CIFAR-10 {ATTACK_NAMES[attack]}{suffix}\n")
        if attack == "adaptive":
            lines.append(
                "† marks a rescanned cell where another budget beats "
                f"{REFERENCE_BUDGET:g} on seed {RESCAN_SEED} (the number after † is "
                "that budget); the n=5 value understates the attacker there.\n"
            )
        lines.append("ASR:\n")
        lines.append(table_vs_f(cells, "cifar10", attack, strength, CIFAR_DEFENSES,
                                CIFAR_FRACTIONS, "asr_final", with_break=True) + "\n")
        lines.append("Clean accuracy:\n")
        lines.append(table_vs_f(cells, "cifar10", attack, strength, CIFAR_DEFENSES,
                                CIFAR_FRACTIONS, "acc_final", with_break=False) + "\n")

    lines.append("## Adaptive budget rescan (max over budgets)\n")
    lines.append(
        f"Seed {RESCAN_SEED} only. The attacker picks the budget that maximises "
        "defended ASR; ties go to the main-grid budget.\n"
    )
    lines.append(table_rescan(cells) + "\n")

    lines.append("## Detection vs aggregation\n")
    lines.append(
        "Oracle weights (true attackers 0) on the weighted-median base against "
        "capped-wmed. aggregation-limited: the base fails even with perfect "
        "detection; detection-limited: the oracle holds and capped-wmed fails.\n"
    )
    lines.append(table_oracle(cells) + "\n")

    lines.append("## Suppression vs co-selected attackers\n")
    lines.append(
        "Malicious weight share divided by the equal share m/K, by the number "
        "of attackers m in the round, pooled over ledger rounds (at least 5 "
        "per entry). 0 = attackers fully down-weighted, 1 = neutral, above 1 "
        "= honest clients down-weighted instead.\n"
    )
    lines.append(table_suppression(cells) + "\n")

    lines.append("## Per-metric AUC vs f\n")
    windows = " / ".join(
        f"{name} (rounds {lo}–{hi if hi is not None else NUM_ROUNDS})"
        for name, (lo, hi) in AUC_WINDOWS.items()
    )
    lines.append(
        f"Malicious-vs-benign AUC (P(malicious > benign), 0.5 = no separation, "
        f"toward 0 = attackers lower), mean over seeds; cells read {windows}.\n"
    )
    for dataset, attack, strength, fractions in (
        ("cifar10", "backdoor", None, CIFAR_FRACTIONS),
        ("cifar10", "adaptive", REFERENCE_BUDGET, CIFAR_FRACTIONS),
        ("femnist", "backdoor", None, FEMNIST_FRACTIONS),
    ):
        lines.append(f"### {dataset} {ATTACK_NAMES[attack]}\n")
        lines.append(table_auc(cells, dataset, attack, strength,
                               ("undefended", "capped-wmed"), fractions) + "\n")

    lines.append("## FEMNIST cliff\n")
    lines.append(
        "'implant round' is the first round from which ASR stays >= "
        f"{ASR_THRESHOLD} for 10 rounds; 'first share>=0.5' the first round "
        f"(from round {FIRST_VALID_ROUND}) in which attackers held half the "
        "aggregation weight.\n"
    )
    lines.append(table_femnist(cells) + "\n")

    factor = _untargeted_strength(cells)
    lines.append("## Untargeted (gradient scaling)\n")
    if factor is None:
        lines.append("_no runs yet_\n")
    else:
        lines.append(f"Clean accuracy, scaling factor {factor:g}:\n")
        lines.append(table_vs_f(cells, "cifar10", "gradient_scaling", factor,
                                UNTARGETED_DEFENSES, CIFAR_FRACTIONS, "acc_final",
                                with_break=False) + "\n")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------


def _curve(
    cells: dict[CellKey, list[RunSummary]],
    dataset: str,
    attack: str,
    strength: float | None,
    defense: str,
    fractions: tuple[float, ...],
    attr: str,
) -> tuple[list[float], list[float], list[float]]:
    xs, ys, sds = [], [], []
    for f in fractions:
        vals = [getattr(r, attr) for r in _find(cells, dataset, attack, strength, f, defense)]
        vals = [v for v in vals if v is not None and not math.isnan(v)]
        if vals:
            xs.append(f)
            ys.append(float(np.mean(vals)))
            sds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
    return xs, ys, sds


def _early_auc(run: RunSummary, metric: str) -> float | None:
    return run.auc.get("early", {}).get(metric)


def _legend(ax: Any, **kwargs: Any) -> None:
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, **kwargs)


def write_figures(cells: dict[CellKey, list[RunSummary]], out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    k = CLIENTS_PER_ROUND

    def kf_axis(ax: Any) -> None:
        top = ax.secondary_xaxis("top", functions=(lambda f: f * k, lambda m: m / k))
        top.set_xlabel("expected attackers per round (Kf)")

    panels: tuple[tuple[str, str, float | None, tuple[str, ...], tuple[float, ...]], ...] = (
        ("cifar10", "backdoor", None, CIFAR_DEFENSES, CIFAR_FRACTIONS),
        ("cifar10", "adaptive", REFERENCE_BUDGET, CIFAR_DEFENSES, CIFAR_FRACTIONS),
        ("femnist", "backdoor", None, FEMNIST_DEFENSES, FEMNIST_FRACTIONS),
    )
    for dataset, attack, strength, defenses, fractions in panels:
        fig, (ax_asr, ax_acc) = plt.subplots(1, 2, figsize=(13, 4.8))
        for defense in defenses:
            for ax, attr in ((ax_asr, "asr_final"), (ax_acc, "acc_final")):
                xs, ys, sds = _curve(cells, dataset, attack, strength, defense, fractions, attr)
                if xs:
                    ax.errorbar(xs, ys, yerr=sds, marker="o", capsize=3, label=defense)
        tail = sampling_tail(cells, dataset, fractions)
        ax_tail = ax_asr.twinx()
        ax_tail.plot([t.fraction for t in tail], [t.p_any for t in tail], "k--", alpha=0.5,
                     label=f"P(any round m>={math.ceil(k / 2)})")
        ax_tail.set_ylim(0, 1.05)
        ax_tail.set_ylabel(f"P(some round has m >= {math.ceil(k / 2)} in {NUM_ROUNDS})")
        ax_tail.legend(loc="lower right", fontsize=8)
        ax_asr.axhline(ASR_THRESHOLD, color="grey", lw=0.8, ls=":")
        ax_asr.set_ylim(0, 1.05)
        ax_asr.set_ylabel("converged ASR")
        ax_acc.set_ylabel("converged clean accuracy")
        for ax in (ax_asr, ax_acc):
            ax.set_xlabel("compromised fraction f")
            kf_axis(ax)
        _legend(ax_asr, loc="upper left")
        fig.suptitle(f"{dataset} {ATTACK_NAMES[attack]}: outcome vs compromise rate")
        fig.tight_layout()
        path = out_dir / f"e2-asr-vs-f-{dataset}-{attack}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (attack, strength) in zip(axes, (("backdoor", None), ("adaptive", REFERENCE_BUDGET))):
        for defense, style in (("oracle-wmed", "s--"), ("capped-wmed", "o-")):
            xs, ys, sds = _curve(cells, "cifar10", attack, strength, defense, CIFAR_FRACTIONS, "asr_final")
            if xs:
                ax.errorbar(xs, ys, yerr=sds, fmt=style, capsize=3, label=defense)
        ax.axhline(ASR_THRESHOLD, color="grey", lw=0.8, ls=":")
        ax.set_title(ATTACK_NAMES[attack])
        ax.set_xlabel("compromised fraction f")
        ax.set_ylim(0, 1.05)
        kf_axis(ax)
        _legend(ax)
    axes[0].set_ylabel("converged ASR")
    fig.suptitle("CIFAR-10: perfect detection vs capped-wmed on the weighted-median base")
    fig.tight_layout()
    path = out_dir / "e2-oracle-vs-capped.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    for dataset, attack, strength, _, fractions in panels:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
        for ax, defense in zip(axes, ("undefended", "capped-wmed")):
            runs_by_f = {f: _find(cells, dataset, attack, strength, f, defense) for f in fractions}
            for metric in METRICS:
                present = [f for f in fractions if runs_by_f[f]]
                aucs = [_mean([_early_auc(r, metric) for r in runs_by_f[f]]) for f in present]
                pts = [(x, y) for x, y in zip(present, aucs) if y is not None]
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=_short_metric(metric))
            ax.axhline(0.5, color="grey", lw=0.8, ls=":")
            lo, hi = AUC_WINDOWS["early"]
            ax.set_title(f"{defense}, rounds {lo}–{hi}")
            ax.set_xlabel("compromised fraction f")
            ax.set_ylim(0, 1.02)
            kf_axis(ax)
            _legend(ax)
        axes[0].set_ylabel("AUC (malicious > benign)")
        fig.suptitle(f"{dataset} {ATTACK_NAMES[attack]}: per-metric separability vs f")
        fig.tight_layout()
        path = out_dir / f"e2-auc-vs-f-{dataset}-{attack}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    tail = [t for t in sampling_tail(cells, "cifar10", CIFAR_FRACTIONS) if t.observed_pmf]
    if tail:
        fig, axes = plt.subplots(1, len(tail), figsize=(2.6 * len(tail), 3.2), sharey=True, squeeze=False)
        ms = np.arange(k + 1)
        for ax, t in zip(axes[0], tail):
            ax.bar(ms - 0.2, t.observed_pmf, width=0.4, label="observed")
            ax.bar(ms + 0.2, t.expected_pmf, width=0.4, label="hypergeometric")
            ax.axvline(t.breakdown_m - 0.5, color="grey", lw=0.8, ls=":")
            ax.set_title(f"f={t.fraction:g}")
            ax.set_xlabel("attackers in round")
            ax.set_xlim(-0.6, 8.6)
        axes[0][0].set_ylabel("fraction of rounds")
        axes[0][0].legend(fontsize=7)
        fig.tight_layout()
        path = out_dir / "e2-malicious-count-hist.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, (dataset, attack, strength, _, fractions) in zip(axes, panels):
        for f in fractions:
            ratios = suppression_by_m(_find(cells, dataset, attack, strength, f, "capped-wmed"))
            if ratios:
                ax.plot(list(ratios), list(ratios.values()), marker="o", label=f"f={f:g}")
        ax.axhline(1.0, color="grey", lw=0.8, ls=":")
        ax.set_title(f"{dataset} {ATTACK_NAMES[attack]}, capped-wmed")
        ax.set_xlabel("attackers in round (m)")
        _legend(ax)
    axes[0].set_ylabel("malicious share / (m/K)")
    fig.suptitle("Down-weighting as a function of co-selected attackers")
    fig.tight_layout()
    path = out_dir / "e2-suppression-vs-m.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)
    return written


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _json_dump(cells: dict[CellKey, list[RunSummary]]) -> list[dict[str, Any]]:
    return [{**asdict(key), **asdict(r)} for key, runs in cells.items() for r in runs]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None, help="per-run dump")
    parser.add_argument("--fig-dir", type=Path, default=None)
    parser.add_argument(
        "--breaking-f",
        action="store_true",
        help="print '<experiment> <override>' lines for the extra-seed block and exit",
    )
    args = parser.parse_args(argv)

    results_root: Path = args.results_root.resolve()
    if not results_root.is_dir():
        raise SystemExit(f"results root not found: {results_root}")
    for target in (args.out, args.json, args.fig_dir):
        if target is not None:
            resolved = target.resolve()
            if results_root == resolved or results_root in resolved.parents:
                raise SystemExit(f"refusing to write inside the results tree: {resolved}")

    cells = load_cells(results_root)
    if args.breaking_f:
        for experiment, override in bump_targets(cells):
            print(f"{experiment} {override}")
        return 0

    report = build_report(cells)
    if args.out is not None:
        args.out.write_text(report)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(report)
    if args.json is not None:
        args.json.write_text(json.dumps(_json_dump(cells), indent=2))
        print(f"wrote {args.json}", file=sys.stderr)
    if args.fig_dir is not None:
        for path in write_figures(cells, args.fig_dir):
            print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
