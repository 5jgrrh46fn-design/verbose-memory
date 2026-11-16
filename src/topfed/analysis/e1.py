"""E1 colluding attackers: how co-selected coordination moves each defense.

Reads the per-run ledgers under ``results/`` (``manifest.json``,
``metrics.json`` with the capture trace, ``client_metrics.json``,
``defense_ledger.json`` when the defense or baseline wrote one, and
``collusion_ledger.json`` for coordinated cells). Only the collusion grid is
in scope: CIFAR-10, Dirichlet(0.5), N=100, K=10, 200 rounds, population
f=0.30, forced co-selection of exactly m attackers per round, the reference
corner trigger, no attack window. A run's cell is keyed by its resolved
config (attack and budget, collusion strategy, m, defense), not by its
directory, so any tag or override spelling lands in the right cell.

Sections of the report, per attack (vanilla backdoor, adaptive backdoor):

* **ASR and clean accuracy vs m** per strategy and defense, converged over the
  last 50 rounds, with the **breaking m** (first m whose mean converged ASR
  reaches 0.5) per strategy and defense.
* **Colluder weight share** per round: the mean share and the fraction of
  rounds with share >= 0.5 (the weighted median's breakdown point), from the
  capture trace, falling back to the defense ledger. A plain median has no
  weights and is shown as an equal vote, m/K.
* **Capture trace**: the fraction of coordinates the bloc set, and the
  aggregate's projection on the malicious mean direction as a fraction of the
  malicious mean's length.
* **Benign metric shift**: the two-sample KS distance between the benign
  clients' M1 / M2 / M5 values in a cell and in the independent m=1 cell of
  the same attack and defense, with the signed shift of the benign median;
  once over the rounds before each run's implant (the reference-shift test
  proper) and once over all rounds.
* **Malicious trust rank**: each malicious client's rank among the round's
  cohort by the defense's trust (the TOPSIS score, FoolsGold's alpha, or the
  weight), 0 = most trusted and 1 = least, averaged over rounds; coordinated
  cells also split it into the round leader (the S3 carrier) and the others.

A run whose parameters become non-finite is reported as diverged: its ASR
is undefined and left out of the cell mean, its clean accuracy (chance)
stays in.

``--fig-dir`` writes ASR / clean accuracy vs m as small multiples per
strategy, and the per-round colluder share per strategy and defense.
``--stage-b`` prints the suggested Stage B m values and strategy (the breaking
m of capped-wmed under the worst Stage A strategy, and its neighbours) for
``experiments/sweep_e1.sh``; the choice stays with the operator.

At m=1 every strategy is the independent attacker, so the independent m=1
cell stands in for every strategy's m=1 point. Pooled rows are not
independent samples, so the KS distance is reported without a p-value.

Read-only over ``results/``; every output path is rejected inside the
results tree.

Usage::

    uv run python -m topfed.analysis.e1 --out tables/e1-tables.md
    uv run python -m topfed.analysis.e1 --fig-dir figures --json e1-runs.json
    uv run python -m topfed.analysis.e1 --stage-b
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
from scipy.stats import ks_2samp, rankdata

from topfed.analysis.e2 import _backdoor_attack
from topfed.analysis.e5 import (
    ASR_THRESHOLD,
    CONVERGENCE_WINDOW,
    _defense_label,
    _fmt,
    _malicious_share,
    _mean_sd,
    _partition_label,
    _per_seed,
    _table,
    _window,
    sustained_implant,
)

NUM_CLIENTS = 100
CLIENTS_PER_ROUND = 10
NUM_ROUNDS = 200
FRACTION = 0.30
PARTITION = "dirichlet(0.5)"
M_VALUES = (1, 2, 3, 4, 5)
REFERENCE_BUDGET = 0.6
ATTACKS: tuple[tuple[str, float | None], ...] = (
    ("backdoor", None),
    ("adaptive", REFERENCE_BUDGET),
)
ATTACK_NAMES = {"backdoor": "vanilla backdoor", "adaptive": "adaptive backdoor"}
INDEPENDENT = "independent"
STRATEGIES = (
    INDEPENDENT,
    "clone",
    "budget_split",
    "reference_shift",
    "coordinate_capture",
)
STRATEGY_NAMES = {
    INDEPENDENT: "independent",
    "clone": "S1 clone",
    "budget_split": "S2 budget split",
    "reference_shift": "S3 reference shift",
    "coordinate_capture": "S4 coordinate capture",
}
STAGE_A_COLLUDE = ("clone", "reference_shift")
DEFENSES = (
    "undefended",
    "ScoreWeight",
    "capped-wmed",
    "median",
    "fltrust",
    "foolsgold",
    "agree-capped-wmed",
)
HEADLINE_DEFENSE = "capped-wmed"
NO_WEIGHT_DEFENSES = ("median",)
METRICS = ("m1_cosine_to_mean", "m2_norm_ratio", "m5_max_layer_deviation")
METRIC_NAMES = {"m1_cosine_to_mean": "M1", "m2_norm_ratio": "M2", "m5_max_layer_deviation": "M5"}
# Trust column read from a ledger, first present wins.
TRUST_COLUMNS = ("score", "alpha", "weight")
# Fixed categorical slots per defense (color follows the entity), and a
# single-hue light-to-dark ramp for the ordinal m.
DEFENSE_COLORS = {
    "undefended": "#2a78d6",
    "ScoreWeight": "#eb6834",
    "capped-wmed": "#1baf7a",
    "median": "#eda100",
    "fltrust": "#e87ba4",
    "foolsgold": "#008300",
    "agree-capped-wmed": "#4a3aa7",
}
M_COLORS = {1: "#86b6ef", 2: "#5598e7", 3: "#2a78d6", 4: "#1c5cab", 5: "#104281"}
SHARE_SMOOTHING = 5


@dataclass(frozen=True)
class CellKey:
    attack: str
    budget: float | None
    strategy: str
    top_fraction: float | None
    m: int
    defense: str


@dataclass
class RunSummary:
    seed: int
    asr_final: float | None
    acc_final: float
    implant_round: int | None
    share: list[float]
    captured: list[float]
    projection_ratio: list[float]
    benign: dict[str, list[float]]
    malicious_rank: float | None
    leader_rank: float | None
    follower_rank: float | None
    # Benign metric values from the rounds before the run's implant (all
    # rounds when it never implants), for the reference-shift test.
    benign_pre: dict[str, list[float]] = field(default_factory=dict)
    # True when the model's parameters became non-finite; ASR is then
    # undefined (``asr_final`` is None) while clean accuracy is chance.
    diverged: bool = False
    source: str = ""

    @property
    def mean_share(self) -> float | None:
        return _nanmean(self.share)

    @property
    def majority_rounds(self) -> float | None:
        xs = [x for x in self.share if not math.isnan(x)]
        return float(np.mean([x >= 0.5 for x in xs])) if xs else None

    @property
    def mean_captured(self) -> float | None:
        return _nanmean(self.captured)

    @property
    def mean_projection_ratio(self) -> float | None:
        return _nanmean(self.projection_ratio)


def _nanmean(values: list[float]) -> float | None:
    xs = [x for x in values if x is not None and not math.isnan(x)]
    return float(np.mean(xs)) if xs else None


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------


def _in_scope(cfg: dict[str, Any]) -> bool:
    fed = cfg["federation"]
    return (
        str(cfg["dataset"]["name"]) == "cifar10"
        and int(fed["num_clients"]) == NUM_CLIENTS
        and int(fed["clients_per_round"]) == CLIENTS_PER_ROUND
        and int(fed["num_rounds"]) == NUM_ROUNDS
        and fed.get("force_malicious_per_round") is not None
        and abs(float(cfg["attack"]["assignment"]["fraction"]) - FRACTION) < 1e-9
        and _partition_label(cfg) == PARTITION
        and int(cfg["dataset"].get("pad_border", 0) or 0) == 0
        and _window(cfg) == (1, None)
    )


def cell_key(manifest: dict[str, Any]) -> CellKey | None:
    """Build the E1 cell key from a manifest; ``None`` for out-of-scope runs."""
    cfg = manifest["config"]
    if cfg["attack"].get("kind") != "backdoor" or not _in_scope(cfg):
        return None
    found = _backdoor_attack(cfg)
    if found is None:
        return None
    attack, budget = found
    collusion = cfg["attack"].get("collusion") or {}
    strategy = str(collusion.get("strategy", INDEPENDENT))
    top = collusion.get("top_fraction") if strategy == "coordinate_capture" else None
    return CellKey(
        attack=attack,
        budget=budget,
        strategy=strategy,
        top_fraction=None if top is None else float(top),
        m=int(cfg["federation"]["force_malicious_per_round"]),
        defense=_defense_label(cfg),
    )


def _trust_ranks(
    ledger: list[dict[str, Any]] | None, leaders: dict[int, str]
) -> tuple[float | None, float | None, float | None]:
    """Mean normalised trust rank of malicious, leader and other malicious rows."""
    if not ledger:
        return None, None, None
    column = next((c for c in TRUST_COLUMNS if c in ledger[0]), None)
    if column is None:
        return None, None, None
    by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in ledger:
        by_round[int(row["round"])].append(row)
    mal: list[float] = []
    lead: list[float] = []
    follow: list[float] = []
    for r, rows in by_round.items():
        if len(rows) < 2:
            continue
        ranks = rankdata([-float(x[column]) for x in rows], method="average")
        norm = (ranks - 1.0) / (len(rows) - 1.0)
        for row, v in zip(rows, norm, strict=True):
            if not row["is_malicious"]:
                continue
            mal.append(float(v))
            if r in leaders:
                (lead if row["client_id"] == leaders[r] else follow).append(float(v))
    return _nanmean(mal), _nanmean(lead), _nanmean(follow)


def summarise_run(
    seed: int,
    metrics: list[dict[str, Any]],
    client_rows: list[dict[str, Any]],
    ledger: list[dict[str, Any]] | None,
    collusion_ledger: list[dict[str, Any]] | None,
    k: int = CLIENTS_PER_ROUND,
) -> RunSummary:
    rounds = np.array([int(m["round"]) for m in metrics])
    acc = np.array([float(m.get("test_accuracy", math.nan)) for m in metrics])
    asr = np.array([float(m.get("asr_accuracy", math.nan)) for m in metrics])
    has_asr = bool(metrics) and "asr_accuracy" in metrics[-1]
    loss = np.array([float(m.get("test_loss", m.get("val_loss", 0.0)) or 0.0) for m in metrics])
    blown = np.flatnonzero(~np.isfinite(loss))
    diverged = blown.size > 0
    # A model with non-finite parameters predicts one class, so its ASR
    # reads 0 or 1 by accident; treat it as undefined from that round on.
    if diverged:
        asr = asr.copy()
        asr[blown[0] :] = math.nan
    implant = sustained_implant(np.nan_to_num(asr, nan=0.0)) if has_asr else None
    last_pre = min(
        implant if implant is not None else len(metrics) + 1,
        int(rounds[blown[0]]) if diverged else len(metrics) + 1,
    )

    def column(key: str) -> list[float]:
        return [
            math.nan if m.get(key) is None else float(m[key]) for m in metrics
        ]

    if metrics and "trace_malicious_share" in metrics[0]:
        share = column("trace_malicious_share")
    elif ledger and "weight" in ledger[0]:
        per_round: dict[int, int] = defaultdict(int)
        for row in client_rows:
            per_round[int(row["round"])] += int(bool(row["is_malicious"]))
        counts = np.array([per_round[int(r)] for r in rounds], dtype=int)
        share = [float(x) for x in _malicious_share(rounds, ledger, counts, k)]
    else:
        share = [math.nan] * len(metrics)
    captured = column("trace_captured_fraction")
    proj = column("trace_projection")
    norm = column("trace_malicious_norm")
    ratio = [
        p / n if not (math.isnan(p) or math.isnan(n)) and n > 0 else math.nan
        for p, n in zip(proj, norm, strict=True)
    ]
    benign_rows = [r for r in client_rows if not r["is_malicious"]]
    benign = {
        metric: [float(r[metric]) for r in benign_rows if r.get(metric) is not None]
        for metric in METRICS
    }
    benign_pre = {
        metric: [
            float(r[metric])
            for r in benign_rows
            if r.get(metric) is not None and int(r["round"]) < last_pre
        ]
        for metric in METRICS
    }
    leaders = {int(r["round"]): str(r["leader"]) for r in (collusion_ledger or [])}
    mal_rank, lead_rank, follow_rank = _trust_ranks(ledger, leaders)
    return RunSummary(
        seed=seed,
        asr_final=(
            float(np.nanmean(asr[-CONVERGENCE_WINDOW:]))
            if has_asr and not diverged
            else None
        ),
        acc_final=float(np.nanmean(acc[-CONVERGENCE_WINDOW:])),
        implant_round=implant,
        share=share,
        captured=captured,
        projection_ratio=ratio,
        benign=benign,
        malicious_rank=mal_rank,
        leader_rank=lead_rank if leaders else None,
        follower_rank=follow_rank if leaders else None,
        benign_pre=benign_pre,
        diverged=diverged,
    )


def _read(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def load_cells(results_root: Path) -> dict[CellKey, list[RunSummary]]:
    """Walk ``results/`` and group completed in-scope runs by cell.

    When one config sits under more than one ``<label>/<hash>`` directory,
    each seed comes from the directory with the most completed seeds, ties
    by path, and the others are reported on stderr.
    """
    candidates: dict[tuple[CellKey, int], list[Path]] = defaultdict(list)
    completed: dict[Path, int] = defaultdict(int)
    for manifest_path in sorted(results_root.glob("*/*/seed_*/manifest.json")):
        run_dir = manifest_path.parent
        if not (run_dir / "metrics.json").exists():
            continue
        key = cell_key(json.loads(manifest_path.read_text()))
        if key is None:
            continue
        completed[run_dir.parent] += 1
        seed = int(run_dir.name.removeprefix("seed_"))
        candidates[(key, seed)].append(run_dir)

    cells: dict[CellKey, list[RunSummary]] = defaultdict(list)
    for (key, seed), run_dirs in candidates.items():
        ranked = sorted(run_dirs, key=lambda d: (-completed[d.parent], str(d)))
        for loser in ranked[1:]:
            print(
                f"note: same config in {loser.relative_to(results_root)}; "
                f"seed {seed} taken from {ranked[0].relative_to(results_root)}",
                file=sys.stderr,
            )
        run_dir = ranked[0]
        metrics = json.loads((run_dir / "metrics.json").read_text())
        if len(metrics) != NUM_ROUNDS:
            continue
        summary = summarise_run(
            seed,
            metrics,
            _read(run_dir / "client_metrics.json") or [],
            _read(run_dir / "defense_ledger.json"),
            _read(run_dir / "collusion_ledger.json"),
        )
        summary.source = str(run_dir.relative_to(results_root))
        cells[key].append(summary)
    return {k: sorted(v, key=lambda r: r.seed) for k, v in cells.items()}


# ----------------------------------------------------------------------------
# Derived quantities
# ----------------------------------------------------------------------------


def find(
    cells: dict[CellKey, list[RunSummary]],
    attack: str,
    budget: float | None,
    strategy: str,
    m: int,
    defense: str,
    top_fraction: float | None = None,
) -> list[RunSummary]:
    """A cell's runs; the independent m=1 cell stands in for any strategy at m=1."""
    runs = cells.get(CellKey(attack, budget, strategy, top_fraction, m, defense), [])
    if not runs and m == 1 and strategy != INDEPENDENT:
        runs = cells.get(CellKey(attack, budget, INDEPENDENT, None, 1, defense), [])
    return runs


def mean_asr(runs: list[RunSummary]) -> float | None:
    return _nanmean([r.asr_final for r in runs if r.asr_final is not None])


def breaking_m(
    cells: dict[CellKey, list[RunSummary]],
    attack: str,
    budget: float | None,
    strategy: str,
    defense: str,
    top_fraction: float | None = None,
) -> tuple[int | None, bool]:
    """First m (ascending) whose mean converged ASR reaches the threshold.

    Returns ``(m, complete)``; ``complete`` is False when a level at or
    below the returned m (or any level, when none breaks) has no runs.
    """
    complete = True
    for m in M_VALUES:
        runs = find(cells, attack, budget, strategy, m, defense, top_fraction)
        if not runs:
            complete = False
            continue
        asr = mean_asr(runs)
        if asr is not None and asr >= ASR_THRESHOLD:
            return m, complete
    return None, complete


def ks_shift(
    runs: list[RunSummary],
    reference: list[RunSummary],
    metric: str,
    window: str = "all",
) -> tuple[float, float] | None:
    """KS distance and signed median shift of pooled benign ``metric`` values.

    ``window`` is ``"all"`` (every round) or ``"pre"`` (rounds before each
    run's implant, on both sides).
    """
    attr = "benign_pre" if window == "pre" else "benign"
    a = [x for r in runs for x in getattr(r, attr).get(metric, [])]
    b = [x for r in reference for x in getattr(r, attr).get(metric, [])]
    if not a or not b:
        return None
    return float(ks_2samp(a, b).statistic), float(np.median(a) - np.median(b))


def neighbours(m: int | None) -> tuple[int, ...]:
    """The breaking m and its two neighbours, kept inside the m grid."""
    lo, hi = M_VALUES[0], M_VALUES[-1]
    if m is None:
        return tuple(range(hi - 2, hi + 1))
    start = min(max(m - 1, lo), hi - 2)
    return tuple(range(start, start + 3))


@dataclass
class StageBSuggestion:
    attack: str
    budget: float | None
    strategy: str | None
    breaking: int | None
    ms: tuple[int, ...]
    complete: bool
    notes: list[str] = field(default_factory=list)


def stage_b_suggestion(cells: dict[CellKey, list[RunSummary]]) -> list[StageBSuggestion]:
    """Per attack: the worst Stage A strategy against capped-wmed and its m values.

    Worst = lowest breaking m; ties (including never breaking) go to the
    higher mean ASR at the largest m.
    """
    out: list[StageBSuggestion] = []
    for attack, budget in ATTACKS:
        best: tuple[float, float, str, int | None, bool] | None = None
        notes: list[str] = []
        for strategy in STAGE_A_COLLUDE:
            m, complete = breaking_m(cells, attack, budget, strategy, HEADLINE_DEFENSE)
            top = mean_asr(find(cells, attack, budget, strategy, M_VALUES[-1], HEADLINE_DEFENSE))
            notes.append(
                f"{strategy}: breaks at {'never' if m is None else f'm={m}'}"
                f"{'' if complete else ' (grid incomplete)'}, ASR at m={M_VALUES[-1]} {_fmt(top)}"
            )
            rank = (math.inf if m is None else float(m), -(top if top is not None else -1.0))
            if best is None or rank < best[:2]:
                best = (*rank, strategy, m, complete)
        assert best is not None
        _, _, strategy, m, complete = best
        out.append(
            StageBSuggestion(attack, budget, strategy, m, neighbours(m), complete, notes)
        )
    return out


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------


def _strategies_present(cells: dict[CellKey, list[RunSummary]], attack: str) -> list[str]:
    found = {k.strategy for k in cells if k.attack == attack}
    return [s for s in STRATEGIES if s in found or s == INDEPENDENT]


def _defenses_present(cells: dict[CellKey, list[RunSummary]], attack: str) -> list[str]:
    found = {k.defense for k in cells if k.attack == attack}
    return [d for d in DEFENSES if d in found] + sorted(found - set(DEFENSES))


def _m_header(first: list[str]) -> list[str]:
    return first + [f"m={m}" for m in M_VALUES]


def _grid(
    cells: dict[CellKey, list[RunSummary]],
    attack: str,
    budget: float | None,
    render: Any,
    by_metric: bool = False,
) -> list[list[str]]:
    rows: list[list[str]] = []
    for strategy in _strategies_present(cells, attack):
        for defense in _defenses_present(cells, attack):
            per_m = [find(cells, attack, budget, strategy, m, defense) for m in M_VALUES]
            if strategy != INDEPENDENT and not any(per_m[1:]):
                continue
            if not any(per_m):
                continue
            rows.append(
                [STRATEGY_NAMES[strategy], defense]
                + [render(runs, m, strategy, defense) if runs else "–" for m, runs in zip(M_VALUES, per_m)]
            )
    return rows


def _outcome_cell(attr: str) -> Any:
    def render(runs: list[RunSummary], m: int, strategy: str, defense: str) -> str:
        blown = sum(r.diverged for r in runs)
        n = f"{len(runs)}, {blown} diverged" if blown else str(len(runs))
        return f"{_mean_sd([getattr(r, attr) for r in runs])} ({n})"

    return render


def _implant_cell(runs: list[RunSummary], m: int, strategy: str, defense: str) -> str:
    return _per_seed(
        [r.implant_round for r in runs]
    ) + ("" if not any(r.diverged for r in runs) else " (diverged: "
         + ",".join(str(r.seed) for r in runs if r.diverged) + ")")


def _share_cell(runs: list[RunSummary], m: int, strategy: str, defense: str) -> str:
    if defense in NO_WEIGHT_DEFENSES:
        return f"{m / CLIENTS_PER_ROUND:.2f} (equal vote)"
    mean = _nanmean([r.mean_share for r in runs if r.mean_share is not None])
    maj = _nanmean([r.majority_rounds for r in runs if r.majority_rounds is not None])
    return f"{_fmt(mean)} / {_fmt(maj)}"


def _trace_cell(runs: list[RunSummary], m: int, strategy: str, defense: str) -> str:
    cap = _nanmean([r.mean_captured for r in runs if r.mean_captured is not None])
    proj = _nanmean([r.mean_projection_ratio for r in runs if r.mean_projection_ratio is not None])
    return f"{_fmt(cap)} / {_fmt(proj, 2)}"


def _rank_cell(runs: list[RunSummary], m: int, strategy: str, defense: str) -> str:
    mal = _nanmean([r.malicious_rank for r in runs if r.malicious_rank is not None])
    if mal is None:
        return "–"
    lead = _nanmean([r.leader_rank for r in runs if r.leader_rank is not None])
    follow = _nanmean([r.follower_rank for r in runs if r.follower_rank is not None])
    if lead is None or m == 1:
        return _fmt(mal, 2)
    return f"{_fmt(mal, 2)} ({_fmt(lead, 2)} / {_fmt(follow, 2)})"


def table_outcome(cells: dict[CellKey, list[RunSummary]], attack: str, budget: float | None, attr: str) -> str:
    rows = _grid(cells, attack, budget, _outcome_cell(attr))
    return _table(_m_header(["strategy", "defense"]), rows) if rows else "_no runs_"


def table_breaking(cells: dict[CellKey, list[RunSummary]], attack: str, budget: float | None) -> str:
    strategies = [s for s in _strategies_present(cells, attack)]
    rows: list[list[str]] = []
    for defense in _defenses_present(cells, attack):
        row = [defense]
        for strategy in strategies:
            m, complete = breaking_m(cells, attack, budget, strategy, defense)
            text = "never" if m is None else str(m)
            row.append(text if complete else f"{text}*")
        rows.append(row)
    if not rows:
        return "_no runs_"
    return _table(["defense"] + [STRATEGY_NAMES[s] for s in strategies], rows)


def table_ks(
    cells: dict[CellKey, list[RunSummary]], attack: str, budget: float | None, window: str = "all"
) -> str:
    rows: list[list[str]] = []
    for strategy in _strategies_present(cells, attack):
        for defense in _defenses_present(cells, attack):
            reference = find(cells, attack, budget, INDEPENDENT, 1, defense)
            if not reference:
                continue
            for metric in METRICS:
                cells_m: list[str] = []
                any_run = False
                for m in M_VALUES:
                    runs = find(cells, attack, budget, strategy, m, defense)
                    shift = ks_shift(runs, reference, metric, window) if runs else None
                    if runs and (strategy == INDEPENDENT or m > 1):
                        any_run = True
                    cells_m.append("–" if shift is None else f"{shift[0]:.3f} ({shift[1]:+.3f})")
                if any_run and (strategy == INDEPENDENT or any(c != "–" for c in cells_m[1:])):
                    rows.append([STRATEGY_NAMES[strategy], defense, METRIC_NAMES[metric]] + cells_m)
    return _table(_m_header(["strategy", "defense", "metric"]), rows) if rows else "_no runs_"


def build_report(cells: dict[CellKey, list[RunSummary]]) -> str:
    out: list[str] = [
        "# E1 colluding attackers: tables",
        "",
        "Generated by `uv run python -m topfed.analysis.e1`. CIFAR-10, "
        "Dirichlet(0.5), N=100, K=10, f=0.30, exactly m attackers per round. "
        "Cells are `mean ± sd (n)` over seeds; converged = last "
        f"{CONVERGENCE_WINDOW} rounds. m=1 is the independent attacker for every "
        "strategy.",
    ]
    for attack, budget in ATTACKS:
        name = ATTACK_NAMES[attack] + ("" if budget is None else f" (budget {budget:g})")
        if not any(k.attack == attack and k.budget == budget for k in cells):
            out += ["", f"## {name}", "", "_no runs_"]
            continue
        out += [
            "",
            f"## {name}",
            "",
            "### Converged ASR vs m",
            "",
            table_outcome(cells, attack, budget, "asr_final"),
            "",
            "### Converged clean accuracy vs m",
            "",
            table_outcome(cells, attack, budget, "acc_final"),
            "",
            "### Implant round (per seed)",
            "",
            "First round from which ASR stays >= 0.5 for 10 rounds. A diverged "
            "run (non-finite parameters) is excluded from the ASR means above; "
            "its clean accuracy (chance) is kept.",
            "",
            _table(_m_header(["strategy", "defense"]), _grid(cells, attack, budget, _implant_cell)),
            "",
            f"### Breaking m (first m with mean ASR >= {ASR_THRESHOLD})",
            "",
            "`*` = a level at or below it has no runs yet, so the first crossing "
            "is not established.",
            "",
            table_breaking(cells, attack, budget),
            "",
            "### Colluder weight share",
            "",
            "Mean per-round share of the aggregation weight held by the bloc / "
            "fraction of rounds with share >= 0.5.",
            "",
            _table(_m_header(["strategy", "defense"]), _grid(cells, attack, budget, _share_cell)),
            "",
            "### Capture trace",
            "",
            "Mean fraction of coordinates where the aggregate lies inside the bloc's "
            "range and outside the honest IQR / mean projection of the aggregate "
            "on the malicious mean direction, as a fraction of that mean's length.",
            "",
            _table(_m_header(["strategy", "defense"]), _grid(cells, attack, budget, _trace_cell)),
            "",
            "### Benign metric shift vs the independent m=1 cell, before implant",
            "",
            "KS distance of the pooled benign values (signed shift of the benign "
            "median), both sides restricted to the rounds before each run's "
            "implant, so the shift is measured while the bloc is still being "
            "resisted. A positive M2 / M5 or negative M1 shift means honest "
            "clients look more anomalous.",
            "",
            table_ks(cells, attack, budget, "pre"),
            "",
            "### Benign metric shift vs the independent m=1 cell, all rounds",
            "",
            "Same statistic over every round; after an implant the attackers' "
            "updates shrink and the cohort references move for that reason.",
            "",
            table_ks(cells, attack, budget, "all"),
            "",
            "### Malicious trust rank",
            "",
            "0 = most trusted in the round, 1 = least; a random ranking averages "
            "0.5. Coordinated cells add (leader / others); under S3 the leader is "
            "the carrier. Defenses without a ledger are absent.",
            "",
            _table(_m_header(["strategy", "defense"]), _grid(cells, attack, budget, _rank_cell)),
        ]
    suggestions = stage_b_suggestion(cells)
    out += ["", "## Stage B suggestion", ""]
    for s in suggestions:
        out.append(
            f"- {ATTACK_NAMES[s.attack]}: worst strategy vs {HEADLINE_DEFENSE} "
            f"`{s.strategy}`, breaking m "
            f"{'never' if s.breaking is None else s.breaking}"
            f"{'' if s.complete else ' (grid incomplete)'}, m values {list(s.ms)}. "
            + "; ".join(s.notes)
        )
    out.append("")
    return "\n".join(out)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or values.size < width:
        return values
    kernel = np.ones(width) / width
    padded = np.concatenate([np.full(width - 1, values[0]), values])
    return np.convolve(padded, kernel, mode="valid")


def write_figures(cells: dict[CellKey, list[RunSummary]], out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for attack, budget in ATTACKS:
        strategies = _strategies_present(cells, attack)
        defenses = _defenses_present(cells, attack)
        if not any(k.attack == attack for k in cells):
            continue
        fig, axes = plt.subplots(
            2, len(strategies), figsize=(3.4 * len(strategies), 6.4),
            sharex=True, sharey="row", squeeze=False,
        )
        for col, strategy in enumerate(strategies):
            for row, attr in enumerate(("asr_final", "acc_final")):
                ax = axes[row][col]
                for defense in defenses:
                    xs, ys, sds = [], [], []
                    for m in M_VALUES:
                        runs = find(cells, attack, budget, strategy, m, defense)
                        vals = [getattr(r, attr) for r in runs if getattr(r, attr) is not None]
                        if vals:
                            xs.append(m)
                            ys.append(float(np.mean(vals)))
                            sds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
                    if xs:
                        ax.errorbar(
                            xs, ys, yerr=sds, marker="o", ms=5, lw=2, capsize=3,
                            color=DEFENSE_COLORS.get(defense, "#6b6b66"), label=defense,
                        )
                ax.grid(True, color="#e4e3dc", lw=0.6)
                ax.set_xticks(M_VALUES)
                if row == 0:
                    ax.axhline(ASR_THRESHOLD, color="#8a897f", lw=0.8, ls=":")
                    ax.set_ylim(0, 1.05)
                    ax.set_title(STRATEGY_NAMES[strategy], fontsize=10)
                else:
                    ax.set_xlabel("attackers per round (m)")
        axes[0][0].set_ylabel("converged ASR")
        axes[1][0].set_ylabel("converged clean accuracy")
        handles, labels = axes[0][0].get_legend_handles_labels()
        for ax in axes.ravel():
            h, lab = ax.get_legend_handles_labels()
            if len(h) > len(handles):
                handles, labels = h, lab
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False)
        fig.suptitle(f"CIFAR-10 f=0.30 {ATTACK_NAMES[attack]}: outcome vs co-selected attackers")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        path = out_dir / f"e1-asr-vs-m-{attack}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

        weighted = [d for d in defenses if d not in NO_WEIGHT_DEFENSES]
        for strategy in strategies:
            present = [
                d for d in weighted
                if any(find(cells, attack, budget, strategy, m, d) for m in M_VALUES[1:])
            ]
            if strategy == INDEPENDENT:
                present = [d for d in weighted if any(find(cells, attack, budget, strategy, m, d) for m in M_VALUES)]
            if not present:
                continue
            fig, axes2 = plt.subplots(
                1, len(present), figsize=(3.6 * len(present), 3.4), sharey=True, squeeze=False
            )
            for ax, defense in zip(axes2[0], present):
                for m in M_VALUES:
                    runs = find(cells, attack, budget, strategy, m, defense)
                    if not runs:
                        continue
                    stack = np.array([r.share for r in runs], dtype=float)
                    if np.isnan(stack).all():
                        continue
                    mean = _smooth(np.nanmean(stack, axis=0), SHARE_SMOOTHING)
                    ax.plot(np.arange(1, mean.size + 1), mean, lw=2, color=M_COLORS[m], label=f"m={m}")
                ax.axhline(0.5, color="#8a897f", lw=0.8, ls=":")
                ax.set_ylim(0, 1.0)
                ax.set_title(defense, fontsize=10)
                ax.set_xlabel("round")
                ax.grid(True, color="#e4e3dc", lw=0.6)
            if axes2[0][0].get_legend_handles_labels()[0]:
                axes2[0][0].legend(fontsize=7, frameon=False)
            axes2[0][0].set_ylabel(f"colluder weight share ({SHARE_SMOOTHING}-round mean)")
            fig.suptitle(f"{ATTACK_NAMES[attack]}, {STRATEGY_NAMES[strategy]}: bloc share per round")
            fig.tight_layout()
            path = out_dir / f"e1-share-{attack}-{strategy}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(path)
    return written


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _json_dump(cells: dict[CellKey, list[RunSummary]]) -> list[dict[str, Any]]:
    return [
        {**asdict(key), **{k: v for k, v in asdict(r).items() if k not in ("benign", "benign_pre")}}
        for key, runs in cells.items()
        for r in runs
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None, help="per-run dump")
    parser.add_argument("--fig-dir", type=Path, default=None)
    parser.add_argument(
        "--stage-b",
        action="store_true",
        help="print suggested STAGE_B_M / STAGE_B_STRATEGIES for sweep_e1.sh and exit",
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
    if args.stage_b:
        suggestions = stage_b_suggestion(cells)
        for s in suggestions:
            print(f"{s.attack}: " + "; ".join(s.notes), file=sys.stderr)
        ms = sorted({m for s in suggestions for m in s.ms})
        strategies = sorted({s.strategy for s in suggestions if s.strategy})
        print(f'STAGE_B_M="{" ".join(map(str, ms))}" STAGE_B_STRATEGIES="{" ".join(strategies)}"')
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
