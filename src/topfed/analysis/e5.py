"""E5 diagnostics: capture, recovery, and the levers that move them.

Reads the per-run ledgers under ``results/`` (``manifest.json``,
``metrics.json``, ``client_metrics.json``, and ``defense_ledger.json`` when a
defense wrote one) and emits one Markdown table per diagnostic block for the
backdoor experiments at a given compromise fraction:

* **REF** — the canonical cells (uniform sampling, K=10, corner trigger,
  the dataset's native partition), shown in every block for comparison.
* **D1** oracle detection, **D2** delayed start, **D3** forced co-selection,
  **D4** heterogeneity dial, **D6** attack-strength dial, **D7** persistence
  window, **D8** trigger placement, **D9** sampling-tail levers (K, carried
  reputation).

A run's block is inferred from its resolved config, not from its directory
name, so a cell reached by any override spelling lands in the right table.
Every row reports, across seeds: converged attack success rate and clean
accuracy, the sustained-implant round, the first round in which the
malicious aggregation-weight share reached the weighted median's breakdown
point (0.5), the maximum share seen, the attack success rate 20 rounds after
the first crossing (recovery), and how many seeds recovered by the end.

Read-only over ``results/``; ``--out`` is rejected inside the results tree.

Usage::

    uv run python -m topfed.analysis.e5 --out tables/e5-tables.md
    uv run python -m topfed.analysis.e5 --fraction 0.2 --json e5-runs.json
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

CONVERGENCE_WINDOW = 50
SUSTAIN_ROUNDS = 10
ASR_THRESHOLD = 0.5
FIRST_VALID_ROUND = 5  # round-1 ASR of a collapsed model is not an implant
RECOVERY_LAG = 20
TIMING_KEYS = (
    "client_train_s",
    "scoring_s",
    "aggregation_s",
    "server_train_s",
    "evaluation_s",
    "round_s",
)

# Per-dataset reference settings; a run deviating from these belongs to the
# block whose knob it moved.
REFERENCE = {
    "femnist": {"partition": "natural", "trigger_amplitude": -5.0},
    "cifar10": {"partition": "dirichlet(0.5)", "trigger_amplitude": 1.0},
    "cifar100": {"partition": "dirichlet(0.5)", "trigger_amplitude": 1.0},
    "cifar100_coarse": {"partition": "dirichlet(0.5)", "trigger_amplitude": 1.0},
}
REFERENCE_COMMON = {
    "k": 10,
    "trigger_position": "bottom_right",
    "pad_border": 0,
    "trigger_size": 6,
    "batch_mode": "augment",
    "scaling": True,
    "window": (1, None),
}

# Integrator keyword defaults; a run whose integrator deviates gets the
# deviation appended to its defense label so sweeps never merge with the
# canonical cell.
INTEGRATOR_DEFAULTS: dict[str, dict[str, Any]] = {
    "CappedScoreWeight": {"cap": 1.0, "alpha": 1.0},
    "ScoreWeight": {"alpha": 1.0},
    "TemporalReputation": {"alpha": 0.5},
    "HardFilter": {"threshold": 0.5, "alpha": 1.0},
    "Quarantine": {"threshold": 0.5, "quarantine_limit": 3, "alpha": 1.0},
    "AdaptiveParticipation": {"alpha": 1.0},
    "OracleFilter": {},
    "ServerGatedCappedScoreWeight": {"cap": 1.0, "alpha": 1.0},
    "M4BandCappedScoreWeight": {"cap": 1.0, "alpha": 1.0, "quantile": 0.75},
}
VANILLA_ATTACK = "model_replacement"

DEFENSE_NAMES = {
    ("CappedScoreWeight", "weighted_median"): "capped-wmed",
    ("ScoreWeight", "weighted_median"): "wmed",
    ("ScoreWeight", "mean"): "ScoreWeight",
    ("TemporalReputation", "weighted_median"): "temporal-wmed",
    ("TemporalReputation", "mean"): "TemporalReputation",
    ("OracleFilter", "weighted_median"): "oracle-wmed",
    ("OracleFilter", "mean"): "oracle-mean",
    ("HardFilter", "mean"): "HardFilter",
    ("Quarantine", "mean"): "Quarantine",
    ("AdaptiveParticipation", "mean"): "AdaptiveParticipation",
    ("ServerGatedCappedScoreWeight", "weighted_median"): "gated-capped-wmed",
    ("M4BandCappedScoreWeight", "weighted_median"): "m4band-capped-wmed",
}
BASELINE_NAMES = {
    "coordinate_median": "median",
    "fltrust_median": "fltrust-median",
    "fltrust_median_capped": "fltrust-median-capped",
    "backdoor_indicator": "indicator",
    "clipped_mean": "clipped-mean",
}
# Config defaults of the parameterised baselines; a cell that deviates gets
# the deviation in its label, as integrators do.
BASELINE_DEFAULTS: dict[str, dict[str, Any]] = {
    "krum": {"num_byzantine": 2, "num_selected": 1},
    "trimmed_mean": {"trim_ratio": 0.2},
    "foolsgold": {"kappa": 1.0},
    "flame": {"noise_factor": 0.001, "min_cluster_size": None},
    "mesas": {"alpha": 0.03, "center": "median"},
    "fltrust_median_capped": {"cap": 1.0},
    "clipped_mean": {"bound": None},
    "backdoor_indicator": {
        "threshold": 95.0, "retrain_epochs": 10, "lr": 0.005, "momentum": 0.9,
        "weight_decay": 0.0005, "lr_gamma": 0.8, "mu": 0.4, "batch_size": 64,
        "eta": 1.0, "start_round": 1, "ood_samples": None,
    },
}
# Prefix for a defense whose scorer adds the server-cosine criterion.
ROOTED_SCORER_PREFIX = "rooted-"
# Prefix for a defense whose scorer adds the pairwise-agreement criterion.
AGREEMENT_SCORER_PREFIX = "agree-"


@dataclass(frozen=True)
class CellKey:
    """Everything that distinguishes one E5 cell from another."""

    dataset: str
    fraction: float
    defense: str
    forced_m: int | None
    k: int
    partition: str
    trigger_position: str
    pad_border: int
    trigger_size: int
    trigger_amplitude: float
    batch_mode: str
    poisoning_rate: float
    attack_variant: str
    window: tuple[int, int | None]
    tag: str | None

    @property
    def scaling(self) -> bool:
        return self.attack_variant != "none"

    @property
    def is_vanilla_attack(self) -> bool:
        """The plain scaled backdoor, or the same with scaling switched off."""
        return self.attack_variant in (VANILLA_ATTACK, "none")

    @property
    def block(self) -> str:
        """Infer the diagnostic block from the knobs this cell moved."""
        ref = {**REFERENCE_COMMON, **REFERENCE.get(self.dataset, {})}
        blocks: list[str] = []
        if self.forced_m is not None:
            blocks.append("D3")
        if (
            self.trigger_position != ref["trigger_position"]
            or self.pad_border != ref["pad_border"]
        ):
            blocks.append("D8")
        start, end = self.window
        if end is not None:
            blocks.append("D7")
        elif start > 1:
            blocks.append("D2")
        if self.k != ref["k"] or self.defense == "temporal-wmed":
            blocks.append("D9")
        if self.defense.startswith("oracle"):
            blocks.append("D1")
        if self.partition != ref.get("partition"):
            blocks.append("D4")
        strength_moved = (
            self.trigger_amplitude != ref.get("trigger_amplitude")
            or self.batch_mode != ref["batch_mode"]
            or self.scaling != ref["scaling"]
            or (self.trigger_size != ref["trigger_size"] and self.pad_border == 0)
        )
        if strength_moved:
            blocks.append("D6")
        return "+".join(blocks) if blocks else "REF"


@dataclass
class RunSummary:
    """Per-seed quantities the E5 tables are built from."""

    seed: int
    num_rounds: int
    asr_final: float
    acc_final: float
    implant_round: int | None
    first_cross: int | None
    asr_after_cross: float | None
    recovered: bool | None
    max_malicious_share: float
    first_majority_round: int | None
    majority_rounds: int
    m_at_implant: int | None
    share_at_implant: float | None
    max_m: int
    asr_at: dict[str, float] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------


def _defense_label(cfg: dict[str, Any]) -> str:
    defense = cfg.get("defense")
    if defense:
        integrator = dict(defense["integrator"])
        target = str(integrator.pop("_target_")).rsplit(".", 1)[-1]
        base = str(defense.get("base", "mean"))
        label = DEFENSE_NAMES.get((target, base), f"{target}/{base}")
        defaults = INTEGRATOR_DEFAULTS.get(target)
        deviations = {
            k: v
            for k, v in integrator.items()
            if defaults is None or defaults.get(k) != v
        }
        if deviations:
            params = ",".join(f"{k}={v}" for k, v in sorted(deviations.items()))
            label = f"{label}({params})"
        if (defense.get("scorer") or {}).get("server_cosine", False):
            label = f"{ROOTED_SCORER_PREFIX}{label}"
        if (defense.get("scorer") or {}).get("pairwise_agreement", False):
            label = f"{AGREEMENT_SCORER_PREFIX}{label}"
        return label
    baseline = cfg.get("baseline")
    if baseline:
        name = str(baseline["name"])
        label = BASELINE_NAMES.get(name, name)
        defaults = BASELINE_DEFAULTS.get(name)
        if defaults is not None:
            deviations = {
                k: v for k, v in baseline.items()
                if k != "name" and defaults.get(k) != v
            }
            if deviations:
                params = ",".join(f"{k}={v}" for k, v in sorted(deviations.items()))
                label = f"{label}({params})"
        return label
    return "undefended"


def _partition_label(cfg: dict[str, Any]) -> str:
    part = cfg["partition"]
    kind = str(part["kind"])
    if kind == "dirichlet":
        return f"dirichlet({float(part['alpha']):g})"
    return kind


def _attack_variant(attack: dict[str, Any]) -> str:
    """Name the update manipulator; ``none`` when the delta is submitted as is.

    The canonical backdoor scales by the round-aware factor (``gamma`` unset);
    anything else (a fixed gamma, a norm bound, direction alignment) is a
    different attacker and is named with its parameters.
    """
    block = attack.get("update_manipulator")
    if not block:
        return "none"
    block = dict(block)
    target = str(block.pop("_target_")).rsplit(".", 1)[-1]
    if target == "ModelReplacementManipulator" and block.get("gamma") is None:
        return VANILLA_ATTACK
    params = ",".join(f"{k}={v}" for k, v in sorted(block.items()))
    return f"{target}({params})" if params else target


def _window(cfg: dict[str, Any]) -> tuple[int, int | None]:
    schedule = cfg["attack"].get("schedule") or {}
    target = str(schedule.get("_target_", "")).rsplit(".", 1)[-1]
    if target == "WindowedAttack":
        end = schedule.get("end_round")
        return int(schedule.get("start_round", 1)), (None if end is None else int(end))
    return (1, None)


def cell_key(manifest: dict[str, Any]) -> CellKey | None:
    """Build the cell key from a manifest; ``None`` for non-backdoor runs.

    Colluding-attacker runs belong to their own analysis and are skipped.
    """
    cfg = manifest["config"]
    attack = cfg["attack"]
    if attack.get("kind") != "backdoor" or not attack.get("poisoner"):
        return None
    if attack.get("collusion"):
        return None
    poisoner = attack["poisoner"]
    fed = cfg["federation"]
    return CellKey(
        dataset=str(cfg["dataset"]["name"]),
        fraction=float(attack["assignment"]["fraction"]),
        defense=_defense_label(cfg),
        forced_m=(
            None
            if fed.get("force_malicious_per_round") is None
            else int(fed["force_malicious_per_round"])
        ),
        k=int(fed["clients_per_round"]),
        partition=_partition_label(cfg),
        trigger_position=str(poisoner.get("trigger_position", "bottom_right")),
        pad_border=int(cfg["dataset"].get("pad_border", 0) or 0),
        trigger_size=int(poisoner["trigger_size"]),
        trigger_amplitude=float(poisoner["trigger_amplitude"]),
        batch_mode=str(poisoner.get("batch_mode", "replace")),
        poisoning_rate=float(poisoner.get("poisoning_rate", 0.5)),
        attack_variant=_attack_variant(attack),
        window=_window(cfg),
        tag=cfg.get("tag"),
    )


def sustained_implant(
    asr: np.ndarray, thr: float = ASR_THRESHOLD, k: int = SUSTAIN_ROUNDS
) -> int | None:
    """First round r such that ASR >= thr on rounds r .. r+k-1 (1-based)."""
    for i in range(len(asr) - k + 1):
        if (asr[i : i + k] >= thr).all():
            return i + 1
    return None


def _malicious_share(
    rounds: np.ndarray, ledger: list[dict[str, Any]] | None, mal_counts: np.ndarray, k: int
) -> np.ndarray:
    """Per-round aggregation-weight share held by malicious clients.

    From the defense ledger when one exists; otherwise every client holds an
    equal share, so it is the malicious count over the cohort size.
    """
    if not ledger or "weight" not in ledger[0]:
        return mal_counts / float(k)
    num: dict[int, float] = defaultdict(float)
    den: dict[int, float] = defaultdict(float)
    for row in ledger:
        den[int(row["round"])] += float(row["weight"])
        if row["is_malicious"]:
            num[int(row["round"])] += float(row["weight"])
    return np.array(
        [num[int(r)] / den[int(r)] if den[int(r)] > 0 else 0.0 for r in rounds]
    )


def summarise_run(
    seed: int,
    metrics: list[dict[str, Any]],
    client_rows: list[dict[str, Any]],
    ledger: list[dict[str, Any]] | None,
    k: int,
    window: tuple[int, int | None],
) -> RunSummary:
    rounds = np.array([int(m["round"]) for m in metrics])
    asr = np.array([float(m.get("asr_accuracy", math.nan)) for m in metrics])
    acc = np.array([float(m.get("test_accuracy", math.nan)) for m in metrics])
    n = len(rounds)

    per_round_mal: dict[int, int] = defaultdict(int)
    for row in client_rows:
        per_round_mal[int(row["round"])] += int(bool(row["is_malicious"]))
    mal_counts = np.array([per_round_mal[int(r)] for r in rounds])
    share = _malicious_share(rounds, ledger, mal_counts, k)

    implant = sustained_implant(asr)
    valid = [i for i in range(FIRST_VALID_ROUND - 1, n) if asr[i] >= ASR_THRESHOLD]
    first_cross = valid[0] + 1 if valid else None
    asr_after = (
        float(asr[min(first_cross - 1 + RECOVERY_LAG, n - 1)])
        if first_cross is not None
        else None
    )
    asr_final = float(np.nanmean(asr[-CONVERGENCE_WINDOW:]))
    recovered = (asr_final < ASR_THRESHOLD) if first_cross is not None else None

    majority = np.flatnonzero(share[FIRST_VALID_ROUND - 1 :] >= 0.5) + FIRST_VALID_ROUND
    first_majority = int(majority[0]) if majority.size else None

    asr_at: dict[str, float] = {}
    start, end = window
    if end is not None and end <= n:
        asr_at["asr_at_window_end"] = float(asr[end - 1])
        asr_at["asr_window_end_plus_50"] = float(asr[min(end - 1 + 50, n - 1)])
    if start > 1 and start <= n:
        asr_at["asr_before_start"] = float(asr[start - 2])

    timings = {
        key: float(np.mean([float(m[key]) for m in metrics]))
        for key in TIMING_KEYS
        if all(key in m for m in metrics)
    }
    return RunSummary(
        seed=seed,
        num_rounds=n,
        asr_final=asr_final,
        acc_final=float(np.nanmean(acc[-CONVERGENCE_WINDOW:])),
        implant_round=implant,
        first_cross=first_cross,
        asr_after_cross=asr_after,
        recovered=recovered,
        max_malicious_share=float(share[FIRST_VALID_ROUND - 1 :].max()) if n >= FIRST_VALID_ROUND else float(share.max()),
        first_majority_round=first_majority,
        majority_rounds=int(majority.size),
        m_at_implant=int(mal_counts[implant - 1]) if implant is not None else None,
        share_at_implant=float(share[implant - 1]) if implant is not None else None,
        max_m=int(mal_counts.max()) if n else 0,
        asr_at=asr_at,
        timings=timings,
    )


def load_cells(results_root: Path) -> dict[CellKey, list[RunSummary]]:
    """Walk ``results/`` and group completed backdoor runs by cell."""
    cells: dict[CellKey, list[RunSummary]] = defaultdict(list)
    for manifest_path in sorted(results_root.glob("*/*/seed_*/manifest.json")):
        run_dir = manifest_path.parent
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        key = cell_key(manifest)
        if key is None:
            continue
        metrics = json.loads(metrics_path.read_text())
        if not metrics or "asr_accuracy" not in metrics[-1]:
            continue
        client_path = run_dir / "client_metrics.json"
        client_rows = json.loads(client_path.read_text()) if client_path.exists() else []
        ledger_path = run_dir / "defense_ledger.json"
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else None
        cells[key].append(
            summarise_run(
                int(manifest["seed"]), metrics, client_rows, ledger, key.k, key.window
            )
        )
    return {k: sorted(v, key=lambda r: r.seed) for k, v in cells.items()}


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------


def hypergeom_tail(N: int, M: int, K: int, m_min: int) -> float:
    """P(at least ``m_min`` of ``K`` drawn without replacement from N hit M)."""
    if m_min > K:
        return 0.0
    return sum(
        math.comb(M, m) * math.comb(N - M, K - m) for m in range(m_min, K + 1)
    ) / math.comb(N, K)


def _fmt(x: float | None, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{nd}f}"


def _mean_sd(values: list[float | None], nd: int = 3) -> str:
    xs = [v for v in values if v is not None and not math.isnan(v)]
    if not xs:
        return "–"
    a = np.asarray(xs, dtype=float)
    sd = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return f"{a.mean():.{nd}f} ± {sd:.{nd}f}"


def _per_seed(values: list[int | None]) -> str:
    return ",".join("never" if v is None else str(v) for v in values)


def _row_common(runs: list[RunSummary]) -> list[str]:
    recovered = [r.recovered for r in runs if r.recovered is not None]
    rec = f"{sum(recovered)}/{len(recovered)}" if recovered else "–"
    return [
        str(len(runs)),
        _mean_sd([r.asr_final for r in runs]),
        _mean_sd([r.acc_final for r in runs]),
        _per_seed([r.implant_round for r in runs]),
        _per_seed([r.first_majority_round for r in runs]),
        _fmt(max(r.max_malicious_share for r in runs), 2),
        _mean_sd([r.asr_after_cross for r in runs], 2),
        rec,
    ]


COMMON_HEADER = [
    "n",
    "ASR final",
    "clean acc",
    "implant round (per seed)",
    "first share>=0.5 (per seed)",
    "max share",
    f"ASR +{RECOVERY_LAG} after cross",
    "recovered",
]


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _select(
    cells: dict[CellKey, list[RunSummary]],
    fraction: float,
    block: str,
    reference_defenses: tuple[str, ...],
) -> list[tuple[CellKey, list[RunSummary]]]:
    """Cells in ``block`` plus reference cells for the named defenses.

    Only the vanilla backdoor (and its scaling-off variant) is in scope;
    adaptive attackers are a different experiment family.
    """
    out = []
    for key, runs in cells.items():
        if abs(key.fraction - fraction) > 1e-9 or not key.is_vanilla_attack:
            continue
        b = key.block
        if block in b.split("+") or (b == "REF" and key.defense in reference_defenses):
            out.append((key, runs))
    return out


def _sort_key(item: tuple[CellKey, list[RunSummary]]) -> tuple:
    key = item[0]
    return (
        key.block != "REF",
        key.dataset,
        key.defense,
        key.forced_m if key.forced_m is not None else -1,
        key.k,
        key.partition,
        key.trigger_position,
        key.pad_border,
        key.window[0],
        key.window[1] if key.window[1] is not None else 10**6,
        key.trigger_amplitude,
        key.batch_mode,
        key.attack_variant,
    )


def table_d3(cells: dict[CellKey, list[RunSummary]], fraction: float) -> str:
    rows = []
    for key, runs in sorted(
        _select(cells, fraction, "D3", ("capped-wmed", "median")), key=_sort_key
    ):
        m = "sampled" if key.forced_m is None else str(key.forced_m)
        predicted = "–" if key.forced_m is None else (
            "hold" if key.forced_m < math.ceil(key.k / 2) else "capture"
        )
        observed = (
            "capture" if any(r.implant_round is not None for r in runs) else "hold"
        )
        rows.append(
            [key.dataset, key.defense, m, predicted, observed]
            + _row_common(runs)
            + [
                _per_seed([r.m_at_implant for r in runs]),
                ",".join(_fmt(r.share_at_implant, 2) for r in runs),
            ]
        )
    header = (
        ["dataset", "defense", "m forced", "predicted", "observed"]
        + COMMON_HEADER
        + ["m at implant", "share at implant"]
    )
    return _table(header, rows)


def table_d8(cells: dict[CellKey, list[RunSummary]], fraction: float) -> str:
    rows = []
    for key, runs in sorted(
        _select(cells, fraction, "D8", ("undefended", "capped-wmed")), key=_sort_key
    ):
        placement = key.trigger_position
        if key.pad_border:
            placement += f" (pad {key.pad_border})"
        rows.append(
            [key.dataset, key.defense, placement, str(key.trigger_size)]
            + _row_common(runs)
        )
    return _table(
        ["dataset", "defense", "trigger placement", "size"] + COMMON_HEADER, rows
    )


def table_d9(cells: dict[CellKey, list[RunSummary]], fraction: float) -> str:
    rows = []
    for key, runs in sorted(
        _select(cells, fraction, "D9", ("capped-wmed",)), key=_sort_key
    ):
        num_malicious = math.ceil(100 * key.fraction)
        breakdown_m = math.ceil(key.k / 2)
        p_round = hypergeom_tail(100, num_malicious, key.k, breakdown_m)
        p_any = 1.0 - (1.0 - p_round) ** runs[0].num_rounds
        rows.append(
            [
                key.dataset,
                key.defense,
                str(key.k),
                str(breakdown_m),
                f"{p_round:.4f}",
                f"{p_any:.3f}",
                str(max(r.max_m for r in runs)),
            ]
            + _row_common(runs)
        )
    return _table(
        [
            "dataset",
            "defense",
            "K",
            "m for share>=0.5",
            "P(m>=that) per round",
            "P(any round)",
            "max m seen",
        ]
        + COMMON_HEADER,
        rows,
    )


def table_d1(cells: dict[CellKey, list[RunSummary]], fraction: float) -> str:
    rows = []
    for key, runs in sorted(
        _select(cells, fraction, "D1", ("undefended", "capped-wmed", "wmed", "ScoreWeight")),
        key=_sort_key,
    ):
        rows.append([key.dataset, key.defense] + _row_common(runs))
    return _table(["dataset", "defense"] + COMMON_HEADER, rows)


def table_window(cells: dict[CellKey, list[RunSummary]], fraction: float) -> str:
    rows = []
    selected = _select(cells, fraction, "D7", ("undefended", "capped-wmed")) + [
        c for c in _select(cells, fraction, "D2", ()) if c[0].block != "REF"
    ]
    for key, runs in sorted(selected, key=_sort_key):
        start, end = key.window
        window = f"{start}–{end if end is not None else 'end'}"
        rows.append(
            [key.dataset, key.defense, window]
            + _row_common(runs)
            + [
                _mean_sd([r.asr_at.get("asr_before_start") for r in runs], 2),
                _mean_sd([r.asr_at.get("asr_at_window_end") for r in runs], 2),
                _mean_sd([r.asr_at.get("asr_window_end_plus_50") for r in runs], 2),
            ]
        )
    return _table(
        ["dataset", "defense", "attack rounds"]
        + COMMON_HEADER
        + ["ASR before start", "ASR at window end", "ASR end+50"],
        rows,
    )


def table_d4(cells: dict[CellKey, list[RunSummary]], fraction: float) -> str:
    rows = []
    for key, runs in sorted(
        _select(cells, fraction, "D4", ("undefended", "capped-wmed")), key=_sort_key
    ):
        rows.append([key.dataset, key.defense, key.partition] + _row_common(runs))
    return _table(["dataset", "defense", "partition"] + COMMON_HEADER, rows)


def table_d6(cells: dict[CellKey, list[RunSummary]], fraction: float) -> str:
    rows = []
    for key, runs in sorted(
        _select(cells, fraction, "D6", ("capped-wmed",)), key=_sort_key
    ):
        mix = key.batch_mode if key.batch_mode == "augment" else f"replace@{key.poisoning_rate:g}"
        rows.append(
            [
                key.dataset,
                key.defense,
                "on" if key.scaling else "off",
                f"{key.trigger_amplitude:g}",
                mix,
                str(key.trigger_size),
            ]
            + _row_common(runs)
        )
    return _table(
        ["dataset", "defense", "scaling", "amplitude", "batch", "size"] + COMMON_HEADER,
        rows,
    )


def table_timing(cells: dict[CellKey, list[RunSummary]], fraction: float) -> str:
    """Mean per-round server seconds by (dataset, defense, K), all blocks."""
    pooled: dict[tuple[str, str, int], list[RunSummary]] = defaultdict(list)
    for key, runs in cells.items():
        if abs(key.fraction - fraction) > 1e-9 or not key.is_vanilla_attack:
            continue
        if any(r.timings for r in runs):
            pooled[(key.dataset, key.defense, key.k)].extend(r for r in runs if r.timings)
    rows = []
    for (dataset, defense, k), runs in sorted(pooled.items()):
        rows.append(
            [dataset, defense, str(k), str(len(runs))]
            + [
                _mean_sd([r.timings.get(t) for r in runs], 2)
                for t in TIMING_KEYS
            ]
        )
    return _table(["dataset", "defense", "K", "runs"] + list(TIMING_KEYS), rows)


SECTIONS: tuple[tuple[str, str, Any, tuple[str, ...]], ...] = (
    (
        "D3 Forced co-selection",
        "Exactly m attackers in every K-client round. Prediction: capped-wmed "
        "holds at m < K/2 and is captured once the malicious share reaches 0.5; "
        "captured CIFAR-10 recovers, captured FEMNIST does not.",
        table_d3,
        ("D3",),
    ),
    (
        "D8 Trigger placement",
        "Corner vs centre (FEMNIST) and padded-border (CIFAR-10). Prediction: "
        "centre-trigger FEMNIST recovers like CIFAR-10; border-trigger CIFAR-10 "
        "becomes permanent like FEMNIST.",
        table_d8,
        ("D8",),
    ),
    (
        "D7 / D2 Attack window",
        "Persistence (attack stops at a round) and delayed start. ASR at the "
        "window end versus 50 rounds later shows whether the base ever erodes "
        "an implant.",
        table_window,
        ("D7", "D2"),
    ),
    (
        "D9 Sampling-tail levers",
        "Larger K shrinks the hypergeometric tail (majority needs m >= K/2); "
        "carried reputation keeps a fresh bloc below equal share.",
        table_d9,
        ("D9",),
    ),
    (
        "D1 Oracle detection",
        "True attackers weight 0. High ASR here means aggregation, not "
        "detection, is the bottleneck.",
        table_d1,
        ("D1",),
    ),
    (
        "D4 Heterogeneity dial",
        "The native partition versus pooled-and-resplit (IID / Dirichlet).",
        table_d4,
        ("D4",),
    ),
    (
        "D6 Attack-strength dial",
        "Scaling, trigger amplitude, and batch composition against capped-wmed.",
        table_d6,
        ("D6",),
    ),
    (
        "Timing",
        "Mean per-round server wall-clock seconds from the metrics ledger.",
        table_timing,
        (),
    ),
)


def build_report(cells: dict[CellKey, list[RunSummary]], fraction: float) -> str:
    lines = [f"# E5 diagnostics (backdoor, f={fraction:g})\n"]
    lines.append(
        f"Converged values average the last {CONVERGENCE_WINDOW} rounds. "
        f"'implant round' is the first round from which ASR stays >= "
        f"{ASR_THRESHOLD} for {SUSTAIN_ROUNDS} consecutive rounds; 'first "
        f"share>=0.5' is the first round (from round {FIRST_VALID_ROUND}) in "
        "which malicious clients held at least half the aggregation weight "
        "(the ledger's weights for a defense, m/K otherwise); 'recovered' "
        f"counts seeds whose ASR crossed {ASR_THRESHOLD} and ended below it.\n"
    )
    in_scope = {
        key: runs
        for key, runs in cells.items()
        if abs(key.fraction - fraction) <= 1e-9 and key.is_vanilla_attack
    }
    n_runs = sum(len(v) for v in in_scope.values())
    lines.append(f"Runs found at this fraction: {n_runs} in {len(in_scope)} cells.\n")
    present = {b for key in in_scope for b in key.block.split("+")}
    for title, blurb, fn, blocks in SECTIONS:
        lines.append(f"## {title}\n")
        lines.append(blurb + "\n")
        table = fn(cells, fraction)
        if (blocks and not present.intersection(blocks)) or table.count("\n") < 2:
            # Reference rows alone do not make a diagnostic table.
            lines.append("_no runs yet_\n")
        else:
            lines.append(table + "\n")
    return "\n".join(lines)


def _json_dump(cells: dict[CellKey, list[RunSummary]]) -> list[dict[str, Any]]:
    out = []
    for key, runs in cells.items():
        for r in runs:
            record = {**asdict(key), "block": key.block, **asdict(r)}
            record["window"] = list(key.window)
            out.append(record)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None, help="per-run dump")
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
    report = build_report(cells, args.fraction)
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
