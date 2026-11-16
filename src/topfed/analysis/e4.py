"""E4 hybrid defenses: bounded influence plus a server-rooted signal.

Scores four variants against five pre-registered success criteria:

* **H1** ``gated-capped-wmed`` -- capped weighted median, each weight times
  ReLU(cosine to the server's root-set delta).
* **H2** ``rooted-capped-wmed`` -- the server cosine as a sixth TOPSIS benefit
  criterion (CRITIC), capped weighted median on top.
* **H3** ``fltrust-median`` -- FLTrust's trust and magnitude normalisation on a
  trust-weighted median.
* **H4** ``m4band-capped-wmed`` -- the control: capped weighted median gated by
  the round's 75th percentile of validation-loss deltas (no server training).

Attacks: the vanilla scaled backdoor, the norm-bounded adaptive backdoor
(budget gamma), the direction-aligned backdoor (cosine c), and the composed
attack (direction-aligned then norm-bounded, both knobs). Each adaptive attack
meets each variant at its **best response**: the knob point with the highest
seed-0 ASR in the rescan grid (NB gamma in {0.6, 1.0}; DA c in {0.9, 0.5};
composed gamma x c), exact ties to the attack config's default. The n=5
headline cell runs at that point. The same function drives the sweep
(``--cells headline``), so the sweep and the report cannot disagree.

Success criteria (thresholds fixed in advance, compared with seed means):

1. worst-case adaptive ASR (max over NB, DA, composed at each variant's best
   response) <= 0.07 at f=0.10 and <= 0.25 at f=0.20 (FLTrust's);
2. vanilla-backdoor ASR <= 0.04 at f=0.10 and <= 0.15 at f=0.20
   (capped-wmed's);
3. clean accuracy >= 0.71 in every f=0.10 CIFAR-10 headline cell;
4. no cell above undefended: in every cell scheduled for the variant, ASR at
   most undefended's (targeted) or clean accuracy at least undefended's
   (untargeted), undefended taken at the identical attack config;
5. FEMNIST backdoor ASR <= 0.05 at f=0.10.

A criterion FAILs as soon as one complete cell (>= 5 seeds) violates it, is
PENDING while a cell it needs is incomplete, and PASSes otherwise. A variant
meeting all five is the hybrid; otherwise the report names what each fails.

Only reference-setting runs count (N=100, K=10, 200 rounds, native
partition, corner trigger, no window, no forced co-selection); cells are keyed
by resolved config, so frozen, E3 and E4 runs of one config pool.

Usage::

    uv run python -m topfed.analysis.e4 --out tables/e4-tables.md
    uv run python -m topfed.analysis.e4 --cells rescan --variants H1,H4
    uv run python -m topfed.analysis.e4 --cells headline --variants H1,H4

``--cells`` prints one tab-separated line per cell to run: tag, experiment,
seed set (``rescan`` or ``all``), then the Hydra overrides.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from topfed.analysis.e3 import (
    NUM_ROUNDS,
    REFERENCE_STRENGTH,
    RunSummary,
    _attack as _e3_attack,
    _is_reference_setting,
    _reference_trigger,
    _target_name,
    load_cells_by,
    summarise_run,
)
from topfed.analysis.e5 import REFERENCE_COMMON, _defense_label, _fmt, _table

__all__ = ["CellKey", "RunSummary", "load_cells", "summarise_run"]

MIN_SEEDS = 5
RESCAN_SEED = 0
FRACTIONS = (0.10, 0.20)
REFERENCE_VAL = {"cifar10": 100, "femnist": 30}

VARIANTS = {
    "H1": "gated-capped-wmed",
    "H2": "rooted-capped-wmed",
    "H3": "fltrust-median",
    "H4": "m4band-capped-wmed",
}
VARIANT_OVERRIDES = {
    "H1": "+defense=topsis_server_gated_capped_wmedian",
    "H2": "+defense=topsis_rooted_capped_weighted_median",
    "H3": "+baseline=fltrust_median",
    "H4": "+defense=topsis_m4band_capped_wmedian",
}
# Variants in the regression and FEMNIST blocks (the control is headline only).
EXTENDED_VARIANTS = ("H1", "H2", "H3")
REFERENCE_DEFENSES = ("fltrust", "capped-wmed", "undefended")
REFERENCE_OVERRIDES = {
    "fltrust": "+baseline=fltrust",
    "capped-wmed": "+defense=topsis_capped_weighted_median",
}

HEADLINE_ATTACKS = ("backdoor", "adaptive", "direction_aligned", "composed")
ADAPTIVE_ATTACKS = ("adaptive", "direction_aligned", "composed")
REGRESSION_ATTACKS = ("gradient_scaling", "sign_flip", "dirty_label", "clean_label")
TARGETED = ("backdoor", "adaptive", "direction_aligned", "composed", "clean_label")
ATTACK_NAMES = {
    "backdoor": "vanilla backdoor",
    "adaptive": "norm-bounded (γ)",
    "direction_aligned": "direction-aligned (c)",
    "composed": "composed (γ, c)",
    "gradient_scaling": "gradient scaling",
    "sign_flip": "sign flip",
    "dirty_label": "dirty label",
    "clean_label": "clean-label backdoor",
}

Knobs = tuple[float, ...]
# Best-response grid per adaptive attack; the attack config's default first.
# Knob tuples: adaptive (budget,), direction_aligned (cosine,), composed
# (budget, cosine).
KNOB_GRID: dict[str, tuple[Knobs, ...]] = {
    "adaptive": ((0.6,), (1.0,)),
    "direction_aligned": ((0.9,), (0.5,)),
    "composed": ((0.6, 0.9), (0.6, 0.5), (1.0, 0.9), (1.0, 0.5)),
}
BASE_EXPERIMENT = {
    "backdoor": "backdoor_cifar10",
    "adaptive": "adaptive_backdoor_cifar10",
    "direction_aligned": "fltrust_backdoor_cifar10",
    "composed": "composed_backdoor_cifar10",
}

WORST_ADAPTIVE_MAX = {0.10: 0.07, 0.20: 0.25}
VANILLA_MAX = {0.10: 0.04, 0.20: 0.15}
CLEAN_ACC_MIN = 0.71
CLEAN_ACC_FRACTION = 0.10
FEMNIST_F10_MAX = 0.05
CRITERIA = (
    ("C1", "worst-case adaptive ASR ≤ FLTrust's (0.07 / 0.25)"),
    ("C2", "vanilla ASR ≤ capped-wmed's (0.04 / 0.15)"),
    ("C3", "clean acc ≥ 0.71 (every f=0.10 headline cell)"),
    ("C4", "no cell above undefended"),
    ("C5", "FEMNIST f=0.10 ASR ≤ 0.05"),
)


@dataclass(frozen=True)
class CellKey:
    """One E4 cell: dataset, attacker and its knobs, f, defense, and setting."""

    dataset: str
    attack: str
    knobs: Knobs | None
    fraction: float
    defense: str
    val_per_class: int
    num_rounds: int
    profile_memory: bool


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------


def _composed_knobs(cfg: dict[str, Any]) -> Knobs | None:
    """(budget, cosine) of a direction-aligned-then-norm-bounded attacker."""
    attack = cfg["attack"]
    poisoner = attack.get("poisoner") or {}
    manipulator = attack.get("update_manipulator") or {}
    if attack.get("kind") != "backdoor" or _target_name(manipulator) != "ComposedManipulator":
        return None
    if not poisoner or not _reference_trigger(cfg, poisoner):
        return None
    if str(poisoner.get("batch_mode", "replace")) != REFERENCE_COMMON["batch_mode"]:
        return None
    stages = list(manipulator.get("manipulators") or [])
    names = [_target_name(s) for s in stages]
    if names != ["DirectionAlignedManipulator", "NormBoundedManipulator"]:
        return None
    return (float(stages[1]["target_norm"]), float(stages[0]["cosine_target"]))


def cell_key(manifest: dict[str, Any]) -> CellKey | None:
    """Build the E4 cell key from a manifest; ``None`` for out-of-scope runs."""
    cfg = manifest["config"]
    if not _is_reference_setting(cfg):
        return None
    knobs = _composed_knobs(cfg)
    if knobs is not None:
        name = "composed"
    else:
        found = _e3_attack(cfg)
        if found is None:
            return None
        name, strength = found
        knobs = None if strength is None else (strength,)
    fed = cfg["federation"]
    return CellKey(
        dataset=str(cfg["dataset"]["name"]),
        attack=name,
        knobs=knobs,
        fraction=round(float(cfg["attack"]["assignment"]["fraction"]), 4),
        defense=_defense_label(cfg),
        val_per_class=int(cfg["dataset"]["val_per_class"]),
        num_rounds=int(fed["num_rounds"]),
        profile_memory=bool(fed.get("profile_memory", False)),
    )


def load_cells(results_root: Path) -> dict[CellKey, list[RunSummary]]:
    return load_cells_by(results_root, cell_key)


# ----------------------------------------------------------------------------
# Cell lookup and best response
# ----------------------------------------------------------------------------


def default_knobs(attack: str) -> Knobs | None:
    if attack in KNOB_GRID:
        return KNOB_GRID[attack][0]
    strength = REFERENCE_STRENGTH.get(attack)
    return None if strength is None else (strength,)


def runs_for(
    cells: dict[CellKey, list[RunSummary]],
    dataset: str,
    attack: str,
    fraction: float,
    defense: str,
    knobs: Knobs | None,
) -> list[RunSummary]:
    key = CellKey(
        dataset, attack, knobs, round(fraction, 4), defense,
        REFERENCE_VAL[dataset], NUM_ROUNDS, False,
    )
    return cells.get(key, [])


def _mean(runs: list[RunSummary], attr: str) -> float | None:
    xs = [float(getattr(r, attr)) for r in runs if getattr(r, attr) is not None]
    xs = [x for x in xs if not math.isnan(x)]
    return float(np.mean(xs)) if xs else None


def _seed0_asr(runs: list[RunSummary]) -> float | None:
    for r in runs:
        if r.seed == RESCAN_SEED:
            return r.asr_final
    return None


@dataclass(frozen=True)
class BestResponse:
    """Seed-0 ASR at each grid point and the chosen point (``None`` if incomplete)."""

    attack: str
    fraction: float
    defense: str
    asr_by_knobs: dict[Knobs, float | None]
    choice: Knobs | None


def best_response(
    cells: dict[CellKey, list[RunSummary]], attack: str, fraction: float, defense: str
) -> BestResponse:
    """The grid point with the highest seed-0 ASR; exact ties go to the default.

    Incomplete (``choice`` is ``None``) until every grid point has a seed-0 run.
    """
    grid = KNOB_GRID[attack]
    asr = {
        k: _seed0_asr(runs_for(cells, "cifar10", attack, fraction, defense, k))
        for k in grid
    }
    if any(v is None for v in asr.values()):
        return BestResponse(attack, fraction, defense, asr, None)
    choice = grid[0]
    for k in grid[1:]:
        value, best = asr[k], asr[choice]
        assert value is not None and best is not None
        if value > best:
            choice = k
    return BestResponse(attack, fraction, defense, asr, choice)


def headline_knobs(
    cells: dict[CellKey, list[RunSummary]], attack: str, fraction: float, defense: str
) -> tuple[bool, Knobs | None]:
    """``(ready, knobs)`` of a headline cell.

    An adaptive attack takes its best response and is not ready until the
    rescan is complete; any other attack takes its attack config's strength.
    """
    if attack in KNOB_GRID:
        choice = best_response(cells, attack, fraction, defense).choice
        return choice is not None, choice
    return True, default_knobs(attack)


# ----------------------------------------------------------------------------
# Sweep cell lists
# ----------------------------------------------------------------------------


def _fmt_knob(x: float) -> str:
    return repr(float(x))


def knob_suffix(attack: str, knobs: Knobs | None) -> str:
    if knobs is None or attack not in KNOB_GRID:
        return ""
    if attack == "adaptive":
        return f"_g{_fmt_knob(knobs[0])}"
    if attack == "direction_aligned":
        return f"_c{_fmt_knob(knobs[0])}"
    return f"_g{_fmt_knob(knobs[0])}_c{_fmt_knob(knobs[1])}"


def knob_overrides(attack: str, knobs: Knobs | None) -> list[str]:
    if knobs is None or attack not in KNOB_GRID:
        return []
    if attack == "adaptive":
        return [f"attack.update_manipulator.target_norm={_fmt_knob(knobs[0])}"]
    if attack == "direction_aligned":
        return [f"attack.update_manipulator.cosine_target={_fmt_knob(knobs[0])}"]
    return [
        f"attack.update_manipulator.manipulators.0.cosine_target={_fmt_knob(knobs[1])}",
        f"attack.update_manipulator.manipulators.1.target_norm={_fmt_knob(knobs[0])}",
    ]


def experiment_name(dataset: str, attack: str, fraction: float) -> str:
    f = f"f{round(fraction * 100):02d}"
    if dataset == "femnist":
        return f"femnist_{attack}_{f}"
    if attack in BASE_EXPERIMENT:
        return f"{BASE_EXPERIMENT[attack]}_{f}"
    return f"{attack}_cifar10_{f}"


@dataclass(frozen=True)
class SweepCell:
    tag: str
    experiment: str
    seeds: str  # "rescan" (seed 0 only) or "all"
    overrides: tuple[str, ...]

    def line(self) -> str:
        return "\t".join([self.tag, self.experiment, self.seeds, *self.overrides])


def _cell(
    dataset: str, attack: str, fraction: float, knobs: Knobs | None,
    defense_override: str | None, seeds: str,
) -> SweepCell:
    exp = experiment_name(dataset, attack, fraction)
    overrides = knob_overrides(attack, knobs)
    if defense_override is not None:
        overrides.append(defense_override)
    return SweepCell(f"e4_{exp}{knob_suffix(attack, knobs)}", exp, seeds, tuple(overrides))


def _undefended_refs(
    cells: dict[CellKey, list[RunSummary]],
    wanted: Iterable[tuple[str, str, float, Knobs | None]],
) -> list[SweepCell]:
    """Undefended cells at the given attack configs that lack ``MIN_SEEDS`` seeds."""
    out: list[SweepCell] = []
    seen: set[tuple[str, str, float, Knobs | None]] = set()
    for dataset, attack, fraction, knobs in wanted:
        key = (dataset, attack, round(fraction, 4), knobs)
        if key in seen:
            continue
        seen.add(key)
        if len(runs_for(cells, dataset, attack, fraction, "undefended", knobs)) < MIN_SEEDS:
            out.append(_cell(dataset, attack, fraction, knobs, None, "all"))
    return out


def sweep_cells(
    cells: dict[CellKey, list[RunSummary]], block: str, variants: tuple[str, ...]
) -> list[SweepCell]:
    """The cells a sweep block runs, given the results found so far.

    ``rescan``: every grid point of every adaptive attack at seed 0.
    ``headline``: the four attacks at n=5, adaptive ones at each variant's best
    response (skipped with a note on stderr while the rescan is incomplete),
    plus undefended at every such config that lacks five seeds.
    ``regression``: the untargeted attacks and the clean-label backdoor at
    f=0.10. ``femnist``: the FEMNIST backdoor at f=0.10 and 0.20.
    ``refs``: FLTrust and capped-wmed against the composed attack at its
    default point (optional context for the table).
    """
    out: list[SweepCell] = []
    wanted: list[tuple[str, str, float, Knobs | None]] = []
    if block == "rescan":
        for v in variants:
            for f in FRACTIONS:
                for attack in ADAPTIVE_ATTACKS:
                    for point in KNOB_GRID[attack]:
                        out.append(_cell("cifar10", attack, f, point, VARIANT_OVERRIDES[v], "rescan"))
        return out
    knobs: Knobs | None
    if block == "headline":
        for v in variants:
            for f in FRACTIONS:
                for attack in HEADLINE_ATTACKS:
                    ready, knobs = headline_knobs(cells, attack, f, VARIANTS[v])
                    if not ready:
                        print(
                            f"note: {v} {attack} f={f:g}: rescan incomplete, "
                            "headline cell not scheduled yet",
                            file=sys.stderr,
                        )
                        continue
                    out.append(_cell("cifar10", attack, f, knobs, VARIANT_OVERRIDES[v], "all"))
                    wanted.append(("cifar10", attack, f, knobs))
        return out + _undefended_refs(cells, wanted)
    if block == "regression":
        for v in variants:
            if v not in EXTENDED_VARIANTS:
                continue
            for attack in REGRESSION_ATTACKS:
                knobs = default_knobs(attack)
                out.append(_cell("cifar10", attack, 0.10, knobs, VARIANT_OVERRIDES[v], "all"))
                wanted.append(("cifar10", attack, 0.10, knobs))
        return out + _undefended_refs(cells, wanted)
    if block == "femnist":
        for v in variants:
            if v not in EXTENDED_VARIANTS:
                continue
            for f in FRACTIONS:
                out.append(_cell("femnist", "backdoor", f, None, VARIANT_OVERRIDES[v], "all"))
                wanted.append(("femnist", "backdoor", f, None))
        return out + _undefended_refs(cells, wanted)
    if block == "refs":
        for f in FRACTIONS:
            knobs = default_knobs("composed")
            for d in ("fltrust", "capped-wmed"):
                out.append(_cell("cifar10", "composed", f, knobs, REFERENCE_OVERRIDES[d], "all"))
            wanted.append(("cifar10", "composed", f, knobs))
        return out + _undefended_refs(cells, wanted)
    raise ValueError(f"unknown block: {block}")


# ----------------------------------------------------------------------------
# Success criteria
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class CellResult:
    """One scheduled cell of a variant: where it stands and what it scored."""

    dataset: str
    attack: str
    fraction: float
    knobs: Knobs | None
    n: int
    asr: float | None
    acc: float | None

    @property
    def complete(self) -> bool:
        return self.n >= MIN_SEEDS

    def label(self) -> str:
        knob = "" if self.knobs is None or self.attack not in KNOB_GRID else f"@{_knob_text(self.knobs)}"
        ds = "" if self.dataset == "cifar10" else f"{self.dataset} "
        return f"{ds}{self.attack}{knob} f={self.fraction:g}"


@dataclass(frozen=True)
class Verdict:
    status: str  # PASS, FAIL, PENDING, N/A
    detail: str


def _result(
    cells: dict[CellKey, list[RunSummary]], dataset: str, attack: str,
    fraction: float, defense: str, knobs: Knobs | None,
) -> CellResult:
    runs = runs_for(cells, dataset, attack, fraction, defense, knobs)
    return CellResult(dataset, attack, fraction, knobs, len(runs),
                      _mean(runs, "asr_final"), _mean(runs, "acc_final"))


def headline_results(
    cells: dict[CellKey, list[RunSummary]], variant: str, fraction: float
) -> dict[str, CellResult | None]:
    """Per attack, the variant's headline cell (``None`` while the rescan is open)."""
    defense = VARIANTS[variant]
    out: dict[str, CellResult | None] = {}
    for attack in HEADLINE_ATTACKS:
        ready, knobs = headline_knobs(cells, attack, fraction, defense)
        out[attack] = _result(cells, "cifar10", attack, fraction, defense, knobs) if ready else None
    return out


def _threshold_verdict(
    values: list[tuple[str, CellResult | None]], attr: str, limit: float, upper: bool
) -> tuple[str, list[str]]:
    """PASS/FAIL/PENDING for "every value <= limit" (``upper``) or ">= limit"."""
    failed, pending, shown = False, False, []
    for name, res in values:
        if res is None:
            pending = True
            shown.append(f"{name} rescan open")
            continue
        value = getattr(res, attr)
        text = f"{name} {_fmt(value)} (n={res.n})"
        if not res.complete or value is None:
            pending = True
            shown.append(text)
            continue
        bad = value > limit if upper else value < limit
        failed |= bad
        shown.append(text + (" ✗" if bad else ""))
    return ("FAIL" if failed else "PENDING" if pending else "PASS"), shown


def _combine(statuses: list[str]) -> str:
    if "FAIL" in statuses:
        return "FAIL"
    if "PENDING" in statuses:
        return "PENDING"
    return "PASS"


def criterion_worst_adaptive(cells: dict[CellKey, list[RunSummary]], variant: str) -> Verdict:
    statuses, parts = [], []
    for f in FRACTIONS:
        head = headline_results(cells, variant, f)
        vals = [(a, head[a]) for a in ADAPTIVE_ATTACKS]
        status, shown = _threshold_verdict(vals, "asr", WORST_ADAPTIVE_MAX[f], upper=True)
        complete = [r.asr for _, r in vals if r is not None and r.complete and r.asr is not None]
        worst = f"worst {_fmt(max(complete))}" if complete else "worst –"
        statuses.append(status)
        parts.append(f"f={f:g}: {worst} vs {WORST_ADAPTIVE_MAX[f]:.2f} [{'; '.join(shown)}]")
    return Verdict(_combine(statuses), " | ".join(parts))


def criterion_vanilla(cells: dict[CellKey, list[RunSummary]], variant: str) -> Verdict:
    statuses, parts = [], []
    for f in FRACTIONS:
        res = headline_results(cells, variant, f)["backdoor"]
        status, shown = _threshold_verdict([("vanilla", res)], "asr", VANILLA_MAX[f], upper=True)
        statuses.append(status)
        parts.append(f"f={f:g}: {shown[0]} vs {VANILLA_MAX[f]:.2f}")
    return Verdict(_combine(statuses), " | ".join(parts))


def criterion_clean_acc(cells: dict[CellKey, list[RunSummary]], variant: str) -> Verdict:
    head = headline_results(cells, variant, CLEAN_ACC_FRACTION)
    vals = [(a, head[a]) for a in HEADLINE_ATTACKS]
    status, shown = _threshold_verdict(vals, "acc", CLEAN_ACC_MIN, upper=False)
    accs = [r.acc for _, r in vals if r is not None and r.complete and r.acc is not None]
    lowest = f"min {_fmt(min(accs))}" if accs else "min –"
    return Verdict(status, f"{lowest} vs {CLEAN_ACC_MIN:.2f} [{'; '.join(shown)}]")


def scheduled_results(cells: dict[CellKey, list[RunSummary]], variant: str) -> list[CellResult | None]:
    """Every cell scheduled for the variant (headline, regression, FEMNIST)."""
    defense = VARIANTS[variant]
    out: list[CellResult | None] = []
    for f in FRACTIONS:
        out.extend(headline_results(cells, variant, f).values())
    if variant in EXTENDED_VARIANTS:
        for attack in REGRESSION_ATTACKS:
            out.append(_result(cells, "cifar10", attack, 0.10, defense, default_knobs(attack)))
        for f in FRACTIONS:
            out.append(_result(cells, "femnist", "backdoor", f, defense, None))
    return out


def criterion_no_backfire(cells: dict[CellKey, list[RunSummary]], variant: str) -> Verdict:
    failed, pending, violations, missing = False, False, [], 0
    for res in scheduled_results(cells, variant):
        if res is None or not res.complete:
            pending = True
            missing += 1
            continue
        ref = _result(cells, res.dataset, res.attack, res.fraction, "undefended", res.knobs)
        if not ref.complete:
            pending = True
            missing += 1
            continue
        if res.attack in TARGETED:
            assert res.asr is not None and ref.asr is not None
            bad, text = res.asr > ref.asr, f"ASR {_fmt(res.asr)} > {_fmt(ref.asr)}"
        else:
            assert res.acc is not None and ref.acc is not None
            bad, text = res.acc < ref.acc, f"acc {_fmt(res.acc)} < {_fmt(ref.acc)}"
        if bad:
            failed = True
            violations.append(f"{res.label()}: {text}")
    status = "FAIL" if failed else "PENDING" if pending else "PASS"
    detail = "; ".join(violations) if violations else "no violation among complete cells"
    if missing:
        detail += f" ({missing} cell(s) or undefended reference(s) incomplete)"
    return Verdict(status, detail)


def criterion_femnist(cells: dict[CellKey, list[RunSummary]], variant: str) -> Verdict:
    if variant not in EXTENDED_VARIANTS:
        return Verdict("N/A", "not scheduled (control variant)")
    res = _result(cells, "femnist", "backdoor", 0.10, VARIANTS[variant], None)
    status, shown = _threshold_verdict([("FEMNIST f=0.1", res)], "asr", FEMNIST_F10_MAX, upper=True)
    return Verdict(status, f"{shown[0]} vs {FEMNIST_F10_MAX:.2f}")


def scorecard(cells: dict[CellKey, list[RunSummary]], variant: str) -> dict[str, Verdict]:
    return {
        "C1": criterion_worst_adaptive(cells, variant),
        "C2": criterion_vanilla(cells, variant),
        "C3": criterion_clean_acc(cells, variant),
        "C4": criterion_no_backfire(cells, variant),
        "C5": criterion_femnist(cells, variant),
    }


def overall(card: dict[str, Verdict]) -> str:
    fails = [c for c, v in card.items() if v.status == "FAIL"]
    if fails:
        return "fails " + ", ".join(fails)
    if any(v.status == "N/A" for v in card.values()):
        pending = [c for c, v in card.items() if v.status == "PENDING"]
        return "control; " + ("pending " + ", ".join(pending) if pending else "no failure")
    if any(v.status == "PENDING" for v in card.values()):
        return "pending " + ", ".join(c for c, v in card.items() if v.status == "PENDING")
    return "meets all five: the hybrid"


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------


def _knob_text(knobs: Knobs) -> str:
    return ",".join(f"{k:g}" for k in knobs)


def _cell_text(runs: list[RunSummary], attr: str) -> str:
    xs = [float(getattr(r, attr)) for r in runs if getattr(r, attr) is not None]
    if not xs:
        return "–"
    a = np.asarray(xs)
    sd = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return f"{a.mean():.3f} ± {sd:.3f} ({a.size})"


def table_rescan(cells: dict[CellKey, list[RunSummary]], variants: tuple[str, ...]) -> str:
    header = ["variant", "f", "attack", "seed-0 ASR by knob", "best response"]
    rows = []
    for v in variants:
        for f in FRACTIONS:
            for attack in ADAPTIVE_ATTACKS:
                br = best_response(cells, attack, f, VARIANTS[v])
                points = ", ".join(f"{_knob_text(k)}: {_fmt(a)}" for k, a in br.asr_by_knobs.items())
                choice = "incomplete" if br.choice is None else _knob_text(br.choice)
                rows.append([v, f"{f:g}", ATTACK_NAMES[attack], points, choice])
    return _table(header, rows)


def table_headline(cells: dict[CellKey, list[RunSummary]], fraction: float) -> str:
    header = ["attack"] + [f"{v} {VARIANTS[v]}" for v in VARIANTS] + list(REFERENCE_DEFENSES)
    rows = []
    for attack in HEADLINE_ATTACKS:
        row = [ATTACK_NAMES[attack]]
        for v in VARIANTS:
            ready, knobs = headline_knobs(cells, attack, fraction, VARIANTS[v])
            if not ready:
                row.append("rescan open")
                continue
            runs = runs_for(cells, "cifar10", attack, fraction, VARIANTS[v], knobs)
            moved = knobs is not None and attack in KNOB_GRID and knobs != default_knobs(attack)
            mark = f" †{_knob_text(knobs)}" if moved and knobs is not None else ""
            row.append(f"{_cell_text(runs, 'asr_final')} / {_fmt(_mean(runs, 'acc_final'))}{mark}")
        for d in REFERENCE_DEFENSES:
            runs = runs_for(cells, "cifar10", attack, fraction, d, default_knobs(attack))
            row.append(f"{_cell_text(runs, 'asr_final')} / {_fmt(_mean(runs, 'acc_final'))}")
        rows.append(row)
    return _table(header, rows)


def table_regression(cells: dict[CellKey, list[RunSummary]]) -> str:
    header = ["attack (f=0.1)"] + [f"{v} {VARIANTS[v]}" for v in EXTENDED_VARIANTS] + list(REFERENCE_DEFENSES)
    rows = []
    for attack in REGRESSION_ATTACKS:
        attr = "asr_final" if attack in TARGETED else "acc_final"
        row = [f"{ATTACK_NAMES[attack]} ({'ASR' if attr == 'asr_final' else 'clean acc'})"]
        for d in [VARIANTS[v] for v in EXTENDED_VARIANTS] + list(REFERENCE_DEFENSES):
            row.append(_cell_text(runs_for(cells, "cifar10", attack, 0.10, d, default_knobs(attack)), attr))
        rows.append(row)
    return _table(header, rows)


def table_femnist(cells: dict[CellKey, list[RunSummary]]) -> str:
    header = ["f"] + [f"{v} {VARIANTS[v]}" for v in EXTENDED_VARIANTS] + list(REFERENCE_DEFENSES)
    rows = []
    for f in FRACTIONS:
        row = [f"{f:g}"]
        for d in [VARIANTS[v] for v in EXTENDED_VARIANTS] + list(REFERENCE_DEFENSES):
            runs = runs_for(cells, "femnist", "backdoor", f, d, None)
            row.append(f"{_cell_text(runs, 'asr_final')} / {_fmt(_mean(runs, 'acc_final'))}")
        rows.append(row)
    return _table(header, rows)


def table_scorecard(cells: dict[CellKey, list[RunSummary]]) -> tuple[str, str]:
    header = ["variant"] + [c for c, _ in CRITERIA] + ["verdict"]
    rows, details = [], []
    for v in VARIANTS:
        card = scorecard(cells, v)
        rows.append([f"{v} {VARIANTS[v]}"] + [card[c].status for c, _ in CRITERIA] + [overall(card)])
        details.append(f"**{v} {VARIANTS[v]}**\n")
        details.extend(f"- {c}: {card[c].status}. {card[c].detail}" for c, _ in CRITERIA)
        details.append("")
    return _table(header, rows), "\n".join(details)


def build_report(cells: dict[CellKey, list[RunSummary]]) -> str:
    lines = ["# E4 hybrid: bounded influence plus a server-rooted signal\n"]
    lines.append(
        "Generated by `uv run python -m topfed.analysis.e4`. ASR and clean accuracy "
        "are means over the last 50 rounds per seed; cells show mean ± sd (seeds). "
        f"A cell is complete at {MIN_SEEDS} seeds.\n"
    )
    lines.append("## Success criteria\n")
    lines.append(
        "Pre-registered thresholds, compared with seed means. A criterion fails as soon "
        "as one complete cell violates it and is pending while a cell it needs is "
        "incomplete. C4 compares each scheduled cell with undefended at the identical "
        "attack config (ASR for targeted attacks, clean accuracy for untargeted ones). "
        "The control (H4) runs only the headline grid, so C5 does not apply to it.\n"
    )
    lines.append("\n".join(f"- {c}: {text}" for c, text in CRITERIA) + "\n")
    card, details = table_scorecard(cells)
    lines.append(card + "\n")
    lines.append(details)
    lines.append("## Best-response rescans (seed 0)\n")
    lines.append(
        "Knobs: norm-bounded budget γ; direction-aligned cosine c; composed (γ, c). "
        "The best response is the highest seed-0 ASR, exact ties to the default "
        "(γ=0.6, c=0.9). The headline cell runs there.\n"
    )
    lines.append(table_rescan(cells, tuple(VARIANTS)) + "\n")
    for f in FRACTIONS:
        lines.append(f"## Headline, CIFAR-10, f = {f:g}\n")
        lines.append(
            "ASR / clean accuracy. Variants at their best response († marks a knob "
            "other than the default); references at the default knob.\n"
        )
        lines.append(table_headline(cells, f) + "\n")
    lines.append("## Regression, CIFAR-10, f = 0.1\n")
    lines.append(table_regression(cells) + "\n")
    lines.append("## FEMNIST backdoor\n")
    lines.append("ASR / clean accuracy.\n")
    lines.append(table_femnist(cells) + "\n")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _json_dump(cells: dict[CellKey, list[RunSummary]]) -> list[dict[str, Any]]:
    out = []
    for key, runs in cells.items():
        if key.num_rounds != NUM_ROUNDS or key.profile_memory:
            continue
        for r in runs:
            record = {**asdict(key), "seed": r.seed, "asr_final": r.asr_final,
                      "acc_final": r.acc_final, "source": r.source}
            out.append(record)
    return out


def _parse_variants(text: str) -> tuple[str, ...]:
    names = tuple(v.strip() for v in text.split(",") if v.strip())
    unknown = [v for v in names if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variant(s): {', '.join(unknown)}")
    return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None, help="per-run dump")
    parser.add_argument(
        "--cells", choices=("rescan", "headline", "regression", "femnist", "refs"),
        default=None, help="print the sweep cells of one block instead of the report",
    )
    parser.add_argument("--variants", default=",".join(VARIANTS))
    args = parser.parse_args(argv)

    results_root: Path = args.results_root.resolve()
    if args.cells is not None:
        cells = load_cells(results_root) if results_root.is_dir() else {}
        for cell in sweep_cells(cells, args.cells, _parse_variants(args.variants)):
            print(cell.line())
        return 0

    if not results_root.is_dir():
        raise SystemExit(f"results root not found: {results_root}")
    for target in (args.out, args.json):
        if target is not None:
            resolved = target.resolve()
            if results_root == resolved or results_root in resolved.parents:
                raise SystemExit(f"refusing to write inside the results tree: {resolved}")

    cells = load_cells(results_root)
    report = build_report(cells)
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
