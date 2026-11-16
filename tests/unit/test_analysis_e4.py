"""E4 analysis: cell keys, best response, sweep cell lists, success criteria."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from topfed.analysis import e4
from topfed.analysis.e4 import (
    KNOB_GRID,
    VARIANTS,
    CellKey,
    SweepCell,
    best_response,
    build_report,
    cell_key,
    load_cells,
    main,
    overall,
    scorecard,
    sweep_cells,
)

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "configs"
TARGETED_KINDS = ("backdoor", "clean_label")


@lru_cache(maxsize=None)
def _compose(experiment: str, overrides: tuple[str, ...]) -> str:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=[f"+experiment={experiment}", *overrides])
    return json.dumps(OmegaConf.to_container(cfg, resolve=True))


def _config(cell: SweepCell) -> dict[str, Any]:
    return json.loads(_compose(cell.experiment, (f"+tag={cell.tag}", *cell.overrides)))


def _write(root: Path, cell: SweepCell, seeds: range | list[int], asr: float | None,
           acc: float) -> None:
    cfg = _config(cell)
    label = f"{cell.tag}__{abs(hash(cell.overrides)) % 10**8}"
    for seed in seeds:
        run = root / label / "h" / f"seed_{seed}"
        run.mkdir(parents=True, exist_ok=True)
        (run / "manifest.json").write_text(json.dumps({"config": cfg, "seed": seed}))
        rows = []
        for r in range(1, e4.NUM_ROUNDS + 1):
            row: dict[str, Any] = {"round": r, "test_accuracy": acc}
            if asr is not None:
                row["asr_accuracy"] = asr
            rows.append(row)
        (run / "metrics.json").write_text(json.dumps(rows))


def _undefended(cell: SweepCell) -> SweepCell:
    overrides = tuple(o for o in cell.overrides if not o.startswith(("+defense", "+baseline")))
    return SweepCell(cell.tag, cell.experiment, "all", overrides)


def _is_targeted(cfg: dict[str, Any]) -> bool:
    return cfg["attack"]["kind"] in TARGETED_KINDS


# ----------------------------------------------------------------------------
# Cell keys
# ----------------------------------------------------------------------------


def test_every_sweep_cell_keys_back_to_its_attack_knobs_and_defense() -> None:
    cells: dict[CellKey, list[Any]] = {}
    for variant in VARIANTS:
        for cell in sweep_cells(cells, "rescan", (variant,)):
            key = cell_key({"config": _config(cell)})
            assert key is not None, cell
            assert key.defense == VARIANTS[variant]
            assert key.attack in KNOB_GRID
            assert key.knobs in KNOB_GRID[key.attack]
            assert cell.tag.endswith(e4.knob_suffix(key.attack, key.knobs))


def test_rooted_criterion_is_not_pooled_with_capped_wmed() -> None:
    base = SweepCell("t", "backdoor_cifar10_f10", "all", ())
    capped = cell_key({"config": _config(SweepCell("t", base.experiment, "all",
                                                   ("+defense=topsis_capped_weighted_median",)))})
    rooted = cell_key({"config": _config(SweepCell("t", base.experiment, "all",
                                                   ("+defense=topsis_rooted_capped_weighted_median",)))})
    assert capped is not None and rooted is not None
    assert capped.defense == "capped-wmed"
    assert rooted.defense == "rooted-capped-wmed"


def test_composed_knobs_are_budget_then_cosine() -> None:
    cell = SweepCell("t", "composed_backdoor_cifar10_f20", "all",
                     tuple(e4.knob_overrides("composed", (1.0, 0.5))))
    key = cell_key({"config": _config(cell)})
    assert key is not None
    assert (key.attack, key.knobs, key.fraction, key.defense) == ("composed", (1.0, 0.5), 0.2, "undefended")


def test_frozen_attacks_keep_their_e3_identity() -> None:
    key = cell_key({"config": _config(SweepCell("t", "adaptive_backdoor_cifar10_f10", "all", ()))})
    assert key is not None and (key.attack, key.knobs) == ("adaptive", (0.6,))
    key = cell_key({"config": _config(SweepCell("t", "gradient_scaling_cifar10_f10", "all", ()))})
    assert key is not None and (key.attack, key.knobs) == ("gradient_scaling", (5.0,))


# ----------------------------------------------------------------------------
# Best response
# ----------------------------------------------------------------------------


def _rescan_cells(variant: str, attack: str, fraction: float) -> list[SweepCell]:
    tag_f = f"f{round(fraction * 100):02d}"
    return [
        c for c in sweep_cells({}, "rescan", (variant,))
        if c.experiment == e4.experiment_name("cifar10", attack, fraction) and tag_f in c.experiment
    ]


def test_best_response_takes_the_highest_seed0_asr(tmp_path: Path) -> None:
    grid = _rescan_cells("H1", "composed", 0.10)
    assert len(grid) == 4
    for cell, asr in zip(grid, [0.10, 0.30, 0.20, 0.05], strict=True):
        _write(tmp_path, cell, [0], asr, 0.72)
    br = best_response(load_cells(tmp_path), "composed", 0.10, VARIANTS["H1"])
    assert br.choice == (0.6, 0.5)


def test_best_response_ties_go_to_the_default(tmp_path: Path) -> None:
    for cell in _rescan_cells("H4", "adaptive", 0.20):
        _write(tmp_path, cell, [0], 0.40, 0.70)
    br = best_response(load_cells(tmp_path), "adaptive", 0.20, VARIANTS["H4"])
    assert br.choice == KNOB_GRID["adaptive"][0]


def test_best_response_is_incomplete_until_every_point_has_seed0(tmp_path: Path) -> None:
    grid = _rescan_cells("H1", "direction_aligned", 0.10)
    _write(tmp_path, grid[0], [0], 0.2, 0.7)
    _write(tmp_path, grid[1], [1, 2, 3], 0.9, 0.7)  # seed 0 missing
    br = best_response(load_cells(tmp_path), "direction_aligned", 0.10, VARIANTS["H1"])
    assert br.choice is None


# ----------------------------------------------------------------------------
# Sweep cell lists
# ----------------------------------------------------------------------------


def test_rescan_block_size_and_seed_set() -> None:
    cells = sweep_cells({}, "rescan", ("H1", "H4"))
    assert len(cells) == 2 * 2 * sum(len(g) for g in KNOB_GRID.values())
    assert {c.seeds for c in cells} == {"rescan"}


def test_headline_waits_for_the_rescan_then_uses_the_worst_point(tmp_path: Path) -> None:
    listed = sweep_cells(load_cells(tmp_path) if tmp_path.exists() else {}, "headline", ("H1",))
    # Only the vanilla backdoor (no knob) is scheduled, with undefended refs.
    assert {c.experiment for c in listed if "+defense" in " ".join(c.overrides)} == {
        "backdoor_cifar10_f10", "backdoor_cifar10_f20"
    }

    for f in e4.FRACTIONS:
        for attack in e4.ADAPTIVE_ATTACKS:
            grid = _rescan_cells("H1", attack, f)
            for i, cell in enumerate(grid):
                _write(tmp_path, cell, [0], 0.5 if i == len(grid) - 1 else 0.1, 0.72)
    listed = sweep_cells(load_cells(tmp_path), "headline", ("H1",))
    defended = [c for c in listed if any(o.startswith("+defense") for o in c.overrides)]
    assert len(defended) == 8
    by_exp = {(c.experiment, c.tag) for c in defended}
    assert ("adaptive_backdoor_cifar10_f10", "e4_adaptive_backdoor_cifar10_f10_g1.0") in by_exp
    assert ("fltrust_backdoor_cifar10_f20", "e4_fltrust_backdoor_cifar10_f20_c0.5") in by_exp
    assert ("composed_backdoor_cifar10_f10", "e4_composed_backdoor_cifar10_f10_g1.0_c0.5") in by_exp
    assert all(c.seeds == "all" for c in listed)
    # Undefended at each chosen config lacking 5 seeds is scheduled too.
    refs = {c.tag for c in listed if c not in defended}
    assert "e4_composed_backdoor_cifar10_f10_g1.0_c0.5" in refs


def test_undefended_reference_not_rescheduled_once_it_has_five_seeds(tmp_path: Path) -> None:
    reg = sweep_cells({}, "regression", ("H1",))
    ref = next(c for c in reg if c.experiment == "sign_flip_cifar10_f10" and not any(
        o.startswith("+defense") for o in c.overrides))
    _write(tmp_path, ref, range(5), None, 0.70)
    again = sweep_cells(load_cells(tmp_path), "regression", ("H1",))
    assert ref not in again
    assert sum(1 for c in again if c.experiment == "sign_flip_cifar10_f10") == 1


def test_control_variant_is_headline_only() -> None:
    assert sweep_cells({}, "regression", ("H4",)) == []
    assert sweep_cells({}, "femnist", ("H4",)) == []
    assert len([c for c in sweep_cells({}, "femnist", ("H1", "H2", "H3"))
                if any(o.startswith(("+defense", "+baseline")) for o in c.overrides)]) == 6


def test_cli_prints_tab_separated_cells(tmp_path: Path, capsys) -> None:
    assert main(["--results-root", str(tmp_path), "--cells", "rescan", "--variants", "H3"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 16
    tag, exp, seeds, *overrides = lines[0].split("\t")
    assert tag.startswith("e4_") and seeds == "rescan" and overrides[-1] == "+baseline=fltrust_median"
    with pytest.raises(SystemExit):
        main(["--cells", "rescan", "--variants", "H9"])


def test_cli_refuses_to_write_inside_results(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="refusing"):
        main(["--results-root", str(tmp_path), "--out", str(tmp_path / "x.md")])


# ----------------------------------------------------------------------------
# Success criteria
# ----------------------------------------------------------------------------


def _populate(root: Path, variant: str, *, asr: dict[tuple[str, float], float] | None = None,
              acc: float = 0.72, undefended_asr: float = 0.9, skip: set[str] | None = None) -> None:
    """Write a complete, passing grid for ``variant`` (with overrides)."""
    asr = asr or {}
    skip = skip or set()
    for cell in sweep_cells({}, "rescan", (variant,)):
        _write(root, cell, [0], 0.01, acc)  # all equal -> defaults chosen
    cells = load_cells(root)
    listed = sweep_cells(cells, "headline", (variant,))
    listed += sweep_cells(cells, "regression", (variant,))
    listed += sweep_cells(cells, "femnist", (variant,))
    for cell in listed:
        cfg = _config(cell)
        key = cell_key({"config": cfg})
        assert key is not None
        if key.defense == "undefended":
            _write(root, cell, range(5), undefended_asr if _is_targeted(cfg) else None, 0.65)
            continue
        if key.attack in skip:
            continue
        value = asr.get((key.attack, key.fraction), 0.01) if _is_targeted(cfg) else None
        _write(root, cell, range(5), value, acc)


def test_a_variant_meeting_every_threshold_is_the_hybrid(tmp_path: Path) -> None:
    _populate(tmp_path, "H1")
    card = scorecard(load_cells(tmp_path), "H1")
    assert {c: v.status for c, v in card.items()} == dict.fromkeys(card, "PASS")
    assert overall(card) == "meets all five: the hybrid"


@pytest.mark.parametrize(
    ("override", "criterion"),
    [
        ({("composed", 0.2): 0.26}, "C1"),
        ({("direction_aligned", 0.1): 0.08}, "C1"),
        ({("backdoor", 0.1): 0.05}, "C2"),
        ({("backdoor", 0.2): 0.16}, "C2"),
    ],
)
def test_each_asr_threshold_fails_its_criterion(tmp_path: Path, override, criterion) -> None:
    _populate(tmp_path, "H2", asr=override)
    card = scorecard(load_cells(tmp_path), "H2")
    assert card[criterion].status == "FAIL"
    assert overall(card) == f"fails {criterion}"


def test_clean_accuracy_below_threshold_fails_c3(tmp_path: Path) -> None:
    _populate(tmp_path, "H3", acc=0.70)
    card = scorecard(load_cells(tmp_path), "H3")
    assert card["C3"].status == "FAIL"
    assert card["C4"].status == "PASS"  # undefended untargeted acc is 0.65


def test_a_cell_above_undefended_fails_c4(tmp_path: Path) -> None:
    _populate(tmp_path, "H1", undefended_asr=0.005)
    card = scorecard(load_cells(tmp_path), "H1")
    assert card["C4"].status == "FAIL"
    assert "ASR 0.010 > 0.005" in card["C4"].detail


def test_femnist_threshold_fails_c5(tmp_path: Path) -> None:
    _populate(tmp_path, "H1", asr={("backdoor", 0.1): 0.03})  # CIFAR and FEMNIST share the key
    card = scorecard(load_cells(tmp_path), "H1")
    assert card["C5"].status == "PASS"
    _populate(tmp_path / "b", "H1", asr={("backdoor", 0.1): 0.06})
    card = scorecard(load_cells(tmp_path / "b"), "H1")
    assert card["C5"].status == "FAIL" and card["C2"].status == "FAIL"


def test_missing_cells_leave_criteria_pending(tmp_path: Path) -> None:
    _populate(tmp_path, "H1", skip={"composed"})
    card = scorecard(load_cells(tmp_path), "H1")
    assert card["C1"].status == "PENDING"
    assert card["C3"].status == "PENDING"
    assert card["C2"].status == "PASS"
    assert overall(card).startswith("pending")


def test_control_has_no_femnist_criterion(tmp_path: Path) -> None:
    _populate(tmp_path, "H4")
    card = scorecard(load_cells(tmp_path), "H4")
    assert card["C5"].status == "N/A"
    assert overall(card) == "control; no failure"


def test_report_renders_every_section(tmp_path: Path) -> None:
    _populate(tmp_path, "H1")
    report = build_report(load_cells(tmp_path))
    for heading in ("## Success criteria", "## Best-response rescans", "## Headline, CIFAR-10, f = 0.1",
                    "## Headline, CIFAR-10, f = 0.2", "## Regression", "## FEMNIST backdoor"):
        assert heading in report
    assert "meets all five: the hybrid" in report
