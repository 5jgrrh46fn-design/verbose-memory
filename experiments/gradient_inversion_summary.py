"""Summarise gradient-inversion ledgers across seeds as Markdown tables.

Reads every ``ledger.json`` under ``--root`` (one per run of
``gradient_inversion.py``) and prints, for each dataset, model state and
client protocol, a table with one row per attacked target (observer and
aggregation rule) giving the mean and standard deviation over seeds of the
reconstruction quality (SSIM and PSNR, averaged over the reconstructed
images) and of the attacker's final matching loss. A seed that was run more
than once for the same target counts once, as the mean of its runs. The
calibration rows
(a flat gray image, unrelated natural images) are included so a reader can
tell a failed reconstruction from a successful one.

Rows written before the ledger carried ``dataset`` and ``seed`` fields are
labelled from their directory name (``recon_<c10|c100>[_s<seed>]``).

Usage::

    uv run python experiments/gradient_inversion_summary.py --root /tmp/recon > summary.md
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

_DIR = re.compile(r"recon_(c10|c100|cifar10|cifar100)(?:_s(\d+))?$")
_DATASET = {"c10": "cifar10", "c100": "cifar100", "cifar10": "cifar10", "cifar100": "cifar100"}

OBSERVER_LABEL = {
    "server": "server view (one client's raw update)",
    "peer": "peer, standard attack",
    "peer_aware": "peer, rule-aware attack",
    "peer_estimated": "peer, rule-aware + estimated weights",
    "peer_oracle": "peer, rule-aware + true weights (oracle)",
    "calibration": "calibration",
}
RULE_ORDER = (
    "client",
    "mean",
    "scoreweight",
    "hardfilter",
    "trimmed_mean",
    "median",
    "topfed",
    "krum",
    "flat_gray",
    "random_other_images",
)
OBSERVER_ORDER = ("server", "peer", "peer_aware", "peer_estimated", "peer_oracle", "calibration")


def _load(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("ledger.json")):
        if "partial" in str(path.relative_to(root)):
            continue
        m = _DIR.search(path.parent.name)
        ledger = json.loads(path.read_text())
        # One run has one label mode; calibration rows of early label-free
        # ledgers did not record it, so take the mode from the attack rows.
        modes = {r["labels"] for r in ledger if "labels" in r}
        ledger_labels = modes.pop() if len(modes) == 1 else "given"
        for row in ledger:
            if "dataset" not in row:
                if m is None:
                    continue  # an unlabelled smoke run; nothing to attribute it to
                row["dataset"] = _DATASET[m.group(1)]
                row["seed"] = int(m.group(2) or 0)
            row.setdefault("labels", ledger_labels)
            rows.append(row)
    return rows


def _fmt(values: list[float], digits: int) -> str:
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{mean(values):.{digits}f} ± {stdev(values):.{digits}f}"


def _by_seed(rows: list[dict[str, Any]], key: str) -> list[float]:
    """One value per seed: the mean of ``key`` over that seed's runs."""
    per_seed: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if key in r:
            per_seed[r["seed"]].append(r[key])
    return [mean(values) for _, values in sorted(per_seed.items())]


def _included_note(done: list[dict[str, Any]]) -> str:
    """For a rule that drops clients: the SSIM over the peers it kept.

    The SSIM column averages over every peer's images, including peers the
    rule gave zero weight, whose images are not in the aggregate at all.
    """
    rs = [r for r in done if "ssim_included_mean" in r]
    if not rs or all(r["n_included_peers"] == len(r["rule_weights"]) - 1 for r in rs):
        return ""
    kept = mean(r["n_included_peers"] for r in rs)
    return (
        f" (SSIM over included peers only: {_fmt([r['ssim_included_mean'] for r in rs], 3)};"
        f" {kept:.1f} of {len(rs[0]['rule_weights']) - 1} peers included on average)"
    )


def summarise(rows: list[dict[str, Any]]) -> str:
    groups: dict[tuple[str, str, str, str], dict[tuple[str, str], list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        labels = row.get("labels", "given")  # rows before the flag existed: given
        key = (row["dataset"], row["state"], row["protocol"], labels)
        groups[key][(row["observer"], row["rule"])].append(row)

    out: list[str] = []
    for (dataset, state, protocol, labels), targets in sorted(groups.items()):
        seeds = sorted({r["seed"] for rs in targets.values() for r in rs})
        tag = "" if labels == "given" else f", labels {labels}"
        out.append(f"### {dataset}, {state} model, protocol `{protocol}`{tag} (seeds {seeds})\n")
        out.append("| observer | rule | SSIM | PSNR (dB) | match loss | n |")
        out.append("|---|---|---|---|---|---|")
        for observer in OBSERVER_ORDER:
            for rule in RULE_ORDER:
                rs = targets.get((observer, rule))
                if not rs:
                    continue
                done = [r for r in rs if "ssim_mean" in r]
                skipped = len(rs) - len(done)
                why = "Krum chose the attacker" if rule == "krum" else "only the attacker was included"
                note = f" ({skipped} skipped: {why})" if skipped else ""
                note += _included_note(done)
                n_seeds = len({r["seed"] for r in done})
                if len(done) > n_seeds:
                    note += f" ({len(done)} runs, averaged within seed)"
                out.append(
                    f"| {OBSERVER_LABEL[observer]} | {rule} | "
                    f"{_fmt(_by_seed(done, 'ssim_mean'), 3)} | "
                    f"{_fmt(_by_seed(done, 'psnr_mean'), 1)} | "
                    f"{_fmt(_by_seed(done, 'rec_loss'), 3)} | "
                    f"{n_seeds}{note} |"
                )
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    print(summarise(_load(Path(args.root))))


if __name__ == "__main__":
    main()
