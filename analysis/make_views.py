"""Build the two views of results/ that the analyses read.

The follow-up runs (labels ``fin_*``) repeat cells of the main grid on other
seeds or on the second GPU. The main tables read the grid without them, so
that no cell pools runs of both GPUs; ``analysis/final_queue.py`` reads only
the follow-up runs. The views are directories of symlinks:

    views/main/results          every label except fin_*
    views/main_no_c100/results  the same without the CIFAR-100 labels
                                (the fraction-sweep module covers CIFAR-10
                                and FEMNIST only)

Run from the repository root after results/ has been extracted.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def build(name: str, keep) -> int:
    view = ROOT / "views" / name / "results"
    if view.parent.exists():
        shutil.rmtree(view.parent)
    view.mkdir(parents=True)
    n = 0
    for label in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
        if keep(label.name):
            os.symlink(os.path.relpath(label, view), view / label.name)
            n += 1
    return n


def main() -> None:
    if not RESULTS.is_dir():
        raise SystemExit("results/ not found; run scripts/fetch_results.sh first")
    a = build("main", lambda s: not s.startswith("fin_"))
    b = build("main_no_c100", lambda s: not s.startswith("fin_") and "cifar100" not in s)
    print(f"views/main: {a} labels; views/main_no_c100: {b} labels")


if __name__ == "__main__":
    main()
