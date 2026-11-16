"""Tables for the follow-up runs (labels ``fin_*`` and ``fin_k_*``), one section per block.

Reads only runs of this box's campaign: a run counts when its manifest says
``reproducibility.campaign == gpu_b_det`` and its label starts with ``fin_``.
Every comparison is paired on shared seeds inside that campaign (two-sided
Wilcoxon signed-rank, Holm within the family named in the section). ASR and
clean accuracy are last-50-round means; cells print mean ± sd (n).

Usage (repository root; see analysis/reproduce.sh)::

    uv run python analysis/final_queue.py --main-seeds 10 > tables/final-queue.md
    uv run python analysis/final_queue.py --blocks A B     # some sections

A section is printed when at least one of its cells has a finished run, so
the note can be regenerated while the queue runs; ``n`` shows what is in.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from topfed.analysis.stats import holm_bonferroni, wilcoxon_vs_baseline

REPO = Path.cwd()
RESULTS = REPO / "results"
CAMPAIGN = "gpu_b_det"
# The second box (RTX 5070 Ti, campaign ``local``) runs whole sections of the
# queue under the tag prefix ``fin_k_``; ``cell(..., box="k")`` reads them. A
# section is built from one box only.
BOX_K = ("fin_k_", "local")
P3_KNOBS = Path(__file__).resolve().parent / "final-queue-p3-knobs.json"
INTEGRATIONS = [
    ("ScoreWeight", "ScoreWeight_{s}"),
    ("TemporalReputation", "TemporalReputation_{s}"),
    ("HardFilter", "HardFilter_{s}"),
    ("Quarantine", "Quarantine_{s}"),
    ("AdaptiveParticipation", "AdaptiveParticipation_{s}"),
    ("Capped weighted median", "CappedScoreWeight_{s}_wmedian"),
]
GATES = INTEGRATIONS[:5]
PEER = "CappedScoreWeight_critic_wmedian"
MEDIAN = "coordinate_median"
ROOT = "fltrust_median"
FLTRUST = "fltrust"
SW = "ScoreWeight_critic"
FRACTIONS = ("f10", "f20")
OUT: list[str] = []
# Seeds a non-held-out cell may use (``--main-seeds``). The items
# run at seeds 0 to 4 first; a stray higher seed from an interrupted stage
# must not enter a five-seed table.
MAIN_SEEDS: set[int] = set(range(5))


def emit(line: str = "") -> None:
    OUT.append(line)


@dataclass
class Cell:
    label: str
    hashes: list[str] = field(default_factory=list)
    asr: dict[int, float] = field(default_factory=dict)
    acc: dict[int, float] = field(default_factory=dict)
    dirs: dict[int, Path] = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.asr)

    @property
    def n(self) -> int:
        return len(self.asr)

    def values(self, metric: str) -> dict[int, float]:
        return self.asr if metric == "asr" else self.acc


def _tail_mean(rows: list[dict], key: str, window: int = 50) -> float:
    """Last-``window`` mean; NaN when the run has no such metric (untargeted attacks)."""
    if key not in rows[-1]:
        return float("nan")
    return st.mean(float(r[key]) for r in rows[-window:])


@lru_cache(maxsize=None)
def cell(tag: str, suffix: str = "", results: str = "", box: str = "") -> Cell:
    """Every finished seed of ``results/<tag>[__<suffix>]/*`` in one campaign.

    ``box="k"`` reads the second box's runs: the tag's ``fin_`` becomes
    ``fin_k_`` and the campaign is that box's.
    """
    want = CAMPAIGN
    if box == "k":
        tag, want = BOX_K[0] + tag.removeprefix("fin_"), BOX_K[1]
    label = f"{tag}__{suffix}" if suffix else tag
    out = Cell(label=label)
    root = (Path(results) if results else RESULTS) / label
    if not root.is_dir():
        return out
    for h in sorted(root.iterdir()):
        for sd in sorted(h.glob("seed_*")):
            mf, me = sd / "manifest.json", sd / "metrics.json"
            if not (mf.exists() and me.exists()):
                continue
            manifest = json.loads(mf.read_text())
            campaign = (manifest["config"].get("reproducibility") or {}).get("campaign")
            if not results and campaign != want:
                continue
            rows = json.loads(me.read_text())
            if len(rows) != int(manifest["config"]["federation"]["num_rounds"]):
                continue
            seed = int(sd.name.removeprefix("seed_"))
            if not results and "holdout" not in tag and seed not in MAIN_SEEDS:
                continue
            if h.name not in out.hashes:
                out.hashes.append(h.name)
            out.asr[seed] = _tail_mean(rows, "asr_accuracy")
            out.acc[seed] = _tail_mean(rows, "test_accuracy")
            out.dirs[seed] = sd
            out.config = manifest["config"]
    if len(out.hashes) > 1:
        print(f"warning: {label} has {len(out.hashes)} hashes: {out.hashes}", file=sys.stderr)
    return out


def msd(c: Cell, metric: str = "asr") -> str:
    v = [100 * x for x in c.values(metric).values() if x == x]
    if not v:
        return "–"
    sd = st.stdev(v) if len(v) > 1 else 0.0
    return f"{st.mean(v):.1f} ± {sd:.1f} ({len(v)})"


@dataclass
class Test:
    delta: float
    p: float
    n: int
    holm: float | None = None


def paired(a: Cell, b: Cell, metric: str = "asr") -> Test | None:
    """a − b on shared seeds, in points."""
    av, bv = a.values(metric), b.values(metric)
    seeds = sorted(set(av) & set(bv))
    if len(seeds) < 2:
        return None
    x, y = [av[s] for s in seeds], [bv[s] for s in seeds]
    t = wilcoxon_vs_baseline(x, y)
    return Test(delta=100 * (st.mean(x) - st.mean(y)), p=t.p_value, n=len(seeds))


def holm(tests: list[Test | None]) -> None:
    live = [t for t in tests if t is not None]
    if not live:
        return
    adjusted, _ = holm_bonferroni([t.p for t in live])
    for t, a in zip(live, adjusted, strict=True):
        t.holm = a


def fmt_test(t: Test | None) -> tuple[str, str, str]:
    if t is None:
        return "–", "–", "–"
    h = "–" if t.holm is None else f"{t.holm:.3f}"
    return f"{t.delta:+.1f} ({t.n})", f"{t.p:.4f}", h


def table(header: list[str], rows: list[list[str]]) -> None:
    emit("| " + " | ".join(header) + " |")
    emit("|" + "---|" * len(header))
    for r in rows:
        emit("| " + " | ".join(r) + " |")
    emit()


# ---------------------------------------------------------------- ledgers


@lru_cache(maxsize=None)
def ledger_stats(label: str) -> dict[str, float] | None:
    """Pooled over the cell's seeds: exclusion rates and the top-score statistic.

    ``top``: share of rounds with a sampled attacker in which an attacker's
    score equals the round's maximum; ``sole_top``: an attacker's score is
    strictly above every honest client's. ``W_A``: the attackers' mean weight
    share in those rounds, ``rel``: that share over their head-count share.
    """
    tag, _, suffix = label.partition("__")
    c = cell(tag, suffix)
    hon = hon_x = mal = mal_x = rounds = top = sole = 0
    shares: list[float] = []
    rel: list[float] = []
    seen = 0
    for sd in c.dirs.values():
        p = sd / "defense_ledger.json"
        if not p.exists():
            continue
        seen += 1
        by_round: dict[int, list[dict]] = defaultdict(list)
        for r in json.loads(p.read_text()):
            by_round[int(r["round"])].append(r)
        for rs in by_round.values():
            for r in rs:
                excluded = float(r["weight"]) <= 0.0
                if r["is_malicious"]:
                    mal += 1
                    mal_x += excluded
                else:
                    hon += 1
                    hon_x += excluded
            m = [r for r in rs if r["is_malicious"]]
            h = [r for r in rs if not r["is_malicious"]]
            if not m:
                continue
            rounds += 1
            best_m = max(float(r["score"]) for r in m)
            best_h = max((float(r["score"]) for r in h), default=float("-inf"))
            top += best_m >= best_h
            sole += best_m > best_h
            total = sum(float(r["weight"]) for r in rs)
            if total > 0:
                w_a = sum(float(r["weight"]) for r in m) / total
                shares.append(w_a)
                rel.append(w_a / (len(m) / len(rs)))
    if not seen:
        return None
    return {
        "seeds": seen,
        "honest_excluded": hon_x / hon if hon else float("nan"),
        "attacker_excluded": mal_x / mal if mal else float("nan"),
        "top": top / rounds if rounds else float("nan"),
        "sole_top": sole / rounds if rounds else float("nan"),
        "W_A": st.mean(shares) if shares else float("nan"),
        "rel": st.mean(rel) if rel else float("nan"),
    }


def pct(x: float | None) -> str:
    return "–" if x is None or x != x else f"{100 * x:.1f}"


# ------------------------------------------------- the Table 1 layout (A, E)


def backfire_section(
    tag_of: dict[str, str], score: str, rows: list[tuple[str, str]], family_note: str,
    extra_rows: list[tuple[str, str, str]] | None = None,
) -> bool:
    """Integrations of one score against undefended, both fractions.

    ``tag_of`` maps a fraction to the cell tag; ``rows`` are (name, suffix
    template); ``extra_rows`` are (name, tag suffix, label suffix) variants
    that live under ``<tag><tag suffix>``. Holm runs over every row that is
    not the capped median, both fractions together.
    """
    cells = {f: {n: cell(tag_of[f], s.format(s=score)) for n, s in rows} for f in FRACTIONS}
    for name, tsuf, lsuf in extra_rows or []:
        for f in FRACTIONS:
            cells[f][name] = cell(tag_of[f] + tsuf, lsuf)
    if not any(c for f in FRACTIONS for c in cells[f].values()):
        return False
    ref = {f: cell(tag_of[f]) for f in FRACTIONS}
    tests = {f: {n: paired(c, ref[f]) for n, c in cells[f].items()} for f in FRACTIONS}
    in_family = {n for n, _ in GATES} | {n for n, *_ in (extra_rows or [])}
    holm([t for f in FRACTIONS for n, t in tests[f].items() if n in in_family])
    header = ["integration"]
    for f in FRACTIONS:
        header += [f"{f} ASR", f"{f} Δ undefended", f"{f} p", f"{f} Holm p", f"{f} clean acc",
                   f"{f} honest excl. %", f"{f} attacker top % (ties / strict)", f"{f} W_A / (m/K)"]
    body = [["Undefended"] + sum(
        ([msd(ref[f]), "", "", "", msd(ref[f], "acc"), "", "", ""] for f in FRACTIONS), [])]
    for name in cells["f10"]:
        row = [name]
        for f in FRACTIONS:
            c = cells[f][name]
            d, p, h = fmt_test(tests[f][name])
            if name not in in_family:
                h = "(not in family)"
            ls = ledger_stats(c.label) if c else None
            row += [msd(c), d, p, h, msd(c, "acc"),
                    pct(ls["honest_excluded"]) if ls else "–",
                    f"{pct(ls['top'])} / {pct(ls['sole_top'])}" if ls else "–",
                    f"{ls['rel']:.2f}" if ls and ls["rel"] == ls["rel"] else "–"]
        body.append(row)
    table(header, body)
    emit(family_note)
    emit()
    return True


NB_TAG = {f: f"fin_adaptive_backdoor_cifar10_{f}" for f in FRACTIONS}
DA_TAG = {f: f"fin_fltrust_backdoor_cifar10_{f}_c0.9" for f in FRACTIONS}
STD_TAG = {f: f"fin_backdoor_cifar10_{f}" for f in FRACTIONS}
FAMILY = ("Δ and p: paired Wilcoxon against the undefended row of the same fraction "
          "(this box, same seeds); Holm over the gate, mean-weight and sampling rows of "
          "both fractions together (the capped median is tested but not in the family). "
          "Honest excl.: share of honest client-rounds given zero weight. Attacker top: "
          "share of rounds with a sampled attacker in which an attacker's score equals "
          "the round's maximum (ties count) / is strictly above every honest score. W_A / (m/K): attackers' weight share over their "
          "head-count share in those rounds.")


def block_a() -> None:
    printed = False
    for w in ("equal", "entropy"):
        if not any(cell(NB_TAG[f], s.format(s=w)) for f in FRACTIONS for _, s in INTEGRATIONS):
            continue
        if not printed:
            emit("## A. The six integrations under other fusions of the five metrics")
            emit()
            emit("CIFAR-10, norm-bounded backdoor at budget 0.6, seeds 0 to 9, this box. "
                 "Reference rows (CRITIC is the paper's fusion; its integrations were not "
                 "re-run here at seeds 0 to 9, see block B for seeds 10 to 19).")
            emit()
            printed = True
        emit(f"### Weighting: {w}")
        emit()
        backfire_section(NB_TAG, w, INTEGRATIONS, FAMILY)
    if printed:
        rows = []
        for f in FRACTIONS:
            und, peer, med = cell(NB_TAG[f]), cell(NB_TAG[f], PEER), cell(NB_TAG[f], MEDIAN)
            t = paired(peer, und)
            d, p, _ = fmt_test(t)
            rows.append([f, msd(und), msd(peer), msd(med), d, p])
        emit("### This box's reference rows (CRITIC capped median = Peer, plain median)")
        emit()
        table(["f", "undefended", "Peer (CRITIC capped median)", "plain median",
               "Peer − undefended", "p"], rows)


# ------------------------------------------------------------------ block B


def block_b() -> None:
    hold = "fin_holdout_"
    t1 = {f: f"{hold}adaptive_backdoor_cifar10_{f}_g0.6" for f in FRACTIONS}
    if not any(cell(t1[f]) or cell(t1[f], PEER) for f in FRACTIONS):
        return
    emit("## B. Held-out evaluation seeds (10 to 19)")
    emit()
    emit("Each cell runs at the knob selected on seeds 0 to 9; nothing was selected "
         "on these seeds. All cells from this box.")
    emit()
    emit("### Table 1 rows (norm-bounded backdoor, budget 0.6), seeds 10 to 19")
    emit()
    backfire_section(t1, "critic", INTEGRATIONS, FAMILY)

    emit("### Best-response cells of the main table, seeds 10 to 19")
    emit()
    rules = [("Undefended", ""), ("Peer", PEER), ("Root", ROOT), ("Median", MEDIAN),
             ("FLTrust", FLTRUST), ("ScoreWeight", SW)]
    nb = {"f10": {"": "1.0", PEER: "0.6", ROOT: "0.6", MEDIAN: "2.0", FLTRUST: "0.6", SW: "0.6"},
          "f20": {"": "1.0", PEER: "0.6", ROOT: "1.0", MEDIAN: "0.6", FLTRUST: "0.6", SW: "0.6"}}
    da = {"": "0.5", PEER: "0.5", ROOT: "0.9", MEDIAN: "0.5", FLTRUST: "0.9", SW: "0.5"}
    header = ["rule"]
    for f in FRACTIONS:
        header += [f"{f} standard", f"{f} NB (budget)", f"{f} DA (cosine)", f"{f} composed (0.6, 0.9)"]
    body = []
    acc_body = []
    for name, suf in rules:
        row, arow = [name], [name]
        for f in FRACTIONS:
            cs = [
                (cell(f"{hold}backdoor_cifar10_{f}", suf), ""),
                (cell(f"{hold}adaptive_backdoor_cifar10_{f}_g{nb[f][suf]}", suf), f" [{nb[f][suf]}]"),
                (cell(f"{hold}fltrust_backdoor_cifar10_{f}_c{da[suf]}", suf), f" [{da[suf]}]"),
                (cell(f"{hold}composed_backdoor_cifar10_{f}_g0.6_c0.9", suf), ""),
            ]
            row += [msd(c) + k for c, k in cs]
            arow += [msd(c, "acc") + k for c, k in cs]
        body.append(row)
        acc_body.append(arow)
    emit("ASR (percent, mean ± sd (n)); the knob in brackets.")
    emit()
    table(header, body)
    emit("Clean accuracy of the same runs.")
    emit()
    table(header, acc_body)

    emit("### Paired comparisons on the held-out seeds")
    emit()
    emit("Matched-knob pairs only (both rows at the same attack configuration). Holm "
         "within each fraction over the rows listed.")
    emit()
    for f in FRACTIONS:
        pairs = [
            ("standard: Peer − undefended", cell(f"{hold}backdoor_cifar10_{f}", PEER), cell(f"{hold}backdoor_cifar10_{f}")),
            ("standard: Peer − median", cell(f"{hold}backdoor_cifar10_{f}", PEER), cell(f"{hold}backdoor_cifar10_{f}", MEDIAN)),
            ("standard: Peer − FLTrust", cell(f"{hold}backdoor_cifar10_{f}", PEER), cell(f"{hold}backdoor_cifar10_{f}", FLTRUST)),
            ("standard: Root − Peer", cell(f"{hold}backdoor_cifar10_{f}", ROOT), cell(f"{hold}backdoor_cifar10_{f}", PEER)),
            ("standard: ScoreWeight − undefended", cell(f"{hold}backdoor_cifar10_{f}", SW), cell(f"{hold}backdoor_cifar10_{f}")),
            ("NB 0.6: Peer − undefended", cell(t1[f], PEER), cell(t1[f])),
            ("NB 0.6: Peer − FLTrust", cell(t1[f], PEER), cell(t1[f], FLTRUST)),
            ("DA 0.5: Peer − undefended", cell(f"{hold}fltrust_backdoor_cifar10_{f}_c0.5", PEER), cell(f"{hold}fltrust_backdoor_cifar10_{f}_c0.5")),
            ("DA 0.5: Peer − median", cell(f"{hold}fltrust_backdoor_cifar10_{f}_c0.5", PEER), cell(f"{hold}fltrust_backdoor_cifar10_{f}_c0.5", MEDIAN)),
            ("DA 0.9: Root − FLTrust", cell(f"{hold}fltrust_backdoor_cifar10_{f}_c0.9", ROOT), cell(f"{hold}fltrust_backdoor_cifar10_{f}_c0.9", FLTRUST)),
            ("composed: Peer − undefended", cell(f"{hold}composed_backdoor_cifar10_{f}_g0.6_c0.9", PEER), cell(f"{hold}composed_backdoor_cifar10_{f}_g0.6_c0.9")),
            ("composed: Root − undefended", cell(f"{hold}composed_backdoor_cifar10_{f}_g0.6_c0.9", ROOT), cell(f"{hold}composed_backdoor_cifar10_{f}_g0.6_c0.9")),
            ("composed: Root − Peer", cell(f"{hold}composed_backdoor_cifar10_{f}_g0.6_c0.9", ROOT), cell(f"{hold}composed_backdoor_cifar10_{f}_g0.6_c0.9", PEER)),
        ]
        tests = [paired(a, b) for _, a, b in pairs]
        holm(tests)
        emit(f"**{f}**")
        emit()
        table(["comparison (a − b)", "a", "b", "Δ (pairs)", "p", "Holm p"],
              [[n, msd(a), msd(b), *fmt_test(t)] for (n, a, b), t in zip(pairs, tests, strict=True)])


# ------------------------------------------------------------------ block C


def _simple_pairs(title: str, pairs: list[tuple[str, Cell, Cell]], note: str = "") -> None:
    if not any(a or b for _, a, b in pairs):
        return
    tests = [paired(a, b) for _, a, b in pairs]
    holm(tests)
    emit(title)
    emit()
    table(["comparison (a − b)", "a", "b", "Δ (pairs)", "p", "Holm p"],
          [[n, msd(a), msd(b), *fmt_test(t)] for (n, a, b), t in zip(pairs, tests, strict=True)])
    if note:
        emit(note)
        emit()


def block_c() -> None:
    cl = {f: f"fin_clean_label_cifar10_{f}" for f in FRACTIONS}
    f34 = ("f30", "f40")
    have = (
        any(cell(DA_TAG[f], FLTRUST) or cell(cl[f], SW) for f in FRACTIONS)
        or any(cell(f"fin_adaptive_backdoor_cifar10_{f}", PEER) for f in f34)
        or any(cell(f"fin_backdoor_cifar10_{f}_root1", ROOT) for f in FRACTIONS)
        or any(cell(f"fin_h500_backdoor_cifar10_{f}") for f in FRACTIONS)
    )
    if not have:
        return
    emit("## C. Ten-seed cells for five-seed rows, Root above 20 percent, root-set size, horizon")
    emit()
    emit("All cells from this box at seeds 0 to 9 (the paper's five-seed cells are from "
         "the other box and were not extended).")
    emit()
    if any(cell(DA_TAG[f], FLTRUST) or cell(cl[f], SW) for f in FRACTIONS):
        emit("### Rows that have five seeds in the paper")
        emit()
        rows = []
        for f in FRACTIONS:
            rows.append([f"FLTrust, direction-aligned c=0.9, {f}", msd(cell(DA_TAG[f], FLTRUST)),
                         msd(cell(DA_TAG[f], FLTRUST), "acc"), msd(cell(DA_TAG[f])),
                         msd(cell(DA_TAG[f], PEER)), msd(cell(DA_TAG[f], MEDIAN))])
        for f in FRACTIONS:
            rows.append([f"ScoreWeight, clean-label, {f}", msd(cell(cl[f], SW)),
                         msd(cell(cl[f], SW), "acc"), msd(cell(cl[f])),
                         msd(cell(cl[f], PEER)), msd(cell(cl[f], MEDIAN))])
        table(["cell", "ASR", "clean acc", "undefended ASR", "Peer ASR", "median ASR"], rows)
        pairs = []
        for f in FRACTIONS:
            pairs += [
                (f"DA 0.9 {f}: FLTrust − undefended", cell(DA_TAG[f], FLTRUST), cell(DA_TAG[f])),
                (f"DA 0.9 {f}: FLTrust − Peer", cell(DA_TAG[f], FLTRUST), cell(DA_TAG[f], PEER)),
                (f"clean-label {f}: ScoreWeight − undefended", cell(cl[f], SW), cell(cl[f])),
                (f"clean-label {f}: ScoreWeight − Peer", cell(cl[f], SW), cell(cl[f], PEER)),
            ]
        _simple_pairs("Paired, Holm over the eight rows:", pairs)

    if any(cell(f"fin_{a}backdoor_cifar10_{f}", s) for f in f34 for a in ("", "adaptive_")
           for s in ("", PEER, MEDIAN, ROOT)):
        emit("### 30 and 40 percent compromise (standard and norm-bounded at 0.6)")
        emit()
        rows = []
        for a, an in (("", "standard"), ("adaptive_", "norm-bounded 0.6")):
            for f in f34:
                tag = f"fin_{a}backdoor_cifar10_{f}"
                rows.append([f"{an}, {f}"] + [msd(cell(tag, s)) for s in ("", PEER, MEDIAN, ROOT)]
                            + [msd(cell(tag, s), "acc") for s in (PEER, ROOT)])
        table(["cell", "undefended", "Peer", "median", "Root", "Peer clean acc", "Root clean acc"], rows)
        pairs = []
        for a, an in (("", "std"), ("adaptive_", "NB")):
            for f in f34:
                tag = f"fin_{a}backdoor_cifar10_{f}"
                pairs += [
                    (f"{an} {f}: Peer − undefended", cell(tag, PEER), cell(tag)),
                    (f"{an} {f}: Peer − median", cell(tag, PEER), cell(tag, MEDIAN)),
                    (f"{an} {f}: Root − undefended", cell(tag, ROOT), cell(tag)),
                    (f"{an} {f}: Root − Peer", cell(tag, ROOT), cell(tag, PEER)),
                ]
        _simple_pairs("Paired, Holm over the sixteen rows:", pairs)

    if any(cell(f"fin_{a}backdoor_cifar10_{f}_root{v}", ROOT) for f in FRACTIONS
           for a in ("", "adaptive_") for v in (1, 10)):
        emit("### Root against the root-set size (val_per_class 1, 10, 100)")
        emit()
        rows, pairs = [], []
        for a, an in (("", "standard"), ("adaptive_", "norm-bounded 0.6")):
            for f in FRACTIONS:
                base = f"fin_{a}backdoor_cifar10_{f}"
                c1, c10, c100 = cell(base + "_root1", ROOT), cell(base + "_root10", ROOT), cell(base, ROOT)
                rows.append([f"{an}, {f}", msd(c1), msd(c10), msd(c100),
                             msd(c1, "acc"), msd(c10, "acc"), msd(c100, "acc")])
                pairs += [(f"{an} {f}: 1/class − 100/class", c1, c100),
                          (f"{an} {f}: 10/class − 100/class", c10, c100)]
        table(["cell", "ASR @ 1/class", "ASR @ 10/class", "ASR @ 100/class",
               "acc @ 1/class", "acc @ 10/class", "acc @ 100/class"], rows)
        _simple_pairs("Paired against the 100-per-class row, Holm over the eight rows:", pairs,
                      "The existing five-seed cells of the other box (f10 only): "
                      "`e3_[adaptive_]backdoor_cifar10_f10_root{1,10}__fltrust_median[_capped]` "
                      "(listed, not paired with these).")

    if any(cell(f"fin_h500_{a}backdoor_cifar10_{f}", s) for f in FRACTIONS
           for a in ("", "adaptive_") for s in ("", PEER, MEDIAN)):
        emit("### 500 rounds (seeds 0 to 4)")
        emit()
        emit("ASR over 50-round windows ending at the round shown (mean over seeds, "
             "percent), then the last-50 mean ± sd and clean accuracy.")
        emit()
        ends = (100, 200, 300, 400, 500)
        rows = []
        for a, an in (("", "standard"), ("adaptive_", "norm-bounded 0.6")):
            for f in FRACTIONS:
                for name, s in (("undefended", ""), ("Peer", PEER), ("median", MEDIAN)):
                    c = cell(f"fin_h500_{a}backdoor_cifar10_{f}", s)
                    traj = []
                    for e in ends:
                        vals = []
                        for sd in c.dirs.values():
                            m = json.loads((sd / "metrics.json").read_text())
                            vals.append(st.mean(float(r["asr_accuracy"]) for r in m[e - 50:e]))
                        traj.append(f"{100 * st.mean(vals):.1f}" if vals else "–")
                    rows.append([f"{an}, {f}, {name}", *traj, msd(c), msd(c, "acc")])
        table(["cell", *[f"≤{e}" for e in ends], "last 50", "clean acc (last 50)"], rows)


# ------------------------------------------------------------------ block D


def _d_cells() -> dict[tuple[str, str, str], list[str]]:
    """(dataset, attack, fraction) -> tags that have any finished run."""
    out: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for p in sorted(RESULTS.glob("fin_*")):
        tag = p.name.split("__")[0]
        for ds, da, comp in (("FEMNIST", "fin_femnist_fltrust_backdoor_", "fin_femnist_composed_backdoor_"),
                             ("CIFAR-100", "fin_fltrust_backdoor_cifar100_", "fin_composed_backdoor_cifar100_")):
            for attack, prefix in (("direction-aligned", da), ("composed", comp)):
                if tag.startswith(prefix):
                    f = tag.removeprefix(prefix)[:3]
                    if tag not in out[(ds, attack, f)]:
                        out[(ds, attack, f)].append(tag)
    return out


def block_d() -> None:
    cells = _d_cells()
    if not cells:
        return
    emit("## D. Direction-aligned and composed attackers on FEMNIST and CIFAR-100")
    emit()
    emit("### Probe (seed 0, undefended)")
    emit()
    rows = []
    for (ds, attack, f), tags in sorted(cells.items()):
        for tag in tags:
            c = cell(tag)
            if 0 in c.asr:
                rows.append([ds, attack, f, tag.split(f"_{f}_")[-1],
                             f"{100 * c.asr[0]:.1f}", f"{100 * c.acc[0]:.1f}"])
    table(["dataset", "attack", "f", "knob", "ASR seed 0", "clean acc seed 0"], rows)
    main = [(k, t) for k, tags in sorted(cells.items()) for t in tags if cell(t, PEER)]
    if not main:
        return
    emit("### Main cells (seeds 0 to 9)")
    emit()
    rules = [("undefended", ""), ("Peer", PEER), ("Root", ROOT), ("median", MEDIAN), ("FLTrust", FLTRUST)]
    table(["dataset", "attack", "f", "knob", *[n for n, _ in rules], "Peer clean acc", "Root clean acc"],
          [[ds, attack, f, t.split(f"_{f}_")[-1], *[msd(cell(t, s)) for _, s in rules],
            msd(cell(t, PEER), "acc"), msd(cell(t, ROOT), "acc")] for (ds, attack, f), t in main])
    for (ds, attack, f), t in main:
        _simple_pairs(
            f"**{ds}, {attack}, {f}** (Holm over the six rows)",
            [("Peer − undefended", cell(t, PEER), cell(t)),
             ("Peer − median", cell(t, PEER), cell(t, MEDIAN)),
             ("Peer − FLTrust", cell(t, PEER), cell(t, FLTRUST)),
             ("Root − undefended", cell(t, ROOT), cell(t)),
             ("Root − Peer", cell(t, ROOT), cell(t, PEER)),
             ("FLTrust − undefended", cell(t, FLTRUST), cell(t))])


# ------------------------------------------------------------------ block E


def block_e() -> None:
    kinds = ("server_cosine", "history_similarity")
    attacks = (("norm-bounded backdoor, budget 0.6", NB_TAG),
               ("direction-aligned backdoor, cosine 0.9", DA_TAG),
               ("standard backdoor", STD_TAG))
    if not any(cell(tag[f], s.format(s=k)) for k in kinds for _, tag in attacks
               for f in FRACTIONS for _, s in INTEGRATIONS):
        return
    emit("## E. Two further trust scores through the same integrations")
    emit()
    emit("`server_cosine`: FLTrust's score, max(0, cos(update, server update)). "
         "`history_similarity`: FoolsGold's per-client weight from accumulated "
         "histories. Both in [0, 1], passed to the integrator unscaled; thresholds "
         "stay at 0.5 unless the row says cohort median. CIFAR-10, seeds 0 to 9, "
         "this box.")
    emit()
    for k in kinds:
        for an, tag in attacks:
            if not any(cell(tag[f], s.format(s=k)) for f in FRACTIONS for _, s in INTEGRATIONS):
                continue
            emit(f"### {k}: {an}")
            emit()
            extra = []
            for name, cls in (("HardFilter", "HardFilter"), ("Quarantine", "Quarantine")):
                if any(cell(tag[f] + "_medthr", f"{cls}_{k}") for f in FRACTIONS):
                    extra.append((f"{name} (cohort-median threshold)", "_medthr", f"{cls}_{k}"))
            backfire_section(tag, k, INTEGRATIONS, FAMILY, extra)
    emit("### Existing cells that are the server-cosine score in another rule (other box, listed only)")
    emit()
    emit("FLTrust is this score as a mean weight with norm rescaling, Root is this "
         "score in an uncapped median, capped Root in a capped one. Their cells are "
         "the other box's (seeds 0 to 9 unless marked) and are not paired with the rows "
         "above; this box's FLTrust and Root rows on seeds 10 to 19 are in block B.")
    emit()
    other = [
        ("NB 0.6", "adaptive_backdoor_cifar10_{f}", "sp_adaptive_backdoor_cifar10_{f}", "e4_adaptive_backdoor_cifar10_{f}_g0.6"),
        ("DA 0.9", "fltrust_backdoor_cifar10_{f}", "sp_fltrust_backdoor_cifar10_{f}_c0.9", "e4_fltrust_backdoor_cifar10_{f}_c0.9"),
        ("standard", "backdoor_cifar10_{f}", "sp_backdoor_cifar10_{f}", "e4_backdoor_cifar10_{f}"),
    ]
    rows = []
    for an, *tags in other:
        for f in FRACTIONS:
            row = [an, f]
            for suf in (FLTRUST, ROOT, "fltrust_median_capped"):
                found = "–"
                for t in tags:
                    c = _other_box(t.format(f=f), suf, an)
                    if c:
                        found = msd(c)
                        break
                row.append(found)
            rows.append(row)
    table(["attack", "f", "FLTrust", "Root", "capped Root"], rows)


def _other_box(tag: str, suffix: str, attack: str) -> Cell | None:
    """A gpu-a cell at the attack config's default knob, for listing only."""
    label = f"{tag}__{suffix}"
    root = RESULTS / label
    if not root.is_dir():
        return None
    want = {"NB 0.6": ("target_norm", 0.6), "DA 0.9": ("cosine_target", 0.9)}.get(attack)
    best: Cell | None = None
    for h in sorted(root.iterdir()):
        c = Cell(label=label)
        for sd in sorted(h.glob("seed_*")):
            mf, me = sd / "manifest.json", sd / "metrics.json"
            if not (mf.exists() and me.exists()):
                continue
            manifest = json.loads(mf.read_text())
            cfg = manifest["config"]
            um = (cfg.get("attack") or {}).get("update_manipulator") or {}
            if want and float(um.get(want[0], -1)) != want[1]:
                continue
            if int(cfg["dataset"].get("val_per_class", 100)) != 100:
                continue
            if (manifest.get("environment") or {}).get("hostname") != "gpu-a":
                continue
            rows = json.loads(me.read_text())
            seed = int(sd.name.removeprefix("seed_"))
            c.asr[seed] = _tail_mean(rows, "asr_accuracy")
            c.acc[seed] = _tail_mean(rows, "test_accuracy")
        if c and (best is None or c.n > best.n):
            best = c
    return best


# ------------------------------------------------------------------ block G


def block_g() -> None:
    attacks = (("LIE", "fin_lie_cifar10_"), ("Fang (median / trimmed mean)", "fin_fang_cifar10_"))
    rules = [("undefended", ""), ("Peer", PEER), ("Root", ROOT), ("median", MEDIAN),
             ("trimmed mean", "trimmed_mean"), ("FLTrust", FLTRUST)]
    tags = sorted({p.name.split("__")[0] for p in RESULTS.glob("fin_*")
                   if p.name.startswith(("fin_lie_", "fin_fang_"))})
    if not tags:
        return
    emit("## G. Untargeted attacks aimed at a median (appendix only)")
    emit()
    emit("Clean accuracy (percent, last-50-round mean ± sd (n)); the benign reference "
         "is `fin_benign_cifar10` where run. Probe rows are seed 0.")
    emit()
    rows = [["benign (`fin_benign_cifar10`)",
             *[msd(cell("fin_benign_cifar10", s), "acc") for _, s in rules]]]
    rows += [[t, *[msd(cell(t, s), "acc") for _, s in rules]] for t in tags]
    table(["cell", *[n for n, _ in rules]], rows)
    emit("Accuracy change against the benign run of the same rule, paired on shared "
         "seeds (points; pairs in brackets; p from the paired Wilcoxon, which cannot "
         "reach 0.05 below six pairs):")
    emit()
    drops = []
    for t in tags:
        row = [t]
        for _, s in rules:
            a, b = cell(t, s), cell("fin_benign_cifar10", s)
            seeds = sorted(set(a.acc) & set(b.acc))
            if not seeds:
                row.append("–")
            elif len(seeds) == 1:
                row.append(f"{100 * (a.acc[seeds[0]] - b.acc[seeds[0]]):+.1f} (1)")
            else:
                d, pv, _ = fmt_test(paired(a, b, "acc"))
                row.append(f"{d}, p={pv}")
        drops.append(row)
    table(["cell", *[n for n, _ in rules]], drops)
    del attacks


# ------------------------------------------------- blocks P1 to P4


MEDIAN_ROW = ("Plain median", MEDIAN)
COSINES = ("0.5", "0.7", "0.9", "0.95")


def _da_tag(c: str) -> dict[str, str]:
    return {f: f"fin_fltrust_backdoor_cifar10_{f}_c{c}" for f in FRACTIONS}


# Composed-attacker knobs (budget, cosine) of the FoolsGold-weight score.
COMPOSED_KNOBS = (("0.6", "0.9"), ("0.6", "0.5"))
# Threshold rules that also ran with the threshold at the cohort's median
# score on the cosine score: (row name, label suffix), under ``<tag>_medthr``.
MEDTHR_ROWS = (("HardFilter (cohort-median threshold)", "HardFilter_server_cosine"),
               ("Quarantine (cohort-median threshold)", "Quarantine_server_cosine"))


def _composed_tag(g: str, c: str) -> dict[str, str]:
    return {f: f"fin_composed_backdoor_cifar10_{f}_g{g}_c{c}" for f in FRACTIONS}


def _excess_table(entries: list[tuple[str, dict[str, str], str, str]]) -> None:
    """One line per cell: its excess over undefended beside the top-score share.

    ``entries`` are (attack and score, tag per fraction, row name, label
    suffix). The p-values are unadjusted; the Holm values are in the tables
    the cells come from.
    """
    body = []
    for what, tag_of, name, suffix in entries:
        for f in FRACTIONS:
            c = cell(tag_of[f], suffix)
            if not c:
                continue
            d, pv, _ = fmt_test(paired(c, cell(tag_of[f].removesuffix("_medthr"))))
            ls = ledger_stats(c.label)
            body.append([what, name, f, msd(c), d, pv,
                         pct(ls["sole_top"]) if ls else "–",
                         pct(ls["honest_excluded"]) if ls else "–"])
    if body:
        table(["attack, score", "integration", "f", "ASR", "Δ undefended", "p",
               "attacker strictly top %", "honest excl. %"], body)


def block_p1() -> None:
    have_m4 = any(cell(NB_TAG[f], s.format(s="m4_only")) for f in FRACTIONS for _, s in INTEGRATIONS)
    have_cos = any(cell(_da_tag(c)[f], s.format(s="server_cosine")) for c in COSINES
                   for f in FRACTIONS for _, s in INTEGRATIONS)
    have_nb = any(cell(NB_TAG[f], s.format(s=k)) for k in ("server_cosine", "history_similarity")
                  for f in FRACTIONS for _, s in INTEGRATIONS)
    if not (have_m4 or have_cos or have_nb):
        return
    emit("## P1. The integrations with other scores (Table 1 extended to integration x score)")
    emit()
    emit("CIFAR-10, this box. Rows: the five gate / mean-weight / sampling "
         "integrations, the capped weighted median with the same score "
         "(Middleweight with that score) and the plain median. The paper's own "
         "score (five metrics, CRITIC) is the last table of this section.")
    emit()
    rows = [*INTEGRATIONS, MEDIAN_ROW]
    if have_m4:
        emit("### Score: M4 alone (validation-loss change); attacker: norm-bounded, budget 0.6")
        emit()
        backfire_section(NB_TAG, "m4_only", rows, FAMILY)
    if have_cos:
        for c in COSINES:
            tag = _da_tag(c)
            if not any(cell(tag[f], s.format(s="server_cosine")) for f in FRACTIONS for _, s in INTEGRATIONS):
                continue
            emit(f"### Score: FLTrust's ReLU-cosine to the server update; attacker: direction-aligned, c = {c}")
            emit()
            extra = [(name, "_medthr", suffix) for name, suffix in MEDTHR_ROWS
                     if any(cell(tag[f] + "_medthr", suffix) for f in FRACTIONS)]
            backfire_section(tag, "server_cosine", rows, FAMILY, extra)
        emit("### Cosine score: the attacker's best response per integration")
        emit()
        emit("For each row the cosine with the highest mean ASR (same seeds as the "
             "tables above, so the selection is not held out), and the undefended "
             "ASR at that cosine.")
        emit()
        body = []
        names = [(n, s.format(s="server_cosine"), "") for n, s in rows]
        names += [(name, suffix, "_medthr") for name, suffix in MEDTHR_ROWS]
        for name, suffix, tsuf in names:
            row = [name]
            for f in FRACTIONS:
                best = None
                for c in COSINES:
                    cc = cell(_da_tag(c)[f] + tsuf, suffix)
                    if cc and (best is None or st.mean(cc.asr.values()) > st.mean(best[1].asr.values())):
                        best = (c, cc)
                if best is None:
                    row += ["–", "–", "–", "–"]
                    continue
                und = cell(_da_tag(best[0])[f])
                d, pv, _ = fmt_test(paired(best[1], und))
                row += [best[0], msd(best[1]), msd(und), f"{d}, p={pv}"]
            body.append(row)
        header = ["integration"]
        for f in FRACTIONS:
            header += [f"{f} c", f"{f} ASR", f"{f} undefended at c", f"{f} Δ (pairs), p"]
        table(header, body)
    for k, title in (("server_cosine", "FLTrust's ReLU-cosine"), ("history_similarity", "FoolsGold's weight")):
        if any(cell(NB_TAG[f], s.format(s=k)) for f in FRACTIONS for _, s in INTEGRATIONS):
            # The cosine-score cells were run at seeds 0 to 4 only (a stray
            # later seed exists in a few); every row, references included, is
            # cut there. The FoolsGold-weight cells are cut the same way until
            # all of them hold every main seed.
            full = k == "history_similarity" and all(
                set(cell(NB_TAG[f], s.format(s=k)).asr) >= MAIN_SEEDS
                for f in FRACTIONS for _, s in INTEGRATIONS)
            seeds = f"seeds 0 to {max(MAIN_SEEDS)}" if full else "seeds 0 to 4"
            emit(f"### Score: {title}; attacker: norm-bounded, budget 0.6 ({seeds})")
            emit()
            if full:
                backfire_section(NB_TAG, k, rows, FAMILY)
                continue
            keep = set(MAIN_SEEDS)
            MAIN_SEEDS.intersection_update(range(5))
            cell.cache_clear()
            ledger_stats.cache_clear()
            backfire_section(NB_TAG, k, rows, FAMILY)
            MAIN_SEEDS.update(keep)
            cell.cache_clear()
            ledger_stats.cache_clear()
    fg = [s.format(s="history_similarity") for _, s in INTEGRATIONS]
    for g, c in COMPOSED_KNOBS:
        tag = _composed_tag(g, c)
        if not any(cell(tag[f], s) for f in FRACTIONS for s in fg):
            continue
        emit(f"### Score: FoolsGold's weight; attacker: composed, budget {g}, c = {c}")
        emit()
        backfire_section(tag, "history_similarity", rows, FAMILY)
    medthr_q = MEDTHR_ROWS[1]
    entries = [(f"direction-aligned c = {c}, cosine", {f: t + "_medthr" for f, t in _da_tag(c).items()},
                medthr_q[0], medthr_q[1]) for c in COSINES]
    entries += [(f"composed ({g}, {c}), FoolsGold weight", _composed_tag(g, c), n, s.format(s="history_similarity"))
                for g, c in COMPOSED_KNOBS for n, s in INTEGRATIONS]
    if any(cell(t[f], s) for _, t, _, s in entries for f in FRACTIONS):
        emit("### Follow-up cells: excess over undefended beside the attacker's top-score share")
        emit()
        emit("Every cell of the two follow-up blocks (Quarantine with the "
             "cohort-median threshold on the cosine score; the FoolsGold weight "
             "under the composed attacker), one line each: Δ and the unadjusted "
             "paired p from the tables above, the share of rounds with a sampled "
             "attacker in which an attacker's score is strictly above every "
             "honest score, and the share of honest client-rounds given zero "
             "weight.")
        emit()
        _excess_table(entries)
    if any(cell(NB_TAG[f], PEER) for f in FRACTIONS):
        emit("### The paper's score on this box (five metrics, CRITIC): reference rows only")
        emit()
        emit("The five gate / mean integrations with the CRITIC score were not re-run "
             "at seeds 0 to 9 on this box (they are the paper's Table 1, other box); "
             "the capped median (Peer) and the plain median are this box's.")
        emit()
        body = []
        for f in FRACTIONS:
            und, peer, med = cell(NB_TAG[f]), cell(NB_TAG[f], PEER), cell(NB_TAG[f], MEDIAN)
            d, pv, _ = fmt_test(paired(peer, und))
            d2, pv2, _ = fmt_test(paired(peer, med))
            body.append([f, msd(und), msd(peer), msd(med), f"{d}, p={pv}", f"{d2}, p={pv2}"])
        table(["f", "undefended", "Peer", "plain median", "Peer − undefended", "Peer − median"], body)


P2_RULES = [("Undefended", ""), ("Median", MEDIAN), ("Trimmed mean", "trimmed_mean"),
            ("FLTrust", FLTRUST), ("Peer", PEER), ("Root", ROOT)]


def block_p2() -> None:
    zs = {"f10": ("0.5", "1.0", "1.5", "0.11"), "f20": ("0.5", "1.0", "1.5", "0.29")}
    zs_k = ("1.5", "2.0", "3.0")
    def alie(f: str, z: str, suf: str) -> Cell:
        return cell(f"fin_alie_backdoor_cifar10_{f}_z{z}", suf)
    def minmax(f: str, suf: str) -> Cell:
        return cell(f"fin_minmax_backdoor_cifar10_{f}", suf)
    def at(f: str, attack: str, suf: str) -> Cell:
        return minmax(f, suf) if attack == "Min-Max" else alie(f, attack, suf)
    if not any(alie(f, z, s) or minmax(f, s) for f in FRACTIONS for z in zs[f] for _, s in P2_RULES):
        return
    emit("## P2. In-range backdoors (ALIE backdoor and Min-Max)")
    emit()
    emit("CIFAR-10, this box; the count in each cell is its number of seeds "
         "(0 to 4 first, 5 to 9 added later). ALIE backdoor: constrain-and-scale "
         "training, then every coordinate clipped into mu ± z sigma; Min-Max: "
         "mu + gamma p along the backdoor direction with gamma at the max-distance "
         "bound. mu, sigma and the benign set are the attacker's own four clean "
         "shard updates. z_max is Baruch's bound for the population counts (0.11 "
         "at f10, 0.29 at f20); for a ten-client cohort it is not positive. Best "
         "response: the column with the highest mean ASR for that rule (selected "
         "on the same seeds).")
    emit()
    for f in FRACTIONS:
        emit(f"### {f}: ASR (clean accuracy)")
        emit()
        header = ["rule", *[f"ALIE z={z}" + (" (z_max)" if z in ("0.11", "0.29") else "") for z in zs[f]],
                  "Min-Max", "best response", "Δ vs undefended at that attack (pairs), p"]
        body = []
        for name, suf in P2_RULES:
            cols = [(f"ALIE z={z}", alie(f, z, suf), alie(f, z, "")) for z in zs[f]]
            cols.append(("Min-Max", minmax(f, suf), minmax(f, "")))
            live = [(n, c, u) for n, c, u in cols if c]
            best = max(live, key=lambda t: st.mean(t[1].asr.values())) if live else None
            row = [name, *[f"{msd(c)} ({msd(c, 'acc').split(' ±')[0]})" if c else "–" for _, c, _ in cols]]
            if best is None:
                row += ["–", "–"]
            else:
                d, pv, _ = fmt_test(paired(best[1], best[2])) if suf else ("", "", "")
                row += [f"{best[0]}: {msd(best[1])}", f"{d}, p={pv}" if suf else ""]
            body.append(row)
        table(header, body)
        attacks = [*zs[f], "Min-Max"]
        _simple_pairs(f"{f}, rule − undefended at the same attack; Holm over the rows:",
                      [(f"{'Min-Max' if a == 'Min-Max' else f'ALIE z={a}'}: {name} − undefended",
                        at(f, a, suf), at(f, a, ""))
                       for a in attacks for name, suf in P2_RULES if suf])
    emit("Reference on the same seeds, this box: the standard-knob attackers of the "
         "paper against Peer are in section P1 (norm-bounded) and P3 (scan).")
    emit()
    def alie_k(f: str, z: str, suf: str) -> Cell:
        return cell(f"fin_alie_backdoor_cifar10_{f}_z{z}", suf, box="k")
    if not any(alie_k(f, z, s) for f in FRACTIONS for z in zs_k for _, s in P2_RULES):
        return
    emit("### ALIE backdoor past z = 1.5 (second box)")
    emit()
    emit("z = 2 and 3 were added after the first three values showed the ASR still "
         "rising. They ran on the second box (RTX 5070 Ti, campaign `local`, tags "
         "`fin_k_*`), which also repeated z = 1.5 so that the three points share a "
         "campaign. No row here is paired with a row of the tables above; the z = "
         "1.5 column is the same cell as above on the other machine.")
    emit()
    for f in FRACTIONS:
        emit(f"**{f}: ASR (clean accuracy)**")
        emit()
        table(["rule", *[f"ALIE z={z}" for z in zs_k]],
              [[name, *[f"{msd(c)} ({msd(c, 'acc').split(' ±')[0]})" if c else "–"
                        for c in (alie_k(f, z, suf) for z in zs_k)]]
               for name, suf in P2_RULES])
        _simple_pairs(f"{f}, rule − undefended at the same z; Holm over the rows:",
                      [(f"z={z}: {name} − undefended", alie_k(f, z, suf), alie_k(f, z, ""))
                       for z in zs_k for name, suf in P2_RULES if suf])


P3_RULES = [("Peer", "PEER", PEER), ("Root", "ROOT", ROOT), ("Median", "MEDIAN", MEDIAN),
            ("FLTrust", "FLTRUST", FLTRUST)]
P3_GRIDS = {
    "NB": ("norm-bounded", ["0.5", "0.6", "0.75", "1.0", "1.5", "2.0"],
           lambda f, k: f"fin_adaptive_backdoor_cifar10_{f}" + ("" if k == "0.6" else f"_g{k}"),
           lambda f, k: f"fin_holdout_adaptive_backdoor_cifar10_{f}_g{k}"),
    "DA": ("direction-aligned", ["0.5", "0.7", "0.9", "0.95"],
           lambda f, k: f"fin_fltrust_backdoor_cifar10_{f}_c{k}",
           lambda f, k: f"fin_holdout_fltrust_backdoor_cifar10_{f}_c{k}"),
    "COMP": ("composed (budget, cosine)", ["0.6,0.5", "0.6,0.9", "1.0,0.5", "1.0,0.9"],
             lambda f, k: f"fin_composed_backdoor_cifar10_{f}_g{k.split(',')[0]}_c{k.split(',')[1]}",
             lambda f, k: f"fin_holdout_composed_backdoor_cifar10_{f}_g{k.split(',')[0]}_c{k.split(',')[1]}"),
}


def _first5(c: Cell) -> str:
    v = [100 * c.asr[s_] for s_ in range(5) if s_ in c.asr]
    return f"{st.mean(v):.1f} ({len(v)})" if v else "–"


def block_p3() -> None:
    env = json.loads(P3_KNOBS.read_text())["knobs"] if P3_KNOBS.exists() else {}
    scan_any = any(cell(tag(f, k), suf, box="k") for _, (_, knobs, tag, _) in P3_GRIDS.items()
                   for f in FRACTIONS for k in knobs for _, _, suf in P3_RULES)
    if not scan_any:
        return
    emit("## P3. Knobs selected on seeds 0 to 4, evaluated on held-out seeds 10 to 19")
    emit()
    emit("### Scan (seeds 0 to 4): mean ASR by knob")
    emit()
    emit("All of P3 ran on the second box (RTX 5070 Ti, campaign `local`, tags "
         "`fin_k_*`), the machine of the paper's CIFAR-10 cells; no row here is "
         "paired with a row of another section.")
    emit()
    emit("Selection rule: another value replaces the default (budget 0.6, cosine "
         "0.9, composed (0.6, 0.9)) only when it raises the ASR on at least three "
         "of the five seeds with a bias-corrected bootstrap interval above zero; "
         "among such values the highest mean. Selected knob in the last column "
         "(from `analysis/final-queue-p3-knobs.json`, written on that box when the "
         "scan ended; empty while the scan is incomplete).")
    emit()
    for key, (title, knobs, tag, _) in P3_GRIDS.items():
        body = []
        for f in FRACTIONS:
            for name, var, suf in P3_RULES:
                body.append([f, name, *[_first5(cell(tag(f, k), suf, box="k")) for k in knobs],
                             env.get(f"P3_{key}_{var}_{f.upper()}", "")])
        emit(f"**{title}**")
        emit()
        table(["f", "rule", *knobs, "selected"], body)
    if not any(k.startswith("P3_") for k in env):
        return
    emit("### Held-out seeds 10 to 19 at the selected knob")
    emit()
    emit("ASR and clean accuracy; the undefended row is run at the same knob, and "
         "the test is the paired Wilcoxon of rule − undefended on the ten held-out "
         "seeds, Holm within each attack and fraction over the four rules.")
    emit()
    for key, (title, _, _, htag) in P3_GRIDS.items():
        body = []
        for f in FRACTIONS:
            tests, rows = [], []
            for name, var, suf in P3_RULES:
                k = env.get(f"P3_{key}_{var}_{f.upper()}")
                if not k:
                    continue
                c, u = cell(htag(f, k), suf, box="k"), cell(htag(f, k), box="k")
                t = paired(c, u)
                tests.append(t)
                rows.append((name, k, c, u, t))
            holm(tests)
            for name, k, c, u, t in rows:
                body.append([f, name, k, msd(c), msd(c, "acc"), msd(u), *fmt_test(t)])
        if body:
            emit(f"**{title}**")
            emit()
            table(["f", "rule", "knob", "ASR", "clean acc", "undefended ASR at knob",
                   "Δ (pairs)", "p", "Holm p"], body)


GEO = "CappedScoreWeight_critic_gmedian"
GEO_UNCAPPED = "ScoreWeight_critic_gmedian"


def block_p4() -> None:
    cells_ = [
        ("standard", lambda f: f"fin_backdoor_cifar10_{f}"),
        ("norm-bounded 0.6", lambda f: f"fin_adaptive_backdoor_cifar10_{f}"),
        ("norm-bounded 1.0", lambda f: f"fin_adaptive_backdoor_cifar10_{f}_g1.0"),
        ("direction-aligned 0.5", lambda f: f"fin_fltrust_backdoor_cifar10_{f}_c0.5"),
        ("direction-aligned 0.9", lambda f: f"fin_fltrust_backdoor_cifar10_{f}_c0.9"),
        ("clean-label", lambda f: f"fin_clean_label_cifar10_{f}"),
    ]
    if not any(cell(tag(f), GEO) for _, tag in cells_ for f in FRACTIONS):
        return
    emit("## P4. Capped weighted geometric median (RFA with the same capped weights)")
    emit()
    emit("Smoothed Weiszfeld, nu = 1e-6, at most 10 iterations, the Peer weights "
         "(five metrics, CRITIC, cap 1). CIFAR-10, this box.")
    emit()
    body, pairs = [], []
    for name, tag in cells_:
        for f in FRACTIONS:
            g, pe, me, un = cell(tag(f), GEO), cell(tag(f), PEER), cell(tag(f), MEDIAN), cell(tag(f))
            body.append([f"{name}, {f}", msd(g), msd(g, "acc"), msd(pe), msd(pe, "acc"), msd(me), msd(un)])
            pairs += [(f"{name} {f}: geometric − Peer", g, pe),
                      (f"{name} {f}: geometric − undefended", g, un)]
    table(["cell", "capped geometric median ASR", "its clean acc", "Peer ASR", "Peer clean acc",
           "plain median ASR", "undefended ASR"], body)
    _simple_pairs("Paired, Holm over all rows listed:", pairs)
    if any(cell(tag(f), GEO_UNCAPPED) for _, tag in cells_[:2] for f in FRACTIONS):
        emit("### Cap ablation on the geometric median")
        emit()
        body, pairs = [], []
        for name, tag in cells_[:2]:
            for f in FRACTIONS:
                g, gu, un = cell(tag(f), GEO), cell(tag(f), GEO_UNCAPPED), cell(tag(f))
                body.append([f"{name}, {f}", msd(g), msd(gu), msd(un)])
                pairs.append((f"{name} {f}: uncapped − capped", gu, g))
        table(["cell", "capped", "uncapped", "undefended"], body)
        _simple_pairs("Paired, Holm over the rows:", pairs)
    rows = []
    for name, suf in (("Undefended", ""), ("Plain median", MEDIAN), ("Peer", PEER),
                      ("Capped geometric median", GEO)):
        c = cell("fin_overhead_cifar10", suf, box="k")
        for sd in c.dirs.values():
            m = json.loads((sd / "metrics.json").read_text())[5:50]
            server = st.mean(r.get("scoring_s", 0.0) + r.get("aggregation_s", 0.0)
                             + r.get("server_train_s", 0.0) for r in m)
            rows.append([name, f"{server:.3f}", f"{st.mean(r['aggregation_s'] for r in m):.3f}",
                         f"{st.mean(r['client_train_s'] for r in m):.2f}"])
    if rows:
        emit("### Server cost (50-round timing runs on the RTX 5070 Ti box, one at a "
             "time, rounds 6 to 50)")
        emit()
        table(["rule", "server s/round", "aggregation s/round", "client training s/round"], rows)


BLOCKS = {"P1": block_p1, "P2": block_p2, "P3": block_p3, "P4": block_p4,
          "A": block_a, "B": block_b, "C": block_c, "D": block_d, "E": block_e, "G": block_g}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="*", default=["P1", "P2", "P3", "P4"])
    ap.add_argument("--main-seeds", type=int, default=5,
                    help="seeds 0..N-1 enter the non-held-out tables (default 5)")
    args = ap.parse_args()
    MAIN_SEEDS.clear()
    MAIN_SEEDS.update(range(args.main_seeds))
    emit("# Final queue: tables per block")
    emit()
    emit("Generated by "
         "`analysis/final_queue.py` from `results/fin_*` on the RTX 4090 box, "
         f"campaign `{CAMPAIGN}` only, except P3 and the server-cost table, which "
         "come from the RTX 5070 Ti box (`results/fin_k_*`, campaign `local`). ASR and clean accuracy are last-50-round "
         "means in percent, mean ± sd (n seeds); non-held-out cells use seeds 0 to "
         f"{args.main_seeds - 1}. Every test is a two-sided paired "
         "Wilcoxon signed-rank test on seed-matched differences inside this campaign; "
         "Holm families are named per table.")
    emit()
    for b in args.blocks:
        BLOCKS[b]()
    print("\n".join(OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
