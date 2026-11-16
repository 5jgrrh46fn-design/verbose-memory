"""Tables for the last CIFAR-100 runs on the RTX 4090 box.

B1: undefended ASR under the norm-bounded attacker by budget (c100_ tags).
B2: Krum's selections and FLTrust's trust weights from the diag_ cells'
``defense_ledger.json``, with a check that every diagnostic run reproduces
the reported run's metrics (the reported run's columns except the
wall-clock ones; a traced run adds trace_* columns).

ASR and clean accuracy are last-50-round means; cells print mean ± sd (n),
sd over seeds (n - 1). Ledger statistics are computed per seed over the
rounds in which at least one attacker was sampled, then averaged over seeds.

Usage (from views/main, see analysis/reproduce.sh): uv run python analysis/last_runs_c100.py
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

RESULTS = Path("results")
TIMING = {"client_train_s", "scoring_s", "aggregation_s", "server_train_s",
          "evaluation_s", "round_s"}


def seeds_of(label: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    hashes = sorted(p.name for p in (RESULTS / label).glob("*") if p.is_dir())
    if len(hashes) > 1:
        print(f"warning: {label} has {len(hashes)} hashes {hashes}")
    for p in sorted((RESULTS / label).glob("*/seed_*/metrics.json")):
        out[int(p.parent.name.split("_")[1])] = p.parent
    return dict(sorted(out.items()))


def tail(run: Path, key: str, window: int = 50) -> float:
    rows = json.loads((run / "metrics.json").read_text())
    return st.fmean(r[key] for r in rows[-window:])


def msd(vals: list[float], scale: float = 100.0, digits: int = 1) -> str:
    if not vals:
        return "–"
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{scale * st.fmean(vals):.{digits}f} ± {scale * sd:.{digits}f} ({len(vals)})"


def b1() -> None:
    print("### B1. Undefended, norm-bounded attacker, by budget\n")
    print("| fraction | budget | tag | ASR % | clean acc % | seeds | min / max ASR % |")
    print("|---|---|---|---|---|---|---|")
    for f, budgets in (("f10", ("1.0", "1.5", "3.0")), ("f20", ("1.0", "1.5", "3.0"))):
        for g in budgets:
            label = f"c100_adaptive_backdoor_cifar100_{f}_g{g}"
            runs = seeds_of(label)
            asr = [tail(r, "asr_accuracy") for r in runs.values()]
            acc = [tail(r, "test_accuracy") for r in runs.values()]
            span = f"{100 * min(asr):.1f} / {100 * max(asr):.1f}" if asr else "–"
            print(f"| {f[1:]} % | {g} | `{label}` | {msd(asr)} | {msd(acc)} | "
                  f"{','.join(map(str, runs))} | {span} |")
    print()


def same_metrics(a: Path, b: Path) -> bool:
    ra = json.loads((a / "metrics.json").read_text())
    rb = json.loads((b / "metrics.json").read_text())
    # A traced run adds trace_* columns; compare the reported run's columns.
    keys = [k for k in rb[0] if k not in TIMING]
    return len(ra) == len(rb) and all(
        all(x.get(k) == y[k] for k in keys) for x, y in zip(ra, rb, strict=True)
    )


def ledger_rounds(run: Path) -> dict[int, list[dict]]:
    by: dict[int, list[dict]] = defaultdict(list)
    for r in json.loads((run / "defense_ledger.json").read_text()):
        by[int(r["round"])].append(r)
    return by


def norms(run: Path, rounds: set[int]) -> tuple[float, float]:
    mal, ben = [], []
    for r in json.loads((run / "client_metrics.json").read_text()):
        if int(r["round"]) in rounds:
            (mal if r["is_malicious"] else ben).append(float(r["update_norm"]))
    return st.fmean(mal), st.fmean(ben)


def krum_stats(run: Path) -> dict[str, float]:
    by = ledger_rounds(run)
    hit, head, rounds = [], [], set()
    for rnd, rs in by.items():
        mal = [r for r in rs if r["is_malicious"]]
        if not mal:
            continue
        rounds.add(rnd)
        assert abs(sum(float(r["weight"]) for r in rs) - 1.0) < 1e-9
        hit.append(any(float(r["weight"]) > 0 for r in mal))
        head.append(len(mal) / len(rs))
    nm, nb = norms(run, rounds)
    return {"rounds": len(rounds), "of": len(by), "selected": st.fmean(hit),
            "head": st.fmean(head), "norm_mal": nm, "norm_ben": nb}


def fltrust_stats(run: Path) -> dict[str, float]:
    by = ledger_rounds(run)
    top, share, head, rounds = [], [], [], set()
    for rnd, rs in by.items():
        mal = [float(r["weight"]) for r in rs if r["is_malicious"]]
        ben = [float(r["weight"]) for r in rs if not r["is_malicious"]]
        if not mal:
            continue
        rounds.add(rnd)
        top.append(max(mal) > max(ben) if ben else True)
        tot = sum(mal) + sum(ben)
        if tot > 0:
            share.append(sum(mal) / tot)
        head.append(len(mal) / len(rs))
    nm, nb = norms(run, rounds)
    return {"rounds": len(rounds), "of": len(by), "top": st.fmean(top),
            "share": st.fmean(share), "head": st.fmean(head),
            "ratio": st.fmean(share) / st.fmean(head), "norm_mal": nm, "norm_ben": nb}


CELLS = [
    ("fltrust", "adaptive_backdoor_cifar100_f10_g1.0"),
    ("fltrust", "adaptive_backdoor_cifar100_f20_g1.0"),
    ("fltrust", "backdoor_cifar100_f10"),
    ("fltrust", "backdoor_cifar100_f20"),
    ("krum", "adaptive_backdoor_cifar100_f20_g1.0"),
    ("krum", "backdoor_cifar100_f20"),
]


def b2() -> None:
    print("### B2. FLTrust's weights and Krum's selections\n")
    rows_f, rows_k, checks = [], [], []
    for rule, exp in CELLS:
        diag = seeds_of(f"diag_{exp}__{rule}")
        ref = seeds_of(f"c100_{exp}__{rule}")
        und = seeds_of(f"c100_{exp}")
        same = [s for s in diag if s in ref and same_metrics(diag[s], ref[s])]
        diff = [s for s in diag if s in ref and s not in same]
        checks.append(f"| `diag_{exp}__{rule}` | {len(diag)} | {len(same)} | "
                      f"{','.join(map(str, diff)) or '–'} |")
        asr = msd([tail(r, "asr_accuracy") for r in diag.values()])
        uasr = msd([tail(r, "asr_accuracy") for r in und.values()])
        fn = krum_stats if rule == "krum" else fltrust_stats
        per = [fn(r) for r in diag.values()]
        if not per:
            continue
        col = lambda k: [p[k] for p in per]  # noqa: E731
        common = (f"{st.fmean(col('rounds')):.0f} of {per[0]['of']:.0f}", msd(col("head")))
        if rule == "krum":
            rows_k.append(f"| `{exp}` | {len(per)} | {asr} | {uasr} | {common[0]} | "
                          f"{msd(col('selected'))} | {common[1]} | "
                          f"{msd(col('norm_mal'), 1, 3)} | {msd(col('norm_ben'), 1, 3)} |")
        else:
            rows_f.append(f"| `{exp}` | {len(per)} | {asr} | {uasr} | {common[0]} | "
                          f"{msd(col('top'))} | {msd(col('share'))} | {common[1]} | "
                          f"{msd(col('ratio'), 1, 2)} | "
                          f"{msd(col('norm_mal'), 1, 3)} | {msd(col('norm_ben'), 1, 3)} |")
    print("FLTrust (rounds = rounds with at least one sampled attacker, mean per seed):\n")
    print("| cell | seeds | ASR % | undefended ASR % | rounds | attacker strictly top % | "
          "attacker weight share % | head-count share % | weight / head-count | "
          "attacker update norm | honest update norm |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows_f))
    print("\nKrum:\n")
    print("| cell | seeds | ASR % | undefended ASR % | rounds | Krum selects an attacker % | "
          "head-count share % | attacker update norm | honest update norm |")
    print("|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows_k))
    print("\nDiagnostic run against the reported run (`c100_<cell>__<rule>`), every "
          "`metrics.json` column except the wall-clock ones:\n")
    print("| diagnostic cell | seeds | identical to reported | differing seeds |")
    print("|---|---|---|---|")
    print("\n".join(checks))
    print()


if __name__ == "__main__":
    b1()
    b2()
