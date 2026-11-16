"""Two further paper figures, as vector PDFs.

  gameability.pdf  x: share of attacker rounds in which an attacker's score is
                   strictly above every honest score; y: last-50-round ASR
                   minus the undefended ASR at the same attack. One point per
                   (score, integration, fraction).
  recon.pdf        true images, what a participating client reconstructs from
                   the FedAvg aggregate, and from the Middleweight aggregate
                   (rule-aware attacker); CIFAR-10 and CIFAR-100, seed 1, warm
                   model, one image per client, one step.

Usage (repository root, see analysis/reproduce.sh):
  uv run python analysis/paper_figures_final.py --out-dir figures

Sources. M4-only, cosine and FoolsGold scores: the tables of
tables/final-queue.md (section P1), parsed here. Five-metric
score: FIVE below, computed from results/adaptive_backdoor_cifar10_f{10,20}__*
(local campaign, budget 0.6, seeds 0 to 9) with the same statistic.
Reconstruction strips: reconstruction/strips.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
BLACK, ORANGE, SKY, GREEN, BLUE, VERMILLION, PURPLE = (
    "#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7")
GATES = ["ScoreWeight", "TemporalReputation", "HardFilter", "Quarantine", "AdaptiveParticipation"]
CAPPED = "Capped weighted median"

# (integration, fraction) -> (delta ASR, strict top-score share), five-metric
# CRITIC score, norm-bounded attacker at budget 0.6, ten seeds, local campaign.
FIVE = {
    ("ScoreWeight", 10): (17.4, 55.2), ("TemporalReputation", 10): (15.6, 55.8),
    ("HardFilter", 10): (20.4, 57.0), ("Quarantine", 10): (8.5, 60.3),
    ("AdaptiveParticipation", 10): (22.2, 57.7), (CAPPED, 10): (-11.9, 54.1),
    ("ScoreWeight", 20): (11.7, 59.4), ("TemporalReputation", 20): (10.8, 59.6),
    ("HardFilter", 20): (12.2, 61.4), ("Quarantine", 20): (6.7, 59.8),
    ("AdaptiveParticipation", 20): (15.5, 64.1), (CAPPED, 20): (-7.3, 50.2),
}

# Same score under the standard backdoor (the score ranks the attacker low):
# (delta ASR, strict top-score share), results/backdoor_cifar10_f{10,20}__*_critic
# against results/backdoor_cifar10_f{10,20} on shared seeds (ten for ScoreWeight
# and the capped median, five for the others); analysis/standard_backdoor_points.py.
STANDARD = {
    ("ScoreWeight", 10): (-66.7, 0.0), ("TemporalReputation", 10): (-59.1, 0.0),
    ("HardFilter", 10): (-74.1, 0.0), ("Quarantine", 10): (-31.4, 0.2),
    ("AdaptiveParticipation", 10): (-77.7, 0.0), (CAPPED, 10): (-77.1, 0.0),
    ("ScoreWeight", 20): (-31.7, 1.0), ("TemporalReputation", 20): (-20.0, 1.7),
    ("HardFilter", 20): (-69.3, 0.3), ("Quarantine", 20): (-6.5, 4.4),
    ("AdaptiveParticipation", 20): (-75.2, 0.0), (CAPPED, 20): (-73.0, 0.3),
}


def _rc() -> None:
    plt.rcParams.update({
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "legend.fontsize": 7, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.compression": 9,
    })


def _sections(text: str) -> dict[str, dict[tuple[str, int], tuple[float, float]]]:
    """Per '### Score: ...' heading: (integration, fraction) -> (delta, strict top share)."""
    out: dict[str, dict[tuple[str, int], tuple[float, float]]] = {}
    title = None
    for line in text.splitlines():
        if line.startswith("### "):
            title = line[4:].strip() if line.startswith("### Score:") else None
            if title:
                out[title] = {}
            continue
        if title is None or not line.startswith("|"):
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 16 or cols[0] in ("integration", "---", "Undefended", "Plain median"):
            continue
        if "cohort-median" in cols[0]:
            continue
        for f, d_i, t_i in ((10, 2, 7), (20, 10, 15)):
            d = re.match(r"([+-][0-9.]+)", cols[d_i])
            t = re.match(r"[0-9.]+ / ([0-9.]+)", cols[t_i])
            if d and t:
                out[title][(cols[0], f)] = (float(d.group(1)), float(t.group(1)))
    return out


XMIN = -6.0  # room left of 0 for the cells that sit at exactly 0


def _spread_left(ax, pts, gap_pt: float = 2.6, columns: int = 5):
    """Move points that share x = 0 into a few columns left of 0.

    Points are placed in order of the vertical value, each in the column
    nearest to 0 that keeps it gap_pt points away from those already placed;
    where no column is free (a dense cluster) it takes the least crowded one.
    The vertical value is unchanged."""
    fig = ax.figure
    fig.canvas.draw()
    to_pt = lambda x, y: ax.transData.transform((x, y)) * 72.0 / fig.dpi  # noqa: E731
    step = gap_pt / (to_pt(1, 0)[0] - to_pt(0, 0)[0])  # x units per column
    placed: list[tuple[float, float]] = []
    out = []
    for key, (d, _) in sorted(pts, key=lambda kv: kv[1][0]):
        best = None
        for k in range(columns):
            x = -(k + 0.6) * step
            px, py = to_pt(x, d)
            near = min(((px - qx) ** 2 + (py - qy) ** 2 for qx, qy in placed), default=1e9) ** 0.5
            if near >= gap_pt:
                best = (near, x, px, py)
                break
            if best is None or near > best[0]:
                best = (near, x, px, py)
        _, x, px, py = best
        placed.append((px, py))
        out.append((key, (d, x)))
    assert min(x for _, (_, x) in out) > XMIN, "spread does not fit left of 0"
    return out


def fig_gameability(out: Path) -> None:
    sec = _sections((BASE / "tables" / "final-queue.md").read_text())
    key = {k.split(";")[0] + ";" + k.split(";")[1]: v for k, v in sec.items()}
    m4 = next(v for k, v in sec.items() if "M4 alone" in k)
    # FoolsGold weight: the norm-bounded table (ten seeds) and both composed tables.
    fools = [item for k, v in sec.items() if "FoolsGold" in k for item in v.items()]
    cos = {c: next(v for k, v in sec.items() if "direction-aligned" in k and f"c = {c}" in k)
           for c in ("0.5", "0.7", "0.9", "0.95")}
    # cosine score: each integration at the cosine with the highest ASR, i.e. the
    # largest delta is not the criterion; pick by the best-response table.
    best = {}
    text = (BASE / "tables" / "final-queue.md").read_text()
    blk = text.split("### Cosine score: the attacker's best response per integration")[1].split("###")[0]
    for line in blk.splitlines():
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) >= 9 and cols[0] in GATES + [CAPPED]:
            best[(cols[0], 10)], best[(cols[0], 20)] = cols[1], cols[5]
    cosine = {k: cos[c][k] for k, c in best.items()}
    del key
    series = [
        ("five metrics (norm-bounded)", list(FIVE.items()), VERMILLION, "o"),
        ("M4 alone (norm-bounded)", list(m4.items()), BLUE, "s"),
        ("server cosine (direction-aligned)", list(cosine.items()), GREEN, "^"),
        ("FoolsGold weight (all at 0, spread left)", fools, PURPLE, "D"),
        ("five metrics (standard backdoor)", list(STANDARD.items()), "#8C8C8C", "v"),
    ]
    _rc()
    fig, ax = plt.subplots(figsize=(3.5, 2.15))
    ax.set_xlim(XMIN, 70)
    ax.set_ylim(-80, 45)
    ax.set_xlabel("rounds in which an attacker tops the score (%)")
    ax.set_ylabel("ASR minus undefended", fontsize=8)
    ax.set_xticks(range(0, 71, 10))
    fig.tight_layout(pad=0.3)
    ax.axhline(0, color=BLACK, lw=0.8, ls="--")
    ax.axvline(0, color="#B0B0B0", lw=0.6, ls=":", zorder=1)
    for name, pts, colour, marker in series:
        for x, y in ((t, d) for (i, _), (d, t) in pts if i in GATES):
            print(f"gameability {name:42s} gate/mean  top={x:5.1f} delta={y:+6.1f}")
        for x, y in ((t, d) for (i, _), (d, t) in pts if i == CAPPED):
            print(f"gameability {name:42s} capped med top={x:5.1f} delta={y:+6.1f}")
        if "FoolsGold" in name:
            # every FoolsGold cell is at exactly 0: spread them to the left of 0
            assert all(t == 0.0 for _, (_, t) in pts)
            pts = _spread_left(ax, pts)
        gx = [t for (i, _), (d, t) in pts if i in GATES]
        gy = [d for (i, _), (d, t) in pts if i in GATES]
        cx = [t for (i, _), (d, t) in pts if i == CAPPED]
        cy = [d for (i, _), (d, t) in pts if i == CAPPED]
        small = "FoolsGold" in name
        ax.scatter(gx, gy, s=9 if small else 16, color=colour, marker=marker, lw=0, label=name, zorder=3)
        ax.scatter(cx, cy, s=10 if small else 20, facecolors="none", edgecolors=colour, marker=marker,
                   lw=0.7 if small else 0.9, zorder=3)
    ax.scatter([], [], s=20, facecolors="none", edgecolors=BLACK, marker="o", lw=0.9,
               label="hollow: capped weighted median")
    ax.legend(loc="lower right", frameon=False, handletextpad=0.2, borderaxespad=0.1)
    fig.savefig(out / "gameability.pdf")
    plt.close(fig)


def fig_recon(out: Path, k: int = 3) -> None:
    """The first k of the nine peers per dataset, the two datasets side by side."""
    img = BASE / "reconstruction" / "strips"
    rows = [("true", "true"), ("mean", "FedAvg"), ("topfed_aware", "MW-Peer")]
    pitch, side = 83.5, 75  # each strip holds nine 75-pixel images
    width = int(round((k - 1) * pitch)) + side
    _rc()
    left, gap, top_in = 0.045, 0.03, 0.15
    w_ax = (1.0 - left - gap) / 2
    h_row_in = 3.5 * w_ax * 74 / width
    fig_h = 3 * h_row_in * 1.04 + top_in
    fig = plt.figure(figsize=(3.5, fig_h))
    for c, (ds, label) in enumerate((("c10", "CIFAR-10"), ("c100", "CIFAR-100"))):
        for r, (name, text) in enumerate(rows):
            h = h_row_in / fig_h
            ax = fig.add_axes((left + c * (w_ax + gap), 1 - top_in / fig_h - (r + 1) * h * 1.04 + 0.04 * h, w_ax, h))
            strip = mpimg.imread(img / f"{ds}_s1_warm_b1_{name}.png")
            strip = strip[-74:, :width]  # the true-image strips carry a caption sliver on top
            ax.imshow(strip, interpolation="lanczos", aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c == 0:
                ax.set_ylabel(text, fontsize=7, labelpad=1.5)
            if r == 0:
                ax.set_title(label, fontsize=7.5, pad=2, loc="left")
    fig.savefig(out / "recon.pdf", dpi=300)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    fig_gameability(a.out_dir)
    fig_recon(a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
