# Middleweight: artifact

Code, configurations and per-seed results for the paper "Bounded Trust:
Capped-Weight Median Aggregation Under Adaptive Poisoning in Federated
Learning" (anonymous submission).

The repository holds the implementation (package `topfed`), every experiment
configuration, the analysis scripts, and the generated tables and figures.
The run outputs (10,714 runs, about 12 GB) are release assets of this
repository; `scripts/fetch_results.sh` downloads and verifies them.

## Contents

| Path | Content |
|---|---|
| `src/topfed/` | Implementation: attacks, trust metrics, scoring, integrations, aggregation rules, server loop, analysis modules |
| `configs/` | Hydra configurations for every experiment, attack, defense, baseline and dataset |
| `experiments/` | Entry points (`run.py`, `gradient_inversion.py`) and the sweep scripts that list every cell that was run |
| `analysis/` | Scripts that turn `results/` into the paper's tables and figures; `reproduce.sh` runs all of them |
| `tables/`, `figures/` | Output of `analysis/reproduce.sh` on the shipped results |
| `tests/` | Unit and integration tests |
| `SHA256SUMS` | Checksums of the result archives |

Names in the code and in the paper:

| Paper | Code |
|---|---|
| Middleweight-Peer (MW-Peer) | `CappedScoreWeight` integration with `wmedian`, CRITIC weighting (label suffix `__CappedScoreWeight_critic_wmedian`, "capped-wmed" in tables) |
| Middleweight-Root (MW-Root) | `fltrust_median`; the capped variant is "capped Root" |
| Norm-bounded attacker (NB) | `adaptive_backdoor` (budget `g`) |
| Direction-aligned attacker (DA) | `direction_aligned` (cosine target `c`) |
| Composed attacker | `composed` (budget and cosine target) |
| Standard backdoor | `backdoor` ("vanilla backdoor" in tables) |

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
uv run pytest
```

The analyses need no GPU. Training runs used CUDA 13.2 with torch 2.12.0
(pinned in `uv.lock`).

## Results

```bash
scripts/fetch_results.sh
```

This extracts `results/<label>/<config_hash>/seed_<k>/` and
`reconstruction/`. Each run directory holds:

- `manifest.json`: resolved configuration, software and hardware
  environment, determinism flags, the compromised client set.
- `metrics.json`: per-round validation and test accuracy, ASR, timing.
- `client_metrics.json`: the five behavioural metrics of every sampled
  client in every round, with the ground-truth label.
- `defense_ledger.json` (runs of rules that assign scores or weights): every
  sampled client's score, weight and label in every round.
- `collusion_ledger.json`, `median_selection.json`: diagnostic records of
  the collusion and coordinate-selection runs.

The config hash excludes the seed, so all seeds of a configuration share a
directory. `reconstruction/` holds the gradient-inversion ledgers
(`recon_*/ledger.json`), the reconstructed image grids and the image strips
of the reconstruction figure.

Notes on the manifests:

- The runs were made on two machines, written as `gpu-a` (one RTX 5070 Ti)
  and `gpu-b` (one RTX 4090) in `environment.hostname`. The hostnames were
  replaced for anonymity, and the campaign tag of `gpu-b` was renamed to
  match.
- `git_sha` is blank. It held commit ids of the development history, and
  this repository is a single commit of the final code. Nothing else in
  the run outputs was changed.
- `config.reproducibility.campaign` separates the machines and phases:
  `local` is `gpu-a`; `gpu_b_det` and a missing value are `gpu-b`.
- All seeds of a configuration share one data partition and one server
  validation split; the seed varies the model initialisation, the client
  sampling, the compromised set and the order of SGD. The validation slice
  is cut from the test set and is excluded from the reported accuracy and
  ASR.
- For the BackdoorIndicator baseline the behavioural metrics in
  `client_metrics.json` are taken against the previous aggregate and so
  include the server's injection step; accuracy and ASR are unaffected.
- Seven runs that did not finish and the preliminary ResNet-18 runs
  mentioned in the paper's limitations are not included.

## Reproducing the tables and figures

```bash
analysis/reproduce.sh
```

This takes about ten minutes and rewrites `tables/` and `figures/`. The
committed copies were produced by the same command, so `git diff` shows any
difference.

The follow-up runs (labels `fin_*`, `fin_k_*`) repeat cells of the main grid
on other seeds or on the other GPU. The main tables are computed without
them so that no cell mixes machines; `analysis/make_views.py` builds that
view as a directory of symlinks (`views/main/results`).

| Paper item | Output | Produced by |
|---|---|---|
| Table 1 (integrations of one score), left group | `tables/supplementary-cells.md`, section A | `analysis/paper_supplementary_cells.py` |
| The same cells as a bar chart (not in the paper) | `figures/backfire.pdf` | `analysis/paper_figures.py` |
| Table 1, middle and right groups | `tables/final-queue.md`, block P1 | `analysis/final_queue.py` |
| Figure 1 (backfire against top-score share) | `figures/gameability.pdf`; points in `tables/top-score-share.md`, `tables/standard-backdoor-points.md`, `tables/gameability-stats.md` | `analysis/paper_figures_final.py`, `top_score_share.py`, `standard_backdoor_points.py`, `gameability_stats.py` |
| Table 2 (ASR, CIFAR-10 and FEMNIST) and the clean-accuracy table of the appendix | `tables/e3-tables.md`, sections cifar10 and femnist | `topfed.analysis.e3` |
| Capped geometric median row of both | `tables/final-queue.md`, block P4 | `analysis/final_queue.py` |
| Table 3 (CIFAR-100) | `tables/e3-tables.md`, section cifar100; undefended by budget in `tables/cifar100-budgets-and-ledgers.md` | `topfed.analysis.e3`, `analysis/last_runs_c100.py` |
| Table 4 (outcome matrix) | Letters assigned by the rule of the paper's statistical protocol from the cells of `e3-tables.md`, `e2-tables.md`, `e4-tables.md`, `final-queue.md` (P2) and `composed-and-knob-scans.md` | see those rows |
| Table 5 (in-range backdoors) | `tables/final-queue.md`, block P2 | `analysis/final_queue.py` |
| Selection rates of Krum and top-score shares of FLTrust | `tables/diag-ledgers.md`, `tables/cifar100-budgets-and-ledgers.md` | `analysis/diag_ledgers.py`, `last_runs_c100.py` |
| Published rules against the undefended model, paired tests | `tables/published-vs-undefended.md` | `analysis/published_vs_undefended.py` |
| Composed attacker against undefended; knob scans | `tables/composed-and-knob-scans.md`, `tables/e4-tables.md` | `analysis/last_runs_analysis.py`, `topfed.analysis.e4` |
| Defense-ledger weight statistics | `tables/amplification-weights.md` | `analysis/amplification_weights.py` |
| Coordinate-level selection of the median | `tables/median-selection.md` | `analysis/median_selection.py` |
| Detector AUC by weighting and leave-one-out | `tables/auc-tables.md` | `analysis/auc_tables.py` |
| Server cost table | `tables/e3-tables.md` (server overhead), `tables/final-queue.md` (server cost) | `topfed.analysis.e3`, `analysis/final_queue.py` |
| Held-out knob selection table | `tables/final-queue.md`, block P3 | `analysis/final_queue.py` with `analysis/final-queue-p3-knobs.json` |
| Reconstruction SSIM table | `tables/reconstruction.md` | `experiments/gradient_inversion_summary.py` |
| Reconstruction figure | `figures/recon.pdf` | `analysis/paper_figures_final.py` from `reconstruction/strips/` |
| Fraction-sweep figure and clean accuracy by fraction | `figures/sweep-cifar10-asr.pdf`, `figures/sweep-cifar10-acc.pdf`, `tables/e2-tables.md` | `analysis/paper_figures.py`, `topfed.analysis.e2` |

Values are fractions in the `e2`, `e3` and `e4` tables and in
`amplification-weights.md` and `median-selection.md`, and percent elsewhere. ASR and clean accuracy are means over the last 50 of 200 rounds.

Limits of the regeneration:

- The five-metric and standard-backdoor points of Figure 1 are constants in
  `analysis/paper_figures_final.py` and `analysis/gameability_stats.py`;
  `top_score_share.py`, `standard_backdoor_points.py` and section A of
  `supplementary-cells.md` print the values they were copied from.
- The scripts in `analysis/` read `results/` relative to the working
  directory. Run them as `reproduce.sh` does (most of them from
  `views/main`); from the repository root they would also pool the
  follow-up seeds.
- Server cost is read from timing columns of single runs; rerunning gives
  other timings.
- The reconstruction image strips were cut from the grids written during
  the GPU runs and are shipped as images.
- `analysis/amplification_weights.py` lists four ResNet-18 cells whose runs
  are not included; their rows are empty.
- `analysis/final-queue-p3-knobs.json` is the knob selection on seeds 0 to
  4; `analysis/final_queue_probe.py p3 --campaign local --prefix fin_k_`
  recomputes it.

## Running experiments

```bash
uv run python experiments/run.py +experiment=backdoor_cifar10_f10
uv run python experiments/run.py +experiment=adaptive_backdoor_cifar10_f10 \
    +defense=topsis_capped_weighted_median federation.seed=3
```

Experiments are selected with `+experiment=<name>` from
`configs/experiment/`, defenses with `+defense=<name>` and published
baselines with `+baseline=<name>`. Outputs go to
`results/<label>/<config_hash>/seed_<k>/`. Overriding `results_root` or
`data_root` changes the config hash; leave them at their defaults to land on
the hash of the shipped run. The `experiments/sweep_*.sh`
scripts list the exact overrides of every block of runs. One 200-round
CIFAR-10 run takes about 10 minutes on the GPUs above. CIFAR-10 and
CIFAR-100 download on first use; FEMNIST is fetched through
`flwr-datasets`.

Runs are deterministic on a given GPU and software stack
(`reproducibility.deterministic: true` by default). Across GPU
architectures results agree statistically, not bit for bit; cells run on
both machines agree within 0.7 points.

## License

MIT, see `LICENSE`.
