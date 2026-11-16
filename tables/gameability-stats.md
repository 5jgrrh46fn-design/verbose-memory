gates, mean weights, sampling: all points: n=70 Spearman rho=0.67 (p=1.7e-10); Kendall tau=0.52 (p=3.6e-09)
same without the standard-backdoor cells: n=60 Spearman rho=0.72 (p=8.1e-11); Kendall tau=0.55 (p=8.9e-09)
capped weighted median only: n=14 Spearman rho=0.33 (p=2.5e-01); Kendall tau=0.30 (p=1.6e-01)
every point incl. capped median: n=84 Spearman rho=0.60 (p=1.4e-09); Kendall tau=0.46 (p=1.2e-08)

cutoff, definition: excess in mean (Δ > 0)
  all four scores: best cutoff anywhere in [29.1, 29.1] %, accuracy 83% of 70 cells
  hold out five metrics     : cutoff from the other three in [29.1, 29.1] (train acc 76%); held-out correct 20/20 at low end, 20/20 at midpoint, 20/20 at high end; positives in held-out 10
  hold out M4 alone         : cutoff from the other three in [40.7, 40.7] (train acc 80%); held-out correct 2/10 at low end, 2/10 at midpoint, 2/10 at high end; positives in held-out 10
  hold out server cosine    : cutoff from the other three in [18.2, 18.2] (train acc 82%); held-out correct 6/10 at low end, 6/10 at midpoint, 6/10 at high end; positives in held-out 1
  hold out FoolsGold weight : cutoff from the other three in [29.1, 29.1] (train acc 98%); held-out correct 19/30 at low end, 19/30 at midpoint, 19/30 at high end; positives in held-out 11

cutoff, definition: excess with unadjusted p < 0.05
  all four scores: best cutoff anywhere in [32.4, 32.4] %, accuracy 97% of 70 cells
  hold out five metrics     : cutoff from the other three in [32.4, 32.4] (train acc 96%); held-out correct 20/20 at low end, 20/20 at midpoint, 20/20 at high end; positives in held-out 10
  hold out M4 alone         : cutoff from the other three in [40.7, 40.7] (train acc 100%); held-out correct 5/10 at low end, 5/10 at midpoint, 5/10 at high end; positives in held-out 7
  hold out server cosine    : cutoff from the other three in [32.4, 32.4] (train acc 97%); held-out correct 10/10 at low end, 10/10 at midpoint, 10/10 at high end; positives in held-out 0
  hold out FoolsGold weight : cutoff from the other three in [32.4, 32.4] (train acc 95%); held-out correct 30/30 at low end, 30/30 at midpoint, 30/30 at high end; positives in held-out 0

| integration | five metrics | M4 alone | server cosine | FoolsGold weight |
|---|---|---|---|---|
| ScoreWeight | +17.4 (f10, norm-bounded, p=0.002, top 55%) | +12.4 (f10, norm-bounded, p=0.019, top 34%) | -0.1 (f20, direction-aligned, p=0.769, top 25%) | +2.1 (f20, composed (0.6, 0.9), p=0.432, top 0%) |
| TemporalReputation | +15.6 (f10, norm-bounded, p=0.002, top 56%) | +10.0 (f10, norm-bounded, p=0.037, top 34%) | -0.6 (f20, direction-aligned, p=0.160, top 22%) | +2.4 (f20, composed (0.6, 0.9), p=0.322, top 0%) |
| HardFilter | +20.4 (f10, norm-bounded, p=0.002, top 57%) | +12.3 (f10, norm-bounded, p=0.037, top 33%) | +0.8 (f10, direction-aligned, p=0.769, top 6%) | +2.3 (f20, composed (0.6, 0.9), p=0.322, top 0%) |
| Quarantine | +8.5 (f10, norm-bounded, p=0.037, top 60%) | +7.1 (f10, norm-bounded, p=0.105, top 32%) | -4.5 (f20, direction-aligned, p=0.002, top 14%) | +2.0 (f20, composed (0.6, 0.5), p=0.375, top 0%) |
| AdaptiveParticipation | +22.2 (f10, norm-bounded, p=0.002, top 58%) | +39.8 (f10, norm-bounded, p=0.002, top 54%) | -21.8 (f20, direction-aligned, p=0.002, top 7%) | +0.7 (f10, composed (0.6, 0.9), p=0.695, top 0%) |
| Capped weighted median | -7.3 (f20, norm-bounded, p=0.002, top 50%) | -12.4 (f10, norm-bounded, p=0.006, top 31%) | -42.6 (f20, direction-aligned, p=0.002, top 5%) | -1.6 (f10, composed (0.6, 0.9), p=0.131, top 0%) |
