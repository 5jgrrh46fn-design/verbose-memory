# E2 compromise-rate sensitivity

Converged values average the last 50 rounds; cells read 'mean ± sd (n)'. 'Breaking f' is the first f whose mean ASR reaches 0.5. Kf is the expected number of attackers among the K=10 sampled clients. Only reference-setting runs (N=100, K=10, 200 rounds, native partition, corner trigger) are included.

Runs found: 2976 in 448 cells.

## Sampling tail

A round with m >= 5 attackers can hold half the weight of an equal-weight cohort, the weighted median's breakdown point. Observed counts come from the client ledgers, one run per seed.

### cifar10

| f | attackers | E[m] = Kf | P(m>=5) per round | E[first such round] | P(any in 200) | observed rounds with m>=5 | seeds with such a round | max m seen |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 5 | 0.5 | 0.0000 | 298760 | 0.001 | 0.0000 | 0/5 | 4 |
| 0.1 | 10 | 1 | 0.0007 | 1489 | 0.126 | 0.0005 | 1/10 | 5 |
| 0.15 | 15 | 1.5 | 0.0063 | 158 | 0.718 | 0.0050 | 5/5 | 5 |
| 0.2 | 20 | 2 | 0.0255 | 39 | 0.994 | 0.0285 | 10/10 | 6 |
| 0.25 | 25 | 2.5 | 0.0675 | 15 | 1.000 | 0.0650 | 5/5 | 7 |
| 0.3 | 30 | 3 | 0.1384 | 7 | 1.000 | 0.1460 | 10/10 | 8 |
| 0.4 | 40 | 4 | 0.3614 | 3 | 1.000 | 0.3540 | 10/10 | 9 |

Per-round malicious count, observed / hypergeometric:

| f | m=0 | m=1 | m=2 | m=3 | m=4 | m=5 | m=6 | m>=7 |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.590 / 0.584 | 0.330 / 0.339 | 0.073 / 0.070 | 0.006 / 0.006 | 0.001 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| 0.1 | 0.341 / 0.330 | 0.382 / 0.408 | 0.217 / 0.202 | 0.051 / 0.052 | 0.009 / 0.008 | 0.001 / 0.001 | 0.000 / 0.000 | 0.000 / 0.000 |
| 0.15 | 0.194 / 0.181 | 0.358 / 0.357 | 0.314 / 0.292 | 0.097 / 0.130 | 0.032 / 0.034 | 0.005 / 0.006 | 0.000 / 0.001 | 0.000 / 0.000 |
| 0.2 | 0.092 / 0.095 | 0.250 / 0.268 | 0.317 / 0.318 | 0.214 / 0.209 | 0.098 / 0.084 | 0.024 / 0.022 | 0.005 / 0.004 | 0.000 / 0.000 |
| 0.25 | 0.054 / 0.048 | 0.195 / 0.181 | 0.296 / 0.292 | 0.254 / 0.264 | 0.136 / 0.147 | 0.044 / 0.053 | 0.019 / 0.012 | 0.002 / 0.002 |
| 0.3 | 0.021 / 0.023 | 0.106 / 0.113 | 0.243 / 0.237 | 0.273 / 0.281 | 0.210 / 0.208 | 0.105 / 0.100 | 0.028 / 0.031 | 0.013 / 0.007 |
| 0.4 | 0.007 / 0.004 | 0.036 / 0.034 | 0.111 / 0.115 | 0.242 / 0.220 | 0.249 / 0.264 | 0.199 / 0.208 | 0.106 / 0.108 | 0.050 / 0.046 |

### femnist

| f | attackers | E[m] = Kf | P(m>=5) per round | E[first such round] | P(any in 200) | observed rounds with m>=5 | seeds with such a round | max m seen |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 5 | 0.5 | 0.0000 | 298760 | 0.001 | 0.0000 | 0/5 | 4 |
| 0.1 | 10 | 1 | 0.0007 | 1489 | 0.126 | 0.0005 | 1/10 | 5 |
| 0.15 | 15 | 1.5 | 0.0063 | 158 | 0.718 | 0.0050 | 5/5 | 5 |
| 0.2 | 20 | 2 | 0.0255 | 39 | 0.994 | 0.0285 | 10/10 | 6 |

Per-round malicious count, observed / hypergeometric:

| f | m=0 | m=1 | m=2 | m=3 | m=4 | m=5 | m=6 | m>=7 |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.590 / 0.584 | 0.330 / 0.339 | 0.073 / 0.070 | 0.006 / 0.006 | 0.001 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| 0.1 | 0.341 / 0.330 | 0.382 / 0.408 | 0.217 / 0.202 | 0.051 / 0.052 | 0.009 / 0.008 | 0.001 / 0.001 | 0.000 / 0.000 | 0.000 / 0.000 |
| 0.15 | 0.194 / 0.181 | 0.358 / 0.357 | 0.314 / 0.292 | 0.097 / 0.130 | 0.032 / 0.034 | 0.005 / 0.006 | 0.000 / 0.001 | 0.000 / 0.000 |
| 0.2 | 0.092 / 0.095 | 0.250 / 0.268 | 0.317 / 0.318 | 0.214 / 0.209 | 0.098 / 0.084 | 0.024 / 0.022 | 0.005 / 0.004 | 0.000 / 0.000 |

## CIFAR-10 vanilla backdoor

ASR:

| defense | f=0.05 (Kf=0.5) | f=0.1 (Kf=1) | f=0.15 (Kf=1.5) | f=0.2 (Kf=2) | f=0.25 (Kf=2.5) | f=0.3 (Kf=3) | f=0.4 (Kf=4) | breaking f |
|---|---|---|---|---|---|---|---|---|
| undefended | 0.627 ± 0.098 (n=5) | 0.810 ± 0.033 (n=10) | 0.808 ± 0.055 (n=5) | 0.881 ± 0.043 (n=10) | 0.861 ± 0.041 (n=5) | 0.924 ± 0.013 (n=5) | 0.937 ± 0.013 (n=5) | 0.05 |
| ScoreWeight | 0.033 ± 0.012 (n=5) | 0.143 ± 0.068 (n=10) | 0.319 ± 0.064 (n=5) | 0.564 ± 0.060 (n=10) | 0.694 ± 0.056 (n=5) | 0.821 ± 0.031 (n=5) | 0.911 ± 0.016 (n=5) | 0.2 |
| capped-wmed | 0.024 ± 0.004 (n=5) | 0.039 ± 0.005 (n=10) | 0.046 ± 0.003 (n=5) | 0.151 ± 0.046 (n=10) | 0.354 ± 0.085 (n=5) | 0.612 ± 0.119 (n=10) | 0.886 ± 0.031 (n=5) | 0.3 |
| median | 0.054 ± 0.007 (n=5) | 0.197 ± 0.028 (n=10) | 0.352 ± 0.054 (n=5) | 0.583 ± 0.052 (n=10) | 0.711 ± 0.037 (n=5) | 0.819 ± 0.045 (n=5) | 0.929 ± 0.015 (n=5) | 0.2 |
| fltrust | 0.038 ± 0.007 (n=5) | 0.057 ± 0.022 (n=10) | 0.174 ± 0.119 (n=5) | 0.239 ± 0.128 (n=10) | 0.322 ± 0.111 (n=5) | 0.463 ± 0.190 (n=5) | 0.683 ± 0.133 (n=10) | 0.4 |

Clean accuracy:

| defense | f=0.05 (Kf=0.5) | f=0.1 (Kf=1) | f=0.15 (Kf=1.5) | f=0.2 (Kf=2) | f=0.25 (Kf=2.5) | f=0.3 (Kf=3) | f=0.4 (Kf=4) |
|---|---|---|---|---|---|---|---|
| undefended | 0.711 ± 0.003 (n=5) | 0.708 ± 0.015 (n=10) | 0.699 ± 0.019 (n=5) | 0.712 ± 0.012 (n=10) | 0.704 ± 0.024 (n=5) | 0.713 ± 0.007 (n=5) | 0.716 ± 0.009 (n=5) |
| ScoreWeight | 0.739 ± 0.005 (n=5) | 0.734 ± 0.005 (n=10) | 0.730 ± 0.007 (n=5) | 0.727 ± 0.005 (n=10) | 0.726 ± 0.005 (n=5) | 0.721 ± 0.006 (n=5) | 0.714 ± 0.008 (n=5) |
| capped-wmed | 0.720 ± 0.005 (n=5) | 0.714 ± 0.003 (n=10) | 0.708 ± 0.006 (n=5) | 0.702 ± 0.005 (n=10) | 0.696 ± 0.008 (n=5) | 0.684 ± 0.007 (n=10) | 0.673 ± 0.008 (n=5) |
| median | 0.720 ± 0.004 (n=5) | 0.712 ± 0.006 (n=10) | 0.704 ± 0.004 (n=5) | 0.693 ± 0.010 (n=10) | 0.685 ± 0.006 (n=5) | 0.675 ± 0.007 (n=5) | 0.666 ± 0.011 (n=5) |
| fltrust | 0.698 ± 0.005 (n=5) | 0.697 ± 0.005 (n=10) | 0.691 ± 0.004 (n=5) | 0.688 ± 0.007 (n=10) | 0.689 ± 0.009 (n=5) | 0.678 ± 0.010 (n=5) | 0.668 ± 0.014 (n=10) |

## CIFAR-10 adaptive backdoor (budget 0.6)

† marks a rescanned cell where another budget beats 0.6 on seed 0 (the number after † is that budget); the n=5 value understates the attacker there.

ASR:

| defense | f=0.05 (Kf=0.5) | f=0.1 (Kf=1) | f=0.15 (Kf=1.5) | f=0.2 (Kf=2) | f=0.25 (Kf=2.5) | f=0.3 (Kf=3) | f=0.4 (Kf=4) | breaking f |
|---|---|---|---|---|---|---|---|---|
| undefended | 0.093 ± 0.035 (n=5) | 0.340 ± 0.097 (n=10) | 0.504 ± 0.060 (n=5) | 0.714 ± 0.055 (n=10) | 0.761 ± 0.040 (n=5) | 0.842 ± 0.027 (n=5) | 0.904 ± 0.020 (n=5) | 0.15 |
| ScoreWeight | 0.172 ± 0.066 (n=5) | 0.514 ± 0.043 (n=10) | 0.681 ± 0.066 (n=5) | 0.831 ± 0.027 (n=10) | 0.867 ± 0.015 (n=5) | 0.915 ± 0.011 (n=5) | 0.952 ± 0.009 (n=5) | 0.1 |
| capped-wmed | 0.049 ± 0.011 (n=5) | 0.222 ± 0.034 (n=10) | 0.396 ± 0.039 (n=5) | 0.641 ± 0.046 (n=10) | 0.733 ± 0.061 (n=5) | 0.855 ± 0.017 (n=5) | 0.953 ± 0.011 (n=5) | 0.2 |
| median | 0.042 ± 0.005 (n=5) | 0.150 ± 0.019 (n=10) | 0.315 ± 0.046 (n=5) | 0.528 ± 0.064 (n=10) | 0.660 ± 0.034 (n=5) | 0.790 ± 0.061 (n=5) | 0.934 ± 0.012 (n=5) | 0.2 |
| fltrust | 0.038 ± 0.008 (n=5) | 0.059 ± 0.024 (n=10) | 0.170 ± 0.114 (n=5) | 0.247 ± 0.124 (n=10) | 0.342 ± 0.135 (n=5) | 0.473 ± 0.168 (n=5) | 0.676 ± 0.122 (n=10) | 0.4 |

Clean accuracy:

| defense | f=0.05 (Kf=0.5) | f=0.1 (Kf=1) | f=0.15 (Kf=1.5) | f=0.2 (Kf=2) | f=0.25 (Kf=2.5) | f=0.3 (Kf=3) | f=0.4 (Kf=4) |
|---|---|---|---|---|---|---|---|
| undefended | 0.747 ± 0.003 (n=5) | 0.743 ± 0.005 (n=10) | 0.739 ± 0.004 (n=5) | 0.737 ± 0.007 (n=10) | 0.735 ± 0.007 (n=5) | 0.731 ± 0.004 (n=5) | 0.725 ± 0.008 (n=5) |
| ScoreWeight | 0.740 ± 0.005 (n=5) | 0.736 ± 0.004 (n=10) | 0.732 ± 0.007 (n=5) | 0.728 ± 0.006 (n=10) | 0.727 ± 0.006 (n=5) | 0.718 ± 0.006 (n=5) | 0.712 ± 0.006 (n=5) |
| capped-wmed | 0.718 ± 0.006 (n=5) | 0.712 ± 0.003 (n=10) | 0.705 ± 0.006 (n=5) | 0.693 ± 0.006 (n=10) | 0.691 ± 0.005 (n=5) | 0.677 ± 0.005 (n=5) | 0.666 ± 0.003 (n=5) |
| median | 0.723 ± 0.005 (n=5) | 0.720 ± 0.005 (n=10) | 0.713 ± 0.003 (n=5) | 0.706 ± 0.008 (n=10) | 0.698 ± 0.006 (n=5) | 0.686 ± 0.005 (n=5) | 0.675 ± 0.008 (n=5) |
| fltrust | 0.698 ± 0.004 (n=5) | 0.696 ± 0.006 (n=10) | 0.692 ± 0.005 (n=5) | 0.689 ± 0.007 (n=10) | 0.689 ± 0.009 (n=5) | 0.679 ± 0.009 (n=5) | 0.668 ± 0.013 (n=10) |

## Adaptive budget rescan (max over budgets)

Seed 0 only. The attacker picks the budget that maximises defended ASR; ties go to the main-grid budget.

| f | defense | ASR seed 0 @ budget 0.6 | ASR seed 0 @ budget 1 | ASR seed 0 @ budget 1.5 | max-ASR budget | max over budgets | main grid ASR @ 0.6 (n=5) | main grid understated |
|---|---|---|---|---|---|---|---|---|
| 0.3 | capped-wmed | 0.834 | 0.523 | 0.444 | 0.6 | 0.834 | 0.855 | no |
| 0.3 | ScoreWeight | 0.911 | 0.773 | 0.774 | 0.6 | 0.911 | 0.915 | no |
| 0.4 | capped-wmed | 0.949 | 0.841 | 0.786 | 0.6 | 0.949 | 0.953 | no |
| 0.4 | ScoreWeight | 0.956 | 0.915 | 0.924 | 0.6 | 0.956 | 0.952 | no |

## Detection vs aggregation

Oracle weights (true attackers 0) on the weighted-median base against capped-wmed. aggregation-limited: the base fails even with perfect detection; detection-limited: the oracle holds and capped-wmed fails.

| attack | f | Kf | oracle-wmed ASR | capped-wmed ASR | oracle-wmed acc | capped-wmed acc | verdict |
|---|---|---|---|---|---|---|---|
| vanilla backdoor | 0.05 | 0.5 | 0.024 ± 0.004 (n=5) | 0.024 ± 0.004 (n=5) | 0.692 ± 0.006 (n=5) | 0.720 ± 0.005 (n=5) | holds |
| vanilla backdoor | 0.1 | 1 | 0.026 ± 0.003 (n=5) | 0.039 ± 0.005 (n=10) | 0.690 ± 0.002 (n=5) | 0.714 ± 0.003 (n=10) | holds |
| vanilla backdoor | 0.15 | 1.5 | 0.023 ± 0.002 (n=5) | 0.046 ± 0.003 (n=5) | 0.684 ± 0.003 (n=5) | 0.708 ± 0.006 (n=5) | holds |
| vanilla backdoor | 0.2 | 2 | 0.024 ± 0.002 (n=5) | 0.151 ± 0.046 (n=10) | 0.682 ± 0.003 (n=5) | 0.702 ± 0.005 (n=10) | holds |
| vanilla backdoor | 0.25 | 2.5 | 0.026 ± 0.004 (n=5) | 0.354 ± 0.085 (n=5) | 0.676 ± 0.008 (n=5) | 0.696 ± 0.008 (n=5) | holds |
| vanilla backdoor | 0.3 | 3 | 0.027 ± 0.006 (n=5) | 0.612 ± 0.119 (n=10) | 0.662 ± 0.013 (n=5) | 0.684 ± 0.007 (n=10) | detection-limited |
| vanilla backdoor | 0.4 | 4 | 0.028 ± 0.004 (n=5) | 0.886 ± 0.031 (n=5) | 0.646 ± 0.006 (n=5) | 0.673 ± 0.008 (n=5) | detection-limited |
| adaptive backdoor | 0.05 | 0.5 | 0.024 ± 0.004 (n=5) | 0.049 ± 0.011 (n=5) | 0.692 ± 0.006 (n=5) | 0.718 ± 0.006 (n=5) | holds |
| adaptive backdoor | 0.1 | 1 | 0.026 ± 0.003 (n=5) | 0.222 ± 0.034 (n=10) | 0.690 ± 0.002 (n=5) | 0.712 ± 0.003 (n=10) | holds |
| adaptive backdoor | 0.15 | 1.5 | 0.023 ± 0.002 (n=5) | 0.396 ± 0.039 (n=5) | 0.684 ± 0.003 (n=5) | 0.705 ± 0.006 (n=5) | holds |
| adaptive backdoor | 0.2 | 2 | 0.024 ± 0.002 (n=5) | 0.641 ± 0.046 (n=10) | 0.682 ± 0.003 (n=5) | 0.693 ± 0.006 (n=10) | detection-limited |
| adaptive backdoor | 0.25 | 2.5 | 0.026 ± 0.004 (n=5) | 0.733 ± 0.061 (n=5) | 0.676 ± 0.008 (n=5) | 0.691 ± 0.005 (n=5) | detection-limited |
| adaptive backdoor | 0.3 | 3 | 0.027 ± 0.006 (n=5) | 0.855 ± 0.017 (n=5) | 0.662 ± 0.013 (n=5) | 0.677 ± 0.005 (n=5) | detection-limited |
| adaptive backdoor | 0.4 | 4 | 0.028 ± 0.004 (n=5) | 0.953 ± 0.011 (n=5) | 0.646 ± 0.006 (n=5) | 0.666 ± 0.003 (n=5) | detection-limited |

## Suppression vs co-selected attackers

Malicious weight share divided by the equal share m/K, by the number of attackers m in the round, pooled over ledger rounds (at least 5 per entry). 0 = attackers fully down-weighted, 1 = neutral, above 1 = honest clients down-weighted instead.

| dataset | attack | defense | f | m=1 | m=2 | m=3 | m=4 | m=5 | m=6 | m=7 | m=8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cifar10 | vanilla backdoor | capped-wmed | 0.05 | 0.00 | 0.28 | 0.28 | – | – | – | – | – |
| cifar10 | vanilla backdoor | capped-wmed | 0.1 | 0.00 | 0.25 | 0.38 | 0.44 | – | – | – | – |
| cifar10 | vanilla backdoor | capped-wmed | 0.15 | 0.00 | 0.20 | 0.34 | 0.46 | 0.72 | – | – | – |
| cifar10 | vanilla backdoor | capped-wmed | 0.2 | 0.00 | 0.23 | 0.36 | 0.54 | 0.78 | 0.94 | – | – |
| cifar10 | vanilla backdoor | capped-wmed | 0.25 | 0.00 | 0.25 | 0.40 | 0.58 | 0.82 | 1.05 | – | – |
| cifar10 | vanilla backdoor | capped-wmed | 0.3 | 0.00 | 0.26 | 0.46 | 0.69 | 0.94 | 1.08 | 1.14 | – |
| cifar10 | vanilla backdoor | capped-wmed | 0.4 | 0.00 | 0.29 | 0.52 | 0.79 | 1.01 | 1.12 | 1.16 | 1.12 |
| cifar10 | vanilla backdoor | wmed | 0.1 | 0.00 | 0.21 | 0.32 | 0.34 | – | – | – | – |
| cifar10 | vanilla backdoor | wmed | 0.2 | 0.00 | 0.18 | 0.28 | 0.40 | 0.61 | 0.82 | – | – |
| cifar10 | adaptive backdoor | capped-wmed | 0.05 | 1.07 | 1.11 | 0.96 | – | – | – | – | – |
| cifar10 | adaptive backdoor | capped-wmed | 0.1 | 1.08 | 1.10 | 1.14 | 1.19 | – | – | – | – |
| cifar10 | adaptive backdoor | capped-wmed | 0.15 | 1.05 | 1.08 | 1.10 | 1.16 | 1.17 | – | – | – |
| cifar10 | adaptive backdoor | capped-wmed | 0.2 | 1.03 | 1.06 | 1.08 | 1.12 | 1.15 | 1.16 | – | – |
| cifar10 | adaptive backdoor | capped-wmed | 0.25 | 1.00 | 1.03 | 1.06 | 1.08 | 1.14 | 1.19 | – | – |
| cifar10 | adaptive backdoor | capped-wmed | 0.3 | 1.02 | 1.02 | 1.07 | 1.08 | 1.12 | 1.12 | 1.10 | – |
| cifar10 | adaptive backdoor | capped-wmed | 0.4 | 0.98 | 1.03 | 1.03 | 1.04 | 1.07 | 1.10 | 1.10 | 1.10 |
| cifar10 | adaptive backdoor | wmed | 0.1 | 1.25 | 1.26 | 1.27 | 1.31 | – | – | – | – |
| cifar10 | adaptive backdoor | wmed | 0.2 | 1.12 | 1.16 | 1.16 | 1.19 | 1.27 | 1.23 | – | – |
| femnist | vanilla backdoor | capped-wmed | 0.05 | 0.02 | 0.39 | 0.66 | – | – | – | – | – |
| femnist | vanilla backdoor | capped-wmed | 0.1 | 0.02 | 0.38 | 0.57 | 0.68 | – | – | – | – |
| femnist | vanilla backdoor | capped-wmed | 0.15 | 0.02 | 0.35 | 0.58 | 0.78 | 0.93 | – | – | – |
| femnist | vanilla backdoor | capped-wmed | 0.2 | 0.04 | 0.43 | 0.73 | 0.90 | 0.97 | 1.01 | – | – |
| femnist | vanilla backdoor | wmed | 0.05 | 0.02 | 0.34 | 0.58 | – | – | – | – | – |
| femnist | vanilla backdoor | wmed | 0.1 | 0.02 | 0.33 | 0.47 | 0.56 | – | – | – | – |
| femnist | vanilla backdoor | wmed | 0.15 | 0.02 | 0.30 | 0.48 | 0.69 | 0.85 | – | – | – |
| femnist | vanilla backdoor | wmed | 0.2 | 0.03 | 0.36 | 0.63 | 0.82 | 0.93 | 0.98 | – | – |

## Per-metric AUC vs f

Malicious-vs-benign AUC (P(malicious > benign), 0.5 = no separation, toward 0 = attackers lower), mean over seeds; cells read early (rounds 1–25) / late (rounds 51–200).

### cifar10 vanilla backdoor

| defense | metric | f=0.05 (Kf=0.5) | f=0.1 (Kf=1) | f=0.15 (Kf=1.5) | f=0.2 (Kf=2) | f=0.25 (Kf=2.5) | f=0.3 (Kf=3) | f=0.4 (Kf=4) |
|---|---|---|---|---|---|---|---|---|
| undefended | M1 | 0.771 / 0.902 | 0.704 / 0.780 | 0.647 / 0.705 | 0.608 / 0.555 | 0.523 / 0.493 | 0.534 / 0.389 | 0.476 / 0.225 |
| undefended | M2 | 0.914 / 0.899 | 0.840 / 0.670 | 0.781 / 0.565 | 0.630 / 0.331 | 0.466 / 0.275 | 0.393 / 0.167 | 0.182 / 0.087 |
| undefended | M3 | 0.578 / 0.636 | 0.552 / 0.667 | 0.565 / 0.647 | 0.581 / 0.665 | 0.577 / 0.653 | 0.591 / 0.661 | 0.601 / 0.651 |
| undefended | M4 | 0.991 / 0.999 | 0.977 / 0.983 | 0.981 / 0.960 | 0.975 / 0.908 | 0.961 / 0.857 | 0.891 / 0.780 | 0.901 / 0.642 |
| undefended | M5 | 0.885 / 0.910 | 0.792 / 0.781 | 0.739 / 0.706 | 0.621 / 0.512 | 0.475 / 0.451 | 0.374 / 0.329 | 0.192 / 0.182 |
| capped-wmed | M1 | 0.975 / 0.996 | 0.946 / 0.993 | 0.961 / 0.991 | 0.906 / 0.965 | 0.901 / 0.913 | 0.811 / 0.678 | 0.644 / 0.352 |
| capped-wmed | M2 | 0.977 / 0.974 | 0.964 / 0.907 | 0.931 / 0.867 | 0.783 / 0.698 | 0.708 / 0.572 | 0.586 / 0.316 | 0.281 / 0.113 |
| capped-wmed | M3 | 0.588 / 0.608 | 0.583 / 0.656 | 0.593 / 0.647 | 0.617 / 0.671 | 0.599 / 0.658 | 0.617 / 0.679 | 0.598 / 0.658 |
| capped-wmed | M4 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 0.999 | 0.994 / 0.984 | 0.975 / 0.916 | 0.919 / 0.738 |
| capped-wmed | M5 | 0.970 / 0.975 | 0.908 / 0.919 | 0.871 / 0.879 | 0.726 / 0.735 | 0.618 / 0.643 | 0.503 / 0.419 | 0.263 / 0.197 |

### cifar10 adaptive backdoor

| defense | metric | f=0.05 (Kf=0.5) | f=0.1 (Kf=1) | f=0.15 (Kf=1.5) | f=0.2 (Kf=2) | f=0.25 (Kf=2.5) | f=0.3 (Kf=3) | f=0.4 (Kf=4) |
|---|---|---|---|---|---|---|---|---|
| undefended | M1 | 0.064 / 0.026 | 0.127 / 0.046 | 0.176 / 0.057 | 0.213 / 0.070 | 0.252 / 0.075 | 0.315 / 0.081 | 0.362 / 0.099 |
| undefended | M2 | 0.012 / 0.001 | 0.014 / 0.002 | 0.024 / 0.003 | 0.023 / 0.009 | 0.040 / 0.022 | 0.066 / 0.025 | 0.069 / 0.051 |
| undefended | M3 | 0.369 / 0.279 | 0.345 / 0.276 | 0.361 / 0.273 | 0.360 / 0.277 | 0.368 / 0.270 | 0.373 / 0.274 | 0.376 / 0.271 |
| undefended | M4 | 0.840 / 0.145 | 0.817 / 0.245 | 0.847 / 0.330 | 0.848 / 0.347 | 0.871 / 0.329 | 0.856 / 0.291 | 0.895 / 0.356 |
| undefended | M5 | 0.179 / 0.113 | 0.145 / 0.066 | 0.132 / 0.052 | 0.107 / 0.051 | 0.087 / 0.050 | 0.074 / 0.046 | 0.067 / 0.049 |
| capped-wmed | M1 | 0.102 / 0.041 | 0.151 / 0.060 | 0.189 / 0.068 | 0.226 / 0.083 | 0.261 / 0.096 | 0.313 / 0.099 | 0.349 / 0.127 |
| capped-wmed | M2 | 0.021 / 0.003 | 0.018 / 0.004 | 0.032 / 0.006 | 0.029 / 0.013 | 0.048 / 0.026 | 0.068 / 0.028 | 0.075 / 0.053 |
| capped-wmed | M3 | 0.384 / 0.276 | 0.339 / 0.278 | 0.361 / 0.273 | 0.359 / 0.279 | 0.367 / 0.271 | 0.373 / 0.278 | 0.377 / 0.273 |
| capped-wmed | M4 | 0.863 / 0.242 | 0.876 / 0.314 | 0.879 / 0.408 | 0.891 / 0.495 | 0.868 / 0.529 | 0.869 / 0.530 | 0.870 / 0.613 |
| capped-wmed | M5 | 0.199 / 0.172 | 0.174 / 0.106 | 0.147 / 0.091 | 0.128 / 0.086 | 0.082 / 0.092 | 0.089 / 0.079 | 0.076 / 0.063 |

### femnist vanilla backdoor

| defense | metric | f=0.05 (Kf=0.5) | f=0.1 (Kf=1) | f=0.15 (Kf=1.5) | f=0.2 (Kf=2) |
|---|---|---|---|---|---|
| undefended | M1 | 0.864 / 0.775 | 0.791 / 0.731 | 0.707 / 0.708 | 0.647 / 0.605 |
| undefended | M2 | 0.940 / 0.844 | 0.855 / 0.749 | 0.775 / 0.714 | 0.670 / 0.582 |
| undefended | M3 | 0.615 / 0.626 | 0.552 / 0.639 | 0.575 / 0.621 | 0.604 / 0.629 |
| undefended | M4 | 0.991 / 0.955 | 0.963 / 0.894 | 0.946 / 0.861 | 0.899 / 0.741 |
| undefended | M5 | 0.921 / 0.832 | 0.861 / 0.765 | 0.819 / 0.734 | 0.728 / 0.633 |
| capped-wmed | M1 | 0.983 / 1.000 | 0.978 / 0.999 | 0.982 / 0.987 | 0.959 / 0.715 |
| capped-wmed | M2 | 0.976 / 0.999 | 0.975 / 0.992 | 0.957 / 0.946 | 0.900 / 0.689 |
| capped-wmed | M3 | 0.596 / 0.561 | 0.574 / 0.629 | 0.603 / 0.623 | 0.618 / 0.642 |
| capped-wmed | M4 | 0.983 / 1.000 | 0.976 / 1.000 | 0.970 / 0.987 | 0.968 / 0.866 |
| capped-wmed | M5 | 0.971 / 0.970 | 0.967 / 0.940 | 0.944 / 0.927 | 0.894 / 0.737 |

## FEMNIST cliff

'implant round' is the first round from which ASR stays >= 0.5 for 10 rounds; 'first share>=0.5' the first round (from round 5) in which attackers held half the aggregation weight.

| defense | f | n | ASR final | clean acc | implant round (per seed) | first share>=0.5 (per seed) | seeds implanted | P(any m>=5 in 200) |
|---|---|---|---|---|---|---|---|---|
| undefended | 0.05 | 5 | 0.989 ± 0.012 | 0.806 ± 0.003 | 10,9,5,4,20 | never,never,never,never,never | 5/5 | 0.00 |
| undefended | 0.1 | 10 | 0.998 ± 0.002 | 0.804 ± 0.004 | 5,5,7,11,5,4,2,10,6,8 | never,never,never,never,never,86,never,never,never,never | 10/10 | 0.13 |
| undefended | 0.15 | 5 | 0.999 ± 0.001 | 0.802 ± 0.005 | 7,4,5,3,8 | 156,145,38,159,124 | 5/5 | 0.72 |
| undefended | 0.2 | 10 | 0.999 ± 0.001 | 0.802 ± 0.005 | 5,5,7,6,7,4,7,4,5,7 | 9,32,7,22,37,15,83,24,95,32 | 10/10 | 0.99 |
| wmed | 0.05 | 5 | 0.005 ± 0.001 | 0.780 ± 0.003 | never,never,never,never,never | never,never,never,never,never | 0/5 | 0.00 |
| wmed | 0.1 | 10 | 0.013 ± 0.007 | 0.784 ± 0.003 | never,never,never,never,never,never,never,never,never,never | never,never,never,never,never,never,never,never,never,never | 0/10 | 0.13 |
| wmed | 0.15 | 5 | 0.307 ± 0.167 | 0.785 ± 0.005 | 191,177,never,never,never | never,never,never,never,never | 2/5 | 0.72 |
| wmed | 0.2 | 10 | 0.983 ± 0.014 | 0.783 ± 0.005 | 9,32,93,81,37,86,83,46,95,34 | 9,32,never,never,37,200,83,34,95,34 | 10/10 | 0.99 |
| capped-wmed | 0.05 | 5 | 0.005 ± 0.001 | 0.783 ± 0.004 | never,never,never,never,never | never,never,never,never,never | 0/5 | 0.00 |
| capped-wmed | 0.1 | 10 | 0.018 ± 0.010 | 0.786 ± 0.003 | never,never,never,never,never,never,never,never,never,never | never,never,never,never,never,never,never,never,never,never | 0/10 | 0.13 |
| capped-wmed | 0.15 | 5 | 0.621 ± 0.259 | 0.786 ± 0.005 | 148,145,never,never,124 | never,never,never,never,124 | 3/5 | 0.72 |
| capped-wmed | 0.2 | 10 | 0.990 ± 0.006 | 0.784 ± 0.004 | 9,32,60,52,37,28,82,34,95,34 | 9,32,60,71,37,28,83,34,95,34 | 10/10 | 0.99 |
| median | 0.05 | 5 | 0.136 ± 0.111 | 0.790 ± 0.004 | never,never,never,never,never | never,never,never,never,never | 0/5 | 0.00 |
| median | 0.1 | 10 | 0.958 ± 0.042 | 0.788 ± 0.003 | 89,54,84,68,106,72,100,114,76,87 | never,never,never,never,never,86,never,never,never,never | 10/10 | 0.13 |
| median | 0.15 | 5 | 0.994 ± 0.002 | 0.789 ± 0.005 | 55,43,38,40,60 | 156,145,38,159,124 | 5/5 | 0.72 |
| median | 0.2 | 10 | 0.997 ± 0.001 | 0.787 ± 0.004 | 9,23,7,15,19,15,32,24,38,21 | 9,32,7,22,37,15,83,24,95,32 | 10/10 | 0.99 |
| ScoreWeight | 0.05 | 5 | 0.814 ± 0.118 | 0.809 ± 0.002 | 58,85,130,120,147 | never,never,never,never,never | 5/5 | 0.00 |
| ScoreWeight | 0.1 | 10 | 0.987 ± 0.009 | 0.809 ± 0.003 | 24,35,38,46,49,42,18,40,24,44 | never,never,never,never,never,never,never,never,never,never | 10/10 | 0.13 |
| ScoreWeight | 0.15 | 5 | 0.995 ± 0.001 | 0.809 ± 0.002 | 30,12,18,21,19 | never,never,never,never,never | 5/5 | 0.72 |
| ScoreWeight | 0.2 | 10 | 0.997 ± 0.001 | 0.807 ± 0.003 | 6,12,18,11,12,14,15,10,7,13 | 9,32,60,never,37,28,37,34,95,32 | 10/10 | 0.99 |

## Untargeted (gradient scaling)

Clean accuracy, scaling factor 5:

| defense | f=0.05 (Kf=0.5) | f=0.1 (Kf=1) | f=0.15 (Kf=1.5) | f=0.2 (Kf=2) | f=0.25 (Kf=2.5) | f=0.3 (Kf=3) | f=0.4 (Kf=4) |
|---|---|---|---|---|---|---|---|
| undefended | – | 0.678 ± 0.033 (n=10) | – | 0.404 ± 0.149 (n=10) | – | – | – |
| capped-wmed | – | 0.719 ± 0.005 (n=10) | – | 0.715 ± 0.004 (n=10) | – | – | – |
