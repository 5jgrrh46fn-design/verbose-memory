### A1. Undefended under the composed attacker (0.6, 0.9)

| fraction | ASR mean | ASR sd | acc mean | acc sd | seeds |
|---|---|---|---|---|---|
| f10 | 7.6 | 2.3 | 74.6 | 0.5 | 10 (0,1,2,3,4,5,6,7,8,9) |
| f20 | 33.4 | 8.3 | 74.2 | 0.5 | 10 (0,1,2,3,4,5,6,7,8,9) |

| fraction | capped wmed ASR | undefended ASR | Δ (pp) | capped above on | Wilcoxon W | p (two-sided) | rank-biserial |
|---|---|---|---|---|---|---|---|
| f10 | 11.6 | 7.6 | +4.1 | 8/10 | 3 | 0.0098 | +0.89 |
| f20 | 48.1 | 33.4 | +14.6 | 10/10 | 0 | 0.0020 | +1.00 |

### A2. Undefended under the norm-bounded attacker

| fraction | budget | ASR mean | ASR sd | acc mean | acc sd | seeds | labels |
|---|---|---|---|---|---|---|---|
| f10 | 0.6 | 34.0 | 9.7 | 74.3 | 0.5 | 10 (0,1,2,3,4,5,6,7,8,9) | adaptive_backdoor_cifar10_f10/d7655368 |
| f10 | 0.75 | 44.6 | 10.0 | 74.3 | 0.5 | 10 (0,1,2,3,4,5,6,7,8,9) | sp_adaptive_backdoor_cifar10_f10_g0.75 |
| f10 | 1.0 | 56.3 | 9.6 | 74.2 | 0.6 | 10 (0,1,2,3,4,5,6,7,8,9) | e4_adaptive_backdoor_cifar10_f10_g1.0, sp_adaptive_backdoor_cifar10_f10_g1.0 |
| f10 | 1.5 | 67.6 | 5.5 | 73.6 | 0.9 | 10 (0,1,2,3,4,5,6,7,8,9) | sp_adaptive_backdoor_cifar10_f10_g1.5 |
| f10 | 2.0 | 74.0 | 2.9 | 72.6 | 1.4 | 10 (0,1,2,3,4,5,6,7,8,9) | sp_adaptive_backdoor_cifar10_f10_g2.0 |
| f20 | 0.6 | 71.4 | 5.5 | 73.7 | 0.7 | 10 (0,1,2,3,4,5,6,7,8,9) | adaptive_backdoor_cifar10_f20/0e7cb95c |
| f20 | 0.75 | 78.9 | 4.1 | 73.6 | 0.7 | 10 (0,1,2,3,4,5,6,7,8,9) | sp_adaptive_backdoor_cifar10_f20_g0.75 |
| f20 | 1.0 | 83.3 | 3.2 | 73.2 | 0.8 | 10 (0,1,2,3,4,5,6,7,8,9) | e4_adaptive_backdoor_cifar10_f20_g1.0, sp_adaptive_backdoor_cifar10_f20_g1.0 |
| f20 | 1.5 | 84.6 | 1.9 | 71.1 | 1.2 | 10 (0,1,2,3,4,5,6,7,8,9) | sp_adaptive_backdoor_cifar10_f20_g1.5 |
| f20 | 2.0 | 86.2 | 1.3 | 67.5 | 2.4 | 10 (0,1,2,3,4,5,6,7,8,9) | sp_adaptive_backdoor_cifar10_f20_g2.0 |

### A3. Krum's selections

| cell | seeds | rounds with a sampled attacker | attacker selected (share of those rounds) | attackers' head-count share in those rounds | selected / head-count | attacker selected, per-seed min-max |
|---|---|---|---|---|---|---|
| diag_adaptive_backdoor_cifar10_f10_g0.6__krum | 10 | 1319 of 2000 | 66.6% | 15.3% | 4.36 | 55.7-76.6% |
| diag_adaptive_backdoor_cifar10_f20_g0.6__krum | 10 | 1815 of 2000 | 82.3% | 22.8% | 3.62 | 76.1-89.9% |
| diag_backdoor_cifar10_f10__krum | 10 | 1319 of 2000 | 1.4% | 15.3% | 0.09 | 0.7-3.2% |
| diag_backdoor_cifar10_f20__krum | 10 | 1815 of 2000 | 18.2% | 22.8% | 0.80 | 12.5-22.3% |

### A3. FLTrust's weights

| cell | seeds | rounds with a sampled attacker | an attacker's trust strictly above every honest client's | attackers' weight share | head-count share | weight / head-count | rounds with all trust zero | attacker norm (before rescaling) | honest norm |
|---|---|---|---|---|---|---|---|---|---|
| diag_femnist_adaptive_backdoor_f10_g0.2__fltrust | 10 | 1319 of 2000 | 17.9% | 14.7% | 15.3% | 0.96 | 0 | 0.200 | 0.950 |
| diag_femnist_backdoor_f10__fltrust | 10 | 1319 of 2000 | 17.1% | 14.7% | 15.3% | 0.96 | 0 | 2.384 | 0.948 |
| diag_adaptive_backdoor_cifar10_f10_g0.6__fltrust | 10 | 1319 of 2000 | 0.5% | 3.2% | 15.3% | 0.21 | 0 | 0.600 | 1.550 |
| diag_adaptive_backdoor_cifar10_f20_g0.6__fltrust | 10 | 1815 of 2000 | 1.0% | 5.6% | 22.8% | 0.24 | 0 | 0.600 | 1.538 |
