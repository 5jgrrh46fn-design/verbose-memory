### B1. Undefended, norm-bounded attacker, by budget

| fraction | budget | tag | ASR % | clean acc % | seeds | min / max ASR % |
|---|---|---|---|---|---|---|
| 10 % | 1.0 | `c100_adaptive_backdoor_cifar100_f10_g1.0` | 23.2 ± 4.4 (10) | 37.6 ± 0.6 (10) | 0,1,2,3,4,5,6,7,8,9 | 17.7 / 32.5 |
| 10 % | 1.5 | `c100_adaptive_backdoor_cifar100_f10_g1.5` | 38.5 ± 5.1 (10) | 37.7 ± 0.6 (10) | 0,1,2,3,4,5,6,7,8,9 | 31.8 / 47.5 |
| 10 % | 3.0 | `c100_adaptive_backdoor_cifar100_f10_g3.0` | 55.2 ± 6.1 (10) | 37.1 ± 1.2 (10) | 0,1,2,3,4,5,6,7,8,9 | 44.8 / 64.6 |
| 20 % | 1.0 | `c100_adaptive_backdoor_cifar100_f20_g1.0` | 55.9 ± 5.0 (10) | 37.2 ± 0.4 (10) | 0,1,2,3,4,5,6,7,8,9 | 47.3 / 62.4 |
| 20 % | 1.5 | `c100_adaptive_backdoor_cifar100_f20_g1.5` | 70.8 ± 7.0 (10) | 37.5 ± 0.4 (10) | 0,1,2,3,4,5,6,7,8,9 | 58.1 / 80.9 |
| 20 % | 3.0 | `c100_adaptive_backdoor_cifar100_f20_g3.0` | 76.8 ± 5.3 (10) | 30.8 ± 4.1 (10) | 0,1,2,3,4,5,6,7,8,9 | 68.7 / 87.4 |

### B2. FLTrust's weights and Krum's selections

FLTrust (rounds = rounds with at least one sampled attacker, mean per seed):

| cell | seeds | ASR % | undefended ASR % | rounds | attacker strictly top % | attacker weight share % | head-count share % | weight / head-count | attacker update norm | honest update norm |
|---|---|---|---|---|---|---|---|---|---|---|
| `adaptive_backdoor_cifar100_f10_g1.0` | 10 | 59.9 ± 5.1 (10) | 23.2 ± 4.4 (10) | 132 of 200 | 47.9 ± 7.1 (10) | 21.5 ± 1.6 (10) | 15.3 ± 0.6 (10) | 1.41 ± 0.13 (10) | 1.000 ± 0.000 (10) | 3.106 ± 0.072 (10) |
| `adaptive_backdoor_cifar100_f20_g1.0` | 10 | 76.1 ± 5.4 (10) | 55.9 ± 5.0 (10) | 182 of 200 | 48.5 ± 3.9 (10) | 28.6 ± 1.2 (10) | 22.8 ± 0.8 (10) | 1.26 ± 0.08 (10) | 1.000 ± 0.000 (10) | 3.102 ± 0.063 (10) |
| `backdoor_cifar100_f10` | 10 | 60.5 ± 5.0 (10) | 62.8 ± 8.6 (10) | 132 of 200 | 48.6 ± 7.0 (10) | 21.8 ± 1.9 (10) | 15.3 ± 0.6 (10) | 1.43 ± 0.14 (10) | 3.641 ± 0.186 (10) | 3.106 ± 0.072 (10) |
| `backdoor_cifar100_f20` | 10 | 76.7 ± 4.1 (10) | 84.4 ± 3.4 (10) | 182 of 200 | 49.0 ± 3.8 (10) | 28.7 ± 1.2 (10) | 22.8 ± 0.8 (10) | 1.26 ± 0.08 (10) | 2.372 ± 0.089 (10) | 3.103 ± 0.069 (10) |

Krum:

| cell | seeds | ASR % | undefended ASR % | rounds | Krum selects an attacker % | head-count share % | attacker update norm | honest update norm |
|---|---|---|---|---|---|---|---|---|
| `adaptive_backdoor_cifar100_f20_g1.0` | 10 | 70.6 ± 15.3 (10) | 55.9 ± 5.0 (10) | 182 of 200 | 39.2 ± 19.4 (10) | 22.8 ± 0.8 (10) | 1.000 ± 0.000 (10) | 1.291 ± 0.397 (10) |
| `backdoor_cifar100_f20` | 10 | 36.2 ± 26.6 (10) | 84.4 ± 3.4 (10) | 182 of 200 | 15.7 ± 2.6 (10) | 22.8 ± 0.8 (10) | 2.789 ± 0.747 (10) | 1.954 ± 0.696 (10) |

Diagnostic run against the reported run (`c100_<cell>__<rule>`), every `metrics.json` column except the wall-clock ones:

| diagnostic cell | seeds | identical to reported | differing seeds |
|---|---|---|---|
| `diag_adaptive_backdoor_cifar100_f10_g1.0__fltrust` | 10 | 10 | – |
| `diag_adaptive_backdoor_cifar100_f20_g1.0__fltrust` | 10 | 10 | – |
| `diag_backdoor_cifar100_f10__fltrust` | 10 | 10 | – |
| `diag_backdoor_cifar100_f20__fltrust` | 10 | 10 | – |
| `diag_adaptive_backdoor_cifar100_f20_g1.0__krum` | 10 | 10 | – |
| `diag_backdoor_cifar100_f20__krum` | 10 | 10 | – |

