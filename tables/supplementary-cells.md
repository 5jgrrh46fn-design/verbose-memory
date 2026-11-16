## A. Table 1: integration mechanisms under the norm-bounded backdoor (CIFAR-10, budget 0.6, results/)

Cells from the Figure 1 loader (`paper_figures.load`, reference setting, hash with most seeds).

| rule | f=10% ASR | f=10% ACC | f=20% ASR | f=20% ACC |
|---|---|---|---|---|
| Undefended | 34.0 ± 9.7 (10) | 74.3 ± 0.5 (10) | 71.4 ± 5.5 (10) | 73.7 ± 0.7 (10) |
| ScoreWeight | 51.4 ± 4.3 (10) | 73.6 ± 0.4 (10) | 83.1 ± 2.7 (10) | 72.8 ± 0.6 (10) |
| TemporalReputation | 49.6 ± 4.4 (10) | 73.7 ± 0.4 (10) | 82.2 ± 2.8 (10) | 73.0 ± 0.6 (10) |
| HardFilter | 54.4 ± 5.4 (10) | 72.2 ± 0.6 (10) | 83.5 ± 2.7 (10) | 71.0 ± 0.8 (10) |
| Quarantine | 42.5 ± 4.9 (10) | 73.4 ± 0.4 (10) | 78.1 ± 2.9 (10) | 72.4 ± 0.6 (10) |
| AdaptiveParticipation | 56.3 ± 7.3 (10) | 73.7 ± 0.3 (10) | 86.9 ± 1.9 (10) | 72.7 ± 0.4 (10) |
| MW-Peer | 22.2 ± 3.4 (10) | 71.2 ± 0.3 (10) | 64.1 ± 4.6 (10) | 69.3 ± 0.6 (10) |

### Backfire family, Holm over the ten comparisons (paired Wilcoxon, ten seeds)

| comparison (a − b) | pairs | mean Δ (pts) | BCa 95% | p | Holm p | reject | r |
|---|---|---|---|---|---|---|---|
| ScoreWeight − Undefended, f=10% | 10 | +17.4 | [+11.5, +23.5] | 0.0020 | 0.0195 | yes | +1.00 |
| TemporalReputation − Undefended, f=10% | 10 | +15.6 | [+9.9, +21.3] | 0.0020 | 0.0195 | yes | +1.00 |
| HardFilter − Undefended, f=10% | 10 | +20.4 | [+13.3, +26.3] | 0.0020 | 0.0195 | yes | +1.00 |
| Quarantine − Undefended, f=10% | 10 | +8.5 | [+3.0, +14.1] | 0.0371 | 0.0371 | yes | +0.75 |
| AdaptiveParticipation − Undefended, f=10% | 10 | +22.2 | [+14.7, +30.7] | 0.0020 | 0.0195 | yes | +1.00 |
| ScoreWeight − Undefended, f=20% | 10 | +11.7 | [+8.6, +13.5] | 0.0020 | 0.0195 | yes | +1.00 |
| TemporalReputation − Undefended, f=20% | 10 | +10.8 | [+7.9, +12.6] | 0.0020 | 0.0195 | yes | +1.00 |
| HardFilter − Undefended, f=20% | 10 | +12.2 | [+9.2, +14.5] | 0.0020 | 0.0195 | yes | +1.00 |
| Quarantine − Undefended, f=20% | 10 | +6.7 | [+4.1, +8.5] | 0.0039 | 0.0195 | yes | +0.96 |
| AdaptiveParticipation − Undefended, f=20% | 10 | +15.5 | [+11.7, +17.8] | 0.0020 | 0.0195 | yes | +1.00 |

### MW-Peer against undefended (not part of the Table 1 family)

| comparison (a − b) | pairs | mean Δ (pts) | BCa 95% | p | Holm p | reject | r |
|---|---|---|---|---|---|---|---|
| MW-Peer − Undefended, f=10% | 10 | -11.9 | [-16.1, -6.7] | 0.0039 | 0.0039 | yes | -0.96 |
| MW-Peer − Undefended, f=20% | 10 | -7.3 | [-9.4, -5.5] | 0.0020 | 0.0039 | yes | -1.00 |

## B. Cap and median ablations at f=10% under the norm-bounded backdoor (results/, local campaign)

Ten-seed cells of the RTX 5070 Ti box.

| cell | label/hash | ASR | ACC |
|---|---|---|---|
| capped weighted median (MW-Peer) | `adaptive_backdoor_cifar10_f10__CappedScoreWeight_critic_wmedian/33b6313e3610b4e2` | 22.2 ± 3.4 (10) | 71.2 ± 0.3 (10) |
| uncapped trust-weighted median | `adaptive_backdoor_cifar10_f10__ScoreWeight_critic_wmedian/405e00ea59b10669` | 32.4 ± 3.9 (10) | 70.8 ± 0.4 (10) |
| cap on a mean (CappedScoreWeight) | `adaptive_backdoor_cifar10_f10__CappedScoreWeight_critic/7cd46974c7a981a9` | 43.6 ± 4.2 (10) | 73.8 ± 0.4 (10) |
| undefended | `adaptive_backdoor_cifar10_f10/d7655368d6eee9dc` | 34.0 ± 9.7 (10) | 74.3 ± 0.5 (10) |
| uncapped weighted median, standard backdoor | `backdoor_cifar10_f10__ScoreWeight_critic_wmedian/b99a97c46eb2691f` | 3.7 ± 0.5 (10) | 71.4 ± 0.4 (10) |
| capped weighted median, standard backdoor | `backdoor_cifar10_f10__CappedScoreWeight_critic_wmedian/42a83f9f8ed5f0b3` | 3.9 ± 0.5 (10) | 71.4 ± 0.3 (10) |

### Ablation family (Holm k=3)

| comparison (a − b) | pairs | mean Δ (pts) | BCa 95% | p | Holm p | reject | r |
|---|---|---|---|---|---|---|---|
| uncapped wmed − capped wmed | 10 | +10.3 | [+9.0, +11.3] | 0.0020 | 0.0059 | yes | +1.00 |
| cap on mean − undefended | 10 | +9.6 | [+4.0, +15.0] | 0.0273 | 0.0547 | no | +0.78 |
| uncapped wmed − undefended | 10 | -1.6 | [-6.5, +3.2] | 0.7695 | 0.7695 | no | -0.13 |

## C. HardFilter ledger: honest client-rounds excluded and attacker share (norm-bounded backdoor, CIFAR-10, results/)

| f | hash | seeds | honest rows excluded | attacker rows excluded | attacker weight share in rounds with an attacker | equal share conditioned on m ≥ 1 |
|---|---|---|---|---|---|---|
| 0.10 | `02f3488e8608e670` | 10 | 34.0% | 11.9% | 21.4% | 15.3% |
| 0.20 | `dd09b9c0e336f5d6` | 10 | 36.6% | 13.6% | 30.3% | 22.8% |

## D. FEMNIST norm-bounded budget probe (f=10%, undefended, seed 0)

| budget | label/hash | last-50 ASR | last-50 ACC | final-round ASR |
|---|---|---|---|---|
| 0.1 | `sp_femnist_adaptive_backdoor_f10_g0.1/b86b703eb4c10953` | 18.6 | 79.8 | 26.6 |
| 0.2 | `sp_femnist_adaptive_backdoor_f10_g0.2/200809b5fc9a1461` | 95.0 | 79.8 | 95.9 |
| 0.3 | `sp_femnist_adaptive_backdoor_f10_g0.3/54afb1a5dcdb31d3` | 98.6 | 79.8 | 98.9 |
| 0.6 | `sp_femnist_adaptive_backdoor_f10/2766a0ebc104922d` | 99.7 | 79.8 | 99.7 |
| 1.0 | `sp_femnist_adaptive_backdoor_f10_g1.0/949371881e40de4e` | 99.9 | 80.0 | 99.8 |
| 1.5 | `sp_femnist_adaptive_backdoor_f10_g1.5/6f04d16c3bae6533` | 99.9 | 79.8 | 99.9 |
| 3.0 | `sp_femnist_adaptive_backdoor_f10_g3.0/0b17303a05d86059` | 100.0 | 79.2 | 100.0 |

## E. MW-Root (fltrust_median) on FEMNIST f=10%, per-seed last-50 ASR

| cell | label/hash | per-seed ASR (seeds 0..9) | mean ± sd | seeds < 20 | seeds > 60 |
|---|---|---|---|---|---|
| standard backdoor, uncapped | `e4_femnist_backdoor_f10__fltrust_median/42f42f15b1e9954a` | 28.3, 58.9, 28.3, 40.6, 5.9, 85.0, 48.2, 6.9, 14.5, 79.0 | 39.6 ± 28.2 (10) | 3 | 2 |
| standard backdoor, capped | `e4_femnist_backdoor_f10__fltrust_median_capped/493b5ff46cf3774c` | 75.5, 95.0, 59.6, 94.8, 29.8, 97.7, 54.6, 62.2, 43.0, 92.5 | 70.5 ± 24.3 (10) | 0 | 6 |
| norm-bounded 0.2, uncapped | `sp_femnist_adaptive_backdoor_f10_g0.2__fltrust_median/a949e2e2b726307a` | 26.3, 58.9, 35.4, 35.9, 5.4, 91.3, 31.4, 8.0, 18.6, 80.5 | 39.2 ± 29.1 (10) | 3 | 2 |
| norm-bounded 0.2, capped | `sp_femnist_adaptive_backdoor_f10_g0.2__fltrust_median_capped/1d23bc53ccc5612c` | 97.7, 93.0, 55.9, 93.0, 24.5, 98.7, 67.8, 66.6, 69.0, 94.4 | 76.1 ± 24.0 (10) | 0 | 8 |

## F. FLAME with all 100 clients per round (CIFAR-10, standard backdoor, f=20%, seed 0)

| cell | label/hash | K | last-50 ASR | last-50 ACC |
|---|---|---|---|---|
| undefended | `sp_flame_full/e098363dda4ca41e` | 100 (f=0.2) | 93.9 | 77.4 |
| FLAME | `sp_flame_full__flame/09fa3ee23327ba78` | 100 (f=0.2) | 88.3 | 64.0 |

## G. BackdoorIndicator probes (seed 0): injection epochs × start round, CIFAR-10 and FEMNIST

Drop shares count ledger rows whose `included`/`admitted` flag is false (weight 0 where neither exists), from the start round on.

| label/hash | experiment | epochs | start | last-10 ACC | last-10 ASR | last-50 ACC | last-50 ASR | honest rows dropped | attacker rows dropped |
|---|---|---|---|---|---|---|---|---|---|
| `probe_indicator_e10_s1__backdoor_indicator/157a83b5d0ba6e19` | benign_cifar10 | 10 | 1 | 70.6 | – | 70.4 | – | 16.4% | – |
| `probe_indicator_e10_s1__backdoor_indicator/ccd863eded69fb05` | backdoor_cifar10_f10 | 10 | 1 | 70.0 | 4.1 | 68.9 | 4.5 | 18.1% | 93.4% |
| `probe_indicator_e10_s50__backdoor_indicator/a290797edf4f81b6` | benign_cifar10 | 10 | 50 | 72.8 | – | 72.2 | – | 13.0% | – |
| `probe_indicator_e10_s50__backdoor_indicator/b2853090405371b0` | backdoor_cifar10_f10 | 10 | 50 | 68.6 | 78.5 | 68.8 | 76.8 | 11.7% | 29.9% |
| `probe_indicator_e200_s1__backdoor_indicator/74e0b340f1335686` | benign_cifar10 | 200 | 1 | 54.9 | – | 53.1 | – | 0.3% | – |
| `probe_indicator_e200_s1__backdoor_indicator/aa2f97c9c892b4b3` | backdoor_cifar10_f10 | 200 | 1 | 47.4 | 6.7 | 44.6 | 7.5 | 2.4% | 87.3% |
| `probe_indicator_e200_s50__backdoor_indicator/6ce95e3f698eec09` | backdoor_cifar10_f10 | 200 | 50 | 70.2 | 68.7 | 69.4 | 65.8 | 5.3% | 47.8% |
| `probe_indicator_e200_s50__backdoor_indicator/c292c371ca8faae8` | benign_cifar10 | 200 | 50 | 73.4 | – | 72.4 | – | 7.9% | – |
| `probe_indicator_femnist_e10_s1__backdoor_indicator/113b598c80827550` | femnist_backdoor_f10 | 10 | 1 | 57.2 | 99.9 | 51.0 | 99.5 | 78.0% | 37.1% |
| `probe_indicator_femnist_e10_s1__backdoor_indicator/de4dce7b37d1978b` | femnist_benign | 10 | 1 | 58.2 | – | 57.6 | – | 90.0% | – |

Undefended seed 0 references: backdoor_cifar10_f10 last-10 ACC 70.3, last-50 ACC 69.9; benign_cifar10 last-10 ACC 75.1, last-50 ACC 74.9; femnist_backdoor_f10 last-10 ACC 80.6, last-50 ACC 80.0; femnist_benign last-10 ACC 80.5, last-50 ACC 80.4; 

## H. Norm-budget scans of the published baselines (CIFAR-10, last-50 ASR, mean (n))

| defense | f | 0.6 | 1.0 | 1.5 | 10 |
|---|---|---|---|---|---|
| flame | 10% | 39.6 (10) | 57.2 (10) | 68.6 (10) | – |
| flame | 20% | 78.2 (10) | 85.0 (10) | 87.2 (10) | – |
| foolsgold | 10% | 7.5 (10) | 24.9 (10) | 51.1 (10) | – |
| foolsgold | 20% | 6.1 (10) | 34.0 (10) | 84.1 (10) | – |
| backdoor_indicator | 10% | 3.1 (10) | 3.1 (10) | 3.2 (10) | – |
| backdoor_indicator | 20% | 3.1 (10) | 3.2 (10) | 17.3 (10) | – |
| trimmed_mean | 10% | 17.5 (10) | 24.6 (10) | 29.9 (10) | 38.5 (5) |
| trimmed_mean | 20% | 57.7 (10) | 68.2 (10) | 73.7 (10) | – |
| clipped_mean | 10% | 40.1 (10) | 64.1 (5) | 75.9 (5) | – |
| clipped_mean | 20% | 76.1 (10) | 87.1 (5) | 88.8 (5) | – |
| krum | 10% | 74.1 (10) | 43.2 (5) | 11.3 (5) | – |
| krum | 20% | 91.8 (10) | 66.8 (5) | 38.9 (5) | – |
| mesas | 10% | 38.3 (10) | 44.7 (5) | 41.5 (10) | – |
| mesas | 20% | 72.4 (10) | 78.3 (5) | 70.7 (5) | – |

## I. Krum clean accuracy against the next-lowest rule, Figure 3 cells (CIFAR-10)

| attack | f | Krum ACC | lowest other rule (ACC) | gap (pts) |
|---|---|---|---|---|
| standard | 5% | 36.7 | MESAS (66.2) | 29.5 |
| standard | 10% | 35.1 | FLAME (65.9) | 30.9 |
| standard | 15% | 29.7 | MESAS (64.2) | 34.5 |
| standard | 20% | 21.8 | FLAME (62.3) | 40.5 |
| standard | 30% | 21.1 | FLAME (61.2) | 40.1 |
| standard | 40% | 22.9 | FLAME (58.7) | 35.8 |
| adaptive | 5% | 32.6 | BackdoorIndicator (68.9) | 36.3 |
| adaptive | 10% | 32.1 | BackdoorIndicator (67.8) | 35.7 |
| adaptive | 15% | 31.6 | BackdoorIndicator (67.1) | 35.6 |
| adaptive | 20% | 30.4 | BackdoorIndicator (66.3) | 35.9 |
| adaptive | 30% | 31.3 | BackdoorIndicator (63.7) | 32.4 |
| adaptive | 40% | 33.3 | BackdoorIndicator (62.6) | 29.2 |

## J. Figure 2 and Figure 3 cells: last-50 ASR / ACC by compromised fraction (CIFAR-10, every rule)


### standard backdoor

| rule | f=5% | f=10% | f=15% | f=20% | f=25% | f=30% | f=40% |
|---|---|---|---|---|---|---|---|
| Undefended | 62.7 ± 9.8 / 71.1 (5) | 81.0 ± 3.3 / 70.8 (10) | 80.8 ± 5.5 / 69.9 (5) | 88.1 ± 4.3 / 71.2 (10) | 86.1 ± 4.1 / 70.4 (5) | 92.4 ± 1.3 / 71.3 (5) | 93.7 ± 1.3 / 71.6 (5) |
| Median | 5.4 ± 0.7 / 72.0 (5) | 19.7 ± 2.8 / 71.2 (10) | 35.2 ± 5.4 / 70.4 (5) | 58.3 ± 5.2 / 69.3 (10) | 71.1 ± 3.7 / 68.5 (5) | 81.9 ± 4.5 / 67.5 (5) | 92.9 ± 1.5 / 66.6 (5) |
| Trimmed mean | 7.6 ± 1.4 / 72.7 (5) | 31.7 ± 3.4 / 72.0 (10) | 49.4 ± 4.5 / 71.2 (5) | 71.5 ± 4.4 / 70.5 (10) | – | 87.2 ± 2.7 / 69.4 (5) | 94.1 ± 0.7 / 68.5 (5) |
| Clipped mean | 49.7 ± 7.2 / 73.9 (5) | 78.9 ± 2.2 / 73.2 (10) | 85.2 ± 2.0 / 72.7 (5) | 91.4 ± 1.3 / 72.2 (10) | – | 94.6 ± 0.8 / 71.2 (5) | 96.4 ± 0.5 / 70.1 (5) |
| Krum | 5.2 ± 1.2 / 36.7 (5) | 7.2 ± 1.8 / 35.1 (10) | 8.0 ± 1.9 / 29.7 (5) | 37.2 ± 19.4 / 21.8 (10) | – | 73.1 ± 11.3 / 21.1 (5) | 87.5 ± 2.2 / 22.9 (5) |
| FoolsGold | 64.2 ± 5.6 / 72.1 (5) | 80.7 ± 4.6 / 71.2 (10) | 85.2 ± 4.2 / 71.4 (5) | 89.1 ± 1.9 / 71.7 (10) | – | 92.1 ± 1.0 / 71.4 (5) | 93.4 ± 0.7 / 71.8 (5) |
| FLTrust | 3.8 ± 0.7 / 69.8 (5) | 5.7 ± 2.2 / 69.7 (10) | 17.4 ± 11.9 / 69.1 (5) | 23.9 ± 12.8 / 68.8 (10) | 32.2 ± 11.1 / 68.9 (5) | 46.3 ± 19.0 / 67.8 (5) | 68.3 ± 13.3 / 66.8 (10) |
| FLAME | 36.7 ± 13.6 / 68.2 (5) | 72.9 ± 4.4 / 65.9 (10) | 80.4 ± 3.3 / 64.5 (5) | 87.0 ± 3.2 / 62.3 (10) | – | 92.8 ± 1.1 / 61.2 (5) | 94.3 ± 1.8 / 58.7 (5) |
| MESAS | 38.9 ± 10.7 / 66.2 (5) | 59.0 ± 11.8 / 66.0 (10) | 63.5 ± 11.7 / 64.2 (5) | 75.9 ± 7.1 / 63.7 (10) | – | 87.7 ± 2.1 / 64.2 (5) | 93.2 ± 3.3 / 59.9 (5) |
| BackdoorIndicator | 3.5 ± 1.4 / 68.7 (5) | 4.5 ± 2.2 / 67.5 (10) | 14.7 ± 17.9 / 67.0 (5) | 21.5 ± 29.9 / 65.9 (10) | – | 53.0 ± 40.6 / 63.3 (5) | 83.2 ± 2.5 / 64.1 (5) |
| MW-Peer | 2.4 ± 0.4 / 72.0 (5) | 3.9 ± 0.5 / 71.4 (10) | 4.6 ± 0.3 / 70.8 (5) | 15.1 ± 4.6 / 70.2 (10) | 35.4 ± 8.5 / 69.6 (5) | 61.2 ± 11.9 / 68.4 (10) | 88.6 ± 3.1 / 67.3 (5) |

### adaptive backdoor (budget 0.6)

| rule | f=5% | f=10% | f=15% | f=20% | f=25% | f=30% | f=40% |
|---|---|---|---|---|---|---|---|
| Undefended | 9.3 ± 3.5 / 74.7 (5) | 34.0 ± 9.7 / 74.3 (10) | 50.4 ± 6.0 / 73.9 (5) | 71.4 ± 5.5 / 73.7 (10) | 76.1 ± 4.0 / 73.5 (5) | 84.2 ± 2.7 / 73.1 (5) | 90.4 ± 2.0 / 72.5 (5) |
| Median | 4.2 ± 0.5 / 72.3 (5) | 15.0 ± 1.9 / 72.0 (10) | 31.5 ± 4.6 / 71.3 (5) | 52.8 ± 6.4 / 70.6 (10) | 66.0 ± 3.4 / 69.8 (5) | 79.0 ± 6.1 / 68.6 (5) | 93.4 ± 1.2 / 67.5 (5) |
| Trimmed mean | 4.2 ± 0.6 / 73.3 (5) | 17.5 ± 2.8 / 72.9 (10) | 34.3 ± 3.4 / 72.3 (5) | 57.7 ± 6.1 / 71.8 (10) | – | 81.4 ± 4.3 / 70.5 (5) | 93.0 ± 0.9 / 69.4 (5) |
| Clipped mean | 9.9 ± 3.0 / 74.3 (5) | 40.1 ± 4.7 / 73.9 (10) | 55.5 ± 4.3 / 73.4 (5) | 76.1 ± 4.4 / 72.9 (10) | – | 88.4 ± 2.2 / 71.9 (5) | 94.3 ± 0.7 / 70.6 (5) |
| Krum | 38.5 ± 14.1 / 32.6 (5) | 74.1 ± 17.4 / 32.1 (10) | 87.0 ± 5.9 / 31.6 (5) | 91.8 ± 3.0 / 30.4 (10) | – | 96.7 ± 0.9 / 31.3 (5) | 97.6 ± 0.6 / 33.3 (5) |
| FoolsGold | 4.3 ± 0.8 / 74.4 (5) | 7.5 ± 1.5 / 73.8 (10) | 7.4 ± 1.8 / 73.4 (5) | 6.1 ± 1.2 / 73.0 (10) | – | 8.2 ± 4.5 / 71.7 (5) | 14.4 ± 19.8 / 71.0 (5) |
| FLTrust | 3.8 ± 0.8 / 69.8 (5) | 5.9 ± 2.4 / 69.6 (10) | 17.0 ± 11.4 / 69.2 (5) | 24.7 ± 12.4 / 68.9 (10) | 34.2 ± 13.5 / 68.9 (5) | 47.3 ± 16.8 / 67.9 (5) | 67.6 ± 12.2 / 66.8 (10) |
| FLAME | 6.2 ± 2.0 / 69.8 (5) | 39.6 ± 3.4 / 69.5 (10) | 54.7 ± 5.6 / 69.0 (5) | 78.2 ± 4.8 / 68.1 (10) | – | 89.8 ± 2.4 / 66.2 (5) | 95.7 ± 0.4 / 63.4 (5) |
| MESAS | 9.6 ± 3.2 / 71.1 (5) | 38.3 ± 4.7 / 71.2 (10) | 51.4 ± 7.3 / 70.5 (5) | 72.4 ± 6.0 / 69.1 (10) | – | 85.9 ± 2.3 / 67.1 (5) | 93.0 ± 1.4 / 63.7 (5) |
| BackdoorIndicator | 3.0 ± 0.4 / 68.9 (5) | 3.1 ± 0.3 / 67.8 (10) | 2.9 ± 0.2 / 67.1 (5) | 3.1 ± 0.4 / 66.3 (10) | – | 3.2 ± 0.7 / 63.7 (5) | 3.6 ± 0.3 / 62.6 (5) |
| MW-Peer | 4.9 ± 1.1 / 71.8 (5) | 22.2 ± 3.4 / 71.2 (10) | 39.6 ± 3.9 / 70.5 (5) | 64.1 ± 4.6 / 69.3 (10) | 73.3 ± 6.1 / 69.1 (5) | 85.5 ± 1.7 / 67.7 (5) | 95.3 ± 1.1 / 66.6 (5) |

## K. Honest update norms (CIFAR-10, f=10%, rounds 151 to 200, client_metrics.json)

| cell | label/hash | seeds | median honest ‖δ‖ | median attacker ‖δ‖ | budget / honest median |
|---|---|---|---|---|---|
| norm-bounded 0.6 vs MW-Peer | `adaptive_backdoor_cifar10_f10__CappedScoreWeight_critic_wmedian/33b6313e3610b4e2` | 10 | 1.85 | 0.60 | 0.32 |
| norm-bounded 0.6, undefended | `adaptive_backdoor_cifar10_f10/d7655368d6eee9dc` | 10 | 1.96 | 0.60 | 0.31 |
| standard backdoor, undefended | `backdoor_cifar10_f10/2246f158ca9949ee` | 10 | 1.94 | 2.33 | – |
