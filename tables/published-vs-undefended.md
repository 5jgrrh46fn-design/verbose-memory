| column | rule | rule ASR | undefended | Δ | seeds above | p | Holm p (column) | Δ at the rule's knob | p | Holm p | Δ clean acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CIFAR-10 NB 10% | Clipped mean (knob 0.6; und. 2) | 40.1 | 74.0 | -34.0 | 0/10 | 0.0020 | 0.018 | +6.0 | 0.0488 | 0.146 | +1.3 |
| CIFAR-10 NB 10% | Krum (knob 0.6; und. 2) | 74.1 | 74.0 | +0.1 | 8/10 | 0.4316 | 0.432 | +40.1 | 0.0039 | 0.020 | -40.5 |
| CIFAR-10 NB 10% | FLAME (knob 1.5; und. 2) | 68.6 | 74.0 | -5.5 | 2/10 | 0.0098 | 0.020 | +0.9 | 0.8457 | 0.846 | -5.6 |
| CIFAR-10 NB 10% | MESAS (knob 0.6; und. 2) | 38.3 | 74.0 | -35.8 | 0/10 | 0.0020 | 0.018 | +4.2 | 0.1934 | 0.387 | -1.4 |
| CIFAR-10 NB 20% | Clipped mean (knob 0.6; und. 2) | 76.1 | 86.2 | -10.1 | 0/10 | 0.0020 | 0.018 | +4.7 | 0.0039 | 0.018 | +5.4 |
| CIFAR-10 NB 20% | Krum (knob 0.6; und. 2) | 91.8 | 86.2 | +5.6 | 9/10 | 0.0039 | 0.018 | +20.4 | 0.0020 | 0.018 | -37.1 |
| CIFAR-10 NB 20% | FLAME (knob 1.5; und. 2) | 87.2 | 86.2 | +1.0 | 8/10 | 0.0840 | 0.168 | +2.6 | 0.0137 | 0.041 | -8.6 |
| CIFAR-10 NB 20% | MESAS (knob 0.6; und. 2) | 72.4 | 86.2 | -13.8 | 0/10 | 0.0020 | 0.018 | +1.0 | 0.4922 | 0.984 | +1.6 |
| CIFAR-10 DA 10% | FLAME (knob 0.5; und. 0.5) | 86.0 | 83.1 | +2.9 | 3/10 | 1.0000 | 1.000 | +2.9 | 1.0000 | 1.000 | -5.3 |
| CIFAR-10 DA 20% | FLAME (knob 0.5; und. 0.5) | 96.9 | 96.7 | +0.3 | 7/10 | 0.1934 | 0.320 | +0.3 | 0.1934 | 0.320 | -5.9 |
| FEMNIST NB 10% | FLTrust (knob 0.2; und. 0.2) | 99.6 | 89.2 | +10.4 | 10/10 | 0.0020 | 0.006 | +10.4 | 0.0020 | 0.006 | -2.3 |
| FEMNIST NB 20% | Median (knob 0.2; und. 0.2) | 99.1 | 98.6 | +0.5 | 9/10 | 0.0059 | 0.012 | +0.5 | 0.0059 | 0.012 | -1.8 |
| FEMNIST NB 20% | FLTrust (knob 0.2; und. 0.2) | 99.8 | 98.6 | +1.2 | 10/10 | 0.0020 | 0.006 | +1.2 | 0.0020 | 0.006 | -2.2 |
| FEMNIST NB 20% | FLAME (knob 0.2; und. 0.2) | 98.9 | 98.6 | +0.3 | 9/10 | 0.0371 | 0.037 | +0.3 | 0.0371 | 0.037 | -0.5 |
| CIFAR-100 NB 10% | Clipped mean (knob 3; und. 3) | 58.1 | 55.2 | +2.9 | 9/10 | 0.0098 | 0.039 | +2.9 | 0.0098 | 0.029 | +0.7 |
| CIFAR-100 NB 10% | FoolsGold (knob 3; und. 3) | 56.6 | 55.2 | +1.4 | 8/10 | 0.0488 | 0.098 | +1.4 | 0.0488 | 0.049 | +0.4 |
| CIFAR-100 NB 10% | FLTrust (knob 1; und. 3) | 59.9 | 55.2 | +4.7 | 6/10 | 0.0840 | 0.098 | +36.7 | 0.0020 | 0.016 | -2.6 |
| CIFAR-100 NB 20% | Clipped mean (knob 1; und. 3) | 57.9 | 76.8 | -18.9 | 0/10 | 0.0020 | 0.016 | +2.0 | 0.0059 | 0.023 | +6.7 |
| CIFAR-100 NB 20% | Krum (knob 1; und. 3) | 70.6 | 76.8 | -6.2 | 5/10 | 0.7695 | 1.000 | +14.7 | 0.0098 | 0.029 | -28.4 |
| CIFAR-100 NB 20% | FLTrust (knob 1; und. 3) | 76.1 | 76.8 | -0.8 | 6/10 | 1.0000 | 1.000 | +20.1 | 0.0020 | 0.016 | +2.2 |
| CIFAR-100 NB 20% | FLAME (knob 1; und. 3) | 61.4 | 76.8 | -15.5 | 1/10 | 0.0039 | 0.016 | +5.4 | 0.0273 | 0.055 | +3.2 |
sig above (column Holm<0.05, best response): [('CIFAR-10 NB 20%', 'Krum'), ('FEMNIST NB 10%', 'FLTrust'), ('FEMNIST NB 20%', 'Median'), ('FEMNIST NB 20%', 'FLTrust'), ('FEMNIST NB 20%', 'FLAME'), ('CIFAR-100 NB 10%', 'Clipped mean')]
sig above (same knob): [('CIFAR-10 NB 10%', 'Krum'), ('CIFAR-10 NB 20%', 'Clipped mean'), ('CIFAR-10 NB 20%', 'Krum'), ('CIFAR-10 NB 20%', 'FLAME'), ('FEMNIST NB 10%', 'FLTrust'), ('FEMNIST NB 20%', 'Median'), ('FEMNIST NB 20%', 'FLTrust'), ('FEMNIST NB 20%', 'FLAME'), ('CIFAR-100 NB 10%', 'Clipped mean'), ('CIFAR-100 NB 10%', 'FoolsGold'), ('CIFAR-100 NB 10%', 'FLTrust'), ('CIFAR-100 NB 20%', 'Clipped mean'), ('CIFAR-100 NB 20%', 'Krum'), ('CIFAR-100 NB 20%', 'FLTrust')]
composed (0.6,0.9) f10: Peer (5070 Ti) vs e4_composed_backdoor_cifar10_f10_g0.6_c0.9: n=10 11.6 vs 7.6, Δ=+4.1, above on 8/10, p=0.0098
composed (0.6,0.9) f10: Peer (5070 Ti) vs fin_composed_backdoor_cifar10_f10_g0.6_c0.9: n=10 11.6 vs 7.4, Δ=+4.3, above on 9/10, p=0.0059
composed (0.6,0.9) f20: Peer (5070 Ti) vs e4_composed_backdoor_cifar10_f20_g0.6_c0.9: n=10 48.1 vs 33.4, Δ=+14.6, above on 10/10, p=0.0020
composed (0.6,0.9) f20: Peer (5070 Ti) vs fin_composed_backdoor_cifar10_f20_g0.6_c0.9: n=10 48.1 vs 33.0, Δ=+15.0, above on 10/10, p=0.0020
