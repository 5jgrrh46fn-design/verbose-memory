f=10%: 10 seeds, 20000 client rows
f=20%: 10 seeds, 20000 client rows

## Detector AUC by criterion weighting (CIFAR-10 standard backdoor, undefended runs)
| f | M4 alone | equal | consistency | magnitude | stability | entropy | critic |
|---|---|---|---|---|---|---|---|
| 10% | 0.983 | 0.921 | 0.943 | 0.931 | 0.909 | 0.935 | 0.957 |
| 20% | 0.926 | 0.904 | 0.951 | 0.865 | 0.908 | 0.914 | 0.949 |

## Single-metric AUC (malicious positive; cost metrics read below 0.5)
| f | M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|---|
| 10% | 0.759 | 0.706 | 0.646 | 0.983 | 0.785 |
| 20% | 0.567 | 0.618 | 0.654 | 0.926 | 0.529 |

## Leave-one-metric-out AUC (critic weighting)
| f | All five | -M1 | -M2 | -M3 | -M4 | -M5 |
|---|---|---|---|---|---|---|
| 10% | 0.957 | 0.955 | 0.956 | 0.953 | 0.846 | 0.958 |
| 20% | 0.949 | 0.943 | 0.946 | 0.943 | 0.655 | 0.954 |
