"""Offline result aggregation and statistical reporting.

These modules are read-only consumers of the ``results/`` ledgers written by
``experiments/run.py``. Nothing here writes back into ``results/``.
"""

from topfed.analysis.stats import (
    PairedTest,
    auc_separability,
    bca_ci,
    holm_bonferroni,
    wilcoxon_vs_baseline,
)

__all__ = [
    "PairedTest",
    "auc_separability",
    "bca_ci",
    "holm_bonferroni",
    "wilcoxon_vs_baseline",
]
