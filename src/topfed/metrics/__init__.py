"""Per-client trustworthiness metric extraction.

Computes the five behavioural metrics M1-M5 from a round's collected
``ClientUpdate``s plus the pre-round global parameters. The extractor is
stateful only in the per-client norm history used by M3.
"""

from topfed.metrics.extraction import MetricsExtractor, RoundMetricRow

__all__ = ["MetricsExtractor", "RoundMetricRow"]
