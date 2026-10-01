"""Canonical window-day resolution shared by the frame (narwhals) and query
(ibis/warehouse) engines, so they agree on where a metric's observation
window ends without either engine depending on the other.

Pure metric-spec logic: no ibis, no narwhals. increment/frame.py must not
import increment/query/, and increment/query/builders.py must not carry a
hand-copied twin of this logic - both import it from here instead.
"""

from __future__ import annotations

import datetime as dt

from increment.errors import (
    UnsupportedRequestError,
    raiser,
    refusals,
)
from increment.semantics.models import (
    ActiveMetric,
    ConversionMetric,
    MeanMetric,
    Metric,
    QuantileMetric,
    RatioMetric,
    RetentionMetric,
    TotalMetric,
)

_REFUSALS = refusals(
    UnsupportedRequestError,
    {
        "facade.window.resolve_window_days": "resolve_window_days for {metric_type}",
    },
)
_raise = raiser(_REFUSALS)

#: Watermark for a measure with zero matching rows; it never wins a
#: `min`/`least` against a real date. native_source.py and artifact_reader.py
#: share it so publish and adopt recognize a zero-row measure identically.
NO_DATA_SIGNAL = dt.date(9999, 12, 31)


def resolve_window_days(metric: Metric) -> int | None:
    """Return the observation band's right edge, in days after exposure
    (exclusive). None means no right edge - callers skip the bound.

    Not the same question as a metric's maturity day: an unbounded
    ``RetentionMetric`` has no band edge here, but its outcome is still
    treated as final at the threshold day there. Callers bounding rows
    want this function; callers deciding whether a unit's outcome is
    final want the maturity day instead.
    """
    if isinstance(metric, RetentionMetric):
        # The band's right edge - None when the band is open on the
        # right. Never the vacated window_days attribute.
        return metric.band[1]
    if isinstance(metric, ConversionMetric | MeanMetric | QuantileMetric):
        # None = variable per-unit window, never censor. A quantile metric
        # carries no window semantics of its own, so it always resolves
        # here rather than raising - MetricSpec already refuses a declared
        # window_days on a quantile metric before this ever runs.
        return metric.window_days
    if isinstance(metric, RatioMetric):
        # numerator.window_days is canonical for window censoring
        return metric.numerator.window_days
    if isinstance(metric, TotalMetric | ActiveMetric):
        # Report-only: no per-unit variance, rejected by estimation before
        # any caller of this function would see one. No window semantics.
        return None
    _raise("facade.window.resolve_window_days", metric_type=type(metric).__name__)


def maturity_days(metric: Metric) -> int | None:
    """Return the maturity offset used to censor an immature unit.

    Retention uses its band's right edge when present; an unbounded retention
    band falls back to ``band_start``. Other metric types use
    :func:`resolve_window_days`.
    """
    if isinstance(metric, RetentionMetric):
        band_start, band_end = metric.band
        return band_end if band_end is not None else band_start
    return resolve_window_days(metric)


def final_maturity_day(metric: Metric) -> int | None:
    """Return the inclusive offset of the last day an outcome depends on.

    Unbounded retention falls back to its ``band_start`` and therefore keeps
    that maturity unchanged. Bounded metrics use one day before their
    exclusive maturity offset; metrics without a fixed maturity return
    ``None``.
    """
    maturity = maturity_days(metric)
    if maturity is None:
        return None
    if isinstance(metric, RetentionMetric) and metric.band[1] is None:
        return maturity
    return maturity - 1
