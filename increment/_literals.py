"""Shared Literal aliases for value domains used across many modules; import
these rather than respelling the values inline."""

from __future__ import annotations

from typing import Literal, get_args

Role = Literal["primary", "secondary", "guardrail", "unassigned"]
# A result row's role: a compiled plan role, or "exploratory" for a metric added after the
# plan was fixed (never a procedure role, so it never enters a compiled family).
RowRole = Literal["primary", "secondary", "guardrail", "unassigned", "exploratory"]
Alternative = Literal["two-sided", "greater", "less"]
ValueScale = Literal["relative", "absolute"]
PreferredDirection = Literal["increase", "decrease", "neutral"]
MultiplicityCorrection = Literal["none", "bh", "bonferroni", "e_bh"]
Correction = Literal["none", "bh", "bonferroni"]
ConversionInference = Literal["auto", "finite_sample"]
# The declared ``Metric.type`` vocabulary (``increment.semantics.models.Metric``).
MetricType = Literal["mean", "conversion", "retention", "ratio", "total", "active", "quantile"]

Backend = Literal["pandas", "polars", "pyarrow"]


def default_frame_backend(preferred: Backend | None = None) -> Backend:
    """Prefer an explicit backend, then pandas, polars, and core pyarrow."""
    if preferred is not None:
        return preferred
    try:
        import pandas  # noqa: F401
    except ModuleNotFoundError:
        try:
            import polars  # noqa: F401
        except ModuleNotFoundError:
            return "pyarrow"
        return "polars"
    return "pandas"


ALTERNATIVE_VALUES = frozenset(get_args(Alternative))
VALUE_SCALE_VALUES = frozenset(get_args(ValueScale))
