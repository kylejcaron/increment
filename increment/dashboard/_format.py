"""Shared dashboard presentation: counts, dates, escaping, interval and reading text.

Every function maps captured readout data to display text or semantic labels. Nothing here
builds a section, reads a source, or knows the theme; ``_html`` and ``_app`` both depend on it.
"""

from __future__ import annotations

import datetime as dt
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from html import escape
from math import inf
from typing import TYPE_CHECKING, Any

from increment.tables import _decision_available, _format_confidence_set

if TYPE_CHECKING:
    from increment.dashboard._data import DashboardSnapshot
    from increment.estimation.diagnostics import SRMResult


def esc(value: object) -> str:
    """Escape dynamic text at the HTML boundary."""
    return escape(str(value), quote=True)


def count_text(value: int | float) -> str:
    return f"{value:,.0f}"


def share_text(value: float) -> str:
    return f"{value:.1%}"


def number_text(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}g}"


def date_label(value: dt.datetime | dt.date | None) -> str:
    if value is None:
        return "open"
    return value.date().isoformat() if isinstance(value, dt.datetime) else value.isoformat()


def timestamp_label(value: dt.datetime) -> str:
    return esc(value.strftime("%Y-%m-%d %H:%M UTC"))


def missing_html(reason: str) -> str:
    """An unavailable value beside the supplied reason."""
    return f'<span class="inc-dashboard-missing">N/A</span> <span class="inc-dashboard-reason">{esc(reason)}</span>'


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def allocation_grain_label(allocation: SRMResult, *, plural: bool = True) -> str:
    """Return the actual sampling grain used by the SRM result."""
    if allocation.grain == "cluster":
        return "clusters" if plural else "cluster"
    return "units" if plural else "unit"


def allocation_count_label(allocation: SRMResult) -> str:
    return f"Enrolled {allocation_grain_label(allocation)}"


def allocation_population_detail(allocation: SRMResult) -> str:
    if allocation.grain == "cluster" and allocation.unit_counts:
        units = sum(allocation.unit_counts.values())
        return f"Assigned clusters · {count_text(units)} member units"
    return "Assigned population"


def allocation_verdict(allocation: SRMResult) -> str:
    if allocation.is_srm:
        return "Sample ratio mismatch detected"
    return "No allocation issues detected"


def allocation_evidence(allocation: SRMResult) -> str:
    """Which statistic decided the check, and at which level.

    The allocation level is stated here and nowhere near a result interval:
    they answer different questions.
    """
    alpha = f"α = {number_text(allocation.alpha, 3)}"
    if allocation.inference == "always_valid":
        if allocation.log_e_value is None:
            return f"Always-valid evidence, e-value unavailable, at {alpha}."
        return (
            f"Always-valid evidence (log e-value {number_text(allocation.log_e_value)}) at {alpha}."
        )
    return f"Fixed-horizon chi-square evidence (p = {number_text(allocation.fixed_p_value)}) at {alpha}."


def inference_word(inference: str) -> str:
    return {
        "always_valid": "always-valid",
        "asymptotic_mean": "asymptotic sequential",
        "fixed": "fixed-horizon",
    }.get(inference, inference)


def tail_word(row: Mapping[str, Any]) -> str:
    alternative = str(row.get("alternative", "two-sided"))
    return {
        "two-sided": "two-sided test",
        "greater": "one-sided test for an increase",
        "less": "one-sided test for a decrease",
    }.get(alternative, f"{alternative} test")


def population_label(snapshot: DashboardSnapshot) -> str:
    populations = sorted(
        {str(row.get("analysis_population", "assigned")) for row in snapshot.readout_rows}
    )
    return esc(", ".join(f"{name} population" for name in populations) or "assigned population")


def inference_label(snapshot: DashboardSnapshot) -> str:
    kinds = sorted({str(row.get("inference", "fixed")) for row in snapshot.readout_rows})
    return esc(", ".join(inference_word(kind) for kind in kinds) or "fixed-horizon")


def interval_endpoints(row: Mapping[str, Any]) -> tuple[Any, Any] | None:
    """Return finite/open endpoints, or None for malformed/unavailable rows."""
    lower, higher, open_side = row.get("lower"), row.get("higher"), row.get("open_side")
    if is_missing(lower):
        lower = None
    if is_missing(higher):
        higher = None
    if open_side == "lower" and higher is not None:
        return -inf, higher
    if open_side == "upper" and lower is not None:
        return lower, inf
    if open_side is None and lower is not None and higher is not None:
        return lower, higher
    return None


def headline_number(value: Any, row: Mapping[str, Any]) -> str:
    text = f"{value:+.1%}" if row.get("value_scale") == "relative" else f"{value:+,.4g}"
    return text.replace("-", "−", 1)


def headline_endpoint(value: Any, row: Mapping[str, Any]) -> str:
    if value == -inf:
        return "−∞"
    if value == inf:
        return "+∞"
    return headline_number(value, row)


def confidence_set_text(row: Mapping[str, Any]) -> str | None:
    relative = row.get("relative_confidence_set")
    if (
        relative is not None
        and not is_missing(relative)
        and relative.geometry == "one_sided"
        and not is_missing(row.get("lift"))
        and interval_endpoints(row) is not None
    ):
        return None

    text = _format_confidence_set(
        row.get("confidence_set"),
        relative=row.get("relative_confidence_set"),
        binomial=row.get("binomial_set"),
        unavailable=row.get("relative_unavailable_reason"),
        scale=str(row.get("value_scale", "relative")),
        lift=row.get("lift"),
    )
    return esc(text) if text else None


def headline_interval(row: Mapping[str, Any]) -> str:
    set_text = confidence_set_text(row)
    if set_text is not None:
        return set_text
    endpoints = interval_endpoints(row)
    if endpoints is None:
        return f"[{missing_html('no interval available')}]"
    lower, higher = endpoints
    opening = "(" if lower == -inf else "["
    closing = ")" if higher == inf else "]"
    return f"{opening}{headline_endpoint(lower, row)}, {headline_endpoint(higher, row)}{closing}"


def primary_tone(row: Mapping[str, Any]) -> str:
    """Semantic headline tone from the row's tested verdict and declared direction."""
    if not row.get("stat_sig"):
        return "inconclusive" if _decision_available(row) else "neutral"
    direction = row.get("preferred_direction")
    if direction not in ("increase", "decrease"):
        return "neutral"

    null_abs = row.get("null_abs")
    if not is_missing(null_abs):
        null, lower, higher = null_abs, row.get("abs_lb"), row.get("abs_ub")
    else:
        null = row.get("null_lift")
        null = 0.0 if null is None else null
        lower, higher = row.get("lower"), row.get("higher")
    if lower is not None and lower > null:
        observed = "increase"
    elif higher is not None and higher < null:
        observed = "decrease"
    else:
        return "neutral"
    return "favorable" if observed == direction else "unfavorable"


def primary_method(row: Mapping[str, Any]) -> str:
    level = row.get("level")
    if level is None:
        level_label = ""
    else:
        percent = f"{float(level) * 100:.2f}".rstrip("0").rstrip(".")
        level_label = f"{percent}% "
    inference = inference_word(str(row.get("inference", "fixed")))
    return f"{level_label}{esc(inference)} interval · {esc(tail_word(row))}"


def effect_html(row: Mapping[str, Any]) -> str:
    """The point estimate on its own scale, or a missing marker with its reason."""
    lift = row.get("lift")
    if is_missing(lift):
        reason = (
            "point estimate unavailable"
            if confidence_set_text(row) is not None
            else "no estimate available"
        )
        return missing_html(str(row.get("note") or reason))
    if row.get("value_scale") == "relative":
        return f"{lift:+.1%}"
    return f"{lift:+,.4g}"


def interval_html(row: Mapping[str, Any]) -> str:
    set_text = confidence_set_text(row)
    if set_text is not None:
        return set_text
    endpoints = interval_endpoints(row)
    if endpoints is None:
        return missing_html("no interval available")
    lower, higher = endpoints
    return f"{headline_endpoint(lower, row)} to {headline_endpoint(higher, row)}"


def null_text(row: Mapping[str, Any]) -> str:
    null_abs = row.get("null_abs")
    if null_abs is not None:
        return f"{null_abs:+,.4g} absolute"
    null_lift = row.get("null_lift")
    if null_lift is None:
        return missing_html("no null boundary reported")
    return f"{float(null_lift):+.1%} relative"


ROLE_LABELS = {
    "primary": "Primary",
    "secondary": "Secondaries",
    "guardrail": "Guardrails",
    "unassigned": "Unassigned",
}


def levels_by_metric(pairs: Iterable[tuple[Any, float]], fmt: Callable[[float], str]) -> str:
    """Captured interval levels, attributed to their metrics only when they differ."""
    by_level: dict[float, list[str]] = {}
    for metric, level in pairs:
        names = by_level.setdefault(float(level), [])
        if str(metric) not in names:
            names.append(str(metric))
    if len(by_level) == 1:
        return fmt(next(iter(by_level)))
    return "; ".join(
        f"{fmt(level)} ({', '.join(names)})" for level, names in sorted(by_level.items())
    )


def monitoring_sentence(rows: Sequence[Any], *, view: str) -> str:
    """What this series is and is not, from the inference it actually carries."""
    if view == "daily_values":
        return (
            "Daily slices are descriptive measurements. They are not independent "
            "sequential lift tests."
        )
    kinds = {str(getattr(row, "inference", "")) for row in rows} - {""}
    if kinds == {"always_valid"}:
        return "Always-valid intervals, so every as-of point is a valid look."
    if kinds == {"asymptotic_mean"}:
        return (
            "Asymptotic sequential intervals support repeated monitoring under the "
            "registered assumptions, without a finite-sample guarantee."
        )
    if kinds:
        return (
            "Descriptive monitoring with fixed-horizon intervals: these points are not "
            "corrected for repeated looks."
        )
    return "Descriptive monitoring of per-arm values."


def result_caveats(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """Keep row caveats and unavailable evidence visible without hiding usable sets."""
    caveats: list[str] = []
    for row in rows:
        metric = esc(row.get("metric", "unknown metric"))
        for label, value in (
            ("note", row.get("note")),
            ("excluded", row.get("excluded")),
            ("unavailable", row.get("unavailable")),
        ):
            if value:
                caveats.append(f"<strong>{metric}</strong>: {label}: {esc(value)}")
        if row.get("low_reliability"):
            caveats.append(f"<strong>{metric}</strong>: flagged low reliability.")
        relative = row.get("relative_confidence_set")
        winsor = row.get("confidence_set")
        unavailable = row.get("relative_unavailable_reason")
        if not is_missing(unavailable):
            caveats.append(f"<strong>{metric}</strong>: unavailable: {esc(str(unavailable))}")
        elif relative is not None and not is_missing(relative):
            if relative.reason or relative.geometry in ("unavailable", "empty"):
                caveats.append(
                    f"<strong>{metric}</strong>: {relative.geometry}: "
                    f"{esc(relative.reason or 'confidence set has no members')}"
                )
        elif winsor is not None and not is_missing(winsor):
            interval = (
                winsor.relative
                if row.get("value_scale", "relative") == "relative"
                else winsor.additive
            )
            for reason in dict.fromkeys((interval.lower.reason, interval.upper.reason)):
                if reason:
                    caveats.append(f"<strong>{metric}</strong>: confidence set: {esc(reason)}")
        elif not is_missing(row.get("binomial_set")):
            continue
        elif is_missing(row.get("lift")):
            caveats.append(f"<strong>{metric}</strong>: {missing_html('no estimate available')}")
        elif interval_endpoints(row) is None:
            caveats.append(f"<strong>{metric}</strong>: no interval available for this estimate.")
    return caveats


def has_open_side(estimates: Sequence[Any]) -> bool:
    """True when any plotted interval is one-sided, so a ribbon cannot be drawn for it."""
    return any(e.lift is not None and e.lift.open_side is not None for e in estimates)


def robust_fence(values: Sequence[float]) -> tuple[float, float] | None:
    """The IQR/Tukey fence CoefTable documents for ``autoscale="robust"``.

    ``None`` when quartiles are not meaningful (fewer than four values or a zero IQR), where the
    chart falls back to a plain min/max fit and clips nothing. Used only to describe clipping.
    """
    if len(values) < 4:
        return None
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    spread = q3 - q1
    if spread == 0:
        return None
    return q1 - 1.5 * spread, q3 + 1.5 * spread
