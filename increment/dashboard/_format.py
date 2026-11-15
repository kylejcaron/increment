"""Shared dashboard presentation: counts, dates, escaping, interval and reading text.

Every function maps captured readout data to display text or semantic labels. Nothing here
builds a section, reads a source, or knows the theme; ``_html`` and ``_app`` both depend on it.
"""

from __future__ import annotations

import datetime as dt
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from html import escape
from math import inf
from typing import TYPE_CHECKING, Any, Literal

from increment._display import _readout_interval, format_coverage, interval_text
from increment.tables import (
    _decision_available,
    _display_confidence_set,
    _format_confidence_set,
    _result_presentation,
    _retained_coverage_alpha,
)

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


def allocation_count_label(
    allocation: SRMResult, *, population: Literal["assigned", "triggered"] = "assigned"
) -> str:
    label = "Triggered" if population == "triggered" else "Enrolled"
    return f"{label} {allocation_grain_label(allocation)}"


def allocation_population_detail(
    allocation: SRMResult, *, population: Literal["assigned", "triggered"] = "assigned"
) -> str:
    label = "Triggered" if population == "triggered" else "Assigned"
    if allocation.grain == "cluster" and allocation.unit_counts:
        units = sum(allocation.unit_counts.values())
        return f"{label} clusters · {count_text(units)} member units"
    return f"{label} population"


def allocation_verdict(
    allocation: SRMResult, *, population: Literal["assigned", "triggered"] = "assigned"
) -> str:
    if population == "triggered":
        if allocation.is_srm:
            return (
                "Triggered cohort differs from declared allocation proportions; imbalance may "
                "reflect treatment-dependent triggering."
            )
        return (
            "Triggered cohort matches declared allocation proportions within the balance-test "
            "threshold; imbalance may reflect treatment-dependent triggering."
        )
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
    interval = _readout_interval(row)
    lower, higher, open_side = interval.lower, interval.upper, interval.open_side
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


def headline_endpoint(value: Any, row: Mapping[str, Any]) -> str | None:
    """An endpoint as headline text; ``None`` for an infinite end."""
    return None if value in (-inf, inf) else headline_number(value, row)


def complete_confidence_set_text(row: Mapping[str, Any]) -> str | None:
    """Escaped full description of the retained confidence set, or None when it adds nothing.

    Unlike ``confidence_set_text`` this keeps a one-sided set that has a point estimate: its
    open direction and own coverage differ from the central interval shown beside the point.
    Only a set the displayed interval already represents exactly is omitted.
    """
    text = _format_confidence_set(
        row.get("confidence_set"),
        relative=row.get("relative_confidence_set"),
        binomial=row.get("binomial_set"),
        unavailable=row.get("relative_unavailable_reason"),
        scale=str(row.get("value_scale", "relative")),
        lift=row.get("lift"),
    )
    return esc(text) if text else None


def confidence_set_text(row: Mapping[str, Any]) -> str | None:
    """Headline set text, using the tested ray for a supported directional guardrail."""
    relative = row.get("relative_confidence_set")
    if (
        relative is not None
        and not is_missing(relative)
        and relative.geometry == "one_sided"
        and not is_missing(row.get("lift"))
        and interval_endpoints(row) is not None
    ):
        return None
    text = _display_confidence_set(row)
    return esc(text) if text else None


def headline_interval(row: Mapping[str, Any]) -> str:
    set_text = confidence_set_text(row)
    if set_text is not None:
        return set_text
    endpoints = interval_endpoints(row)
    if endpoints is None:
        return f"[{missing_html('no interval available')}]"
    lower, higher = endpoints
    return interval_text(headline_endpoint(lower, row), headline_endpoint(higher, row))


def primary_tone(row: Mapping[str, Any]) -> str:
    """Semantic headline tone shared with readout and standalone tables."""
    return _result_presentation(row)[0]


def displayed_coverage(row: Mapping[str, Any]) -> tuple[float, float | None] | None:
    """Coverage of the displayed bounds, separate from the captured interval's budget."""
    interval = _readout_interval(row)
    return None if interval.level is None else (interval.level, interval.alpha)


def percent_text(level: float, alpha: float | None = None) -> str:
    """A compact coverage percentage, retaining precision near zero and one."""
    return format_coverage(level, alpha)


def retained_coverage_alpha(estimate: Any) -> float | None:
    """Return the noncoverage budget for the representation supplying the shown level."""
    return _retained_coverage_alpha(estimate)


def retained_coverage_alpha_for_row(
    snapshot: DashboardSnapshot, row: Mapping[str, Any]
) -> float | None:
    """Match a displayed readout row to its captured estimate and retain its own budget."""
    estimate = next(
        (
            estimate
            for estimate, readout in zip(snapshot.estimates, snapshot.readout_rows, strict=True)
            if readout is row
        ),
        None,
    )
    return retained_coverage_alpha(estimate)


def primary_method(row: Mapping[str, Any]) -> str:
    """Describe the displayed interval using its retained coverage budget."""
    coverage = displayed_coverage(row)
    level_label = "" if coverage is None else f"{percent_text(*coverage)} "
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
    return interval_text(headline_endpoint(lower, row), headline_endpoint(higher, row))


def decision_scale(row: Mapping[str, Any]) -> str:
    """The scale the row's own decision is made on, as the readout adapter persisted it
    (``decision_scale`` from ``increment.estimation.results.row_decision_scale``); a row
    without the column decides on its declared absolute margin when one is set."""
    scale = row.get("decision_scale")
    if not is_missing(scale):
        return str(scale)
    return "relative" if is_missing(row.get("null_abs")) else "absolute"


def _absolute_null(row: Mapping[str, Any]) -> Any:
    null = row.get("null_abs")
    if not is_missing(null):
        return null
    if row.get("value_scale") == "absolute":
        return row.get("null_lift") or 0.0
    return 0.0


def null_text(row: Mapping[str, Any]) -> str:
    """The tested null on the row's own decision scale."""
    if decision_scale(row) == "absolute":
        return f"{_absolute_null(row):+,.4g} absolute"
    null_lift = row.get("null_lift")
    if null_lift is None:
        return missing_html("no null boundary reported")
    return f"{float(null_lift):+.1%} relative"


def guardrail_text(row: Mapping[str, Any]) -> str:
    """Describe the declared guardrail boundary, not the observed outcome."""
    if row.get("role") != "guardrail":
        return ""
    absolute = decision_scale(row) == "absolute"
    boundary = _absolute_null(row) if absolute else row.get("null_lift")
    if boundary is None or is_missing(boundary):
        return "Guardrail: threshold unavailable"
    boundary = float(boundary)
    amount = (
        f"{abs(boundary):.4g} absolute units of"
        if absolute
        else f"a {abs(boundary) * 100:.4g}% relative"
    )
    alternative = row.get("alternative")
    if alternative == "greater":
        return (
            f"Guardrail: no more than {amount} decrease"
            if boundary <= 0
            else f"Guardrail: at least {amount} increase"
        )
    if alternative == "less":
        return (
            f"Guardrail: no more than {amount} increase"
            if boundary >= 0
            else f"Guardrail: at least {amount} decrease"
        )
    return f"Guardrail threshold: {null_text(row)}"


@dataclass(frozen=True)
class ResultOutcome:
    """Semantic row verdict for dashboard consumers."""

    verdict: Literal["reject", "not_reject", "unavailable"]
    unavailable_reason: str | None = None


def result_outcome(row: Mapping[str, Any]) -> ResultOutcome:
    """Read the row's persisted verdict without inferring it from interval geometry."""
    significant = row.get("stat_sig")
    if significant is True:
        return ResultOutcome("reject")
    if significant is False and _decision_available(row):
        return ResultOutcome("not_reject")
    reason = row.get("failure_code") or row.get("relative_unavailable_reason")
    return ResultOutcome("unavailable", str(reason) if reason is not None else None)


def outcome_label(row: Mapping[str, Any]) -> str:
    """Format emitted decision and family fields; do not infer either from effect geometry."""
    verdict = {
        "reject": "rejects the declared null",
        "not_reject": "does not reject the declared null",
        "unavailable": "inference unavailable",
    }[result_outcome(row).verdict]
    discovery = row.get("discovery")
    selection = (
        "selected by the family procedure"
        if discovery is True
        else "not selected by the family procedure"
        if discovery is False
        else "family selection not applied"
    )
    return f"{tail_word(row)} {verdict}; {selection}"


ROLE_LABELS = {
    "primary": "Primary",
    "secondary": "Secondaries",
    "guardrail": "Guardrails",
    "unassigned": "Unassigned",
}


def levels_by_metric(pairs: Iterable[tuple[Any, Any]], fmt: Callable[[Any], str]) -> str:
    """Captured interval coverage, attributed to its metric only when displays differ."""
    by_level: dict[Any, list[str]] = {}
    for metric, level in pairs:
        names = by_level.setdefault(level, [])
        if str(metric) not in names:
            names.append(str(metric))
    if len(by_level) == 1:
        return fmt(next(iter(by_level)))
    return "; ".join(f"{fmt(level)} ({', '.join(names)})" for level, names in by_level.items())


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


def without_advisory_notes(row: Mapping[str, Any], messages: Iterable[str]) -> dict[str, Any]:
    """Remove only exact copies of advisories displayed as structured diagnostics."""
    result = dict(row)
    note = result.get("note")
    if isinstance(note, str):
        for message in messages:
            note = note.replace(f"{message}; ", "").replace(f"; {message}", "")
            note = note.replace(message, "")
        result["note"] = note or None
    return result


def result_caveats(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """Keep row caveats and unavailable evidence visible without hiding usable sets."""
    return _result_caveats(rows, omit_table_footnotes=False)


def _result_table_caveats(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """Keep non-table warnings visible when note/reliability text is already in table footnotes."""
    return _result_caveats(rows, omit_table_footnotes=True)


def _result_caveats(rows: Iterable[Mapping[str, Any]], *, omit_table_footnotes: bool) -> list[str]:
    caveats: list[str] = []
    for row in rows:
        metric = esc(row.get("metric", "unknown metric"))
        if not omit_table_footnotes:
            note = row.get("note")
            if note:
                caveats.append(f"<strong>{metric}</strong>: note: {esc(note)}")
            if row.get("low_reliability"):
                caveats.append(f"<strong>{metric}</strong>: flagged low reliability.")
        for label, value in (
            ("excluded", row.get("excluded")),
            ("unavailable", row.get("unavailable")),
        ):
            if value:
                caveats.append(f"<strong>{metric}</strong>: {label}: {esc(value)}")
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
