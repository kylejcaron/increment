"""Section-level rendering for the dashboard.

Every function here takes a prepared snapshot and returns one complete
section. No function opens a connection, builds a query, or creates widget
state: notebooks own their controls and pass the values in.
"""

from __future__ import annotations

import datetime as dt
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from html import escape
from importlib import resources
from math import ceil, inf, isfinite, log10
from typing import TYPE_CHECKING, Any

import coeftable as ct
import marimo as mo
import pandas as pd
from coeftable import Theme
from scipy.stats import norm

from increment.dashboard._data import (
    DashboardSnapshot,
    decision_rows,
    estimate_for_metric,
    group_data_rows,
    require_metric,
    row_for_metric,
)
from increment.tables import _format_confidence_set, estimates_to_readout, readout_table

if TYPE_CHECKING:
    from increment.breakout.estimates import (
        BreakoutEstimates,
        DailyLiftEstimates,
        DailyMetricValues,
    )
    from increment.dashboard._data import ExploreView
    from increment.estimation.diagnostics import SRMResult

__all__ = [
    "dashboard_styles",
    "render_details",
    "render_header",
    "render_health",
    "render_explore",
    "render_metric_details",
    "render_results",
]

# CoefTable's public theme, not selectors into its generated table ids.
# series_palette follows series order, which is control then treatment, so
# arm identity stays separate from favorable/unfavorable estimate colour.
DASHBOARD_THEME = Theme(
    favorable="#386647",
    unfavorable="#A23C3C",
    inconclusive="#74816F",
    neutral="#42644D",
    header_bg="#FFFEF9",
    header_fg="#292D26",
    column_label_bg="#ECEFE4",
    band="#F5F5ED",
    surface="#FFFEF9",
    rule="#DCDED1",
    border_color="#DCDED1",
    axis="#62695D",
    muted="#62695D",
    text="#292D26",
    value_size="14px",
    ci_size="12px",
    table_font_size="14px",
    border_style="minimal",
    na_text="N/A",
    series_palette=("#737B6D", "#42644D"),
)


def dashboard_styles() -> mo.Html:
    """The packaged, scoped stylesheet as one style block.

    Read from package data, so an installed wheel needs no stylesheet path
    and no repository checkout.
    """
    css = resources.files(__package__).joinpath("_dashboard.css").read_text(encoding="utf-8")
    return mo.Html(f"<style>{css}</style>")


def _esc(value: object) -> str:
    """Escape dynamic text at the HTML boundary."""
    return escape(str(value), quote=True)


def _count(value: int | float) -> str:
    return f"{value:,.0f}"


def _allocation_grain_label(allocation: SRMResult, *, plural: bool = True) -> str:
    """Return the actual sampling grain used by the SRM result."""
    if allocation.grain == "cluster":
        return "clusters" if plural else "cluster"
    return "units" if plural else "unit"


def _allocation_count_label(allocation: SRMResult) -> str:
    return f"Enrolled {_allocation_grain_label(allocation)}"


def _allocation_population_detail(allocation: SRMResult) -> str:
    if allocation.grain == "cluster" and allocation.unit_counts:
        units = sum(allocation.unit_counts.values())
        return f"Assigned clusters · {_count(units)} member units"
    return "Assigned population"


def _share(value: float) -> str:
    return f"{value:.1%}"


def _number(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}g}"


def _date(value: dt.datetime | dt.date | None) -> str:
    if value is None:
        return "open"
    return value.date().isoformat() if isinstance(value, dt.datetime) else value.isoformat()


def _missing(reason: str) -> str:
    """An unavailable value beside the supplied reason."""
    return f'<span class="inc-dashboard-missing">N/A</span> <span class="inc-dashboard-reason">{_esc(reason)}</span>'


def _section(anchor: str, heading: str, body: str, *, subtitle: str = "") -> mo.Html:
    sub = f'<p class="inc-dashboard-subtitle">{_esc(subtitle)}</p>' if subtitle else ""
    return mo.Html(
        f'<section class="inc-dashboard-root inc-dashboard-section" id="{_esc(anchor)}">'
        f'<h2 class="inc-dashboard-heading">{_esc(heading)}</h2>{sub}{body}</section>'
    )


def _disclosure(label: str, body: str) -> str:
    return (
        f'<details class="inc-dashboard-disclosure"><summary>{_esc(label)}</summary>'
        f"{body}</details>"
    )


def _chips(entries: Sequence[tuple[str, str]]) -> str:
    items = "".join(f"<li><b>{_esc(label)}</b> {value}</li>" for label, value in entries)
    return f'<ul class="inc-dashboard-chips">{items}</ul>'


def _card(label: str, value: str, detail: str = "", *, worded: bool = False) -> str:
    """One summary card. ``worded`` values read as a sentence, not a figure."""
    foot = f'<p class="inc-dashboard-card-detail">{detail}</p>' if detail else ""
    variant = " inc-dashboard-card-value--text" if worded else ""
    return (
        '<div class="inc-dashboard-card">'
        f'<p class="inc-dashboard-card-label">{_esc(label)}</p>'
        f'<p class="inc-dashboard-card-value{variant}">{value}</p>{foot}</div>'
    )


def _cards(cards: Iterable[str]) -> str:
    return f'<div class="inc-dashboard-cards">{"".join(cards)}</div>'


def _status(tone: str, text: str) -> str:
    """A status line whose colour is always accompanied by its wording."""
    return f'<p class="inc-dashboard-status inc-dashboard-status--{tone}">{_esc(text)}</p>'


def _list(items: Sequence[str], *, css_class: str) -> str:
    if not items:
        return ""
    rows = "".join(f"<li>{item}</li>" for item in items)
    return f'<ul class="{css_class}">{rows}</ul>'


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "".join(f"<th scope='col'>{_esc(header)}</th>" for header in headers)
    body = "".join(
        "<tr>"
        + "".join(
            f"<th scope='row'>{cell}</th>" if index == 0 else f"<td>{cell}</td>"
            for index, cell in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    return (
        '<div class="inc-dashboard-table-wrap"><table class="inc-dashboard-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


# Header


def render_header(snapshot: DashboardSnapshot) -> mo.Html:
    """Experiment identity, analysis policy, and the headline summary."""
    lede = (
        f'<p class="inc-dashboard-lede">{_esc(snapshot.description)}</p>'
        if snapshot.description
        else ""
    )
    source = snapshot.config.source_label
    source_html = f'<span class="inc-dashboard-source">{_esc(source)}</span>' if source else ""
    body = (
        '<div class="inc-dashboard-headline">'
        f'<h1 class="inc-dashboard-title">{_esc(snapshot.title)}</h1>{source_html}</div>'
        f"{lede}{_header_chips(snapshot)}{_header_cards(snapshot)}"
    )
    return mo.Html(
        f'<section class="inc-dashboard-root inc-dashboard-section" id="overview">{body}</section>'
    )


def _header_chips(snapshot: DashboardSnapshot) -> str:
    chips = [
        ("Window", f"{_date(snapshot.start)} → {_date(snapshot.end)}"),
        ("Population", _population_label(snapshot)),
        ("Inference", _inference_label(snapshot)),
        (
            "Arms",
            f'<span class="inc-dashboard-arm">{_esc(snapshot.control_group)}</span> → '
            f'<span class="inc-dashboard-arm">{_esc(snapshot.treatment_group)}</span>',
        ),
        ("Computed", _timestamp(snapshot.computed_at)),
    ]
    return _chips(chips)


def _header_cards(snapshot: DashboardSnapshot) -> str:
    allocation = snapshot.allocation
    if allocation is None:
        reason = (snapshot.allocation_refusal or ("", ""))[1]
        enrolled_card = _card(
            "Enrolled units",
            _missing("allocation check unavailable"),
            worded=True,
        )
        check_card = _card(
            "Allocation check",
            _missing(reason or "refused by the source"),
            "Not a passing check.",
            worded=True,
        )
    else:
        enrolled = sum(allocation.observed.values())
        enrolled_card = _card(
            _allocation_count_label(allocation),
            _count(enrolled),
            _allocation_population_detail(allocation),
        )
        check_card = _card(
            "Observed arm split",
            " / ".join(
                _share(allocation.observed.get(arm, 0) / enrolled) if enrolled else "N/A"
                for arm in (snapshot.control_group, snapshot.treatment_group)
            ),
            f"{_esc(snapshot.control_group)} / {_esc(snapshot.treatment_group)}",
        )
    return _cards([enrolled_card, check_card, _primary_card(snapshot)])


def _primary_card(snapshot: DashboardSnapshot) -> str:
    row = row_for_metric(snapshot, snapshot.primary_metric)
    if row is None:
        return _card(
            "Relative lift vs control",
            _missing(f"no decision result for {snapshot.primary_metric}"),
            worded=True,
        )
    label = (
        "Absolute effect vs control"
        if row.get("value_scale") == "absolute"
        else "Relative lift vs control"
    )
    has_point = not _is_missing(row.get("lift"))
    tone = _primary_tone(row)
    status = _primary_status(tone, significant=bool(row.get("stat_sig")))
    point = _headline_number(row["lift"], row) if has_point else _effect(row)
    caption = _primary_method(row)
    if not has_point and _confidence_set_text(row) is not None:
        point = ""
        caption = f"Point estimate unavailable · {caption}"
    point_html = f'<span class="inc-dashboard-primary-estimate">{point}</span>' if point else ""
    return (
        f'<div class="inc-dashboard-card inc-dashboard-primary '
        f'inc-dashboard-primary--{tone}">'
        f'<p class="inc-dashboard-card-label">{label}</p>'
        '<p class="inc-dashboard-primary-inline">'
        f"{point_html}"
        f'<span class="inc-dashboard-primary-interval">{_headline_interval(row)}</span></p>'
        '<div class="inc-dashboard-primary-caption">'
        f"<span>{caption}</span>{status}</div></div>"
    )


def _primary_tone(row: Mapping[str, Any]) -> str:
    """Semantic headline tone from the row's tested verdict and declared direction."""
    from increment.tables import _decision_available

    if not row.get("stat_sig"):
        return "inconclusive" if _decision_available(row) else "neutral"
    direction = row.get("preferred_direction")
    if direction not in ("increase", "decrease"):
        return "neutral"

    null_abs = row.get("null_abs")
    if not _is_missing(null_abs):
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


def _primary_status(tone: str, *, significant: bool) -> str:
    if tone == "neutral":
        if not significant:
            return ""
        return (
            '<span class="inc-dashboard-primary-status" '
            'title="The result is statistically significant but has no declared favorable direction.">'
            "Significant</span>"
        )
    if tone == "inconclusive":
        label = "Not significant"
        title = (
            "This result did not meet its declared significance threshold. "
            "This does not establish that there is no effect."
        )
    else:
        label = f"Significant · {tone}"
        title = f"The result is statistically significant and {tone} in the declared direction."
    return f'<span class="inc-dashboard-primary-status" title="{_esc(title)}">{_esc(label)}</span>'


def _headline_number(value: Any, row: Mapping[str, Any]) -> str:
    text = f"{value:+.1%}" if row.get("value_scale") == "relative" else f"{value:+,.4g}"
    return text.replace("-", "−", 1)


def _confidence_set_text(row: Mapping[str, Any]) -> str | None:
    from increment.tables import _format_confidence_set

    relative = row.get("relative_confidence_set")
    if (
        relative is not None
        and not _is_missing(relative)
        and relative.geometry == "one_sided"
        and not _is_missing(row.get("lift"))
        and _interval_endpoints(row) is not None
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
    return _esc(text) if text else None


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def _interval_endpoints(row: Mapping[str, Any]) -> tuple[Any, Any] | None:
    """Return finite/open endpoints, or None for malformed/unavailable rows."""
    lower, higher, open_side = row.get("lower"), row.get("higher"), row.get("open_side")
    if _is_missing(lower):
        lower = None
    if _is_missing(higher):
        higher = None
    if open_side == "lower" and higher is not None:
        return -inf, higher
    if open_side == "upper" and lower is not None:
        return lower, inf
    if open_side is None and lower is not None and higher is not None:
        return lower, higher
    return None


def _headline_endpoint(value: Any, row: Mapping[str, Any]) -> str:
    if value == -inf:
        return "−∞"
    if value == inf:
        return "+∞"
    return _headline_number(value, row)


def _headline_interval(row: Mapping[str, Any]) -> str:
    set_text = _confidence_set_text(row)
    if set_text is not None:
        return set_text
    endpoints = _interval_endpoints(row)
    if endpoints is None:
        return f"[{_missing('no interval available')}]"
    lower, higher = endpoints
    opening = "(" if lower == -inf else "["
    closing = ")" if higher == inf else "]"
    return f"{opening}{_headline_endpoint(lower, row)}, {_headline_endpoint(higher, row)}{closing}"


def _primary_method(row: Mapping[str, Any]) -> str:
    level = row.get("level")
    if level is None:
        level_label = ""
    else:
        percent = f"{float(level) * 100:.2f}".rstrip("0").rstrip(".")
        level_label = f"{percent}% "
    inference = _inference_word(str(row.get("inference", "fixed")))
    return f"{level_label}{_esc(inference)} interval · {_esc(_tail_word(row))}"


def _effect(row: Mapping[str, Any]) -> str:
    """The point estimate on its own scale, or a missing marker with its reason."""
    lift = row.get("lift")
    if _is_missing(lift):
        reason = (
            "point estimate unavailable"
            if _confidence_set_text(row) is not None
            else "no estimate available"
        )
        return _missing(str(row.get("note") or reason))
    if row.get("value_scale") == "relative":
        return f"{lift:+.1%}"
    return f"{lift:+,.4g}"


def _interval(row: Mapping[str, Any]) -> str:
    set_text = _confidence_set_text(row)
    if set_text is not None:
        return set_text
    endpoints = _interval_endpoints(row)
    if endpoints is None:
        return _missing("no interval available")
    lower, higher = endpoints
    return f"{_headline_endpoint(lower, row)} to {_headline_endpoint(higher, row)}"


def _effect_detail(row: Mapping[str, Any]) -> str:
    level = row.get("level")
    level_label = "Interval" if level is None else f"{level:.1%} interval"
    parts = [
        f"{level_label} {_interval(row)}",
        f"{_inference_word(str(row.get('inference', 'fixed')))} · {_tail_word(row)}",
    ]
    return " · ".join(parts)


def _inference_word(inference: str) -> str:
    return {
        "always_valid": "always-valid",
        "asymptotic_mean": "asymptotic sequential",
        "fixed": "fixed-horizon",
    }.get(inference, inference)


def _tail_word(row: Mapping[str, Any]) -> str:
    alternative = str(row.get("alternative", "two-sided"))
    return {
        "two-sided": "two-sided test",
        "greater": "one-sided test for an increase",
        "less": "one-sided test for a decrease",
    }.get(alternative, f"{alternative} test")


def _population_label(snapshot: DashboardSnapshot) -> str:
    populations = sorted(
        {str(row.get("analysis_population", "assigned")) for row in snapshot.readout_rows}
    )
    return _esc(", ".join(f"{name} population" for name in populations) or "assigned population")


def _inference_label(snapshot: DashboardSnapshot) -> str:
    kinds = sorted({str(row.get("inference", "fixed")) for row in snapshot.readout_rows})
    return _esc(", ".join(_inference_word(kind) for kind in kinds) or "fixed-horizon")


def _timestamp(value: dt.datetime) -> str:
    return _esc(value.strftime("%Y-%m-%d %H:%M UTC"))


# Health


def render_health(snapshot: DashboardSnapshot) -> mo.Html:
    """The allocation and caveat section, shown above the results."""
    return _section(
        "health",
        "Experiment health",
        _allocation_block(snapshot) + _flags_block(snapshot) + _caveats_block(snapshot),
        subtitle="",
    )


def _allocation_verdict(allocation: SRMResult) -> str:
    if allocation.is_srm:
        return "Sample ratio mismatch detected"
    return "No allocation issues detected"


def _allocation_block(snapshot: DashboardSnapshot) -> str:
    allocation = snapshot.allocation
    if allocation is None:
        code, reason = snapshot.allocation_refusal or ("", "")
        return (
            _status("warn", "Allocation check unavailable; this is not a passing check.")
            + f'<p class="inc-dashboard-code">{_esc(code)}</p>'
            + f'<p class="inc-dashboard-reason">{_esc(reason)}</p>'
        )
    enrolled = sum(allocation.observed.values())
    verdict = _allocation_verdict(allocation)
    return (
        _status("bad" if allocation.is_srm else "ok", verdict)
        + _allocation_table(snapshot, allocation, enrolled)
        + _disclosure(
            "Allocation over time and evidence",
            _allocation_history_table(snapshot)
            + f'<p class="inc-dashboard-note">{_allocation_evidence(allocation)}</p>',
        )
    )


def _allocation_history_table(snapshot: DashboardSnapshot) -> str:
    if snapshot.allocation_history_refusal is not None:
        code, reason = snapshot.allocation_history_refusal
        return _status("warn", "Allocation history unavailable.") + _kv(
            [("Code", _esc(code)), ("Reason", _esc(reason))]
        )
    if not snapshot.allocation_history:
        return f'<p class="inc-dashboard-note">{_missing("no enrollment history")}</p>'
    allocation = snapshot.allocation
    assert allocation is not None
    frame = pd.DataFrame([dict(row) for row in snapshot.allocation_history]).rename(
        columns={"group_id": "Variant"}
    )
    totals = frame.groupby("ds")["n_cumulative"].transform("sum").astype(float)
    n = totals.where(totals > 0)
    share = frame["n_cumulative"] / n
    frame["share"] = share
    z = norm.isf(0.025)
    denominator = 1 + z * z / n
    center = (share + z * z / (2 * n)) / denominator
    half = z * (share * (1 - share) / n + (z / (2 * n)) ** 2) ** 0.5 / denominator
    frame["share_lower"] = (center - half).clip(0, 1)
    frame["share_upper"] = (center + half).clip(0, 1)
    arms = [snapshot.control_group, snapshot.treatment_group]
    latest = (
        frame.sort_values("ds")
        .drop_duplicates("Variant", keep="last")
        .set_index("Variant")
        .reindex(arms)
        .reset_index()
    )
    total_weight = sum(snapshot.config.expected_allocation.values())
    latest["target"] = [snapshot.config.expected_allocation[arm] / total_weight for arm in arms]
    table = (
        ct.CoefTable(latest, rows="Variant")
        .estimate(_allocation_count_label(allocation), "n_cumulative", fmt=_count)
        .estimate("Share", "share", ci=("share_lower", "share_upper"), fmt=_share)
        .estimate("Target", "target", fmt=_share)
        .sparkline(
            "Cumulative allocation",
            value="share",
            ci=("share_lower", "share_upper"),
            x="ds",
            data=frame,
            ref=None,
            annotations=[ct.Rule(at="target", axis="y", color=DASHBOARD_THEME.muted)],
            scale="table",
            width=420,
            height=76,
            axis_fmt=ct.DateAxis(),
            fmt=_share,
        )
        .with_theme(DASHBOARD_THEME)
    )
    return (
        f'<div class="inc-dashboard-table-wrap">{table.as_raw_html()}</div>'
        '<p class="inc-dashboard-note">Cumulative enrolled share by enrollment date. '
        "Dashed lines show each variant's target; shaded bands show pointwise 95% Wilson "
        "intervals. These bands are not corrected for repeated looks and are not sequential "
        "SRM thresholds.</p>"
    )


def _allocation_evidence(allocation: SRMResult) -> str:
    """Which statistic decided the check, and at which level.

    The allocation level is stated here and nowhere near a result interval:
    they answer different questions.
    """
    alpha = f"α = {_number(allocation.alpha, 3)}"
    if allocation.inference == "always_valid":
        if allocation.log_e_value is None:
            return f"Always-valid evidence, e-value unavailable, at {alpha}."
        return f"Always-valid evidence (log e-value {_number(allocation.log_e_value)}) at {alpha}."
    return (
        f"Fixed-horizon chi-square evidence (p = {_number(allocation.fixed_p_value)}) at {alpha}."
    )


def _allocation_table(snapshot: DashboardSnapshot, allocation: SRMResult, enrolled: int) -> str:
    target = snapshot.config.expected_allocation
    total_weight = sum(target.values())
    rows = []
    for arm in (snapshot.control_group, snapshot.treatment_group):
        units = allocation.observed.get(arm, 0)
        observed_share = _share(units / enrolled) if enrolled else _missing("no enrolled units")
        weight = target.get(arm)
        expected_share = (
            _share(weight / total_weight) if weight is not None else _missing("no target weight")
        )
        role = "control" if arm == snapshot.control_group else "treatment"
        meter = ""
        if enrolled and weight is not None:
            meter = (
                '<span class="inc-dashboard-meter" aria-hidden="true">'
                f'<span class="inc-dashboard-meter-fill inc-dashboard-meter-fill--{role}" '
                f'style="width:{units / enrolled:.6%}"></span>'
                f'<span class="inc-dashboard-meter-target" '
                f'style="left:{weight / total_weight:.6%}"></span></span>'
            )
        rows.append(
            [
                f"{_esc(arm)} <span class='inc-dashboard-tag'>{role}</span>",
                _count(units),
                meter + observed_share,
                expected_share,
            ]
        )
    return _table(
        ["Arm", _allocation_count_label(allocation), "Observed share", "Target share"], rows
    )


def _flags_block(snapshot: DashboardSnapshot) -> str:
    allocation = snapshot.allocation
    if allocation is None:
        return ""
    flags: list[str] = []
    if allocation.unassigned_units:
        flags.append(
            f"{_count(allocation.unassigned_units)} units are not assigned to any arm. "
            "They are excluded from the arm counts above."
        )
    if allocation.mixed_assignment_units:
        flags.append(
            f"{_count(allocation.mixed_assignment_units)} units appear in more than one arm. "
            "Every arm count above is reduced by the mixed units it contained."
        )
    if allocation.low_expected_count:
        smallest = (
            _number(allocation.min_expected_count)
            if allocation.min_expected_count is not None
            else "unknown"
        )
        flags.append(
            f"The smallest expected arm count is {smallest}. A small expected count is a "
            "separate caution: an undetected mismatch does not make the split balanced."
        )
    if not flags:
        return ""
    return '<h3 class="inc-dashboard-subheading">Assignment warnings</h3>' + _list(
        [_esc(flag) for flag in flags], css_class="inc-dashboard-flags"
    )


def _caveats_block(snapshot: DashboardSnapshot) -> str:
    """Keep actual caveats visible without an empty-state paragraph."""
    caveats = _result_caveats(snapshot.readout_rows)
    return _caveats_list(caveats)


def _caveats_list(caveats: Sequence[str]) -> str:
    """A caveat list beside a table; nothing at all when there are none."""
    if not caveats:
        return ""
    return '<h3 class="inc-dashboard-subheading">Result caveats</h3>' + _list(
        caveats, css_class="inc-dashboard-caveats"
    )


def _result_caveats(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """Keep row caveats and unavailable evidence visible without hiding usable sets."""
    caveats: list[str] = []
    for row in rows:
        metric = _esc(row.get("metric", "unknown metric"))
        for label, value in (
            ("note", row.get("note")),
            ("excluded", row.get("excluded")),
            ("unavailable", row.get("unavailable")),
        ):
            if value:
                caveats.append(f"<strong>{metric}</strong>: {label}: {_esc(value)}")
        if row.get("low_reliability"):
            caveats.append(f"<strong>{metric}</strong>: flagged low reliability.")
        relative = row.get("relative_confidence_set")
        winsor = row.get("confidence_set")
        unavailable = row.get("relative_unavailable_reason")
        if not _is_missing(unavailable):
            caveats.append(f"<strong>{metric}</strong>: unavailable: {_esc(str(unavailable))}")
        elif relative is not None and not _is_missing(relative):
            if relative.reason or relative.geometry in ("unavailable", "empty"):
                caveats.append(
                    f"<strong>{metric}</strong>: {relative.geometry}: "
                    f"{_esc(relative.reason or 'confidence set has no members')}"
                )
        elif winsor is not None and not _is_missing(winsor):
            interval = (
                winsor.relative
                if row.get("value_scale", "relative") == "relative"
                else winsor.additive
            )
            for reason in dict.fromkeys((interval.lower.reason, interval.upper.reason)):
                if reason:
                    caveats.append(f"<strong>{metric}</strong>: confidence set: {_esc(reason)}")
        elif not _is_missing(row.get("binomial_set")):
            continue
        elif _is_missing(row.get("lift")):
            caveats.append(f"<strong>{metric}</strong>: {_missing('no estimate available')}")
        elif _interval_endpoints(row) is None:
            caveats.append(f"<strong>{metric}</strong>: no interval available for this estimate.")
    return caveats


def _format_group_value(value: Any, unit: str, *, count: bool = False) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    if count:
        return _count(value)
    if unit == "%":
        return f"{float(value):.1%}"
    if unit == "USD":
        return f"${float(value):,.2f}"
    if unit == "ms":
        return f"{float(value):,.2f} ms"
    return f"{float(value):,.4g}"


def _group_data_table(snapshot: DashboardSnapshot, metric: str) -> str:
    """Render the immutable aggregate rows captured with this snapshot."""
    rows = group_data_rows(snapshot, metric=metric)
    if not rows:
        return _missing("no group aggregates available")
    unit = str(rows[0].get("unit", "value"))
    model = require_metric(snapshot, metric)
    headers = ["Arm", "Eligible units", f"Observed value ({unit})"]
    keys: list[tuple[str, str, bool]] = [("Assigned units", "assigned_units", True)]
    if model.type in ("conversion", "retention"):
        keys.append(("Retained / converted units", "retained_units", True))
    if model.type in ("mean", "quantile"):
        label = (
            "Sum of per-unit event averages"
            if getattr(model, "aggregation", None) == "avg_event"
            else "Sum of unit aggregates"
        )
        keys.append((label, "sum_value", False))
    keys.append(("Qualifying events", "event_count", True))
    if model.type == "ratio":
        label = "Ratio numerator total"
        if model.numerator.aggregation == "avg_event":
            label += " (sum of per-unit event averages)"
        keys.append((label, "numerator", False))
    if model.type == "ratio":
        label = "Ratio denominator total"
        if model.denominator.aggregation == "avg_event":
            label += " (sum of per-unit event averages)"
        keys.append((label, "denominator", False))
    if any(row.get("analysis_input_value") is not None for row in rows):
        keys.append(("Analysis input (transformed)", "analysis_input_value", False))
    keys.extend(
        [
            ("Excluded: not mature", "excluded_not_mature", True),
            ("Excluded: no observed day", "excluded_no_observed_day", True),
            ("Excluded: other", "excluded_other", True),
            ("Observation cutoff", "observation_end", False),
        ]
    )
    if any(row.get("window_start_days") is not None for row in rows):
        keys.extend(
            [
                ("Window start (days)", "window_start_days", True),
                ("Window end (days)", "window_end_days", True),
            ]
        )
    keys.append(("Evidence source", "source_kind", False))
    if any(row.get("source_kind") == "retained_checkpoint" for row in rows):
        keys.append(("Retained checkpoint", "prefix_id", False))
    headers.extend(label for label, _, _ in keys)
    reason_labels = {
        "eligible_units": "Eligible units",
        "observed_value": headers[2],
        **{key: label for label, key, _ in keys},
    }
    has_unavailable = any(
        key in reason_labels for row in rows for key in (row.get("unavailable") or {})
    )
    if has_unavailable:
        headers.append("Unavailable / not applicable")
    body: list[list[str]] = []
    for row in rows:
        cells = [
            str(row.get("group_id", "")),
            _format_group_value(row.get("eligible_units"), "count", count=True),
            _format_group_value(row.get("observed_value"), unit),
        ]
        for _, key, is_count in keys:
            value = row.get(key)
            if key == "observation_end" and value is not None:
                value = value.isoformat() if hasattr(value, "isoformat") else str(value)
            if key == "source_kind":
                value = "pinned warehouse" if value == "pinned_warehouse" else "retained checkpoint"
            cells.append(
                _format_group_value(value, "count", count=True)
                if is_count
                else ("" if value is None else str(value))
            )
        if has_unavailable:
            unavailable = row.get("unavailable") or {}
            cells.append(
                "; ".join(
                    f"{reason_labels[key]}: {value}"
                    for key, value in unavailable.items()
                    if key in reason_labels
                )
            )
        escaped_cells = [_esc(cell) for cell in cells]
        if has_unavailable:
            escaped_cells[-1] = (
                f'<span class="inc-dashboard-unavailable-reasons">{escaped_cells[-1]}</span>'
            )
        body.append(escaped_cells)
    return _table(headers, body)


_RELATIVE_SET_EXPLANATIONS = {
    "one_sided": (
        "The relative confidence set is open on one side. Any separately reported central "
        "interval is distinct from that set."
    ),
    "disconnected": (
        "The relative confidence set contains separated ranges. Values in the gap are excluded; "
        "joining the ranges would misrepresent the evidence."
    ),
    "all_real": "The relative confidence set includes the whole real line, without a finite bound.",
    "empty": "The retained relative confidence set is empty.",
    "unavailable": "The retained relative confidence set is unavailable.",
}


def _geometry_explanations(snapshot: DashboardSnapshot, row: Mapping[str, Any]) -> list[str]:
    """Describe retained confidence-set geometry without changing its evidence."""
    explanations: list[str] = []
    retained_set = _format_confidence_set(
        row.get("confidence_set"),
        relative=row.get("relative_confidence_set"),
        binomial=row.get("binomial_set"),
        unavailable=row.get("relative_unavailable_reason"),
        scale=str(row.get("value_scale") or "relative"),
        lift=row.get("lift"),
    )
    if retained_set:
        explanations.append(f"Retained confidence set: {retained_set}.")
    alternative = row.get("alternative")
    open_side = row.get("open_side")
    binomial = row.get("binomial_set")
    relative = row.get("relative_confidence_set")
    if (
        _is_missing(open_side)
        and binomial is not None
        and not _is_missing(binomial)
        and binomial.geometry == "lower_bound"
    ):
        open_side = "upper"
    if (
        _is_missing(open_side)
        and relative is not None
        and not _is_missing(relative)
        and relative.geometry == "one_sided"
    ):
        open_side = "lower" if relative.intervals[0][0] is None else "upper"
    if binomial is not None and not _is_missing(binomial) and binomial.geometry == "upper_bound":
        explanations.append(
            "This is a one-sided binary test for a decrease. Its lower endpoint is the physical "
            "−100% relative-lift floor, not an open lower side."
        )
    elif alternative in ("greater", "less") and open_side in ("upper", "lower"):
        bounded = "lower" if alternative == "greater" else "upper"
        opposite = "upper" if alternative == "greater" else "lower"
        explanations.append(
            f"This is a declared {_tail_word(row)}: it bounds the {bounded} direction; "
            f"the {opposite} direction is intentionally unconstrained by this test."
        )
    if binomial is not None and not _is_missing(binomial) and binomial.x_c == 0:
        explanations.append(
            f"The control arm has 0/{_count(binomial.n_c)} retained/converted units, versus "
            f"{_count(binomial.x_t)}/{_count(binomial.n_t)} in {row.get('group_id', snapshot.treatment_group)}. "
            "Its zero control count supplies no finite upper relative-effect bound."
        )
    if relative is not None and not _is_missing(relative):
        description = _RELATIVE_SET_EXPLANATIONS.get(relative.geometry)
        if description:
            explanations.append(
                description + (f" Reported reason: {relative.reason}." if relative.reason else "")
            )
    if not _is_missing(row.get("relative_unavailable_reason")):
        explanations.append(
            f"Relative evidence is unavailable: {row['relative_unavailable_reason']}."
        )
    if (
        _is_missing(row.get("lift"))
        or not _is_missing(row.get("relative_unavailable_reason"))
        or (
            relative is not None
            and not _is_missing(relative)
            and relative.geometry == "unavailable"
        )
    ):
        absolute = [
            (label, row[field])
            for label, field in (
                ("point", "abs_diff"),
                ("lower bound", "abs_lb"),
                ("upper bound", "abs_ub"),
            )
            if not _is_missing(row.get(field)) and isfinite(row[field])
        ]
        if absolute:
            groups = group_data_rows(snapshot, metric=str(row["metric"]))
            unit = str(groups[0]["unit"]) if groups else "metric units"
            scale = 100 if unit == "%" else 1
            unit = "percentage points" if unit == "%" else unit
            values = ", ".join(f"{label} {value * scale:.6g}" for label, value in absolute)
            explanations.append(f"Absolute difference ({unit}): {values}.")
    model = require_metric(snapshot, str(row["metric"]))
    if model.type in ("conversion", "retention"):
        explanations.append(
            "Observed rates are bounded by 0–100%. Relative lift is a different quantity: "
            "its upper bound can be infinite, and its confidence set need not be one interval."
        )
    return explanations


# Results


_ROLE_LABELS = {
    "primary": "Primary",
    "secondary": "Secondaries",
    "guardrail": "Guardrails",
    "unassigned": "Unassigned",
}


def _display_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Omit the family selection column only from the visual readout."""
    return [{key: value for key, value in row.items() if key != "discovery"} for row in rows]


# Numeric confidence-set evidence lives in interval disclosures and the per-level wording beside
# each table, so the verdict, set and per-row level columns are not repeated in the grid. The
# unrounded values stay in the readout frame and CSV.
_REDUNDANT_COLUMNS = frozenset({"Significant", "Confidence set", "Interval"})


def _levels_by_metric(pairs: Iterable[tuple[Any, float]], fmt: Callable[[float], str]) -> str:
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


INTERVALS_DISCLOSURE = "How to read these intervals"
_TABLE_WRAP = '<div class="inc-dashboard-table-wrap">{}</div>'


def _drop_redundant_columns(table: ct.CoefTable) -> None:
    table.columns = tuple(
        column
        for column in table.columns
        if getattr(column, "label", None) not in _REDUNDANT_COLUMNS
    )


def _geometry_disclosure(
    snapshot: DashboardSnapshot, rows: Iterable[Mapping[str, Any]], *, label: str
) -> str:
    geometry = _list(
        [
            f"<strong>{_esc(row.get('metric'))}</strong>: {_esc(text)}"
            for row in rows
            for text in _geometry_explanations(snapshot, row)
        ],
        css_class="inc-dashboard-reasons",
    )
    return _disclosure(label, geometry) if geometry else ""


def results_table(snapshot: DashboardSnapshot) -> str:
    """The whole declared family as one native CoefTable, without redundant columns."""
    table = readout_table(
        _display_rows(snapshot.readout_rows),
        title="",
        subtitle=f"{snapshot.treatment_group} vs {snapshot.control_group}",
        theme=DASHBOARD_THEME,
        nest_by="arm",
        advisory=False,
        show_interval_level=True,
    )
    _drop_redundant_columns(table)
    table.columns = tuple(
        replace(column, width=300, height=42) if isinstance(column, ct.Forest) else column
        for column in table.columns
    )
    return _TABLE_WRAP.format(table.as_raw_html())


def results_notes(snapshot: DashboardSnapshot) -> str:
    """Interval reading guide and the declared inference per role."""
    return _geometry_disclosure(
        snapshot, snapshot.readout_rows, label=INTERVALS_DISCLOSURE
    ) + _disclosure(
        "Statistical interpretation",
        _list(_role_disclosures(snapshot), css_class="inc-dashboard-reasons"),
    )


def render_results(snapshot: DashboardSnapshot) -> mo.Html:
    """The whole declared family, its readout table, and its disclosures."""
    body = (
        results_table(snapshot)
        + _adverse_block(snapshot)
        + results_notes(snapshot)
        + _caveats_list(_result_caveats(snapshot.readout_rows))
    )
    return _section("results", "Results", body)


def _is_adverse(row: Mapping[str, Any]) -> bool:
    """True when the whole interval sits on the unfavorable side of the null.

    Independent of ``stat_sig``: a one-sided improvement test that did not
    reject says nothing about an interval lying wholly in the wrong direction.
    """
    direction = row.get("preferred_direction")
    null = row.get("null_abs")
    if null is not None:
        lower, higher = row.get("abs_lb"), row.get("abs_ub")
    else:
        null = row.get("null_lift") or 0.0
        lower, higher = row.get("lower"), row.get("higher")
    if direction == "increase" and higher is not None:
        return higher < null
    if direction == "decrease" and lower is not None:
        return lower > null
    return False


def _adverse_block(snapshot: DashboardSnapshot) -> str:
    adverse = [row for row in decision_rows(snapshot) if _is_adverse(row)]
    if not adverse:
        return ""
    named = ", ".join(str(row.get("metric")) for row in adverse)
    return _status(
        "bad",
        f"Unfavorable evidence: the whole interval is on the adverse side for {named}. "
        "This holds whether or not the declared test rejected.",
    )


def _role_disclosures(snapshot: DashboardSnapshot) -> list[str]:
    rows_by_role: dict[str, list[Mapping[str, Any]]] = {}
    for row in snapshot.readout_rows:
        rows_by_role.setdefault(str(row.get("role") or "unassigned"), []).append(row)
    items = []
    for role in ("primary", "secondary", "guardrail", "unassigned"):
        rows = rows_by_role.get(role)
        if rows:
            items.append(_role_disclosure(role, rows))
    return items


def _role_disclosure(role: str, rows: Sequence[Mapping[str, Any]]) -> str:
    label = _ROLE_LABELS.get(role, role)
    tails = "; ".join(f"{_esc(row.get('metric'))} {_tail_word(row)}" for row in rows)
    sentences = [f"<strong>{_esc(label)}</strong>: {tails}."]
    several_methods = len({(row.get("method"), row.get("method_role")) for row in rows}) > 1
    levelled = [
        (
            f"{row.get('metric')} ({row.get('method')}, {row.get('method_role')})"
            if several_methods
            else row.get("metric"),
            float(row["level"]),
        )
        for row in rows
        if row.get("level") is not None
    ]
    if levelled:
        shown = _esc(_levels_by_metric(levelled, lambda level: f"{level:.1%}"))
        sentences.append(f"Interval level {shown}.")
        if any(row.get("alternative") in ("greater", "less") for row in rows):
            sentences.append(
                "A one-sided test and a two-sided interval at a different level "
                "are not the same statement."
            )
    if role == "secondary":
        sentences.append(_discovery_sentence(rows))
    if role == "guardrail":
        sentences.append(
            "A guardrail that did not reject means the declared test did not reject. "
            "It is not evidence of no harm, and not a two-sided conclusion."
        )
    return " ".join(sentences)


def _discovery_sentence(rows: Sequence[Mapping[str, Any]]) -> str:
    axes = next((row.get("family_axes") for row in rows if row.get("family_axes")), None)
    q = next((row.get("family_q") for row in rows if row.get("family_q") is not None), None)
    if axes is None and q is None:
        return "No discovery family was applied to these rows."
    parts = []
    if axes:
        parts.append(f"family axes {_esc(', '.join(str(axis) for axis in axes))}")
    if q is not None:
        parts.append(f"q = {_number(float(q), 3)}")
    return (
        f"Family selection ({'; '.join(parts)}) is distinct from the row's tested-alternative "
        "verdict. Its metadata is retained in metric details and CSV."
    )


def render_details(snapshot: DashboardSnapshot) -> mo.Html:
    """Declared policy and optional provenance, with no source access."""
    entries = [
        ("Experiment", _esc(snapshot.experiment_name)),
        ("Control arm", _esc(snapshot.control_group)),
        ("Treatment arm", _esc(snapshot.treatment_group)),
        ("Declared window", f"{_date(snapshot.start)} → {_date(snapshot.end)}"),
        ("Primary metric", _esc(snapshot.primary_metric)),
        ("Computed", _esc(snapshot.computed_at.isoformat())),
        (
            "Declared breakouts",
            _esc(
                ", ".join(
                    f"{dimension} ({source or 'source resolved by the readout'})"
                    for source, dimension in snapshot.breakouts
                )
                or "No declared breakouts"
            ),
        ),
    ]
    if snapshot.config.source_label:
        entries.append(("Source", _esc(snapshot.config.source_label)))
    entries.extend((label, _esc(value)) for label, value in snapshot.config.provenance.items())
    policies = []
    for model in snapshot.metrics:
        row = row_for_metric(snapshot, model.name)
        if row is not None:
            policies.append(
                f"<details><summary>{_esc(model.name)}</summary>"
                + _kv(_metric_detail_entries(snapshot, model, row))
                + "</details>"
            )
    return _section(
        "details",
        "Details",
        _disclosure("Experiment and provenance", _kv(entries))
        + _disclosure("Metric policies", "".join(policies))
        + _disclosure(
            "About this snapshot",
            '<p class="inc-dashboard-note">Computed is the analysis timestamp, not a data '
            "freshness guarantee. Data by group shows each metric's captured arm values, "
            "eligible counts, observation window and evidence source.</p>"
            '<p class="inc-dashboard-note">The CSV contains all unrounded headline rows, '
            "including tested alternatives and unavailable values. Static HTML captures the "
            "rendered evidence; prepared dashboard tabs switch among captured views without "
            "rerunning the analysis.</p>",
        ),
        subtitle="",
    )


# Metric details


def render_metric_details(snapshot: DashboardSnapshot, *, metric: str) -> mo.Html:
    """Definition, tested tail, policy, and observed group evidence for one metric."""
    model = require_metric(snapshot, metric)
    estimate = estimate_for_metric(snapshot, metric)
    row = row_for_metric(snapshot, metric)
    group_body = _group_data_disclosure(snapshot, metric)
    if estimate is None or row is None:
        body = (
            f'<p class="inc-dashboard-note">{_missing("no decision result for this metric")}</p>'
            + group_body
        )
        return _section("metric-details", f"Metric: {metric}", body)
    return _section(
        "metric-details",
        f"Metric: {model.name}",
        _chips(
            [
                ("Lift", _effect(row)),
                ("Interval", _interval(row)),
                ("Favorable", _esc(row.get("preferred_direction") or "not declared")),
            ]
        )
        + _disclosure(
            "Definition and analysis policy", _kv(_metric_detail_entries(snapshot, model, row))
        )
        + _geometry_disclosure(snapshot, [row], label=INTERVALS_DISCLOSURE)
        + group_body
        + _list(_result_caveats([row]), css_class="inc-dashboard-caveats"),
        subtitle=str(getattr(model, "description", "") or ""),
    )


def _group_data_disclosure(snapshot: DashboardSnapshot, metric: str) -> str:
    """Captured arm evidence for one metric's inspection view."""
    return _disclosure(
        "Data by group",
        _group_data_table(snapshot, metric)
        + '<p class="inc-dashboard-note">Observed values are pre-adjustment aggregates. '
        "Eligibility uses the same cohort and metric window as the analysis. For fixed-horizon "
        "retention, assigned = eligible + not mature + no observed day + other exclusions; "
        "the observation cutoff and window endpoints are shown when captured. "
        "CUPED, a prior, or winsorization can make the reported effect differ from these raw values. "
        "Only retained transformed inputs are shown in the separate analysis-input column; "
        "pre-transform outcomes absent from a checkpoint remain unavailable.</p>",
    )


def _metric_detail_entries(
    snapshot: DashboardSnapshot, model: Any, row: Mapping[str, Any]
) -> list[tuple[str, str]]:
    declared = getattr(model, "declared_preferred_direction", None)
    direction = row.get("preferred_direction")
    direction_text = _esc(direction or "not declared")
    if declared is None and direction is not None:
        direction_text += ' <span class="inc-dashboard-reason">(library default)</span>'
    entries: list[tuple[str, str]] = [
        (
            "Role",
            _esc(_ROLE_LABELS.get(str(row.get("role")), str(row.get("role") or "unassigned"))),
        ),
        ("Relative lift", _effect(row)),
        (
            "Interval",
            f"{_interval(row)}"
            + (f" at {row['level']:.1%}" if row.get("level") is not None else ""),
        ),
        ("Favorable direction", direction_text),
        ("Tested alternative", f"{_esc(row.get('alternative'))} ({_tail_word(row)})"),
        ("Null boundary", _null_text(row)),
        ("Inference", _esc(_inference_word(str(row.get("inference", "fixed"))))),
        ("Analysis population", _esc(row.get("analysis_population", "assigned"))),
        ("Decision method", _esc(row.get("method", "unknown"))),
        ("Estimand", _esc(row.get("estimand", "itt"))),
        ("Measurement window", _metric_window(model)),
        ("Arms", f"{_esc(snapshot.control_group)} → {_esc(snapshot.treatment_group)}"),
    ]
    entries.extend(_family_entries(row))
    return entries


def _null_text(row: Mapping[str, Any]) -> str:
    null_abs = row.get("null_abs")
    if null_abs is not None:
        return f"{null_abs:+,.4g} absolute"
    null_lift = row.get("null_lift")
    if null_lift is None:
        return _missing("no null boundary reported")
    return f"{float(null_lift):+.1%} relative"


def _family_entries(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    discovery = row.get("discovery")
    if discovery is not None:
        verdict = "in the discovery set" if discovery else "not in the discovery set"
        entries.append(("Family discovery", _esc(verdict)))
    axes = row.get("family_axes")
    if axes:
        entries.append(("Family axes", _esc(", ".join(str(axis) for axis in axes))))
    for label, key in (("Family q", "family_q"), ("Family threshold", "family_threshold")):
        value = row.get(key)
        if value is not None:
            entries.append((label, _number(float(value), 4)))
    return entries


def _metric_window(model: Any) -> str:
    """The declared measurement window, in the metric's own terms."""
    threshold = getattr(model, "threshold_days", None)
    if threshold is not None:
        if isinstance(threshold, int):
            return f"retention from day {threshold} onward"
        start, end = threshold
        return f"retention band days {start} to {end - 1}"
    window = getattr(model, "window_days", None)
    if window is not None:
        return f"{window} days post-exposure"
    parts = [
        f"{label} {getattr(part, 'window_days', None) or 'unbounded'}"
        for label, part in (
            ("numerator", getattr(model, "numerator", None)),
            ("denominator", getattr(model, "denominator", None)),
        )
        if part is not None
    ]
    if parts:
        return _esc(" / ".join(f"{part} days" for part in parts))
    return _missing("no declared window")


def _kv(entries: Sequence[tuple[str, str]]) -> str:
    rows = "".join(f"<dt>{_esc(label)}</dt><dd>{value}</dd>" for label, value in entries)
    return f'<dl class="inc-dashboard-kv">{rows}</dl>'


# Explore: one requested view at a time.


_VIEW_TITLES: dict[str, str] = {
    "cumulative_lift": "Cumulative lift",
    "daily_values": "Daily metric values",
    "cumulative_values": "Cumulative metric values",
    "segments": "Segments",
}


def render_explore(
    snapshot: DashboardSnapshot,
    data: DailyLiftEstimates | DailyMetricValues | BreakoutEstimates,
    *,
    metric: str | None,
    view: ExploreView,
    completed_windows_only: bool = False,
) -> mo.Html:
    """The requested CoefTable view, with context and unavailable-point reasons."""
    if metric is not None:
        require_metric(snapshot, metric)
    if view == "segments":
        body = _segments_body(snapshot, data, metric=metric)
    else:
        body = _temporal_body(
            snapshot,
            data,
            metric=metric,
            view=view,
            completed_windows_only=completed_windows_only,
        )
    rows = (
        snapshot.readout_rows
        if metric is None
        else tuple(row for row in snapshot.readout_rows if row.get("metric") == metric)
    )
    headline = _geometry_disclosure(snapshot, rows, label=f"{INTERVALS_DISCLOSURE} (headline)")
    body = headline + body
    return _section(
        "explore",
        f"Explore: {_VIEW_TITLES.get(view, view)}",
        body,
    )


def _maturity_label(*, completed_windows_only: bool) -> str:
    return "Completed windows only" if completed_windows_only else "Provisional monitoring"


def _monitoring_sentence(rows: Sequence[Any], *, view: str) -> str:
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


def temporal_figure(
    snapshot: DashboardSnapshot, data: Any, *, metric: str | None, view: str
) -> tuple[str, str]:
    """The native table(s) for a non-empty temporal view, then its unavailable-point block.

    A metric whose points mix calendar and cohort dates cannot share one axis, so it gets a
    warning in place of a chart.
    """
    frame = data.to_frame()
    if (
        not frame["ds_basis"].dropna().nunique()
        or (frame.groupby("metric")["ds_basis"].nunique() != 1).any()
    ):
        return (
            _status(
                "warn",
                "A metric mixes calendar and cohort date bases, so its points cannot share one axis.",
            ),
            "",
        )
    return (
        _temporal_table(snapshot, data, metric=metric, view=view),
        _gaps_block(frame, view=view),
    )


def _temporal_body(
    snapshot: DashboardSnapshot,
    data: Any,
    *,
    metric: str | None,
    view: str,
    completed_windows_only: bool,
) -> str:
    rows = list(data)
    if not rows:
        return f'<p class="inc-dashboard-note">{_missing("this view returned no points")}</p>'
    frame = data.to_frame()
    bases = sorted({str(basis) for basis in frame["ds_basis"].dropna().unique()})
    caption = _temporal_caption(
        snapshot,
        frame,
        bases=bases,
        view=view,
        completed_windows_only=completed_windows_only,
        monitoring=_monitoring_sentence(rows, view=view),
    )
    figure, gaps = temporal_figure(snapshot, data, metric=metric, view=view)
    return figure + caption + gaps


def _temporal_caption(
    snapshot: DashboardSnapshot,
    frame: Any,
    *,
    bases: Sequence[str],
    view: str,
    completed_windows_only: bool,
    monitoring: str,
) -> str:
    dates = frame["ds"].dropna()
    observed = f"{_date(dates.min())} → {_date(dates.max())}" if len(dates) else "no dated points"
    basis_label = ", ".join(_axis_title(basis) for basis in bases) or "unknown basis"
    entries = [
        ("Series covers", _esc(observed)),
        ("Date basis", _esc(basis_label)),
        ("Points", _count(len(frame))),
    ]
    if "dimension_value" in frame.columns and frame["dimension_value"].notna().any():
        segments = sorted(str(value) for value in frame["dimension_value"].dropna().unique())
        entries.insert(
            0,
            (
                "Broken out by",
                f"{_esc(frame['dimension'].dropna().iloc[0])}: {_esc(', '.join(segments))}",
            ),
        )
    if view != "daily_values":
        entries.append(
            ("Maturity", _esc(_maturity_label(completed_windows_only=completed_windows_only)))
        )
    frozen_note = (
        '<p class="inc-dashboard-note">An as-of point freezes each unit at its last '
        "in-window value, so a flat tail is not evidence of recent observations.</p>"
        if view != "daily_values"
        else ""
    )
    return (
        _chips(entries)
        + f'<p class="inc-dashboard-note">{_esc(monitoring)}</p>'
        + _disclosure(
            "Window and maturity",
            _kv([("Declared window", f"{_date(snapshot.start)} → {_date(snapshot.end)}")])
            + frozen_note,
        )
    )


def _axis_title(basis: str) -> str:
    if basis == "cohort":
        return "cohort date (each unit's own exposure date)"
    return "calendar date (observation date)"


def _metric_value_format(
    snapshot: DashboardSnapshot,
    model: Any,
    *,
    span: float = 1.0,
) -> ct.Number:
    """Keep nearby axis ticks distinct in the metric's displayed units."""
    unit = snapshot.config.metric_units.get(model.name, "")
    rate = model.type in ("conversion", "retention") or unit == "%"
    decimals = 1 if rate else 2
    if 0 < span < inf:
        decimals = max(decimals, ceil(-log10(span) - (2 if rate else 0)) + 1)
    if rate:
        return ct.Percent(scale=100.0, decimals=decimals, signed=False)
    if unit == "USD":
        return ct.Currency(decimals=decimals)
    if unit == "count":
        return ct.Number(decimals=decimals, suffix=" events/unit")
    return ct.Number(decimals=decimals, suffix=f" {unit}" if unit else "")


def _absolute_metric_table(
    snapshot: DashboardSnapshot,
    frame: Any,
    *,
    metric: str,
) -> str:
    """Render one absolute-value metric: both arms with their uncertainty, per segment if any."""
    metric_frame = frame.loc[frame["metric"] == metric].copy()
    if metric_frame.empty:
        return ""
    segmented = (
        "dimension_value" in metric_frame.columns and metric_frame["dimension_value"].notna().any()
    )
    if segmented:
        metric_frame["Segment"] = metric_frame["dimension_value"].astype(str)
        nest = "Segment"
    else:
        metric_frame["Date basis"] = metric_frame["ds_basis"].map(
            {"calendar": "Observation date", "cohort": "Exposure cohort"}
        )
        nest = "Date basis"
    labels = metric_frame[["metric", nest]].drop_duplicates(ignore_index=True)
    model = next(model for model in snapshot.metrics if model.name == metric)
    # Ticks must stay distinct on the tightest plotted segment.
    spans: list[float] = []
    for _, part in metric_frame.groupby(nest):
        pooled = [
            value
            for column in ("value", "lb", "ub")
            for value in part[column]
            if not _is_missing(value) and isfinite(value)
        ]
        if pooled:
            spans.append((max(pooled) - min(pooled)) or abs(max(pooled)) / 10)
    formatter = _metric_value_format(snapshot, model, span=min(spans, default=0.0))
    table = (
        ct.CoefTable(labels, rows="metric", nest=nest)
        .sparkline(
            "Value over time",
            value="value",
            ci=("lb", "ub"),
            x="ds",
            data=metric_frame,
            ref=None,
            scale="row",
            series="group_id",
            show_ribbon=True,
            show_y_axis=True,
            y_axis_fmt=formatter,
            fmt=formatter,
            axis_fmt=ct.DateAxis(),
            width=600,
            height=160,
            series_colors={
                snapshot.control_group: DASHBOARD_THEME.series_palette[0],
                snapshot.treatment_group: DASHBOARD_THEME.series_palette[1],
            },
        )
        .header("", f"{snapshot.control_group} vs {snapshot.treatment_group}")
        .with_theme(DASHBOARD_THEME)
    )
    return _TABLE_WRAP.format(table.as_raw_html())


def _absolute_tables(snapshot: DashboardSnapshot, frame: Any, *, selected: Sequence[str]) -> str:
    available = set(frame["metric"])
    return "".join(
        _absolute_metric_table(snapshot, frame, metric=metric)
        for metric in selected
        if metric in available
    )


def _latest_points(estimates: Sequence[Any]) -> list[Any]:
    """Each series' latest emitted point, in first-appearance order."""
    latest: dict[tuple[Any, ...], Any] = {}
    for estimate in estimates:
        key = (
            estimate.metric,
            estimate.group_id,
            estimate.method,
            estimate.estimand,
            estimate.dimension_value,
        )
        if key not in latest or estimate.ds >= latest[key].ds:
            latest[key] = estimate
    return list(latest.values())


def _plotted_values(estimates: Sequence[Any]) -> list[float]:
    """Every finite estimate and bound a trajectory plots; open sides contribute none."""
    return [
        value
        for estimate in estimates
        if estimate.lift is not None
        for value in (estimate.lift.value, estimate.lift.lb, estimate.lift.ub)
        if value is not None
    ]


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


def _percent_axis(estimates: Sequence[Any]) -> ct.Percent:
    """Tick precision for the typical plotted range, so nearby ticks never repeat."""
    values = _plotted_values(estimates)
    fence = robust_fence(values)
    span = (fence[1] - fence[0]) if fence else max(values, default=0.0) - min(values, default=0.0)
    decimals = min(4, max(0, ceil(-log10(span * 100)) + 1)) if span > 0 else 1
    return ct.Percent(scale=100.0, decimals=decimals, signed=True)


def _with_bound_lines(column: ct.Sparkline) -> ct.Sparkline:
    """Draw each finite interval bound as its own line beside the estimate.

    The native ribbon needs both bounds; an open side has none to draw and nothing is invented
    for it, so finite sides become labelled lines and absent ones simply have no line.
    """
    data = column.data
    assert data is not None
    palette = DASHBOARD_THEME.series_palette
    parts = [data.assign(__series="Estimate", __plot=data[column.value])]
    colors = {"Estimate": palette[1]}
    for label, field, color in (
        ("Lower CI", "lb", palette[0]),
        ("Upper CI", "ub", DASHBOARD_THEME.inconclusive),
    ):
        if data[field].notna().any():
            parts.append(data.assign(__series=label, __plot=data[field]))
            colors[label] = color
    return replace(
        column,
        data=pd.concat(parts, ignore_index=True),
        value="__plot",
        ci=None,
        series="__series",
        series_colors=colors,
    )


def _lift_trajectory_table(
    snapshot: DashboardSnapshot, estimates: Sequence[Any], *, width: int
) -> str:
    """Cumulative lift as native CoefTable trajectories: latest reading beside its history."""
    segmented = any(estimate.dimension is not None for estimate in estimates)
    latest_rows = estimates_to_readout(_latest_points(estimates))
    table = readout_table(
        _display_rows(latest_rows),
        title="",
        subtitle=f"{snapshot.treatment_group} vs {snapshot.control_group}",
        theme=DASHBOARD_THEME,
        nest_by="segment" if segmented else "arm",
        trend=list(estimates),
        trend_label="Cumulative lift",
        advisory=False,
        show_interval_level=True,
    )
    axis = _percent_axis(estimates)
    columns = []
    for column in table.columns:
        if isinstance(column, ct.Forest) or getattr(column, "label", None) in _REDUNDANT_COLUMNS:
            continue
        if isinstance(column, ct.Sparkline):
            column = replace(
                column,
                width=width,
                height=156,
                scale="row",
                show_y_axis=True,
                y_axis_fmt=axis,
                fmt=axis,
                show_endpoint=False,
            )
            if has_open_side(estimates):
                column = _with_bound_lines(column)
        columns.append(column)
    table.columns = tuple(columns)
    return _TABLE_WRAP.format(table.as_raw_html()) + _geometry_disclosure(
        snapshot, latest_rows, label=INTERVALS_DISCLOSURE
    )


def _temporal_table(
    snapshot: DashboardSnapshot, data: Any, *, metric: str | None, view: str
) -> str:
    """Render cumulative lift or independent absolute-value metric tables."""
    if view == "cumulative_lift":
        decisions = [row for row in data if row.method_role == "decision"]
        if not decisions:
            return f'<p class="inc-dashboard-note">{_missing("no decision estimates in this view")}</p>'
        if (
            any(row.dimension is not None for row in decisions)
            and len({row.method for row in decisions}) > 1
        ):
            return "".join(
                _lift_trajectory_table(
                    snapshot, [row for row in decisions if row.metric == name], width=380
                )
                for name in dict.fromkeys(row.metric for row in decisions)
            )
        return _lift_trajectory_table(snapshot, decisions, width=380 if metric is None else 560)
    frame = data.to_frame()
    selected = [model.name for model in snapshot.metrics if metric is None or model.name == metric]
    return _absolute_tables(snapshot, frame, selected=selected)


def _gaps_block(frame: Any, *, view: str) -> str:
    """Why points are missing, and that they are left as gaps."""
    if "unavailable" not in frame.columns:
        return ""
    reasons = frame["unavailable"].dropna()
    if not len(reasons):
        return ""
    counts = reasons.value_counts().to_dict()
    items = [
        f"{_esc(reason)}: {_count(count)} of {_count(len(frame))} points"
        for reason, count in counts.items()
    ]
    heading = "Unavailable points" if view != "segments" else "Excluded segments"
    return (
        f'<h3 class="inc-dashboard-subheading">{_esc(heading)}</h3>'
        + _list(items, css_class="inc-dashboard-reasons")
        + '<p class="inc-dashboard-note">Missing observations remain gaps, not zeros.</p>'
    )


def _segments_body(snapshot: DashboardSnapshot, data: Any, *, metric: str | None) -> str:
    if not snapshot.breakouts:
        return (
            '<p class="inc-dashboard-note">No declared breakouts on this experiment, so '
            "there is no segment view.</p>"
        )
    rows = estimates_to_readout(list(data))
    if not rows:
        return f'<p class="inc-dashboard-note">{_missing("this breakout returned no segments")}</p>'
    table = readout_table(
        _display_rows(rows),
        title=f"{metric or 'All metrics'} by {rows[0].get('dimension')}",
        subtitle=f"Relative lift of {snapshot.treatment_group} vs {snapshot.control_group}",
        theme=DASHBOARD_THEME,
        nest_by="segment",
        show_interval_level=True,
    )
    _drop_redundant_columns(table)
    return (
        f'<div class="inc-dashboard-table-wrap">{table.as_raw_html()}</div>'
        + _segments_caption(rows)
        + _caveats_list(_segment_caveats(rows))
        + _geometry_disclosure(snapshot, rows, label=INTERVALS_DISCLOSURE)
    )


def _segments_caption(rows: Sequence[Mapping[str, Any]]) -> str:
    segments = sorted({str(row.get("segment")) for row in rows})
    levelled = [
        (row.get("metric"), float(row["level"])) for row in rows if row.get("level") is not None
    ]
    entries = [
        ("Dimension", _esc(rows[0].get("dimension"))),
        ("Source", _esc(rows[0].get("source") or "resolved by the readout")),
        ("Metrics", _count(len({row.get("metric") for row in rows}))),
        ("Segments", f"{_count(len(segments))}: {_esc(', '.join(segments))}"),
        (
            "Interval level",
            _esc(_levels_by_metric(levelled, lambda level: f"{level:.2%}"))
            if levelled
            else _missing("no interval"),
        ),
    ]
    notes = [
        "The shown interval levels come from the breakout readout and retain its "
        "declared multiplicity policy; they may differ from the headline interval.",
        "One significant segment is not evidence of an interaction.",
    ]
    if len(segments) == 1:
        notes.append(
            "This dimension has one segment with exposure in the window. That is a valid "
            "result, not a missing table."
        )
    return _chips(entries) + _disclosure(
        "Segment interpretation",
        _list([_esc(note) for note in notes], css_class="inc-dashboard-reasons"),
    )


def _segment_caveats(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    caveats: list[str] = []
    for row in rows:
        label = (
            f"{_esc(row.get('metric'))}: {_esc(row.get('dimension'))} = {_esc(row.get('segment'))}"
        )
        if row.get("excluded"):
            caveats.append(f"<strong>{label}</strong>: excluded: {_esc(row['excluded'])}")
        if row.get("low_reliability"):
            caveats.append(f"<strong>{label}</strong>: flagged low reliability.")
        if row.get("lift") is None and not row.get("excluded"):
            caveats.append(f"<strong>{label}</strong>: {_missing('no estimate available')}")
    return caveats
