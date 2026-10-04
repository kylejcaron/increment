"""Section-level rendering for the dashboard.

Every function here takes a prepared snapshot and returns one complete
section. No function opens a connection, builds a query, or creates widget
state: notebooks own their controls and pass the values in.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import fields, replace
from importlib import resources
from math import ceil, inf, isfinite, log10
from typing import TYPE_CHECKING, Any, Literal

import coeftable as ct
import marimo as mo
import pandas as pd
from scipy.stats import norm

from increment.dashboard._data import (
    DashboardSnapshot,
    all_metrics,
    decision_rows,
    enriched_rows,
    estimate_for_metric,
    group_data_rows,
    require_metric,
    row_for_metric,
)
from increment.dashboard._format import (
    ROLE_LABELS,
    allocation_count_label,
    allocation_evidence,
    allocation_population_detail,
    allocation_verdict,
    confidence_set_text,
    count_text,
    date_label,
    effect_html,
    esc,
    has_open_side,
    headline_interval,
    headline_number,
    inference_label,
    inference_word,
    interval_html,
    is_missing,
    levels_by_metric,
    missing_html,
    monitoring_sentence,
    null_text,
    number_text,
    percent_text,
    population_label,
    primary_method,
    primary_tone,
    result_caveats,
    result_outcome_text,
    robust_fence,
    share_text,
    tail_word,
    timestamp_label,
)
from increment.dashboard._native import (
    drop_columns,
    native_html,
    readout_native,
    resize_forest,
)
from increment.dashboard._theme import (
    MIDNIGHT,
    DashboardTheme,
    coeftable_theme,
    standalone_css,
)
from increment.tables import _format_confidence_set, estimates_to_readout

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


def dashboard_styles(*, theme: DashboardTheme = MIDNIGHT) -> mo.Html:
    """The configured theme tokens and packaged section stylesheet as one style block.

    Standalone sections scope the theme to ``.inc-dashboard-root``, so the notebook canvas is
    untouched and the typography, layout and palette are the caller's preset. Read from package
    data, so an installed wheel needs no stylesheet path and no repository checkout.
    """
    css = resources.files(__package__).joinpath("_dashboard.css").read_text(encoding="utf-8")
    return mo.Html(f"<style>{standalone_css(theme)}\n{css}</style>")


def _section(anchor: str, heading: str, body: str, *, subtitle: str = "") -> mo.Html:
    sub = f'<p class="inc-dashboard-subtitle">{esc(subtitle)}</p>' if subtitle else ""
    return mo.Html(
        f'<section class="inc-dashboard-root inc-dashboard-section" id="{esc(anchor)}">'
        f'<h2 class="inc-dashboard-heading">{esc(heading)}</h2>{sub}{body}</section>'
    )


def _disclosure(label: str, body: str) -> str:
    return (
        f'<details class="inc-dashboard-disclosure"><summary>{esc(label)}</summary>{body}</details>'
    )


def _chips(entries: Sequence[tuple[str, str]]) -> str:
    items = "".join(f"<li><b>{esc(label)}</b> {value}</li>" for label, value in entries)
    return f'<ul class="inc-dashboard-chips">{items}</ul>'


def _card(label: str, value: str, detail: str = "", *, worded: bool = False) -> str:
    """One summary card. ``worded`` values read as a sentence, not a figure."""
    foot = f'<p class="inc-dashboard-card-detail">{detail}</p>' if detail else ""
    variant = " inc-dashboard-card-value--text" if worded else ""
    return (
        '<div class="inc-dashboard-card">'
        f'<p class="inc-dashboard-card-label">{esc(label)}</p>'
        f'<p class="inc-dashboard-card-value{variant}">{value}</p>{foot}</div>'
    )


def _cards(cards: Iterable[str]) -> str:
    return f'<div class="inc-dashboard-cards">{"".join(cards)}</div>'


def _status(tone: str, text: str) -> str:
    """A status line whose colour is always accompanied by its wording."""
    return f'<p class="inc-dashboard-status inc-dashboard-status--{tone}">{esc(text)}</p>'


def _list(items: Sequence[str], *, css_class: str) -> str:
    if not items:
        return ""
    rows = "".join(f"<li>{item}</li>" for item in items)
    return f'<ul class="{css_class}">{rows}</ul>'


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "".join(f"<th scope='col'>{esc(header)}</th>" for header in headers)
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
        f'<p class="inc-dashboard-lede">{esc(snapshot.description)}</p>'
        if snapshot.description
        else ""
    )
    source = snapshot.config.source_label
    source_html = f'<span class="inc-dashboard-source">{esc(source)}</span>' if source else ""
    body = (
        '<div class="inc-dashboard-headline">'
        f'<h1 class="inc-dashboard-title">{esc(snapshot.title)}</h1>{source_html}</div>'
        f"{lede}{_header_chips(snapshot)}{_header_cards(snapshot)}"
    )
    return mo.Html(
        f'<section class="inc-dashboard-root inc-dashboard-section" id="overview">{body}</section>'
    )


def _header_chips(snapshot: DashboardSnapshot) -> str:
    chips = [
        ("Window", f"{date_label(snapshot.start)} → {date_label(snapshot.end)}"),
        ("Population", population_label(snapshot)),
        ("Inference", inference_label(snapshot)),
        (
            "Arms",
            f'<span class="inc-dashboard-arm">{esc(snapshot.control_group)}</span> → '
            f'<span class="inc-dashboard-arm">{esc(snapshot.treatment_group)}</span>',
        ),
        ("Computed", timestamp_label(snapshot.computed_at)),
    ]
    return _chips(chips)


def _header_cards(snapshot: DashboardSnapshot) -> str:
    allocation = snapshot.allocation
    if allocation is None:
        reason = (snapshot.allocation_refusal or ("", ""))[1]
        enrolled_card = _card(
            "Enrolled units",
            missing_html("allocation check unavailable"),
            worded=True,
        )
        check_card = _card(
            "Allocation check",
            missing_html(reason or "refused by the source"),
            "Not a passing check.",
            worded=True,
        )
    else:
        enrolled = sum(allocation.observed.values())
        enrolled_card = _card(
            allocation_count_label(allocation),
            count_text(enrolled),
            allocation_population_detail(allocation),
        )
        check_card = _card(
            "Observed arm split",
            " / ".join(
                share_text(allocation.observed.get(arm, 0) / enrolled) if enrolled else "N/A"
                for arm in (snapshot.control_group, snapshot.treatment_group)
            ),
            f"{esc(snapshot.control_group)} / {esc(snapshot.treatment_group)}",
        )
    return _cards([enrolled_card, check_card, _primary_card(snapshot)])


def _primary_card(snapshot: DashboardSnapshot) -> str:
    row = row_for_metric(snapshot, snapshot.primary_metric)
    if row is None:
        return _card(
            "Relative lift vs control",
            missing_html(f"no decision result for {snapshot.primary_metric}"),
            worded=True,
        )
    label = (
        "Absolute effect vs control"
        if row.get("value_scale") == "absolute"
        else "Relative lift vs control"
    )
    has_point = not is_missing(row.get("lift"))
    tone = primary_tone(row)
    status = _primary_status(tone, significant=bool(row.get("stat_sig")))
    point = headline_number(row["lift"], row) if has_point else effect_html(row)
    caption = primary_method(row)
    if not has_point and confidence_set_text(row) is not None:
        point = ""
        caption = f"Point estimate unavailable · {caption}"
    point_html = f'<span class="inc-dashboard-primary-estimate">{point}</span>' if point else ""
    return (
        f'<div class="inc-dashboard-card inc-dashboard-primary '
        f'inc-dashboard-primary--{tone}">'
        f'<p class="inc-dashboard-card-label">{label}</p>'
        '<p class="inc-dashboard-primary-inline">'
        f"{point_html}"
        f'<span class="inc-dashboard-primary-interval">{headline_interval(row)}</span></p>'
        '<div class="inc-dashboard-primary-caption">'
        f"<span>{caption}</span>{status}</div></div>"
    )


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
    return f'<span class="inc-dashboard-primary-status" title="{esc(title)}">{esc(label)}</span>'


# Health


def render_health(snapshot: DashboardSnapshot) -> mo.Html:
    """The allocation and caveat section, shown above the results."""
    return _section(
        "health",
        "Experiment health",
        _allocation_block(snapshot) + _flags_block(snapshot) + _caveats_block(snapshot),
        subtitle="",
    )


def _allocation_block(snapshot: DashboardSnapshot) -> str:
    allocation = snapshot.allocation
    if allocation is None:
        code, reason = snapshot.allocation_refusal or ("", "")
        return (
            _status("warn", "Allocation check unavailable; this is not a passing check.")
            + f'<p class="inc-dashboard-code">{esc(code)}</p>'
            + f'<p class="inc-dashboard-reason">{esc(reason)}</p>'
        )
    enrolled = sum(allocation.observed.values())
    verdict = allocation_verdict(allocation)
    return (
        _status("bad" if allocation.is_srm else "ok", verdict)
        + _allocation_table(snapshot, allocation, enrolled)
        + _disclosure(
            "Allocation over time and evidence",
            _allocation_history_table(snapshot)
            + f'<p class="inc-dashboard-note">{allocation_evidence(allocation)}</p>',
        )
    )


def _allocation_history_table(snapshot: DashboardSnapshot) -> str:
    if snapshot.allocation_history_refusal is not None:
        code, reason = snapshot.allocation_history_refusal
        return _status("warn", "Allocation history unavailable.") + _kv(
            [("Code", esc(code)), ("Reason", esc(reason))]
        )
    if not snapshot.allocation_history:
        return f'<p class="inc-dashboard-note">{missing_html("no enrollment history")}</p>'
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
    theme = snapshot.config.theme
    native_theme = coeftable_theme(theme)
    table = (
        ct.CoefTable(latest, rows="Variant")
        .estimate(allocation_count_label(allocation), "n_cumulative", fmt=count_text)
        .estimate("Share", "share", ci=("share_lower", "share_upper"), fmt=share_text)
        .estimate("Target", "target", fmt=share_text)
        .sparkline(
            "Cumulative allocation",
            value="share",
            ci=("share_lower", "share_upper"),
            x="ds",
            data=frame,
            ref=None,
            annotations=[ct.Rule(at="target", axis="y", color=native_theme.muted)],
            scale="table",
            width=theme.charts.allocation_width,
            height=theme.charts.allocation_height,
            axis_fmt=ct.DateAxis(),
            fmt=share_text,
        )
        .with_theme(native_theme)
    )
    return (
        f'<div class="inc-dashboard-table-wrap">{table.as_raw_html()}</div>'
        '<p class="inc-dashboard-note">Cumulative enrolled share by enrollment date. '
        "Dashed lines show each variant's target; shaded bands show pointwise 95% Wilson "
        "intervals. These bands are not corrected for repeated looks and are not sequential "
        "SRM thresholds.</p>"
    )


def _allocation_table(snapshot: DashboardSnapshot, allocation: SRMResult, enrolled: int) -> str:
    target = snapshot.config.expected_allocation
    total_weight = sum(target.values())
    rows = []
    for arm in (snapshot.control_group, snapshot.treatment_group):
        units = allocation.observed.get(arm, 0)
        observed_share = (
            share_text(units / enrolled) if enrolled else missing_html("no enrolled units")
        )
        weight = target.get(arm)
        expected_share = (
            share_text(weight / total_weight)
            if weight is not None
            else missing_html("no target weight")
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
                f"{esc(arm)} <span class='inc-dashboard-tag'>{role}</span>",
                count_text(units),
                meter + observed_share,
                expected_share,
            ]
        )
    return _table(
        ["Arm", allocation_count_label(allocation), "Observed share", "Target share"], rows
    )


def _flags_block(snapshot: DashboardSnapshot) -> str:
    allocation = snapshot.allocation
    if allocation is None:
        return ""
    flags: list[str] = []
    if allocation.unassigned_units:
        flags.append(
            f"{count_text(allocation.unassigned_units)} units are not assigned to any arm. "
            "They are excluded from the arm counts above."
        )
    if allocation.mixed_assignment_units:
        flags.append(
            f"{count_text(allocation.mixed_assignment_units)} units appear in more than one arm. "
            "Every arm count above is reduced by the mixed units it contained."
        )
    if allocation.low_expected_count:
        smallest = (
            number_text(allocation.min_expected_count)
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
        [esc(flag) for flag in flags], css_class="inc-dashboard-flags"
    )


def _caveats_block(snapshot: DashboardSnapshot) -> str:
    """Keep actual caveats visible without an empty-state paragraph."""
    caveats = result_caveats(snapshot.readout_rows)
    return _caveats_list(caveats)


def _caveats_list(caveats: Sequence[str]) -> str:
    """A caveat list beside a table; nothing at all when there are none."""
    if not caveats:
        return ""
    return '<h3 class="inc-dashboard-subheading">Result caveats</h3>' + _list(
        caveats, css_class="inc-dashboard-caveats"
    )


def _format_group_value(value: Any, unit: str, *, count: bool = False) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    if count:
        return count_text(value)
    if unit == "%":
        return f"{float(value):.1%}"
    if unit == "USD":
        return f"${float(value):,.2f}"
    if unit == "ms":
        return f"{float(value):,.2f} ms"
    return f"{float(value):,.4g}"


def _group_data_table(snapshot: DashboardSnapshot, metric: str) -> str:
    """Render the immutable aggregate rows captured with this snapshot.

    One row per measure and one column per arm, so the table stays as narrow as the arm count
    however many measures a metric type adds.
    """
    rows = group_data_rows(snapshot, metric=metric)
    if not rows:
        return missing_html("no group aggregates available")
    unit = str(rows[0].get("unit", "value"))
    model = require_metric(snapshot, metric)
    measures = ["Eligible units", f"Observed value ({unit})"]
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
    measures.extend(label for label, _, _ in keys)
    reason_labels = {
        "eligible_units": "Eligible units",
        "observed_value": measures[1],
        **{key: label for label, key, _ in keys},
    }
    has_unavailable = any(
        key in reason_labels for row in rows for key in (row.get("unavailable") or {})
    )
    if has_unavailable:
        measures.append("Unavailable / not applicable")
    columns: list[list[str]] = []
    for row in rows:
        cells = [
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
        escaped_cells = [esc(cell) for cell in cells]
        if has_unavailable:
            unavailable = row.get("unavailable") or {}
            reasons = "; ".join(
                f"{reason_labels[key]}: {value}"
                for key, value in unavailable.items()
                if key in reason_labels
            )
            escaped_cells.append(
                f'<span class="inc-dashboard-unavailable-reasons">{esc(reasons)}</span>'
            )
        columns.append(escaped_cells)
    headers = ["Measure", *(str(row.get("group_id", "")) for row in rows)]
    body = [
        [esc(measure), *(column[index] for column in columns)]
        for index, measure in enumerate(measures)
    ]
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
        is_missing(open_side)
        and binomial is not None
        and not is_missing(binomial)
        and binomial.geometry == "lower_bound"
    ):
        open_side = "upper"
    if (
        is_missing(open_side)
        and relative is not None
        and not is_missing(relative)
        and relative.geometry == "one_sided"
    ):
        open_side = "lower" if relative.intervals[0][0] is None else "upper"
    if binomial is not None and not is_missing(binomial) and binomial.geometry == "upper_bound":
        explanations.append(
            "This is a one-sided binary test for a decrease. Its lower endpoint is the physical "
            "−100% relative-lift floor, not an open lower side."
        )
    elif alternative in ("greater", "less") and open_side in ("upper", "lower"):
        bounded = "lower" if alternative == "greater" else "upper"
        opposite = "upper" if alternative == "greater" else "lower"
        explanations.append(
            f"This is a declared {tail_word(row)}: it bounds the {bounded} direction; "
            f"the {opposite} direction is intentionally unconstrained by this test."
        )
    if binomial is not None and not is_missing(binomial) and binomial.x_c == 0:
        explanations.append(
            f"The control arm has 0/{count_text(binomial.n_c)} retained/converted units, versus "
            f"{count_text(binomial.x_t)}/{count_text(binomial.n_t)} in {row.get('group_id', snapshot.treatment_group)}. "
            "Its zero control count supplies no finite upper relative-effect bound."
        )
    if relative is not None and not is_missing(relative):
        description = _RELATIVE_SET_EXPLANATIONS.get(relative.geometry)
        if description:
            explanations.append(
                description + (f" Reported reason: {relative.reason}." if relative.reason else "")
            )
    if not is_missing(row.get("relative_unavailable_reason")):
        explanations.append(
            f"Relative evidence is unavailable: {row['relative_unavailable_reason']}."
        )
    if (
        is_missing(row.get("lift"))
        or not is_missing(row.get("relative_unavailable_reason"))
        or (
            relative is not None and not is_missing(relative) and relative.geometry == "unavailable"
        )
    ):
        absolute = [
            (label, row[field])
            for label, field in (
                ("point", "abs_diff"),
                ("lower bound", "abs_lb"),
                ("upper bound", "abs_ub"),
            )
            if not is_missing(row.get(field)) and isfinite(row[field])
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


def _display_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Omit the family selection column only from the visual readout."""
    return [{key: value for key, value in row.items() if key != "discovery"} for row in rows]


# Numeric confidence-set evidence lives in interval disclosures and the per-level wording beside
# each table, so the verdict, set and per-row level columns are not repeated in the grid. The
# unrounded values stay in the readout frame and CSV.
_REDUNDANT_COLUMNS = frozenset({"Significant", "Confidence set", "Interval"})


INTERVALS_DISCLOSURE = "How to read these intervals"
_TABLE_WRAP = '<div class="inc-dashboard-table-wrap">{}</div>'


def _geometry_disclosure(
    snapshot: DashboardSnapshot,
    rows: Iterable[Mapping[str, Any]],
    *,
    label: str,
    name: Callable[[Mapping[str, Any]], str] = lambda row: str(row.get("metric")),
) -> str:
    geometry = _list(
        [
            f"<strong>{esc(name(row))}</strong>: {esc(text)}"
            for row in rows
            for text in _geometry_explanations(snapshot, row)
        ],
        css_class="inc-dashboard-reasons",
    )
    return _disclosure(label, geometry) if geometry else ""


def results_table(snapshot: DashboardSnapshot) -> str:
    """The whole declared family as one native CoefTable, without redundant columns."""
    table, metrics = readout_native(
        _display_rows(snapshot.readout_rows),
        title="",
        subtitle=f"{snapshot.treatment_group} vs {snapshot.control_group}",
        theme=coeftable_theme(snapshot.config.theme),
        nest_by="arm",
        advisory=False,
        show_interval_level=True,
    )
    drop_columns(table, _REDUNDANT_COLUMNS)
    charts = snapshot.config.theme.charts
    resize_forest(table, width=charts.forest_width, height=charts.forest_height)
    return _TABLE_WRAP.format(native_html(table, metrics))


_WHOLE = "Whole experiment"


def overview_rows(
    snapshot: DashboardSnapshot, breakout: tuple[str | None, str] | None
) -> list[dict[str, Any]]:
    """The Explore overview's decision rows for one scope, metric by metric.

    Declared metrics' whole-experiment rows are the Readout's rows. Exploratory rows (added
    metrics and every segment) carry the overview family's correction and the ``_exploratory``
    mark; their significance is their BH discovery, not their unadjusted interval.
    """
    overview = snapshot.overview

    def exploratory(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [
            {**row, "stat_sig": bool(row.get("discovery")), "_exploratory": True} for row in rows
        ]

    by_metric: dict[str, list[dict[str, Any]]] = {}
    for row in decision_rows(snapshot):
        by_metric.setdefault(str(row["metric"]), []).append(
            {**row, "segment": f"{_WHOLE} · confirmatory"}
        )
    if overview is not None and overview.family_refusal is None:
        for row in exploratory(enriched_rows(overview.exploratory)):
            by_metric.setdefault(str(row["metric"]), []).append({**row, "segment": _WHOLE})
        segments = overview.segments.get(breakout, ()) if breakout is not None else ()
        for row in exploratory(enriched_rows(segments)):
            by_metric.setdefault(str(row["metric"]), []).append(row)
    # Segments nest under their metric, so every row takes its metric's declared role group;
    # added metrics form the Exploratory group.
    roles = {str(row["metric"]): row.get("role") for row in decision_rows(snapshot)}
    ordered = [
        {**row, "role": roles.get(model.name, "exploratory")}
        for model in all_metrics(snapshot)
        for row in by_metric.get(model.name, [])
    ]
    if breakout is None:
        return [{key: value for key, value in row.items() if key != "segment"} for row in ordered]
    # Segment nesting needs one method column; each row is its metric's decision method.
    return [{**row, "method": "decision"} for row in ordered]


def overview_table(snapshot: DashboardSnapshot, breakout: tuple[str | None, str] | None) -> str:
    """The Explore overview as one native CoefTable, segments nested under each metric."""
    rows = overview_rows(snapshot, breakout)
    if not rows:
        return f'<p class="inc-dashboard-note">{missing_html("no decision rows to show")}</p>'
    table, metrics = readout_native(
        _display_rows(rows),
        title="",
        subtitle=f"{snapshot.treatment_group} vs {snapshot.control_group}",
        theme=coeftable_theme(snapshot.config.theme),
        nest_by="arm" if breakout is None else "segment",
        advisory=False,
        show_interval_level=True,
    )
    drop_columns(table, _REDUNDANT_COLUMNS)
    charts = snapshot.config.theme.charts
    resize_forest(table, width=charts.forest_width, height=charts.forest_height)
    _colour_discoveries_only(table)
    declared = {model.name for model in snapshot.metrics}
    explained = [
        row for row in rows if row["metric"] not in declared or row.get("dimension") is not None
    ]

    def name(row: Mapping[str, Any]) -> str:
        segmented = row.get("dimension") is not None
        return f"{row['metric']} · {row['segment']}" if segmented else str(row["metric"])

    return (
        _TABLE_WRAP.format(native_html(table, metrics))
        + _geometry_disclosure(snapshot, explained, label=INTERVALS_DISCLOSURE, name=name)
        + _caveats_list(result_caveats([{**row, "metric": name(row)} for row in explained]))
    )


def _colour_discoveries_only(table: ct.CoefTable) -> None:
    """Draw exploratory rows that are not BH discoveries in the inconclusive colour.

    The forest colours any interval that clears zero; an exploratory row's unadjusted interval
    may clear zero without the family selecting it, and must not look like a finding.
    """
    frame = pd.DataFrame(table.data)
    if "_exploratory" not in frame.columns:
        return
    marked = frame["_exploratory"].fillna(False).astype(bool)
    neutral = frozenset(int(i) for i in frame.index[marked & ~frame["stat_sig"].astype(bool)])
    if not neutral:
        return

    def inconclusive(*_: Any) -> Literal["inconclusive"]:
        return "inconclusive"

    def neutral_forest(column: Any) -> Any:
        base = type(column)

        class DiscoveryForest(base):
            def cell(self, ctx: Any) -> str:
                if ctx.index in neutral:
                    ctx = replace(ctx, color_rule=inconclusive)
                return super().cell(ctx)

        return DiscoveryForest(
            **{field.name: getattr(column, field.name) for field in fields(column)}
        )

    table.columns = tuple(
        neutral_forest(column) if isinstance(column, ct.Forest) else column
        for column in table.columns
    )


def overview_notes(
    snapshot: DashboardSnapshot, breakout: tuple[str | None, str] | None
) -> list[str]:
    """Plain-text statements a reader needs beside the overview table."""
    overview = snapshot.overview
    notes = [
        "Whole-experiment rows of the experiment's own metrics are the Readout's confirmatory "
        "results, never re-corrected here."
    ]
    if overview is None:
        return notes
    for metric, place, refusal in overview.refusals:
        notes.append(f"{metric} ({place}) is unavailable: {refusal} ({refusal.code}).")
    if overview.family_refusal is not None:
        refusal = overview.family_refusal
        notes.append(
            f"Exploratory cells are not shown: {refusal} ({refusal.code}). Uncorrected "
            "exploratory estimates are not substituted."
        )
        return notes
    notes.append(
        f"Exploratory family: Benjamini-Hochberg at q = {overview.family_q:.3g} across "
        f"{count_text(overview.family_size)} comparisons (added metrics and every segment cell "
        "of every declared breakout), fixed for this snapshot."
    )
    notes.append(
        "Only BH discoveries are coloured as findings, with FCR-adjusted intervals. Other "
        "exploratory intervals are unadjusted and drawn in the neutral colour even when they "
        "exclude zero. Exploratory results generate hypotheses; they are not confirmatory."
    )
    if overview.exclusions:
        cells = "; ".join(
            f"{metric} ({place}): {reason}" for metric, place, reason in overview.exclusions
        )
        notes.append(
            f"Not in the exploratory family, so shown unadjusted and never marked: {cells}."
        )
    return notes


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
        + results_warnings(snapshot)
        + results_notes(snapshot)
        + _caveats_list(result_caveats(snapshot.readout_rows))
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


def results_warnings(snapshot: DashboardSnapshot) -> str:
    """Status markup for decision rows whose whole interval is adverse."""
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
    several_arms = len({row.get("group_id") for row in snapshot.readout_rows}) > 1
    items = []
    for role in ("primary", "secondary", "guardrail", "unassigned"):
        rows = rows_by_role.get(role)
        if rows:
            items.append(_role_disclosure(role, rows, several_arms=several_arms))
    return items


def _role_disclosure(role: str, rows: Sequence[Mapping[str, Any]], *, several_arms: bool) -> str:
    label = ROLE_LABELS.get(role, role)
    tails = "; ".join(f"{esc(row.get('metric'))} {tail_word(row)}" for row in rows)
    sentences = [f"<strong>{esc(label)}</strong>: {tails}."]
    outcomes = "; ".join(
        f"{esc(row.get('metric'))}, {esc(result_outcome_text(row, arm=several_arms))}"
        for row in rows
    )
    sentences.append(f"Outcomes: {outcomes}.")
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
        shown = esc(levels_by_metric(levelled, percent_text))
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
        parts.append(f"family axes {esc(', '.join(str(axis) for axis in axes))}")
    if q is not None:
        parts.append(f"q = {number_text(float(q), 3)}")
    return (
        f"Family selection ({'; '.join(parts)}) is distinct from the row's tested-alternative "
        "verdict. Its metadata is retained in metric details and CSV."
    )


def render_details(snapshot: DashboardSnapshot) -> mo.Html:
    """Declared policy and optional provenance, with no source access."""
    entries = [
        ("Experiment", esc(snapshot.experiment_name)),
        ("Control arm", esc(snapshot.control_group)),
        ("Treatment arm", esc(snapshot.treatment_group)),
        ("Declared window", f"{date_label(snapshot.start)} → {date_label(snapshot.end)}"),
        ("Primary metric", esc(snapshot.primary_metric)),
        ("Computed", esc(snapshot.computed_at.isoformat())),
        (
            "Declared breakouts",
            esc(
                ", ".join(
                    f"{dimension} ({source or 'source resolved by the readout'})"
                    for source, dimension in snapshot.breakouts
                )
                or "No declared breakouts"
            ),
        ),
    ]
    if snapshot.config.source_label:
        entries.append(("Source", esc(snapshot.config.source_label)))
    entries.extend((label, esc(value)) for label, value in snapshot.config.provenance.items())
    policies = []
    for model in snapshot.metrics:
        row = row_for_metric(snapshot, model.name)
        if row is not None:
            policies.append(
                f"<details><summary>{esc(model.name)}</summary>"
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
            f'<p class="inc-dashboard-note">{missing_html("no decision result for this metric")}</p>'
            + group_body
        )
        return _section("metric-details", f"Metric: {metric}", body)
    return _section(
        "metric-details",
        f"Metric: {model.name}",
        _chips(
            [
                ("Lift", effect_html(row)),
                ("Interval", interval_html(row)),
                ("Favorable", esc(row.get("preferred_direction") or "not declared")),
            ]
        )
        + _disclosure(
            "Definition and analysis policy", _kv(_metric_detail_entries(snapshot, model, row))
        )
        + _geometry_disclosure(snapshot, [row], label=INTERVALS_DISCLOSURE)
        + group_body
        + _list(result_caveats([row]), css_class="inc-dashboard-caveats"),
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
        "Only retained transformed inputs are shown in the separate analysis-input row; "
        "pre-transform outcomes absent from a checkpoint remain unavailable.</p>",
    )


def _metric_detail_entries(
    snapshot: DashboardSnapshot, model: Any, row: Mapping[str, Any]
) -> list[tuple[str, str]]:
    declared = getattr(model, "declared_preferred_direction", None)
    direction = row.get("preferred_direction")
    direction_text = esc(direction or "not declared")
    if declared is None and direction is not None:
        direction_text += ' <span class="inc-dashboard-reason">(library default)</span>'
    entries: list[tuple[str, str]] = [
        (
            "Role",
            esc(ROLE_LABELS.get(str(row.get("role")), str(row.get("role") or "unassigned"))),
        ),
        ("Relative lift", effect_html(row)),
        (
            "Interval",
            f"{interval_html(row)}"
            + (f" at {percent_text(row['level'])}" if row.get("level") is not None else ""),
        ),
        ("Favorable direction", direction_text),
        ("Tested alternative", f"{esc(row.get('alternative'))} ({tail_word(row)})"),
        ("Null boundary", null_text(row)),
        ("Inference", esc(inference_word(str(row.get("inference", "fixed"))))),
        ("Analysis population", esc(row.get("analysis_population", "assigned"))),
        ("Decision method", esc(row.get("method", "unknown"))),
        ("Estimand", esc(row.get("estimand", "itt"))),
        ("Measurement window", _metric_window(model)),
        ("Arms", f"{esc(snapshot.control_group)} → {esc(snapshot.treatment_group)}"),
    ]
    entries.extend(_family_entries(row))
    return entries


def _family_entries(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    discovery = row.get("discovery")
    if discovery is not None:
        verdict = "in the discovery set" if discovery else "not in the discovery set"
        entries.append(("Family discovery", esc(verdict)))
    axes = row.get("family_axes")
    if axes:
        entries.append(("Family axes", esc(", ".join(str(axis) for axis in axes))))
    for label, key in (("Family q", "family_q"), ("Family threshold", "family_threshold")):
        value = row.get(key)
        if value is not None:
            entries.append((label, number_text(float(value), 4)))
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
        return esc(" / ".join(f"{part} days" for part in parts))
    return missing_html("no declared window")


def _kv(entries: Sequence[tuple[str, str]]) -> str:
    rows = "".join(f"<dt>{esc(label)}</dt><dd>{value}</dd>" for label, value in entries)
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


def temporal_figure(
    snapshot: DashboardSnapshot, data: Any, *, metric: str | None, view: str
) -> str:
    """The native table(s) for a non-empty temporal view.

    A metric whose points mix calendar and cohort dates cannot share one axis, so it gets a
    warning in place of a chart.
    """
    frame = data.to_frame()
    if (
        not frame["ds_basis"].dropna().nunique()
        or (frame.groupby("metric")["ds_basis"].nunique() != 1).any()
    ):
        return _status(
            "warn",
            "A metric mixes calendar and cohort date bases, so its points cannot share one axis.",
        )
    return _temporal_table(snapshot, data, metric=metric, view=view)


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
        return f'<p class="inc-dashboard-note">{missing_html("this view returned no points")}</p>'
    frame = data.to_frame()
    bases = sorted({str(basis) for basis in frame["ds_basis"].dropna().unique()})
    caption = _temporal_caption(
        snapshot,
        frame,
        bases=bases,
        view=view,
        completed_windows_only=completed_windows_only,
        monitoring=monitoring_sentence(rows, view=view),
    )
    figure = temporal_figure(snapshot, data, metric=metric, view=view)
    return figure + caption + _gaps_block(frame, view=view)


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
    observed = (
        f"{date_label(dates.min())} → {date_label(dates.max())}"
        if len(dates)
        else "no dated points"
    )
    basis_label = ", ".join(_axis_title(basis) for basis in bases) or "unknown basis"
    entries = [
        ("Series covers", esc(observed)),
        ("Date basis", esc(basis_label)),
        ("Points", count_text(len(frame))),
    ]
    if "dimension_value" in frame.columns and frame["dimension_value"].notna().any():
        segments = sorted(str(value) for value in frame["dimension_value"].dropna().unique())
        entries.insert(
            0,
            (
                "Broken out by",
                f"{esc(frame['dimension'].dropna().iloc[0])}: {esc(', '.join(segments))}",
            ),
        )
    if view != "daily_values":
        entries.append(
            ("Maturity", esc(_maturity_label(completed_windows_only=completed_windows_only)))
        )
    frozen_note = (
        '<p class="inc-dashboard-note">An as-of point freezes each unit at its last '
        "in-window value, so a flat tail is not evidence of recent observations.</p>"
        if view != "daily_values"
        else ""
    )
    return (
        _chips(entries)
        + f'<p class="inc-dashboard-note">{esc(monitoring)}</p>'
        + _disclosure(
            "Window and maturity",
            _kv([("Declared window", f"{date_label(snapshot.start)} → {date_label(snapshot.end)}")])
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
    model = require_metric(snapshot, metric)
    # Ticks must stay distinct on the tightest plotted segment.
    spans: list[float] = []
    for _, part in metric_frame.groupby(nest):
        pooled = [
            value
            for column in ("value", "lb", "ub")
            for value in part[column]
            if not is_missing(value) and isfinite(value)
        ]
        if pooled:
            spans.append((max(pooled) - min(pooled)) or abs(max(pooled)) / 10)
    formatter = _metric_value_format(snapshot, model, span=min(spans, default=0.0))
    theme = snapshot.config.theme
    native_theme = coeftable_theme(theme)
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
            width=theme.charts.absolute_width,
            height=theme.charts.absolute_height,
            series_colors={
                snapshot.control_group: native_theme.series_palette[0],
                snapshot.treatment_group: native_theme.series_palette[1],
            },
        )
        .header("", f"{snapshot.control_group} vs {snapshot.treatment_group}")
        .with_theme(native_theme)
    )
    return _TABLE_WRAP.format(native_html(table, {metric: metric}))


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


def _percent_axis(estimates: Sequence[Any]) -> ct.Percent:
    """Tick precision for the typical plotted range, so nearby ticks never repeat."""
    values = _plotted_values(estimates)
    fence = robust_fence(values)
    span = (fence[1] - fence[0]) if fence else max(values, default=0.0) - min(values, default=0.0)
    decimals = min(4, max(0, ceil(-log10(span * 100)) + 1)) if span > 0 else 1
    return ct.Percent(scale=100.0, decimals=decimals, signed=True)


# Several metrics or segments share one view, so each trajectory draws at this share of the
# theme's time-chart width.
_COMPACT_CHART_SHARE = 380 / 560


def _with_bound_lines(column: ct.Sparkline, theme: ct.Theme) -> ct.Sparkline:
    """Draw each finite interval bound as its own line beside the estimate.

    The native ribbon needs both bounds; an open side has none to draw and nothing is invented
    for it, so finite sides become labelled lines and absent ones simply have no line.
    """
    data = column.data
    assert data is not None
    palette = theme.series_palette
    parts = [data.assign(__series="Estimate", __plot=data[column.value])]
    colors = {"Estimate": palette[1]}
    for label, field, color in (
        ("Lower CI", "lb", palette[0]),
        ("Upper CI", "ub", theme.inconclusive),
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
    snapshot: DashboardSnapshot, estimates: Sequence[Any], *, compact: bool
) -> str:
    """Cumulative lift as native CoefTable trajectories: latest reading beside its history."""
    segmented = any(estimate.dimension is not None for estimate in estimates)
    # Day-axis rows carry no preferred direction; take it from each metric's definition so an
    # adverse move is never coloured as a favourable one.
    latest_rows = [
        {
            **row,
            "preferred_direction": require_metric(snapshot, str(row["metric"])).preferred_direction,
        }
        for row in estimates_to_readout(_latest_points(estimates))
    ]
    charts = snapshot.config.theme.charts
    native_theme = coeftable_theme(snapshot.config.theme)
    table, metrics = readout_native(
        _display_rows(latest_rows),
        title="",
        subtitle=f"{snapshot.treatment_group} vs {snapshot.control_group}",
        theme=native_theme,
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
                width=round(charts.time_width * _COMPACT_CHART_SHARE)
                if compact
                else charts.time_width,
                height=charts.time_height,
                scale="row",
                show_y_axis=True,
                y_axis_fmt=axis,
                fmt=axis,
                show_endpoint=False,
            )
            if has_open_side(estimates):
                column = _with_bound_lines(column, native_theme)
        columns.append(column)
    table.columns = tuple(columns)
    return _TABLE_WRAP.format(native_html(table, metrics)) + _geometry_disclosure(
        snapshot, latest_rows, label=INTERVALS_DISCLOSURE
    )


def _temporal_table(
    snapshot: DashboardSnapshot, data: Any, *, metric: str | None, view: str
) -> str:
    """Render cumulative lift or independent absolute-value metric tables."""
    if view == "cumulative_lift":
        decisions = [row for row in data if row.method_role == "decision"]
        if not decisions:
            return f'<p class="inc-dashboard-note">{missing_html("no decision estimates in this view")}</p>'
        if (
            any(row.dimension is not None for row in decisions)
            and len({row.method for row in decisions}) > 1
        ):
            return "".join(
                _lift_trajectory_table(
                    snapshot, [row for row in decisions if row.metric == name], compact=True
                )
                for name in dict.fromkeys(row.metric for row in decisions)
            )
        return _lift_trajectory_table(snapshot, decisions, compact=metric is None)
    frame = data.to_frame()
    selected = [
        model.name for model in all_metrics(snapshot) if metric is None or model.name == metric
    ]
    return _absolute_tables(snapshot, frame, selected=selected)


def _unavailable_counts(frame: Any) -> tuple[dict[str, int], int]:
    if "unavailable" not in frame.columns:
        return {}, len(frame)
    return frame["unavailable"].dropna().value_counts().to_dict(), len(frame)


def unavailable_point_note(frame: Any) -> str | None:
    """One plain-text note on why points are missing, or ``None`` when none are."""
    counts, total = _unavailable_counts(frame)
    if not counts:
        return None
    reasons = "; ".join(f"{reason}: {count_text(count)}" for reason, count in counts.items())
    return (
        f"{count_text(sum(counts.values()))} of {count_text(total)} points are unavailable "
        f"({reasons}). They remain gaps in the chart, not zeros."
    )


def _gaps_block(frame: Any, *, view: str) -> str:
    """Why points are missing, and that they are left as gaps."""
    counts, total = _unavailable_counts(frame)
    if not counts:
        return ""
    items = [
        f"{esc(reason)}: {count_text(count)} of {count_text(total)} points"
        for reason, count in counts.items()
    ]
    heading = "Unavailable points" if view != "segments" else "Excluded segments"
    return (
        f'<h3 class="inc-dashboard-subheading">{esc(heading)}</h3>'
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
        return f'<p class="inc-dashboard-note">{missing_html("this breakout returned no segments")}</p>'
    table, metrics = readout_native(
        _display_rows(rows),
        title=f"{metric or 'All metrics'} by {rows[0].get('dimension')}",
        subtitle=f"Relative lift of {snapshot.treatment_group} vs {snapshot.control_group}",
        theme=coeftable_theme(snapshot.config.theme),
        nest_by="segment",
        show_interval_level=True,
    )
    drop_columns(table, _REDUNDANT_COLUMNS)
    charts = snapshot.config.theme.charts
    resize_forest(table, width=charts.segment_forest_width, height=charts.segment_forest_height)
    return (
        _TABLE_WRAP.format(native_html(table, metrics))
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
        ("Dimension", esc(rows[0].get("dimension"))),
        ("Source", esc(rows[0].get("source") or "resolved by the readout")),
        ("Metrics", count_text(len({row.get("metric") for row in rows}))),
        ("Segments", f"{count_text(len(segments))}: {esc(', '.join(segments))}"),
        (
            "Interval level",
            esc(levels_by_metric(levelled, lambda level: f"{level:.2%}"))
            if levelled
            else missing_html("no interval"),
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
        _list([esc(note) for note in notes], css_class="inc-dashboard-reasons"),
    )


def _segment_caveats(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    caveats: list[str] = []
    for row in rows:
        label = f"{esc(row.get('metric'))}: {esc(row.get('dimension'))} = {esc(row.get('segment'))}"
        if row.get("excluded"):
            caveats.append(f"<strong>{label}</strong>: excluded: {esc(row['excluded'])}")
        if row.get("low_reliability"):
            caveats.append(f"<strong>{label}</strong>: flagged low reliability.")
        if row.get("lift") is None and not row.get("excluded"):
            caveats.append(f"<strong>{label}</strong>: {missing_html('no estimate available')}")
    return caveats
