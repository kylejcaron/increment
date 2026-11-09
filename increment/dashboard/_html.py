"""Section-level rendering for the dashboard.

Every function here takes a prepared snapshot and returns one complete
section. No function opens a connection, builds a query, or creates widget
state: notebooks own their controls and pass the values in.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import fields, replace
from importlib import resources
from math import ceil, inf, isfinite, log10
from typing import TYPE_CHECKING, Any, Literal, cast

import coeftable as ct
import marimo as mo
import pandas as pd
from coeftable import plots

from increment._interpretation import interpret_readout, render_readout_text
from increment.dashboard._data import (
    _OVERVIEW_WHOLE,
    DashboardSnapshot,
    _dashboard_allocation_qualification,
    _default_population,
    _population_snapshot,
    added_row,
    all_metrics,
    decision_rows,
    enriched_rows,
    estimate_for_metric,
    group_data_rows,
    readout_interpretation_text,
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
    outcome_label,
    percent_text,
    population_label,
    primary_method,
    primary_tone,
    result_caveats,
    retained_coverage_alpha,
    robust_fence,
    share_text,
    tail_word,
    timestamp_label,
)
from increment.dashboard._native import (
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
from increment.tables import (
    _binomial_qualification_disclosure,
    _failure_disclosure,
    _format_confidence_set,
    _format_metadata_stat,
    _markdown_literal,
    _readout_evidence_notes,
    _readout_metadata_notes,
    _segment_label,
    estimates_to_readout,
)

if TYPE_CHECKING:
    from increment.breakout.estimates import (
        BreakoutEstimates,
        DailyLiftEstimates,
        DailyMetricValues,
    )
    from increment.dashboard._data import ExploreView
    from increment.estimation.diagnostics import SRMResult
    from increment.estimation.health import ExperimentHealth

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
    # Headers here are data (arm IDs), so the native mount never aliases or pins them.
    head = "".join(f"<th scope='col' data-inc-literal>{esc(header)}</th>" for header in headers)
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


def render_header(
    snapshot: DashboardSnapshot, *, population: Literal["assigned", "triggered"] | None = None
) -> mo.Html:
    """Experiment identity, analysis policy, and the selected population's headline summary.

    Readout qualifications are shown by ``render_results`` and ``render_health``, not here.
    """
    snapshot = _population_snapshot(snapshot, population or _default_population(snapshot))
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


def _qualification_fragment(text: str) -> str:
    if not text:
        return ""
    content = esc(text).replace("\n", "<br>")
    return (
        '<div class="inc-dashboard-status">'
        f"<strong>Readout qualifications</strong><p>{content}</p></div>"
    )


def _qualification_block(snapshot: DashboardSnapshot) -> str:
    return _qualification_fragment(readout_interpretation_text(snapshot))


def _added_qualification_block(snapshot: DashboardSnapshot, metric: str) -> str:
    population = _default_population(snapshot)
    overview = snapshot.overview
    if overview is None:
        return _qualification_fragment(
            f"Readout qualifications unavailable for added metric {metric}: "
            "its collection evidence was not captured."
        )
    capture = next(
        (
            capture
            for key, capture in snapshot.explore.items()
            if key[0] == population
            and key[1] == _OVERVIEW_WHOLE
            and key[2] is None
            and key[4] is None
        ),
        None,
    )
    if capture is None:
        return _qualification_fragment(
            f"Readout qualifications unavailable for added metric {metric}: "
            "its collection evidence was not captured."
        )
    collection = capture.load()
    rows = tuple(row for row in overview.exploratory if row.metric == metric)
    if not rows:
        return _qualification_fragment(
            f"Readout qualifications unavailable for added metric {metric}: "
            "its retained result was not captured."
        )
    interpretation = interpret_readout(rows, getattr(collection, "metadata", None))
    rendered = []
    for group in interpretation.groups:
        if group.population not in (None, population):
            continue
        allocation, _ = _dashboard_allocation_qualification(snapshot, population)
        evidence = render_readout_text(replace(interpretation, groups=(group,)))
        rendered.append(
            f"Source snapshot: {group.source_snapshot_id or 'unknown'}\n"
            + allocation
            + "\n"
            + evidence
        )
    return _qualification_fragment("\n\n".join(rendered))


def _header_cards(snapshot: DashboardSnapshot) -> str:
    population = _default_population(snapshot)
    allocation = snapshot.allocation
    check_label = "Triggered balance check" if population == "triggered" else "Allocation check"
    if allocation is None:
        code, reason = snapshot.allocation_refusal or ("", "")
        not_checked = code == "dashboard.allocation_not_applicable"
        enrolled_card = _card(
            "Triggered units" if population == "triggered" else "Enrolled units",
            missing_html("not checked" if not_checked else "allocation check unavailable"),
            worded=True,
        )
        check_card = _card(
            check_label,
            missing_html(reason or "refused by the source"),
            "Not checked. Not a passing check." if not_checked else "Not a passing check.",
            worded=True,
        )
    else:
        enrolled = sum(allocation.observed.values())
        enrolled_card = _card(
            allocation_count_label(allocation, population=_default_population(snapshot)),
            count_text(enrolled),
            allocation_population_detail(allocation, population=_default_population(snapshot)),
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
    row_estimate = next(
        (
            estimate
            for estimate, readout in zip(snapshot.estimates, snapshot.readout_rows, strict=True)
            if readout is row
        ),
        None,
    )
    method_row = {**row, "coverage_alpha": retained_coverage_alpha(row_estimate)}
    caption = primary_method(method_row)
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
    """The allocation, applicable checks, and caveat section shown above results."""
    return _section(
        "health",
        "Experiment health",
        _allocation_block(snapshot, population="assigned")
        + (
            _allocation_block(snapshot, population="triggered")
            if snapshot.triggered_allocation is not None
            or snapshot.triggered_allocation_refusal is not None
            else ""
        )
        + _health_checks_block(snapshot.health)
        + _technical_details_block(snapshot)
        + _flags_block(snapshot)
        + _caveats_block(snapshot),
    )


def _allocation_block(
    snapshot: DashboardSnapshot, *, population: Literal["assigned", "triggered"]
) -> str:
    if population == "triggered":
        allocation = snapshot.triggered_allocation
        refusal = snapshot.triggered_allocation_refusal
        title = "Triggered-cohort balance diagnostic (not assignment integrity)"
    else:
        allocation = snapshot.allocation
        refusal = snapshot.allocation_refusal
        title = "Assigned-population assignment integrity"
    heading = f"<h3>{esc(title)}</h3>"
    if allocation is None:
        code = refusal[0] if refusal is not None else ""
        not_checked = code == "dashboard.allocation_not_applicable"
        label = (
            (
                "Triggered balance not checked; not applicable and not a passing check."
                if not_checked
                else "Triggered balance unavailable; this is not a passing check."
            )
            if population == "triggered"
            else (
                "Assignment integrity not checked; not applicable and not a passing check."
                if not_checked
                else "Assignment integrity unavailable; this is not a passing check."
            )
        )
        return heading + _status("warn", label)
    enrolled = sum(allocation.observed.values())
    verdict = allocation_verdict(allocation, population=population)
    return (
        heading
        + _status("bad" if allocation.is_srm else "ok", verdict)
        + _allocation_table(snapshot, allocation, enrolled, population=population)
        + _disclosure(
            "Population allocation over time and evidence",
            _allocation_history_table(snapshot, population=population)
            + f'<p class="inc-dashboard-note">{allocation_evidence(allocation)}</p>',
        )
    )


def _health_checks_block(report: ExperimentHealth | None) -> str:
    if report is None:
        return ""
    inapplicable_trigger = [
        check
        for check in report.checks
        if check.analysis_population == "triggered"
        and check.availability == "not_applicable"
        and check.context.get("reason") == "This experiment does not declare a trigger."
    ]
    checks = [
        check
        for check in report.checks
        if check not in inapplicable_trigger
        and not (check.check == "assigned_allocation" and check.availability == "available")
    ]
    blocks: list[str] = []
    if checks:
        blocks.append('<h3 class="inc-dashboard-subheading">Additional checks</h3>')
    for check in checks:
        identity = [check.analysis_population]
        if check.metric is not None:
            identity.append(f"metric {check.metric}")
        if check.feed is not None:
            identity.append(f"feed {check.feed}")
        heading = (
            f"<h4>{esc(check.check.replace('_', ' ').title())} "
            f'<span class="inc-dashboard-note">({esc(", ".join(identity))})</span></h4>'
        )
        if check.availability == "unavailable":
            line = "Unavailable; this is not a passing check."
            tone = "warn"
        elif check.availability == "not_applicable":
            line = "Not applicable; this is not an evaluated check."
            tone = "warn"
        elif check.availability == "warning" or check.conclusion == "warning":
            line = "Warning; inspect the available diagnostic evidence."
            tone = "bad"
        elif check.conclusion == "not_rejected":
            line = "Not rejected by this diagnostic; this is not proof of validity."
            tone = "ok"
        else:
            line = "Descriptive evidence; it does not test the assumption."
            tone = "neutral"
        body = heading + _status(tone, line)
        route = check.context.get("route")
        if route is not None:
            body += (
                f'<p class="inc-dashboard-note"><strong>Next step:</strong> {esc(str(route))}</p>'
            )
        if check.rates is not None:
            body += _table(
                ["Arm", "Observed trigger rate"],
                [[esc(arm), f"{rate:.1%}"] for arm, rate in sorted(check.rates.items())],
            )
        blocks.append(body)
    if inapplicable_trigger:
        blocks.append(
            '<p class="inc-dashboard-note">'
            f"{len(inapplicable_trigger)} trigger checks don't apply: no trigger declared."
            "</p>"
        )
    return "".join(blocks)


def _technical_details_block(snapshot: DashboardSnapshot) -> str:
    details: list[str] = []
    for label, refusal in (
        ("Assigned allocation", snapshot.allocation_refusal),
        ("Assigned allocation history", snapshot.allocation_history_refusal),
        ("Triggered allocation", snapshot.triggered_allocation_refusal),
        ("Triggered allocation history", snapshot.triggered_allocation_history_refusal),
    ):
        if refusal is not None:
            code, reason = refusal
            details.append(
                f"<h4>{esc(label)}</h4><dl><dt>Code</dt><dd>{esc(code)}</dd>"
                f"<dt>Reason</dt><dd>{esc(reason)}</dd></dl>"
            )
    if snapshot.health is not None:
        for check in snapshot.health.checks:
            if check.code is None and not check.context:
                continue
            identity = [check.analysis_population]
            if check.metric is not None:
                identity.append(f"metric {check.metric}")
            if check.feed is not None:
                identity.append(f"feed {check.feed}")
            values = []
            if check.code is not None:
                values.append(f"<dt>Code</dt><dd>{esc(check.code)}</dd>")
            values.extend(
                f"<dt>{esc(key.replace('_', ' ').title())}</dt><dd>{esc(value)}</dd>"
                for key, item in check.context.items()
                for value in [str(item)]
            )
            details.append(
                f"<h4>{esc(check.check.replace('_', ' ').title())} "
                f"({esc(', '.join(identity))})</h4><dl>{''.join(values)}</dl>"
            )
    if not details:
        return ""
    return _disclosure("Technical details", "".join(details))


def _allocation_history_table(
    snapshot: DashboardSnapshot, *, population: Literal["assigned", "triggered"]
) -> str:
    history_refusal = (
        snapshot.triggered_allocation_history_refusal
        if population == "triggered"
        else snapshot.allocation_history_refusal
    )
    history = (
        snapshot.triggered_allocation_history
        if population == "triggered"
        else snapshot.allocation_history
    )
    if history_refusal is not None:
        code, reason = history_refusal
        return _status("warn", "Population allocation history unavailable.") + _kv(
            [("Code", esc(code)), ("Reason", esc(reason))]
        )
    if not history:
        label = "trigger history" if population == "triggered" else "enrollment history"
        return f'<p class="inc-dashboard-note">{missing_html(f"no {label}")}</p>'
    allocation = snapshot.triggered_allocation if population == "triggered" else snapshot.allocation
    assert allocation is not None
    frame = pd.DataFrame([dict(row) for row in history]).rename(columns={"group_id": "Variant"})

    totals = frame.groupby("ds")["n_cumulative"].transform("sum").astype(float)
    n = totals.where(totals > 0)
    share = frame["n_cumulative"] / n
    frame["share"] = share
    from scipy.stats import norm

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
        .estimate(
            allocation_count_label(allocation, population=population),
            "n_cumulative",
            fmt=count_text,
        )
        .estimate("Target", "target", fmt=share_text)
        .estimate(
            "Share",
            "share",
            ci=("share_lower", "share_upper"),
            fmt=share_text,
        )
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
        f'<p class="inc-dashboard-note">Cumulative {population} share by '
        f"{'first trigger' if population == 'triggered' else 'first exposure'} date. "
        "Dashed lines show each variant's target; shaded bands show pointwise 95% Wilson "
        "intervals. These bands are not corrected for repeated looks and are not sequential "
        "SRM thresholds.</p>"
    )


def _allocation_table(
    snapshot: DashboardSnapshot,
    allocation: SRMResult,
    enrolled: int,
    *,
    population: Literal["assigned", "triggered"],
) -> str:
    target = snapshot.config.expected_allocation
    total_weight = sum(target.values())
    rows = []
    for arm in (snapshot.control_group, snapshot.treatment_group):
        units = allocation.observed.get(arm, 0)
        observed_share = (
            share_text(units / enrolled)
            if enrolled
            else missing_html(
                "no triggered units" if population == "triggered" else "no enrolled units"
            )
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
        [
            "Arm",
            allocation_count_label(allocation, population=population),
            "Observed share",
            "Target share",
        ],
        rows,
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
    return _qualification_block(snapshot)


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
    population = str(rows[0].get("analysis_population", "assigned"))
    eligible_label = "Triggered eligible units" if population == "triggered" else "Eligible units"
    measures = [eligible_label, f"Observed value ({unit})"]
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
    qualification = _binomial_qualification_disclosure(binomial)
    if qualification:
        explanations.append(qualification)
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
    """Preserve readout metadata so the core-column filter can choose what to display."""
    return [dict(row) for row in rows]


# Keep only the dashboard's core readout and declared advisory columns.
_CORE_COLUMNS = frozenset(
    {
        "Lift (absolute)",
        "Lift %",
        "Lift Plot",
        "Failure",
        "Discovery",
        "Posterior chance to beat",
        "Posterior risk if shipped",
        "Posterior P(favorable)",
    }
)

# The Readout leaves family selection to its notes, the Report and the metric inspector.
_READOUT_COLUMNS = _CORE_COLUMNS - {"Discovery"}


def _keep_core_columns(table: ct.CoefTable, columns: frozenset[str] = _CORE_COLUMNS) -> None:
    table.columns = tuple(
        column for column in table.columns if getattr(column, "label", None) in columns
    )


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


def _selection_qualification_notes(
    rows: Iterable[Mapping[str, Any]],
    *,
    name: Callable[[Mapping[str, Any]], str],
) -> str:
    """Keep selected-row qualifications beside a table without adding a per-row column."""
    affected: dict[str, list[str]] = {}
    for row in rows:
        qualification = row.get("selection_qualification")
        if is_missing(qualification):
            continue
        affected.setdefault(str(qualification), []).append(name(row))
    return _list(
        [
            f"Selection evidence for {esc(', '.join(dict.fromkeys(names)))}: {esc(qualification)}."
            for qualification, names in affected.items()
        ],
        css_class="inc-dashboard-family-qualifications",
    )


def results_table(snapshot: DashboardSnapshot) -> str:
    """The whole declared family as one native CoefTable, without redundant columns."""
    table, metrics = readout_native(
        _display_rows(snapshot.readout_rows),
        title="",
        subtitle=f"{snapshot.treatment_group} vs {snapshot.control_group}",
        theme=coeftable_theme(snapshot.config.theme),
        nest_by="arm",
        advisory=True,
        show_interval_level=True,
    )
    _keep_core_columns(table, _READOUT_COLUMNS)
    charts = snapshot.config.theme.charts
    resize_forest(table, width=charts.forest_width, height=charts.forest_height)
    return _TABLE_WRAP.format(native_html(table, metrics)) + _selection_qualification_notes(
        snapshot.readout_rows,
        name=lambda row: f"{row.get('metric')} · {row.get('group_id')}",
    )


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
            {**row, "segment": f"{_WHOLE} · as in Readout"}
        )
    if overview is not None and overview.family_refusal is None:
        for row in exploratory(enriched_rows(overview.exploratory)):
            by_metric.setdefault(str(row["metric"]), []).append({**row, "segment": _WHOLE})
        segments = overview.segments.get(breakout, ()) if breakout is not None else ()
        for row in exploratory(enriched_rows(segments)):
            # Segment cells sit in their metric's role section so they can nest beneath it, but
            # they belong to the exploratory family, never the metric's declared one.
            labelled = {**row, "segment": f"{row['segment']} · exploratory"}
            by_metric.setdefault(str(row["metric"]), []).append(labelled)
    # readout_table sections rows only by ``role``, so every row takes its metric's declared
    # role section (added metrics form the Exploratory section); the segment label above, not
    # the section, states each cell's inferential role.
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
        advisory=True,
        show_interval_level=True,
    )
    _keep_core_columns(table)
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
        + _selection_qualification_notes(rows, name=name)
    )


def _colour_discoveries_only(table: ct.CoefTable) -> None:
    """Draw exploratory rows that are not BH discoveries in the inconclusive colour.

    The forest colours any interval that clears zero; an exploratory row's unadjusted interval
    may clear zero without the family selecting it, and must not look like a finding.
    """
    frame = pd.DataFrame(table.data)
    if "_exploratory" not in frame.columns:
        return
    marked = frame["_exploratory"].astype("boolean").fillna(False).astype(bool)
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
        "Whole-experiment rows of the experiment's own metrics are the Readout's results, with "
        "their declared roles and corrections (the secondaries are their own discovery family), "
        "never re-corrected here."
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
    shown = {model.name for model in snapshot.exploratory_metrics}
    hidden = [model.name for model in snapshot.offered_metrics if model.name not in shown]
    not_shown = f" ({count_text(len(hidden))} not shown: {', '.join(hidden)})" if hidden else ""
    notes.append(
        f"Exploratory family: Benjamini-Hochberg at q = {overview.family_q:.3g} across "
        f"{count_text(overview.family_size)} comparisons: every eligible whole-experiment cell of "
        f"every saved metric the definitions offer{not_shown}, and every eligible segment cell of "
        "every metric and declared breakout; cells it cannot take are listed with their reasons. "
        "Every eligible comparison counts whether or not its metric is shown, so choosing what "
        "to display never changes the correction. "
        "BH controls the false discovery rate under independence or positive dependence; these "
        "comparisons share units and the control arm, and that condition is not checked."
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
            "Not in the exploratory family, so never marked as discoveries (shown unadjusted "
            f"when their metric is shown): {cells}."
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
        _qualification_block(snapshot)
        + results_table(snapshot)
        + results_warnings(snapshot)
        + results_notes(snapshot)
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
    all_rows = snapshot.readout_rows
    several_arms = len({row.get("group_id") for row in all_rows}) > 1
    several_methods = len({(row.get("method"), row.get("method_role")) for row in all_rows}) > 1
    items = []
    for role in ("primary", "secondary", "guardrail", "unassigned"):
        rows = rows_by_role.get(role)
        if rows:
            items.append(
                _role_disclosure(
                    role,
                    rows,
                    several_arms=several_arms,
                    several_methods=several_methods,
                )
            )
    return items


def _role_disclosure(
    role: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    several_arms: bool,
    several_methods: bool,
) -> str:
    label = ROLE_LABELS.get(role, role)
    tails = "; ".join(f"{esc(row.get('metric'))} {tail_word(row)}" for row in rows)
    sentences = [f"<strong>{esc(label)}</strong>: {tails}."]

    def name(row: Mapping[str, Any]) -> str:
        metric = str(row.get("metric"))
        qualifiers = [str(row.get("group_id"))] if several_arms else []
        if several_methods:
            qualifiers.append(f"{row.get('method')}, {row.get('method_role')}")
        return f"{metric} ({'; '.join(qualifiers)})" if qualifiers else metric

    outcomes = "; ".join(f"{esc(name(row))}: {esc(outcome_label(row))}" for row in rows)
    sentences.append(f"Outcomes: {outcomes}.")
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


def render_metric_details(
    snapshot: DashboardSnapshot,
    *,
    metric: str,
    population: Literal["assigned", "triggered"] | None = None,
) -> mo.Html:
    """Definition, tested tail, policy, and one population's observed group evidence."""
    snapshot = _population_snapshot(snapshot, population or _default_population(snapshot))
    model = require_metric(snapshot, metric)
    if metric not in {declared.name for declared in snapshot.metrics}:
        return _added_metric_details(snapshot, model)
    estimate = estimate_for_metric(snapshot, metric)
    row = row_for_metric(snapshot, metric)
    group_body = _group_data_disclosure(snapshot, metric)
    if estimate is None or row is None:
        body = (
            f'<p class="inc-dashboard-note">{missing_html("no decision result for this metric")}</p>'
            + group_body
        )
        return _section(
            "metric-details", f"Metric: {metric}", body, subtitle=model.description or ""
        )
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
        + _qualification_block(snapshot),
        subtitle=model.description or "",
    )


_ADDED_GROUP_NOTE = (
    "Data by group is captured for the experiment's own metrics; an added metric shows its "
    "Overview result."
)


def _added_family_state(snapshot: DashboardSnapshot, metric: str, row: Mapping[str, Any]) -> str:
    """What the exploratory family did to this added metric's whole-experiment interval."""
    overview = snapshot.overview
    reason = next(
        (
            why
            for name, place, why in (overview.exclusions if overview else ())
            if name == metric and place == "whole experiment"
        ),
        None,
    )
    if reason is not None:
        return f"Not in the exploratory family ({reason}): the interval is unadjusted."
    if row.get("discovery"):
        return "A discovery in the exploratory family: the interval is FCR-adjusted."
    return "In the exploratory family, not a discovery: the interval is at its nominal level."


def _added_metric_details(snapshot: DashboardSnapshot, model: Any) -> mo.Html:
    """An added metric's whole-experiment Overview result, with its family state."""
    overview = snapshot.overview
    row = added_row(snapshot, model.name)
    note = f'<p class="inc-dashboard-note">{esc(_ADDED_GROUP_NOTE)}</p>'
    if row is None:
        refused = overview.family_refusal if overview is not None else None
        if refused is None:
            refused = next(
                (
                    error
                    for metric, place, error in (overview.refusals if overview else ())
                    if metric == model.name and place == "whole experiment"
                ),
                None,
            )
        reason = (
            "no whole-experiment result for this added metric"
            if refused is None
            else f"{refused} ({refused.code})"
        )
        body = f'<p class="inc-dashboard-note">{missing_html(reason)}</p>' + note
        return _section(
            "metric-details",
            f"Metric: {model.name}",
            body,
            subtitle=model.description or "",
        )
    state = (
        f'<p class="inc-dashboard-note">{esc(_added_family_state(snapshot, model.name, row))}</p>'
    )
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
        + state
        + _disclosure(
            "Definition and analysis policy", _kv(_metric_detail_entries(snapshot, model, row))
        )
        + _geometry_disclosure(snapshot, [row], label=INTERVALS_DISCLOSURE)
        + note
        + _added_qualification_block(snapshot, model.name),
        subtitle=model.description or "",
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
    """The requested chart or table, with context and unavailable-point reasons."""
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
    """Standalone temporal SVGs with the latest captured readings and qualifications."""
    bases: dict[str, set[str]] = {}
    for row in data:
        basis = getattr(row, "ds_basis", None)
        if basis is None:
            return _status("warn", "Date-basis metadata is unavailable for this captured view.")
        bases.setdefault(row.metric, set()).add(basis)
    if not bases or any(len(found) != 1 for found in bases.values()):
        return _status(
            "warn",
            "A metric mixes calendar and cohort date bases, so its points cannot share one axis.",
        )
    return _temporal_plots(snapshot, data, metric=metric, view=view)


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


def _date_coordinate(value: Any) -> float:
    if isinstance(value, dt.datetime):
        if value.tzinfo is not None:
            value = value.astimezone(dt.UTC).replace(tzinfo=None)
        return (value - dt.datetime(1970, 1, 1)).total_seconds()
    if isinstance(value, dt.date):
        return (value - dt.date(1970, 1, 1)).days * 86400.0
    return float(value)


def _trajectory_domain(
    traces: Sequence[plots.Trace], ref: float | None, *, robust: bool
) -> tuple[float, float]:
    """Fit finite evidence with the same padding and optional Tukey fence as native plots."""
    values = [
        value
        for trace in traces
        for values in ((trace.y, trace.lower, trace.upper) if trace.show_ribbon else (trace.y,))
        for value in values
        if value is not None and isfinite(value)
    ]
    fence = robust_fence(values) if robust else None
    if fence is not None:
        values = [value for value in values if fence[0] <= value <= fence[1]]
    if ref is not None:
        values.append(ref)
    if not values:
        return -1.0, 1.0
    low, high = min(values), max(values)
    if low == high:
        return low - 1.0, high + 1.0
    margin = (high - low) * 0.08
    return low - margin, high + margin


def _trajectory_svg(
    snapshot: DashboardSnapshot,
    traces: Sequence[plots.Trace],
    *,
    x_domain: tuple[float, float],
    domain: tuple[float, float],
    ref: float | None,
    formatter: ct.Number,
    width: int,
    height: int,
) -> str:
    """Keep the plot and date axis on identical domains, gutters, and endpoint reserves."""
    theme = coeftable_theme(snapshot.config.theme)
    chart = plots.sparkline_multi(
        traces,
        x_domain=x_domain,
        domain=domain,
        ref=ref,
        ref_color=theme.muted,
        fmt=formatter,
        show_y_axis=True,
        y_axis_fmt=formatter,
        width=width,
        height=height,
        show_endpoint=False,
        theme=theme,
    )
    axis = plots.sparkline_axis(
        x_domain=x_domain,
        fmt=ct.DateAxis(),
        temporal=True,
        theme=theme,
        width=width,
        show_endpoint=False,
        y_axis_domain=domain,
        y_axis_fmt=formatter,
        legend=[
            (trace.label, trace.color)
            for trace in traces
            if any(value is not None for value in trace.y)
        ],
    )
    return _TABLE_WRAP.format(
        f'<div class="inc-dashboard-trajectory-svg" style="width:{width}px">{chart}{axis}</div>'
    )


def _point_bounds(row: Any, *, lift: bool) -> tuple[float | None, float | None]:
    point = row.lift if lift else row.value
    if point is not None:
        return point.lb, point.ub
    if lift and row.binomial_set is not None:
        return row.binomial_set.lower, row.binomial_set.upper
    return None, None


def _point_trace(rows: Sequence[Any], *, lift: bool, color: str, label: str) -> plots.Trace:
    points = [row.lift if lift else row.value for row in rows]
    bounds = [_point_bounds(row, lift=lift) for row in rows]
    return plots.Trace(
        x=[_date_coordinate(row.ds) for row in rows],
        y=[point.value if point is not None else None for point in points],
        lower=[lower for lower, _ in bounds],
        upper=[upper for _, upper in bounds],
        color=color,
        show_ribbon=True,
        label=label,
    )


def _lift_traces(snapshot: DashboardSnapshot, rows: Sequence[Any]) -> list[plots.Trace]:
    theme = coeftable_theme(snapshot.config.theme)
    last = next((row for row in reversed(rows) if row.lift is not None), None)
    point = last.lift if last is not None else None
    model = require_metric(snapshot, rows[0].metric)
    direction = (
        "lower_is_better"
        if model.preferred_direction == "decrease"
        else "higher_is_better"
        if model.preferred_direction == "increase"
        else "neutral"
    )
    ref = (rows[-1].null_abs or 0.0) if rows[-1].value_scale == "absolute" else rows[-1].null_lift
    color = theme.color(
        ct.role_for(
            point.lb if point is not None else None,
            point.ub if point is not None else None,
            ref,
            direction,
        )
    )
    return [_point_trace(rows, lift=True, color=color, label="Estimate")]


def _latest_lift_qualifications(
    row: Mapping[str, Any],
) -> tuple[list[tuple[str, str]], list[str]]:
    entries = []
    if not is_missing(row.get("discovery")):
        entries.append(("Discovery", "Yes" if row["discovery"] else "No"))
    if not is_missing(row.get("failure_code")):
        entries.append(
            ("Failure", esc(_failure_disclosure(row["failure_code"], row.get("failure_context"))))
        )
    for key, label in (
        ("posterior_chance_to_beat", "Posterior chance to beat"),
        ("posterior_risk_if_shipped", "Posterior risk if shipped"),
        ("posterior_prob_favorable", "Posterior P(favorable)"),
    ):
        value = row.get(key)
        if is_missing(value):
            continue
        if key == "posterior_prob_favorable" and not (
            row.get("null_lift") or row.get("null_abs") is not None
        ):
            continue
        percentage = key != "posterior_risk_if_shipped" or row.get("value_scale") != "absolute"
        entries.append((label, esc(_format_metadata_stat(value, percentage=percentage))))
    frame = pd.DataFrame([row])
    notes = _readout_metadata_notes(frame) + _readout_evidence_notes(frame)
    return entries, notes


def _attributed_trajectory_notes(notes: Mapping[str, Sequence[str]]) -> list[str]:
    return [
        f"<strong>{esc(', '.join(dict.fromkeys(names)))}</strong>: {text}"
        for text, names in notes.items()
    ]


def _lift_reading_label(row: Mapping[str, Any]) -> str:
    return " · ".join(str(row[key]) for key in ("segment", "group_id") if row.get(key) is not None)


def _lift_trajectory_table(
    snapshot: DashboardSnapshot,
    panels: Sequence[tuple[Any, str]],
    *,
    caption_context: Sequence[str] = (),
) -> str:
    readings = []
    cells = []
    advisory = []
    latest: dict[str, list[str]] = {}
    notes: dict[str, list[str]] = {}
    geometry: dict[str, list[str]] = {}
    caveats: dict[str, list[str]] = {}
    for row, svg in panels:
        reading = estimates_to_readout([row])[0]
        reading["alternative"] = row.alternative
        reading["coverage_alpha"] = retained_coverage_alpha(row)
        readings.append(reading)
        name = _lift_reading_label(reading)
        cells.append(
            {
                "Series": _segment_label(row.dimension_value)
                if row.dimension_value is not None
                else "Whole experiment",
                "Arm": _markdown_literal(row.group_id),
                "Trajectory": svg,
            }
        )
        point = (
            f"{esc(str(reading['ds']))}: <strong>{effect_html(reading)}</strong>; "
            f"{interval_html(reading)} · {primary_method(reading)} · "
            f"Tested null: {esc(null_text(reading))}"
        )
        latest.setdefault(point, []).append(name)
        entries, qualifications = _latest_lift_qualifications(reading)
        advisory.extend((f"{name} · {label}", value) for label, value in entries)
        for note in qualifications:
            notes.setdefault(esc(note), []).append(name)
        for explanation in _geometry_explanations(snapshot, reading):
            geometry.setdefault(esc(explanation), []).append(name)
        for caveat in result_caveats([reading]):
            caveats.setdefault(caveat, []).append(name)
    row = panels[0][0]
    basis = "Exposure cohort" if row.ds_basis == "cohort" else "Observation date"
    dates = list(dict.fromkeys(str(reading["ds"]) for reading in readings))
    coverages = list(
        dict.fromkeys((reading.get("level"), reading.get("coverage_alpha")) for reading in readings)
    )
    caption = [*caption_context, basis]
    caption.append(f"Latest {dates[0]}" if len(dates) == 1 else "Latest dates vary by series")
    if len(coverages) == 1 and not is_missing(coverages[0][0]):
        level, alpha = coverages[0]
        caption.append(
            f"{percent_text(cast(float, level), alpha if isinstance(alpha, float) else None)} "
            "intervals"
        )
    elif len(coverages) > 1:
        caption.append("Confidence levels vary by series")
    context = _chips(
        [
            ("Metric", esc(row.metric)),
            ("Population", esc(row.analysis_population)),
            ("Date axis", basis),
            ("Scale", esc(row.value_scale)),
            ("Method", esc(row.method)),
            ("Estimand", esc(row.estimand)),
            *([("Source", esc(row.source))] if row.source is not None else []),
        ]
    )
    table = ct.CoefTable(
        pd.DataFrame(cells),
        rows="Series",
        nest="Arm",
        columns=(ct.Passthrough("Trajectory", "Trajectory"),),
        theme=coeftable_theme(snapshot.config.theme),
    )
    details = (
        context
        + _list(_attributed_trajectory_notes(latest), css_class="inc-dashboard-reasons")
        + (_chips(advisory) if advisory else "")
        + _list(
            _attributed_trajectory_notes(notes), css_class="inc-dashboard-family-qualifications"
        )
        + _list(_attributed_trajectory_notes(geometry), css_class="inc-dashboard-reasons")
        + _selection_qualification_notes(readings, name=_lift_reading_label)
        + _caveats_list(_attributed_trajectory_notes(caveats))
    )
    return (
        '<div class="inc-dashboard-temporal-table">'
        f'<p class="inc-dashboard-note">{esc(" · ".join(caption))}</p>'
        + _TABLE_WRAP.format(native_html(table, {}))
        + _disclosure("Latest readings and qualifications", details)
        + "</div>"
    )


def _trajectory_label(row: Any) -> str:
    basis = "Exposure cohort" if row.ds_basis == "cohort" else "Observation date"
    parts = [row.metric, row.analysis_population, basis]
    if row.dimension is not None:
        parts.append(f"{row.dimension}: {row.dimension_value}")
    return " · ".join(parts)


def _trajectory_scale_key(row: Any, *, lift: bool) -> tuple[Any, ...]:
    return (
        row.metric,
        row.source,
        row.dimension,
        row.analysis_population,
        row.value_scale,
        row.method if lift else None,
        row.estimand if lift else None,
    )


def _temporal_plots(
    snapshot: DashboardSnapshot, data: Any, *, metric: str | None, view: str
) -> str:
    """Project captured series once, with a small table for compatible lift readings."""
    lift = view == "cumulative_lift"
    rows = [
        row
        for row in data
        if (metric is None or row.metric == metric) and (not lift or row.method_role == "decision")
    ]
    if not rows:
        return f'<p class="inc-dashboard-note">{missing_html("no decision estimates in this view")}</p>'
    grouped: dict[tuple[Any, ...], list[Any]] = {}
    for row in rows:
        key = _trajectory_scale_key(row, lift=lift) + (
            row.dimension_value,
            row.group_id if lift else None,
        )
        grouped.setdefault(key, []).append(row)
    prepared = []
    theme = coeftable_theme(snapshot.config.theme)
    for group in grouped.values():
        group.sort(key=lambda row: _date_coordinate(row.ds))
        if lift:
            traces = _lift_traces(snapshot, group)
        else:
            traces = [
                _point_trace(
                    [row for row in group if row.group_id == arm],
                    lift=False,
                    color=theme.series_palette[0 if arm == snapshot.control_group else 1],
                    label=arm,
                )
                for arm in dict.fromkeys(row.group_id for row in group)
            ]
        prepared.append((group, traces))
    charts = snapshot.config.theme.charts
    fragments = []
    lift_panels: dict[tuple[Any, ...], list[tuple[Any, str]]] = {}
    for group, traces in prepared:
        row = group[-1]
        peers = [
            trace
            for other, series in prepared
            if _trajectory_scale_key(other[0], lift=lift) == _trajectory_scale_key(row, lift=lift)
            for trace in (series[:1] if lift else series)
        ]
        ref = (
            ((row.null_abs or 0.0) if row.value_scale == "absolute" else row.null_lift)
            if lift
            else None
        )
        domain = _trajectory_domain(peers, ref, robust=lift)
        formatter = (
            _percent_axis(group)
            if lift and row.value_scale == "relative"
            else _metric_value_format(
                snapshot, require_metric(snapshot, row.metric), span=domain[1] - domain[0]
            )
        )
        dates = [_date_coordinate(other.ds) for other in rows if lift or other.metric == row.metric]
        width = (
            (
                round(charts.time_width * _COMPACT_CHART_SHARE)
                if metric is None
                else charts.time_width
            )
            if lift
            else charts.absolute_width
        )
        svg = _trajectory_svg(
            snapshot,
            traces,
            x_domain=(min(dates), max(dates)),
            domain=domain,
            ref=ref,
            formatter=formatter,
            width=width,
            height=charts.time_height if lift else charts.absolute_height,
        )
        if lift:
            lift_panels.setdefault(_trajectory_scale_key(row, lift=True), []).append((row, svg))
            continue
        latest = _latest_value_reading(group, formatter)
        fragments.append(
            f'<figure class="inc-dashboard-trajectory"><figcaption>'
            f"{esc(_trajectory_label(row))}</figcaption>{svg}{latest}</figure>"
        )
    if lift:
        varying = (
            [
                (name, label)
                for name, label in (
                    ("source", "Source"),
                    ("dimension", "Breakout"),
                    ("analysis_population", "Population"),
                    ("value_scale", "Scale"),
                    ("method", "Method"),
                    ("estimand", "Estimand"),
                )
                if len({getattr(panels[0][0], name) for panels in lift_panels.values()}) > 1
            ]
            if len(lift_panels) > 1
            else []
        )
        for panels in lift_panels.values():
            row = panels[0][0]
            caption_context = [row.metric] if metric is None else []
            caption_context.extend(f"{label}: {getattr(row, name)!r}" for name, label in varying)
            fragments.append(
                _lift_trajectory_table(snapshot, panels, caption_context=caption_context)
            )
    return "".join(fragments)


def _value_interval(point: Any, formatter: ct.Number) -> str:
    if point is None or point.level is None:
        return missing_html("no interval available")
    lower = esc(formatter(point.lb)) if point.lb is not None else "−∞"
    upper = esc(formatter(point.ub)) if point.ub is not None else "∞"
    opening = "[" if point.lb is not None else "("
    closing = "]" if point.ub is not None else ")"
    level = f"{percent_text(point.level)} " if point.level is not None else ""
    return f"{level}{opening}{lower}, {upper}{closing}"


def _latest_value_reading(rows: Sequence[Any], formatter: ct.Number) -> str:
    latest: dict[str, Any] = {}
    for row in rows:
        latest[row.group_id] = row
    return _table(
        ["Arm", "Latest", "Value", "Interval"],
        [
            [
                esc(row.group_id),
                esc(str(row.ds)),
                esc(formatter(row.value.value))
                if row.value is not None
                else missing_html(row.unavailable),
                _value_interval(row.value, formatter),
            ]
            for row in latest.values()
        ],
    )


def _percent_axis(estimates: Sequence[Any]) -> ct.Percent:
    values = [
        value
        for estimate in estimates
        for value in (
            estimate.lift.value if estimate.lift is not None else None,
            *_point_bounds(estimate, lift=True),
        )
        if value is not None and isfinite(value)
    ]
    fence = robust_fence(values)
    span = (fence[1] - fence[0]) if fence else max(values, default=0.0) - min(values, default=0.0)
    decimals = min(4, max(0, ceil(-log10(span * 100)) + 1)) if span > 0 else 1
    return ct.Percent(scale=100.0, decimals=decimals, signed=True)


_COMPACT_CHART_SHARE = 380 / 560


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
        advisory=True,
        show_interval_level=True,
    )
    _keep_core_columns(table)
    charts = snapshot.config.theme.charts
    resize_forest(table, width=charts.segment_forest_width, height=charts.segment_forest_height)
    return (
        _TABLE_WRAP.format(native_html(table, metrics))
        + _segments_caption(rows)
        + _caveats_list(_segment_caveats(rows))
        + _geometry_disclosure(snapshot, rows, label=INTERVALS_DISCLOSURE)
        + _selection_qualification_notes(
            rows, name=lambda row: f"{row.get('metric')} · {row.get('segment')}"
        )
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
