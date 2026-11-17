"""The production dashboard: one self-contained iframe from a snapshot and its experiment.

``render_dashboard`` renders captured metric, scope and view choices with native CoefTable helpers
and embeds them as JSON in a packaged HTML shell. The confirmatory snapshot is never touched:
Explore selects captured views of the same pinned source read. A state the engine refused is shown
with its code and reason; it is never replaced by another series.
"""

from __future__ import annotations

import functools
import html
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from importlib import resources
from typing import TYPE_CHECKING, Any

import anywidget
import marimo as mo
import traitlets

from increment._readout_html import render_reference_group
from increment.dashboard._data import (
    DashboardSnapshot,
    ExploreView,
    _require_same_experiment,
    all_metrics,
    enriched_rows,
    group_data_csv,
    load_explore,
    metric_names,
    readout_csv,
    row_for_metric,
    show_exploratory_metrics,
)
from increment.dashboard._format import (
    _result_presentation,
    allocation_evidence,
    allocation_grain_label,
    complete_confidence_set_text,
    count_text,
    date_label,
    displayed_coverage,
    effect_html,
    esc,
    has_open_side,
    headline_interval,
    inference_word,
    is_missing,
    levels_by_metric,
    missing_html,
    monitoring_sentence,
    null_text,
    outcome_label,
    percent_text,
    population_label,
    primary_method,
    retained_coverage_alpha_for_row,
    robust_fence,
    tail_word,
    timestamp_label,
)
from increment.dashboard._html import (
    _UNCERTAINTY_CODES,
    _allocation_warnings,
    _dashboard_health,
    _dashboard_results,
    _table,
    overview_explanation,
    overview_table,
    render_comparison_summary,
    render_details,
    render_metric_summary,
    render_result_reference,
    results_table,
    results_warnings,
    temporal_figure,
    temporal_refusal_table,
    unavailable_point_note,
)
from increment.dashboard._references import (
    ReferenceCatalog,
    build_reference_catalog,
)
from increment.dashboard._references import (
    view_key as reference_view_key,
)
from increment.dashboard._theme import theme_css, theme_from_payload
from increment.declarations import thaw
from increment.errors import CodedError
from increment.tables import _binomial_qualification_disclosure

if TYPE_CHECKING:
    from increment.kernel.experiment import Experiment

__all__ = ["build_payload", "document", "health_status", "render_dashboard"]

_MARKER = "<!-- DASHBOARD_DATA -->"
_FRAME_HEIGHT = "1400px"

# Shell view key -> (load_explore view, completed windows only).
_VIEWS: dict[str, tuple[ExploreView, bool]] = {
    "cumulative_lift": ("cumulative_lift", False),
    "cumulative_lift_complete": ("cumulative_lift", True),
    "cumulative_values": ("cumulative_values", False),
    "cumulative_values_complete": ("cumulative_values", True),
    "daily_values": ("daily_values", False),
}
_VIEW_TITLES = {
    "cumulative_lift": "Cumulative relative lift",
    "cumulative_values": "Cumulative per-arm values",
    "daily_values": "Daily per-arm values",
}


def render_dashboard(experiment: Experiment, *, snapshot: DashboardSnapshot) -> mo.ui.anywidget:
    """The complete interactive dashboard for one prepared snapshot.

    ``experiment`` proves the snapshot belongs to the caller's experiment; every tab reads only
    captured evidence. In a running notebook, adding or removing exploratory metrics in Explore
    re-renders the same snapshot showing them; nothing is read again. A static export shows its
    captured selection.
    """
    return mo.ui.anywidget(DashboardWidget(experiment, snapshot=snapshot))


class DashboardWidget(anywidget.AnyWidget):
    """The dashboard page, plus the one request it can send back: a new set of added metrics.

    ``document`` is the rendered page. Setting ``exploratory_metrics`` re-renders the same
    snapshot showing those metrics: every offered metric was read and corrected with the
    headline, so nothing is read or recomputed and the results cannot change. A refused or
    failed update keeps the current page, restores the selection, and reports the failure in
    ``status``; an unexpected failure is re-raised after the rollback so the notebook shows it.
    """

    _esm = resources.files(__package__).joinpath("_bridge.js").read_text(encoding="utf-8")
    _css = f".inc-dashboard-frame{{display:block;width:100%;height:{_FRAME_HEIGHT};border:0}}"
    document = traitlets.Unicode("").tag(sync=True)
    exploratory_metrics = traitlets.List(traitlets.Unicode()).tag(sync=True)
    status = traitlets.Unicode("").tag(sync=True)

    def __init__(self, experiment: Experiment, *, snapshot: DashboardSnapshot) -> None:
        super().__init__(
            document=document(build_payload(experiment, snapshot=snapshot)),
            exploratory_metrics=list(snapshot.config.exploratory_metrics),
        )
        self._experiment = experiment
        self._snapshot = snapshot
        self.observe(self._show, names="exploratory_metrics")

    def _show(self, change: Mapping[str, Any]) -> None:
        requested = tuple(change["new"])
        current = self._snapshot.config.exploratory_metrics
        if requested == current:
            return
        self.status = "updating"
        try:
            snapshot = show_exploratory_metrics(self._snapshot, requested)
            page = document(build_payload(self._experiment, snapshot=snapshot))
        except CodedError as exc:
            self.status = f"{exc} ({exc.code})"
            self.exploratory_metrics = list(current)
            return
        except Exception as exc:
            self.status = f"{type(exc).__name__}: {exc}"
            self.exploratory_metrics = list(current)
            raise
        self._snapshot = snapshot
        self.document = page
        self.status = ""


def document(payload: Mapping[str, Any]) -> str:
    """The packaged shell with ``payload`` embedded as inert JSON.

    ``<``, ``>``, ``&`` and the JavaScript line separators are escaped inside the JSON strings,
    so no payload text can close the script element or be read as markup.
    """
    template = resources.files(__package__).joinpath("_shell.html").read_text(encoding="utf-8")
    assert template.count(_MARKER) == 1, "the packaged dashboard shell requires one data marker"
    views = payload.get("populationViews")
    if isinstance(views, Mapping) and payload.get("defaultPopulation") in views:
        payload = {
            **payload,
            "populationViews": {
                population: view
                for population, view in views.items()
                if population != payload["defaultPopulation"]
            },
        }
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for raw, escaped in (
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("&", "\\u0026"),
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
    ):
        text = text.replace(raw, escaped)
    replacements = {
        _MARKER: text,
        "<!-- DASHBOARD_THEME -->": theme_css(theme_from_payload(payload.get("theme"))),
        "<!-- DASHBOARD_STYLES -->": _stylesheet(),
        "<!-- DASHBOARD_NATIVE -->": resources.files(__package__)
        .joinpath("_native.js")
        .read_text(encoding="utf-8"),
    }
    for marker, value in replacements.items():
        assert template.count(marker) == 1, f"the packaged dashboard shell requires one {marker}"
        template = template.replace(marker, value, 1)
    return template


def build_payload(experiment: Experiment, *, snapshot: DashboardSnapshot) -> dict[str, Any]:
    """Build a coherent dashboard payload for each captured population."""
    default = "triggered" if "triggered" in snapshot.population_estimates else "assigned"
    populations = tuple(snapshot.population_estimates) or ("assigned",)
    views: dict[str, dict[str, Any]] = {}
    for population in populations:
        estimates = (
            snapshot.estimates
            if population == default
            else snapshot.population_estimates.get(population, snapshot.estimates)
        )
        collection = snapshot.population_collections.get(population, snapshot.readout_collection)
        rows = (
            snapshot.readout_rows
            if population == default
            else enriched_rows(collection if collection is not None else estimates)
        )
        scoped = replace(
            snapshot,
            estimates=tuple(estimates),
            readout_collection=collection,
            population_collections=snapshot.population_collections or {population: collection},
            population_estimates={population: tuple(estimates)},
            readout_rows=rows,
            group_data=tuple(
                row for row in snapshot.group_data if row.analysis_population == population
            ),
            explore={
                key: capture for key, capture in snapshot.explore.items() if key[0] == population
            },
            overview=snapshot.overviews.get(population, snapshot.overview),
        )
        payload = _build_payload_for_population(experiment, snapshot=scoped)
        payload["population"] = population
        payload["populationLabel"] = "Triggered" if population == "triggered" else "Assigned"
        payload["populationDisclaimer"] = (
            "Triggered-cohort inference is appropriate only when triggering is unaffected by treatment."
            if population == "triggered"
            else ""
        )
        views[population] = payload
    result = dict(views[default])
    result["defaultPopulation"] = default
    result["populationOptions"] = list(populations)
    result["populationViews"] = views
    return result


def _build_payload_for_population(
    experiment: Experiment, *, snapshot: DashboardSnapshot
) -> dict[str, Any]:
    # A mismatched pair is a caller error, not an engine refusal to display per state.
    _require_same_experiment(experiment, snapshot, metric=None, view="dashboard")
    correction = _segment_correction(snapshot)
    scopes: dict[str, dict[str, Any]] = {
        "overall": {"label": "Whole experiment", "dimension": None, "source": None}
    }
    breakouts: dict[str, tuple[str | None, str] | None] = {"overall": None}
    for key, (source, dimension), label in _declared_scopes(snapshot):
        scopes[key] = {"label": label, "dimension": dimension, "source": source}
        breakouts[key] = (source, dimension)
    catalog = build_reference_catalog(snapshot)
    table = results_table(
        snapshot, catalog=catalog, reference_href="#report-table"
    ) + results_warnings(snapshot)
    explanation = report_explanation(snapshot, catalog=catalog)
    return {
        "title": snapshot.title,
        "theme": asdict(snapshot.config.theme),
        "description": snapshot.description or "",
        "meta": _meta(snapshot),
        "primary": _primary(snapshot),
        "healthStatus": health_status(snapshot),
        "healthReport": (
            None if snapshot.health is None else snapshot.health.model_dump(mode="json")
        ),
        "report": {
            "results": _styled(table),
            **{key: _styled(value) if value else "" for key, value in explanation.items()},
            "context": _report_context(snapshot),
        },
        "results": _styled(_dashboard_results(snapshot, catalog).text),
        "references": _reference_payload(snapshot, catalog),
        "health": _styled(_dashboard_health(snapshot, catalog).text),
        "provenance": _styled(render_details(snapshot).text + _source_reference_picker(catalog)),
        "metrics": [_metric_payload(snapshot, model.name) for model in all_metrics(snapshot)],
        "readoutCsv": readout_csv(snapshot).decode("utf-8"),
        "scopes": scopes,
        "overview": {
            key: _overview_payload(snapshot, breakout, catalog)
            for key, breakout in breakouts.items()
        },
        "exploratory": {
            "available": [
                {
                    "key": model.name,
                    "label": _label(model.name),
                    "description": str(getattr(model, "description", "") or ""),
                }
                for model in snapshot.offered_metrics
            ],
            "selected": [model.name for model in snapshot.exploratory_metrics],
        },
        "explore": {
            key: _explore_scope(experiment, snapshot, breakout, correction, catalog)
            for key, breakout in breakouts.items()
        },
        "wholeExperiment": _whole_explore_scope(experiment, snapshot, correction, catalog),
    }


def _overview_payload(
    snapshot: DashboardSnapshot, breakout: tuple[str | None, str] | None, catalog: ReferenceCatalog
) -> dict[str, Any]:
    explanation = overview_explanation(snapshot, breakout)
    return {
        "html": _styled(overview_table(snapshot, breakout, catalog=catalog)),
        "limitationsHtml": _styled(explanation["limitationsHtml"]),
        "methodsHtml": _styled(explanation["methodsHtml"]),
        "hiddenMetrics": explanation["hiddenMetrics"],
    }


def _reference_payload(snapshot: DashboardSnapshot, catalog: ReferenceCatalog) -> dict[str, Any]:
    memberships: dict[str, list[str]] = {}
    for key, identities in catalog.views.items():
        coordinates = json.loads(key)
        kind = coordinates[0]
        if kind == "readout":
            label = "Readout"
        elif kind == "overview":
            label = "Explore Overview"
            if coordinates[2] is not None:
                label += f" · {coordinates[1] or 'unspecified source'} · {coordinates[2]}"
        else:
            label = _VIEW_TITLES.get(coordinates[2], "Original Overview input")
            label += f" · {coordinates[1]}"
            if coordinates[4]:
                label += " · completed windows only"
            if coordinates[5] is not None:
                label += " · " + " · ".join(str(value) for value in coordinates[5])
        for identity in identities:
            memberships.setdefault(identity, []).append(label)
    results = {}
    for identity, reference in catalog.results.items():
        raw = {
            "reading": dict(reference.original) if reference.original is not None else None,
            "record": reference.record.model_dump(mode="json") if reference.record else None,
        }
        results[identity] = {
            "cell": reference.cell.model_dump(mode="json"),
            "sourceId": reference.source_snapshot_id,
            "viewLabels": memberships.get(identity, ()),
            "selectionLabel": (
                "Overview correction"
                if any("correction" in catalog.groups[group] for group in reference.group_ids)
                else "Original analysis"
            ),
            "groupIds": reference.group_ids,
            "readingHtml": _styled(
                render_result_reference(
                    snapshot,
                    reference,
                    views=memberships.get(identity, ()),
                )
            ),
            "diagnosticCodes": reference.diagnostic_codes,
            "recordJson": json.dumps(raw, indent=2, ensure_ascii=False, default=str),
        }
    return {
        "results": results,
        "diagnosticLabels": {code: text.label for code, text in _UNCERTAINTY_CODES.items()},
        "groups": {
            identity: {
                "previewHtml": render_reference_group(group),
                "recordJson": json.dumps(dict(group), indent=2, ensure_ascii=False, default=str),
            }
            for identity, group in catalog.groups.items()
        },
        "views": dict(catalog.views),
        "refusals": thaw(catalog.view_refusals),
    }


def _source_reference_picker(catalog: ReferenceCatalog) -> str:
    groups = dict.fromkeys(
        group
        for identity in catalog.views.get(reference_view_key(("readout",)), ())
        for group in catalog.results[identity].group_ids
        if "source" in catalog.groups[group]
    )
    if not groups:
        return '<p class="inc-dashboard-note">No source or check records were captured.</p>'
    options = "".join(
        f'<option value="{esc(identity)}">Capture {index}</option>'
        for index, identity in enumerate(groups, 1)
    )
    return (
        "<details><summary>Source and capture records</summary>"
        '<label>Captured source <select class="source-record-choice">'
        '<option value="">Choose a capture…</option>' + options + "</select></label>"
        '<div class="source-record-output native-output"></div></details>'
    )


def _whole_explore_scope(
    experiment: Experiment,
    snapshot: DashboardSnapshot,
    correction: str | None,
    catalog: ReferenceCatalog,
) -> dict[str, dict[str, Any]]:
    """Render one captured whole-experiment packet per temporal view."""
    packets: dict[str, dict[str, Any]] = {}
    declared = metric_names(snapshot)
    selected = [model.name for model in snapshot.exploratory_metrics]
    for view_key, (view, complete) in _VIEWS.items():
        refusals: dict[str, dict[str, str]] = {}
        metric_rows: dict[str, Any] = {}
        reference_metrics: set[str | None] = set()
        if view == "cumulative_lift":
            # Each captured lift family retains its metric-specific declared multiplicity.
            for metric in (*declared, *selected):
                label = metric
                try:
                    series = load_explore(
                        experiment,
                        snapshot=snapshot,
                        metric=metric,
                        view=view,
                        completed_windows_only=complete,
                        breakout=None,
                    )
                except CodedError as exc:
                    refusals[label] = _whole_refusal_detail(exc)
                else:
                    metric_rows[label] = series
                    reference_metrics.add(metric)
        else:
            try:
                batch = load_explore(
                    experiment,
                    snapshot=snapshot,
                    metric=None,
                    view=view,
                    completed_windows_only=complete,
                    breakout=None,
                )
            except CodedError:
                batch = None
            else:
                for metric in declared:
                    metric_rows[metric] = batch.filter(
                        lambda row, metric=metric: row.metric == metric
                    )
                reference_metrics.add(None)
            if batch is None:
                for metric in declared:
                    try:
                        series = load_explore(
                            experiment,
                            snapshot=snapshot,
                            metric=metric,
                            view=view,
                            completed_windows_only=complete,
                            breakout=None,
                        )
                    except CodedError as exc:
                        refusals[metric] = _whole_refusal_detail(exc)
                    else:
                        metric_rows[metric] = series
                        reference_metrics.add(metric)
            for metric in selected:
                label = metric
                try:
                    series = load_explore(
                        experiment,
                        snapshot=snapshot,
                        metric=metric,
                        view=view,
                        completed_windows_only=complete,
                        breakout=None,
                    )
                except CodedError as exc:
                    refusals[label] = _whole_refusal_detail(exc)
                else:
                    metric_rows[label] = series
                    reference_metrics.add(metric)
        for metric, series in metric_rows.items():
            metric_rows[metric] = series.filter(
                lambda row: getattr(row, "method_role", "decision") == "decision"
            )
        for metric in (*declared, *selected):
            if metric in metric_rows and not metric_rows[metric] and metric not in refusals:
                refusals[metric] = {
                    "reason": "The captured view returned no points for this metric.",
                    "code": "",
                    "context": "",
                }
        parts = [series for series in metric_rows.values() if series]
        if not parts:
            packets[view_key] = _whole_refusal_packet(snapshot, view_key, refusals)
            continue
        data = parts[0]
        for part in parts[1:]:
            data = data.concat(part)
        packets[view_key] = _render_entry(
            snapshot,
            data,
            metric=None,
            view=view,
            complete=complete,
            breakout=None,
            correction=correction,
            whole=True,
            refusals=refusals,
            reference_context=(
                catalog,
                tuple(
                    reference_view_key(
                        ("explore", data[0].analysis_population, view, name, complete, None)
                    )
                    for name in reference_metrics
                ),
            ),
        )
    return packets


def _whole_refusal_detail(exc: CodedError) -> dict[str, str]:
    return {
        "reason": str(exc),
        "code": exc.code,
        "context": repr(dict(exc.context)),
    }


def _whole_refusal_packet(
    snapshot: DashboardSnapshot,
    view_key: str,
    refusals: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    title = _VIEW_TITLES[_VIEWS[view_key][0]]
    notes = [f"{metric}: unavailable" for metric in refusals]
    body = (
        temporal_refusal_table(snapshot, refusals)
        if refusals
        else f'<p class="inc-dashboard-note">{esc(title)} unavailable.</p>'
    )
    return {
        "html": _styled(body),
        "caption": f"{title} · no core metric data available",
        "notes": notes,
        "pointCount": 0,
    }


# Hero, health and metric payloads


def _label(name: str) -> str:
    text = name.replace("_", " ")
    return text[:1].upper() + text[1:]


def _plain(markup: str) -> str:
    """Text of a production HTML fragment, for fields the shell sets as textContent."""
    return html.unescape(re.sub(r"<[^>]+>", "", markup))


def _primary(snapshot: DashboardSnapshot) -> dict[str, Any]:
    key = snapshot.primary_metric
    row = row_for_metric(snapshot, key)
    if row is None:
        return {
            "key": key,
            "label": _label(key),
            "effect": "N/A",
            "interval": "",
            "levelLabel": None,
            "tone": "neutral",
            "result": "Unavailable",
        }
    coverage_alpha = retained_coverage_alpha_for_row(snapshot, row)
    method_row = {**row, "coverage_alpha": coverage_alpha}
    coverage = displayed_coverage(method_row)
    tone, result = _result_presentation(row)
    return {
        "key": key,
        "label": _label(key),
        "effect": _plain(effect_html(row)),
        "interval": _plain(headline_interval(row)),
        "levelLabel": (None if coverage is None else percent_text(*coverage)),
        "method": _plain(primary_method(method_row)),
        "tone": tone,
        "result": result,
    }


def _meta(snapshot: DashboardSnapshot) -> str:
    parts = [f"{date_label(snapshot.start)} → {date_label(snapshot.end)}"]
    population = "triggered" if "triggered" in snapshot.population_estimates else "assigned"
    allocation = snapshot.triggered_allocation if population == "triggered" else snapshot.allocation
    if allocation is not None:
        units = sum(allocation.observed.values())
        parts.append(f"{count_text(units)} {population} {allocation_grain_label(allocation)}")
    kinds = sorted({str(row.get("inference", "fixed")) for row in snapshot.readout_rows})
    parts.append(", ".join(inference_word(kind) for kind in kinds) or "fixed-horizon")
    if snapshot.config.source_label:
        parts.append(snapshot.config.source_label)
    return " · ".join(parts)


def health_status(snapshot: DashboardSnapshot) -> dict[str, str]:
    """A qualified allocation status, never a recommendation to ship.

    ``healthy`` means the currently selected population's balance check ran, found no mismatch,
    and nothing else in Health needs a caveat. It is never a recommendation to ship.
    """
    population = "triggered" if "triggered" in snapshot.population_estimates else "assigned"
    allocation = snapshot.triggered_allocation if population == "triggered" else snapshot.allocation
    refusal = (
        snapshot.triggered_allocation_refusal
        if population == "triggered"
        else snapshot.allocation_refusal
    )
    if allocation is None:
        code, reason = refusal or ("", "")
        not_applicable = code == "dashboard.allocation_not_applicable"
        return {
            "kind": "not_checked" if not_applicable else "unavailable",
            "population": population,
            "label": "Allocation check not checked"
            if not_applicable
            else "Allocation check unavailable",
            "detail": f"{reason} ({code}). This is not a passing check.",
        }
    population_label = "Triggered cohort" if population == "triggered" else "Assigned"
    weights = snapshot.config.expected_allocation
    total = sum(weights.values())
    observed = " / ".join(
        f"{arm} {count_text(units)}" for arm, units in allocation.observed.items()
    )
    target = " / ".join(f"{arm} {weight / total:.0%}" for arm, weight in weights.items())
    summary = f"{population_label} {observed} (target {target}). {allocation_evidence(allocation)}"
    warnings = _allocation_warnings(snapshot, allocation, population=population)
    if allocation.is_srm:
        label = (
            "Triggered-cohort imbalance" if population == "triggered" else "Sample ratio mismatch"
        )
    elif warnings:
        label = (
            "Triggered balance needs review"
            if population == "triggered"
            else "Allocation needs review"
        )
    else:
        qualifier = (
            "This is a triggered-cohort balance diagnostic, not assignment integrity."
            if population == "triggered"
            else "This checks assignment balance only; it does not validate any other experiment assumption."
        )
        return {
            "kind": "healthy",
            "population": population,
            "label": "Triggered balance check passed"
            if population == "triggered"
            else "Allocation check passed",
            "detail": (
                f"No population balance issue detected among {count_text(sum(allocation.observed.values()))} "
                f"{population_label.lower()} units. {summary} {qualifier}"
            ),
        }
    return {
        "kind": "warning",
        "population": population,
        "label": label,
        "detail": " ".join(warnings) + " " + summary,
    }


def _metric_payload(snapshot: DashboardSnapshot, metric: str) -> dict[str, Any]:
    declared = metric in metric_names(snapshot)
    row = row_for_metric(snapshot, metric)
    return {
        "key": metric,
        "label": _label(metric),
        "role": (
            str(row.get("role") or "unassigned")
            if row is not None
            else "unassigned"
            if declared
            else "exploratory"
        ),
        "detail": _styled(render_metric_summary(snapshot, metric)),
        # Group data is captured for declared metrics only; no download is offered otherwise.
        "csv": group_data_csv(snapshot, metric=metric).decode("utf-8") if declared else "",
    }


# Report


def _report_context(snapshot: DashboardSnapshot) -> str:
    """Comparison, population, capture time and source for the printed report."""
    parts = [
        f"{snapshot.treatment_group} vs {snapshot.control_group}",
        _plain(population_label(snapshot)),
        f"captured {_plain(timestamp_label(snapshot.computed_at))}",
    ]
    if snapshot.config.source_label:
        parts.append(snapshot.config.source_label)
    return " · ".join(parts)


_INFERENCE_STATEMENTS = {
    "always_valid": "You can monitor repeatedly under the stated assumptions without invalidating these intervals.",
    "asymptotic_mean": (
        "You can monitor repeatedly under the stated assumptions, but the guarantee is "
        "approximate rather than valid for every sample size."
    ),
    "fixed": "These intervals assume analysis at the planned stopping point; repeatedly checking and stopping early can invalidate them.",
}


def _report_interval_reference(
    groups: Mapping[tuple[tuple[float, float | None] | None, str, str], list[str]],
    displayed_groups: Mapping[tuple[tuple[float, float | None] | None, str, str], list[str]],
) -> str:
    """Keep displayed confidence visible and exact captured budgets in reference."""
    if not groups:
        return ""
    rows = []
    for (coverage, kind, tail), names in groups.items():
        interval = f"{kind} intervals, {tail}"
        budget = "Not recorded"
        if coverage is not None:
            interval = f"{percent_text(*coverage)} {interval}"
            budget = repr(coverage[1]) if coverage[1] is not None else "Not recorded"
        rows.append([esc(interval), esc(", ".join(dict.fromkeys(names))), esc(budget)])
    displayed_rows = [
        [
            esc(
                f"{percent_text(*coverage)} {kind} intervals, {tail}"
                if coverage is not None
                else f"Confidence not recorded; {kind} intervals, {tail}"
            ),
            esc(", ".join(dict.fromkeys(names))),
        ]
        for (coverage, kind, tail), names in displayed_groups.items()
    ]
    return (
        _table(["Displayed interval", "Applies to"], displayed_rows)
        + '<details class="report-reference"><summary>Stored interval specifications</summary>'
        + _table(["Stored interval", "Applies to", "Retained error budget (α)"], rows)
        + "</details>"
    )


def report_explanation(
    snapshot: DashboardSnapshot, *, catalog: ReferenceCatalog | None = None
) -> dict[str, str]:
    """Keep material qualifications visible and routine specifications in reference."""
    rows: tuple[Mapping[str, Any], ...] = snapshot.readout_rows
    methods: list[str] = []
    limitations: list[str] = []
    several_arms = len({row.get("group_id") for row in rows}) > 1
    several_methods = len({(row.get("method"), row.get("method_role")) for row in rows}) > 1

    def name(row: Mapping[str, Any]) -> str:
        qualifiers = [str(row.get("group_id"))] if several_arms else []
        if several_methods:
            qualifiers.append(f"{row.get('method')}, {row.get('method_role')}")
        return str(row.get("metric")) + (f" ({'; '.join(qualifiers)})" if qualifiers else "")

    groups: dict[tuple[tuple[float, float | None] | None, str, str], list[str]] = {}
    qualifications: dict[str, list[str]] = {}
    displayed_groups: dict[tuple[tuple[float, float | None] | None, str, str], list[str]] = {}
    for row in rows:
        level = row.get("level")
        coverage = (
            None
            if level is None or is_missing(level)
            else (float(level), retained_coverage_alpha_for_row(snapshot, row))
        )
        key = (coverage, inference_word(str(row.get("inference", "fixed"))), tail_word(row))
        groups.setdefault(key, []).append(name(row))
        shown = displayed_coverage(
            {**row, "coverage_alpha": None if coverage is None else coverage[1]}
        )
        displayed_groups.setdefault((shown, key[1], key[2]), []).append(name(row))
        confidence_set = complete_confidence_set_text(row)
        qualification = _binomial_qualification_disclosure(row.get("binomial_set"), technical=False)
        details = []
        if confidence_set:
            details.append(confidence_set)
        if qualification and qualification != "No numerical qualification claim":
            qualifications.setdefault(qualification, []).append(name(row))
        if details:
            limitations.append(f"<p><strong>{esc(name(row))}.</strong> {' '.join(details)}</p>")
    limitations.extend(
        f"<p><strong>{esc(', '.join(names))}.</strong> {esc(qualification)}</p>"
        for qualification, names in qualifications.items()
    )

    reference = _report_interval_reference(groups, displayed_groups)
    for kind in dict.fromkeys(str(row.get("inference", "fixed")) for row in rows):
        methods.insert(
            0, esc(_INFERENCE_STATEMENTS.get(kind, f"Inference: {inference_word(kind)}."))
        )
    if any(row.get("alternative") in ("greater", "less") for row in rows):
        limitations.insert(
            0,
            "<p>A one-sided limit answers only the stated directional question; the opposite direction remains unconstrained.</p>",
        )
    if any(row.get("role") == "guardrail" for row in rows):
        limitations.insert(
            0, "<p>An inconclusive guardrail does not establish that the change is safe.</p>"
        )
    if any(row.get("family_guarantee") == "asymptotic_sequential" for row in rows):
        limitations.insert(
            0,
            "<p>When these results are judged together, the repeated-monitoring guarantee is approximate, not valid for every sample size.</p>",
        )
    if any(row.get("family_axes") or row.get("family_q") is not None for row in rows):
        methods.append(
            "When several results are judged together, the group's finding and an individual test can disagree."
        )
    if len({coverage for coverage, _, _ in groups}) > 1:
        methods.append("Interval levels differ; use each result's own confidence level.")
    catalog = build_reference_catalog(snapshot) if catalog is None else catalog
    decisions = (
        _table(
            ["Result", "Individual test and group selection", "Test boundary"],
            [[esc(name(row)), esc(outcome_label(row)), esc(null_text(row))] for row in rows],
        )
        if rows
        else ""
    )
    if decisions:
        reference = (
            '<details class="report-decisions"><summary>Individual tests and group selections</summary>'
            + decisions
            + "</details>"
            + reference
        )
    for identity in catalog.views.get(reference_view_key(("readout",)), ()):
        result = catalog.results[identity]
        if not result.diagnostic_codes:
            continue
        consequences = " ".join(
            dict.fromkeys(_UNCERTAINTY_CODES[code].consequence for code in result.diagnostic_codes)
        )
        label = name(result.row or {"metric": result.cell.metric})
        limitations.append(
            f"<p><strong>{esc(label)}.</strong> {esc(consequences)} "
            f'<button type="button" class="evidence-select report-check-link" data-result-ref="{esc(identity)}">'
            "Inspect this result's checks</button></p>"
        )
    health = render_comparison_summary(snapshot, catalog)
    return {
        "limitationsHtml": "".join(limitations),
        "methodsHtml": "".join(f"<p>{text}</p>" for text in dict.fromkeys(methods)) + reference,
        "healthHtml": health,
    }


# Scoped HTML


@functools.cache
def _stylesheet() -> str:
    """The dashboard stylesheet, comments and indentation removed to keep payloads small."""
    css = resources.files(__package__).joinpath("_dashboard.css").read_text(encoding="utf-8")
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    return re.sub(r"\s*([{};])\s*", r"\1", re.sub(r"\s+", " ", css))


def _styled(fragment: str) -> str:
    """Wrap a native fragment; the document owns shared section styles."""
    return f'<div class="inc-dashboard-root">{fragment}</div>'


# Explore


def _declared_scopes(
    snapshot: DashboardSnapshot,
) -> list[tuple[str, tuple[str | None, str], str]]:
    """Unique shell keys for each declared (source, dimension), in declaration order."""
    breakouts = list(dict.fromkeys(snapshot.breakouts))
    repeats = Counter(dimension for _, dimension in breakouts)
    scopes: list[tuple[str, tuple[str | None, str], str]] = []
    used = {"overall"}
    for source, dimension in breakouts:
        shared = repeats[dimension] > 1
        key = (
            f"{source or 'resolved'}:{dimension}" if shared or dimension == "overall" else dimension
        )
        while key in used:
            key += "'"
        used.add(key)
        label = _label(dimension) + (f" · {source or 'resolved source'}" if shared else "")
        scopes.append((key, (source, dimension), label))
    return scopes


def _segment_correction(snapshot: DashboardSnapshot) -> str | None:
    multiplicity = snapshot.view_multiplicity
    return None if multiplicity is None else multiplicity.correction


def _explore_scope(
    experiment: Experiment,
    snapshot: DashboardSnapshot,
    breakout: tuple[str | None, str] | None,
    correction: str | None,
    catalog: ReferenceCatalog,
) -> dict[str, dict[str, Any]]:
    names = [model.name for model in all_metrics(snapshot)]
    entries: dict[str, dict[str, Any]] = {name: {} for name in names}
    declared = set(metric_names(snapshot))
    for view_key, (view, complete) in _VIEWS.items():
        batch = None
        batch_by_metric: dict[str, Any] = {}
        if view != "cumulative_lift":
            # Absolute values are independent per metric, so one call serves them all; the
            # lift family is read per metric so each keeps its own declared multiplicity.
            batch = _attempt(
                lambda complete=complete, view=view: load_explore(
                    experiment,
                    snapshot=snapshot,
                    metric=None,
                    view=view,
                    completed_windows_only=complete,
                    breakout=breakout,
                )
            )
            if batch is not None:
                batch_by_metric = {
                    metric: batch.filter(lambda row, metric=metric: row.metric == metric)
                    for metric in names
                }
        for metric in names:
            # The metric=None batch covers declared metrics only; added metrics load their own.
            entries[metric][view_key] = _explore_entry(
                experiment,
                snapshot,
                metric=metric,
                view_key=view_key,
                breakout=breakout,
                correction=correction,
                batch=batch_by_metric.get(metric) if metric in declared else None,
                catalog=catalog,
            )
    return entries


def _attempt(load: Callable[[], Any]) -> Any:
    """The loaded data, or ``None`` for a refusal the per-metric retry will report exactly."""
    try:
        return load()
    except CodedError:
        return None


def _explore_entry(
    experiment: Experiment,
    snapshot: DashboardSnapshot,
    *,
    metric: str,
    view_key: str,
    breakout: tuple[str | None, str] | None,
    correction: str | None,
    batch: Any,
    catalog: ReferenceCatalog,
) -> dict[str, Any]:
    view, complete = _VIEWS[view_key]
    scope = "the whole experiment" if breakout is None else f"{breakout[1]} segments"
    try:
        if batch is not None:
            data = batch
        else:
            data = load_explore(
                experiment,
                snapshot=snapshot,
                metric=metric,
                view=view,
                completed_windows_only=complete,
                breakout=breakout,
            )
        return _render_entry(
            snapshot,
            data,
            metric=metric,
            view=view,
            complete=complete,
            breakout=breakout,
            correction=correction,
            reference_context=(
                catalog,
                (
                    reference_view_key(
                        (
                            "explore",
                            data[0].analysis_population,
                            view,
                            None if batch is not None else metric,
                            complete,
                            breakout,
                        )
                    ),
                ),
            )
            if data
            else None,
        )
    except CodedError as exc:
        return _refusal_entry(
            view_key,
            scope,
            exc,
            segmented=breakout is not None,
            population="triggered" if "triggered" in snapshot.population_estimates else "assigned",
        )


def _refusal_entry(
    view_key: str,
    scope: str,
    exc: CodedError,
    *,
    segmented: bool,
    population: str,
) -> dict[str, Any]:
    title = _VIEW_TITLES[_VIEWS[view_key][0]]
    reason = f"{title} for {scope} is unavailable: {exc} ({exc.code})."
    substitute = (
        "The whole-experiment series is not substituted."
        if segmented
        else "No other series is substituted."
    )
    route = (
        "Select Assigned for assignment-level daily estimates; the triggered series remains unavailable."
        if population == "triggered"
        else ""
    )
    return {
        "html": _styled(
            f'<p class="inc-dashboard-note">{missing_html(reason)}</p>'
            f'<p class="inc-dashboard-note">{esc(substitute)}</p>'
            + (f'<p class="inc-dashboard-note">{esc(route)}</p>' if route else "")
        ),
        "caption": reason,
        "notes": [reason, substitute] + ([route] if route else []),
        "routeForward": route or None,
        "pointCount": 0,
    }


def _day(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _render_entry(
    snapshot: DashboardSnapshot,
    data: Any,
    *,
    metric: str | None,
    view: ExploreView,
    complete: bool,
    breakout: tuple[str | None, str] | None,
    correction: str | None,
    whole: bool = False,
    refusals: Mapping[str, Mapping[str, str]] | None = None,
    reference_context: tuple[ReferenceCatalog, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    data = data.filter(lambda row: getattr(row, "method_role", "decision") == "decision")
    rows = list(data)
    subject = "Whole experiment" if breakout is None else f"By {breakout[1]}"
    title = _VIEW_TITLES[view] + (" · completed windows only" if complete else "")
    if not rows:
        reason = "The engine returned no points for this state."
        return {
            "html": _styled(f'<p class="inc-dashboard-note">{missing_html(reason)}</p>'),
            "caption": f"{subject} · {title} · no points",
            "notes": [reason],
            "pointCount": 0,
        }
    if getattr(rows[0], "ds", None) is None:
        reason = "The captured response has no date axis, so this time series is unavailable."
        return {
            "html": _styled(f'<p class="inc-dashboard-note">{missing_html(reason)}</p>'),
            "caption": f"{subject} · {title} · date axis unavailable",
            "notes": [reason],
            "pointCount": 0,
        }
    figure = temporal_figure(
        snapshot,
        data,
        metric=metric,
        view=view,
        metric_refusals=refusals if whole else None,
        reference_context=reference_context,
    )
    notes = _notes(
        snapshot, rows, metric=metric, view=view, complete=complete, correction=correction
    )
    if refusals:
        notes.extend(f"{name}: unavailable" for name in refusals)
    gaps = unavailable_point_note(data.to_frame())
    if gaps is not None:
        notes.append(gaps)
    first, last = min(row.ds for row in rows), max(row.ds for row in rows)
    series = len({(row.metric, row.dimension_value, row.group_id) for row in rows})
    caption = (
        f"{subject} · {title} · {_day(first)} → {_day(last)} · "
        f"{series} series · {monitoring_sentence(rows, view=view)}"
    )
    return {
        "html": _styled(figure),
        "caption": caption,
        "notes": notes,
        "pointCount": len(rows),
        "referenceViews": reference_context[1] if reference_context is not None else (),
    }


def _temporal_row_coverage(row: Any, *, lift: bool) -> tuple[float, float | None, bool] | None:
    if lift:
        sequential = getattr(row, "sequential_result", None)
        if sequential is not None:
            alpha = float(sequential.bounds.alpha)
            return 1.0 - alpha, alpha, False
        point = row.lift
        if point is not None and point.level is not None:
            return point.level, point.alpha, False
        binomial = getattr(row, "binomial_set", None)
        if binomial is not None:
            return binomial.level, binomial.alpha, True
        relative = getattr(row, "relative_confidence_set", None)
        if relative is not None:
            return 1.0 - relative.alpha, relative.alpha, True
        return None
    point = row.value
    if point is None or point.level is None:
        return None
    return point.level, point.alpha, False


def _notes(
    snapshot: DashboardSnapshot,
    rows: list[Any],
    *,
    metric: str | None,
    view: str,
    complete: bool,
    correction: str | None,
) -> list[str]:
    lift = view == "cumulative_lift"
    segmented = rows[0].dimension is not None
    coverages = [
        (row, coverage)
        for row in rows
        if (coverage := _temporal_row_coverage(row, lift=lift)) is not None
    ]
    distinct_coverages = {coverage[:2] for _, coverage in coverages}
    if len(distinct_coverages) > 1:
        levelled = [
            (
                f"{row.metric} · {row.dimension}={row.dimension_value} · {row.group_id}"
                if row.dimension is not None
                else f"{row.metric} · {row.group_id}",
                coverage[:2],
            )
            for row, coverage in coverages
        ]
        level_text = levels_by_metric(levelled, lambda value: percent_text(*value))
    else:
        level_text = percent_text(*next(iter(distinct_coverages))) if distinct_coverages else ""
    has_point_free_set = any(coverage[2] for _, coverage in coverages)
    segments = sorted({str(row.dimension_value) for row in rows}) if segmented else []
    notes = [monitoring_sentence(rows, view=view)]
    if view != "daily_values":
        notes.append(
            "Completed windows only: a unit joins once its whole outcome window has closed, so "
            "early dates are absent rather than provisional."
            if complete
            else "Provisional monitoring: a unit joins as soon as it is post-exposure, so early "
            "points rest on partial outcome windows."
        )
        notes.append(
            "Each point is the as-of reading from the first exposure date to that date. A unit "
            "freezes at its last in-window value, so a flat tail is not evidence of recent "
            "observations."
        )
    cohorts = list(dict.fromkeys(row.metric for row in rows if row.ds_basis == "cohort"))
    calendar = list(dict.fromkeys(row.metric for row in rows if row.ds_basis != "cohort"))
    if cohorts:
        names = ", ".join(_label(name) for name in cohorts)
        notes.append(
            (f"{names}: " if calendar else "")
            + "Dates are exposure cohorts: each unit's own exposure date, not the observation date."
        )
        if calendar:
            names = ", ".join(_label(name) for name in calendar)
            notes.append(f"{names}: dates are observation dates.")
    if level_text:
        notes.append(
            (
                "Intervals and confidence sets are at level "
                if has_point_free_set
                else "Intervals are at level "
            )
            + f"{level_text}. "
            + (
                "They describe the lift of each point, one look at a time."
                if lift
                else "Bands are per-arm Wald intervals for each arm's value, not for the difference."
            )
        )
    if segmented:
        assert metric is not None
        notes.append(
            _segment_note(
                snapshot,
                metric,
                segments,
                level_text,
                correction,
                lift=lift,
                has_point_free_set=has_point_free_set,
            )
        )
    if lift:
        notes.extend(
            note
            for name in dict.fromkeys(row.metric for row in rows)
            for note in _lift_notes(
                snapshot,
                [row for row in rows if row.metric == name],
                name,
                segmented=segmented,
            )
        )
    return notes


def _segment_note(
    snapshot: DashboardSnapshot,
    metric: str,
    segments: list[str],
    level_text: str,
    correction: str | None,
    *,
    lift: bool,
    has_point_free_set: bool,
) -> str:
    if not lift:
        return (
            f"Segment values are descriptive measurements across {len(segments)} segments; they do "
            "not test differences between segments."
        )
    if len(segments) == 1:
        return (
            f"This dimension has one segment with exposure in the window ({segments[0]}), so no "
            "segment adjustment applies. Segment lines are pointwise and unadjusted for repeated "
            "looks over time."
        )
    headline = row_for_metric(snapshot, metric)
    coverage = (
        None
        if headline is None
        else displayed_coverage(
            {
                **headline,
                "coverage_alpha": retained_coverage_alpha_for_row(snapshot, headline),
            }
        )
    )
    versus = "" if coverage is None else (f" (the headline uses {percent_text(*coverage)})")
    declared = f" ({correction})" if correction else ""
    return (
        f"Segment "
        f"{'intervals and confidence sets' if has_point_free_set else 'intervals'} "
        f"carry the experiment's declared segment multiplicity{declared} across "
        f"{len(segments)} segments, at "
        f"{'coverage' if has_point_free_set else 'interval level'} {level_text}{versus}. "
        "Segment lines are pointwise and unadjusted for repeated looks over time. One significant "
        "segment is not evidence of an interaction."
    )


def _lift_notes(
    snapshot: DashboardSnapshot, rows: list[Any], metric: str, *, segmented: bool
) -> list[str]:
    notes = []
    values = [
        value
        for row in rows
        if row.lift is not None
        for value in (row.lift.value, row.lift.lb, row.lift.ub)
        if value is not None
    ]
    fence = robust_fence(values)
    if fence is not None:
        outside = sum(1 for value in values if value < fence[0] or value > fence[1])
        if outside:
            notes.append(
                f"{outside} estimate or bound values lie outside the typical range "
                f"({fence[0] * 100:+.0f}% to {fence[1] * 100:+.0f}%) the chart is fitted to. They are "
                "clipped at the plot edge and marked, with a faint trace of the full series, never "
                "dropped; the exact latest interval is in Latest readings and qualifications."
            )
    if has_open_side(rows):
        notes.append(
            "One-sided intervals have no finite endpoint on the open side. Shading is limited "
            "to dates with a finite estimate and two finite bounds; no separate bound lines "
            "are drawn. The retained latest interval is in Latest readings and qualifications."
        )
    headline = row_for_metric(snapshot, metric)
    if not segmented and headline is not None and not is_missing(headline.get("lift")):
        latest = max(
            (row for row in rows if row.group_id == headline["group_id"]),
            key=lambda row: row.ds,
            default=None,
        )
        if (
            latest is not None
            and latest.lift is not None
            and abs(latest.lift.value - headline["lift"]) > 1e-9
        ):
            notes.append(
                "The latest as-of point differs from the confirmatory headline estimate, which "
                "analyses each unit's whole window, because as-of series freeze units at their last "
                "in-window value."
            )
    return notes
