"""The production dashboard: one self-contained iframe from a snapshot and its analysis.

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
    allocation_evidence,
    allocation_grain_label,
    complete_confidence_set_text,
    count_text,
    date_label,
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
    percent_text,
    population_label,
    primary_method,
    primary_tone,
    result_caveats,
    result_outcome_text,
    robust_fence,
    tail_word,
    timestamp_label,
)
from increment.dashboard._html import (
    overview_notes,
    overview_table,
    render_details,
    render_health,
    render_metric_details,
    results_notes,
    results_table,
    results_warnings,
    temporal_figure,
    unavailable_point_note,
)
from increment.dashboard._theme import theme_css, theme_from_payload
from increment.errors import CodedError

if TYPE_CHECKING:
    from increment.analysis import Analysis
    from increment.estimation.diagnostics import SRMResult

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


def render_dashboard(analysis: Analysis, *, snapshot: DashboardSnapshot) -> Any:
    """The complete interactive dashboard for one prepared snapshot.

    ``analysis`` proves the experiment binding; every tab reads only captured evidence. In a
    running notebook, adding or removing exploratory metrics in Explore re-renders the same
    snapshot showing them; nothing is read again. A static export shows its captured selection.
    """
    return mo.ui.anywidget(DashboardWidget(analysis, snapshot=snapshot))


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

    def __init__(self, analysis: Analysis, *, snapshot: DashboardSnapshot) -> None:
        super().__init__(
            document=document(build_payload(analysis, snapshot=snapshot)),
            exploratory_metrics=list(snapshot.config.exploratory_metrics),
        )
        self._analysis = analysis
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
            page = document(build_payload(self._analysis, snapshot=snapshot))
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


def build_payload(analysis: Analysis, *, snapshot: DashboardSnapshot) -> dict[str, Any]:
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
        rows = snapshot.readout_rows if population == default else enriched_rows(estimates)
        scoped = replace(
            snapshot,
            estimates=tuple(estimates),
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
        payload = _build_payload_for_population(analysis, snapshot=scoped)
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
    analysis: Analysis, *, snapshot: DashboardSnapshot
) -> dict[str, Any]:
    # A mismatched pair is a caller error, not an engine refusal to display per state.
    _require_same_experiment(analysis, snapshot, metric=None, view="dashboard")
    correction = _segment_correction(analysis)
    scopes: dict[str, dict[str, Any]] = {
        "overall": {"label": "Whole experiment", "dimension": None, "source": None}
    }
    breakouts: dict[str, tuple[str | None, str] | None] = {"overall": None}
    for key, (source, dimension), label in _declared_scopes(snapshot):
        scopes[key] = {"label": label, "dimension": dimension, "source": source}
        breakouts[key] = (source, dimension)
    table = results_table(snapshot) + results_warnings(snapshot)
    return {
        "title": snapshot.title,
        "theme": asdict(snapshot.config.theme),
        "description": snapshot.description or "",
        "meta": _meta(snapshot),
        "primary": _primary(snapshot),
        "healthStatus": health_status(snapshot),
        "report": {
            "results": _styled(table),
            "notes": _report_notes(snapshot),
            "context": _report_context(snapshot),
        },
        "results": _styled(table + results_notes(snapshot)),
        "health": _styled(render_health(snapshot).text),
        "provenance": _styled(render_details(snapshot).text),
        "metrics": [_metric_payload(snapshot, model.name) for model in all_metrics(snapshot)],
        "readoutCsv": readout_csv(snapshot).decode("utf-8"),
        "scopes": scopes,
        "overview": {
            key: {
                "html": _styled(overview_table(snapshot, breakout)),
                "notes": overview_notes(snapshot, breakout),
            }
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
            key: _explore_scope(analysis, snapshot, breakout, correction)
            for key, breakout in breakouts.items()
        },
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
            "method": f"No decision result for {key}",
            "tone": "neutral",
        }
    level = row.get("level")
    return {
        "key": key,
        "label": _label(key),
        "effect": _plain(effect_html(row)),
        "interval": _plain(headline_interval(row)),
        "levelLabel": None if level is None or is_missing(level) else percent_text(float(level)),
        "method": _plain(primary_method(row)),
        "tone": primary_tone(row),
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


def _allocation_warnings(
    snapshot: DashboardSnapshot,
    allocation: SRMResult,
    *,
    population: str,
) -> list[str]:
    warnings = []
    population_label = "triggered cohort" if population == "triggered" else "assigned units"
    if allocation.is_srm:
        warnings.append(f"Population imbalance detected in {population_label}.")
    if allocation.unassigned_units:
        warnings.append(
            f"{count_text(allocation.unassigned_units)} units are outside a declared arm."
        )
    if allocation.mixed_assignment_units:
        warnings.append(
            f"{count_text(allocation.mixed_assignment_units)} units appear in more than one arm."
        )
    if allocation.low_expected_count:
        warnings.append("A small expected arm count limits the population balance check.")
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
        warnings.append(f"{population_label.capitalize()} history is unavailable.")
    elif not history:
        warnings.append(f"No {population_label} history was captured.")
    caveats = len(result_caveats(snapshot.readout_rows))
    if caveats:
        warnings.append(f"{caveats} result caveat{'s' if caveats != 1 else ''} listed in Health.")
    return warnings


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
        "detail": _styled(render_metric_details(snapshot, metric=metric).text),
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
    "always_valid": "Always-valid intervals remain valid at every look.",
    "asymptotic_mean": (
        "Asymptotic sequential intervals support repeated monitoring under the registered "
        "assumptions, without a finite-sample guarantee."
    ),
    "fixed": "Fixed-horizon intervals are not corrected for repeated looks.",
}


def _report_notes(snapshot: DashboardSnapshot) -> list[str]:
    """The inference statements a reader needs beside the report table, as plain text.

    Levels, inference kinds and tested directions are read from the captured rows, never derived
    from policy. Confidence sets not represented by the central interval keep every endpoint,
    and each unavailability reason is kept as the engine reported it.
    """
    rows = snapshot.readout_rows
    if not rows:
        return []
    several_arms = len({row.get("group_id") for row in rows}) > 1
    several_methods = len({(row.get("method"), row.get("method_role")) for row in rows}) > 1

    def name(row: Mapping[str, Any]) -> str:
        metric = str(row.get("metric"))
        qualifiers = [str(row.get("group_id"))] if several_arms else []
        if several_methods:
            qualifiers.append(f"{row.get('method')}, {row.get('method_role')}")
        return f"{metric} ({'; '.join(qualifiers)})" if qualifiers else metric

    groups: dict[str, list[str]] = {}
    for row in rows:
        level = row.get("level")
        prefix = "" if level is None or is_missing(level) else f"{percent_text(float(level))} "
        kind = inference_word(str(row.get("inference", "fixed")))
        names = groups.setdefault(f"{prefix}{kind} intervals, {tail_word(row)}", [])
        if name(row) not in names:
            names.append(name(row))
    if len(groups) == 1:
        notes = [f"All results: {next(iter(groups))}."]
    else:
        notes = [
            "Intervals by metric: "
            + "; ".join(f"{label} ({', '.join(names)})" for label, names in groups.items())
            + "."
        ]
    levels = {float(row["level"]) for row in rows if not is_missing(row.get("level"))}
    if len(levels) > 1:
        notes.append("Interval levels differ; compare each interval only with its own level.")
    kinds = dict.fromkeys(str(row.get("inference", "fixed")) for row in rows)
    notes.extend(
        _INFERENCE_STATEMENTS.get(kind, f"Inference: {inference_word(kind)}.") for kind in kinds
    )
    if any(row.get("alternative") in ("greater", "less") for row in rows):
        notes.append(
            "A one-sided test constrains only the tested direction; the opposite direction is "
            "unconstrained."
        )
    if any(row.get("role") == "guardrail" for row in rows):
        notes.append("A guardrail that does not reject is not evidence of no harm.")
    shifted = [
        f"{name(row)} {null_text(row)}"
        for row in rows
        if not is_missing(row.get("null_abs"))
        or not (is_missing(row.get("null_lift")) or row.get("null_lift") == 0)
    ]
    if shifted:
        notes.append(f"Non-zero null boundaries: {'; '.join(shifted)}.")
    axes = next((row.get("family_axes") for row in rows if row.get("family_axes")), None)
    q = next((row.get("family_q") for row in rows if row.get("family_q") is not None), None)
    if axes or q is not None:
        family = "; ".join(
            ([f"axes {', '.join(str(axis) for axis in axes)}"] if axes else [])
            + ([f"q = {float(q):.3g}"] if q is not None else [])
        )
        notes.append(
            f"Family selection ({family}) is separate from each row's tested-alternative verdict."
        )
    if any(row.get("family_guarantee") == "asymptotic_sequential" for row in rows):
        notes.append("The sequential family guarantee is asymptotic, not finite-sample.")
    for row in rows:
        notes.append(f"{row.get('metric')}: {result_outcome_text(row, arm=several_arms)}")
        confidence_set = complete_confidence_set_text(row)
        retained = _plain(confidence_set) if confidence_set else ""
        if retained:
            notes.append(f"{name(row)}: {retained}")
        for caveat in result_caveats([row]):
            reason = _plain(caveat).partition(": ")[2]
            if reason and reason not in retained:
                notes.append(f"{name(row)}: {reason}")
    return notes


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


def _segment_correction(analysis: Analysis) -> str | None:
    plan = getattr(analysis.experiment, "plan", None)
    return getattr(getattr(plan, "view_multiplicity", None), "correction", None)


def _explore_scope(
    analysis: Analysis,
    snapshot: DashboardSnapshot,
    breakout: tuple[str | None, str] | None,
    correction: str | None,
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
                    analysis,
                    snapshot=snapshot,
                    metric=None,
                    view=view,
                    completed_windows_only=complete,
                    breakout=breakout,
                )
            )
            if batch is not None:
                grouped: dict[str, list[Any]] = {metric: [] for metric in names}
                for row in batch:
                    if row.metric in grouped:
                        grouped[row.metric].append(row)
                batch_by_metric = {metric: type(batch)(rows) for metric, rows in grouped.items()}
        for metric in names:
            # The metric=None batch covers declared metrics only; added metrics load their own.
            entries[metric][view_key] = _explore_entry(
                analysis,
                snapshot,
                metric=metric,
                view_key=view_key,
                breakout=breakout,
                correction=correction,
                batch=batch_by_metric.get(metric) if metric in declared else None,
            )
    return entries


def _attempt(load: Callable[[], Any]) -> Any:
    """The loaded data, or ``None`` for a refusal the per-metric retry will report exactly."""
    try:
        return load()
    except CodedError:
        return None


def _explore_entry(
    analysis: Analysis,
    snapshot: DashboardSnapshot,
    *,
    metric: str,
    view_key: str,
    breakout: tuple[str | None, str] | None,
    correction: str | None,
    batch: Any,
) -> dict[str, Any]:
    view, complete = _VIEWS[view_key]
    scope = "the whole experiment" if breakout is None else f"{breakout[1]} segments"
    try:
        if batch is not None:
            data = batch
        else:
            data = load_explore(
                analysis,
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
    metric: str,
    view: ExploreView,
    complete: bool,
    breakout: tuple[str | None, str] | None,
    correction: str | None,
) -> dict[str, Any]:
    rows = [row for row in data if getattr(row, "method_role", "decision") == "decision"]
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
    data = type(data)(rows)
    figure = temporal_figure(snapshot, data, metric=metric, view=view)
    notes = _notes(
        snapshot, rows, metric=metric, view=view, complete=complete, correction=correction
    )
    gaps = unavailable_point_note(data.to_frame())
    if gaps is not None:
        notes.append(gaps)
    first, last = min(row.ds for row in rows), max(row.ds for row in rows)
    series = len({(row.dimension_value, row.group_id) for row in rows})
    caption = (
        f"{subject} · {title} · {_day(first)} → {_day(last)} · "
        f"{series} series · {monitoring_sentence(rows, view=view)}"
    )
    return {
        "html": _styled(figure),
        "caption": caption,
        "notes": notes,
        "pointCount": len(rows),
    }


def _notes(
    snapshot: DashboardSnapshot,
    rows: list[Any],
    *,
    metric: str,
    view: str,
    complete: bool,
    correction: str | None,
) -> list[str]:
    lift = view == "cumulative_lift"
    segmented = rows[0].dimension is not None
    points = [row.lift if lift else row.value for row in rows]
    levelled = [
        (row.metric, point.level)
        for row, point in zip(rows, points, strict=True)
        if point is not None and point.level
    ]
    level_text = ", ".join(
        percent_text(level) for level in sorted({level for _, level in levelled})
    )
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
    if rows[0].ds_basis == "cohort":
        notes.append(
            "Dates are exposure cohorts: each unit's own exposure date, not the observation date."
        )
    if level_text:
        notes.append(
            f"Intervals are at level {levels_by_metric(levelled, percent_text)}. "
            + (
                "They describe the lift of each point, one look at a time."
                if lift
                else "Bands are per-arm Wald intervals for each arm's value, not for the difference."
            )
        )
    if segmented:
        notes.append(_segment_note(snapshot, metric, segments, level_text, correction, lift=lift))
    if lift:
        notes.extend(_lift_notes(snapshot, rows, metric, segmented=segmented))
    return notes


def _segment_note(
    snapshot: DashboardSnapshot,
    metric: str,
    segments: list[str],
    level_text: str,
    correction: str | None,
    *,
    lift: bool,
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
    level = None if headline is None else headline.get("level")
    versus = (
        ""
        if level is None or is_missing(level)
        else f" (the headline uses {percent_text(float(level))})"
    )
    declared = f" ({correction})" if correction else ""
    return (
        f"Segment intervals carry the experiment's declared segment multiplicity{declared} across "
        f"{len(segments)} segments, at interval level {level_text}{versus}. Segment lines are "
        "pointwise and unadjusted for repeated looks over time. One significant segment is not "
        "evidence of an interaction."
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
                "dropped; the table beside the chart shows the exact latest interval."
            )
    if has_open_side(rows):
        notes.append(
            "One-sided interval: the open side has no finite bound, so only the finite side is "
            "drawn, as its own labelled line."
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
