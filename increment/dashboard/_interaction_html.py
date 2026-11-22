"""Explore interaction comparisons and their bounded assignment evidence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from html import escape
from itertools import batched
from math import isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING

from increment.dashboard._native import InteractionNativeRow, interaction_native
from increment.dashboard._theme import MIDNIGHT
from increment.estimation.interaction_scan import _relative_interaction_rows
from increment.semantics.interaction_scan import SCAN_CONTRASTS

if TYPE_CHECKING:
    from increment.dashboard._native import InteractionTone
    from increment.dashboard._theme import DashboardTheme
    from increment.semantics.interaction_scan import (
        CombinationResult,
        CombinationRow,
        InteractionScan,
        InventoryEntry,
        PairDetail,
        ScanReason,
        ScanRow,
        ScanScope,
    )
    from increment.semantics.overlap import CellId, OverlapCell, OverlapMetric

_CELLS: tuple[CellId, ...] = ("00", "10", "01", "11")
_SWAPPED: tuple[CellId, ...] = ("00", "01", "10", "11")
_ASSIGNMENTS = {
    "00": "both off",
    "10": "focal on, other off",
    "01": "focal off, other on",
    "11": "both on",
}
_REASONS = {
    "allocation_switch": "an assignment changed during the study",
    "archive_missing": "the assignment record was not captured",
    "incompatible_window": "the experiments did not overlap for the full study window",
    "missing_outcomes": "outcomes are missing",
    "no_target": "nobody qualified for this comparison",
    "pair_evidence_missing": "pair evidence was not captured",
    "pending_outcomes": "outcomes are still pending",
    "singleton_cell": "only one outcome was observed",
    "small_cell": "too few outcomes were observed",
    "relative_baseline_nonpositive": "the baseline is zero or negative",
    "relative_baseline_uncertain": "the baseline is too uncertain for a finite relative interval",
    "relative_moments_unavailable": "the outcome data needed for relative lift are missing",
    "sparse_binary_events": "too few events or non-events were observed",
    "unit_namespace_mismatch": "the experiments assigned different kinds of units",
    "unsupported_arms": "the experiment did not have two assignment groups",
    "zero_cell_variance": "all observed outcomes were the same",
}

_COMBINATION_REASONS = {
    **_REASONS,
    "empty_cell": "Missing group",
    "joint_feed_missing": "Joint assignments were not captured",
    "joint_registration_unverified": "Registration was not verified",
    "incomplete_inventory": "The experiment list was not verified",
}


@dataclass(frozen=True, slots=True)
class InteractionTables:
    """One renderable overview and its expanded fragments keyed by pair ID."""

    overview: str
    details: Mapping[str, str]
    has_interaction: bool
    detected_pairs: frozenset[str]
    absolute_unit: str


def interaction_tables(
    scan: InteractionScan,
    *,
    metric_key: str,
    theme: DashboardTheme = MIDNIGHT,
    relative: bool = False,
) -> InteractionTables:
    """Render the selected metric without refitting or changing the scan family."""
    metric = {item.name: item for item in scan.plan.metrics}[metric_key]
    rows = (
        _relative_interaction_rows(scan, metric_key)
        if relative
        else tuple(row for row in scan.rows if row.key.metric == metric_key)
    )
    extent = (
        max(
            (
                abs(value)
                for row in rows
                for value in (row.estimate, row.lower, row.upper)
                if value is not None
            ),
            default=0.0,
        )
        or 1.0
    )
    binary = metric.kind == "binary_mean"
    unit = "percentage points" if binary else metric.unit
    metric_name = escape(metric.name.replace("_", " "))
    focal = escape(scan.plan.scope.focal.entry.display_name)
    basis = "lift" if relative else f"effect on {metric_name}"
    effect_suffix = "%" if relative else " pp" if binary else f" {escape(unit)}"
    change_suffix = " pp" if relative else f" pp of {metric_name}" if binary else effect_suffix

    def number(value: float) -> str:
        places = 1 if relative else 2 if binary else None
        return _number(value, percent=relative or binary, signed=True, places=places)

    def axis_number(value: float) -> str:
        return _number(value, percent=relative or binary, signed=True)

    names = Counter(detail.candidate.entry.display_name for detail in scan.details)
    summary = []
    details = {}
    groups = batched(rows, len(SCAN_CONTRASTS))
    for detail, group in zip(scan.details, groups, strict=True):
        pair_id = detail.candidate.key.pair_id
        direct = group[-1]
        entry = detail.candidate.entry
        other = escape(entry.display_name)
        labels = (
            f"{focal}’s {basis} with {other} off",
            f"{focal}’s {basis} with {other} on",
            f"How {other} changed {focal}’s {basis}",
        )
        reasons = (*direct.reasons, *detail.reasons, *detail.candidate.reasons)
        status = _status(direct, detail, reasons)
        summary.append(
            _native_row(
                direct,
                pair_id,
                _pair_button(entry, pair_id, repeated=names[entry.display_name] > 1),
                status,
                _effect_tone(direct, metric.better),
                suffix=change_suffix,
            )
        )
        contrast_rows = [
            _native_row(
                row,
                key=f"{pair_id}-{row.key.contrast}",
                label=f'<span data-inc-contrast="{row.key.contrast}">{label}</span>',
                status=status
                if row is direct
                else _status(row, detail, (*row.reasons, *detail.reasons), better=metric.better),
                tone=_effect_tone(row, metric.better),
                suffix=change_suffix if row is direct else effect_suffix,
            )
            for row, label in zip(group, labels, strict=True)
        ]
        contrasts = interaction_native(
            contrast_rows,
            row_heading="Comparison",
            estimate_heading="Estimate",
            extent=extent,
            number=number,
            axis_number=axis_number,
            theme=theme,
        )
        details[pair_id] = _pair_detail(scan.plan.scope, metric, detail, pair_id, group, contrasts)

    if summary:
        overview = interaction_native(
            summary,
            row_heading="Other experiment",
            estimate_heading="Change in lift" if relative else f"Change in effect on {metric_name}",
            extent=extent,
            number=number,
            axis_number=axis_number,
            theme=theme,
        )
    else:
        overview = (
            "<p>No concurrent experiments were in scope.</p>"
            if scan.no_candidates
            else "<p>No candidates are shown; the experiment inventory was not verified as complete.</p>"
        )
    return InteractionTables(
        overview,
        MappingProxyType(details),
        any(row.key.contrast == "interaction" and row.lower is not None for row in rows),
        frozenset(row.key for row in summary if row.tone == "detected"),
        unit,
    )


def _number(
    value: float, *, percent: bool = False, signed: bool = False, places: int | None = None
) -> str:
    if not value:
        return "0"
    scaled = value * 100 if percent else value
    # A finite raw number can exceed the float range in display units.
    display = scaled if isfinite(scaled) else Decimal(str(value)).scaleb(2)
    sign = "+" if signed else ""
    if places is not None and 10.0**-places <= abs(display) < 1000:
        text = format(display, f"{sign}.{places}f").rstrip("0").rstrip(".")
    else:
        text = format(display, f"{sign}.3g")
    return f"−{text[1:]}" if text.startswith("-") else text


def _native_row(
    row: ScanRow,
    key: str,
    label: str,
    status: str,
    tone: InteractionTone = "neutral",
    *,
    suffix: str = "",
) -> InteractionNativeRow:
    return InteractionNativeRow(
        key, label, row.estimate, row.lower, row.upper, status, tone, suffix
    )


def _pair_button(entry: InventoryEntry, pair_id: str, *, repeated: bool) -> str:
    identity = entry.identity
    reference = f"{identity.target} / {identity.experiment_id} / {identity.revision}"
    name = escape(entry.display_name)
    qualification = (
        f' <small class="inc-interaction-pair-reference">{escape(reference)}</small>'
        if repeated
        else ""
    )
    return (
        '<button type="button" class="inc-interaction-pair" '
        f'data-inc-pair-id="{pair_id}" aria-label="{name} ({escape(reference)}) pair evidence">'
        f"{name}{qualification}</button>"
    )


def _assignment(detail: PairDetail, raw_cell: CellId) -> str:
    source = _SWAPPED if detail.focal_side == "B" else _CELLS
    return _ASSIGNMENTS[_CELLS[source.index(raw_cell)]]


def _reason_text(reason: ScanReason, detail: PairDetail, *, locate: bool = False) -> str:
    if reason.code == "empty_cell":
        assignment = (
            _assignment(detail, reason.cell) if reason.cell is not None else "to this group"
        )
        return f"nobody was assigned {assignment}"
    message = _REASONS.get(reason.code, reason.code.replace("_", " "))
    if locate and reason.cell is not None:
        return f"{message} ({_assignment(detail, reason.cell)})"
    return message


def _effect_tone(row: ScanRow, better: str) -> InteractionTone:
    if row.decision != "detected":
        return "neutral"
    if row.key.contrast == "interaction":
        return "detected"
    rises = row.lower is not None and row.lower > 0
    return "favorable" if rises == (better == "higher") else "unfavorable"


def _status(
    row: ScanRow, detail: PairDetail, reasons: tuple[ScanReason, ...], *, better: str = ""
) -> str:
    if row.availability in ("unavailable", "fitted_small"):
        label = (
            "Couldn’t estimate"
            if row.availability == "unavailable"
            else "Insufficient evidence for a conclusion"
        )
        reason = next(
            (reason for reason in reasons if reason.code.startswith("relative_")),
            reasons[0] if reasons else None,
        )
        return f"{label} · {_reason_text(reason, detail, locate=True)}" if reason else label
    if row.lower is None:
        return "Estimate only · no interval"
    if row.decision == "detected":
        if row.key.contrast == "interaction":
            return (
                "Higher with other on"
                if row.lower is not None and row.lower > 0
                else "Lower with other on"
            )
        return "Favorable" if _effect_tone(row, better) == "favorable" else "Unfavorable"
    if row.decision == "none_detected":
        return "None detected"
    return "No conclusion"


def _experiment(entry: InventoryEntry) -> str:
    identity = entry.identity
    return (
        f"{escape(entry.display_name)} <small>({escape(identity.target)} / "
        f"{escape(identity.experiment_id)} / {escape(identity.revision)})</small>"
    )


def _pair_detail(
    scope: ScanScope,
    metric: OverlapMetric,
    detail: PairDetail,
    pair_id: str,
    rows: tuple[ScanRow, ...],
    contrasts: str,
) -> str:
    reasons = tuple(
        dict.fromkeys(
            (
                *rows[-1].reasons,
                *detail.reasons,
                *detail.candidate.reasons,
                *(reason for row in rows[:-1] for reason in row.reasons),
            )
        )
    )
    target = "Eligible for both experiments"
    if scope.focal.target == "jointly_eligible_and_enrolled":
        target += " and enrolled in both"
    context = (
        f"<dt>Focal experiment</dt><dd>{_experiment(scope.focal.entry)}</dd>"
        f"<dt>Other experiment</dt><dd>{_experiment(detail.candidate.entry)}</dd>"
        f"<dt>Metric</dt><dd>{escape(metric.name)} ({escape(metric.unit)})</dd>"
        f"<dt>Entry opened</dt><dd>{scope.entry_open.isoformat(sep=' ')}</dd>"
        f"<dt>Outcome window</dt><dd>{scope.start.isoformat(sep=' ')} to "
        f"{scope.end.isoformat(sep=' ')} (end excluded)</dd>"
        f"<dt>Data cutoff</dt><dd>{scope.cutoff.isoformat(sep=' ')}</dd>"
        f"<dt>Target</dt><dd>{target} <small>({escape(scope.focal.population_id)})</small></dd>"
    )
    general = _reason_list(tuple(reason for reason in reasons if reason.cell is None), detail)
    assignments = _assignment_table(metric, detail, rows, reasons)
    return (
        f'<section class="inc-interaction-detail" data-inc-pair-detail="{pair_id}">'
        '<div class="inc-interaction-contrasts" role="region" aria-label="Pair estimates" '
        f'tabindex="0">{contrasts}</div>{general}{assignments}<details class="inc-interaction-scope">'
        f"<summary>Experiment scope and dates</summary><dl>{context}</dl></details></section>"
    )


def _reason_list(reasons: tuple[ScanReason, ...], detail: PairDetail) -> str:
    if not reasons:
        return ""
    notes = "".join(
        f'<li data-inc-reason="{escape(reason.code)}">{escape(_reason_text(reason, detail))}</li>'
        for reason in reasons
    )
    return f"<ul>{notes}</ul>"


def _assignment_table(
    metric: OverlapMetric,
    detail: PairDetail,
    rows: tuple[ScanRow, ...],
    reasons: tuple[ScanReason, ...],
) -> str:
    cells = {cell.cell: cell for cell in detail.cells if cell.metric == metric.name}
    counts = {cell.cell: cell for row in rows for cell in row.cells}
    binary = metric.kind == "binary_mean"
    heading = "Observed events" if binary else f"Observed mean ({escape(metric.unit)})"
    source = _SWAPPED if detail.focal_side == "B" else _CELLS
    body = []
    for position, raw_cell in zip(_CELLS, source, strict=True):
        cell = cells.get(raw_cell)
        count = cell if cell is not None else counts.get(raw_cell)
        matched = tuple(reason for reason in reasons if reason.cell == raw_cell)
        notes = _reason_list(matched, detail)
        assigned = observed = "—"
        if count is not None:
            assigned, observed = f"{count.n_assigned:,}", f"{count.n_observed:,}"
        body.append(
            f'<tr data-inc-cell="{position}"><th scope="row">{_ASSIGNMENTS[position]}</th>'
            f"<td>{assigned}</td><td>{observed}</td><td>{_observed_value(cell, binary=binary)}</td>"
            f"<td>{notes or ('Counts not captured' if count is None else '—')}</td></tr>"
        )
    return (
        '<div class="table-surface"><table class="inc-interaction-cells">'
        "<caption>Outcomes by assignment</caption>"
        '<thead><tr><th scope="col">Assignment</th><th scope="col">Assigned</th>'
        f'<th scope="col">Observed</th><th scope="col">{heading}</th>'
        f'<th scope="col">Notes</th></tr></thead><tbody>{"".join(body)}</tbody></table></div>'
    )


def _observed_value(cell: OverlapCell | None, *, binary: bool) -> str:
    if cell is not None:
        if binary and cell.n_success is not None:
            return f"{cell.n_success:,}"
        if not binary and cell.anchor is not None and cell.mean_delta is not None:
            # Decode the captured mean from its stored origin and offset.
            return _number(cell.anchor + cell.mean_delta)
    return "—"


def _combination_row(row: CombinationRow, label: str, *, better: str) -> InteractionNativeRow:
    status = "; ".join(
        dict.fromkeys(
            _COMBINATION_REASONS.get(reason.code, "The comparison could not be fully assessed")
            for reason in row.reasons
        )
    )
    tone: InteractionTone = "neutral"
    if row.kind == "effect" and row.decision == "detected" and row.estimate is not None:
        tone = "favorable" if (row.estimate > 0) == (better == "higher") else "unfavorable"
    return InteractionNativeRow(
        f"{row.kind}:{row.assignment or ''}",
        escape(label),
        row.estimate,
        row.lower,
        row.upper,
        status,
        tone,
        "",
        row.assignment,
    )


def _joint_assignment_table(result: CombinationResult, names: tuple[str, ...]) -> str:
    if not result.cells:
        return ""
    heading = (
        "".join(f"<th scope='col'>{escape(name)}</th>" for name in names)
        + "<th scope='col'>Assigned</th><th scope='col'>Observed</th>"
        "<th scope='col'>Missing</th><th scope='col'>Pending</th>"
    )
    body = "".join(
        "<tr>"
        + "".join(f"<td>{'On' if bit == '1' else 'Off'}</td>" for bit in cell.cell)
        + f"<td>{cell.n_assigned:,}</td><td>{cell.n_observed:,}</td>"
        f"<td>{cell.n_missing:,}</td><td>{cell.n_pending:,}</td></tr>"
        for cell in result.cells
    )
    return (
        "<details class='metric-detail'><summary>Assignment groups</summary>"
        "<div class='table-surface inc-combination-groups' tabindex='0'>"
        f"<table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table></div></details>"
    )


def combination_packet(
    result: CombinationResult,
    scan: InteractionScan,
    theme: DashboardTheme,
) -> dict[str, str]:
    """Render both questions from saved contrasts, without additional estimation."""
    entries = (
        scan.plan.scope.focal.entry,
        *(candidate.entry for candidate in scan.plan.candidates),
    )
    counts = Counter(entry.display_name for entry in entries)
    names_by_id = {
        entry.identity: (
            entry.display_name
            if counts[entry.display_name] == 1
            else f"{entry.display_name} ({' / '.join(entry.identity.sort_key)})"
        )
        for entry in entries
    }
    focal = names_by_id[scan.plan.scope.focal.entry.identity]
    names = tuple(names_by_id[candidate] for candidate in result.declaration.candidates)
    last = names[-1]
    rate = result.metric.kind == "binary_mean"
    unit = "pp" if rate else result.metric.unit

    def number(value: float) -> str:
        return _number(value, percent=rate, signed=True, places=2)

    extent = (
        max(
            (
                abs(value)
                for row in result.rows
                for value in (row.estimate, row.lower, row.upper)
                if value is not None
            ),
            default=0.0,
        )
        or 1.0
    )
    effects = [
        _combination_row(row, "", better=result.metric.better)
        for row in result.rows
        if row.kind == "effect"
    ]
    contrast_rows = tuple(row for row in result.rows if row.kind != "effect")
    labels = (f"Without {last}", f"With {last}", "Change (with − without)")
    change = [
        _combination_row(row, label, better=result.metric.better)
        for row, label in zip(contrast_rows, labels, strict=True)
    ]
    chart = interaction_native(
        effects,
        row_heading="",
        estimate_heading=unit,
        extent=extent,
        number=number,
        axis_number=number,
        theme=theme,
        assignment_columns=names,
    )
    detail = interaction_native(
        change,
        row_heading="Comparison",
        estimate_heading=unit,
        extent=extent,
        number=number,
        axis_number=number,
        theme=theme,
    )
    diff = contrast_rows[-1]
    explanation = (
        "The full change couldn’t be assessed."
        if diff.decision is None
        else "No clear change. That does not rule out interactions among the other experiments."
        if diff.decision == "none_detected"
        else "The interval for the change excludes zero."
    )
    method = (
        f"{result.family_size} comparisons were registered across sets and metrics, including "
        "unavailable comparisons. The adjusted intervals are approximate and apply at the fixed cutoff."
    )
    return {
        "id": "|".join(
            candidate.key.pair_id
            for candidate in scan.plan.candidates
            if candidate.entry.identity in result.declaration.candidates
        ),
        "label": " + ".join(names),
        "slice": last if len(names) > 2 else "",
        "effects": (
            f"<h4>When does {escape(focal)} help or hurt?</h4>"
            f"<p>Effect of adding {escape(focal)} to each combination.</p>"
            "<p aria-label='Filled dot: on. Hollow dot: off.'>● On &nbsp; ○ Off</p>"
            f"<div class='table-surface inc-combination-chart' tabindex='0'>{chart}</div>"
        ),
        "change": (
            f"<h4>Did {escape(last)} change how the other experiments work together?</h4>"
            f"<p>{explanation}</p><div class='table-surface' tabindex='0'>{detail}</div>"
        ),
        "groups": _joint_assignment_table(result, (focal, *names)),
        "method": method,
    }
