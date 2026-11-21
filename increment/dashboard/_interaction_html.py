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
from increment.semantics.interaction_scan import SCAN_CONTRASTS

if TYPE_CHECKING:
    from increment.dashboard._theme import DashboardTheme
    from increment.semantics.interaction_scan import (
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
_CONTRAST_LABELS = {
    "focal_given_other_off": "Focal effect with other off",
    "focal_given_other_on": "Focal effect with other on",
    "interaction": "Change in focal effect (other on minus off)",
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
    "sparse_binary_events": "too few events or non-events were observed",
    "unit_namespace_mismatch": "the experiments assigned different kinds of units",
    "unsupported_arms": "the experiment did not have two assignment groups",
    "zero_cell_variance": "all observed outcomes were the same",
}


@dataclass(frozen=True, slots=True)
class InteractionTables:
    """One renderable overview and its expanded fragments keyed by pair ID."""

    overview: str
    details: Mapping[str, str]


def interaction_tables(
    scan: InteractionScan, *, metric_key: str, theme: DashboardTheme = MIDNIGHT
) -> InteractionTables:
    """Render the selected metric without refitting or changing the scan family."""
    metric = {item.name: item for item in scan.plan.metrics}[metric_key]
    rows = tuple(row for row in scan.rows if row.key.metric == metric_key)
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

    def number(value: float) -> str:
        return _number(value, binary=binary, signed=True)

    def axis_number(value: float) -> str:
        return _number(value, binary=binary, signed=True, digits=3)

    names = Counter(detail.candidate.entry.display_name for detail in scan.details)
    summary = []
    details = {}
    groups = batched(rows, len(SCAN_CONTRASTS))
    for detail, group in zip(scan.details, groups, strict=True):
        pair_id = detail.candidate.key.pair_id
        direct = group[-1]
        entry = detail.candidate.entry
        reasons = (*direct.reasons, *detail.reasons, *detail.candidate.reasons)
        status = _status(direct, detail, reasons)
        summary.append(
            _native_row(
                direct,
                pair_id,
                _pair_button(entry, pair_id, repeated=names[entry.display_name] > 1),
                status,
            )
        )
        contrast_rows = [
            _native_row(
                row,
                f"{pair_id}-{row.key.contrast}",
                f'<span data-inc-contrast="{row.key.contrast}">{_CONTRAST_LABELS[row.key.contrast]}</span>',
                status if row is direct else _status(row, detail, (*row.reasons, *detail.reasons)),
            )
            for row in group
        ]
        contrasts = interaction_native(
            contrast_rows,
            row_heading="Comparison",
            extent=extent,
            number=number,
            axis_number=axis_number,
            theme=theme,
        )
        details[pair_id] = _pair_detail(
            scan.plan.scope, metric, detail, pair_id, unit, group, contrasts
        )

    if summary:
        table = interaction_native(
            summary,
            row_heading="Other experiment",
            extent=extent,
            number=number,
            axis_number=axis_number,
            theme=theme,
        )
        overview = (
            '<p class="inc-interaction-units">Change in the focal effect with the other '
            f"experiment on rather than off. Units: {escape(unit)}.</p>{table}"
        )
    else:
        overview = (
            "<p>No concurrent experiments were in scope.</p>"
            if scan.no_candidates
            else "<p>No candidates are shown; the experiment inventory was not verified as complete.</p>"
        )
    return InteractionTables(overview, MappingProxyType(details))


def _number(value: float, *, binary: bool = False, signed: bool = False, digits: int = 8) -> str:
    if not value:
        return "0"
    scaled = value * 100 if binary else value
    # A finite raw number can exceed the float range in display units.
    display = scaled if isfinite(scaled) else Decimal(str(value)).scaleb(2)
    text = format(display, f"{'+' if signed else ''}.{digits}g")
    return f"−{text[1:]}" if text.startswith("-") else text


def _native_row(row: ScanRow, key: str, label: str, status: str) -> InteractionNativeRow:
    return InteractionNativeRow(key, label, row.estimate, row.lower, row.upper, status)


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


def _status(row: ScanRow, detail: PairDetail, reasons: tuple[ScanReason, ...]) -> str:
    if row.availability in ("unavailable", "fitted_small"):
        label = (
            "Couldn’t estimate"
            if row.availability == "unavailable"
            else "Insufficient evidence for a conclusion"
        )
        return f"{label} · {_reason_text(reasons[0], detail, locate=True)}" if reasons else label
    if row.lower is None:
        return "Estimate only · no interval"
    if row.decision == "detected":
        return "Change detected"
    if row.decision == "none_detected":
        return "None detected"
    return "No conclusion"


def _conclusion(direct: ScanRow) -> str:
    if direct.decision == "detected":
        return "The other experiment changed the focal experiment’s effect on this metric."
    if direct.decision == "none_detected":
        return "No change in the focal effect was detected. This does not establish that the effects are equal."
    if direct.availability == "fitted_small":
        return "The interval is shown, but there is insufficient evidence to conclude whether the other experiment changed the focal effect."
    return "There is no conclusion about whether the other experiment changed the focal effect."


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
    unit: str,
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
        f"<dt>Metric</dt><dd>{escape(metric.name)} ({escape(unit)})</dd>"
        f"<dt>Entry opened</dt><dd>{scope.entry_open.isoformat(sep=' ')}</dd>"
        f"<dt>Outcome window</dt><dd>{scope.start.isoformat(sep=' ')} to "
        f"{scope.end.isoformat(sep=' ')} (end excluded)</dd>"
        f"<dt>Data cutoff</dt><dd>{scope.cutoff.isoformat(sep=' ')}</dd>"
        f"<dt>Target</dt><dd>{target} <small>({escape(scope.focal.population_id)})</small></dd>"
    )
    direct = rows[-1]
    if direct.availability in ("unavailable", "fitted_small") and reasons:
        why = f"<p>{escape(_status(direct, detail, reasons))}.</p>"
    else:
        why = ""
    general = _reason_list(tuple(reason for reason in reasons if reason.cell is None), detail)
    assignments = _assignment_table(metric, detail, rows, reasons)
    return (
        f'<section class="inc-interaction-detail" data-inc-pair-detail="{pair_id}">'
        f"<p>{_conclusion(direct)}</p>{why}"
        f'<p class="inc-interaction-units">Units: {escape(unit)}.</p>'
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
