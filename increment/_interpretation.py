"""Evidence-only interpretation for scoped readout collections."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from increment.estimation.readout_types import (
    CellKey,
    ReadoutMetadata,
    render_readout_failure,
)
from increment.estimation.results import (
    _selection_qualification,
    row_chance_to_beat,
    row_verdict,
)


@dataclass(frozen=True, slots=True)
class InterpretedCell:
    key: CellKey
    required: bool
    visible: bool
    sampling_available: bool
    point_available: bool
    failure_code: str | None
    failure_message: str | None
    failure_context: Any
    advisories: tuple[Any, ...]
    posterior_available: bool | None
    posterior_reason_code: str | None
    posterior_reason_context: Any

    @property
    def available(self) -> bool:
        return self.sampling_available


@dataclass(frozen=True, slots=True)
class InterpretedGroup:
    source_snapshot_id: str | None
    population: str | None
    control_group: str | None
    treatment_group: str | None
    integrity_status: str
    integrity_code: str | None
    integrity_context: Any
    decision_complete: bool | None
    roster_complete: bool | None
    roster_source: str | None
    source_cells: tuple[InterpretedCell, ...]
    required_cells: tuple[InterpretedCell, ...]
    available_cells: tuple[InterpretedCell, ...]
    unavailable_cells: tuple[InterpretedCell, ...]
    hidden_cells: tuple[InterpretedCell, ...]
    retained_families: tuple[Any, ...]
    partial: bool
    advisories: tuple[Any, ...]
    visible_rows: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class ReadoutInterpretation:
    groups: tuple[InterpretedGroup, ...]
    metadata_known: bool
    partial: bool


@dataclass(frozen=True, slots=True)
class EffectEstimate:
    point: float | None
    lower: float | None
    upper: float | None
    alpha: float | None
    value_scale: str
    reason: str | None = None
    geometry: str | None = None
    intervals: tuple[tuple[float | None, float | None], ...] = ()
    qualification: str | None = None


def _unscoped_cell(row) -> InterpretedCell:
    failure_code = getattr(row, "failure_code", None)
    failure_context = getattr(row, "failure_context", None)
    failure_message = render_readout_failure(
        failure_code,
        failure_context,
        identity={"metric": getattr(row, "metric", None)},
    )
    if failure_code is not None and failure_message is None:
        failure_message = f"{failure_code}: {_json(failure_context) or '{}'}"
    elif failure_message is None and getattr(row, "unavailable", None) is not None:
        failure_message = f"Unavailable: {row.unavailable}"
    return InterpretedCell(
        key=CellKey.from_row(row),
        required=False,
        visible=True,
        sampling_available=_has_sampling_result(row),
        point_available=_has_point(row),
        failure_code=failure_code,
        failure_message=failure_message,
        failure_context=failure_context,
        advisories=(),
        posterior_available=getattr(row, "posterior_available", None),
        posterior_reason_code=getattr(row, "posterior_reason_code", None),
        posterior_reason_context=getattr(row, "posterior_reason_context", None),
    )


def _unscoped_group(source, population, control, treatment, rows):
    cells = tuple(_unscoped_cell(row) for row in rows)
    return InterpretedGroup(
        source_snapshot_id=source,
        population=population,
        control_group=control,
        treatment_group=treatment,
        decision_complete=None,
        roster_complete=None,
        roster_source=None,
        integrity_status="unknown",
        integrity_code=None,
        integrity_context=None,
        source_cells=cells,
        required_cells=(),
        available_cells=tuple(cell for cell in cells if cell.available),
        unavailable_cells=tuple(cell for cell in cells if not cell.available),
        hidden_cells=(),
        retained_families=(),
        partial=False,
        advisories=(),
        visible_rows=tuple(rows),
    )


def _interpret_unscoped(visible_rows):
    grouped: dict[tuple[str | None, str | None, str | None, str | None], list[Any]] = {}
    for row in visible_rows:
        key = (
            getattr(row, "source_snapshot_id", None),
            getattr(row, "analysis_population", None),
            getattr(row, "control_group", None),
            getattr(row, "group_id", getattr(row, "treatment_group", None)),
        )
        grouped.setdefault(key, []).append(row)
    groups = tuple(
        _unscoped_group(source, population, control, treatment, rows)
        for (source, population, control, treatment), rows in sorted(
            grouped.items(), key=lambda item: repr(item[0])
        )
    )
    if not groups:
        groups = (
            InterpretedGroup(
                source_snapshot_id=None,
                population=None,
                control_group=None,
                treatment_group=None,
                decision_complete=None,
                roster_complete=None,
                roster_source=None,
                integrity_status="unknown",
                integrity_code=None,
                integrity_context=None,
                source_cells=(),
                required_cells=(),
                available_cells=(),
                unavailable_cells=(),
                hidden_cells=(),
                retained_families=(),
                partial=False,
                advisories=(),
                visible_rows=(),
            ),
        )
    return ReadoutInterpretation(groups, False, False)


def _failure_and_availability(row, record):
    failure = record.failure
    sampling = record.sampling
    if failure is not None:
        failure_code, failure_context = failure.code, failure.context
    elif row is not None and getattr(row, "sampling_available", None) is False:
        failure_code = getattr(row, "failure_code", None) or getattr(
            row, "sampling_reason_code", None
        )
        failure_context = getattr(row, "failure_context", None) or getattr(
            row, "sampling_reason_context", None
        )
    elif sampling is not None and not sampling.available:
        failure_code, failure_context = sampling.reason_code, sampling.reason_context
    elif row is not None:
        failure_code = getattr(row, "failure_code", None)
        failure_context = getattr(row, "failure_context", None)
    else:
        failure_code = failure_context = None
    sampling_available = (
        False
        if row is not None and getattr(row, "sampling_available", None) is False
        else sampling.available
        if sampling is not None
        else _has_sampling_result(row)
    )
    return failure_code, failure_context, sampling_available


def interpret_readout(rows, metadata: ReadoutMetadata | None) -> ReadoutInterpretation:
    """Interpret retained collection evidence without recomputing inference."""
    visible_rows = tuple(rows)
    if metadata is None:
        return _interpret_unscoped(visible_rows)

    row_by_key: dict[tuple[str, CellKey], Any] = {}
    for row in visible_rows:
        try:
            key = CellKey.from_row(row)
        except (AttributeError, TypeError, ValueError):
            continue
        source = getattr(row, "source_snapshot_id", None)
        if source is not None:
            row_by_key[(source, key)] = row

    groups: list[InterpretedGroup] = []
    for source_id, source in sorted(metadata.scope.by_source.items()):
        integrity_by_population = {item.analysis_population: item for item in source.integrity}
        roster_by_population = {item.analysis_population: item for item in source.rosters}
        for population in sorted({cell.analysis_population for cell in source.cells}):
            roster = roster_by_population.get(population)
            by_comparison: dict[tuple[str | None, str | None], list[CellKey]] = {}
            for cell in source.cells:
                if cell.analysis_population != population:
                    continue
                treatment = cell.group_id if cell.kind != "contrast" else cell.treatment_group
                row = row_by_key.get((source_id, cell))
                control = (
                    None
                    if _is_daily_mean_cell(cell) or _is_daily_value(row)
                    else (
                        cell.control_group
                        or getattr(row, "control_group", None)
                        or getattr(roster, "control_group", None)
                    )
                )
                by_comparison.setdefault((control, treatment), []).append(cell)
            for (control, treatment), cells in sorted(
                by_comparison.items(), key=lambda item: repr(item[0])
            ):
                integrity = integrity_by_population.get(population)
                required_keys = {cell for cell in source.decision_cells if cell in cells}
                interpreted: list[InterpretedCell] = []
                for cell in cells:
                    record = metadata.record(source_id, cell)
                    row = row_by_key.get((source_id, cell))
                    failure_code, failure_context, sampling_available = _failure_and_availability(
                        row, record
                    )
                    failure_message = render_readout_failure(
                        failure_code,
                        failure_context,
                        identity={"metric": cell.metric},
                    )
                    if failure_code is not None and failure_message is None:
                        failure_message = f"{failure_code}: {_json(failure_context) or '{}'}"
                    elif (
                        failure_message is None
                        and row is not None
                        and getattr(row, "unavailable", None) is not None
                    ):
                        failure_message = f"Unavailable: {row.unavailable}"
                    posterior = record.posterior
                    interpreted.append(
                        InterpretedCell(
                            key=cell,
                            required=cell in required_keys,
                            visible=row is not None,
                            sampling_available=sampling_available,
                            point_available=_has_point(row),
                            failure_code=failure_code,
                            failure_message=failure_message,
                            failure_context=failure_context,
                            advisories=record.advisories,
                            posterior_available=None if posterior is None else posterior.available,
                            posterior_reason_code=None
                            if posterior is None
                            else posterior.reason_code,
                            posterior_reason_context=None
                            if posterior is None
                            else posterior.reason_context,
                        )
                    )
                all_cells = tuple(interpreted)
                required = tuple(item for item in all_cells if item.required)
                available = tuple(item for item in all_cells if item.available)
                unavailable = tuple(item for item in all_cells if not item.available)
                hidden = tuple(item for item in all_cells if not item.visible)
                families = tuple(
                    family
                    for family in metadata.scope.families
                    if family.source_snapshot_id == source_id
                    and family.analysis_population == population
                    and any(member in cells for member in family.members)
                )
                rows_here = tuple(
                    row_by_key[(source_id, cell)]
                    for cell in cells
                    if (source_id, cell) in row_by_key
                )
                advisories = tuple(
                    sorted(
                        {
                            (
                                advisory.code,
                                json.dumps(advisory.context, sort_keys=True, default=str),
                            ): advisory
                            for item in all_cells
                            for advisory in item.advisories
                        }.values(),
                        key=lambda item: (
                            item.code,
                            json.dumps(item.context, sort_keys=True, default=str),
                        ),
                    )
                )
                groups.append(
                    InterpretedGroup(
                        source_snapshot_id=source_id,
                        population=population,
                        control_group=control,
                        treatment_group=treatment,
                        integrity_status="unknown" if integrity is None else integrity.status.value,
                        integrity_code=None if integrity is None else integrity.code,
                        integrity_context=None if integrity is None else integrity.context,
                        decision_complete=source.decision_complete_by_population.get(population),
                        roster_complete=None if roster is None else roster.complete,
                        roster_source=None if roster is None else roster.source,
                        source_cells=all_cells,
                        required_cells=required,
                        available_cells=available,
                        unavailable_cells=unavailable,
                        hidden_cells=hidden,
                        retained_families=families,
                        partial=metadata.partial,
                        advisories=advisories,
                        visible_rows=rows_here,
                    )
                )
    return ReadoutInterpretation(tuple(groups), True, metadata.partial)


def readout_evidence_columns(
    interpretation: ReadoutInterpretation,
    cell_or_row,
    *,
    source_snapshot_id: str | None = None,
) -> dict[str, object]:
    """Return portable scalar/JSON qualification columns for a source-qualified cell."""
    row = None
    if isinstance(cell_or_row, CellKey):
        cell = cell_or_row
    else:
        row = cell_or_row
        cell = CellKey.from_row(row)
        source_snapshot_id = source_snapshot_id or getattr(row, "source_snapshot_id", None)
    matches = [
        (group, item)
        for group in interpretation.groups
        if source_snapshot_id is None or group.source_snapshot_id == source_snapshot_id
        for item in group.source_cells
        if item.key == cell
    ]
    if source_snapshot_id is None and len(matches) != 1:
        matches = []
    if matches:
        group, item = matches[0]
        if row is None:
            row = next(
                (
                    candidate
                    for candidate in group.visible_rows
                    if CellKey.from_row(candidate) == cell
                    and (
                        source_snapshot_id is None
                        or getattr(candidate, "source_snapshot_id", None) == source_snapshot_id
                    )
                ),
                None,
            )
        status = getattr(row, "multiplicity_status", None)
        return {
            "source_snapshot_id": group.source_snapshot_id,
            "analysis_population": group.population,
            "control_group": group.control_group,
            "treatment_group": group.treatment_group,
            "comparison": (
                None
                if group.treatment_group is None
                else f"{group.treatment_group} vs {group.control_group}"
                if group.control_group is not None
                else f"arm {group.treatment_group}"
            ),
            "integrity_status": group.integrity_status,
            "integrity_code": group.integrity_code,
            "integrity_context": _json(group.integrity_context),
            "required_cell_count": len(group.required_cells),
            "available_cell_count": sum(entry.available for entry in group.required_cells),
            "source_cell_count": len(group.source_cells),
            "unavailable_cell_count": len(group.unavailable_cells),
            "hidden_cell_count": len(group.hidden_cells),
            "decision_scope_complete": group.decision_complete,
            "roster_complete": group.roster_complete,
            "roster_source": group.roster_source,
            "sampling_available": item.available,
            "point_available": item.point_available,
            "posterior_available": item.posterior_available,
            "posterior_reason_code": item.posterior_reason_code,
            "posterior_reason_context": _json(item.posterior_reason_context),
            "failure_code": item.failure_code,
            "failure_context": _json(item.failure_context),
            "failure_message": item.failure_message,
            "unavailable_cells": _json(
                [
                    {
                        "cell": entry.key.model_dump(mode="json"),
                        "code": entry.failure_code,
                        "context": _jsonable(entry.failure_context),
                        "message": entry.failure_message,
                        "available": entry.available,
                        "hidden_by_filter": False,
                    }
                    for entry in group.unavailable_cells
                    if entry.visible
                ]
            ),
            "hidden_cells": _json(
                [
                    {
                        "cell": entry.key.model_dump(mode="json"),
                        "code": entry.failure_code,
                        "context": _jsonable(entry.failure_context),
                        "message": entry.failure_message,
                        "available": entry.available,
                        "hidden_by_filter": True,
                    }
                    for entry in group.hidden_cells
                ]
            ),
            "advisories": _json(
                [advisory.model_dump(mode="json") for advisory in group.advisories]
            ),
            "view_partial": group.partial,
            "multiplicity_status": status
            or (
                "unknown"
                if not group.retained_families
                else "adjusted"
                if any(family.family is not None for family in group.retained_families)
                else "exploratory_unadjusted"
            ),
            "retained_family": _json(
                [family.model_dump(mode="json") for family in group.retained_families]
            ),
        }
    return {
        "source_snapshot_id": source_snapshot_id,
        "analysis_population": getattr(row, "analysis_population", None),
        "control_group": getattr(row, "control_group", None),
        "treatment_group": getattr(row, "treatment_group", None) or getattr(row, "group_id", None),
        "comparison": (
            f"{getattr(row, 'treatment_group', None) or getattr(row, 'group_id', None)} vs "
            f"{getattr(row, 'control_group', None)}"
            if getattr(row, "control_group", None) is not None
            else f"arm {getattr(row, 'group_id', None)}"
            if getattr(row, "group_id", None) is not None
            else None
        ),
        "integrity_status": "unknown",
        "integrity_code": None,
        "integrity_context": None,
        "required_cell_count": 0,
        "available_cell_count": 0,
        "source_cell_count": 0,
        "unavailable_cell_count": 0,
        "hidden_cell_count": 0,
        "decision_scope_complete": None,
        "roster_complete": None,
        "roster_source": None,
        "sampling_available": False,
        "point_available": False,
        "posterior_available": None,
        "posterior_reason_code": None,
        "posterior_reason_context": None,
        "failure_code": None,
        "failure_context": None,
        "failure_message": None,
        "unavailable_cells": "[]",
        "hidden_cells": "[]",
        "advisories": "[]",
        "view_partial": interpretation.partial,
        "multiplicity_status": "unknown",
        "retained_family": "[]",
    }


_ROLE_ORDER = {"primary": 0, "guardrail": 1, "secondary": 2}


def _row_display_order(row):
    role = getattr(row, "role", None) or "outcome"
    return (
        _ROLE_ORDER.get(role, 3),
        getattr(row, "metric", ""),
        getattr(row, "group_id", None) or getattr(row, "treatment_group", None) or "",
        getattr(row, "method", ""),
    )


def _is_daily_value(row) -> bool:
    return hasattr(row, "value") and hasattr(row, "ds") and not hasattr(row, "lift")


def _is_daily_mean_cell(cell: CellKey) -> bool:
    return (
        cell.kind in {"arm", "segment"}
        and cell.ds is not None
        and cell.method is None
        and cell.estimand is None
    )


def _row_identity(row) -> tuple[str, ...]:
    details = []
    for attribute, label in (
        ("dimension_value", getattr(row, "dimension", None) or "segment"),
        ("ds", "date"),
        ("method", "method"),
        ("method_role", "method role"),
        ("estimand", "estimand"),
    ):
        value = getattr(row, attribute, None)
        if value is not None:
            details.append(f"{label} {value}")
    if getattr(row, "discovery", None) is not None:
        details.append("selected" if row.discovery else "not selected")
    return tuple(details)


def _group_comparison(group, *, metadata_known: bool, include_dates: bool = False) -> str:
    daily = (
        bool(group.source_cells)
        and all(_is_daily_mean_cell(cell.key) for cell in group.source_cells)
    ) or (bool(group.visible_rows) and all(_is_daily_value(row) for row in group.visible_rows))
    if daily:
        comparison = f"arm {group.treatment_group or 'unknown'}"
        dates = sorted({str(row.ds) for row in group.visible_rows})
        if include_dates and dates:
            comparison += f"; date {', '.join(dates)}"
        return comparison
    if group.control_group is not None and group.treatment_group is not None:
        return f"{group.treatment_group} vs {group.control_group}"
    if group.treatment_group is not None:
        control = (
            "control identity unknown"
            if not metadata_known
            else "control identity unavailable in retained scope"
        )
        return f"treatment {group.treatment_group}; {control}"
    return (
        "comparison unknown"
        if not metadata_known
        else "comparison not represented by retained scope"
    )


def _render_group_label(group, *, metadata_known: bool, show_source: bool) -> str:
    comparison = _group_comparison(group, metadata_known=metadata_known, include_dates=True)
    source = (
        f" [{_display_source(group.source_snapshot_id)}]"
        if show_source and group.source_snapshot_id is not None
        else ""
    )
    return f"{group.population or 'population unknown'} — {comparison}{source}"


def render_brief_qualification(row, family=None, *, alpha=None) -> str | None:
    """Return compact confidence and inference labels from retained result evidence."""
    labels = []
    estimate = getattr(row, "lift", None) or getattr(row, "estimate", None)
    display_alpha = alpha if alpha is not None else getattr(estimate, "alpha", None)
    family_definition = None if family is None else family.family
    procedure = getattr(row, "family_procedure", None) or getattr(
        family_definition, "correction", None
    )
    family_size = getattr(row, "family_size", None)
    if family_size is None and family is not None:
        family_size = len(family.members)
    if display_alpha is not None:
        labels.append(f"{1 - display_alpha:.1%} confidence")
    if family_size is not None:
        procedure_label = (
            str(procedure).replace("_", " ").title() if procedure is not None else "family"
        )
        labels.append(f"{family_size}-member {procedure_label}")
    if getattr(getattr(row, "sequential_result", None), "validity_regime", None) == (
        "asymptotic_sequential"
    ):
        labels.append("asymptotic sequential")
    relative_set = getattr(row, "relative_confidence_set", None)
    if relative_set is not None and relative_set.qualification == "working_approximation":
        labels.append("asymptotic joint confidence set")
    return "; ".join(labels) if labels else None


def render_readout_brief(cells, interpretation: ReadoutInterpretation) -> tuple[str, ...]:
    """Project cells and retained disclosures into source-qualified brief lines."""
    rows_by_identity = {
        (row.source_snapshot_id, CellKey.from_row(row)): row
        for group in interpretation.groups
        for row in group.visible_rows
    }
    source_ids = {cell.source_snapshot_id for cell in cells if cell.source_snapshot_id is not None}
    comparisons = {
        (group.source_snapshot_id, CellKey.from_row(row)): (
            group.control_group,
            group.treatment_group,
        )
        for group in interpretation.groups
        for row in group.visible_rows
    }
    lines = []
    for cell in cells:
        key = cell.identity.key
        label = _cell_label(key)
        comparison = comparisons.get((cell.source_snapshot_id, key))
        if comparison is not None and all(comparison):
            label += f", comparison={comparison[1]} vs {comparison[0]}"
        if len(source_ids) > 1 and cell.source_snapshot_id is not None:
            label += f", source={_display_source(cell.source_snapshot_id)}"
        status = cell.components["sampling"][0]
        row = rows_by_identity.get((cell.source_snapshot_id, key))
        text = _brief_summary_text(cell.summary)
        text += f"; status={status.state}"
        if status.code is not None:
            text += f"; code={status.code}"
        reasons = tuple(
            dict.fromkeys(
                reason
                for reason in (status.reason, _reason(status.context))
                if reason and reason != "reason unavailable"
            )
        )
        if reasons:
            text += f"; reason={'; '.join(reasons)}"
        if row is not None:
            discovery = getattr(row, "discovery", None)
            if discovery is not None:
                text += "; " + ("selected" if discovery else "not selected")
            selection = _selection_qualification(row)
            if selection is not None:
                text += f"; {selection}"
            qualification = render_brief_qualification(
                row, cell.family, alpha=None if cell.summary is None else cell.summary.alpha
            )
            if qualification is not None:
                text += f"; {qualification}"
        lines.append(f"{label}: {text}")

    lines.extend(_brief_disclosure_lines(interpretation))
    lines.extend(f"Warning: {line}" for line in render_readout_qualifications(interpretation))
    for group in interpretation.groups:
        lines.extend(
            line
            for line in _render_group_scope(group, group.visible_rows)
            if line.startswith("Integrity:")
            or line.startswith("SRM not checked")
            or line.startswith("SRM not run")
        )
    return tuple(lines)


def _brief_summary_text(summary) -> str:
    if summary is None:
        return "unavailable"
    text = f"{summary.point!r} [{summary.lower!r}, {summary.upper!r}] ({summary.value_scale}"
    if summary.alpha is not None:
        text += f", {1 - summary.alpha:.1%} confidence"
    if summary.geometry is not None:
        text += f", {summary.geometry} set {summary.intervals!r}"
    if summary.reason is not None:
        text += f", {summary.reason}"
    if summary.qualification is not None:
        text += f", {summary.qualification}"
    return text + ")"


def _brief_disclosure_lines(interpretation: ReadoutInterpretation) -> tuple[str, ...]:
    source_ids = {group.source_snapshot_id for group in interpretation.groups}
    lines = []
    for group in interpretation.groups:
        warning_codes = {
            cell.failure_code
            for cell in (*group.hidden_cells, *group.unavailable_cells)
            if cell.failure_code is not None
        }
        required_keys = {cell.key for cell in group.required_cells}
        warning_codes.update(
            f"readout.hidden_required_cell:{cell.key.metric}"
            for cell in group.hidden_cells
            if cell.key in required_keys
        )
        if group.integrity_status not in {"not_rejected", "not_applicable"}:
            warning_codes.add(f"integrity.{group.integrity_status}")
        warning_codes.update(advisory.code for advisory in group.advisories)
        if group.partial:
            warning_codes.add("readout.partial_view")
        qualifications = []
        if len(source_ids) > 1 and group.source_snapshot_id is not None:
            qualifications.append(f"source={_display_source(group.source_snapshot_id)}")
        if group.control_group is not None and group.treatment_group is not None:
            qualifications.append(f"comparison={group.treatment_group} vs {group.control_group}")
        suffix = "" if not qualifications else f" ({'; '.join(qualifications)})"
        lines.extend(f"Warning: {code}{suffix}" for code in sorted(warning_codes))
    if not interpretation.groups and not interpretation.metadata_known:
        lines.append("Warning: integrity.unknown")
    if interpretation.partial and not any(group.partial for group in interpretation.groups):
        lines.append("Warning: readout.partial_view")
    return tuple(lines)


def render_readout_text(interpretation: ReadoutInterpretation) -> str:
    """Render metric evidence first, followed by relevant qualifications."""
    if not interpretation.groups:
        return "Scope retained; no source-qualified cells recorded."
    lines: list[str] = []
    show_source = len({group.source_snapshot_id for group in interpretation.groups}) > 1
    for group in interpretation.groups:
        lines.append(
            _render_group_label(
                group,
                metadata_known=interpretation.metadata_known,
                show_source=show_source,
            )
        )
        visible_rows = sorted(group.visible_rows, key=_row_display_order)
        lines.extend(_render_group_rows(group, visible_rows))
        lines.extend(_render_group_scope(group, visible_rows))
    return "\n".join(lines)


def _render_integrity_disclosure(group) -> str | None:
    if group.integrity_status == "failed":
        return f"Integrity: failed ({group.integrity_code or 'mismatch'})."
    if group.integrity_status == "not_rejected":
        return "Integrity: not_rejected — no detected mismatch; not proof of valid randomization."
    if (
        group.integrity_status == "not_checked_missing_declaration"
        and group.integrity_code == "integrity.allocation_missing_or_nonconstant"
    ):
        return (
            "SRM not checked; declare allocation={...} and "
            'allocation_scheme="independent" on Randomized or Encouragement.'
        )
    if (
        group.integrity_status == "not_checked_missing_declaration"
        and group.integrity_code == "integrity.allocation_scheme_missing"
    ):
        return (
            'SRM not checked; declare allocation_scheme="independent" on '
            "Randomized or Encouragement; it must describe how units were actually assigned."
        )
    if group.integrity_status == "not_checked_missing_declaration":
        reason = group.integrity_code or _reason(group.integrity_context)
        return (
            f"SRM not checked ({reason}); declare allocation={{...}} and "
            'allocation_scheme="independent" matching the assignment actually used.'
        )
    if group.integrity_code == "integrity.triggered_population_not_checked":
        return (
            "SRM not checked on the triggered population; post-trigger counts cannot "
            "validate assignment. Check the assigned population instead."
        )
    if group.integrity_status == "not_checked_missing_counts":
        return "SRM not checked; assignment counts are not recorded."
    if group.integrity_status == "unsupported_assignment":
        return "SRM not run for this assignment design."
    if group.integrity_status.startswith("not_checked"):
        reason = group.integrity_code or group.integrity_status
        return f"SRM not checked ({reason})."
    return None


def _render_group_scope(group, visible_rows) -> list[str]:
    lines = []
    integrity = _render_integrity_disclosure(group)
    if integrity is not None:
        lines.append(integrity)
    if group.partial:
        lines.append("View: partial; do not treat visible cells as a complete decision.")
    if group.decision_complete is False:
        lines.append("Decision scope completeness: incomplete.")
    if group.roster_complete is False:
        lines.append("Population roster completeness: incomplete.")
    if group.required_cells:
        required_available = sum(item.available for item in group.required_cells)
        if required_available < len(group.required_cells):
            lines.append(
                f"Required cells: {len(group.required_cells)}; available: {required_available}."
            )
    for item in group.unavailable_cells:
        if item.visible:
            lines.append(
                f"Unavailable {item.key.metric}: "
                f"{item.failure_message or item.failure_code or 'reason unavailable'}"
            )
    for item in group.hidden_cells:
        if item.failure_message:
            lines.append(f"Cell {item.key.metric} hidden by filter; {item.failure_message}")
        else:
            lines.append(f"Cell {item.key.metric} hidden by filter.")
    families_by_id = {family.family_id: family for family in group.retained_families}
    multiplicity = set()
    for row in visible_rows:
        role = getattr(row, "role", None) or "outcome"
        family = families_by_id.get(getattr(row, "family_id", None))
        status = getattr(row, "multiplicity_status", None)
        if status is None and family is not None:
            status = "adjusted" if family.family is not None else "exploratory_unadjusted"
        if status is not None or family is not None:
            multiplicity.add(
                (role, status or "unknown", "retained scope" if family is None else family.name)
            )
    for role, status, family_name in sorted(
        multiplicity, key=lambda item: (_ROLE_ORDER.get(item[0], 3), item[2], item[1])
    ):
        lines.append(f"Multiplicity ({role}): {status}; family {family_name}.")
    if not multiplicity and group.retained_families:
        labels = sorted({family.name for family in group.retained_families})
        lines.append(f"Multiplicity: retained scope; family {', '.join(labels)}.")
    for advisory in group.advisories:
        lines.append(f"Advisory {advisory.code}: {advisory.message}")
    return lines


def _render_group_rows(group, visible_rows) -> list[str]:
    lines = []
    advisory_codes = {item.code for item in group.advisories}
    cells_by_key = {item.key: item for item in group.source_cells}
    methods_by_metric: dict[str, set[tuple[str | None, str | None]]] = {}
    for row in visible_rows:
        methods_by_metric.setdefault(row.metric, set()).add(
            (getattr(row, "method", None), getattr(row, "method_role", None))
        )
    for row in visible_rows:
        if not _is_daily_value(row):
            lines.append(
                _decision_line(
                    row,
                    control_group=group.control_group,
                    treatment_group=group.treatment_group,
                )
            )
        if getattr(row, "posterior_available", False) is True:
            posterior_estimate = getattr(row, "posterior_estimate", None)
            chance_to_beat = row_chance_to_beat(row)
            lines.append(
                f"{row.metric} posterior: estimate={posterior_estimate!r}; "
                f"chance_to_beat={chance_to_beat!r}"
            )
        cell = cells_by_key.get(CellKey.from_row(row))
        if cell is not None and cell.posterior_available is False:
            lines.append(
                f"{row.metric} posterior unavailable {cell.posterior_reason_code}: "
                f"{_reason(cell.posterior_reason_context)}"
            )
        if getattr(row, "note", None) and not any(code in row.note for code in advisory_codes):
            lines.append(f"{row.metric} qualification: {row.note}")
        selection_qualification = _selection_qualification(row)
        if selection_qualification is not None:
            lines.append(f"{row.metric} selection qualification: {selection_qualification}.")
        lines.extend(
            _render_effect(
                row,
                qualify_method=len(methods_by_metric[row.metric]) > 1,
            )
        )
        identity = _row_identity(row)
        if identity:
            lines.append(f"{row.metric} result identity: {'; '.join(identity)}.")
        if getattr(row, "role", None) == "secondary":
            discovery = getattr(row, "discovery", None)
            selection = (
                "selected"
                if discovery is True
                else "not selected"
                if discovery is False
                else "not recorded"
            )
            lines.append(f"{row.metric} selection: {selection}; separate from interval evidence.")
    return lines


def _decision_line(row, *, control_group: str | None, treatment_group: str | None) -> str:
    projection = project_effect(row)
    control = control_group if control_group is not None else "control identity unknown"
    treatment = treatment_group if treatment_group is not None else "treatment identity unknown"
    parts = [f"{row.metric}: {control} → {treatment}"]
    for name, label in (
        ("n_control", "n control"),
        ("n_treat", "n treatment"),
        ("mean_control", "mean control"),
        ("mean_treat", "mean treatment"),
    ):
        value = getattr(row, name, None)
        if value is not None:
            parts.append(f"{label}={value:.6g}" if name.startswith("mean_") else f"{label}={value}")
    factor = 100 if projection.value_scale == "relative" else 1
    if projection.point is not None:
        suffix = "%" if factor == 100 else ""
        parts.append(f"effect={projection.point * factor:.6g}{suffix}")
        if projection.lower is not None or projection.upper is not None:
            low = "−∞" if projection.lower is None else f"{projection.lower * factor:.6g}"
            high = "+∞" if projection.upper is None else f"{projection.upper * factor:.6g}"
            parts.append(f"interval=[{low}, {high}]{suffix}")
    if projection.qualification is not None:
        parts.append(f"scale qualification={projection.qualification}")
    estimand = getattr(row, "estimand", None)
    if estimand is not None:
        parts.append(f"estimand={estimand}")
    dimension = getattr(row, "dimension", None)
    dimension_value = getattr(row, "dimension_value", None)
    if dimension is not None and dimension_value is not None:
        parts.append(f"{dimension}={dimension_value}")
    if hasattr(row, "control_group") and hasattr(row, "estimate"):
        from increment.estimation.contrast import contrast_stat_sig

        verdict = contrast_stat_sig(row)
    else:
        verdict = row_verdict(row)
    parts.append(f"stat_sig={verdict if verdict is not None else 'unavailable'}")
    return "; ".join(parts)


def _display_source(source_id: str | None) -> str | None:
    if source_id is None:
        return None
    prefix, separator, digest = source_id.partition(":")
    if not separator:
        return source_id[:12]
    return f"{prefix}:{digest[:12]}"


def render_readout_qualifications(interpretation: ReadoutInterpretation) -> tuple[str, ...]:
    """Return compact, source-attributed qualifications for a rendered table header."""
    if not interpretation.groups:
        return ("Scope and integrity unknown.",)
    comparisons = [
        _group_comparison(group, metadata_known=interpretation.metadata_known)
        for group in interpretation.groups
    ]
    attribution_count = len(
        {
            (group.population, group.source_snapshot_id, comparison)
            for group, comparison in zip(interpretation.groups, comparisons, strict=True)
        }
    )
    repeat_attribution = attribution_count > 1
    bullets: list[str] = []
    for group, comparison in zip(interpretation.groups, comparisons, strict=True):
        attribution = [
            value
            for value in (
                group.population,
                _display_source(group.source_snapshot_id),
                comparison if repeat_attribution else None,
            )
            if value
        ]
        suffix = "" if not attribution else f" [{', '.join(attribution)}]"

        def add_bullet(text: str, group_suffix: str = suffix) -> None:
            attribution_suffix = group_suffix if repeat_attribution or not bullets else ""
            bullets.append(text + attribution_suffix)

        add_bullet(f"Comparison: {comparison}")
        integrity = f"Integrity: {group.integrity_status}"
        if group.integrity_code:
            integrity += f" ({group.integrity_code})"
        if group.integrity_status == "not_rejected":
            integrity += "; no detected mismatch, not proof of valid randomization"
        add_bullet(integrity)
        decision = (
            "unknown"
            if group.decision_complete is None
            else "complete"
            if group.decision_complete
            else "incomplete"
        )
        roster = (
            "unknown"
            if group.roster_complete is None
            else "complete"
            if group.roster_complete
            else "incomplete"
        )
        if group.roster_source:
            roster += f" ({group.roster_source})"
        counts = ""
        if group.required_cells:
            required_available = sum(item.available for item in group.required_cells)
            counts = (
                f"; Required cells: {len(group.required_cells)}; "
                f"available: {required_available}; hidden: {len(group.hidden_cells)}; "
                f"unavailable: {len(group.unavailable_cells)}"
            )
        add_bullet(
            f"Scope: decision {decision}; roster {roster}"
            f"{'; partial view' if group.partial else ''}{counts}"
        )
        for item in group.unavailable_cells:
            if not item.visible:
                continue
            identity = _cell_label(item.key)
            reason = item.failure_message or item.failure_code or "reason unavailable"
            add_bullet(f"Unavailable {identity}: {reason}")
        for item in group.hidden_cells:
            identity = _cell_label(item.key)
            reason = item.failure_message or (
                "availability unknown" if not item.available else "result retained in scope"
            )
            add_bullet(f"Hidden by filter {identity}: {reason}")
        family_by_id = {family.family_id: family for family in group.retained_families}
        multiplicity = set()
        for row in group.visible_rows:
            role = getattr(row, "role", None) or "outcome"
            family = family_by_id.get(getattr(row, "family_id", None))
            status = getattr(row, "multiplicity_status", None)
            if status is None and family is not None:
                status = "adjusted" if family.family is not None else "exploratory_unadjusted"
            if status is not None or family is not None:
                multiplicity.add(
                    (role, status or "unknown", "retained scope" if family is None else family.name)
                )
        if multiplicity:
            for role, status, family_name in sorted(
                multiplicity,
                key=lambda item: (_ROLE_ORDER.get(item[0], 3), item[2], item[1]),
            ):
                add_bullet(f"Multiplicity ({role}): {status}; family {family_name}")
        elif group.retained_families:
            family_names = ", ".join(sorted({family.name for family in group.retained_families}))
            add_bullet(f"Multiplicity: retained scope; family {family_names}")
        for advisory in group.advisories:
            context = _json(advisory.context)
            detail = "" if not context else f"; context {context}"
            add_bullet(f"Advisory {advisory.code}: {advisory.message}{detail}")
        cells_by_key = {item.key: item for item in group.source_cells}
        advisory_codes = {advisory.code for advisory in group.advisories}
        for row in group.visible_rows:
            cell = cells_by_key.get(CellKey.from_row(row))
            if cell is not None and cell.posterior_available is False:
                add_bullet(
                    f"Posterior unavailable {cell.posterior_reason_code}: "
                    f"{_reason(cell.posterior_reason_context)}"
                )
            note = getattr(row, "note", None)
            if note and not any(code in note for code in advisory_codes):
                add_bullet(f"Qualification: {note}")
    return tuple(bullets)


def _cell_label(cell: CellKey) -> str:
    parts = [cell.metric]
    if cell.kind == "contrast":
        if cell.control_group is not None:
            parts.append(f"control={cell.control_group}")
        if cell.treatment_group is not None:
            parts.append(f"treatment={cell.treatment_group}")
    elif cell.group_id is not None:
        parts.append(f"arm={cell.group_id}")
    if cell.dimension is not None:
        parts.append(f"dimension={cell.dimension}")
    if cell.dimension_value is not None:
        parts.append(f"segment={cell.dimension_value}")
    for attribute, label in (
        (cell.method, "method"),
        (cell.method_role, "role"),
        (cell.estimand, "estimand"),
        (cell.analysis_population, "population"),
        (cell.value_scale, "scale"),
        (cell.inference, "inference"),
        (cell.alternative, "alternative"),
        (cell.ds, "date"),
        (cell.ds_basis, "date basis"),
    ):
        if attribute is not None:
            parts.append(f"{label}={attribute}")
    return " / ".join(parts)


def _retained_intervals(row, estimate, scale: str):
    if (
        getattr(row, "failure_code", None) is not None
        or getattr(row, "sampling_available", None) is False
    ):
        return (), None
    sequential = getattr(row, "sequential_result", None)
    if sequential is not None:
        bounds = sequential.bounds
        if bounds.status == "unavailable":
            return (), None
        components = getattr(bounds, "components", None)
        ratio_intervals = (
            tuple((component.lower, component.upper) for component in components)
            if components is not None
            else ((bounds.lower, bounds.upper),)
        )
        return tuple(
            (
                None if lower is None else lower - 1,
                None if upper is None else upper - 1,
            )
            for lower, upper in ratio_intervals
        ), getattr(bounds, "level", None)
    confidence_set = getattr(row, "confidence_set", None)
    if confidence_set is not None:
        interval = getattr(confidence_set, "additive" if scale == "absolute" else "relative", None)
        if interval is not None:
            lower = interval.lower
            upper = interval.upper
            if (
                getattr(lower, "status", None) == "undefined"
                or getattr(upper, "status", None) == "undefined"
            ):
                return (), getattr(confidence_set, "level", None)
            return (
                (
                    getattr(lower, "value", None)
                    if getattr(lower, "status", None) == "finite"
                    else None,
                    getattr(upper, "value", None)
                    if getattr(upper, "status", None) == "finite"
                    else None,
                ),
            ), getattr(confidence_set, "level", None)
    if scale == "absolute":
        if hasattr(row, "control_group"):
            if estimate is None or (estimate.lb is None and estimate.ub is None):
                return (), getattr(estimate, "level", None)
            return ((estimate.lb, estimate.ub),), getattr(estimate, "level", None)
        if getattr(row, "null_abs", None) is not None:
            lower = getattr(row, "abs_lb", None)
            upper = getattr(row, "abs_ub", None)
        elif estimate is not None:
            lower, upper = estimate.lb, estimate.ub
        else:
            lower = upper = None
        if lower is None and upper is None:
            return (), getattr(estimate, "level", None)
        return ((lower, upper),), getattr(estimate, "level", None)
    relative_set = getattr(row, "relative_confidence_set", None)
    if relative_set is not None:
        return tuple(relative_set.intervals), getattr(relative_set, "level", None)
    binomial = getattr(row, "binomial_set", None)
    if binomial is not None:
        return ((binomial.lower, binomial.upper),), binomial.level
    if estimate is None or (estimate.lb is None and estimate.ub is None):
        return (), None
    return ((estimate.lb, estimate.ub),), getattr(estimate, "level", None)


def _interval_interpretation(row, estimate, scale: str, *, contrast: bool):
    null_abs = getattr(row, "null_abs", None)
    null_lift = getattr(row, "null_lift", None)
    absolute_native = scale == "absolute" and null_abs is None
    null = null_abs if null_abs is not None else 0 if absolute_native or contrast else null_lift
    alternative = getattr(row, "alternative", None)
    if null is None or alternative is None:
        return None
    intervals, level = _retained_intervals(row, estimate, scale)
    if not intervals:
        if null_abs is not None:
            return "Interpretation: unavailable; the additive interval is not retained."
        return None
    contains_null = any(
        (lower is None or lower <= null) and (upper is None or null <= upper)
        for lower, upper in intervals
    )
    sequential_result = getattr(row, "sequential_result", None)
    if sequential_result is not None and not sequential_result.rejects():
        return (
            "Interpretation: does not reject; the retained sequential procedure has not rejected."
        )
    if sequential_result is None and bool(getattr(row, "low_reliability", False)):
        return (
            "Interpretation: inconclusive; the nominal interval is not trustworthy "
            "because this row is flagged low reliability."
        )
    if sequential_result is None and getattr(row, "stat_sig", None) is False and not contains_null:
        return (
            "Interpretation: does not reject; the retained statistical verdict "
            "does not support an inferential claim."
        )
    if contains_null:
        return "Interpretation: does not reject; inconclusive, not evidence of no effect."
    level_note = "" if level is None else f" at {level:.1%} confidence"
    if null == 0:
        outcome = f"excludes zero{level_note}"
    else:
        outcome = f"rejects the null{level_note} (margin met)"

    alternative_direction = {"greater": "increase", "less": "decrease"}.get(alternative)
    if alternative_direction is not None:
        sides = []
        for lower, upper in intervals:
            if lower is not None and lower > null:
                sides.append("above")
            elif upper is not None and upper < null:
                sides.append("below")
            else:
                sides.append("contains")
        supported_side = "above" if alternative_direction == "increase" else "below"
        if not all(side == supported_side for side in sides):
            if set(sides) == {"above", "below"}:
                return (
                    "Interpretation: confidence set spans both sides of the tested null; "
                    "effect direction not established."
                )
            opposite_side = "below" if supported_side == "above" else "above"
            return (
                f"Interpretation: does not support the declared alternative; "
                f"interval lies entirely {opposite_side} the tested null."
            )

    zero_directions = set()
    for lower, upper in intervals:
        if lower is not None and lower > 0:
            zero_directions.add("increase")
        elif upper is not None and upper < 0:
            zero_directions.add("decrease")
        else:
            zero_directions.add("undetermined")
    if zero_directions == {"increase"}:
        direction_note = "effect direction supported (increase)"
    elif zero_directions == {"decrease"}:
        direction_note = "effect direction supported (decrease)"
    else:
        direction_note = "effect direction not established"
    return f"Interpretation: {outcome}; {direction_note}."


def render_effect_interpretation(row) -> str | None:
    """Render a restrained interpretation from a row's retained interval and null."""
    if hasattr(row, "value") and not hasattr(row, "lift") and not hasattr(row, "estimate"):
        return None
    contrast = hasattr(row, "control_group")
    scale = (
        "absolute"
        if contrast
        or getattr(row, "null_abs", None) is not None
        or getattr(row, "value_scale", "relative") == "absolute"
        else "relative"
    )
    estimate = getattr(row, "lift", None) or getattr(row, "estimate", None)
    return _interval_interpretation(row, estimate, scale, contrast=contrast)


def _reason(context):
    if context is None:
        return "reason unavailable"
    if isinstance(context, dict):
        return str(context.get("reason", context.get("display", context)))
    return (
        str(context.get("reason", context.get("display", context)))
        if hasattr(context, "get")
        else str(context)
    )


def _json(value):
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    elif hasattr(value, "items"):
        value = {key: _jsonable(item) for key, item in value.items()}
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def _jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "items"):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _sequential_lift_bound(value, *, lower: bool) -> float | None:
    if value is None:
        return None
    exact = Fraction(value) - 1
    try:
        result = float(exact)
    except OverflowError:
        return None
    if not math.isfinite(result):
        return None
    if (lower and Fraction(result) > exact) or (not lower and Fraction(result) < exact):
        result = math.nextafter(result, -math.inf if lower else math.inf)
    return result if math.isfinite(result) else None


def project_effect(row) -> EffectEstimate:
    """Choose one retained estimate on the row's requested scale."""
    sequential = (
        row.require_asymptotic_sequential_result()
        if getattr(row, "inference", None) == "asymptotic_mean"
        and getattr(row, "sequential_result", None) is not None
        and hasattr(row, "require_asymptotic_sequential_result")
        else None
    )
    if sequential is not None:
        bounds = sequential.bounds
        available = bounds.available
        point = row.lift.value if sequential.point_reason is None and row.lift is not None else None
        disconnected = available and len(bounds.components) > 1
        return EffectEstimate(
            point,
            None
            if not available or disconnected
            else _sequential_lift_bound(bounds.lower, lower=True),
            None
            if not available or disconnected
            else _sequential_lift_bound(bounds.upper, lower=False),
            float(sequential.decision_alpha),
            getattr(row, "value_scale", "relative"),
            sequential.point_reason
            or (
                "disconnected sequential confidence set"
                if disconnected
                else None
                if available
                else bounds.reason
            ),
            "disconnected" if disconnected else None,
            tuple(
                (
                    _sequential_lift_bound(item.lower, lower=True),
                    _sequential_lift_bound(item.upper, lower=False),
                )
                for item in bounds.components
            )
            if disconnected
            else (),
        )
    if (
        getattr(row, "inference", None) == "always_valid"
        and getattr(row, "sequential_result", None) is not None
        and hasattr(row, "require_sequential_result")
    ):
        sequential = row.require_sequential_result()
        bounds = sequential.bounds
        lower = _sequential_lift_bound(bounds.lower, lower=True)
        upper = _sequential_lift_bound(bounds.upper, lower=False)
        return EffectEstimate(
            row.lift.value if sequential.point_reason is None and row.lift is not None else None,
            lower,
            upper,
            float(sequential.decision_alpha),
            getattr(row, "value_scale", "relative"),
            sequential.point_reason,
            intervals=() if bounds.empty else ((lower, upper),),
        )

    contrast = hasattr(row, "control_group") and hasattr(row, "estimate")
    requested = (
        "absolute"
        if contrast
        else getattr(row, "value_scale", "absolute" if hasattr(row, "control") else "relative")
    )
    lift = getattr(row, "lift", None)
    relative_set = getattr(row, "relative_confidence_set", None)
    if requested == "relative" and relative_set is not None:
        intervals = relative_set.intervals
        lower, upper = intervals[0] if len(intervals) == 1 else (None, None)
        point = lift.value if lift is not None else None
        reason = None
        if point is None:
            reason = (
                getattr(row, "relative_unavailable_reason", None)
                or relative_set.point_unavailable_reason
                or getattr(row, "relative_error_code", None)
            )
        if relative_set.geometry == "unavailable":
            reason = reason or relative_set.reason
        return EffectEstimate(
            point,
            lower,
            upper,
            relative_set.alpha,
            "relative",
            reason,
            relative_set.geometry,
            intervals,
        )

    binomial = getattr(row, "binomial_set", None)
    if requested == "relative" and binomial is not None:
        point = None if lift is None else lift.value
        reason = (
            None
            if point is not None
            else (
                getattr(row, "relative_unavailable_reason", None)
                or getattr(row, "failure_code", None)
            )
        )
        return EffectEstimate(
            point,
            binomial.lower,
            binomial.upper,
            binomial.alpha,
            "relative",
            reason,
            binomial.geometry,
            ((binomial.lower, binomial.upper),),
        )

    if contrast:
        estimate = row.estimate
        return EffectEstimate(estimate.value, estimate.lb, estimate.ub, estimate.alpha, "absolute")

    if requested == "absolute":
        absolute = getattr(row, "abs_diff", None)
        if (
            absolute is None
            and getattr(row, "value_scale", None) == "absolute"
            and lift is not None
        ):
            return EffectEstimate(
                lift.value,
                lift.lb,
                lift.ub,
                lift.alpha,
                "absolute",
                getattr(row, "relative_unavailable_reason", None),
            )
        if absolute is not None:
            return EffectEstimate(
                absolute,
                row.abs_lb,
                row.abs_ub,
                row.abs_alpha,
                "absolute",
                getattr(row, "absolute_error_code", None),
            )
        if lift is not None and getattr(row, "value_scale", "relative") == "relative":
            reason = getattr(row, "absolute_error_code", None) or "absolute interval unavailable"
            return EffectEstimate(
                lift.value,
                lift.lb,
                lift.ub,
                lift.alpha,
                "relative",
                None,
                qualification=f"requested absolute scale unavailable; showing relative ({reason})",
            )

    if requested == "relative" and lift is not None:
        absolute = getattr(row, "abs_diff", None)
        if (
            absolute is not None
            and lift.lb is None
            and lift.ub is None
            and (
                getattr(row, "abs_lb", None) is not None or getattr(row, "abs_ub", None) is not None
            )
        ):
            reason = getattr(row, "relative_error_code", None) or getattr(
                row, "relative_unavailable_reason", None
            )
            return EffectEstimate(
                absolute,
                row.abs_lb,
                row.abs_ub,
                row.abs_alpha,
                "absolute",
                reason,
                qualification=(
                    "requested relative scale unavailable; showing absolute "
                    f"({reason or 'relative interval unavailable'})"
                ),
            )
        return EffectEstimate(
            lift.value,
            lift.lb,
            lift.ub,
            lift.alpha,
            "relative",
            getattr(row, "relative_unavailable_reason", None)
            or getattr(row, "relative_error_code", None),
        )

    absolute = getattr(row, "abs_diff", None)
    if absolute is not None:
        reason = getattr(row, "relative_unavailable_reason", None) or getattr(
            row, "relative_error_code", None
        )
        return EffectEstimate(
            absolute,
            row.abs_lb,
            row.abs_ub,
            row.abs_alpha,
            "absolute",
            reason,
            qualification=f"requested relative scale unavailable; showing absolute ({reason or 'relative interval unavailable'})",
        )
    reason = (
        getattr(row, "relative_unavailable_reason", None)
        or getattr(row, "relative_error_code", None)
        or getattr(row, "failure_code", None)
    )
    return EffectEstimate(None, None, None, None, requested, reason)


def _has_point(row) -> bool:
    if row is None:
        return False
    return (
        getattr(row, "lift", None) is not None
        or getattr(row, "estimate", None) is not None
        or getattr(row, "value", None) is not None
    )


def _has_sampling_result(row) -> bool:
    if row is None or getattr(row, "failure_code", None) is not None:
        return False
    if _has_point(row):
        return True
    if any(
        getattr(row, name, None) is not None
        for name in (
            "binomial_set",
            "relative_confidence_set",
            "confidence_set",
            "sequential_result",
        )
    ):
        return True
    return any(getattr(row, name, None) is not None for name in ("abs_diff", "abs_lb", "abs_ub"))


def _format_bound(value, *, lower=False):
    return ("−∞" if lower else "+∞") if value is None else f"{value:.6g}"


def _endpoint_text(endpoint, *, lower: bool):
    status = getattr(endpoint, "status", None)
    value = getattr(endpoint, "value", None)
    reason = getattr(endpoint, "reason", None)
    if status == "finite":
        return f"{value:.6g}"
    if status == "unbounded":
        return "−∞" if lower else "+∞"
    if status == "undefined":
        return f"undefined ({reason or 'reason unavailable'})"
    return _format_bound(endpoint)


def _render_effect(row, *, qualify_method: bool = False):
    lines = []
    role = getattr(row, "role", None) or "outcome"
    if _is_daily_value(row):
        value = getattr(row, "value", None)
        if value is not None:
            effect = (
                f"{row.metric} (absolute mean, arm {row.group_id}, date {row.ds}): "
                f"{value.value:.6g}"
            )
            if value.lb is not None or value.ub is not None:
                low = "−∞" if value.lb is None else f"{value.lb:.6g}"
                high = "+∞" if value.ub is None else f"{value.ub:.6g}"
                effect += f", set [{low}, {high}]"
                if value.level is not None:
                    effect += f" at {value.level:.3g} confidence"
            lines.append(effect + ".")
        return lines
    contrast = hasattr(row, "control_group")
    projection = project_effect(row)
    scale = projection.value_scale
    label = (
        "absolute contrast" if contrast else "absolute" if scale == "absolute" else "relative lift"
    )
    if projection.point is not None:
        method = getattr(row, "method", None)
        method_role = getattr(row, "method_role", None)
        method_label = (
            f", method {method or 'unknown'}, method role {method_role or 'unknown'}"
            if qualify_method
            else ""
        )
        effect = f"{row.metric} ({role}, {label}{method_label}): {projection.point:.6g}"
        if projection.lower is not None or projection.upper is not None:
            low = "−∞" if projection.lower is None else f"{projection.lower:.6g}"
            high = "+∞" if projection.upper is None else f"{projection.upper:.6g}"
            effect += f", set [{low}, {high}]"
            if projection.alpha is not None:
                effect += f" at {1 - projection.alpha:.3g} confidence"
        lines.append(effect + ".")
    if projection.qualification is not None:
        lines.append(f"{row.metric} scale qualification: {projection.qualification}.")
    interpretation = render_effect_interpretation(row)
    if interpretation is not None:
        lines.append(f"{row.metric} {interpretation[0].lower()}{interpretation[1:]}")
    relative_reason = getattr(row, "relative_unavailable_reason", None)
    if relative_reason:
        lines.append(f"{row.metric} relative unavailable: {relative_reason}.")
    if getattr(row, "abs_diff", None) is not None:
        low = getattr(row, "abs_lb", None)
        high = getattr(row, "abs_ub", None)
        typed_additive = getattr(getattr(row, "confidence_set", None), "additive", None)
        if low is not None and high is not None:
            lines.append(
                f"{row.metric} additive effect: {row.abs_diff:.6g}, set [{low:.6g}, {high:.6g}]."
            )
        elif typed_additive is not None:
            # The typed confidence set below carries the endpoint states.
            lines.append(f"{row.metric} additive effect: {row.abs_diff:.6g}.")
        else:
            lines.append(
                f"{row.metric} additive effect: {row.abs_diff:.6g}; additive interval unavailable."
            )

    lines.extend(_render_confidence_geometry(row))
    lines.extend(_render_test_null(row, contrast=contrast, scale=scale))
    return lines


def _render_confidence_geometry(row) -> list[str]:
    lines = []
    binomial = getattr(row, "binomial_set", None)
    if binomial is not None:
        high = "+∞" if binomial.upper is None else f"{binomial.upper:.6g}"
        lines.append(
            f"{row.metric} binomial set {binomial.geometry}: "
            f"[{binomial.lower:.6g}, {high}] at {binomial.level:.3g} confidence."
        )
    relative = getattr(row, "relative_confidence_set", None)
    if relative is not None:
        intervals = (
            " ∪ ".join(
                f"[{_format_bound(lower, lower=True)}, {_format_bound(upper)}]"
                for lower, upper in relative.intervals
            )
            or "∅"
        )
        lines.append(
            f"{row.metric} relative confidence set: "
            f"{getattr(relative, 'geometry', 'unknown')} {intervals}."
        )
    sequential = getattr(row, "sequential_result", None)
    if sequential is not None:
        bounds = sequential.bounds
        components = getattr(bounds, "components", None)
        intervals = (
            " ∪ ".join(
                f"[{_format_bound(component.lower, lower=True)}, {_format_bound(component.upper)}]"
                for component in components
            )
            if components is not None
            else f"[{_format_bound(bounds.lower, lower=True)}, {_format_bound(bounds.upper)}]"
        ) or "∅"
        lines.append(f"{row.metric} sequential ratio confidence set {bounds.status}: {intervals}.")
    confidence_set = getattr(row, "confidence_set", None)
    if confidence_set is not None:
        geometry = getattr(confidence_set, "geometry", None)
        if geometry is not None:
            lines.append(
                f"{row.metric} confidence set {geometry}; "
                f"qualification {confidence_set.qualification}."
            )
        else:
            for scale_name in ("relative", "additive"):
                interval = getattr(confidence_set, scale_name, None)
                if interval is not None:
                    lines.append(
                        f"{row.metric} {scale_name} confidence interval: "
                        f"[{_endpoint_text(interval.lower, lower=True)}, "
                        f"{_endpoint_text(interval.upper, lower=False)}]; "
                        f"qualification {confidence_set.qualification}."
                    )
    return lines


def _render_test_null(row, *, contrast: bool, scale: str) -> list[str]:
    null_abs = getattr(row, "null_abs", None)
    null_lift = getattr(row, "null_lift", None)
    null = null_abs if null_abs is not None else null_lift
    if not getattr(row, "alternative", None) or null is None:
        return []
    unit = (
        "absolute units"
        if null_abs is not None or contrast
        else "relative lift"
        if null_lift is not None and getattr(row, "value_scale", None) == "relative"
        else "absolute units"
        if scale == "absolute" or getattr(row, "value_scale", None) == "absolute"
        else "relative lift"
    )
    return [f"{row.metric} test alternative: {row.alternative}; null ({unit}): {null:.6g}."]
