"""Evidence-only interpretation for scoped readout collections."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from increment.estimation.readout_types import (
    CellKey,
    ReadoutMetadata,
    render_readout_failure,
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


def _unscoped_cell(row) -> InterpretedCell:
    failure_code = getattr(row, "failure_code", None)
    failure_context = getattr(row, "failure_context", None)
    failure_message = render_readout_failure(failure_code, failure_context)
    if failure_code is not None and failure_message is None:
        failure_message = f"{failure_code}: {_json(failure_context) or '{}'}"
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
                    cell.control_group
                    or getattr(row, "control_group", None)
                    or getattr(roster, "control_group", None)
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
                    failure = record.failure
                    sampling = record.sampling
                    if failure is not None:
                        failure_code = failure.code
                        failure_context = failure.context
                    elif sampling is not None and not sampling.available:
                        failure_code = sampling.reason_code
                        failure_context = sampling.reason_context
                    elif row is not None:
                        failure_code = getattr(row, "failure_code", None)
                        failure_context = getattr(row, "failure_context", None)
                    else:
                        failure_code = None
                        failure_context = None
                    failure_message = render_readout_failure(failure_code, failure_context)
                    if failure_code is not None and failure_message is None:
                        failure_message = f"{failure_code}: {_json(failure_context) or '{}'}"
                    sampling_available = (
                        sampling.available if sampling is not None else _has_sampling_result(row)
                    )
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


def render_readout_text(interpretation: ReadoutInterpretation) -> str:
    """Render concise integrity/completeness qualifications before effects."""
    if not interpretation.groups:
        return "Scope retained; no source-qualified cells recorded."
    lines: list[str] = []
    for group in interpretation.groups:
        comparison = (
            f"{group.treatment_group} vs {group.control_group}"
            if group.control_group is not None and group.treatment_group is not None
            else f"treatment {group.treatment_group}; "
            + (
                "control identity unknown"
                if not interpretation.metadata_known
                else "control identity unavailable in retained scope"
            )
            if group.treatment_group is not None
            else (
                "comparison unknown"
                if not interpretation.metadata_known
                else "comparison not represented by retained scope"
            )
        )
        lines.append(f"{group.population or 'population unknown'} — {comparison}")
        visible_rows = sorted(group.visible_rows, key=_row_display_order)
        if group.integrity_status == "failed":
            lines.append(f"Integrity: failed ({group.integrity_code or 'mismatch'}).")
        elif group.integrity_status == "not_rejected":
            lines.append(
                "Integrity: not_rejected — no detected mismatch; not proof of valid randomization."
            )
        else:
            lines.append(f"Integrity: {group.integrity_status} (not success).")
        if group.partial:
            lines.append("View: partial; do not treat visible cells as a complete decision.")
        if group.decision_complete is None:
            lines.append("Decision scope completeness: unknown.")
        else:
            state = "complete" if group.decision_complete else "incomplete"
            lines.append(f"Decision scope completeness: {state}.")
        if group.roster_complete is None:
            lines.append("Population roster completeness: unknown.")
        else:
            state = "complete" if group.roster_complete else "incomplete"
            source = group.roster_source or "source unknown"
            lines.append(f"Population roster completeness: {state} ({source}).")
        if group.required_cells:
            required_available = sum(item.available for item in group.required_cells)
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
        advisory_codes = {item.code for item in group.advisories}
        cells_by_key = {item.key: item for item in group.source_cells}
        for row in visible_rows:
            cell = cells_by_key.get(CellKey.from_row(row))
            if cell is not None and cell.posterior_available is False:
                lines.append(
                    f"Posterior unavailable {cell.posterior_reason_code}: "
                    f"{_reason(cell.posterior_reason_context)}"
                )
            if getattr(row, "note", None) and not any(code in row.note for code in advisory_codes):
                lines.append(f"Qualification: {row.note}")
            lines.extend(_render_effect(row))
            if getattr(row, "role", None) == "secondary":
                discovery = getattr(row, "discovery", None)
                selection = (
                    "selected"
                    if discovery is True
                    else "not selected"
                    if discovery is False
                    else "not recorded"
                )
                lines.append(f"Selection: {selection}; separate from interval evidence.")
    return "\n".join(lines)


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
    bullets: list[str] = []
    for group in interpretation.groups:
        attribution = [
            value
            for value in (group.population, _display_source(group.source_snapshot_id))
            if value
        ]
        suffix = "" if not attribution else f" [{', '.join(attribution)}]"
        comparison = (
            f"{group.treatment_group} vs {group.control_group}"
            if group.control_group is not None and group.treatment_group is not None
            else f"treatment {group.treatment_group}; control identity "
            + ("unknown" if not interpretation.metadata_known else "unavailable in retained scope")
            if group.treatment_group is not None
            else "comparison unknown"
            if not interpretation.metadata_known
            else "comparison not represented by retained scope"
        )
        bullets.append(f"Comparison: {comparison}{suffix}")
        integrity = f"Integrity: {group.integrity_status}"
        if group.integrity_code:
            integrity += f" ({group.integrity_code})"
        if group.integrity_status == "not_rejected":
            integrity += "; no detected mismatch, not proof of valid randomization"
        bullets.append(integrity + suffix)
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
        bullets.append(
            f"Scope: decision {decision}; roster {roster}"
            f"{'; partial view' if group.partial else ''}{counts}{suffix}"
        )
        for item in group.unavailable_cells:
            if not item.visible:
                continue
            identity = _cell_label(item.key)
            reason = item.failure_message or item.failure_code or "reason unavailable"
            bullets.append(f"Unavailable {identity}: {reason}{suffix}")
        for item in group.hidden_cells:
            identity = _cell_label(item.key)
            reason = item.failure_message or (
                "availability unknown" if not item.available else "result retained in scope"
            )
            bullets.append(f"Hidden by filter {identity}: {reason}{suffix}")
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
                bullets.append(f"Multiplicity ({role}): {status}; family {family_name}{suffix}")
        elif group.retained_families:
            family_names = ", ".join(sorted({family.name for family in group.retained_families}))
            bullets.append(f"Multiplicity: retained scope; family {family_names}{suffix}")
        for advisory in group.advisories:
            context = _json(advisory.context)
            detail = "" if not context else f"; context {context}"
            bullets.append(f"Advisory {advisory.code}: {advisory.message}{detail}{suffix}")
    return tuple(bullets)


def _cell_label(cell: CellKey) -> str:
    parts = [cell.metric]
    if cell.kind == "contrast" and cell.treatment_group is not None:
        comparison = (
            f"{cell.treatment_group} vs {cell.control_group}"
            if cell.control_group is not None
            else f"treatment {cell.treatment_group}"
        )
        parts.append(comparison)
    elif cell.group_id is not None:
        parts.append(cell.group_id)
    if cell.method is not None:
        parts.append(cell.method)
    return " / ".join(parts)


def _retained_intervals(row, estimate, scale: str):
    if (
        getattr(row, "failure_code", None) is not None
        or getattr(row, "sampling_available", None) is False
    ):
        return (), None
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
    relative_set = getattr(row, "relative_confidence_set", None)
    if relative_set is not None and scale != "absolute":
        return tuple(relative_set.intervals), getattr(relative_set, "level", None)
    binomial = getattr(row, "binomial_set", None)
    if binomial is not None and scale != "absolute":
        return ((binomial.lower, binomial.upper),), binomial.level
    sequential = getattr(row, "sequential_result", None)
    if sequential is not None:
        bounds = sequential.bounds
        if bounds.status == "unavailable":
            return (), None
        components = getattr(bounds, "components", None)
        intervals = (
            tuple((component.lower, component.upper) for component in components)
            if components is not None
            else ((bounds.lower, bounds.upper),)
        )
        return intervals, getattr(bounds, "level", None)
    if estimate is None or (estimate.lb is None and estimate.ub is None):
        return (), None
    return ((estimate.lb, estimate.ub),), getattr(estimate, "level", None)


def _interval_interpretation(row, estimate, scale: str, *, contrast: bool):
    null = (
        getattr(row, "null_abs", None)
        if contrast or scale == "absolute"
        else getattr(row, "null_lift", None)
    )
    alternative = getattr(row, "alternative", None)
    if null is None or alternative is None:
        return None
    intervals, level = _retained_intervals(row, estimate, scale)
    if not intervals:
        return None
    contains_null = any(
        (lower is None or lower <= null) and (upper is None or null <= upper)
        for lower, upper in intervals
    )
    if contains_null:
        return (
            "Interpretation: interval contains the tested null; inconclusive about "
            "direction, not evidence of no effect."
        )
    directions = {
        "increase" if lower is not None and lower > null else "decrease"
        for lower, upper in intervals
        if (lower is not None and lower > null) or (upper is not None and upper < null)
    }
    level_note = "" if level is None else f" at {level:.1%} confidence"
    if len(directions) != 1:
        return f"Interpretation: confidence set excludes the tested null{level_note} but spans opposing directions."
    direction = next(iter(directions))
    if alternative == "greater" and direction != "increase":
        return (
            f"Interpretation: interval excludes the tested null{level_note} in the "
            "direction opposite the declared alternative."
        )
    if alternative == "less" and direction != "decrease":
        return (
            f"Interpretation: interval excludes the tested null{level_note} in the "
            "direction opposite the declared alternative."
        )
    return (
        f"Interpretation: interval excludes the tested null{level_note}; "
        f"direction supported ({direction})."
    )


def render_effect_interpretation(row) -> str | None:
    """Render a restrained interpretation from a row's retained interval and null."""
    contrast = hasattr(row, "control_group")
    scale = "absolute" if contrast else getattr(row, "value_scale", "relative")
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


def _has_point(row) -> bool:
    if row is None:
        return False
    return getattr(row, "lift", None) is not None or getattr(row, "estimate", None) is not None


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


def _render_effect(row):
    lines = []
    role = getattr(row, "role", None) or "outcome"
    contrast = hasattr(row, "control_group")
    scale = "absolute" if contrast else getattr(row, "value_scale", "relative")
    label = (
        "absolute contrast" if contrast else "absolute" if scale == "absolute" else "relative lift"
    )
    estimate = getattr(row, "lift", None) or getattr(row, "estimate", None)
    if estimate is not None:
        effect = f"{row.metric} ({role}, {label}): {estimate.value:.6g}"
        if estimate.lb is not None or estimate.ub is not None:
            low = "−∞" if estimate.lb is None else f"{estimate.lb:.6g}"
            high = "+∞" if estimate.ub is None else f"{estimate.ub:.6g}"
            effect += f", set [{low}, {high}]"
            if estimate.level is not None:
                effect += f" at {estimate.level:.3g} confidence"
        lines.append(effect + ".")
    interpretation = _interval_interpretation(row, estimate, scale, contrast=contrast)
    if interpretation is not None:
        lines.append(interpretation)
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
    null_abs = getattr(row, "null_abs", None)
    null_lift = getattr(row, "null_lift", None)
    null = null_abs if null_abs is not None else null_lift
    if getattr(row, "alternative", None) and null is not None:
        unit = (
            "absolute units"
            if null_abs is not None or contrast or scale == "absolute"
            else "relative lift"
        )
        lines.append(f"Test alternative: {row.alternative}; null ({unit}): {null:.6g}.")
    return lines
