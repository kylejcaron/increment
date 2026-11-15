"""Structured, dependency-neutral evidence presentation for experiment tables."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from typing import Any

from increment._display import _readout_interval, format_coverage
from increment._interpretation import _assignment_check_warning

MULTIPLICITY_LABELS = {
    "declared_plan": "Declared multiple-testing plan",
    "undeclared_plan": "No multiple-testing adjustment (no plan)",
    "unassigned_in_plan": "No multiple-testing adjustment (outside the plan)",
    "exploratory_unadjusted": "Exploratory; no multiple-testing adjustment",
    "exploratory_family": "Exploratory testing group",
}


def _metadata_value(value: Any) -> str:
    """Render scalar metadata without exposing Python container reprs."""
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _metadata_state(value: Any, *, yes: str, no: str) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return yes if value else no
    return _metadata_value(value)


def _metadata_context(value: Any, *, encoded: bool = True) -> str:
    if value is None:
        return "unknown"
    if value == "":
        return ""
    if encoded and isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return html.escape(value)
    if isinstance(value, Mapping):
        fields = "".join(
            (
                "<dt>"
                + html.escape(str(key))
                + "</dt><dd>"
                + _metadata_context(item, encoded=False)
                + "</dd>"
                for key, item in value.items()
            )
        )
        return f'<dl class="inc-readout-kv">{fields}</dl>'
    if isinstance(value, (list, tuple)):
        fields = "".join(f"<li>{_metadata_context(item, encoded=False)}</li>" for item in value)
        return f"<ul>{fields}</ul>"
    return html.escape(_metadata_value(value))


def _evidence_pair(label: str, value: Any) -> str:
    return "<dt>" + html.escape(label) + "</dt><dd>" + html.escape(_metadata_value(value)) + "</dd>"


def _evidence_label(row: Mapping[str, Any]) -> str:
    parts = [str(row["metric"])] if row.get("metric") is not None else []
    coordinates = (
        ("segment", "Segment"),
        ("group_id", "Arm"),
        ("estimand", "Estimand"),
        ("value_scale", "Scale"),
        ("ds", "Date"),
        ("ds_basis", "Date axis"),
    )
    for key, label in coordinates:
        if row.get(key) is not None:
            parts.append(f"{label}: {row[key]}")
    method = row.get("_evidence_method", row.get("method"))
    if method is not None:
        role = row.get("method_role")
        parts.append(f"Method: {method}" + (f" ({role})" if role is not None else ""))
    return " · ".join(parts)


def _reading_evidence(row: Mapping[str, Any]) -> str:
    level = row.get("level")
    alpha = row.get("coverage_alpha", row.get("alpha"))
    confidence = (
        format_coverage(float(level), float(alpha) if isinstance(alpha, (int, float)) else None)
        if isinstance(level, (int, float))
        else None
    )
    fields = [
        _evidence_pair("Confidence", confidence),
        _evidence_pair("Retained noncoverage budget (α)", alpha),
        _evidence_pair(
            "Family selection",
            _metadata_state(row.get("discovery"), yes="selected", no="not selected"),
        ),
    ]
    displayed = _readout_interval(row)
    if displayed.one_sided and displayed.level is not None:
        fields[0] = _evidence_pair("Stored confidence", confidence)
        fields.insert(
            0,
            _evidence_pair(
                "Displayed confidence",
                f"{format_coverage(displayed.level, displayed.alpha)} one-sided",
            ),
        )
    plan = row.get("multiplicity_status")
    fields.append(_evidence_pair("Multiple-testing plan", MULTIPLICITY_LABELS.get(plan, plan)))
    for key, label in (("dimension", "Breakout"), ("source", "Breakout source")):
        if row.get(key) is not None:
            fields.append(_evidence_pair(label, row[key]))
    if row.get("note"):
        fields.append(_evidence_pair("Interval qualification", row["note"]))
    for key, label in (
        ("effect_interpretation", "Interpretation"),
        ("numerical_qualification", "Numerical qualification"),
        ("selection_qualification", "Selection qualification"),
    ):
        if row.get(key):
            fields.append(_evidence_pair(label, row[key]))
    if any(
        row.get(key) is not None
        for key in ("posterior_available", "posterior_reason_code", "posterior_reason_context")
    ):
        fields.append(
            _evidence_pair(
                "Prior advisory",
                _metadata_state(row.get("posterior_available"), yes="available", no="unavailable"),
            )
        )
        if row.get("posterior_reason_code") is not None:
            fields.append(_evidence_pair("Reason code", row["posterior_reason_code"]))
        if row.get("posterior_reason_context") is not None:
            fields.append(
                "<dt>Reason context</dt><dd>"
                + _metadata_context(row["posterior_reason_context"])
                + "</dd>"
            )
    return (
        '<section class="inc-readout-evidence-card inc-readout-cell-evidence"><h5>'
        + html.escape(_evidence_label(row))
        + '</h5><dl class="inc-readout-kv">'
        + "".join(fields)
        + "</dl></section>"
    )


def _cell_evidence(
    row: Mapping[str, Any], fields: list[str], references: list[str], seen: set[Any]
) -> None:
    failure_code = row.get("failure_code")
    failure_context = row.get("failure_context")
    failure_message = row.get("failure_message")
    label = _evidence_label(row)
    failure = (label, failure_code, failure_context, failure_message)
    failure_key = json.dumps(failure, sort_keys=True, default=str)
    if any(value is not None for value in failure[1:]) and failure_key not in seen:
        seen.add(failure_key)
        references.append(_evidence_pair(f"Failure code · {label}", failure_code))
        if failure_message is not None:
            fields.append(_evidence_pair(f"Unavailable · {label}", failure_message))
        if failure_context is not None:
            references.append(
                "<dt>"
                + html.escape(f"Failure context · {label}")
                + "</dt><dd>"
                + _metadata_context(failure_context)
                + "</dd>"
            )
    for key, label in (
        ("advisories", "Advisory"),
        ("unavailable_cells", "Unavailable cell"),
        ("hidden_cells", "Hidden cell"),
    ):
        raw_records = row.get(key)
        if not raw_records:
            continue
        try:
            records = json.loads(raw_records) if isinstance(raw_records, str) else raw_records
        except json.JSONDecodeError:
            records = raw_records
        if isinstance(records, list):
            for item in records:
                rendered = _metadata_context(item, encoded=False)
                record = (key, rendered)
                if rendered and record not in seen:
                    seen.add(record)
                    fields.append("<dt>" + html.escape(label) + "</dt><dd>" + rendered + "</dd>")
        else:
            fields.append(
                "<dt>"
                + html.escape(label)
                + "</dt><dd>"
                + _metadata_context(records, encoded=False)
                + "</dd>"
            )


def _family_evidence(
    row: Mapping[str, Any], fields: list[str], references: list[str], seen: set[Any]
) -> None:
    raw_families = row.get("retained_family")
    if not raw_families:
        return
    try:
        families = json.loads(raw_families) if isinstance(raw_families, str) else raw_families
    except json.JSONDecodeError:
        families = raw_families
    if not isinstance(families, list):
        references.append(
            "<dt>Family record</dt><dd>" + _metadata_context(families, encoded=False) + "</dd>"
        )
        return
    for family in families:
        if not isinstance(family, Mapping):
            references.append(
                "<dt>Family record</dt><dd>" + _metadata_context(family, encoded=False) + "</dd>"
            )
            continue
        model = family.get("family") or {}
        name = (
            (model.get("name") if isinstance(model, Mapping) else None)
            or family.get("name")
            or family.get("family_id")
            or "unknown"
        )
        record = json.dumps(family, sort_keys=True, default=str)
        if record in seen:
            continue
        seen.add(record)
        fields.append(
            _evidence_pair(
                f"Multiple-testing group · {name}",
                _metadata_state(family.get("complete"), yes="complete", no="incomplete"),
            )
        )
        references.append(_evidence_pair(f"Family reference · {name}", family.get("family_id")))
        references.append(
            "<dt>"
            + html.escape(f"Plan record · {name}")
            + "</dt><dd>"
            + _metadata_context(family, encoded=False)
            + "</dd>"
        )


def _readout_evidence(rows: Sequence[dict[str, Any]], subtitle: Any) -> str:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        identity = tuple(
            row.get(key)
            for key in (
                "analysis_population",
                "source_snapshot_id",
                "comparison",
                "treatment_group",
                "control_group",
            )
        )
        groups.setdefault(identity, []).append(row)
    cards = []
    for identity, members in groups.items():
        population, source, comparison, treatment, control = identity
        comparison = comparison or (
            f"{treatment} vs {control}" if treatment is not None and control is not None else None
        )
        heading = comparison or population or "Readout checks"
        if population and comparison and (population != comparison):
            heading = f"{comparison} — {population}"
        integrity_label = (
            "Triggered-population integrity" if population == "triggered" else "Assignment check"
        )
        fields: list[str] = []
        references: list[str] = []
        other_seen = set()
        integrity_seen = set()
        roster_seen = set()
        family_seen = set()
        readings = []
        for row in members:
            integrity = tuple(
                row.get(key)
                for key in (
                    "integrity_status",
                    "integrity_code",
                    "integrity_context",
                    "required_cell_count",
                    "available_cell_count",
                )
            )
            integrity_key = json.dumps(integrity, sort_keys=True, default=str)
            if (
                any(value is not None for value in integrity)
                and integrity_key not in integrity_seen
            ):
                integrity_seen.add(integrity_key)
                status, code, context, required, available = integrity
                description = {
                    "not_rejected": "No mismatch detected; not proof of valid randomization",
                    "failed": "Assignment check failed",
                    "not_applicable": "Not an assignment check for this population",
                    "not_checked_missing_declaration": "Not checked: assignment declaration incomplete",
                    "not_checked_missing_counts": "Not an assignment check: triggered counts are selected after assignment"
                    if population == "triggered"
                    else "Not checked: assignment counts are missing",
                }.get(status, status)
                fields.extend(
                    [
                        _evidence_pair(integrity_label, description),
                        _evidence_pair("Required results", required),
                        _evidence_pair("Available results", available),
                    ]
                )
                references.append(_evidence_pair("Recorded integrity status", status))
                if code is not None:
                    references.append(_evidence_pair("Integrity code", code))
                if context is not None:
                    references.append(
                        "<dt>Integrity context</dt><dd>" + _metadata_context(context) + "</dd>"
                    )
            roster = (row.get("roster_complete"), row.get("roster_source"))
            if any(value is not None for value in roster) and roster not in roster_seen:
                roster_seen.add(roster)
                complete, roster_source = roster
                fields.extend(
                    [
                        _evidence_pair(
                            "Roster completeness",
                            _metadata_state(complete, yes="complete", no="incomplete"),
                        ),
                        _evidence_pair("Roster source", roster_source),
                    ]
                )
            for key, label in (
                ("decision_scope_complete", "Decision scope completeness"),
                ("view_partial", "View partial"),
                ("multiplicity_status", "Multiple-testing plan"),
            ):
                value = row.get(key)
                record = (key, value)
                if value is not None and record not in other_seen:
                    other_seen.add(record)
                    display = (
                        _metadata_state(value, yes="complete", no="incomplete")
                        if key == "decision_scope_complete"
                        else _metadata_state(value, yes="partial", no="not partial")
                        if key == "view_partial"
                        else MULTIPLICITY_LABELS.get(value, value)
                    )
                    fields.append(_evidence_pair(label, display))
                    if key == "multiplicity_status":
                        references.append(_evidence_pair("Recorded plan status", value))
            _cell_evidence(row, fields, references, other_seen)
            _family_evidence(row, fields, references, family_seen)
            for key, label in (
                ("coherence", "Snapshot coherence"),
                ("provider_owned", "Provider-owned evidence"),
                ("independently_captured", "Independently captured sources"),
                ("effective_modes", "Evidence acquisition modes"),
            ):
                value = row.get(key)
                record = json.dumps(("capture", key, value), sort_keys=True, default=str)
                if value is not None and record not in other_seen:
                    other_seen.add(record)
                    references.append(
                        "<dt>"
                        + html.escape(label)
                        + "</dt><dd>"
                        + _metadata_context(value)
                        + "</dd>"
                    )
            if row.get("metric") is not None:
                readings.append(_reading_evidence(row))
        if not integrity_seen:
            fields.insert(0, _evidence_pair(integrity_label, None))
        if source is not None:
            references.insert(0, _evidence_pair("Source reference", source))
        reference = (
            '<details class="inc-readout-disclosure"><summary>Source references and check context</summary><dl class="inc-readout-kv">'
            + "".join(references)
            + "</dl></details>"
            if references
            else ""
        )
        cards.append(
            '<section class="inc-readout-evidence-card"><h4>'
            + html.escape(str(heading))
            + '</h4><dl class="inc-readout-kv">'
            + "".join(fields)
            + "</dl>"
            + '<details class="inc-readout-disclosure"><summary>Per-result evidence</summary>'
            + '<div class="inc-readout-evidence-stack">'
            + "".join(readings)
            + "</div></details>"
            + reference
            + "</section>"
        )
    context = (
        '<section class="inc-readout-evidence-card"><h4>Table context</h4><dl class="inc-readout-kv">'
        + _evidence_pair("Subtitle", subtitle)
        + "</dl></section>"
        if subtitle
        else ""
    )
    return '<div class="inc-readout-evidence-stack">' + context + "".join(cards) + "</div>"


_EVIDENCE_STYLE = """
<style>
.inc-readout-evidence {font-size:12px;font-weight:400;line-height:1.5;text-align:left;white-space:normal}
.inc-readout-evidence summary {cursor:pointer;font-weight:600}
.inc-readout-evidence-stack {display:grid;gap:10px;margin-top:10px}
.inc-readout-evidence-card {border:1px solid color-mix(in srgb,currentColor 18%,transparent);border-radius:8px;padding:12px;min-width:0}
.inc-readout-evidence-card h4,.inc-readout-evidence-card h5 {font-size:inherit;font-weight:600;margin:0 0 10px}
.inc-readout-kv {display:grid;grid-template-columns:minmax(8rem,1fr) minmax(0,3fr);gap:6px 14px;margin:0}
.inc-readout-kv dt {font-weight:600}
.inc-readout-kv dd {margin:0;min-width:0;overflow-wrap:anywhere}
.inc-readout-kv ul {margin:0;padding-left:18px}
.inc-readout-disclosure {margin-top:12px}
.inc-readout-status {display:flex;flex-wrap:wrap;gap:4px 18px;list-style:none;padding:0;margin:0 0 8px}
.inc-readout-caption {margin:0 0 8px}
@media (max-width:640px) {.inc-readout-kv {grid-template-columns:minmax(0,1fr);gap:4px}.inc-readout-kv dd {margin-bottom:8px}}
</style>
"""


def _status_summary(
    rows: Sequence[dict[str, Any]], *, empty: bool, markers: Mapping[str, str]
) -> str:
    statuses: dict[str, str] = {}
    if empty:
        statuses["empty"] = "No results in this view"
    if not rows:
        statuses["unknown_scope"] = "Scope and assignment checks unknown"
    for row in rows:
        integrity = row.get("integrity_status")
        assignment_warning = _assignment_check_warning(integrity, row.get("analysis_population"))
        if row.get("decision_scope_complete") is None and integrity in (None, "unknown"):
            statuses["unknown_scope"] = "Scope and assignment checks not retained"
        elif assignment_warning is not None:
            statuses[integrity or "unknown"] = (
                "Assignment check failed"
                if assignment_warning == "failed"
                else "Assignment not checked"
            )
        if row.get("coherence") == "live":
            statuses["live"] = "Live read; no pinned snapshot guaranteed"
        if row.get("provider_owned") is True:
            statuses["provider_owned"] = "Snapshot guarantees depend on the evidence provider"
        if row.get("independently_captured") is True:
            statuses["independently_captured"] = (
                "Independent captures; no common snapshot guaranteed"
            )
        for key, state, label in (
            ("decision_scope_complete", False, "Incomplete decision scope"),
            ("roster_complete", False, "Incomplete assignment roster"),
            ("view_partial", True, "Partial view"),
        ):
            if row.get(key) is state:
                statuses[key] = label
        if row.get("failure_code"):
            statuses["unavailable"] = "Some results unavailable"
        plan = row.get("multiplicity_status")
        if plan in MULTIPLICITY_LABELS:
            statuses[plan] = f"{markers.get(plan, '')} {MULTIPLICITY_LABELS[plan]}".strip()
    if not statuses:
        return ""
    items = "".join(
        f'<li data-inc-status="{html.escape(code, quote=True)}">{html.escape(label)}</li>'
        for code, label in statuses.items()
    )
    return f'<ul class="inc-readout-status">{items}</ul>'


def readout_header(
    rows: Sequence[dict[str, Any]],
    *,
    subtitle: str = "",
    level_note: str | None = None,
    empty: bool = False,
    markers: Mapping[str, str] | None = None,
) -> str:
    """Keep safety status visible and the complete evidence expandable."""
    caption = "".join(
        f'<p class="inc-readout-caption">{html.escape(text)}</p>'
        for text in (subtitle, level_note)
        if text
    )
    body = (
        '<div class="inc-readout-evidence">'
        + caption
        + _status_summary(rows, empty=empty, markers=markers or {})
        + "<details><summary>Table evidence and checks</summary>"
        + _readout_evidence(rows, "")
        + "</details></div>"
    )
    # Great Tables treats double braces as units, including inside HTML headers.
    # Encode data braces, not the raw-text CSS rules.
    return _EVIDENCE_STYLE + body.replace("{", "&#123;").replace("}", "&#125;")
