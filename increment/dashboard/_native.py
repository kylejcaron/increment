"""Native CoefTable rendering with explicit metric identity.

The browser needs to know which declared metric a row label belongs to, and the rendered label
cannot say: estimand, population and scale qualifiers are appended to it, humanised labels can
repeat, and a metric name may be hostile text. The renderer therefore pairs each row label with
the original model key *before* HTML exists and emits the key as ``data-inc-metric`` on a span
inside the label cell. Nothing downstream recovers a key from label text.

Everything that touches CoefTable internals lives here: the label formatter and the collapsible
transform that ``CoefTable.as_raw_html`` applies, and the Forest/column adjustments the
dashboard needs. ``_html`` sees only the functions below.
"""

from __future__ import annotations

import html
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import coeftable as ct
import narwhals as nw
from coeftable.collapsible import make_collapsible
from coeftable.labels import ROW_LABEL
from great_tables import html as gt_html

from increment.dashboard._format import guardrail_text, percent_text
from increment.tables import readout_table

if TYPE_CHECKING:
    # Annotation-only: great_tables is a transitive dependency and this module is private.
    from great_tables._gt_data import FormatFn, FormatterSkipElement

__all__ = [
    "drop_columns",
    "native_html",
    "readout_native",
    "resize_forest",
    "row_metrics",
]

METRIC_ATTRIBUTE = "data-inc-metric"


def row_metrics(table: ct.CoefTable, keys: Sequence[object]) -> dict[str, str]:
    """Map each row label of ``table`` to the one model key it stands for.

    ``keys[i]`` is the original metric key of the table's ``i``-th input row. A label shared by
    rows of different metrics would make one cell stand for two metrics, so it gets no identity
    rather than an arbitrary one.
    """
    assert table.rows is not None
    labels = nw.from_native(table.data, eager_only=True)[table.rows].to_list()
    assert len(labels) == len(keys), "metric keys must align with the table's input rows"
    metrics: dict[str, str | None] = {}
    for label, key in zip(labels, keys, strict=True):
        label, key = str(label), str(key)
        metrics[label] = key if metrics.get(label, key) == key else None
    return {label: key for label, key in metrics.items() if key is not None}


def _metric_label(metrics: Mapping[str, str], guardrails: Mapping[str, str]) -> FormatFn:
    """The row-label formatter, wrapping each identified label in its metric key."""
    formatter = ROW_LABEL.html
    assert formatter is not None

    def to_html(text: Any) -> str | FormatterSkipElement:
        label = formatter(text)
        # Row keys reach the formatter entity-escaped (blank on a repeated key).
        key = metrics.get(html.unescape(text)) if isinstance(text, str) and text else None
        if key is None or not isinstance(label, str) or not label:
            return label
        boundary = guardrails.get(html.unescape(text), "")
        note = (
            '<small class="inc-dashboard-guardrail">'
            '<span class="inc-dashboard-guardrail-label">Guardrail</span>'
            f"{html.escape(boundary.removeprefix('Guardrail'))}</small>"
            if boundary
            else ""
        )
        return f'<span {METRIC_ATTRIBUTE}="{html.escape(key, quote=True)}">{label}</span>{note}'

    return to_html


def native_html(table: ct.CoefTable, metrics: Mapping[str, str]) -> str:
    """``table`` as an HTML fragment whose identified row labels carry ``data-inc-metric``."""
    rendered = table.gt()
    if table.rows is not None and metrics:
        rows = nw.from_native(table.data, eager_only=True)
        guardrails = {
            str(row[table.rows]): guardrail_text(row) for row in rows.iter_rows(named=True)
        }
        rendered = rendered.fmt(_metric_label(metrics, guardrails), columns=table.rows)
    markup = rendered.as_raw_html()
    return make_collapsible(markup) if table.collapsible_groups else markup


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
            "<dt>"
            + html.escape(str(key))
            + "</dt><dd>"
            + _metadata_context(item, encoded=False)
            + "</dd>"
            for key, item in value.items()
        )
        return f'<dl class="inc-dashboard-kv">{fields}</dl>'
    if isinstance(value, (list, tuple)):
        fields = "".join(f"<li>{_metadata_context(item, encoded=False)}</li>" for item in value)
        return f"<ul>{fields}</ul>"
    return html.escape(_metadata_value(value))


def _evidence_pair(label: str, value: Any) -> str:
    return "<dt>" + html.escape(label) + "</dt><dd>" + html.escape(_metadata_value(value)) + "</dd>"


def _evidence_label(row: Mapping[str, Any]) -> str:
    return " · ".join(
        str(value)
        for value in (
            row.get("metric"),
            row.get("segment"),
            row.get("_evidence_method", row.get("method")),
            row.get("method_role"),
        )
        if value is not None
    )


def _reading_evidence(row: Mapping[str, Any]) -> str:
    level = row.get("level")
    alpha = row.get("coverage_alpha", row.get("alpha"))
    confidence = (
        percent_text(float(level), float(alpha) if isinstance(alpha, (int, float)) else None)
        if isinstance(level, (int, float))
        else None
    )
    fields = [
        _evidence_pair("Confidence", confidence),
        _evidence_pair("Retained noncoverage budget (α)", alpha),
    ]
    for key, label in (("dimension", "Breakout"), ("source", "Breakout source")):
        if row.get(key) is not None:
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
        '<section class="inc-dashboard-evidence-card inc-dashboard-cell-evidence"><h5>'
        + html.escape(_evidence_label(row))
        + '</h5><dl class="inc-dashboard-kv">'
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
        record = (family.get("family_id"), name, family.get("complete"))
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
        if population and comparison and population != comparison:
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
                    "not_checked_missing_counts": (
                        "Not an assignment check: triggered counts are selected after assignment"
                        if population == "triggered"
                        else "Not checked: assignment counts are missing"
                    ),
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
                ("multiplicity_status", "Multiplicity status"),
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
                        else value
                    )
                    fields.append(_evidence_pair(label, display))
            _cell_evidence(row, fields, references, other_seen)
            _family_evidence(row, fields, references, family_seen)
            readings.append(_reading_evidence(row))

        if not integrity_seen:
            fields.insert(0, _evidence_pair(integrity_label, None))
        if source is not None:
            references.insert(0, _evidence_pair("Source reference", source))
        reference = (
            '<details class="inc-dashboard-disclosure"><summary>Source references and check context</summary>'
            '<dl class="inc-dashboard-kv">' + "".join(references) + "</dl></details>"
            if references
            else ""
        )
        cards.append(
            '<section class="inc-dashboard-evidence-card"><h4>'
            + html.escape(str(heading))
            + '</h4><dl class="inc-dashboard-kv">'
            + "".join(fields)
            + "</dl>"
            + '<details class="inc-dashboard-disclosure"><summary>Per-result confidence and prior evidence</summary>'
            + '<div class="inc-dashboard-evidence-stack">'
            + "".join(readings)
            + "</div></details>"
            + reference
            + "</section>"
        )

    context = (
        '<section class="inc-dashboard-evidence-card"><h4>Table context</h4>'
        '<dl class="inc-dashboard-kv">' + _evidence_pair("Subtitle", subtitle) + "</dl></section>"
        if subtitle
        else ""
    )
    return '<div class="inc-dashboard-evidence-stack">' + context + "".join(cards) + "</div>"


def readout_native(
    rows: Sequence[dict[str, Any]], **options: Any
) -> tuple[ct.CoefTable, dict[str, str]]:
    """``readout_table`` for ``rows`` plus the identity of each label it renders.

    ``rows`` are readout row mappings; ``options`` pass through to ``readout_table``.
    """
    title = options.pop("title", "")
    subtitle = options.pop("subtitle", "")
    table = readout_table(list(rows), title="", subtitle="", **options)
    evidence = _readout_evidence(rows, subtitle)
    # Great Tables interprets double braces as units even inside an HTML header.
    evidence = evidence.replace("{", "&#123;").replace("}", "&#125;")
    disclosure = gt_html(
        '<details class="inc-dashboard-table-evidence" '
        'style="font-size:12px;font-weight:normal;line-height:1.5;text-align:left">'
        "<summary>Table evidence and checks</summary>"
        f"{evidence}</details>"
    )
    table = table.header(title or cast(Any, disclosure), cast(Any, disclosure) if title else "")
    return table, row_metrics(table, [row["metric"] for row in rows])


def drop_columns(table: ct.CoefTable, labels: Iterable[str]) -> None:
    """Remove the declared columns whose header is in ``labels``."""
    hidden = frozenset(labels)
    table.columns = tuple(
        column for column in table.columns if getattr(column, "label", None) not in hidden
    )


def resize_forest(table: ct.CoefTable, *, width: int, height: int) -> None:
    """Give every Forest column the theme's chart size."""
    table.columns = tuple(
        replace(column, width=width, height=height) if isinstance(column, ct.Forest) else column
        for column in table.columns
    )
