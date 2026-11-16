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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import fields, replace
from typing import TYPE_CHECKING, Any

import coeftable as ct
import narwhals as nw
from coeftable.collapsible import make_collapsible
from coeftable.frame import SPLIT_JOINER
from coeftable.labels import ROW_LABEL

from increment.dashboard._format import guardrail_text
from increment.tables import readout_table

if TYPE_CHECKING:
    # Annotation-only: great_tables is a transitive dependency and this module is private.
    from great_tables._gt_data import FormatFn, FormatterSkipElement

__all__ = [
    "append_diagnostics",
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


def _metric_label(
    metrics: Mapping[str, str], guardrails: Mapping[str, str], labels: Mapping[str, str]
) -> FormatFn:
    """The row-label formatter, wrapping each identified label in its metric key."""
    formatter = ROW_LABEL.html
    assert formatter is not None

    def to_html(text: Any) -> str | FormatterSkipElement:
        raw = html.unescape(text) if isinstance(text, str) and text else None
        display = html.escape(labels[raw]) if raw is not None and raw in labels else text
        label = formatter(display)
        # Row keys reach the formatter entity-escaped (blank on a repeated key).
        key = metrics.get(raw) if raw is not None else None
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


def native_html(
    table: ct.CoefTable, metrics: Mapping[str, str], *, labels: Mapping[str, str] | None = None
) -> str:
    """``table`` as an HTML fragment whose identified row labels carry ``data-inc-metric``."""
    rendered = table.gt()
    html_columns = [
        column.label
        for column in table.columns
        if isinstance(column, ct.Passthrough)
        and (column.label, column.column)
        in {("Diagnostics", "_uncertainty_diagnostics"), ("Trajectory", "Trajectory")}
    ]
    if html_columns:
        if table.split_columns is not None:
            splits = nw.from_native(table.data, eager_only=True)[table.split_columns].unique()
            html_columns = [
                label if split is None else f"{split}{SPLIT_JOINER}{label}"
                for split in splits.to_list()
                for label in html_columns
            ]
        # Generated HTML already escapes data; do not reinterpret it as units or Markdown.
        rendered = rendered.fmt(lambda value: value, columns=html_columns)
    if table.rows is not None and (metrics or labels):
        rows = nw.from_native(table.data, eager_only=True)
        guardrails = {
            str(row[table.rows]): guardrail_text(row) for row in rows.iter_rows(named=True)
        }
        rendered = rendered.fmt(
            _metric_label(metrics, guardrails, labels or {}), columns=table.rows
        )
    markup = rendered.as_raw_html()
    return make_collapsible(markup) if table.collapsible_groups else markup


def readout_native(
    rows: Sequence[dict[str, Any]], **options: Any
) -> tuple[ct.CoefTable, dict[str, str]]:
    """``readout_table`` for ``rows`` plus the identity of each label it renders.

    ``rows`` are readout row mappings; ``options`` pass through to ``readout_table``.
    """
    table = readout_table(list(rows), **options)
    return table, row_metrics(table, [row["metric"] for row in rows])


def append_diagnostics(table: ct.CoefTable) -> None:
    """Append captured, row-specific disclosures as a native HTML column."""
    data = nw.from_native(table.data, eager_only=True)
    column = "_uncertainty_diagnostics"
    if column in data.columns and any(data[column].to_list()):
        table.columns = (*table.columns, ct.Passthrough("Diagnostics", column))


def _mark_lift_caveats(table: ct.CoefTable) -> None:
    """Keep row-specific footnote references visible when the Result column is hidden."""
    data = nw.from_native(table.data, eager_only=True)
    if "__result_markers" not in data.columns:
        return
    markers = data["__result_markers"].to_list()
    if not any(markers):
        return
    absolute_rows = (
        [scale == "absolute" for scale in data["value_scale"].to_list()]
        if "value_scale" in data.columns
        else [False] * len(markers)
    )

    def marked(column: ct.Estimate) -> ct.Estimate:
        absolute = column.label == "Lift (absolute)"

        class CaveatMarkedEstimate(ct.Estimate):
            def cell(self, ctx: Any) -> str:
                rendered = super().cell(ctx)
                marker = markers[ctx.index]
                if marker and absolute_rows[ctx.index] == absolute:
                    return f"{rendered}<sup>{html.escape(marker)}</sup>"
                return rendered

        return CaveatMarkedEstimate(
            **{field.name: getattr(column, field.name) for field in fields(column)}
        )

    table.columns = tuple(
        marked(column)
        if isinstance(column, ct.Estimate) and column.label in {"Lift %", "Lift (absolute)"}
        else column
        for column in table.columns
    )


def drop_columns(table: ct.CoefTable, labels: Iterable[str]) -> None:
    """Remove the declared columns whose header is in ``labels``."""
    hidden = frozenset(labels)
    if "Result" in hidden:
        _mark_lift_caveats(table)
    table.columns = tuple(
        column for column in table.columns if getattr(column, "label", None) not in hidden
    )


def resize_forest(table: ct.CoefTable, *, width: int, height: int) -> None:
    """Give every Forest column the theme's chart size."""
    table.columns = tuple(
        replace(column, width=width, height=height) if isinstance(column, ct.Forest) else column
        for column in table.columns
    )
