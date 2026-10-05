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
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import coeftable as ct
import narwhals as nw
from coeftable.collapsible import make_collapsible
from coeftable.labels import ROW_LABEL

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


def _metric_label(metrics: Mapping[str, str]) -> FormatFn:
    """The row-label formatter, wrapping each identified label in its metric key."""
    formatter = ROW_LABEL.html
    assert formatter is not None

    def to_html(text: Any) -> str | FormatterSkipElement:
        label = formatter(text)
        # Row keys reach the formatter entity-escaped (blank on a repeated key).
        key = metrics.get(html.unescape(text)) if isinstance(text, str) and text else None
        if key is None or not isinstance(label, str) or not label:
            return label
        return f'<span {METRIC_ATTRIBUTE}="{html.escape(key, quote=True)}">{label}</span>'

    return to_html


def native_html(table: ct.CoefTable, metrics: Mapping[str, str]) -> str:
    """``table`` as an HTML fragment whose identified row labels carry ``data-inc-metric``."""
    rendered = table.gt()
    if table.rows is not None and metrics:
        rendered = rendered.fmt(_metric_label(metrics), columns=table.rows)
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
