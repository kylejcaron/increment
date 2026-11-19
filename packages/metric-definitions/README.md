# Metric Definitions

Define measures and metrics once, independently of a warehouse connection or an
analysis. This package validates definitions and selects the dependencies needed
by a consumer. It does not read warehouse data or evaluate metrics.

```python
from metric_definitions import Catalog, load

catalog = load("examples/shop/metrics.yaml")
selected = catalog.select(["average_order_value"])
assert set(selected.measures) == {"revenue", "order_count"}

schema = Catalog.model_json_schema()  # JSON Schema for the YAML input
```

Use plain YAML mappings, lists, and scalars. Explicit YAML tags, aliases, and
merge keys are not supported.

## Version 1

- A **source** names a table, its unique non-null `row_key`, entity columns,
  and optionally its event-time column. Use simple ASCII identifiers for
  columns and each segment of a qualified `table` name, not SQL fragments.
- A **dimension** names a column and its type (`string`, `boolean`, `integer`,
  `number`, `date`, or `timestamp`). A named join maps a local `foreign_key` to
  another source's `row_key`. Use `join.dimension` to refer to a joined
  dimension. Only many-to-one joins to a unique key are supported.
- A **filter** names its source and a nonempty list of conditions, all of which
  must hold. A condition names a dimension and uses `eq`, `ne`, `lt`, `le`,
  `gt`, `ge`, or `in`. Use a typed scalar, or a nonempty list for `in`. Use
  ISO dates (`YYYY-MM-DD`) and ISO timestamps. Null is not a comparison value.
- A **measure** sums a column (`sum`) or counts qualifying rows
  (`count_rows: true`). Its optional filter applies only to that measure.
- A **population** names a source keyed by the declared entity. This is the
  eligible roster, not the set of entities with activity. The roster and the
  measured source must name the same entity; their physical key names may differ.
- A **metric** is a `formula`, `per_unit_mean`, or `conversion`. Formulas combine
  named measures and metrics with finite numbers, `+`, `-`, `*`, `/` and
  parentheses. Means average over an explicit population. Conversions count an
  eligible entity once if it has any qualifying events.

Compute formula operands at the requested unit or grouping, then apply the
arithmetic. For example, average order value is total revenue divided by total
orders, not the average of per-customer order values. Zero orders means the
ratio is undefined. A customer with known-complete coverage and no qualifying
orders contributes zero to a count or sum; missing coverage is not zero.

Validate physical keys and data coverage in the consumer. Included amounts must
be non-null; row, roster and dimension keys must be unique and non-null in the
same input snapshot. A consumer must reject a population or grouping it cannot
represent. A valid catalog does not establish that any workload can execute it.

The [shop fixture](examples/shop/) contains the definitions, input SQL, matched
scope, and hand-derived results. Existing product-specific settings such as
exposure rules and snapshot windows belong in the consumer, not in this file.

## Use the same metric in another tool

| Consumer | Entry point | Scope |
| --- | --- | --- |
| Increment | `from_definitions` with `metric_definitions` | Additive ratios, eligible-unit means, and event-presence conversions with matching native bindings |
| Terrine | `Model.from_metric_definitions`, then an ordinary `Design` | Named formulas over aligned window aggregates; readiness and roster binding stay native |
| Malloy | `metric_definitions.malloy.export_malloy` | Native fields and grouping over compatible fact or roster sources |

The [migration and conformance guide](conformance/) shows how to reuse AOV
without rewriting an experiment or a Terrine Design. The comparison uses matched
DuckDB inputs and candidate builds, not released package compatibility.
