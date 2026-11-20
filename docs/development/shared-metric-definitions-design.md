# Shared metric definitions: Increment, Terrine, and Malloy

**Proposed contract, 2026-11-18.** A team authors a metric once in versioned YAML. Increment and Terrine read the same file without importing each other or running Malloy. Analysts import generated `.malloy` into hand-written models. Compatibility means matching the metric on the same population, period, grouping and inputs; it does not mean the three tools answer different questions with identical numbers.

## What is shared

Extract a standalone, non-executing Python package: the YAML schema, strict loader, immutable typed model, and optional Malloy exporter. Reuse the vocabulary in Increment's `increment/semantics/models.py` and Terrine's `terrine/semantic/models.py` where meanings agree. The core has no database connection, Ibis, estimator, or Malloy runtime dependency. Neither product's full native schema becomes the shared schema. Do not distribute copies of the parser in consumer repositories.

The portable model names physical table sources, row keys (what one row is), entity keys (who it belongs to), event time, dimensions, typed filters, aggregate measures and arithmetic formulas. A declared eligible roster supplies the denominator population for per-unit metrics. Revenue per eligible user and revenue per buyer are different definitions; neither may be inferred from a fact-table join. Start with event-local dimensions and attributes joined many-to-one on declared keys; do not infer joins, historical lookups or compatibility from column names. Refuse ambiguous or fanout-producing joins rather than multiplying amounts. Keep connection credentials and actual window/anchor bindings outside the definition.

Illustrative authoring shape (the orders table has one row per order; amounts are in one currency):

```yaml
version: 1
sources:
  orders:
    table: orders
    row_key: order_id
    entities: {customer: customer_id}
    time: placed_at
    dimensions: {channel: {column: channel, type: string}}
measures:
  revenue: {source: orders, sum: gross_amount}
  refunds: {source: orders, sum: refunded_amount}
  order_count: {source: orders, count_rows: true}
metrics:
  average_order_value: {formula: revenue / order_count}
  net_revenue: {formula: revenue - refunds}
```

Formulas evaluate **after aggregating their operands at the requested grain**. Support references to named measures/metrics, finite numeric literals, parentheses, unary minus and `+ - * /`. No arbitrary Python, SQL, function calls or Malloy text in formulas. Reject unknown references, cycles and incompatible groupings; keep the operand tree available to callers. Divide by zero yields numeric null; missing or incomplete input must not be fabricated as zero. A complete population member with no qualifying events contributes zero to an additive count or sum, not zero to an undefined ratio. Roll up additive components, then reevaluate the formula; never average displayed ratios by default.

## Consumer boundaries

| Consumer | What it owns | What happens with a formula |
|---|---|---|
| Increment | Existing Ibis builders, exposure, assigned/triggered population, CUPED and inference | Bind existing supported means, conversions and ratio-of-totals to their native kinds when their requirements match. A catalog may contain `net_revenue` without breaking another metric's experiment. If selected for inference without a supported construction, refuse before loading data; do not add generic formula inference. |
| Terrine | Its Ibis builders, unit roster binding, snapshot/anchor design, readiness, window completeness | Compute the named operands at the same unit/snapshot and window and then evaluate the formula. Retain null and availability reasons. Do not imply that an arbitrary Terrine feature or design is exportable. |
| Malloy | Exploration, grouping, nested views, dashboards, dialect compilation | Export supported measures and formulas as ordinary Malloy fields referencing named operands. Join from the declared population when required; count real order keys rather than the synthetic row from a left join. Analysts extend a separate file; regenerating the base must not touch it. |

If a requested definition needs native-only behavior that the target cannot represent faithfully (such as an unbound exposure-relative population), return a diagnostic naming the metric, requirement and target. No partial file presented as a successful export. A catalog containing unrelated native definitions must not prevent selecting a portable metric. No Malloy importer or runtime replacement. Formula decomposition exposes arithmetic and dependencies; ranking drivers or explaining causation is not part of this project.

## Migration and proof

1. Inventory Increment's existing YAML and Terrine's Catalog against the portable subset. For each existing field record whether it is shared, consumer-specific or explicitly unsupported. Do not reinterpret a current native file as shared by changing the default loader. New shared documents have their own versioned entry point. Make an explicit migration for portable definitions; leave native analysis/design settings in place. Keep the documented `Model.from_increment` input contract for existing users through its supported release lifecycle, without using it as the new shared path.
2. Prove the contract on one DuckDB fixture with customers A, B and C. A has orders for 20 and 40 with refunds of 0 and 5; B has one order for 60 with a refund of 10; C has no orders. Use complete coverage and a matched population and window. Expected parts by customer are `(60,2)`, `(60,1)`, `(0,0)`; overall AOV is `120/3 = 40`, not the unweighted buyer average `45`. Overall net revenue is `105`. Restricting the eligible roster to C gives zero orders and undefined AOV. A filtered numerator must not accidentally filter an unfiltered denominator. Compile **and execute** generated Malloy; matching file text is not evidence of parity.
3. Verify Terrine's per-snapshot formula and the Malloy result at the matched grain. Verify Increment's matched per-unit components on the shared-definition warehouse route, its supported experimental ratios against today's native definitions, and its explicit refusal of an unsupported experimental formula. A second timestamp-boundary example must retain the intentional difference: a Terrine outcome includes the event at its start; Increment's ordinary raw post-exposure outcome excludes an event exactly at exposure. Incomplete follow-up is not inactivity.
4. Publish a capability table with tested consumer versions, entry points, and semantic conditions, not a blanket “Malloy-compatible” claim. DuckDB is the first conformance backend; it is not evidence of equivalence on every warehouse dialect. Roll out the core contract first, then independent consumer integrations and exporter, then a pinned three-way acceptance run before recommending migration.

## Starting evidence

- [Increment realistic metrics](https://github.com/kylejcaron/increment/blob/fad84bc19d1127f1360b0a756c913161979c4be3/examples/realistic_demo/definitions/metrics.yaml): order-value ratio, filtered checkout latency, per-user metrics and exposure-relative retention are distinct existing cases.
- [Terrine's existing adapter](https://github.com/kylejcaron/terrine/blob/2d6758ea751505daef84dcbe073c5dfe9c5cc8e3/docs/guides/increment-compatibility.md): compatibility documented for Increment 0.1.0a2; metadata translation alone does not execute filters, windows or transforms.
- [Terrine's shared-dataset pilot](https://github.com/kylejcaron/terrine/blob/2d6758ea751505daef84dcbe073c5dfe9c5cc8e3/docs/guides/shared-dataset-contract.md): compares matched components and documents genuine differences in time grain and coverage.
- [Malloy fields](https://docs.malloydata.dev/documentation/language/fields) and [expressions](https://docs.malloydata.dev/documentation/language/expressions): named measures and arithmetic are native. The proposed YAML is not a Malloy input format.