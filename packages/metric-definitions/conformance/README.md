# Verify shared metric definitions

Run this check when changing how Increment, Terrine, or the Malloy exporter reads the shared YAML. It uses the single [shop fixture](../examples/shop/): three customers, three orders, two operand-specific filters, and an explicit coverage interval. This checks candidate source builds on DuckDB, not compatibility of released packages or other warehouses.

## Run the candidate comparison

From the Increment checkout, set `TERRINE_CHECKOUT` to a Terrine checkout at the commit in `versions.json`. Install the locked dependencies, then install **the same built core wheel** into both environments.

```sh
core="$PWD/packages/metric-definitions"
receipt="$core/conformance/versions.json"
test "$(git -C "$TERRINE_CHECKOUT" rev-parse HEAD)" = "$(jq -r '.terrine.commit' "$receipt")"
python_version=$(jq -r '.python' "$receipt")
uv sync --locked --python "$python_version" --extra demo
uv sync --project "$TERRINE_CHECKOUT" --locked --python "$python_version" --extra duckdb --extra metric-definitions
wheel_dir=$(mktemp -d)
SOURCE_DATE_EPOCH=$(jq -r '.core.source_date_epoch' "$receipt") \
  uv build --directory "$core" --wheel --out-dir "$wheel_dir"
sha256sum "$wheel_dir"/*.whl
export INCREMENT_PYTHON="$PWD/.venv/bin/python"
export TERRINE_PYTHON="$TERRINE_CHECKOUT/.venv/bin/python"
uv pip install --no-deps --reinstall-package increment-metrics --python "$INCREMENT_PYTHON" "$wheel_dir"/*.whl
uv pip install --no-deps --reinstall-package increment-metrics --python "$TERRINE_PYTHON" "$wheel_dir"/*.whl
node_version=$(jq -r '.malloy.node' "$receipt")
npm exec --yes --package="node@$node_version" --package=npm@10.9.4 -- npm ci --prefix "$core/conformance/malloy"
MALLOY_NODE=$(npm exec --yes --package="node@$node_version" -- which node)
export MALLOY_NODE
uv run --project "$core" --locked --python "$python_version" pytest -q "$core/conformance/test_three_consumers.py"
```

Each consumer runs in its own process. The suite fails rather than skipping a missing runtime. The Malloy runner checks keys and runs the report on the same read-only database connection. Increment compares exposures and measure rows from a verified artifact, not a second query on the raw events. A zero for an enrolled unit without events is only inferred after both operands are certified complete.

## What must agree

| Request | Known answer |
|---|---|
| Revenue and order counts at the same customer and window | A: `(60, 2)`; B: `(60, 1)`; C: `(0, 0)` |
| Average order value at the reporting grain | `120 / 3 = 40`, not the average of 30 and 60 |
| Web revenue per all orders | `80 / 3`; the web filter does not restrict the denominator |
| Net revenue and purchase conversion | Net 105; 2 of 3 eligible customers purchased |
| Customer C without orders, with complete coverage | Undefined AOV; not zero |

An event exactly at the noon anchor is a separate **unmatched** case. Terrine's `[start, end)` window counts three A orders; Increment's ordinary post-exposure window counts two. When coverage ends before the requested window, Terrine retains all three customers and reports null with `incomplete_coverage`. When an experiment selects general `net_revenue` arithmetic, Increment refuses; descriptive consumers can still calculate it.

The three-customer fixture does not validate statistical inference. The Increment parity case `shared-completed-order-value` separately compares native and shared definitions on a populated experiment, including intervals and retained monitoring state. Terrine's pinned bridge check runs separately against released Increment `0.1.0a2`.

| Consumer path | Compared in the shop fixture | Refused or kept native |
|---|---|---|
| Increment `from_definitions` → `publish_unit_day_artifact` | Order-value and web-only revenue ratios: eligible units, numerator and denominator | General formula experiment inference refuses; assignments and analysis settings stay native |
| Terrine `Model.from_metric_definitions` → `Design.build` | Named formulas, ratios, means, conversion, empty and incomplete outcomes | Snapshot and readiness rules stay native |
| Malloy `export_malloy` → handwritten report | Customer grouping, ratios, arithmetic, eligible and buyer conversion | Only groupings defined for every operand; no import of Malloy definitions |

## Move an existing definition

| Field or behavior | Where it belongs |
|---|---|
| Table, row identifier, entity columns, event time, typed dimension, simple filter, sum/count, or arithmetic | Shared YAML when the consumer can represent it |
| Eligible denominator for a mean or conversion | Explicit shared roster; the native binding must identify that roster |
| Assignment, trigger, analysis plan, inference method and exposure-relative window | Existing Increment definitions; use `metric_definitions` to bind selected measures and metrics |
| Native source SQL, historical dimensions, unit roster bounds, window and snapshot design, readiness | Existing Terrine bindings and Design; use `Model.from_metric_definitions` for portable definitions |
| Legacy Increment YAML or SDK objects used by Terrine | Keep `Model.from_increment`; no migration required for existing users |
| Browser/report grouping and custom query layout | Handwritten Malloy report importing generated fields; generation does not overwrite the report |

For AOV, move the formula and its revenue/count operands into the shared file. Keep the experiment plan and completed-order source in [Increment's existing definitions](../../../examples/realistic_demo/definitions/metrics.yaml); the opt-in binds the shared operands to the native `order` event. In Terrine, change the model constructor, not the existing `Design` or window. For a Malloy report, export `generated.malloy` beside the [handwritten report](../examples/shop/report.malloy) and import it. Keep other native definitions in place when their semantics don't fit the shared format.

`versions.json` records the tested runtime revisions, dependency versions and wheel hash. CI builds the wheel from the checkout under test and logs its hash; the recorded hash identifies the candidate used for these results. A registry install and a new report against the released artifacts are separate checks.
