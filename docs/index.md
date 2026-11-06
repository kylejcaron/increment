# Increment

Analyze A/B tests in Python from dataframes or SQL warehouses. Increment provides reusable metric definitions, effect estimates, uncertainty intervals, and shareable reports.

Methods include exact constructions (such as Bernoulli e-processes), asymptotic approximations (such as `asymptotic_mean` confidence sequences), and empirically qualified procedures. Results may include diagnostics instead of refusing; see [Statistical limitations](limitations.md) for guarantees and known calibration gaps.

Requires Python 3.12 or newer. Core and dataframe analysis are tested on Python
3.12–3.15; warehouse workflows are tested on 3.12–3.14. DuckDB-backed extras on
Python 3.15 need a compatible DuckDB wheel.

## Install

Install the core package for dataframe analysis:

```bash
pip install increment
```

For a local DuckDB backend, use `increment[demo]`, which installs a compatible
Ibis/DuckDB combination. For another SQL warehouse, add warehouse support and
the matching Ibis backend:

```bash
pip install "increment[demo]"  # local DuckDB
pip install "increment[warehouse]" "ibis-framework[snowflake]"  # remote warehouse
```

| Extra | Adds | For |
|---|---|---|
| `increment` (core) | pydantic, numpy, scipy, Arrow, narwhals, pyyaml | Dataframe and moments analysis |
| `increment[warehouse]` | Ibis, SQLGlot | SQL warehouses; install a backend driver separately |
| `ibis-framework[<backend>]` | your chosen backend driver | Connecting to remote warehouses such as Snowflake, BigQuery, or Postgres |
| `increment[demo]` | Warehouse support and compatible Ibis/DuckDB dependencies | Local DuckDB backend for demos, examples, and simulation |
| `increment[tables]` | `coeftable`, `pandas` | CoefTable-based experiment and metric trend tables |
| `increment[dashboard]` | `marimo`, `coeftable`, `pandas` | Reusable HTML dashboard over one bound experiment; see the [dashboard guide](guides/dashboard.md) |

Any Ibis SQL backend is supported. DuckDB, PostgreSQL, Snowflake, and BigQuery
are verified by execution test suites; other backends show a one-time
`UnverifiedWarehouseWarning`. An analysis runs when Ibis can translate its
queries for that backend, and stops with `query.backend.operation_unsupported`
when it cannot. The default run and warehouse CUPED need TEMP tables; unit-day
artifacts need a verified backend. See
[Other warehouses](guides/data-model.md#other-warehouses).

## Quickstart

Start without YAML or a warehouse connection: one row per unit is all it
takes. Not sure which design or estimator fits your data? Start with [Which
method should I use?](guides/choose-a-method.md).

Before acting on a result, read [Statistical limitations](limitations.md). It
states in one place what each estimator assumes and where its approximations
bite.

Which estimators have frozen third-party references, and which regimes have none, is
inventoried in [External validation coverage](validation.md#reference-inventory).

The [Evidence status](validation.md#evidence-status) section of that page states, per capability,
its statistical guarantee and what has been checked, kept apart from the software compatibility
policy.

<!-- invisible-code-block: python
import random

import polars as pl

random.seed(0)
n = 60
df = pl.DataFrame(
    {
        "user_id": [f"u{i:03d}" for i in range(2 * n)],
        "variant": ["control"] * n + ["treatment"] * n,
        "revenue": [max(0.0, random.gauss(10, 4)) for _ in range(2 * n)],
        "converted": [int(random.random() < 0.3) for _ in range(2 * n)],
    }
)
-->

```python
import increment

experiment, reader = increment.from_frame(
    df,
    shape="summary",
    unit="user_id",
    group="variant",
    control="control",
    metrics={"revenue": "mean", "converted": "conversion"},
)
results = increment.run(experiment, reader).readout()

for r in results:
    if r.lift is None:  # e.g. an exact conversion row with zero control events
        print(repr(r))  # the repr shows the row's set or its unavailable reason
        continue
    print(f"{r.metric} / {r.group_id}: lift={r.lift.value:.2%}")
```

## Analyze from your warehouse

Declare sources, exposures, metrics, and experiments in YAML, then bind the
definitions to an Ibis connection to inspect the generated SQL and run the same
analysis:

<!-- skip: next "requires a configured warehouse and project definitions" -->

```python
import ibis

import increment
from increment.query import summary_sql

con = ibis.snowflake.connect(...)
experiment, reader = increment.from_definitions(
    "definitions/", con, experiment_name="checkout_redesign"
)

print(summary_sql(experiment, reader)["revenue"])
results = increment.run(experiment, reader)
```

The [data model guide](guides/data-model.md) covers the YAML setup. The
[warehouse analysis example](examples/analysis_from_a_warehouse.md) runs end to end
against an in-memory DuckDB database.

## What do you want to do next?

| Job | Where to go |
|---|---|
| Add CUPED variance reduction | [CUPED on the dataframe path](guides/cuped.md#on-the-dataframe-path) |
| Monitor while the experiment runs (conversion or retention) | [Sequential monitoring of a conversion metric](guides/sequential-inference.md#sequential-monitoring-of-a-conversion-metric) |
| Monitor while the experiment runs (mean or ratio) | [Ordinary continuous monitoring](guides/sequential-inference.md#ordinary-continuous-monitoring) |
| Run a switchback | [Switchback supported contract](guides/switchback.md#supported-contract) |
| Estimate an opt-in or self-selected treatment | [Observational inference](guides/observational.md) |
| Plan sample size or power | [Power analysis](guides/power-analysis.md) |
| Use daily panels, a published unit-day artifact, or a moments file | [One row per unit per day](guides/quickstart.md#one-row-per-unit-per-day), [Canonical unit-day artifacts](guides/data-model.md#canonical-unit-day-artifacts), [Moments wire migration](guides/data-model.md#moments-wire-migration); which entry points support what: [What runs where](limitations.md#what-runs-where) |
| Anything else | [Which method should I use?](guides/choose-a-method.md) |

Method assumptions are in [Statistical limitations](limitations.md). Unit-day
artifacts, publication, and recovery are in [Canonical unit-day
artifacts](guides/data-model.md#canonical-unit-day-artifacts), and decision and
sensitivity methods are in [Choosing decision and sensitivity
methods](guides/choose-a-method.md#choosing-decision-and-sensitivity-methods). See
the [API Reference](api.md) for the full surface.
