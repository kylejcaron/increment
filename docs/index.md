# Increment

Increment is a Python library for warehouse-native A/B testing, causal inference, and experiment analysis. Analyze an experiment from a dataframe with no backend, or define experiments, metrics, and fact sources in YAML and run the same estimation engine against DuckDB, Snowflake, BigQuery, Postgres, or another Ibis backend.

Methods include exact constructions (such as Bernoulli e-processes), asymptotic
approximations (such as `asymptotic_mean` confidence sequences), and empirically
qualified procedures. Approximate results can carry advisory diagnostics rather
than a refusal; see [Statistical limitations](limitations.md) for their guarantees
and unresolved calibration limits.

Requires Python 3.12, 3.13, or 3.14.

## Install

`increment` core has no bundled database driver — you bring your own
[ibis](https://ibis-project.org) backend:

```bash
pip install increment
pip install 'ibis-framework[duckdb]'    # or [snowflake], [bigquery], [postgres], ...
```

| Extra | Adds | For |
|---|---|---|
| `increment` (core) | ibis, pydantic, numpy, scipy, sqlglot, narwhals, pyyaml | Always required; no backend included |
| `ibis-framework[<backend>]` | your chosen driver | Connecting to DuckDB, Snowflake, BigQuery, Postgres, etc. |
| `increment[demo]` | `ibis-framework[duckdb]` | Local DuckDB backend for demos, examples, and simulation |
| `increment[tables]` | `coeftable`, `pandas` | CoefTable-based experiment and metric trend tables |
| `increment[dashboard]` | `marimo`, `coeftable`, `pandas` | Reusable HTML dashboard over one bound experiment; see the [dashboard guide](guides/dashboard.md) |

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
import increment as inc

results = inc.Analysis.from_unit_summary(
    df,
    unit="user_id",
    group="variant",
    control="control",
    metrics={"revenue": "mean", "converted": "conversion"},
).run()

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
from increment import Analysis

con = ibis.snowflake.connect(...)
analysis = Analysis.from_definitions("checkout_redesign", "definitions/", con)

print(analysis.summary_sql()["revenue"])
results = analysis.run()
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
