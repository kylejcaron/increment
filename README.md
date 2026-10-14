# increment

[![CI](https://github.com/kylejcaron/increment/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/kylejcaron/increment/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/increment?include_prereleases)](https://pypi.org/project/increment/)
[![Python](https://img.shields.io/pypi/pyversions/increment)](https://pypi.org/project/increment/)
[![License](https://img.shields.io/pypi/l/increment)](https://github.com/kylejcaron/increment/blob/main/LICENSE)

> [!WARNING]
> **Increment is currently in pre-release.** APIs and supported combinations may
> change before a stable release; see the
> [compatibility policy](https://getincrement.io/docs/api/pre-1-0-compatibility/). Review the
> [statistical limitations](https://getincrement.io/docs/limitations/) before relying on an analysis.

**A/B testing that shows its work.**

Increment is an open-source Python library for A/B test analysis. Analyze
experiments from dataframes or SQL warehouses with reusable metrics, effect
estimates, uncertainty intervals, and reports. Increment does not assign traffic,
ship feature flags, or maintain a separate event store.

[Website](https://getincrement.io/) · [Documentation](https://getincrement.io/docs/) ·
[Quickstart](#quickstart) · [Warehouse analysis](#connect-your-warehouse) ·
[Guides](#explore-the-guides) · [Statistical limitations](https://getincrement.io/docs/limitations/)

## What you can do

- **Analyze experiments** — estimate effects, reduce variance with CUPED, apply
  multiple-comparison corrections, and check sample-ratio mismatch after
  declaring `allocation_scheme="independent"` or explicitly requesting a
  one-look `srm(inference="fixed")` diagnostic. `health_check()` reports when
  assignment checks make SRM unavailable.
- **Use Bayesian inference** — combine prior beliefs with experiment data to
  estimate effects and probabilities of benefit.
- **Plan and monitor** — calculate sample size, power, and minimum detectable
  effect; use a sequential method for repeated looks when its assumptions fit.
- **Estimate causal effects** — use IPTW, DML, or AIPW for observational
  treatments, or analyze encouragement and switchback designs.
- **Explore and report** — inspect breakouts, trends, and treatment-effect
  differences; create readout tables, dashboards, and sitewide-impact estimates.

Methods support different metrics and designs. Use the
[method guide](https://getincrement.io/docs/guides/choose-a-method/) to choose;
the [compatibility matrix](https://getincrement.io/docs/guides/compatibility/)
lists supported combinations.

## Quickstart

Requires **Python 3.12–3.14**. Install Increment's pre-release package and
Polars for this example:

```bash
pip install --pre increment polars
```

Use one row per randomization unit. This example creates control and treatment
data, then estimates the relative change in average revenue:

```python
from random import Random

import polars as pl
from increment import Analysis

rng = Random(0)
df = pl.DataFrame(
    {
        "user_id": [f"u{i}" for i in range(200)],
        "variant": ["control"] * 100 + ["treatment"] * 100,
        "revenue": [rng.gauss(20 + (i >= 100), 5) for i in range(200)],
    }
)

results = Analysis.from_unit_summary(
    df,
    unit="user_id",
    group="variant",
    control="control",
    metrics={"revenue": "mean"},
).run()

for result in results:
    if result.lift is None:  # e.g. an exact conversion row with zero control events
        print(repr(result))  # the repr shows the row's set or its unavailable reason
        continue
    print(f"{result.metric}: lift={result.lift.value:+.1%}")
```

<!-- output -->
```text
revenue: lift=+7.5%
```
**Use the result to support a decision.** Check that required metrics and
guardrails are present, assignment integrity was assessed, and the right
multiplicity status applies. The
[result-reading guide](https://github.com/kylejcaron/increment/blob/main/docs/guides/reading-results.md)
covers unavailable cells, posterior availability, and triggered-population
readouts. For a refusal, follow its code and context in the
[refusal guide](https://github.com/kylejcaron/increment/blob/main/docs/guides/refusals.md).


Replace the generated data with your unit-level dataframe. No YAML, warehouse
connection, or DuckDB is needed. Other eager dataframes supported by Narwhals
also work. For daily data, use `Analysis.from_unit_panel`.

The [dataframe quickstart](https://getincrement.io/docs/guides/quickstart/)
covers metrics, missing values, analysis plans, uncertainty, and monitoring.

Next: [add CUPED](https://getincrement.io/docs/guides/cuped/#on-the-dataframe-path) ·
monitor sequentially ([mean or ratio](https://getincrement.io/docs/guides/sequential-inference/#ordinary-continuous-monitoring),
[conversion or retention](https://getincrement.io/docs/guides/sequential-inference/#sequential-monitoring-of-a-conversion-metric)) ·
[run a switchback](https://getincrement.io/docs/guides/switchback/#supported-contract) ·
[adjust for opt-in treatment](https://getincrement.io/docs/guides/observational/) ·
[other data shapes](https://getincrement.io/docs/guides/choose-a-method/).

## Connect your warehouse

Define sources, exposures, metrics, and experiments in YAML. Connect the
definitions to an Ibis backend to inspect generated SQL and run the analysis:

<!-- skip: next "requires a configured warehouse and project definitions" -->

```python
import ibis
from increment import Analysis

con = ibis.snowflake.connect(...)
analysis = Analysis.from_definitions("checkout_redesign", "definitions/", con)

print(analysis.summary_sql()["revenue"])
results = analysis.run()
```

Install the Ibis backend for your warehouse separately, such as
`pip install 'ibis-framework[snowflake]'`. The core package includes no database
driver. DuckDB, PostgreSQL, Snowflake, and BigQuery have execution test suites;
that does not mean every method works on every backend.

See the [runnable warehouse example](https://github.com/kylejcaron/increment/blob/main/examples/analysis_from_a_warehouse.py)
and [data-model guide](https://getincrement.io/docs/guides/data-model/) for YAML,
query inspection, portable unit-day artifacts, and moments exports.

## Optional extras

| Install | Adds |
|---|---|
| `increment[demo]` | DuckDB for local examples and simulation |
| `increment[tables]` | CoefTable and pandas for rendered readouts |
| `increment[dashboard]` | marimo, CoefTable, and pandas for an HTML experiment dashboard |

## Explore the guides

| Goal | Guide |
|---|---|
| Choose an analysis method | [Method selection](https://getincrement.io/docs/guides/choose-a-method/) |
| Set up dataframe inputs | [Quickstart](https://getincrement.io/docs/guides/quickstart/) |
| Define warehouse data and experiments | [Data model](https://getincrement.io/docs/guides/data-model/) |
| Understand metrics and supported combinations | [Metric types](https://getincrement.io/docs/guides/metric-types/) · [Compatibility](https://getincrement.io/docs/guides/compatibility/) |
| Use priors and probability-based decisions | [Bayesian inference](https://getincrement.io/docs/guides/priors-and-decisions/) |
| Reduce variance | [CUPED](https://getincrement.io/docs/guides/cuped/#on-the-dataframe-path) |
| Correct multiple comparisons | [Multiplicity](https://getincrement.io/docs/guides/multiplicity/) |
| Monitor an experiment | [Mean or ratio metrics](https://getincrement.io/docs/guides/sequential-inference/#ordinary-continuous-monitoring) · [Conversion or retention metrics](https://getincrement.io/docs/guides/sequential-inference/#sequential-monitoring-of-a-conversion-metric) |
| Plan sample size or power | [Power analysis](https://getincrement.io/docs/guides/power-analysis/) |
| Analyze non-randomized treatments | [Observational inference](https://getincrement.io/docs/guides/observational/) |
| Analyze encouragement or switchback designs | [Encouragement](https://getincrement.io/docs/guides/encouragement/) · [Switchback](https://getincrement.io/docs/guides/switchback/#supported-contract) |
| Explore treatment-effect differences | [Heterogeneity and rollout](https://getincrement.io/docs/guides/heterogeneity-and-rollout/) |
| Build an experiment dashboard | [Dashboard](https://getincrement.io/docs/guides/dashboard/) |
| Look up public APIs | [API reference](https://getincrement.io/docs/api/) |

The [`examples/`](https://github.com/kylejcaron/increment/tree/main/examples)
directory has runnable marimo notebooks for warehouse and dataframe analyses,
power, CUPED, causal inference, and reporting. Use the
[examples guide](https://github.com/kylejcaron/increment/blob/main/examples/README.md)
to choose one.

## Scope and statistical assumptions

Increment analyzes randomized experiments from dataframes or SQL warehouses.
It also supports observational treatments from dataframes and warehouse
definitions, but causal conclusions require identification assumptions.

Exact and approximate methods have different guarantees. Repeated looks need a
compatible sequential procedure for frequentist error control; Bayesian
probabilities do not provide that guarantee by themselves. Unsupported
combinations produce coded errors or warnings. A result may include supported
cells when others are unavailable.

Before production use, read the
[statistical limitations](https://getincrement.io/docs/limitations/),
[compatibility matrix](https://getincrement.io/docs/guides/compatibility/), and
[backend verification notes](https://github.com/kylejcaron/increment/blob/main/CONTRIBUTING.md#backend-verification).

The [validation inventory](https://github.com/kylejcaron/increment/blob/main/docs/validation.md#reference-inventory)
lists methods with frozen third-party references and regimes without one.
[Evidence status](https://github.com/kylejcaron/increment/blob/main/docs/validation.md#evidence-status)
explains each capability's guarantee and evidence, separately from software
compatibility.

## Contributing and license

See [CONTRIBUTING.md](https://github.com/kylejcaron/increment/blob/main/CONTRIBUTING.md)
for setup and contribution requirements. Contributions require a signed
[Contributor License Agreement](https://github.com/kylejcaron/increment/blob/main/.github/CLA.md).

Apache License 2.0. See [LICENSE](https://github.com/kylejcaron/increment/blob/main/LICENSE)
and [NOTICE](https://github.com/kylejcaron/increment/blob/main/NOTICE).
Copyright 2026 Kyle Caron.
