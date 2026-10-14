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

Increment is an open-source Python library for A/B test analysis. It helps teams
measure the impact of product changes using dataframes or SQL warehouses, with
reusable metric definitions, effect estimates, uncertainty intervals, and
shareable reports. Increment does not assign traffic, ship feature flags, or
maintain a separate event store.

[Website](https://getincrement.io/) · [Documentation](https://getincrement.io/docs/) ·
[Quickstart](#quickstart) · [Warehouse analysis](#connect-your-warehouse) ·
[Guides](#explore-the-guides) · [Statistical limitations](https://getincrement.io/docs/limitations/)

## What you can do

- **Analyze experiments** — estimate effects, reduce variance with CUPED, apply
  multiple-comparison corrections, and check sample-ratio mismatch when the
  assignment law is declared (`allocation_scheme="independent"`).
- **Bayesian inference** — combine prior beliefs with experiment data to estimate effects and
  probabilities of benefit.
- **Plan and monitor** — solve for sample size, power, and minimum detectable
  effect; use sequential methods for repeated looks under their stated assumptions.
- **Estimate causal effects** — use IPTW, DML, or AIPW for observational treatments
  from dataframes or warehouse definitions, or analyze encouragement designs and
  switchback contrasts.
- **Explore and report** — inspect breakouts, trends, and heterogeneous treatment
  effects; produce readout tables, dashboards, and sitewide-impact estimates.

Not every method works with every metric or design. The
[method guide](https://getincrement.io/docs/guides/choose-a-method/) helps you choose; the
[compatibility matrix](https://getincrement.io/docs/guides/compatibility/) shows supported combinations.

## Quickstart

Requires **Python 3.12–3.14**. Install the pre-release package and Polars for this
example:

```bash
pip install --pre increment polars
```

Start with one row per randomization unit. This example generates a small
control/treatment dataset and estimates the relative change in mean revenue:

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

```text
revenue: lift=+7.5%
```
**Read the result as a decision, not just a lift.** Check whether required
metrics and guardrails are present, whether assignment integrity was assessed,
and which multiplicity status applies before acting. The
[result-reading guide](https://github.com/kylejcaron/increment/blob/main/docs/guides/reading-results.md) explains failed or
unavailable cells, posterior availability, and triggered-population readouts.
For a coded refusal, follow its code and context with the
[refusal navigation guide](https://github.com/kylejcaron/increment/blob/main/docs/guides/refusals.md).


Replace the generated data with your own unit-level dataframe. No YAML, warehouse
connection, or DuckDB is required. Other Narwhals-supported eager dataframes work
as well. For unit-by-day data, use `Analysis.from_unit_panel`.

The [dataframe quickstart](https://getincrement.io/docs/guides/quickstart/) covers metric declarations,
missing values, analysis plans, uncertainty intervals, and monitoring.

Next: [add CUPED](https://getincrement.io/docs/guides/cuped/#on-the-dataframe-path) ·
monitor sequentially ([mean or ratio](https://getincrement.io/docs/guides/sequential-inference/#ordinary-continuous-monitoring),
[conversion or retention](https://getincrement.io/docs/guides/sequential-inference/#sequential-monitoring-of-a-conversion-metric)) ·
[run a switchback](https://getincrement.io/docs/guides/switchback/#supported-contract) ·
[adjust for opt-in treatment](https://getincrement.io/docs/guides/observational/) ·
[other data shapes](https://getincrement.io/docs/guides/choose-a-method/).

## Connect your warehouse

Declare sources, exposures, metrics, and experiments in YAML. Bind the definitions
to an Ibis connection to inspect the generated SQL and run an analysis:

<!-- skip: next "requires a configured warehouse and project definitions" -->

```python
import ibis
from increment import Analysis

con = ibis.snowflake.connect(...)
analysis = Analysis.from_definitions("checkout_redesign", "definitions/", con)

print(analysis.summary_sql()["revenue"])
results = analysis.run()
```

Install your warehouse's Ibis backend separately—for example,
`pip install 'ibis-framework[snowflake]'`. The core package bundles no database
driver. DuckDB, PostgreSQL, Snowflake, and BigQuery have execution test suites;
coverage is not a promise that every method works on every backend.

See the [runnable warehouse example](https://github.com/kylejcaron/increment/blob/main/examples/analysis_from_a_warehouse.py) and
[data-model guide](https://getincrement.io/docs/guides/data-model/) for YAML setup, query inspection,
portable unit-day artifacts, and moments exports.

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

The [`examples/`](https://github.com/kylejcaron/increment/tree/main/examples) directory contains runnable marimo notebooks for
warehouse and dataframe analyses, power, CUPED, causal inference, and reporting.
See the [examples guide](https://github.com/kylejcaron/increment/blob/main/examples/README.md) to choose one.

## Scope and statistical assumptions

Increment analyzes randomized experiments from dataframes or SQL warehouses;
observational treatments are supported from dataframes or warehouse definitions. It estimates effects—it
does not make an observational comparison causal without identification assumptions.

Exact and asymptotic methods have different guarantees. Repeated looks require a
compatible sequential procedure to claim frequentist error control; Bayesian
posterior probabilities do not automatically provide that guarantee. Unsupported
combinations surface coded errors or warnings; partial results may be returned
when only some requested cells are unsupported.

Read the [statistical limitations](https://getincrement.io/docs/limitations/),
[compatibility matrix](https://getincrement.io/docs/guides/compatibility/), and
[backend verification notes](https://github.com/kylejcaron/increment/blob/main/CONTRIBUTING.md#backend-verification) before choosing
an analysis for production use.

The [validation coverage inventory](https://github.com/kylejcaron/increment/blob/main/docs/validation.md#reference-inventory) lists which
estimators have frozen third-party references and which regimes have none.
How guarantee, evidence and software compatibility differ for each capability is stated in
[Evidence status](https://github.com/kylejcaron/increment/blob/main/docs/validation.md#evidence-status).

## Contributing and license

See [CONTRIBUTING.md](https://github.com/kylejcaron/increment/blob/main/CONTRIBUTING.md) for development setup and contribution
requirements. Contributions require a signed
[Contributor License Agreement](https://github.com/kylejcaron/increment/blob/main/.github/CLA.md).

Apache License 2.0. See [LICENSE](https://github.com/kylejcaron/increment/blob/main/LICENSE) and [NOTICE](https://github.com/kylejcaron/increment/blob/main/NOTICE).
Copyright 2026 Kyle Caron.
