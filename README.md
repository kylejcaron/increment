# increment

[![CI](https://github.com/kylejcaron/increment/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/kylejcaron/increment/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/increment?include_prereleases)](https://pypi.org/project/increment/)
[![Python](https://img.shields.io/pypi/pyversions/increment)](https://pypi.org/project/increment/)
[![License](https://img.shields.io/pypi/l/increment)](LICENSE)

> [!WARNING]
> **Increment is currently in pre-release.** APIs and supported combinations may
> change before a stable release. Review the [statistical limitations](docs/limitations.md)
> before relying on an analysis.

**Analyze experiments directly from your warehouse or a dataframe.**

Estimate treatment effects, quantify uncertainty, and inform decisions using data
that already exists. Define warehouse analyses in version-controlled YAML, or
start directly from a dataframe. Increment does not assign traffic, ship feature
flags, or maintain a separate event store.

[Quickstart](#quickstart) · [Warehouse analysis](#connect-your-warehouse) ·
[Documentation](#explore-the-guides) · [Statistical limitations](docs/limitations.md)

## What you can do

- **Analyze experiments** — estimate effects, reduce variance with CUPED, apply
  multiple-comparison corrections, and check sample-ratio mismatch.
- **Bayesian inference** — combine prior beliefs with experiment data to estimate effects and
  probabilities of benefit.
- **Plan and monitor** — solve for sample size, power, and minimum detectable
  effect; use sequential methods for repeated looks under their stated assumptions.
- **Estimate causal effects** — use IPTW, DML, or AIPW for observational treatments
  from dataframes, or analyze encouragement designs and switchback contrasts.
- **Explore and report** — inspect breakouts, trends, and heterogeneous treatment
  effects; produce readout tables, dashboards, and sitewide-impact estimates.

Not every method works with every metric or design. The
[method guide](docs/guides/choose-a-method.md) helps you choose; the
[compatibility matrix](docs/guides/compatibility.md) shows supported combinations.

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
    print(f"{result.metric}: lift={result.lift.value:+.1%}")
```

```text
revenue: lift=+7.5%
```

Replace the generated data with your own unit-level dataframe. No YAML, warehouse
connection, or DuckDB is required. Other Narwhals-supported eager dataframes work
as well. For unit-by-day data, use `Analysis.from_unit_panel`.

The [dataframe quickstart](docs/guides/quickstart.md) covers metric declarations,
missing values, analysis plans, uncertainty intervals, and monitoring.

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

See the [runnable warehouse example](examples/analysis_from_a_warehouse.py) and
[data-model guide](docs/guides/data-model.md) for YAML setup, query inspection,
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
| Choose an analysis method | [Method selection](docs/guides/choose-a-method.md) |
| Set up dataframe inputs | [Quickstart](docs/guides/quickstart.md) |
| Define warehouse data and experiments | [Data model](docs/guides/data-model.md) |
| Understand metrics and supported combinations | [Metric types](docs/guides/metric-types.md) · [Compatibility](docs/guides/compatibility.md) |
| Use priors and probability-based decisions | [Bayesian inference](docs/guides/priors-and-decisions.md) |
| Reduce variance | [CUPED](docs/guides/cuped.md) |
| Correct multiple comparisons | [Multiplicity](docs/guides/multiplicity.md) |
| Monitor an experiment | [Sequential inference](docs/guides/sequential-inference.md) |
| Plan sample size or power | [Power analysis](docs/guides/power-analysis.md) |
| Analyze non-randomized treatments | [Observational inference](docs/guides/observational.md) |
| Analyze encouragement or switchback designs | [Encouragement](docs/guides/encouragement.md) · [Switchback](docs/guides/switchback.md) |
| Explore treatment-effect differences | [Heterogeneity and rollout](docs/guides/heterogeneity-and-rollout.md) |
| Build an experiment dashboard | [Dashboard](docs/guides/dashboard.md) |
| Look up public APIs | [API reference](docs/api.md) |

The [`examples/`](examples/) directory contains runnable marimo notebooks for
warehouse and dataframe analyses, power, CUPED, causal inference, and reporting.
See the [examples guide](examples/README.md) to choose one.

## Scope and statistical assumptions

Increment analyzes randomized experiments from dataframes or SQL warehouses;
observational treatments are supported from dataframes. It estimates effects—it
does not make an observational comparison causal without identification assumptions.

Exact and asymptotic methods have different guarantees. Repeated looks require a
compatible sequential procedure to claim frequentist error control; Bayesian
posterior probabilities do not automatically provide that guarantee. Unsupported
combinations surface coded errors or warnings; partial results may be returned
when only some requested cells are unsupported.

Read the [statistical limitations](docs/limitations.md),
[compatibility matrix](docs/guides/compatibility.md), and
[backend verification notes](CONTRIBUTING.md#backend-verification) before choosing
an analysis for production use.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution
requirements. Contributions require a signed
[Contributor License Agreement](.github/CLA.md).

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
Copyright 2026 Kyle Caron.
