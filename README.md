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
estimates, and uncertainty intervals. Increment does not assign traffic, ship
feature flags, or maintain a separate event store.

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
- **Explore** — inspect breakouts, trends, and treatment-effect differences;
  create readout tables, dashboards, and sitewide-impact estimates, and track
  metric trends over calendar time.

Methods support different metrics and designs. Use the
[method guide](https://getincrement.io/docs/guides/choose-a-method/) to choose;
the [compatibility matrix](https://getincrement.io/docs/guides/compatibility/)
lists supported combinations.

## Quickstart

Requires **Python 3.12–3.14**. Install Increment's pre-release package and
Polars for this example:

```bash
pip install "increment>=0.1.0a2" polars
```

Start with one row per randomization unit. This example declares independent
50/50 assignment, makes revenue the primary metric, and includes complaints as
a guardrail. It deliberately omits treatment-arm complaint values so the
rendered result shows that missing guardrails are unavailable rather than
treated as safe.

```python
from random import Random
import warnings

import polars as pl

from increment import AnalysisPlan, MetricSpec, Randomized, from_frame, render, run, runnable
from increment.errors import IncrementWarning

rng = Random(7)
n = 500
arms = ["treatment" if rng.random() < 0.5 else "control" for _ in range(2 * n)]
revenue = [max(0.0, rng.gauss(10 + (arm == "treatment"), 4)) for arm in arms]
complaints = [int(rng.random() < 0.04) for _ in arms]
df = pl.DataFrame(
    {
        "user_id": [f"u{i:04d}" for i in range(2 * n)],
        "variant": arms,
        "revenue": revenue,
        "complaints": [
            None if arm == "treatment" else value
            for arm, value in zip(arms, complaints, strict=True)
        ],
    }
)

with warnings.catch_warnings(record=True) as data_warnings:
    warnings.simplefilter("always", IncrementWarning)
    exp, resolver = from_frame(
        df,
        shape="summary",
        unit="user_id",
        group="variant",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics=[
            "revenue",
            MetricSpec(
                name="complaints",
                type="conversion",
                missing="drop",
                preferred_direction="decrease",
            ),
        ],
        plan=AnalysisPlan(primary="revenue", guardrails=["complaints"]),
    )
    feasibility = runnable(exp, "revenue")
    result = run(exp, resolver)
for warning in data_warnings:
    print(warning.message)
rendered = render(result, "brief")
print(rendered)
```

`exp` describes the experiment; `resolver` binds its dataframe when `run`
reads the evidence. `runnable(exp, "revenue")` checks known incompatibilities
without reading data, including for every declared view. It is sound, not
complete: no refusal does not guarantee the run will succeed, because some
checks need data. Top-level `run` currently accepts one declared view at a
time; separate calls can read changed data. The experiment can be saved with
`exp.to_json()` and identified with `exp.hash()`. Rendering retains warnings
about missing guardrails, checks that were not completed, and what the data can
establish.

`shape="summary"` accepts summary-specific options such as
`intervention_grain="cluster"`. With one row per unit per day, use
`shape="panel"`, supply `date="event_day"`, and optionally set
`observation_end="2025-01-31"`. Editors and type checkers check the options
for the selected shape; a summary has no `date` or `observation_end`.
For an available guardrail and recovery, see the
[quickstart guide](https://getincrement.io/docs/guides/quickstart/#recover-the-unavailable-guardrail).

The separate compatibility route `Analysis.run()` still returns
`LiftEstimates` or `ContrastResults`; top-level `run(exp, resolver)` returns a
`RunResult` with result cells, effects, and decisions. `result.readout()` returns
the same typed collection, and `result.to_frame()` and `readout_table(result)`
read the result directly.

The [dataframe quickstart](https://getincrement.io/docs/guides/quickstart/) continues
with CUPED, monitoring, and other data shapes.

Next: [add CUPED](https://getincrement.io/docs/guides/cuped/#on-the-dataframe-path) ·
monitor sequentially ([mean or ratio](https://getincrement.io/docs/guides/sequential-inference/#ordinary-continuous-monitoring),
[conversion or retention](https://getincrement.io/docs/guides/sequential-inference/#sequential-monitoring-of-a-conversion-metric)) ·
[run a switchback](https://getincrement.io/docs/guides/switchback/#supported-contract) ·
[adjust for opt-in treatment](https://getincrement.io/docs/guides/observational/) ·
[other data shapes](https://getincrement.io/docs/guides/choose-a-method/).

## Connect your warehouse

Define sources, exposures, metrics, and experiments in YAML. Connect the
definitions to an Ibis backend and build the experiment:

<!-- skip: next "requires a configured warehouse and project definitions" -->

```python
import ibis
from increment import from_definitions, render, run

con = ibis.snowflake.connect(...)
experiment, resolver = from_definitions("definitions/", con, experiment_name="checkout_redesign")
with resolver:
    results = run(experiment, resolver)
print(render(results, "brief"))
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
