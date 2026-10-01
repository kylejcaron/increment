# increment

> **NOTICE**
> This package is currently in its pre-release phase.

**A Python library for warehouse-native A/B testing, causal inference, and experiment analysis.**

Increment analyzes randomized experiments from dataframes or SQL warehouses and observational treatments from dataframes. It compiles inspectable Ibis queries, estimates effects with explicit assumptions, and returns diagnostics and readouts. Exact constructions are identified as exact; asymptotic and empirically qualified methods carry those qualifications, while unsupported or unfinished capabilities refuse rather than proceeding silently — see [Statistical limitations](docs/limitations.md).

It analyzes data that already exists; it does not assign traffic, ship feature flags, ingest events, or maintain a separate event store.

## Core features

- **Warehouse-native A/B testing** — define experiments and metrics in YAML, then compile
  inspectable queries through Ibis to your warehouse.
- **A/B testing from a dataframe** — analyze unit-summary or unit-by-day data without a
  warehouse connection.
- **Power analysis** — solve for sample size, achieved power, and minimum detectable effect.
- **Multiple-comparison corrections** — control metric and view families through the declared
  analysis plan.
- **CUPED** — reduce variance with pre-treatment covariates: fixed-horizon mean and
  ratio CUPED, and asymptotic sequential adjustment under `asymptotic_mean`. The
  exact Bernoulli e-process route and sources lacking the required covariate moments
  refuse; see the [CUPED guide](docs/guides/cuped.md) for each path.
- **Bayesian inference** — combine prior beliefs with experiment data to estimate effects and
  probabilities of benefit.
- **Switchback contrasts** — estimate fixed-horizon additive treatment-versus-control
  differences over randomized within-unit cycles.
- **Sequential analysis and always-valid inference** — monitor experiments with confidence
  sequences (exact Bernoulli e-processes or asymptotic confidence sequences, under
  their stated assumptions) when the declared methods are compatible.
- **Encouragement designs** — report intent-to-treat, compliance, and local average treatment
  effects.
- **Observational causal inference** — estimate non-randomized treatments with IPTW, DML, or
  doubly robust AIPW.
- **Sample-ratio mismatch diagnostics** — check allocation with anytime-valid or fixed-look
  methods.
- **Heterogeneous treatment effects** — estimate CATE, inspect GATES/CLAN profiles, and validate
  targeting rules.
- **Reporting and decision support** — produce breakouts, trends, treatment recommendations,
  and sitewide-impact estimates.

## Install

Requires Python 3.12, 3.13, or 3.14.

```bash
pip install increment
```

The core package has no bundled database driver. Install the Ibis backend for your warehouse
only when you need it:

```bash
pip install 'ibis-framework[duckdb]'    # or [snowflake], [bigquery], [postgres], ...
```

| Extra | Adds | Use it for |
|---|---|---|
| `increment` | Statistics, dataframe support, and query construction | All analyses |
| `increment[demo]` | DuckDB | Local demos, examples, and simulation |
| `increment[tables]` | CoefTable and pandas | Rendered experiment and trend tables |
| `increment[dashboard]` | marimo, CoefTable, and pandas | A reusable HTML dashboard over one bound experiment |

**Backend support tiers.** DuckDB executes in the default test suite on every run.
PostgreSQL, Snowflake and BigQuery each have a live-execution probe suite under
`integration/warehouse_execution/`. Snowflake and BigQuery additionally have SQL
compilation checks in the default suite. CI runs the PostgreSQL 16 probes on every pull
request; all three backends run weekly and on manual dispatch through the
[warehouse gate](.github/workflows/warehouse-backends.yml), which fails, never skips, when
credentials are absent. These hosted gates have not yet been observed running, so read the
tier as verified by developer runs.

The live probes cover ratio aggregation (+25% lift), materialized mean correction
(`2/11` to `15/11` lift), and conversion artifact parity (-50% lift, 60/60 arm counts)
with cluster identity and assigned-population counts. They check physical relations
and cleanup isolation, not just compiled SQL. This is not execution coverage for
retention, quantiles, encouragement/switchback designs, or all artifact extensions.

## Governed, warehouse-backed analysis

Declare fact and dimension sources, exposures, metrics, and experiments in version-controlled
YAML. These definitions make enrollment, metric windows, and analysis plans shared and reviewable:

```text
definitions/
  fact_sources.yaml   # SQL exposing event tables
  dim_sources.yaml    # optional dimension tables joined into fact sources
  exposures.yaml      # what enrolls a unit
  metrics.yaml        # mean, conversion, retention, and ratio metrics
  experiments.yaml    # schedules, analysis plans, and metric lists
```

Bind those definitions to any Ibis connection. Increment builds the analysis queries, compiles
them to the backend dialect, and returns the same `LiftEstimate` result models as the dataframe
path.

<!-- skip: next "requires a configured warehouse and project definitions" -->

```python
import ibis
from increment import Analysis

con = ibis.snowflake.connect(...)
analysis = Analysis.from_definitions("checkout_redesign", "definitions/", con)

print(analysis.summary_sql()["revenue"])
results = analysis.run()
```

Fact-source SQL declares its authoring dialect and is transpiled to the connection's dialect at
execution time. There is no default warehouse and no separate Increment event store.

See the runnable [warehouse example](examples/analysis_from_a_warehouse.py) and the
[data-model guide](docs/guides/data-model.md).

### Portable unit-day artifacts

Definitions-backed analyses can publish the canonical unit-day evidence once
through `analysis.publish_unit_day_artifact(store, extensions=...)`, receiving a
caller-pinned `UnitDayArtifactRef`. A later process adopts that immutable
generation with `Analysis.from_unit_day_artifact(store, ref,
expected_context=...)`; adoption performs no upstream fact or exposure build.
The artifact carries context format 2, a manifest with base exposures and
`measure_stats` relations, metric bindings, context, freshness, and digest
metadata, and keeps each read on one immutable snapshot.

If publication raises after the manifest is written, `WarehouseArtifactStore`
invalidates that generation through `drop_generation`: a fresh store lists
nothing for it and any read is refused with `artifact.generation.dropped`. If the
invalidation itself fails, the original exception carries a note naming the
`artifact_id` and `generation_id`; run `store.drop_generation(artifact_id,
generation_id)` from a fresh process, which is safe to repeat. If the manifest
insert was submitted but its outcome is unknown, the abort tombstones the
generation before erasing its relations, so a manifest row that commits late stays
hidden. Erasure continues past a failed drop, so every relation is attempted. If
that tombstone cannot be written, or a relation cannot be dropped afterwards, the
relations are kept: an aborting exception carries a note naming them (qualified by
catalog and schema) with the `artifact_id` and `generation_id`, and a publication
that ends normally without a manifest raises
`query.session.warehouse_artifact.publication_state_unknown` with the same
qualified names, the identifiers and a `route` in its context. A failure before the
manifest insert is submitted that leaves relations behind, including a relation
whose write failed and whose cleanup drop also failed, is reported the same way: an
aborting exception carries the note, and a publication that ends normally raises
`query.session.warehouse_artifact.publication_cleanup_incomplete` instead. No row
can appear, so drop the listed relations by name. When the outcome is unknown,
call `store.abandon_generation(artifact_id, generation_id)` from a fresh process
first (durable, safe to repeat), then drop the listed relations by name; if a
manifest row exists, `drop_generation` also works. See the
[data model guide](docs/guides/data-model.md#canonical-unit-day-artifacts)
for the operator recovery procedure.

`materialize()` is only a session-local TEMP realization and is not portable.
`from_moments()` is a separate moments-cube seam that reads formats 7–9 and
requires a compiled decision plan. Current exports write format 8 for
fixed-horizon cubes and format 9 for sequential checkpoints; a sequential plan
replays only from format 9, and this wire version is distinct from the artifact
format above. Legacy formats 1–6 cannot be converted without the raw
definitions: reconstruct the definitions and re-export, or pin the old
Increment revision that wrote the cube.

## A/B testing from a dataframe

The shortest path starts with one row per randomization unit. No YAML, warehouse connection, or
DuckDB required. This snippet uses [polars](https://pola.rs); any
[narwhals](https://narwhals-dev.github.io/narwhals/)-supported eager dataframe works.

```bash
pip install increment polars
```

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
from increment import Analysis

results = Analysis.from_unit_summary(
    df,
    unit="user_id",
    group="variant",
    control="control",
    metrics={"revenue": "mean", "converted": "conversion"},
).run()

for result in results:
    print(f"{result.metric} / {result.group_id}: lift={result.lift.value:.2%}")
```

`Analysis.from_unit_panel` accepts one row per unit per day and adds ratio metrics and SRM
checks. The [quickstart](docs/guides/quickstart.md) covers both dataframe shapes, missing-value
policies, analysis plans, and always-valid monitoring.

## Causal inference for non-randomized treatments

An opt-in or phased treatment is not a randomized experiment. Replace `control=` with an
`Observational` design, declare the adjustment set, and choose IPTW, DML, or AIPW. Increment
checks identification and overlap rather than silently treating the raw arm difference as a
causal effect.

<!-- invisible-code-block: python
import numpy as np
import polars as pl

rng = np.random.default_rng(5)
n = 400
tenure_days = rng.gamma(shape=4.0, scale=90.0, size=n)
plan_tier_rank = rng.integers(0, 3, size=n).astype(float)
logit = -0.6 + 0.002 * (tenure_days - 360) + 0.3 * (plan_tier_rank - 1)
treated = rng.random(n) < 1 / (1 + np.exp(-logit))
base = 20 + 0.02 * tenure_days + 4.0 * plan_tier_rank
revenue = base * np.where(treated, 1.05, 1.0) + rng.normal(0, 4, size=n)
df = pl.DataFrame(
    {
        "user_id": [f"u{i}" for i in range(n)],
        "variant": np.where(treated, "opted_in", "control"),
        "revenue": revenue,
        "tenure_days": tenure_days,
        "plan_tier_rank": plan_tier_rank,
    }
)
-->

```python
from increment import AdjustmentSet, Analysis, IdentificationGate, Method, Observational

design = Observational(
    control_group="control",
    adjustment=AdjustmentSet(covariates=("tenure_days", "plan_tier_rank")),
    gate=IdentificationGate(overlap="trim"),
)

results = Analysis.from_unit_summary(
    df,
    unit="user_id",
    group="variant",
    metrics={"revenue": "mean"},
    design=design,
).run(decision_method=Method(name="aipw"))
```

The [observational guide](docs/guides/observational.md) documents estimands, overlap policies,
learner protocols, diagnostics, and the cases Increment deliberately refuses to estimate.


## One engine across the analysis lifecycle

| Need | Entry point | What Increment owns |
|---|---|---|
| Analyze an experiment already in memory | `Analysis.from_unit_summary` | Validation, estimation, diagnostics, results |
| Analyze a unit-by-day panel | `Analysis.from_unit_panel` | Metric aggregation, ratio inference, SRM, estimation |
| Estimate a switchback contrast | `Analysis.from_switchback_panel` | Retained-window additive contrast; unit-t approximation by default for unit-cycle orders (optional prospective envelope), block-t for `shared_schedule` |
| Estimate a non-randomized treatment | `Observational` + AIPW/DML/IPTW | Adjustment, overlap gates, causal estimation |
| Run governed warehouse metrics | `Analysis.from_definitions` | Definitions, query construction, estimation |
| Track metrics without an experiment | `Report.from_definitions` | Calendar and rolling-window metric reports |
| Plan an experiment | `increment.power` | Sample size, power, and minimum detectable effect |
| Choose an estimator | [Which method should I use?](docs/guides/choose-a-method.md) | Assignment, monitoring, estimand, and identification guidance |

## Explicit decision and sensitivity roles

Arm analyses resolve one `decision_method` and zero or more
`sensitivity_methods` per metric. Set these on a definitions-plan binding or
`MetricSpec`; a call-wide `run(decision_method=..., sensitivity_methods=...)`
override applies to every selected metric and takes precedence over bindings.
Sensitivity rows are reporting comparisons and do not create additional decision
evidence. The legacy method-list keyword is not supported.

Switchback analyses are a separate contrast evidence family. Independent unit-cycle
orders use the qualified unit-t approximation by default, with no reference argument:
it needs at least two independent units and reports `dof=n_units-1` with
`switchback_unit_t_approximation` / `unit_t_approximation`. Passing
`UnitCycleTApproximation()` explicitly is equivalent. A separately supplied valid
prospective `UnitCycleVarianceEnvelope` is a stronger route and can support one unit;
a pilot or sample variance is not such an envelope. Shared schedules use
`switchback_block_t` / `block_t` with at least two blocks and a fixed roster of one
or more units; units, cycles and blocks are distinct independence grains. They accept no role,
sensitivity, or prior overrides. See the [switchback migration guide](docs/guides/switchback.md).

## Inspect SQL and render results

Warehouse analyses expose their generated queries before execution:

<!-- skip: next "continues the configured warehouse example above" -->

```python
panel_queries = analysis.panel_sql()
summary_queries = analysis.summary_sql()

print(panel_queries["purchase_rate"])
print(summary_queries["revenue"])
```

Install `increment[tables]` for notebook-friendly readouts:

```python
from increment.tables import estimates_to_readout, readout_table

readout = readout_table(estimates_to_readout(results))
readout  # renders in Jupyter and marimo
```

## Power analysis

Use the same metric assumptions before launch to solve for sample size, achieved power, or
minimum detectable effect.

```python
from increment import Baseline
from increment.power import ArmPlanningProcedure, required_sample_size

# `standard` fills the ordinary choices -- randomized parallel test, total
# grain, unadjusted, fixed horizon -- and names them in its docstring.
procedure = ArmPlanningProcedure.standard("mean")
result = required_sample_size(
    relative_lift=0.10,
    baseline=Baseline(mean=1.0, var=2.0),
    procedure=procedure,
)
print(f"Need {result.n_total} total units ({result.n_per_arm} per arm)")
```

See the [power-analysis guide](docs/guides/power-analysis.md) for the
procedure-first solver API, CUPED-adjusted fixed-horizon planning, and the
sequential/cluster constraints.

## A/B testing dashboard

Install `increment[dashboard]` for a reusable HTML dashboard over one bound experiment:
allocation health, the full role-grouped CoefTable readout, optional metric time trends and
segments, and a downloadable CSV. It is section-level `mo.Html` helpers over the public
`Analysis`/readout API, not a private example — bind your own experiment, prepare one
snapshot, and reuse the same rendering calls.

<!-- skip: next "requires increment[dashboard] and a configured warehouse" -->

```python
import ibis
from increment import Analysis
from increment.dashboard import (
    DashboardConfig,
    dashboard_styles,
    prepare_dashboard,
    render_header,
    render_health,
    render_results,
)

con = ibis.snowflake.connect(...)
analysis = Analysis.from_definitions("checkout_redesign", "definitions/", con)
config = DashboardConfig(
    expected_allocation={"control": 0.5, "treatment": 0.5},
    source_label="Production warehouse",
)
snapshot = prepare_dashboard(analysis, config=config)

dashboard_styles()
render_header(snapshot)
render_health(snapshot)
render_results(snapshot)
```

Put each rendering call in its own notebook cell, including `dashboard_styles()`.

`prepare_dashboard` is the headline computation boundary: it runs the assigned-population
allocation check and the full headline family once. Renamed arms, unequal
`expected_allocation`, a different declared primary metric, and experiments with no declared
breakouts are ordinary supported cases — there is no implicit 50/50 default anywhere in the
package. The primary headline keeps its interval beside the point estimate:
inconclusive results are muted amber, while statistically significant results
are green or red according to the metric's declared preferred direction.
Unavailable values render as `—` with their reason rather than a zero-filled
result, and a `stat_sig=False` guardrail is never relabeled "safe". Baseline
arm values and per-metric sample sizes are not part of this first readout: the
current public lift-result model does not carry them.

See the [dashboard guide](docs/guides/dashboard.md) and the bundled
[`examples/ab_testing_dashboard.py`](examples/ab_testing_dashboard.py) for the full
section-by-section recipe, optional time/segment exploration, and the native CSV download.

## Documentation and examples

| Start here | Covers |
|---|---|
| [Warehouse analysis](examples/analysis_from_a_warehouse.py) | End-to-end definitions, queries, diagnostics, and readouts |
| [Data model](docs/guides/data-model.md) | Fact sources, exposures, metrics, and experiments |
| [Quickstart](docs/guides/quickstart.md) | Dataframe summary and panel workflows |
| [Observational comparisons](docs/guides/observational.md) | AIPW, DML, IPTW, overlap, and estimands |
| [Encouragement designs](docs/guides/encouragement.md) | ITT, compliance, and LATE |
| [CUPED](docs/guides/cuped.md) | Pre-period variance reduction |
| [Switchback contrasts](docs/guides/switchback.md) | Fixed-horizon within-unit order contrasts |
| [Metric types](docs/guides/metric-types.md) | Metric semantics and YAML forms |
| [Power analysis](docs/guides/power-analysis.md) | Sample size, power, and MDE |
| [A/B testing dashboard](docs/guides/dashboard.md) | Reusable HTML dashboard over one bound experiment |
| [API reference](docs/api.md) | Complete public Python API |

The [`examples/`](examples/) directory contains runnable marimo notebooks for warehouse and
dataframe analysis, observational and encouragement designs, CUPED, power, breakouts, and
HTE:

```bash
uv run --extra demo --extra tables marimo edit examples/analysis_from_a_warehouse.py
```

```text
YAML definitions ──> semantics ──> query (Ibis) ──> warehouse SQL
                                                        │
                                                        ▼
                                          canonical unit-day artifact
                                                        │
Dataframe ──────────────────────────────────────────────┤
                                                        ▼
                                             arm moments / switchback contrasts
                                                        │
                                                        └─> estimation ──> typed results
```

- `increment.semantics` validates version-controlled definitions.
- `increment.query` constructs backend-independent Ibis expressions.
- `increment.estimation` consumes additive moments or explicit per-unit dataframes.
- `increment.readouts` coordinates methods, diagnostics, breakouts, and trends.
- `increment.power` performs design-time calculations without warehouse dependencies.
- `increment.tables` optionally renders results through CoefTable.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution requirements.
Increment is licensed under the [Apache License 2.0](LICENSE).

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Copyright 2026 Kyle Caron.

Contributions require a signed [Contributor License Agreement](.github/CLA.md); see
[CONTRIBUTING.md](CONTRIBUTING.md).