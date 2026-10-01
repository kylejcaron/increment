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
    print(f"{r.metric} / {r.group_id}: lift={r.lift.value:.2%}")
```

Monitoring daily? [Register a raw sequential model](guides/sequential-inference.md)
before reading outcomes, including predictive priors, a fixed cell roster, and the
joint finalized reveal contract. Pass that registration through `InferenceSpec`
and explicitly capture finalized checkpoints. The public exact `always_valid` route is
limited to raw Bernoulli observations with a proper Beta predictive prior and a
common joint-unit filtration. Scalar Gaussian and paired-Gaussian kernels are
private research diagnostics only and do not produce public evidence or
decisions; that does not affect the public asymptotic route below. Sequential secondary families use current or frozen e-values and
reinvert selected intervals at `min(q * R / m, nominal alpha)` on that same stopped state.
Name primary and guardrail roles explicitly when constructing the plan.

`InferenceSpec(kind="asymptotic_mean")` is a separate public route: an asymptotic
confidence sequence for mean, ratio, conversion and retention metrics that needs no prior
and holds only under its stated iid, fixed-Bernoulli-assignment and moment
assumptions (see [sequential inference](guides/sequential-inference.md)). It is
distinct from the exact Bernoulli `always_valid` e-process above.

See the [API Reference](api.md) for the full surface.

## Switchback contrasts

For randomized treatment orders within independent units or shared across a
fixed roster, use the fixed-horizon [switchback guide](guides/switchback.md).
Switchback applies washout and declared carryover discards, then estimates an
additive retained-window difference. Independent unit-cycle orders use the qualified
unit-t approximation by default (equivalent to passing `UnitCycleTApproximation()`):
it requires at least two independent units and reports
`switchback_unit_t_approximation` / `unit_t_approximation` with `dof=n_units-1`.
A separately supplied valid prospective `UnitCycleVarianceEnvelope` is a stronger
route and can support one unit; a pilot or sample variance is not such an envelope.
Shared schedules keep block-t semantics: `switchback_block_t` with a `block_t`
reference and at least two independent blocks; a one-unit roster is supported.
Units, cycles and blocks are distinct independence grains. Switchback does
not support sequential inference, CUPED, multiplicity families, or arm-method
overrides.

## Evidence and method roles

Parallel arm analyses compile one `decision_method` and optional
`sensitivity_methods` per metric. Declare these on a plan binding or
`MetricSpec`, or override them call-wide with `run(decision_method=...,
sensitivity_methods=...)`; sensitivity rows are reporting-only and do not
create decision evidence. The former method-list keyword is removed.

Definitions-backed analyses may publish a canonical context-format-2 unit-day
artifact once and later adopt it from a caller-pinned `UnitDayArtifactRef`.
Session `materialize()` creates only a local TEMP realization. See the [data
model guide](guides/data-model.md#canonical-unit-day-artifacts) for the
manifest, extensions, trust boundary, and immutable snapshot contract.

If artifact publication raises after its manifest is written,
`WarehouseArtifactStore` invalidates the generation through `drop_generation`,
so a fresh store lists nothing for it and reads are refused with
`artifact.generation.dropped`. If the invalidation itself fails, the exception
carries a note with the `artifact_id` and `generation_id` to pass to
`drop_generation` from a fresh process. If the manifest insert was submitted but
its outcome is unknown, the generation is tombstoned before its relations are
erased, so a row that commits late stays hidden. Erasure attempts every relation
even if one drop fails. If the tombstone cannot be written, or a relation cannot be
dropped afterwards, the relations are kept and the note (or, when the publication
ends normally without a manifest, the context of the
`query.session.warehouse_artifact.publication_state_unknown` refusal) lists them
by qualified name. A failed drop before the manifest insert is submitted is reported
the same way: the aborting exception carries the note, and a publication that ends
normally without a manifest raises
`query.session.warehouse_artifact.publication_cleanup_incomplete`; drop the listed
relations by name. When the outcome is unknown, call
`store.abandon_generation(artifact_id, generation_id)` from a fresh process first
(durable, safe to repeat), then drop the listed relations by name. The
[data
model guide](guides/data-model.md#canonical-unit-day-artifacts) describes recovery.
