# When a request is refused

A refusal explains why a request cannot run. Read its code, check the named
input or assumption, then correct the cause. Do not change the experiment you
claim to have run just to make an error disappear.

## Start with the code and context

Catch `CodedError` and branch on its stable `code`; use `context` to inspect
the relevant fields. The human-readable message can change, so do not parse
it. See the [API error reference](../api.md#errors) for typed
refusals and examples.

```python
from increment.errors import CodedError


def refusal_record(exc: CodedError) -> tuple[str, dict[str, object]]:
    return exc.code, dict(exc.context)
```

## Find the next step by code family

The first code segment helps you find a family; it is not a separate public
enum. Check the specific refusal's message, context, and linked contract:
similar codes can mean different things.

| What failed? | Start here |
|---|---|
| Data or definitions | [Warehouse and dataframe inputs](data-model.md) |
| Method or required data | [Choose a method](choose-a-method.md), [supported inputs](../reference/capabilities-by-entry-point.md) |
| A warehouse query | [Backend translation](data-model.md#when-ibis-cannot-translate-an-operation) |
| Overlapping experiments | [Check the design and four input feeds](overlapping-experiments.md#first-establish-who-could-have-received-either-treatment) |
| An estimate or comparison family | [Reading results](reading-results.md), [multiplicity](multiplicity.md) |
| A saved run | [Saved formats](../reference/saved-formats.md) |
| A display or export | [Dashboard requirements](dashboard.md) |

Search the full code in the [error reference](../reference/error-codes.md).
The recovery cases below explain choices that need more than a corrected argument.

A missing treatment arm can prevent a contrast without preventing arm counts
or a separate allocation check. An unavailable or refused estimate is **not a
zero effect**. Likewise, a refused triggered-population result does not make an
assigned-population estimate a substitute for it.

## Recover from a missing sequential allocation

Sequential registration uses the assignment probabilities. If a frame analysis
omitted them, declare the **actual** allocation; do not invent a convenient split.
This example catches the refusal, then reruns with its real 1:1 assignment.
It uses the asymptotic mean method: four observations illustrate recovery,
not reliable inference.

```python
import pandas as pd

import increment
from increment.errors import CapabilityError

frame = pd.DataFrame(
    [
        {"unit": "c1", "arm": "control", "outcome": 2.0, "exposure": 1},
        {"unit": "c2", "arm": "control", "outcome": 3.0, "exposure": 2},
        {"unit": "t1", "arm": "treatment", "outcome": 4.0, "exposure": 3},
        {"unit": "t2", "arm": "treatment", "outcome": 5.0, "exposure": 4},
    ]
)
plan = increment.AnalysisPlan(
    primary="outcome", inference=increment.InferenceSpec(kind="asymptotic_mean")
)
metrics = [increment.MetricSpec(name="outcome", type="mean")]

try:
    increment.from_frame(
        frame,
        shape="summary",
        unit="unit",
        group="arm",
        design=increment.Randomized(control_group="control"),
        metrics=metrics,
        experiment_id="example",
        plan=plan,
        exposure_date="exposure",
    )
except CapabilityError as exc:
    assert exc.code == "sequential.source.invalid"
    assert exc.context["route"] == "design.allocation"
    assert exc.context["mechanism"] == "randomized"
    assert exc.context["required"] == "actual assignment allocation"
else:
    raise AssertionError("the missing allocation should be refused")

# This example's units were assigned 1:1; use the real allocation.
experiment, reader = increment.from_frame(
    frame,
    shape="summary",
    unit="unit",
    group="arm",
    design=increment.Randomized(
        control_group="control", allocation={"control": 0.5, "treatment": 0.5}
    ),
    metrics=metrics,
    experiment_id="example",
    plan=plan,
    exposure_date="exposure",
)
results = increment.run(experiment, reader).readout()
assert len(results) == 1
assert results[0].metric == "outcome"
```

## Specific recovery cases

### The source cannot supply the requested data

For `evidence.*`, read `requested`, `offered` and `route` in the context.
The code identifies missing grouping, unit rows, raw outcomes, clusters,
covariates, assignment records or an operation. Use a source that contains
the required data, or an operation it supports; do not fabricate missing data.
Compliance declarations distinguish total from as-of data.

- `evidence.assignment.unavailable`: the source has no assignment evidence.
- `evidence.assignment.selection_unavailable`: the requested experiment,
  version, period or population was not retained.
- `evidence.assignment.identity_unavailable`: replay, whole-ledger exposure or
  allocation history needs row-level assignment records. Supply a ledger through
  `from_definitions` with `experiment.assignment_ledger`, or a `FrameLedger`.
- `evidence.assignment.invalid`: inspect the reported malformed roles,
  duplicate identities, inconsistent cluster membership, invalid periods or
  replay specification.
- `evidence.assignment.replay_mismatch`: replay disagrees with the selected ledger.
- `evidence.joint_grouping_unavailable`: declare all requested grouping
  properties together on one source, or use a source that already retains them.
- `evidence.operation_unavailable`: the source does not retain the requested
  operation.

For `source.*`, `frame.*`, `artifact.*`, `artifact_contract.*`, `definition.*`,
`model.*` and `compatibility.*`, repair the reported column, grain, identity,
assignment or version at its source. Follow [data preparation](data-model.md)
and the [compatibility policy](../api.md#compatibility-policy).

### A warehouse cannot compile or verify a query

**`query.backend.operation_unsupported`** means Ibis cannot translate a query.
`backend` names the dialect; `detail` names the failed operation. If the detail
instead describes converting timezone-aware event times to UTC, the backend
is not one of the four verified ones. Convert times in the source SQL as
[event-time handling](data-model.md#event-times-with-a-time-zone) describes.
Earlier queries may already have read data, but no result is produced.
Use `from_frame`, or add the [missing translation](data-model.md#when-ibis-cannot-translate-an-operation).

**`query.backend.compiler_unavailable`** means a SQL preview requested a
dialect the installed Ibis cannot write. It refuses before reading anything.
Pass an Ibis SQL backend name to `from_definitions(..., backend=...)`, or
install an Ibis release supporting that dialect.

**`query.session.warehouse_artifact.verification_unavailable`** means this
warehouse route cannot verify the stored relation format. Context names the
relation and digest format. Do not read or relabel its hashes; republish from
trusted definitions with warehouse-side integrity enabled.

**`query.cuped.magnitude_range`** means an arm's root-mean-square outcome
component (including a ratio denominator) or continuous covariate is outside
`[1e-50, 1e50]`. Context identifies the metric, arm, component and bound.
Rescale the units, or supply per-unit data with `from_frame(shape="summary")`
or `from_frame(shape="panel")`; local per-unit fitters are unchanged.

### A comparison family has an unusable result

**`family.evidence.incomplete`** names unusable cells in `failed`, `missing`
and `invalid`. Whole-window families refuse for any cell without evidence,
including a missing control arm. Fixed-horizon breakout families instead
leave out cells with too-small or missing arms and keep outcome-based
exclusions as non-rejections; the refusal names other failures.

For breakout or as-of views, inspect cell reasons with
`MultiplicitySpec(correction="none")`, then repair the cause and rerun the
intended family. This diagnostic view is not a replacement for the
[multiple-comparison procedure](multiplicity.md).

For other inference refusals, check the effect, population, method and
comparison scope in [Reading results](reading-results.md). SRM integrity
codes require the actual target allocation and, for always-valid checks,
the assignment law. `sensitivity.*` also requires the row's metric type and
scale; see [omitted-confounder sensitivity](observational.md#how-strong-would-an-omitted-confounder-need-to-be).

### The experiment and reader do not match

- `kernel.run.reader_required`: `received` names the wrong input type.
  Pass the reader returned by a builder. A `BoundResolver` verifies a custom
  provider; it is not a reader.
- `kernel.run.reader_closed`: build a new reader.
- `kernel.run.reader_offline`: build it with a connection.
- `kernel.run.execution_unavailable` with `reader_experiment_id`: the reader
  was built for a different experiment; `field` names the first differing
  declaration. Pass the original pair, or rebuild both from the changed declarations.
- `kernel.provider.digest_mismatch` or `kernel.provider.capability_mismatch`:
  the source no longer matches the recorded declarations. Rebuild the pair.
- `kernel.definitions.catalog_missing`: bind definitions with a connection
  to write the offline catalog.
- `kernel.definitions.catalog_stale`: rebind after changing definitions.

Other `kernel.*` codes name an inconsistent or nonportable description through
`field` and `kind`. Rebuild it from the analysis rather than editing stored
identifiers. Request identity datetimes need a timezone; aware offsets are
normalized to UTC.

### A request option belongs elsewhere

**`kernel.request.capture_not_supported`**: remove sequential capture or
checkpoint adoption from saved request options. Call
`increment.workflows.capture_sequential(experiment, reader, ...)` with the
builder pair, then run; or pass `capture=SequentialCaptureRequest(...)` to `run`.

**`kernel.request.population_scope_only`**: set population on the view, such
as `View.overall(population=...)`, not in request options. A call-wide
`run(population=...)` must not disagree with a selected scope.

**`kernel.scope.family_mismatch`**: a switchback view was requested from an
ordinary arm experiment. Use `from_switchback_frame`, or choose the ordinary
experiment's run, daily, asof or breakout view.

### A saved run or interval cannot be reused

**`results.operation.alpha_above_allocation`**: the reissued interval exceeds
its stored declared or selected family allocation. Request an equal or
smaller alpha, or use an unadjusted row.

For **`results.readout.*`**, request a view the run executed; `scope_hashes`
lists them. A stored or pickled result keeps only its saved cells. Run the
experiment again for its full collections.

For **`store.*`**, keep run IDs unique, publish a complete roster once, and
read with the pinned `RunHandle`. Correct the source run and write a new ID
rather than replacing a committed run.

- `store.run.empty`: select at least one metric before saving. An empty
  result can still be inspected in memory.
- `store.run.manifest_unsupported`: a newer release wrote the run. Compare
  `found` with `supported` in context and upgrade Increment; see [format versions](../reference/saved-formats.md).
- `store.run.schema_unavailable`: `WarehouseResultsStore` could neither
  find nor create its schema; nothing was saved. Create it or grant permission.
- `store.schema_name_too_long`: shorten the schema or catalog name. The
  limit is 63 bytes on PostgreSQL, 255 characters on Snowflake and 1,024 bytes
  on BigQuery. The store named in context refuses at construction, before
  reads or writes; PostgreSQL would otherwise silently truncate names.

**`wire.procedure.contrast_context`**: a switchback run's decisions have no
portable arm decision plan. Save cells alone with `run(..., results_store=...)`,
keep the result in memory, or read `ContrastResults` with `result.readout()`.
Other `wire.*` refusals mean the result cannot be represented in that wire format.

### An optional package is missing

For `install.*`, context gives `feature` and the install command. These errors
are also `ImportError`s.

- `install.warehouse_extra_missing`: install `increment[warehouse]` for Ibis
  and SQLGlot, plus your Ibis backend driver separately.
- `install.dashboard_extra_missing`: install `increment[dashboard]`.
- `install.tables_extra_missing`: install `increment[tables]` for
  `readout_table` or `trend_table`.

Use `python -m pip install --pre "increment[<extra>]"` for the current
prerelease. Dataframe and moments builders need none of these extras.
`package.*` instead means the distribution is incomplete or inconsistent:
reinstall a matching package, rather than changing the experiment.

### A specialized method or extension refuses

For `analysis.*`, `facade.*`, `plan.*`, `power.*` or `arm_planning.*`,
check inputs and supported methods in [capabilities](../reference/capabilities-by-entry-point.md),
[power planning](power-analysis.md) or [method selection](choose-a-method.md).
For `cate.*` and `logged_policy.*`, check identification and support
assumptions in [heterogeneity](heterogeneity-and-rollout.md) or [logged policies](logged-policy.md).
Choose a supported effect to estimate, not a weaker validity check.

For `simulate.*`, check the simulated design and operating characteristic.
Simulation results do not establish a guarantee for a different design.
For `dashboard.*`, `report.*`, `reporting.*` and `tables.*`, check projection
columns and the view's [input requirements](dashboard.md); use the underlying
readout if the display is unavailable.

**`estimation.estimator.adjustment.callback_contract`** means a registered
callback does not accept `AdjustmentInputs`, metric, design and the required
keyword arguments. Migrate to the [current callback contract](../development/adding-an-estimator.md#custom-evidence-providers);
do not register the legacy positional signature to bypass the check.
