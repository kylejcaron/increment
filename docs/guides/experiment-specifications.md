# Experiment specifications

An `Experiment` describes the analysis: its design, metrics, decision plan and requested views. A reader supplies the data; `run(experiment, reader)` returns the results. Saving the experiment saves the settings, not the data or a warehouse connection.

Dataframe readers keep the rows present when the builder ran. Warehouse readers read current data on each call. Identical declarations do not establish an identical data snapshot; use [saved data or results](../reference/artifacts-and-replay.md) when replay must use the same input.

## Building and running

Each module builder returns an experiment and the reader that reads its
source:

```python
import increment
import pandas as pd

units = pd.DataFrame(
    {
        "user_id": range(40),
        "variant": ["control"] * 20 + ["treatment"] * 20,
        "revenue": [float(1 + index % 7) for index in range(40)],
    }
)

experiment, reader = increment.from_frame(
    units,
    shape="summary",
    unit="user_id",
    group="variant",
    control="control",
    metrics={"revenue": "mean"},
)
saved = experiment.to_json()
restored = increment.Experiment.from_json(saved)
result = increment.run(restored, reader)
print(increment.render(result, "brief"))
```

The declaration is immutable. Methods such as `with_views` return a new experiment. Portable declarations retain their canonical identity through JSON. A raw learner factory is usable in memory but cannot be exported or given a portable hash; recording an `ImplementationRef` does not make that factory portable.

## Add result views

A builder declares the overall view. Add daily lift and per-arm daily values with `with_views`; adding an already-declared view refuses with `kernel.request.invalid`. Daily views need each unit’s exposure date. Here every unit was exposed on the panel’s only day:

```python
import pandas as pd

import increment

panel = pd.DataFrame(
    {
        "user_id": range(40),
        "variant": ["control"] * 20 + ["treatment"] * 20,
        "event_day": [pd.Timestamp("2025-01-01")] * 40,
        "revenue": [float(1 + index % 7) for index in range(40)],
    }
)
experiment, reader = increment.from_frame(
    panel,
    shape="panel",
    unit="user_id",
    group="variant",
    date="event_day",
    exposure_date="event_day",
    control="control",
    metrics={"revenue": "mean"},
)
total = increment.View.overall()
daily = increment.View.daily()
daily_values = increment.View.daily(values=True)
experiment = experiment.with_views(daily, daily_values)  # The builder declared `total`.
combined = increment.run(experiment, reader)  # One capture, all three views.
only_daily = increment.run(experiment, reader, views=(daily,))
```

`values=True` requests descriptive per-arm values on daily or as-of views. Overall, breakout and switchback views do not support it. Select declared views with `run(..., views=(...))`; execution follows declaration order. Empty, duplicate or undeclared selections refuse before reading data.

Selected views share one capture, but combining them cannot supply missing time, breakout or cluster information. Whole-window overall views are evaluated against the decision plan; daily, as-of, breakout, compliance-only and exploratory cells do not decide the overall experiment. Read each collection with `result.readout(view)`.

## Save and reopen results

Pass `results_store=` to save a run and `decision_log=` to record its final
family decisions; either may be used alone, and `run_id=` names the run (an ID
is generated when it is omitted, and an empty ID refuses with
`store.run.identity_incomplete` before any data is read). Both are checked
against the experiment's registration before any data is read, and a refused
run writes nothing: methods without a portable implementation reference refuse
with `wire.procedure.runtime_method`, and a registration that differs from the
analysis plan or readout options that change registered decisions or omit
registered metrics with `store.decision.reference_invalid`.
A switchback contrast run saves with `results_store=` alone: its cells,
metadata and method identities are available to `handle.envelope()`,
`store.query` and `store.history`, but it keeps no decision plan, so
`decision_log=` refuses with `wire.procedure.contrast_context` and
`RunResult.from_handle` refuses it with `store.decision.reference_invalid`.
Saving a live raw-set confidence set refuses with
`wire.result.raw_outcomes_unavailable`; it remains available in memory. A saved
run reopens with its registration, which must match the one it was saved
under:

```python
from tempfile import TemporaryDirectory

import increment
from increment.store import ParquetResultsStore

with TemporaryDirectory() as directory:
    store = ParquetResultsStore(directory)
    saved = increment.run(experiment, reader, results_store=store)
    handle = store.open_run(saved.run.experiment_id, saved.run.analysis_spec_hash, saved.run.run_id)
    reopened = increment.RunResult.from_handle(handle, registration=experiment.registration)
assert reopened.cells == saved.cells
assert reopened.decisions == saved.decisions
```

A reopened result recomputes its decisions from the saved cells; live readouts
are not saved, so its `live_rows` holds `None` for every cell.

## Save selected segments with the complete decision

`View.breakout(..., metrics=(metric, ...))` selects registered metric definitions
for that breakout only; an empty tuple keeps all declared metrics. Add the
breakout with `with_views` to retain the overall view, including every primary
and guardrail procedure. The complete decision and selected segments share one
run, source snapshot and evidence capture:

```python
from tempfile import TemporaryDirectory

import increment
import pandas as pd
from increment.evidence import BreakoutIdentity
from increment.store import ParquetResultsStore
from increment.workflows import metrics

units = pd.DataFrame(
    {
        "unit": range(80),
        "arm": ["control"] * 40 + ["treatment"] * 40,
        "country": ["US", "GB"] * 40,
        "revenue": [10.0 + index % 7 for index in range(80)],
        "latency": [100.0 + index % 5 for index in range(80)],
    }
)
experiment, reader = increment.from_frame(
    units,
    shape="summary",
    unit="unit",
    group="arm",
    control="control",
    metrics=[
        increment.MetricSpec(name="revenue", type="mean", preferred_direction="increase"),
        increment.MetricSpec(name="latency", type="mean", preferred_direction="decrease"),
    ],
    plan=increment.AnalysisPlan(
        primary="revenue",
        guardrails=[increment.ExperimentMetric(metric="latency", margin_abs=5.0)],
    ),
    breakouts=["country"],
)
revenue = next(metric for metric in metrics(experiment, reader) if metric.name == "revenue")
segments = increment.View.breakout((BreakoutIdentity("country", source=None),), metrics=(revenue,))
experiment = experiment.with_views(segments)
with reader, TemporaryDirectory() as directory:
    store = ParquetResultsStore(directory)
    result = increment.run(experiment, reader, results_store=store)
    handle = store.open_run(
        result.run.experiment_id, result.run.analysis_spec_hash, result.run.run_id
    )
    restored = increment.RunResult.from_handle(handle, registration=experiment.registration)
    assert restored.decisions == result.decisions
```

A saved or logged selective breakout must accompany the complete overall
decision in the same invocation. A global incomplete `run(metrics=...)`
selection or a selective breakout without that overall view refuses with
`store.decision.reference_invalid` before reading or writing. Selected metrics
must match their registered definitions; this is not a route for adding
`exploratory_metrics`.

Breakout evidence remains monitoring evidence: its cells retain
`decision.not_evaluated.monitoring_view`, and even a rejecting segment cannot
change the overall program verdict. Saved family events retain their segment
references and replay that evidence. `ParquetResultsStore` and
`WarehouseResultsStore` preserve the same cells, unavailable reasons and source
identity; `RunResult.from_handle` restores them for decisions, frames and
reports without a reader. Selecting a metric does not bypass its existing
breakout compatibility checks.

## Check a saved declaration before reading data

`runnable` checks known compatibility without opening a source. Inspect all three fields: unknown requirements do not cancel known refusals, and no static refusal is not a guarantee that the data will be usable. Denominators and sample floors still need execution.

```python
import increment

status = increment.runnable(experiment, "revenue")["randomized.unadjusted"]
print("cannot run:", [error.code for error in status.static_refusals])
print("not established offline:", status.unknown)
print("checks deferred until execution:", status.deferred_checks)
```

Calling `from_definitions(path)` without a connection reads a previously generated catalog. It can support offline inspection, not execution. Rebind with a warehouse connection to run, or after changing the parsed definitions. Comment and whitespace edits do not invalidate the catalog.

## Reader lifetime and replay details

Use a reader as a context manager when it owns TEMP tables or an artifact snapshot. Exiting `with reader:` releases those resources without closing a warehouse connection you supplied. A closed reader cannot reopen; build another. Closing an artifact reader does not delete its published data.

- [Identity, view selection and reader matching](../reference/artifacts-and-replay.md#experiment-identity).
- [Builder options and offline catalogs](../reference/artifacts-and-replay.md#builder-options-and-offline-catalogs).
- [Sequential capture and continuation](sequential-inference.md): capture is an execution step, not a saved request option.
- [Saved-format versions and migration](../reference/saved-formats.md).

<span id="frozen-capture-construction-coverage"></span>

[Specification round-trip coverage](../reference/artifacts-and-replay.md#frozen-capture-construction-coverage) distinguishes captured analysis constructions from refusal-only probes.
