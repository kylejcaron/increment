# Experiment specifications

`increment.Experiment` is an immutable description of a registered analysis. It records a design, metrics, source references, and selected scopes. For portable declarations, `to_json()` and `from_json()` preserve its canonical identity. In-memory analyses can also declare learner factories and fold counts: these remain usable during construction and execution. Exporting a raw learner factory refuses rather than replacing the learner. Source references can record a digest, a column-name/type schema, and a versioned `EvidenceCapabilities` declaration; a missing declaration is not evidence that a source supports a request.

Raw learner factories use exact object identity while bound in memory, so
different closures, partials, or callable instances cannot be substituted for
one another. Their registration identity is process-local, not a portable
implementation/configuration identifier. Stable experiment hashes and portable
exports refuse raw factories even when `implementation_refs` records their
provenance. An `ImplementationRef` does not replace or reconstruct a raw
factory; experiments containing one remain in-memory-only.

`run(experiment, resolver)` returns one `RunResult` for all declared supported
scopes, sharing one physical evidence capture. `views=(scope, ...)` selects
declared scopes by their hashes; execution follows declaration order, not
selection order. Empty, duplicate or undeclared selections refuse before reads.
`result.readout(scope)` returns the typed collection each executed scope
produced; see [reading results](reading-results.md).

Declare the result views on the experiment with `with_views`; it returns a new
experiment and preserves the supplied view order:

```python
import pandas as pd
from increment import Scope, from_frame

frame = pd.DataFrame(
    {
        "unit": ["u1", "u2", "u3", "u4"],
        "arm": ["control", "treatment", "control", "treatment"],
        "day": ["2025-01-01"] * 4,
        "value": [1.0, 2.0, 3.0, 4.0],
    }
)
experiment, resolver = from_frame(
    frame,
    shape="panel",
    unit="unit",
    group="arm",
    date="day",
    control="control",
    metrics={"value": "mean"},
    observation_end="2025-01-02",
)
daily_lift = Scope(view="daily", grain="daily")
daily_values = Scope(view="daily", grain="daily", values=True)
experiment = experiment.with_views(daily_lift, daily_values)
```

`values=True` requests per-arm descriptive values for daily or as-of views.
Without it, those views request lift. Values and lift at the same time
coordinates are separate scopes; the default lift scope keeps its existing
canonical identity. Run, breakout and switchback scopes do not support values.
If an in-memory learner configuration cannot be serialized, `RunResult`
remains usable in memory, while storing or serializing it refuses with
`wire.procedure.runtime_method`.

`Experiment.compile()` returns one run-wide `EvidenceScope` inventory and one `AnalysisRequest` for each selected scope. Both contain the declared metrics, with any switchback scope-specific selection applied. Each request retains its scope, registration hash, and source-reference name.

`runnable(experiment, metric)` is an offline preflight. It does not open a source. If a source has no recorded capabilities, the returned estimator entries name those capabilities as `unknown`; known data-dependent conditions such as usable denominators and sample floors remain deferred until execution. Its `static_refusals` contain only known incompatibilities.

An unknown source declaration does not erase refusals already established by
another source. Alternative evidence routes include their count and named
covariate requirements. For switchback experiments, `switchback.contrast`
identifies the supported contrast procedure; arm estimators refuse with
`readout.contrast.override`.

Builder specifications retain the compiled plan, metric configurations,
cluster and trigger declarations, and the bound operations' accepted arguments.
Readout operations are selected through the declared scopes: daily values and
daily lift are separate scopes even when they share time coordinates, and have
different compatibility requirements. Preflight is conservative and sound rather than complete: it reports
known static refusals using the execution refusal code, but does not estimate
data-dependent conditions or guarantee that an unrefused request will execute.


The analysis specification identity includes the effective readout operation for
each scope and saved request options, including priors and estimands. Omitting
an operation and explicitly selecting its default have the same specification
identity. These declarations also contribute to the execution identity; they
do not change evidence identities.

Preflight restores saved request options and checks known metric selection and
configuration overrides, including an explicit `prior=None` that clears a
declared prior. It cannot run every execution validator offline, so some
configuration checks may first appear during execution. Daily and as-of values
use descriptive validation, without restrictions that apply only to lift or
causal inference. A registered as-of lift projects the current finalized
checkpoint, so it requires sequential history rather than separately retained
as-of moments.

Estimator selection retains declared method settings such as conversion
inference and CUPED covariates where available; multicovariate CUPED requires
its covariate configuration. Switchback scopes require a switchback experiment.
Switchback requests reject arm-only arguments and role or prior overrides with
the same checks as execution.
`capture` is an execution step, not a saved request option. To capture or adopt
a sequential checkpoint, call
`increment.workflows.capture_sequential(experiment, resolver, ...)` with the
pair returned by a module builder, then run the experiment. Pass an `as_of`
horizon when required by the source. `run(experiment, resolver,
capture=SequentialCaptureRequest(...))` instead takes the capture inside the
run's own shared evidence capture, before any view reads.
Specifications containing `capture` refuse with
`kernel.request.capture_not_supported`. Population is declared on `Scope` via
`Scope(population=...)`; a request option named `population` refuses with
`kernel.request.population_scope_only`, as does a `run(population=...)` that
differs from a selected scope's population.

`Resolver` binds the named sources and verifies their recorded declarations.
`BoundResolver` adapts a custom `EvidenceProvider` for verification only; it
does not execute experiments and refuses with
`kernel.run.provider_execution_unavailable`. Use the bound resolver returned by
a module builder to execute an experiment. The top-level
`run(experiment, resolver)` verifies the binding before calling the resolver's
experiment execution adapter.

The specification format describes declarations; it does not itself open a warehouse or infer a source snapshot. Reproducible execution still requires the resolver and execution context to establish the actual snapshot identity.

## Building and running

Each module builder returns an experiment and the resolver that reads its
source:

```python
from increment import Experiment, from_frame, render, run
import pandas as pd

units = pd.DataFrame(
    {
        "user_id": range(40),
        "variant": ["control"] * 20 + ["treatment"] * 20,
        "revenue": [float(1 + index % 7) for index in range(40)],
    }
)

experiment, resolver = from_frame(
    units,
    shape="summary",
    unit="user_id",
    group="variant",
    control="control",
    metrics={"revenue": "mean"},
)
saved = experiment.to_json()
restored = Experiment.from_json(saved)
result = run(restored, resolver)
print(render(result, "brief"))
```

For a registered sequential definition, capture through the builder's pair
before running. Supply an `as_of` horizon when required by the source:

<!-- skip: next "requires a configured warehouse and a registered sequential definition" -->
```python
from increment import from_definitions, run
from increment.workflows import capture_sequential

experiment, resolver = from_definitions(
    "definitions.yml", con, experiment_name="sequential_experiment"
)
capture_sequential(experiment, resolver, finalized=True, as_of=as_of)
result = run(experiment, resolver)
```

`from_frame` requires `shape="summary"` or `shape="panel"`. Each shape has
its own typed options. A summary accepts `intervention_grain="unit"` or
`"cluster"`; a panel requires `date` and accepts `observation_end`. Both accept
`unit`, `group`, `metrics`, `control` or `design`, `plan`, `uptake`,
`exposure_date`, `breakouts`, `cluster`, and `on_unassigned`.
`control` and `design` are alternative ways to declare the comparison.

The definitions call is path-first:
`from_definitions(definitions_path, con=None, *, experiment_name=None,
backend=None, store="auto", on_mixed_assignment="error",
source_snapshot_evidence=None)`. Omit `experiment_name` only when the definitions
or offline catalog contain exactly one experiment. Binding with a connection
records a versioned catalog file; a no-connection call reads that catalog and
refuses when it is missing or stale. An offline specification can be inspected
with `runnable`, but needs a connection-bound source to execute.

The other source builders expose only the options their data support:

<!-- skip: next "requires a published artifact, exported moments, and a switchback frame" -->
```python
from increment import from_unit_day_artifact, from_moments, from_switchback_frame

artifact_experiment, artifact_resolver = from_unit_day_artifact(
    artifact_store,
    artifact_ref,
    expected_context=artifact_context,
    verification="lazy_digest",
)
moments_experiment, moments_resolver = from_moments(
    moment_rows,
    metrics={"revenue": "mean"},
    control="control",
    experiment_id="checkout_redesign",
)
switchback_experiment, switchback_resolver = from_switchback_frame(
    switchback_rows,
    unit="user_id",
    cycle="cycle",
    period="period",
    step="step",
    group="variant",
    metrics={"revenue": "mean"},
    identification=randomized_design,
    assignment=switchback_assignment,
    contrast_references=contrast_references,
)
```

Every builder returns an experiment declaration and a resolver for reading its
source. Saving and loading the declaration preserves
its hash; it does not serialize the data or a warehouse connection.

Use the resolver as a context manager when it owns temporary warehouse tables or
an opened artifact snapshot. Exiting the block releases those resources; it
does not close the warehouse connection you supplied:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import ibis
import pyarrow as pa

from increment import from_definitions, run

rows = [
    {
        "user_id": f"u{index}",
        "group_id": "control" if index < 4 else "treatment",
        "experiment_id": "demo",
        "ts": f"2025-01-01 12:{event:02}:00",
        "event": kind,
        "revenue": float(index + 1) if kind == "purchase" else None,
    }
    for index in range(8)
    for event, kind in ((0, "page_view"), (1, "purchase"))
]
con = ibis.duckdb.connect()
con.create_table("demo_events", pa.Table.from_pylist(rows))

with TemporaryDirectory() as definitions:
    Path(definitions, "demo.yaml").write_text(
        """\
dialect: duckdb
fact_sources:
  - name: demo_events
    sql: SELECT * FROM demo_events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - {name: page_view, column: null}
      - {name: purchase, column: revenue}
exposures:
  - {name: enrolled, fact: page_view}
metrics:
  - {name: revenue, type: mean, entity: user_id, fact: purchase, aggregation: sum, window_days: 1}
experiments:
  - name: demo
    exposure: enrolled
    unit: user_id
    start: 2025-01-01
    end: 2025-01-01
    observation_end: 2025-01-02
    control_group: control
    plan: {primary: revenue}
""",
        encoding="utf-8",
    )
    experiment, resolver = from_definitions(
        definitions, con, experiment_name="demo", store="always"
    )
    with resolver:
        result = run(experiment, resolver)
assert result
assert con.raw_sql("SELECT 1").fetchone()[0] == 1
con.disconnect()
```

An offline definitions resolver can inspect a saved specification, but it has
no warehouse connection and cannot run it. Connect to the warehouse and build
the resolver again when you are ready to run.

The bound built-in resolver executes selected run, daily, as-of, breakout,
or switchback scopes using each scope's declared metric selection and
population. It validates every selected operation before opening the shared
capture. Existing source limitations still apply: combining scopes does not
create temporal, breakout, raw-outcome or clustering evidence.
Whole-window `run` scopes, including the switchback contrast view, are evaluated
against the registered decision plan once per result. Daily, as-of, breakout,
compliance-only and exploratory-metric cells retain their own scope/role and
receive `not_evaluated` verdicts. Cells, effects and any live confidence-set rows
stay aligned in one flat result with one run ID. Changed designs, decision
plans, metric definitions or unsupported scope coordinates refuse with
`kernel.run.execution_unavailable`, before reading evidence.
Source names, URIs, recorded digests, and capabilities must identify the
bound source.
`run` accepts readout options as keywords:
`decision_method`, `sensitivity_methods`, `prior`, `metrics`, `estimands`,
`value_scale`, `exploratory_metrics`, `population` and `coherence`. Only the
options you pass apply, so `prior=None` removes the registered prior while an
omitted `prior` keeps it. A `population` must match every selected scope;
declare a different population with `Scope(population=...)`.
`result.readout(scope)` returns the `LiftEstimates`, `ContrastResults` or
other typed collection one scope produced.

For example, a unit-panel source that supports daily data can declare whole-window
and daily views together:

```python
from dataclasses import replace
import pandas as pd

from increment import Scope, from_frame, run

panel = pd.DataFrame(
    {
        "user_id": range(40),
        "variant": ["control"] * 20 + ["treatment"] * 20,
        "event_day": [pd.Timestamp("2025-01-01")] * 40,
        "revenue": [float(1 + index % 7) for index in range(40)],
    }
)
exp, resolver = from_frame(
    panel,
    shape="panel",
    unit="user_id",
    group="variant",
    date="event_day",
    control="control",
    metrics={"revenue": "mean"},
)
total = Scope(view="run", grain="total")
daily = Scope(view="daily", grain="daily")
exp = replace(exp, scopes=(total, daily))
combined = run(exp, resolver)  # One capture, both scopes.
only_daily = run(exp, resolver, views=(daily,))
```

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

from increment import RunResult
from increment.store import ParquetResultsStore

with TemporaryDirectory() as directory:
    store = ParquetResultsStore(directory)
    saved = run(exp, resolver, results_store=store)
    handle = store.open_run(saved.run.experiment_id, saved.run.analysis_spec_hash, saved.run.run_id)
    reopened = RunResult.from_handle(handle, registration=exp.registration)
assert reopened.cells == saved.cells
assert reopened.decisions == saved.decisions
```

A reopened result recomputes its decisions from the saved cells; live readouts
are not saved, so its `live_rows` holds `None` for every cell.

Definitions-backed `run(experiment, resolver, exploratory_metrics=...)` also shares one
physical capture across declared and exploratory families. Each family retains
the horizon of its original standalone declaration, including unselected
declared metrics needed to determine a running calendar-day horizon. Such
capture-only dependencies do not become authorized readout metrics. Request-
and evidence-bound `source_snapshot_id` values remain distinct when appropriate;
the shared physical capture is recorded separately in `metadata.captures`.
No source snapshot hash or wire-format version changes.

Connected definitions builders record dtypes lazily from the selected input
relations during the first execution, then refresh their offline catalog.
Construction performs no additional schema query. The definitions builder takes
the path first and the experiment name as the `experiment_name=` keyword.
Catalog refreshes retain the definitions digest from the original bind. Changing
those files before execution cannot republish the old experiment as current;
the refresh refuses with `kernel.definitions.catalog_stale`.
Catalog updates lock the complete read-merge-replace operation, so concurrent
bindings or schema refreshes retain each experiment's entry.

The catalog records each loader's `declaration_identity()` digest alongside its
schema and capabilities. A bound resolver recomputes that declaration digest
at `verify()` and refuses a changed or unavailable recorded digest with
`kernel.resolver.digest_mismatch`. It does not hash live rows: snapshot and
content digests belong to the runtime evidence, not the saved specification.

Use `runnable` when a caller needs the static refusal codes, and inspect all
three `Runnable` fields rather than treating unknown requirements as supported:

```python
from increment import runnable

status = runnable(restored, "revenue")["randomized.unadjusted"]
if status.unknown:
    print("not established offline:", status.unknown)
elif status.static_refusals:
    print("cannot run:", [error.code for error in status.static_refusals])
else:
    print("checks still deferred:", status.deferred_checks)
```

`ResultCell` preserves one cell's identity and outcome in a stored run;
`Decisions` records the verdicts computed from those cells. The top-level
`run(experiment, resolver)` returns a `RunResult` containing cells and decisions;
`result.readout(scope)` returns each view's estimate collection.

## Frozen-capture construction coverage

The specification round-trip tests enumerate the frozen capture's own
scenario/constructor inventory, including its as-of variants. In-memory definitions
are written through the public definitions constructor; explicitly registered
segment families are retained in the serialized inference declaration rather
than replaced with an automatic, unsegmented registration.

An absent constructor is not a failed round-trip: it is not declared
for that captured scenario. A constructor refusal also produces no `Experiment`.
Both outcomes are recorded with their reason or refusal code. Among the
analysis-construction scenarios, `edge-nonfinite-ratio-denominator` has no
successful captured constructor: its unit-summary input refuses with
`source.frame.non_finite`, and the other constructor routes are absent.

The following frozen scenarios are isolated refusal probes, not analysis
constructions. They supply invalid arm statistics, compatibility requests or
contrast objects directly to the owning statistical function; there is no
builder-produced specification to serialize:

- `allocation.alpha.underflow`
- `arm.adjustment.cluster_cuped`, `arm.adjustment.cluster_prior`,
  `arm.adjustment.sequential_prior`
- `arm.baseline.cluster_size_required`, `arm.baseline.compliance_undeclared`,
  `arm.baseline.iid_cluster_knobs`
- `arm.inference.cluster`
- `arm.metric.quantile_cluster`, `arm.metric.quantile_cuped`,
  `arm.metric.quantile_sequential`
- `compatibility.support_unsupported`
- `contrast.decision`, `contrast.inference`, `contrast.metric`
- `estimation.engine.arm.duplicate_rows`, `estimation.engine.arm.invalid_count`
- `estimation.engine.carries_format_moments`,
  `estimation.engine.carries_unsupported_weighted`
- `estimation.engine.cluster.denominator`, `estimation.engine.control_group_found`
- `estimation.engine.cuped_linear_cluster`,
  `estimation.engine.cuped_linear_encouragement`,
  `estimation.engine.cuped_linear_observational`,
  `estimation.engine.cuped_linear_sequential`,
  `estimation.engine.cuped_linear_unit_evidence`,
  `estimation.engine.cuped_linear_winsorization`
- `estimation.engine.group_summary_missing`, `estimation.engine.group_summary_row`
- `estimation.engine.method_name_observational`,
  `estimation.engine.method_names_unique`, `estimation.engine.metric_found_group`
- `estimation.engine.ratio.cuped`, `estimation.engine.ratio.negative_variance`
- `estimation.engine.winsorization_metadata_differs`,
  `estimation.engine.winsorization_metadata_present`
- `estimation.engine.x_role_declared`

The remaining non-construction probes exercise planning-only admission
coordinates, rather than experiment declarations:

| Frozen scenario | Why no builder specification exists |
| --- | --- |
| `compatibility.denominator` | Only the private conservative-ratio helper accepts an arbitrary denominator; public plans derive it from validated inputs. |
| `compatibility.multiplier` | Only that private helper accepts an arbitrary multiplier; public plans derive it from validated inputs. |
| `arm.adjustment.sequential_cuped` | The public power planner rejects always-valid inference before this admission point, and admits CUPED under asymptotic inference. |
| `arm.baseline.absorption_undeclared` | The public sample-size planner derives factor absorption from nonzero baseline ICC before admission. |
| `arm.baseline.triggering_undeclared` | It derives the triggered population from the baseline trigger rate before admission. |
| `arm.baseline.cuped_undeclared` | It derives the CUPED method from the baseline correlation before admission. |
| `arm.metric.unsupported` | Validated planning-procedure metric capabilities exclude the probe's unsupported metric shape before admission. |
