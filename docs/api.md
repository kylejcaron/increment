# API Reference

Use `increment` for workflow entry points and caller-authored configuration.
Receive-only result types are in `increment.results`; specialized power solvers
are in `increment.power`.

## Start here

::: increment.from_frame

::: increment.from_definitions

::: increment.run

::: increment.RunResult

::: increment.estimators.MetricTrends

::: increment.Definitions

`increment.__version__` reports the installed package version.

## Pre-1.0 compatibility

Increment is pre-release software. This section covers compatibility between
releases, persisted formats, and migrations—not statistical certification. A
stable interface does not establish that a method's assumptions fit your
experiment. See [Statistical limitations](limitations.md) for method assumptions
and the [capability table](reference/capabilities-by-entry-point.md) for what
runs on each entry point.

### Promised surface

The policy covers:

- documented workflow and configuration imports from `increment`;
- documented result types in `increment.results` and solvers in `increment.power`;
- stable artifact, digest, estimator, and evidence exports from
  `increment.artifacts`, `increment.digest`, `increment.estimators`, and
  `increment.evidence`. Existing family namespaces such as `increment.power`
  remain available;
- `increment.query` promises only `build_panel_for_metric`, `panel_sql`, and
  `summary_sql`. Each operates on an `(experiment, resolver)` pair; SQL
  previews preserve their optional-breakout defaults and use adopted SQL as
  the summary fallback.
- `increment.workflows` promises only `materialize`, `export_moments`,
  `breakout_summaries`, `factor_summaries`, `experiment_definition`, `metrics`,
  `compiled_decision_plan`, `available_metrics`, `unit_counts`, `srm`,
  `trigger_rates`, `allocation_history`, `health_check`,
  `assignment_diagnostic`, `planning_baseline`, `capture_sequential`, and
  `sequential_snapshot`.
  These functions operate on an `(experiment, resolver)` pair; the experiment
  definition inspection returns the semantic source declaration, not the
  kernel execution request. A sequential capture is kept by the resolver's
  source, so the next `run(experiment, resolver)` and `export_moments` with
  that resolver read it.
- `increment.artifacts` promises only `artifact_context` and
  `publish_unit_day_artifact`; each operates on an `(experiment, resolver)`
  pair, and publication retains trusted extension selection and refresh semantics.
- `increment.estimators` conditional-effect operations `estimate_cate`,
  `validate_cate`, `targeting_rule`, and `select_targeting_rule`, and the
  whole-site estimate `sitewide`, operate on an `(experiment, resolver)` pair;
  their remaining parameters are unchanged. The data-only array kernels in
  `increment.estimation.targeting` and `increment.estimation.sitewide`, and the
  `increment.power` solvers, take no source.
- result fields and schemas, including row sets, null reasons, and interval
  reference/guarantee labels;
- stable refusal codes and named context fields;
- each versioned portable or artifact format listed below.

Underscore modules and private research kernels are not public APIs and may
change without notice.

### Alpha semantics

- Pin the exact prerelease and dependency versions (Narwhals, Ibis, Arrow,
  DuckDB, and any pandas/Polars versions) when you need to reproduce a result.
- Breaking changes may occur between 0.x prereleases. Each is listed in the
  release notes (the `Breaking changes` section, from pull requests labelled
  `breaking`) with the affected paths and a migration route.
- 1.0 stability is not promised. Alpha API cutovers remove obsolete paths
  directly and document their replacements without deprecation warnings.
- A bug fix may change a previously incorrect numerical answer. Release notes
  describe the correction; compatibility does not preserve wrong values.

- Release note (breaking API cutover): specialist names are no longer exported
  from the package root. Import artifact, digest, estimator, and evidence names
  from `increment.artifacts`, `increment.digest`, `increment.estimators`, and
  `increment.evidence` respectively. Source-loading estimation facades remain
  available without deprecation warnings; data-only estimation kernels do not
  load sources.
- Release note (breaking API cutover): artifact declarations are no longer
  re-exported by `increment.semantics.models`. Use `increment.artifacts` for
  documented artifact workflows. Schema definitions remain in
  `increment.semantics.artifact`; that defining module is not a promised import
  surface. Artifact formats and class identities are unchanged.
- Release note (bug fix): portable cell summaries follow the row's requested
  effect scale and retain an available relative lift when an additive sidecar
  cannot be projected. A different-scale fallback is labeled explicitly.
- Release note (bug fix): re-issuing a row beyond its registered plan or selected
  family alpha refuses with `results.operation.alpha_above_allocation`; register
  the new alpha in the analysis plan and rerun instead.
- Release note (API addition): `Experiment` and `SourceReference` describe a registered
  analysis and its recorded evidence declarations; `runnable()` reports offline
  unknowns and deferred data checks. Module builders `from_frame`,
  `from_definitions`, `from_unit_day_artifact`, `from_switchback_frame`, and
  `from_moments` return an experiment/resolver pair, and top-level
  `run(experiment, resolver)` returns `RunResult`. See the
  [experiment specification guide](guides/experiment-specifications.md).
- Release note (breaking API cutover): the `Analysis` class, its
  `increment.Analysis` export and the `increment.analysis` module are removed
  without an alias. Build an experiment and resolver with a module builder and
  read results with `run(experiment, resolver)`; source operations moved to
  `increment.workflows`, `increment.query`, `increment.artifacts`,
  `increment.estimators` and `increment.dashboard`. The narrow builder aliases
  `increment.builders.from_unit_summary` and `from_unit_panel`, the
  experiment-name-first `from_definitions(experiment_name, definitions_path,
  con)` form, the `run(analysis, registration, compiled_plan, ...)` overload and
  `increment.readouts.run_scoped` are removed too. Results, refusal codes and
  their context fields are unchanged; refusal messages that named an
  `Analysis` method name the replacement call. See
  [Migrating from `Analysis`](#migrating-from-analysis) for every replacement.
- Release note (breaking API cutover): the module-builder resolver no longer
  has a `capture_sequential` method. Call
  `increment.workflows.capture_sequential(experiment, resolver, ...)`, and read
  the retained checkpoint with `increment.workflows.sequential_snapshot`.
- Release note (breaking API cutover): `increment.estimators.estimate_cate`,
  `validate_cate`, `targeting_rule`, and `select_targeting_rule` take the
  `(experiment, resolver)` pair returned by a builder such as `from_frame` or
  `from_definitions` instead of a source. Their remaining parameters, defaults,
  results, and refusal codes are unchanged; `estimate_cate`'s `control` defaults
  to the declared control arm, and a declared trigger restricts every operation
  to triggered units.
- Release note (breaking API cutover): `increment.workflows.planning_baseline(experiment, resolver, metric)`
  and `increment.estimators.sitewide(experiment, resolver, metric, *, arm=None, alpha=0.05, population="assigned")`
  replace `Analysis.planning_baseline` and `Analysis.sitewide`. Results,
  trigger/CUPED/cluster/quantile/compliance handling, switchback baselines,
  calendar-window site volume, randomized-only restrictions, and refusal codes
  are unchanged. `increment.power` gains no source-reading function.
- Release note (breaking API cutover): `run(experiment, resolver, ...)` no
  longer accepts `analysis_options`. Pass the same readout options as keywords:
  `decision_method`, `sensitivity_methods`, `prior`, `metrics`, `estimands`,
  `value_scale`, `exploratory_metrics`, `population` and `coherence`; only the
  options you pass apply, so `prior=None` still differs from an omitted prior.
  Pass a sequential capture as `capture=SequentialCaptureRequest(...)` and an
  open shared evidence capture as `execution_context=`. The `store` option is
  not accepted. A `population` that differs from a selected scope refuses with
  `kernel.request.population_scope_only`; declare it on `Scope(population=...)`.
- Release note (API addition): `run(experiment, resolver, ...)` saves and logs
  runs. `results_store=`, `decision_log=` and `run_id=` are checked against the
  experiment's registration before any data is read, and a refused run writes
  nothing; an ID is generated when `run_id` is omitted.
  `RunResult.from_handle(handle, registration=...)` decides a saved run again
  and refuses a registration the run was not saved under.
- Release note (API addition): `RunResult` is read directly. `result.readout(scope)`
  returns the full typed collection one executed view produced, including live
  confidence sets; omit `scope` only for a one-view run. It refuses with
  `results.readout.scope_required` when several views ran,
  `results.readout.scope_unknown` for a view the run did not execute, and
  `results.readout.not_retained` on a result restored by `RunResult.from_handle`
  or a pickle. `result.metadata`, `result.envelope` and
  `result.to_frame(backend=..., columns=...)` read the run's metadata, portable
  envelope and one qualified frame of every row; `readout_table` accepts a
  `RunResult`. Every `result.to_frame()` row leads with `scope_view` (string)
  and `scope_values` (nullable boolean), the `view` and `values` of the scope
  that produced it, and `scope_key` (string), which identifies the view a row
  came from: `Scope.hash()` of the row's scope without its day, segment and
  population. Rows of several views stay distinct in live, stored and pickled
  results, and a `columns` selection keeps all three. Write full-row JSON with
  `result.readout(scope).model_dump_json()` and portable cells with
  `result.envelope.to_payload()` or `.to_arrow()`; constructing
  `AnalysisEnvelope(result.cells, result.run.metadata)` is no longer needed.
- Release note (bug fix): `DecisionLog.replay` accepts the logged decisions of a
  Bonferroni breakout family instead of refusing them with
  `store.decision.reference_invalid`. Existing decision logs replay unchanged.
- Release note (bug fix): a breakout read under a registered sequential plan
  refuses with `sequential.route.unsupported` when the declared or selected
  breakouts include a property other than the registered segment, instead of
  returning only the registered segment's rows.
- Release note (bug fix): a one-view `run(..., capture=SequentialCaptureRequest(...))`
  takes the capture inside the run's own shared evidence capture, as a
  several-view run does, so a registered sequential breakout read reports the
  run's `coherence` and `effective_modes` in its metadata.
- Release note: compliance-only as-of and daily views run through `run`
  request no outcome metrics, so their `source_snapshot_id` differs from the
  removed `Analysis.run_asof_lift(estimands=("compliance",))`; the estimates
  are unchanged.
- Compatibility policy is the same across methods; guarantees and evidence
  differ. See [Evidence status](validation.md#evidence-status).

### Migrating from `Analysis`

Every removed `Analysis` member has one replacement. In the table, `experiment,
resolver` is the pair a builder returns, `workflows`, `query`, `artifacts`,
`estimators` and `dashboard` are the `increment.*` namespaces of those names,
and parameters not shown keep their names, defaults and refusals.

| Removed call | Replacement |
|---|---|
| `Analysis(experiment_name, definitions_path, con, ...)`, `Analysis.from_definitions(experiment_name, definitions_path, con, ...)` | `experiment, resolver = from_definitions(definitions_path, con, experiment_name=..., ...)`; the path comes first and the name is a keyword |
| `Analysis.from_unit_summary(frame, ...)`, `increment.builders.from_unit_summary(frame, ...)` | `from_frame(frame, shape="summary", ...)` |
| `Analysis.from_unit_panel(frame, ...)`, `increment.builders.from_unit_panel(frame, ...)` | `from_frame(frame, shape="panel", ...)` |
| `Analysis.from_unit_day_artifact(store, ref, expected_context=..., ...)` | `from_unit_day_artifact(store, ref, expected_context=..., ...)` |
| `Analysis.from_moments(rows, ...)` | `from_moments(rows, ...)` |
| `Analysis.from_switchback_panel(frame, ...)` | `from_switchback_frame(frame, ...)` |
| `with analysis:`, `analysis.close()` | `with resolver:`, `resolver.close()`; neither closes a connection you supplied |
| `analysis.run(**options)` | `run(experiment, resolver, **options).readout()`, the same `LiftEstimates` or `ContrastResults` |
| `analysis.run(population="triggered")` | `scope = Scope(view="run", grain="total", population="triggered")`, then `run(experiment.with_views(scope), resolver).readout(scope)`; declare one run scope per population to read both |
| `analysis.run(store=store)` | `run(experiment, resolver, results_store=store)`; `decision_log=` and `run_id=` are checked before any read |
| `run(analysis, registration, compiled_plan, *, experiment_id=..., analysis_spec_hash=..., execution_identity_hash=..., run_id=..., ...)` | `run(experiment, resolver, run_id=..., results_store=..., decision_log=..., ...)`; the experiment supplies its registration and identities |
| `analysis.run_breakout(...)` | `Scope(view="breakout", grain="total")`; `segment=(BreakoutIdentity(property, source), ...)` selects declared breakouts, and the empty default selects all |
| `analysis.run_daily(dimension=property, ...)` | `Scope(view="daily", grain="daily", values=True, segment=BreakoutIdentity(property, None))` |
| `analysis.run_daily_lift(dimension=property, ...)` | `Scope(view="daily", grain="daily", segment=BreakoutIdentity(property, None))` |
| `analysis.run_asof(completed_windows_only=flag, ...)` | `Scope(view="asof", grain="asof", values=True, completion_policy=flag)` |
| `analysis.run_asof_lift(completed_windows_only=flag, ...)` | `Scope(view="asof", grain="asof", completion_policy=flag)` |
| `population=` on a day-axis or breakout method | `Scope(..., population=...)` |
| `increment.readouts.run_scoped(analysis, ...)` | `run(experiment.with_views(*scopes), resolver)`: every declared view reads one capture of the data; `views=` selects a subset |
| `analysis.materialize()`, `analysis.materialize().run()` | `workflows.materialize(experiment, resolver)`, then `run(experiment, resolver)` |
| `analysis.export(path)` | `workflows.export_moments(experiment, resolver, path)` |
| `analysis.experiment` | `workflows.experiment_definition(experiment, resolver)`, the semantic declaration |
| `analysis.metrics`, `analysis.compiled_decision_plan`, `analysis.available_metrics` | `workflows.metrics(...)`, `workflows.compiled_decision_plan(...)`, `workflows.available_metrics(...)`, each taking `(experiment, resolver)` |
| `analysis.unit_counts()`, `srm(...)`, `trigger_rates()`, `allocation_history(...)`, `health_check(...)`, `assignment_diagnostic()`, `planning_baseline(metric)`, `breakout_summaries(...)`, `factor_summaries()`, `capture_sequential(...)`, `sequential_snapshot(...)` | `workflows.<name>(experiment, resolver, ...)` with the same remaining parameters |
| `analysis.artifact_context` | `artifacts.artifact_context(experiment, resolver)` |
| `analysis.publish_unit_day_artifact(store, ...)` | `artifacts.publish_unit_day_artifact(experiment, resolver, store, ...)` |
| `analysis.build_panel_for_metric(...)`, `panel_sql(...)`, `summary_sql(...)` | `query.<name>(experiment, resolver, ...)` |
| `analysis.estimate_cate(...)`, `validate_cate(...)`, `targeting_rule(...)`, `select_targeting_rule(...)`, `sitewide(...)` | `estimators.<name>(experiment, resolver, ...)` |
| `analysis.dashboard_snapshot(...)`, `dashboard_breakout_reads(...)`, `dashboard_group_data(...)` | `dashboard.prepare_dashboard(experiment, resolver, config=...)` captures the headline, group data and Explore reads once; `load_explore(experiment, snapshot=...)` and `render_dashboard(experiment, snapshot=...)` read the snapshot |
| `increment.analysis.fit_predeclared_adjustment` | `estimators.fit_predeclared_adjustment` |

Readout options such as `decision_method`, `prior`, `metrics`, `estimands`,
`exploratory_metrics` and `coherence` are `run` keywords and apply to every
selected view. A view's population, breakout selection and completed-window
gate belong to its `Scope`. One `run` reads every declared view from one
capture, so the daily, as-of and whole-window rows agree on the data they
describe:

```python
import polars as pl

from increment import Scope, from_frame, run
from increment.evidence import BreakoutIdentity

panel = pl.DataFrame(
    {
        "user_id": [f"u{index}" for index in range(8)] * 2,
        "variant": ["control", "treatment"] * 8,
        "day": ["2025-01-01"] * 8 + ["2025-01-02"] * 8,
        "country": ["US", "US", "GB", "GB"] * 4,
        "revenue": [float(index % 5) for index in range(16)],
    }
)
experiment, resolver = from_frame(
    panel,
    shape="panel",
    unit="user_id",
    group="variant",
    date="day",
    control="control",
    metrics={"revenue": "mean"},
    breakouts=["country"],
)
whole = Scope(view="run", grain="total")
daily_by_country = Scope(
    view="daily", grain="daily", values=True, segment=BreakoutIdentity("country", None)
)
asof_lift = Scope(view="asof", grain="asof")
result = run(experiment.with_views(whole, daily_by_country, asof_lift), resolver)
estimates = result.readout(whole)  # was analysis.run()
values = result.readout(daily_by_country)  # was analysis.run_daily(dimension="country")
trend = result.readout(asof_lift)  # was analysis.run_asof_lift()
```

### Machine-readable contracts

- A refusal code always keeps its meaning. Message text may change; branch on
  `.code` and context fields.
- Retired codes remain in `increment.errors.RETIRED_CODES`, mapped to a
  replacement, replacement tuple, or `None` if there is no replacement. For
  example, `estimation.binomial.arm_too_large_for_exact_enumeration` maps to
  `estimation.binomial.finite_sample_arm_ceiling_exceeded`. Retirement is a
  contract change listed in the release notes.
- Preserve numeric nulls and their reason, guarantee, and reference metadata.
  Dropping these changes what a result claims.
- Removing a result field or changing its type requires a migration notice.

### Persisted formats

Persisted files use versioned formats; they are not upgraded silently. Re-export
legacy artifacts from trusted definitions. The details below document supported
moments, artifact, and checkpoint formats.

### Portable result cells

Decoders accept saved metadata cells in either the previous source-snapshot-first
order or the current population-first order, and normalize reopened metadata to
the current order.

Serialize the collection and reopen it with its dedicated decoder:

```python
import polars as pl

from increment import MetricSpec, Randomized, from_frame, run
from increment.estimation.readout_types import ReadoutResults

frame = pl.DataFrame(
    {
        "unit": ["c0", "c1", "t0", "t1"],
        "arm": ["control", "control", "treatment", "treatment"],
        "revenue": [1.0, 2.0, 1.5, 2.5],
    }
)
experiment, resolver = from_frame(
    frame,
    shape="summary",
    unit="unit",
    group="arm",
    design=Randomized(
        control_group="control",
        allocation={"control": 0.5, "treatment": 0.5},
        allocation_scheme="independent",
    ),
    metrics=[MetricSpec(name="revenue", type="mean")],
)
rows = run(experiment, resolver).readout()
payload = rows.model_dump_json()
restored = ReadoutResults.model_validate_json(payload)
```

This preserves the rows and collection-level metadata. A row or collection JSON
payload is distinct from a portable `ResultCell` payload; use the envelope
format below when you need per-cell operation metadata, summaries, or Arrow
interchange.

For portable cells, use `ResultCell` and `AnalysisEnvelope` from
`increment.result_envelope`. `ResultCell.to_payload()` / `from_payload()` read
and write JSON-compatible cells; `ResultCell.to_arrow()` / `from_arrow()` read
and write Arrow tables. The matching envelope methods also preserve collection
metadata.

Estimated result collections expose `as_cells()`: this includes run readouts,
`BreakoutEstimates`, `DailyLiftEstimates`, `DailyMetricValues`, and
`ContrastResults`. Individual report objects expose `as_cell()`, including
`CateResult`, `CateValidation`, `TargetingRule`, `TargetingSelection`, and
`ConfoundingSensitivity`.
Store identity is based on the capture (`source_snapshot_id`) and analysis
specification, not on equality of `source_content_digest`. The content digest
records exact canonical estimation inputs as provenance; it is not a store
deduplication key. Moment-row metadata such as `experiment_id` is excluded,
but a user-data column with that name remains part of the evidence.

Different constructors or aggregation orders can produce numerically equivalent
moments with slightly different floating-point representations, and therefore
different exact hashes. Compare estimates and intervals at the numerical
tolerance appropriate to the quantity, not by content-hash equality. The digest
does not round evidence or promise universal cross-constructor equality.

The Arrow `se` column in portable cells, envelopes, and run history uses the
estimator's native uncertainty scale: `log_se` for log-scale relative lift and
`abs_se` when no log-scale standard error is retained. It is not a standard
error on the displayed relative lift (`exp(log_mean) - 1`).

Continuing with the `rows` collection created above, save and reopen an
envelope to render its table or text report:

```python
import json

from increment.result_envelope import AnalysisEnvelope, render
from increment.tables import readout_table

envelope = AnalysisEnvelope(tuple(rows.as_cells()), rows.metadata)
saved = json.dumps(envelope.to_payload())
reopened = AnalysisEnvelope.from_payload(json.loads(saved))
table = readout_table(reopened)
brief = render(reopened)
```

Version-2 cells also cover breakout rows, daily/as-of lift and absolute metric
values, switchback contrasts, honest CATE and targeting reports, and sensitivity
reports. These rows preserve typed report data and identity; their retained
payloads do not recreate fit state or grant portable inference operations.
Check each cell's `supports` before requesting an operation. Raw source outcomes
are not retained: cells that would carry raw outcomes preserve identity and the
reason without serializing those outcomes. The envelope retains no live analysis
or fit object.

`relevel(cell, alpha=...)` is available only for supported retained `t`
intervals, at the same or a lower alpha than already stored; it can preserve or
widen the interval, but cannot narrow it. Directional intervals with an open
endpoint are not relevelable. Requests that would narrow an interval refuse
with `results.operation.alpha_above_allocation`. This cell operation is
separate from `LiftEstimate.reintervalize(alpha)`, which rebuilds from the
retained row but refuses requests above the row's registered plan or selected
family allocation with `results.operation.alpha_above_allocation`. Register a
new alpha in the analysis plan and rerun rather than widening an existing result.
For multiplicity correction, pass retained lift rows to
`select_exploratory_family()`; it derives the family cutoff and returns corrected
estimates.

Choose the format by what must survive. `ReadoutResults.model_dump_json()` keeps
the typed readout collection and its collection metadata. `AnalysisEnvelope` and
`ResultCell` provide portable per-cell summaries, supported operations,
rendering, and Arrow interchange. A persisted execution belongs in
`ResultsStore`: its `Run` keeps the cells alongside execution identity, plan,
and manifest metadata in Arrow-backed storage. These formats serve different
lifecycles; none substitutes for live analysis or fit state.

See the [retained-state inventory](development/retained-state.md) for
construction details and operation limits.



!!! warning "A checkpoint belongs to one monitoring process"
    Do not restart monitoring and spend alpha again: retain the old checkpoint
    or register a defensible new protocol.

??? info "Statistical and wire-format details"
    **Moments cubes.** `increment.workflows.export_moments(experiment, resolver, path)` stamps each row with `moments_format`;
    `from_moments` reads it. Formats vary by writer:

    | Format | Written by | Read by `from_moments` |
    |---|---|---|
    | 13 | Fixed-horizon `export_moments` when method-scoped CUPED evidence is needed: scalar families, categorical cells, or fitted continuous/mixed multicovariate QR state | Yes |
    | 12 | Fixed-horizon `export_moments`; source identity, exact counts, ratio denominator third moment `cden3` | Yes |
    | 11 | Prior fixed-horizon export; source identity and exact counts | Yes; ratio denominator-skew coverage is unavailable |
    | 10 | Earlier fixed-horizon export or current sequential checkpoint export | Yes; fixed-horizon ratio denominator-skew coverage is unavailable |
    | 9 | Prior sequential checkpoint export | Yes |
    | Below 9 | No longer written | Refused: `moments.format.unsupported_legacy` |
    | Above 13 | Not written by this release | Refused: `moments.format.unsupported_future` |

    A cube must use one format (`moments.format.mixed`) and have no duplicate
    `(metric, group_id)` rows (`moments.rows.duplicate`). Fixed and sequential
    records use typed envelopes. Formats 10 and 11 fixed-horizon cubes and
    format 9 sequential cubes remain readable. Every row below format 9 is
    refused before a plan is read, including sequential format 8. A sequential
    checkpoint stamped older than the supported format is refused with
    `sequential.continuation.legacy`. Re-export from trusted definitions or use
    the release that wrote the file; formats are never silently upgraded.

    Historical fixed-horizon format 10 does not synthesize missing
    `assignment_counts`; operations that require those counts refuse with
    `source.moments.assignment_counts`. Its rows remain distinct from typed
    sequential format-10 checkpoints. A format-12 ratio
    row without `cden3` is refused with
    `moments.denominator_third_moment_missing`. Formats 10 and 11 have no
    `cden3`; their lift rows mark denominator-skew coverage unavailable, not
    passed.

    Fixed formats 11, 12, and 13 and sequential format 10 carry source identity.
    `from_moments` restores it rather than using the replay object's default
    study name. Snapshot identity also binds the canonical request, pinned
    evidence, and integer sufficient statistics—not floating aggregates.
    Fixed format 10 and sequential format 9 lack source identity; format 11
    may omit it. Replay identity for these cubes is namespaced by legacy format
    and study ID because provenance is unrecoverable. `source_identity` is
    self-asserted, not authenticated.

    Every ordinary row needs integer `n >= 1` and, if present, integer
    `0 <= successes <= n`. Import and export refuse invalid values, including
    unselected rows, with `moments.count_out_of_range`. Counts remain integers;
    floating centered moments only check count agreement and cannot recover
    counts. Keep count columns integer when storing or merging partitions.

    For per-arm summaries, call
    `moments_from_summaries(rows, metrics=..., control=..., study_id=..., assignment_counts=...)`
    and pass its rows to `from_moments`. It accepts strict mean rows
    (`group_id`, `metric`, `n >= 2`, `mean`, ddof=1 `var`) and conversion rows
    (`group_id`, `metric`, `n`, exact integer `successes`). Extra fields are
    refused with `moments.summaries.invalid`. Without `assignment_counts`,
    `workflows.unit_counts` and health allocation checks refuse with
    `source.moments.assignment_counts`; `workflows.srm` does too once a target
    allocation is declared, and refuses with an `integrity.allocation_*` code
    before that. For ratios and richer summaries, use `from_frame(..., shape="summary")`.

    `LiftEstimate`, `BreakoutEstimate`, `DailyLiftEstimate`, `DailyMetricValue` and
    `ContrastResult` carry `metric_type`: the declared `Metric.type` of the row's metric
    (`mean`, `conversion`, `retention`, `ratio`, `total`, `active` or `quantile`), stamped
    by every `run` view and carried through saved results and frame exports. It is
    `None` when the type is unknown: rows built directly by the low-level estimators, the
    design-level compliance row, and results saved before the field existed.

    `LiftEstimate`, `BreakoutEstimate`, `DailyLiftEstimate` and `ContrastResult` carry
    `metric_type`: the declared `Metric.type` of the row's metric (`mean`, `conversion`,
    `retention`, `ratio`, `total`, `active` or `quantile`), stamped by every `run` view
    and carried through saved results and frame exports. It is `None` when the
    type is unknown: rows built directly by the low-level estimators, the design-level
    compliance row, and results saved before the field existed.

    **Site-volume coverage ends at `end`.** An experiment that declares
    `observation_end` after `end` now publishes site-volume coverage through `end`,
    the window its rows and the estimand use, instead of through `observation_end`.
    Its context and site-volume extension digests change, so republish such
    artifacts. Experiments whose `observation_end` is not on a later local day than
    `end` are byte-identical.

    **Unit-day artifacts.** Artifact context format 2 is the only supported
    format. It contains no source SQL; source recipes use domain-tagged SHA-256
    digests. Opening or publishing format 1 is refused with
    `artifact.format.unsupported`, naming the received and supported versions.
    Republish from trusted definitions; existing format-1 manifests still
    contain their original SQL. See [Canonical unit-day artifacts](guides/data-model.md#canonical-unit-day-artifacts).

    publications use digest format 4 and verify it with the SQL codec on
    DuckDB, PostgreSQL, Snowflake, and BigQuery. Format 2 remains a
    verify-only codec for existing DuckDB relations. Relations stored with
    digest formats 1 or 3 cannot be adopted on a route without a warehouse-side
    verifier; they refuse before unit transfer with
    `query.session.warehouse_artifact.verification_unavailable`.
    Republish from trusted definitions with warehouse-side integrity enabled;
    never relabel a stored digest.

    **Republish and re-register after window-day and label fixes.** Artifact
    context binds `window_days`, the boundary-local day of each declared window
    edge. Contexts missing it are refused with `artifact.context.mismatch`
    before any manifest, relation, or pin is used; republish from trusted
    definitions. Older native sequential registrations bind a different source
    recipe and are refused with `sequential.source.invalid`; register again.
    Frame registrations are unaffected. Older artifacts label Boolean
    breakouts and factors `True`/`False`; republishing uses
    `true`/`false`/`__null__`.

    **Names added in this release.** `increment.semantics.models.local_day`,
    `window_days`, `Experiment.start_day`, `Experiment.end_day`,
    `Experiment.observation_horizon_day`,
    `increment.query.artifact_contract.open_trusted_manifest_snapshot`,
    `WarehouseArtifactStore.abandon_generation(artifact_id, generation_id)`,
    and the `roles=` argument of `increment.estimators.impute.pooled_mean`. The abandon
    call durably hides a generation even if no manifest row exists and is safe
    to repeat. `pooled_mean` without `roles=` still fills but warns with
    `impute.pooled_mean_role_undeclared`.

    `LiftEstimate`, `BreakoutEstimate`, and `DailyLiftEstimate` expose read-only
    `reason` from persisted coded-cell failure fields: the context reason, or
    otherwise the failure code. It does not combine other sampling or interval
    reasons.

    **Site-volume coverage ends at `end`.** If `observation_end` is later than
    `end`, coverage now ends at `end`, matching the rows and estimand. This
    changes context and extension digests; republish affected artifacts.
    Experiments whose observation end is on or before `end`'s local day remain
    byte-identical.

    **New refusal codes.** `query.session.warehouse_artifact.publication_cleanup_incomplete`
    (relations could not be dropped before the manifest insert began) and
    `query.session.warehouse_artifact.publication_state_unknown` (cleanup was
    incomplete after it began: the tombstone could not be written, or relation
    erasure failed after it was) are raised when a publication block exits normally
    without a manifest and cleanup could not finish. Propagating errors are
    re-raised unchanged with a note. Both codes include qualified relation names
    and recovery steps.

    **Sequential checkpoints and segment labels.** A checkpoint belongs to one
    monitoring process; never relabel or rehash it. Frame segment labels use
    `true`/`false` for Booleans and `__null__` for missing values. Older
    checkpoints still replay, but continuing them with canonical source labels
    is refused with `sequential.continuation.rewrite`. Do not restart monitoring
    and spend alpha again: retain the old checkpoint or register a defensible
    new protocol. See [Sequential inference](guides/sequential-inference.md).

    **Triggered chains in sequential checkpoints.** A checkpoint captured with
    a declared trigger contains a nested triggered `SequentialSnapshot`, with
    its own registration, records, states, and `prefix_id`. Optional
    `SequentialRegistration.population`, `SequentialCheckpoint.population`,
    `InferenceSpec.triggered_registration`, and compiled-plan
    `triggered_registration` fields are omitted when unused. The derived
    registration is not added to the hashed artifact context. Existing assigned
    record, prefix, and checkpoint identities do not change; older checkpoints,
    registrations, exports, and artifact contexts replay as before. Older
    artifacts derive the same registration when opened. A checkpoint without
    a triggered chain remains assigned-only; backfilling one is refused with
    `sequential.continuation.legacy`.

    **Frame labels and shared schedules.** Noncanonical segment labels in
    frame registrations now use canonical labels instead of producing
    `missing_arm`. Implausibly lopsided shared schedules declared as unit-cycle
    orders are refused before statistics are built, with
    `source.frame.switchback.schedule` and reason `implausible_realized_split`.

    **Encodings are format-specific.** Each format defines its aliases, tuple
    encoding, collection order, duplicate handling, and runtime-only values.
    There is no general guarantee that a value is "JSON compatible."

### Hashes and pickle

A digest shows that bytes match the hashed encoding. It does not identify the
writer, establish statistical validity, or hide guessable secrets. For
artifacts, authenticity depends on the caller-pinned reference and store
authorization.

Moments files, JSON, and artifacts are the portable interchange formats.
Python pickle is only for trusted objects you produced; it has no cross-release
interchange guarantee. Never unpickle untrusted data.

### `TabularPolicy` persistence

`TabularPolicy` accepts any hashable context key, but JSON round-trips only
string-keyed tables. Other key types are refused at dump time with
`logged_policy.policy.json_context_key`; JSON could otherwise change `1` to
`"1"` and alter the policy. Use `model_dump(mode="python")` or trusted pickle
for typed keys. Built-in `TARGET_POLICY_V1` uses integer keys and is Python-only.

```python
from increment.estimators import TabularPolicy

policy = TabularPolicy(
    policy_id="p",
    version="v1",
    probabilities={"web": {"A": 0.5, "B": 0.5}},
    default={"A": 1.0},
)
assert TabularPolicy.model_validate_json(policy.model_dump_json()) == policy

typed = TabularPolicy(
    policy_id="p", version="v2", probabilities={1: {"A": 1.0}}, default={"A": 1.0}
)
assert TabularPolicy.model_validate(typed.model_dump(mode="python")) == typed
try:
    typed.model_dump_json()
except Exception as exc:
    assert exc.code == "logged_policy.policy.json_context_key"
```

See [Persisting a `TabularPolicy`](guides/logged-policy.md#persisting-a-tabularpolicy).




## Analysis and reporting

`increment.workflows.unit_counts(experiment, resolver)` returns enrolled counts
by canonical group label, including excluded units as `"(unassigned)"`. The
source must retain total-grain counts; day-axis views refuse.

`increment.artifacts.artifact_context(experiment, resolver)` returns the trusted
context for a definitions-backed experiment. Pass it as `expected_context` when
opening an artifact published from that experiment.

`increment.workflows.allocation_history(experiment, resolver, population=...)`
returns a PyArrow table with `experiment_id`, `analysis_population`, `ds`,
`group_id`, `n_daily`, and `n_cumulative`, ordered by population, date, and
arm. Native assigned cohorts are binned by first exposure date; triggered
cohorts start at their first eligible trigger. Artifact-backed assigned history
is available only when the artifact retains assignment identity rows, and
remains aggregate rather than per-unit replay data; otherwise it refuses with
`evidence.assignment.identity_unavailable`. Triggered artifact history is not
available. Clustered sources refuse with `CapabilityError`.

### Mixed-covariate fixed-horizon CUPED

On per-unit fixed-horizon inputs such as `from_frame(..., shape="summary")`,
declare inputs on `Method.cuped_multicovariate(covariates=...)`, using
`CupedCovariate` or the binding-aware `CupedCovariateBinding` for each input.
Each declaration has a `name`, `kind="continuous"` or `"categorical"`, and a
`missing` policy (`"refuse"` or `"impute-indicator"`). Categorical values
must be strings. Multicovariate CUPED supports mean, conversion, and ratio
metrics; ratio metrics use the same declared covariates for numerator and
denominator. The scalar `Method.cuped(covariates=...)` factory is for one
continuous covariate. Both factories accept a column name in place of a
declaration: `"pre"` means `CupedCovariate(name="pre", kind="continuous")`, so
categorical covariates are always declared explicitly. See the runnable
[mixed-covariate CUPED recipe](guides/cuped.md#a-mixed-covariate-dataframe-recipe).

Multicovariate CUPED uses HC2 uncertainty and a Bell–McCaffrey Student-t
approximation, unlike scalar `cuped`, which retains its separate
Welch–Satterthwaite method. It is fixed-horizon only. Native definitions and
verified unit-day artifacts fit continuous/mixed models in the warehouse
without transferring unit rows; only the small triangular solve and
finalization run client-side. A fitted continuous/mixed fit can be exported in
a format-13 fitted envelope and replayed from `from_moments`, including
reintervalization at a changed alpha. A moments cube cannot refit a model or
reconstruct a missing fitted envelope. Clustered assignment and sequential
inference are unsupported. Fits outside M1 are returned as unavailable with
`estimation.cuped_linear.support_region`, not silently replaced by an
unadjusted result. Ratio fits also enforce the M1 raw denominator-precision
bound. See [M1 and statistical limits](guides/cuped.md#mixed-numeric-and-categorical-covariates).

The preregistered coverage campaign in `calibration/cuped_linear.py` has not
been run; its coverage is not established. External-prediction training
(tracker #72) and longitudinal dependence (tracker #80) are not implemented.
See [campaign status and commands](validation.md#mixed-covariate-linear-cuped).

The day-axis collections returned by daily and as-of views, values or lift, are
ordered chronologically by day, then by requested metric and a stable row
identity order. The order is independent of backend row order.

`increment.workflows.available_metrics(experiment, resolver)` lists undeclared
saved per-unit metrics on the experiment's unit, in definitions order. It
excludes report-only `total` and `active` metrics and metrics for other
entities. Whole-window run, breakout, as-of lift, as-of values and daily
values views accept selected names through `run(..., exploratory_metrics=...)`.
Lift rows from the first three carry `role="exploratory"`, join no plan family,
and use two-sided full-plan `alpha`; as-of and daily absolute values have no
role. Declared rows are unchanged. Use `metrics=[]` to read only exploratory
metrics. Unknown or declared names refuse before querying with
`facade.analysis_config.exploratory_metric_unavailable`. Other sources refuse
with `facade.analysis.exploratory_metrics_source_limited`; registered
sequential plans refuse with `facade.analysis.exploratory_metrics_sequential`.
`prepare_dashboard` reads every offered added metric in the same pinned snapshot as the headline.

::: increment.Metric

::: increment.Method

::: increment.estimators.Normal

## Dataframe entry points

::: increment.MetricSpec


::: increment.estimators.to_frame

::: increment.estimators.impute

## Designs

::: increment.Design

::: increment.Randomized

::: increment.Observational

::: increment.Encouragement

::: increment.UptakeSpec

::: increment.ExclusionRestriction

::: increment.AdjustmentSet

## Assignment and study contracts

::: increment.estimators.Assignment

::: increment.estimators.ParallelAssignment

::: increment.estimators.IndependentBernoulliOrder

::: increment.estimators.SharedScheduleOrder

::: increment.estimators.SwitchbackWindow

::: increment.estimators.SwitchbackAssignment

::: increment.estimators.ParallelStudyEnvelope

::: increment.estimators.SwitchbackStudyEnvelope

::: increment.estimators.StudyEnvelope

The artifact API controls how immutable unit-day data is published and adopted.
The store authorizes namespaces; callers provide trusted context and pinned
generation references. Artifact contract modules provide the context, catalog,
and digest helpers.


| Model | Purpose |
| --- | --- |
| `UnitDayArtifactRef` | Pins an immutable artifact generation. |
| `ArtifactRelationRef` | Binds a relation and its digests to that generation. |
| `MeasureManifest` | Records measure provenance and freshness. |
| `ArtifactContext` | Binds canonical context JSON (format 2, hash-only source recipes, no source SQL) to its SHA-256 digest (encoding integrity, not authenticity). Format 1 is retired; republish. |
| `UnitDayArtifactManifest` | Captures the complete artifact manifest. |
| `ExtensionRefBase` | Carries relation, definition, and source digests. |
| `BreakoutDimensionExtension` | Describes a pre-exposure breakout dimension. |
| `FactorDimensionExtension` | Describes a pre-exposure factor dimension. |
| `ClusterIdentityExtension` | Identifies the clustering relation. |
| `CupedPreperiodExtension` | Supplies a pre-period measure. |
| `AssignmentCountsExtension` | Captures assignment population counts. |
| `TriggerPopulationExtension` | Version-3, cutoff-bound trigger membership with trigger-feed completeness evidence. |
| `TriggerMeasureStatsExtension` | Singleton metric trigger-relative per-unit/day sufficient statistics bound to trigger membership. |
| `EncouragementUptakeExtension` | Captures encouragement uptake data. |
| `SiteVolumeExtension` | Records site-level measure volume and freshness. |
| `UnitCovariateExtension` | Carries one declared numeric per-unit covariate. |
| `UnitCovariateLevelExtension` | Carries one declared categorical per-unit covariate, including nulls. |
| `ArtifactExtensionRef` | Tagged union of the eleven concrete extension references. |
| `ArtifactExtensionRequest` | Tagged union of the eleven concrete extension requests. |

These are Pydantic models and tagged unions importable from `increment`; see
`increment.semantics.artifact` for full field detail.

::: increment.artifacts.RelationLocator

::: increment.artifacts.RelationRole

::: increment.artifacts.BaseRelations

::: increment.artifacts.Freshness

::: increment.artifacts.SimpleMetricMeasure

::: increment.artifacts.RatioMetricMeasure

::: increment.artifacts.MetricMeasure

### Artifact extensions

These references describe artifact dimensions, assignment, trigger membership
and outcomes, uptake, and site-volume extensions.

### Artifact requests

Requests validate extension definitions before publication.

::: increment.artifacts.ExtensionRequestBase

::: increment.artifacts.BreakoutDimensionRequest

::: increment.artifacts.FactorDimensionRequest

::: increment.artifacts.ClusterIdentityRequest

::: increment.artifacts.CupedPreperiodRequest

::: increment.artifacts.AssignmentCountsRequest

::: increment.artifacts.TriggerPopulationRequest
::: increment.artifacts.TriggerMeasureStatsRequest

::: increment.artifacts.EncouragementUptakeRequest

::: increment.artifacts.SiteVolumeRequest

::: increment.artifacts.UnitCovariateRequest

::: increment.artifacts.UnitCovariateLevelRequest

### Artifact publication and storage

`from_definitions(..., source_snapshot_evidence=...)` accepts
`increment.artifacts.SourceSnapshotEvidence` only when the source provides a timezone-aware
event-time cutoff and, optionally, certified per-feed completeness times.
Without a cutoff, Increment does not invent one: assigned-only operations remain
available, but triggered membership, triggered outcomes, and trigger-evidence
publication refuse with `source.native.trigger_evidence_required`. Missing feed
certification stays unknown; maximum observed timestamps, publication time, and
experiment end do not prove completeness. Triggered artifact outcome requests
select one `trigger_measure_stats` item per `(trigger, metric)`, bound to the
version-3 membership digest and the same pinned cutoff.

::: increment.artifacts.SourceSnapshotEvidence


Publication connects validated context to immutable snapshots in the
authorized artifact store.

::: increment.artifacts.ArtifactPublication

::: increment.artifacts.ArtifactSnapshot

::: increment.artifacts.ArtifactStore

::: increment.artifacts.compile_unit_day_artifact_context

::: increment.artifacts.unit_day_artifact_extension_catalog

### Digests and hashing

Canonical encodings and SHA-256 helpers keep artifact identities deterministic
across publication and requests.

::: increment.query.artifact_digest.ArtifactDigestError

::: increment.digest.DigestType

::: increment.digest.FieldSpec

::: increment.digest.RelationDigests

::: increment.digest.canonical_schema_bytes

::: increment.digest.canonical_row_bytes

::: increment.digest.canonical_rows_bytes

::: increment.digest.canonical_json_bytes

::: increment.digest.schema_sha256

::: increment.digest.content_sha256

::: increment.digest.digest_relation

::: increment.digest.manifest_sha256

::: increment.digest.context_sha256

::: increment.digest.request_sha256

::: increment.digest.extension_definition_sha256

::: increment.digest.extension_source_provenance_sha256

## Logged-policy evaluation

Off-policy contrast compares two registered policies using a decision trace
logged under one fixed policy version. It uses `LoggedTrace.from_records` or
`from_frame`, not an experiment builder. See the
[guide](guides/logged-policy.md) for the estimand, assumptions, and floors.
Each constructor requires exactly one logging-law route: `registry=`, complete
record-level `logging_distributions=`, or frame
`logging_distribution_column=`. The estimator and admission bounds are the
same for either route.


::: increment.estimators.estimate_policy_contrast

::: increment.estimators.LoggedTrace

::: increment.estimators.PolicyRegistry

::: increment.estimators.TabularPolicy

::: increment.results.PolicyValueContrast

## Switchback results

::: increment.estimation.contrast_results.ContrastResult

::: increment.estimation.contrast_results.ContrastResults

## Declared analysis policy

::: increment.AnalysisPlan

::: increment.MultiplicitySpec

::: increment.ExperimentMetric

::: increment.InferenceSpec

::: increment.Winsorization

## Sequential inference

Continuous means use `InferenceSpec(kind="asymptotic_mean")` with an
ordinary fixed-allocation design. The source binds registration before capture;
`expected_decision_sample_size` optionally sets pre-outcome tuning.
For conversion and retention, `InferenceSpec(kind="always_valid",
baseline_rate=...)` binds an exact Bernoulli e-process from the plan alone.
Ratios may be convertible to means, but that conversion does not itself provide
the same sequential guarantee. See
[sequential conversion monitoring](guides/sequential-inference.md#sequential-monitoring-of-a-conversion-metric)
and the [route decision table](guides/sequential-inference.md#choose-the-route-by-the-metric-and-the-guarantee-you-need).
Definitions-based experiments specify arm weights in `Experiment.allocation`.
See [ordinary continuous monitoring](guides/sequential-inference.md#ordinary-continuous-monitoring)
for assumptions, finalization, and the qualified asymptotic guarantee.


::: increment.estimators.AlwaysValid

::: increment.estimators.AsymptoticMean

::: increment.estimators.ScalarMeanModel

::: increment.estimators.AsymptoticSequentialEvidence

::: increment.estimators.AsymptoticSequentialResult

::: increment.estimators.SequentialRegistration

::: increment.estimators.PredeclaredAdjustment

::: increment.estimators.fit_predeclared_adjustment

::: increment.estimators.SequentialModel

::: increment.estimators.SequentialCell

::: increment.estimators.SequentialCompliancePolicy

::: increment.estimators.SequentialSnapshot

::: increment.estimators.snapshot_from_json

::: increment.estimators.capture_sequential_snapshot

::: increment.estimators.declare_sequential_freeze

::: increment.estimators.estimate_sequential

::: increment.estimators.sequential_definition_id

::: increment.estimators.PredictivePrior

::: increment.estimators.JointReveal

## Power planning

Plan fixed-horizon sample size, power, or minimum detectable effect (MDE) with
the runtime's deployed decision method. Binomial and switchback planning have
different approximations and limits.

!!! warning "Judge enumerated binomial targets by their lower bound"
    For enumerated binomial results, a sample size or effect reaches a target
    only when `power_lower` reaches it, even when `power` reports an enumerated
    point. Closed-form and asymptotic planning routes have no enclosure
    (`power_lower` and `power_upper` are `None`); assess their targets using the
    reported model `power`.

??? info "Statistical details"
    Fixed-horizon binomial planning estimates rejection probability over possible
    event counts. Enumerated results include an enclosure (`power_lower`,
    `power_upper`) conditional on the deployed SciPy/Boost special-function
    error model—not a cross-build floating-point proof. `power` is the
    enumerated point when its maximum absolute distance to either endpoint is
    at most `1e-6` (`power_basis="exact"`); otherwise it is the enclosure's
    lower end (`power_basis="lower_bound"`). A size or effect reaches its target
    only when that lower end does. The dense closed-form route gives a
    model-based point estimate without an enclosure.

    `PowerResult.numerical_qualification` records this scope:
    `scipy_special_function_error_model_conditional_v1` for enumerated results
    and `closed_form_model_only_v1` for the asymptotic model. Requesting an
    unresolved point or reaching the sample-size ceiling with an enclosure
    covering the target refuses with `power.binomial_probability_unresolved`;
    the latter includes `n_per_arm`. There is no accuracy-tuning option.

    Binomial MDE search finds the earliest detectable region to `1e-8` absolute
    and `1e-8` relative tolerance on the relative-effect scale, not the first
    representable float. It reports power at the returned effect. A resolved
    supplied-effect result may return `mde_relative=None` and
    `mde_unavailable_reason="numerical_resolution"`; an unresolved standalone
    MDE refuses with `power.minimum_detectable_effect.numerical_resolution`.
    The planning identity is `hybrid_finite_plus_delta_v3`; recompute persisted
    v1 and v2 plans. See the [planning guide](guides/power-analysis.md#conversion-and-retention-planning-follows-the-runtimes-route)
    for route assumptions and a statsmodels comparison.

    Switchback users can fit a baseline with `source.planning_baseline(metric)`
    and pass it to the three `switchback_*` solvers. No manually supplied
    reference effect, covariance, or population certificate is needed. The
    result remains a model-conditioned moment-t approximation; see
    [switchback planning](guides/power-analysis.md#switchback-contrasts).

    Complete-law simulation declarations and oracle solvers are in
    `increment.semantics.unit_cycle` and `increment.power.unit_cycle`, outside
    the package-root and ordinary power-planning exports.

::: increment.estimation.arm_contract.ArmPlanningProcedure

::: increment.estimators.SummaryStats

::: increment.estimators.Baseline

::: increment.estimators.SwitchbackBaseline

::: increment.estimators.ProspectiveAssumptionProvenance

::: increment.estimators.UnitCycleReference

::: increment.estimators.UnitCycleTApproximation

::: increment.estimators.UnitCycleVarianceEnvelope

::: increment.estimators.PowerDesign

::: increment.estimators.required_sample_size

::: increment.estimators.achieved_power

::: increment.estimators.minimum_detectable_effect

::: increment.estimators.power_curve

::: increment.estimators.switchback_required_blocks_or_units

::: increment.estimators.switchback_achieved_power

::: increment.estimators.switchback_minimum_detectable_effect

::: increment.estimators.unit_cycle_power_lower_bound

## Diagnostics and robustness

::: increment.estimators.sample_ratio_mismatch

::: increment.estimators.allocation_posterior_bands

::: increment.estimators.absorb_factor

`increment.estimators.confounding_sensitivity` reports a marginal conversion risk-ratio interval for an observational `LiftEstimate`; `increment.results.ConfoundingSensitivity` is its receive-only result type.

`increment.workflows.health_check(experiment, resolver)` combines assignment
balance, triggered-cohort balance, descriptive trigger rates, retained
source-maturity/completeness evidence, and declared trigger assumptions. Use it
alongside estimates; it is not a general safety verdict or causal-validity test.
Missing source evidence remains unavailable with its identity and reason. A
non-rejected SRM does not prove valid assignment.

For triggered experiments, inspect assigned and triggered checks separately.
Balanced rates do not remove the assumptions of treatment-independent
triggering, no pre-trigger treatment effect, and stable trigger order.
Sequential triggered analyses also assume `triggered_look_predictability`:
looks, stops, and freezes use only already-revealed triggered-chain data.
Assigned results must not reveal future triggered outcomes, including for the
same metric. The health check does not verify an analyst's decision rule.

::: increment.estimators.confounding_sensitivity

::: increment.results.ConfoundingSensitivity

For fixed-horizon IPTW/AIPW conversion risk ratios, confounding sensitivity
transforms an existing `LiftEstimate`. Binary outcome status is caller-attested,
not verified; robustness is not identification.

```python
import pandas as pd

from increment import from_frame
from increment.workflows import health_check

units = pd.DataFrame(
    {
        "unit": ["c1", "c2", "t1", "t2"],
        "arm": ["control", "control", "treatment", "treatment"],
        "value": [1.0, 0.0, 1.5, 1.0],
    }
)
experiment, resolver = from_frame(
    units,
    shape="summary",
    unit="unit",
    group="arm",
    control="control",
    metrics={"value": "mean"},
)
report = health_check(experiment, resolver)
for check in report.checks:
    print(check.check, check.analysis_population, check.availability, check.conclusion)
```

::: increment.workflows.health_check

## Conditional effects and targeting

`estimate_cate` estimates how treatment effects vary with user characteristics.
Validation and targeting refit through honest or nested splits; their reports
give evidence for those procedures, not for a separate `estimate_cate` fit.
Each operation takes the `(experiment, resolver)` pair from a builder, reads
the experiment's unit-grain rows (triggered units when a trigger is declared),
and records the source identity on its report.

!!! warning "Prediction is not permission to deploy"
    `TargetingRule.predict` scores a candidate; `recommendation` remains the
    deployment evidence gate. Supply the full member roster for each cluster
    when predicting with a cluster rule.

??? info "Statistical and deployment details"
    `estimate_cate(..., cluster_weight="member_count")` gives every member equal
    weight. Set `cluster_weight="equal"` to give each observed cluster equal
    total weight; this requires a cluster column. Intervention grain does not
    change the default. Centering, regression, covariance, and ARD use the
    selected weights.

    Clustered fits use a grouped-score sandwich scaled by `K/(K-1)`, counting
    all observed cluster IDs, including singleton and constant-residual
    clusters. They use no row HC2 divisors or extra row-count corrections.
    Scalar intervals use `t(K-1)`; the interaction Wald quadratic divided by
    dimension `q` uses `F(q, K-1)`. Without clusters, HC2 and ARD are unchanged.

    `CateResult.dimension`, `n_clusters`, `reference_df`, and immutable `vcov`
    hold fit-level inference metadata reused by CATE and contrast projections.
    `se_unadjusted` fits `[1, d]` separately with the same cluster weights and
    sandwich, stored as immutable `unadjusted_vcov`. Covariances stay in the
    live result, not report-only dumps; reports include counts, dimension,
    reference degrees of freedom, and selected weighting.

    Clustered fits raise `InvalidRequestError` with `estimation.cate.*` codes:
    `insufficient_clusters` (fewer than two clusters), `rank_deficient`,
    `invalid_covariance`, `single_cluster_direction`, `singular_wald`, and
    `uncertainty_unavailable`. Singular full covariance alone does not block a
    supported scalar query. An unavailable required interaction Wald test does
    refuse the fit.

    Validation and policy APIs keep `cluster_weight` and cluster IDs through
    the honest split and nested selection. Validation reports member counts
    separately from independent held-out clusters. JSON preserves numeric
    nulls and `unavailable_reason`; unavailable AUTOC evidence keeps the gate
    closed. A clustered constant score reports
    `estimation.targeting.degenerate_rank_distribution`, rather than
    manufacturing evidence from repeated members. Its rank evidence is
    unavailable, not an exception. Training or holdout masks that split
    clusters refuse with `estimation.crossfit.cluster_split`.

    `targeting_rule` and `select_targeting_rule` accept
    `deploy_grain: Literal["unit", "cluster"] | None = None`. Only a declared
    `intervention_grain="cluster"` defaults to cluster deployment. Dependence
    clusters alone keep unit policies, including mixed-treatment observational
    clusters. A unit policy for a cluster intervention refuses before reading
    data with `estimation.targeting.unsupported_unit_deployment`.

    Cluster deployment averages member scores, orders clusters by descending
    score and canonical ID, then takes the longest prefix that fits the budget.
    `cluster_weight="member_count"` budgets members; `"equal"` budgets clusters.
    The same weights apply to fitting, selection, and policy evaluation. A
    cluster is never split or skipped to fill unused capacity. Fractions `0`
    and `1` select nobody and everybody; an oversized first cluster selects
    nobody. `fraction` is requested share; `achieved_fraction` is realized
    share, also retained by inner `FractionScore` rows.

    `CateResult.score(cols, *, cluster_ids=None, deploy_grain=None)` returns an
    aligned unit array or an immutable tuple of `ClusterScore(cluster_id, score,
    member_count)` records in canonical cluster-ID order. IDs are separate
    metadata. `TargetingRule.predict` returns aligned Boolean actions from its
    saved `CateScoreState`: unit rules use a frozen cutoff; cluster rules pool
    the supplied batch, apply the budget, and broadcast each cluster's action.
    Supply the full member roster for each cluster. `predict` scores a
    candidate; `recommendation` remains the deployment evidence gate. Empty
    candidates have no conditional policy effect and return
    `estimation.targeting.empty_group`. A cluster rule's `threshold` is
    descriptive; its ID tie-break and prefix budget determine selection.

    Policy JSON includes the fitted basis, knots, category levels, centering and
    coefficients, so `TargetingRule.model_validate_json(rule.model_dump_json())`
    predicts without fitting again.
    CATE report dumps are report-only; serialize `score_state` for portable
    point prediction.

::: increment.estimators.Covariate

::: increment.estimators.ClusterBootstrap

::: increment.estimators.estimate_cate

::: increment.estimators.validate_cate

::: increment.estimators.select_targeting_rule

::: increment.estimators.targeting_rule

## Segment analysis and rollout

::: increment.estimators.segment_heterogeneity

::: increment.estimators.segment_contrast


Rollout recommendations give each usable segment equal weight. Exposure-
weighted recommendations are not supported.

??? info "Why rollout recommendations are unweighted"
    An exposure-weighted policy value was evaluated and left out. Under
    weighting, the winner's-curse correction did not meet the accuracy bar of
    the adopted unweighted correction. A weighted headline without an honest
    correction for selection bias would be misleading.

::: increment.estimators.segment_rollout_recommendation

## Exploratory multiplicity

Use `select_exploratory_family` to correct eligible whole-window and segment
results together. It accepts only uncorrected fixed-horizon decision rows.

??? info "Family membership and interval details"
    `select_exploratory_family` corrects whole-window `LiftEstimate` rows and
    segment `BreakoutEstimate` rows as one Benjamini-Hochberg family at FDR
    level `q`. Each returned row records `discovery`, `family_axes`, `family_q`,
    `family_threshold`, and `family_size`. A selected row's interval is
    reissued at Benjamini-Yekutieli level `1 - R*q/m`, capped at its nominal
    level, and matches the breakout view's row for the same cell under
    `AnalysisPlan(view_multiplicity=MultiplicitySpec(correction="bh", q=q))`.
    Unselected rows keep nominal intervals.

    Rows retain canonical `family_procedure` (`"bh"`, `"e_bh"`, or
    `"bonferroni"`). Older serialized rows without this field remain `None`;
    presentation reports it as unrecorded rather than inferring it.

    Every refusal names offending rows in its `rows` context:

    | Combination | Classification |
    |---|---|
    | Mean, ratio, and CUPED rows (Wald log interval) | Supported: reissued from persisted statistics and reference. |
    | Conversion and retention (exact binomial) | Supported: reinverted from persisted counts; one-sided geometry is kept. |
    | Clustered ratio (Fieller set) | Supported: reinverted from persisted joint reference. |
    | No relative interval (non-positive arm mean), selected by an absolute margin or additive evidence against zero | Supported: additive interval uses persisted `abs_diff`, `abs_se`, reference, and `abs_alpha`, never narrower than nominal. Rows serialized before `abs_alpha` are refused with `estimation.family.exploratory_construction`. |
    | Breakout segments and whole-window rows | Supported: one family across metric, arm, and segment. |
    | Design-excluded cells (too few units or no control) | Not hypotheses; returned unchanged outside the family. Outcome-based exclusions remain in `m` as non-rejections. |
    | Informative prior | Posterior fields may change; selection uses persisted sampling evidence. Prior-bound rows without a proven sampling-availability marker refuse with `readout.legacy.sampling_unreconstructible`. |
    | Sequential inference | Unfinished; refused with `estimation.family.exploratory_sequential`. |
    | Quantile, percentile-winsorized, and additive-scale rows | Unfinished; refused with `estimation.family.exploratory_construction` because intervals cannot be reissued without approximation. |
    | Already-corrected, sensitivity, or day-axis rows | Refused with `estimation.family.exploratory_pre_corrected`, `estimation.family.exploratory_non_decision`, or `estimation.family.exploratory_row`. |

::: increment.estimation.family.select_exploratory_family

`exploratory_family_exclusion` applies the same admissibility test to one row and
returns the reason it cannot join the family, or `None`. Exclude ineligible rows
and report the returned reason instead of refusing the whole family; refusal
context uses the same values in `reasons` or `constructions`.
One-sided (`greater`/`less`) Wald, exact binomial and Fieller rows are reissued
exactly, with their one-sided geometry, and are not excluded.

::: increment.estimation.family.exploratory_family_exclusion

## Errors

::: increment.IdentificationGate

::: increment.estimators.IdentificationError

::: increment.estimators.CapabilityError

### Errors and refusals

Typed errors and refusal specifications describe invalid definitions,
unsupported requests, and wire-format failures.

An arm-lift readout with no observed treatment arm raises
`InvalidRequestError` with code `readout.arms.no_treatment`. Its context
identifies `control_group` and `observed_arms`; supply at least one
non-control arm to estimate a contrast. This does not prevent reading
arm-level counts or running an otherwise defined allocation diagnostic.

Heterogeneity estimation checks its identification contract before reading
outcomes. `InvalidRequestError` codes and their `context`:

- `cate.identification.randomized_only` (`caller`, `mechanism`):
  `estimate_cate` needs a randomized source.
- `cate.identification.unsupported_mechanism` (`caller`, `mechanism`,
  `supported`): the design is neither randomized nor observational.
- `cate.identification.unsupported_missing_policy` (`caller`, `missing`): an
  observational design must declare `missing="refuse"` for this path.
- `cate.identification.unsupported_max_smd` (`caller`, `max_smd`): the
  observational `max_smd` balance gate is not implemented for this path.

::: increment.estimators.CodedError

`CodedModel`, `CodedValidationMixin` and `unwrap_coded` are public from
`increment.errors`, not the root `increment` namespace.

`CodedModel` and every public input model that inherits it surface known
field-validation failures from direct construction and direct `model_validate`,
`model_validate_json`, and `model_validate_strings` calls as
`InvalidRequestError` refusals with stable `model.field.*` codes and structured
context. Definition-specific validators raise `DefinitionError`; shared
finite-sample metric-type and CUPED refusals raise `InvalidRequestError` with
the same `conversion_inference.finite_sample.*` code and structured context as
frame, estimation and planning requests. Catch `CodedError` when handling both
kinds by code. Pydantic schema boundaries such
as `TypeAdapter` and an ordinary `BaseModel` containing one of these models
still raise `ValidationError`: Pydantic captures the refusal because
`CodedError` intentionally remains compatible with `ValueError`.
Recover the coded refusal explicitly at those boundaries:

Only `CodedModel` (including `Definitions`) also translates declared-field
`Field` constraint failures: they raise
`InvalidRequestError` (a `ValueError`), with a `model.field.*` code, rather
than `pydantic.ValidationError`. Code that catches validation failures from
these models must catch `(pydantic.ValidationError,
increment.errors.InvalidRequestError)` (or `ValueError`) rather than
replacing one with the other. Uncoded-validator, root-level, and nested
non-coded-model failures are reported as `definition.validation` with field
locations when loading YAML definitions.

```python
from pydantic import TypeAdapter, ValidationError

from increment.estimators import DefinitionError
from increment import Definitions
from increment.errors import unwrap_coded

try:
    TypeAdapter(Definitions).validate_python({"day_boundary": "EST"})
except ValidationError as exc:
    try:
        unwrap_coded(exc)
    except DefinitionError as coded:
        refusal = {"code": coded.code, "context": dict(coded.context)}
    else:
        raise
```

::: increment.errors.CodedModel

::: increment.estimators.DefinitionError

::: increment.estimators.InvalidRequestError

::: increment.estimators.RefusalSpec

::: increment.errors.RETIRED_CODES

::: increment.estimators.UnsupportedRequestError

::: increment.estimators.WireFormatError

::: increment.estimators.refuse

::: increment.errors.refusals

::: increment.errors.raiser

::: increment.errors.unwrap_coded

### Warnings

Library advisories are instances of `IncrementWarning` (or
`IncrementRuntimeWarning`/`IncrementDeprecationWarning`, which also satisfy
`filterwarnings`/`pytest.warns` matched against `RuntimeWarning`/
`DeprecationWarning`), carrying a stable `.code` and immutable `.context`
alongside the free-text message -- the warning counterpart to `CodedError`.
Warnings with a declared deduplication scope also carry an immutable
`.dedupe_key`. Randomized lift-cell readouts suppress repeated emissions with
the same key while replaying captured estimator warnings within one operation;
warnings without a key are not suppressed by that replay path.

Data-quality warnings from dataframe sources respect the current Python warning
filter on every run, even when intermediate calculations are cached. Each cached
calculation emits its warnings at most once per run, whether the run reads live
data or takes a snapshot.

Filter and assert on `.code` rather than message text:

```python
import warnings

from increment.estimators import IncrementWarning

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    ...  # a call that may emit a library advisory
codes = {w.message.code for w in caught if isinstance(w.message, IncrementWarning)}
```

::: increment.errors.IncrementWarning

::: increment.errors.IncrementRuntimeWarning

::: increment.errors.IncrementDeprecationWarning

::: increment.errors.WarningSpec

::: increment.errors.warn

## Results (`increment.results`)

Sitewide behavior is exposed through
`increment.estimators.sitewide(experiment, resolver, metric=...)`, with output
values available from the receive-only results namespace.

`sitewide(experiment, resolver, metric, population="assigned")` reports an assigned
intent-to-treat whole-site impact, including when the experiment declares a
trigger. For triggered experiments, `population="triggered"` reports a
conditional impact scaled to enrolled units in finalized trigger membership.
Both populations require an unwindowed metric whose components use sum or
count aggregation. Conversion metrics refuse with
`facade.analysis.sitewide_metric_unaligned` before evidence is loaded because
the converter indicator is not additive across events; declare a count metric
on the conversion fact to read conversion events site-wide. Windowed and other
non-additive aggregations use the same refusal.
Triggered sitewide additionally requires a declared experiment end. It uses
calendar-window outcomes (not `run()`'s trigger-anchored windows) and
discloses membership assumptions, allocation-balance evidence, provenance,
and its exploratory unadjusted multiplicity status on the result.
`SitewideImpact` and `SitewideRatioImpact`
include `analysis_population`, `window`, `membership_cutoff`, `snapshot_id`,
`scaled_units`, `unobserved_members`, `assumptions`, `allocation_balance`, and
`allocation_balance_unavailable_reason` (the integrity code when a triggered
balance cannot run because allocation is undeclared).
The paired-provenance check is content-bound only for artifacts; native
`snapshot_id` reflects declared `SourceSnapshotEvidence`, not table contents.

`increment.results.LiftEstimate.family_procedure` identifies the canonical correction
procedure retained with the result's family (`e_bh`, `bonferroni`, or `bh`), when
applicable. Its read-only `reason` exposes only the persisted reason for a coded cell
failure (the context reason, or the failure code); it does not combine other unavailable
sampling or interval reasons.

::: increment.results.Estimate

::: increment.results.LiftEstimate

Binomial rows persist `binomial_set.numerical_qualification`:
`scipy_special_function_error_model_conditional_v1` states the finite-sample
arithmetic qualification conditional on the deployed SciPy/Boost special-function
error model, not a cross-build proof. Supported rows saved before this field was
introduced load as `legacy_unrecorded_v1`, which makes no arithmetic qualification
claim.

::: increment.estimation.results.JointContrastReference

::: increment.estimation.results.RelativeConfidenceSet

::: increment.results.LiftEstimates

::: increment.results.BreakoutEstimate

::: increment.results.BreakoutEstimates

::: increment.results.DailyMetricValue

::: increment.results.DailyMetricValues

::: increment.results.DailyLiftEstimate

::: increment.results.DailyLiftEstimates

::: increment.results.HeterogeneitySummary

::: increment.results.HeterogeneitySummaries

::: increment.results.SegmentEstimate

::: increment.results.SegmentEstimates

::: increment.results.SegmentRolloutResult

::: increment.results.RolloutRecommendation

::: increment.results.RolloutRecommendations

::: increment.results.RolloutSegment

::: increment.results.RolloutSegments

::: increment.results.PowerResult

::: increment.results.PowerCurvePoint

::: increment.results.PowerCurve

::: increment.results.SRMResult

::: increment.results.AllocationBand

::: increment.results.ExperimentHealth

::: increment.results.HealthCheck

::: increment.results.NotApplicable

::: increment.results.AbsorptionResult

::: increment.results.ClusterScore

::: increment.results.CateScoreState

::: increment.results.CateResult

Pass `include_evaluation_population=True` to CATE validation or targeting APIs
to retain the immutable evaluation roster and its actual base weights.
The default retains no identifiers. Read the snapshot from
`validation.evaluation_population`, or from `rule.validation.evaluation_population`
for fixed and selected rules. Selection captures only the outer evaluation split;
its overlap provenance is independent of the inner selection population.
Equal-cluster weights use retained cluster sizes after overlap trimming, before
policy selection. Snapshot rows, cluster identities, and weights stay aligned
through JSON serialization.

For synthetic validation,
`increment.simulate.cluster_dgp.evaluation_policy_truth(rule, population)` consumes
a saved rule and its `ClusteredCATEResult`. It returns exact policy truth
and the retained-population ATE, using the stored IDs and actual base weights.
Missing or mismatched evaluation rosters are refused, not reconstructed.
This is truth for the same retained population, not an independent evaluation batch.

::: increment.results.CateEvaluationPopulation

::: increment.results.CateValidation

::: increment.results.TargetingRule

::: increment.results.TargetingSelection

::: increment.results.MetricTrend

::: increment.results.SitewideImpact

::: increment.results.SitewideRatioImpact

## Advanced power (`increment.power`)

Specialized segment and heterogeneity solvers are advanced power entry
points:

::: increment.power.segment_pairwise_required_sample_size

::: increment.power.segment_pairwise_achieved_power

::: increment.power.segment_pairwise_minimum_detectable_effect

::: increment.power.joint_q_power_fixed

::: increment.power.joint_q_power_random

## Compliance sufficient state

::: increment.sources.ComplianceArm

::: increment.sources.ComplianceSummary

## Optional dashboard (`increment.dashboard`)

`increment.dashboard` is an optional presentation layer over one bound
experiment: a complete interactive four-tab dashboard, section-level
`mo.Html` helpers, and thin adapters over the public `run` results and
CoefTable APIs. Install it with `pip install
"increment[dashboard]"`; core Increment, estimation, power, and the
dataframe entry points never import marimo, CoefTable, or pandas
through this package. See the [dashboard guide](guides/dashboard.md) for
the caller-owned source/config/prepare/render recipe, supported experiment
shapes, and statistical disclosures.

::: increment.dashboard.DashboardConfig

::: increment.dashboard.DashboardSnapshot

`increment.dashboard.ExploreView` is the exported literal type
`"cumulative_lift" | "daily_values" | "cumulative_values" | "segments"`,
accepted by `load_explore` and `render_explore`.

::: increment.dashboard.prepare_dashboard

::: increment.dashboard.render_dashboard

::: increment.dashboard.dashboard_styles

::: increment.dashboard.render_header

::: increment.dashboard.render_health

::: increment.dashboard.render_results

::: increment.dashboard.render_metric_details

::: increment.dashboard.load_explore

::: increment.dashboard.render_explore

::: increment.dashboard.render_details

::: increment.dashboard.readout_csv

::: increment.dashboard.group_data_csv
