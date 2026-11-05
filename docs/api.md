# API Reference

Use `increment` for workflow entry points and caller-authored configuration.
Receive-only result types are in `increment.results`; specialized power solvers
are in `increment.power`.

## Start here

::: increment.from_frame

::: increment.from_definitions

::: increment.run

::: increment.ExperimentReader

::: increment.RunResult

::: increment.estimators.MetricTrends

::: increment.Definitions

`increment.__version__` reports the installed package version.

`Experiment` is the immutable execution specification. `ExperimentDefinition`
is the experiment declaration written in definitions and returned by
`increment.workflows.experiment_definition`; custom evidence providers use it
for experiment metadata.

::: increment.ExperimentDefinition

## Compatibility policy

This policy defines the compatibility promise beginning with release 1.0.0. It
does not claim that the current alpha release is 1.0.0 or already has this
promise. Before 1.0.0, compatibility between prereleases is not guaranteed.
This section covers interfaces, persisted formats, and migrations—not
statistical certification. A stable interface does not establish that a
method's assumptions fit your experiment. See [Statistical limitations](limitations.md)
for method assumptions and the [capability table](reference/capabilities-by-entry-point.md)
for what runs on each entry point.

### Covered contracts

The machine-checked [public manifest](https://github.com/kylejcaron/increment/blob/main/tests/public_exports.txt)
is the import promise from 1.0.0. It lists one user task per name, grouped by
namespace:

- `increment` for workflow entry points and caller-authored declarations;
- `increment.estimators`, `increment.power`, `increment.results`, and
  `increment.evidence` for estimation, planning, received results, and custom providers;
- `increment.artifacts`, `increment.digest`, and `increment.store` for
  artifact publication, integrity verification, and saving and reopening runs;
- `increment.dashboard`, `increment.tables`, and `increment.simulate` for
  presentation and synthetic data;
- `increment.errors` for errors and warnings callers catch;
- `increment.query` for `build_panel_for_metric`, `panel_sql`, and `summary_sql`.
  Each operates on an `(experiment, reader)` pair; SQL previews preserve
  their optional-breakout defaults and use adopted SQL as the summary fallback.
- `increment.workflows` promises only `materialize`, `export_moments`,
  `breakout_summaries`, `factor_summaries`, `experiment_definition`, `metrics`,
  `compiled_decision_plan`, `available_metrics`, `unit_counts`, `srm`,
  `trigger_rates`, `allocation_history`, `health_check`,
  `assignment_diagnostic`, `planning_baseline`, `capture_sequential`, and
  `sequential_snapshot`.
  These functions operate on an `(experiment, reader)` pair; the experiment
  definition inspection returns the semantic source declaration, not the
  kernel execution request. A sequential capture is kept by the reader's
  source, so the next `run(experiment, reader)` and `export_moments` with
  that reader read it.
- `increment.artifacts.artifact_context` and `publish_unit_day_artifact`
  operate on an `(experiment, reader)` pair; publication retains trusted
  extension selection and refresh semantics. `WarehouseArtifactStore`
  implements the artifact store contract for warehouse publication.
- `increment.estimators` conditional-effect operations `estimate_cate`,
  `validate_cate`, `targeting_rule`, and `select_targeting_rule`, and the
  whole-site estimate `sitewide`, operate on an `(experiment, reader)` pair;
  their remaining parameters are unchanged. The data-only array kernels in
  `increment.estimation.targeting` and `increment.estimation.sitewide`, and the
  `increment.power` solvers, take no source.

The promise covers the listed public names and their signatures (new optional
parameters only); result fields and schemas, including row sets, null reasons,
and interval reference/guarantee labels (new fields only, never renamed or
reinterpreted); stable error codes and their named context keys (message text
may change); each saved format and the versions its readers accept, listed in
the [saved-format table](reference/saved-formats.md); and identity hashes,
including `Experiment.hash()`, `View.hash()`, and `scope_key`.

Other modules, underscore names, registry and dispatch helpers, private
research kernels, and the `increment.experimental` namespace are not covered
by this promise and may change without notice.

### Change and forward-compatibility rules

Public additions may ship in minor releases. To remove or change a promised
contract, first deprecate it with a warning and state the release in which it
will be removed. Removal happens in the next major release. Clean cutovers
without an alias remain appropriate for private code only.

An older release refuses data written in a newer saved format with a coded
upgrade-required error. It does not guess how to interpret unknown data or
silently upgrade it. The saved-format table identifies the supported versions
and refusal codes.

### Support matrix

The package requires Python 3.12 or newer (`>=3.12`). Core and dataframe analysis
are tested on Python 3.12–3.15; warehouse workflows are tested on 3.12–3.14.
DuckDB-backed extras on Python 3.15 need a compatible DuckDB wheel. The dependency
floors below describe supported configurations, not a claim that the current
alpha release is 1.0.0. Dependency changes are reflected in `pyproject.toml`.

| Installation | Dependency floors |
| --- | --- |
| Core | Pydantic 2.14.1, packaging 24.0, NumPy 2.5.4, SciPy 1.18.1, Narwhals 2.27.1, PyArrow 26.0.0 (25.0.0 excluded); PyYAML 6.0.1 on Python 3.12, 6.0.2 on 3.13, and 6.0.3 on 3.14 and newer |
| `warehouse` extra | Ibis 10.1.0 on Python 3.12–3.13 and 11.0.0 on 3.14; SQLGlot `>=30.14`, excluding `30.18.0` through `30.27.0` inclusive and `30.29.0` (the declared warehouse-extra constraint) |

These floors describe supported dependency combinations, not statistical
certification. Optional extras have their own dependencies in `pyproject.toml`.

### Numerical corrections

A bug fix may change a previously incorrect numerical answer. Disclose
numerical corrections in release notes; compatibility does not preserve wrong
values.

### Earlier release notes

- Release note (usability): reading a public submodule such as
  `increment.workflows` from the package before importing it raises an
  `AttributeError` that names the import, `from increment.workflows import ...`;
  the submodule is not imported for you. Guides and examples call top-level
  names as `increment.<name>` after `import increment`, import submodule names
  with `from increment.<submodule> import ...`, and name a builder's pair
  `experiment, reader`.
- Release note (bug fix): a `from_frame` reader keeps the rows its dataframe
  held when the builder ran. Previously a quantile or other raw-row read of a
  summary, or a CUPED covariate of a panel, could read a Polars dataframe
  changed in place after the build, such as with `replace_column`. A Polars
  dataframe is cloned without copying its columns, an Arrow table is kept as
  is, a pandas 3 dataframe is shallow-copied under copy-on-write, and other
  pandas-like dataframes are copied. Estimates from unchanged dataframes and
  experiment identity hashes are unchanged.
- Release note (bug fix): a `missing="skip"` metric's breakout and daily lift
  rows now state that they cover only the units with a value, as its
  whole-window rows do, in a new optional `population` field on
  `BreakoutEstimate` and `DailyLiftEstimate` (as-of lift rows included) that
  reaches `to_frame()`, the rendered row, saved cells and pickles. No source
  records a segment's or a day's enrolled units, so the label reads
  `observed-only population; counts not recorded per segment` (or `per day`);
  the row's `n_control` and `n_treat` are its observed units. Rows of other
  metrics keep `population=None` and an unchanged saved payload; estimates,
  source snapshots and experiment identity are unchanged.
- Release note (breaking API cutover): `MetricSpec(missing="drop")` is now
  spelled `missing="skip"`, with the same computation: the metric uses only the
  units with a value, and those units stay in every other metric. Estimates are
  unchanged. `missing="drop"` refuses with `frame.metric.missing_drop_renamed`,
  whose `route` names `missing='skip'`; the panel refusal
  `frame.missing_policy.panel_drop` is retired to
  `frame.missing_policy.panel_skip`. A skip metric's whole-window rows state in
  `LiftEstimate.population` how many units each group observed, for example
  `observed score only (control 150 of 200 units, treatment 100 of 200 units)`;
  the label reaches `RunResult.to_frame()` and the rendered row, and a row that
  already carries an observational population restriction keeps both, the skip
  label first, joined by `"; "`. A `from_moments` experiment records no unit
  counts, so its rows say the counts were not recorded. When the share of
  missing values differs between groups (Pearson chi-square, p < 0.001), the
  warning `frame.validation.metric_missing_differs_by_arm` reports each group's
  `n_missing` and `n_units` and the `p_value`; the check counts units, so it is
  not run when `cluster=` is declared or when an expected count is below 5. The
  per-metric warning
  `frame.validation.metric_missing_drop` is removed without replacement. When
  every value in one group is missing, the metric's comparisons with that group
  are unavailable with `readout.cell.missing_metric_observations` while other
  metrics still estimate, and in a multiplicity family such a comparison counts
  as a non-rejection, so the family keeps its size on both the whole-window and
  the breakout readout. Under `"drop"`, missing control values aborted the run
  and missing treatment values shrank the family. An experiment loaded with
  `from_moments` records no unit counts, so a moments input with no row for a
  secondary's treatment group drops that comparison from the family, and one
  with no control row refuses: `estimation.engine.control_group_found` on the
  whole window, `family.evidence.incomplete` for a secondary family.
  Experiments that declared `missing="drop"` now declare `"skip"`, so their
  `to_json()`, `hash()` and `analysis_spec_hash` change once, and a
  `from_moments` experiment declaring `"skip"` also records the policy in its
  source declaration. Experiments with any other policy keep their identity.
- Release note (bug fix): an `ExperimentDefinition` prints as
  `ExperimentDefinition(...)`, the name it is imported under, instead of
  `Experiment(...)`, the name of the execution specification. Its fields, saved
  JSON and identity hashes are unchanged.
- Release note (feature): an `Experiment` now pickles and deep-copies, keeping
  its scopes, source references and implementation references typed. An
  in-memory-only experiment that declares a raw learner factory copies too when
  the factory itself pickles, such as a module-level function or class.
  `hash(experiment)` now works and agrees with `==`, so equal experiments are
  one dictionary key; it is not `experiment.hash()`, the portable identity,
  which can tell apart declarations that compare equal, such as `Fraction(1, 2)`
  and `0.5`. Readers still do not pickle. `to_json()`, `hash()` and
  `analysis_spec_hash` are unchanged.
- Release note (bug fix): an experiment's `hash()` and `analysis_spec_hash`, and
  the identities of its runs, no longer include the column types recorded for
  its sources. A connected and an offline `from_definitions` experiment now
  keep one identity before and after the first run records those types;
  `to_json()` still saves them and `Experiment.provides()` lists the recorded columns.
  Identities of definitions-built experiments, and of experiments declared with
  `SourceReference(schema=...)`, change once, and so do the cell identities of
  their new runs, because a cell's identity includes its run's
  `analysis_spec_hash`. Runs saved earlier still reopen from their saved
  `store.open_run` coordinates with `RunResult.from_handle`. Dataframe, moments,
  switchback and artifact experiments record no column types, so their
  identities are unchanged.
- Release note (bug fix): after the definitions behind a `from_definitions`
  reader change, `run(experiment, reader)` and every function that reads
  data through that reader refuse with `kernel.definitions.catalog_stale`
  before reading anything, instead of reading the data and then refusing.
  Comment and whitespace edits no longer count as changes. The offline catalog
  format moves to schema 2; a schema-1 catalog refuses offline with
  `kernel.definitions.catalog_stale`, and rebinding the definitions with a
  connection rebuilds it. Experiment identity hashes are unchanged.
- Release note (bug fix): an exact binomial conversion or retention row reports
  its arm unit counts and observed rates in `n_control`, `n_treat`,
  `mean_control` and `mean_treat`, as other unclustered conversion rows do.
  Estimates, intervals and p-values are unchanged. Because those rows now carry
  arm counts, the snapshot and family IDs of daily and as-of lift readouts that
  contain them change once; runs saved before the change still reopen.
- Release note (breaking API cutover): every builder returns an
  `increment.ExperimentReader` with the experiment, including an offline
  `from_definitions(path)`, whose object previously had a default repr and no
  `with` support. Its public members are `experiment`, `closed` and `close()`,
  and it is a context manager; the `binding`, `resolve`, `verify`,
  `validate_experiment`, `run_experiment` and `catalog_writer` attributes are
  private. A closed reader stays closed: after `close()` or a `with` block, `run`
  and every function that reads data refuse with `kernel.run.reader_closed`
  (context `experiment_id`) instead of reopening the source; build a new reader
  to read again. An offline reader refuses every read with
  `kernel.run.reader_offline`. `run` and the data-reading functions refuse
  anything but a reader, such as `None`, swapped arguments, a `BoundResolver` or
  another object with `resolve` and `verify` methods, with
  `kernel.run.reader_required`, whose `received` context names the type passed.
  `increment.errors.RETIRED_CODES` maps the retired `kernel.run.resolver_missing`
  to `kernel.run.reader_required` and `kernel.run.reader_offline`, and
  `kernel.run.provider_execution_unavailable` to `kernel.run.reader_required`.
  `repr(reader)` reads `ExperimentReader(experiment_id=..., reads=...,
  closed=...)`, where `reads` is `"dataframe snapshot"`, `"warehouse, read live"`,
  `"artifact"`, `"moments file"` or `"offline: cannot read"`. Results and
  experiment identity hashes are unchanged.
- Release note (breaking API cutover): `run` and every function that reads data
  name their second parameter `reader`, the `ExperimentReader` a builder
  returns, instead of `resolver`; positional calls are unchanged, and refusal
  and health-check routes that name a call, such as `run(experiment, reader)`,
  use the new name. Because a health report's `source_snapshot_id` covers its
  checks, a report whose allocation check carries that route gets a new ID once.
  `Resolver` and `BoundResolver` move from the package root to
  `increment.evidence` under the same names. The four `kernel.resolver.*`
  codes, which readers and `BoundResolver` raise for the same hazards, become
  `kernel.provider.digest_mismatch`, `kernel.provider.capability_mismatch`,
  `kernel.provider.mismatch` (was `kernel.resolver.provider_mismatch`) and
  `kernel.provider.capabilities_missing`; `increment.errors.RETIRED_CODES` maps
  each retired code to its replacement. `kernel.provider.digest_mismatch` now
  carries one context from every source, `source_refs` (the names of the
  source references that did not verify); the recorded and provider digests,
  previously the `source_ref`, `expected` and `actual` keys, are in its message.
  The other three codes keep their context keys.
  When the experiment passed with a reader is not the one the reader was built
  for, `kernel.run.execution_unavailable` now names the first differing
  declaration in `field` (`experiment_id`, `design`, `registration`,
  `analysis_context` or `metrics`, previously `declarations`) and the reader's
  experiment in a new `reader_experiment_id` context key. Results and
  experiment identity hashes are unchanged.
- Release note (breaking API cutover): `Scope` is renamed `View` and its
  overall constructor `Scope.run()` is now `View.overall()`; `View.daily`,
  `View.asof`, `View.breakout` and `View.switchback` and every constructor
  keyword are unchanged. `Experiment.scopes` is now `Experiment.views`, also as
  the `Experiment(...)` keyword, and `RunResult.readout(scope)` is now
  `readout(view)`; positional calls are unchanged. `Experiment.with_views(*views)`
  now adds views after the declared ones instead of replacing them:
  `experiment.with_views(View.daily())` declares the builder's overall view and
  the daily view, and `run` returns both. Pass `views=` to `run` to execute a
  subset. An empty call, or a view already declared, refuses with
  `kernel.request.invalid`, so omit the builder's `View.overall()` from the
  views you add. Refusal routes that named a `Scope(...)` construction, such as
  the `route` of a day-axis refusal, now name the `View` constructor, such as
  `View.asof()`. Saved experiment JSON keeps its `"scopes"` key, and every view
  hash, the identity of an experiment whose views did not change, and the
  `RunResult.to_frame()` columns `scope_view`, `scope_values` and `scope_key`
  are unchanged; an experiment that used `with_views` to replace the builder's
  view now also declares that view, so its hash changes.

- Release note (typing): public signatures and fields no longer use `Any`;
  `make lint` now refuses any new `Any` in a promised name. Results, refusal
  codes, saved formats and identity hashes are unchanged; a recorded `source_name`
  that is not a string now raises `TypeError`. Type checkers now see the precise
  types, so custom providers and callers may need to
  narrow values they previously treated as `Any`: moment and breakout row cells
  are typed scalars, `UnitRows` is a dataframe, `MethodFitAnswer.envelope` is a
  protocol with `fit`, `arm_order` and `withheld`, saved payloads are JSON
  values, and `EvidenceExecutionContext.open()`/`borrow()` return the evidence
  source a scope yields. `RunResult.run` exposes `analysis_spec_hash`.

- Release note (bug fix): `Experiment.from_json` now refuses unknown design
  mechanisms with `design.mechanism_unsupported` before accepting the saved
  declaration.

- Release note (breaking API cutover): the public manifest now covers every
  promised namespace, including `errors`, `tables`, `store`, `simulate`, and
  the three `query` workflows. The following names leave the import promise;
  their defining modules continue to serve library internals. No refusal
  codes, structured context keys, validation order, saved formats, or identity
  hashes change.

    | Namespace | Names no longer promised | Criterion and supported route |
    | --- | --- | --- |
    | `increment.estimators` | `ESTIMATORS`, `estimator_id`, `load`, `observational_estimator_id`, `randomized_estimator_id`, `routes`, `resolve` | Registry and dispatch plumbing; use `runnable` for offline preflight and `Method` for method selection. |
    | `increment.estimators` | `refuse`, `warn`, `WarningSpec` | Refusal/warning emission plumbing; catch `CodedError` or filter `IncrementWarning` instead. Custom providers may return `RefusalSpec` with their breakout rows. |
    | `increment.artifacts` | `compile_unit_day_artifact_context` | Context-compilation plumbing; use `artifact_context(experiment, reader)`. |
    | `increment.evidence` | `BreakoutRoute`, `BreakoutSlice`, `BreakoutSlices`, `BreakoutSlicesRequest`, `BreakoutUnitRows`, `BreakoutUnitRowsRequest`, `DayAxisRequest`, `DayAxisRoute`, `DayAxisRows`, `MetricEvidence`, `MetricEvidenceRequest`, `PreparedMulticovariateFit`, `EvidenceSource`, `warehouse_cuped_fit_evidence` | Compound-request and acquisition plumbing; custom providers implement the primitive `EvidenceProvider` requests. |
    | `increment.errors` | `CodedValidationMixin`, `MODEL_FIELD_REFUSALS`, `PACKAGE_FRAMES`, `WarningSpec`, `raiser`, `refusals`, `refuse`, `warn`, `unwrap_coded_serialization` | Model-validation, registry, and emission plumbing; use catchable errors and warnings, `CodedModel`, and `unwrap_coded`. |

- Release note (API addition): import priors, fixed inference, dataframe
  sequential-registration helpers, and exploratory-family operations from
  `increment.estimators`; saved readout decoders, `AnalysisEnvelope`, and `ResultCell` from
  `increment.results`; `WarehouseArtifactStore` from `increment.artifacts`;
  planning-procedure field types from `increment.power`; and `ComplianceArm`
  from `increment.evidence`. `ExperimentDefinition` at the root names the
  semantic declaration separately from the execution `Experiment`.
  These are re-exports of the existing objects, not new statistical behavior.

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
  `from_moments` return an experiment/reader pair, and top-level
  `run(experiment, reader)` returns `RunResult`. See the
  [experiment specification guide](guides/experiment-specifications.md).
- Release note (breaking API cutover): the `Analysis` class, its
  `increment.Analysis` export and the `increment.analysis` module are removed
  without an alias. Build an experiment and reader with a module builder and
  read results with `run(experiment, reader)`; source operations moved to
  `increment.workflows`, `increment.query`, `increment.artifacts`,
  `increment.estimators` and `increment.dashboard`. The narrow builder aliases
  `increment.builders.from_unit_summary` and `from_unit_panel`, the
  experiment-name-first `from_definitions(experiment_name, definitions_path,
  con)` form, the `run(analysis, registration, compiled_plan, ...)` overload and
  `increment.readouts.run_scoped` are removed too. Results, refusal codes and
  context keys are unchanged. Context values that named an `Analysis` member
  (`route`, `constructor`, `reason`) now name the replacement builder, `run`
  view or workflow function, and refusal messages name the replacement call.
  No refusal code is retired with `Analysis`, so `increment.errors.RETIRED_CODES`
  is unchanged.
  `repr(experiment)` is one bounded line naming the experiment, its design,
  metrics, view count and identity prefix; `repr(reader)` names its source
  and whether it is closed without reopening it. See
  [Migrating from `Analysis`](#migrating-from-analysis) for every replacement.
- Release note (breaking API cutover): `ResultsStore.persist_results`,
  `ParquetResultsStore.persist_results`, `WarehouseResultsStore.persist_results`
  and the `increment.store.producer` module with its `PersistenceIdentity` are
  removed without an alias; save a run with
  `run(experiment, reader, results_store=store)`. The saved run
  and cell identities are on `RunResult.run` and `RunResult.cells` rather than
  on the returned rows. A compliance-only selection made with a call-time
  `estimands=("compliance",)` cannot be saved, and a live percentile-winsorized
  confidence set refuses to save with `wire.result.raw_outcomes_unavailable`
  before any read. Readout JSON written with persisted cells stays readable.
  `dashboard.prepare_dashboard` replaces the dashboard capture members; a
  dataframe or moments experiment refuses with `dashboard.unsupported_experiment`.
- Release note (breaking API cutover): the module-builder reader no longer
  has a `capture_sequential` method. Call
  `increment.workflows.capture_sequential(experiment, reader, ...)`, and read
  the retained checkpoint with `increment.workflows.sequential_snapshot`.
- Release note (breaking API cutover): `increment.estimators.estimate_cate`,
  `validate_cate`, `targeting_rule`, and `select_targeting_rule` take the
  `(experiment, reader)` pair returned by a builder such as `from_frame` or
  `from_definitions` instead of a source. Their remaining parameters, defaults,
  results, and refusal codes are unchanged; `estimate_cate`'s `control` defaults
  to the declared control arm, and a declared trigger restricts every operation
  to triggered units.
- Release note (breaking API cutover): `increment.workflows.planning_baseline(experiment, reader, metric)`
  and `increment.estimators.sitewide(experiment, reader, metric, *, arm=None, alpha=0.05, population="assigned")`
  replace `Analysis.planning_baseline` and `Analysis.sitewide`. Results,
  trigger/CUPED/cluster/quantile/compliance handling, switchback baselines,
  calendar-window site volume, randomized-only restrictions, and refusal codes
  are unchanged. `increment.power` gains no source-reading function.
- Release note (breaking API cutover): `run(experiment, reader, ...)` no
  longer accepts `analysis_options`. Pass the same readout options as keywords:
  `decision_method`, `sensitivity_methods`, `prior`, `metrics`, `estimands`,
  `value_scale`, `exploratory_metrics`, `population` and `coherence`; only the
  options you pass apply, so `prior=None` still differs from an omitted prior.
  Pass a sequential capture as `capture=SequentialCaptureRequest(...)` and an
  open shared evidence capture as `execution_context=`. The `store` option is
  not accepted. A `population` that differs from a selected scope refuses with
  `kernel.request.population_scope_only`; declare it on `Scope(population=...)`.
- Release note (API addition): `run(experiment, reader, ...)` saves and logs
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
- Release note (bug fix): a breakout view of an experiment that declares no
  breakouts returns no rows when it runs beside other views, as it does alone,
  instead of refusing a metric that only a breakout read could not serve.
- Release note (API addition): `run(..., results_store=)` saves switchback
  contrast runs, replacing `Analysis.run(store=)`. A saved switchback run keeps
  its cells, metadata and method identities for `store.query` and
  `store.history`, but no decision plan: `RunResult.from_handle` refuses it
  with `store.decision.reference_invalid`, and `decision_log=` refuses with
  `wire.procedure.contrast_context`. Runs saved by `Analysis.run(store=)`
  remain readable; history does not join them with new runs, because the
  registration, scope and method identities differ.
- Release note (bug fix): an experiment saved with `to_json` keeps the view
  parameters it declared, so its hash does not change with a private executor
  parameter and it runs against a new reader. A `coherence` saved in an
  experiment's request options applies to every selected view, and a
  `coherence` passed to `run` replaces the saved one in either direction.
- Release note (breaking packaging change): Ibis and SQLGlot are optional.
  Install `increment` for dataframe and moments builders; they do not import
  either package. For `from_definitions`, `from_unit_day_artifact`, SQL previews,
  artifact publication, metric trends, and warehouse-only workflows, install
  `pip install "increment[warehouse]"` plus your Ibis backend driver, such as
  `"ibis-framework[snowflake]"`. The `demo` extra includes warehouse support
  and DuckDB. Without warehouse support, those entry points and direct imports
  from `increment.query.session` or `increment.query.artifact_digest` refuse with
  `install.warehouse_extra_missing`; its `feature` and `install` context fields
  identify the requested feature or module and the install command.
- Compatibility policy is the same across methods; guarantees and evidence
  differ. See [Evidence status](validation.md#evidence-status).

### Migrating from `Analysis`

Every removed `Analysis` member, every public name moved since, and every renamed
option has one replacement. In the table, `experiment, reader` is the pair a
builder returns, `workflows`, `query`, `artifacts`, `estimators` and `dashboard`
are the `increment.*` namespaces of those names, and parameters not shown keep
their names, defaults and refusals.

| Removed call | Replacement |
|---|---|
| `Analysis(experiment_name, definitions_path, con, ...)`, `Analysis.from_definitions(experiment_name, definitions_path, con, ...)` | `experiment, reader = from_definitions(definitions_path, con, experiment_name=..., ...)`; the path comes first and the name is a keyword |
| `Analysis.from_unit_summary(frame, ...)`, `increment.builders.from_unit_summary(frame, ...)` | `from_frame(frame, shape="summary", ...)` |
| `Analysis.from_unit_panel(frame, ...)`, `increment.builders.from_unit_panel(frame, ...)` | `from_frame(frame, shape="panel", ...)` |
| `Analysis.from_unit_day_artifact(store, ref, expected_context=..., ...)` | `from_unit_day_artifact(store, ref, expected_context=..., ...)` |
| `Analysis.from_moments(rows, ...)` | `from_moments(rows, ...)` |
| `Analysis.from_switchback_panel(frame, ...)` | `from_switchback_frame(frame, ...)` |
| `with analysis:`, `analysis.close()` | `with reader:`, `reader.close()`; neither closes a connection you supplied |
| `reader.run_experiment(experiment, provider, ...)`, `reader.resolve(...)`, `reader.verify(...)`, `reader.validate_experiment(experiment)` | `run(experiment, reader, ...)`, which checks the experiment against the reader and verifies its source declarations before reading |
| `reader.binding` and the source, plan, design and metrics it held | `reader.experiment`, and the `workflows`, `query`, `artifacts` and `estimators` functions taking `(experiment, reader)`, such as `workflows.metrics`, `workflows.compiled_decision_plan` and `workflows.experiment_definition` |
| `reader.catalog_writer(experiment)` | `run(experiment, reader)` on a connected `from_definitions` pair, which records the offline catalog |
| reading through a reader after `close()` or its `with` block | build a new pair with the builder that returned it; a closed reader refuses with `kernel.run.reader_closed` |
| `analysis.run(**options)` | `run(experiment, reader, **options).readout()`, the same `LiftEstimates` or `ContrastResults` |
| `analysis.run(population="triggered")` | `view = View.overall(population="triggered")`, then `run(experiment.with_views(view), reader, views=(view,)).readout(view)`; omit `views=` to read the builder's assigned view beside it |
| `analysis.run(store=store)` | `run(experiment, reader, results_store=store)`; `decision_log=` and `run_id=` are checked before any read |
| `run(analysis, registration, compiled_plan, *, experiment_id=..., analysis_spec_hash=..., execution_identity_hash=..., run_id=..., ...)` | `run(experiment, reader, run_id=..., results_store=..., decision_log=..., ...)`; the experiment supplies its registration and identities |
| `analysis.run_breakout(...)` | `View.breakout(segment)`; `segment=(BreakoutIdentity(property, source), ...)` selects declared breakouts, and `()` selects all |
| `analysis.run_daily(dimension=property, ...)` | `View.daily(values=True, segment=(BreakoutIdentity(property, None),))` |
| `analysis.run_daily_lift(dimension=property, ...)` | `View.daily(segment=(BreakoutIdentity(property, None),))` |
| `analysis.run_asof(completed_windows_only=flag, ...)` | `View.asof(values=True, completion_policy=flag)` |
| `analysis.run_asof_lift(completed_windows_only=flag, ...)` | `View.asof(completion_policy=flag)` |
| `population=` on a day-axis or breakout method | `population=` on `View.daily`, `View.asof` or `View.breakout` |
| `increment.readouts.run_scoped(analysis, ...)` | `declared = {view.hash() for view in experiment.views}` and `added = [view for view in views if view.hash() not in declared]`, then `run(experiment.with_views(*added) if added else experiment, reader, views=views)`: one capture of the data serves every selected view; `with_views` refuses a view the experiment already declares, such as the builder's `View.overall()` |
| `analysis.materialize()`, `analysis.materialize().run()` | `workflows.materialize(experiment, reader)`, then `run(experiment, reader)` |
| `analysis.export(path)` | `workflows.export_moments(experiment, reader, path)` |
| `analysis.experiment` | `workflows.experiment_definition(experiment, reader)`, the semantic declaration |
| `analysis.metrics`, `analysis.compiled_decision_plan`, `analysis.available_metrics` | `workflows.metrics(...)`, `workflows.compiled_decision_plan(...)`, `workflows.available_metrics(...)`, each taking `(experiment, reader)` |
| `analysis.unit_counts()`, `srm(...)`, `trigger_rates()`, `allocation_history(...)`, `health_check(...)`, `assignment_diagnostic()`, `planning_baseline(metric)`, `breakout_summaries(...)`, `factor_summaries()`, `capture_sequential(...)`, `sequential_snapshot(...)` | `workflows.<name>(experiment, reader, ...)` with the same remaining parameters |
| `analysis.artifact_context` | `artifacts.artifact_context(experiment, reader)` |
| `analysis.publish_unit_day_artifact(store, ...)` | `artifacts.publish_unit_day_artifact(experiment, reader, store, ...)` |
| `analysis.build_panel_for_metric(...)`, `panel_sql(...)`, `summary_sql(...)` | `query.<name>(experiment, reader, ...)` |
| `analysis.estimate_cate(...)`, `validate_cate(...)`, `targeting_rule(...)`, `select_targeting_rule(...)`, `sitewide(...)` | `estimators.<name>(experiment, reader, ...)` |
| `analysis.dashboard_snapshot(...)`, `dashboard_breakout_reads(...)`, `dashboard_group_data(...)` | `dashboard.prepare_dashboard(experiment, reader, config=...)` captures the headline, group data and Explore reads once; `load_explore(experiment, snapshot=...)` and `render_dashboard(experiment, snapshot=...)` read the snapshot |
| `increment.analysis.fit_predeclared_adjustment` | `estimators.fit_predeclared_adjustment` |
| `increment.Resolver`, `increment.BoundResolver` | `increment.evidence.Resolver`, `increment.evidence.BoundResolver`, under the same names |
| `resolver=` on `run` and on every function that reads data | `reader=`; positional calls are unchanged |
| `MetricSpec(missing="drop")` | `MetricSpec(missing="skip")`, the same computation; the rows' `population` names the units each group observed |
| `increment.Scope` | `increment.View` |
| `Scope.run(population=...)` | `View.overall(population=...)` |
| `experiment.scopes`, `Experiment(..., scopes=...)` | `experiment.views`, `Experiment(..., views=...)`; saved JSON keeps its `"scopes"` key |
| `experiment.with_views(*scopes)`, which replaced the declared views | `experiment.with_views(*views)`, which adds after the declared views: omit a view already declared, such as the builder's `View.overall()`, and pass `views=` to `run` to execute only some of them |
| `result.readout(scope=...)` | `result.readout(view=...)`; positional calls are unchanged |

Readout options such as `decision_method`, `prior`, `metrics`, `estimands`,
`exploratory_metrics` and `coherence` are `run` keywords and apply to every
selected view. A view's population, breakout selection and completed-window
gate belong to its `View`. One `run` reads every declared view from one
capture, so the daily, as-of and whole-window rows agree on the data they
describe:

```python
import polars as pl

import increment
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
experiment, reader = increment.from_frame(
    panel,
    shape="panel",
    unit="user_id",
    group="variant",
    date="day",
    control="control",
    metrics={"revenue": "mean"},
    breakouts=["country"],
)
whole = increment.View.overall()
daily_by_country = increment.View.daily(values=True, segment=(BreakoutIdentity("country", None),))
asof_lift = increment.View.asof()
result = increment.run(experiment.with_views(daily_by_country, asof_lift), reader)
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
legacy artifacts from trusted definitions. The [saved-format table](reference/saved-formats.md)
lists the version each writer produces and the versions each reader accepts. The
details below document supported moments, artifact, and checkpoint formats.

### Portable result cells

Decoders accept saved metadata cells in either the previous source-snapshot-first
order or the current population-first order, and normalize reopened metadata to
the current order.

Serialize the collection and reopen it with its dedicated decoder:

```python
import polars as pl

import increment
from increment.results import ReadoutResults

frame = pl.DataFrame(
    {
        "unit": ["c0", "c1", "t0", "t1"],
        "arm": ["control", "control", "treatment", "treatment"],
        "revenue": [1.0, 2.0, 1.5, 2.5],
    }
)
experiment, reader = increment.from_frame(
    frame,
    shape="summary",
    unit="unit",
    group="arm",
    design=increment.Randomized(
        control_group="control",
        allocation={"control": 0.5, "treatment": 0.5},
        allocation_scheme="independent",
    ),
    metrics=[increment.MetricSpec(name="revenue", type="mean")],
)
rows = increment.run(experiment, reader).readout()
payload = rows.model_dump_json()
restored = ReadoutResults.model_validate_json(payload)
```

This preserves the rows and collection-level metadata. A row or collection JSON
payload is distinct from a portable `ResultCell` payload; use the envelope
format below when you need per-cell operation metadata, summaries, or Arrow
interchange.

For portable cells, use `ResultCell` and `AnalysisEnvelope` from
`increment.results`. `ResultCell.to_payload()` / `from_payload()` read
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

import increment
from increment.results import AnalysisEnvelope, ResultCell
from increment.tables import readout_table

cells: tuple[ResultCell, ...] = tuple(rows.as_cells())
envelope = AnalysisEnvelope(cells, rows.metadata)
saved = json.dumps(envelope.to_payload())
reopened = AnalysisEnvelope.from_payload(json.loads(saved))
table = readout_table(reopened)
brief = increment.render(reopened)
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
    **Moments cubes.** `increment.workflows.export_moments(experiment, reader, path)` stamps each row with `moments_format`;
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

`increment.workflows.unit_counts(experiment, reader)` returns enrolled counts
by canonical group label, including excluded units as `"(unassigned)"`. The
source must retain total-grain counts; day-axis views refuse.

`increment.artifacts.artifact_context(experiment, reader)` returns the trusted
context for a definitions-backed experiment. Pass it as `expected_context` when
opening an artifact published from that experiment.

`increment.workflows.allocation_history(experiment, reader, population=...)`
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

`increment.workflows.available_metrics(experiment, reader)` lists undeclared
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

::: increment.estimators.MixturePrior

::: increment.estimators.StudentTPrior

::: increment.estimators.moments_from_summaries

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

These models and tagged unions are importable from `increment.artifacts`.
They describe the published format without exposing source-loading machinery.

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

::: increment.artifacts.WarehouseArtifactStore

::: increment.artifacts.artifact_context

::: increment.artifacts.publish_unit_day_artifact

::: increment.artifacts.ArtifactExtensionCatalogEntry

::: increment.artifacts.unit_day_artifact_extension_catalog

### Digests and hashing

Canonical encodings and SHA-256 helpers keep artifact identities deterministic
across publication and requests.

::: increment.digest.ArtifactDigestError

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

::: increment.results.ContrastResult

::: increment.results.ContrastResults

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

For a dataframe registration, `synthesise_metric` derives a metric from its
`MetricSpec`, and `frame_observation_mapping` records the physical column roles
passed to `sequential_definition_id`. Declare this binding before reading outcomes.

::: increment.estimators.synthesise_metric

::: increment.estimators.frame_observation_mapping

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

::: increment.power.ArmPlanningProcedure

::: increment.power.AnalysisAxes

::: increment.power.FamilyPolicy

::: increment.power.MetricCapabilities

::: increment.power.PlanningFamilyExpansion

::: increment.power.RelativeDecisionPolicy

::: increment.estimators.FixedInference

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

`increment.workflows.health_check(experiment, reader)` combines assignment
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

import increment
from increment.workflows import health_check

units = pd.DataFrame(
    {
        "unit": ["c1", "c2", "t1", "t2"],
        "arm": ["control", "control", "treatment", "treatment"],
        "value": [1.0, 0.0, 1.5, 1.0],
    }
)
experiment, reader = increment.from_frame(
    units,
    shape="summary",
    unit="unit",
    group="arm",
    control="control",
    metrics={"value": "mean"},
)
report = health_check(experiment, reader)
for check in report.checks:
    print(check.check, check.analysis_population, check.availability, check.conclusion)
```

::: increment.workflows.health_check

## Conditional effects and targeting

`estimate_cate` estimates how treatment effects vary with user characteristics.
Validation and targeting refit through honest or nested splits; their reports
give evidence for those procedures, not for a separate `estimate_cate` fit.
Each operation takes the `(experiment, reader)` pair from a builder, reads
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

::: increment.estimators.select_exploratory_family

`exploratory_family_exclusion` applies the same admissibility test to one row and
returns the reason it cannot join the family, or `None`. Exclude ineligible rows
and report the returned reason instead of refusing the whole family; refusal
context uses the same values in `reasons` or `constructions`.
One-sided (`greater`/`less`) Wald, exact binomial and Fieller rows are reissued
exactly, with their one-sided geometry, and are not excluded.

::: increment.estimators.exploratory_family_exclusion

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

`CodedModel` and `unwrap_coded` are public from `increment.errors`, not the
root `increment` namespace.

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
import increment
from increment.errors import unwrap_coded

try:
    TypeAdapter(increment.Definitions).validate_python({"day_boundary": "EST"})
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

## Results (`increment.results`)

Sitewide behavior is exposed through
`increment.estimators.sitewide(experiment, reader, metric=...)`, with output
values available from the receive-only results namespace.

`sitewide(experiment, reader, metric, population="assigned")` reports an assigned
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

::: increment.results.BinomialConfidenceSet

::: increment.results.WinsorConfidenceSet

::: increment.results.WinsorInferenceSpec

::: increment.results.BootstrapReference

::: increment.results.InfluenceReference

::: increment.results.RankReference

::: increment.results.WinsorPermutationTest

::: increment.results.IndependentMeanComponent

::: increment.results.IndependentMeanReference

::: increment.results.SetEndpoint

::: increment.results.SetInterval

::: increment.results.SwitchbackAssignmentDiagnostic

::: increment.results.SwitchbackPowerResult

::: increment.results.UnitCycleFailureCount

::: increment.results.UnitCycleLawMoments

::: increment.results.UnitCyclePowerLowerBoundResult

::: increment.results.UnitCycleModelPowerResult

::: increment.results.UnitCycleModelMdeResult

::: increment.results.UnitCycleModelRequiredUnitsResult

::: increment.results.AllocationBalance

::: increment.results.PolicyValueContrast

::: increment.results.AnalysisEnvelope

::: increment.results.ResultCell

::: increment.results.ReadoutResults

Binomial rows persist `binomial_set.numerical_qualification`:
`scipy_special_function_error_model_conditional_v1` states the finite-sample
arithmetic qualification conditional on the deployed SciPy/Boost special-function
error model, not a cross-build proof. Supported rows saved before this field was
introduced load as `legacy_unrecorded_v1`, which makes no arithmetic qualification
claim.

::: increment.results.JointContrastReference

::: increment.results.RelativeConfidenceSet

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

::: increment.evidence.ComplianceArm

::: increment.evidence.ComplianceSummary

## Custom evidence providers (`increment.evidence`)

Implement the primitive requests in the [provider guide](development/adding-an-estimator.md#custom-evidence-providers).
Capabilities are declared before reads; compound readout acquisition remains internal.

::: increment.evidence.BreakoutCell

::: increment.evidence.MomentRows

::: increment.evidence.ScalarEvidenceDeclaration

::: increment.evidence.EvidenceExecutionContext

## Saved runs (`increment.store`)

Use `ParquetResultsStore` for the
[save-and-reopen journey](guides/experiment-specifications.md), or
`WarehouseResultsStore` for warehouse-backed runs.

::: increment.store.ParquetResultsStore

::: increment.store.WarehouseResultsStore

::: increment.store.ResultsStore

::: increment.store.Run

::: increment.store.RunHandle

::: increment.store.HistorySelector

::: increment.store.DecisionLog

::: increment.store.DecisionEvent

::: increment.store.DecisionReference

## Tables (`increment.tables`)

::: increment.tables.readout_table

::: increment.tables.estimates_to_readout

::: increment.tables.contrast_results_to_readout

::: increment.tables.trend_table

## Synthetic data (`increment.simulate`)

Generate example data or a logged policy trace without a warehouse.
These generators do not certify an analysis method's assumptions.

::: increment.simulate.Scenario

::: increment.simulate.simulate_raw_logs

::: increment.simulate.SwitchbackScenario

::: increment.simulate.simulate_switchback_panel

::: increment.simulate.simulate_logged_run

::: increment.simulate.simulate_logged_trace

::: increment.simulate.LoggedRun

::: increment.simulate.RewardModel

::: increment.simulate.true_policy_value

::: increment.simulate.run_end_to_end

::: increment.simulate.EvalResult

::: increment.simulate.run_switchback_end_to_end

::: increment.simulate.SwitchbackEvalResult

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
