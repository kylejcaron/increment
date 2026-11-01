# Extending an estimator or its evidence

This guide follows two completed changes: refining exact conversion-interval endpoints and adding experiment-health diagnostics. These are repository examples, not a universal extension protocol. For new methods or designs, follow the [Adding a method or design checklist](https://github.com/kylejcaron/increment/blob/main/CONTRIBUTING.md#adding-a-method-or-design); it does not block fixes to shipped behavior.

## First locate the owner and the boundary

For randomized lift, `increment/declarations.py` owns the lightweight `Method`, variance-reduction ids, and static estimator declarations; `increment/estimation/engine.py` owns `estimate_lift` dispatch. Observational estimator identities and lazy implementation lookup also live in `increment/declarations.py` and `increment/estimators/registry.py`; `increment/estimation/adjust.py` owns observational `estimate_ate` orchestration. Built-in implementations are declared by module path rather than eagerly imported. Keep these boundaries separate: don't add observational adjustments to randomized lift or warehouse queries to the estimator. Sources and reductions obtain evidence; estimators consume `ArmStats` or the per-unit data their construction requires.

Analysis CATE routes and registered run persistence use narrow contracts in `increment.kernel.ports` and resolve installed operation providers lazily from distribution entry-point metadata. Keep estimator and store implementations in their specialist namespaces; do not import them from core orchestration or add another runtime registry.
Missing bundled provider declarations refuse with
`package.builtin_provider_unavailable`; reinstall Increment to restore them.

`Method.name` is a selector for observational estimator refs and a result label; randomized lift uses `variance_reduction` to choose its estimator id. Observational plugins can use `ADJUSTMENTS.register(name, fn)` as the compatibility registration API, or register an `EstimatorRef` and its implementation directly. Do not infer a method from display text or add aliases.

`increment/_moment_plan.py` distinguishes counts, references, centered first moments, and centered second/cross moments. These cannot replace unit records or reconstruct membership and arbitrary distributions. Evidence is distinct too: reasons, guarantees, diagnostics, and source availability answer different questions from numeric moments. Preserve these distinctions through source, estimator, result, serialization, and presentation.

Artifact query contracts live in `increment.query.artifact_ports`; core loaders
call those contracts rather than artifact implementations. The installed
distribution declares the `increment.artifacts._query` adapter through its
`increment.artifact_ports` entry point, loaded only when an artifact operation
is requested. Keep this composition out of root imports, and preserve saved
source identities when changing runtime implementation classes.
Missing bundled provider declarations refuse with
`package.builtin_provider_unavailable`; reinstall Increment to restore them.

## Custom evidence providers

For an `Analysis` backed by custom data, implement `EvidenceProvider` from
`increment.evidence`, declare `evidence: EvidenceCapabilities`, and preserve
analysis metadata in `context: SourceContext`. Its required primitive methods
are `moments(MomentsRequest)`, `breakout_method_moments(BreakoutMethodMomentsRequest)`,
`unit_rows(UnitRowsRequest)`, `method_fit(MethodFitRequest)`,
`compliance(ComplianceRequest)`, `compliance_series(ComplianceSeriesRequest)`,
`counts(CountsRequest)`, `capture_sequential(SequentialCaptureRequest)`, and
`scope(EvidenceScope)`. Route requests through `request_evidence(provider,
request)` so shared capability checks run before a read. The
[runnable custom-provider example](https://github.com/kylejcaron/increment/blob/main/examples/custom_evidence_provider.py)
shows the provider interface. The removed `increment.sources.MomentSource`
custom-source protocol has no compatibility adapter.

When `evidence.assignment_relation` is `"ledger"` or `"reduced"`, implement
`assignment(request: AssignmentRequest) -> AssignmentEvidence` to serve
`AssignmentRequest`s. It returns reduced assignment metadata (arm counts,
randomization grain, and any declared probabilities), not ledger rows or event
timestamps. For example:

```python
from increment.evidence import AssignmentEvidence, AssignmentRequest


class Provider:
    def assignment(self, request: AssignmentRequest) -> AssignmentEvidence:
        return AssignmentEvidence(
            experiment=request.experiment,
            population=request.population,
            versions=(1,),
            periods=(None,),
            arm_counts=((1, None, "control", 4), (1, None, "treatment", 4)),
            independence_grain="unit",
        )
```

Providers without assignment evidence do not need this optional method. A
provider that declares assignment evidence but omits it refuses the request
with `evidence.assignment.unavailable`.

The optional `sequential_snapshot(SequentialSnapshotRequest(previous=...))`
returns the provider's exact current checkpoint linked to the supplied
checkpoint. Without this method, acquiring a current checkpoint refuses with
`sequential.continuation.legacy`.

`scope(EvidenceScope)` must keep one data version for nested scopes entered
inside a snapshot scope. This lets all reads in one readout use the same view
of the data. Breakout slices open an individual scope for each breakout
identity, nested inside one snapshot scope; each slice's reads must stay bound
to its identity and the same snapshot. Breakout rows name the source the
experiment declares for each breakout; when the experiment leaves a breakout's
source unnamed, a provider's rows leave it unnamed too.

Triggered daily and as-of readouts need two separate kinds of evidence:
trigger observation edges for each metric, and the experiment reconstruction
metadata used to build the complete day axis (including zero-population early
look cells). The provider must expose that metadata as `experiment` or
`artifact_experiment`, or pass it as `DayAxisRequest.experiment` when serving a
request. Supply the complete experiment metadata, including `trigger`, `unit`,
`exposure`, `start`, `control_group`, plan, and breakouts for requested
dimensions. The provider must also implement `triggered_observation_edges(metric)`
and declare the triggered population. Missing reconstruction metadata or trigger
edges refuses before reading.

For an existing `EvidenceProvider`, the metadata fields and edge method can be
provided like this:

```python
from datetime import date, datetime
from types import SimpleNamespace

from increment import AnalysisPlan
from increment.semantics.models import Experiment

provider = SimpleNamespace()
provider.experiment = Experiment(
    name="trial",
    exposure="assigned",
    trigger="seen",
    unit="user_id",
    start=datetime(2025, 1, 1),
    control_group="control",
    plan=AnalysisPlan(),
)
provider.triggered_observation_edges = lambda metric: (
    date(2025, 1, 1),
    date(2025, 1, 7),
)
```

The [runnable custom-provider example](https://github.com/kylejcaron/increment/blob/main/examples/custom_evidence_provider.py)
shows the provider interface; add the trigger metadata and edge method when
serving triggered day-axis requests. A source that neither declares
`EvidenceCapabilities` nor implements `EvidenceProvider` refuses before
reading with `evidence.provider_unavailable`.

The same declared provider/acquisition boundary serves lower-level estimator
facades such as `estimate_ate()`. Its supplied source must declare
`EvidenceCapabilities` or implement `EvidenceProvider`; it does not need legacy
`moments(metric)` or `unit_frame(metric, ...)` methods.

| Former custom `MomentSource` member or source method | Implement instead | Request example |
| --- | --- | --- |
| `increment.sources.MomentSource` | `increment.evidence.EvidenceProvider` | — |
| `capabilities` | Declare `EvidenceCapabilities.grains` | `grains=frozenset({"total"})` |
| `breakouts` | Declare `EvidenceCapabilities.breakouts` | `breakouts=("country",)` |
| `shape` | Declare the matching `EvidenceCapabilities.kind` | `kind="unit_summary"` |
| `moments_for_method(metric, method, by=...)` and `categorical_evidence(metric, method)` | `moments(MomentsRequest)` for scalar method rows; `method_fit(MethodFitRequest)` for retained/warehouse method evidence | `MomentsRequest(metric, method=method)`; `MethodFitRequest(metric, method, control="control", arm_order=("control", "treatment"), population="assigned")` |
| `breakout_source(...)` / `breakout_sources(...)` | `scope(EvidenceScope)` | `EvidenceScope(metrics=(metric,), breakouts=(BreakoutIdentity("country", None),), include_breakouts=True)` |
| `moments(metric, grain=..., by=..., completed_windows_only=..., include_covariate=...)` | `moments(MomentsRequest)` | `MomentsRequest(metric, grain="total", by=("country",))` |
| `breakout_moments_for_method(metric, method, by=...)` | `breakout_method_moments(BreakoutMethodMomentsRequest)` | `BreakoutMethodMomentsRequest(metric, method, BreakoutIdentity("country", None))` |
| `unit_frame(metric, covariates=...)` | `unit_rows(UnitRowsRequest)` | `UnitRowsRequest(metric, covariates=("tenure",))` |
| `unit_counts()` | `counts(CountsRequest)` | `CountsRequest(kind="unit")` |
| `cluster_counts()` | `counts(CountsRequest)` | `CountsRequest(kind="cluster")` |
| `assignment_counts()` | `counts(CountsRequest)` | `CountsRequest(kind="assignment")` |
| `triggered_counts()` | `counts(CountsRequest)` | `CountsRequest(kind="triggered", population="triggered")` |
| `compliance_summary(design, as_of=..., completed_windows_only=...)` | `compliance(ComplianceRequest)` | `ComplianceRequest(design, as_of=day)` |
| `compliance_dates()` | `compliance_series(ComplianceSeriesRequest)` | `ComplianceSeriesRequest(design, completed_windows_only=True)` |
| `readout_snapshot(...)` / `readout_execution(...)` | `scope(EvidenceScope)` | `EvidenceScope(metrics=(metric,), snapshot=True)` / `EvidenceScope(metrics=(metric,), execution=True)` |
| `day_source(...)` | `scope(EvidenceScope)` with a day grain | `EvidenceScope(metrics=(metric,), grain="daily")` |
| `triggered_source(...)` | `scope(EvidenceScope)` with the triggered population | `EvidenceScope(metrics=(metric,), population="triggered")` |
| `capture_sequential(...)` | `capture_sequential(SequentialCaptureRequest)` | `SequentialCaptureRequest(finalized=True, as_of=day)` |
| `sequential_snapshot(previous=...)` | Optionally implement `sequential_snapshot(SequentialSnapshotRequest)` | `SequentialSnapshotRequest(previous=checkpoint)` |


Fields such as `context` remain provider metadata; evidence fields such as
`operations`, SQL access, and `close()` are not replacements for evidence
requests and stay on their separately documented lifecycle or product-operation
interfaces. Do not claim a capability you cannot answer. If a requested grain,
dimension, row set, or operation is not available, let the shared request
validation raise its coded refusal rather than returning an empty or fabricated
result. A request to unsupported compliance-series evidence uses the same
shared refusal path.



`ADJUSTMENTS` callbacks receive one contract across built-ins and extensions:
`(inputs: AdjustmentInputs, metric: Metric, design: Observational, **kwargs)`.
`inputs.frame` is the unit-level data for the metric and adjustment covariates,
`inputs.moment_rows` are the metric's reduced arm summaries, and
`inputs.cluster` is the declared cluster column or `None`. Every callback must
accept the shared keyword arguments `prior`, `alpha`, `alternative`,
`value_scale`, `null_lift`, `null_abs`, and `preferred_direction`;
method-specific keywords are supplied when applicable. Registration validates
that signature and refuses incompatible callbacks with
`estimation.estimator.adjustment.callback_contract`. Migrate legacy callbacks
from `(frame, moment_rows, cluster, metric, design, **kwargs)` by accepting
`inputs` and reading those three fields from it. Keep kernels data-only; they
must not reach back into a source.

Every estimator ref carries alternative evidence routes as grouped request specs in `EstimatorRef.needs`. Observational registrations must declare at least one route containing total moments and design-covariate unit rows; the registry converts those specs to the typed evidence requests and checks source capabilities before a run. Extra requests gate runnability, but dispatch does not yet deliver them to callbacks. Add a ref and implementation registration in one plugin module; built-in entries are maintained in `increment/declarations.py`.

`increment/estimators/manifest.json` is generated from the built-in refs. Update it with `uv run --no-sync python scripts/render_estimator_manifest.py`; check it in CI with the same command plus `--check`. The affected-test selector reads the manifest to retain the required tests when an estimator implementation module changes.

## Set up before running the focused commands

Install the test dependencies used below:

```bash
uv sync --group dev --extra demo --extra tables --extra dashboard
```

`make install` also synchronizes dependencies and installs repository hooks.
Check shared-hook ownership first, as described in CONTRIBUTING. Hooks are not
needed to run the focused tests.

## Walkthrough 1: refine exact-count binomial endpoints

**Historical change:** `0f272c4` (`Resolve exact risk-ratio endpoints to a relative log width (#40)`), compared with parent `0f272c4^`. This refined an existing exact-count estimator. The route already had exact successes and trial counts; the change improved endpoint search without adding evidence or changing the statistic.

### Contract and assumptions

The route computes an exact independent-binomial risk-ratio confidence set for eligible conversion contrasts. It uses exact successes and trial counts from both arms and canonicalizes Bernoulli arms. These counts differ from ordinary centered mean/ratio moments. A new sufficient statistic needs coordinated source, reduction, and wire changes, plus an independent construction and an independent data check. This construction does not extend to clustered units, sequential monitoring, or unsupported adjustments. `increment/estimation/binomial_rr.py` owns the construction and its eligibility/refusals.

The change made endpoint search relative in log-risk-ratio space instead of using a fixed absolute step. That gives finer resolution near zero. Preserve outward containment and finite or unbounded endpoint representation; never hide the resolution limit by rounding inward.

### Trace the implementation and independent check

1. Trace `increment/estimation/engine.py::_contrast_counts` from `estimate_lift` through `_lift_for_method` to `_infer_binomial_lift_result`. It builds the contrast counts and returns the result or typed failure. `binomial_rr.confidence_interval` owns the construction and its endpoint qualifications.
2. Read `tests/estimation/test_binomial_rr.py::TestEndpointContainment::test_endpoints_enclose_the_finer_reference` and `tests/estimation/_binomial_endpoint_reference.py`. The reference uses a separate, finer doubling-and-bisection search. Decimal polynomial and enumeration checks independently test binomial tails; production endpoint bits are not an oracle.
3. Check fixed-horizon conversion cases in `tests/parity_harness/cases.py`, such as `_fixed_horizon_case` and `mean_ratio_conversion_cuped_fixed_horizon`. Unsupported inputs must retain their explicit refusal, not invent counts from rounded moments. See the six entry-point dispositions in the [capability table](../reference/capabilities-by-entry-point.md).
4. `tests/estimation/test_engine.py::TestBinomialDirectionalAlphaConvention::test_a_produced_row_round_trips_from_a_dump_without_defaults` checks that results and metadata survive serialization and reinversion. Check it alongside parity and result assertions, not only the point estimate.

Focused commands for this slice:

```bash
make test TESTS=tests/estimation/test_binomial_rr.py::TestEndpointContainment::test_endpoints_enclose_the_finer_reference
make test TESTS=tests/estimation/test_engine.py
make test TESTS=tests/test_parity_harness.py
```

`test_binomial_rr.py` checks the numerical route against an independent reference; the named `test_engine.py` case checks result round-tripping; the parity harness checks source support. Use `make test-affected BASE=0f272c4^` when changes span more owners. Do not start a calibration campaign for a documentation edit; calibration obligations for a numerical change remain those recorded by the owners.

## Walkthrough 2: add experiment-health evidence

**Historical changes:** `baf3d7a` introduced `Analysis.health_check()` and its composed report; `b32338a` added source and snapshot evidence, compared with `b32338a^`. The feature combines assignment-integrity, trigger-rate, and retained source-maturity evidence. It does not add an estimator or determine causal validity.

### Contract, assumptions and source requirements

`Analysis.health_check()` returns a frozen `ExperimentHealth` report (`increment/estimation/health.py`) with ordered `HealthCheck` entries. Availability is not a conclusion: missing trigger membership or feed-completeness evidence remains unavailable with its exact code and context, not a failure or pass. Assigned allocation integrity and triggered-population balance are separate.

!!! warning "Balanced triggers do not establish causal validity"
    Balanced trigger rates do not establish treatment-independent membership,
    stationarity, randomization integrity or causal validity; report the
    assumption risk rather than converting the diagnostic into a guarantee.

The report is available from all public analysis factories, but trigger diagnostics need specific evidence:

- `Analysis.from_definitions` needs retained native trigger and source-snapshot evidence.
- `Analysis.from_unit_day_artifact` needs `trigger_population`, `assignment_counts`, and `trigger_measure_stats`; without them, trigger checks are unavailable.
- `Analysis.from_unit_summary`, `from_unit_panel`, `from_switchback_panel`, and `from_moments` do not retain trigger membership. They report the exact unavailable or not-applicable status; they do not infer trigger diagnostics.

`tests/capability_entry_point_catalog.py` and `tests/capability_consumers_auxiliary.py` record dispositions across six entry points. `increment/_source_diagnostics.py` owns the source-bound diagnostics and `increment/_health.py` composes their evidence; `increment/estimation/health.py` defines reports; `increment/dashboard/_data.py` and `increment/dashboard/_html.py` render them.

This evidence is not a moment: aggregates cannot restore unit membership or feed watermarks. Nor is it a substitute for unit records; source identity and timing evidence must remain attached to the check that uses them.

### Trace dispatch, independent checks and consumers

1. Trace `increment.workflows.health_check()` to `check_health()` in `increment/_health.py`. It pins one source snapshot, combines the allocation, trigger-rate, maturity, feed and assumption checks, retains source identity and cutoff, and builds the report. Dashboard capture reuses the same collection inside its own pinned snapshot rather than duplicating its policy.
2. `tests/test_experiment_health.py::test_balanced_trigger_rates_do_not_clear_assumption_risk` calculates each arm's expected trigger share directly from the fixture's raw `enrolled` and `saw_surface` user IDs, then compares that independent count to the report. The same test confirms that balanced rates leave assumption-risk fields marked as assumed. `test_health_missing_certificates_are_unknown` checks the refusal and missing watermark.
3. Sentinel tests cover unavailable and not-applicable paths: `test_untriggered_summary_has_explicit_unavailable_triggered_rows`, `test_health_unit_panel_reports_only_supported_population`, `test_health_switchback_does_not_substitute_unit_arm_srm`, `test_health_moments_replay_does_not_reconstruct_trigger_population`, and `test_health_artifact_without_trigger_extensions_is_unavailable`. `test_health_artifact_trigger_extensions_are_optional_but_explicit` compares retained native/artifact evidence.
4. Preserve identity and provenance through frozen models, JSON/pickle/deepcopy round trips, and dashboard projection. `test_health_json_replay_preserves_checks_and_risks` and `test_health_models_round_trip_pickle_deepcopy_and_nested_immutability` check restoration; dashboard tests check rendering. Serialization restores a report; it cannot make an unavailable check available.
5. Check the capability row and [entry-point reference](../reference/capabilities-by-entry-point.md) before changing support. Pair each newly supported input path with evidence tests; a constructor alone is not parity evidence.

Focused commands for this slice:

```bash
make test TESTS=tests/test_experiment_health.py
make test TESTS=tests/test_dashboard.py
make test TESTS=tests/test_dashboard_overview.py
```

## Extraction decision

No extraction is proposed. The apparent overlap does not demonstrate a shared policy that can move without changing behavior: moment planning/reduction, estimator dispatch, multiplicity, and result presentation own separate contracts. Moving modules or adding abstraction for navigation or file size alone would be cosmetic.

For this historical walkthrough, inspect the relevant implementation and sentinel tests, then run the focused commands above. For documentation changes, run the strict docs session and reference checker after adding this page to navigation. New methods must follow the contributor checklist; these examples do not replace it.
