# API Reference

Use `increment` for workflow entry points and caller-authored configuration.
Receive-only result types are in `increment.results`; specialized power solvers
are in `increment.power`.

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
  `breaking`) with the affected paths and a migration route. There is no
  1.0 stability promise or deprecation window.
- A bug fix may change a previously incorrect numerical answer. Release notes
  describe the correction; compatibility does not preserve wrong values.
- Compatibility policy is the same across methods; guarantees and evidence
  differ. See [Evidence status](validation.md#evidence-status).

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

!!! warning "A checkpoint belongs to one monitoring process"
    Do not restart monitoring and spend alpha again: retain the old checkpoint
    or register a defensible new protocol.

??? info "Statistical and wire-format details"
    **Moments cubes.** `Analysis.export` stamps each row with `moments_format`;
    `Analysis.from_moments` reads it. Formats vary by writer:

    | Format | Written by | Read by `from_moments` |
    |---|---|---|
    | 12 | Fixed-horizon `export`; source identity, exact counts, ratio denominator third moment `cden3` | Yes |
    | 11 | Prior fixed-horizon export; source identity and exact counts | Yes; ratio denominator-skew coverage is unavailable |
    | 10 | Earlier fixed-horizon export or current sequential checkpoint export | Yes; fixed-horizon ratio denominator-skew coverage is unavailable |
    | 9 | Prior sequential checkpoint export | Yes |
    | Below 9 | No longer written | Refused: `moments.format.unsupported_legacy` |
    | Above 12 | Not written by this release | Refused: `moments.format.unsupported_future` |

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

    Fixed formats 11 and 12 and sequential format 10 carry source identity.
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
    and pass its rows to `Analysis.from_moments`. It accepts strict mean rows
    (`group_id`, `metric`, `n >= 2`, `mean`, ddof=1 `var`) and conversion rows
    (`group_id`, `metric`, `n`, exact integer `successes`). Extra fields are
    refused with `moments.summaries.invalid`. Without `assignment_counts`,
    `unit_counts()`, `srm()`, and health allocation checks refuse with
    `source.moments.assignment_counts`. For ratios and richer summaries, use
    `from_unit_summary`.

    **Unit-day artifacts.** Artifact context format 2 is the only supported
    format. It contains no source SQL; source recipes use domain-tagged SHA-256
    digests. Opening or publishing format 1 is refused with
    `artifact.format.unsupported`, naming the received and supported versions.
    Republish from trusted definitions; existing format-1 manifests still
    contain their original SQL. See [Canonical unit-day artifacts](guides/data-model.md#canonical-unit-day-artifacts).

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
    and the `roles=` argument of `increment.impute.pooled_mean`. The abandon
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
from increment import TabularPolicy

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

## Start here

::: increment.Analysis
    options:
      inherited_members: true

::: increment.Report

::: increment.Definitions

`increment.__version__` reports the installed package version.

## Analysis and reporting

`Analysis.unit_counts()` returns enrolled counts by canonical group label,
including excluded units as `"(unassigned)"`. The source must retain total-grain
counts; day-axis views refuse.

`Analysis.artifact_context` returns the trusted context for a
definitions-backed analysis. Pass it as `expected_context` when opening an
artifact published from that analysis.

`Analysis.allocation_history(population=...)` returns a PyArrow table with
`experiment_id`, `analysis_population`, `ds`, `group_id`, `n_daily`, and
`n_cumulative`, ordered by population, date, and arm. Assigned cohorts start on
their first assignment; triggered cohorts start at their first eligible trigger.
Both require a native definitions-backed analysis. Non-native, artifact, and
clustered sources refuse with `CapabilityError`.

Day-axis results from `run_daily`, `run_daily_lift`, `run_asof`, and
`run_asof_lift` sort by day, requested metric, and stable row identity—not
backend row order.

`Analysis.available_metrics` lists undeclared saved per-unit metrics on the
experiment's unit, in definitions order. It excludes report-only `total` and
`active` metrics and metrics for other entities. `run`, `run_breakout`,
`run_asof_lift`, `run_asof`, and `run_daily` accept selected names through
`exploratory_metrics=`. Lift rows from the first three carry
`role="exploratory"`, join no plan family, and use two-sided full-plan `alpha`;
absolute values from `run_asof` and `run_daily` have no role. Declared rows are
unchanged. Use `metrics=[]` to read only exploratory metrics. Unknown or
declared names refuse before querying with
`facade.analysis_config.exploratory_metric_unavailable`. Other sources refuse
with `facade.analysis.exploratory_metrics_source_limited`; registered
sequential plans refuse with `facade.analysis.exploratory_metrics_sequential`.
Pass added metrics in `metrics=` to pin them in `Analysis.dashboard_snapshot`.

::: increment.Metric

::: increment.Method

::: increment.Normal

## Dataframe entry points

::: increment.MetricSpec


::: increment.to_frame

::: increment.impute

## Designs

::: increment.Design

::: increment.Randomized

::: increment.Observational

::: increment.Encouragement

::: increment.UptakeSpec

::: increment.ExclusionRestriction

::: increment.AdjustmentSet

## Assignment and study contracts

::: increment.Assignment

::: increment.ParallelAssignment

::: increment.IndependentBernoulliOrder

::: increment.SharedScheduleOrder

::: increment.SwitchbackWindow

::: increment.SwitchbackAssignment

::: increment.ParallelStudyEnvelope

::: increment.SwitchbackStudyEnvelope

::: increment.StudyEnvelope

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

::: increment.RelationLocator

::: increment.RelationRole

::: increment.BaseRelations

::: increment.Freshness

::: increment.SimpleMetricMeasure

::: increment.RatioMetricMeasure

::: increment.MetricMeasure

### Artifact extensions

These references describe artifact dimensions, assignment, trigger membership
and outcomes, uptake, and site-volume extensions.

### Artifact requests

Requests validate extension definitions before publication.

::: increment.ExtensionRequestBase

::: increment.BreakoutDimensionRequest

::: increment.FactorDimensionRequest

::: increment.ClusterIdentityRequest

::: increment.CupedPreperiodRequest

::: increment.AssignmentCountsRequest

::: increment.TriggerPopulationRequest
::: increment.TriggerMeasureStatsRequest

::: increment.EncouragementUptakeRequest

::: increment.SiteVolumeRequest

::: increment.UnitCovariateRequest

::: increment.UnitCovariateLevelRequest

### Artifact publication and storage

`Analysis.from_definitions(..., source_snapshot_evidence=...)` accepts
`increment.SourceSnapshotEvidence` only when the source provides a timezone-aware
event-time cutoff and, optionally, certified per-feed completeness times.
Without a cutoff, Increment does not invent one: assigned-only operations remain
available, but triggered membership, triggered outcomes, and trigger-evidence
publication refuse with `source.native.trigger_evidence_required`. Missing feed
certification stays unknown; maximum observed timestamps, publication time, and
experiment end do not prove completeness. Triggered artifact outcome requests
select one `trigger_measure_stats` item per `(trigger, metric)`, bound to the
version-3 membership digest and the same pinned cutoff.

::: increment.SourceSnapshotEvidence


Publication connects validated context to immutable snapshots in the
authorized artifact store.

::: increment.ArtifactPublication

::: increment.ArtifactSnapshot

::: increment.ArtifactStore

::: increment.compile_unit_day_artifact_context

::: increment.unit_day_artifact_extension_catalog

### Digests and hashing

Canonical encodings and SHA-256 helpers keep artifact identities deterministic
across publication and requests.

::: increment.query.artifact_digest.ArtifactDigestError

::: increment.DigestType

::: increment.FieldSpec

::: increment.RelationDigests

::: increment.canonical_schema_bytes

::: increment.canonical_row_bytes

::: increment.canonical_rows_bytes

::: increment.canonical_json_bytes

::: increment.schema_sha256

::: increment.content_sha256

::: increment.digest_relation

::: increment.manifest_sha256

::: increment.context_sha256

::: increment.request_sha256

::: increment.extension_definition_sha256

::: increment.extension_source_provenance_sha256

## Logged-policy evaluation

Off-policy contrast compares two registered policies using a decision trace
logged under one fixed policy version. It uses `LoggedTrace.from_records` or
`from_frame`, not an `Analysis` constructor. See the
[guide](guides/logged-policy.md) for the estimand, assumptions, and floors.
Each constructor requires exactly one logging-law route: `registry=`, complete
record-level `logging_distributions=`, or frame
`logging_distribution_column=`. The estimator and admission bounds are the
same for either route.


::: increment.estimate_policy_contrast

::: increment.LoggedTrace

::: increment.PolicyRegistry

::: increment.TabularPolicy

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

For continuous means, use `InferenceSpec(kind="asymptotic_mean")` with an
ordinary fixed-allocation design. The source binds registration before capture;
`expected_decision_sample_size` optionally sets pre-outcome tuning.
For conversion and retention, `InferenceSpec(kind="always_valid",
baseline_rate=...)` binds an exact Bernoulli e-process from the plan alone. See
[sequential conversion monitoring](guides/sequential-inference.md#sequential-monitoring-of-a-conversion-metric).
Definitions-based experiments specify arm weights in `Experiment.allocation`.
See [ordinary continuous monitoring](guides/sequential-inference.md#ordinary-continuous-monitoring)
for assumptions, finalization, and the qualified asymptotic guarantee.


::: increment.AlwaysValid

::: increment.AsymptoticMean

::: increment.ScalarMeanModel

::: increment.AsymptoticSequentialEvidence

::: increment.AsymptoticSequentialResult

::: increment.SequentialRegistration

::: increment.PredeclaredAdjustment

::: increment.fit_predeclared_adjustment

::: increment.SequentialModel

::: increment.SequentialCell

::: increment.SequentialCompliancePolicy

::: increment.SequentialSnapshot

::: increment.snapshot_from_json

::: increment.capture_sequential_snapshot

::: increment.declare_sequential_freeze

::: increment.estimate_sequential

::: increment.sequential_definition_id

::: increment.PredictivePrior

::: increment.JointReveal

## Power planning

Plan fixed-horizon sample size, power, or minimum detectable effect (MDE) with
the runtime's deployed decision method. Binomial and switchback planning have
different approximations and limits.

!!! warning "Use the lower bound to judge a power target"
    A sample size or effect reaches a target only when `power_lower` reaches
    it, even when `power` reports an enumerated point.

??? info "Statistical details"
    Fixed-horizon binomial planning estimates rejection probability over possible
    event counts. Enumerated results include an enclosure (`power_lower`,
    `power_upper`) conditional on the deployed SciPy/Boost special-function
    error model—not a cross-build floating-point proof. `power` is the
    enumerated point when the enclosure resolves to absolute error at most
    `1e-6` (`power_basis="exact"`) and the enclosure's lower end otherwise
    (`power_basis="lower_bound"`); a size or effect reaches a target only when
    the lower end does. The dense closed-form route gives a model-based point
    estimate without an enclosure.

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

::: increment.SummaryStats

::: increment.Baseline

::: increment.SwitchbackBaseline

::: increment.ProspectiveAssumptionProvenance

::: increment.UnitCycleReference

::: increment.UnitCycleTApproximation

::: increment.UnitCycleVarianceEnvelope

::: increment.PowerDesign

::: increment.required_sample_size

::: increment.achieved_power

::: increment.minimum_detectable_effect

::: increment.power_curve

::: increment.switchback_required_blocks_or_units

::: increment.switchback_achieved_power

::: increment.switchback_minimum_detectable_effect

::: increment.unit_cycle_power_lower_bound

## Diagnostics and robustness

::: increment.sample_ratio_mismatch

::: increment.allocation_posterior_bands

::: increment.absorb_factor

`Analysis.health_check()` combines assignment balance, triggered-cohort
balance, descriptive trigger rates, retained source-maturity/completeness
evidence, and declared trigger assumptions. Use it alongside estimates; it is
not a general safety verdict or causal-validity test. Missing source evidence
remains unavailable with its identity and reason. A non-rejected SRM does not
prove valid assignment.

For triggered experiments, inspect assigned and triggered checks separately.
Balanced rates do not remove the assumptions of treatment-independent
triggering, no pre-trigger treatment effect, and stable trigger order.
Sequential triggered analyses also assume `triggered_look_predictability`:
looks, stops, and freezes use only already-revealed triggered-chain data.
Assigned results must not reveal future triggered outcomes, including for the
same metric. The health check does not verify an analyst's decision rule.

```python
import pandas as pd

from increment import Analysis

units = pd.DataFrame(
    {
        "unit": ["c1", "c2", "t1", "t2"],
        "arm": ["control", "control", "treatment", "treatment"],
        "value": [1.0, 0.0, 1.5, 1.0],
    }
)
analysis = Analysis.from_unit_summary(
    units, unit="unit", group="arm", control="control", metrics={"value": "mean"}
)
report = analysis.health_check()
for check in report.checks:
    print(check.check, check.analysis_population, check.availability, check.conclusion)
analysis.close()
```

::: increment.Analysis.health_check

## Conditional effects and targeting

`estimate_cate` estimates how treatment effects vary with user characteristics.
Validation and targeting APIs use the same fitted result and preserve its
uncertainty and evidence limits.

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
    point prediction. `Analysis.estimate_cate`, `validate_cate`,
    `targeting_rule`, and `select_targeting_rule` delegate to the same source
    APIs.

::: increment.Covariate

::: increment.ClusterBootstrap

::: increment.estimate_cate

::: increment.validate_cate

::: increment.select_targeting_rule

::: increment.targeting_rule

## Segment analysis and rollout

::: increment.segment_heterogeneity

::: increment.segment_contrast


Rollout recommendations give each usable segment equal weight. Exposure-
weighted recommendations are not supported.

??? info "Why rollout recommendations are unweighted"
    An exposure-weighted policy value was evaluated and left out. Under
    weighting, the winner's-curse correction did not meet the accuracy bar of
    the adopted unweighted correction. A weighted headline without an honest
    correction for selection bias would be misleading.

::: increment.segment_rollout_recommendation

## Exploratory multiplicity

Use `select_exploratory_family` to correct eligible whole-window and segment
results together. It accepts only uncorrected fixed-horizon decision rows.

??? info "Family membership and interval details"
    `select_exploratory_family` corrects whole-window `LiftEstimate` rows and
    segment `BreakoutEstimate` rows as one Benjamini-Hochberg family at FDR
    level `q`. Each returned row records `discovery`, `family_axes`, `family_q`,
    `family_threshold`, and `family_size`. A selected row's interval is
    reissued at Benjamini-Yekutieli level `1 - R*q/m`, capped at its nominal
    level, and matches `run_breakout(correction="bh")` for the same cell.
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

::: increment.IdentificationError

::: increment.CapabilityError

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

::: increment.CodedError

`CodedModel`, `CodedValidationMixin` and `unwrap_coded` are public from
`increment.errors`, not the root `increment` namespace.

`CodedModel` and every definition model in `increment.semantics.models`
(which use `CodedValidationMixin`) surface coded validator refusals from direct
construction and direct `model_validate`, `model_validate_json`, and
`model_validate_strings` calls. Definition-specific validators raise
`DefinitionError`; shared finite-sample metric-type and CUPED refusals raise
`InvalidRequestError` with the same `conversion_inference.finite_sample.*` code
and structured context as frame, estimation and planning requests. Catch
`CodedError` when handling both kinds by code. Pydantic schema boundaries such
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
non-coded-model failures remain `ValidationError`, as do plain field
constraints on the other definition models.

```python
from pydantic import TypeAdapter, ValidationError

from increment import DefinitionError, Definitions
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

::: increment.DefinitionError

::: increment.InvalidRequestError

::: increment.RefusalSpec

::: increment.errors.RETIRED_CODES

::: increment.UnsupportedRequestError

::: increment.WireFormatError

::: increment.refuse

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
Filter and assert on `.code` rather than message text:

```python
import warnings

from increment import IncrementWarning

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

Sitewide behavior is exposed through `Analysis.sitewide`, with output values
available from the receive-only results namespace.

`Analysis.sitewide(metric, population="assigned")` reports an assigned
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
`scaled_units`, `unobserved_members`, `assumptions`, and `allocation_balance`.
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
`mo.Html` helpers, and thin adapters over the public `Analysis` and
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
