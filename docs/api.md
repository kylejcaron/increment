# API Reference

The package root contains workflow entry points and caller-authored
configuration. Receive-only values live in `increment.results`; specialized
power solvers live in `increment.power`.

## Pre-1.0 compatibility

Increment is alpha software. This section says what may change between
prereleases, what is written to disk or a warehouse, and how to migrate. It is
a software-compatibility policy, not scientific certification: a stable
interface says nothing about whether a method's assumptions hold for your
experiment. Method assumptions and known gaps live in
[Statistical limitations](limitations.md); the
[capabilities by entry point](reference/capabilities-by-entry-point.md) table remains the
single statement of what runs where.

### Promised surface

These are the interfaces this policy covers:

- documented root-package workflow and configuration imports (`from increment
  import ...`);
- the documented receive-only types in `increment.results` and the solvers in
  `increment.power`;
- result schemas: field names, types, row sets, null reasons and the
  reference/guarantee labels that qualify an interval;
- stable refusal codes and the named fields of their context;
- each versioned portable or artifact format listed below.

Underscore modules and private research kernels are not public APIs and may
change without notice.

### Alpha semantics

- Pin the exact prerelease and the versions of its dependencies (Narwhals,
  Ibis, Arrow, DuckDB, pandas/Polars as used) for any decision you need to
  reproduce.
- Breaking changes are permitted between 0.x prereleases. Each is listed in the
  release notes (the `Breaking changes` section, from pull requests labelled
  `breaking`) with the affected paths and a migration route. Nothing here
  implies 1.0 stability or a deprecation window.
- A released bug fix may change a numerical answer that was wrong. The
  release notes name the correction; the incorrect value is not preserved for
  compatibility.
- Compatibility promises do not vary by method; guarantees and evidence do, see
  [Evidence status](validation.md#evidence-status).

### Machine-readable contracts

- A refusal code's meaning is never reassigned to a different condition.
  Message prose is not stable; branch on `.code` and the context fields.
- A code that is retired keeps its entry in
  `increment.errors.RETIRED_CODES`, which maps it to its replacement code, a
  tuple of replacements when it split, or `None` when it was removed with no
  replacement. For example,
  `estimation.binomial.arm_too_large_for_exact_enumeration` maps to
  `estimation.binomial.finite_sample_arm_ceiling_exceeded`. Retiring a code is
  a contract change and is listed in the release notes.
- Consumers of results must preserve numeric nulls and the reason, guarantee
  and reference metadata that accompany a number. Dropping a null reason or
  reading a value without its reference label changes what the number claims.
- Removing a result field or changing its type requires an explicit migration
  notice.

### Persisted formats

**Moments cubes.** `Analysis.export` writes a `moments_format` stamp on every
row; `Analysis.from_moments` reads it. There is one format number per kind of
writer:

| Format | Written by | Read by `from_moments` |
|---|---|---|
| 9 | registered sequential `export` (one typed checkpoint envelope) | yes |
| 10 | fixed-horizon `export`, with integer `n` and nullable integer `successes` | yes |
| below 9 | no longer written | refused, `moments.format.unsupported_legacy` |
| above 10 | not written by this release | refused, `moments.format.unsupported_future` |

A cube must use one format (`moments.format.mixed`) and repeat no
`(metric, group_id)` row (`moments.rows.duplicate`). Every row stamped below 9,
including each row of a sequential analysis written as format 8, is refused with
`moments.format.unsupported_legacy` before any plan is read. Only a
checkpoint envelope restamped below 9 reaches `sequential.continuation.legacy`:
a checkpoint resumes only from the current format.
Re-export from the raw definitions with the
current release, or pin the release that wrote the file. Nothing is
downgraded or filled in silently.

Fixed-horizon formats 7 and 8 must be re-exported from their original data.
Format 10 carries exact success counts for eligible unit-grain, declared
conversion and retention outcomes. Cluster-aggregate rows and other outcomes
carry `successes=null`. Every ordinary row requires integer `n >= 1` and,
when present, integer `0 <= successes <= n`; import and export refuse invalid
ranges with `moments.count_out_of_range`, including unselected imported rows.
Exact counts travel as these integers; the floating centered moments are used only
to check that a count agrees with them and are not a source from which to recover
one. Preserve count columns as integers through storage and partition
merges; do not cast them through floating point. Sequential format 9 is unchanged.

**Unit-day artifacts.** Artifact context format 2 is the only supported
context. It carries no fact, dimension or exposure SQL; source recipes are
represented by domain-tagged SHA-256 digests. Opening or publishing with a
format-1 context is refused with `artifact.format.unsupported`, which names the
received and supported versions. The refusal does not upgrade an old artifact:
republish from trusted definitions. Copies of a format-1 manifest that you
already stored or shared still contain the SQL they were written with. See
[Canonical unit-day artifacts](guides/data-model.md#canonical-unit-day-artifacts).

**Republish and re-register after the window-day and label fixes.** An
artifact context now binds `window_days`, the boundary-local day of each
declared window edge. A context without it is refused with
`artifact.context.mismatch` (missing `window_days`) before any manifest,
relation or pin is used; republish from trusted definitions. A native
sequential registration made before this change binds a different source
recipe and refuses with `sequential.source.invalid`; register again. Frame
registrations are unaffected. Artifacts published earlier also label Boolean
breakout and factor values `True`/`False`; republishing yields
`true`/`false`/`__null__`.

**Names added in this release.** `increment.semantics.models.local_day` and
`window_days`; `Experiment.start_day`, `Experiment.end_day` and
`Experiment.observation_horizon_day`;
`increment.query.artifact_contract.open_trusted_manifest_snapshot`;
`WarehouseArtifactStore.abandon_generation(artifact_id, generation_id)`, which
durably hides a generation that may or may not have a manifest row yet (safe to
repeat); the `roles=` argument of `increment.impute.pooled_mean`. A
`pooled_mean` call without `roles=` still fills but now emits the
`impute.pooled_mean_role_undeclared` warning.

**Site-volume coverage ends at `end`.** An experiment that declares
`observation_end` after `end` now publishes site-volume coverage through `end`,
the window its rows and the estimand use, instead of through `observation_end`.
Its context and site-volume extension digests change, so republish such
artifacts. Experiments whose `observation_end` is not on a later local day than
`end` are byte-identical.

**New refusal codes.** `query.session.warehouse_artifact.publication_cleanup_incomplete`
(relations could not be dropped before the manifest insert began) and
`query.session.warehouse_artifact.publication_state_unknown` (cleanup was
incomplete after it began: the tombstone could not be written, or relation
erasure failed after it was) are raised when a publication block exits normally
without a manifest and cleanup could not finish. A propagating error is re-raised
unchanged with a note instead. Both codes carry the qualified relation names and a
recovery route in their context.

**Sequential checkpoints and segment labels.** A saved checkpoint is evidence
for one monitoring process and is never relabelled or rehashed. Frame segment
labels are canonical strings: `true`/`false` for Booleans and `__null__` for
missing values. A checkpoint captured under an earlier spelling (`True`,
`None`) still replays unchanged, but continuing it from a source that labels
the same units canonically is refused with `sequential.continuation.rewrite`.
That refusal is not permission to restart monitoring and spend alpha again:
keep the earlier checkpoint as the record, or register a new protocol you can
defend. See [Sequential inference](guides/sequential-inference.md).

**Frame segment labels and degenerate shared schedules.** A frame sequential registration
whose segment labels were not canonical used to report `missing_arm` on the unit summary;
labels are now canonical, as above. A degenerate shared schedule declared as a unit-cycle
order could previously enter the unit-cycle path; a realized schedule too lopsided to be
plausible now refuses before statistics are constructed with
`source.frame.switchback.schedule` and reason `implausible_realized_split`.

**Encodings are format-specific.** Aliases, tuple encoding, collection order,
duplicate rejection and which values are runtime-only are defined by each
format's own contract, not by a general promise that a value is "JSON
compatible".

### Hashes and pickle

A digest proves that bytes match the encoding that was hashed. It does not
prove who wrote them or that a result is statistically valid, and it does not
hide a secret that can be guessed. The caller-pinned reference and the store's
own authorization are the authenticity boundary for artifacts.

Portable files (moments parquet/rows, JSON, artifacts) are the interchange
formats above. Python pickle is not: it is a trusted-input-only convenience
for objects you produced yourself, and carries no versioned-interchange
guarantee across releases. Never unpickle data from an untrusted source.

### `TabularPolicy` persistence

`TabularPolicy` accepts any hashable context key, but only string-keyed tables
roundtrip through JSON. A table with any other key type refuses at dump time
with `logged_policy.policy.json_context_key`, rather than have JSON turn `1`
into `"1"` and change the policy. Persist typed keys with
`model_dump(mode="python")` or trusted pickle. The built-in integer-keyed
`TARGET_POLICY_V1` is in the Python-only group.

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

`Analysis.allocation_history()` returns a PyArrow table with
`experiment_id`, `ds`, `group_id`, `n_daily`, and `n_cumulative`, ordered
by date and arm. It counts first-assignment enrollments independently of
metric maturity, respecting the experiment's day boundary and mixed-assignment
policy. This unit-level history requires a native definitions-backed
analysis; non-native and clustered sources raise a coded `CapabilityError`.

`Analysis.available_metrics` lists the saved per-unit metrics on the experiment's unit that the
experiment does not declare, in definitions order; report-only `total`/`active` metrics and other
entities' metrics are not offered. `run`, `run_breakout`, `run_asof_lift`, `run_asof` and `run_daily` accept
`exploratory_metrics=` naming some of them. Lift rows from `run`, `run_breakout` and
`run_asof_lift` carry `role="exploratory"`, join no plan family, and are tested two-sided at
the plan's full `alpha` in every view; the absolute values from
`run_asof` and `run_daily` carry no role. Every declared row is unchanged. `metrics=[]` with
`exploratory_metrics=` reads only the added metrics. An unknown or already-declared name
(`facade.analysis_config.exploratory_metric_unavailable`) is refused before any query, as is
any source other than `Analysis.from_definitions`
(`facade.analysis.exploratory_metrics_source_limited`) and a registered sequential plan
(`facade.analysis.exploratory_metrics_sequential`). `Analysis.dashboard_snapshot` pins the
added metrics when they are passed in `metrics=`.

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

## Canonical unit-day artifacts

The artifact API is an explicit publication/adoption boundary. Namespace
authorization belongs to the store; callers supply trusted context and pinned
generation references. Context, catalog, and digest helpers are available from
the artifact contract modules.

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
| `TriggerPopulationExtension` | Captures the trigger population relation. |
| `EncouragementUptakeExtension` | Captures encouragement uptake data. |
| `SiteVolumeExtension` | Records site-level measure volume and freshness. |
| `UnitCovariateExtension` | Carries one declared numeric per-unit covariate. |
| `UnitCovariateLevelExtension` | Carries one declared categorical per-unit covariate, including nulls. |
| `ArtifactExtensionCatalogEntry` | Stores a canonical extension definition and source recipe. |
| `ArtifactExtensionRef` | Tagged union of the ten concrete extension references. |
| `ArtifactExtensionRequest` | Tagged union of the ten concrete extension requests. |

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

Extension references and descriptors define the supported dimensions,
assignment, uptake, and site-volume extensions.

### Artifact requests

Requests validate caller-supplied extension definitions before publication.

::: increment.ExtensionRequestBase

::: increment.BreakoutDimensionRequest

::: increment.FactorDimensionRequest

::: increment.ClusterIdentityRequest

::: increment.CupedPreperiodRequest

::: increment.AssignmentCountsRequest

::: increment.TriggerPopulationRequest

::: increment.EncouragementUptakeRequest

::: increment.SiteVolumeRequest

::: increment.UnitCovariateRequest

::: increment.UnitCovariateLevelRequest

### Artifact publication and storage

Publication handles connect validated context to immutable snapshots in the
authorized artifact store.

::: increment.ArtifactPublication

::: increment.ArtifactSnapshot

::: increment.ArtifactStore

::: increment.compile_unit_day_artifact_context

::: increment.unit_day_artifact_extension_catalog

### Digests and hashing

Canonical byte encodings and SHA-256 helpers make artifact identity
deterministic across publication and request boundaries.

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

Off-policy contrast compares two registered policies from a decision trace
logged under one fixed policy version. This family uses its own ingress
(`LoggedTrace.from_records` / `from_frame`) rather than an `Analysis`
constructor; see the [guide](guides/logged-policy.md) for the estimand, the
assumptions, and the floors.
Both constructors accept either `registry=` or complete recorded logging laws:
`logging_distributions=` on records, or `logging_distribution_column=` on a
frame. Exactly one route is required; the estimator and admission bounds are unchanged.


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

Continuous means use `InferenceSpec(kind="asymptotic_mean")` with an ordinary
fixed-allocation design. The source binds the registration before capture;
`expected_decision_sample_size` optionally controls pre-outcome tuning.
Conversion and retention metrics may instead use `InferenceSpec(kind="always_valid",
baseline_rate=...)`, which binds the exact Bernoulli e-process from the plan alone; see
[sequential monitoring of a conversion metric](guides/sequential-inference.md#sequential-monitoring-of-a-conversion-metric).
Definitions-based experiments carry assignment weights in `Experiment.allocation`.
See [ordinary continuous monitoring](guides/sequential-inference.md#ordinary-continuous-monitoring)
for method assumptions, finalization, and the qualified asymptotic guarantee.


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

Fixed-horizon binomial planning reports model-based point power. Enumerated
rejection mass is published only when its computed enclosure is resolved to
absolute error at most `1e-6`, conditional on the deployed SciPy/Boost
special-function error model; this is not a cross-build floating-point proof.
A materially unresolved probability refuses with
`power.binomial_probability_unresolved`, rather than returning its lower bound.
The dense closed-form route reports its model's point probability.
`PowerResult.numerical_qualification` serializes that scope:
`scipy_special_function_error_model_conditional_v1` for a resolved finite-sample
enclosure, `closed_form_model_only_v1` for the asymptotic model, and
`unclaimed_approximation_diagnostic_v1` for an unresolved diagnostic. No public
numerical-bracket field or accuracy tuning knob is exposed.
Binomial MDE searches the earliest detectable region to `1e-8` absolute plus
`1e-8` relative tolerance on the relative-effect scale, not the first
representable float, and reports power evaluated at the returned effect.
A resolved supplied-effect result may have `mde_relative=None` and
`mde_unavailable_reason="numerical_resolution"`; a standalone unresolved MDE
refuses with `power.minimum_detectable_effect.numerical_resolution`.
The planning identity is `hybrid_finite_plus_delta_v3`; persisted v1/v2
binomial plans must be recomputed. See the
[planning guide](guides/power-analysis.md#conversion-and-retention-planning-follows-the-runtimes-route)
for route assumptions and comparison with statsmodels.

Switchback users can obtain a fitted baseline with
`source.planning_baseline(metric)` and pass it to the three `switchback_*`
solvers. No manually supplied reference effect, covariance, or population
certificate is required. The result remains a model-conditioned moment-t
approximation; see [switchback planning](guides/power-analysis.md#switchback-contrasts).

Complete-law simulation declarations and oracle solvers remain in
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

## Conditional effects and targeting

`estimate_cate(..., cluster_weight="member_count")` uses every member equally.
Choose `cluster_weight="equal"` explicitly to give each observed cluster
unit total weight; this requires a declared cluster column. Intervention grain
never changes the default weighting. Centering, regression, covariance and ARD
all use the selected weights.

Clustered fits use the grouped score sandwich with scale `K/(K-1)`, counting
**all** observed cluster IDs, including singleton and constant-residual clusters.
There are no row HC2 divisors or additional row-count corrections. Scalar intervals
use `t(K-1)`; the interaction Wald quadratic divided by its dimension `q` uses
`F(q, K-1)`. Without clusters, the existing HC2 and ARD calculations are unchanged.

`CateResult.dimension`, `n_clusters`, `reference_df`, and immutable `vcov` hold
fit-level inference metadata; CATE and contrast projections reuse that state.
`se_unadjusted` uses a separate fit on `[1, d]` with the same cluster weights and
sandwich, stored as immutable `unadjusted_vcov`. Covariances remain live-object
state excluded from the report-only dump; the counts, dimension, reference df
and selected weighting are included in reports.

Clustered fits raise `InvalidRequestError` with stable `estimation.cate.*` codes
for fewer than two clusters (`insufficient_clusters`), rank deficiency
(`rank_deficient`), invalid covariance (`invalid_covariance`), directions identified
only by one cluster (`single_cluster_direction`), singular interaction Wald
covariance (`singular_wald`), or unavailable required uncertainty
(`uncertainty_unavailable`). Singular full covariance alone does not invalidate
an otherwise supported scalar query. Because the interaction Wald result is
required, an unavailable joint test refuses the fit.

Public validation and both policy APIs retain the same `cluster_weight` and
cluster IDs through the honest split and nested selection. Validation reports
member counts separately from independent heldout clusters. Numeric nulls and
`unavailable_reason` survive policy JSON round trips; unavailable AUTOC evidence
keeps the gate closed. A clustered constant score reports
`estimation.targeting.degenerate_rank_distribution`, rather than manufacturing
evidence from repeated members. A supplied randomized training mask that splits
a cluster raises `estimation.crossfit.cluster_split`, as does a split holdout mask.

`targeting_rule` and `select_targeting_rule` accept
`deploy_grain: Literal["unit", "cluster"] | None = None`. Only a declared
`intervention_grain="cluster"` defaults to cluster deployment. Dependence clusters
alone retain unit policies, including mixed-treatment observational clusters.
An explicit unit policy for a cluster intervention refuses before reading data
with `estimation.targeting.unsupported_unit_deployment`.

Cluster deployment averages member scores, orders clusters by descending score
and canonical ID, and takes the longest prefix fitting the requested budget.
`cluster_weight="member_count"` budgets members; `"equal"` budgets clusters. The
same weights govern fitting, selection and policy evaluation. The next cluster
is never split or skipped to backfill unused capacity. Fractions `0` and `1`
select nobody and everybody; an oversized first cluster selects nobody.
`fraction` records the requested share, `achieved_fraction` the realized share;
inner `FractionScore` rows also retain achieved shares.

`CateResult.score(cols, *, cluster_ids=None, deploy_grain=None)` returns an aligned
unit array in unit mode, or an immutable tuple of `ClusterScore(cluster_id, score,
member_count)` records in canonical ID order in cluster mode. IDs are separate
metadata. `TargetingRule.predict` returns aligned Boolean actions from its saved
`CateScoreState`: unit policies use their frozen cutoff, cluster policies pool
and budget the supplied batch and broadcast cluster actions. Supply the complete
member roster for each deployment cluster. `predict` evaluates the candidate;
`recommendation` remains the evidence gate for deploying it. An empty candidate
has no conditional policy effect, with reason `estimation.targeting.empty_group`.
A cluster policy's reported `threshold` is descriptive; it does not replace its
ID tie break and prefix budget.

Policy JSON includes the fitted basis, knots, category levels, centering and
coefficients, so `TargetingRule.model_validate_json(rule.model_dump_json())`
predicts without fitting again. CATE report dumps remain report-only; their
separately serializable `score_state` supplies portable point prediction.
`Analysis.estimate_cate`, `validate_cate`, `targeting_rule` and
`select_targeting_rule` delegate to the same source APIs.

::: increment.Covariate

::: increment.ClusterBootstrap

::: increment.estimate_cate

::: increment.validate_cate

::: increment.select_targeting_rule

::: increment.targeting_rule

## Segment analysis and rollout

::: increment.segment_heterogeneity

::: increment.segment_contrast


Every value below is unweighted: each usable segment contributes equally,
regardless of the exposure it carries. An exposure-weighted policy value was
measured before this shipped and deliberately left out -- under weighting the
winner's-curse correction could not meet the accuracy bar the unweighted
correction is adopted against, and a weighted headline whose selection bias
cannot be removed honestly is worse than none.

::: increment.segment_rollout_recommendation

## Exploratory multiplicity

`select_exploratory_family` corrects whole-window `LiftEstimate` rows and segment
`BreakoutEstimate` rows as one Benjamini-Hochberg family at FDR level `q`. Every
returned row records `discovery`, `family_axes`, `family_q`, `family_threshold` and
`family_size`, so a row read alone states which family corrected it. A selected
row's interval is reissued at the Benjamini-Yekutieli level `1 - R*q/m`, capped at
the row's own nominal level, and equals the interval `run_breakout(correction="bh")`
returns for the same cell; unselected rows keep their nominal interval.

Rows must be uncorrected fixed-horizon decision rows. Each refusal names every
offending row in its `rows` context:

| Combination | Classification |
|---|---|
| Mean, ratio and CUPED rows (Wald log-scale interval) | Supported: reissued from the persisted raw statistics and reference. |
| Conversion and retention rows (exact binomial) | Supported: reinverted from the persisted counts, one-sided geometry kept. |
| Clustered ratio rows (Fieller set) | Supported: reinverted from the persisted joint reference. |
| Rows with no relative interval (non-positive arm mean) selected by an absolute margin | Supported: the additive interval is reissued from the persisted `abs_diff`, `abs_se`, reference and `abs_alpha`, never narrower than the nominal interval. A row serialized before `abs_alpha` existed has no recorded level to cap at, so it is refused, `estimation.family.exploratory_construction`. |
| Breakout segments, and whole-window rows beside them | Supported: one family over metric, arm and segment. |
| Cells excluded by design (too few units, no control arm) | Not hypotheses: returned unchanged, outside the family. Outcome-based exclusions stay in `m` as non-rejections. |
| Informative prior | Mathematically unsound for BH (a posterior tail is not a frequentist p-value): refused, `breakout.run_breakout_bh_excludes_prior`. |
| Sequential inference | Unfinished: refused, `estimation.family.exploratory_sequential`. |
| Quantile, percentile-winsorized and additive-scale rows | Unfinished: their interval cannot be reissued from persisted state without approximation, so they are refused, `estimation.family.exploratory_construction`. |
| Rows already corrected, sensitivity rows, day-axis rows | Refused: `estimation.family.exploratory_pre_corrected`, `estimation.family.exploratory_non_decision`, `estimation.family.exploratory_row`. |

::: increment.estimation.family.select_exploratory_family

`exploratory_family_exclusion` applies the same admissibility test to one row and
returns the reason it cannot join the family, or `None`. Leave excluded rows out of
the family and report the reason, instead of letting one cell refuse the call; the
refusals carry the same strings in their `reasons` (or `constructions`) context.
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
Filter or assert on `.code` rather than message text:

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
