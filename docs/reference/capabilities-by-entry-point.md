# Capabilities by entry point

This page records, for each of the six `Analysis` entry points, which capabilities run and which refuse.
The [Method compatibility reference](../guides/compatibility.md) is organised by metric type and method; neither page restates the other's cells.
Where they disagree, the probe and harness results govern: `scripts/probe_capability_table.py`, `tests/test_composition_matrix.py` and `tests/parity_harness/matrix.py`.

## What runs where

A capability is a claim only when its path is named. `Analysis` has six entry
points in three families: **dataframe** (`from_unit_summary`, one row per unit;
`from_unit_panel`, one row per unit per day; `from_switchback_panel`);
**warehouse** (`from_definitions`, compiling from raw event facts;
`from_unit_day_artifact`, reading unit × day relations previously published
into the warehouse); and **portable** (`from_moments`, a file of pre-reduced
moments exported from any of the others).

The table records measured behavior on
every path, using the dataframe unit-summary path as the oracle against which the other matched-arm paths are checked. The switchback panel estimates a
different quantity (a fixed-horizon contrast over a switchback schedule), so it is
exercised on its own and never compared row for row with that oracle.

Reproduce this table with `uv run --extra demo --extra tables --extra dashboard python scripts/probe_capability_table.py`.

| Capability | Unit summary | Unit panel | Switchback panel | Definitions | Unit-day artifact | Portable moments |
|---|---|---|---|---|---|---|
| Mean CUPED, fixed horizon | yes | yes | refused | yes | yes | yes |
| Ratio CUPED, fixed horizon | yes | yes | refused | yes | yes | yes |
| Ratio denominator-precision advisory, fixed horizon | yes | yes | refused | yes | yes | yes |
| Informative Normal/Student-t/mixture log-lift priors, fixed-horizon mean/ratio/conversion/mean CUPED | yes | yes | not comparable (SOURCE — matched parallel-arm data has no switchback schedule) | yes | yes | yes |
| Fixed-threshold winsorization under sequential inference | yes | yes | refused | yes | yes | checkpoint replay |
| Ratio metric under sequential inference | yes | yes | refused | yes | yes | checkpoint replay |
| CUPED under sequential inference | asymptotic route | refused | refused | asymptotic route | asymptotic route | checkpoint replay |
| Ratio CUPED under sequential inference | asymptotic route | refused | refused | asymptotic route | asymptotic route | checkpoint replay |
| Bounded conversion under asymptotic sequential inference | yes, pre-windowed per-unit outcomes with `exposure_date` | yes | refused | yes | yes | checkpoint replay |
| Bounded retention (`threshold_days=(start, end)`) under asymptotic sequential inference (parity: the enumerated variant cells run by `tests/test_parity_matrix.py::test_asymptotic_retention_cell`, ids ending `-asymptotic_mean`, counts in `tests/parity_harness/COVERAGE.md`; in the `run` cells the four live ingresses agree and each retains the `scalar_mean` law) | refused (SOURCE, `source.frame.constructor` -- the summary carries no dates to resolve `threshold_days` against) | yes | refused | yes | yes | checkpoint replay |
| Bounded conversion under automatic exact Bernoulli monitoring (`always_valid` without a registration) | yes, pre-windowed per-unit outcomes with `exposure_date` | yes | refused | yes | yes | checkpoint replay |
| Bounded retention under automatic exact Bernoulli monitoring (`always_valid` without a registration) | refused (SOURCE, `source.frame.constructor` -- same as above) | yes | refused | yes | yes | checkpoint replay |
| Automatic exact multi-arm bounded-conversion monitoring | yes, pre-windowed per-unit outcomes with `exposure_date` | yes | refused (SOURCE, `source.frame.switchback.identification` -- a switchback frame takes exactly one treatment arm; a sequential plan is refused separately, `source.frame.switchback.plan`) | yes | yes | checkpoint replay |
| Automatic exact multi-arm bounded-retention monitoring | refused (SOURCE, `source.frame.constructor` -- same as above) | not measured: no parity case exercises multi-arm retention | refused (SOURCE, `source.frame.switchback.identification` -- same as above) | not measured | not measured | not measured |
| Unbounded conversion under sequential inference (`conversion-run-sequential-utc-error`; unfinished routes tracked in `1cr4`) | refused (`sequential.source.invalid`; unfinished) | refused (`sequential.route.unsupported`; unfinished) | refused (SOURCE, `source.frame.switchback.plan`) | refused (`sequential.route.unsupported`; unfinished) | refused (`sequential.route.unsupported`; unfinished) | refused (SOURCE, `sequential.source.invalid` -- no checkpoint can be captured for replay) |
| Unbounded retention (`threshold_days=int`) under sequential inference | refused (SOURCE, `source.frame.constructor`) | no recorded disposition | refused | refused (`sequential.route.unsupported` -- an unbounded band has no finalized observation window) | no recorded disposition | no sequential process can start |
| Randomized registered segmented sequential family (automatic predeclared levels or explicit roster) | yes | yes | refused (SOURCE -- no sequential construction) | refused (SOURCE, `sequential.route.unsupported` -- relational capture has no immutable segment-property contract) | refused (SOURCE, `sequential.route.unsupported` -- artifact capture has no immutable segment-property contract) | retained-state replay only; breakout readout refused (SOURCE, `readout.source.dimension` -- no breakout catalog) |
| Boolean and null segment labels in a registered segmented sequential family (canonical `true`/`false`/`__null__`, matching the string-labelled oracle) | yes (measured on pandas, Polars and Arrow Boolean columns, automatic and explicit rosters) | yes (same three column kinds) | refused (SOURCE -- no sequential construction) | refused (SOURCE, `sequential.route.unsupported`) | refused (SOURCE, `sequential.route.unsupported`) | retained-state replay only; breakout readout refused (SOURCE, `readout.source.dimension`) |
| `increment.impute.pooled_mean` with declared `roles=`; `MetricSpec(missing="impute")` | roles honored: an `outcome` role refuses (CONSTRUCTION -- filling an outcome with its pooled mean shrinks variance, `impute.pooled_mean_outcome`); an undeclared role warns (`impute.pooled_mean_role_undeclared`); `MetricSpec(missing="impute")` refuses (CONSTRUCTION, `frame.metric.missing_impute`) | same | same | not comparable (SOURCE -- a warehouse `Metric` cannot express `missing="impute"`; the helper prepares dataframes only) | not comparable (SOURCE -- same) | not comparable (SOURCE -- no per-unit rows) |
| Encouragement ITT, compliance and LATE, fixed horizon, declared via `design:` | yes | yes | not expressible as `design=` (no such parameter; `TypeError`, no `.code`); passed as `identification=`, refused (SOURCE, `source.frame.switchback.identification`) | yes | yes | yes |
| Explicit ITT/compliance without a LATE exclusion declaration (fixed horizon or supported sequential monitoring) | yes | yes | not expressible as `design=` (no such parameter; `TypeError`, no `.code`); passed as `identification=`, refused (SOURCE, `source.frame.switchback.identification`) | yes | yes | yes; sequential checkpoint replay |
| Fixed-horizon design-level compliance with an empty outcome catalog | yes, `metrics=[]` | yes, `metrics=[]` | not expressible as `design=` (no such parameter; `TypeError`, no `.code`); passed as `identification=`, refused (SOURCE, `source.frame.switchback.identification`) | yes, `metrics=[]` | yes, `metrics=[]` | yes, format-11 `design_summary` with source identity and `metrics=[]` |
| Encouragement guardrail with a declared non-inferiority margin | yes, ITT row carries the margin verdict; refused (CONSTRUCTION, `readout.encouragement.margin`) if a request excludes `itt` | yes, ITT row carries the margin verdict; refused (CONSTRUCTION) if `itt` is excluded | not expressible as `design=` (no such parameter; `TypeError`, no `.code`); passed as `identification=`, refused (SOURCE, `source.frame.switchback.identification`) | yes, ITT row carries the margin verdict; refused (CONSTRUCTION) if `itt` is excluded | yes, ITT row carries the margin verdict; refused (CONSTRUCTION) if `itt` is excluded | yes, ITT row carries the margin verdict; refused (CONSTRUCTION) if `itt` is excluded |
| Encouragement BH-selected secondary's LATE row re-estimated at the corrected level | yes | yes | not expressible as `design=` (no such parameter; `TypeError`, no `.code`); passed as `identification=`, refused (SOURCE, `source.frame.switchback.identification`) | yes | yes | yes |
| Encouragement ITT on a conversion metric takes the count-routed conversion route (dense counts the delta-method route, `reference_kind="t"`; sparse counts the finite-sample route, `"binomial"`) | yes | yes | not expressible as `design=` (no such parameter; `TypeError`, no `.code`); passed as `identification=`, refused (SOURCE, `source.frame.switchback.identification`) | yes | yes | yes |
| Conversion/retention counts one below, at and one above the dense threshold, on the success and failure side, route identically and report the same interval and p-value (`conversion_route_straddles_the_dense_threshold`) | yes | yes | not comparable (SOURCE -- no switchback schedule) | yes | yes | yes |
| Conversion/retention breakout with dense, sparse and threshold-straddling segments in one metric: each segment row is routed by its own counts (`conversion_route_breakout_segments`) | refused (SOURCE -- no declared breakout dimension) | yes | not comparable (SOURCE -- no switchback schedule) | yes | yes | refused (SOURCE -- no per-unit breakout dimension) |
| Encouragement ITT rows of a BH secondary family are routed at the family's smallest level, `q` over its hypotheses: counts dense at the row's own tail but not at that one take the finite-sample route (`conversion_route_encouragement_family`) | yes | yes | not expressible as `design=` (no such parameter; `TypeError`, no `.code`); passed as `identification=`, refused (SOURCE, `source.frame.switchback.identification`) | yes | yes | yes |
| Observational unadjusted rows of a BH secondary family are routed at the family's smallest level (`conversion_route_observational_family`) | yes | yes | not expressible as `design=` (no such parameter; `TypeError`, no `.code`); passed as `identification=`, refused (SOURCE, `source.frame.switchback.identification`) | yes | yes | yes |
| Encouragement design with a retention metric (ITT and LATE readouts) | refused (SOURCE, `source.frame.constructor` -- no per-unit dates to resolve `threshold_days` against) | refused (CONSTRUCTION, `readout.encouragement.retention`) | not expressible as `design=` (no such parameter; `TypeError`, no `.code`); passed as `identification=`, refused (SOURCE, `source.frame.switchback.identification`) | refused (CONSTRUCTION, `readout.encouragement.retention`) | refused (CONSTRUCTION, `readout.encouragement.retention`) | refused (CONSTRUCTION, `readout.encouragement.retention`) |
| Encouragement design with a retention metric, compliance-only request (`estimands=("compliance",)`, which ignores the retention outcome) | refused at construction (SOURCE, `source.frame.constructor` -- the retention declaration itself) | refused at construction (CONSTRUCTION, `readout.encouragement.retention` -- the retention declaration itself) | not expressible as `design=` (no such parameter; `TypeError`, no `.code`); passed as `identification=`, refused (SOURCE, `source.frame.switchback.identification`) | yes (`tests/test_compliance_summary.py::test_compliance_only_ignores_outcome_configuration`) | not measured | not measured |
| Encouragement conversion ITT breakouts retain exact confidence sets at zero control counts | refused (SOURCE -- no declared breakout dimension) | yes | not comparable (SOURCE -- no switchback schedule) | yes | yes | refused (SOURCE -- no per-unit breakout dimension) |
| Bounded-retention breakouts with mature exposure cohorts (`audit-retention-breakout-cohorts`) | refused (SOURCE -- no exposure-date cohort stream, `source.frame.constructor`) | yes, `run_breakout` | not comparable (SOURCE -- no switchback schedule) | yes, `run_breakout` and cohort-aware `breakout_summaries` | yes, `run_breakout` and cohort-aware `breakout_summaries` | refused (SOURCE -- no breakout operation, `facade.analysis.operation`) |
| Encouragement continuous ITT + Bernoulli uptake compliance, jointly monitored (composed sequential family, e-BH selected) | yes | yes | not expressible as `design=` (no such parameter; `TypeError`, no `.code`); passed as `identification=`, refused (SOURCE, `source.frame.switchback.identification`); a sequential plan is also refused, `source.frame.switchback.plan` | yes | yes | checkpoint replay |
| Uptake-only sequential compliance with an empty outcome catalog (`always_valid` and a compliance policy) | yes, `metrics=[]` | yes, `metrics=[]` | not expressible as `design=` (no such parameter; `TypeError`, no `.code`); passed as `identification=`, refused (SOURCE, `source.frame.switchback.identification`); a sequential plan is also refused, `source.frame.switchback.plan` | yes | yes | checkpoint replay with `metrics=[]` |
| Uptake-only sequential compliance that retains an unbounded retention metric in the outcome catalog (explicit uptake-only registration over that catalog; compliance-only `run_asof_lift`/`run` succeeds; an ITT request refuses with `breakout.retention.encouragement` on `run_asof_lift`, or `sequential.source.invalid` on `run` because the outcome has no registered sampling law) | refused (SOURCE -- a one-row-per-unit summary has no per-unit dates to resolve a retention band against, `source.frame.constructor`) | refused (COMBINATION -- a unit panel refuses a retention metric under an Encouragement design at construction, `readout.encouragement.retention`) | not expressible as `design=` (no such parameter; `TypeError`, no `.code`); passed as `identification=`, refused (SOURCE, `source.frame.switchback.identification`); a sequential plan is also refused, `source.frame.switchback.plan` | yes, catalog retained | yes, the retention metric declared on the experiment and its catalog retained | checkpoint replay with the retention catalog retained |
| Observational weight diagnostics (IPTW/AIPW; unit-level mean and conversion parity across four supported ingresses; cluster grain measured on unit summary) | yes; mean/conversion parity | yes; mean/conversion parity | not expressible (`from_switchback_panel` has no `design=` parameter; observational identification refused) | yes; mean/conversion parity; cluster grain | yes; mean/conversion parity | refused (SOURCE, `source.moments.covariate_unavailable` -- no per-unit covariates) |
| Assignment-integrity scope (status, observed counts, randomization grain and retained metadata; triggered populations are not assignment-law evidence) | yes; unit counts or declared cluster counts; unassigned totals are separate | yes at unit grain; no cluster key in the panel constructor | not comparable (a switchback contrast does not have a parallel-arm assignment law) | yes; unit or cluster-grain counts with mixed/unassigned audit totals | yes; cluster grain requires cluster identity, and mixed/unassigned totals require assignment-count evidence | yes at unit grain; clustered counts require the complete cluster/compliance payload used by portable moments |
| Observational IPTW covariate adjustment | yes | yes | not expressible as `design=` (no such parameter; `TypeError`, no `.code`); passed as `identification=`, refused (SOURCE, `source.frame.switchback.identification`) | yes | yes | refused (SOURCE -- a moments cube carries no per-unit rows to attach a covariate to) |
| Observational AIPW/DML covariate adjustment | yes | yes | not expressible as `design=` (no such parameter; `TypeError`, no `.code`); passed as `identification=`, refused (SOURCE, `source.frame.switchback.identification`) | yes | yes | refused (SOURCE -- same as above) |
| Categorical observational adjustment (string covariates) | yes | yes | not expressible as `design=` (no such parameter; `TypeError`, no `.code`); passed as `identification=`, refused (SOURCE, `source.frame.switchback.identification`) | yes | yes | refused (SOURCE, `source.moments.covariate_unavailable` -- no per-unit covariates) |
| Categorical nulls with a nondefault missing-value policy | yes | yes | not expressible as `design=` (no such parameter; `TypeError`, no `.code`); passed as `identification=`, refused (SOURCE, `source.frame.switchback.identification`) | default missing-value refusal; YAML does not accept policy overrides | default missing-value refusal inherited from definitions | refused (SOURCE -- no per-unit covariates) |
| Observational primary + secondary + guardrail together (role multiplicity) | yes | yes | not expressible as `design=` (no such parameter; `TypeError`, no `.code`); passed as `identification=`, refused (SOURCE, `source.frame.switchback.identification`) | yes | yes | refused (SOURCE -- same as above) |
| Quantile inference on tied/rounded outcomes | yes | yes (unwindowed only -- the panel's per-unit collapse is the same per-unit total mean/ratio already use; a windowed quantile metric can never be declared on the frame path at all, `frame.metric.window_days_supported`) | refused (SOURCE -- a quantile metric is refused at the metric-type gate before any frame is read) | yes | yes | genuine quantile export refused (`source.frame.quantile_no_moments`); a quantile declared over an existing scalar cube refuses on `run()` with `source.moments.unit_grain`, on `run_breakout()` with `facade.analysis.operation`, and on day-axis methods with `facade.analysis.no_definitions` |
| Registered conversion checkpoint retaining an unmodeled, unwindowed quantile catalog (parity: `sequential-mixed-quantile-catalog-checkpoint`) | yes, export and replay | yes, export and replay | not comparable (SOURCE -- no quantile catalog or sequential construction) | yes, export and replay | yes, export and replay | checkpoint replay only; re-export refused (`facade.analysis.operation`) |
| Quantile metric under an `Observational` design (two-sided, zero-null request; one-sided or absolute-margin requests first refuse with `readout.metric.quantile_alternative`, relative margins at construction with `plan.observational.relative_margin`) | refused (CONSTRUCTION, `readout.observational.quantile` -- no observational quantile estimator exists) | refused (CONSTRUCTION, same code) | not expressible as `design=` (no such parameter; `TypeError`, no `.code`); passed as `identification=`, refused (SOURCE, `source.frame.switchback.identification`) | refused (CONSTRUCTION, same code) | refused (CONSTRUCTION, same code) | a two-sided zero-null quantile declared over an existing scalar cube refuses on `run()` with `readout.observational.quantile` before using the moments; genuine observational quantile export raises the same estimator refusal. `run_breakout()` raises `facade.analysis.operation`, and day-axis methods raise `facade.analysis.no_definitions` (SOURCE limits before any estimator). Earlier margin and alternative gates apply as stated in the first column. |
| Clustered ratio metric with a non-positive control-arm numerator | yes | refused (SOURCE -- `from_unit_panel(cluster=...)` refuses unconditionally; build clustered arm totals with `from_unit_summary(cluster=...)`) | not expressible (`from_switchback_panel` has no `cluster=` parameter: `TypeError`, no `.code`; a clustered arm-lift ratio has no switchback contrast shape) | yes | yes | refused (SOURCE, `source.moments.cluster_grain` -- clustered moments transport needs a complete encouragement compliance payload) |
| Clustered zero relative variance retains its point and any available additive interval | yes | refused (SOURCE -- no clustered arm moments; use `from_unit_summary`) | not expressible (`from_switchback_panel` has no `cluster=` parameter: `TypeError`, no `.code`; a switchback contrast has no arm-cluster shape) | yes | yes | refused (SOURCE, `source.moments.cluster_grain` -- clustered moments transport needs a complete encouragement compliance payload) |
| `Analysis.planning_baseline`, mean metric | yes | yes | yes (the pilot-fitted `SwitchbackBaseline` for the `switchback_*` solvers) | yes | yes | yes |
| `Analysis.planning_baseline`, quantile metric | yes | yes (same unwindowed-only reach as the row above) | refused (SOURCE -- a quantile metric is refused at the switchback metric-type gate before any frame is read) | yes | yes | refused (SOURCE, `analysis.planning_baseline.quantile_source_unavailable` -- no per-unit control-arm values for a pilot; genuine quantile export separately refuses with `source.frame.quantile_no_moments`) |
| `Analysis.planning_baseline` with a declared cluster (per-unit mean/var; ICC on the metric's per-unit score) | yes | refused (SOURCE -- `from_unit_panel` takes no cluster) | not expressible (`from_switchback_panel` has no `cluster=` parameter: `TypeError`, no `.code`; a switchback contrast has no arm cluster) | yes | yes | refused (SOURCE, `source.moments.cluster_grain` -- clustered moments are not exported) |
| `Analysis.planning_baseline` under a declared trigger (triggered-population moments and `trigger_rate`) | refused (SOURCE -- `from_unit_summary` accepts no `Experiment` declaration, so `planning_baseline`'s `trigger` always resolves `None`; the assigned population is analyzed instead) | refused (SOURCE -- same as `from_unit_summary`) | refused (SOURCE -- same as `from_unit_summary`) | yes | yes with `trigger_population`, triggered `assignment_counts`, and metric-scoped `trigger_measure_stats` extensions, plus explicit `SourceSnapshotEvidence`; otherwise refused (SOURCE, `analysis.planning_baseline.trigger_evidence_unavailable`) | refused (SOURCE -- same as `from_unit_summary`) |
| Triggered daily/as-of outcome value and lift history (`run_daily`, `run_daily_lift`, `run_asof`, `run_asof_lift`; membership is limited to units whose first eligible trigger is observed by each day, and outcome windows anchor at that unit's trigger) | refused (SOURCE -- no experiment trigger declaration or per-unit trigger-evidence operation) | refused (SOURCE -- no experiment trigger declaration or per-unit trigger-evidence operation) | refused (SOURCE -- switchback contrasts do not carry trigger-anchored arm day evidence) | yes; requires explicit `SourceSnapshotEvidence`; outcome events are strictly after each unit's first eligible trigger and are cutoff by source evidence | yes; requires `trigger_population` and metric-compatible `trigger_measure_stats` evidence | refused (SOURCE -- portable moments have no trigger anchor) |
| Triggered encouragement compliance (`run_daily_lift` / `run_asof_lift`, `estimands=("compliance",)`; uptake stays assignment-anchored while membership enters at first eligible trigger) | refused (SOURCE -- no experiment trigger declaration or per-unit trigger anchors) | refused (SOURCE -- panel lacks trigger declaration and trigger dates) | refused (SOURCE -- switchback contrasts carry no triggered assignment population) | supported with explicit `SourceSnapshotEvidence`; uptake windows use assignment time and triggered membership uses first-trigger day | supported with `trigger_population` and encouragement uptake evidence; membership uses first-trigger day and uptake windows use assignment time | refused (SOURCE -- portable moments have no trigger anchors) |
| Triggered clustered daily/as-of outcome readouts | refused (SOURCE -- no trigger declaration) | refused (SOURCE -- no trigger declaration) | not comparable | refused for outcome histories (`facade.analysis.clustered_day_axis`); compliance-only uptake histories remain supported | refused for outcome histories (`facade.analysis.clustered_day_axis`); cluster-grain compliance-only histories remain supported with trigger and uptake evidence | refused (SOURCE -- no trigger membership evidence) |

| Logged-policy contrast (`estimate_policy_contrast`) | not expressible (no `Analysis` constructor or method carries a decision trace -- the per-decision chosen action, exact logged propensity, logging policy version and reward boundary -- so there is no call to write and no `.code`; the family's own ingress is `LoggedTrace.from_records` / `from_frame`) | not expressible (same) | not expressible (same) | not expressible (same) | not expressible (same) | not expressible (same) |
| TabularPolicy persistence (`model_dump_json` / `model_validate_json`) | not comparable (SOURCE -- no `Analysis` constructor carries a policy table; the policy is a `LoggedTrace` input) | not comparable (SOURCE -- same) | not comparable (SOURCE -- same) | not comparable (SOURCE -- same) | not comparable (SOURCE -- same) | not comparable (SOURCE -- same). Within the family: JSON roundtrips string-keyed tables only; any other key type refuses (CONSTRUCTION -- JSON object keys are strings, `logged_policy.policy.json_context_key`); typed keys persist through `model_dump(mode="python")` or trusted pickle |

Triggered encouragement compliance histories are dense calendar series from
the first observed trigger-entry date through the observed uptake/observation
horizon. Uptake remains assignment-anchored; `completed_windows_only` also
requires the assignment-anchored window to fit within both the requested
as-of day and the inclusive certified uptake-feed edge. Dimensioned daily
compliance requests are explicitly refused with
`facade.analysis.daily_compliance_dimension_unsupported`.

## Triggered auxiliary readouts

Triggered segmented readouts are separate populations, not a filter silently
applied to the assigned view. `run_breakout()` and `breakout_summaries()` emit
assigned and triggered results with population-qualified family/table identity.
Definitions are supported; a unit-day artifact is supported for breakouts only
when it contains the declared breakout-dimension extension and current trigger
evidence. Segment properties declared `as_of: pre_exposure` are resolved at
assignment, before either population's outcome anchor. The consumer parity
regression in `tests/test_analysis_trigger.py` compares population-qualified
breakout/factor summaries and public breakout rows across definitions and
unit-day artifact ingresses.

| Read | Definitions | Unit-day artifact | Other entry points |
|---|---|---|---|
| `run_breakout()` / `breakout_summaries()` with a declared trigger | supported with explicit `SourceSnapshotEvidence` | supported with breakout-dimension and trigger evidence; otherwise `artifact.extension.missing` or `artifact.evidence.unavailable` | no constructor carries a declared trigger membership for this read |
| `factor_summaries()` with a declared trigger | supported on native definitions evidence with explicit `SourceSnapshotEvidence` | supported with the declared `factor_dimension` extension and trigger evidence; otherwise `artifact.extension.missing` or `artifact.evidence.unavailable` | no trigger declaration or per-unit factor evidence |
| `dashboard_group_data(population=...)` | supports assigned and triggered populations; pre-exposure group properties remain assignment-anchored | refused (`facade.analysis.operation`; no source-owned dashboard aggregation operation) | no trigger declaration |
| `allocation_history(population=...)` | supports assigned enrollment dates and first-trigger cohort dates; rows carry `analysis_population` and sort by population, date, then arm | refused (`facade.analysis.operation`; artifact retains no raw enrollment timeline) | no trigger declaration |
| `sitewide()` with a declared trigger | refused (`analysis.sitewide.triggered_population_unsupported`; whole-site evidence cannot isolate trigger-eligible units) | same source-specific refusal | no trigger declaration |
| `estimate_cate()`, `validate_cate()`, `targeting_rule()`, `select_targeting_rule()` with a declared trigger | supported when the source can supply triggered unit outcomes and pre-assignment covariates | supported with triggered-population and unit-covariate evidence; source-specific missing-evidence refusals remain | no trigger declaration |

Triggered sequential breakout inference remains explicitly unsupported
(`sequential.route.unsupported`): the registered construction has no
triggered-population breakout roster. It does not fall back to assigned
segments.

For `from_definitions`, pinning preserves dimensions joined by freshness-bearing
fact sources. Qualifying non-enrolled events can establish that later data has
arrived without entering the enrolled outcome population. In `from_unit_panel`,
natural day labels such as `d9` and `d10` use numeric order when inferring
maturity, matching the same explicitly declared observation bound.
Windowed ratios use the earlier numerator/denominator observation bound for
totals and completed as-of cohorts. An entirely null component under
`missing="zero"` retains the panel-extent fallback; an explicitly supplied
`observation_end` remains authoritative.

For clustered triggered planning, `from_definitions` and `from_unit_day_artifact`
retain both populations: assigned mean cluster size determines recruitment;
contributing-cluster participation, analyzed size CV, and analyzed-score ICC
determine the planning approximation. The artifact needs `cluster_identity`
as well as the trigger and assignment-count extensions. These pilot estimates
do not certify future participation or finite-sample coverage. Other
constructors cannot supply the declared trigger membership identified in the table.
Both clustered trigger routes refuse a pilot whose metric rows omit
trigger-eligible control units (SOURCE,
`analysis.planning_baseline.trigger_metric_population_unavailable`).
Missing rows can reflect unfinished windows or outcomes undefined even after
full follow-up, such as `avg_event` for a unit with no events. Complete
unfinished windows; for structurally undefined outcomes, choose a metric
defined for every eligible unit. The adapter does not silently substitute a
complete-case population for the declared trigger population.

For trigger-aware planning through `from_definitions` and an adopted unit-day
artifact, the consumer regression covers mean and ratio CUPED moments,
an unadjusted median baseline, and the observed trigger rate. Each uses the
first eligible trigger as the outcome-window anchor; mean and ratio CUPED
retain covariates measured before each unit's own assignment, even with
staggered assignment dates. Enrollment counts remain assignment-based, while
the analyzed-trigger count remains distinct in baseline-derived sample-size
results. These claims apply to those two warehouse ingresses, not to the
summary, panel, switchback, or portable-moment routes marked refused above.

Mean/Ratio CUPED on the unit panel resolves the declared covariate the same
way observational adjustment already does: through the per-unit value
`unit_frame()`/`moments(grain="total")` join in, constant across a unit's
own rows, refused by name (`frame.frame_panel.unit_covariate_varies`) when
it genuinely varies. A windowed or retention metric still refuses a CUPED
covariate (CONSTRUCTION, `frame.validation.from_unit_panel`): those metrics
have no per-unit collapse for that value to attach to. For a windowed
metric, compute each unit's windowed value upstream and declare it as an
unwindowed metric on `from_unit_summary`; no frame source serves a per-unit
retention value, so remove the covariate to run retention without CUPED.
The switchback panel supports neither sequential inference
nor CUPED; its contrasts are fixed-horizon. "Asymptotic route" means CUPED
is admitted under `InferenceSpec(kind="asymptotic_mean")`: fitted coefficients
use retained joint moments, whose confidence sequence accounts for their
estimation; pre-period coefficients use the scalar-mean adjustment. The exact
Bernoulli e-process route (`AlwaysValid`) admits neither form of CUPED.
"Checkpoint replay" means `from_moments` cannot start a sequential process,
because it holds no per-unit source to register against, but reproduces an
exported checkpoint exactly. On the warehouse the covariate for a ratio metric
is the numerator's pre-period total; a separate denominator covariate is not
expressible, and the CUPED guide says why. An encouragement design's continuous
ITT and Bernoulli-uptake compliance, monitored jointly, compose into one
sequential family automatically (no explicit registration to declare); the
family reports one shared `family_guarantee`, equal to the weakest regime any
member carries -- with an asymptotic scalar-mean ITT cell in the family, that
value is `asymptotic_sequential` on every selected row, including the exact
Bernoulli compliance cell's own row, not "exact for one cell, asymptotic for
the other." The `from_switchback_panel` encouragement and observational cells are a structural
limit of that constructor, not a gap. It has no `design=` parameter, so a `design:` declaration
cannot be written and Python raises `TypeError` (no `.code`). It accepts only a
switchback-shaped frame and `identification: Randomized`; an `Encouragement`/`Observational`
object passed as `identification=` (as the parity matrix's observational cells do) is refused
with `source.frame.switchback.identification` (reason `unsupported_identification`), as is a
quantile metric (`source.frame.switchback.metric`, reason `unsupported_metric_type`).

Parity is asserted, not assumed: identical per-unit data through each path must
reproduce the same adjusted moments and interval, and the same retained
sequential state byte for byte. That check runs against DuckDB in the ordinary
suite and against a live PostgreSQL in the warehouse gate. For the two binary
rows it reads one per-unit dataset through the unit panel (under the same
two-day window the definitions declare), the definitions reader, the unit-day
artifact reader and a replayed portable checkpoint, and requires each to retain
the unit-summary oracle's state, registration and interval exactly.

The informative-prior row uses the existing approximate Normal likelihood
on log lift, including for conversion metrics; it is not a binomial
posterior or a finite-sample sampling guarantee. The three enumerated
prior cases compare posterior bounds, probabilities, metadata and row
identity across all five applicable constructors on DuckDB and PostgreSQL.
See [Priors and Bayesian decisions](../guides/priors-and-decisions.md) for
binary boundary behavior and the distinction from prior-free sampling.

Binary sequential monitoring (either automatic route) composes as follows, measured on
`from_unit_summary` and pinned per axis. Multiplicity: every in-family secondary on the
metric/arm axis -- Bernoulli, scalar-mean, or a mix of both -- is registered into the
family at a nominal per-cell allocation and selected by e-BH at the plan's `q`
(supported); a selected member's interval is reinverted at `min(q * R / m, nominal_alpha)`,
and the family reports one shared `family_guarantee` (`finite_sample` only when every
member is exact, `asymptotic_sequential` when any member is asymptotic). This is
unrelated to the registered breakout/segment axis, whose own correction is declared per
registration: a continuous (scalar-mean) breakout registers `correction="bonferroni"` and
keeps fixed per-cell Bonferroni (FWER, no reselection); a discrete (Bernoulli) breakout
registers `correction="bh"` and is selected by the same e-BH-at-`q` family as an ordinary
metric/arm secondary, reinverted the same way. This is also unrelated to guardrails, which test at the
full plan `alpha` outside every family. CUPED: a pre-period coefficient and centre declared through
`InferenceSpec.adjustments` is supported on the asymptotic route; an in-experiment
coefficient is refused on the exact route (`sequential.transform.unpredictable`) because
it re-weights past increments with information unavailable when they were revealed, a
construction this package does not have; the asymptotic route retains the joint moments
instead. Clustered assignment: refused before any observation is read
(`sequential.route.unsupported`); no cluster-robust sequential boundary is built.
Winsorization: it does not apply to a 0/1 outcome, so a binary metric refuses a
percentile or fixed clip at declaration (`frame.metric.winsorization_applies_type`); on a
mean metric a fixed clip is supported and a percentile clip refuses
(`sequential.transform.unpredictable`) because its threshold depends on accumulated data.
Quantile metrics: refused on the exact route (`definition.inference.always_valid_metric_type`)
because the Bernoulli law does not describe them; use fixed-horizon inference (valid for one planned analysis, not repeated looks).
Breakouts: `InferenceSpec.segments` fixes one string-valued dimension and its
allowed levels before outcomes on `from_unit_summary` and `from_unit_panel`;
an explicit registration is optional. Empty levels stay in the roster.
Asymptotic segment membership must be determined before assignment and uses
the existing fixed-roster Bonferroni construction, not BH. Segment labels are
canonical strings on both frame routes: `true`/`false` for Booleans and
`__null__` for missing values, so summary and panel sources label the same
units identically. A checkpoint recorded under an earlier raw spelling (`True`,
`None`) replays unchanged but cannot be continued by a canonically labelled
source (COMBINATION -- earlier recorded prefix with a different roster
spelling; `sequential.continuation.rewrite`). This is a stored-state boundary, not an
inherent limitation; recovery is in [Persisted formats](../api.md#persisted-formats).
`from_definitions` and `from_unit_day_artifact` still refuse segmented capture
(`sequential.route.unsupported`). A moments cube can replay retained segment
state but has no breakout catalog, so `run_breakout` refuses
(`readout.source.dimension`). Switchback: refused before any frame is read
(`source.frame.switchback.plan`, reason `unsupported_inference`) for every sequential kind,
automatic or explicit; the switchback contrast has no sequential construction.
Continuous intent-to-treat under an encouragement design registers under the public asymptotic
mean law exactly as a randomized design does (`sequential.route.unsupported` still names the
Bernoulli route on any other mechanism).

### Stored-state compatibility by entry point

These rows describe what happens when stored state meets the current reader, or the artifact store's recovery path. Recovery steps are in [Persisted formats](../api.md#persisted-formats).

| Capability | Unit summary | Unit panel | Switchback panel | Definitions | Unit-day artifact | Portable moments |
|---|---|---|---|---|---|---|
| Continuing a checkpoint recorded under earlier raw Boolean/null spellings (`True`, `None`) with canonically labelled units | refused (COMBINATION -- earlier prefix by canonical source, `sequential.continuation.rewrite`); the same string spellings still continue, and the checkpoint replays unchanged | refused (COMBINATION, `sequential.continuation.rewrite`) | refused (SOURCE -- no sequential construction) | refused (SOURCE, `sequential.route.unsupported`) | refused (SOURCE, `sequential.route.unsupported`) | replay only (SOURCE -- no per-unit source to continue from) |
| Artifact context format 2 (hash-only source recipes; no source SQL) | not comparable (SOURCE -- no warehouse artifact) | not comparable (SOURCE -- no warehouse artifact) | not comparable (SOURCE -- no warehouse artifact) | yes, publishes format 2 | yes, reads format 2 only; a format-1 context is refused (COMBINATION -- stored older artifact by current reader, `artifact.format.unsupported`) and must be republished from trusted definitions | not comparable (SOURCE -- moments cubes carry no artifact context) |
| Experiment window days (`start_day`, `end_day`, `observation_horizon_day`) follow `day_boundary`: naive value = wall clock at the boundary, aware value converted | not comparable (SOURCE -- `from_unit_summary` accepts no `Experiment` declaration, so there is no window to place) | not comparable (SOURCE -- same; a frame has no window lever) | not comparable (SOURCE -- same) | yes; a native sequential registration made before this change is refused (COMBINATION -- earlier registration by current recipe, `sequential.source.invalid`) and must be re-registered | yes, the context binds `window_days`; a context without it is refused (COMBINATION -- stored older artifact by current reader, `artifact.context.mismatch`) and must be republished | not comparable (SOURCE -- moments cubes carry no experiment window) |
| Publication abort after the manifest insert began invalidates the generation (tombstone first, then relation erase; reads refused with `artifact.generation.dropped`; if the tombstone cannot be written the relations are kept and named, and `abandon_generation` from a fresh process hides the generation before they are dropped) | not comparable (SOURCE -- no artifact store) | not comparable (SOURCE -- no artifact store) | not comparable (SOURCE -- no artifact store) | yes, `WarehouseArtifactStore` publication | yes, reads of a dropped generation refuse | not comparable (SOURCE -- no artifact store) |
