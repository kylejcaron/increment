# Statistical limitations

Every number this library reports rests on assumptions. This page puts them in
one place, so you can see what a result assumes without reading the source.

Most entries mark deliberate limits: a method that is standard but
approximate, an estimand that is narrower than it looks, or a guarantee that
holds under a condition the library cannot check for you. A few identify gaps:
a safeguard or diagnostic that one entry point provides and a neighboring one
does not. Where the package offers a better route, this page names it.

Read the section that matches the decision you are making. Each entry says what
the library does, what it assumes, and when the assumption matters.

## What runs where

A capability is a claim only when its path is named. `Analysis` has six entry
points in three families: **dataframe** (`from_unit_summary`, one row per unit;
`from_unit_panel`, one row per unit per day; `from_switchback_panel`);
**warehouse** (`from_definitions`, compiling from raw event facts;
`from_unit_day_artifact`, reading unit × day relations previously published
into the warehouse); and **portable** (`from_moments`, a file of pre-reduced
moments exported from any of the others).

The warehouse is the primary route. The table records measured behavior on
every path, using the dataframe unit-summary path as the oracle against which
the others are checked.

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
| Conversion/retention metric under asymptotic sequential inference | yes | yes | refused | yes | yes | checkpoint replay |
| Conversion/retention metric under automatic exact Bernoulli monitoring (`always_valid` without a registration) | yes | yes | refused | yes | yes | checkpoint replay |
| Automatic exact multi-arm conversion/retention monitoring | yes | yes | refused (SOURCE -- no sequential construction) | yes | yes | checkpoint replay |
| Randomized registered segmented sequential family (automatic predeclared levels or explicit roster) | yes | yes | refused (SOURCE -- no sequential construction) | refused (SOURCE, `sequential.route.unsupported` -- relational capture has no immutable segment-property contract) | refused (SOURCE, `sequential.route.unsupported` -- artifact capture has no immutable segment-property contract) | retained-state replay only; breakout readout refused (SOURCE, `readout.source.dimension` -- no breakout catalog) |
| Boolean and null segment labels in a registered segmented sequential family (canonical `true`/`false`/`__null__`, matching the string-labelled oracle) | yes (previously `missing_arm` on summary; measured on pandas, Polars and Arrow Boolean columns, automatic and explicit rosters) | yes (same three column kinds) | refused (SOURCE -- no sequential construction) | refused (SOURCE, `sequential.route.unsupported`) | refused (SOURCE, `sequential.route.unsupported`) | retained-state replay only; breakout readout refused (SOURCE, `readout.source.dimension`) |
| Continuing a checkpoint recorded under earlier raw Boolean/null spellings (`True`, `None`) with canonically labelled units | refused (COMBINATION -- earlier prefix by canonical source, `sequential.continuation.rewrite`); the same string spellings still continue, and the checkpoint replays unchanged | refused (COMBINATION, `sequential.continuation.rewrite`) | refused (SOURCE -- no sequential construction) | refused (SOURCE, `sequential.route.unsupported`) | refused (SOURCE, `sequential.route.unsupported`) | replay only (SOURCE -- no per-unit source to continue from) |
| Artifact context format 2 (hash-only source recipes; no source SQL) | not comparable (SOURCE -- no warehouse artifact) | not comparable (SOURCE -- no warehouse artifact) | not comparable (SOURCE -- no warehouse artifact) | yes, publishes format 2 | yes, reads format 2 only; a format-1 context is refused (COMBINATION -- stored older artifact by current reader, `artifact.format.unsupported`) and must be republished from trusted definitions | not comparable (SOURCE -- moments cubes carry no artifact context) |
| Experiment window days (`start_day`, `end_day`, `observation_horizon_day`) follow `day_boundary`: naive value = wall clock at the boundary, aware value converted | not comparable (SOURCE -- `from_unit_summary` accepts no `Experiment` declaration, so there is no window to place) | not comparable (SOURCE -- same; a frame has no window lever) | not comparable (SOURCE -- same) | yes; a native sequential registration made before this change is refused (COMBINATION -- earlier registration by current recipe, `sequential.source.invalid`) and must be re-registered | yes, the context binds `window_days`; a context without it is refused (COMBINATION -- stored older artifact by current reader, `artifact.context.mismatch`) and must be republished | not comparable (SOURCE -- moments cubes carry no experiment window) |
| Publication abort after the manifest insert began invalidates the generation (tombstone first, then relation erase; reads refused with `artifact.generation.dropped`; if the tombstone cannot be written the relations are kept and named, and `abandon_generation` from a fresh process hides the generation before they are dropped) | not comparable (SOURCE -- no artifact store) | not comparable (SOURCE -- no artifact store) | not comparable (SOURCE -- no artifact store) | yes, `WarehouseArtifactStore` publication | yes, reads of a dropped generation refuse | not comparable (SOURCE -- no artifact store) |
| `increment.impute.pooled_mean` with declared `roles=`; `MetricSpec(missing="impute")` | roles honored: an `outcome` role refuses (CONSTRUCTION -- filling an outcome with its pooled mean shrinks variance, `impute.pooled_mean_outcome`); an undeclared role warns (`impute.pooled_mean_role_undeclared`); `MetricSpec(missing="impute")` refuses (CONSTRUCTION, `frame.metric.missing_impute`) | same | same | not comparable (SOURCE -- a warehouse `Metric` cannot express `missing="impute"`; the helper prepares dataframes only) | not comparable (SOURCE -- same) | not comparable (SOURCE -- no per-unit rows) |
| Encouragement ITT, compliance and LATE, fixed horizon, declared via `design:` | yes | yes | refused (SOURCE -- `from_switchback_panel` takes no `design=`) | yes | yes | yes |
| Explicit ITT/compliance without a LATE exclusion declaration (fixed horizon or supported sequential monitoring) | yes | yes | refused (SOURCE -- `from_switchback_panel` takes no `design=`) | yes | yes | yes; sequential checkpoint replay |
| Fixed-horizon design-level compliance with an empty outcome catalog | yes, `metrics=[]` | yes, `metrics=[]` | refused (SOURCE -- no `design=`) | yes | yes | yes, format-8 `design_summary` and `metrics=[]` |
| Encouragement guardrail with a declared non-inferiority margin | yes, ITT row carries the margin verdict; refused (CONSTRUCTION, `readout.encouragement.margin`) if a request excludes `itt` | yes, ITT row carries the margin verdict; refused (CONSTRUCTION) if `itt` is excluded | refused (SOURCE -- no `design=`) | yes, ITT row carries the margin verdict; refused (CONSTRUCTION) if `itt` is excluded | yes, ITT row carries the margin verdict; refused (CONSTRUCTION) if `itt` is excluded | yes, ITT row carries the margin verdict; refused (CONSTRUCTION) if `itt` is excluded |
| Encouragement BH-selected secondary's LATE row re-estimated at the corrected level | yes | yes | refused (SOURCE -- no `design=`) | yes | yes | yes |
| Encouragement ITT on a conversion/retention metric keeps the exact binomial route | yes | yes | refused (SOURCE -- no `design=`) | yes | yes | yes |
| Encouragement conversion ITT breakouts retain exact confidence sets at zero control counts | refused (SOURCE -- no declared breakout dimension) | yes | not comparable (SOURCE -- no switchback schedule) | yes | yes | refused (SOURCE -- no per-unit breakout dimension) |
| Bounded-retention breakouts with mature exposure cohorts (`audit-retention-breakout-cohorts`) | refused (SOURCE -- no exposure-date cohort stream, `source.frame.constructor`) | yes, `run_breakout` | not comparable (SOURCE -- no switchback schedule) | yes, `run_breakout` and cohort-aware `breakout_summaries` | yes, `run_breakout` and cohort-aware `breakout_summaries` | refused (SOURCE -- no breakout operation, `facade.analysis.operation`) |
| Encouragement continuous ITT + Bernoulli uptake compliance, jointly monitored (composed sequential family, e-BH selected) | yes | yes | refused (SOURCE -- no `design=`, no sequential construction) | yes | yes | checkpoint replay |
| Uptake-only sequential compliance with an empty outcome catalog (`always_valid` and a compliance policy) | yes, `metrics=[]` | yes, `metrics=[]` | refused (SOURCE -- no `design=`, no sequential construction) | yes | yes | checkpoint replay with `metrics=[]` |
| Observational IPTW covariate adjustment | yes | yes | refused (SOURCE -- no `design=`) | yes | yes | refused (SOURCE -- a moments cube carries no per-unit rows to attach a covariate to) |
| Observational AIPW/DML covariate adjustment | yes | yes | refused (SOURCE -- no `design=`) | yes | yes | refused (SOURCE -- same as above) |
| Categorical observational adjustment (string covariates) | yes | yes | refused (SOURCE -- no `design=`) | yes | yes | refused (SOURCE, `source.moments.covariate_unavailable` -- no per-unit covariates) |
| Categorical nulls with a nondefault missing-value policy | yes | yes | refused (SOURCE -- no `design=`) | default missing-value refusal; YAML does not accept policy overrides | default missing-value refusal inherited from definitions | refused (SOURCE -- no per-unit covariates) |
| Observational primary + secondary + guardrail together (role multiplicity) | yes | yes | refused (SOURCE -- no `design=`) | yes | yes | refused (SOURCE -- same as above) |
| Quantile inference on tied/rounded outcomes | yes | yes (unwindowed only -- the panel's per-unit collapse is the same per-unit total mean/ratio already use; a windowed quantile metric can never be declared on the frame path at all, `frame.metric.window_days_supported`) | refused (SOURCE -- a quantile metric is refused at the metric-type gate before any frame is read) | yes | yes | refused (SOURCE, `source.frame.quantile_no_moments` -- a quantile has no moments representation) |
| Clustered ratio metric with a non-positive control-arm numerator | yes | refused (SOURCE -- `from_unit_panel(cluster=...)` refuses unconditionally; build clustered arm totals with `from_unit_summary(cluster=...)`) | refused (SOURCE -- a clustered arm-lift ratio has no switchback contrast shape) | yes | yes | refused (SOURCE, `source.moments.cluster_grain` -- clustered moments transport needs a complete encouragement compliance payload) |
| Clustered zero relative variance retains its point and any available additive interval | yes | refused (SOURCE -- no clustered arm moments; use `from_unit_summary`) | refused (SOURCE -- a switchback contrast has no arm-cluster shape) | yes | yes | refused (SOURCE, `source.moments.cluster_grain` -- clustered moments transport needs a complete encouragement compliance payload) |
| `Analysis.planning_baseline`, mean metric | yes | yes | yes (the pilot-fitted `SwitchbackBaseline` for the `switchback_*` solvers) | yes | yes | yes |
| `Analysis.planning_baseline`, quantile metric | yes | yes (same unwindowed-only reach as the row above) | refused (SOURCE -- a quantile metric is refused at the switchback metric-type gate before any frame is read) | yes | yes | refused (SOURCE, `source.frame.quantile_no_moments` -- no per-unit rows to draw a pilot sample from) |
| `Analysis.planning_baseline` with a declared cluster (per-unit mean/var; ICC on the metric's per-unit score) | yes | refused (SOURCE -- `from_unit_panel` takes no cluster) | refused (SOURCE -- a switchback contrast has no arm cluster) | yes | yes | refused (SOURCE, `source.moments.cluster_grain` -- clustered moments are not exported) |
| `Analysis.planning_baseline` under a declared trigger (triggered-population moments and `trigger_rate`) | refused (SOURCE -- `from_unit_summary` accepts no `Experiment` declaration, so `planning_baseline`'s `trigger` always resolves `None`; the assigned population is analyzed instead) | refused (SOURCE -- same as `from_unit_summary`) | refused (SOURCE -- same as `from_unit_summary`) | yes | yes with the `trigger_population` and `assignment_counts` extensions; otherwise refused (SOURCE, `analysis.planning_baseline.trigger_evidence_unavailable`) | refused (SOURCE -- same as `from_unit_summary`) |
| Logged-policy contrast (`estimate_policy_contrast`) | refused (SOURCE -- no `Analysis` constructor carries a decision trace: the per-decision chosen action, exact logged propensity, logging policy version and reward boundary; the family's own ingress is `LoggedTrace.from_records` / `from_frame`) | refused (SOURCE -- same) | refused (SOURCE -- same) | refused (SOURCE -- same) | refused (SOURCE -- same) | refused (SOURCE -- same) |
| TabularPolicy persistence (`model_dump_json` / `model_validate_json`) | not comparable (SOURCE -- no `Analysis` constructor carries a policy table; the policy is a `LoggedTrace` input) | not comparable (SOURCE -- same) | not comparable (SOURCE -- same) | not comparable (SOURCE -- same) | not comparable (SOURCE -- same) | not comparable (SOURCE -- same). Within the family: JSON roundtrips string-keyed tables only; any other key type refuses (CONSTRUCTION -- JSON object keys are strings, `logged_policy.policy.json_context_key`); typed keys persist through `model_dump(mode="python")` or trusted pickle |
| Metrics added after the plan: `Analysis.available_metrics` and `exploratory_metrics=` on `run`, `run_breakout`, `run_asof_lift`, `run_asof`, `run_daily` | refused (SOURCE, `facade.analysis.exploratory_metrics_source_limited` -- no saved definitions to add from; use `Analysis.from_definitions`) | refused (SOURCE, same code) | refused (SOURCE, same code) | yes, `role="exploratory"` under the default unassigned procedure; refused under registered sequential inference (`facade.analysis.exploratory_metrics_sequential` -- an added metric is chosen after data are visible, so no registered model covers it) | refused (SOURCE, same code) | refused (SOURCE, same code) |

For `from_definitions`, pinning preserves dimensions joined by freshness-bearing
fact sources. Qualifying non-enrolled events can establish that later data has
arrived without entering the enrolled outcome population. In `from_unit_panel`,
natural day labels such as `d9` and `d10` use numeric order when inferring
maturity, matching the same explicitly declared observation bound.
Windowed ratios use the earlier numerator/denominator observation bound for
totals and completed as-of cohorts. An entirely null component under
`missing="zero"` retains the panel-extent fallback; an explicitly supplied
`observation_end` remains authoritative.

An added metric is read through the same estimators as a declared one: CUPED, ratio
and winsorized metrics follow the call-wide `decision_method` you pass (the metric has no
declared binding, so the default unadjusted estimator applies), clustering and breakouts
follow the experiment's declaration, and the day-axis refusals (clustered, winsorized,
quantile, dimensioned `run_daily` of an undeclared metric) keep their codes. It joins
no plan family and never changes a declared row; under a breakout multiplicity plan its
segment cells form a family of their own.

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

Mean/Ratio CUPED on the unit panel resolves the declared covariate the same
way observational adjustment already does: through the per-unit value
`unit_frame()`/`moments(grain="total")` join in, constant across a unit's
own rows, refused by name (`frame.frame_panel.unit_covariate_varies`) when
it genuinely varies. A windowed or retention metric still refuses a CUPED
covariate (CONSTRUCTION, `frame.validation.from_unit_panel`): those metrics
have no per-unit collapse for that value to attach to; the refusal names
`from_unit_summary` as the route that does.
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
the other." The `from_switchback_panel` SOURCE refusal on the encouragement,
observational and quantile rows above is structural, not a gap: that
constructor accepts only a switchback-shaped frame and an
`identification: Randomized` design, so it never reaches an `Encouragement`,
`Observational`, or quantile-metric case built for the other five paths.

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
See [Priors and Bayesian decisions](guides/priors-and-decisions.md) for
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
because the Bernoulli law does not describe them; use fixed-horizon inference.
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
spelling; `sequential.continuation.rewrite`). That is a correction of an earlier
inconsistency, not an inherent limitation.
`from_definitions` and `from_unit_day_artifact` still refuse segmented capture
(`sequential.route.unsupported`). A moments cube can replay retained segment
state but has no breakout catalog, so `run_breakout` refuses
(`readout.source.dimension`). Switchback: refused before any frame is read
(`source.frame.switchback.plan`, reason `unsupported_inference`) for every sequential kind,
automatic or explicit; the switchback contrast has no sequential construction.
Continuous intent-to-treat under an encouragement design registers under the public asymptotic
mean law exactly as a randomized design does (`sequential.route.unsupported` still names the
Bernoulli route on any other mechanism).

## What the intervals assume

### Uncertainty is estimated on every path

No estimator here is handed a known variance; every one estimates uncertainty from the
data. What differs is the approximation each family layers on to get there:

| Family | How uncertainty is obtained |
|---|---|
| Conversion and retention arm lift (unadjusted, unit-grain, no informative prior) | Exact independent-binomial risk-ratio test inversion (Berger-Boos restricted-nuisance construction) -- no delta method, no Normal reference |
| Unit-grain mean/ratio lift and CUPED, no informative prior | Delta-method log-scale uncertainty against a Welch-Satterthwaite t reference, whose degrees of freedom are persisted on the row |
| Clustered unadjusted arm lift | Joint additive/control covariance, Fieller relative set; separate additive uncertainty |
| Sequential | Separate registered observation-model contract; fixed-horizon approximations do not establish anytime validity |
| Quantiles | Inversion of an order-statistic bracket (Woodruff), not a delta method |
| Switchback | Independent unit-cycle orders: default `UnitCycleTApproximation` (sample variance, at least two units, `dof = n_units - 1`); a valid prospective `UnitCycleVarianceEnvelope` is a separate stronger route supporting one unit. Shared schedules: block-t on at least two blocks |
| Observational adjusted methods | Influence-function sandwich |

Estimating the variance does not by itself forfeit finite-sample validity — under
parametric assumptions it need not. But it does mean each family's interval inherits the
quality of its own approximation, and those are not interchangeable. The gap matters most
at small n, for heavy-tailed outcomes, and for ratio metrics, where a first-order
approximation is furthest from the truth.

Absolute effects on the switchback and observational paths are estimated natively on the
additive scale rather than reconstructed from a log-scale estimate, so they carry their own
path's approximation and not an additional one.

Approximate inference paths do not promise finite-sample exactness for a discrete outcome.
Eligible fixed-horizon conversion/retention results with `reference_kind="binomial"`
retain a finite-sample guarantee for the relative-risk confidence set and its
relative-scale decisions. Their additive `abs_lb`/`abs_ub` sidecar, when available,
uses a Normal-Wald approximation; that interval is not exact.
Asymptotic and empirically qualified methods are labeled as such rather than treated as
exact; an unfinished or unsupported capability refuses before producing a result.
No result field diagnoses model misspecification.
For approximate results, no field certifies calibration. A procedure declares a coarse
sampling floor — two units per arm for most arm inference, twenty for a quantile.
Cluster methods require estimable independent contributions, not a universal
ten-cluster admission rule. A feasibility minimum is not a calibration guarantee.

It is not the estimator's own feasibility check either, and the two can disagree in both
directions. The quantile order-statistic bracket needs a sample that depends on the
requested quantile and the alpha it is evaluated at: a median at 95% is feasible on six
units per arm, well under the declared twenty, while a 99th percentile needs 368.

The quantile interval stays valid as the recording grid coarsens relative to the
sample size, rather than refusing on ties: while an arm's order-statistic bracket
holds a tie and does not yet span a dozen repeated values, the reported interval
contains that bracket, so the bracket's distribution-free coverage carries over
whatever the grid (values recorded off it, prices ending in both .99 and .00). The
price is width, about 1.1 to 3 times the classical formula's on coarse grids
(milliseconds, seconds or low-value cents at large sample sizes; counts). It has no
fixed bound, because a bracket collapsed onto one value has zero classical width, so
each widened row states its own ratio in its note.

This feasibility bound applies to a single reported interval, evaluated at one caller alpha.
The p-value computation (see below) scans across alpha internally and never surfaces this
refusal to the caller: past the infeasible boundary it treats the construction as "does not
exclude the null" and stops searching there, rather than raising.

Calibration is metric-specific for approximate methods. Unadjusted unit-grain
conversion/retention uses the exact binomial method below at every event count.
CUPED and ratio-denominator routes retain their delta approximation. Clustered
unadjusted arm lift instead retains joint covariance and relative-set geometry;
that representation prevents misleading finite intervals near zero controls,
but does not establish finite-sample coverage for rare events or few clusters.

A unit-grain ratio row (unadjusted or CUPED, fixed horizon) carries a
`ratio_denominator_precision` note when either arm's denominator mean is resolved to a
relative standard error `sqrt(var_d / n) / d_bar` above 0.15. This is an advisory
qualification, not a refusal or a corrected interval: the interval is emitted unchanged.
The threshold comes from `calibration/ratio_precision.py`, a seeded coverage
grid over sample sizes 20-2000 and lognormal, exponential and gamma denominators with an
independent numerator. Every cell covering below 93% at nominal 95% exceeds it in most
draws, and no cell covering at least 94.5% does. The note names the arm, the statistic and
the threshold. The unchanged interval is still anticonservative under a right-skewed
denominator (85.5% coverage at n=20 and 89.6% at n=50 under lognormal(sigma=1.5)).
The statistic sees only the denominator's own precision: a numerator that moves with the
denominator covers close to nominal even when flagged, and a numerator independent of it
undercovers most. The carried moments end at second order, so the heavy-tail correction
and broader small-sample coverage remain unresolved.

### The switchback t reference is itself an approximation

Independent unit-cycle orders use `UnitCycleTApproximation()` by default
(equivalent to passing it) unless a valid prospective `UnitCycleVarianceEnvelope`
is supplied; the shared-block reference is separate. Both use the independent
contributions' sample variance. That is not a claim of finite-sample coverage, and the
approximation is not confined to discrete outcomes.

Under independent unit-cycle orders, each mean-outcome contribution is inverse-probability
weighted, so it takes
one of only two values, and those two are asymmetric whenever the declared CT/TC probability
is not 0.5. The per-unit deltas are means of such contributions, so the t reference relies on
those means being approximately Normal — a condition that is not among the contract's
declared assumptions, which name only `no_residual_carryover_after_discarded_steps` and
`independent_units`. Coverage therefore degrades with few units, with an assignment
probability far from 0.5, and with skewed, heteroskedastic, or high-leverage outcomes, and the
declared floor admits as few as two units — the regime where the approximation is weakest. The
asymmetric/skew calibration includes a small-N counterexample to nominal coverage.
Unit-cycle variance envelopes instead give a finite-sample bound under the declared
prospective residual-variance assumption, including at one unit. That assumption
must be justified externally; a fitted pilot variance is not a valid envelope. Shared schedules instead apply
`switchback_block_t` with a `block_t` reference to block contributions; their floor is two
independent blocks, even with a one-unit roster. More roster units do not increase the
replication count or establish finite-sample coverage.

### Cluster references are qualified working approximations

Disjoint unadjusted arms retain each arm's cluster-ratio uncertainty. The relative
Fieller set uses a fixed t reference with `min(K_treatment - 1, K_control - 1)`
degrees of freedom; the additive interval separately uses Welch–Satterthwaite.
Shared-arm dependence retains the complete response-corrected covariance.
Observational IPTW/AIPW/DML use their joint influence-function covariance and an
asymptotic Normal reference. Neither a Bessel multiplier nor a generic t choice
establishes finite-sample validity for arbitrary small, unbalanced, or skewed clusters.
Encouragement LATE and sitewide impacts retain their own component-based references.
The cluster-asymptotic justification also needs no cluster to dominate its arm's
variance. Neither a large total row count nor crossing the 40-cluster warning
threshold establishes that condition. Concentrated cluster masses can bias the
disjoint-arm variance estimate downward; a t reference alone does not correct it.

The clustered encouragement LATE cuts its additive interval at a Welch–Satterthwaite t
reference over the two arms' own cluster-ratio variance components. Measured by
`tests/calibration/test_cluster_late_welch_df.py` (2000 replications per cell over 5/5,
5/25 and 10/50 treatment/control clusters, treatment between-cluster variance 1, 4 and 16
times the control's, mean cluster size 20; binomial Monte-Carlo standard error 0.5 points
at 95% and 0.9 points at 82%): with equal cluster sizes the interval covers the true LATE
94.3–96.0% at nominal 95% in every cell, including 94.55% at 5/25 clusters with 16 times
the variance in the small arm, where a pooled `t_(K_treatment + K_control - 2)` reference
over the same components would cover 89.15%. Log-normally dispersed cluster sizes (sigma 0.75)
cost about 3–6 points that the reference does not recover: 88.5–92.7% across those nine
cells, 89.7% at 5/25 with the 16x ratio, against 82.0% pooled. The first-stage gate withheld
at most 4 of 2000 replications in any cell.

`relative_confidence_set` preserves bounded, disconnected, one-sided, full-real,
empty, or unavailable geometry. A zero control mean can leave a confidence set
without a finite point. Numeric nulls are not infinities. A genuinely indefinite
or unrepresentable joint covariance sets `relative_unavailable_reason`; additive
output survives rather than being replaced by a clipped covariance or fabricated
relative precision. Scalar posterior decision probabilities are not recovered
from a joint frequentist set.

Representative release checks are diagnostics, not certification of the original
stress matrix. Run the preserved cases and paired-width criteria explicitly:

```bash
uv run python -m calibration.cluster_diagnostics --output /tmp/cluster-diagnostics --repetitions 4
uv run python -m calibration.cluster_diagnostics --selection full --output /tmp/cluster-stress
```

Output directories must be new. Budgets are bounded at one hour; started,
completed, refused, failed, unfinished, and unstarted work remain distinguishable.
The full route preserves original seeds, repetition rules, and paired-width
thresholds. Interrupted or diagnostic prefixes never pass those historical gates.

### CUPED treats its adjustment coefficient as known

The CUPED standard error conditions on a theta estimated from the same data it adjusts. The
uncertainty in that estimate is not propagated, so the reported interval is slightly
narrower than one holding the coefficient fixed. Measured by
`tests/calibration/test_cuped_theta_uncertainty.py` (16 cells: covariate correlation 0.1 to
0.9 by 20 to 2000 units per arm, equal allocation, one homogeneous slope, 3000 replications
each, intervals cut at the row's Welch–Satterthwaite t reference; binomial Monte-Carlo
standard error 0.4 points): the fitted interval is 1.3–1.4% narrower on the log scale than
a known-coefficient comparator at 20 units per arm, 0.5% at 50, 0.1% at 200 and 0.01% at
2000, the same at every covariate correlation, and within Monte-Carlo error of
`sqrt(1 - 1/(N - 2))` with `N` the total sample. Coverage of the fitted interval is
93.4–95.3% at nominal 95% across the 16 cells, against 93.8–95.6% for the comparator, and
the reported standard error runs 0.94–1.01 times the empirical spread of the log estimate
(0.95–0.98 at 20 per arm). The comparator is the same log-scale delta method with the true
coefficient substituted, not exact finite-sample inference, so the width ratio measures the
residual-variance shrinkage from fitting theta and does not isolate the added sampling
variance of the estimate. Unequal allocation and arm-specific slopes are outside that design.

## Metric types with no supported estimand

### Rare events on an unadjusted conversion/retention arm are estimated exactly, not refused

An unadjusted (no CUPED, no declared cluster), unit-grain conversion or retention arm pair
uses an exact independent-binomial risk-ratio method (Berger-Boos restricted-nuisance test
inversion; see `increment/estimation/binomial_rr.py`), not the log-scale delta method. It
handles every event count directly, with no `log_se >= 0.5` admission rule and no
continuity correction:

An Encouragement design's ITT on a conversion/retention metric keeps
this route: the design's uptake (first-stage compliance) moments are a
different random variable over the same units and do not change the
ITT's own sufficient statistics, so they are stripped before this route
reads the arm rather than disqualifying it.

| Observations | Point estimate | Confidence set |
|---|---:|---|
| Both arms have events | Empirical ratio | Finite two-sided (or one-sided-plus-unbounded-ceiling) set |
| Treatment has zero events | Exactly -1 (total loss) | Finite two-sided set, lower endpoint exactly -1 |
| Control has zero events | **Unavailable** (the empirical ratio's denominator is zero) | A typed, point-less confidence set: finite lower endpoint, genuinely unbounded ceiling |
| Both arms have zero events | **Unavailable** | The full relative-lift support, `[-1, +inf)` |

A row with no finite point (`LiftEstimate.lift is None`) still carries a typed
`LiftEstimate.binomial_set` (a `BinomialConfidenceSet`: sufficient counts, the frozen
nuisance budget, and finite/unbounded endpoints on the relative-lift scale) and answers
`stat_sig()`/`p_value()` exactly from it — point availability, confidence-set availability,
and reportability are three distinct, independently tracked properties; a point-less row is
never a failed one. An unbounded ceiling is represented as `upper=None` on the set, never as
a serialized infinity.

This exact method is fixed-horizon: it does not itself prove validity under optional
stopping or repeated peeking (see the sequential-inference limitations elsewhere on this
page for that separate guarantee). A conversion/retention request with an informative
prior uses the established Normal approximation. A registered sequential specification
instead uses its declared raw-observation likelihood and matching sequential inversion,
subject to the registered sampling and finalized-window contract.

It also has two further boundaries, both refusals rather than silent degradation:

* **Arm size.** Each arm is capped at `binomial_rr.MAX_ARM_SIZE` (4,000,000). This is a
  compute-resource applicability boundary, not a statistical one: per-call cost keeps
  growing with arm size past it, so a call with either arm above the cap refuses
  immediately (`estimation.binomial.arm_too_large_for_exact_enumeration`) rather than
  running an increasingly expensive search. The cap is sized to admit every arm size this
  method is asked to support today; a workload with a genuinely larger arm would need a
  higher cap or a closed-form/recurrence tail evaluation (out of scope here).
* **Latency.** Cost grows with arm size: measured on commodity hardware
  (two-sided, `alpha=0.05`, cold), roughly 1.7s at 100,000 per arm and
  an extrapolated 6-9s at 1,000,000 (`MAX_ARM_SIZE`) -- down from an
  unoptimized ~13s and ~60s respectively, a measured 7-8x reduction from
  tightening the boundary-search iteration budget without weakening the
  certified interval in any tested regime, including rare events. A
  readout multiplies this across metrics, arms, and breakout cells.
  There is no opt-out: every eligible unadjusted conversion/retention
  contrast takes this route. A further speedup to sub-few-second
  latency at the largest admitted arm size would need a genuinely
  different tail-evaluation construction (a closed-form or recurrence
  update between adjacent risk-ratio candidates); none is implemented
  today. Reduce the number of eligible contrasts in a single call
  (narrow `metrics=`, run large-arm breakouts separately) if latency
  matters more than exactness at your arm sizes.
* **Extreme alpha.** The frozen nuisance tail budget passed to the Clopper-Pearson
  endpoint solver is `min(1e-6, alpha / 32)`. Below `1e-9` (i.e. `alpha < 3.2e-8`), SciPy's
  iterative beta-quantile solver has demonstrated large relative error against an exact
  oracle for small `n` in the validated regime, so the endpoint is refused
  (`estimation.binomial.tail_unrepresentable`) rather than certified outside that regime.
  Ordinary use (including FCR-adjusted alpha) is far above this floor.

CUPED and unit-grain ratio-denominator conversion/retention retain their log-scale
guards; their sufficient statistics are not raw Bernoulli count pairs. A clustered
unadjusted arm instead uses joint relative inference, without the scalar log-SE
admission rule. None of those approximate routes inherits exact-binomial validity.
The remaining scalar log-path refusal is:

```text
log-scale relative-lift inference is undefined (zero events, or a signed metric
that needs an absolute-scale estimand)
```

### Signed effects require a suitable scale and reference

Prior-free adjusted and clustered unadjusted routes preserve additive inference
and signed joint-relative sets. A negative control mean is allowed; a zero one
can leave no finite ratio point. Ordinary unclustered log-scale arm lift still
requires positive means. Absolute-scale estimation is supported directly by
observational adjusted methods, encouragement LATE, switchback contrasts,
and sitewide impact APIs; `Analysis.run(value_scale="absolute")` remains an
observational selector rather than a general randomized arm-lift option.

A randomized experiment is not shut out, though. `Analysis.sitewide()` derives an additive
effect directly from enrolled-arm moments, so it is a conditional randomized route for a
signed quantity. It is conditional on where the analysis came from: a definitions-backed
analysis carries the site-volume evidence it needs, and an artifact-backed one qualifies
when the published artifact includes a site-volume evidence extension. It also has its own
baseline requirements. Otherwise reach for one of the designs above, or compare a
transformed quantity that stays positive.

### A quantile's reported interval depends on the alpha you asked for

The quantile standard error is derived from an order-statistic bracket evaluated at the
caller's alpha, then divided by the corresponding normal critical value. Change the alpha —
which multiplicity allocation does — and the same data and the same quantile yield a
different standard error and a different reported interval. This is necessary, not a defect:
the bracket must be built at the caller's own alpha for the reported interval to cover at
that alpha; a fixed-reference-alpha SE, rescaled to other alphas, was measured to fail
coverage on the same grid that validates this construction.

The p-value used to decide significance does not share this dependence: it is computed once,
by inverting this same construction (the smallest alpha at which the construction's own
interval excludes the null), so it is a function of the data alone and does not move when a
multiplicity allocation changes the alpha assigned to the row. Do not compare quantile
standard errors or reported intervals across analyses that allocated different alpha
budgets — they are not on the same scale — but the p-value is comparable across them.
Fixed-horizon family selection uses that same inversion rather than reconstructing
a p-value from the alpha-dependent standard error. A null that is never excluded
has p-value exactly one, so every admissible family threshold leaves it unselected.

### A quantile metric's planned variance is a projection from the pilot, not a certified bound

`Analysis.planning_baseline(metric)` builds a `QuantileBaseline` from the pilot's own
control-arm values; at each candidate sample size the solver applies the runtime's own
half-width rule to a bracket projected from the pilot -- on continuous data the pilot's
classical order-statistic standard error scaled by the asymptotic `1/sqrt(n)` rate, on
rounded data the pilot's recorded distribution, averaged over where the quantile falls within
its recording cell -- since resampling at every candidate size the search visits is not
possible from a single pilot draw. The projection is close when the candidate size is a
modest multiple of the pilot's own size; extrapolating far past it, especially toward an
extreme quantile where the pilot has few points, inherits that region's own sampling noise
and can be off by a larger margin. This is a property of the SOURCE: a finite pilot is one
draw, not a certified estimate of the runtime's exact variance at an arbitrary future size.
A bigger or more representative pilot narrows it; planning a design still sizes an
experiment, it does not certify a final result. Measured directly (Monte Carlo, 1500
two-arm draws at the planned size): a 90th-percentile metric with a 5,000-unit pilot,
15% relative lift, and target power 0.80 achieved empirical power as low as 0.73 on one
draw -- a real, several-point shortfall the deterministic projection does not eliminate,
not merely simulation noise.

On rounded data the runtime's interval never narrows below the log gap from the quantile to
the next recorded value, however large the sample. Power therefore has a ceiling below one:
`required_sample_size` refuses a target above it with
`power.quantile_size_search_unreachable`, naming in `maximum_power` the largest power any
size reaches (`limiting_condition="recording_grid"`). This is a property of the SOURCE's
recording resolution; recording the metric more finely raises the ceiling.

## Design assumptions

### Switchback carryover is assumed away, not tested

The switchback contract's identifying assumption is
`no_residual_carryover_after_discarded_steps`. You declare a washout window, and a
`SwitchbackWindow.carryover_order` (validated `0 <= carryover_order < observation_steps`,
default `0`) names how many additional post-washout steps are still assumed contaminated.
The retained window is `step >= washout_steps + carryover_order`, of length
`observation_steps - carryover_order`. Declaring a nonzero order records the assumption
precisely and is applied by `from_switchback_panel`'s runtime reduction; it does not prove
that carryover has vanished.

Nothing verifies that `washout_steps + carryover_order` was long enough. The library
validates the logged schedule, but cannot establish that the scheduler was random or that
carryover has actually vanished by the retained window's start. If residual carryover
remains past that point, the contrast can be biased and the interval need not cover.

`from_switchback_panel` retains `step >= washout_steps + carryover_order` for declared orders
0, 1, and 2 (whenever enough observation steps remain) under both `IndependentBernoulliOrder`
and `SharedScheduleOrder`; declaring an order only records the assumption; nothing here proves
carryover actually vanished by that point.

### Switchback inference requires independent units or blocks

Under independent unit-cycle orders, the estimator averages all cycles within a labelled
unit and applies a t test
with `n_units - 1` degrees of freedom. This accommodates arbitrary serial correlation
*within* a unit, which is the point of the design. It assumes the units are independent of
each other, and that each unit's order was drawn independently.

Several common patterns bear on that assumption. What each one does to the reported interval
depends on the randomization law and on which target you are inferring about, and none of it
is characterized here:

- **Correlated schedules.** If unit orders are not drawn independently — a shared seed, or a
  balanced or stratified schedule — independent-unit inference is not valid. The direction of
  the error depends on both the randomization law and how units' potential outcomes respond to
  order, not the law alone: when units respond to order in the same direction, shared orders
  tend to induce positive covariance between contributions and balanced schedules tend to
  induce negative covariance (leaving the independent-unit standard error conservative); when
  units respond to order in opposite directions, these signs can reverse.
- **Shared time shocks.** Conditional on the outcomes actually realized, an additive shock
  shared across units does not couple contributions, because the only randomness is each
  unit's independent order draw. Whether such a shock matters for a target beyond the periods
  you ran depends on the mechanism: what makes that broader target random is a treatment
  effect that varies with the shock, not the shock level on its own. The library does not
  state which target it addresses.
- **Interference.** If one unit's assignment changes another unit's outcome, the estimator is
  no longer targeting the intended contrast, so the point estimate can be biased. This is
  fundamentally an identification failure, and there is no exposure-mapping facility to
  address it — but because one unit's outcome now depends on another unit's assignment, it can
  also couple unit contributions and invalidate the independent-unit standard error, so the
  reported interval is not reliably conservative either.
- **Geographic or network correlation of outcomes, without interference.** Nearby units
  having similar outcomes is not the same as one unit affecting another. On its own it does
  not bias the point estimate; as with a shared shock, whether it matters depends on the
  target.

A `SharedScheduleOrder` sequence declares a coarser randomization: one shared Bernoulli
CT/TC order per two-period block, drawn once for a fixed, complete unit roster rather than
independently per unit. `from_switchback_panel` reduces it with a block-level Student-t
reference (`n_blocks - 1` degrees of freedom): the independent replicate is the block, not
the roster unit, so a uniformly cloned roster changes neither the estimate nor its width.

Both laws target the mean-unit retained-window additive effect. Results record
`observation_steps` and `retained_steps=observation_steps-carryover_order`; the
total/conversion estimand labels explicitly name that retained window. A schedule
integrity check cannot establish either independent randomization or absence of
carryover. The original degenerate all-CT witness (`ct=320, tc=0` under a unit-cycle
declaration) now refuses before statistics are constructed. Earlier refusal of
unsupported shared declarations did not prevent de facto shared data from entering
the unit-cycle path. Correctly declared, nondegenerate shared schedules are supported.

### Sequential information fractions are unit counts, not Fisher information

Alpha-spending boundaries are computed at information fractions taken as cumulative
analyzed-unit fractions of the planned maximum. That equals the true information fraction
only when information is proportional to analyzed count: stable allocation, stable
variance, comparable independent increments.

Under drifting allocation or drifting variance the spending schedule is mis-timed, which
can over- or under-spend error — in exactly the non-stationary conditions that motivate
continuous monitoring in the first place.

### The sequential certification campaign has not been executed

The public Bernoulli likelihood route has a finite-sample anytime-valid
derivation under its registered model, assignment and reveal assumptions.
That argument and verification of the deployed numerical implementation are
separate obligations. Continuous-mean, adjusted-mean and ratio sequential
routes carry asymptotic qualifications; they do not inherit the Bernoulli
finite-sample guarantee.

The complete empirical campaign over the registered runtime manifest has
not been executed. Its recorded cost is months on twelve cores even after
the measured speedups; budgeted profiles deliberately cover only subsets.
The presence of this machinery is a specification of the check, not evidence
that its cells passed. Complete clustered and unit-cycle research matrices
likewise remain uncertified.

Release review for public sequential and difficult-cluster methods instead
uses a bounded claim audit, independent numerical oracles and preselected
risk-focused calibration, with unchanged per-case thresholds. Those checks
can identify concrete failures and establish evidence for their stated cases;
they cannot establish universal calibration, resolve unrun regimes, or turn
small-cluster/asymptotic approximations into finite-sample guarantees.
Incomplete or insufficiently precise checks remain unverified. This narrower
release evidence requirement does not claim completion of the full campaigns
or change the separate unit-cycle research obligations.

### Off-policy evaluation of logged decisions is calibrated only under a fixed logger

`estimate_policy_contrast` evaluates a target policy against a reference policy from a
logged decision trace with a trajectory-level self-normalized importance-weighted
estimator, cumulative likelihood ratios, and a unit-clustered sandwich interval on a `t`
reference. The interval is asymptotic in the number of units. It runs on no `Analysis`
path (SOURCE: none carries a decision trace); its ingress is `LoggedTrace.from_records`
/ `from_frame`, and the [guide](guides/logged-policy.md) states the estimand.

Admission requires every positive registered or recorded logging probability, chosen or not,
to meet the declared `0.05` floor (`logged_policy.trace.propensity_below_floor`).
Exact zero support is allowed only where the evaluated policies also assign zero
mass. A sub-floor positive probability violates the promised importance-weight
bound; it does not imply mathematically unbounded weights.
The recorded-law constructor is an alternative to registry reconstruction,
not a certificate of logger truthfulness or permission to relabel adaptive data.

A trace logged under more than one policy version is refused (CONSTRUCTION,
`logged_policy.inference.adaptive_logging_unsupported`). Measured before that gate
existed, with the same estimator on batch-refit adaptive loggers at horizon 5 under
context dependence 0.8 (`tests/calibration/test_logged_policy_adaptive.py`'s design, 400
replications at 200 units): an epsilon-greedy logger covered 86.4%, with the estimate's
bias within its Monte-Carlo error but the sandwich standard error 27% smaller than the
estimate's realized spread; a Thompson-sampling logger covered 89.6% over the 67% of
replications its own propensities kept above the floor; and a contextual Thompson logger
at horizon 2, 1000 units, covered 87.2% over the 24% of replications that survived the
floor, with a bias of seven Monte-Carlo standard errors in that surviving slice. The
variance understatement comes from units coupled through the batch refits and from
heavy-tailed cumulative weights; the surviving-slice bias is selection on the outcomes
that drove the propensities down. The construction that repairs both (stabilized or
adaptively weighted estimators with a martingale variance; Hadad et al. 2021, Zhang,
Janson and Murphy 2021) is not implemented, so the refusal names one policy version per
trace as the route forward.

Under a fixed logger the interval is nominal from the effective-sample-size floor up.
Measured on 26 fixed-logger cells over target divergence, horizon and unit count (9188
emitted intervals; `tests/calibration/test_logged_policy_ess_floor.py`), coverage by band
of the smallest per-index ESS was 0.862 / 0.892 / 0.933 / 0.933 / 0.943 / 0.947 / 0.949 for
`[2,5)`, `[5,10)`, `[10,20)`, `[20,40)`, `[40,80)`, `[80,160)` and `[160,inf)`, with
700 to 2190 intervals per band; `[20,40)` is the smallest band from which every band is
within three Monte-Carlo standard errors of nominal, so `ESS_FLOOR = 20` and, since the
ESS never exceeds the unit count, the independent-unit floor is 20 as well. The two
bands just above the floor sit 1 to 2 points under nominal -- within tolerance, not at
it -- because the sandwich standard error is about 7% too small there; from 80 effective
units on it is within 1 point. On the nine fixed-logger cells of the adaptive study
(T in {1, 2, 5}, 50 to 1000 units, 160 to 800 replications each) coverage ranged from
0.931 to 0.960, every cell inside its own three-MCSE band; at horizon 5 with 100 units
the floor screened 126 of 600 replications and the emitted intervals covered 0.941.

The cumulative-product weighting is what makes the estimator target the dynamic policy
value. Measured against a current-step-only comparator at horizon 5, 1000 units, under
context dependence 0.8 (truth 0.0275): the current-step estimator is biased by +0.0026,
more than three Monte-Carlo standard errors from zero, while the shipped estimator's
bias of +0.0006 stays within its own.

## Power and planning

### Arm power planning depends on declared alternative-arm variance shapes

The three arm solvers evaluate the treatment and control terms at their own
means under the alternative. Starting from the control-arm effective variance
`v`, mean-like metrics assume equal absolute variance in the two arms
(`v_treatment = v`). Conversion and retention metrics that the runtime does not
decide with the exact binomial test (CUPED, clustered, absorbed-factor,
sequential) instead rescale `v` by the Bernoulli shape at the alternative rate:
`v_treatment = v * p_treatment * (1 - p_treatment) / (p_control * (1 - p_control))`.
These are explicit planning assumptions (`power_basis="asymptotic"`). They do
not guarantee that a future data-generating process has either variance shape.

### Conversion planning matches the exact binomial decision only within a budget

Unadjusted, unclustered, fixed-horizon conversion and retention plans report
the rejection probability of the runtime's exact binomial risk-ratio decision.
With at most 16,000 retained (control, treatment) cells at the null rate the
decision set is replayed exactly (`power_basis="exact"`); beyond that the
replay uses Normal conditional tails (`power_basis="approximate"`), measured
at up to 0.8 percentage points below the runtime's power (unequal allocation,
shifted null) and able to misclassify rare-event count pairs near the tail
allocation. The runtime's own control-arm window and floating-point
allowance carry over to both routes. A binomial plan's call takes seconds
rather than milliseconds. Measured on an Apple M3 Pro: exact route at 701 per
arm, about 0.8 s for achieved power or MDE and 1.9 s for sizing; approximate
route at 10,000 / 20,000 / 35,000 per arm (10% baseline), 1.2 / 2.6 / 3.9 s
for achieved power and 6.3 / 15.5 / 22 s for sizing, and at 20,000 per arm
with a 50% baseline 6.4 s and 67 s. Every count pair's decision comes from its
own replay, except counts the replay's first step would settle: those are
inferred from a neighbouring count's margin through the step's monotonicity
in the treatment count, which assumes each computed tail lies within
`5e-11` of its exact-arithmetic value. Sizing returns a verified bracket
crossing, not a proven global minimum; effect searches exclude earlier
effects to within `2e-12` of the target power. Triggered plans use the
rounded analyzed counts.

Bounded-metric baselines and implied null/alternative rates must stay strictly
positive and at most 1. A requested rate above 1 is refused rather than treated
as a conservative approximation. At a fixed sample size, the directed
noncentrality can peak before the effect-domain boundary, so a target power can
be genuinely unattainable. In that case an achieved-power query still reports
power for its supplied effect, but its companion `mde_relative` is null and
`mde_unavailable_reason` states `unattainable`, `unrepresentable`, or
`numerical_resolution`.

The segment-pairwise solvers deliberately retain their baseline-only
four-arm approximation. Their numbers should not be compared with the changed
arm trio as though the two models were identical.

Sequential MDE inversion recomputes the alternative-arm standard error and
boundary at each candidate. Its crossing calculation uses composite
eight-point Gauss–Legendre panels. Classical derivative-remainder bounds are
propagated through every surviving-density and exit-mass integral; floating
arithmetic and normal-tail allowances are added separately. Expected
information has its own propagated bound. Observed differences between
quadrature levels are diagnostics, not the basis of the enclosure.

Public scalar power and expected-information values require their respective
enclosures to meet independent convergence tolerances. An inverse that cannot
distinguish the target within the declared quadrature, interval, or resource
limits refuses with `power.minimum_detectable_effect.numerical_resolution`
instead of consuming an unresolved midpoint or claiming an effect or
unattainability. Information fractions remain the analyzed-unit-count
approximation described above.

### Switchback planning requires its own model

Ordinary switchback planning derives its reference effect and centered
contribution/order-slope covariance from a validated pilot, at the independent
unit or shared-block grain. Moment-t power conditions on those fitted estimates;
it does not account for their estimation uncertainty or certify finite-sample
calibration. Optional prospective envelopes and complete-law research oracles
retain separate lower-bound and Monte Carlo meanings. Existing small-sample
failures and the unresolved computational-certification work remain; see the
[switchback guide](guides/switchback.md).
Parallel-arm `Baseline.icc` does not describe within-unit switchback dependence.

## Observational estimands

Increment checks overlap and post-adjustment balance. It cannot validate conditional
ignorability, and no diagnostic on this page substitutes for a defensible identification
argument.

Categorical strings are internally encoded with a modal reference and one
indicator per remaining training level. High-cardinality columns therefore
produce many columns; no automatic pooling or cohort trimming is applied.
Encoding and per-level balance diagnostics do not establish overlap.
Definitions and artifact ingress preserve their default missing-value refusal
for numeric and categorical nulls alike. YAML rejects `missing`/`gate` tuning;
other policies use a full `Observational` design on the two dataframe routes.

### DML reports a partially-linear slope, not an average treatment effect

The double machine-learning estimator identifies the slope of the partially linear model,
which weights conditional effects by the conditional variance of treatment. That equals the
average treatment effect only under homogeneous effects or a constant weight
(`e(1-e)` with one treatment, `p_a p_0 / (p_a + p_0)` with several). The result labels its estimand
`plr_slope` and carries a note saying so.

A relative DML row divides that slope by the augmented control-arm mean of the analysed
population, with their joint influence covariance; it is not `mu1 / mu0 - 1`. With several
treatments, each row keeps its own comparison-specific slope over one shared control mean.
Do not read a DML relative lift as a population ATE when you expect effect heterogeneity,
and do not compare it against IPTW or AIPW as though all three targeted the same quantity.
Its interval is asymptotic: finite-sample calibration of multi-arm and small-cluster
designs is not established.

### Only untrimmed native-logistic IPTW accounts for fitting its propensity

The propensity model is fit on the same data used for estimation. Untrimmed IPTW with the
default pooled logistic learner includes those estimating equations, and with several
treatments the equations of every treatment-versus-control model that shares the control.
Generic, pattern-specific and trimmed fits still treat the fitted propensity as known, so
their standard error omits the propensity-estimation contribution, with no guarantee of
being conservative once trimming is applied. High-capacity or custom learners can overfit
assignment and destabilize the weights.

The multi-arm construction couples pairwise treatment-versus-control fits into marginal
arm propensities. With the default logistic learner this is consistent for a multinomial
logit propensity, but it is not that model's joint maximum-likelihood fit.

### Trimming changes which population you estimated

When overlap trimming removes units, IPTW and AIPW relabel the estimand to
`overlap_subpopulation_ate` and record the retained population on the result, so the change
of target is visible. The standard error conditions on that fitted retained set: it does not
include the variability of the trim boundary itself, which depends on estimated
propensities.

DML reports `plr_slope` whether or not trimming occurred, so a DML result alone does not
tell you that the population changed. Check the recorded population.

### The positivity gate is a fixed threshold with no weight diagnostics

The default identification gate requires propensities within `[0.01, 0.99]`, which caps a
raw inverse weight at 100. Passing that gate is not evidence of good overlap. The result
reports no effective sample size, no leverage, no weight-tail concentration, and no
sensitivity across thresholds.

A handful of near-boundary units can dominate an IPTW or AIPW estimate while the gate still
passes, producing an estimate that looks precise and is not stable. If you are relying on
weighting, compute weight diagnostics yourself.

## Multiple comparisons

### Fixed-horizon FDR control assumes a dependence condition that is not checked

Selection across the metric × arm × segment family uses ordinary Benjamini–Hochberg, which
controls FDR at the stated level under independence or positive dependence (PRDS). An
exploratory family sharing a control arm and correlated outcomes can violate that
condition, and there is no Benjamini–Yekutieli option at selection time.

This applies to the fixed-horizon path. Registered sequential inference uses
current or explicitly frozen likelihood e-values, whose family validity requires
the declared common joint-unit filtration. Distributional assumptions are
pre-data declarations; timestamps and observed means do not establish them.

### Sequential selected intervals use the same stopped likelihood

Sequential families retain every registered cell, including missing or abstaining
cells. Selected intervals are reinverted at `min(q * R / m, nominal alpha)` on
the checkpoint used for selection. This selected-set statement is distinct from
coverage conditional on selecting one particular metric. Unbounded, full, empty,
and numerically abstaining sets remain explicit in the result.

### Stopping-date guarantees require the registered reveal contract

Native, frame-panel and artifact monitoring require explicit finalized capture.
All relevant metrics must be revealed together after their longest required window,
in outcome-independent unit order. Previously finalized values and assignments
cannot be corrected during continuation. A changed generation or freshness hash
is insufficient proof of append. Current as-of readouts expose the retained labeled
checkpoint; historical curves require retained per-date checkpoints.

The finite-sample public route uses raw Bernoulli/Beta likelihoods and matching
inversion. The public mean, adjusted-mean and ratio routes instead provide
asymptotic confidence sequences and family approximations; estimated variance
does not preserve the exact e-value martingale argument. Scalar Gaussian NIG
and paired-Gaussian NIW kernels are private research diagnostics and do not
produce public anytime-valid evidence or decision rows. The filtration qualification is the declared common,
outcome-independent joint-unit reveal order; a timestamp or observed mean does
not establish it. Primary references are
[Ville's maximal inequality](https://doi.org/10.1007/BF01503646) and
[Howard et al.'s time-uniform concentration framework](https://arxiv.org/abs/1810.08240).
The bounded release evidence review is separate from these mathematical claims;
the full sequential research matrix remains uncertified.

The union across dates is not controlled. "This metric was flagged at least once this week"
has inflated FDR and no stated guarantee, and neither does picking the date with the most
flags. Act on the current date's discovery set; do not accumulate flags across days.

### Clustered CATE uncertainty is cluster-asymptotic

Clustered `estimate_cate` uses a weighted CR1/HC0 score sandwich and `K-1`
reference degrees of freedom, with every declared observed cluster counted.
It does not implement CR2. Small K, imbalance and high leverage still require
scientific calibration; matching the intended sandwich calculation does not
establish coverage.

Cluster ARD uses `trace(G @ V_JJ) / q` as its noise variance, with `G` the
weighted residualized interaction Gram matrix. This is a directional-average
precision approximation, not a full correlated-likelihood ARD model. Unclustered
ARD retains its existing residual-variance calculation.

Uniformly cloning rows **within the same cluster IDs** preserves fixed-design
point estimates, covariance and cluster ARD precision. Assigning new IDs to the
copies declares new independent clusters: under complete doubling the specified
sandwich covariance is multiplied by `(K-1)/(2*K-1)`. Invariance to that operation
is incompatible with the specified sandwich and literal cluster count; copies
that remain dependent must retain their original cluster identity. Learned
continuous bases can also change when refitted on duplicated rows.

Exact duplication checks therefore freeze scores and nuisance fits and replicate
the entire within-cluster row pattern. A fully refitted pipeline, or a roster-size
change that also changes member noise, needs separate behavioral and Monte Carlo
evidence. Neither an unchanged independent cluster count nor one replayed p-value
establishes null calibration. The existing unclustered calibration figures do not
certify clustered AUTOC, Qini, GATES, CLAN or selected policy values.

### Targeting validation depends on identification and overlap

With an `Observational` design, `validate_cate`, `targeting_rule`, and `select_targeting_rule` use doubly-robust scores; `Randomized` keeps IPW scores.
Observational GATES and the holdout ATE summarize score means, not raw-outcome Welch contrasts; identification still requires conditional ignorability and overlap.
When holdout trimming removes units, `CateValidation.population == "overlap_subpopulation"`: AUTOC, Qini, and GATES describe that retained population, not the full one.
[`estimate_cate`][increment.estimate_cate] remains randomized-only (`cate.identification.randomized_only`); its per-unit point predictions are not doubly robust.

Declared clusters retain unit-level scores and outcomes. `cluster_weight="member_count"`
weights members equally; `"equal"` gives each retained cluster total weight one and
requires cluster IDs. Smooth means use centered cluster contributions with the
`K/(K-1)` scale: disjoint randomized arms use their own cluster counts and Welch
reference degrees, overlapping contrasts combine signed contributions within each
cluster, and observational DR scores use pooled cluster influence.

Clustered DR nuisance models are fitted once on the training half, with frozen
heldout predictions. Ranks, empirical GATES/CLAN cutoffs, and policy values are
recomputed in a whole-cluster bootstrap (default
[`ClusterBootstrap(seed=0, repetitions=999)`][increment.ClusterBootstrap]).
Randomized pure-arm clusters are resampled within
arms; pooled DR and mixed-arm clusters have no arm stratum. Repeated cluster
draws are separate sampled instances. Weighted empirical cutoffs keep tied scores
together; the clustered AUTOC/Qini integrate the empirical weighted TOC with
uniform interpolation within each tied block, making uniform row cloning invariant.
Unclustered rank calculations retain their existing discrete definition.

Clustered intervals and tests use `bootstrap-t+cluster-jackknife-t`. The interval
is the smallest interval containing both component intervals; the one-sided
p-value is the larger component p-value. Thus a rejection requires both tests to
reject, and the reported interval cannot be narrower than either component.
This preserves a component's coverage when that component is valid; it does not
prove finite-cluster validity of either component.

For the bootstrap component, use the complete statistic
\(T\), its bootstrap draw \(T_b^*\), and positive covariance reference scales
\(s,s_b^*\) to form \(R_b^*=(T_b^*-T)/s_b^*\). Centering at \(T\) retains the
bootstrap estimate of ratio and cutoff bias; centering at the bootstrap mean
would erase it. Dividing by each draw's own scale retains the dependence between
estimation error and uncertainty, which matters for skewed cluster contributions
and unequal cluster masses. Even for an ordinary mean of \(K\) IID clusters,
the raw empirical-bootstrap variance is \((K-1)/K\) times the usual unbiased
variance estimate; more repetitions do not remove that finite-\(K\) difference.
For nonlinear ratios, replicates that omit a large cluster can also have both a
large estimation error and a much smaller reference scale. Intervals invert the root tails as
\([T-sR^*_{\mathrm{upper}},T-sR^*_{\mathrm{lower}}]\); one-sided rank tests compare
these same roots with \(T/s\). The reported `se` is the standard deviation of the
complete bootstrap statistics, including rank and cutoff variation, rather than
the conditional reference scale used inside the roots.

For GATES, CLAN, and policy values, the reference is the centered ratio-mean
covariance described above, recomputed for each draw's groups. For ranks, let a
tied score block occupy probability interval \((a,b]\). Integrating the response
weights gives \(r_A=(-b\log b+a\log a)/(b-a)\) for AUTOC (with \(0\log0=0\))
and \(r_Q=(1-a-b)/2\) for Qini. Holding these response weights fixed, the
functional \(E[r\psi]-E[r]E[\psi]\) has influence
\((r-E[r])(\psi-E[\psi])-\operatorname{Cov}(r,\psi)\).
The reference scale sums these signed, member-weighted influences inside each
cluster before squaring, with the \(K/(K-1)\) correction. It is a conditional
response covariance, **not** a claim that estimated ranks or cutoffs are fixed.
Inner policy selection similarly uses the cluster covariance of each member's
targeted net-benefit contribution as its reference. All deployed statistics and
reference scales are recomputed on every whole-cluster draw; there is no nested
bootstrap or extra rank sort for the rank reference.

The second component deletes each observed cluster once and recomputes the
**complete** statistic \(T_{(-g)}\), including ranks, cutoffs, groups, weights,
and policy budgets. It uses the full-estimate-centered cluster jackknife scale
\[
s_J^2=\frac{K-1}{K}\sum_{g=1}^K (T_{(-g)}-T)^2.
\]
This is the CV3 construction in
[MacKinnon, Nielsen and Webb, equation (18)](https://doi.org/10.1002/jae.2969).
That paper treats regression; applying the construction here requires regularity
of the complete nonlinear statistic. For a ratio mean with cluster target-mass
share \(h_g\) and signed contribution \(u_g\), the exact deletion identity is
\(T-T_{(-g)}=u_g/(1-h_g)\). This captures observed leverage without a fitted
multiplier. At equal masses it reduces to the usual cluster-mean standard error.
For shared-cluster contrasts, deletion differences combine both sides before
squaring, preserving their signed covariance. Estimated cuts and ranks are
recomputed, so their deletion changes also enter the scale.

The jackknife interval is \(T\pm t_{\nu,\alpha/(2m)}s_J\), where \(m\) is
the existing family size and \(t\) denotes an upper-tail quantile. Its test uses
the Student-t survival probability at \(T/s_J\). The reference has
\(\nu=\min_h(K_h-1)\), using the bootstrap arm/fold strata; without strata it
is \(K-1\). Deletions retain all remaining clusters and their declared target
weights, without compensating other clusters in the omitted cluster's stratum.
This unrestricted deletion scale can include composition variation absent from
the stratified bootstrap. The smaller stratum reference is conservative relative
to using \(K-1\), but is not an exact small-sample law. This adds \(K\) full
evaluations and no random draws; the bootstrap stream and repetition count are
unchanged. `reference_df` on rank/group/CLAN results describes this jackknife
component, while `covariance_method` describes the bootstrap response reference.

The bootstrap component has a first-order asymptotic justification when the complete
cluster bootstrap is consistent and \(\sqrt K s\) and \(\sqrt K s^*\) converge
to the same positive limit: Slutsky's theorem then gives the same limiting law
for the observed and bootstrap roots. The reference need not equal the full
rank/cutoff asymptotic standard error for this argument. It does not establish a
higher-order accuracy improvement for these nonlinear statistics. Jackknife-t
also requires an asymptotically linear statistic and consistent deletion variance;
it is not generally valid for a nonsmooth quantile statistic. Independent
clusters, adequate moments, no dominating cluster, and regular population
quantiles are needed; stable tied blocks can be evaluated, but a quantile exactly
at a jump boundary can be nonregular. The AUTOC logarithmic tail additionally
needs an integrable squared response-weight envelope. Training-only nuisance
predictions remain frozen. Neither this argument nor a calibration screen
establishes distribution-free finite-cluster coverage, particularly with few
skewed, high-leverage clusters. See the
[bootstrap-t inversion](https://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/ADAfaEPoV.pdf)
and [RATE asymptotics](https://arxiv.org/abs/2111.07966) for the underlying methods;
the latter does not by itself certify this clustered implementation.

Measured behaviour on a null screen with frozen oracle scores (constant effect,
independent CLAN covariate), 256 replications and 199 draws per cell: every one
of 72 designs stays within its family-aware binomial miss bound for AUTOC, Qini,
each GATES interval, the joint GATES family and CLAN, and within the rejection
bound for the AUTOC and Qini tests. The designs cross 10, 40 or 200 clusters,
equal sizes of 20 or repeating sizes 5/20/100, intracluster correlation 0, 0.2
or 0.5, Gaussian or skewed errors with one high-leverage score per cluster, and
both weightings. The tightest design has 10 unequal clusters, skewed errors,
member weighting and correlation 0.5: each GATES interval missed 18 of 256
times against a nominal 2.5% (6.4 expected, bound 19). Few unequal clusters
therefore still undercover; the screen is a regression check, not a guarantee.

GATES and CLAN retain separate Bonferroni budgets. Tail indices use the exact
binary64 input alpha divided rationally by twice the family size, rounded outward
on the \(B+1\) order-statistic grid. Monte Carlo p-values also round upward.
A failed bootstrap statistic makes uncertainty unavailable instead of conditioning
on successful draws. A finite draw whose reference scale is nonpositive or
nonfinite uses the observed positive reference scale in its root denominator.
This fallback retains the complete draw; it alone supplies no coverage correction.
`bootstrap_valid_repetitions` counts finite complete bootstrap statistics,
independently of their scale availability and of jackknife evaluations. Repeated
draws of one source cluster are still distinct instances. The observed response
and deletion scales must be positive and finite. An undefined deletion reports
`estimation.targeting.jackknife_unavailable_replicate`; it is never dropped from
the sum or replaced by a successful deletion. Small cluster
counts, empty groups, constant rankings, zero variance, and inadequate tail
resolution retain nullable fields and explicit `unavailable_reason` codes.
Results record cluster count, weighting, method, seed, and requested and valid
repetitions. Increasing repetitions resolves smaller tails; it does not remedy
insufficient independent clusters.

Deployment grain follows the explicitly declared intervention, independently of
uncertainty clustering. A cluster intervention cannot request unit deployment
(`estimation.targeting.unsupported_unit_deployment`). Cluster policies pool
member scores and select a score-ordered whole-cluster prefix, breaking ties by
canonical ID. Member/equal weighting determines both the budget and the reported
population. This prefix is feasible, not a knapsack optimum: unused budget stays
unused when the next cluster does not fit. Empty prefixes have no conditional
policy effect; zero and one fractions select nobody and everybody. Bootstrap
replicates recompute the pooled ranking and budget, counting repeated draws as
separate cluster instances while preserving canonical source IDs for score ties.
Overlap trimming that retains only part of a cluster is refused for cluster
deployment (`estimation.targeting.overlap.partial_cluster`).

Saved policy JSON retains its fitted scoring basis and coefficients.
`predict(cols, cluster_ids=...)` returns the candidate actions; consult
`recommendation` before deployment. Cluster prediction needs a complete member
roster for each deployment cluster and computes the budget over the supplied
batch. Unit prediction retains its fitted threshold even with dependence IDs.
Neither new-data outcomes nor nuisance refitting enter prediction.

Fraction selection freezes all inner-fold scores before resampling within folds
(and arms for randomized designs). The selected inner winner and a policy value
reported after the outer validation gate still have no nominal confidence interval;
the bootstrap does not remove this selection restriction. A declared cluster
intervention defaults to whole-cluster prefix deployment and rejects explicit
unit deployment.

### The targeting workflow carries no joint error control

Within the targeting workflow, group-effect tests are Bonferroni-corrected inside their own
family, and characteristic tests inside theirs. The rank-based tests are not corrected
alongside them, and no guarantee spans the whole workflow.

If you scan every interval and test the workflow displays and act on whichever looks
strongest, you are not protected at your stated alpha. The per-family scope is real but easy
to over-read.

### Meta-analysis can be anticonservative at small K

Pooling uses a DerSimonian–Laird heterogeneity estimate followed by an unmodified
Hartung–Knapp–Sidik–Jonkman adjustment, with no variance floor. Measured by
`tests/calibration/test_hksj_small_k.py` (10000 replications per cell, unequal segment
variances with log-scale standard errors 0.10–0.20, between-segment variance 0, 0.5 and 2
times the average sampling variance; binomial Monte-Carlo standard error 0.2 points at 95%
and 0.4 points at 80%): the HKSJ interval covers the random-effects mean 93.7–95.2% at
nominal 95% for every K from 2 to 12 and every heterogeneity level. The plug-in
random-effects interval covers 80.0% at K = 2, 85.2% at K = 3, 87.2% at K = 4, 88.6% at
K = 5, 90.9% at K = 8 and 92.2% at K = 12 when the between-segment variance is twice the
average sampling variance, and over-covers (96.0–96.4%) when there is none. HKSJ is the
narrower of the two in 12–35% of replications without heterogeneity and 2–13% with it. The
degeneracy fallback to the plug-in interval fired in 1 of 10000 replications in two of the
K = 2 cells and never otherwise on continuous data. The anticonservatism at small K is real
but bounded at about 1.3 points in this design; treat a pooled interval over two or three
segments as indicative for that reason, not because it is narrower than plug-in.

Bayesian segment shrinkage places a half-normal prior on the heterogeneity scale, with
a configurable default scale of 0.30. This prior remains consequential: a scale suitable
for log relative effects may strongly shrink effects measured in dollars or other absolute
units. Adaptive integration now resolves posterior mass beyond the old fixed grid for both
shrinkage and rollout pricing; it does not remove prior sensitivity. The reported
Wald intervals use posterior means and variances, rather than posterior quantiles;
they carry no finite-sample frequentist coverage guarantee.

For segment-specific inference, prefer the marginalized segment intervals, which integrate
over the heterogeneity posterior instead of plugging in a point estimate. They are a better
per-segment answer; they are not a substitute for the pooled interval, which answers a
different question.

## Operational limits

### The read-only SQL gate is a screen, not a sandbox

Every SQL string in a definitions file is admitted as exactly one read-only query. The check
parses the string with the declared (or the connection's) dialect and rejects:

- more than one statement (`definition.sql.statement_count`);
- any statement that is not a query, including `INSERT`, `UPDATE`, `DELETE`, DDL, and DML/DDL
  nested under a `WITH`;
- `SELECT ... INTO`, `FOR UPDATE`/`FOR SHARE`, `LOCK IN SHARE MODE`, and locking table hints such
  as T-SQL `WITH (UPDLOCK)`, `SERIALIZABLE` and `REPEATABLEREAD` (`definition.sql.not_read_only`);
  benign hints such as `NOLOCK`, `READPAST` and `INDEX(...)` are admitted;
- calls to a fixed list of known side-effecting functions (advisory locks, sleeps, large-object
  and file writers, `dblink`, session and query cancellation, and similar).

That list cannot be complete. A side-effecting function outside it, including any user-defined
function, is admitted, because syntactically it is an ordinary `SELECT`. Increment makes no claim
that an unlisted function is safe. The connection and its privileges belong to you.

Treat the gate as protection against a mistake in a definitions file, not as a security
boundary against a definitions author you do not trust. Definitions are trusted, governed code:
review them like any other code that runs against your warehouse, and never load a definitions
file from an untrusted source. Connect with read-only, least-privilege warehouse credentials
(no write, DDL, or execute grants beyond what the queries need).

Repository code cannot tell you how a downstream application accepts definitions, or what grants
your warehouse role actually holds. Before relying on the gate, verify both yourself: that
definitions can be changed only through your review process, that the application never builds
definitions from end-user input, and that the warehouse role is read-only on the tables in scope
(for example by attempting a write with that role and confirming it fails).
