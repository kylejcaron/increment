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
For how evidence is labelled see [Evidence status](validation.md#evidence-status).

## How to read a refusal

A refusal is a coded early stop, not a silent fallback: branch on `.code`
([Machine-readable contracts](api.md#machine-readable-contracts)). The labels in the
[capabilities table](reference/capabilities-by-entry-point.md#what-runs-where) say where a
refusal comes from, not whether it is permanent.

Not every unavailable capability is a refusal. Where the entry point has no parameter or
constructor for the request, the table says **not expressible**: the call cannot be written, so
Python raises `TypeError` (or there is nothing to call) and there is no `.code` to branch on.
`from_switchback_panel(..., design=...)` and `from_switchback_panel(..., cluster=...)` are
examples, as is every logged-policy request made through an `Analysis` entry point.

- **SOURCE**: the entry point's input does not carry what the capability needs, as when a
  moments cube carries no per-unit rows to attach a covariate to. This is a current absence
  unless the text says structural, as the `from_switchback_panel` encouragement, observational
  and cluster rows (not expressible) and quantile requests (a coded refusal) do.
- **CONSTRUCTION**: no admissible construction exists for the request on that path. Five
  reasons are stated: filling an outcome with its pooled mean shrinks variance; the margin
  verdict is carried by `itt`; JSON object keys are strings; windowed and retention metrics
  have no per-unit collapse for a CUPED covariate (use `from_unit_summary`); and the repair
  for adaptive logging is not implemented (tracker-owned).
- **COMBINATION**: stored state from an earlier release meets the current reader, recipe or
  roster, as when a format-1 artifact context reaches the current reader.
- **not comparable**: the capability concerns something that entry point lacks, such as an
  artifact store, so there is no result to compare. The cell also names a SOURCE reason.

The parity harness (`tests/parity_harness/matrix.py`) calls SOURCE `source_limited` and
CONSTRUCTION `construction_limited`; its waiver strings label the "no switchback schedule"
cells `COMBINATION:` where this page says SOURCE.

## At a glance

One row per entry on this page. Rows carry no numbers or codes; the entry is authoritative.

| Entry | Kind | Check it when |
|---|---|---|
| [Uncertainty is estimated on every path](#uncertainty-is-estimated-on-every-path) | approximation | Reading any interval; approximations are weakest at small samples. |
| [The switchback t reference is itself an approximation](#the-switchback-t-reference-is-itself-an-approximation) | approximation | Few switchback units, skewed outcomes, or assignment far from even. |
| [Cluster references are qualified working approximations](#cluster-references-are-qualified-working-approximations) | approximation | Clusters are few, unbalanced, or skewed. |
| [CUPED treats its adjustment coefficient as known](#cuped-treats-its-adjustment-coefficient-as-known) | approximation | Coefficient treated as known: interval narrower than a known-coefficient comparator in the tested design (equal allocation, one homogeneous slope); unequal allocation and arm-specific slopes untested. |
| [Rare events on an unadjusted conversion/retention arm are estimated exactly, not refused](#rare-events-on-an-unadjusted-conversionretention-arm-are-estimated-exactly-not-refused) | scope of guarantee | Reading a conversion or retention interval's route, size limits, or extreme alpha. |
| [Signed effects require a suitable scale and reference](#signed-effects-require-a-suitable-scale-and-reference) | assumption | A control mean can be zero or negative. |
| [A quantile's reported interval depends on the alpha you asked for](#a-quantiles-reported-interval-depends-on-the-alpha-you-asked-for) | scope of guarantee | Comparing quantile intervals across analyses that allocated different alpha. |
| [A quantile metric's planned variance is a projection from the pilot, not a certified bound](#a-quantile-metrics-planned-variance-is-a-projection-from-the-pilot-not-a-certified-bound) | approximation | Sizing a quantile experiment from a pilot, or extrapolating far past it. |
| [Switchback carryover is assumed away, not tested](#switchback-carryover-is-assumed-away-not-tested) | assumption | Choosing a switchback washout; carryover is declared, not verified. |
| [Switchback inference requires independent units or blocks](#switchback-inference-requires-independent-units-or-blocks) | assumption | Switchback units may correlate, interfere, or share schedules. |
| [Sequential information fractions are unit counts, not Fisher information](#sequential-information-fractions-are-unit-counts-not-fisher-information) | approximation | Allocation or variance drifts during sequential monitoring. |
| [The sequential certification campaign has not been executed](#the-sequential-certification-campaign-has-not-been-executed) | tracked work | Relying on sequential or cluster evidence beyond its bounded release checks; see [Unvalidated regimes](validation.md#unvalidated-regimes). |
| [Off-policy evaluation of logged decisions is calibrated only under a fixed logger](#off-policy-evaluation-of-logged-decisions-is-calibrated-only-under-a-fixed-logger) | scope of guarantee | Evaluating logged decisions: interval is asymptotic and needs a fixed logger. |
| [Arm power planning depends on declared alternative-arm variance shapes](#arm-power-planning-depends-on-declared-alternative-arm-variance-shapes) | assumption | Planning with alternative-arm variances that future data may not follow. |
| [Conversion planning matches the exact binomial decision only within a budget](#conversion-planning-matches-the-exact-binomial-decision-only-within-a-budget) | approximation | Planning conversion or retention beyond the exact-replay budget. |
| [Switchback planning requires its own model](#switchback-planning-requires-its-own-model) | approximation | Sizing a switchback from a pilot; estimation uncertainty is not included. |
| [DML reports a partially-linear slope, not an average treatment effect](#dml-reports-a-partially-linear-slope-not-an-average-treatment-effect) | narrower estimand | Effects are heterogeneous and DML is read as an average effect. |
| [Only untrimmed native-logistic IPTW accounts for fitting its propensity](#only-untrimmed-native-logistic-iptw-accounts-for-fitting-its-propensity) | approximation | Trimmed or generic propensity fits: standard error omits propensity estimation. |
| [Trimming changes which population you estimated](#trimming-changes-which-population-you-estimated) | narrower estimand | Overlap trimming removes units; the estimand changes. |
| [The positivity gate is a fixed threshold with no weight diagnostics](#the-positivity-gate-is-a-fixed-threshold-with-no-weight-diagnostics) | scope of guarantee | Relying on weighting; passing the gate is not evidence of overlap. See [Unvalidated regimes](validation.md#unvalidated-regimes). |
| [Fixed-horizon FDR control assumes a dependence condition that is not checked](#fixed-horizon-fdr-control-assumes-a-dependence-condition-that-is-not-checked) | assumption | Fixed-horizon selection over correlated outcomes; the dependence condition is not checked. |
| [Sequential selected intervals use the same stopped likelihood](#sequential-selected-intervals-use-the-same-stopped-likelihood) | scope of guarantee | Reading a selected interval as coverage for one metric picked afterwards. |
| [Stopping-date guarantees require the registered reveal contract](#stopping-date-guarantees-require-the-registered-reveal-contract) | assumption | Monitoring departs from the registered reveal contract, or flags accumulate across dates. |
| [Clustered CATE uncertainty is cluster-asymptotic](#clustered-cate-uncertainty-is-cluster-asymptotic) | approximation | Clustered CATE with few, unbalanced, or high-leverage clusters. |
| [Targeting validation depends on identification and overlap](#targeting-validation-depends-on-identification-and-overlap) | assumption | Validating targeting on observational data; ignorability cannot be checked. |
| [The targeting workflow carries no joint error control](#the-targeting-workflow-carries-no-joint-error-control) | scope of guarantee | Acting on the strongest of many targeting-workflow tests. |
| [Meta-analysis can be anticonservative at small K](#meta-analysis-can-be-anticonservative-at-small-k) | approximation | Pooling few segments; HKSJ can undercover in the tested design. |
| [The read-only SQL gate is a screen, not a sandbox](#the-read-only-sql-gate-is-a-screen-not-a-sandbox) | operational | Definitions could come from anyone you do not fully trust. |

## What runs where

A capability is a claim only when its path is named. `Analysis` has six entry
points in three families: **dataframe** (`from_unit_summary`, one row per unit;
`from_unit_panel`, one row per unit per day; `from_switchback_panel`);
**warehouse** (`from_definitions`, compiling from raw event facts;
`from_unit_day_artifact`, reading unit × day relations previously published
into the warehouse); and **portable** (`from_moments`, a file of pre-reduced
moments exported from any of the others).

The table records measured behavior on every path, using the dataframe unit-summary
path as the oracle against which the other matched-arm paths are checked. The switchback panel estimates a
different quantity (a fixed-horizon contrast over a switchback schedule), so it is
exercised on its own and never compared row for row with that oracle. It lives in
[Capabilities by entry point](reference/capabilities-by-entry-point.md#what-runs-where).

## What the intervals assume

### Uncertainty is estimated on every path

No estimator here is handed a known variance; every one estimates uncertainty from the
data. What differs is the approximation each family layers on to get there:

| Family | How uncertainty is obtained |
|---|---|
| Conversion and retention arm lift (unadjusted, unit-grain, no informative prior), all four success and failure counts dense | Delta-method log-scale uncertainty against a Welch-Satterthwaite t reference (`reference_kind="t"`), as every unadjusted mean -- an approximation with no finite-sample guarantee, checked at the dense-count threshold (`conversion_inference="auto"`, the default) |
| Conversion and retention arm lift (same eligibility), any sparse count, or `conversion_inference="finite_sample"` | Exact independent-binomial risk-ratio test inversion (Berger-Boos restricted-nuisance construction) -- no delta method, no Normal reference (`reference_kind="binomial"`) |
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
relative-scale decisions. Only those rows do: the default `conversion_inference="auto"`
labels a row with dense counts `reference_kind="t"`, an approximation whose noncoverage was
measured to stay within a stated tolerance at the dense-count threshold (see the next
section) and is not a finite-sample guarantee; the default is never a universally
finite-sample route. Their additive `abs_lb`/`abs_ub` sidecar, when available,
uses a Normal-Wald approximation on either route; that interval is not exact.
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
conversion/retention uses the delta method only where every success and failure count is
dense and the exact binomial method below at every other event count.
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

### Conversion and retention arms: dense counts use the delta method, sparse counts are estimated exactly, not refused

An unadjusted (no CUPED, no declared cluster), unit-grain conversion or retention arm pair
has two routes, and a rule that reads only the four per-arm success and failure counts and
the tail allocation (`alpha / 2` per tail two-sided, `alpha` directional) picks one before any
interval is computed. It never compares an interval or a p-value, and it labels the row:

* **Delta-method route** (`reference_kind="t"`, `scale="log"`, no `binomial_set`): the log
  risk ratio against a Welch-Satterthwaite t reference that every unadjusted mean uses,
  taken when the smallest of `x_c`, `n_c - x_c`, `x_t` and `n_t - x_t` is at least
  `dense_min_count(tail)`. It has no arm-size ceiling and no `log_se >= 0.5` risk (every
  count is at least 9 there, which keeps the combined log standard error below
  `sqrt(2/9)`), and it carries no finite-sample guarantee.
* **Finite-sample route** (`reference_kind="binomial"`, `scale="linear"`, a `binomial_set`):
  an exact independent-binomial risk-ratio method (Berger-Boos restricted-nuisance test
  inversion; see `increment/estimation/binomial_rr.py`) for every other count pair. It
  handles every event count directly, with no admission rule and no continuity correction,
  and only these rows carry the finite-sample guarantee. The hybrid default is never a
  universally finite-sample route.

`Method.conversion_inference` selects: `"auto"` (the default, on `Method`, `MethodSpec` and the
wire format) routes by counts, and `"finite_sample"` always takes the finite-sample route
(refused by code, `estimation.binomial.finite_sample_unavailable`, with an informative prior,
clustered units, sequential inference or an observational adjustment; and on a metric that is
not a conversion or retention rate or on a CUPED-adjusted method, with
`conversion_inference.finite_sample.metric_type` and `conversion_inference.finite_sample.cuped`,
the one code per hazard whether a definition, a frame metric, a method or a plan carries it).
There is no forced
delta-method value and no analysis-wide knob. A multiplicity family that reads p-values at
levels below the row's own routes each row at the smallest level it can be decided at (for a
BH family, `q` over the hypotheses it tests), never at a looser one. Plans and wire payloads
written before the field existed decode as `"finite_sample"`, the route they were produced
under.

`dense_min_count(tail) = max(412, ceil(145 z^4))` with `z = Phi^-1(1 - tail)`. It is an envelope
of a measured requirement, not a fit: the requirement is the least count at which the
delta-method interval's one-sided noncoverage in the boundary cells below stays within
`tests.mc.scientific_delta(tail)` of the tail (a tenth of the tail below 0.05, 0.005 at and above
it) at that count and at every larger step of a geometric ladder (ratio 1.15) through 1.25 times
it. The law is at least 1.25 times the requirement at every production tail; the 0.1 tail sets
its floor and the 0.0005 tail its slope. The tail error of a log risk ratio Wald interval grows
like `z^3` relative to the tail, so extreme tails need thousands of events. The requirement
grows more slowly than `z^4` below them, so the middle tails carry a margin well above 1.25
(about 3.7 times at 0.025 and 0.05), and counts the interval already covers take the
finite-sample route there, which is valid at every count. A tail below 0.0005 is extrapolated by
the same formula, not measured. Per tail, the requirement is a step of a ladder of that ratio
with every step of that candidate's own ladder through 1.25 times it measured; the ladders are
anchored at 10 for the tails from 0.025 up and at 1,637 below (the 0.005 tail's on the
continuation of an interrupted scan, at 2,489.5). A scan records its ladder with every step.
The same numeric rung counts regardless of which scan measured it, but a differently numbered
rung cannot replace a missing candidate rung. `python -m calibration.conversion_route required`
reads saved scans, inferring compatible anchors for legacy rows and refusing a legacy scan
that no single ladder fits:

| One-sided tail | 0.0005 | 0.001 | 0.005 | 0.01 | 0.025 | 0.05 | 0.1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Measured requirement | 13,320 | 10,072 | 2,863 | 2,165 | 576 | 286 | 329 |
| `dense_min_count` | 17,000 | 13,224 | 6,384 | 4,247 | 2,140 | 1,062 | 412 |
| Law over requirement | 1.28 | 1.31 | 2.23 | 1.96 | 3.72 | 3.71 | 1.25 |

`python -m calibration.conversion_route select` integrates the interval's noncoverage over
the exact binomial lattice in rare, failure-limited and central boundary cells (control sizes
1e3 to 1e7 at every 1, 2 and 5 of a decade, allocations 1:1, 1:4 and 4:1, risk ratios 0.5 to 2)
and reports, per tail, the worst excess over the tail in tolerance units (at most 1 passes); the
failure-limited cells bind at every tail. The excess just below each requirement and at it:

| One-sided tail | 0.0005 | 0.001 | 0.005 | 0.01 | 0.025 | 0.05 | 0.1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Last failing step | 11,583 | 8,758 | 2,490 | 1,883 | 501 | 249 | 286 |
| Worst excess there | 1.15 | 1.09 | 1.29 | 1.01 | 1.16 | 1.21 | 1.03 |
| Requirement | 13,320 | 10,072 | 2,863 | 2,165 | 576 | 286 | 329 |
| Worst excess there | 0.87 | 0.98 | 0.97 | 0.99 | 0.98 | 0.93 | 0.97 |

and, at counts every tail was measured at (worst excess in tolerance units):

| Count `m` | 0.0005 | 0.001 | 0.005 | 0.01 | 0.025 | 0.05 | 0.1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 435 | 6.84 | 5.47 | 3.07 | 2.25 | 1.39 | 0.88 | 0.95 |
| 1,637 | 2.86 | 2.35 | 1.37 | 1.03 | 0.64 | 0.41 | 0.45 |
| 2,165 | 2.73 | 2.25 | 1.31 | 0.99 | 0.62 | 0.39 | 0.42 |

`verify` re-checks the shipped count at each tail and at 1.25 times it in the same cells; every
cell passes (worst excess in tolerance units, at the count and at 1.25 times it):

| One-sided tail | 0.0005 | 0.001 | 0.005 | 0.01 | 0.025 | 0.05 | 0.1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| At `dense_min_count` | 0.84 | 0.72 | 0.62 | 0.69 | 0.62 | 0.56 | 0.97 |
| At 1.25 times it | 0.83 | 0.71 | 0.65 | 0.60 | 0.47 | 0.42 | 0.76 |

The 0.1 tail passes by 3% at its floor. `hybrid` sums the production pipeline's noncoverage
exactly over the count lattice at design counts two below to two above the shipped count (the
three cells with the most routed noncoverage at each offset, two-sided at `alpha = 2 * tail`,
plus the worst cell read directionally at `alpha = tail`: `greater`, and `less` at the tails a
production alpha reaches directionally, run at the 0.1 and 0.05 tails). At every one of the
seven tails every cell stays within tolerance; the worst excess over each tail's cells is
0.50 / +0.34 / +0.40 / +0.48 / +0.53 / +0.63 / +0.67 at the 0.1 / 0.05 / 0.025 / 0.01 / 0.005 /
0.001 / 0.0005 tails. The production pipeline on 57,601, 30,400 and 62,400 seeded draws at the
0.1, 0.05 and 0.025 tails agrees with the exact value within four Monte Carlo standard errors;
that check was not run below the 0.025 tail. The calibration-only vectorised interval agreed
with `estimate_lift` within `2.1e-14` (relative), and the interpolated Student quantile agreed
with `t.isf` within `1.7e-15`, on the measured `conformance` cells. These empirical comparisons
are not numerical certificates; planning and exact enumeration use the runtime calculation
directly. A bounded grid is evidence at those cells, not coverage for every data-generating
process.

A BH-selected row is re-estimated at its own corrected level and routed there, so its label can
differ from the label of the nominal row whose p-value selected it.

An Encouragement design's ITT on a conversion metric takes the same
routes (ITT and LATE readouts of a retention metric under an Encouragement design are refused
with `readout.encouragement.retention`; a compliance-only request, `estimands=("compliance",)`,
ignores the retention outcome and is measured to succeed on `from_definitions` only, see the
[capabilities table](reference/capabilities-by-entry-point.md#what-runs-where)): the
design's uptake (first-stage compliance) moments are a
different random variable over the same units and do not change the
ITT's own sufficient statistics, so they are stripped before either route
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

Both routes are fixed-horizon: neither proves validity under optional
stopping or repeated peeking (see the sequential-inference limitations elsewhere on this
page for that separate guarantee). A conversion/retention request with an informative
prior uses the established Normal approximation on every count. A registered sequential specification
instead uses its declared raw-observation likelihood and matching sequential inversion,
subject to the registered sampling and finalized-window contract.

The finite-sample route also has three boundaries, each a refusal rather than silent degradation (dense counts under `auto` meet none of them):

* **Arm size.** Each arm is capped at `binomial_rr.FINITE_SAMPLE_MAX_ARM_SIZE`
  (1,000,000,000; it replaces `binomial_rr.MAX_ARM_SIZE`, which capped arms at 4,000,000 and is
  removed without an alias -- code that imported the old name imports the new one and gets the
  new cap). This is a compute-resource applicability boundary, not a statistical
  one: a call with either arm above the cap refuses immediately
  (`estimation.binomial.finite_sample_arm_ceiling_exceeded`, the cap in `max_arm_size`; a dense cell runs at any size under `auto`)
  rather than running a search whose cost keeps growing with the arm. The cap is the
  largest arm the numerical safeguards were validated at against an independent decimal
  oracle (`calibration/binomial_oracle.py`, run by `scripts/measure_binomial_ceiling.py`
  `ulp` and `recovery` at 4,000,000, 16,000,000, 64,000,000, 100,000,000 and 1,000,000,000
  per arm): the SciPy binomial primitives' error allowance (the worst relative error of a
  pmf, cdf or sf that bears weight, as a share of `n` units of `2^-52`, was 0.198, 0.245,
  0.212, 0.236 and 0.226 at those sizes, against an allowance of one), the Clopper-Pearson
  enclosure, the support window's omitted mass (each held at every size), and count recovery from the
  producer's float moments, whose Bernoulli second-moment check scales with `n`: the
  DuckDB producer's counts were accepted at every size and rate measured (0.5, 0.002, 1e-4
  and a single success), the second-moment error reaching at most 0.24 of the check's
  rounding bound (2,442 times `variance_slack` at a billion units), and a constant-0.5
  arm was refused at every size. Count thresholds are computed in exact
  integers, which a float quotient cannot keep above about 67,000,000 per arm. The error
  model was measured with fused multiply-subtract on arm64; an x86 build has not been run.
* **Latency.** Cost grows with arm size. Cold CPU seconds and peak resident set of one
  two-sided `alpha=0.05` `confidence_interval` with the treatment count at a risk ratio of
  1.02 of the control rate, measured on an Apple M3 Pro with
  `scripts/measure_binomial_ceiling.py latency` (commits `d9b136b` to `2cad484`, none touching
  the estimation code; wall time depends on machine load):

  | Units per arm | 5% control rate | 1e-4 | 50% |
  |---:|---:|---:|---:|
  | 4,000,000 | 5.16 s / 178 MiB | 0.07 s / 122 MiB | 13.7 s / 207 MiB |
  | 16,000,000 | 10.8 s / 198 MiB | 0.227 s / 128 MiB | 44.2 s / 237 MiB |
  | 64,000,000 | 44.5 s / 228 MiB | 0.731 s / 148 MiB | 159 s / 242 MiB |
  | 100,000,000 | 82.3 s / 222 MiB | 1.21 s / 150 MiB | 179 s / 415 MiB |
  | 1,000,000,000 | 234 s / 336 MiB | 2.92 s / 165 MiB | 787 s / 424 MiB |

  The cost follows the control-count window, which follows the nuisance rate, so a rare rate
  is cheap at any size and the window is widest at a 50% control rate.
  Each p-value bounds a supremum over the control rate's Clopper-Pearson
  interval. The search splits that interval until the bound's distance above a
  witnessed lower end of the supremum is within 2^-14 of the larger of the p-value
  and the tail level it is compared with, plus the certificate's own float noise
  (2.9e-12 at 1,000 per arm, 8.9e-10 at 1,000,000, 3.6e-9 at 4,000,000, 8.9e-8 at
  100,000,000 and 8.9e-7 at 1,000,000,000, which no search narrows), or
  after 2,048 splits. The reported p-value is therefore a certified upper bound that
  exceeds the supremum-based p-value by no more than that gap: 2^-14 of the p-value
  above the tail level and 2^-14 of the tail level below it, plus the noise. For one tail
  at a billion units per arm that is at most 2.4e-6 at a 0.025 tail (1.5e-6 plus 8.9e-7) and
  1.5e-6 at a 0.01 tail (6.1e-7 plus 8.9e-7); at 1,000 units the noise is negligible and it
  is 1.5e-6 and 6.1e-7. A two-sided p-value is twice the smaller certified tail, so its gap
  is twice those. The target is never tighter than a fixed 1e-6 stop at a tail
  level of 0.0164 or more. A certificate whose bounds still straddle the tail level is
  conservatively non-rejecting; a certified upper bound below the tail rejects even
  within the declared gap. The endpoint search only compares p-values with the tail
  level, so each probe also stops once that comparison is certified either way.
  A search the cap ends is not an error: its p-value stays a valid,
  conservative bound, and the row's `note` says how many probes ended so and the largest
  gap they left, in units of the p-value the row reports (twice a directional gap on a
  two-sided row). No tolerance beyond that disclosure is promised. Every
  `BinomialConfidenceSet` records `binomial_bb_difference_v3`: its Clopper–Pearson
  widening is relative to the smaller endpoint/complement, its numerical allowance
  scales with arm size, and its lattice thresholds use integer arithmetic. A row
  persisted under v1 (the absolute-gap stop), v2 (the earlier nuisance envelope),
  another construction, or no construction is refused when read
  (`estimation.results.binomial.obsolete_construction`), rather than combining its
  endpoints with a verdict recomputed by a different rule. Re-run the analysis to cut
  it again. Each endpoint is the outer end of a search
  bracket no wider than 0.05% (2^-11) and no wider than 1/128 of the log risk
  ratio's standard error, whichever is finer, so the search works harder as
  arms grow and the standard error shrinks. The stop is the resolution of the
  evaluated tail envelope only, not a bound on SciPy's primitive error; a
  search that cannot reach it keeps its conservative
  endpoint and says so in the row's `note`. The process caches up to 24 MiB of control and
  24 MiB of treatment tail vectors between searches. A readout multiplies this across
  metrics, arms, and breakout cells. Only sparse counts, or an explicit
  `conversion_inference="finite_sample"`, take this route. A further large
  speedup would need a genuinely different tail-evaluation construction (a
  closed-form or recurrence update between adjacent risk-ratio candidates);
  none is implemented today. Reduce the number of eligible contrasts in a single
  call (narrow `metrics=`, run large-arm breakouts separately) if latency
  matters more than exactness at your arm sizes.
  FCR-selected rows refresh this disclosure for the returned interval; an
  obsolete nominal disclosure is removed while unrelated notes are retained.
* **Extreme alpha.** Two floors bound the tail level. The frozen nuisance tail budget passed
  to the Clopper-Pearson endpoint solver is `min(1e-6, alpha / 32)`; below `1e-9` (i.e.
  `alpha < 3.2e-8`), SciPy's iterative beta-quantile solver has demonstrated large relative
  error against an exact oracle for small `n` in the validated regime, so the endpoint is
  refused (`estimation.binomial.tail_unrepresentable`) rather than certified outside that
  regime. And every certified tail carries a float margin that grows with the arms,
  `(A(n_c) + A(n_t) + m) * 2^-52` with `A(n) = max(2048, n)` (SciPy's binomial primitives
  were measured against an exact decimal oracle to a billion trials, where their relative
  error stays within a quarter of `n` units of `2^-52`): once it reaches what the tail
  level leaves after the nuisance budget (`alpha / 2 - min(1e-6, alpha / 32)` two-sided,
  `alpha - min(1e-6, alpha / 32)` one-sided) no p-value read from an evaluated tail can be
  certified below the tail, and the call refuses with the same code (context `alpha`,
  `margin`, `n_c`, `n_t`) instead of returning a set that extends to wherever
  the structure alone stops it. The one exception is a null the structure alone rejects: a
  two-sided or "less" test of a null ratio above `1 / a`, `a` the control arm's
  Clopper-Pearson lower bound at the nuisance budget, has an empty nuisance domain, so its
  p-value is the budget itself (doubled two-sided) with no tail evaluated, and the upper
  endpoint is the first candidate above `1 / a`, which the control arm certifies by itself;
  that call is answered, the lower endpoint at zero. A row persisted from such a call
  carries that guard: `stat_sig()`/`p_value()` and the table adapters recompute its verdict
  from the persisted counts, so re-reading them at a null the structure does not reject (the
  unshifted null, for one) refuses with the same code and context a fresh call at that null
  does, never a margin-floored non-rejection. For equal arms the two-sided threshold is
  about `3.8e-9` at 4,000,000 per arm, `9.5e-8` at 100,000,000 and `9.5e-7` at
  1,000,000,000 (one-sided, about half of that); it passes the `3.2e-8` solver floor at
  about 34,000,000 per arm, so below that size the solver floor binds. The margin is
  absent from ordinary levels. Its effect depends on its ratio to the tail level, so it was
  measured on the production search at 10,000 units per arm with the margin set to the
  ratio a billion-unit arm has (it was not run at a billion): at `alpha >= 1e-4` it does not
  move an interval, at `1e-5` it widens it by about 1%, and from about `3e-6` it degrades
  it (+2%, then +43% at `1.5e-6`). Planning mirrors both floors: a tail level they refuse
  leaves the runtime deciding no count pair, so it has no power to plan and is refused, not
  planned as zero. Under a dominating margin the runtime decides only the count pairs whose
  control count rejects the null on its Clopper-Pearson bound alone, and refuses the rest;
  a plan whose control window at the baseline rate holds a refused count is refused too, never
  planned with those pairs as non-rejections. `achieved_power`, `minimum_detectable_effect`
  and each `power_curve` row refuse it with `power.binomial_tail_level_unrepresentable`
  (context `alpha`, `beta`, `tail_alpha`, `margin`, `n_c`, `n_t`, `p_c`, `decided_from` -- the
  smallest control count the runtime decides, `None` when it decides none -- `solver_floor`,
  `cause`, `solver_floor` or `float_margin`, and `scope`, `requested`) and an arm above the
  ceiling with `power.binomial_arm_ceiling_exceeded` (context `n_c`, `n_t`, `max_arm_size`),
  before any replay. `required_sample_size` searches only sizes whose decision reads the
  treatment arm: one the margin dominates even at the smallest design is refused before
  searching, with `power.binomial_tail_level_unrepresentable` (`scope` `smallest`); a float
  margin that starts dominating only at larger arms ends the search at that size with
  `power.binomial_size_search_unreachable` (context `power`, `maximum_power`, `n_per_arm`,
  `max_arm_size`). An allocation so lopsided that the smallest design has an arm above the arm
  ceiling is refused with `power.binomial_arm_ceiling_below_smallest_design`. There is no
  finite-sample route at a smaller alpha; use a larger alpha, or `conversion_inference="auto"`,
  which takes the delta-method route at counts dense for the tail and has neither floor.

For these fixed-horizon binomial plans, public power is a model-based point
probability, not a conservative numerical lower bound. Enumeration publishes
computed rejection mass only when its internal enclosure establishes maximum
absolute error at most `1e-6`; otherwise it refuses with
`power.binomial_probability_unresolved`. The dense closed-form route instead
reports its model's point probability; its validation tolerance remains
`0.005`, not a universal bound or finite-sample calibration guarantee.
Binomial MDE locates the earliest detectable region to `1e-8` absolute plus
`1e-8` relative tolerance on the relative-effect scale. It does not promise the
first IEEE-representable effect or silently skip materially earlier unresolved
bands. Unresolved MDE searches refuse with
`power.minimum_detectable_effect.numerical_resolution`, or leave a companion
MDE unavailable with `mde_unavailable_reason="numerical_resolution"`.
The reported power is evaluated at the returned effect. Internal accuracy
bounds address numerical evaluation, not model misspecification; calibration
remains a separate, deferred diagnostic. The planning model is
`hybrid_finite_plus_delta_v3`, and persisted v1/v2 plans require recomputation.
See [power planning](guides/power-analysis.md#conversion-and-retention-planning-follows-the-runtimes-route)
for replay limits and comparisons with other tests.

CUPED and unit-grain ratio-denominator conversion/retention retain their log-scale
guards; their sufficient statistics are not raw Bernoulli count pairs. A clustered
unadjusted arm instead uses joint relative inference, without the scalar log-SE
admission rule. None of those approximate routes inherits finite-sample validity.
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
total/conversion estimand labels explicitly name that retained window.

A schedule integrity check cannot establish either independent randomization or absence of
carryover. Under a unit-cycle declaration, a realized schedule too lopsided to be plausible
(for example 320 CT and 0 TC unit-cycle draws) refuses before statistics are constructed
with `source.frame.switchback.schedule` and reason `implausible_realized_split`. Correctly
declared, nondegenerate shared schedules are supported.

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

Evidence status for these campaigns is recorded under
[Unvalidated regimes](validation.md#unvalidated-regimes).

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

The record of where these figures come from is under
[Unvalidated regimes](validation.md#unvalidated-regimes).

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
decide with the finite-sample binomial test (CUPED, clustered, absorbed-factor,
sequential, and counts the runtime takes the delta-method route at) instead rescale `v` by the Bernoulli shape at the alternative rate:
`v_treatment = v * p_treatment * (1 - p_treatment) / (p_control * (1 - p_control))`.
These are explicit planning assumptions (`power_basis="asymptotic"`). They do
not guarantee that a future data-generating process has either variance shape.

### Conversion planning follows the runtime's route; the finite-sample replay is bounded

Unadjusted, unclustered, fixed-horizon conversion and retention plans mirror the runtime's
count rule. A count pair is decided by the delta method when its four per-arm success and
failure counts are all at least `m = dense_min_count` for the plan's tail allocation and by the
finite-sample test otherwise, so a plan's power is the rejection probability of that union
over the count lattice: the production delta decision on the routed rectangle plus the replayed
finite-sample decision on the rest (`power_basis="exact"` or `"approximate"`).
Public `power` is the computed rejection mass, admitted only when its internal
enclosure establishes absolute error at most `1e-6`. It is neither
route's power: a plan whose counts straddle the threshold is not the smaller of the two. A plan
whose counts are dense with probability at least `1 - 1e-6` (`P(m <= X <= n - m)` per arm, the
arms independent) is enumerated while its lattice has at most 100,000 cells (a cost limit,
the same for every design), retaining any undecided finite-route mass in the upper bound.
Larger dense lattices use the closed-form model above (`power_basis="asymptotic"`, no replay,
no cell budget, no arm ceiling, so a dense plan above a billion units per arm is planned).
Its figures have the measured 0.005 agreement criterion, not a runtime lower-bound guarantee.
A decision the runtime refuses in full on the finite-sample
route (a tail level its margin dominates, an arm above its ceiling) is refused for a plan that
keeps more than half of `1e-6` of its counts there, never reported as zero power. A minimum
detectable effect whose whole path from the null is closed-form is solved without a lattice.
Every enumerated plan is subject to the cell bound below, including sparse and borderline
`auto` plans. `conversion_inference="finite_sample"` always uses enumeration; sufficiently
dense `auto` plans can instead use the closed-form route.

For these eligible dense rows, the runtime reconstructs Bernoulli moments from the validated
integer counts before computing the delta interval. Accepted producer-rounding differences
therefore cannot change the decision for identical counts. Other metrics and adjusted
conversion rows continue to use their supplied moments.

`python -m calibration.conversion_route bound` compares each route's plan with the pipeline's
exact rejection probability, summed over the count lattice, and judges each by the claim of its
route. The 534 designs run so far reach every production tail at central and rare-event rates,
with falls (negative lifts) at the 0.01, 0.025, 0.05 and 0.1 tails; the `tails` and `window`
grids define every production request at its own alpha, two-sided against a rise and
directional in both directions, and the directional requests have been run at the 0.05 and 0.1
tails (central and rare-event) and two 0.01 designs. Those figures were measured against the
interim planner (the smaller of the replayed and the closed-form power for a plan whose counts
straddle the threshold, and a Normal-tail replay above 16,000 cells): a sparse plan was
between 1.8e-5 above and 3.3e-4 below the pipeline's exact probability, and a borderline plan
between 1.6e-5 above and 0.072 below it. Neither is a claim of the hybrid planner, whose
enclosure is checked against the same enumeration with no tolerance beyond its own undecided
mass and numerical error (the dense closed form keeps its 0.005 ceiling; its two failures, at
1,236 per arm and a 50% baseline with a 6% lift, are now enumerated).

A `bound --out` checkpoint records each design's enumeration beside the finite-sample
construction and routing floor it was summed under, and its plan beside the planner model. A
resumed run retains compatible runtime enumerations and replans retired-model records with the
current planner, `hybrid_finite_plus_delta_v3`. Its v1 and v2 predecessors must be
recomputed, not relabelled. Unknown models and earlier undated layouts refuse with
the file line and a route to a new `--out`. Saved earlier files remain unchanged as evidence.

A plan not evaluated in closed form integrates the count lattice with a bounded decision
budget. Evaluated pairs use the runtime's delta-method calculation on the routed rectangle
and the finite-sample replay elsewhere. Each evaluation selects at most 100,000 routed pairs
and 150,000 finite-sample directional pairs from its own count windows, prioritising heavier
cells and retaining unprocessed probability in the upper bound. Selection depends on the
request, not cached decisions: shared, repeated and fresh evaluations report the same enclosure.
An enumeration that cannot establish maximum point error at most `1e-6` refuses
with `power.binomial_probability_unresolved`; it does not substitute the lower
endpoint. Diagnostic enclosures can remain wide. The Normal-tail replay only proposes sizes.
The runtime's control-arm window and floating-point allowance carry over.

On a loaded 12-core Apple machine, evaluating a dense 62,500-cell plan (1,236 per arm) with
the runtime calculation took 10 to 20 seconds. Public calls that also search for an effect
require further evaluations. Broader calibration certification of this implementation is
pending; the saved campaign results above describe the interim planner.

Each evaluated finite-sample pair uses its own replay or a root exit settled by the replay's
first step; no pair inherits a neighbour's decision, because the runtime's stopped search
need not be monotone in a count. Sizing returns a verified bracket crossing,
not a proven global minimum.

The internal numerical enclosure accounts for the runtime's SciPy allowance,
summation error, omitted support and unresolved decisions. Public solvers compare
the admitted computed point power with the target, not an enclosure endpoint.
The closed-form route reports its own model probability; its dense agreement
criterion remains `0.005`, not a universal runtime error bound.

MDE searches the earliest detectable region to `1e-8` absolute plus `1e-8`
relative effect tolerance, not the first representable floating-point effect.
It excludes an earlier interval only with a valid upper bound and cannot skip
a materially earlier unresolved region. Reported power is evaluated at the
reported effect. If the search cannot resolve that region, a supplied-effect
answer preserves its valid power and has `mde_relative=None` with
`mde_unavailable_reason="numerical_resolution"`. A direct
`minimum_detectable_effect` refusal additionally retains structured context.
Triggered plans use the rounded analyzed counts.

A solve's replay is bounded by the cells it stores: at most 10,000,000 (control, treatment)
count cells, the control window by the treatment windows at the null rate and at every
alternative that solve evaluates. A supplied effect, an effect search (the union of the
windows it evaluates) and each row of a power curve are separate solves: the cells an
earlier row or the supplied effect left on the shared geometry are a cache, dropped when
the next solve needs the room (the solve's own cells never are, and its union is what the
bound counts), so a curve row answers as its scalar call does. The work
follows that count, not the arm size, which the runtime decides up to a billion units.
Planning refuses a decision whose null rectangle exceeds the bound before any replay
(`power.binomial_replay_bound_exceeded`), and with the same code (its context then names
the alternative treatment rate `p_t`) an alternative whose window would take its solve past
it, before any mask is allocated. An effect search that would exceed it ends unresolved:
the companion `mde_relative` of a supplied-effect answer is null with `numerical_resolution`
(the supplied effect's power is unchanged), and `minimum_detectable_effect` is refused with
the bound after the work that preceded it. An effect search stores more than the null
rectangle, so it reaches the bound at smaller arms than a supplied effect does.
`required_sample_size` searches only sizes about 1/128 under the largest whose null and
supplied-effect rectangles fit, and a search that reaches that ceiling refuses with
`power.binomial_replay_bound_exceeded`, whose context adds the target `power` and
`power_reached` there (a skipped size just above may reach more). The other ends of a size
search have their own codes (see the sizing table in the power-analysis guide): the arm
ceiling or a float margin that dominates the tail level at larger arms ends it with
`power.binomial_size_search_unreachable` and `maximum_power`, and a decision refused at the
smallest design is refused before any search (see **Extreme alpha**). Under `conversion_inference="finite_sample"` a supplied effect
reaches the bound at about a million units per arm at a 5% baseline, about 190,000 at 50%,
and at any arm the runtime admits at a rate expecting up to about 48,000 events per arm.
Under default `"auto"`, dense plans using the closed-form route have no replay and no
such bound; small enumerated dense plans remain subject to the replay bound. Measured with
`scripts/measure_binomial_ceiling.py planning` (`--conversion-inference finite_sample` or
`auto`) on a 12-core machine, cold CPU seconds of the call and the process's peak resident set,
under a shared load that varied by several times between cells (null cells are the null
rectangle). Explicit `finite_sample` plans at 1e5, 1e6, 5e6 and 5e7 units per arm, dense (a 5%
baseline) and sparse (about a hundred events per arm):

| Baseline, units per arm | Call | CPU / peak | Outcome |
|---|---|---:|---|
| 5%, 100,000 | `achieved_power`, lift 5% | 9.8 s / 1.34 GiB | 0.97M null cells, companion effect available |
| 5%, 1,000,000 | `achieved_power`, lift 1.5% | 198 s / 1.72 GiB | 9.67M null cells; companion effect `numerical_resolution` |
| 5%, 5,000,000 | `achieved_power` or `minimum_detectable_effect` | refused within 1 ms / 0.12 GiB | 48.3M null cells |
| 5%, 50,000,000 | `achieved_power` or `minimum_detectable_effect` | refused within 1 ms / 0.12 GiB | 483M null cells |
| 0.1% at 1e5, 0.01% at 1e6, 0.002% at 5e6, 0.0002% at 5e7 (100 events) | `achieved_power`, lift 50%, and `minimum_detectable_effect` | 0.19 to 0.20 s / 0.21 to 0.26 GiB | 20,164 null cells at each size, answered (`approximate`) |

The same calls under `"auto"`:

| Baseline, units per arm | Call | CPU / peak | Outcome |
|---|---|---:|---|
| 5%, 1e5 to 5e7 | `achieved_power`, `minimum_detectable_effect` and `required_sample_size` | 1 ms / 0.12 GiB | `power_basis="asymptotic"`; at 5e6 the power for lift 1% is 0.951 and the MDE 0.77%, at 5e7 the power for lift 0.3% is 0.930 and the MDE 0.24% |
| 50%, 1e5 and 1.9e5 | `minimum_detectable_effect`, `achieved_power` | 1 ms / 0.12 GiB | asymptotic, answered where the replay refuses |
| 100 events per arm at 1e5, 1e6, 5e6, 5e7 | `achieved_power`, lift 50% | 0.19 to 0.21 s / 0.23 to 0.25 GiB | the sparse replay, unchanged (`approximate`) |

Earlier measurements of the replay at other sizes (an Apple M3 Pro at `45829f1`, `2da355f`),
which an explicit `finite_sample` plan still takes and which were not repeated at this commit.
The cells marked † were re-measured at `45829f1`, after a companion effect search became a solve
of its own (it no longer inherits the supplied effect's cells, so an unresolved companion costs
what a refused `minimum_detectable_effect` does); the others are from `2da355f`:

| Baseline, units per arm | Call | CPU / peak | Outcome |
|---|---|---:|---|
| 5%, 250,000 | `achieved_power`, lift 3% | 16.5 s / 1.6 GiB | 2.4M null cells, companion effect available |
| 5%, 500,000 † | `achieved_power`, lift 2% | 35.3 s / 1.8 GiB | 4.8M null cells, companion effect available |
| 5%, 500,000 † | `minimum_detectable_effect` | 35.7 s / 1.9 GiB | answered |
| 5%, 1,000,000 † | `minimum_detectable_effect` | refused after 61.0 s / 1.8 GiB | needs 19.7M cells |
| 5%, lift 1.75% | `required_sample_size` | 277 s / 1.7 GiB | 993,064 per arm; companion effect `numerical_resolution` (not re-measured) |
| 50%, 100,000 | `minimum_detectable_effect` | refused after 57.5 s / 1.3 GiB | needs 10.6M cells (5.1M null) |
| 50%, 190,000 † | `achieved_power`, lift 3% | 201 s / 1.7 GiB | 9.7M null cells; companion effect `numerical_resolution` |
| 50%, 190,000 † | `minimum_detectable_effect` | refused after 210 s / 1.1 GiB | needs 10.8M cells |
| 1e-4 at 1e6, 1e-6 at 1e8, 1e-7 at 1e9 (100 events) | `achieved_power`, lift 50% | 0.85 s / 0.2 GiB | 20,164 null cells, answered |

Not measured: `required_sample_size` at a 50% baseline or beyond a million units per arm under
`finite_sample`. Planning a design above the bound has no replay: under `auto` a dense one is
planned in closed form, an explicit `finite_sample` one is refused. Before the bound existed a
call took 574 s and 3.4 GiB at 4,000,000 per arm (39 million cells).

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

For the evidence record see [Unvalidated regimes](validation.md#unvalidated-regimes).

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

What this assumes is the family's declared common joint-unit filtration. It matters when a
selected interval is read as coverage for one metric chosen afterwards, which is a different
statement.
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

The gate assumes definitions are trusted code; it matters whenever a definitions file could
come from someone you do not trust.

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
