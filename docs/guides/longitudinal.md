# Randomized longitudinal experiments

Analyze repeated additive outcomes without counting repeated periods as new
independent participants. Supply a separate assigned roster, the scheduled
outcomes, and a schedule declared before treatment. The estimator reports one
fixed-horizon additive effect for that roster, using each unit's pre-period
history. Precision improvement is not guaranteed: noisy pre-period outcomes can
make the change-score estimate less precise than a post-only comparison.

## Start with the assigned roster

The roster has exactly one row per unit: its randomized arm, predeclared cohort,
and scheduled exposure time. A control unit has the same hypothetical start
as treatment units in its cohort. Cohorts, observation schedules, exposure
anchors, and covariates must be fixed before treatment or by the randomization
design. Do not construct them from realized engagement, outcome availability,
or treatment uptake.

The observation policy is `complete_scheduled_roster`: each roster member has
exactly one finite outcome at every declared pre and post time. Missing,
duplicate, unscheduled, and unknown-unit observations refuse with
`longitudinal.observation`. A genuine measured zero is valid; an absent outcome
is not zero. Units with missing history remain in the roster and cause an
explicit refusal; do not remove them and claim the original roster effect.

Different cohorts may have different complete schedules, including staggered
entry and different pre/post lengths. Both arms within a cohort use the same
schedule. This supports predeclared unbalanced panels across cohorts. Selective
within-cohort attrition requires a separate missingness/weighting construction
and is outside this estimator. Fix incomplete feeds before analysis.

## Eager dataframe example

```python
import pandas as pd
from increment.longitudinal import (
    LongitudinalCohort,
    LongitudinalDesign,
    prepare_longitudinal_panel,
    fit_longitudinal,
)

schedule = LongitudinalCohort(
    name="launch",
    exposure=0,
    pre_periods=(-2, -1),
    post_periods=(0, 1),
    post_weights=(0.25, 0.75),
)
design = LongitudinalDesign(
    control_group="control",
    treatment_group="treatment",
    cohorts=(schedule,),
)
roster = pd.DataFrame(
    {
        "unit": list(range(12)),
        "group": ["control"] * 6 + ["treatment"] * 6,
        "cohort": ["launch"] * 12,
        "exposure": [0] * 12,
    }
)
outcomes = pd.DataFrame(
    [
        {"unit": u, "time": t, "outcome": 10 + u + t + (u % 3) * t * 0.1 + (u >= 6 and t >= 0)}
        for u in range(12)
        for t in (-2, -1, 0, 1)
    ]
)
summary = prepare_longitudinal_panel(outcomes, roster, design=design)
result = fit_longitudinal(summary)
assert result.n_units == result.n_independent == 12
assert result.n_observations == 48
assert result.effect > 0
```

Integer time indices and naive UTC datetimes are supported; normalize timezone
and sub-microsecond timestamps explicitly. Narwhals eager Pandas, Polars and
Arrow inputs are accepted. IDs cannot be missing or collide under canonical
string encoding. Warehouse IDs must be integer or string columns.

For cluster randomization, set `independence_grain="cluster"` on the design and
supply `cluster="cluster_id"` to the producer. Every cluster must be in one arm.
Clusters may span cohorts for inference. At least two independent grains must
occur in every cohort/arm, and covariate regressions must be estimable.

## Pre-treatment covariates

Use the existing `increment.estimation.cate.Covariate` declaration:

```python
from increment.estimation.cate import Covariate

roster["previous_spend"] = [2, 7, 3, 8, 4, 9] * 2
adjusted = prepare_longitudinal_panel(
    outcomes,
    roster,
    design=design,
    covariates=[Covariate(name="previous_spend")],
)
adjusted_result = fit_longitudinal(adjusted)
assert adjusted_result.covariates == ("previous_spend",)
```

Continuous columns standardize using the whole roster. String categorical
columns use modal-level reference coding, with lexical tie-breaking. Bases
then center within cohort. Each cohort/arm gets its own fitted slope, allowing
unequal allocation and heterogeneous covariate relationships. The sandwich
propagates fitted-coefficient uncertainty. Knots, missing covariates, constant
predictors, redundant columns, one-arm-only levels and fitted directions
supported by only one independent grain refuse; no automatic
feature selection or silent dropping occurs. These conventions align with
[#67](https://github.com/kylejcaron/increment/issues/67), without extending the
ordinary CUPED ingress or implementing its missing-history construction.

## Warehouse compression

`increment.query.longitudinal.prepare_longitudinal_query(outcomes, roster,
design=design, ...)` accepts Ibis relations bound to the same backend. Use
immutable snapshot relations: validation and reduction execute multiple
queries. Backend execution parity is tested on DuckDB; PostgreSQL, Snowflake
and BigQuery execution is not certified for this new adapter.

The adapter computes each unit's weighted change in the warehouse, encodes
covariates, and exports one Gram/XY record per independent grain. Raw outcome
rows do not leave the warehouse. With K regression columns, the export is
O(G K²) rather than O(N T); without cluster randomization G=N. This retains
participant/cluster dependence, not merely frequency counts over identical
regressors. Numeric/category metadata and validation summaries also execute
in the warehouse. Transfer decreases when the period count exceeds the retained
moment width; wide covariate models can increase the payload. Memory still grows
with the roster and covariate dimension. Capacity is checked before basis expansion.

Both producers return the same owned `LongitudinalSummary` and use
`fit_longitudinal`. Trusted pickle/deepcopy replay is covered; a portable or
untrusted wire format is not promised. Public results record the roster,
observation and independent counts, schedule/weights, cohort effects,
covariance, reference and multiplicity status. Regression intercepts describe
weighted change scores; slopes use the standardized, cohort-centered basis.
The computational design is capped at 48 columns (1224 exported moment fields)
to bound warehouse projection width and dense covariance work.

## Estimand and uncertainty

For unit i in predeclared cohort c, compute

\[
D_i=\sum_{t\in post_c}w_{ct}Y_{it}
       -\frac{1}{|pre_c|}\sum_{t\in pre_c}Y_{it}.
\]

Fit D to separate cohort/arm intercepts and cohort-centered pre-treatment
covariates, with slopes separated by arm. The target is
\(\sum_c N_c/N\,(\widehat\mu_{Tc}-\widehat\mu_{Cc})\): equal weight per
assigned unit and the declared post-period weights within cohort. It is a
mean effect per weighted observation period, not a post-period total.

Randomization, no anticipation, treatment-unaffected membership/observation,
and no spillovers between independent grains identify this contrast. Calendar
shocks cancel within cohort; arbitrary dependence within grain is retained.
This is a change-score contrast of a fully saturated cohort/arm/calendar
panel representation with unit intercepts. It includes all pre-period terms;
it is not a generic unbalanced two-way fixed-effects adoption estimator. The
[Mundlak compression paper](https://arxiv.org/abs/2410.09952) motivates scalable
panel reduction, while this implementation retains additional grain evidence
needed for uncertainty.

For grain g retain \(A_g=X_g'X_g\), \(h_g=X_g'D_g\). With
\(A=\sum_g A_g\), \(\widehat\beta=A^{-1}\sum_g h_g\), the score is
\(s_g=h_g-A_g\widehat\beta\). Covariance is

\[
\frac{G}{G-1}A^{-1}\left(\sum_g s_gs_g'\right)A^{-1}.
\]

The contrast variance is computed from its projected grain scores to avoid
cancellation in covariance quadratic forms. Near zero, an unadjusted contrast
uses exact arithmetic on the retained floating-point sufficient statistics;
small positive uncertainty is not rounded into a zero-variance status.

Large fitted covariate effects with tiny residual noise can lose accuracy in
floating-point Gram/XY compression. A cancellation screen refuses unresolved
requests with `longitudinal.numeric`; it is not a forward-error certification.
Subtracting a known treatment-unaffected outcome baseline before preparation
can help. Merely changing units cannot recover precision already lost.

There is no panel-row degrees-of-freedom multiplier. Intervals and two-sided
p-values use a working Student-t reference with G-1 df. The guarantee is
`asymptotic`; this does not certify small, skewed or highly unbalanced clusters.
A zero estimated variance produces numeric null uncertainty and
`unavailable_reason="zero_variance"`, never automatic significance.

## Pilot-based power and MDE

```python
from increment.power.longitudinal import (
    LongitudinalBaseline,
    longitudinal_achieved_power,
    longitudinal_minimum_detectable_effect,
    longitudinal_required_independent,
)

pilot = LongitudinalBaseline.from_summary(summary)
mde = longitudinal_minimum_detectable_effect(pilot, n_independent=120)
power = longitudinal_achieved_power(pilot, n_independent=120, delta=mde)
required = longitudinal_required_independent(pilot, delta=1.0)
assert abs(power - 0.8) < 1e-8
assert required >= 4
```

Planning projects SE as `pilot_SE * sqrt(pilot_G / new_G)`, with a noncentral-t
reference at new G-1 df. It conditions on unchanged allocation, cohort mix,
cluster-size distribution, schedule, weighting, and adjustment design under a
homogeneous additive shift. Per-cohort/arm expected independent counts must
remain sufficient for the adjustment design. Unit-randomized pilots require
independent replication of each fitted direction. Cluster pilots may retain
two grains per cohort/arm when covariate directions are replicated within each
grain; projections also condition on that unchanged within-cluster structure.
Runtime still checks the actual fitted design, rather than certifying future
estimability from an expected grain count alone. Planning currently requires
clusters nested within cohorts; crossing-cohort participation needs a separate
planning model. The approximation is not exact future-design variance.

More periods are not more independent units. Changing periods or covariance,
including extending the follow-up horizon, requires a newly fitted pilot.
Zero-variance pilots and unattainable or unresolved requests refuse explicitly.

## Compatibility and ingress

This is a standalone family, like logged-policy evaluation. Use its producers;
none of the six `Analysis` factories select this estimator. Existing factories
retain their current methods. Frame/query parity covers estimates, covariance,
intervals and p-values; the full dense panel reference also checks uncertainty.

| Interaction | Disposition |
|---|---|
| Pre-outcome history and mixed pre-treatment adjustment | Supported, separately fitted cohort/arm slopes |
| Cluster randomization | Supported, pure-arm cluster scores; working asymptotic inference |
| Breakouts | Supported only as predeclared cohorts with roster-standardized weights |
| Sequential inference | Not expressible; fixed-horizon analysis only |
| Ratio metrics | Not expressible; supply additive outcomes, not a ratio-effect claim |
| Winsorization | Only a fixed treatment-unaffected transformation supplied as the outcome; no learned thresholds |
| Multiplicity roles | One `exploratory_unadjusted` contrast; no confirmatory family adjustment |
| Priors | Not expressible; sampling inference uses no posterior |

The remaining broader longitudinal integrations and missing-history adjustment
are tracked by [#80](https://github.com/kylejcaron/increment/issues/80) and
[#67](https://github.com/kylejcaron/increment/issues/67). Numerical equivalence
and calibration evidence, including their limits, are listed in
[validation](../validation.md#reference-inventory).
