# Confounding sensitivity for a cohort risk ratio

An analyst has a cohort comparison and wants to know how strong omitted
confounding would need to be to overturn a decision. Use
`increment.sensitivity.risk_ratio_sensitivity` for a supplied **unadjusted
cohort risk ratio**, comparing a binary exposure and binary outcome over one
common follow-up in a declared population. The causal target is the population
risk ratio under exposure versus no exposure. The observed association is
preserved; sensitivity does not identify that causal target.

This is a standalone summary-input workflow. Its supported ingress is
`CohortRiskRatio` followed by `risk_ratio_sensitivity`, with an optional supplied
fixed-horizon confidence interval. It does not fit an estimator or run on any
of the six `Analysis` entry points. Increment's observational IPTW, AIPW and
DML readouts report different estimands/scales; their effect differences and
relative lifts cannot be used as cohort risk ratios.

## Declare the study and decision

```python
from increment.sensitivity import CohortRiskRatio, risk_ratio_sensitivity

study = CohortRiskRatio(
    risk_ratio=2.0,
    confidence_interval=(1.5, 3.0),
    confidence_level=0.95,
    population="eligible cohort at enrollment",
    exposure="opted in",
    outcome="converted within 30 days",
)

sensitivity = risk_ratio_sensitivity(study, threshold=1.25, direction="above")
assert sensitivity.study is study
assert round(sensitivity.point_evalue, 3) == 2.580
assert round(sensitivity.confidence_evalue, 3) == 1.690
```

The decision here is strictly `risk_ratio > 1.25`. The point E-value is the
minimum common strength of the two confounding associations that could move
the estimate to that threshold. The confidence E-value challenges the lower
endpoint, which currently also exceeds the threshold. Equality already fails
a strict decision; if an estimate or endpoint does not satisfy it, its E-value
is one. The upper endpoint remains in `sensitivity.study.confidence_interval`.

With `direction="below"`, the decision is strictly `risk_ratio < threshold`
and the upper interval endpoint is challenged. This handles protective
associations by reversing the exposure comparison. By default, direction
follows the estimate's side of one (`above` at one), and the threshold is one.
An above threshold must be at least one, and a below threshold at most one:
this construction challenges an association toward the null. Other threshold
requests refuse with `sensitivity.decision.threshold`.

An E-value describes a **minimax/equal-strength tipping point**. With unequal
associations, at least one must reach the E-value; both need not individually
reach it. It is not a probability of hidden confounding or a universal
robustness score. Good measured balance does not establish that omitted
confounding is absent.

## Inspect assumed confounding strengths

```python
scenario = risk_ratio_sensitivity(
    study,
    threshold=1.25,
    exposure_confounder_rr=2.0,
    confounder_outcome_rr=3.0,
)
assert scenario.bias_factor == 1.5
assert round(scenario.point_bound, 3) == 1.333
assert scenario.confidence_limit_bound == 1.0
assert scenario.study.risk_ratio == 2.0
```

Both inputs are **assumed upper bounds**, at least one, for a hypothetical
confounder `U` (which can be a vector of confounders). For an above decision:

- `exposure_confounder_rr` bounds the largest
  `P(U=u | exposure=1) / P(U=u | exposure=0)` over confounder categories.
- `confounder_outcome_rr` bounds the largest outcome risk ratio between any
  two confounder categories within either exposure arm.

For a below decision, the first ratio reverses the exposure arms. The second
remains the maximum within either arm. These are not odds ratios, regression
coefficients, or associations estimated from measured covariates.

The bounding factor is

\[
B = \frac{ab}{a+b-1}.
\]

For above decisions, `point_bound` is `risk_ratio / B`, a lower sensitivity
bound; for below decisions it is `risk_ratio * B`, an upper bound. The
corresponding supplied interval endpoint is transformed in
`confidence_limit_bound`. This is a bound under the specified confounding
limits, not an estimate of the actual bias or a corrected causal estimate.
The bounds may cross one and are not clipped. Default strengths of one give
`B=1` and reproduce the original estimate and directional interval endpoint.

The scenario above retains a point bound exceeding 1.25 while the transformed
lower confidence limit is 1.0. These are different conclusions; point
robustness does not establish confidence-level robustness.

## Assumptions and uncertainty

The result exposes its assumptions and retains study identities, estimator,
estimand, effect scale, confidence level, decision direction, threshold and
assumed confounding strengths. Causal interpretation requires consistency,
positivity and no interference, and exchangeability after conditioning on the
hypothetical confounders. Selection bias, measurement error and estimator
misspecification are outside the construction.

`guarantee="exact"` describes the **population sensitivity inequality and its
algebra**, conditional on those assumptions. It does not certify the supplied
estimate, confidence interval, or identification. Confidence-limit sensitivity
inherits the validity and coverage level of the supplied fixed-horizon
interval; Increment neither recalibrates it nor produces a new causal
confidence interval. Without a supplied interval, `confidence_evalue` and
`confidence_limit_bound` are numeric nulls, with
`confidence_unavailable_reason="confidence_interval_not_supplied"`.

The tipped ratio is `R = risk_ratio / threshold` for above decisions and
`R = threshold / risk_ratio` for below decisions. When the decision holds,
its E-value is `R + sqrt(R*(R-1))`. Numerically unrepresentable E-values or
bounds refuse with `sensitivity.numerical.range`; infinity and underflowed
zero are not returned as ordinary sensitivity results.

## Compatibility of this construction

| Interaction | Status | Behavior |
|---|---|---|
| CUPED | `not_expressible` | The ingress accepts an unadjusted risk ratio, not CUPED-adjusted outcomes or mean differences. |
| Ratio metrics | `construction_limited` | General ratios of means, odds ratios, hazard ratios and relative lifts refuse as scales; only binary cohort risks are supported. |
| Clustering | `source_limited` | The algebra needs no independence assumption, but the summary has no cluster design or covariance evidence. Supplied interval validity remains the caller's responsibility; no cluster inference claim is made. |
| Sequential inference | `construction_limited` | Non-fixed-horizon inference declarations refuse; no sequential sensitivity or repeated-look guarantee is implemented. |
| Winsorization | `not_expressible` | This ingress carries a binary outcome risk, not a transformed continuous outcome. |
| Breakouts | `supported` | Supply a separate cohort summary with the subgroup population declared. It targets that subgroup only; no automatic subgroup selection or family correction is performed. |
| Multiplicity roles | `not_expressible` | This diagnostic creates no p-values, posterior decisions or `Analysis` family memberships. It preserves the supplied interval and level without claiming a new family guarantee. |

Adjusted, standardized and regression-based estimates are outside this first
implemented slice. That is an implementation boundary, not a claim that
E-values are mathematically invalid for every adjusted risk estimand. They
refuse with `sensitivity.study.adjusted` or `sensitivity.study.unsupported`;
there is no automatic IPTW/AIPW/DML adapter or measured-covariate benchmark.
Inputs declare the estimator, estimand, scale and inference explicitly, with
defaults for the supported construction. Recognized unsupported combinations
raise `UnsupportedRequestError`; invalid numbers, identities, intervals and
decisions raise `InvalidRequestError`, with immutable structured context.
Refusal contexts use bounded descriptions for invalid objects that cannot be
safely retained.

## References and validation

- Ding and VanderWeele (2016), [Sensitivity Analysis Without Assumptions](https://doi.org/10.1097/EDE.0000000000000457): bounding factor, sharpness and exposure reversal.
- VanderWeele and Ding (2017), [Sensitivity Analysis in Observational Research: Introducing the E-Value](https://doi.org/10.7326/M16-2607): E-value interpretation.
- [EValue package examples](https://louisahsmith.github.io/evalue/reference/evalues.RR.html) and its [confounding vignette](https://louisahsmith.github.io/evalue/articles/unmeasured-confounding.html): frozen independent numerical comparisons.

The [reference inventory](../validation.md#reference-inventory) records the
published-example tolerances and uncovered regimes. Independent deterministic
confounded-population witnesses verify sharpness, causal bounds and the
no-confounding limit. No new sampling coverage or measured-covariate
benchmarking campaign is claimed.
