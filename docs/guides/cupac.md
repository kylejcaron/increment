# CUPAC: externally trained predictions as a CUPED covariate

## What CUPAC means here

In this guide, CUPAC is not a separate estimator or public method. It is ordinary
scalar CUPED with a covariate column containing a prediction from a model trained
outside the randomized experiment. Train and freeze the model on historical units,
score each experiment unit before assignment, and declare the resulting per-unit
column as `MetricSpec(covariate="prediction")`. Request CUPED with
`Method(name="cuped", variance_reduction="cuped")`.

The analysis still estimates the randomized experiment's effect. The prediction
is a precision covariate; it does not define treatment assignment, eligibility,
or the estimand. A useful prediction can reduce interval width, but a weak or
shifted predictor may do little. Calibration or model quality alone does not prove
that a particular confidence interval is calibrated.

The [runnable CUPAC notebook](../examples/cupac.md) builds separate historical
and experiment samples, records provenance, compares both methods, and shows the
missingness refusal and an arm-dependent-imputation failure mode.

## Declarations the analyst owns

Increment receives a numeric prediction column. It cannot infer when the model
was trained, what data or features it used, whether the prediction was available
before assignment, or whether the historical population is independent. Record
and retain evidence for these analyst-owned facts alongside the experiment.

| Declaration | What to record and check |
|---|---|
| Training cutoff | The latest time included in training. It must precede the experiment's assignment period; use an explicit timezone when timestamps require one. |
| Feature timing | Every feature used to score an experiment unit must be available before that unit's assignment. A feature's name or a frozen model does not prove its timestamp. |
| Training independence | Training units and labels must be independent of experiment assignment and outcomes. Keep the historical cohort definition and the disjoint-unit check. |
| Model identity | Record a stable model/version identifier, training code and configuration, and the feature schema. Do not silently replace the fitted model between arms or scoring batches. |
| Prediction coverage | Measure scored and missing units by assigned arm before analysis. Investigate any imbalance; an overall coverage percentage can hide assignment-dependent gaps. |
| Reproducibility | Preserve the frozen model, input transformations, scoring code, random seeds where relevant, and per-unit output used for the analysis. Re-scoring later should reproduce the stored prediction. |

### Missing predictions

Use `covariate_missing="error"` for a CUPAC analysis. A missing or null prediction
then refuses with the stable code
`frame.validation.metric_covariate_column`, so the analyst can repair scoring or
make a deliberate policy decision before interpreting an estimate. The
[notebook](../examples/cupac.md) shows this refusal.

The default `impute` policy fills missing covariates with the pooled mean of
observed covariates. That is defensible only when predictions were meant to be
scored before assignment and missingness cannot depend on assignment. Imputation
is not a correction for arm-dependent coverage: if treatment units are more or
less likely to be scored, or missingness selects units based on post-assignment
information, the covariate distributions used by the adjustment can be distorted.
Repair pre-assignment scoring, re-check coverage by arm, and then analyze. The
notebook uses selectively missing high treatment predictions to illustrate the
resulting biased lift; that single example demonstrates a hazard, not a universal
imputation-bias estimate.

A table of coverage by arm is diagnostic, not a substitute for the assignment
and feature-timing checks. Under the `error` policy, a missing value is refused
rather than silently changing the analyzed cohort or imputed covariate.

## A dataframe analysis

The per-unit dataframe needs one outcome, assignment label, stable unit id, and a
frozen prediction for every unit. The prediction must be present on every unit
under the recommended `error` policy.

```python
import pandas as pd
from increment import Analysis, Method, MetricSpec

units = pd.DataFrame(
    {
        "user_id": ["c1", "c2", "t1", "t2"],
        "variant": ["control", "control", "treatment", "treatment"],
        "revenue": [18.2, 20.5, 19.6, 22.4],
        "prediction": [17.9, 20.1, 19.3, 21.8],
    }
)

analysis = Analysis.from_unit_summary(
    units,
    unit="user_id",
    group="variant",
    control="control",
    metrics=[
        MetricSpec(
            name="revenue",
            type="mean",
            covariate="prediction",
            covariate_missing="error",
        )
    ],
)
rows = analysis.run(
    decision_method=Method(name="unadjusted"),
    sensitivity_methods=(Method(name="cuped", variance_reduction="cuped"),),
)
```

The call keeps the unadjusted method as the decision method and reports CUPED as
a sensitivity method. Inspect the result method, estimate, interval, and declared
scope together. A CUPED row is still CUPED; the source of its covariate is an
analyst-owned prediction rather than a warehouse pre-period calculation.

For ratio metrics, the same prediction column is supplied to CUPED, which adjusts
the numerator and denominator separately using separate slopes. Do not substitute
a post-treatment or assignment-derived feature. See the [CUPED guide](cuped.md#ratio-metrics)
for the ratio construction and its covariance terms.

## Measured evidence

The tests use an independently generated historical sample, fit a frozen least-
squares model, and score an independent randomized experiment. They check useful,
irrelevant-feature, and shifted predictors for mean and ratio metrics. Coverage
and bias are repeated-draw checks; calibration slopes and interval-width ratios
come from a separate large fixed-seed precision draw. These Monte Carlo results
characterize the tested regimes and do not provide a finite-sample guarantee.

Each coverage value below is based on 1500 replications with 1000 units per arm.
The coverage MCSE is the binomial standard error; the width ratio is CUPED interval
width divided by unadjusted interval width on a 20000-units-per-arm precision draw.
Mean signed errors are lift estimate minus the known relative lift.

| Metric | Predictor scenario | Coverage | Mean signed error | Coverage MCSE | Width ratio |
|---|---|---:|---:|---:|---:|
| Mean | Useful | 0.958667 | 0.000019268 | 0.005140 | 0.432736 |
| Mean | Useless (irrelevant feature) | 0.950667 | 0.000032438 | 0.005592 | 1.000003 |
| Mean | Shifted | 0.950667 | -0.000160344 | 0.005592 | 0.890227 |
| Ratio | Useful | 0.946000 | 0.000150112 | 0.005836 | 0.444461 |
| Ratio | Useless (irrelevant feature) | 0.946667 | 0.000174823 | 0.005802 | 1.000002 |
| Ratio | Shifted | 0.946667 | -0.000198347 | 0.005802 | 0.889534 |

The signed-error Monte Carlo standard errors (sample standard deviation divided
by the square root of 1500), in the same row order, are 0.00011805, 0.00026523,
0.00023064, 0.00012263, 0.00026732, and 0.00022403. Every rate fell within the
prespecified nominal coverage band, and every absolute signed error was within
three signed-error MCSEs. The mean-width relation
`width_ratio ~= sqrt(1 - rho**2)` held for each predictor scenario; see
`test_cupac_precision_tracks_prediction_quality`.

The separate control-arm calibration slopes of revenue on prediction were:

| Metric | Useful | Useless | Shifted |
|---|---:|---:|---:|
| Mean | 0.998615 | 0.839446 | 0.475312 |
| Ratio numerator (revenue) | 4.212958 | -1.808274 | 1.990418 |
| Ratio denominator (sessions) | 0.007192 | -0.274786 | 0.002354 |

Calibration describes predictor/outcome association in this draw; it is not
interval coverage. The repeated-draw tests are `tests/test_cupac.py::test_cupac_coverage_and_bias`
and `tests/test_cupac.py::test_cupac_coverage_smoke`. Precision and calibration
are checked in `tests/test_cupac.py::test_cupac_precision_tracks_prediction_quality`.
Summary/panel parity and refusal codes are covered by the other tests in that
module and by the parity case
`helper:tests/parity_harness/cases.py` (case `cupac_external_prediction_fixed_horizon`).

## Entry-point dispositions

The supported and unavailable paths are measured by the parity case above. These
dispositions concern an externally supplied prediction column, not CUPED as a
whole: warehouse CUPED continues to use its own pre-period covariate route.

| Entry point | External prediction disposition | Notes |
|---|---|---|
| `Analysis.from_unit_summary` | Supported | Declare the `prediction` column in `MetricSpec(covariate=...)`. |
| `Analysis.from_unit_panel` | Supported | The prediction must remain constant across every row for a unit; variation refuses with `frame.frame_panel.unit_covariate_varies`. |
| `Analysis.from_moments` | Supported | Replay moments exported with the covariate's sufficient statistics. The moments must actually carry the covariate. |
| `Analysis.from_definitions` | Not expressible today | CUPED binds the metric's own pre-period window; an external prediction column cannot be bound. Declared covariate sets are tracked by GitHub #67. |
| `Analysis.from_unit_day_artifact` | Not expressible today | Same pre-period binding limitation as definitions; the external column cannot be bound. This is not a refusal of ordinary warehouse CUPED. |
| `Analysis.from_switchback_panel` | Refused | Switchback procedures do not apply CUPED; the existing refusal code is `source.frame.switchback.metric`. |

The definitions/artifact restriction is an ingress limitation tracked for the
separate declared-covariate work. It is not a statistical limitation of CUPED,
and it should not be read as saying that warehouse CUPED with a pre-period
window is unavailable.

## Interactions inherited from CUPED

CUPAC changes only the covariate's provenance. It inherits CUPED's supported
metric, source, method-role, and inference behavior; this recipe does not add
combinations or guarantees. Consult the [method compatibility reference](compatibility.md)
and [CUPED guide](cuped.md) for current path- and method-specific rules.

| Interaction | CUPAC behavior |
|---|---|
| Mean and ratio metrics | Fixed-horizon mean and ratio are measured by the Task 1 Monte Carlo campaign; ratio numerator and denominator slopes are separate. Other metric types inherit CUPED's own availability rules. |
| Unit panel | Supported for the fixed-horizon per-unit collapse when the prediction is unit-constant; a varying prediction is refused by code. |
| Clustering and breakouts | No new support is implied; apply the existing CUPED compatibility rules for the requested source and metric. |
| Winsorization and multiplicity | The external prediction does not change method composition or multiplicity roles. Any combination must satisfy CUPED's existing rules and retain the declared decision/sensitivity roles. |
| Sequential inference | CUPED's existing predictability and asymptotic/exact-route rules apply unchanged. This guide makes no new sequential guarantee for a frozen external model. |

In particular, historical training independence does not itself make an
experiment-trained coefficient predictable for an exact sequential process.
Use the existing CUPED guidance for sequential estimands and registrations.

## CUPAC, multivariate adjustment, and surrogates

CUPAC is not multivariate linear adjustment. In CUPAC, the analyst fits a model
outside the experiment, freezes it, and supplies one scalar prediction per unit.
CUPED estimates its usual scalar adjustment coefficient from the experiment's
outcome and prediction moments. By contrast, multivariate adjustment uses
multiple covariates fitted jointly in the experiment; support for that feature
set and estimator is a separate capability (see GitHub #67 and the method
compatibility reference). A prediction that compresses many historical features
into one score is still one CUPED covariate at analysis time.

A post-treatment surrogate is different again. It is measured after assignment
and can itself respond to treatment, absorb part of the treatment effect, or
create selection bias. Do not use a post-treatment variable as a CUPAC/CUPED
covariate merely because it predicts the outcome well. The feature timing must
precede assignment for every experiment unit.

## What is not included

This recipe does not support training or refitting the prediction model on the
randomized experiment, cross-fitting, automatic model selection or AutoML, or new
sequential guarantees. It does not establish transportability from historical to
experiment populations, prove that analyst-declared training units are disjoint,
or certify feature timestamps. Those are design and data-lineage responsibilities.

It also does not extend CUPED to a new metric, source, or combination. Use the
parity case and capability table for observed ingress behavior and the existing
CUPED documentation for restrictions outside this guide's fixed-horizon mean and
ratio calibration evidence.

## References

- Deng, A., Xu, Y., Kohavi, R., and Walker, T. (2013). “Improving the Sensitivity
  of Online Controlled Experiments by Utilizing Pre-Experiment Data.” DOI:
  [10.1145/2433396.2433413](https://doi.org/10.1145/2433396.2433413).
- Li, J. (2020-06-08). “Improving Experimental Power through Control Using
  Predictions as Covariate (CUPAC).” DoorDash Engineering.
  [Article](https://careersatdoordash.com/blog/improving-experimental-power-through-control-using-predictions-as-covariate-cupac).
- [CUPED variance reduction for A/B tests](cuped.md), including ratio metrics
  and sequential inference.
