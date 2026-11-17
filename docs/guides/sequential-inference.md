# Registered sequential inference

You can make repeated decisions as finalized experiment data arrives, but only
with a sequential plan registered **before looking at outcomes**. Repeated
fixed-horizon intervals do not provide that protection.

## Choose the route by the metric and the guarantee you need

- **Conversion or retention — `always_valid`:** finite-sample, anytime-valid
  under the registered sampling and reveal assumptions.
- **Mean, ratio, conversion or retention — `asymptotic_mean`:** an asymptotic
  approximation, not a finite-sample guarantee.

Exact Bernoulli monitoring supports multiple treatment arms but not CUPED.
The asymptotic route supports one treatment/control comparison and supported
CUPED adjustments. Ratios need its ratio model and denominator checks;
relabeling a ratio as a mean does not transfer the guarantee.

Exact monitoring can have very low power for rare conversions. That does
not establish the asymptotic route's calibration at those rates.
[Sample-size planning](power-analysis.md#fixed-and-sequential-planning)
is a separate approximation, not a runtime guarantee; exact-route planning
is unsupported.

## Sequential monitoring of a conversion metric

Declare the assignment law, metric role and inference before reading data.
This example uses independent 50/50 assignment and finalized 0/1 outcomes.
`exposure_date` sets reveal order: ascending enrollment date, then canonical
string unit ID for ties—not dataframe row order.

Reveal every registered metric for a unit together, after the longest required
outcome or uptake window. The order must be outcome-independent and finalized
values must never change. A summary frame asserts that its outcomes are final;
panel, warehouse and artifact sources instead require an explicit
[capture with an `as_of` horizon](../reference/sequential-contracts.md#capture-and-continuation).
Use an explicit `design=` with fixed `allocation`, not the `control=` shortcut.
The table requires the `tables` extra.

```python
from random import Random

import polars as pl

import increment
from increment.tables import readout_table

design = increment.Randomized(
    control_group="control",
    allocation={"control": 0.5, "treatment": 0.5},
    allocation_scheme="independent",
)
plan = increment.AnalysisPlan(
    primary="purchase",
    inference=increment.InferenceSpec(kind="always_valid", baseline_rate=0.03),
)

rng = Random(21)
n = 2000
arms = ["treatment" if rng.random() < 0.5 else "control" for _ in range(n)]
finalized_frame = pl.DataFrame(
    {
        "user_id": [f"u{i:04d}" for i in range(n)],
        "variant": arms,
        "enrolled_on": list(range(n)),
        "purchase": [int(rng.random() < (0.045 if arm == "treatment" else 0.03)) for arm in arms],
    }
)
experiment, reader = increment.from_frame(
    finalized_frame,
    shape="summary",
    unit="user_id",
    group="variant",
    design=design,
    metrics=[increment.MetricSpec(name="purchase", type="conversion")],
    experiment_id="checkout-experiment",
    plan=plan,
    exposure_date="enrolled_on",
)
results = increment.run(experiment, reader).readout()
readout_table(results, title="Sequential purchase results")
print(results[0].stat_sig())
```

<!-- output -->
```text
False
```

This look does not reject the null; it does not establish no effect.
The table displays relative lift in percent. Authoritative bounds on
`results[0].require_sequential_result().bounds` use ratios: ratio is
`1 + lift`, so a ratio of 0.681539 means about −31.8461% lift.
Bounds may be unbounded, full or empty. A missing point does not erase a
confidence set; see [sequential result fields](../reference/result-fields.md#sequential-inference-two-coordinate-systems).

The finite-sample claim requires Bernoulli outcomes, independent
fixed-probability assignment and the reveal rules above. An assignment check
marked `not_rejected` does not prove those assumptions.
`baseline_rate=0.03` uses a proper Beta prior centered on a pre-experiment 3%
control rate. Omit it when that information is not justified. It changes
power, not validity.

## Ordinary continuous monitoring

<a id="continuous-scalar-means"></a>

Use `asymptotic_mean` for a continuous mean. It also accepts conversion and
retention as scalar 0/1 outcomes, and ratios through the dedicated ratio
model. The source derives the registration; no manual population flags are
needed.

The method assumes independent, identically distributed stationary
potential-outcome vectors; independent fixed Bernoulli assignment; finite
absolute moments of order `2+delta` for some `delta>0`; positive limiting
arm variances; a positive population control mean; consistency and no
interference; and finalized, outcome-independent joint reveal.
Treatment means may be negative. Adaptive allocation, calendar drift and
outcome-dependent reveal do not meet this contract.

Reuse the assignment and unit IDs above, but declare a new continuous-outcome
experiment before reading its outcomes:

```python
mean_plan = increment.AnalysisPlan(
    primary="revenue",
    inference=increment.InferenceSpec(
        kind="asymptotic_mean",
        expected_decision_sample_size=5000,
    ),
)
revenue_frame = finalized_frame.select("user_id", "variant", "enrolled_on").with_columns(
    pl.Series("revenue", [10.0 + rng.uniform(-2, 2) + (arm == "treatment") for arm in arms])
)
mean_experiment, mean_reader = increment.from_frame(
    revenue_frame,
    shape="summary",
    unit="user_id",
    group="variant",
    design=design,
    metrics=[increment.MetricSpec(name="revenue", type="mean")],
    experiment_id="revenue-experiment",
    plan=mean_plan,
    exposure_date="enrolled_on",
)
mean_row = increment.run(mean_experiment, mean_reader).readout()[0]
confidence_set = mean_row.require_asymptotic_sequential_result().bounds
```

Read `confidence_set.components`, not just two endpoints: the set can be
disconnected or unbounded, with no −1 lift floor. A zero observed variance
or fewer than two observations in an arm withholds an informative result;
it does not establish zero population variance.

!!! warning "Two observations are not a calibration guarantee"
    The operational minimum is two per arm. Selected calibration checks start
    at 40 per arm; smaller `expected_decision_sample_size` targets increase
    early-look approximation risk. Neither the default target of 5000 nor a
    minimum count protects against every heavy-tailed distribution.

For ratios, population denominator means must be positive and bounded away
from zero. CUPED needs a supported per-unit pre-period covariate: panels
support bounded-window unadjusted ratios, not adjusted laws; artifacts need
the published `cuped_preperiod` extension. See
[joint-law requirements](../reference/sequential-contracts.md#adjusted-means-and-ratio-metrics-the-joint-laws).
The library commits before its own capture; it cannot know whether someone
looked at outcomes elsewhere.

## Error control by plan role

Primaries share the plan's `alpha`; guardrails test at full `alpha` and must
all be certified for the compound safety claim. Secondaries use e-BH at `q`,
with a finite-sample guarantee only when every member is exact.
Filtering does not shrink the family. Freezing a secondary stops its
evidence, not its family selection: its discovery can change at later looks.
See [Multiplicity](multiplicity.md) and the
[exact role allocations](../reference/family-inference.md#error-control-by-plan-role).

## Triggered populations

Triggered monitoring is available on declared warehouse/artifact sources;
frames have no trigger declaration, and moments can only replay an existing
chain. Assigned and triggered results describe separate populations and
separate families.

Trigger membership and timing must be treatment-independent, with
counterfactual logging and no pre-trigger treatment effect. Trigger timing
must carry no information about the retained outcome/covariate vector.
Balanced trigger rates cannot establish these assumptions.

Choose triggered family looks, stops and freezes using only information
already revealed by its joint triggered chain. Assigned results—including
the same metric—can expose future triggered outcomes and invalidate a
family claim. Review `increment.workflows.health_check(experiment, reader)`,
feed completeness, outcome maturity and each unavailable reason; a health
check is not proof of causal validity.
See [triggered capture and guarantees](../reference/sequential-contracts.md#triggered-populations).

## Registration and method reference

<a id="manual-registration-for-low-level-integrations"></a>

- [Manual registration](../reference/sequential-contracts.md#manual-registration):
  exact declarations, source identity, capture, continuation and checkpoint encoding.

<a id="exact-conversion-route-details"></a>
<a id="choosing-the-beta-prior-weight-for-a-declared-baseline-rate"></a>

- [Exact conversion registration and prior weight](../reference/sequential-contracts.md#exact-conversion-route-details).

<a id="adjusted-means-and-ratio-metrics-the-joint-laws"></a>

- [Adjusted means and ratio laws](../reference/sequential-contracts.md#adjusted-means-and-ratio-metrics-the-joint-laws).

<a id="guardrails-one-sided-non-inferiority-uncorrected-across-guardrails"></a>

- [Guardrail allocation and the all-guardrails rule](../reference/family-inference.md#guardrails-one-sided-non-inferiority-uncorrected-across-guardrails).

<a id="secondaries-false-discovery-rate-control-at-q-for-every-in-family-secondary"></a>

- [Secondary selection and unavailable members](../reference/family-inference.md#e-bh-under-sequential-inference).

<a id="what-the-triggered-claim-assumes"></a>

- [Triggered assumptions](../reference/sequential-contracts.md#what-the-triggered-claim-assumes).

<a id="a-practical-triggered-health-review"></a>

- [Triggered health review](../reference/sequential-contracts.md#a-practical-triggered-health-review).

<a id="per-cell-coverage-and-the-family-verdict"></a>

- [Per-cell coverage versus family decisions](../reference/sequential-contracts.md#per-cell-coverage-and-the-family-verdict).

<a id="certified-feeds-not-evidence-presence"></a>

- [Certified feed requirements](../reference/sequential-contracts.md#certified-feeds-not-evidence-presence).

<a id="processes-begun-without-a-triggered-commitment"></a>

- [Continuing an assigned-only checkpoint](../reference/sequential-contracts.md#processes-begun-without-a-triggered-commitment).

<a id="explicit-research-campaign"></a>

- [Research campaign interpretation](../reference/sequential-contracts.md#explicit-research-campaign).

<a id="primary-references"></a>

- [Primary references](../reference/sequential-contracts.md#primary-references).
