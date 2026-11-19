# Power analysis and MDE for A/B tests

Before launch, choose a control-arm baseline, a useful effect and the decision
rule you will run. Power is the chance that rule rejects under the planning
model—not a guarantee that the experiment will succeed.

## Size a conversion experiment

The standard conversion plan, a 10% baseline rate with a 10% relative lift (to
11%), tested two-sided at `alpha=0.05` for 80% power, needs 15,597 units per
arm:

<!-- skip: next "enumerating the conversion count lattice takes about a minute" -->
```python
from increment.estimators import Baseline
from increment.power import ArmPlanningProcedure, required_sample_size

plan = required_sample_size(
    relative_lift=0.10,
    baseline=Baseline.from_proportion(0.10),
    procedure=ArmPlanningProcedure.standard("conversion"),
)
print(plan.n_per_arm, plan.n_total, plan.power_basis)
print(f"power={plan.power:.4f} in [{plan.power_lower:.4f}, {plan.power_upper:.4f}]")
```

```text
15597 31194 lower_bound
power=0.8000 in [0.8000, 0.8003]
```

Recruit 31,194 assigned units at equal allocation. The reported `power` is
the lower end of the computed range, not its midpoint. This calculation
replays an eligible binary decision; its numerical qualification is
conditional on the deployed special-function error model. It is not a
guarantee against model misspecification.

## What `power` and `power_lower` mean

- `exact`: the enumerated point is within `1e-6` of both reported endpoints.
- `lower_bound`: `power` is `power_lower`; the numerical range is wider.
- `asymptotic`: a model-based probability, with no `power_lower`/`power_upper`.

Read `power_basis` and `numerical_qualification` before quoting an answer.
“Exact” describes the decision-law calculation, not error-free arithmetic or
a finite-sample guarantee for a hybrid test. A size or MDE meets its stated
target but need not be a global minimum.
[The planning reference](../reference/planning-methods.md#what-power-and-power_lower-mean)
gives the enclosure, search and model qualifications.

## Size a mean metric

Supply the control mean and **per-unit variance**, not the variance of its
sample mean. These calls solve size, power and detectable lift respectively:

```python
from increment.estimators import Baseline, PowerDesign
from increment.power import (
    ArmPlanningProcedure,
    achieved_power,
    minimum_detectable_effect,
    required_sample_size,
)

baseline = Baseline(mean=20.0, var=400.0)
# The default procedure is a fixed-horizon standard for this baseline's metric.
size = required_sample_size(
    relative_lift=0.02,
    baseline=baseline,
    design=PowerDesign(power=0.80, allocation=0.5),
)
power = achieved_power(25_000, 0.02, baseline)
mde = minimum_detectable_effect(25_000, baseline)
```

`relative_lift=0.02` means a 2% change: from 20 to 20.4 in this example.
`n_per_arm` always counts assigned treatment units; control size follows
from `PowerDesign.allocation`. At 0.5 allocation the arms are equal.
`n_total` counts both arms.

The baseline type selects the standard procedure when `procedure` is omitted.
Use an explicit `ArmPlanningProcedure` to change alpha, sidedness, inference
or decision method; those are not direct solver arguments. `PowerDesign`
sets only target power (default 0.80) and treatment allocation (default 0.5).

## Build a baseline

- `Baseline(mean=..., var=...)`: assumed or externally measured control moments.
- `Baseline.from_proportion`: a conversion or retention rate.
- `Baseline.from_summary`: pilot moments, raising variance to its 80%
  one-sided chi-square upper estimate.
- `Baseline.from_ratio`: numerator/denominator moments and covariance.
- `Baseline.from_absorption`: a measured one-way factor reduction.
- `increment.workflows.planning_baseline(experiment, reader, metric)`: read a
  prior/pilot experiment or pre-treatment data with its declared design.

Without such data, `planning_baseline` has nothing to read; use an assumed
value instead. It returns `QuantileBaseline` for quantiles and
`SwitchbackBaseline` for switchbacks. For ordinary pilot metrics it raises
variance as `from_summary` does; that is a model-based allowance, not a
heavy-tail guarantee.

A triggered pilot must retain its trigger and assignment data. A clustered
pilot also needs cluster identities and outcomes for every eligible control
unit. Finish incomplete windows; waiting cannot define an outcome such as
`avg_event` for a unit with no events. See
[pilot inputs and refusal routes](../reference/planning-methods.md#what-planning_baseline-reads).

### Will CUPED help at our expected traffic?

You expect 10,000 users per arm. Can you detect a 7% lift in revenue per user
with 80% power? Compare two fixed-horizon plans using the same pilot users.
Use one row per user, including nonbuyers. Measure `pre_revenue` before
assignment and `revenue` over the outcome window you intend to use.

Here is a synthetic pilot you can replace with your own dataframe:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 4_000
purchase_chance = rng.beta(0.1, 0.4, n)
pilot = pd.DataFrame({
    "user": range(n),
    "arm": ["control", "treatment"] * (n // 2),
    "pre_revenue": 20 * rng.binomial(1, purchase_chance),
    "revenue": 20 * rng.binomial(1, purchase_chance),
})
```

Estimate the baseline from the pilot's control users. For the unadjusted
comparison, keep the same mean and variance but turn off the CUPED reduction:

```python
import increment as inc
from increment.power import achieved_power, minimum_detectable_effect
from increment.workflows import planning_baseline

experiment, reader = inc.from_frame(
    pilot, shape="summary", unit="user", group="arm", control="control",
    metrics=[inc.MetricSpec(
        name="revenue", type="mean",
        decision_method=inc.Method.cuped(covariates="pre_revenue"),
    )],
)
cuped = planning_baseline(experiment, reader, "revenue")
plain = cuped.model_copy(update={"cuped_rho": 0.0})

for label, candidate in [("Unadjusted", plain), ("CUPED", cuped)]:
    mde = minimum_detectable_effect(10_000, candidate).mde_relative
    power = achieved_power(10_000, 0.07, candidate).power
    print(f"{label}: MDE {mde:.1%}, power at +7% {power:.1%}")
```

<!-- output -->
```text
Unadjusted: MDE 7.9%, power at +7% 69.6%
CUPED: MDE 5.7%, power at +7% 93.3%
```

Only the CUPED plan reaches the 80% target for a 7% lift. Choose it as the
**decision method** if prior revenue is available at launch and you expect
it to predict revenue about as well as it did in the pilot.

### Quantile metrics

A quantile baseline keeps the real per-unit pilot values and projects the
readout at each candidate size; a variance alone is insufficient. Quantile
results are relative-lift only, two-sided against zero, without CUPED or
segment-pairwise planning.

!!! warning "Check a quantile plan against the readout"
    Rounded data can impose a power ceiling, and projected power need not
    increase monotonically with size. Near the pilot size, rounded-data checks
    measured 0.71–0.76 power for a planned 0.80. Passing the projected
    log-SE guard also does not guarantee a real readout reports a lift.
    Run the [complete pilot-resampling check](../reference/planning-methods.md#checking-a-quantile-plan-against-the-readout)
    before trusting a small-size plan.

The [quantile construction and refusal codes](../reference/planning-methods.md#quantile-planning)
explain recording-grid ceilings, feasibility floors and the 0.5 log-SE limit.

### Save and reload a baseline

`model_dump_json()` is a baseline's saved form, and
`Baseline.model_validate_json` restores it as the class it was saved from. A
quantile baseline comes back with its pilot values and plans exactly as
before:

```python
import numpy as np
import pandas as pd

import increment
from increment.estimators import Baseline
from increment.workflows import planning_baseline

latency = np.round(np.random.default_rng(3).lognormal(3.0, 0.5, 400), 2)
pilot = pd.DataFrame(
    {"unit": range(400), "group": ["control", "treatment"] * 200, "latency": latency}
)
experiment, reader = increment.from_frame(
    pilot,
    shape="summary",
    unit="unit",
    group="group",
    control="control",
    metrics=[{"name": "latency", "type": "quantile", "quantile": 0.5}],
)
latency_baseline = planning_baseline(experiment, reader, "latency")

text = latency_baseline.model_dump_json()
restored = Baseline.model_validate_json(text)
assert restored == latency_baseline
```

A quantile dump without its pilot values, such as one saved by an earlier
release, cannot plan; restoring it refuses with
`power.quantile_baseline.pilot_values_missing`. Rebuild it with
`planning_baseline`.

## Adjust the baseline for your design

CUPED credit follows the **decision** method: `cuped_rho` reduces its planning
variance by `1 - cuped_rho**2`. A sensitivity-only CUPED result earns no credit
for an unadjusted decision. Scalar CUPED can compose with prospective
`asymptotic_mean` planning. Multicovariate `cuped_linear` planning refuses
with `power.cuped_linear_unavailable`; a scalar substitute is an approximation,
not guaranteed conservative when extra fitted columns cost degrees of freedom.

`compliance` accounts for diluted encouragement effects; `trigger_rate` is
the assigned share analyzed. Clustered plans also need assigned mean cluster
size, analyzed-score ICC and the size CV across **contributing** clusters.
For triggered clusters, `cluster_participation` is the fraction of recruited
clusters contributing, not the unit trigger rate.

Cluster counts and t references are pilot-conditioned approximations. They
require at least two represented clusters per arm, not ten; two does not
establish calibration. See the [cluster formulas and reference](../reference/planning-methods.md#cluster-design-effect-and-reference).

### Secondary metrics and guardrails

Use `role="secondary"`, `secondaries=<count>` and `q=<FDR target>` for a
Bonferroni-at-`q/m` planning calculation. For a binary secondary, enumerating
that count decision can **upper-bound**, not lower-bound, the runtime member
rejection probability; see [secondary-member power](../reference/planning-methods.md#secondary-member-power).

A guardrail uses `role="guardrail"` and a declared favorable direction or
one-sided alternative. It gets full alpha, never an arm-split share.

## Curves

Supply exactly one of `relative_lift` and `target_power`:

```python
from increment.power import power_curve

curve = power_curve(
    n_per_arm=[5_000, 10_000, 20_000],
    relative_lift=[0.01, 0.02, 0.05],
    baseline=baseline,
    units_per_week=50_000,
)
print(curve.to_frame())

mde_curve = power_curve(
    n_per_arm=[5_000, 10_000, 20_000],
    target_power=[0.80, 0.90],
    baseline=baseline,
)
```

Rows keep input product order and duplicates. `units_per_week` adds
`duration_days` for assigned units; it is not a recruitment forecast.
Sequential curves also report expected stopping size/duration beside the
worst-case totals. See the [curve fields](../reference/planning-methods.md#power-and-mde-curves).

## Fixed and sequential planning

Use a matching procedure before planning repeated looks. This prospective
`asymptotic_mean` calculation uses a Gaussian approximation on an equal-look
schedule (`planned_looks`, default 14):

```python
import increment
from increment.power import ArmPlanningProcedure

procedure = ArmPlanningProcedure.standard(
    "mean", inference=increment.InferenceSpec(kind="asymptotic_mean")
)
sized = required_sample_size(0.02, baseline, procedure, planned_looks=14)
# Declare the same boundary at runtime registration:
runtime_inference = sized.inference_to_declare
```

Pass `inference_to_declare` into runtime registration so its target sample
size matches the plan. This does not certify power for the continuous
data-dependent construction or make unplanned fixed-horizon peeking valid.
Exact `AlwaysValid` likelihood-policy planning and clustered sequential
planning are unsupported. See [sequential calculation and limits](../reference/planning-methods.md#fixed-and-sequential-planning).

## Reading `PowerResult`

Read the answer at its actual integer size. `mde_relative=None` means no
numeric MDE was certified; its `mde_unavailable_reason` distinguishes
unattainability, representability and numerical uncertainty. It is not zero
lift. `effective_var` reflects the decision method, not sensitivity-only
adjustments. The [field reference](../reference/planning-methods.md#reading-powerresult)
defines every count, bound and missing-value reason.

### Small samples and the runtime’s log-scale guard

For eligible log-ratio routes, size planning also keeps the projected
combined log-scale SE below 0.5. A real sample can still fail the readout’s
guard. See the [mean example and exact eligibility](../reference/planning-methods.md#small-samples-and-the-runtimes-log-scale-guard).

### Checking a quantile plan against the readout

[Resample two distinct arms from a large enough pilot](../reference/planning-methods.md#checking-a-quantile-plan-against-the-readout)
and measure the share of unavailable lifts. The reference’s heavy-tail
example reports no lift in 78% of draws at the planned 45 units per arm,
versus 2% at 100. Those are scenario measurements, not a universal size rule.

### An MDE against a shifted null

`mde_relative` is measured from the declared null, not control. At full
compliance, compose it as `(1 + null_lift) * (1 + mde_relative) - 1` before
passing it back as `relative_lift`; see the [complete example](../reference/planning-methods.md#an-mde-against-a-shifted-null).

## Switchback contrasts

Use the pilot-fitted `SwitchbackBaseline` and a `ContrastDecisionProcedure`
with the `switchback_*` solvers. Their `n` counts independent **units** for
independent orders or **blocks** for shared schedules, never cycles times
roster size. Effects and MDEs are absolute, not relative.

Power includes the probability that the schedule admits a result. The
noncentral-t approximation conditions on pilot moments: it does not integrate
pilot uncertainty, certify population moments or establish no residual
carryover. Conversion aggregation `"any"` refuses with
`power.switchback.bounded_model_required`.

Start with the [switchback workflow](switchback.md); consult
[baseline, result and inversion contracts](../reference/planning-methods.md#switchback-contrasts)
for moment-t planning versus certified envelopes and complete joint models.

<span id="conversion-and-retention-planning-follows-the-runtimes-route"></span>

- [Conversion/retention routing and numerical budgets](../reference/planning-methods.md#conversion-and-retention-planning-follows-the-runtimes-route).

<span id="when-it-refuses"></span>

- [Binomial search refusals and contexts](../reference/planning-methods.md#when-it-refuses).

<span id="comparing-with-statsmodels"></span>

- [Compare aligned tests with statsmodels](../reference/planning-methods.md#comparing-with-statsmodels).
