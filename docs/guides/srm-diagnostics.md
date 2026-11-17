# SRM diagnostics

A sample-ratio-mismatch (SRM) check asks whether assigned arm counts fit the
declared allocation. An alarm can reflect broken assignment, asymmetric
logging or filtering; it does not identify the cause. A non-rejection does
not prove assignment was valid, and metric intervals do not replace this check.
CUPED has a separate [covariate balance diagnostic](cuped.md#when-cuped-misleads-a-covariate-that-differs-by-arm).

<span id="assignment-protocol-requirements"></span>

## Declare the assignment law

Use expected allocation shares fixed before observing counts. Neither fixed
nor always-valid SRM infers the null from observed shares. For example,
`allocation={"control": 4, "treatment": 1}` means 80/20, not 50/50.

Declare `allocation_scheme="independent"` only when every assignment is an
independent draw with the same known arm probabilities. Constant marginal
shares alone are insufficient: blocked, adaptive, dependent, quota, exact-
balance, without-replacement, ramped or reset streams do not meet the
always-valid contract. Do not substitute repeated fixed-look tests.

For cluster randomization, count distinct assigned clusters with
`grain="cluster"`; descriptive `unit_counts` do not change `is_srm`.
Encouragement counts randomized assignment, not uptake. Switchbacks have
their own assignment law; observational comparisons have no randomized
allocation. Unsupported or inapplicable is not “no SRM.”

Check all assigned/targeted units, or a demonstrably pre-treatment,
arm-invariant exposure. A treatment-affected triggered subset can reflect
selection or telemetry rather than assignment failure.

## Check a bound experiment

`srm` reads one population’s result; `health_check` reads one source snapshot
and labels each assignment/trigger check by population:

```python
import pyarrow as pa

import increment
from increment.workflows import health_check, srm

units = pa.table(
    {
        "unit": [f"u{i}" for i in range(40)],
        "arm": ["control", "treatment"] * 20,
        "value": [float(i % 7) for i in range(40)],
    }
)
experiment, reader = increment.from_frame(
    units,
    shape="summary",
    unit="unit",
    group="arm",
    metrics={"value": "mean"},
    design=increment.Randomized(
        control_group="control",
        allocation={"control": 0.5, "treatment": 0.5},
        allocation_scheme="independent",
    ),
)
fixed = srm(experiment, reader, inference="fixed")
print(f"fixed: is_srm={fixed.is_srm} grain={fixed.grain}")
for check in health_check(experiment, reader).checks:
    print(check.check, check.analysis_population, check.availability, check.conclusion)
```

<!-- output -->
```text
fixed: is_srm=False grain=unit
assigned_allocation assigned available not_rejected
triggered_balance triggered not_applicable None
trigger_rates triggered not_applicable None
trigger_maturity triggered not_applicable None
feed_completeness triggered not_applicable None
trigger_assumptions triggered not_applicable None
```

Each check is `available`, `unavailable` (the source cannot answer; the check
carries the refusal code), or `not_applicable` (the design does not call for
it). Only an available check has a conclusion, and `not_rejected` does not
prove valid assignment. A source that retains only portable moments reports
its triggered checks as unavailable rather than reconstructing the trigger
population.

Automatic `run()` checks use cumulative assigned counts at randomization
grain and require declared constant allocation plus the independent scheme.
Otherwise `health_check()` and `print(results)` retain an unavailable status,
such as `not_checked_missing_declaration`. Registered sequential reads use
source-owned assigned counts when present, never checkpoint sample sizes.
Clustered artifacts use assigned clusters, not member units. Trigger counts
are not assignment-integrity evidence.

To investigate an alarm, compare assignment records with downstream counts
at the same grain and population. Check for missing logs, arm-dependent
filters and incorrect allocation declarations before interpreting outcome
differences. Balanced trigger rates do not validate trigger-selection assumptions.

## `sample_ratio_mismatch`: fixed vs always-valid

For counts you already have, pass `expected` explicitly:

```python
from increment.estimators import allocation_posterior_bands, sample_ratio_mismatch

expected = {"control": 0.5, "treatment": 0.5}
observed = {"control": 5120, "treatment": 4880}

fixed = sample_ratio_mismatch(observed, expected=expected, inference="fixed", alpha=0.001)
print(
    f"fixed: is_srm={fixed.is_srm} p={fixed.fixed_p_value:.4f} "
    f"low_expected_count={fixed.low_expected_count}"
)

always_valid = sample_ratio_mismatch(observed, expected=expected)
print(f"always_valid: is_srm={always_valid.is_srm} log_e_value={always_valid.log_e_value:.3f}")
```

<!-- output -->
```text
fixed: is_srm=False p=0.0164 low_expected_count=False
always_valid: is_srm=False log_e_value=-1.499
```

A fixed check uses the chi-square `fixed_p_value` at one predeclared look.
The default always-valid check uses a uniform-Dirichlet mixture, alarming
when `log_e_value >= -log(alpha)`. Repeated peeking is valid only under the
constant-conditional-probability assignment law above. Its e-process is
a distinct construction from a metric’s `AlwaysValid` interval; neither
establishes the other’s assumptions.

`increment.workflows.srm(experiment, reader)` accepts `expected` or the design’s
allocation. Without either, fixed and always-valid requests both refuse.
For one predeclared fixed look, declare the target and pass `inference="fixed"`;
declaring shares alone does not assert independent assignment.

## `allocation_posterior_bands`: a descriptive complement

```python
counts = [
    {"ds": "2025-01-01", "group_id": "control", "n_cumulative": 500},
    {"ds": "2025-01-01", "group_id": "treatment", "n_cumulative": 480},
    {"ds": "2025-01-02", "group_id": "control", "n_cumulative": 1010},
    {"ds": "2025-01-02", "group_id": "treatment", "n_cumulative": 970},
]
bands = allocation_posterior_bands(counts)
for b in bands:
    print(f"{b.ds} {b.group_id:<10} mean={b.mean:.3f} [{b.lower:.3f}, {b.upper:.3f}]")
```

<!-- output -->
```text
2025-01-01 control    mean=0.510 [0.479, 0.541]
2025-01-01 treatment  mean=0.490 [0.459, 0.521]
2025-01-02 control    mean=0.510 [0.488, 0.532]
2025-01-02 treatment  mean=0.490 [0.468, 0.512]
```

These bands are descriptive and have no pass/fail status. Use
`sample_ratio_mismatch` for the decision; use the per-day Beta credible bands
to visualize allocation shares over time.

## Reading `low_expected_count`

`low_expected_count=True` means the smallest expected arm count
(`expected[arm] * total`) is below 5, where the fixed chi-square p-value may be
unreliable. This flag does not change `is_srm`; it applies only to the fixed
asymptotic test, not the always-valid e-process.
