# SRM diagnostics

Unequal arm counts can mean assignment failed, or that downstream filtering or
logging lost data unevenly. Either can bias results without changing a metric's
interval.

The sample-ratio-mismatch (SRM) check is a separate assignment-integrity
diagnostic. It flags the count imbalance; it does not explain its cause or
replace a metric's decision interval.

## `sample_ratio_mismatch`: fixed vs always-valid

```python
from increment import allocation_posterior_bands, sample_ratio_mismatch

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

`inference="fixed"` tests one predeclared look using `fixed_p_value` and
`alpha`; it is not valid for repeated peeking. The default
`inference="always_valid"` tracks `log_e_value`, evidence from a
uniform-Dirichlet mixture against the declared allocation. It alarms at
`-log(alpha)` and supports peeking only when assignment prefixes have the same
known arm probabilities at every assignment.

Static marginal shares are not enough: blocked, adaptive, dependent, quota,
exact-balance, without-replacement, ramped, and reset assignment streams do
not meet this contract. Independently assigned clusters do, when checked at
cluster grain.


## Declare the assignment law

Proportions alone do not identify the assignment law. Declare
`allocation_scheme="independent"` only when each `Randomized` or
`Encouragement` assignment is an independent draw with the same arm
probabilities throughout the experiment. Older definitions may omit this
field; declaring `allocation` alone does not imply independence.

```python
from increment import Randomized

design = Randomized(
    control_group="control",
    allocation={"control": 4, "treatment": 1},
    allocation_scheme="independent",
)
```

For cluster randomization, count distinct assigned clusters by arm and use
`grain="cluster"`. For encouragement, count randomized assignment, not
uptake. Expected shares may be unequal and come from the declared allocation.

Blocked, adaptive, quota, and fixed-count assignment do not meet the
independent-draw e-process contract. Do not replace it with repeated fixed-look
Pearson tests. Switchbacks have their own assignment law; observational
comparisons have no randomized allocation. Report these as unsupported or not
applicable, not "no SRM."

`sample_ratio_mismatch(inference="fixed")` assumes equal shares when
`expected` is omitted. `Analysis.srm(inference="fixed")` uses the design's
declared `allocation` when present, falling back to equal shares only when
neither the design nor the call supplies an allocation. This example's 4:1
allocation therefore means 80/20, not 50/50.
Automatic SRM checking requires `allocation_scheme="independent"`, declared
constant `allocation` weights, and cumulative assigned counts at the
randomization grain. Otherwise `health_check()` records an unavailable status
(for example, missing allocation is `not_checked_missing_declaration`). You can
also call `srm(inference="fixed")` for a single predeclared look.

??? info "How the always-valid SRM guarantee differs"
    That e-process is a distinct construction from the `AlwaysValid` guarantee
    a metric's own interval carries (see
    [Sequential inference](sequential-inference.md)), but it earns the same
    kind of time-uniform, peek-anytime validity. Its precondition is that
    cumulative assignment prefixes carry the same known conditional arm
    probabilities at every assignment.

## Assignment-protocol requirements

`inference="always_valid"` REQUIRES a predeclared `expected` allocation -- it
cannot infer equal shares from the observed counts the way fixed inference
can, since an always-valid check that adapted its own null to the data it is
checking would not be a check at all. Omitting `expected` raises. For
cluster-randomized designs, use `grain="cluster"` with cluster-level `counts`.
`unit_counts` is descriptive only: cluster-size imbalance does not change
`is_srm`.

Check all assigned or targeted units, or a demonstrably pre-treatment,
arm-invariant exposure. A treatment-affected triggered subset can reflect
selection or telemetry, not assignment failure.

`Analysis.run()` attaches automatic integrity results to the assigned
population. Triggered counts are not checked as assignment integrity and say
nothing about trigger-selection validity. Clustered artifact readouts use
assigned clusters, not member units. Registered sequential reads use
source-owned assigned counts when available, never checkpoint sample sizes.

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
