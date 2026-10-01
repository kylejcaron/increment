# SRM diagnostics

A wrong arm allocation can have two distinct causes:
a broken assignment mechanism, such as a bad hash or bucketing step, that
puts the wrong share of units into each arm; or an asymmetric downstream
selection or telemetry gap, such as a bot filter or logging failure, that
skews observed counts without changing assignment. Either can silently bias
downstream estimates without the estimates' intervals revealing that bias.

Increment's sample-ratio-mismatch (SRM) check is an independent
assignment-integrity gate. It flags the observed-count symptom regardless of
which failure produced it and remains separate from each metric's decision
interval.

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

```text
fixed: is_srm=False p=0.0164 low_expected_count=False
always_valid: is_srm=False log_e_value=-1.499
```

`inference="fixed"` (opt in explicitly) reads `fixed_p_value` against
`alpha` at one look -- valid once, not under repeated peeking.
`inference="always_valid"` (the default) instead tracks
`log_e_value`, the running log evidence of a uniform-Dirichlet mixture
e-process against the declared `expected` allocation, and alarms when it
reaches `-log(alpha)`. That e-process is a distinct construction from
the `AlwaysValid` guarantee a metric's own interval carries (see
[Sequential inference](sequential-inference.md)), but it earns the same
kind of time-uniform, peek-anytime validity -- and that validity has its
own precondition: cumulative assignment prefixes must carry the same
known conditional arm probabilities at every assignment. Static
marginal shares alone do not suffice, and blocked, adaptive, dependent,
quota, exact-balance, without-replacement, and ramped or reset
assignment streams are unsupported (independently assigned clusters
satisfy the contract at cluster grain).

## Assignment-protocol requirements

`inference="always_valid"` REQUIRES a predeclared `expected` allocation
-- it cannot infer equal shares from the observed counts the way fixed
inference can, since an always-valid check that adapted its own null to
the data it is checking would not be a check at all. Omitting `expected`
under always-valid inference raises. For cluster-randomized designs,
pass `grain="cluster"` with cluster-level `counts`; `unit_counts` then
carries per-arm unit totals purely for descriptive context -- cluster
size imbalance earns no degree of freedom in this test, so it never
changes `is_srm`.

Apply the check to all assigned or targeted units, or to a demonstrably
pre-treatment, arm-invariant exposure. A treatment-affected triggered
subset -- e.g. only units that reached some later page after
assignment -- is selection or telemetry evidence, not evidence that
randomization itself failed; running SRM on that subset can move
`is_srm` for reasons that have nothing to do with the assignment
mechanism.

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

```text
2025-01-01 control    mean=0.510 [0.479, 0.541]
2025-01-01 treatment  mean=0.490 [0.459, 0.521]
2025-01-02 control    mean=0.510 [0.488, 0.532]
2025-01-02 treatment  mean=0.490 [0.468, 0.512]
```

This is purely descriptive: it carries no pass/fail flag.
`sample_ratio_mismatch` above is the actual gate; use these per-day Beta
credible bands to visualize how the allocation share trended, not to
decide whether an experiment has SRM.

## Reading `low_expected_count`

`low_expected_count=True` flags that the smallest per-arm expected count
(`expected[arm] * total`) fell below 5 -- the standard Cochran rule of
thumb for when the fixed chi-square test's asymptotic p-value may be
unreliable. It is informational only: it does not change `is_srm`, and
it applies to the fixed-inference branch's asymptotic approximation, not
to the always-valid e-process (which has no such small-count asymptotic
requirement).
