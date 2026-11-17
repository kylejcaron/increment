# Logged-policy evaluation

Use `estimate_policy_contrast` to compare two policies offline from decisions
collected under one **fixed** logging policy. You need each unit's complete
decision history, pre-decision context, chosen action, exact decision-time
action probability, and closed reward.

The result is the target policy's mean reward per decision minus the reference
policy's, averaged over a fixed horizon. It is not a percentage lift.
The logger must support every action either evaluated policy could take.
An updating bandit is unsupported; the interval is asymptotic in independent
units, not a finite-sample guarantee.

## Example

This complete example simulates a uniform fixed logger. The target chooses `B`
with probability 0.8 when `x=0` and 0.2 when `x=1`; the reference chooses `B`
with probability 0.5 at every history.

```python
from increment.estimators import TabularPolicy, estimate_policy_contrast
from increment.simulate import simulate_logged_run

run = simulate_logged_run("fixed_random", n_units=200, horizon=3, seed=7)
target = TabularPolicy(
    policy_id="context-aware",
    version="v1",
    probabilities={0: {"A": 0.2, "B": 0.8}, 1: {"A": 0.8, "B": 0.2}},
)
reference = TabularPolicy(policy_id="uniform", version="v1", default={"A": 0.5, "B": 0.5})
contrast = estimate_policy_contrast(run.trace, target, reference, alpha=0.05)
assert contrast.estimator == "trajectory_hajek_ipw"
assert contrast.n_units == 200 and contrast.horizon == 3
assert contrast.lb < contrast.estimate < contrast.ub
assert min(contrast.ess_by_time) >= contrast.ess_floor
print(f"Estimated target reward exceeds reference reward: {contrast.estimate > 0}")
print(f"Difference: {contrast.estimate:.3f}; 95% interval: [{contrast.lb:.3f}, {contrast.ub:.3f}]")
```

The sign tells you which policy has the higher estimated reward; `lb` and `ub`
show the uncertainty on that reward scale. An interval containing zero does not
establish an improvement. These are estimates for this simulated setup, not
proof that the policy will improve a new population.

## What it estimates

The estimator weights each decision by the whole action history up to that
point, so earlier actions can affect later contexts. It compares
self-normalized weighted rewards and uses a unit-clustered standard error
with a Student-t reference.

Read `target_value` and `reference_value` for the separate policy values;
`ess_by_time` is the smaller effective sample size (ESS) of the two policies
at each decision. `logging_policy_versions` identifies the fixed logger.
[Equations and result fields](../reference/causal-methods.md#policy-value-and-uncertainty)
give the exact construction.

## Assumptions and guarantee

- Keep one fixed `(policy_id, version)` throughout the trace. Refitting during
  collection raises `logged_policy.inference.adaptive_logging_unsupported`.
  Relabelling an adaptive logger as one version does not make it fixed.
- Supply exact recorded probabilities, checked against either a registered
  logger or complete recorded action distributions. Increment does not infer,
  normalize, clip, or trim them; your records must truthfully describe collection.
- Every positive candidate-action logging probability must be at least
  `PROPENSITY_FLOOR = 0.05`, including for unchosen actions. Zero is allowed only
  when both evaluated policies assign zero. Set support in the logger; filtering
  traces afterward can select on outcomes and change the population.
- You need at least 20 independent units and ESS of at least 20 for **both**
  policies at **every** decision. These measured admission floors do not promise
  nominal coverage on all data.

A shortfall reports `logged_policy.inference.cluster_floor` or
`logged_policy.support.ess_floor`, with per-decision ESS, maximum cumulative
weight and minimum propensity. Read the [support rules](../reference/causal-methods.md#fixed-logger-and-support)
and [ESS calibration evidence](../reference/causal-methods.md#ess-calibration-evidence)
before changing data collection.

Real data enter through `LoggedTrace.from_records` or `LoggedTrace.from_frame`,
not an experiment builder or `run`. The
[complete import example](../reference/causal-methods.md#importing-decision-records)
shows both registered loggers and recorded distributions, including every
required input.

### Persisting a `TabularPolicy`

Python persistence preserves supported context-key types; pickle is for trusted
data only. JSON requires string table keys and refuses other types rather than
silently merging keys such as `1` and `"1"`. See
[persistence rules](../reference/causal-methods.md#persisting-a-tabularpolicy).

## Interaction with the rest of the package

- One `alpha` per call; comparing several policies adds no family correction.
- Units are independent trajectories; shared clusters beyond the unit are
  not modelled.
- Every unit must have all horizon rewards closed. Re-running on a growing
  trace is repeated fixed-horizon inference, with no anytime validity.
- No CUPED, winsorization, ratio rewards or breakouts.
