# Logged-policy evaluation

`estimate_policy_contrast` compares two decision policies on a trace of
decisions that a *different*, fixed policy actually made and logged. Use it
when a bandit or rule ran in production under one frozen policy version and
recorded the exact action probability for every decision.

The estimator answers how much better a candidate policy would have done than a
reference policy over the same horizon, without deploying either.

This family has its own ingress. It does not run through any `Analysis`
constructor, because none of them carries a decision trace; the "What runs
where" table in [Statistical limitations](../limitations.md) records that as a
SOURCE refusal on every path.

## What it estimates

A trace holds, for each unit, `T` decisions in order. At decision `t` the
logging policy saw a history `H_t`, chose an action `A_t` with probability
`p_t = P(A_t | H_t)` (the logged **propensity**), and a reward `Y_t` closed
later. A candidate policy `pi` assigns its own probability `pi(A_t | H_t)` to the
same action at the same history.

The estimand is the **finite-horizon dynamic policy value contrast**

```text
Delta_T = V_T(target) - V_T(reference),    V_T(pi) = (1/T) sum_t E_pi[Y_t]
```

the difference in mean reward per decision, averaged over the horizon, that the
target policy would have produced against the reference policy. Each `V_T(pi)`
is estimated by a self-normalized importance-weighted mean at every decision
index with the cumulative likelihood ratio
`W_t = prod_{s <= t} pi(A_s | H_s) / p_s`, so later contexts that depend on
earlier actions are re-weighted through the whole trajectory, not only the
current step. The interval is a two-sided Wald interval from a unit-clustered
sandwich standard error on a `t` reference with `n_units - 1` degrees of
freedom.

## Assumptions and guarantee

- **One fixed logging policy per trace.** Every logged decision must come from
  a single registered `(policy_id, version)` pair. IDs and versions are opaque:
  slashes are allowed, and identical display labels do not merge distinct pairs.
  A trace whose logging policy was refit
  while it was being collected -- an epsilon-greedy or Thompson-sampling
  bandit updating between batches -- is refused with
  `logged_policy.inference.adaptive_logging_unsupported`. The construction
  that makes the interval valid under adaptive collection (stabilized or
  adaptively weighted estimators with a martingale variance, as in Hadad et
  al. 2021 and Zhang, Janson and Murphy 2021) is not implemented; the route
  forward is one policy version per trace. An adaptive logger that never
  refit inside the trace logged one version and is admitted.
- **Exact logged propensities.** The propensity is evidence recorded at
  decision time. `LoggedTrace.from_records` / `from_frame` check every chosen
  propensity against either the registered logging policy or its complete
  recorded action distribution; nothing is inferred, normalized, clipped or trimmed.
- **A design-time propensity floor.** Every positive candidate-action logging
  probability must be at least `PROPENSITY_FLOOR = 0.05`, including unchosen
  actions. Admission checks the whole supplied distribution, not just the
  observed action. An exact zero is allowed only where the evaluated policies
  also assign zero mass. A smaller positive probability violates the declared
  weight bound, not the existence of a finite likelihood ratio. Enforce the
  floor in the logger: screening realized traces after the fact selects on
  the outcomes that drove the propensity down, and the traces that survive
  are not a representative sample.
- **Guarantee.** The interval is **asymptotic** in the number of independent
  units (trajectories). Nothing about it is finite-sample; the floors below
  are where measured coverage stops being nominal.

## Floors

Two floors refuse a trace whose interval would not be calibrated.

- **Independent units.** At least 20 units
  (`logged_policy.inference.cluster_floor`). The effective sample size of a
  policy can never exceed the unit count, so fewer units could never pass the
  ESS floor either; the unit count is named first because that is the actual
  shortfall.
- **Effective sample size.** For both policies and at every decision index,
  `ESS_t = (sum_i W_it)^2 / sum_i W_it^2` must be at least
  `ESS_FLOOR = 20` (`logged_policy.support.ess_floor`). The floor was
  measured, not chosen: on fixed-logger traces spanning target divergence,
  horizon and unit count (26 cells, 9188 emitted intervals), coverage of the
  95% interval by band of the smallest per-index ESS was 0.862 for
  `[2, 5)`, 0.892 for `[5, 10)`, 0.933 for `[10, 20)`, 0.933 for
  `[20, 40)`, 0.943 for `[40, 80)`, 0.947 for `[80, 160)` and 0.949 above
  160, each band holding 700 to 2190 intervals. `[20, 40)` is the smallest
  band from which every band stays within three Monte-Carlo standard errors
  of nominal. The shortfall below the floor is variance, not bias: the
  sandwich standard error is about 20% too small when the ESS is under five
  and 7% too small in `[10, 20)`, while the estimate's bias stays within its
  Monte-Carlo error everywhere.

Both refusals carry the ESS, maximum cumulative weight and minimum
propensity at every decision index in their context, so you can see how far
a trace is from admission.

## Example

The simulator in `increment.simulate` produces a trace under a fixed
uniform logger together with the registry that reproduces its logging law.
The target policy plays `B` with probability 0.8 when the context `x` is 0
and 0.2 when it is 1; the reference plays `B` with probability 0.5 at every
history.

```python
from increment import TabularPolicy, estimate_policy_contrast
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
```

`contrast.estimate` is `Delta_T` on the reward scale, with `lb`, `ub`,
`se` and `p_value` for the null of no difference. `target_value` and
`reference_value` are the two policy values, `ess_by_time` the smaller of
the two policies' effective sample sizes at each decision index, and
`logging_policy_versions` the single version the trace was logged under.

Real traces enter through `LoggedTrace.from_records` (an iterable of
`DecisionRecord` or mappings) or `LoggedTrace.from_frame` (a pandas,
polars or pyarrow frame with the record columns plus the context columns).
Supply exactly one of two alternatives. Existing calls can use a
`PolicyRegistry` holding every logging policy version the trace names:

<!-- skip: next "requires a caller-prepared decision frame" -->
```python
from increment import LoggedTrace, PolicyRegistry

registry = PolicyRegistry([logger_v1])
trace = LoggedTrace.from_frame(frame, registry=registry, context_columns=("x",))
```

If the logger stored the complete action probabilities at decision time, import
them directly instead of rebuilding an executable policy:

<!-- skip: next "requires caller-prepared records and their recorded logging laws" -->
```python
trace = LoggedTrace.from_records(
    records,
    logging_distributions=recorded_laws,
)
```

`recorded_laws` contains one mapping per input record, for example
`{"A": 0.2, "B": 0.3, "C": 0.5}` for candidates `("A", "B", "C")`.
The record and its mapping are paired before ordering by unit and decision
index, then copied into immutable snapshots. Every mapping must cover exactly
the candidate actions, contain finite nonnegative probabilities summing to one
within the existing tolerance, meet the positive-probability floor, and agree
with the chosen-action propensity. A chosen propensity alone is insufficient.

For a dataframe, store those mappings in a column:

<!-- skip: next "requires a caller-prepared decision frame with complete logging-law mappings" -->
```python
trace = LoggedTrace.from_frame(
    frame,
    logging_distribution_column="logging_distribution",
    context_columns=("x",),
)
```

Both alternatives retain support, fixed-logger identity/version, unit-count and
ESS checks. Recorded distributions are claims about the original logger, not
proof of their truthfulness. Do not relabel adaptive collection as one fixed
policy version; this admission route does not change the estimator or its guarantee.

Frame unit IDs may be strings or integers. Distinct native IDs that stringify
identically, such as `1` and `"1"`, refuse with
`estimation.crossfit.identity_collision` rather than becoming one trajectory.
The records constructor continues to require string IDs.

A `TabularPolicy` is an action distribution over one context key; any object
with `policy_id`, `version` and `probability(action, context)` satisfies the
`StochasticPolicy` protocol and can stand in for a hand-written rule.

### Persisting a `TabularPolicy`

`probabilities` and `default` are immutable copies of what you pass in.

- **Python persistence** (`model_dump(mode="python")` then `model_validate`,
  `copy.deepcopy`, and pickle) keeps every supported context key type,
  including distinct `1` and `"1"`, tuples and datetimes, and rebuilds a
  policy with the same action probabilities and the same `default` (or none).
  Pickle is for trusted data only.
- **JSON persistence** (`model_dump(mode="json")`, `model_dump_json`,
  `model_validate_json`) roundtrips only when every table key is a string,
  including default-only and no-`default` policies. Any other key type refuses
  at dump time with `logged_policy.policy.json_context_key`, because JSON
  would silently turn `1` into `"1"` or merge the two. Construction accepts
  such keys; use Python persistence or string-keyed tables. This includes the
  built-in integer-keyed `TARGET_POLICY_V1`.

## Interaction with the rest of the package

- No multiplicity: one `alpha` per call. Comparing several target policies
  against one reference is several calls, and no family correction is
  applied across them.
- No clusters beyond the unit: the trajectory is the independent replicate.
  Units nested in a shared cluster are not modelled.
- Fixed horizon only: every admitted unit must have all `T` rewards closed.
  Re-running on a growing trace is repeated fixed-horizon inference and has
  no anytime validity.
- No CUPED, winsorization, ratio rewards or breakouts.
