# Reading results

`Analysis.run()` returns one of two result families. The family fixes the row
type, but the scale of a value is a property of each row, so check every row
before comparing or combining rows.

| Evidence | Returns | Row type | Scale |
|---|---|---|---|
| Arm comparisons: `from_unit_summary`, `from_unit_panel`, `from_unit_day_artifact`, `from_definitions`, `from_moments` | `LiftEstimates` | `LiftEstimate` | Usually relative lift: `0.05` is 5% above control. Read each row's `value_scale`. |
| Switchback: `from_switchback_panel` | `ContrastResults` | `ContrastResult` | Additive difference in the metric's own units. |

A `LiftEstimate` can be absolute. An encouragement design's additive LATE row
and an observational metric reported with `run(value_scale={metric: "absolute"})`
carry `r.value_scale == "absolute"`, and then `r.lift.value` and its interval are
in the metric's own units. An encouragement design can also emit a relative LATE
sibling with the same estimand, so a LATE does not imply a scale. Check
`r.value_scale` (and `r.estimand`) on every row, and never compare a relative lift
with an additive value numerically.
`LiftEstimates.to_frame` and the readout rows have a `value_scale` column.
`ContrastResults.to_frame` does not, because every contrast is additive; only
the readout adapter below adds `value_scale="absolute"` to contrast rows.

## What to read

For a `LiftEstimate` row `r`:

| Question | Read | Notes |
|---|---|---|
| How big is the effect? | `r.lift.value` | `r.lift` is `None` when no finite point exists. |
| How uncertain is it? | `r.lift.lb`, `r.lift.ub`, `r.lift.level` | An endpoint is `None` on the side named by `r.lift.open_side`. Both are `None` when relative inference is unavailable. |
| Where is the interval when `r.lift` is `None`? | `r.binomial_set`, `r.relative_confidence_set`, `r.confidence_set`, `r.sequential_result.bounds`; the additive `r.abs_lb`, `r.abs_ub` | The set that owns the row's interval, if the row has one. A `None` upper bound on a binomial set means unbounded. Some rows have no relative set at all; see below. |
| Is it significant? | `r.stat_sig()` | Honors one-sided tests, nonzero nulls, and the row's own inference. |
| Was it selected within a family? | `r.role`, `r.discovery`, `r.family_q` | See [Multiplicity](multiplicity.md). |
| Why is something missing? | `r.relative_unavailable_reason`, `r.relative_confidence_set.point_unavailable_reason`, `r.confidence_set.relative` endpoint `status` and `reason`, `r.sequential_result.point_reason` | The exact persisted reason. |
| Which inference regime? | `r.inference` | `"fixed"` is a single-look interval. The sequential values are valid at every look. |

For a `ContrastResult` row `c`, read `c.estimate.value`, `c.estimate.lb`,
`c.estimate.ub`, `c.null_abs`, and `c.alternative`. All are additive.

## Fixed horizon: guard the missing point

Most rows carry `r.lift`. A few reachable rows do not. The loop below guards for
them and runs on data where one occurs: no control user converts, so the
relative lift has no finite point.

```python
import polars as pl

from increment import Analysis

n = 100
df = pl.DataFrame(
    {
        "user_id": [f"u{i:03d}" for i in range(2 * n)],
        "variant": ["control"] * n + ["treatment"] * n,
        "revenue": [10.0 + (i * 7) % 11 for i in range(n)]
        + [11.0 + (i * 7) % 11 for i in range(n)],
        "converted": [0] * n + [1] * 40 + [0] * 60,
    }
)
results = Analysis.from_unit_summary(
    df,
    unit="user_id",
    group="variant",
    control="control",
    metrics={"revenue": "mean", "converted": "conversion"},
).run()

for r in results:
    if r.lift is None:  # e.g. an exact conversion row with zero control events
        reason = r.relative_unavailable_reason
        print(f"{r.metric} / {r.group_id}: lift unavailable ({r.reference_kind}, {reason})")
        continue
    print(
        f"{r.metric} / {r.group_id}: "
        f"lift={r.lift.value:+.2%} [{r.lift.lb:+.2%}, {r.lift.ub:+.2%}] "
        f"significant={r.stat_sig()}"
    )
```

```text
revenue / treatment: lift=+6.69% [+0.69%, +13.05%] significant=True
converted / treatment: lift unavailable (binomial, None)
```

`r.lift` is `None` when no finite point estimate exists. The causes, and where
each row keeps its remaining evidence, are:

- An exact conversion row whose control arm has zero events. The ratio has a
  zero denominator, but the exact binomial set `r.binomial_set` still bounds the
  lift.
- A joint ratio set with no representable point. The set is
  `r.relative_confidence_set`, and `r.relative_confidence_set.point_unavailable_reason`
  names the reason.
- A fixed-horizon mean row whose arm mean is not positive:
  `r.relative_unavailable_reason == "nonpositive_arm_mean"`. The log-scale
  relative lift is undefined and there is **no** relative set. The additive
  difference survives in `r.abs_diff`, `r.abs_lb`, and `r.abs_ub`.
- A row whose joint relative covariance cannot be represented as a float, as
  with extremely large scores:
  `r.relative_unavailable_reason == "joint_covariance_unrepresentable"`. There is
  **no** relative set, and the additive interval stays in `r.abs_lb` and
  `r.abs_ub`.
- A winsorized metric with a `confidence_set`. A missing point leaves
  `r.confidence_set.relative` with a status and reason per endpoint (for
  example `unbounded` with `denominator_nonseparation`), and the additive
  interval stays in `r.abs_lb` and `r.abs_ub`.
- A sequential row whose point is withheld. The confidence set stays in
  `r.sequential_result.bounds`, and `r.sequential_result.point_reason` names the
  reason.

Do not assume a missing point comes with a set. Read
`r.relative_unavailable_reason` first: `nonpositive_arm_mean` and
`joint_covariance_unrepresentable` mean no relative set, only the additive
interval. Otherwise read whichever of `r.binomial_set`,
`r.relative_confidence_set`, `r.confidence_set`, and `r.sequential_result` is
present, as named above.

A missing point does not erase the evidence. The `converted` row is
significant. Its interval lives on `r.binomial_set`, and the upper end is
unbounded:

```python
from increment import CodedError

row = next(r for r in results if r.metric == "converted")
assert row.lift is None and row.reference_kind == "binomial"
assert row.stat_sig()
bset = row.binomial_set
assert bset.lower > 0 and bset.upper is None  # lift scale; None means unbounded

try:
    row.require_lift()
except CodedError as err:
    assert err.code == "estimation.results.lift.binomial_lift_availability"
else:
    raise AssertionError("a set-only row has no point to require")
```

`require_lift()` returns the point or raises a coded refusal. Use it when a
missing point should stop the program rather than be handled.

Two of the other causes are reproducible with small frames. A mean metric whose
control mean is negative has no relative lift, no relative set, and a reported
additive interval:

```python
k = 40
negative = pl.DataFrame(
    {
        "user_id": [f"u{i:03d}" for i in range(2 * k)],
        "variant": ["control"] * k + ["treatment"] * k,
        "delta": [-3.0 + (i * 7) % 5 for i in range(k)] + [-1.0 + (i * 7) % 5 for i in range(k)],
    }
)
flat_mean = Analysis.from_unit_summary(
    negative,
    unit="user_id",
    group="variant",
    control="control",
    metrics={"delta": "mean"},
).run()[0]

assert flat_mean.lift is None and flat_mean.relative_confidence_set is None
assert flat_mean.relative_unavailable_reason == "nonpositive_arm_mean"
assert flat_mean.abs_lb < flat_mean.abs_diff < flat_mean.abs_ub
```

A winsorized metric whose control arm sums to zero keeps a confidence set with a
finite lower and an unbounded upper relative endpoint, each with its own status:

```python
from increment import MetricSpec

zero_control = pl.DataFrame(
    {
        "user_id": [f"u{i:03d}" for i in range(2 * k)],
        "variant": ["control"] * k + ["treatment"] * k,
        "revenue": [0.0] * k + [float(1 + (i * 7) % 5) for i in range(k)],
    }
)
winsorized = Analysis.from_unit_summary(
    zero_control,
    unit="user_id",
    group="variant",
    control="control",
    metrics=[
        MetricSpec(
            name="revenue",
            type="mean",
            winsorization={
                "upper_percentile": 0.9,
                "support": {"lower": 0.0, "provenance": "Revenue is nonnegative by definition"},
                "inference": {"method": "joint-rank-projection-v1"},
            },
        )
    ],
).run()[0]

relative = winsorized.confidence_set.relative
assert winsorized.lift is None and winsorized.reference_kind == "confidence_set"
assert relative.lower.status == "finite" and relative.upper.status == "unbounded"
assert winsorized.abs_lb < winsorized.abs_diff < winsorized.abs_ub
```

### A point without an interval

A point can exist while its relative interval does not. When the estimated
variance of the relative contrast is exactly zero, the row keeps the point,
withholds the bounds and the decision, and names the reason. This happens, for
example, with clustered data whose values are identical within each arm:

```python
m = 100
degenerate = pl.DataFrame(
    {
        "user_id": [f"u{i:03d}" for i in range(2 * m)],
        "variant": ["control"] * m + ["treatment"] * m,
        "store": [f"s{i // 2:03d}" for i in range(2 * m)],
        "revenue": [10.0] * m + [11.0] * m,
    }
)
flat = Analysis.from_unit_summary(
    degenerate,
    unit="user_id",
    group="variant",
    control="control",
    metrics={"revenue": "mean"},
    cluster="store",
).run()[0]

assert flat.lift is not None and flat.lift.lb is None and flat.lift.ub is None
assert flat.relative_unavailable_reason == "zero_relative_variance"
assert not flat.stat_sig()  # no interval, so no decision
assert flat.abs_diff == 1.0  # the additive difference is still reported
```

Check `lb` and `ub` against `None`, or read `r.relative_unavailable_reason`
first, before formatting them.

## One-sided tests

A directional conversion test returns an open interval. `open_side` names the
unbounded end, and that bound is `None`:

```python
from increment import AnalysisPlan, MetricSpec

purchases = pl.DataFrame(
    {
        "user_id": [f"u{i:03d}" for i in range(2 * n)],
        "variant": ["control"] * n + ["treatment"] * n,
        "purchased": [1] * 30 + [0] * 70 + [1] * 45 + [0] * 55,
    }
)
directional = Analysis.from_unit_summary(
    purchases,
    unit="user_id",
    group="variant",
    control="control",
    metrics=[MetricSpec(name="purchased", type="conversion")],
    plan=AnalysisPlan(primary="purchased", alternative="greater"),
).run()[0]

lift = directional.lift
assert lift.open_side == "upper" and lift.ub is None
assert lift.lb <= lift.value  # the finite bound and the point share one scale
assert lift.excludes(0.0) and directional.stat_sig()
```

`excludes()` reads only the finite side of an open interval. A one-sided test
spends all of `alpha` on its single tail, so its `level` is the central
equivalent: `level=0.9` for the default `alpha=0.05`. Mean metrics test the same
direction but return a closed interval at that level, so read `open_side`
instead of assuming it is set.

## Sequential inference: two coordinate systems

A sequential row stores its confidence set in exact **ratio** coordinates, where
`1.0` means no effect, and displays it on the **lift** scale, where `0.0` means
no effect. Know which one you are reading:

| Where | Scale |
|---|---|
| `r.sequential_result.bounds` and the `ratio_set` in the row's `repr` | Ratio, exact `Fraction` endpoints. |
| `r.lift.value`, `r.lift.lb`, `r.lift.ub` | Lift: the ratio bounds minus one, rounded outward to floats. |
| Frame columns `lb`, `ub`, `sequential_lower`, `sequential_upper`; readout `lower`, `higher` | Lift, same as `r.lift`. |

The lift-scale endpoints are never narrower than the ratio bounds, because
rounding moves them outward. Asymptotic sets can be disconnected, empty, or
unbounded. For a disconnected or empty set `r.lift` carries only the point,
because collapsing a set to its hull would misstate it. Read
`sequential_status` and `sequential_components` for the geometry; for a
disconnected set, `sequential_lower` and `sequential_upper` are the outer
endpoints, not an interval. A missing endpoint is `None` and is never replaced by
a number on the wrong scale.

```python
import polars as pl

from increment import Analysis, AnalysisPlan, InferenceSpec, MetricSpec, Randomized

k = 200
monitored = pl.DataFrame(
    {
        "unit": [f"u{i:04d}" for i in range(2 * k)],
        "arm": ["control"] * k + ["treatment"] * k,
        "enrolled_on": [1] * (2 * k),
        "revenue": [10.0 + (i * 37) % 17 for i in range(k)]
        + [11.5 + (i * 37) % 17 for i in range(k)],
    }
)
sequential = Analysis.from_unit_summary(
    monitored,
    unit="unit",
    group="arm",
    design=Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5}),
    metrics=[MetricSpec(name="revenue", type="mean")],
    experiment_id="revenue-monitor",
    plan=AnalysisPlan(
        primary="revenue",
        inference=InferenceSpec(kind="asymptotic_mean", expected_decision_sample_size=400),
    ),
    exposure_date="enrolled_on",
).run()
row = sequential[0]
assert row.inference != "fixed"

lift = row.lift  # lift scale
bounds = row.sequential_result.bounds  # ratio scale
assert lift.lb <= lift.value <= lift.ub
assert float(bounds.lower) <= 1 + lift.value <= float(bounds.upper)
assert lift.lb <= float(bounds.lower) - 1 and float(bounds.upper) - 1 <= lift.ub

frame = sequential.to_frame(backend="polars").row(0, named=True)
assert frame["sequential_status"] == "bounded"
# The frame's sequential endpoints are lift-scale: they surround the lift
# point, which a ratio-scale interval around 1.0 would not.
assert frame["sequential_lower"] <= frame["lift"] <= frame["sequential_upper"]
```

The exact `always_valid` route for conversion metrics reads the same way. Its
bounds are `Fraction` ratios and its certificate adds `log_e`, the evidence
against the registered null:

```python
from fractions import Fraction

registered = pl.DataFrame(
    [
        {"unit": f"{i:08d}-{arm}", "arm": arm, "outcome": value, "exposure": i}
        for i, pair in enumerate(zip([0, 0, 0, 1] * 50, [0, 1, 1, 1] * 50, strict=True))
        for arm, value in zip(("control", "treatment"), pair, strict=True)
    ]
)
exact = Analysis.from_unit_summary(
    registered,
    unit="unit",
    group="arm",
    exposure_date="exposure",
    metrics=[MetricSpec(name="outcome", type="conversion")],
    design=Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5}),
    experiment_id="experiment",
    plan=AnalysisPlan(
        primary="outcome",
        inference=InferenceSpec(kind="always_valid", baseline_rate=Fraction(3, 10)),
    ),
).run()[0]

result = exact.require_sequential_result()
assert exact.inference == "always_valid" and exact.stat_sig()
assert result.bounds.lower <= 1 + Fraction(exact.lift.value) <= result.bounds.upper
assert exact.lift.lb <= float(result.bounds.lower) - 1
assert result.log_e > 0
```

See [Registered sequential inference](sequential-inference.md) for how these
sets are built, including the e-BH secondary family.

## Switchback contrasts are additive

`from_switchback_panel` returns `ContrastResults`. Each `ContrastResult` carries
an additive `estimate` in the metric's own units, with `null_abs` as its null.
This panel has four independent blocks over a three-unit roster:

```python
import math

from increment import (
    Randomized,
    SharedScheduleOrder,
    SwitchbackAssignment,
    SwitchbackWindow,
)

orders = [("control", "treatment")] * 2 + [("treatment", "control")] * 2
shifts = [-1, 1, -2, 2]
panel = pl.DataFrame(
    [
        {
            "unit": f"u{unit}",
            "cycle": block,
            "period": period,
            "step": 0,
            "group": group,
            "value": 10.0 + (effect + shifts[block]) * (group == "treatment"),
        }
        for unit, effect in enumerate([4, 6, 8])
        for block, order in enumerate(orders)
        for period, group in enumerate(order)
    ]
)
contrasts = Analysis.from_switchback_panel(
    panel,
    unit="unit",
    cycle="cycle",
    period="period",
    step="step",
    group="group",
    metrics={"value": "mean"},
    identification=Randomized(
        control_group="control",
        allocation={"control": 0.5, "treatment": 0.5},
    ),
    assignment=SwitchbackAssignment(
        sequence=SharedScheduleOrder(probability_ct=0.5),
        window=SwitchbackWindow(washout_steps=0, observation_steps=1),
    ),
).run()
c = contrasts[0]

estimate = c.estimate
assert estimate.lb <= estimate.value <= estimate.ub and estimate.open_side is None
assert c.alternative == "two-sided" and c.null_abs == 0.0
assert math.isclose(estimate.value, 6.0)  # the mean unit effect, in outcome units
```

The estimate is not a relative lift. `ContrastResult` has no `stat_sig()`; the
readout adapter below supplies the verdict.

For the other designs, see [Switchback contrasts](switchback.md) for the
estimand and its unit-cycle `standard_error_unavailable_reason`,
[Multiplicity](multiplicity.md) for `role`, `discovery`, and `family_q`, and
[Observational comparisons](observational.md#near-zero-baselines-reporting-the-additive-effect) for
`relative_unavailable_reason`.

## Results as frames and flat rows

Every result list converts to a frame in your preferred library. The call is
`to_frame(backend="polars")` for `LiftEstimates` and, by keyword only, for
`ContrastResults`. A set-only exact binomial row keeps its interval in the
`set_lower`, `set_upper`, and `set_level` columns. Those columns come from
`binomial_set` only. A sequential row's set is in `sequential_lower`,
`sequential_upper`, `sequential_status`, and `sequential_components`. A
`relative_confidence_set` or winsor `confidence_set` appears in the frame only
as a `repr` text column of the same name, so read the set from the result
object, and use `abs_lb` and `abs_ub` for the additive interval:

```python
table = results.to_frame(backend="polars")
converted = table.filter(pl.col("metric") == "converted").row(0, named=True)
assert converted["lift"] is None
assert converted["set_lower"] is not None and converted["set_upper"] is None

assert "estimate" in contrasts.to_frame(backend="polars").columns
```

`increment.tables.estimates_to_readout` adapts either family to one flat
`dict` per row, so one loop or one dataframe reads a mixed list. It accepts
`LiftEstimate`, `BreakoutEstimate`, `DailyLiftEstimate`, and `ContrastResult`
rows. Each dict carries `lift`, `lower`, `higher`, `level`, `open_side`,
`stat_sig`, `null_lift`, `null_abs`, `value_scale`, `estimand`, `inference`,
`role`, and `discovery`, plus the sequential columns when the row is
sequential. Contrast rows set `value_scale="absolute"` and
`group_id=treatment_group`.

```python
from increment.tables import estimates_to_readout

rows = estimates_to_readout([*results, *contrasts])
by_metric = {row["metric"]: row for row in rows}

unbounded = by_metric["converted"]
assert unbounded["lift"] is None and unbounded["lower"] > 0
assert unbounded["higher"] is None and unbounded["stat_sig"]

additive = by_metric["value"]
assert additive["value_scale"] == "absolute"
assert additive["group_id"] == additive["treatment_group"] == "treatment"
assert additive["lower"] <= additive["lift"] <= additive["higher"]
```

`lift` holds the contrast's additive estimate on a contrast row, which is why
`value_scale` must be read with it. `open_side` comes from the row's `lift`, so
a set-only row's open upper end shows only as `higher is None`, as above.

The adapter needs only the core install. `estimates_to_readout` imports
without `pandas` or `coeftable`; in a wheel-only environment with `pyarrow` and
neither of them, it converted a zero-control conversion row to
`lift=None`, a finite `lower`, `higher=None`, and `stat_sig=True`. Rendering the
rows with `readout_table` requires the `tables` extra. `to_frame` defaults to
`backend="pandas"`, so without pandas pass `backend="pyarrow"`, which the core
install provides.
