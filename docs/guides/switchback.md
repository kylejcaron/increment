# Fixed-horizon switchback contrasts

Use a switchback when treatment changes over time within a store, city or
market. The question is additive: *how many more orders per store and
retained window?* An 80% treatment schedule does not make the answer a
percent lift. The example below reads the assignment log, compares three
metrics and uses the orders pilot to size a future study.

## Supported contract

Declare one control arm, one treatment arm, equal-duration periods and a
fixed analysis horizon. Use an eager pandas, Polars or PyArrow frame.
The randomization determines both the arm allocation and what counts as
an independent replicate.

### Choose the order law

Choose the declaration that matches how the schedule was actually drawn.
CT means control then treatment; TC means treatment then control.

- **One period per arm:** `IndependentBernoulliOrder(probability_ct=p)`
  draws an order independently for each unit-cycle;
  `SharedScheduleOrder(probability_ct=p)` draws it once for the entire
  roster in each two-period block. The allocation is 50/50 in either case.
  Even `p=0.8` means an 80% chance of CT, **not** 80% exposure to treatment.
- **A fixed number of treated periods:** `FixedCountOrder` draws exactly
  `m` of `K` positions uniformly without replacement (`K >= 3`, `0 < m < K`).
  Set treatment allocation to `m/K`. Use `independence_unit="unit_cycle"`
  for a new draw per unit-cycle, or `"shared_block"` for a common draw
  across the roster. Don't use this for a deterministic rotation, a
  calendar restriction or a scheduler that favors particular positions.

With independent orders, inference averages each unit's cycles and uses
the variation between units. With a shared order, it averages the roster
in each block and uses the variation between blocks. Adding stores to the
same blocks does not add independent draws. Neither t interval guarantees
finite-sample coverage: in one paired four-unit study design, the nominal
95% interval covered zero on only 302 of 512 null attempts. See the
[paired measurements](../reference/switchback-inference.md#coverage-with-few-units-or-blocks);
they are not a calibration of the fixed-count design below.

### Washout and carryover

Each period has `washout_steps + observation_steps` scheduled steps. Discard the
washout and then `carryover_order` extra steps; the retained count is
`observation_steps - carryover_order`. Orders 0, 1 and 2 are supported
when enough steps remain. A mean sums the retained values in each period.
A conversion is one indicator: did **any** event occur in those steps?

!!! warning "Discard enough time for carryover to end"
    The assumption `no_residual_carryover_after_discarded_steps` must hold
    for the retained steps. Increment checks the logged schedule, not
    whether randomization happened or treatment carryover has vanished.
    Residual carryover can bias the estimate and invalidate the interval.

## Analyze an 80% treatment schedule

Two stores share a schedule. In each of 48 blocks the scheduler chooses
one of five positions uniformly for control; the other four get treatment.
Count orders per store during the retained step. Also ask whether adjusting
orders for forecast demand changes the estimate, and how the average
refund amount differs between arms.

### 1. Declare the design and the decisions

```python
from datetime import date
from random import Random

import polars as pl

from increment import (
    AnalysisPlan,
    Method,
    MetricSpec,
    Randomized,
    SwitchbackObservationSchedule,
    from_switchback_frame,
    run,
)
from increment.estimators import FixedCountOrder, SwitchbackAssignment, SwitchbackWindow
from increment.power import ContrastDecisionProcedure, switchback_profile_plan
from increment.workflows import planning_baseline

stores = ("north", "south")
n_blocks = 48
assignment = SwitchbackAssignment(
    periods_per_cycle=5,
    sequence=FixedCountOrder(treated_periods=4, independence_unit="shared_block"),
    window=SwitchbackWindow(washout_steps=1, observation_steps=1),
)
design = Randomized(control_group="control", allocation={"control": 0.2, "treatment": 0.8})
metrics = [
    MetricSpec(name="orders", preferred_direction="increase"),
    MetricSpec(
        name="adjusted_orders",
        value_column="orders",
        decision_method=Method.cuped(covariates="forecast"),
    ),
    MetricSpec(name="refund_per_return", type="ratio", numerator="refund", denominator="returns"),
]
alpha = 0.05
plan = AnalysisPlan(primary=[m.name for m in metrics], alternative="two-sided", alpha=alpha)
```

### 2. Keep the forecasts, assignments and responses separate

CUPED needs one numeric forecast of the retained-window total per
store, block and position. Here demand varies by store and position;
the forecast is fixed before anyone knows the assigned arm. The latest
`forecast_at` must be strictly earlier than the earliest `assigned_at`.
Neither washout outcomes nor a forecast copied from the outcome table
establish that timing.
If pandas stores Python dates as `object`, convert both clock columns with
`pd.to_datetime` first. An unreadable date is not evidence the forecast was late.
Increment compares the two clock columns you supply; it cannot verify
either timestamp against the scheduler.

```python
forecasts = pl.DataFrame(
    [
        {
            "store": store,
            "block": block,
            "position": pos,
            "forecast": 40 + 3 * i + 2 * pos,
            "forecast_at": date(2025, 1, 1),
            "assigned_at": date(2025, 1, 2),
        }
        for i, store in enumerate(stores)
        for block in range(n_blocks)
        for pos in range(5)
    ]
)

rng = Random(29)
controls = [rng.randrange(5) for _ in range(n_blocks)]
assignments = pl.DataFrame(
    [
        {
            "store": store,
            "block": block,
            "position": pos,
            "arm": "control" if pos == control else "treatment",
        }
        for block, control in enumerate(controls)
        for store in stores
        for pos in range(5)
    ]
)
schedule = SwitchbackObservationSchedule(assignments=assignments, units=stores, cycles=n_blocks)
```

For this example, generate the response table separately. Only step 1
is retained. Every retained step was measured; in treatment, there were
no returns. Those are observed zeros, not missing data.

```python
keys = ["store", "block", "position"]
records = []
for row in assignments.join(forecasts, on=keys).iter_rows(named=True):
    treated = row["arm"] == "treatment"
    noise = (row["block"] + 2 * row["position"]) % 7 - 3
    returns = int(not treated)
    records.append(
        {
            **{key: row[key] for key in keys},
            "step": 1,
            "orders": row["forecast"] + noise + 2 * treated,
            "returns": returns,
            "refund": 12 * returns,
        }
    )
observations = pl.DataFrame(records)
```

### 3. Read the three decisions and keep the orders pilot

```python
import json

experiment, reader = from_switchback_frame(
    observations,
    unit="store",
    cycle="block",
    period="position",
    step="step",
    group="arm",
    metrics=metrics,
    identification=design,
    assignment=assignment,
    plan=plan,
    schedule=schedule,
    pre_assignment_forecasts=forecasts,
)
with reader:
    results = run(experiment, reader).readout().to_frame(backend="polars")
    pilot = planning_baseline(experiment, reader, "orders")

rows = {row["metric"]: row for row in results.to_dicts()}
assert rows["orders"]["sampling_available"] and rows["orders"]["dof"] == n_blocks - 1
assert (
    rows["adjusted_orders"]["sampling_available"] and rows["adjusted_orders"]["dof"] == n_blocks - 2
)
refund = rows["refund_per_return"]
assert refund["estimate"] is None and not refund["sampling_available"]
reason = json.loads(refund["failure_context"])
assert (reason["reason"], reason["arm"]) == ("zero_denominator", "treatment")
print(results.select("metric", "estimate", "standard_error", "dof"))
```

Read the orders row in **orders per store per retained window**. CUPED
compares outcome contrasts at zero forecast contrast, refitting when each
block is left out; it does not treat the forecast adjustment as known.
Average refund per return is undefined in treatment: no point, interval
or decision. The orders rows still use the full roster. A weak but
nonzero denominator can instead give a descriptive point without a test.

If one store has a blank demand forecast for a scheduled period,
`adjusted_orders` has no estimate and reports `missing_forecast`. `orders`
still uses every scheduled period and the same share of alpha. If instead
all blocks show exactly +2 orders, the ordinary row keeps that +2 estimate
but reports `zero_standard_error`: the significance test **did not run**.
That is different from a completed test whose interval includes zero.

For a fixed-count schedule, a reported t-test is approximate: even with
random assignment it can call a change more often than the chosen alpha.
The [48-block example](../reference/switchback-inference.md#unadjusted-means-and-conversions)
shows when that happens. More transactions per block do not fix it.

The reader keeps the pilot values measured at construction. Editing the
original dataframe later doesn't change planning or the readout; build
a new reader to use corrected measurements.

### Check the schedule before launch

Every position must appear in both arms across the independent draws.
For a shared schedule those draws are blocks, not stores. Before drawing,
check the chance of a refusal under the declared order:

```python
from increment.power import switchback_schedule_refusal_probability

print("Blocks  Refusal probability")
for blocks in (12, 24, 36, 48):
    chance = switchback_schedule_refusal_probability(assignment, blocks)
    print(f"{blocks:6d}  {chance:.9f}")
```

<!-- output -->
```text
Blocks  Refusal probability
    12  0.321997312
    24  0.023564451
    36  0.001622490
    48  0.000111504
```

After drawing, validate the actual assignment log **before collecting
responses**. The loader accepts an empty response table with declared
column types; no outcome values are needed. Here the drawn log passes,
but a log that always gives position zero to control does not:

```python
from increment.errors import InvalidRequestError

empty_responses = pl.DataFrame(
    schema={
        "store": pl.String,
        "block": pl.Int64,
        "position": pl.Int64,
        "step": pl.Int64,
        "orders": pl.Int64,
    }
)


def check_schedule(candidate):
    return from_switchback_frame(
        empty_responses,
        unit="store",
        cycle="block",
        period="position",
        step="step",
        group="arm",
        metrics=[metrics[0]],
        identification=design,
        assignment=assignment,
        schedule=candidate,
    )


_, checked_reader = check_schedule(schedule)
with checked_reader:
    print("Schedule accepted; no outcomes supplied.")

unsupported_assignments = assignments.with_columns(
    pl.when(pl.col("position") == 0)
    .then(pl.lit("control"))
    .otherwise(pl.lit("treatment"))
    .alias("arm")
)
unsupported_schedule = SwitchbackObservationSchedule(
    assignments=unsupported_assignments, units=stores, cycles=n_blocks
)
try:
    check_schedule(unsupported_schedule)
except InvalidRequestError as exc:
    print(exc.code)
```

<!-- output -->
```text
Schedule accepted; no outcomes supplied.
source.frame.switchback.schedule
```

More blocks are the primary remedy. Analysed studies are already only
those whose schedule passes. A rule fixed before drawing that redraws
only when this check fails yields the same set of analysable schedules;
redrawing for any other reason does not. Passing this check says nothing
about carryover, missing responses or the eventual test's precision.

### 4. Size a future study with the same randomization

For orders, assume treatment adds the same amount to each retained-window
total. The planner subtracts the fitted shift from treated pilot windows,
resamples entire blocks (both stores together), then redraws the positions
under the 4-of-5 law. Match the prospective decision: the zero null is
two-sided, and the plan splits its alpha across the three primary metrics.
Sizing a fixed-count mean needs a completed pilot or hand-built profiles;
a first study without either relies on the schedule check alone.

```python
procedure = ContrastDecisionProcedure(
    metric="orders",
    role="primary",
    alternative="two-sided",
    preferred_direction="increase",
    null_abs=0.0,
    alpha=alpha / len(metrics),
    reference=None,
)
grid = switchback_profile_plan(
    pilot,
    procedure,
    sizes=(8, 32, 64),
    effects=(0.0, 0.25, 6.0),
    target_power=0.8,
    repetitions=512,
    seed=29,
)
for point in grid.points:
    print(
        point.n, point.delta, point.rejected, point.attempts, point.power_lower, point.power_upper
    )
size = grid.grid_size(6.0)
mde = grid.grid_mde(64)
print("First block count at +6 orders:", size.status, size.value)
print("First effect at 64 blocks:", mde.status, mde.value)
```

Eight blocks can miss a position entirely in control. Such attempts,
zero standard errors and other failed decisions stay in the denominator:
power is **rejections / attempts**, not rejections / successful fits.
`attained` means the lower Monte Carlo bound reaches 80% and every earlier
candidate's upper bound is below 80%. `undecided` is not a recommendation
to take a later point. Neither answer says what happens between grid points.
The bounds describe simulation error under this pilot model, not uncertainty
about the pilot or the real test's false-positive rate.

For the same 4-of-5 design, suppose treatment does nothing and one slot
is always busier than the others. Even with 48 blocks, the nominal 5% test
can call a difference on 7.14% of assignments. See the [calendar and exact
count](../reference/switchback-inference.md#unadjusted-means-and-conversions).
More transactions in each block don't create more independent draws.

## When it refuses

- **No separate log?** Supply every unit × cycle × period × step with its
  arm and outcome, including discarded steps. In this dense format a gap
  cannot distinguish missing assignment from missing measurement.
- **A separate log?** A missing assignment, conflicting arm or duplicate
  key invalidates the study. With a valid log, an unobserved retained
  response leaves only its metric unavailable (`unobserved_response`).
  An absent event row can count as zero only with `missing="zero"` and that
  metric in `complete_event_metrics`. A null in a recorded row, an absent
  denominator or a missing forecast is never filled with zero.
- **No support at a position?** Fixed-count analysis requires each position
  to appear in both arms across the independent draws. Shared paired
  schedules need both CT and TC. More roster members cannot repair either:
  [check the schedule before launch](#check-the-schedule-before-launch).

`increment.workflows.assignment_diagnostic` reports schedule integrity, not
a sample-ratio test or proof the scheduler drew uniformly. For exact reasons
and result fields, see the [inference reference](../reference/switchback-inference.md).

## Optional finite-sample references and research laws

For a paired unadjusted mean with independent unit-cycle orders, a
prospective `UnitCycleVarianceEnvelope` supports a different finite-sample
interval under its declared model. A pilot variance is **not** such a bound.
See the [envelope assumptions](../reference/switchback-inference.md#optional-finite-sample-references-and-research-laws)
and the [planning guide](power-analysis.md#switchback-contrasts) for the
distinction between paired mean, trajectory and conversion pilots.

## Response meaning and independent certification

You may supply totals or pre-normalized means, including from unequal
underlying micro-observation counts. The analysis does not receive those
counts or reweight by them. Totals and means define different response
models and effect units, not interchangeable encodings; retained schedule
row counts must still be equal and complete.

The [response and certification reference](../reference/switchback-inference.md#response-meaning-and-independent-certification)
records the simulation model, known counterexample and independent checks.

### Representative release checks

The unit-cycle campaign has **not been executed**. Its coverage,
availability and power thresholds are acceptance criteria, not measured
results. See the [representative checks](../reference/switchback-inference.md#representative-release-checks).

### Bounded full research campaign

The [full campaign reference](../reference/switchback-inference.md#bounded-full-research-campaign)
defines the manifest, execution budgets and records. A smoke run or an
inconclusive, failed or timed-out campaign is not a calibration certificate.
