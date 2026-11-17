# Fixed-horizon switchback contrasts

Use a switchback when a store, city or market must alternate between
treatment and control over time instead of splitting users into groups.
`from_switchback_frame` estimates an **additive difference over retained
windows**, not per-user relative lift.

## Supported contract

Prepare an eager pandas, Polars or PyArrow frame with one row for each
unit, cycle, period and step. A cycle has two periods: one control and one
treatment. Every period must contain the complete declared step range.
Choose mean or conversion metrics and a `Randomized` design with exactly
one control and one treatment arm, allocated `0.5`/`0.5`.

### Choose the order law

Choose the declaration that matches how the schedule was actually drawn.
CT means control then treatment; TC means treatment then control.

- **Each unit-cycle draws its own order:** declare
  `IndependentBernoulliOrder(probability_ct=...)`. The independent replicate
  is a unit; ordinary inference needs at least two.
- **A whole roster follows one order per block:** declare
  `SharedScheduleOrder(probability_ct=...)`. The independent replicate is a
  two-period block; at least two are needed, with both orders realized.

The default independent-order analysis uses a qualified unit-t
approximation with `n_units - 1` degrees of freedom. A shared schedule uses
an approximate block-t interval with `n_blocks - 1`. Neither is an exact
finite-sample coverage guarantee or a boundary for repeated decisions.
Small-sample unit-t calibration failures remain unresolved; see
[certification limits](../reference/switchback-inference.md#response-meaning-and-independent-certification).

`probability_ct` describes independent Bernoulli order draws, not fixed
CT/TC counts. There is no finite-population correction. A shared schedule
requires a fixed, complete roster; duplicating that roster uniformly
changes neither the effect scale nor the number of independent blocks.

### Washout and carryover

Each period has `washout_steps + observation_steps` rows. Discard the
washout and then `carryover_order` extra steps; the retained count is
`observation_steps - carryover_order`. Orders 0, 1 and 2 are supported
under both laws when enough steps remain. Mean metrics sum retained
values; conversion metrics use `any` across the retained steps.

!!! warning "Discard enough time for carryover to end"
    The assumption `no_residual_carryover_after_discarded_steps` must hold
    for the retained steps. Increment checks the logged schedule, not
    whether randomization happened or treatment carryover has vanished.
    Residual carryover can bias the estimate and invalidate the interval.

## A nondegenerate shared example

This complete, hand-constructed example represents four independent block
draws shared by three stores. Each period has one washout step and three
observation steps; order one discards the first observation step too.
The large values in discarded steps do not enter the result.

```python
import math

import pandas as pd

import increment
from increment.estimators import (
    SharedScheduleOrder,
    SwitchbackAssignment,
    SwitchbackWindow,
)
from increment.workflows import assignment_diagnostic

orders = [("control", "treatment")] * 2 + [("treatment", "control")] * 2
shifts = [-1, 1, -2, 2]
panel = pd.DataFrame(
    [
        {
            "unit": f"u{unit}",
            "cycle": block,
            "period": period,
            "step": step,
            "group": group,
            "value": 999 if step < 2 else 10 + (effect + shifts[block]) * (group == "treatment"),
        }
        for unit, effect in enumerate([4, 6, 8])
        for block, order in enumerate(orders)
        for period, group in enumerate(order)
        for step in range(4)
    ]
)
experiment, reader = increment.from_switchback_frame(
    panel,
    unit="unit",
    cycle="cycle",
    period="period",
    step="step",
    group="group",
    metrics={"value": "mean"},
    identification=increment.Randomized(
        control_group="control",
        allocation={"control": 0.5, "treatment": 0.5},
    ),
    assignment=SwitchbackAssignment(
        sequence=SharedScheduleOrder(probability_ct=0.5),
        window=SwitchbackWindow(washout_steps=1, observation_steps=3, carryover_order=1),
    ),
)
result = increment.run(experiment, reader).readout()[0]
diagnostics = assignment_diagnostic(experiment, reader)
assert math.isclose(result.estimate.value, 12)
assert math.isclose(result.standard_error, math.sqrt(40 / 12))
assert result.n_blocks == 4 and result.n_units == 3 and result.dof == 3
assert result.observation_steps == 3 and result.retained_steps == 2
print(f"retained-window difference: {result.estimate.value:g}")
print(f"independent blocks: {result.n_blocks}; roster units: {result.n_units}")
print(f"degrees of freedom: {result.dof:g}")
```

<!-- output -->
```text
retained-window difference: 12
independent blocks: 4; roster units: 3
degrees of freedom: 3
```

The block contributions are 10, 14, 8 and 16, averaging **12 additional
outcome units per store's two-step retained window**. This is not 12% lift
or 12 per user. The standard error is `sqrt(40 / 12)`, based on four
blocks—not the twelve store-block combinations.

## Results and diagnostics

`readout()` returns one `ContrastResult` per selected metric. Read
`estimate`, `lb`, `ub` and `standard_error` on the additive scale. Conversion
contrasts measure a probability difference; they are not event-count gains.
The result also records the method, order law, retained window and counts.
See [result fields and diagnostics](../reference/switchback-inference.md#result-fields-and-diagnostics)
for their exact meanings and portable schema.

`assignment_diagnostic(experiment, reader)` checks the schedule and reports
realized CT/TC counts. It is not an SRM test, has no p-value, and gives no
anytime-valid guarantee. `health_check` marks assignment balance unavailable
and `srm` refuses for this design; their exact codes are in the reference.

## Randomizing at a market or geo grain

For independent orders, set `unit` to the entity that drew its own order.
Pre-aggregate to `geo × cycle × period × step` before passing geo data.
The independent replicates are geos, not their underlying users.

When the whole market flips together, use `SharedScheduleOrder`. Even one
geo can be a roster, but you still need independent two-period blocks and
both realized orders. More roster members do not add degrees of freedom.

Independent per-geo orders make the contrast design-unbiased, but do not
make outcomes independent. Shared time shocks, correlated schedules and
interference can violate the inference assumptions; see
[switchback independence](../limitations.md#switchback-inference-requires-independent-units-or-blocks).

## When it refuses

Check for missing or duplicate cells, incomplete steps, inconsistent orders
across a shared roster, and an implausible CT/TC split. Shared schedules
also refuse if only CT or only TC occurs, even when that split is plausible.
If block contributions have zero spread, decision evidence is unavailable:
bounds and confidence level are numeric nulls, not an invented interval.

For skewed shared schedules, plan enough blocks to realize both orders.
The power solvers already account for that admission probability. See
[schedule refusals](../reference/switchback-inference.md#when-it-refuses).

## Deliberate refusals

Find exact codes, exception types, unsupported metric/plan options and
`run()` override rules in the
[refusal inventory](../reference/switchback-inference.md#deliberate-refusals).
Use `from_frame` or `from_definitions` for a parallel A/B analysis, not a
switchback declaration.

## Optional finite-sample references and research laws

Ordinary inference and pilot planning need no certificate. If you can
justify a prospective residual-variance upper bound, a
`UnitCycleVarianceEnvelope` provides a different, finite-sample construction
under its declared model. A pilot variance or observed standard error is
**not** such a bound. See the
[envelope assumptions and interval](../reference/switchback-inference.md#optional-finite-sample-references-and-research-laws)
and [complete-law power planning](../reference/switchback-inference.md#complete-law-power-planning).

For ordinary [switchback power planning](power-analysis.md#switchback-contrasts),
pilot moments give a moment-t approximation. Two-sided MDE planning keeps
the signed additive effect: it searches below the null for
`preferred_direction="decrease"` and above otherwise, within any declared
`effect_bounds`.

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
