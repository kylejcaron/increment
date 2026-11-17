# Reading results

What changed, how uncertain is it, and can I act? Read the effect together
with its interval, the checks on the data, and the experiment's decision.
An available estimate is not necessarily a convincing result.

## Start with the report

This small example compares a metric between control and treatment.
It declares independent 50/50 assignment and an increase as favorable.

```python
import increment
import polars as pl

frame = pl.DataFrame(
    {
        "unit": ["c1", "c2", "c3", "c4", "t1", "t2", "t3", "t4"],
        "arm": ["control"] * 4 + ["treatment"] * 4,
        "value": [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0],
    }
)
experiment, reader = increment.from_frame(
    frame,
    shape="summary",
    unit="unit",
    group="arm",
    design=increment.Randomized(
        control_group="control",
        allocation={"control": 0.5, "treatment": 0.5},
        allocation_scheme="independent",
    ),
    metrics=[increment.MetricSpec(name="value", type="mean", preferred_direction="increase")],
    plan=increment.AnalysisPlan(primary="value"),
)
result = increment.run(experiment, reader)
print(increment.render(result, "brief"))
print(result.decisions.program)
```

The report's first line, wrapped for readability:

```text
value / arm=treatment,
comparison=treatment vs control:
effect=40% [-36.8823%, 210.531%]
(relative, 95.0% confidence);
status=available; 95.0% confidence;
1-member family
```

Treatment's average is 40% higher, but the interval includes a decrease of
about 37% and an increase of about 211%. These eight observations do not
establish a difference from control. `available` means the calculation
provided evidence; it does not mean the treatment worked.

The decision confirms that distinction, also wrapped here:

```text
ProgramVerdict(
  status='block',
  reason_codes=(
    'decision.primary.not_rejected',
  )
)
```

The primary metric did not meet its test. This blocks the declared decision;
it is not evidence that the treatment has no effect.

## What to read

| Need | Use |
|---|---|
| Text report | `render` |
| Notebook or HTML table | `readout_table` |
| Dataframe | `result.to_frame` |

Tables require the `tables` extra. In a notebook:

```python
from increment.tables import readout_table

table = readout_table(result)
table
```

Keep the report's warnings and table notes. They identify missing results,
unchecked assignment, partial selections and other limits on the conclusion.
Filtering what you display does not change the comparisons originally tested.

## Read the row's scale

**Relative lift** compares treatment with control as a proportion:
`0.05` means 5% higher, and zero means no difference. **Additive effects**
use the metric's original units. Never compare those numbers without their
scale.

Read `value_scale` on each lift row rather than inferring it from its type.
Some encouragement and observational results are additive.
Switchback contrasts are always additive.

Also check who the result describes. An **assigned** result and a
**triggered** result answer different population questions. Treatment can
affect who triggers, so conditioning on triggering can introduce selection
bias. Do not substitute a triggered result for an assignment-based comparison.

## Read the result as a decision, not just a number

### Interpret status and scope

| Result | What it tells you |
|---|---|
| Favorable or adverse finding | The applicable test establishes a direction, interpreted against the metric's preference. |
| Difference detected | A difference is established, but favorability is not. |
| Inconclusive | The applicable test or family selection did not establish the requested claim. |
| Unavailable | Required evidence is missing or the calculation could not supply it. Read the reason. |
| Guardrail passed | The safety test rules out the declared unacceptable loss. |

A passed guardrail can allow a small decline within its margin.
An inconclusive guardrail proves neither safety nor harm. An unavailable one
cannot be checked at all; the [guardrail troubleshooting example](../reference/result-fields.md#report-a-guardrail-that-cannot-be-checked)
shows how to repair the source and rerun.

An assignment check marked `not_rejected` found no mismatch; it does **not**
prove valid randomization. `failed` needs investigation. An unchecked,
unsupported or inapplicable check is not a pass. Use checks on the assigned
population, not triggered counts, to assess assignment.
See the [exact status meanings](../reference/result-fields.md#integrity-and-posterior-status).

### Program verdict

`result.decisions.program` combines the registered primary metrics,
guardrails and assignment checks:

| Status | Meaning |
|---|---|
| `pass` | Primaries meet their favorable-side tests, guardrails are certified, and assignment checks report no detected mismatch. |
| `block` | A decision requirement was not met; inspect `reason_codes`. |
| `not_evaluated` | A required result or declaration was unavailable; inspect `reason_codes`. |

This verdict applies to the declared whole-experiment decision. Daily,
as-of, breakout and exploratory results are not substitutes for it.
A `pass` does not establish assumptions that the data cannot check.

## Significance, p-values and selection

A row's test and its selection among multiple comparisons are different.
`stat_sig()` answers the row's test; `discovery` records family selection.
Use the report's final interpretation rather than cherry-picking one of them.
Unadjusted exploratory findings are not confirmatory discoveries.
See [Multiplicity](multiplicity.md) when testing several metrics or segments.

A confidence interval is not a posterior probability. A sampling result
does not imply that a posterior is available; use its recorded reason
instead of reconstructing one from the interval.
See [priors and Bayesian decisions](priors-and-decisions.md).

## Results as frames and flat rows

Export the complete result, including qualifications:

```python
full = result.to_frame(backend="polars")
compact = result.to_frame(backend="polars", columns="summary")
```

Use `result.readout(view)` when working with a particular view. With several
views, make a table from each view's collection rather than pooling their
conclusions. Exact columns, saved-result behavior and custom projections are
in [Result fields](../reference/result-fields.md).

## Before you act

1. Confirm the metric, population, comparison, scale and analysis window.
2. Read the estimate with its interval or confidence set—not the point alone.
3. Check assignment, required results and guardrails. Do not drop unavailable
   rows or treat a partial report as a complete decision.
4. Confirm the planned test and multiple-testing procedure. Repeated decisions
   need [sequential inference](sequential-inference.md), not repeated
   fixed-horizon intervals.
5. Share the qualifications alongside the conclusion.

## Detailed result contracts

Missing numbers, unusual confidence sets and exports need more precise
handling. The reference keeps the fields, reasons and runnable examples:

<a id="what-to-read_1"></a>
<a id="what-to-read-1"></a>

- [Result field lookup](../reference/result-fields.md#what-to-read).

<a id="fixed-horizon-guard-the-missing-point"></a>
<a id="when-a-point-is-missing"></a>

- [Missing points and zero-control conversions](../reference/result-fields.md#fixed-horizon-guard-the-missing-point):
  a missing point can still have a useful confidence set.

<a id="a-point-without-an-interval"></a>

- [A point without an interval](../reference/result-fields.md#a-point-without-an-interval):
  a finite estimate alone does not establish uncertainty.

<a id="small-segments-are-flagged-not-withheld"></a>

- [Small-segment reliability flags](../reference/result-fields.md#small-segments-are-flagged-not-withheld).

<a id="one-sided-tests"></a>

- [One-sided tests and open bounds](../reference/result-fields.md#one-sided-tests).

<a id="sequential-inference-two-coordinate-systems"></a>

- [Sequential ratios, lift and disconnected sets](../reference/result-fields.md#sequential-inference-two-coordinate-systems).

<a id="switchback-contrasts-are-additive"></a>

- [Additive switchback contrasts](../reference/result-fields.md#switchback-contrasts-are-additive).
