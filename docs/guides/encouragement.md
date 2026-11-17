# Encouragement Designs

Use an encouragement design when randomized assignment changes treatment uptake
but does not force it. You need each unit's assignment and uptake; add outcomes
to estimate effects on those outcomes.

Choose the question before running:

- **Intent to treat (ITT):** what did being encouraged change?
- **Compliance:** how much did encouragement change treatment uptake?
- **Local average treatment effect (LATE):** what did treatment change among
  people whose uptake changed because of encouragement—not everyone?

ITT and compliance do not require an exclusion restriction. LATE additionally
requires a relevant first stage, exclusion (assignment affects the outcome
only through uptake), and monotonicity (no defiers). A declaration records your
justification; it cannot validate these assumptions.

## Declare assignment and uptake

This deliberately balanced toy dataset makes the scales easy to read: no control
unit takes up treatment, half the encouraged units do, and treatment adds 3
revenue units. Real assignment must be randomized.

```python
import numpy as np
import pandas as pd
import increment

n = 400
clicked = np.concatenate([np.zeros(200), np.ones(100), np.zeros(100)])
encouragement_df = pd.DataFrame(
    {
        "user_id": [f"u{i:03d}" for i in range(n)],
        "variant": ["control"] * 200 + ["encouraged"] * 200,
        "clicked": clicked,
        "revenue": 8.0 + 3.0 * clicked + np.tile([-1.0, 1.0], 200),
    }
)
experiment, reader = increment.from_frame(
    encouragement_df,
    shape="summary",
    unit="user_id",
    group="variant",
    metrics={"revenue": "mean"},
    design=increment.Encouragement(
        control_group="control",
        uptake=increment.UptakeSpec(fact="clicked"),
        one_sided=True,
        exclusion_restriction=increment.ExclusionRestriction(
            justification="the prompt affects revenue only through help-button clicks",
        ),
    ),
)
results = increment.run(experiment, reader).readout()
itt = next(row for row in results if row.estimand == "itt")
compliance = next(row for row in results if row.estimand == "compliance")
late = next(row for row in results if row.estimand == "late" and row.value_scale == "absolute")
print(f"ITT revenue lift: {itt.lift.value:+.2%}")
print(f"Uptake increase: {100 * compliance.lift.value:+.0f} percentage points")
print(f"LATE revenue difference: {late.lift.value:+.2f}")
```

<!-- output -->
```text
ITT revenue lift: +18.75%
Uptake increase: +50 percentage points
LATE revenue difference: +3.00
```

Encouragement increases revenue by 18.75% here. Uptake increases by 50 percentage
points; the additive ITT difference of 1.5 divided by 0.5 gives LATE of 3 revenue
units per complier. That is not an effect of 3 units on every enrolled person.
Read each row's interval as well as its point; these toy numbers do not establish
coverage or validate a real encouragement mechanism.

If exclusion is not credible, omit `exclusion_restriction` and request only
`estimands=("itt", "compliance")` on `run`. The default requests all three;
any request including LATE without exclusion fails before outcomes are read
with `identification.encouragement.exclusion_required`, even if a weak first
stage would later prevent LATE.

## Declare on `from_definitions`

Keep assignment's `control_group` and `allocation` on the experiment. Its
`design` declares `mechanism: encouragement`, the uptake fact/window, and,
for LATE, the exclusion justification. The uptake fact must belong to a
source containing the experiment's unit.

[Definitions and supported inputs](../reference/causal-methods.md#definitions-and-supported-inputs)
cover dataframe summaries/panels, warehouse definitions, artifacts and moments.
The [complete daily warehouse example](../reference/causal-methods.md#daily-warehouse-example)
creates its own data and definitions.

## Read the estimands

Check `estimand` and `value_scale`: a relative LATE row uses the **complier**
control mean, while additive LATE uses outcome units. Compliance is design-level,
using enrolled units and the uptake window; each outcome's LATE uses the first
stage on its own outcome cohort. Missing outcomes can make those cohorts differ.

### When LATE is unavailable

A weak first stage retains the requested LATE row with `lift=None` and
`failure_code="estimation.encouragement.late.weak_first_stage"`.
Strongly reversed uptake has its own
`estimation.encouragement.late.negative_first_stage` failure. Neither means
“no treatment effect,” and neither silently drops the row.
Inspect the [first-stage context](../reference/causal-methods.md#outcome-cohorts-and-first-stage-failures).

### Margins and selection

A guardrail margin applies to fixed-horizon whole-window ITT, not LATE or
design-level compliance. A margined metric requested as LATE without ITT
refuses. As-of, breakout and sequential margins are unsupported.
Secondary discovery is ITT-based, not a separate LATE hypothesis test;
fixed-horizon compliance is outside selection.
See [selection, margin and binary-ITT details](../reference/causal-methods.md#estimands-selection-and-margins).

### Uptake without outcomes

For fixed-horizon uptake alone, request `estimands=("compliance",)`;
pass `metrics=[]` to dataframe/portable constructors when there are no outcomes.
It needs no outcome catalog or sequential policy. Do not replace undefined
outcomes with zeros to evade the panel's `missing="skip"` refusal;
mixed ITT/LATE requests still need their outcome data.

### Daily and sequential monitoring

Daily uptake is a first-stage rate; per-day and sequential LATE are unsupported.
Triggered sequential compliance and clustered daily outcome histories are
unavailable.

Uptake-only monitoring needs `InferenceSpec(kind="always_valid")`, an explicit
`SequentialCompliancePolicy`, and matching registration—not just an empty
catalog. Registered ITT/compliance also require joint finalization and
immutable-prefix evidence. Use `completed_windows_only=True` for sequential
as-of readouts. Compliance enters discovery selection only when explicitly
included by policy.

The binary compliance construction is exact under its conditions; continuous
ITT uses an asymptotic mean construction. A family containing continuous ITT
has an asymptotic guarantee even if its compliance member is exact. Declaring
exclusion does not enable sequential LATE.

Read [uptake-only request requirements](../reference/causal-methods.md#uptake-only-and-sequential-requests)
and [windows, timelines and saved state](../reference/causal-methods.md#windows-timelines-and-retained-state)
before monitoring or replaying results.

## Run the example

The [Encouragement notebook](../examples/encouragement.md) provides a larger
simulation and rendered results. See the [API reference](../api.md) for
`Encouragement`, `UptakeSpec`, and `ExclusionRestriction`.

## References and assumptions

LATE follows Imbens and Angrist,
[“Identification and Estimation of Local Average Treatment Effects”](https://doi.org/10.2307/2951620).
Random assignment, relevance, exclusion and monotonicity are assumptions,
not conclusions from passing the first-stage check.
