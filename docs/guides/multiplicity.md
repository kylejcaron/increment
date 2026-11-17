# Multiplicity

Testing more metrics gives chance more opportunities to look convincing.
Declare which results drive your decision before reading outcomes, then keep
the whole tested family when interpreting a selected result.

## Declaring families: primary, guardrails, secondaries

`AnalysisPlan` assigns three jobs:

- **Primary:** the result your experiment is meant to establish. Primaries
  share familywise `alpha`, divided across primary metrics and then treatment
  arms. Five primaries and one treatment arm each receive `0.01` when
  `alpha=0.05`.
- **Guardrail:** an unacceptable loss to rule out. Declare a `margin` or
  `margin_abs`; each guardrail uses full `alpha`, not a share. The compound
  safety claim requires **every** guardrail to be available and certified.
  An inconclusive guardrail establishes neither safety nor harm.
- **Secondary:** a discovery among additional metrics. The family uses
  false discovery rate (FDR) level `q`: a bound on the expected fraction of
  discoveries that are false positives, not a per-metric error rate.

With fixed-horizon inference, secondaries use Benjamini–Hochberg (BH). Its
FDR claim assumes independence or positive dependence (PRDS) across the
family; Increment does not check that condition.

This example declares one primary and one secondary. Both illustrative
columns are continuous means, not individual binary signup indicators.
The table requires the `tables` extra.

```python
import numpy as np
import polars as pl
import increment
from increment.tables import readout_table

design = increment.Randomized(
    control_group="control",
    allocation={"control": 0.5, "treatment": 0.5},
    allocation_scheme="independent",
)
plan = increment.AnalysisPlan(primary="sessions", secondaries=["signups"], q=0.10)
rng = np.random.default_rng(9)
n = 3000
variant = np.where(rng.random(n) < 0.5, "treatment", "control")
sessions = 4.0 + np.where(variant == "treatment", 0.6, 0.0) + rng.normal(0, 2.0, n)
signups = 0.10 + rng.normal(0, 0.02, n)
df = pl.DataFrame(
    {
        "user_id": [f"u{i}" for i in range(n)],
        "variant": variant,
        "enrolled_on": np.arange(n),
        "sessions": sessions,
        "signups": signups,
    }
)
experiment, reader = increment.from_frame(
    df,
    shape="summary",
    unit="user_id",
    group="variant",
    design=design,
    metrics=[
        increment.MetricSpec(name="sessions", type="mean"),
        increment.MetricSpec(name="signups", type="mean"),
    ],
    plan=plan,
)
results = increment.run(experiment, reader).readout()
readout_table(results, title="Metric families")
print(sorted((row.metric, row.role) for row in results))
```

<!-- output -->
```text
[('sessions', 'primary'), ('signups', 'secondary')]
```

Expand **Table evidence and checks** to inspect the families.
The primary uses its allocated test and has `discovery=None`; `discovery`
records selection of secondaries or breakouts. Selection does not turn an
exploratory finding into a confirmatory one.

## Reading multiplicity provenance

Check `multiplicity_status`, `family_id`, population and source scope.
`declared_plan` means a role was declared, not that every role shares joint
error control. Without a declared plan, metrics remain unadjusted; none is
silently promoted to primary.

**Filtering changes what you see, not the family that was tested.**
`results.filter(...)` and slicing retain the full family member lists and
partial-view qualifications. Choosing metrics for a read is not permission
to redefine a registered decision family; adding another metric does not
reallocate the primary family's alpha.

Assigned and triggered results have separate family identities. Switching
population or concatenating readouts does not create experiment-wide error
control. See [family metadata and metric selection](../reference/family-inference.md#family-metadata-and-metric-selection).

Missing results are not automatically removable. Some unavailable cells stay
in the family as non-rejections; other failures stop family selection with
`family.evidence.incomplete`. Read the reason instead of silently dropping a
cell. The reference includes runnable
[unavailable-member and incomplete-family examples](../reference/family-inference.md#missing-and-incomplete-families).

## `correction` and `MultiplicitySpec` for segmented views

`AnalysisPlan.view_multiplicity` controls segmented breakout/as-of views,
separately from the whole-experiment secondary family.

- **Exploratory FDR selection:** `MultiplicitySpec(correction="bh", q=0.10)`.
- **Divide alpha across eligible cells:** `MultiplicitySpec(correction="bonferroni")`.
- **No view correction:** `MultiplicitySpec(correction="none")`.

`q` is accepted only with `"bh"` and defaults to `0.10` there. With no
`view_multiplicity`, randomized breakouts default to BH, encouragement
breakouts to no correction, and as-of views to no correction unless segmented.
Sequential asymptotic breakouts instead use fixed-roster Bonferroni.

Fixed-horizon randomized breakouts use each metric's declared alternative
and relative or absolute margin. Absolute margins use metric units, not
relative-interval exclusion of zero. Segments remain exploratory—even for a
guardrail—and cannot change the headline verdict.

Sequential dataframe breakouts need a complete `InferenceSpec.segments`
declaration before outcomes. Asymptotic breakouts refuse BH. Encouragement
cannot combine segments with `SequentialCompliancePolicy(family=True)`:
use separately allocated uptake (`family=False`) or an unsegmented joint
family. See [eligible cells and source limits](../reference/family-inference.md#segmented-views).

## e-BH under sequential inference

Sequential secondaries use e-BH rather than p-value BH. Its finite-sample
stopping-time guarantee requires **valid e-processes**: evidence valid at
every stopping time under the registered joint reveal rules. The selector
cannot establish those assumptions. `always_valid` supplies the supported
exact conversion/retention route; `asymptotic_mean` supplies an asymptotic
approximation, not finite-sample error control.

For continuous outcomes, declare the following plan before collecting them.
Here the synthetic frame above is reused only to illustrate the call:

```python
av_plan = increment.AnalysisPlan(
    primary="sessions",
    secondaries=["signups"],
    q=0.10,
    inference=increment.InferenceSpec(kind="asymptotic_mean"),
)
experiment, reader = increment.from_frame(
    df,
    shape="summary",
    unit="user_id",
    group="variant",
    design=design,
    metrics=[
        increment.MetricSpec(name="sessions", type="mean"),
        increment.MetricSpec(name="signups", type="mean"),
    ],
    plan=av_plan,
    exposure_date="enrolled_on",
)
av_results = increment.run(experiment, reader).readout()
readout_table(av_results, title="Sequential metric families")
```

This needs explicit fixed allocation, finalized unit outcomes, and
outcome-independent exposure-date ordering—not the `control=` shortcut or
caller row order. Follow the [sequential route's assumptions and capture steps](sequential-inference.md)
before using it for decisions. The discovery set is controlled at the current
stopping date; **the union of discoveries across dates is not controlled**.

For triggered families, triggering must be arm-independent, there must be no
pre-trigger treatment effect, and trigger order must be stationary.
Looks/stops/freezes may use only already-revealed joint triggered information.
Assigned results can expose future triggered outcomes, including the same
metric's result. See [the triggered stopping condition](../reference/sequential-contracts.md#per-cell-coverage-and-the-family-verdict).

## FCR re-estimation of discovered cells

A selected secondary or breakout already carries a selection-adjusted
interval. It is not generally the same width as an unselected row's interval.
Read `discovery` together with the interval and family level, not as a
synonym for the row's individual test.
The [FCR reference](../reference/family-inference.md#fcr-re-estimation-of-discovered-cells)
defines the adjustment, nominal cap, quantile behavior and sequential geometry.

## Caveats

These are fixed-family procedures, not online FDR across experiments.
The plan fixes hypotheses, alternatives, margins and sequential registration
before data is read; final evaluation does not authorize changing them.
See the [fixed-horizon dependence condition](../limitations.md#fixed-horizon-fdr-control-assumes-a-dependence-condition-that-is-not-checked)
and [stopping-date limits](../limitations.md#stopping-date-guarantees-require-the-registered-reveal-contract).

## Two-phase stored runs

Saving and replaying a run preserves its registered family, compiled plan and
decision inputs. It does not create a new family or online correction across
runs. See [stored-run contracts](../reference/family-inference.md#two-phase-stored-runs).
