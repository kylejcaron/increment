# Overlapping experiments and joint launches

`estimate_overlap_effects` answers launch questions about two experiments that
ran at the same time on overlapping users:

- Should both changes launch together?
- Should one launch only if the other is also launched?
- Does the effect of one depend on the other (their interaction)?

Analyzing each experiment on its own reports an effect averaged over whatever
mix of the other experiment's arms existed during the test. After launch,
that mix no longer exists: the other experiment will be fully on or fully off.
This family estimates the effects for each specific launch combination
instead.

This family has its own ingress. It does not run through any `Analysis`
constructor, because those carry one experiment's arms rather than two
experiments' joint assignments; the "What runs where" table in
[Capabilities by entry point](../reference/capabilities-by-entry-point.md#what-runs-where)
records it as not expressible on every `Analysis` entry point. Build an
`OverlapCells` instead (see below).

## What it estimates

Each jointly eligible unit falls in one of four **joint assignment cells**,
written `(first_treated, second_treated)`: `00` (both control), `10`, `01`,
and `11` (both treated). Let `mu_ab` be the mean outcome in cell `ab` over the
target population. Every contrast is a fixed linear combination of the four
cell means:

| Contrast | Meaning | Weights on `mu` |
| --- | --- | --- |
| `joint_launch` | Launch both versus launch neither | `mu_11 - mu_00` |
| `first_only_launch` | Launch the first alone; the second stays at control | `mu_10 - mu_00` |
| `second_only_launch` | Launch the second alone; the first stays at control | `mu_01 - mu_00` |
| `first_given_second_treatment` | The first's effect once the second has launched | `mu_11 - mu_01` |
| `second_given_first_treatment` | The second's effect once the first has launched | `mu_11 - mu_10` |
| `interaction` | How much the first's effect changes when the second launches | `mu_11 - mu_10 - mu_01 + mu_00` |
| `marginal_first` | The first's effect averaged over the second's declared treatment mix `p2` | `p2 (mu_11 - mu_01) + (1 - p2)(mu_10 - mu_00)` |
| `marginal_second` | The second's effect averaged over the first's declared treatment mix `p1` | `p1 (mu_11 - mu_10) + (1 - p1)(mu_01 - mu_00)` |

The marginal effects use the **declared** treatment probability, so they
describe the experiment as designed rather than the mix that happened to be
observed. A marginal effect is not biased merely because it differs from a
conditional launch effect: the two answer different questions. Only
`interaction` tells you whether they differ.

The estimate of each contrast is the same weighted sum of the observed cell
means. Its standard error is `sqrt(sum_k w_k^2 s_k^2 / n_k)` from the per-cell
sample variances, which is the HC2 sandwich of the saturated regression on
cell indicators. The interval uses a Student-t reference with
Welch-Satterthwaite degrees of freedom and is asymptotic in the per-cell
counts.

## Inputs and what is checked

Provide one row per unit with, for each experiment:

- a pre-treatment **eligibility** flag;
- the **assignment** arm, recorded only for eligible units and null otherwise;
- the **assignment time**, null when unassigned;

plus the start and end of the unit's **outcome window** and one numeric
column per metric.

Absence from an experiment is never read as its control arm. Units eligible
for only one experiment, or for neither, are counted in the eligibility audit
and never pooled into the target. The target population is the units eligible
for both experiments.

Before any mean is computed, both ingresses check, in this order, and refuse
with a coded error on the first failure:

1. every row has a unit identifier and no unit appears twice (the join is
   one row per unit);
2. eligibility is recorded for every row; eligible units are assigned and
   ineligible ones are not; arms use the declared labels; assigned units have
   an assignment time;
3. at least one unit is eligible for both experiments;
4. within the target, every outcome window opens no earlier than the later of
   the two assignments and all windows have the same length (a fixed horizon);
5. every outcome is present and finite.

```python
import datetime as dt

import numpy as np
import pandas as pd

from increment import (
    ExperimentColumns,
    OverlapCells,
    OverlapDesign,
    OverlapExperiment,
    estimate_overlap_effects,
)

rng = np.random.default_rng(83)
n = 4000
t0 = dt.datetime(2026, 3, 2)
elig_a = rng.random(n) < 0.9
elig_b = rng.random(n) < 0.8
treat_a = rng.random(n) < 0.5
treat_b = rng.random(n) < 0.3
cell_mean = np.array([10.0, 11.0, 10.5, 13.0])  # cells 00, 10, 01, 11
y = cell_mean[treat_a.astype(int) + 2 * treat_b.astype(int)] + rng.normal(0, 2, n)

frame = pd.DataFrame(
    {
        "user_id": np.arange(n),
        "eligible_checkout": elig_a,
        "eligible_search": elig_b,
        "checkout_arm": np.where(elig_a, np.where(treat_a, "treatment", "control"), None),
        "search_arm": np.where(elig_b, np.where(treat_b, "new", "old"), None),
        "checkout_assigned_at": pd.to_datetime(np.where(elig_a, t0, pd.NaT)),
        "search_assigned_at": pd.to_datetime(np.where(elig_b, t0, pd.NaT)),
        "window_start": t0 + dt.timedelta(days=1),
        "window_end": t0 + dt.timedelta(days=15),
        "revenue": y,
    }
)

design = OverlapDesign(
    first=OverlapExperiment(name="checkout", treatment_probability=0.5),
    second=OverlapExperiment(
        name="search", control="old", treatment="new", treatment_probability=0.3
    ),
)
cells = OverlapCells.from_frame(
    frame,
    design,
    unit="user_id",
    columns={
        "checkout": ExperimentColumns(
            eligible="eligible_checkout",
            assignment="checkout_arm",
            assigned_at="checkout_assigned_at",
        ),
        "search": ExperimentColumns(
            eligible="eligible_search",
            assignment="search_arm",
            assigned_at="search_assigned_at",
        ),
    },
    outcome_start="window_start",
    outcome_end="window_end",
    metrics={"revenue": "revenue"},
)

readout = estimate_overlap_effects(
    cells,
    contrasts=("interaction", "joint_launch", "first_only_launch", "second_only_launch"),
    roles={"revenue": "primary"},
)
joint = readout.row("revenue", "joint_launch")
assert joint.available and joint.lb < 3.0 < joint.ub
assert joint.multiplicity_status == "bonferroni_fwer" and joint.family_size == 4
assert readout.audit.jointly_eligible + readout.audit.first_only_eligible <= n
assert not readout.assignment_check.mismatch
```

`OverlapCells.from_frame` accepts pandas, polars, or pyarrow frames.
`OverlapCells.from_table` accepts an Ibis table and runs the same checks and
per-cell summaries as warehouse aggregates; only counts, a few example values
for refusal messages, and four summaries per metric leave the warehouse. The
same per-unit data yields the same rows through either ingress, and every
check raises the same refusal code on both.

## Reading the readout

`readout.rows` holds one `OverlapEffect` per declared `(metric, contrast)`,
in declaration order. Each row carries its `weights` over `cells`
(`00, 10, 01, 11`), so it can be recomputed from `readout.cells`, the per-cell
counts, means, and variances it was estimated from.

A row whose contrast needs an empty cell, or a cell with a single unit (no
variance), is **unavailable**: it keeps its identity and multiplicity fields,
sets `available=False`, reports `unavailable_reason` (`cell_empty` or
`cell_variance_unidentified`) and the `unsupported_cells`, and has null
estimates. It is never filled by summing separate scorecards or extrapolating
across the empty cell. Other contrasts that do not need that cell are still
reported.

`readout.assignment_check` compares the four cell counts with the shares
implied by the declared independent assignment law (a fixed-look Pearson
chi-square at `0.001`). A `mismatch` signals a broken law or a join that lost
units by arm. `readout.design` and `readout.audit` keep the assignment
provenance and the eligibility counts with the result.

A readout is a frozen Pydantic model and round-trips through
`model_dump_json` / `OverlapReadout.model_validate_json`, pickle, and
`copy.deepcopy`.

## Declaring contrasts and multiplicity

Only the contrasts you declare are estimated and corrected for; there is no
search over launch combinations. Declare the family before looking at results.

| Role | Procedure | Interval | `significant` means |
| --- | --- | --- | --- |
| `primary` | Bonferroni across every primary metric times every declared contrast at `alpha` | Simultaneous, at `alpha / family_size` | `p_value <= alpha / family_size` |
| `secondary` | Benjamini-Hochberg across secondary metrics times declared contrasts at `q` | Marginal, at `alpha` | BH-selected at `q` |
| `unassigned` (default) | None (`exploratory_unadjusted`) | Marginal, at `alpha` | `p_value <= alpha` |

Unavailable rows stay in their family, so they never shrink it; under BH they
enter with `p = 1`. `guardrail` is refused
(`overlap.multiplicity.guardrail_unfinished`): read a guardrail per experiment
through `Analysis` for now.

## Planning

The interaction uses all four cells with unit weights, so at the same sample
size it is less precise than any launch contrast. Plan for both the
interaction you want to detect and the launch contrast you will decide on:

```python
from increment import OverlapDesign, OverlapExperiment
from increment import overlap_contrast_power, overlap_minimum_detectable_effect

design = OverlapDesign(
    first=OverlapExperiment(name="checkout", treatment_probability=0.5),
    second=OverlapExperiment(name="search", treatment_probability=0.3),
)
interaction = overlap_minimum_detectable_effect(
    design, "interaction", n_total=20_000, sd=2.0, family_size=4
)
launch = overlap_minimum_detectable_effect(
    design, "joint_launch", n_total=20_000, sd=2.0, family_size=4
)
assert interaction > launch
assert (
    abs(
        overlap_contrast_power(
            design, "interaction", effect=interaction, n_total=20_000, sd=2.0, family_size=4
        )
        - 0.8
    )
    < 1e-9
)
```

`n_total` is the number of jointly eligible units, and `family_size` is the
Bonferroni family the readout will use. `sd` may be one value or a mapping per
cell. These solvers use the estimator's own variance at the expected
allocation with a normal reference: a planning approximation that ignores the
Welch correction and the chance of a sparse cell. `simulate_overlap_power`
instead draws independent Bernoulli assignments and runs the runtime
estimator, including its interval, multiplicity share, and unavailable cells.

## Compatibility and limits

| Combination | Status | Note |
| --- | --- | --- |
| Mean metrics, two independently randomized binary experiments | `supported` | Fixed horizon, unit randomization |
| Multiplicity roles `primary`, `secondary`, `unassigned` | `supported` | See the table above |
| Guardrail role | `unfinished` | Refused: `overlap.multiplicity.guardrail_unfinished` |
| Dependent or mutually exclusive assignment layers | `construction_limited` | Refused: `overlap.design.joint_law_unsupported` |
| Sequential monitoring | `construction_limited` | No look-structure construction for joint contrasts; not expressible |
| More than two experiments, or more than two arms | `construction_limited` | Not expressible |
| CUPED, ratio metrics, clustered randomization, winsorization, breakouts | `unfinished` | Not expressible on this family; none of the single-experiment guarantees is inherited |

## Not in scope

This family does not allocate traffic or serve feature flags. Network
spillovers between units and synthesis across replicated experiments are
separate problems. A joint launch contrast is evidence for one decision, not a
portfolio return on investment.
