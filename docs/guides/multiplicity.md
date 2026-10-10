# Multiplicity

Testing several metrics at once inflates the chance that at least one
looks significant by pure noise. Increment's `AnalysisPlan` separates
metrics into roles with different multiplicity treatment: a `primary`
takes a share `alpha / n_primaries` of the familywise `alpha`, split
again across that metric's own non-control arms at estimation; a
`guardrail` tests unsplit, at the full `alpha`, since it must not move
adversely; a `secondaries` family is judged as a false-discovery-rate
problem at level `q` instead of a per-metric significance test.

## Reading multiplicity provenance

Lift, breakout and daily/as-of lift rows expose `multiplicity_status`
alongside the existing `role`, interval alpha and (where the route has a
source-scoped family) `family_id`. The status discloses how the metric was
declared or selected; it does not apply a new correction:

| `role` | `multiplicity_status` | Interpretation |
|---|---|---|
| `None` | `undeclared_plan` | No metric role was declared; unadjusted. |
| `unassigned` | `unassigned_in_plan` | A plan exists, but this metric has no assigned role; unadjusted. |
| `primary`, `secondary`, `guardrail` | `declared_plan` | Retains the existing role-specific allocation and procedure. |
| `exploratory`, outside a BH/e-BH or Bonferroni family | `exploratory_unadjusted` | Exploratory, without family adjustment. |
| `exploratory`, inside a BH/e-BH or Bonferroni family | `exploratory_family` | Member of the named adjusted family: BH/e-BH selection or Bonferroni alpha division. |

`readout_table` keeps this disclosure in the header when all rows share an
informative status. For mixed statuses, it adds a superscript marker to
each known unadjusted or exploratory row and explains the marker and row
count in the header; declared-plan rows need no marker. The original
`multiplicity_status` and family fields remain available on result objects
and in `to_frame()`.

For example, five metrics without a declared plan each retain alpha `0.05`
and report `undeclared_plan`; five declared primaries share alpha `0.05`,
so each receives `0.01` for a single treatment arm. `declared_plan` is
not a blanket claim of joint error control: guardrails still use their
existing unsplit alpha. No metric is silently promoted to primary.

Family membership comes from the original source-scoped readout, never
the number of visible rows. `results.filter(...)` and slicing retain the
full `results.metadata.scope.families` member lists; `to_frame()` carries
the row status and `family_id`, and saved readout envelopes preserve them.
Adding an exploratory metric does not enlarge the declared primary
family or reallocate its alpha.

Assigned and triggered populations have separate family identities.
Switching the visible population only changes the rows shown, not either
population's allocation or membership. Concatenating readouts preserves
each source's families; it does not create one jointly controlled
experiment-wide family. Use `family_id` and the source/population scope,
not a shared family name, to identify which guarantee applies.
Under a registered sequential plan the triggered secondary family is selected
by e-BH on its own chain with the same guarantee and status labels as the
assigned family; do not time triggered stops or freezes on other metrics'
assigned results (see
[Triggered populations](sequential-inference.md#triggered-populations)).

## Declaring families: primary, guardrails, secondaries

```python
import numpy as np
import polars as pl
from increment import Analysis, AnalysisPlan, MetricSpec

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

plan = AnalysisPlan(primary="sessions", secondaries=["signups"], q=0.10)
results = Analysis.from_unit_summary(
    df,
    unit="user_id",
    group="variant",
    control="control",
    metrics=[MetricSpec(name="sessions", type="mean"), MetricSpec(name="signups", type="mean")],
    plan=plan,
).run()
for r in results:
    print(f"{r.metric:>10} role={r.role:<10} discovery={r.discovery} lift={r.lift.value:+.2%}")
```

```text
  sessions role=primary    discovery=None lift=+12.94%
   signups role=secondary  discovery=False lift=-0.85%
```

`q=0.10` is a false-discovery level, not a second `alpha` or a
quantile. It bounds the expected fraction of *discovered* secondaries
that are false positives, not each metric's type-I rate. A `primary`
row always has `discovery=None`: family selection applies only to
secondaries and breakouts.

## `correction` and `MultiplicitySpec` for segmented views

`AnalysisPlan.view_multiplicity: MultiplicitySpec | None` governs
segmented breakout and as-of views specifically, independent of the
top-level `secondaries`/`q` family above.
`MultiplicitySpec(correction="bh", q=0.10)`,
`MultiplicitySpec(correction="bonferroni")`, and
`MultiplicitySpec(correction="none")` are the three options; `q` is only
accepted under `correction="bh"` (and defaults to `0.10` there if
omitted) -- setting `q` under `"bonferroni"` or `"none"` is refused.
Leaving `view_multiplicity=None` keeps the per-route defaults:
randomized breakout BH at `q` (fixed-roster Bonferroni for a sequential
`AsymptoticMean` registration), encouragement breakout uncorrected, and
as-of uncorrected unless segmented.

For randomized sequential dataframe breakouts, `InferenceSpec.segments` declares the
complete level roster before outcomes; automatic registration keeps absent
levels in the family. Exact e-BH uses the breakout view's `q`, including a
`MultiplicitySpec` override, not a second family inferred from observed rows.
Asymptotic breakouts retain fixed-roster Bonferroni and refuse BH. Exact
multi-arm whole-window monitoring counts every secondary metric/arm cell in
the family and splits each primary's allocation across its treatment arms.
An encouragement plan cannot combine predeclared segments with
`SequentialCompliancePolicy(family=True)`: its breakout policy is uncorrected.
Use separately allocated uptake cells (`family=False`) or an unsegmented
joint family; existing encouragement breakout-readout restrictions remain.

## e-BH under sequential inference

Under `AnalysisPlan(inference=InferenceSpec(kind="asymptotic_mean"))` or
`InferenceSpec(kind="always_valid")`, the secondary family uses e-BH
(Wang & Ramdas 2022: ordinary BH applied to `1/e`) instead of ordinary
p-value BH. e-BH controls FDR at `<= q` under arbitrary dependence and
stopping time, but only when every input is a valid e-process: an e-value
that remains valid at every stopping time, not just at one fixed look.
`e_bh_select` assumes that property; it does not establish it. The
asymptotic-mean route builds its e-process from the same delta-method
sampling distribution as the fixed-horizon interval, so its time-uniform
guarantee is asymptotic and empirical, not finite-sample. The exact
`kind="always_valid"` route, limited to conversion and retention metrics,
is the finite-sample alternative.

```python
from increment import AnalysisPlan, InferenceSpec, Randomized

design = Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5})
av_plan = AnalysisPlan(
    primary="sessions",
    secondaries=["signups"],
    q=0.10,
    inference=InferenceSpec(kind="asymptotic_mean"),
)
av_results = Analysis.from_unit_summary(
    df,
    unit="user_id",
    group="variant",
    design=design,
    metrics=[MetricSpec(name="sessions", type="mean"), MetricSpec(name="signups", type="mean")],
    plan=av_plan,
    exposure_date="enrolled_on",
).run()
for r in av_results:
    print(f"{r.metric:>10} inference={r.inference:<13} discovery={r.discovery}")
```

```text
  sessions inference=asymptotic_mean discovery=None
   signups inference=asymptotic_mean discovery=False
```

A sequential `AnalysisPlan` needs a `design=` with an explicit
`allocation` (not the dataframe shortcut `control=`), since the
registered asymptotic-mean runtime must be declared before any outcome
is read. A unit summary also names each unit's `exposure_date` (a date or
day index); units are revealed in that order, never in row order. See
[Sequential inference](sequential-inference.md) for the
registration, capture and error-control mechanics this route shares
with an ordinary (non-multiplicity) sequential primary.

## FCR re-estimation of discovered cells

A secondary or breakout cell selected under BH (fixed-horizon p-values) or
e-BH (sequential e-values) gets a reported interval re-estimated at
`fcr_alpha = min(q * R / m, nominal_alpha)`. Here, `R` is the number of
selected cells and `m` is the family size. The interval is wider than the
nominal interval when few cells are selected from a large family, and
narrower as more cells are selected, up to the nominal-alpha cap.

Treat a discovered secondary's interval as already carrying this
correction; it is not the same width as a non-discovered row's interval.
`family_guarantee` reports `"finite_sample"` only when every family member
is exact (`always_valid`) and `"asymptotic_sequential"` when any member is
asymptotic.
## Caveats

Fixed-horizon BH assumes independence or positive dependence (PRDS)
across the metric x arm x segment family, and this is not checked --
read
[Fixed-horizon FDR control assumes a dependence condition that is not checked](../limitations.md#fixed-horizon-fdr-control-assumes-a-dependence-condition-that-is-not-checked).
An always-valid discovery set's guarantee holds for the current date;
the union across dates is not controlled -- read
[Stopping-date guarantees require the registered reveal contract](../limitations.md#stopping-date-guarantees-require-the-registered-reveal-contract).
