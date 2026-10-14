# Multiplicity

When you test many metrics, chance can produce a false positive. Assign roles in `AnalysisPlan`: primaries share familywise `alpha` across metrics and then treatment arms; guardrails test adverse effects at full `alpha`; secondaries use false-discovery-rate level `q`.

## Reading multiplicity provenance

Lift, breakout, and daily/as-of rows expose `multiplicity_status` and, where applicable, `family_id`. These fields explain how a row was declared or selected; they do not add a correction.

| Role / status | Meaning |
|---|---|
| `None` / `undeclared_plan` | No role declared; unadjusted. |
| `unassigned` / `unassigned_in_plan` | Plan exists, but this metric has no role; unadjusted. |
| `primary`, `secondary`, `guardrail` / `declared_plan` | Uses its declared role's existing procedure. |
| `exploratory` / `exploratory_unadjusted` | Outside an adjusted family; no family adjustment. |
| `exploratory` / `exploratory_family` | In a named BH/e-BH or Bonferroni family. |

`readout_table` shows a common informative status in its header. For mixed statuses, it marks each known unadjusted or exploratory row and gives the marked row count. Result objects and `to_frame()` retain the original status and family fields.

Five undeclared metrics each keep alpha `0.05`; five declared primaries share `0.05`, so each gets `0.01` with one treatment arm. `declared_plan` does not mean every role has joint error control: guardrails use their existing unsplit alpha. No metric is silently promoted to primary.

Family membership comes from the original source-scoped readout, not from visible rows. Filtering or slicing keeps the full family; adding exploratory metrics does not enlarge the primary family. Assigned and triggered populations have separate family identities. Concatenating sources does not create an experiment-wide correction. Use `family_id` with source and population scope to identify the guarantee.

A selected result is not automatically confirmatory. Fixed-horizon BH retains its q level and FCR-adjusted intervals, but depends on independence or positive dependence (PRDS), which is not checked. A triggered sequential family uses e-BH on its own chain and has conditional guarantees only: triggering must be arm-independent, there must be no pre-trigger treatment effect, trigger order must be stationary, and looks, stops, and freezes must follow the joint triggered filtration. Do not time these decisions using assigned results that reveal future triggered outcomes. Filtering does not shrink a family or remove a partial-view indication. See [Triggered populations](sequential-inference.md#triggered-populations).


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

<!-- output -->
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

<!-- output -->
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
