# Multiplicity

When you test many metrics, chance can produce a false positive. Assign roles in `AnalysisPlan`: primaries share familywise `alpha` across metrics and then treatment arms; guardrails test adverse effects at full `alpha`; secondaries use false-discovery-rate level `q`.

## Declaring families: primary, guardrails, secondaries

```python
import numpy as np
import polars as pl
from increment import Analysis, AnalysisPlan, MetricSpec, Randomized

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
design = Randomized(
    control_group="control",
    allocation={"control": 0.5, "treatment": 0.5},
    allocation_scheme="independent",
)
plan = AnalysisPlan(primary="sessions", secondaries=["signups"], q=0.10)
results = Analysis.from_unit_summary(
    df,
    unit="user_id",
    group="variant",
    design=design,
    metrics=[MetricSpec(name="sessions", type="mean"), MetricSpec(name="signups", type="mean")],
    plan=plan,
).run()
print(results)
```

<!-- output -->
```text
assigned — treatment vs control
sessions: control → treatment; n control=1494; n treatment=1506; mean control=4.01529; mean treatment=4.53469; effect=12.9354%; interval=[9.12107, 16.883]%; estimand=itt; stat_sig=True
sessions (primary, relative lift): 0.129354, set [0.0912107, 0.16883] at 0.95 confidence.
sessions interpretation: excludes zero at 95.0% confidence; effect direction supported (increase).
sessions additive effect: 0.519393, set [0.373532, 0.665254].
sessions test alternative: two-sided; null (relative lift): 0.
sessions result identity: method unadjusted; method role decision; estimand itt.
signups: control → treatment; n control=1494; n treatment=1506; mean control=0.100916; mean treatment=0.100055; effect=-0.85373%; interval=[-2.24037, 0.552583]%; estimand=itt; stat_sig=False
signups (secondary, relative lift): -0.0085373, set [-0.0224037, 0.00552583] at 0.95 confidence.
signups interpretation: does not reject; inconclusive, not evidence of no effect.
signups additive effect: -0.000861552, set [-0.00227681, 0.000553707].
signups test alternative: two-sided; null (relative lift): 0.
signups result identity: method unadjusted; method role decision; estimand itt; not selected.
signups selection: not selected; separate from interval evidence.
Integrity: not_rejected — no detected mismatch; not proof of valid randomization.
Multiplicity (primary): declared_plan; family primary.
Multiplicity (secondary): declared_plan; family secondary.
```

## Reading multiplicity provenance

Check `multiplicity_status` and `family_id` to see which correction applies. Filtering changes what you see, not the family that was tested.

!!! warning "A selected result is not automatically confirmatory"
    Guardrails use unsplit alpha; `declared_plan` does not mean every role has joint error control. Exploratory discoveries stay exploratory. Assigned and triggered populations have separate families; concatenating readouts does not create an experiment-wide correction.

    A triggered sequential family's guarantees are conditional: triggering must be arm-independent, there must be no pre-trigger treatment effect, trigger order must be stationary, and looks, stops, and freezes must follow the joint triggered filtration. Do not time these decisions using assigned results that reveal future triggered outcomes, including the same metric's result. See [Triggered populations](sequential-inference.md#triggered-populations).

??? info "Family metadata and disclosure"
    Lift, breakout and daily/as-of lift rows expose `multiplicity_status` alongside the existing `role`, interval alpha and (where the route has a source-scoped family) `family_id`. The status discloses how the metric was declared or selected; it does not apply a new correction:

    | `role` | `multiplicity_status` | Interpretation |
    |---|---|---|
    | `None` | `undeclared_plan` | No metric role was declared; unadjusted. |
    | `unassigned` | `unassigned_in_plan` | A plan exists, but this metric has no assigned role; unadjusted. |
    | `primary`, `secondary`, `guardrail` | `declared_plan` | Retains the existing role-specific allocation and procedure. |
    | `exploratory`, outside a BH/e-BH or Bonferroni family | `exploratory_unadjusted` | Exploratory, without family adjustment. |
    | `exploratory`, inside a BH/e-BH or Bonferroni family | `exploratory_family` | Member of the named adjusted family: BH/e-BH selection or Bonferroni alpha division. |

    `readout_table` keeps this disclosure in the header when all rows share an informative status. For mixed statuses, it adds a superscript marker to each known unadjusted or exploratory row and explains the marker and row count in the header; declared-plan rows need no marker. The original `multiplicity_status` and family fields remain available on result objects and in `to_frame()`.

    For example, five metrics without a declared plan each retain alpha `0.05` and report `undeclared_plan`; five declared primaries share alpha `0.05`, so each receives `0.01` for a single treatment arm. `declared_plan` is not a blanket claim of joint error control: guardrails still use their existing unsplit alpha. No metric is silently promoted to primary.

    Family membership comes from the original source-scoped readout, never the number of visible rows. `results.filter(...)` and slicing retain the full `results.metadata.scope.families` member lists; `to_frame()` carries the row status and `family_id`, and saved readout envelopes preserve them. Adding an exploratory metric does not enlarge the declared primary family or reallocate its alpha.

    Assigned and triggered populations have separate family identities. Switching the visible population only changes the rows shown, not either population's allocation or membership. Concatenating readouts preserves each source's families; it does not create one jointly controlled experiment-wide family. Use `family_id` and the source/population scope, not a shared family name, to identify which guarantee applies.

    Under a registered sequential plan the triggered secondary family is selected by e-BH on its own chain with the assigned family's guarantee and status labels, conditional on arm-independent triggering, no pre-trigger treatment effect, trigger-order stationarity, and looks/stops/freezes adapted to the joint triggered filtration.

    An affirmative selection is still not automatically confirmatory. A fixed Benjamini-Hochberg exploratory family retains its q level, whole-family size, FCR-adjusted selected intervals, and the independence/positive-dependence qualification for its FDR statement; it is shown as exploratory, not as an unadjusted result or a confirmatory discovery. A triggered sequential family's guarantee labels state conditional control, not verification of the analyst's look, stopping or freezing rule. Filtering a result view does not shrink either family or remove its partial-view indication.



??? info "What the secondary q level means"
    Testing several metrics at once inflates the chance that at least one looks significant by pure noise. Increment's `AnalysisPlan` separates metrics into roles with different multiplicity treatment: a `primary` takes a share `alpha / n_primaries` of the familywise `alpha`, split again across that metric's own non-control arms at estimation; a `guardrail` tests unsplit, at the full `alpha`, since it must not move adversely; a `secondaries` family is judged as a false-discovery-rate problem at level `q` instead of a per-metric significance test.

    `q=0.10` is a false-discovery level, not a second `alpha` or a quantile. It bounds the expected fraction of *discovered* secondaries that are false positives, not each metric's type-I rate. A `primary` row always has `discovery=None`: family selection applies only to secondaries and breakouts.

## `correction` and `MultiplicitySpec` for segmented views

Use `AnalysisPlan.view_multiplicity` to choose the correction for segmented breakout and as-of views, separately from the whole-experiment secondary family.

Leaving `view_multiplicity=None` keeps the per-route defaults: randomized breakout BH at `q` (fixed-roster Bonferroni for a sequential `AsymptoticMean` registration), encouragement breakout uncorrected, and as-of uncorrected unless segmented.

!!! warning "Choose a supported view correction"
    `q` is accepted only with `correction="bh"`; setting it with `"bonferroni"` or `"none"` is refused. Sequential dataframe breakouts require a complete `InferenceSpec.segments` roster before outcomes. Asymptotic breakouts refuse BH. Encouragement cannot combine predeclared segments with `SequentialCompliancePolicy(family=True)`; use separately allocated uptake cells (`family=False`) or an unsegmented joint family.

??? info "View defaults and family construction"
    `AnalysisPlan.view_multiplicity: MultiplicitySpec | None` governs segmented breakout and as-of views specifically, independent of the top-level `secondaries`/`q` family above. `MultiplicitySpec(correction="bh", q=0.10)`, `MultiplicitySpec(correction="bonferroni")`, and `MultiplicitySpec(correction="none")` are the three options; `q` is only accepted under `correction="bh"` (and defaults to `0.10` there if omitted) -- setting `q` under `"bonferroni"` or `"none"` is refused.

    For randomized sequential dataframe breakouts, `InferenceSpec.segments` declares the complete level roster before outcomes; automatic registration keeps absent levels in the family. Exact e-BH uses the breakout view's `q`, including a `MultiplicitySpec` override, not a second family inferred from observed rows. Asymptotic breakouts retain fixed-roster Bonferroni and refuse BH. Exact multi-arm whole-window monitoring counts every secondary metric/arm cell in the family and splits each primary's allocation across its treatment arms. An encouragement plan cannot combine predeclared segments with `SequentialCompliancePolicy(family=True)`: its breakout policy is uncorrected. Use separately allocated uptake cells (`family=False`) or an unsegmented joint family; existing encouragement breakout-readout restrictions remain.

## e-BH under sequential inference

Sequential plans use e-BH to select secondaries. Declare the sampling and reveal rules before reading outcomes, as in this example.

!!! warning "The guarantee depends on the inputs"
    e-BH does not make an invalid sequential estimate valid. The asymptotic-mean route has an asymptotic, empirical guarantee, not a finite-sample one; exact `kind="always_valid"` is limited to supported conversion and retention routes.

??? info "e-BH construction"
    Under `AnalysisPlan(inference=InferenceSpec(kind="asymptotic_mean"))` or `InferenceSpec(kind="always_valid")`, the secondary family uses e-BH (Wang & Ramdas 2022: ordinary BH applied to `1/e`) instead of ordinary p-value BH. e-BH controls FDR at `<= q` under arbitrary dependence and stopping time, but only when every input is a valid e-process: an e-value that remains valid at every stopping time, not just at one fixed look. `e_bh_select` assumes that property; it does not establish it. The asymptotic-mean route builds its e-process from the same delta-method sampling distribution as the fixed-horizon interval, so its time-uniform guarantee is asymptotic and empirical, not finite-sample. The exact `kind="always_valid"` route, limited to conversion and retention metrics, is the finite-sample alternative.

```python
from increment import AnalysisPlan, InferenceSpec, Randomized

design = Randomized(
    control_group="control",
    allocation={"control": 0.5, "treatment": 0.5},
    allocation_scheme="independent",
)
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
print(av_results)
```

<!-- output -->
```text
assigned — treatment vs control
sessions: control → treatment; effect=12.9354%; interval=[7.06467, 19.1773]%; estimand=itt; stat_sig=True
sessions (primary, relative lift): 0.129354, set [0.0706467, 0.191773] at 0.95 confidence.
sessions interpretation: excludes zero; effect direction supported (increase).
sessions sequential ratio confidence set bounded: [1.07065, 1.19177].
sessions test alternative: two-sided; null (relative lift): 0.
sessions result identity: method unadjusted; method role decision; estimand itt.
signups: control → treatment; effect=-0.85373%; interval=[-3.00719, 1.34716]%; estimand=itt; stat_sig=False
signups (secondary, relative lift): -0.0085373, set [-0.0300719, 0.0134716] at 0.95 confidence.
signups interpretation: does not reject; the retained sequential procedure has not rejected.
signups sequential ratio confidence set bounded: [0.969928, 1.01347].
signups test alternative: two-sided; null (relative lift): 0.
signups result identity: method unadjusted; method role decision; estimand itt; not selected.
signups selection: not selected; separate from interval evidence.
Integrity: not_rejected — no detected mismatch; not proof of valid randomization.
Multiplicity (primary): declared_plan; family primary.
Multiplicity (secondary): declared_plan; family secondary.
```

!!! warning "Sequential dataframe inputs"
    A sequential `AnalysisPlan` needs a `design=` with an explicit `allocation` (not the dataframe shortcut `control=`), since the registered asymptotic-mean runtime must be declared before any outcome is read. A unit summary also names each unit's `exposure_date` (a date or day index); units are revealed in that order, never in row order. See [Sequential inference](sequential-inference.md) for the registration, capture and error-control mechanics this route shares with an ordinary (non-multiplicity) sequential primary.

## FCR re-estimation of discovered cells

A discovered secondary or breakout already carries a selection-adjusted interval. Do not treat it as the same width as an unselected row's interval.

For routes that filter estimation evidence to selected hypotheses, only selected
hypotheses are re-estimated. Quantile families are re-estimated without filtering
the unit frame to selected arms, so a stricter corrected level can make a
quantile estimate unavailable even for an unselected arm.

??? info "FCR interval construction"
    A secondary or breakout cell selected under BH (fixed-horizon p-values) or e-BH (sequential e-values) gets a reported interval re-estimated at `fcr_alpha = min(q * R / m, nominal_alpha)`. Here, `R` is the number of selected cells and `m` is the family size. The interval is wider than the nominal interval when few cells are selected from a large family, and narrower as more cells are selected, up to the nominal-alpha cap.

    Treat a discovered secondary's interval as already carrying this correction; it is not the same width as a non-discovered row's interval. `family_guarantee` reports `"finite_sample"` only when every family member is exact (`always_valid`) and `"asymptotic_sequential"` when any member is asymptotic.

## Caveats

!!! warning "Dependence and repeated discovery sets"
    Fixed-horizon BH assumes independence or positive dependence (PRDS) across the metric x arm x segment family, and this is not checked -- read [Fixed-horizon FDR control assumes a dependence condition that is not checked](../limitations.md#fixed-horizon-fdr-control-assumes-a-dependence-condition-that-is-not-checked). An always-valid discovery set's guarantee holds for the current date; the union across dates is not controlled -- read [Stopping-date guarantees require the registered reveal contract](../limitations.md#stopping-date-guarantees-require-the-registered-reveal-contract).
