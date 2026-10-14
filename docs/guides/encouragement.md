# Encouragement Designs

Use an encouragement design when random assignment changes the chance that people receive treatment, but does not force them to take it. Compare assignment effects directly; estimate effects among people whose uptake changed only when the causal assumptions are credible.

## Declare assignment and uptake

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(9)
n = 300
variant = np.array(["control"] * (n // 2) + ["encouraged"] * (n // 2))
rng.shuffle(variant)
clicked = np.where(variant == "encouraged", rng.random(n) < 0.6, False).astype(int)
encouragement_df = pd.DataFrame(
    {
        "user_id": [f"u{i:03d}" for i in range(n)],
        "variant": variant,
        "revenue": 8.0 + 3.0 * clicked + rng.normal(0, 2.0, n),
        "clicked": clicked,
    }
)

from increment import Analysis, Encouragement, ExclusionRestriction, UptakeSpec

design = Encouragement(
    control_group="control",
    uptake=UptakeSpec(fact="clicked"),
    exclusion_restriction=ExclusionRestriction(
        acknowledged=True,
        justification=("the randomized prompt affects revenue only through help-button clicks"),
    ),
    one_sided=False,
)

results = Analysis.from_unit_summary(
    encouragement_df,
    unit="user_id",
    group="variant",
    metrics={"revenue": "mean"},
    design=design,
).run()
```

LATE needs the exclusion restriction: assignment must affect the outcome only through treatment receipt.

ITT and compliance do not require exclusion. Omit it when requesting only those estimands:

```python
assignment_and_uptake = Analysis.from_unit_summary(
    encouragement_df,
    unit="user_id",
    group="variant",
    metrics={"revenue": "mean"},
    design=Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked"),
    ),
).run(estimands=("itt", "compliance"))
```

The default requests all three estimands. If exclusion is not declared, any request containing `late` raises `identification.encouragement.exclusion_required` before reading outcomes, even if a weak first stage would later make LATE unavailable. The same omission works in YAML and survives artifact and moments round trips. Supported sequential ITT/compliance monitoring also needs no exclusion declaration; declaring one does not make sequential LATE supported.

## Declare on `from_definitions`

The control arm and target allocation come from the experiment's own
`control_group`/`allocation`, exactly as a randomized experiment declares
them; `design:` adds only the mechanism-specific facts:

```yaml
experiments:
  - name: help_button_encouragement
    exposure: assigned_button
    unit: user_id
    start: 2026-04-01
    control_group: control
    plan:
      secondaries: [ticket_deflection]
    design:
      mechanism: encouragement
      uptake:
        fact: clicked_help
        window_days: 7
      one_sided: true
      exclusion_restriction:
        acknowledged: true
        justification: "the button can only move deflection by being clicked"
```

See the [capabilities by entry point](../reference/capabilities-by-entry-point.md#what-runs-where) table for supported constructors.

??? info "Warehouse declarations and supported paths"
    `uptake.fact` must name a fact declared on a source whose `entities` include the experiment's `unit`, validated at load like `exposure`/`trigger`. A guardrail's declared margin (`Metric.margin`/`margin_abs`, or a plan-bound `ExperimentMetric.margin`) applies to its ITT row -- the same non-inferiority construction a randomized design uses; a margin-less guardrail tests its adverse tail against zero. Statistical tuning (the first-stage weak-instrument threshold, overlap policy) is not a YAML key: declare a full `Encouragement`/`Observational` object from `Analysis.from_unit_summary`/`from_unit_panel` to override it.

    `Observational` parses the same way -- `design: {mechanism: observational, covariates: [{property: tenure_days, source: users}]}` -- and its adjustment covariates run from `from_definitions` identically to `from_unit_summary`; see [observational.md](observational.md#from-definitions-the-warehouse-path). Five entry points support an `Encouragement` design: `from_unit_summary`, `from_unit_panel`, `from_definitions`, `from_unit_day_artifact`, and `from_moments`. They retain ITT, compliance, LATE, guardrails, and BH/e-BH-selected secondaries across artifact and moments round trips. `from_switchback_panel` accepts only `identification: Randomized`, never `Encouragement` or `Observational`.


## Read the estimands

- **ITT**: the effect of being encouraged, whether or not uptake changes.
- **Compliance**: how much encouragement changes treatment uptake.
- **LATE**: the effect of treatment on people who comply because of encouragement, not everyone.

!!! warning "When LATE is not available"
    LATE requires randomized assignment, a relevant first stage, the exclusion restriction, and monotonicity (no defiers). A weak first stage leaves a visible typed failure rather than a numeric estimate. Requesting LATE without declaring exclusion raises `identification.encouragement.exclusion_required`; sequential and per-day LATE are unsupported.

??? info "Statistical details"
    LATE is ITT divided by the compliance first stage: the effect among compliers whose treatment receipt changed because of encouragement.

    The notebook's registered uptake confidence sequence is shown in **ratio coordinates**, not as a lift interval. Its default display keeps the bounds, confidence level, and geometry status concise; exact rational endpoints and the endpoint certificates remain available from the result's `sequential_result.bounds` when detailed numerical evidence is needed.

    Discovery and the BH/e-BH-corrected interval level are ITT-based: a selected secondary's LATE row is re-estimated at the same corrected level as its ITT row (both test the same null), but `discovery` itself is reported on the ITT row only -- LATE is a derived presentation, not a second hypothesis test.

    Unadjusted, independent-unit binary ITT takes the same count-routed conversion route as an ordinary arm pair even when the source also carries uptake moments: dense counts the delta-method route (`reference_kind="t"`), sparse counts the exact binomial reference (`"binomial"`; `Method(name="unadjusted", conversion_inference="finite_sample")` selects it at every size). This includes conversion breakouts from `from_definitions`, `from_unit_day_artifact`, `from_unit_panel`, and `from_unit_summary`: zero control conversions can leave the point lift unavailable while retaining a confidence set. Summary sources can therefore carry a declared breakout dimension.

    Compliance rows diagnose the first stage. Fixed-horizon compliance rows have `discovery is None`. Registered sequential compliance also stays outside selection unless its own compiled policy explicitly includes it in the retained family. A guardrail reports the same estimands as every other metric on every constructor, and `run(estimands=...)` returns exactly the estimands requested. LATE carries no margin verdict for a margined guardrail -- a plain interval, same construction as an unmargined LATE row. Design-level compliance alone does not apply an outcome margin.

!!! warning "Margins and compliance scope"
    A declared guardrail margin applies to its fixed-horizon, whole-window ITT row. LATE has no margin verdict; requesting `late` without `itt` for a margined metric refuses with `readout.encouragement.margin`. As-of (`readout.metric_declare_non`), breakout (`readout.margin.breakout`), and sequential (`plan.encouragement.margin`) margins are refused. Compliance cannot be segmented (`readout.run.segment_unsupported`). In sequential plans it is outside selection unless explicitly included by policy; triggered sequential compliance is unavailable.

For fixed-horizon uptake alone, use `run(estimands=["compliance"])`; no outcome catalog or sequential policy is needed. Pass `metrics=[]` to dataframe and portable constructors when there are no outcomes.

!!! warning "Outcome data and sequential monitoring"
    The existing unit-panel refusal of `missing="drop"` remains; do not replace undefined outcomes with zeros to bypass it. Mixed ITT/LATE requests still require their outcome data.

    For uptake-only sequential monitoring, use `InferenceSpec(kind="always_valid")` and an explicit `SequentialCompliancePolicy`; no scalar outcome law is needed. An empty catalog alone does not enable monitoring: without a compliance policy or an uptake registration it is refused. Request `estimands=("compliance",)`; an `("itt",)` request contains no outcome rows to scope, and the default request also includes unsupported LATE. Outcome-consuming requests still refuse an unbounded retention metric (`sequential.route.unsupported`).

    A nonempty catalog with unbounded retention needs explicit uptake-only registration; automatic registration would monitor and refuse that metric. Only `from_definitions`, `from_unit_day_artifact` (metric declared on the experiment), and `from_moments` (metric in `metrics=`) can retain it; dataframe constructors cannot declare it.

??? info "Design and sequential details"
    Design-level compliance uses the enrolled population and the design's own `uptake.window_days`, independently of outcome missingness, metric order, or metric count. A fixed-horizon `run(estimands=["compliance"])` needs no available outcome arm: `from_definitions`, `from_unit_day_artifact`, `from_unit_summary` and `from_moments` agree even when treatment outcomes are entirely missing. That request skips outcome compatibility checks and reductions, including for a declared triggered population; metric names and source/design state remain validated.

    With no outcome catalog, fixed-horizon compliance also works on `from_definitions`, `from_unit_day_artifact`, `from_unit_summary`, `from_unit_panel`, and `from_moments`. Pass `metrics=[]` to the dataframe and portable constructors; only assignment and uptake inputs are needed. Exports retain the complete design-level state without inventing an outcome row.

    The same five constructors support the empty catalog, with `from_moments` replaying the retained checkpoint. A registered compliance-only checkpoint reads uptake and no outcome, so an unbounded retention metric may stay in the catalog when the explicit uptake-only registration is declared over that catalog (an automatic registration would monitor the metric and refuse it, `sequential.route.unsupported`). `from_definitions`, `from_unit_day_artifact` (with the metric declared on the experiment) and `from_moments` (with the metric in `metrics=`) retain it; the dataframe constructors cannot declare it. Requests that consume outcomes, such as ITT, still refuse the retention metric. The empty-catalog path above does not exercise a nonempty catalog.

    Each metric's LATE uses the first stage on its matching outcome cohort. As-of compliance includes units assigned by the requested day and qualifying uptake through that day; a triggered-population compliance series instead admits units by their first eligible trigger day while retaining assignment-anchored uptake windows.

    In fixed-horizon `run()`, a weak first stage for one metric's outcome cohort suppresses that cohort's LATE estimate but retains its planned LATE row as a typed failure (`failure_code="estimation.encouragement.late.weak_first_stage"` and `lift=None`). Other metrics use their own outcome cohorts and can still report LATE when their first stages pass.

!!! warning "Daily estimates"
    Daily encouragement uptake is a first-stage rate; per-day LATE is unsupported. Request only an estimand the daily view supports:

```python
import datetime as dt
from pathlib import Path

import ibis
import pyarrow as pa

from increment import Analysis, SourceSnapshotEvidence

definitions = Path("encouragement.yml")
definitions.write_text(
    """
dialect: duckdb
fact_sources:
  - name: events
    sql: SELECT * FROM events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: revenue
        column: value
      - name: enrolled
        column: null
      - name: saw_surface
        column: null
      - name: clicked
        column: null
exposures:
  - name: assignment
    fact: enrolled
  - name: saw_surface
    fact: saw_surface
metrics:
  - name: revenue
    type: mean
    entity: user_id
    fact: revenue
    window_days: 3
    aggregation: sum
experiments:
  - name: encouragement_example
    exposure: assignment
    trigger: saw_surface
    unit: user_id
    start: 2026-01-01T00:00:00
    end: 2026-01-03
    observation_end: 2026-01-04
    control_group: control
    plan:
      secondaries: [revenue]
    design:
      mechanism: encouragement
      uptake:
        fact: clicked
    """
)
con = ibis.duckdb.connect()
con.create_table(
    "events",
    obj=pa.table(
        {
            "experiment_id": ["encouragement_example"] * 8,
            "user_id": ["c1", "t1", "c1", "t1", "c1", "t1", "c1", "t1"],
            "group_id": ["control", "encouraged"] * 4,
            "ts": [
                dt.datetime(2026, 1, 1),
                dt.datetime(2026, 1, 1),
                dt.datetime(2026, 1, 2),
                dt.datetime(2026, 1, 2),
                dt.datetime(2026, 1, 2),
                dt.datetime(2026, 1, 2),
                dt.datetime(2026, 1, 3),
                dt.datetime(2026, 1, 3),
            ],
            "event": [
                "enrolled",
                "enrolled",
                "saw_surface",
                "saw_surface",
                "clicked",
                "clicked",
                "revenue",
                "revenue",
            ],
            "value": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0],
        }
    ),
)
analysis = Analysis.from_definitions(
    "encouragement_example",
    definitions,
    con,
    source_snapshot_evidence=SourceSnapshotEvidence(
        dt.datetime(2026, 1, 4, tzinfo=dt.UTC),
        {"events": dt.datetime(2026, 1, 4, tzinfo=dt.UTC)},
    ),
)
analysis.run_daily_lift(
    estimands=("compliance",),
    population="triggered",
)
analysis.close()
con.disconnect()
```

!!! warning "Outcome and compliance caveats"
    Triggered sequential compliance is unavailable (`triggered_uptake_unsupported`), and clustered daily outcome histories are unavailable. A weak first stage leaves a typed LATE failure (`estimation.encouragement.late.weak_first_stage`); it is never silently dropped.

!!! warning "Sequential request requirements"
    Registered sequential ITT and relative Bernoulli compliance additionally require joint finalization and immutable prefix proofs; see [Sequential inference](sequential-inference.md). Sequential compliance additionally needs an explicit `SequentialCompliancePolicy` on the plan, matching its registered alpha, direction, null and family membership. The default policy excludes compliance from discovery selection; a registered family must explicitly include it. For sequential as-of readouts, pass `completed_windows_only=True`.

??? info "As-of, sequential, and artifact details"
    `from_definitions` and `from_unit_day_artifact` apply uptake windows to raw timestamps before day bucketing. For a one-day window starting at noon, uptake at exposure or 21 hours later qualifies; uptake exactly 24 hours later does not. Qualifying uptake after the last outcome day remains visible in as-of readouts. Later uptake alone does not add outcome-observation days, change calendar-day averages, or admit newer outcome cohorts. Completed outcome rows still require outcome data through their last required day. A `from_unit_panel` input carries day-reduced flags, not raw event timestamps: use elapsed exposure days when reducing the same events for cross-path comparison.

    With `completed_windows_only=True`, compliance includes only units whose uptake window has closed. Continuous intent-to-treat alone monitors under the public asymptotic mean law (the same construction a randomized design uses): assignment to encouragement is itself randomized, so the confidence-sequence assumptions only need the allocation, not the mechanism label. Binary-uptake LATE remains fixed-horizon because it is not a bivariate Gaussian observation model. LATE suppression diagnostics use the affected outcome's first stage, even when design-level compliance gives a different weak/strong classification.

    When compliance joins the family, the joint continuous-ITT/Bernoulli-compliance roster is selected by the same e-BH rule as any other in-family sequential secondary -- exact for the Bernoulli compliance member, asymptotic for the continuous ITT member -- with one shared `family_guarantee` (`asymptotic_sequential` whenever the continuous member is present) reported on every row; see [Sequential inference](sequential-inference.md).

    Triggered readouts restrict compliance to the triggered enrollment population. Frame as-of summaries retain the panel's calendar, numeric, or structured string day labels and use the same chronological ordering as the metric moments. For open-ended native experiments, compliance dates extend through the latest enrollment or post-enrollment uptake event, including repeats outside the uptake window. Outcome filters and missing values do not shorten this axis. An explicit observation horizon takes precedence. Custom `MomentSource` implementations must provide `compliance_dates()` from enrollment/uptake state; sources without as-of compliance support must raise a coded capability refusal. Outcome dates are not a substitute for this axis.

    For clustered assignment, compliance is the ratio of total uptake to total members, with uncertainty from the centered cluster uptake-total/size moments. It does not average cluster uptake rates. `ComplianceArm` and `ComplianceSummary` in `increment.sources` retain immutable sufficient state and separate member counts from independent cluster counts.

    Native, frame, and artifact `export()` preserve compliance points and uncertainty. Reload with the same `Encouragement` design; omit `experiment_id` to recover the exported identity, or supply that same identity explicitly.

    Current cubes carry a version-1 `compliance_summary` payload on every row. New encouragement artifacts include assignment counts, cluster identities when needed, and a version-2 uptake relation with the first qualifying timestamp. The uptake extension also persists its enrollment/uptake observation edge from the publication snapshot. Timestamp-bearing artifacts without that field use enrollment and first qualifying uptake dates as a conservative coverage bound. Artifacts with only a Boolean uptake flag, and cubes without the payload, raise `source.compliance_summary.legacy_uptake_state` with a re-export remedy.

    Malformed current state is rejected separately, including disagreement between compliance member counts and the cube's assignment unit counts. Mixed and unassigned accounting entries are excluded from that comparison. Standalone summary serialization retains ISO calendar dates, numeric values, and tagged string or datetime values so day-axis identity survives a round trip. As-of lift results also preserve these labels through JSON serialization.

## Run the example

See the [Encouragement notebook](../examples/encouragement.md) for a complete
simulation and rendered results. The [API reference](../api.md) documents
`Encouragement`, `UptakeSpec`, and `ExclusionRestriction`.

## References and assumptions

The LATE interpretation follows Imbens and Angrist, [“Identification and Estimation of Local Average Treatment Effects”](https://doi.org/10.2307/2951620). LATE requires random assignment, a relevant first stage, the exclusion restriction, and monotonicity; Increment reports diagnostics and refuses weak first stages but cannot validate these assumptions.
