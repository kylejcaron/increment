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
    `uptake.fact` must be declared on a source whose entities include the experiment's unit. A guardrail margin applies to ITT; a margin-less guardrail tests its adverse tail against zero. Tune the weak-instrument threshold and overlap policy through full `Encouragement` or `Observational` objects on dataframe constructors, not YAML.

    `Observational` can be declared in YAML and its adjustment covariates work on both warehouse and dataframe paths; see [observational.md](observational.md#from-definitions-the-warehouse-path). Five entry points support an `Encouragement` design: `from_unit_summary`, `from_unit_panel`, `from_definitions`, `from_unit_day_artifact`, and `from_moments`. They retain ITT, compliance, LATE, guardrails, and BH/e-BH-selected secondaries across artifact and moments round trips. `from_switchback_panel` accepts only `identification: Randomized`, never `Encouragement` or `Observational`.


## Read the estimands

- **ITT**: the effect of being encouraged, whether or not uptake changes.
- **Compliance**: how much encouragement changes treatment uptake.
- **LATE**: the effect of treatment on people who comply because of encouragement, not everyone.

!!! warning "When LATE is not available"
    LATE requires randomized assignment, a relevant first stage, the exclusion restriction, and monotonicity (no defiers). A weak first stage leaves a visible typed failure rather than a numeric estimate. Requesting LATE without declaring exclusion raises `identification.encouragement.exclusion_required`; sequential and per-day LATE are unsupported.

??? info "Statistical details"
    The registered uptake confidence sequence uses ratio coordinates. Exact endpoints and certificates are available in `sequential_result.bounds`. Secondary discovery is ITT-based: the selected LATE row uses the same corrected interval level as ITT, but only ITT reports `discovery`. For unadjusted independent-unit binary ITT, dense counts use the delta-method t reference and sparse counts use exact binomial inference; `conversion_inference="finite_sample"` selects the exact route at every size. Zero control conversions can leave the lift unavailable while retaining a confidence set.

!!! warning "Margins and compliance scope"
    A declared guardrail margin applies to its fixed-horizon, whole-window ITT row. LATE has no margin verdict; requesting `late` without `itt` for a margined metric refuses with `readout.encouragement.margin`. As-of (`readout.metric_declare_non`), breakout (`readout.margin.breakout`), and sequential (`plan.encouragement.margin`) margins are refused. Compliance cannot be segmented (`readout.run.segment_unsupported`). In sequential plans it is outside selection unless explicitly included by policy; triggered sequential compliance is unavailable.

??? info "Design and sequential details"
    Fixed-horizon design-level compliance uses the enrolled population and `uptake.window_days`, regardless of outcome missingness or metric count; it can run with no outcome catalog on all five supported design entry points. A compliance-only empty catalog needs an explicit policy or uptake registration. Sequential uptake-only monitoring uses `InferenceSpec(kind="always_valid")` and `SequentialCompliancePolicy`; it reads no outcome. A compliance-only registered checkpoint can retain an unbounded retention metric, but automatic registration includes that metric and refuses it with `sequential.route.unsupported`. Outcome-consuming requests such as ITT still refuse the unbounded metric.

    Unit-panel `missing="drop"` remains refused. Mixed ITT/LATE requests need outcome data; each metric's LATE uses its own outcome cohort. As-of compliance includes units assigned by the requested day and uptake through that day. Triggered compliance admits units by first eligible trigger day but keeps uptake windows anchored to assignment. Compliance-only requests skip outcome reductions but continue validating metric names and source/design state.

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

??? info "As-of, sequential, and artifact details"
    `from_definitions` and `from_unit_day_artifact` apply uptake windows to raw timestamps before day bucketing. A one-day window starting at noon includes uptake at exposure or 21 hours later, but not exactly 24 hours later. Later uptake remains visible in as-of readouts but does not add outcome-observation days or admit newer outcome cohorts. Completed outcome rows still need outcome data through their last required day; `from_unit_panel` takes day-reduced flags, so use elapsed exposure days when comparing paths.

    `completed_windows_only=True` includes only closed uptake windows. Sequential ITT and relative Bernoulli compliance require joint finalization and immutable-prefix proofs. Continuous ITT uses the asymptotic mean law; binary-uptake LATE is fixed-horizon. Sequential compliance requires an explicit `SequentialCompliancePolicy` matching registered alpha, direction, null, and family membership. By default, compliance is excluded from discovery; add it explicitly to the family. When included, joint continuous-ITT/Bernoulli-compliance selection uses e-BH, with `asymptotic_sequential` family guarantee when continuous ITT is present. Sequential as-of readouts need `completed_windows_only=True`.

    Triggered readouts restrict compliance to the triggered enrollment population. Open-ended compliance dates extend through the latest enrollment or uptake event, including repeated out-of-window uptake; outcome filters do not shorten the axis. An explicit observation horizon takes precedence. Custom `MomentSource` implementations must provide `compliance_dates()`; outcome dates are not a substitute.

    Clustered compliance is total uptake divided by total members, with uncertainty from cluster totals, not the average of cluster rates. Exports preserve compliance points and uncertainty and must be reloaded with the same `Encouragement` design and experiment identity. Current cubes use a version-1 `compliance_summary`; new artifacts also store assignment counts, cluster IDs when needed, and a version-2 uptake relation. Legacy artifacts without sufficient uptake state refuse with `source.compliance_summary.legacy_uptake_state`; malformed current state is rejected. Serialized summaries and as-of readouts preserve day-label identity.

## Run the example

See the [Encouragement notebook](../examples/encouragement.md) for a complete
simulation and rendered results. The [API reference](../api.md) documents
`Encouragement`, `UptakeSpec`, and `ExclusionRestriction`.

## References and assumptions

The LATE interpretation follows Imbens and Angrist, [“Identification and Estimation of Local Average Treatment Effects”](https://doi.org/10.2307/2951620). LATE requires random assignment, a relevant first stage, the exclusion restriction, and monotonicity; Increment reports diagnostics and refuses weak first stages but cannot validate these assumptions.
