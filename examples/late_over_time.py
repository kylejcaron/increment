import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    from increment import (
        Analysis,
        AnalysisPlan,
        Encouragement,
        ExclusionRestriction,
        InferenceSpec,
        JointReveal,
        MetricSpec,
        PredictivePrior,
        SequentialCell,
        SequentialCompliancePolicy,
        SequentialModel,
        SequentialRegistration,
        UptakeSpec,
        sequential_definition_id,
    )
    from increment.frame import synthesise_metric
    from increment.sequential_source import frame_observation_mapping
    from increment.simulate.dgp import Scenario, simulate_raw_logs

    return (
        Analysis,
        AnalysisPlan,
        Encouragement,
        ExclusionRestriction,
        InferenceSpec,
        JointReveal,
        MetricSpec,
        PredictivePrior,
        Scenario,
        SequentialCell,
        SequentialCompliancePolicy,
        SequentialModel,
        SequentialRegistration,
        UptakeSpec,
        frame_observation_mapping,
        mo,
        np,
        pd,
        sequential_definition_id,
        simulate_raw_logs,
        synthesise_metric,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # LATE over time

    A re-engagement prompt is randomized once, but real-world enrollment
    staggers exposure over the following 20 days -- each unit sees the
    prompt on its own day. From there, people click it -- and their
    revenue outcome resolves -- on different days over the following two
    weeks, measured relative to each unit's own exposure. This notebook
    watches the LATE (the effect of clicking, for compliers) mature as
    that window fills in, and shows why the per-day view Increment offers
    everywhere else is refused for an encouragement design.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Simulate a staggered encouragement panel

    `increment.simulate.dgp.simulate_raw_logs` draws a raw event log: one
    `exposure` row per unit, plus `uptake`/`outcome` rows at a random day
    offset in `[0, n_days)` for each unit. `W` bounds both the outcome
    window and the uptake window, so every simulated event lands strictly
    inside the window the analysis later gates on.
    """)
    return


@app.cell
def _(Scenario, simulate_raw_logs):
    W = 10  # outcome/uptake window, in days since each unit's own exposure
    scenario = Scenario(
        n_units=900,
        n_days=W,
        true_lift={},
        uptake_compliance_t=0.55,
        uptake_compliance_c=0.10,
        tau_complier=15.0,
        seed=7,
    )
    raw = simulate_raw_logs(scenario).to_pandas()
    return W, raw


@app.cell
def _(np, pd, raw):
    start = pd.Timestamp("2025-02-01").normalize()

    exposures = (
        raw[raw["event"] == "exposure"][["unit_id", "group_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    # Stagger real-world enrollment over 20 calendar days, independent of the
    # DGP's relative day-offsets (always measured from each unit's own exposure).
    enroll_offsets = np.arange(len(exposures)) % 20
    exposures["exposed_on"] = [start + pd.Timedelta(days=int(o)) for o in enroll_offsets]
    return (exposures,)


@app.cell
def _(exposures, pd, raw):
    outcome = raw[raw["event"] == "outcome"][["unit_id", "ts", "value"]].rename(
        columns={"value": "revenue"}
    )
    outcome["rel_day"] = (outcome["ts"] - pd.Timestamp("2025-01-01")).dt.days
    uptake = raw[raw["event"] == "uptake"][["unit_id", "ts"]]
    uptake["rel_day"] = (uptake["ts"] - pd.Timestamp("2025-01-01")).dt.days
    uptake = uptake[["unit_id", "rel_day"]].assign(clicked=1.0)

    outcome_rows = exposures.merge(
        outcome[["unit_id", "rel_day", "revenue"]], on="unit_id", how="left"
    )
    outcome_rows["ds"] = (
        outcome_rows["exposed_on"] + pd.to_timedelta(outcome_rows["rel_day"], unit="D")
    ).dt.date
    outcome_rows["clicked"] = 0.0

    uptake_rows = exposures.merge(uptake, on="unit_id", how="inner")
    uptake_rows["ds"] = (
        uptake_rows["exposed_on"] + pd.to_timedelta(uptake_rows["rel_day"], unit="D")
    ).dt.date
    uptake_rows["revenue"] = 0.0
    return outcome_rows, uptake_rows


@app.cell
def _(W, exposures, outcome_rows, pd, uptake_rows):
    completion_rows = exposures.assign(revenue=0.0, clicked=0.0)
    completion_rows["ds"] = (completion_rows["exposed_on"] + pd.Timedelta(days=W)).dt.date

    cols = ["unit_id", "group_id", "ds", "exposed_on", "revenue", "clicked"]
    frames = []
    for source_frame in (outcome_rows, uptake_rows, completion_rows):
        source_frame = source_frame.copy()
        source_frame["exposed_on"] = source_frame["exposed_on"].dt.date
        frames.append(source_frame[cols])
    # from_unit_panel expects at most one row per (unit, day): a unit whose
    # outcome and uptake land on the same day merges here before it is seen.
    panel = (
        pd.concat(frames, ignore_index=True)
        .groupby(["unit_id", "group_id", "ds", "exposed_on"], as_index=False)[
            ["revenue", "clicked"]
        ]
        .sum()
    )
    return completion_rows, panel


@app.cell
def _(mo):
    mo.md(r"""
    ## Declare the encouragement design and build the analysis
    """)
    return


@app.cell
def _(Encouragement, ExclusionRestriction, UptakeSpec, W):
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked", window_days=W),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True,
            justification="the randomized prompt affects revenue only through the click.",
        ),
        one_sided=False,
    )
    return (design,)


@app.cell
def _(Analysis, MetricSpec, W, design, panel):
    metrics = [MetricSpec(name="revenue", type="mean", window_days=W)]
    analysis = Analysis.from_unit_panel(
        panel,
        unit="unit_id",
        group="group_id",
        date="ds",
        metrics=metrics,
        design=design,
        exposure_date="exposed_on",
        experiment_id="reengagement_email",
    )
    return analysis, metrics


@app.cell
def _(mo):
    mo.md(r"""
    ## `run_daily_lift` refuses under encouragement

    A single day's clicks are too small a first stage to divide by; the
    per-day LATE it would imply is not statistically meaningful. Increment
    refuses the call by name instead of returning a noisy number.
    """)
    return


@app.cell
def _(analysis):
    try:
        analysis.run_daily_lift()
        daily_lift_message = "no error raised"
    except ValueError as daily_lift_error:
        daily_lift_message = str(daily_lift_error)
    daily_lift_message
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The as-of LATE trend

    `run_asof_lift(completed_windows_only=True)` admits each unit only once
    both its outcome and uptake windows have closed, so every plotted date
    reports the LATE over units mature enough to trust.
    """)
    return


@app.cell
def _(analysis):
    def _late_trend(results):
        return [r for r in results if r.estimand == "late" and r.value_scale == "relative"]

    fixed_late_trend = _late_trend(analysis.run_asof_lift(completed_windows_only=True))
    return (fixed_late_trend,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Monitoring relative uptake instead: an always-valid compliance trend

    LATE itself is a Wald ratio of two Gaussian-referenced ITTs, so it has
    no registered exact likelihood to monitor against. Binary uptake does:
    declaring a `SequentialRegistration` with a Bernoulli/Beta model for
    the `uptake` observable, and reading `estimand="compliance"` off the
    resulting rows, gives a relative-uptake series that stays valid no
    matter how many of these dates are actually inspected -- a genuine
    always-valid complement to the fixed-horizon LATE trend above, not a
    sequential LATE itself. `InferenceSpec(kind="always_valid",
    registration=...)` is declared on the source's `AnalysisPlan` before
    any outcome is read; always-valid inference is read from that
    declaration, never passed at call time.
    """)
    return


@app.cell
def _(
    Analysis,
    AnalysisPlan,
    InferenceSpec,
    JointReveal,
    PredictivePrior,
    SequentialCell,
    SequentialCompliancePolicy,
    SequentialModel,
    SequentialRegistration,
    W,
    completion_rows,
    design,
    frame_observation_mapping,
    metrics,
    panel,
    sequential_definition_id,
    synthesise_metric,
):
    _prior = PredictivePrior(kind="beta", a=1, b=1)
    registration = SequentialRegistration(
        source_id="reengagement_email",
        control_group="control",
        committed_before_data=True,
        definitions_id=sequential_definition_id(
            [synthesise_metric(spec) for spec in metrics],
            design,
            transformations=metrics,
            source_mapping=frame_observation_mapping(
                unit="unit_id",
                group="group_id",
                date="ds",
                exposure_date="exposed_on",
                uptake="clicked",
            ),
        ),
        reveal=JointReveal(
            filtration_id="reengagement-finalized-units",
            independent_unit_vectors=True,
            simultaneous_metrics=True,
            outcome_independent_order=True,
            immutable_finalized_outcomes=True,
            longest_window_days=W,
        ),
        models=(
            SequentialModel(
                metric="uptake",
                observable="uptake",
                law="bernoulli",
                control_prior=_prior,
                treatment_prior=_prior,
                positive_population_control=True,
            ),
        ),
        roster=(SequentialCell(metric="uptake", group_id="treatment", estimand="compliance"),),
    )
    av_plan = AnalysisPlan(
        inference=InferenceSpec(kind="always_valid", registration=registration),
        compliance=SequentialCompliancePolicy(alpha=registration.roster[0].alpha),
    )
    av_analysis = Analysis.from_unit_panel(
        panel,
        unit="unit_id",
        group="group_id",
        date="ds",
        metrics=metrics,
        design=design,
        plan=av_plan,
        exposure_date="exposed_on",
        experiment_id="reengagement_email",
    )
    uptake_trend = []
    for _day in sorted(set(completion_rows["ds"])):
        av_analysis.capture_sequential(finalized=True, as_of=_day)
        uptake_trend.extend(
            av_analysis.run_asof_lift(estimands=("compliance",), completed_windows_only=True)
        )
    return (uptake_trend,)


@app.cell(hide_code=True)
def _(fixed_late_trend, pd):
    pd.DataFrame(
        {
            "ds": r.ds,
            "late_lift": f"{r.lift.value:+.2%}",
            "lb": f"{r.lift.lb:+.2%}",
            "ub": f"{r.lift.ub:+.2%}",
            "inference": r.inference,
        }
        for r in fixed_late_trend
    )
    return


@app.cell(hide_code=True)
def _(uptake_trend, pd):
    pd.DataFrame(
        {
            "ds": r.ds,
            "compliance_lift": f"{r.lift.value:+.2%}" if r.lift is not None else None,
            "inference": r.inference,
        }
        for r in uptake_trend
    )
    return


@app.cell(hide_code=True)
def _(fixed_late_trend, mo, uptake_trend):
    _fixed_last = fixed_late_trend[-1]
    _uptake_last = uptake_trend[-1]

    mo.md(f"""
    - **Fixed-horizon LATE (last date):** [{_fixed_last.lift.lb:+.1%}, {_fixed_last.lift.ub:+.1%}] -- repeated-look warning
    - **Always-valid compliance (last date):** {_uptake_last.inference} inference, safe to inspect on any date

    The two series answer related but distinct questions: the fixed-horizon
    series is the LATE trend itself, valid at the one date you commit to in
    advance; the always-valid series is relative uptake, the LATE's own
    first stage, and stays valid under unlimited peeking.
    """)
    return


if __name__ == "__main__":
    app.run()
