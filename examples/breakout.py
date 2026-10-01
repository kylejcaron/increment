import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import coeftable as ct
    import ibis
    import marimo as mo
    from _seed import seed_event_log

    from increment import Analysis
    from increment.tables import estimates_to_readout, readout_table

    return (
        Analysis,
        ct,
        estimates_to_readout,
        ibis,
        mo,
        readout_table,
        seed_event_log,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Breakouts: per-segment lifts and metric trends over time
    """)
    return


@app.cell
def _():
    DEFINITIONS = "examples/definitions/"
    EXPERIMENT = "new_onboarding_v2"
    return DEFINITIONS, EXPERIMENT


@app.cell
def _(ibis, seed_event_log):
    con = ibis.duckdb.connect()
    # Keep this documentation example compact while retaining every displayed
    # daily, as-of, and country-breakout path. The generator default remains
    # 4,000 units for production-like examples; this fixed seed and 800-unit
    # slice are for informative API/figure rendering, not power claims.
    seed_event_log(con, n_units=800, seed=2025)
    return (con,)


@app.cell
def _(Analysis, DEFINITIONS, EXPERIMENT, con):
    analysis = Analysis(experiment_name=EXPERIMENT, definitions_path=DEFINITIONS, con=con)
    return (analysis,)


@app.cell
def _(mo):
    mo.md("""
    # Segment and Date Breakout

    While its typically ideal to have a well planned and well powered experiment with a set duration and minimal peeking, lifes not always so simple. There could be issues with a hashing algorithm, interactions with another experiment or a bug, novelty effects, segment level effects (maybe the feature doesnt render well on mobile), or even just a stakeholder who wants to dive deeper.

    This notebook highlights metric breakouts for daily, segment, and daily-segment level views. It uses a compact deterministic 800-unit seed so every view stays quick and informative; the generated data is for demonstrating the APIs and figures, not for making a power or significance claim.

    ## Per-segment lift: does the effect vary by country?

    `new_onboarding_v2` declares one breakout, `country`
    (`examples/definitions/experiments.yaml`).
    """)
    return


@app.cell
def _(analysis, estimates_to_readout, readout_table):
    # new_onboarding_v2 declares Bonferroni segmented views in
    # examples/definitions/experiments.yaml; its plan keeps fixed-horizon
    # inference because the declared retention guardrail carries a prior.
    breakout_estimates = analysis.run_breakout()
    breakout_table = readout_table(
        estimates_to_readout(breakout_estimates),
        title="new_onboarding_v2 readout, by country",
        subtitle="Relative lift vs control, Bonferroni-adjusted intervals (95% family-wise coverage per metric)",
        nest_by="segment",
        show_interval_level=False,
        # visualize time series
        trend=analysis.run_asof_lift(dimension="country"),
        trend_label="As-of lift trend",
    )
    breakout_table
    return


@app.cell
def _(mo):
    mo.md("""
    ## Metric values over time

    The section above focused on the running lift. Below, we're focusing on the actual metric values over time
    """)
    return


@app.cell
def _(analysis, ct):
    daily_values = analysis.run_daily()
    daily_frame = daily_values.to_frame()

    value_table = (
        ct.CoefTable(daily_frame[["metric"]].drop_duplicates(), rows="metric")
        .sparkline(
            "Value over time",
            value="value",
            ci=("lb", "ub"),
            x="ds",
            data=daily_frame,
            ref=None,
            series="group_id",
            show_ribbon=True,  # 2 overlapping arms at low opacity stays legible
            axis_fmt=ct.DateAxis(),
        )
        .header("Metric values over time", "One line per arm, per metric")
    )
    value_table
    return


@app.cell
def _(mo):
    mo.md("""
    ## Metric values over time, by segment

    `Analysis.run_daily(dimension="country")` breaks the view
    above out by `new_onboarding_v2`'s declared `country` breakout instead of
    collapsing across it
    """)
    return


@app.cell
def _(analysis, ct):
    segment_daily_values = analysis.run_daily(dimension="country")
    segment_daily_frame = segment_daily_values.to_frame()
    segment_daily_frame = segment_daily_frame[
        segment_daily_frame["group_id"] != analysis.experiment.control_group
    ]

    segment_value_table = (
        ct.CoefTable(segment_daily_frame[["metric"]].drop_duplicates(), rows="metric")
        .sparkline(
            "Value by segment (treatment arm)",
            value="value",
            ci=("lb", "ub"),
            x="ds",
            data=segment_daily_frame,
            ref=None,
            series="dimension_value",
            axis_fmt=ct.DateAxis(),
        )
        .header(
            "Metric values over time, by segment",
            "Treatment arm only, one line per country",
        )
    )
    segment_value_table
    return


@app.cell
def _(mo):
    mo.md("""
    ## As-of metric values over time

    This view shows the same per-arm credible-interval band as the section above, but each
    day's value is a RUNNING TOTAL "as of day N" instead of that day's independent snapshot.

    This is  particularly useful to see how a metric within the experiment is changing over time based on time, enrollment, or window dynamics.
    """)
    return


@app.cell
def _(analysis, ct):
    # its series starts late, absent until the first cohort matures.
    asof_values = analysis.run_asof()
    asof_frame = asof_values.to_frame()

    asof_table = (
        ct.CoefTable(asof_frame[["metric"]].drop_duplicates(), rows="metric")
        .sparkline(
            "As-of value over time",
            value="value",
            ci=("lb", "ub"),
            x="ds",
            data=asof_frame,
            ref=None,
            series="group_id",
            show_ribbon=True,  # 2 overlapping arms at low opacity stays legible
            axis_fmt=ct.DateAxis(),
        )
        .header("As-of metric values over time", "One line per arm, per metric, running total")
    )
    asof_table
    return


@app.cell
def _(mo):
    mo.md("""
    ## Lift over time

    This view shows the independent daily lift each day. Note, this is different than the running lift (`analysis.run_asof_lift()`), as this approach just looks at only data from a single day. It is particularly useful for identifying **novelty effects**
    """)
    return


@app.cell
def _(analysis, estimates_to_readout, readout_table):
    daily_lift = analysis.run_daily_lift()
    calendar_daily_lift = [e for e in daily_lift if e.ds_basis == "calendar"]
    estimates = analysis.run()
    lift_table = readout_table(
        estimates_to_readout(estimates),
        title="new_onboarding_v2 readout",
        subtitle="Relative lift vs control, 95 % credible interval",
        trend=calendar_daily_lift,
        trend_label="Daily lift",
    )
    lift_table
    return estimates


@app.cell
def _(mo):
    mo.md("""
    ## Running lift over time

    The running-total counterpart to the section above: `Analysis.run_asof_lift()` tells us "what was the experiment lift as of a certain date?"

    It is typically the best starting point for diving into an experiment
    """)
    return


@app.cell
def _(analysis, estimates_to_readout, readout_table, estimates):
    readout_table(
        estimates_to_readout(estimates),
        title="new_onboarding_v2 readout, as-of trend",
        subtitle="Relative lift vs control, 95 % credible interval",
        trend=analysis.run_asof_lift(),
        trend_label="As-of lift trend",
    )
    return


if __name__ == "__main__":
    app.run()
