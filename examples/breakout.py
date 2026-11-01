import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import coeftable as ct
    import ibis
    import marimo as mo
    from _seed import seed_event_log

    from increment import Method, Scope, from_definitions, run
    from increment.evidence import BreakoutIdentity
    from increment.tables import estimates_to_readout, readout_table
    from increment.workflows import experiment_definition

    return (
        BreakoutIdentity,
        Method,
        Scope,
        ct,
        estimates_to_readout,
        experiment_definition,
        from_definitions,
        ibis,
        mo,
        readout_table,
        run,
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
def _(DEFINITIONS, EXPERIMENT, con, from_definitions):
    experiment, resolver = from_definitions(DEFINITIONS, con, experiment_name=EXPERIMENT)
    return experiment, resolver


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
def _(
    BreakoutIdentity,
    Method,
    Scope,
    estimates_to_readout,
    experiment,
    readout_table,
    resolver,
    run,
):
    # new_onboarding_v2 declares Bonferroni segmented views in
    # examples/definitions/experiments.yaml; its plan keeps fixed-horizon
    # inference because the declared retention guardrail carries a prior.
    by_country = Scope(view="breakout", grain="total")
    asof_lift_by_country = Scope(
        view="asof", grain="asof", segment=BreakoutIdentity("country", None)
    )
    segment_result = run(
        experiment.with_views(by_country, asof_lift_by_country),
        resolver,
        decision_method=Method(name="unadjusted"),
    )
    breakout_table = readout_table(
        estimates_to_readout(segment_result.readout(by_country)),
        title="new_onboarding_v2 readout, by country",
        subtitle="Relative lift vs control, Bonferroni-adjusted intervals (95% family-wise coverage per metric)",
        nest_by="segment",
        show_interval_level=False,
        # visualize time series
        trend=segment_result.readout(asof_lift_by_country),
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
def _(Scope, ct, experiment, resolver, run):
    daily_scope = Scope(view="daily", grain="daily", values=True)
    daily_values = run(experiment.with_views(daily_scope), resolver).readout(daily_scope)
    daily_frame = daily_values.to_frame(backend="pandas")

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

    A daily values view with `segment=BreakoutIdentity("country", None)` breaks the view
    above out by `new_onboarding_v2`'s declared `country` breakout instead of
    collapsing across it
    """)
    return


@app.cell
def _(BreakoutIdentity, Scope, ct, experiment, experiment_definition, resolver, run):
    segment_daily_scope = Scope(
        view="daily", grain="daily", values=True, segment=BreakoutIdentity("country", None)
    )
    segment_daily_values = run(experiment.with_views(segment_daily_scope), resolver).readout(
        segment_daily_scope
    )
    segment_daily_frame = segment_daily_values.to_frame(backend="pandas")
    control_group = experiment_definition(experiment, resolver).control_group
    segment_daily_frame = segment_daily_frame[segment_daily_frame["group_id"] != control_group]

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
def _(Scope, ct, experiment, resolver, run):
    # its series starts late, absent until the first cohort matures.
    asof_scope = Scope(view="asof", grain="asof", values=True)
    asof_values = run(experiment.with_views(asof_scope), resolver).readout(asof_scope)
    asof_frame = asof_values.to_frame(backend="pandas")

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

    This view shows the independent daily lift each day. Note, this is different than the running lift (the as-of view, `Scope(view="asof", grain="asof")`), as this approach just looks at only data from a single day. It is particularly useful for identifying **novelty effects**
    """)
    return


@app.cell
def _(Method, Scope, estimates_to_readout, experiment, readout_table, resolver, run):
    whole_window = Scope(view="run", grain="total")
    daily_lift_scope = Scope(view="daily", grain="daily")
    lift_result = run(
        experiment.with_views(whole_window, daily_lift_scope),
        resolver,
        decision_method=Method(name="unadjusted"),
    )
    calendar_daily_lift = [
        e for e in lift_result.readout(daily_lift_scope) if e.ds_basis == "calendar"
    ]
    estimates = lift_result.readout(whole_window)
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

    The running-total counterpart to the section above: the as-of view (`Scope(view="asof", grain="asof")`) tells us "what was the experiment lift as of a certain date?"

    It is typically the best starting point for diving into an experiment
    """)
    return


@app.cell
def _(Method, Scope, estimates, estimates_to_readout, experiment, readout_table, resolver, run):
    asof_lift_scope = Scope(view="asof", grain="asof")
    readout_table(
        estimates_to_readout(estimates),
        title="new_onboarding_v2 readout, as-of trend",
        subtitle="Relative lift vs control, 95 % credible interval",
        trend=run(
            experiment.with_views(asof_lift_scope),
            resolver,
            decision_method=Method(name="unadjusted"),
        ).readout(asof_lift_scope),
        trend_label="As-of lift trend",
    )
    return


if __name__ == "__main__":
    app.run()
