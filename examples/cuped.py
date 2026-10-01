import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import ibis
    import marimo as mo
    from _seed import seed_event_log

    from increment import Analysis, Method
    from increment.results import LiftEstimate
    from increment.tables import estimates_to_readout, readout_table

    return (
        Analysis,
        LiftEstimate,
        Method,
        estimates_to_readout,
        ibis,
        mo,
        readout_table,
        seed_event_log,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # CUPED variance reduction

    CUPED uses pre-exposure activity as a covariate to reduce outcome variance.
    This notebook runs the same experiment unadjusted and CUPED-adjusted, then
    compares the estimates in one coeftable. The gain is clearest for continuous
    metrics; binary metrics usually have less variance to remove.

    The method is derived in [`docs/guides/cuped.md`](../docs/guides/cuped.md).
    """)
    return


@app.cell
def _(Method):
    DEFINITIONS = "examples/definitions/"
    EXPERIMENT = "new_onboarding_v2"

    UNADJUSTED = Method(name="unadjusted")
    CUPED = Method(name="cuped", variance_reduction="cuped")

    # Put the continuous metric first: it shows the clearest variance reduction.
    METRIC_ORDER = ("avg_session_duration", "purchase_rate", "d7_retention")
    return CUPED, DEFINITIONS, EXPERIMENT, METRIC_ORDER, UNADJUSTED


@app.cell(hide_code=True)
def _(LiftEstimate):
    def interval_width(estimate: LiftEstimate) -> float:
        """Width of a lift interval.  ``lb``/``ub`` are always populated here."""
        lift = estimate.lift
        assert lift.lb is not None and lift.ub is not None, "run() always returns an interval"
        return lift.ub - lift.lb

    return (interval_width,)


@app.cell
def _(Analysis, DEFINITIONS, EXPERIMENT, ibis, seed_event_log):
    con = ibis.duckdb.connect()

    # generate events
    seed_event_log(con, with_pre_period=True)

    analysis = Analysis(experiment_name=EXPERIMENT, definitions_path=DEFINITIONS, con=con)
    return (analysis,)


@app.cell
def _(CUPED, UNADJUSTED, analysis):
    # One run computes both methods from the same queried moments.
    estimates = analysis.run(decision_method=UNADJUSTED, sensitivity_methods=(CUPED,))

    by_method = {(est.metric, est.group_id, est.method): est for est in estimates}
    arms = sorted({est.group_id for est in estimates})
    return arms, by_method, estimates


@app.cell(hide_code=True)
def _(CUPED, METRIC_ORDER, UNADJUSTED, arms, by_method, interval_width, mo):
    _metric = METRIC_ORDER[0]
    _widths = [
        (
            interval_width(by_method[_metric, arm, UNADJUSTED.name]),
            interval_width(by_method[_metric, arm, CUPED.name]),
        )
        for arm in arms
    ]
    _reductions = [1 - adjusted / plain for plain, adjusted in _widths]

    _width_text = "; ".join(
        f"treatment: {plain:.2%} → {adjusted:.2%}" for plain, adjusted in _widths
    )

    mo.md(f"""
    ## Readout

    CUPED reduces the `{_metric}` interval width by
    **{_reductions[0]:.1%}** ({_width_text}). The coeftable below shows the full
    unadjusted-versus-CUPED estimate and interval comparison.
    """)
    return


@app.cell
def _(estimates, estimates_to_readout, readout_table):
    readout_table(
        estimates_to_readout(estimates),
        title="Unadjusted vs CUPED",
        subtitle="Relative lift vs control, 95% credible interval",
        nest_by="method",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Takeaway

    CUPED is most useful for continuous outcomes such as duration, revenue, and
    counts. Binary outcomes such as conversion and retention usually see a smaller
    reduction because less of their variance is predictable from pre-period data.
    """)
    return


if __name__ == "__main__":
    app.run()
