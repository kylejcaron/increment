import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import ibis
    import marimo as mo
    from _seed import seed_event_log

    import increment
    from increment.results import LiftEstimate
    from increment.tables import estimates_to_readout, readout_table

    return (
        LiftEstimate,
        estimates_to_readout,
        ibis,
        increment,
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
    compares their estimated variance for each metric and treatment arm.
    The gain depends on how well each metric's pre-period activity predicts
    its outcome, not on the displayed interval's width.

    The method is derived in [`docs/guides/cuped.md`](../docs/guides/cuped.md).
    """)
    return


@app.cell
def _(increment, mo):
    UNADJUSTED = increment.Method(name="unadjusted")
    CUPED = increment.Method(name="cuped", variance_reduction="cuped")
    mo.show_code(position="above")
    return CUPED, UNADJUSTED


@app.cell(hide_code=True)
def _():
    DEFINITIONS = "examples/definitions/"
    EXPERIMENT = "new_onboarding_v2"
    # Put the continuous metric first: it shows the clearest variance reduction.
    METRIC_ORDER = ("avg_session_duration", "purchase_rate", "d7_retention")
    return DEFINITIONS, EXPERIMENT, METRIC_ORDER


@app.cell(hide_code=True)
def _(LiftEstimate):
    def variance_gain(plain: LiftEstimate, adjusted: LiftEstimate) -> float | None:
        """Estimated variance reduction for the absolute treatment-control difference."""
        plain_se, adjusted_se = plain.abs_se, adjusted.abs_se
        if plain_se is None or adjusted_se is None or plain_se <= 0 or adjusted_se < 0:
            return None
        return 1 - (adjusted_se / plain_se) ** 2

    return (variance_gain,)


@app.cell
def _(DEFINITIONS, EXPERIMENT, ibis, increment, seed_event_log):
    con = ibis.duckdb.connect()

    # generate events
    seed_event_log(con, with_pre_period=True)

    experiment, reader = increment.from_definitions(DEFINITIONS, con, experiment_name=EXPERIMENT)
    return experiment, reader


@app.cell
def _(CUPED, UNADJUSTED, experiment, increment, mo, reader):
    # One run computes both methods from the same queried moments.
    estimates = increment.run(
        experiment, reader, decision_method=UNADJUSTED, sensitivity_methods=(CUPED,)
    ).readout()
    mo.show_code(position="above")
    return (estimates,)


@app.cell(hide_code=True)
def _(estimates):
    by_method = {(est.metric, est.group_id, est.method): est for est in estimates}
    arms = sorted({est.group_id for est in estimates})
    return arms, by_method


@app.cell(hide_code=True)
def _(CUPED, METRIC_ORDER, UNADJUSTED, arms, by_method, mo, variance_gain):
    precision_comparisons = [
        (
            metric,
            arm,
            variance_gain(
                by_method[metric, arm, UNADJUSTED.name],
                by_method[metric, arm, CUPED.name],
            ),
        )
        for metric in METRIC_ORDER
        for arm in arms
    ]
    comparison_text = "\n".join(
        f"- `{metric}` / `{arm}`: **{gain:.1%} variance reduction**."
        if gain is not None
        else f"- `{metric}` / `{arm}`: variance reduction unavailable."
        for metric, arm, gain in precision_comparisons
    )

    mo.md(
        "## Readout\n\n"
        "Compare the same absolute treatment-control difference under both methods:\n"
        "variance reduction = **1 − (CUPED absolute SE / unadjusted absolute SE)²**.\n\n"
        f"{comparison_text}\n\n"
        "A negative gain means CUPED increased the estimated variance. A missing "
        "absolute SE or a zero unadjusted SE makes this comparison unavailable, "
        "not a zero gain. Relative-lift interval widths are not this variance "
        "comparison: they also depend on the inference method, confidence level, "
        "and whether the interval is one-sided."
    )
    return


@app.cell
def _(estimates, estimates_to_readout, readout_table):
    readout_table(
        estimates_to_readout(estimates),
        title="Unadjusted vs CUPED",
        subtitle="Relative lift vs control; sampling confidence intervals, one-sided for guardrails",
        nest_by="method",
        show_interval_level=True,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Takeaway

    - **Session duration:** pre-period activity strongly predicts this continuous
      outcome, so CUPED removes a large share of its estimated variance.
    - **Purchase rate:** the binary purchase outcome retains more unpredictable
      variation, so its gain is smaller.
    - **D7 retention:** pre-period page views predict only a little of this
      binary outcome's variation; its modest variance gain is not a dramatic
      narrowing of the one-sided guardrail interval.

    The pre-history signal is synthetic: the seed drives pre- and post-exposure
    activity with a shared per-user propensity. These gains illustrate
    metric-specific predictability, not a promise about real experiment data.
    """)
    return


if __name__ == "__main__":
    app.run()
