import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    from increment import CupedCovariate, Method, MetricSpec, Scope, from_frame, run
    from increment.estimators import Normal
    from increment.results import LiftEstimate
    from increment.tables import estimates_to_readout, readout_table
    from increment.workflows import srm

    return (
        CupedCovariate,
        LiftEstimate,
        Method,
        MetricSpec,
        Normal,
        Scope,
        estimates_to_readout,
        from_frame,
        mo,
        np,
        pd,
        readout_table,
        run,
        srm,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Analysing an experiment straight from a dataframe

    While the core workflow of this package is connecting straight to a warehouse, we also support analysis straight from a dataframe, making it easy for data scientists to do adhoc analyses.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## An Example Checkout Experiment

    We simulated a fake experiment with the following metrics:
    * revenue over the 5-day experiment window
    * revenue over the 5 days *before* the experiment (the CUPED covariate)
    * an order count (for a ratio metric)
    * and a pre-computed conversion indicator

    The treatment raises the value of each order by 15% and touches nothing
    else, so order counts carry no true effect. One row per user, exactly what
    `from_frame(..., shape="summary")` expects.
    """)
    return


@app.cell(hide_code=True)
def simulate(np, pd):
    _rng = np.random.default_rng(7)
    _n_users = 20_000
    _n_days = 5

    # Persistent per-user traits: they drive both the pre-period and the
    # experiment window, which is what makes pre_revenue a useful covariate.
    _quality = _rng.normal(0, 1, _n_users)
    _countries = _rng.choice(["US", "CA", "GB"], size=_n_users, p=[0.5, 0.25, 0.25])
    _country_scale = np.select(
        [_countries == "CA", _countries == "GB"],
        [0.9, 1.1],
        default=1.0,
    )
    _active_probability = np.clip(0.35 + 0.15 * _quality, 0.05, 0.95)
    _order_rate = np.clip(0.4 + 0.2 * _quality, 0.05, None)
    _order_value = (10 + 4 * _quality) * _country_scale

    # One population randomized into arms; the lift applies to order value only.
    _variant = np.where(_rng.random(_n_users) < 0.5, "treatment", "control")
    _lift = np.where(_variant == "treatment", 0.15, 0.0)

    # Each order's value is an independent Gamma draw with CV 1/sqrt(shape):
    # real basket sizes are right-skewed and vary order to order. The sum of
    # k iid Gamma(shape, scale) draws is Gamma(k * shape, scale), so a user's
    # daily revenue is one draw per user.
    _order_value_shape = 1.0
    _mean_order_value = np.clip(_order_value, 1, None)

    def _simulate_day(rng, lift):
        active = rng.uniform(0, 1, _n_users) < _active_probability
        orders = np.where(active, 1 + rng.poisson(_order_rate), 0)
        revenue = np.where(
            orders > 0,
            rng.gamma(
                np.maximum(orders, 1) * _order_value_shape,
                _mean_order_value / _order_value_shape,
            ),
            0.0,
        )
        return orders, revenue * (1 + lift)

    # Pre-period: same users, same process, before anyone was treated.
    _pre_revenue = np.zeros(_n_users)
    for _day in range(_n_days):
        _pre_revenue += _simulate_day(_rng, 0.0)[1]

    _daily_frames = []
    for _day in range(_n_days):
        _orders, _revenue = _simulate_day(_rng, _lift)
        _daily_frames.append(
            pd.DataFrame(
                {
                    "user_id": [f"u{i:05d}" for i in range(_n_users)],
                    "variant": _variant,
                    "day": f"2026-02-{_day + 1:02d}",
                    "country": _countries,
                    "revenue": _revenue,
                    "orders": _orders,
                    "converted": (_orders > 0).astype(float),
                    "pre_revenue": _pre_revenue,
                }
            )
        )

    panel_df = pd.concat(_daily_frames, ignore_index=True)

    checkout_df = panel_df.groupby(["user_id", "variant", "country"], as_index=False).agg(
        revenue=("revenue", "sum"),
        pre_revenue=("pre_revenue", "first"),
        orders=("orders", "sum"),
        converted=("converted", "max"),
    )
    checkout_df.head().round(2)
    return checkout_df, panel_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The core workflow

    Analyzing an experiment is easy:
    """)
    return


@app.cell
def _(checkout_df, estimates_to_readout, from_frame, readout_table, run):
    # pass in your experiment
    experiment, resolver = from_frame(
        checkout_df,
        shape="summary",
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean", "converted": "conversion"},
    )
    baseline_results = run(experiment, resolver).readout()

    # visualize the readout
    readout_table(estimates_to_readout(baseline_results), title="Checkout experiment")
    return experiment, resolver


@app.cell
def _(mo):
    mo.md(r"""
    ## CUPED: adjusting for the pre-period covariate

    CUPED is also implemented. With just one call, both methods are supported.
    The moments are computed once, and the CUPED adjustment is applied to the
    same per-arm sums the unadjusted estimate uses, so any difference below is
    the adjustment and nothing else. How much it buys is set by one number: the
    correlation between the covariate and the metric. Variance falls by roughly
    that correlation squared, so a weak covariate is not worth the column.
    """)
    return


@app.cell
def cuped(
    CupedCovariate,
    Method,
    MetricSpec,
    checkout_df,
    estimates_to_readout,
    from_frame,
    readout_table,
    run,
):
    UNADJUSTED = Method(name="unadjusted")
    CUPED = Method.cuped(covariates=CupedCovariate(name="pre_revenue", kind="continuous"))

    _experiment, _resolver = from_frame(
        checkout_df,
        shape="summary",
        unit="user_id",
        group="variant",
        control="control",
        metrics=[
            MetricSpec(
                name="revenue",
                type="mean",
                sensitivity_methods=(CUPED,),
            )
        ],
    )
    cuped_estimates = run(
        _experiment, _resolver, decision_method=UNADJUSTED, sensitivity_methods=(CUPED,)
    ).readout()
    readout_table(estimates_to_readout(cuped_estimates), title="CUPED Comparison", nest_by="method")
    return (cuped_estimates,)


@app.cell(hide_code=True)
def _(LiftEstimate, checkout_df, cuped_estimates, mo):
    def interval_width(estimate: LiftEstimate) -> float:
        lift = estimate.lift
        assert lift.lb is not None and lift.ub is not None, "run() always returns an interval"
        return lift.ub - lift.lb

    unadjusted_estimate = cuped_estimates[0]
    cuped_estimate = cuped_estimates[1]
    _plain_width = interval_width(unadjusted_estimate)
    _adjusted_width = interval_width(cuped_estimate)
    _rho = checkout_df["revenue"].corr(checkout_df["pre_revenue"])

    mo.md(f"""
    `pre_revenue` correlates with `revenue` at **{_rho:.2f}**, so the ceiling on
    variance reduction is about {_rho**2:.0%} — and CUPED narrows the interval by
    **{(1 - _adjusted_width / _plain_width):.1%}** (width shrinks with the square
    root of variance). The point estimate moves from
    **{unadjusted_estimate.lift.value:+.1%}** to **{cuped_estimate.lift.value:+.1%}**:
    the arms differ slightly in pre-period revenue by chance, and the adjustment
    prices that imbalance out.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Adding a prior

    Priors on the lift can be set via

    ```python
    run(experiment, resolver, prior=Normal(mu=mu, sigma=sigma))
    ```

    This adds a separate posterior summary; it does not replace the sampling inference or the declared multiplicity/sequential procedure.
    """)
    return


@app.cell
def _(Normal, estimates_to_readout, experiment, readout_table, resolver, run):
    prior_results = run(experiment, resolver, prior=Normal(mu=0.0, sigma=0.05)).readout()
    readout_table(
        estimates_to_readout(prior_results),
        title="Checkout experiment — informative prior",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## A ratio metric

    `orders` is the denominator: `MetricSpec(type="ratio", numerator=...,
    denominator=...)` estimates lift on **revenue per order**, not on revenue
    or order count separately.

    `Revenue per User` below is the same estimand as the `revenue` row in the
    first table — a `mean` metric is already a per-unit average, so the two
    always agree. It sits here to show the decomposition: revenue per user is
    revenue per order times orders per user, and the three lifts multiply out
    the same way.
    """)
    return


@app.cell
def _(MetricSpec, checkout_df, estimates_to_readout, from_frame, readout_table, run):
    _experiment, _resolver = from_frame(
        checkout_df,
        shape="summary",
        unit="user_id",
        group="variant",
        control="control",
        metrics=[
            MetricSpec(
                name="Avg. Order Value",
                type="ratio",
                numerator="revenue",
                denominator="orders",
            ),
            MetricSpec(
                name="Revenue per User",
                type="mean",
                value_column="revenue",
            ),
            MetricSpec(
                name="Avg. Orders",
                type="mean",
                value_column="orders",
            ),
        ],
    )
    ratio_result = run(_experiment, _resolver).readout()

    readout_table(
        estimates_to_readout(ratio_result),
        title="Analyzing a Ratio Metric",
        # nest_by='method'
    )
    return (ratio_result,)


@app.cell(hide_code=True)
def _(mo, ratio_result):
    _lift = {estimate.metric: estimate.lift.value for estimate in ratio_result}
    _implied = (1 + _lift["Avg. Order Value"]) * (1 + _lift["Avg. Orders"]) - 1

    mo.md(f"""
    The treatment only raises the value of each order, and that is what the
    ratio metric isolates: **{_lift["Avg. Order Value"]:+.1%}** on revenue per
    order against **{_lift["Avg. Orders"]:+.1%}** on orders per user, whose
    interval covers zero. Multiplying them gives **{_implied:+.1%}**, matching
    the **{_lift["Revenue per User"]:+.1%}** measured directly on revenue per
    user. Revenue per user is the blunter metric here: it carries the order-count
    noise on top of the effect.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Sample-ratio mismatch

    Free on this shape: `srm()` reads the per-group unit count straight out of
    the moments already computed above, no extra pass over the data. Its
    anytime-valid default requires known, constant conditional assignment
    probabilities; `expected` declares their arm support and shares.
    """)
    return


@app.cell
def _(experiment, resolver, srm):
    from rich.pretty import pprint

    # Request one fixed-look check on the simulated population.
    srm_result = srm(
        experiment, resolver, expected={"control": 0.5, "treatment": 0.5}, inference="fixed"
    )
    pprint(srm_result, expand_all=True)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Analyzing a panel dataset

    `from_frame(..., shape="panel")` takes the same experiment used above at user-day grain.
    The unit-summary examples aggregate this panel to one row per user, so their
    whole-window effects match exactly. `country` is a stable per-user dimension
    carried on every daily row.
    """)
    return


@app.cell(hide_code=True)
def _(panel_df):
    panel_df.head().round(2)
    return


@app.cell
def _(Scope, estimates_to_readout, from_frame, mo, panel_df, readout_table, run):
    _experiment, _resolver = from_frame(
        panel_df,
        shape="panel",
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
    )
    # One run reads the whole-window readout and the per-day values together.
    _whole_window = Scope(view="run", grain="total")
    _daily = Scope(view="daily", grain="daily", values=True)
    _panel_run = run(_experiment.with_views(_whole_window, _daily), _resolver)
    daily_values = _panel_run.readout(_daily)
    panel_results = _panel_run.readout(_whole_window)
    panel_readout = readout_table(
        estimates_to_readout(panel_results),
        title="Analyzing Panel Data",
    )
    mo.vstack(
        [
            panel_readout,
            mo.md(f"`daily_values` produced **{len(daily_values)}** per-day metric values."),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Windows, retention, and censoring on the panel path


    - Any windowed or retention metric needs `exposure_date=...` which is day 0 for that
      metric's window/band arithmetic.

    - **Late enrollees are censored, not zeroed.** A unit whose window/band
      has not had time to close by `observation_end` (explicit, or for a
      running experiment, the latest date that metric's own column was
      actually observed) is dropped entirely rather than read as "observed,
      did not return." A `UserWarning` fires if this drops more than 10% of
      enrolled units.
    """)
    return


@app.cell
def _(MetricSpec, estimates_to_readout, from_frame, panel_df, readout_table, run):
    # Every unit in this synthetic panel enrolled on the same day; a real
    # dataset would carry its own per-unit exposure_date column instead.
    panel_df_exposed = panel_df.assign(exposure_date="2026-02-01")

    _experiment, _resolver = from_frame(
        panel_df_exposed,
        shape="panel",
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[
            MetricSpec(name="revenue", type="mean"),
            MetricSpec(
                name="revenue (3d window)", value_column="revenue", type="mean", window_days=3
            ),
        ],
        exposure_date="exposure_date",
    )
    windowed_lift_breakout = run(_experiment, _resolver).readout()

    readout_table(
        estimates_to_readout(windowed_lift_breakout),
        title="Windowed Lift",
        # nest_by='method'
    )
    return (panel_df_exposed,)


@app.cell
def _(
    MetricSpec,
    estimates_to_readout,
    from_frame,
    panel_df_exposed,
    readout_table,
    run,
):
    _experiment, _resolver = from_frame(
        panel_df_exposed,
        shape="panel",
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[
            MetricSpec(
                name="returned",
                type="retention",
                value_column="revenue",
                threshold_days=(1, 5),
            )
        ],
        exposure_date="exposure_date",
    )
    retention_lift = run(_experiment, _resolver).readout()

    readout_table(
        estimates_to_readout(retention_lift),
        title="Retention Lift",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What's still not here

    Panel-frame covariates (CUPED) remain unsupported - collapsing a
    pre-period value from per-day rows is ambiguous; aggregate to one row
    per unit yourself and use `from_frame(..., shape="summary")` instead (see
    [`docs/guides/cuped.md`](../docs/guides/cuped.md)).

    `from_definitions` remains the richer retention home: a
    bounded `RetentionMetric` supports all five readout views there.
    Panel-frame retention supports the whole-window readout and the as-of
    value and lift views, but raises `CapabilityError` from the daily value
    and daily lift views - the frame-backed panel has no per-day
    cohort reduction, bounded band or not.
    """)
    return


if __name__ == "__main__":
    app.run()
