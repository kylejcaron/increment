import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import coeftable as ct
    import marimo as mo
    import numpy as np
    import pandas as pd
    from scipy.special import expit

    from increment import (
        AdjustmentSet,
        Analysis,
        IdentificationGate,
        Method,
        Observational,
    )
    from increment.tables import estimates_to_readout, readout_table

    return (
        AdjustmentSet,
        Analysis,
        IdentificationGate,
        Method,
        Observational,
        ct,
        estimates_to_readout,
        expit,
        mo,
        np,
        pd,
        readout_table,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Observational inference: IPTW and DML

    Treatment was chosen, not randomized. Here, `x1` and `x2` affect both treatment
    and revenue, so raw group means are confounded.

    IPTW and DML adjust for those measured differences. Both assume the adjustment
    set contains every confounder; overlap checks can diagnose poor comparisons but
    cannot prove that assumption.
    """)
    return


@app.cell(hide_code=True)
def _(expit, np, pd):
    _rng = np.random.default_rng(42)
    _n = 3000
    _effect = 0.2

    _x1 = _rng.normal(size=_n)
    _x2 = _rng.normal(size=_n)
    _propensity = expit(-0.3 + 1.2 * _x1 - 0.8 * _x2)
    _d = _rng.binomial(1, _propensity)
    _y = 1.0 + 0.9 * _x1 + 0.6 * _x2 + _effect * _d + _rng.normal(scale=0.5, size=_n)

    df = pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(_n)],
            "variant": np.where(_d == 1, "treatment", "control"),
            "revenue": _y,
            "x1": _x1,
            "x2": _x2,
        }
    )
    df.head()
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Simulate confounded data

    The true effect is **+0.2**, or **+20%** relative to the untreated mean.
    `x1` and `x2` drive both treatment and revenue.
    """)
    return


@app.cell
def _(Analysis, df):
    naive_results = Analysis.from_unit_summary(
        df,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
    ).run()
    return (naive_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Naive comparison
    """)
    return


@app.cell
def _(estimates_to_readout, naive_results, readout_table):
    readout_table(estimates_to_readout(naive_results), title="Naive comparison (confounded)")
    return


@app.cell
def _(mo, naive_results):
    _naive_lift = naive_results[0].lift.value

    mo.md(f"""
    The naive estimate is **{_naive_lift:+.2%}**, not the true +20%. Treated users
    already had higher expected revenue because of `x1` and `x2`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. IPTW: reweight comparable users

    IPTW weights users by their treatment propensity. The default overlap gate
    refuses extreme weights; `overlap="trim"` instead drops poorly overlapping users
    and names the resulting overlap population on the estimate.
    """)
    return


@app.cell
def _(AdjustmentSet, Analysis, IdentificationGate, Observational, df):
    iptw_analysis = Analysis.from_unit_summary(
        df,
        unit="user_id",
        group="variant",
        metrics={"revenue": "mean"},
        design=Observational(
            control_group="control",
            adjustment=AdjustmentSet(covariates=("x1", "x2")),
            gate=IdentificationGate(overlap="trim"),
        ),
    )
    iptw_results = iptw_analysis.run()
    return iptw_analysis, iptw_results


@app.cell
def _(estimates_to_readout, iptw_results, readout_table):
    readout_table(estimates_to_readout(iptw_results), title="IPTW-adjusted")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. DML: adjust treatment and outcome

    DML cross-fits both a propensity model and an outcome model: every user's
    prediction comes from folds that excluded that user. It uses the same declared
    adjustment set; only the method changes.
    """)
    return


@app.cell
def _(Method, iptw_analysis):
    dml_results = iptw_analysis.run(decision_method=Method(name="dml"))
    return (dml_results,)


@app.cell
def _(dml_results, estimates_to_readout, readout_table):
    readout_table(estimates_to_readout(dml_results), title="DML-adjusted")
    return


@app.cell
def final_comparison(ct, dml_results, iptw_results, mo, naive_results, np, pd):
    _comparison = pd.DataFrame(
        {
            "estimator": ["Naive", "IPTW", "DML"],
            "estimate": [
                naive_results[0].lift.value,
                iptw_results[0].lift.value,
                dml_results[0].lift.value,
            ],
            "lb": [
                np.nan,
                iptw_results[0].lift.lb,
                dml_results[0].lift.lb,
            ],
            "ub": [
                np.nan,
                iptw_results[0].lift.ub,
                dml_results[0].lift.ub,
            ],
            "truth": 0.2,
        }
    )
    _comparison_table = (
        ct.CoefTable(_comparison, rows="estimator")
        .estimate(
            "Lift",
            "estimate",
            ci=("lb", "ub"),
            fmt=ct.Percent(scale=100.0, decimals=2),
        )
        .forest(
            "Lift comparison",
            of="Lift",
            annotations=(ct.Rule("truth", axis="x", color="red"),),
            width=240,
            symmetric=True,
        )
        .header("Compare with truth", "Known truth: +20% in red")
    )

    mo.vstack(
        [
            _comparison_table,
            mo.md("""
            Both adjusted estimates recover the known effect. DML can be more precise
            when its outcome model explains variation that weighting alone cannot.
            """),
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
