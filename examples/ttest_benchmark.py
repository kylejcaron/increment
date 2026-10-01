import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import coeftable as ct
    import marimo as mo
    import numpy as np
    import pandas as pd
    from scipy.stats import norm, t, ttest_ind

    from increment import Analysis

    return Analysis, ct, mo, norm, np, pd, t, ttest_ind


@app.cell
def _(mo):
    mo.md(r"""
    # A simple sanity check: Increment vs. a t-test

    This is Increment's simplest case: independent units, one mean metric, and
    fixed-horizon inference. The same simulated experiment is analyzed three ways:
    Increment, an explicit log-scale z-test, and Welch's two-sample t-test.

    The red line is the data-generating effect. Increment and the explicit z-test
    should agree; Welch's interval converges to them as the sample grows.
    """)
    return


@app.cell(hide_code=True)
def simulate(np):
    max_units_per_arm = 4_000
    sample_sizes = range(100, max_units_per_arm + 1, 100)
    true_effects = np.arange(-0.10, 0.1001, 0.005).round(3)
    return max_units_per_arm, sample_sizes, true_effects


@app.cell
def effect_selection():
    def selected_true_effect(percent: float) -> float:
        return round(percent / 100, 3)

    def format_true_effect(effect: float) -> str:
        return f"{effect:+.1%}"

    return format_true_effect, selected_true_effect


@app.cell(hide_code=True)
def _(Analysis, mo, norm, np, pd, sample_sizes, t, true_effects, ttest_ind):
    rows = []
    total_calculations = len(sample_sizes) * len(true_effects)

    with mo.status.progress_bar(
        total=total_calculations,
        title="Calculating benchmark estimates",
        subtitle="Starting...",
    ) as progress:
        for _n_per_arm in sample_sizes:
            for _effect_index, _true_effect in enumerate(true_effects):
                # Fresh experiment for each sample-size/effect combination.
                rng = np.random.default_rng(11_000_000 + _n_per_arm * 100 + _effect_index)
                control_baseline = np.clip(rng.normal(20.0, 8.0, size=_n_per_arm), 0, None)
                treatment_baseline = np.clip(rng.normal(20.0, 8.0, size=_n_per_arm), 0, None)

                # simulate dataframe
                control = control_baseline
                treatment = treatment_baseline * (1.0 + _true_effect)
                data = pd.DataFrame(
                    {
                        "user_id": np.arange(2 * _n_per_arm),
                        "variant": ["control"] * _n_per_arm + ["treatment"] * _n_per_arm,
                        "revenue": np.concatenate([control, treatment]),
                    }
                )

                analysis = Analysis.from_unit_summary(
                    data,
                    unit="user_id",
                    group="variant",
                    control="control",
                    metrics={"revenue": "mean"},
                )
                (increment_estimate,) = analysis.run()
                increment_lift = increment_estimate.lift
                rows.append(
                    {
                        "n_per_arm": _n_per_arm,
                        "true_effect": _true_effect,
                        "method": "Increment",
                        "estimate": increment_lift.value,
                        "lb": increment_lift.lb,
                        "ub": increment_lift.ub,
                    }
                )

                control_mean = float(control.mean())
                treatment_mean = float(treatment.mean())
                relative_point = treatment_mean / control_mean - 1.0
                log_ratio = float(np.log(treatment_mean / control_mean))
                se_log_ratio = float(
                    np.sqrt(
                        control.var(ddof=1) / (_n_per_arm * control_mean**2)
                        + treatment.var(ddof=1) / (_n_per_arm * treatment_mean**2)
                    )
                )
                z_critical = float(norm.ppf(0.975))
                rows.append(
                    {
                        "n_per_arm": _n_per_arm,
                        "true_effect": _true_effect,
                        "method": "Frequentist z-test",
                        "estimate": relative_point,
                        "lb": float(np.expm1(log_ratio - z_critical * se_log_ratio)),
                        "ub": float(np.expm1(log_ratio + z_critical * se_log_ratio)),
                    }
                )

                test = ttest_ind(treatment, control, equal_var=False)
                difference = treatment_mean - control_mean
                se_difference = float(
                    np.sqrt(treatment.var(ddof=1) / _n_per_arm + control.var(ddof=1) / _n_per_arm)
                )
                t_critical = float(t.ppf(0.975, float(test.df)))
                rows.append(
                    {
                        "n_per_arm": _n_per_arm,
                        "true_effect": _true_effect,
                        "method": "Welch t-test",
                        "estimate": relative_point,
                        "lb": (difference - t_critical * se_difference) / control_mean,
                        "ub": (difference + t_critical * se_difference) / control_mean,
                    }
                )
                progress.update(
                    subtitle=f"Effect {_true_effect:+.0%}; {_n_per_arm:,} units per arm"
                )

    benchmark_results = pd.DataFrame(rows)
    return (benchmark_results,)


@app.cell
def _(max_units_per_arm, mo):
    sample_size_slider = mo.ui.slider(
        start=100,
        stop=max_units_per_arm,
        step=100,
        value=1_000,
        label="Units per arm",
        show_value=True,
    )
    true_effect_slider = mo.ui.slider(
        start=-10,
        stop=10,
        step=0.5,
        value=5,
        label="True effect (%)",
        show_value=True,
    )
    mo.vstack(
        [
            mo.md("## Explore the benchmark"),
            sample_size_slider,
            true_effect_slider,
        ]
    )
    return sample_size_slider, true_effect_slider


@app.cell
def _(
    benchmark_results,
    ct,
    format_true_effect,
    mo,
    sample_size_slider,
    selected_true_effect,
    true_effect_slider,
):
    _n_per_arm = sample_size_slider.value
    _true_effect = selected_true_effect(true_effect_slider.value)
    _selected = benchmark_results[
        (benchmark_results["n_per_arm"] == _n_per_arm)
        & (benchmark_results["true_effect"] == _true_effect)
    ]
    comparison_table = (
        ct.CoefTable(_selected, rows="method")
        .estimate(
            "Relative lift",
            "estimate",
            ci=("lb", "ub"),
            fmt=ct.Percent(scale=100.0, decimals=2),
        )
        .forest(
            "Relative lift comparison",
            of="Relative lift",
            width=260,
            symmetric=True,
            annotations=(ct.Rule("true_effect", axis="x", color="red"),),
            ylim=(-0.3, 0.3),
        )
        .header(
            "Benchmark",
            f"95% intervals — {_n_per_arm:,} units/arm; true effect {format_true_effect(_true_effect)}",
        )
    )
    mo.vstack([comparison_table])
    return


@app.cell
def _(mo, sample_size_slider):
    _n_per_arm = sample_size_slider.value
    mo.md(f"""
    The red line marks the **true effect** used to generate this experiment.
    Increment and the explicit z-test are the direct sanity-check pair; Welch's
    t-test is the familiar additive-scale reference. At **{_n_per_arm:,} units per
    arm**, their point estimates and intervals should be close, with the t/z
    difference shrinking as the sample grows.
    """)
    return


if __name__ == "__main__":
    app.run()
