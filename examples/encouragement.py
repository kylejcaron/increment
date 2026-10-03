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
        MetricSpec,
        UptakeSpec,
    )
    from increment.tables import estimates_to_readout, readout_table

    return (
        Analysis,
        AnalysisPlan,
        InferenceSpec,
        Encouragement,
        ExclusionRestriction,
        MetricSpec,
        UptakeSpec,
        estimates_to_readout,
        mo,
        np,
        pd,
        readout_table,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Encouragement design

    Randomize a **prompt**, then estimate the effect of the behavior it changes.
    Everyone has the help button; only the encouragement is assigned.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.Html(r"""
    <style>
    .enc-diagram { max-width: 920px; margin: 0 auto 1.5rem; }
    .enc-legend { display: flex; flex-wrap: wrap; gap: 1rem; justify-content: center; margin: .5rem 0 1.25rem; font-size: .85rem; }
    .enc-key { display: flex; align-items: center; gap: .4rem; }
    .enc-dot { width: .75rem; height: .75rem; border-radius: .2rem; display: inline-block; }
    .enc-stage { display: grid; grid-template-columns: 9rem minmax(0, 1fr) 9rem; gap: 1rem; align-items: center; margin: .8rem 0; }
    .enc-stage-label { text-align: right; font-weight: 650; color: var(--slate-11); }
    .enc-stage-content { grid-column: 2; display: flex; gap: 1rem; justify-content: center; align-items: stretch; }
    .enc-box { flex: 1; max-width: 19rem; border: 1.5px solid var(--slate-7); border-radius: .8rem; padding: 1rem; text-align: center; background: var(--slate-1); }
    .enc-box strong { display: block; margin-bottom: .25rem; }
    .enc-instrument { border-color: #6c5ce7; background: #f4f1ff; }
    .enc-treatment { border-color: #00a884; background: #edfbf7; }
    .enc-outcome { border-color: #e67e22; background: #fff5eb; }
    .enc-arrow { text-align: center; color: var(--slate-9); font-size: 1.3rem; line-height: 1; }
    .enc-phone { width: 5.5rem; margin: .6rem auto 0; border: 2px solid #34495e; border-radius: .7rem; padding: .55rem .45rem .45rem; background: white; }
    .enc-line { height: .35rem; background: #dfe6e9; border-radius: .25rem; margin: .3rem 0; }
    .enc-button { background: #00a884; color: white; border-radius: .35rem; padding: .35rem .2rem; margin-top: .55rem; font-size: .7rem; }
    .enc-prompt { background: #6c5ce7; color: white; border-radius: .4rem; padding: .35rem; font-size: .65rem; margin-bottom: .4rem; transform: rotate(-2deg); }
    .enc-results { display: grid; grid-template-columns: repeat(3, 1fr); gap: .65rem; margin: 1.25rem auto .8rem; max-width: 47rem; }
    .enc-result { border-top: 4px solid #6c5ce7; border-radius: .55rem; padding: .8rem; background: var(--slate-1); }
    .enc-assumption { max-width: 47rem; margin: 1rem auto 0; padding: .75rem 1rem; border-left: 4px solid #e67e22; background: #fff5eb; border-radius: .4rem; }
    @media (max-width: 700px) {
      .enc-stage { grid-template-columns: 1fr; }
      .enc-stage-label { text-align: center; }
      .enc-stage-content { grid-column: 1; flex-direction: column; align-items: center; }
      .enc-box { width: 100%; box-sizing: border-box; }
      .enc-results { grid-template-columns: 1fr; }
    }
    </style>
    <div class="enc-diagram">
      <div class="enc-legend">
        <span class="enc-key"><i class="enc-dot" style="background:#6c5ce7"></i> Randomized encouragement</span>
        <span class="enc-key"><i class="enc-dot" style="background:#00a884"></i> Chosen treatment</span>
        <span class="enc-key"><i class="enc-dot" style="background:#e67e22"></i> Outcome</span>
      </div>
      <div class="enc-stage">
        <div class="enc-stage-label">Available to everyone</div>
        <div class="enc-stage-content"><div class="enc-box"><strong>Help button</strong>Every user can click it.</div></div>
      </div>
      <div class="enc-arrow">↓ randomize</div>
      <div class="enc-stage">
        <div class="enc-stage-label">Encouragement</div>
        <div class="enc-stage-content">
          <div class="enc-box enc-instrument"><strong>Control</strong>No prompt<div class="enc-phone"><div class="enc-line"></div><div class="enc-line"></div><div class="enc-button">Help</div></div></div>
          <div class="enc-box enc-instrument"><strong>Encouraged</strong>Strong prompt<div class="enc-phone"><div class="enc-prompt">Need help? Try this</div><div class="enc-line"></div><div class="enc-button">Help</div></div></div>
        </div>
      </div>
      <div class="enc-arrow">↓ users choose</div>
      <div class="enc-stage">
        <div class="enc-stage-label">Uptake</div>
        <div class="enc-stage-content">
          <div class="enc-box enc-treatment"><strong>12% click</strong>Baseline uptake</div>
          <div class="enc-box enc-treatment"><strong>55% click</strong>Encouraged uptake</div>
        </div>
      </div>
      <div class="enc-arrow">↓</div>
      <div class="enc-stage">
        <div class="enc-stage-label">Outcome</div>
        <div class="enc-stage-content"><div class="enc-box enc-outcome"><strong>Revenue</strong>Measured for every user</div></div>
      </div>
      <div class="enc-results">
        <div class="enc-result"><strong>ITT</strong><br>Effect of the prompt</div>
        <div class="enc-result" style="border-color:#00a884"><strong>Uptake</strong><br>55% − 12% = 43 pp</div>
        <div class="enc-result" style="border-color:#e67e22"><strong>LATE</strong><br>Effect of clicking for compliers</div>
      </div>
      <div class="enc-assumption"><strong>Required assumptions:</strong> the prompt affects revenue only through clicking, and no user would click without the prompt but refuse when prompted.</div>
    </div>
    """),
            mo.md(r"""
    ## Simulate user-level data

    One row per user: assigned prompt, click, and revenue. A shared latent uptake threshold makes the simulated click response monotone: the prompt can create clicks, not deter them.
    """),
        ]
    )
    return


@app.cell
def _(np, pd):
    def simulate_encouragement(
        rng,
        n_per_arm,
        control_uptake,
        encouraged_uptake,
        tau,
        base=20.0,
        noise_sd=6.0,
    ):
        """Two-sided encouragement with monotone latent uptake thresholds."""
        rows = []
        for group, uptake_rate in (
            ("control", control_uptake),
            ("encouraged", encouraged_uptake),
        ):
            uptake_rank = rng.uniform(size=n_per_arm)
            clicked = (uptake_rank < uptake_rate).astype(int)
            revenue = base + tau * clicked + rng.normal(0, noise_sd, size=n_per_arm)
            rows.extend(
                (f"{group[0]}{i:04d}", group, float(revenue[i]), int(clicked[i]))
                for i in range(n_per_arm)
            )
        return pd.DataFrame(rows, columns=["user_id", "variant", "revenue", "clicked"])

    encouragement_df = simulate_encouragement(
        np.random.default_rng(42),
        n_per_arm=1500,
        control_uptake=0.12,
        encouraged_uptake=0.55,
        tau=18.0,
    )
    encouragement_df.head()
    return encouragement_df, simulate_encouragement


@app.cell
def _(
    Analysis,
    Encouragement,
    ExclusionRestriction,
    UptakeSpec,
    encouragement_df,
):
    encouragement_design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True,
            justification="the randomized prompt affects revenue only through help-button clicks",
        ),
        one_sided=False,
    )
    encouragement_results = Analysis.from_unit_summary(
        encouragement_df,
        unit="user_id",
        group="variant",
        metrics={"revenue": "mean"},
        design=encouragement_design,
    ).run()
    return encouragement_design, encouragement_results


@app.cell(hide_code=True)
def _(encouragement_df, encouragement_results, mo):
    _itt = next(r for r in encouragement_results if r.estimand == "itt")
    _uptake = encouragement_df.groupby("variant")["clicked"].mean()
    _control_uptake = float(_uptake["control"])
    _encouraged_uptake = float(_uptake["encouraged"])
    _uptake_difference = _encouraged_uptake - _control_uptake
    _late = next(
        r for r in encouragement_results if r.estimand == "late" and r.value_scale == "relative"
    )

    mo.md(
        f"""
    ## Results

    - **ITT — prompt:** {_itt.lift.value:+.1%} revenue lift
    - **Uptake — clicking:** {_uptake_difference * 100:+.1f} percentage points ({_encouraged_uptake:.1%} vs {_control_uptake:.1%})
    - **LATE — clicking for compliers:** {_late.lift.value:+.1%} revenue lift
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### Intervals and assumptions
    """)
    return


@app.cell
def _(encouragement_results, estimates_to_readout, readout_table):
    readout_table(
        estimates_to_readout([r for r in encouragement_results if r.estimand == "itt"]),
        title="ITT — effect of the prompt",
    )
    return


@app.cell
def _(encouragement_results, estimates_to_readout, readout_table):
    readout_table(
        estimates_to_readout([r for r in encouragement_results if r.estimand == "compliance"]),
        title="Uptake — relative lift in clicking",
    )
    return


@app.cell
def _(encouragement_results, estimates_to_readout, readout_table):
    readout_table(
        estimates_to_readout(
            [
                r
                for r in encouragement_results
                if r.estimand == "late" and r.value_scale == "relative"
            ]
        ),
        title="LATE — effect of clicking for compliers",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Weak encouragement

    LATE divides ITT by the change in uptake. If that change is too uncertain,
    increment omits LATE but still reports the randomized ITT.
    """)
    return


@app.cell
def _(Analysis, encouragement_design, np, simulate_encouragement):
    weak_df = simulate_encouragement(
        np.random.default_rng(7),
        n_per_arm=40,
        control_uptake=0.12,
        encouraged_uptake=0.16,
        tau=18.0,
    )
    weak_results = Analysis.from_unit_summary(
        weak_df,
        unit="user_id",
        group="variant",
        metrics={"revenue": "mean"},
        design=encouragement_design,
    ).run()
    return (weak_results,)


@app.cell
def _(mo, weak_results):
    _estimands = sorted({r.estimand for r in weak_results})
    _compliance = next(r for r in weak_results if r.estimand == "compliance")

    mo.md(f"""
    Reported: **{", ".join(_estimands)}**. LATE is omitted.

    > {_compliance.note}
    """)
    return


@app.cell
def _(estimates_to_readout, readout_table, weak_results):
    readout_table(
        estimates_to_readout([r for r in weak_results if r.estimand == "itt"]),
        title="Weak encouragement — ITT remains valid",
    )
    return


@app.cell
def _(estimates_to_readout, readout_table, weak_results):
    readout_table(
        estimates_to_readout([r for r in weak_results if r.estimand == "compliance"]),
        title="Weak encouragement — uncertain uptake change",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Monitor LATE over time

    Fixed-horizon LATE trends carry a repeated-look warning. Binary uptake
    makes the joint LATE score non-Gaussian, so it cannot use the registered
    Gaussian likelihood. The certified series below instead monitors relative
    uptake with an Always valid Bernoulli/Beta model, declared as
    `InferenceSpec(kind="always_valid", registration=...)` on the source's
    `AnalysisPlan` before the simulation runs -- `always_valid` inference is
    read from that declaration, never passed at call time. Both series use
    `completed_windows_only=True`, so every displayed date admits a unit
    only once its finalized outcome and uptake windows have closed.
    """)
    return


@app.cell(hide_code=True)
def _(
    Analysis,
    AnalysisPlan,
    Encouragement,
    ExclusionRestriction,
    InferenceSpec,
    MetricSpec,
    UptakeSpec,
    np,
    pd,
    simulate_encouragement,
):
    from increment import (
        JointReveal,
        PredictivePrior,
        SequentialCell,
        SequentialCompliancePolicy,
        SequentialModel,
        SequentialRegistration,
        sequential_definition_id,
    )
    from increment.frame import synthesise_metric
    from increment.sequential_source import frame_observation_mapping

    _window_days = 7
    _design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked", window_days=_window_days),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True,
            justification="the randomized prompt affects revenue only through help-button clicks",
        ),
        one_sided=False,
    )
    _specs = [MetricSpec(name="revenue", type="mean", window_days=_window_days)]
    _prior = PredictivePrior(kind="beta", a=1, b=1)
    _registration = SequentialRegistration(
        source_id="help_button_prompt",
        control_group="control",
        committed_before_data=True,
        definitions_id=sequential_definition_id(
            [synthesise_metric(spec) for spec in _specs],
            _design,
            transformations=_specs,
            source_mapping=frame_observation_mapping(
                unit="user_id",
                group="variant",
                date="ds",
                exposure_date="exposed_on",
                uptake="clicked",
            ),
        ),
        reveal=JointReveal(
            filtration_id="help-button-finalized-units",
            independent_unit_vectors=True,
            simultaneous_metrics=True,
            outcome_independent_order=True,
            immutable_finalized_outcomes=True,
            longest_window_days=_window_days,
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
        roster=(SequentialCell(metric="uptake", group_id="encouraged", estimand="compliance"),),
    )
    _registered_plan = AnalysisPlan(
        inference=InferenceSpec(kind="always_valid", registration=_registration),
        compliance=SequentialCompliancePolicy(alpha=_registration.roster[0].alpha),
    )
    _rng = np.random.default_rng(3)
    # Staggered enrollment over 21 days, so day_index varies across units
    # even though `simulate_encouragement` realizes each outcome once.
    _sim = simulate_encouragement(
        _rng,
        n_per_arm=250,
        control_uptake=0.12,
        encouraged_uptake=0.55,
        tau=18.0,
    ).reset_index(drop=True)
    _enroll = list(pd.date_range("2026-03-01", periods=21, freq="D").date)
    _sim["user_id"] = "u" + _sim.index.astype(str)
    _sim["exposed_on"] = np.array(_enroll)[_sim.index % len(_enroll)]
    # Keep outcome and completion rows; ``from_unit_panel`` densifies the
    # intervening days and admits each cohort once both windows are complete.
    _exposure_rows = _sim.assign(ds=_sim["exposed_on"])
    _completion_rows = _sim.assign(
        ds=_sim["exposed_on"] + pd.Timedelta(days=_window_days), revenue=0.0, clicked=0.0
    )
    _panel = pd.concat([_exposure_rows, _completion_rows], ignore_index=True)[
        ["user_id", "variant", "ds", "exposed_on", "revenue", "clicked"]
    ]
    _analysis = Analysis.from_unit_panel(
        _panel,
        unit="user_id",
        group="variant",
        date="ds",
        metrics=[MetricSpec(name="revenue", type="mean", window_days=_window_days)],
        design=_design,
        exposure_date="exposed_on",
        experiment_id="help_button_prompt",
    )
    _av_analysis = Analysis.from_unit_panel(
        _panel,
        unit="user_id",
        group="variant",
        date="ds",
        metrics=[MetricSpec(name="revenue", type="mean", window_days=_window_days)],
        design=_design,
        plan=_registered_plan,
        exposure_date="exposed_on",
        experiment_id="help_button_prompt",
    )

    def _late_trend(results):
        return [row for row in results if row.estimand == "late" and row.value_scale == "relative"]

    fixed_late_trend = _late_trend(_analysis.run_asof_lift(completed_windows_only=True))
    uptake_trend = []
    for _day in sorted(set(_completion_rows["ds"])):
        _av_analysis.capture_sequential(finalized=True, as_of=_day)
        uptake_trend.extend(
            _av_analysis.run_asof_lift(estimands=("compliance",), completed_windows_only=True)
        )
    return uptake_trend, fixed_late_trend


@app.cell(hide_code=True)
def _(uptake_trend, pd):
    pd.DataFrame(
        {
            "ds": r.ds,
            "uptake_lift": None if r.lift is None else r.lift.value,
            "lb": r.sequential_result.bounds.lower,
            "ub": r.sequential_result.bounds.upper,
            "inference": r.inference,
        }
        for r in uptake_trend
    )
    return


@app.cell(hide_code=True)
def _(uptake_trend, fixed_late_trend, mo):
    _fixed_last = fixed_late_trend[-1]
    _av_last = uptake_trend[-1]

    mo.md(f"""
    - **Fixed alpha:** [{_fixed_last.lift.lb:+.1%}, {_fixed_last.lift.ub:+.1%}] — repeated-look warning
    - **Registered uptake:** {_av_last.sequential_result.bounds} — ratio-coordinate confidence sequence

    The rows answer different questions: the causal effect among compliers and the randomized uptake contrast.
    """)
    return


if __name__ == "__main__":
    app.run()
