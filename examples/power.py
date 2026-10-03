import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from rich.pretty import pprint

    from increment import (
        Baseline,
        InferenceSpec,
        ParallelAssignment,
        PowerDesign,
        achieved_power,
        minimum_detectable_effect,
        power_curve,
        required_sample_size,
    )
    from increment.decision import FixedInference
    from increment.estimation.arm_contract import (
        AnalysisAxes,
        FamilyPolicy,
        MetricCapabilities,
        PlanningFamilyExpansion,
        RelativeDecisionPolicy,
    )
    from increment.power import ArmPlanningProcedure
    from increment.semantics.models import MethodSpec

    return (
        AnalysisAxes,
        ArmPlanningProcedure,
        Baseline,
        FamilyPolicy,
        FixedInference,
        InferenceSpec,
        MetricCapabilities,
        MethodSpec,
        ParallelAssignment,
        PlanningFamilyExpansion,
        PowerDesign,
        RelativeDecisionPolicy,
        achieved_power,
        minimum_detectable_effect,
        mo,
        power_curve,
        pprint,
        required_sample_size,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Power analysis

    Power analysis is used to size an experiment. It can help answer
    * how many users does a target lift need?
    * how much power does a fixed budget buy?
    * what is the smallest effect a given sample size could even detect?


    We additionally support power estimation beyond just the simple case, such as CUPED and multiple variants.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Assumptions

    `mean` and `var` describe the *control* arm's per-unit outcome. In real life
    they come from one of two places:
    * a historical query (`SELECT avg(revenue), var_samp(revenue) FROM ...` over the population you plan
    to expose)
    * or `Baseline.from_summary(stats)` where `stats` is a `SummaryStats` produced by a prior increment analysis (a pilot or an earlier readout).

    For a conversion-style metric, `Baseline.from_proportion(p)` fills in the Bernoulli variance `p * (1 - p)` for you.

    Here: average 14-day revenue per exposed user. Heavy right tail, so the variance dwarfs the mean - which is exactly why sample sizes get large. CUPED can be used to help with this later
    """)
    return


@app.cell
def _():
    MEAN_REVENUE = 12.40  # dollars per user
    VAR_REVENUE = 430.0  # dollars^2 per user (sd ~ $20.7)

    TARGET_LIFT = 0.03  # +3% revenue per user
    BUDGET_N = 20_000
    BIGGER_N = 80_000

    # `cuped_rho` is the assumed correlation between the outcome and a pre-period covariate
    CUPED_RHO = 0.5
    N_VARIANTS = 4
    return (
        BIGGER_N,
        BUDGET_N,
        CUPED_RHO,
        MEAN_REVENUE,
        N_VARIANTS,
        TARGET_LIFT,
        VAR_REVENUE,
    )


@app.cell
def _(
    AnalysisAxes,
    ArmPlanningProcedure,
    Baseline,
    FamilyPolicy,
    FixedInference,
    MetricCapabilities,
    MethodSpec,
    ParallelAssignment,
    PlanningFamilyExpansion,
    PowerDesign,
    RelativeDecisionPolicy,
    MEAN_REVENUE,
    VAR_REVENUE,
):
    baseline = Baseline(mean=MEAN_REVENUE, var=VAR_REVENUE)
    design = PowerDesign(power=0.80)
    procedure = ArmPlanningProcedure(
        assignment=ParallelAssignment(),
        analysis=AnalysisAxes(
            identification="randomized",
            view="total",
            segmented=False,
            completed_windows_only=True,
            population="assigned",
            variance_adjustment="none",
        ),
        dependence="iid",
        inference=FixedInference(),
        estimand="mean",
        metric=MetricCapabilities(
            metric_type="mean",
            value_scale="relative",
            winsorization="none",
            outcome_window="bounded",
            uptake_window="not_applicable",
        ),
        decision=RelativeDecisionPolicy(
            alternative="two-sided",
            null_lift=0.0,
            family=FamilyPolicy(kind="none", axes=(), nominal_alpha=0.05),
        ),
        family_expansion=PlanningFamilyExpansion(family_size=1),
        # The decision method is the estimator the ship decision will use;
        # sensitivity methods only add reporting-only comparison rows.
        decision_method=MethodSpec(name="unadjusted"),
        sensitivity_methods=(),
        prior_present=False,
    )
    return baseline, design, procedure


@app.cell
def _(mo):
    mo.md("""
    ## 1. How many users do I need to detect a target lift?
    """)
    return


@app.cell
def _(TARGET_LIFT, baseline, design, pprint, procedure, required_sample_size):
    rss = required_sample_size(TARGET_LIFT, baseline, procedure, design)
    pprint(rss)
    return (rss,)


@app.cell
def _(mo):
    mo.md("""
    ## 2. I can only get a fixed budget of users per arm - what power is that?
    """)
    return


@app.cell
def _(BIGGER_N, BUDGET_N, TARGET_LIFT, achieved_power, baseline, design, procedure):
    ap = achieved_power(BUDGET_N, TARGET_LIFT, baseline, procedure, design)
    ap_bigger = achieved_power(BIGGER_N, TARGET_LIFT, baseline, procedure, design)
    return ap, ap_bigger


@app.cell(hide_code=True)
def _(BIGGER_N, BUDGET_N, TARGET_LIFT, ap, ap_bigger, mo):
    mo.md(f"""
    | | |
    |--|--:|
    | Power, {BUDGET_N:,}/arm | {ap.power:.1%} |
    | Power, {BIGGER_N:,}/arm | {ap_bigger.power:.1%} (more N -> more power) |

    A real {TARGET_LIFT:+.0%} effect at {BUDGET_N:,} users/arm would be missed
    {1 - ap.power:.0%} of the time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. What is the smallest lift I could detect?
    """)
    return


@app.cell
def _(BIGGER_N, BUDGET_N, baseline, design, minimum_detectable_effect, procedure):
    mde_small = minimum_detectable_effect(BUDGET_N, baseline, procedure, design)
    mde_big = minimum_detectable_effect(BIGGER_N, baseline, procedure, design)
    return mde_big, mde_small


@app.cell(hide_code=True)
def _(BIGGER_N, BUDGET_N, MEAN_REVENUE, mde_big, mde_small, mo):
    def _mde_row(label, result):
        if result.mde_relative is None:
            return f"| {label} | not available ({result.mde_unavailable_reason}) | - |"
        return (
            f"| {label} | {result.mde_relative:+.2%} | {MEAN_REVENUE * result.mde_relative:+.3f} |"
        )

    small_row = _mde_row(f"{BUDGET_N:,} users/arm", mde_small)
    big_row = _mde_row(f"{BIGGER_N:,} users/arm", mde_big)
    mo.md(f"""
    | Budget | MDE (relative) | MDE (dollars/user) |
    |--|--:|--:|
    {small_row}
    {big_row}

    The MDE shrinks as N grows: bigger tests resolve finer effects. A row
    reads "not available" when no effect reaches the target power at that
    budget.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Extended features

    ## Estimating a Power Curve
    """)
    return


@app.cell
def _(baseline, design, power_curve, procedure):
    import altair as alt

    sample_sizes = range(5_000, 55_001, 5_000)
    effect_sizes = [0.01, 0.02, 0.03, 0.05]
    alphas = [0.01, 0.05, 0.10]
    procedures = [
        procedure.model_copy(
            update={
                "decision": procedure.decision.model_copy(
                    update={
                        "family": procedure.decision.family.model_copy(
                            update={"nominal_alpha": nominal_alpha}
                        )
                    }
                )
            }
        )
        for nominal_alpha in alphas
    ]

    curve = power_curve(
        n_per_arm=sample_sizes,
        relative_lift=effect_sizes,
        baseline=baseline,
        procedure=procedures,
        design=design,
        units_per_week=30000,
    )

    curve_df = curve.to_frame(backend="pandas")

    alt.Chart(curve_df).mark_line().encode(
        x=alt.X("duration_days:Q", title="Duration (Days)"),
        y=alt.Y("power:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("relative_lift:N", title="Relative lift"),
        strokeDash=alt.StrokeDash("alpha:N", title="Alpha"),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multiple variants + Bonferroni - more users for the same power
    """)
    return


@app.cell
def _(
    N_VARIANTS,
    PlanningFamilyExpansion,
    TARGET_LIFT,
    baseline,
    design,
    procedure,
    required_sample_size,
    rss,
):
    procedure_mc = procedure.model_copy(
        update={
            "decision": procedure.decision.model_copy(
                update={
                    "family": procedure.decision.family.model_copy(
                        update={"kind": "bonferroni", "axes": ("arm",)}
                    )
                }
            ),
            "family_expansion": PlanningFamilyExpansion(family_size=N_VARIANTS),
        }
    )
    rss_mc = required_sample_size(TARGET_LIFT, baseline, procedure_mc, design)

    extra = rss_mc.n_per_arm - rss.n_per_arm
    return procedure_mc, extra, rss_mc


@app.cell(hide_code=True)
def _(N_VARIANTS, design, extra, mo, procedure, procedure_mc, rss, rss_mc):
    mo.md(f"""
    | | | |
    |--|--:|--:|
    | 1 variant, no correction | {rss.n_per_arm:,} users/arm | alpha {procedure.compiled_alpha:.4f} |
    | {N_VARIANTS} variants, bonferroni | {rss_mc.n_per_arm:,} users/arm | alpha {procedure_mc.compiled_alpha:.4f} |

    {extra:,} more users/arm, a {extra / rss.n_per_arm:.1%} increase - and that is
    per arm, across {N_VARIANTS + 1} arms.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## CUPED variance reduction - fewer users for the same power

    Planning with a nonzero `cuped_rho` requires a procedure that declares
    CUPED; make it the *decision* method, since the savings are only real
    if the ship decision itself uses the adjusted estimator.
    """)
    return


@app.cell
def _(
    Baseline,
    CUPED_RHO,
    MEAN_REVENUE,
    MethodSpec,
    TARGET_LIFT,
    VAR_REVENUE,
    design,
    procedure,
    required_sample_size,
    rss,
):
    baseline_cuped = Baseline(mean=MEAN_REVENUE, var=VAR_REVENUE, cuped_rho=CUPED_RHO)
    procedure_cuped = procedure.model_copy(
        update={"decision_method": MethodSpec(name="cuped", variance_reduction="cuped")}
    )
    rss_cuped = required_sample_size(TARGET_LIFT, baseline_cuped, procedure_cuped, design)

    saved = rss.n_per_arm - rss_cuped.n_per_arm
    return procedure_cuped, rss_cuped, saved


@app.cell(hide_code=True)
def _(CUPED_RHO, mo, rss, rss_cuped, saved):
    mo.md(f"""
    | | |
    |--|--:|
    | rho = 0.0 (no adjustment) | {rss.n_per_arm:,} users/arm |
    | rho = {CUPED_RHO:.1f} (pre-period cov.) | {rss_cuped.n_per_arm:,} users/arm |
    | Effective variance | {rss.effective_var:,.1f} -> {rss_cuped.effective_var:,.1f} |

    {saved:,} fewer users/arm, a {saved / rss.n_per_arm:.1%} reduction - theory says
    required N scales by `1 - rho^2` = {1 - CUPED_RHO**2:.2f}.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Non-inferiority sizing - how many users to CONFIRM no loss

    A non-inferiority design asks a different question than a headline metric: not "how many
    users to detect a lift," but "how many users to CONFIRM we didn't lose more
    than a tolerance, if the true effect is actually zero." That's
    `relative_lift=0.0` against a one-sided design shifted to a nonzero
    `null_lift`.
    """)
    return


@app.cell
def _(
    Baseline,
    CUPED_RHO,
    MEAN_REVENUE,
    RelativeDecisionPolicy,
    VAR_REVENUE,
    design,
    procedure_cuped,
    required_sample_size,
):
    non_inferiority_procedure = procedure_cuped.model_copy(
        update={
            "decision": RelativeDecisionPolicy(
                alternative="greater",
                null_lift=-0.025,
                family=procedure_cuped.decision.family,
            )
        }
    )
    non_inferiority_baseline = Baseline(mean=MEAN_REVENUE, var=VAR_REVENUE, cuped_rho=CUPED_RHO)
    non_inferiority_plan = required_sample_size(
        relative_lift=0.0,
        baseline=non_inferiority_baseline,
        procedure=non_inferiority_procedure,
        design=design,
    )

    non_inferiority_mde = non_inferiority_plan.mde_relative
    true_lift_at_boundary = (
        None
        if non_inferiority_mde is None
        else (1 + non_inferiority_procedure.decision.null_lift) * (1 + non_inferiority_mde) - 1
    )
    return (
        non_inferiority_mde,
        non_inferiority_plan,
        non_inferiority_procedure,
        true_lift_at_boundary,
    )


@app.cell(hide_code=True)
def _(
    mo,
    non_inferiority_mde,
    non_inferiority_plan,
    non_inferiority_procedure,
    true_lift_at_boundary,
):
    margin = -non_inferiority_procedure.decision.null_lift
    if non_inferiority_mde is None or true_lift_at_boundary is None:
        mde_line = f"| MDE beyond the margin | not available ({non_inferiority_plan.mde_unavailable_reason}) |"
        mde_note = "No effect beyond the margin reaches the target power at this N."
    else:
        mde_line = f"| MDE beyond the margin | {non_inferiority_mde:+.2%} |"
        mde_note = (
            "`mde_relative` here is the distance DETECTABLE BEYOND the margin, not an "
            f"absolute effect size: {non_inferiority_mde:+.2%} beyond a {margin:.0%} margin "
            f"is a true lift of about {true_lift_at_boundary:+.2%}."
        )
    mo.md(f"""
    | | |
    |--|--:|
    | Margin (tolerated loss) | {margin:.0%} |
    | Treatment arm | {non_inferiority_plan.n_per_arm:,} users |
    | Total (both arms) | {non_inferiority_plan.n_total:,} users |
    | Power at that N | {non_inferiority_plan.power:.1%} |
    {mde_line}

    {mde_note}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sequential planning with asymptotic_mean inference

    Declaring `InferenceSpec(kind="asymptotic_mean")` on the procedure plans the
    boundary the runtime monitors with under that same declaration, at 14
    equally spaced looks. `kind="always_valid"` has no planning construction:
    plan such a test at a fixed horizon and monitor it at runtime.
    """)
    return


@app.cell
def _(
    ArmPlanningProcedure,
    InferenceSpec,
    TARGET_LIFT,
    achieved_power,
    baseline,
    minimum_detectable_effect,
    pprint,
    required_sample_size,
):
    sequential_procedure = ArmPlanningProcedure.standard(
        "mean", inference=InferenceSpec(kind="asymptotic_mean")
    )

    sample_size = required_sample_size(
        relative_lift=TARGET_LIFT,
        baseline=baseline,
        procedure=sequential_procedure,
        planned_looks=14,
    )

    power = achieved_power(
        n_per_arm=25_000,
        relative_lift=TARGET_LIFT,
        baseline=baseline,
        procedure=sequential_procedure,
        planned_looks=14,
    )

    mde = minimum_detectable_effect(
        n_per_arm=25_000,
        baseline=baseline,
        procedure=sequential_procedure,
        planned_looks=14,
    )

    pprint(
        {
            "required_sample_size": sample_size,
            "achieved_power": power,
            "minimum_detectable_effect": mde,
        }
    )
    return


if __name__ == "__main__":
    app.run()
