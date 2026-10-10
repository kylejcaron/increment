import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import coeftable as ct
    import marimo as mo
    import numpy as np
    import pandas as pd

    from increment import Analysis, Covariate, estimate_cate, targeting_rule, validate_cate
    from increment.frame import from_unit_summary

    return (
        Analysis,
        Covariate,
        ct,
        estimate_cate,
        from_unit_summary,
        mo,
        np,
        pd,
        targeting_rule,
        validate_cate,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Heterogeneous treatment effects

    The average effect asks **did it work?** HTE asks **who benefits more?**

    1. Confirm the overall effect.
    2. Estimate how effects vary with pre-treatment covariates.
    3. Validate the ranking on held-out users.
    4. Build a targeting rule only if validation passes.

    A model always finds a “best” group. Held-out validation tells us whether that
    ranking is real. Because this notebook uses simulated data, every estimate can
    be compared with the true per-user effect, `tau_true`.
    """)
    return


@app.cell(hide_code=True)
def _(np, pd):
    PLATFORMS = ("ios", "android", "web")
    REGIONS = ("na", "emea", "apac", "latam")

    # The known truth: tau(x) = BASE_TAU + SPEND_SLOPE * z(spend) + IOS_BONUS * 1[ios],
    # in absolute revenue per user - the scale `estimate_cate` reports on.
    BASE_TAU = 0.15
    SPEND_SLOPE = 0.10
    IOS_BONUS = 0.08
    NOISE_SD = 0.40

    def simulate(n, seed, *, heterogeneous):
        """One row per user, with the per-unit truth carried alongside."""
        rng = np.random.default_rng(seed)

        # Strictly pre-exposure covariates: measured before assignment, so
        # nothing here can be touched by the treatment.
        spend = rng.lognormal(3.0, 0.35, n)  # dollars in the 30 days before
        tenure = rng.gamma(2.0, 3.0, n)  # months on the platform
        platform = rng.choice(PLATFORMS, size=n, p=(0.35, 0.45, 0.20))
        sessions = rng.poisson(8.0, n).astype(float)
        referrals = rng.poisson(0.6, n).astype(float)
        region = rng.choice(REGIONS, size=n, p=(0.40, 0.30, 0.20, 0.10))
        z_spend = (spend - spend.mean()) / spend.std()

        tau = np.full(n, BASE_TAU)
        if heterogeneous:
            tau = tau + SPEND_SLOPE * z_spend + IOS_BONUS * (platform == "ios")

        variant = np.where(rng.random(n) < 0.5, "treatment", "control")
        treated = (variant == "treatment").astype(float)

        # Prognostic, not moderating: these four move revenue in BOTH arms and
        # leave tau alone, which is exactly what `adjust=` covariates are for.
        # The `min(tenure, 6)` kink is why tenure is declared with knots below.
        baseline = (
            1.0
            + 0.30 * z_spend
            + 0.12 * np.minimum(tenure, 6.0)
            + 0.10 * (platform == "web")
            + 0.02 * sessions
        )
        revenue = baseline + treated * tau + rng.normal(0.0, NOISE_SD, n)

        return pd.DataFrame(
            {
                "user_id": [f"u{i:05d}" for i in range(n)],
                "variant": variant,
                "revenue": revenue,
                "spend": spend,
                "tenure": tenure,
                "platform": platform,
                "sessions": sessions,
                "referrals": referrals,
                "region": region,
                # Never fed to any estimator - the answer key, on screen.
                "tau_true": tau,
            }
        )

    return BASE_TAU, IOS_BONUS, SPEND_SLOPE, simulate


@app.cell(hide_code=True)
def _(simulate):
    hetero_df = simulate(20_000, 20260812, heterogeneous=True)
    hetero_df.head(8).round(3)
    return (hetero_df,)


@app.cell(hide_code=True)
def _(BASE_TAU, IOS_BONUS, SPEND_SLOPE, hetero_df, mo):
    _tau = hetero_df["tau_true"]
    _shares = hetero_df["variant"].value_counts()
    mo.md(f"""
    ## Simulated cohorts

    **Cohort A:** {len(hetero_df):,} users ({_shares["control"]:,} control,
    {_shares["treatment"]:,} treatment).

    `tau(x) = {BASE_TAU:.2f} + {SPEND_SLOPE:.2f} × z(spend) + {IOS_BONUS:.2f} × 1[ios]`

    True effects average **{_tau.mean():+.4f}** with standard deviation
    **{_tau.std():.4f}**. Spend and iOS drive heterogeneity; the other covariates
    improve precision only.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Confirm the average effect

    Start with the ordinary experiment readout. HTE is a follow-up, not a substitute
    for the headline effect.
    """)
    return


@app.cell
def _(Analysis, hetero_df):
    headline = Analysis.from_unit_summary(
        hetero_df,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
    ).run()
    return (headline,)


@app.cell(hide_code=True)
def _(headline, hetero_df, mo):

    _control_mean = hetero_df.loc[hetero_df["variant"] == "control", "revenue"].mean()
    _truth_relative = hetero_df["tau_true"].mean() / _control_mean

    mo.md(
        "\n".join(
            [
                *(
                    f"- **Estimated lift:** {r.lift.value:+.2%} "
                    f"[{r.lift.lb:+.2%}, {r.lift.ub:+.2%}]"
                    for r in headline
                ),
                f"- **True lift:** {_truth_relative:+.2%}",
            ]
        )
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Estimate how effects vary

    - Every `interact=` covariate also enters as a main effect, so it both adjusts
      for that covariate and allows the treatment effect to vary with it.
    - `adjust=` only controls. Use it when you deliberately assume the
      covariate can explain variance but not be an effect modifier.

    Real users rarely know that distinction in advance. Tenure is a plausible effect
    modifier, so this example includes it in `interact=` rather than using simulated
    knowledge to hide it in `adjust=`. The fit can then show whether the data
    supports tenure HTE, and held-out validation checks whether the resulting ranking
    generalizes.

    HTE results use the absolute outcome scale: revenue per user.
    """)
    return


@app.cell
def _(Covariate, estimate_cate, from_unit_summary, hetero_df):
    INTERACT = [
        "spend",
        Covariate(name="tenure", knots=4),
        Covariate(name="platform", kind="categorical"),
    ]
    ADJUST = []

    hetero_src = from_unit_summary(
        hetero_df,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
    )
    fit = estimate_cate(
        hetero_src,
        "revenue",
        control="control",
        interact=INTERACT,
        adjust=ADJUST,
    )
    return ADJUST, INTERACT, fit, hetero_src


@app.cell(hide_code=True)
def _(IOS_BONUS, SPEND_SLOPE, ct, fit, hetero_df, mo, pd):
    _truth_by_column = {
        "d:spend": SPEND_SLOPE,
        "d:platform=ios": IOS_BONUS,
        "d:platform=web": 0.0,
    }
    _interaction_frame = pd.DataFrame(
        {
            "interaction": [e.name for e in fit.interactions],
            "estimate": [e.coef for e in fit.interactions],
            "lb": [e.lb for e in fit.interactions],
            "ub": [e.ub for e in fit.interactions],
            "truth": [_truth_by_column.get(e.name, 0.0) for e in fit.interactions],
        }
    )
    _interaction_table = (
        ct.CoefTable(_interaction_frame, rows="interaction")
        .estimate(
            "Estimate",
            "estimate",
            ci=("lb", "ub"),
            fmt=ct.Number(decimals=4, signed=True),
        )
        .forest(
            "Estimate plot",
            of="Estimate",
            annotations=(ct.Rule("truth", axis="x", color="red"),),
            width=240,
        )
        .header("Estimated Interaction effects", "Ground-truth in red")
    )

    mo.vstack(
        [
            mo.md(
                "\n".join(
                    [
                        f"**Average effect:** {fit.ate:+.4f} [{fit.lb:+.4f}, {fit.ub:+.4f}] "
                        f"(truth {hetero_df['tau_true'].mean():+.4f})",
                        "",
                        f"Covariate adjustment reduced the standard error from {fit.se_unadjusted:.5f} "
                        f"to {fit.se:.5f} (**{fit.se_reduction:.1%}**).",
                        "",
                        f"**Joint heterogeneity test:** χ²={fit.heterogeneity.statistic:.1f}, "
                        f"df={fit.heterogeneity.df}, p={fit.heterogeneity.p_value:.2g}",
                        f"**Pruned columns:** `{fit.pruned or '()'}`",
                    ]
                )
            ),
            _interaction_table,
        ]
    )
    return


@app.cell(hide_code=True)
def _(BASE_TAU, IOS_BONUS, SPEND_SLOPE, ct, fit, hetero_df, mo, np, pd):
    _p10, _p90 = (float(np.quantile(hetero_df["spend"], q)) for q in (0.10, 0.90))
    _tenure = float(hetero_df["tenure"].median())
    _high = {"spend": _p90, "tenure": _tenure, "platform": "ios"}
    _low = {"spend": _p10, "tenure": _tenure, "platform": "android"}

    _z = (np.array([_p10, _p90]) - hetero_df["spend"].mean()) / hetero_df["spend"].std()
    _truth_high = BASE_TAU + SPEND_SLOPE * _z[1] + IOS_BONUS
    _truth_low = BASE_TAU + SPEND_SLOPE * _z[0]

    _high_est = fit.cate(_high)
    _low_est = fit.cate(_low)
    _gap = fit.contrast(_high, _low)
    _compare_frame = pd.DataFrame(
        {
            "user": [
                f"iOS, 90th-pct spend (${_p90:,.0f})",
                f"Android, 10th-pct spend (${_p10:,.0f})",
            ],
            "estimate": [_high_est.value, _low_est.value],
            "lb": [_high_est.lb, _low_est.lb],
            "ub": [_high_est.ub, _low_est.ub],
            "truth": [_truth_high, _truth_low],
        }
    )
    _compare_table = (
        ct.CoefTable(_compare_frame, rows="user")
        .estimate(
            "Estimate",
            "estimate",
            ci=("lb", "ub"),
            fmt=ct.Number(decimals=4, signed=True),
        )
        .forest(
            "Estimate plot",
            of="Estimate",
            annotations=(ct.Rule("truth", axis="x", color="red"),),
            width=240,
            symmetric=True,
        )
        .header("Compare two users", "True conditional effects in red")
    )

    mo.vstack(
        [
            mo.md(
                "Holding tenure fixed at its median isolates the spend and platform contrast."
                f"\n\n**Direct contrast:** {_gap.value:+.4f} [{_gap.lb:+.4f}, {_gap.ub:+.4f}] "
                f"(truth {_truth_high - _truth_low:+.4f})"
            ),
            _compare_table,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Validate on held-out users

    `validate_cate` fits on one half and reports effects from the other. The gate
    passes when the held-out ranking has significant **AUTOC** (Area Under the Targeting-Operator Characteristic curve).
    Until then, report only the average effect.


    AUTOC, measures whether the model’s held-out ranking is useful:

    1. Score users by predicted treatment effect.
    2. Sort users from lowest to highest predicted effect.
    3. Compare observed treatment effects as you move through that ranking.
    4. Integrate that curve into one statistic.

    **Interpretation**:

    - AUTOC > 0: higher-ranked users tend to have larger treatment effects.
    - AUTOC ≈ 0: ranking is no better than random.
    - AUTOC < 0: ranking is backwards.
    - Its one-sided p-value tests whether the ranking carries real signal.
    - The validation gate passes when autoc.p_value < alpha.
    """)
    return


@app.cell
def _(ADJUST, INTERACT, hetero_src, validate_cate):
    validation = validate_cate(
        hetero_src,
        "revenue",
        control="control",
        interact=INTERACT,
        adjust=ADJUST,
        n_groups=5,
    )
    return (validation,)


@app.cell
def _():
    def display_number(value, spec="+.4f"):
        return "unavailable" if value is None else format(value, spec)

    def display_estimate(estimate):
        if estimate is None:
            return "unavailable"
        point = display_number(estimate.value)
        if estimate.lb is None and estimate.ub is None:
            return f"{point} (point only; interval unavailable)"
        return f"{point} [{display_number(estimate.lb)}, {display_number(estimate.ub)}]"

    return display_estimate, display_number


@app.cell(hide_code=True)
def _(display_estimate, display_number, mo, validation):
    mo.md(f"""
    - **Train / holdout:** {validation.n_train:,} / {validation.n_holdout:,} users
    - **Gate:** {"PASSED" if validation.passed else "FAILED"} (α={validation.alpha})
    - **AUTOC:** {display_number(validation.autoc.estimate)} (p={display_number(validation.autoc.p_value, ".2g")})
    - **Qini:** {display_number(validation.qini.estimate)} (p={display_number(validation.qini.p_value, ".2g")})
    - **Holdout ATE:** {display_estimate(validation.holdout_ate)}
    - **Rank uncertainty:** {validation.autoc.unavailable_reason or "available"}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### GATES on held-out users

    GATES (Group Average Treatment Effects on the Selected) evaluates whether the
    model’s treatment-effect ranking generalizes beyond the data used to fit it.

    We fit the model on a training sample, use it to predict treatment effects for
    untouched holdout users, sort those users by predicted effect, and divide them into
    K=5 equally sized score groups. Within each group, we estimate the treatment effect
    using only the held-out outcomes.

    The group estimates will not match the predictions exactly; they are noisy estimates
    with their own confidence intervals. The important pattern is that the measured
    effect increases steadily from the lowest- to the highest-ranked group. That
    monotone relationship indicates that the model’s ranking is informative out of
    sample, rather than merely fitting noise in the training data.
    """)
    return


@app.cell(hide_code=True)
def _(ct, display_number, pd):
    def gates_table(validation, truth, title):
        """Held-out effect per predicted-effect group, with its interval."""
        groups = list(validation.groups)
        group_labels = [str(group.group) for group in groups]
        group_labels[0] += " (lowest)"
        group_labels[-1] += " (highest)"

        frame = pd.DataFrame(
            {
                "group": group_labels,
                "n_display": [f"{group.n:,}" for group in groups],
                "effect": [group.effect for group in groups],
                "lb": [group.lb for group in groups],
                "ub": [group.ub for group in groups],
                "mean_score_display": [display_number(group.mean_score) for group in groups],
                "uncertainty": [group.unavailable_reason or "available" for group in groups],
            }
        )
        return (
            ct.CoefTable(
                frame,
                rows="group",
                direction="neutral",
                title=title,
                subtitle=f"Dashed line: true average effect ({truth:+.4f})",
                sort_rows=False,
            )
            .estimate(
                "Held-out effect",
                "effect",
                ci=("lb", "ub"),
                fmt=ct.Number(decimals=4, signed=True),
            )
            .forest(
                "Effect plot",
                of="Held-out effect",
                ref=truth,
                width=220,
            )
            .passthrough("Holdout n", "n_display")
            .passthrough("Mean predicted effect", "mean_score_display")
            .passthrough("Uncertainty", "uncertainty")
        )

    return (gates_table,)


@app.cell
def _(gates_table, hetero_df, validation):
    gates_table(
        validation,
        float(hetero_df["tau_true"].mean()),
        "Cohort A: GATES holdout ranking",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### CLAN profile table

    The CLAN profile is a descriptive explanation of who the model ranked as most versus
    least affected - it tries to ask of GATES, "who is in each group?"

    For each covariate, it compares the mean in the most affected group (group 5) against the mean in the least affected group (group 1).



    - Most affected: Group 5, the highest predicted-effect quintile.
    - Least affected: Group 1, the lowest predicted-effect quintile.
    - Difference: Most affected minus least affected.
    - 95% interval: Uncertainty around that covariate difference.

    It does not estimate another treatment effect. It profiles the covariates associated
    with the model’s ranking, using held-out users.
    """)
    return


@app.cell(hide_code=True)
def _(ct, display_number, hetero_df, pd, validation):
    _groups = validation.groups
    _pairs = list(zip(_groups, _groups[1:], strict=False))
    _monotone = all(
        a.effect is not None and b.effect is not None and a.effect <= b.effect for a, b in _pairs
    )

    # Normalize only the visualization: keep the raw CLAN contrasts readable
    # while putting mixed covariates on a common, dimensionless forest scale.
    def _profile_values(name):
        if "=" in name:
            column, level = name.split("=", 1)
            return (hetero_df[column].astype(str) == level).astype(float)
        return hetero_df[name].astype(float)

    _clan_frame = pd.DataFrame(
        {
            "covariate": [r.covariate for r in validation.clan],
            "uncertainty": [r.unavailable_reason or "available" for r in validation.clan],
            "difference_display": [
                display_number(r.diff, "+.1%" if "=" in r.covariate else "+.3f")
                for r in validation.clan
            ],
            "standardized": [
                r.diff / float(_profile_values(r.covariate).std(ddof=1))
                if r.diff is not None
                else None
                for r in validation.clan
            ],
            "standardized_lb": [
                r.lb / float(_profile_values(r.covariate).std(ddof=1)) if r.lb is not None else None
                for r in validation.clan
            ],
            "standardized_ub": [
                r.ub / float(_profile_values(r.covariate).std(ddof=1)) if r.ub is not None else None
                for r in validation.clan
            ],
            "most_display": [
                display_number(r.mean_most, ".1%" if "=" in r.covariate else ".3f")
                for r in validation.clan
            ],
            "least_display": [
                display_number(r.mean_least, ".1%" if "=" in r.covariate else ".3f")
                for r in validation.clan
            ],
        }
    )
    _clan_table = (
        ct.CoefTable(_clan_frame, rows="covariate")
        .estimate(
            "Difference (SD)",
            "standardized",
            ci=("standardized_lb", "standardized_ub"),
            fmt=ct.Number(decimals=2, signed=True),
        )
        .forest("Standardized difference", of="Difference (SD)", ref=0.0, width=220, symmetric=True)
        .passthrough("Raw difference", "difference_display")
        .passthrough("Most affected", "most_display")
        .passthrough("Least affected", "least_display")
        .passthrough("Uncertainty", "uncertainty")
        .header(
            "CLAN profile",
            "Forest plot uses covariate differences in pooled SD units; raw values remain in the table",
        )
    )

    _clan_table
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Check the false-discovery trap

    We simulated another example to see what happens when there is no heterogeneity - do we report false positives?

    Cohort B has a constant true effect: no heterogeneity exists. A wide model on a
    smaller cohort still produces an in-sample “top” group. Held-out
    validation should reject that story, however.
    """)
    return


@app.cell
def _(Covariate, np):
    CANDIDATES = [
        Covariate(name="spend", knots=4),
        Covariate(name="tenure", knots=4),
        Covariate(name="platform", kind="categorical"),
        Covariate(name="sessions", knots=2),
        Covariate(name="referrals"),
        Covariate(name="region", kind="categorical"),
    ]
    COVARIATE_COLUMNS = ("spend", "tenure", "platform", "sessions", "referrals", "region")

    def top_group_in_sample(df, cate_result, n_groups=5):
        """The naive number: rank the fitted units by their OWN predicted
        effect, then read the difference in means inside the top group.

        Same quantile binning `validate_cate` uses - the only difference is
        that these units chose the ranking with the very outcomes being
        averaged back.
        """
        score = cate_result.score(
            {c: df[c].to_numpy() for c in COVARIATE_COLUMNS}, deploy_grain="unit"
        )
        edges = np.quantile(score, np.arange(1, n_groups) / n_groups)
        top = np.searchsorted(edges, score, side="left") == n_groups - 1
        y = df["revenue"].to_numpy()
        treated = df["variant"].to_numpy() == "treatment"
        return float(y[top & treated].mean() - y[top & ~treated].mean())

    return CANDIDATES, COVARIATE_COLUMNS, top_group_in_sample


@app.cell
def _(
    CANDIDATES,
    estimate_cate,
    from_unit_summary,
    simulate,
    top_group_in_sample,
    validate_cate,
):
    null_df = simulate(1_200, 20260813, heterogeneous=False)
    null_src = from_unit_summary(
        null_df,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
    )

    # The naive path: one fit on everything, then read its own top quintile.
    null_fit = estimate_cate(null_src, "revenue", control="control", interact=CANDIDATES)
    null_in_sample = top_group_in_sample(null_df, null_fit)

    # The honest path: fit on half, re-estimate on the other half.
    null_validation = validate_cate(
        null_src, "revenue", control="control", interact=CANDIDATES, n_groups=5
    )
    return null_fit, null_in_sample, null_src, null_validation


@app.cell(hide_code=True)
def _(BASE_TAU, display_estimate, display_number, mo, null_fit, null_in_sample, null_validation):
    _honest_top = null_validation.groups[-1]
    mo.md(
        "\n".join(
            [
                f"Every user's true effect is **{BASE_TAU:+.4f}**; the model fits "
                f"{len(null_fit.interactions)} interaction columns on 1,200 users.",
                "",
                "| estimate | value | distance from truth |",
                "| -- | -- | -- |",
                f"| In-sample top quintile | {null_in_sample:+.4f} | "
                f"**{null_in_sample - BASE_TAU:+.4f}** |",
                f"| Held-out top group | {display_number(_honest_top.effect)} "
                f"[{display_number(_honest_top.lb)}, {display_number(_honest_top.ub)}] | "
                f"{display_number(_honest_top.effect - BASE_TAU if _honest_top.effect is not None else None)} |",
                f"| Held-out ATE | {display_estimate(null_validation.holdout_ate)} | "
                f"{display_number(null_validation.holdout_ate.value - BASE_TAU if null_validation.holdout_ate is not None else None)} |",
                "",
                f"- Joint-test p-value: {null_fit.heterogeneity.p_value:.2f}",
                f"- Validation gate: **{'PASSED' if null_validation.passed else 'FAILED'}** "
                f"(AUTOC p={display_number(null_validation.autoc.p_value, '.2f')})",
            ]
        )
    )
    return


@app.cell
def _(BASE_TAU, gates_table, null_validation):
    gates_table(
        null_validation,
        BASE_TAU,
        "Cohort B - no heterogeneity at all",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    Held-out effects are flat and non-monotone, with every interval covering the
    truth. In-sample ranking creates a staircase from noise. Next, repeat the same
    comparison across 40 null cohorts.
    """)
    return


@app.cell
def _(
    CANDIDATES,
    estimate_cate,
    from_unit_summary,
    np,
    simulate,
    top_group_in_sample,
    validate_cate,
):
    _in_sample, _honest, _fired = [], [], 0
    for _seed in range(800_000, 800_040):
        df_i = simulate(1_200, _seed, heterogeneous=False)
        analysis_i = from_unit_summary(
            df_i,
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
        )
        _in_sample.append(
            top_group_in_sample(
                df_i, estimate_cate(analysis_i, "revenue", control="control", interact=CANDIDATES)
            )
        )
        _run = validate_cate(analysis_i, "revenue", control="control", interact=CANDIDATES)
        _honest.append(_run.groups[-1].effect)
        _fired += _run.passed

    replications = {
        "reps": len(_in_sample),
        "in_sample_mean": float(np.mean(_in_sample)),
        "in_sample_mc_se": float(np.std(_in_sample, ddof=1) / np.sqrt(len(_in_sample))),
        "honest_mean": float(np.mean(_honest)) if all(x is not None for x in _honest) else None,
        "honest_mc_se": (
            float(np.std(_honest, ddof=1) / np.sqrt(len(_honest)))
            if all(x is not None for x in _honest)
            else None
        ),
        "gate_fired": _fired,
    }
    return (replications,)


@app.cell(hide_code=True)
def _(BASE_TAU, display_number, mo, replications):
    _r = replications
    _gap = (
        f"{(_r['honest_mean'] - BASE_TAU) / _r['honest_mc_se']:+.1f}"
        if _r["honest_mean"] is not None and _r["honest_mc_se"]
        else None
    )
    _comparison = (
        f"{_gap} Monte Carlo SEs {'low' if _gap.startswith('-') else 'high'} of truth"
        if _gap is not None
        else "uncertainty unavailable"
    )
    mo.md(
        f"""
    Across **{_r["reps"]} null cohorts**:

    - **In-sample top group:** {_r["in_sample_mean"]:+.4f}
      ({_r["in_sample_mean"] / BASE_TAU:.1f}× truth)
    - **Held-out top group:** {display_number(_r["honest_mean"])}
      ({_comparison})
    - **Gate passes:** {_r["gate_fired"]} / {_r["reps"]} cohorts

    > A CATE model always finds a winner. Held-out re-estimation makes the ranking honest.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Build a targeting rule

    We can create a targeting rule off of a CATE model - this can be used to decide **"who should the treatment be applied to?"**

    **fraction** here decides the share of units the rule would treat. Typically it should be pre-registered to avoid bias; looking at the group table and selecting the cut based off of it can bias the policy value upwards of 60-120%. Cross validation procedures can also be used to select the optimal targeting fraction, though not shown here.
    """)
    return


@app.cell
def _(ADJUST, INTERACT, hetero_src, targeting_rule):
    rule = targeting_rule(
        hetero_src,
        "revenue",
        control="control",
        interact=INTERACT,
        adjust=ADJUST,
        fraction=0.40,
    )
    return (rule,)


@app.cell(hide_code=True)
def _(COVARIATE_COLUMNS, hetero_df, mo, rule):
    _selected = rule.predict({c: hetero_df[c].to_numpy() for c in COVARIATE_COLUMNS})
    _rule_truth = float(hetero_df.loc[_selected, "tau_true"].mean())
    _best_possible = float(
        hetero_df.loc[
            hetero_df["tau_true"] >= hetero_df["tau_true"].quantile(1.0 - rule.fraction),
            "tau_true",
        ].mean()
    )

    mo.md(
        f"""
    **Recommendation:** `{rule.recommendation}`. Treat the top {rule.fraction:.0%}
    at scores ≥ **{rule.threshold:+.4f}**.

    - **Policy value:** {rule.policy_value.value:+.4f}
      (full-sample rule truth {_rule_truth:+.4f}; oracle {_best_possible:+.4f})
    - **Uplift over treating everyone:** {rule.uplift_vs_average.value:+.4f}
    - **Validation passed:** `{rule.validation.passed}`

    Both policy numbers are point estimates with no interval, and that is
    deliberate: the same holdout both selects the rule (via the significance
    gate) and reports its value, so any nominal interval here would be
    optimistically biased by the selection. Treat them as descriptive of the
    chosen rule, not as confirmatory evidence about its magnitude.
    """
    )
    return


@app.cell
def _(CANDIDATES, null_src, targeting_rule):
    null_rule = targeting_rule(
        null_src,
        "revenue",
        control="control",
        interact=CANDIDATES,
        fraction=0.40,
    )
    return (null_rule,)


@app.cell(hide_code=True)
def _(display_estimate, display_number, mo, null_rule):
    _validation = null_rule.validation
    _autoc = _validation.autoc
    gate_detail = (
        f"AUTOC p={display_number(_autoc.p_value, '.2f')} vs alpha={_validation.alpha:.2f}"
        if _autoc.p_value is not None
        else f"AUTOC unavailable ({_autoc.unavailable_reason})"
    )
    if _validation.passed:
        threshold = display_number(null_rule.threshold)
        policy_value = display_estimate(null_rule.policy_value)
        uplift = display_estimate(null_rule.uplift_vs_average)
    else:
        threshold = "unavailable (validation gate did not authorize targeting)"
        policy_value = "unavailable (validation gate did not pass)"
        uplift = "unavailable (validation gate did not pass)"
    mo.md(
        f"""
    ### The same rule on Cohort B

    - **Recommendation:** `{null_rule.recommendation}`
    - **Validation:** {gate_detail}
    - **Targeting threshold:** {threshold}
    - **Conditional policy value:** {policy_value}
    - **Uplift versus average:** {uplift}

    Conditional policy values are points only when the gate passes; the same holdout
    both selects and reports them, so no interval is claimed.
    """
    )
    return


@app.cell(hide_code=True)
def _(fit, mo):
    _spend = next(e for e in fit.interactions if e.name == "d:spend")
    _mde = 2.80 * _spend.se
    _rows = [f"| {int(20_000 * k):,} | {_mde / k**0.5:+.4f} |" for k in (1, 4, 16)]

    mo.md(
        "\n".join(
            [
                "## 6. Practical constraints",
                "",
                "| users | smallest detectable interaction |",
                "| -- | -- |",
                *_rows,
                "",
                "- Halving the detectable interaction requires **4×** the users.",
                "- Every `interact=` and `adjust=` covariate must be measured before assignment.",
                "- Use this order: average effect → `estimate_cate` → `validate_cate` → "
                "`targeting_rule`.",
                "- If validation fails, report the average effect and stop.",
            ]
        )
    )
    return


if __name__ == "__main__":
    app.run()
