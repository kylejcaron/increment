"""Seeded repeated-assignment bias/coverage and pilot power checks."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest

from increment.estimation.cate import Covariate
from increment.longitudinal import (
    LongitudinalCohort,
    LongitudinalDesign,
    fit_longitudinal,
    prepare_longitudinal_panel,
)
from increment.power.longitudinal import (
    LongitudinalBaseline,
    longitudinal_minimum_detectable_effect,
)

pytestmark = [pytest.mark.slow, pytest.mark.parameter_recovery]


def simulated_panel(rng, *, cluster=False, adjusted=False, unbalanced=False, delta=1.0):
    cohorts = [LongitudinalCohort(name="a", exposure=0, pre_periods=(-2, -1), post_periods=(0, 1))]
    if unbalanced:
        cohorts.append(
            LongitudinalCohort(
                name="b",
                exposure=2,
                pre_periods=(0,),
                post_periods=(2, 3, 4),
                post_weights=(0.2, 0.3, 0.5),
            )
        )
    design = LongitudinalDesign(
        control_group="C",
        treatment_group="T",
        cohorts=tuple(cohorts),
        independence_grain="cluster" if cluster else "unit",
    )
    roster, rows = [], []
    for j, c in enumerate(cohorts):
        grains = 72
        treated = set(rng.choice(grains, 24, replace=False).tolist())
        for g in range(grains):
            shared = rng.normal(size=len(c.pre_periods) + len(c.post_periods))
            for member in range(1 + g % 3 if cluster else 1):
                unit = f"{j}_{g}_{member}"
                x = rng.normal()
                arm = "T" if g in treated else "C"
                roster.append(
                    {
                        "unit": unit,
                        "cluster": f"{j}_{g}",
                        "cohort": c.name,
                        "group": arm,
                        "exposure": c.exposure,
                        "x": x,
                    }
                )
                intercept = rng.normal(scale=5)
                error = rng.normal()
                for t_index, period in enumerate((*c.pre_periods, *c.post_periods)):
                    assert isinstance(period, int) and isinstance(c.exposure, int)
                    error = 0.7 * error + rng.normal()
                    effect = delta + (0.3 * x if adjusted else 0)
                    y = intercept + 2 * np.sin(period) + error + shared[t_index]
                    if adjusted:
                        y += 2.0 * x * period
                    if arm == "T" and period in c.post_periods:
                        y += effect
                    rows.append({"unit": unit, "time": period, "outcome": y})
    return pd.DataFrame(rows), pd.DataFrame(roster), design


@pytest.mark.parametrize(
    "cluster,adjusted,unbalanced",
    [(False, False, False), (False, True, True), (True, False, True), (True, True, True)],
)
def test_coverage_and_bias(cluster, adjusted, unbalanced):
    rng = np.random.default_rng(800801)
    errors = []
    covered = 0
    repetitions = 300
    for _ in range(repetitions):
        o, r, d = simulated_panel(rng, cluster=cluster, adjusted=adjusted, unbalanced=unbalanced)
        summary = prepare_longitudinal_panel(
            o,
            r,
            design=d,
            cluster="cluster" if cluster else None,
            covariates=[Covariate(name="x")] if adjusted else [],
        )
        result = fit_longitudinal(summary)
        truth = 1.0 + 0.3 * r.x.mean() if adjusted else 1.0
        errors.append(result.effect - truth)
        assert result.lower is not None and result.upper is not None
        covered += result.lower <= truth <= result.upper
    assert abs(np.mean(errors)) < 3.5 * np.std(errors, ddof=1) / np.sqrt(repetitions)
    # Exact MC uncertainty around measured coverage must include nominal 0.95.
    interval = binomtest(covered, repetitions).proportion_ci(confidence_level=0.999)
    assert interval.low <= 0.95 <= interval.high
    assert covered / repetitions >= 0.90


def test_pilot_mde_recovery():
    rng = np.random.default_rng(800802)
    o, r, d = simulated_panel(rng, delta=0.0)
    pilot = LongitudinalBaseline.from_summary(prepare_longitudinal_panel(o, r, design=d))
    delta = longitudinal_minimum_detectable_effect(pilot, n_independent=72)
    rejected = 0
    repetitions = 300
    for _ in range(repetitions):
        o, r, d = simulated_panel(rng, delta=delta)
        result = fit_longitudinal(prepare_longitudinal_panel(o, r, design=d))
        assert result.p_value is not None
        rejected += result.p_value < 0.05
    # Pilot variance is estimated once; this is a plug-in check, not exact power.
    assert abs(rejected / repetitions - 0.8) < 0.13
