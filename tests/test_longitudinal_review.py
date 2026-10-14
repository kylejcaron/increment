from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from increment.errors import InvalidRequestError
from increment.estimation.cate import Covariate
from increment.longitudinal import (
    LongitudinalCohort,
    LongitudinalDesign,
    fit_longitudinal,
    prepare_longitudinal_panel,
)
from increment.power.longitudinal import (
    LongitudinalBaseline,
    longitudinal_achieved_power,
    longitudinal_minimum_detectable_effect,
)
from increment.query.longitudinal import prepare_longitudinal_query
from tests._longitudinal_data import panel_data
from tests.oracles._longitudinal_decimal import balanced_adjusted_reference


@pytest.mark.parametrize("producer", [prepare_longitudinal_panel, prepare_longitudinal_query])
def test_empty_cluster_column_refuses_before_reading_data(producer):
    with pytest.raises(InvalidRequestError) as error:
        producer(None, None, design=single_cohort(cluster=True), cluster="")
    assert error.value.code == "longitudinal.input"


def test_tiny_pilot_variance_keeps_representable_projected_standard_error():
    o, r, d = panel_data()
    o.outcome *= 1e-155
    pilot = LongitudinalBaseline.from_summary(prepare_longitudinal_panel(o, r, design=d))
    expected = np.sqrt(pilot.variance_per_independent) / np.sqrt(2**53)
    assert pilot.standard_error(2**53) == pytest.approx(expected, rel=1e-12, abs=0)
    assert pilot.standard_error(2**53) > 0
    power = longitudinal_achieved_power(pilot, n_independent=2**53, delta=1e-164)
    assert 0.05 <= power <= 1


def test_overflowing_period_weights_use_coded_refusal():
    with pytest.raises(InvalidRequestError) as error:
        LongitudinalCohort(
            name="a",
            exposure=0,
            pre_periods=(-1,),
            post_periods=(0, 1),
            post_weights=(1e308, 1e308),
        )
    assert error.value.code == "longitudinal.input"


def single_cohort(*, cluster=False):
    return LongitudinalDesign(
        control_group="C",
        treatment_group="T",
        cohorts=(LongitudinalCohort(name="a", exposure=1, pre_periods=(0,), post_periods=(1,)),),
        independence_grain="cluster" if cluster else "unit",
    )


@pytest.mark.integration
@pytest.mark.parametrize("slope", [1e14, 1e15])
def test_large_slope_small_residual_refuses_unresolved_accuracy(slope):
    import ibis

    x = np.tile(np.arange(20), 2)
    y = slope * x + np.random.default_rng(36).normal(0, 4, 40) + np.repeat([0, 2], 20)
    effect, se = balanced_adjusted_reference(y)
    if slope == 1e15:
        assert effect == pytest.approx(2.576970301057194, abs=1e-12)
        assert se == pytest.approx(0.9087987057231192, abs=1e-12)
    r = pd.DataFrame(
        {
            "unit": range(40),
            "group": ["C"] * 20 + ["T"] * 20,
            "cohort": ["a"] * 40,
            "exposure": [1] * 40,
            "x": x,
        }
    )
    o = pd.DataFrame(
        [
            {"unit": u, "time": t, "outcome": 0.0 if t == 0 else y[u]}
            for u in range(40)
            for t in (0, 1)
        ]
    )
    d = single_cohort()
    kwargs: dict[str, Any] = {"design": d, "covariates": [Covariate(name="x")]}
    eager = prepare_longitudinal_panel(o, r, **kwargs)
    con = ibis.duckdb.connect()
    try:
        warehouse = prepare_longitudinal_query(
            con.create_table("o", o), con.create_table("r", r), **kwargs
        )
    finally:
        con.disconnect()
    for summary in (eager, warehouse):
        with pytest.raises(InvalidRequestError) as error:
            fit_longitudinal(summary)
        assert error.value.code == "longitudinal.numeric"


def test_adjusted_cluster_pilot_reproduces_its_own_standard_error():
    r = pd.DataFrame(
        [
            {
                "unit": 3 * g + x,
                "group": "C" if g < 2 else "T",
                "cluster": str(g),
                "cohort": "a",
                "exposure": 1,
                "x": x,
            }
            for g in range(4)
            for x in range(3)
        ]
    )
    o = pd.DataFrame(
        [
            {"unit": 3 * g + x, "time": t, "outcome": float(2 * x + g + (g == 3)) if t else 0.0}
            for g in range(4)
            for x in range(3)
            for t in (0, 1)
        ]
    )
    summary = prepare_longitudinal_panel(
        o,
        r,
        design=single_cohort(cluster=True),
        cluster="cluster",
        covariates=[Covariate(name="x")],
    )
    result = fit_longitudinal(summary)
    assert result.effect == pytest.approx(2.5)
    assert result.standard_error == pytest.approx(0.9128709291752768)
    baseline = LongitudinalBaseline.from_summary(summary)
    assert baseline.standard_error(4) == pytest.approx(result.standard_error)
    delta = longitudinal_minimum_detectable_effect(baseline, n_independent=4)
    assert longitudinal_achieved_power(baseline, n_independent=4, delta=delta) == pytest.approx(
        0.8, abs=1e-9
    )


@pytest.mark.integration
@pytest.mark.parametrize("multipliers", [(1, 2, -3), (1, 3, -4), (1, 7, -8), (1, 1, -2)])
def test_zero_variance_across_cohorts_stays_unavailable(multipliers):
    import ibis

    r = []
    o = []
    for j, q in enumerate(multipliers):
        for g in range(6):
            u = f"{j}-{g}"
            r.append(
                {
                    "unit": u,
                    "group": "C" if g < 3 else "T",
                    "cluster": str(g),
                    "cohort": str(j),
                    "exposure": 1,
                }
            )
            for t in (0, 1):
                o.append(
                    {"unit": u, "time": t, "outcome": float(q * (g % 3) + (g >= 3)) if t else 0.0}
                )
    r = pd.DataFrame(r)
    o = pd.DataFrame(o)
    d = LongitudinalDesign(
        control_group="C",
        treatment_group="T",
        cohorts=tuple(
            LongitudinalCohort(name=str(j), exposure=1, pre_periods=(0,), post_periods=(1,))
            for j in range(3)
        ),
        independence_grain="cluster",
    )
    eager = prepare_longitudinal_panel(o, r, design=d, cluster="cluster")
    con = ibis.duckdb.connect()
    try:
        warehouse = prepare_longitudinal_query(
            con.create_table("o", o), con.create_table("r", r), design=d, cluster="cluster"
        )
    finally:
        con.disconnect()
    for summary in (eager, warehouse):
        result = fit_longitudinal(summary)
        assert result.effect == pytest.approx(1.0, abs=1e-14)
        assert result.standard_error is result.lower is result.upper is result.p_value is None
        assert result.unavailable_reason == "zero_variance"


def test_small_positive_contrast_variance_is_not_clipped_to_zero():
    r = []
    o = []
    for j, q in enumerate((1, 2, -3)):
        for g in range(6):
            u = f"{j}-{g}"
            r.append(
                {
                    "unit": u,
                    "group": "C" if g < 3 else "T",
                    "cluster": str(g),
                    "cohort": str(j),
                    "exposure": 1,
                }
            )
            for t in (0, 1):
                value = float(q * (g % 3) + (g >= 3)) if t else 0.0
                if j == 0 and g == 1 and t == 1:
                    value = np.nextafter(value, np.inf)
                o.append({"unit": u, "time": t, "outcome": value})
    d = LongitudinalDesign(
        control_group="C",
        treatment_group="T",
        cohorts=tuple(
            LongitudinalCohort(name=str(j), exposure=1, pre_periods=(0,), post_periods=(1,))
            for j in range(3)
        ),
        independence_grain="cluster",
    )
    result = fit_longitudinal(
        prepare_longitudinal_panel(pd.DataFrame(o), pd.DataFrame(r), design=d, cluster="cluster")
    )
    assert result.standard_error is not None and result.standard_error > 0
    assert result.unavailable_reason is None


@pytest.mark.parametrize("scale", [3e154, 1e-200])
def test_unrepresentable_contrast_variance_refuses_with_numeric_code(scale):
    r = pd.DataFrame(
        {"unit": range(6), "group": ["C"] * 3 + ["T"] * 3, "cohort": ["a"] * 6, "exposure": [1] * 6}
    )
    y = [0.0, scale, 2 * scale] * 2
    o = pd.DataFrame(
        [{"unit": u, "time": t, "outcome": y[u] if t else 0.0} for u in range(6) for t in (0, 1)]
    )
    with pytest.raises(InvalidRequestError) as error:
        fit_longitudinal(prepare_longitudinal_panel(o, r, design=single_cohort()))
    assert error.value.code == "longitudinal.numeric"
