from __future__ import annotations

import copy
import datetime as dt
import pickle
from typing import Any

import numpy as np
import pytest

from increment import InvalidRequestError
from increment.estimation.cate import Covariate
from increment.longitudinal import (
    LongitudinalCohort,
    LongitudinalDesign,
    LongitudinalSummary,
    fit_longitudinal,
    prepare_longitudinal_panel,
)
from increment.power.longitudinal import (
    LongitudinalBaseline,
    longitudinal_achieved_power,
    longitudinal_minimum_detectable_effect,
    longitudinal_required_independent,
)
from increment.query.longitudinal import prepare_longitudinal_query
from tests._longitudinal_data import panel_data
from tests.oracles._longitudinal_reference import dense_panel_reference


@pytest.mark.parametrize(
    "unbalanced,clustered,adjusted,heterogeneous",
    [
        (False, False, False, False),
        (True, False, False, True),
        (True, True, False, False),
        (True, False, True, True),
        (False, True, True, True),
    ],
)
def test_dense_saturated_panel_reference(unbalanced, clustered, adjusted, heterogeneous):
    o, r, d = panel_data(
        unbalanced=unbalanced, clustered=clustered, adjusted=adjusted, heterogeneous=heterogeneous
    )
    covariates = (
        [Covariate(name="x"), Covariate(name="kind", kind="categorical")] if adjusted else []
    )
    summary = prepare_longitudinal_panel(
        o, r, design=d, cluster="cluster" if clustered else None, covariates=covariates
    )
    result = fit_longitudinal(summary)
    effect, se = dense_panel_reference(o, r, d, adjusted=adjusted, clustered=clustered)
    assert result.effect == pytest.approx(effect, abs=1e-10)
    assert result.standard_error == pytest.approx(se, rel=1e-9)
    assert result.n_independent == len(r) // (2 if clustered else 1)
    assert result.n_units == len(r)
    assert result.n_observations == len(o)
    assert result.multiplicity_status == "exploratory_unadjusted"


@pytest.mark.integration
@pytest.mark.parametrize(
    "clustered,adjusted,timestamp",
    [(False, False, False), (True, True, False), (False, True, True)],
)
def test_frame_query_parity(clustered, adjusted, timestamp):
    import ibis

    o, r, d = panel_data(
        unbalanced=True, clustered=clustered, adjusted=adjusted, heterogeneous=True
    )
    if timestamp:

        def convert(t):
            return dt.datetime(2026, 1, 1) + dt.timedelta(days=t)

        o.time = o.time.map(convert)
        r.exposure = r.exposure.map(convert)
        d = d.model_copy(
            update={
                "cohorts": tuple(
                    LongitudinalCohort(
                        name=c.name,
                        exposure=convert(c.exposure),
                        pre_periods=tuple(map(convert, c.pre_periods)),
                        post_periods=tuple(map(convert, c.post_periods)),
                        post_weights=c.post_weights,
                    )
                    for c in d.cohorts
                )
            }
        )
    covariates = (
        [Covariate(name="x"), Covariate(name="kind", kind="categorical")] if adjusted else []
    )
    args: dict[str, Any] = {
        "design": d,
        "cluster": "cluster" if clustered else None,
        "covariates": covariates,
    }
    frame = prepare_longitudinal_panel(o, r, **args)
    con = ibis.duckdb.connect()
    try:
        query = prepare_longitudinal_query(
            con.create_table("outcomes", o), con.create_table("roster", r), **args
        )
    finally:
        con.disconnect()
    left, right = fit_longitudinal(frame), fit_longitudinal(query)
    for field in ("effect", "standard_error", "lower", "upper", "p_value"):
        assert getattr(left, field) == pytest.approx(getattr(right, field), abs=1e-10)
    assert np.asarray(left.covariance) == pytest.approx(np.asarray(right.covariance), abs=1e-10)
    assert left.design == right.design
    assert left.n_independent == right.n_independent


@pytest.mark.parametrize(
    "change,code",
    [
        ("missing", "longitudinal.observation"),
        ("extra", "longitudinal.observation"),
        ("duplicate", "longitudinal.observation"),
        ("unknown", "longitudinal.observation"),
        ("exposure", "longitudinal.input"),
        ("arm", "longitudinal.input"),
        ("nan", "longitudinal.input"),
        ("cluster", "longitudinal.input"),
    ],
)
@pytest.mark.integration
def test_observation_and_roster_refusals_match(change, code):
    import ibis
    import pandas as pd

    o, r, d = panel_data(clustered=True)
    if change == "missing":
        o = o.iloc[1:]
    if change == "extra":
        o.loc[0, "time"] = 99
    if change == "duplicate":
        o = pd.concat([o, o.iloc[:1]])
    if change == "unknown":
        o.loc[0, "unit"] = "unknown"
    if change == "exposure":
        r.loc[0, "exposure"] = 1
    if change == "arm":
        r.loc[0, "group"] = "other"
    if change == "nan":
        o.loc[0, "outcome"] = np.nan
    if change == "cluster":
        r.loc[0, "group"] = "T"
    with pytest.raises(InvalidRequestError) as eager:
        prepare_longitudinal_panel(o, r, design=d, cluster="cluster")
    con = ibis.duckdb.connect()
    try:
        with pytest.raises(InvalidRequestError) as warehouse:
            prepare_longitudinal_query(
                con.create_table("o", o), con.create_table("r", r), design=d, cluster="cluster"
            )
    finally:
        con.disconnect()
    assert eager.value.code == warehouse.value.code == code
    restored = pickle.loads(pickle.dumps(eager.value))
    assert restored.code == copy.deepcopy(eager.value).code == code
    assert restored.context == eager.value.context


def test_owned_evidence_replays_after_pickle_and_input_mutation():
    o, r, d = panel_data(adjusted=True)
    summary = prepare_longitudinal_panel(o, r, design=d, covariates=[Covariate(name="x")])
    expected = fit_longitudinal(summary)
    o.outcome = 0
    r.x = 0
    for restored in (
        pickle.loads(pickle.dumps(summary)),
        copy.deepcopy(summary),
        LongitudinalSummary.model_validate(summary.model_dump()),
    ):
        assert fit_longitudinal(restored) == expected


def test_pilot_runtime_and_solver_equivalence():
    o, r, d = panel_data(unbalanced=True, adjusted=True)
    summary = prepare_longitudinal_panel(o, r, design=d, covariates=[Covariate(name="x")])
    result = fit_longitudinal(summary)
    pilot = LongitudinalBaseline.from_summary(summary)
    assert result.standard_error is not None
    assert pilot.standard_error(result.n_independent) == pytest.approx(result.standard_error)
    assert pilot.standard_error(result.n_independent * 4) == pytest.approx(
        result.standard_error / 2
    )
    delta = longitudinal_minimum_detectable_effect(pilot, n_independent=192)
    assert longitudinal_achieved_power(pilot, n_independent=192, delta=delta) == pytest.approx(
        0.8, abs=1e-9
    )
    n = longitudinal_required_independent(pilot, delta=0.8)
    assert longitudinal_achieved_power(pilot, n_independent=n, delta=0.8) >= 0.8
    assert longitudinal_achieved_power(pilot, n_independent=n - 1, delta=0.8) < 0.8
    with pytest.raises(InvalidRequestError):
        pilot.standard_error(4)


def test_covariate_estimability_and_missing_history():
    o, r, d = panel_data()
    r.x = 1.0
    with pytest.raises(InvalidRequestError):
        prepare_longitudinal_panel(o, r, design=d, covariates=[Covariate(name="x")])
    r.x = np.arange(len(r), dtype=float)
    r["other"] = r.x
    summary = prepare_longitudinal_panel(
        o, r, design=d, covariates=[Covariate(name="x"), Covariate(name="other")]
    )
    with pytest.raises(InvalidRequestError, match="rank deficient"):
        fit_longitudinal(summary)
    r.loc[0, "x"] = np.nan
    with pytest.raises(InvalidRequestError):
        prepare_longitudinal_panel(o, r, design=d, covariates=[Covariate(name="x")])


def test_large_offsets_and_order_preserve_inference():
    o, r, d = panel_data(unbalanced=True)
    o.outcome = np.round(o.outcome * 8) / 8
    r.x = np.arange(len(r), dtype=float) / 8
    args: dict[str, Any] = {"design": d, "covariates": [Covariate(name="x")]}
    expected = fit_longitudinal(prepare_longitudinal_panel(o, r, **args))
    o.outcome += 2**40
    r.x += 2**40
    result = fit_longitudinal(
        prepare_longitudinal_panel(
            o.sample(frac=1, random_state=1), r.sample(frac=1, random_state=2), **args
        )
    )
    assert result.effect == pytest.approx(expected.effect, abs=1e-10)
    assert result.standard_error == pytest.approx(expected.standard_error, abs=1e-10)


def test_degenerate_uncertainty_is_unavailable():
    o, r, d = panel_data()
    o.outcome = 1.0
    result = fit_longitudinal(prepare_longitudinal_panel(o, r, design=d))
    assert result.effect == 0
    assert result.standard_error is result.lower is result.upper is result.p_value is None
    assert result.unavailable_reason == "zero_variance"
    with pytest.raises(InvalidRequestError):
        LongitudinalBaseline.from_summary(prepare_longitudinal_panel(o, r, design=d))


def test_no_precision_gain_is_a_valid_outcome():
    o, r, d = panel_data()
    # Unrelated noisy history can worsen precision: preserve its actual uncertainty.
    rng = np.random.default_rng(8009)
    o.outcome = rng.normal(size=len(o))
    result = fit_longitudinal(prepare_longitudinal_panel(o, r, design=d))
    post = o[o.time >= 0].merge(r[["unit", "group"]], on="unit")
    post_mean = post.groupby(["unit", "group"]).outcome.mean().reset_index()
    variance = (
        sum(post_mean[post_mean.group == arm].outcome.var(ddof=0) / 24 for arm in ("C", "T"))
        * 48
        / 47
    )
    assert result.standard_error is not None
    assert result.standard_error > np.sqrt(variance)
    effect, se = dense_panel_reference(o, r, d)
    assert result.effect == pytest.approx(effect, abs=1e-10)
    assert result.standard_error == pytest.approx(se, rel=1e-9)


def test_more_timepoints_are_not_more_independent_grains():
    o, r, d = panel_data()
    initial = fit_longitudinal(prepare_longitudinal_panel(o, r, design=d))
    extra = o[o.time == 1].copy()
    extra.time = 2
    import pandas as pd

    o = pd.concat([o, extra])
    c = d.cohorts[0]
    updated = LongitudinalCohort(
        name=c.name,
        exposure=c.exposure,
        pre_periods=c.pre_periods,
        post_periods=(0, 1, 2),
        post_weights=(0.25, 0.375, 0.375),
    )
    extended = d.model_copy(update={"cohorts": (updated,)})
    result = fit_longitudinal(prepare_longitudinal_panel(o, r, design=extended))
    assert result.n_independent == initial.n_independent
    assert result.n_observations > initial.n_observations
    assert result.effect == pytest.approx(initial.effect, abs=1e-12)
    assert result.standard_error == pytest.approx(initial.standard_error, abs=1e-12)


@pytest.mark.parametrize("backend", ["polars", "arrow"])
def test_native_frame_ingress_parity(backend):
    o, r, d = panel_data(adjusted=True)
    args: dict[str, Any] = {
        "design": d,
        "covariates": [Covariate(name="x"), Covariate(name="kind", kind="categorical")],
    }
    expected = fit_longitudinal(prepare_longitudinal_panel(o, r, **args))
    if backend == "polars":
        import polars as pl

        o_native, r_native = pl.from_pandas(o), pl.from_pandas(r)
    else:
        import pyarrow as pa

        o_native, r_native = pa.Table.from_pandas(o), pa.Table.from_pandas(r)
    result = fit_longitudinal(prepare_longitudinal_panel(o_native, r_native, **args))
    assert result == expected


def test_schedules_and_planning_refuse_invalid_requests_before_data():
    with pytest.raises(InvalidRequestError):
        LongitudinalCohort(name="a", exposure=0, pre_periods=(1,), post_periods=(2,))
    with pytest.raises(InvalidRequestError):
        LongitudinalCohort(
            name="a", exposure=0, pre_periods=(-1,), post_periods=(0, 1), post_weights=(0.4, 0.4)
        )
    o, r, d = panel_data()
    summary = prepare_longitudinal_panel(o, r, design=d)
    for alpha in (0, 1, float("nan"), True, np.nextafter(0.0, 1.0)):
        with pytest.raises(InvalidRequestError):
            fit_longitudinal(summary, alpha=alpha)
    pilot = LongitudinalBaseline.from_summary(summary)
    for n in (True, 3, 4.5):
        with pytest.raises(InvalidRequestError):
            pilot.standard_error(n)  # ty: ignore[invalid-argument-type]
    with pytest.raises(InvalidRequestError):
        longitudinal_required_independent(pilot, delta=0)


@pytest.mark.integration
def test_large_offsets_query_covariance_parity():
    import ibis

    o, r, d = panel_data(adjusted=True)
    o.outcome = np.round(o.outcome * 8) / 8 + 2**40
    r.x = np.round(r.x * 8) / 8 + 2**40
    args: dict[str, Any] = {"design": d, "covariates": [Covariate(name="x")]}
    expected = fit_longitudinal(prepare_longitudinal_panel(o, r, **args))
    con = ibis.duckdb.connect()
    try:
        result = fit_longitudinal(
            prepare_longitudinal_query(con.create_table("o", o), con.create_table("r", r), **args)
        )
    finally:
        con.disconnect()
    assert result.effect == pytest.approx(expected.effect, abs=1e-10)
    assert np.array(result.covariance) == pytest.approx(np.array(expected.covariance), abs=1e-10)


def test_neighboring_large_changes_preserve_small_effect():
    import pandas as pd

    origin = 2**46
    values = np.array([0.0, 0.125, 0.0, 0.25, 0.5, 0.25])
    r = pd.DataFrame(
        {"unit": range(6), "group": ["C"] * 3 + ["T"] * 3, "cohort": ["a"] * 6, "exposure": [0] * 6}
    )
    o = pd.DataFrame(
        [
            {"unit": i, "time": t, "outcome": 0.0 if t == -1 else origin + values[i]}
            for i in range(6)
            for t in (-1, 0)
        ]
    )
    d = LongitudinalDesign(
        control_group="C",
        treatment_group="T",
        cohorts=(LongitudinalCohort(name="a", exposure=0, pre_periods=(-1,), post_periods=(0,)),),
    )
    result = fit_longitudinal(prepare_longitudinal_panel(o, r, design=d))
    assert result.effect == pytest.approx(values[3:].mean() - values[:3].mean(), abs=1e-14)
    assert result.cohort_effects == pytest.approx((result.effect,), abs=1e-14)


def test_sparse_levels_cannot_hide_independent_uncertainty():
    import pandas as pd

    r = pd.DataFrame(
        {
            "unit": range(8),
            "group": ["C"] * 4 + ["T"] * 4,
            "cohort": ["a"] * 8,
            "exposure": [0] * 8,
            "kind": ["a", "a", "a", "b"] * 2,
        }
    )
    o = pd.DataFrame(
        [
            {"unit": i, "time": t, "outcome": 0.0 if t == -1 else [0, 1, 2, 0, 0, 1, 2, 100][i]}
            for i in range(8)
            for t in (-1, 0)
        ]
    )
    d = LongitudinalDesign(
        control_group="C",
        treatment_group="T",
        cohorts=(LongitudinalCohort(name="a", exposure=0, pre_periods=(-1,), post_periods=(0,)),),
    )
    summary = prepare_longitudinal_panel(
        o, r, design=d, covariates=[Covariate(name="kind", kind="categorical")]
    )
    with pytest.raises(InvalidRequestError, match="one independent grain"):
        fit_longitudinal(summary)


def test_summary_cannot_change_arms_or_roster_counts():
    o, r, d = panel_data()
    summary = prepare_longitudinal_panel(o, r, design=d)
    changed = summary.model_dump()
    changed["clusters"] = tuple({**g, "arm": "C"} for g in changed["clusters"])
    with pytest.raises(InvalidRequestError):
        LongitudinalSummary.model_validate(changed)
    changed = summary.model_dump()
    changed["cohort_counts"] = (len(r) + 1,)
    with pytest.raises(InvalidRequestError):
        LongitudinalSummary.model_validate(changed)


@pytest.mark.integration
@pytest.mark.parametrize("metadata", ["group", "cohort", "unit", "cluster"])
def test_nullable_roster_metadata_uses_matching_coded_refusals(metadata):
    import ibis
    import pandas as pd

    o, r, d = panel_data(clustered=True)
    r[metadata] = r[metadata].astype("string")
    r.loc[0, metadata] = pd.NA
    with pytest.raises(InvalidRequestError) as frame:
        prepare_longitudinal_panel(o, r, design=d, cluster="cluster")
    con = ibis.duckdb.connect()
    try:
        with pytest.raises(InvalidRequestError) as query:
            prepare_longitudinal_query(
                con.create_table("o", o), con.create_table("r", r), design=d, cluster="cluster"
            )
    finally:
        con.disconnect()
    assert frame.value.code == query.value.code == "longitudinal.input"


def test_summary_cannot_change_covariate_target():
    o, r, d = panel_data()
    summary = prepare_longitudinal_panel(o, r, design=d, covariates=[Covariate(name="x")])
    changed = summary.model_dump()
    # Reparameterize every covariate x to x+1 while keeping valid PSD/count blocks.
    transform = np.eye(4)
    transform[0, 1] = transform[2, 3] = 1.0
    for g in changed["clusters"]:
        gram = np.asarray(g["gram"])
        g["gram"] = tuple(tuple(row) for row in transform.T @ gram @ transform)
        g["cross"] = tuple(transform.T @ np.asarray(g["cross"]))
    with pytest.raises(InvalidRequestError, match="centered"):
        LongitudinalSummary.model_validate(changed)


def test_unrepresentable_covariance_uses_coded_refusal():
    import pandas as pd

    values = [1e308, 1e308, 1e308, 1.1e308, 1.2e308, 1.3e308]
    r = pd.DataFrame(
        {"unit": range(6), "group": ["C"] * 3 + ["T"] * 3, "cohort": ["a"] * 6, "exposure": [0] * 6}
    )
    o = pd.DataFrame(
        [
            {"unit": i, "time": t, "outcome": 0.0 if t == -1 else values[i]}
            for i in range(6)
            for t in (-1, 0)
        ]
    )
    d = LongitudinalDesign(
        control_group="C",
        treatment_group="T",
        cohorts=(LongitudinalCohort(name="a", exposure=0, pre_periods=(-1,), post_periods=(0,)),),
    )
    with pytest.raises(InvalidRequestError) as error:
        fit_longitudinal(prepare_longitudinal_panel(o, r, design=d))
    assert error.value.code == "longitudinal.numeric"


@pytest.mark.integration
def test_high_cardinality_refuses_before_expanding_basis():
    import ibis

    o, r, d = panel_data()
    r.kind = r.unit
    args: dict[str, Any] = {"design": d, "covariates": [Covariate(name="kind", kind="categorical")]}
    with pytest.raises(InvalidRequestError, match="48 regression") as eager:
        prepare_longitudinal_panel(o, r, **args)
    con = ibis.duckdb.connect()
    try:
        with pytest.raises(InvalidRequestError, match="48 regression") as query:
            prepare_longitudinal_query(con.create_table("o", o), con.create_table("r", r), **args)
    finally:
        con.disconnect()
    assert eager.value.code == query.value.code == "longitudinal.input"
