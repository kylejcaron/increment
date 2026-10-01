"""Conversion and retention metrics on the automatic asymptotic scalar-mean route."""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from increment.breakout.estimates import LiftEstimates
from increment.frame import MetricSpec
from increment.semantics.models import AnalysisPlan, InferenceSpec
from tests.binary_sequential_cases import (
    DESIGN,
    check_path_parity,
    event_rows,
    frame_analysis,
    unit_rows,
)

SPEC = InferenceSpec(kind="asymptotic_mean", expected_decision_sample_size=400)


@pytest.fixture(scope="module")
def duck():
    import ibis

    con = ibis.duckdb.connect()
    con.create_table("events", obj=event_rows(unit_rows()))
    return con


def test_conversion_metric_registers_scalar_mean_and_retains_bernoulli_moments():
    # 600 per arm: at the helper's default 200 the anytime-valid boundary for
    # this draw still includes zero (lb = -0.031), so no decision to assert.
    units = unit_rows(n=600)
    analysis = frame_analysis(units, SPEC)
    snapshot = analysis.capture_sequential(finalized=True)
    assert snapshot.registration.models[0].law == "scalar_mean"
    control = snapshot.arm("purchase", "control")
    successes = sum(u["purchase"] for u in units if u["variant"] == "control")
    rate = F(successes, control.n)
    assert control.mean[0] == rate
    assert control.scatter[0][0] == control.n * rate * (1 - rate)
    row = analysis.run()[0]
    assert row.inference == "asymptotic_mean"
    assert row.require_lift().lb is not None
    assert row.stat_sig()


def test_binary_secondary_family_uses_e_bh_selection_at_q():
    """In-family asymptotic-mean secondaries face the same e-BH selection as
    every other registered family: selection at ``q`` over the retained
    roster's log evidence, reinversion at ``min(q*R/m, nominal_alpha)`` for a
    selected cell, and a family-wide guarantee label. Only the primary's own
    per-cell allocation is unaffected; it never joins the family.
    """
    import pandas as pd

    from increment import Analysis
    from increment.estimation.decision_types import sequential_hypothesis_key
    from increment.estimation.family import e_bh_select, select_sequential_family
    from increment.estimation.sequential import AsymptoticMean
    from increment.estimation.sequential_runtime import estimate_sequential

    rows = []
    for i in range(200):
        for arm in ("control", "treatment"):
            t = arm == "treatment"
            rows.append(
                {
                    "unit": f"{i:04d}-{arm}",
                    "arm": arm,
                    "checkout": int(i % 10 < (6 if t else 3)),
                    "signups": int(i % 5 < (4 if t else 2)),
                    "sessions": int(i % 2 == 0),
                    "exposure": i,
                }
            )
    specs = [MetricSpec(name=n, type="conversion") for n in ("checkout", "signups", "sessions")]
    plan = AnalysisPlan(
        primary="checkout", secondaries=["signups", "sessions"], q=0.08, inference=SPEC
    )
    analysis = Analysis.from_unit_summary(
        pd.DataFrame(rows),
        unit="unit",
        group="arm",
        metrics=specs,
        experiment_id="exp",
        exposure_date="exposure",
        design=DESIGN,
        plan=plan,
    )
    rows_out = analysis.run()
    assert isinstance(rows_out, LiftEstimates)
    rows_by_metric = {row.metric: row for row in rows_out}
    primary = rows_by_metric["checkout"].require_asymptotic_sequential_result()
    assert primary.decision_alpha == F(0.05)
    assert rows_by_metric["checkout"].discovery is None
    assert rows_by_metric["checkout"].family_threshold is None

    # Recompute the family's selection and reinversion independently of
    # `run()`, from the same verified snapshot, so this pins the e-BH formula
    # rather than whatever value `run()` happened to emit.
    snapshot = analysis.capture_sequential(finalized=True)
    policy = AsymptoticMean(registration=snapshot.registration)
    computation = estimate_sequential(snapshot, policy)
    by_metric = {row.metric: row for row in computation.results}
    cells = [
        (sequential_hypothesis_key(row.require_sequential_result().checkpoint.cell), row)
        for row in computation.results
        if row.require_sequential_result().checkpoint.cell.family
    ]
    order = [key for key, _ in cells]
    log_e_values = [row.require_asymptotic_sequential_result().log_e for _, row in cells]
    expected_selected = {order[i] for i in e_bh_select(log_e_values, snapshot.registration.q)}
    outcome = select_sequential_family(
        cells, snapshot.registration.q, policy, F(0.05), computation=computation
    )
    assert set(outcome.selected) == expected_selected
    # Exactly one of the two secondaries clears q * R / m at this draw -- a
    # selection of 0 or 2 would not exercise the FCR reinversion at all.
    assert len(expected_selected) == 1
    realized = snapshot.registration.q * len(expected_selected) / len(order)
    assert outcome.realized_threshold == float(realized)
    assert outcome.fcr_alpha == min(realized, F(0.05))
    assert outcome.guarantee == "asymptotic_sequential"

    for name in ("signups", "sessions"):
        row = rows_by_metric[name]
        assert row.family_axes == ("metric", "arm")
        assert row.family_q == float(snapshot.registration.q)
        assert row.family_threshold == outcome.realized_threshold
        assert row.family_guarantee == "asymptotic_sequential"
        assert row.family_nominal_alpha == 0.05
        selected = (
            sequential_hypothesis_key(row.require_asymptotic_sequential_result().checkpoint.cell)
            in expected_selected
        )
        assert row.discovery == selected
        result = row.require_asymptotic_sequential_result()
        if selected:
            assert result.decision_alpha == outcome.fcr_alpha
            # An unselected cell keeps its own registered allocation, never
            # the family's reinverted alpha.
            registered = by_metric[name].require_asymptotic_sequential_result()
            assert result.decision_alpha == registered.decision_alpha
    assert {rows_by_metric[n].discovery for n in ("signups", "sessions")} == {True, False}


def test_conversion_state_and_interval_agree_on_every_path(duck):
    check_path_parity(duck, "duckdb", "events", "asymptotic_mean")
