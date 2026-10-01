"""Mechanical parity: every PARITY_CASES row agrees across every
constructor it attempts, and every waived constructor refuses with the
exact code recorded for it. See tests/parity_harness/ for the harness
itself and docs/limitations.md's "What runs where" table for the
capability claims these cases back."""

from __future__ import annotations

import pytest

from tests.parity_harness.cases import PARITY_CASES
from tests.parity_harness.runner import assert_parity, run_case


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(case, id=case.id, marks=pytest.mark.slow if case.slow else ())
        for case in PARITY_CASES
    ],
)
def test_parity_case(case):
    result = run_case(case)
    assert_parity(case, result)


def test_parity_case_ids_are_unique():
    ids = [c.id for c in PARITY_CASES]
    assert len(ids) == len(set(ids)), f"duplicate parity case ids: {ids}"


@pytest.mark.slow
@pytest.mark.parametrize(
    "constructor",
    [
        "from_definitions",
        "from_unit_day_artifact",
        "from_unit_summary",
        "from_unit_panel",
        "from_moments",
    ],
)
def test_prior_reset_matches_independent_prior_free_analysis(constructor):
    from increment import Analysis, AnalysisPlan, MetricSpec
    from tests.analysis_factory import lift_rows
    from tests.parity_harness import dataset as ds
    from tests.parity_harness.cases import _prior_reset_case

    rows = ds.event_rows()
    treated = {row["user_id"] for row in rows if row["group_id"] == "treatment"}
    for row in rows:
        if row["user_id"] in treated and row["revenue"] is not None:
            row["revenue"] *= 2
    con = ds.duckdb_connection(rows)
    try:
        frame = ds.unit_summary_frame(con)
    finally:
        con.disconnect()
    oracle = Analysis.from_unit_summary(
        frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[
            MetricSpec(name="revenue", type="mean", missing="zero", preferred_direction="increase")
        ],
        plan=AnalysisPlan(secondaries=["revenue"]),
    )
    try:
        expected = lift_rows(oracle.run())[0]
    finally:
        oracle.close()
    assert expected.discovery is True
    assert expected.family_axes == ("metric", "arm")

    analysis = _prior_reset_case().build[constructor]()
    try:
        inherited = lift_rows(analysis.run(metrics=["revenue"]))[0]
        for _ in range(2):
            cleared = lift_rows(analysis.run(metrics=["revenue"], prior=None))[0]
            assert cleared.discovery == expected.discovery
            assert cleared.family_axes == expected.family_axes
            assert cleared.prior_shrunk == expected.prior_shrunk is False
            assert cleared.prior_spec == expected.prior_spec is None
            for field in ("value", "lb", "ub"):
                wanted = getattr(expected.require_lift(), field)
                assert wanted is not None
                assert getattr(cleared.require_lift(), field) == pytest.approx(wanted)
        restored = lift_rows(analysis.run(metrics=["revenue"]))[0]
        for row in (inherited, restored):
            assert row.discovery is None
            assert row.family_axes is None
            assert row.prior_shrunk is True
        for field in ("value", "lb", "ub"):
            assert getattr(restored.require_lift(), field) == pytest.approx(
                getattr(inherited.require_lift(), field)
            )
    finally:
        analysis.close()
        for connection in getattr(analysis, "_parity_connections", ()):
            connection.disconnect()
