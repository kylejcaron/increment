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


def _summary_builder(metrics):
    import pandas as pd

    from increment import Analysis

    def build():
        frame = pd.DataFrame(
            {
                "unit": [f"u{i}" for i in range(8)],
                "arm": ["control"] * 4 + ["treatment"] * 4,
                "y": [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 6.0],
            }
        )
        return Analysis.from_unit_summary(
            frame, unit="unit", group="arm", control="control", metrics=metrics
        )

    return build


def _retention_summary_builder():
    from increment import MetricSpec

    return _summary_builder([MetricSpec(name="y", type="retention", threshold_days=1)])


def _no_design_switchback_builder():
    import pandas as pd

    from increment import Analysis

    def build():
        return Analysis.from_switchback_panel(
            pd.DataFrame(),
            unit="u",
            cycle="c",
            period="p",
            step="s",
            group="g",
            metrics={"y": "mean"},
            identification=None,
            assignment=None,
            **{"design": None},
        )

    return build


def _refusal_only_case(**overrides):
    from tests.parity_harness.cases import ParityCase

    fields = {
        "id": "refusal-only",
        "build": {"from_unit_summary": _retention_summary_builder()},
        "waive": {"from_unit_summary": "SOURCE: the summary seam carries no dates"},
        "waived_refusal_codes": {"from_unit_summary": "source.frame.constructor"},
        "refusal_only": True,
    }
    fields.update(overrides)
    return ParityCase(**fields)


def _unattempted(case, *names):
    from tests.parity_harness.cases import CONSTRUCTORS

    reason = "SOURCE: not attempted for this runner-contract case"
    return {
        **case.waive,
        **{name: reason for name in CONSTRUCTORS if name not in case.build and name not in names},
    }


def test_refusal_only_case_passes_when_every_attempted_ingress_refuses():
    case = _refusal_only_case()
    case = _refusal_only_case(waive=_unattempted(case))
    assert_parity(case, run_case(case))


def test_all_waived_case_without_refusal_only_still_fails():
    case = _refusal_only_case()
    case = _refusal_only_case(waive=_unattempted(case), refusal_only=False)
    with pytest.raises(pytest.fail.Exception, match="nothing to compare"):
        assert_parity(case, run_case(case))


def test_refusal_only_case_fails_when_a_waived_ingress_produces_rows():
    from increment import MetricSpec

    case = _refusal_only_case(
        build={"from_unit_summary": _summary_builder([MetricSpec(name="y", type="mean")])}
    )
    case = _refusal_only_case(build=case.build, waive=_unattempted(case))
    with pytest.raises(AssertionError, match="produced rows instead of refusing"):
        assert_parity(case, run_case(case))


def test_expected_absence_is_attempted_and_accounted_for():
    from increment import MetricSpec

    case = _refusal_only_case(
        build={
            "from_unit_summary": _summary_builder([MetricSpec(name="y", type="mean")]),
            "from_switchback_panel": _no_design_switchback_builder(),
        },
        waive={"from_switchback_panel": "SOURCE: from_switchback_panel takes no design="},
        waived_refusal_codes={},
        expected_absence={"from_switchback_panel": TypeError},
        refusal_only=False,
    )
    case = _refusal_only_case(
        build=case.build,
        waive=_unattempted(case, "from_switchback_panel") | dict(case.waive),
        waived_refusal_codes={},
        expected_absence=case.expected_absence,
        refusal_only=False,
    )
    result = run_case(case)
    assert result.absences == {"from_switchback_panel": TypeError}
    assert_parity(case, result)


def test_expected_absence_that_never_occurs_fails():
    from increment import MetricSpec

    case = _refusal_only_case(
        build={"from_unit_summary": _summary_builder([MetricSpec(name="y", type="mean")])},
        waived_refusal_codes={},
        expected_absence={"from_unit_summary": TypeError},
        refusal_only=False,
    )
    case = _refusal_only_case(
        build=case.build,
        waive=_unattempted(case, "from_unit_summary") | {"from_unit_summary": "SOURCE: absent"},
        waived_refusal_codes={},
        expected_absence=case.expected_absence,
        refusal_only=False,
    )
    with pytest.raises(AssertionError, match="declared absent"):
        assert_parity(case, run_case(case))
