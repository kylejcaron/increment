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


def _mean_summary_builder():
    from increment import MetricSpec

    return _summary_builder([MetricSpec(name="y", type="mean")])


def _retention_summary_builder():
    from increment import MetricSpec

    return _summary_builder([MetricSpec(name="y", type="retention", threshold_days=1)])


def _no_design_switchback_builder():
    import pandas as pd

    from increment import Analysis
    from increment.semantics.assignment import (
        IndependentBernoulliOrder,
        SwitchbackAssignment,
        SwitchbackWindow,
    )
    from increment.semantics.design import Randomized

    def build():
        # `from_switchback_panel` takes no `design=`: the unsupported keyword is the attempt.
        return Analysis.from_switchback_panel(
            pd.DataFrame(),
            unit="u",
            cycle="c",
            period="p",
            step="s",
            group="g",
            metrics={"y": "mean"},
            identification=Randomized(control_group="control"),
            assignment=SwitchbackAssignment(
                sequence=IndependentBernoulliOrder(probability_ct=0.5),
                window=SwitchbackWindow(washout_steps=0, observation_steps=1),
            ),
            **{"design": None},  # ty: ignore[invalid-argument-type]
        )

    return build


def _case(build, *, waive=None, codes=None, absence=None, refusal_only=False):
    """A runner-contract case: every constructor outside *build* is recorded as not attempted."""
    from tests.parity_harness.cases import CONSTRUCTORS, ParityCase

    waive = dict(waive or {})
    for name in CONSTRUCTORS:
        if name not in build and name not in waive:
            waive[name] = "SOURCE: not attempted for this runner-contract case"
    return ParityCase(
        id="runner-contract",
        build=build,
        waive=waive,
        waived_refusal_codes=codes or {},
        expected_absence=absence or {},
        refusal_only=refusal_only,
    )


_RETENTION_REFUSAL = {"from_unit_summary": "SOURCE: the summary seam carries no dates"}


def test_refusal_only_case_passes_when_every_attempted_ingress_refuses():
    case = _case(
        {"from_unit_summary": _retention_summary_builder()},
        waive=_RETENTION_REFUSAL,
        codes={"from_unit_summary": "source.frame.constructor"},
        refusal_only=True,
    )
    assert_parity(case, run_case(case))


def test_all_waived_case_without_refusal_only_still_fails():
    case = _case(
        {"from_unit_summary": _retention_summary_builder()},
        waive=_RETENTION_REFUSAL,
        codes={"from_unit_summary": "source.frame.constructor"},
    )
    result = run_case(case)
    assert result.rows == {}
    assert result.refusals == {"from_unit_summary": "source.frame.constructor"}
    with pytest.raises(pytest.fail.Exception):
        assert_parity(case, result)


def test_refusal_only_case_fails_when_a_waived_ingress_produces_rows():
    case = _case(
        {"from_unit_summary": _mean_summary_builder()},
        waive=_RETENTION_REFUSAL,
        codes={"from_unit_summary": "source.frame.constructor"},
        refusal_only=True,
    )
    result = run_case(case)
    assert set(result.rows) == {"from_unit_summary"}
    assert result.refusals == {}
    with pytest.raises(AssertionError):
        assert_parity(case, result)


def test_refusal_only_case_fails_when_an_unwaived_ingress_produces_rows():
    """`refusal_only` promises every attempted ingress refuses or is absent; an ingress
    nobody waived that returns rows self-compares as `live` and must still fail."""
    case = _case({"from_unit_summary": _mean_summary_builder()}, refusal_only=True)
    result = run_case(case)
    assert set(result.rows) == {"from_unit_summary"}
    with pytest.raises(AssertionError):
        assert_parity(case, result)


def test_expected_absence_is_attempted_and_accounted_for():
    case = _case(
        {
            "from_unit_summary": _mean_summary_builder(),
            "from_switchback_panel": _no_design_switchback_builder(),
        },
        waive={"from_switchback_panel": "SOURCE: from_switchback_panel takes no design="},
        absence={"from_switchback_panel": TypeError},
    )
    result = run_case(case)
    assert result.absences == {"from_switchback_panel": TypeError}
    assert_parity(case, result)


def test_expected_absence_that_never_occurs_fails():
    case = _case(
        {"from_unit_summary": _mean_summary_builder()},
        waive={"from_unit_summary": "SOURCE: declared absent"},
        absence={"from_unit_summary": TypeError},
    )
    result = run_case(case)
    assert set(result.rows) == {"from_unit_summary"}
    assert result.absences == {}
    with pytest.raises(AssertionError):
        assert_parity(case, result)


def test_a_downstream_error_of_the_absent_type_is_not_recorded_as_absence():
    """Absence is the constructor attempt failing; a `TypeError` raised after the
    constructor accepted the request is a defect, not a structural absence."""
    from tests.parity_harness.cases import CONSTRUCTORS, ParityCase

    def failing_probe(_results):
        raise TypeError("raised while reading, after the constructor accepted the request")

    case = ParityCase(
        id="downstream-type-error",
        build={"from_unit_summary": _mean_summary_builder()},
        waive={n: "SOURCE: not attempted" for n in CONSTRUCTORS if n != "from_unit_summary"}
        | {"from_unit_summary": "SOURCE: declared absent"},
        expected_absence={"from_unit_summary": TypeError},
        readout_probe=failing_probe,
    )
    with pytest.raises(TypeError, match="raised while reading"):
        run_case(case)


def _unavailable_lift_row(reason):
    from increment.breakout.estimates import DailyLiftEstimate

    return DailyLiftEstimate(
        metric="m",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        ds="2025-01-10",
        lift=None,
        unavailable=reason,
    )


def test_unavailable_day_axis_rows_with_different_reasons_do_not_compare_equal():
    from tests.parity_harness.runner import _normalize

    few = _normalize([_unavailable_lift_row("few_units")])
    zero = _normalize([_unavailable_lift_row("zero_variance")])
    (key,) = few
    assert few.keys() == zero.keys()
    with pytest.raises(AssertionError):
        _assert_equal(few, zero, key)


def _assert_equal(left, right, key):
    from tests.parity_harness.runner import _assert_payload_equal

    for method, payload in left[key].items():
        _assert_payload_equal("case", "left", "right", key, method, payload, right[key][method])


def _binomial_set(**overrides):
    from increment.estimation.binomial_rr import nuisance_beta
    from increment.estimation.results import BinomialConfidenceSet

    fields = {
        "lower": 1.232421875,
        "upper": None,
        "alpha": 0.05,
        "level": 0.95,
        "decision_alpha": 0.05,
        "geometry": "central",
        "x_c": 0,
        "n_c": 10,
        "x_t": 2,
        "n_t": 10,
        "nuisance_beta": nuisance_beta(0.05),
    }
    return BinomialConfidenceSet(**(fields | overrides))


@pytest.mark.parametrize(
    "changed",
    [
        {"low_reliability": True},
        {"n_treat": 3},
        {"n_control": 4},
        {"policy_name": "compiled_plan"},
        {"binomial_set": _binomial_set()},
    ],
    ids=["low_reliability", "n_treat", "n_control", "policy_name", "binomial_set"],
)
def test_day_axis_rows_differing_in_a_consumer_visible_field_do_not_compare_equal(changed):
    from tests.parity_harness.runner import _normalize

    base = _unavailable_lift_row("few_units")
    left = _normalize([base])
    right = _normalize([base.model_copy(update=changed)])
    (key,) = left
    with pytest.raises(AssertionError):
        _assert_equal(left, right, key)


def test_day_axis_exact_binomial_bounds_compare_within_numeric_tolerance():
    from tests.parity_harness.runner import _normalize

    base = _unavailable_lift_row("few_units")
    left = _normalize([base.model_copy(update={"binomial_set": _binomial_set()})])
    ulp = _normalize(
        [base.model_copy(update={"binomial_set": _binomial_set(lower=1.232421875 + 1e-12)})]
    )
    moved = _normalize([base.model_copy(update={"binomial_set": _binomial_set(lower=1.3)})])
    (key,) = left
    _assert_equal(left, ulp, key)
    with pytest.raises(AssertionError):
        _assert_equal(left, moved, key)


class _SplitDayAxis:
    """An analysis whose value methods return rows while its lift methods refuse, so a
    runner that routes a view to the wrong method, or reads two methods atomically, is
    visible in the calls it made and in the rows and refusals it kept."""

    def __init__(self):
        self.calls = []

    def _values(self, name):
        from datetime import date

        from increment.breakout.estimates import DailyMetricValue

        self.calls.append(name)
        return [
            DailyMetricValue(
                ds=date(2025, 1, 10),
                metric="m",
                group_id="treatment",
                value=None,
                unavailable="few_units",
                n=0,
            )
        ]

    def _refuse(self, name):
        from increment.errors import CapabilityError

        self.calls.append(name)
        raise CapabilityError("lift refuses", code="facade.analysis.lift_refused", context={})

    def run_daily(self, **_):
        return self._values("run_daily")

    def run_asof(self, **_):
        return self._values("run_asof")

    def run_daily_lift(self, **_):
        self._refuse("run_daily_lift")

    def run_asof_lift(self, **_):
        self._refuse("run_asof_lift")

    def close(self):
        pass


@pytest.mark.parametrize(
    ("view", "method", "refuses"),
    [
        ("daily", "run_daily", False),
        ("daily_lift", "run_daily_lift", True),
        ("asof", "run_asof", False),
        ("asof_lift", "run_asof_lift", True),
    ],
)
def test_each_day_axis_view_reads_only_its_own_method_and_keeps_its_own_outcome(
    view, method, refuses
):
    """`daily` reads `run_daily` and `daily_lift` reads `run_daily_lift` as separate legs,
    so a refusing lift never hides its value sibling's rows and vice versa."""
    from tests.parity_harness.cases import CONSTRUCTORS, ParityCase

    analysis = _SplitDayAxis()
    waive = {n: "SOURCE: not attempted" for n in CONSTRUCTORS if n != "from_unit_summary"}
    if refuses:
        waive["from_unit_summary"] = "SOURCE: the lift leg refuses"
    case = ParityCase(
        id=f"leg-{view}",
        build={"from_unit_summary": lambda: analysis},
        waive=waive,
        waived_refusal_codes=(
            {"from_unit_summary": "facade.analysis.lift_refused"} if refuses else {}
        ),
        refusal_only=refuses,
        view=view,
    )
    result = run_case(case)
    assert analysis.calls == [method]
    if refuses:
        assert result.rows == {}
        assert result.refusals == {"from_unit_summary": "facade.analysis.lift_refused"}
    else:
        assert set(result.rows) == {"from_unit_summary"}
        assert result.refusals == {}
    assert_parity(case, result)
