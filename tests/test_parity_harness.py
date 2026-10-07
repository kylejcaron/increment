"""Mechanical parity: every PARITY_CASES row agrees across every
constructor it attempts, and every waived constructor refuses with the
exact code recorded for it. See tests/parity_harness/ for the harness
itself and docs/reference/capabilities-by-entry-point.md's "What runs where" table for
the capability claims these cases back."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from increment import Analysis

from tests.parity_harness.cases import PARITY_CASES, Absence
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


def test_refusal_only_case_that_attempts_no_ingress_fails():
    """Both sides of the refusal set equality are empty when nothing is built, so the case
    would pass having attempted no ingress at all."""
    case = _case({}, refusal_only=True)
    result = run_case(case)
    assert result.rows == {}
    assert result.refusals == {}
    assert result.absences == {}
    with pytest.raises(AssertionError):
        assert_parity(case, result)


def test_expected_absence_is_attempted_and_accounted_for():
    case = _case(
        {
            "from_unit_summary": _mean_summary_builder(),
            "from_switchback_panel": _no_design_switchback_builder(),
        },
        waive={"from_switchback_panel": "SOURCE: from_switchback_panel takes no design="},
        absence={"from_switchback_panel": Absence(TypeError, "design")},
    )
    result = run_case(case)
    assert result.absences == {"from_switchback_panel": Absence(TypeError, "design")}
    assert_parity(case, result)


def test_expected_absence_that_never_occurs_fails():
    case = _case(
        {"from_unit_summary": _mean_summary_builder()},
        waive={"from_unit_summary": "SOURCE: declared absent"},
        absence={"from_unit_summary": Absence(TypeError, "design")},
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
        expected_absence={"from_unit_summary": Absence(TypeError, "design")},
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


def _binomial_set(*, lower: float = 1.232421875):
    """A zero-control-count exact-binomial set (no finite point); only `lower` varies."""
    from increment.estimation.binomial_rr import nuisance_beta
    from increment.estimation.results import (
        BINOMIAL_METHOD,
        BINOMIAL_NUMERICAL_QUALIFICATION,
        BinomialConfidenceSet,
    )

    return BinomialConfidenceSet(
        lower=lower,
        upper=None,
        alpha=0.05,
        level=0.95,
        decision_alpha=0.05,
        geometry="central",
        method=BINOMIAL_METHOD,
        numerical_qualification=BINOMIAL_NUMERICAL_QUALIFICATION,
        x_c=0,
        n_c=10,
        x_t=2,
        n_t=10,
        nuisance_beta=nuisance_beta(0.05),
    )


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


_INF = float("inf")


@pytest.mark.parametrize(
    ("expected", "actual"),
    [
        (-_INF, 0.0),
        (-_INF, -1e300),
        (_INF, 1e300),
        (0.0, -_INF),
        (1.0, _INF),
        (-_INF, _INF),
        (_INF, -_INF),
        (-_INF, 3),
        (float("nan"), float("nan")),
        (float("nan"), 1.0),
        (1.0, float("nan")),
        (float("nan"), _INF),
        (None, 0.0),
        (0.0, None),
    ],
)
def test_nonfinite_evidence_does_not_compare_close_to_anything_but_itself(expected, actual):
    """`sequential_log_e` is `-inf` once a boundary is certain; tolerance arithmetic on an
    infinite operand is `inf <= inf`, which must never make it equal a finite or opposite
    evidence value. A NaN or a null equals nothing but a null."""
    from tests.parity_harness.comparison import nested_close

    assert not nested_close(expected, actual)
    assert not nested_close({"sequential_log_e": [expected]}, {"sequential_log_e": [actual]})


@pytest.mark.parametrize("value", [_INF, -_INF])
def test_same_signed_infinity_and_nulls_compare_equal_and_finite_values_keep_their_tolerance(
    value,
):
    from tests.parity_harness.comparison import nested_close

    assert nested_close(value, value)
    assert nested_close({"sequential_log_e": value}, {"sequential_log_e": value})
    assert nested_close(None, None)
    assert nested_close(1.0, 1.0 + 1e-12)
    assert nested_close(1e6, 1e6 * (1 + 1e-12))
    assert not nested_close(1.0, 1.0 + 1e-6)
    assert not nested_close(1, 2)


def test_a_row_whose_sequential_evidence_is_infinite_on_one_path_only_does_not_agree():
    from tests.parity_harness.runner import _assert_payload_equal

    certain = {"sequential_log_e": -_INF, "point": 1.0}
    for other in (0.0, -5.0, _INF):
        with pytest.raises(AssertionError, match="sequential_log_e"):
            _assert_payload_equal(
                "case",
                "left",
                "right",
                ("m",),
                "unadjusted",
                certain,
                {**certain, "sequential_log_e": other},
            )
    _assert_payload_equal("case", "left", "right", ("m",), "unadjusted", certain, dict(certain))


def _post_exposure_events(unit_id: str, event: str) -> int:
    """How many *event* rows the matrix log holds for *unit_id* at or after its exposure."""
    from tests.parity_harness import matrix_data as md

    exposed_at = next(u.exposed_at for u in md.units() if u.id == unit_id)
    return sum(
        1
        for row in md.event_rows()
        if row["user_id"] == unit_id and row["event"] == event and row["event_at"] >= exposed_at
    )


def test_matrix_fixture_nulls_mark_exactly_the_inputs_the_event_log_lacks():
    """A frame NULL is an input the warehouse reads as an absent event (zero), for every
    metric input -- conversion, revenue, sessions, latency -- and the `error` fixture is that
    same data with the zero written out, so the three policies differ only in what they do
    with the NULLs, never in the values underneath."""
    from tests.parity_harness import matrix_data as md

    nullable = md.frame(md.summary_rows("utc", nulls=True))
    explicit = md.frame(md.summary_rows("utc", nulls=False))
    assert not explicit.isna().any().any()
    events = {
        "revenue": "purchase",
        "converted": "purchase",
        "sessions": "session_end",
        "latency": "latency",
    }
    for column, event in events.items():
        null_units = set(nullable.loc[nullable[column].isna(), "user_id"])
        assert null_units, f"{column} has no missing input"
        for unit_id in null_units:
            assert _post_exposure_events(unit_id, event) == 0, (
                f"{unit_id}: {column} is NULL but the log has {event} events"
            )
        assert (nullable[column].fillna(0.0) == explicit[column]).all(), column


def test_matrix_fixture_missing_inputs_are_distinct_and_cover_every_ratio_pattern():
    """Each metric input is missing on its own units, so a policy applied to the wrong column
    changes a result, and a ratio has units missing its numerator only, its denominator only,
    both, and neither, in each arm."""
    from collections import Counter

    from tests.parity_harness import matrix_data as md

    units = md.units()
    for arm in ("control", "treatment"):
        mine = [u for u in units if u.arm == arm]
        sets = {
            "conversion": {u.id for u in mine if md.conversion_missing(u)},
            "revenue": {u.id for u in mine if md.revenue_missing(u)},
            "sessions": {u.id for u in mine if md.sessions_missing(u)},
            "latency": {u.id for u in mine if md.latency_missing(u)},
        }
        assert all(sets.values()), sets
        assert not sets["conversion"] & sets["revenue"]
        patterns = Counter((md.revenue_missing(u), md.sessions_missing(u)) for u in mine)
        assert set(patterns) == {(False, False), (True, False), (False, True), (True, True)}


def test_matrix_fixture_panel_nulls_reach_every_input_a_panel_metric_reads():
    from tests.parity_harness import matrix_data as md

    nulls = md.frame(md.panel_rows("utc", nulls=True))
    explicit = md.frame(md.panel_rows("utc", nulls=False))
    assert not explicit.isna().any().any()
    for column in ("revenue", "converted", "returned", "sessions", "latency"):
        assert nulls[column].isna().any(), column
        assert (nulls[column].fillna(0.0) == explicit[column]).all(), column
    daily = md.frame(md.panel_rows("utc", nulls=True, daily_conversion=True))
    assert daily["converted"].isna().any()


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
        build={"from_unit_summary": lambda: cast("Analysis", analysis)},
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


def _lift_row(**changed):
    """A fixed-horizon lift row; each test changes one public field of it."""
    from increment.estimation.results import Estimate, LiftEstimate

    lift = Estimate(
        value=0.1, lb=0.01, ub=0.19, level=0.95, alpha=0.05, log_mean=0.0953, log_se=0.04
    )
    base = LiftEstimate(
        metric="m", group_id="treatment", method="unadjusted", method_role="decision", lift=lift
    )
    lift_changes = {k: changed.pop(k) for k in ("alpha", "log_mean", "log_se") if k in changed}
    if lift_changes:
        changed["lift"] = lift.model_copy(update=lift_changes)
    return base.model_copy(update=changed)


def _rank_set(
    *, alpha: float = 0.05, treatment=(1, 2, 3, 5), study_id: str = "e", missingness: str = "error"
):
    from increment.estimation.winsor import joint_confidence_set
    from increment.winsor import RawArm, WinsorInferenceSpec, WinsorRawState, WinsorSupport

    raw = WinsorRawState(
        metric="m",
        study_id=study_id,
        missingness=missingness,
        quantile=0.75,
        inference=WinsorInferenceSpec(method="joint-rank-projection-v1"),
        support=WinsorSupport(lower=0, upper=None, provenance="External finite oracle support"),
        arms=(RawArm(group_id="C", values=(1, 2, 3, 9)), RawArm(group_id="T", values=treatment)),
    )
    return joint_confidence_set(raw, "C", "T", alpha)


def _assert_lift_rows_equal(left, right):
    from tests.parity_harness.runner import _normalize

    expected, actual = _normalize([left]), _normalize([right])
    (key,) = expected
    assert expected.keys() == actual.keys()
    _assert_equal(expected, actual, key)


@pytest.mark.parametrize(
    "changed",
    [
        {"alpha": 0.1},
        {"log_mean": 0.09},
        {"log_se": 0.05},
        {"n_clusters": 40},
        {"population": "overlap e in [0.01, 0.99] (90 of 100 units)"},
        {"winsor_upper_percentile": 0.99},
        {"winsor_upper_bound": 12.0},
        {"winsor_control_n_upper": 3},
        {"winsor_treatment_n": 50},
        {"preferred_direction": "increase"},
        {"relative_unavailable_reason": "zero_relative_variance"},
        {"confidence_set": _rank_set()},
    ],
    ids=lambda changed: next(iter(changed)),
)
def test_lift_rows_differing_in_any_public_field_do_not_compare_equal(changed):
    with pytest.raises(AssertionError):
        _assert_lift_rows_equal(_lift_row(), _lift_row(**changed))


def test_winsor_confidence_sets_differing_only_in_construction_do_not_compare_equal():
    """Identical displayed bounds from a different raw outcome pool or error budget are a
    different reusable inference state."""
    left = _lift_row(confidence_set=_rank_set())
    with pytest.raises(AssertionError):
        _assert_lift_rows_equal(left, _lift_row(confidence_set=_rank_set(alpha=0.1)))
    with pytest.raises(AssertionError):
        _assert_lift_rows_equal(left, _lift_row(confidence_set=_rank_set(treatment=(1, 2, 3, 6))))
    # The source's identity and its declared missing-policy label name the path, not the pool.
    labelled = _rank_set(study_id="frame", missingness="measure-unit-inclusion-v1:sum")
    _assert_lift_rows_equal(left, _lift_row(confidence_set=labelled))


def test_numeric_metadata_agrees_within_the_documented_tolerance():
    """Floats compare at 1e-9 relative: a last-place difference in a standard error passes and
    a visible one does not."""
    _assert_lift_rows_equal(_lift_row(), _lift_row(log_se=0.04 * (1 + 1e-12)))
    with pytest.raises(AssertionError):
        _assert_lift_rows_equal(_lift_row(), _lift_row(log_se=0.04 * (1 + 1e-6)))


def test_a_duplicate_emitted_row_is_refused_not_collapsed():
    """A join fan-out returning a row twice must not compare equal to returning it once."""
    from tests.parity_harness.runner import _normalize

    row = _lift_row()
    (key,) = _normalize([row])
    assert _normalize([row]).keys() == {key}
    with pytest.raises(AssertionError, match="more than one row"):
        _normalize([row, row])


def test_a_duplicate_day_axis_value_is_refused_not_collapsed():
    from datetime import date

    from increment.breakout.estimates import DailyMetricValue
    from tests.parity_harness.runner import _normalize

    value = DailyMetricValue(
        ds=date(2025, 1, 10),
        metric="m",
        group_id="treatment",
        value=None,
        unavailable="few_units",
        n=0,
    )
    with pytest.raises(AssertionError, match="more than one row"):
        _normalize([value, value])


@pytest.mark.parametrize("seed", range(6))
def test_the_parity_verdict_does_not_depend_on_which_ingress_is_the_oracle(seed):
    """Rows compare the same whichever ingress is read first: a disagreeing ingress fails in
    every order, and agreeing ones pass in every order."""
    import itertools

    from tests.parity_harness.cases import CONSTRUCTORS
    from tests.parity_harness.runner import CaseResult, _normalize

    names = ("from_definitions", "from_unit_summary", "from_unit_panel")
    agreeing = [_lift_row(), _lift_row(log_se=0.04 * (1 + 1e-12)), _lift_row()]
    disagreeing = [*agreeing[:2], _lift_row(n_clusters=40)]
    case = _case({name: cast("Callable[[], Analysis]", None) for name in names})
    assert set(CONSTRUCTORS) >= set(names)
    order = list(itertools.permutations(range(3)))[seed]
    for rows, fails in ((agreeing, False), (disagreeing, True)):
        result = CaseResult(rows={names[i]: _normalize([rows[i]]) for i in order}, refusals={})
        if fails:
            with pytest.raises(AssertionError):
                assert_parity(case, result)
        else:
            assert_parity(case, result)


def _definitions_schema_builder(*, metric_type: str, winsorization: bool, missing: str | None):
    """A `Definitions` declaration carrying only the named unsupported inputs; validating it
    raises before an `Analysis` exists."""
    from increment.semantics.models import Definitions
    from tests.parity_harness.matrix import Cell
    from tests.parity_harness.matrix_cases import definitions_payload

    cell = Cell(metric_type, "run", "winsor_fixed" if winsorization else "none", "utc", "error")

    def build():
        payload = definitions_payload(cell)
        if missing is not None:
            payload["metrics"][0]["missing"] = missing
        return cast("Analysis", Definitions.model_validate(payload))

    return build


def _schema_absence_case(build, field):
    from pydantic import ValidationError

    return _case(
        {"from_definitions": build},
        waive={"from_definitions": "SOURCE: schema absence"},
        absence={"from_definitions": Absence(ValidationError, field)},
        refusal_only=True,
    )


def test_schema_absence_passes_when_the_error_rejects_the_declared_field():
    winsor = _definitions_schema_builder(metric_type="conversion", winsorization=True, missing=None)
    case = _schema_absence_case(winsor, "winsorization")
    assert_parity(case, run_case(case))
    both = _definitions_schema_builder(metric_type="conversion", winsorization=True, missing="drop")
    case = _schema_absence_case(both, "winsorization")
    assert_parity(case, run_case(case))


def test_schema_absence_fails_when_the_error_is_about_another_field():
    """A validation error is not evidence for a different declared absence: with the
    winsorization guard gone, the unrelated missing-policy error must not satisfy it."""
    other = _definitions_schema_builder(metric_type="mean", winsorization=False, missing="drop")
    case = _schema_absence_case(other, "winsorization")
    with pytest.raises(AssertionError, match="none rejects 'winsorization'"):
        run_case(case)


def test_keyword_absence_must_name_a_keyword_the_signature_does_not_accept():
    from tests.parity_harness.cases import CONSTRUCTORS

    assert "from_switchback_panel" in CONSTRUCTORS
    for field, message in (("cluster", "without naming 'cluster'"), ("unit", "accepts 'unit'")):
        case = _case(
            {"from_switchback_panel": _no_design_switchback_builder()},
            waive={"from_switchback_panel": "SOURCE: declared absent"},
            absence={"from_switchback_panel": Absence(TypeError, field)},
            refusal_only=True,
        )
        with pytest.raises(AssertionError, match=message):
            run_case(case)


def _sequential_row():
    """A real sequential row from the asymptotic-mean scenario's frame ingress."""
    case = next(c for c in PARITY_CASES if c.id == "sequential_asymptotic_mean_revenue")
    analysis = case.build["from_unit_summary"]()
    try:
        as_of = getattr(analysis, "_sequential_as_of", None)
        analysis.capture_sequential(finalized=True, **({"as_of": as_of} if as_of else {}))
        rows = [r for r in analysis.run() if getattr(r, "sequential_result", None) is not None]
    finally:
        analysis.close()
    assert rows
    return rows[0]


def _with_checkpoint(row, **changed):
    result = row.sequential_result
    checkpoint = result.checkpoint.model_copy(update=changed)
    return row.model_copy(
        update={"sequential_result": result.model_copy(update={"checkpoint": checkpoint})}
    )


@pytest.mark.slow
def test_sequential_rows_compare_every_public_field_except_route_identifiers():
    """A sequential row's evidence is compared whole: a different valid `alpha_ceiling`, a
    different `point_reason` or different checkpoint metadata is a different result. Only the
    three identifiers hashed from the path's observation mapping name the route, not the result."""
    from fractions import Fraction

    from increment.estimation.sequential_runtime import evaluate_checkpoint

    row = _sequential_row()
    result = row.sequential_result
    ceiling = Fraction(1, 2)
    assert result.alpha_ceiling < ceiling
    widened = evaluate_checkpoint(result.checkpoint, alpha=result.decision_alpha, ceiling=ceiling)
    assert widened.log_e == result.log_e
    assert widened.bounds == result.bounds
    for different in (
        row.model_copy(update={"sequential_result": widened}),
        row.model_copy(
            update={"sequential_result": result.model_copy(update={"point_reason": "other"})}
        ),
        _with_checkpoint(row, revealed_units=result.checkpoint.revealed_units + 1),
        _with_checkpoint(row, cell=result.checkpoint.cell.model_copy(update={"family": True})),
    ):
        with pytest.raises(AssertionError):
            _assert_lift_rows_equal(row, different)

    renamed = _with_checkpoint(
        row, registration_id="other", filtration_id="other", prefix_id="other"
    )
    assert renamed.sequential_result.checkpoint.prefix_id != result.checkpoint.prefix_id
    _assert_lift_rows_equal(row, renamed)
