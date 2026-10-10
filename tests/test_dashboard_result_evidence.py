"""Dashboard result evidence: per-row outcomes, complete confidence sets, levels, warnings.

Rows come from real estimates and the real confidence-set constructors, so every assertion
reads emitted numbers, geometry and outcome classes rather than prose.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

import pytest

pytest.importorskip("coeftable")
pytest.importorskip("marimo")

from increment.dashboard import DashboardSnapshot, render_details, render_metric_details
from increment.dashboard._format import (
    complete_confidence_set_text,
    confidence_set_text,
    headline_interval,
    result_outcome,
    result_outcome_text,
)
from increment.dashboard._html import results_notes
from increment.estimation.inference import _joint_additive_bounds
from increment.estimation.results import (
    BINOMIAL_METHOD,
    BINOMIAL_NUMERICAL_QUALIFICATION,
    BinomialConfidenceSet,
    Estimate,
    JointContrastReference,
    relative_confidence_set,
)
from increment.tables import estimates_to_readout
from tests import test_dashboard as dashboard_tests

# The dashboard suite's real DuckDB-backed fixtures, shared by name.
dashboard_con = dashboard_tests.dashboard_con
dashboard_definitions = dashboard_tests.dashboard_definitions
storefront = dashboard_tests.storefront
_guardrail_snapshot = dashboard_tests._guardrail_snapshot


PRIMARY, SECONDARY, GUARDRAIL = "checkout_conversion", "revenue_per_user", "session_seconds"


def _rebuilt(snapshot: DashboardSnapshot, estimates: tuple[Any, ...]) -> DashboardSnapshot:
    rows = estimates_to_readout(list(estimates))
    return dataclasses.replace(
        snapshot,
        estimates=estimates,
        readout_rows=tuple(
            {**row, "alternative": est.alternative}
            for row, est in zip(rows, estimates, strict=True)
        ),
    )


def _decision(snapshot: DashboardSnapshot, metric: str) -> Any:
    return next(e for e in snapshot.estimates if e.metric == metric and e.method_role == "decision")


def _with(snapshot: DashboardSnapshot, metric: str, **updates: Any) -> DashboardSnapshot:
    """The snapshot with one decision estimate changed."""
    original = _decision(snapshot, metric)
    estimates = tuple(
        original.model_copy(update=updates) if e is original else e for e in snapshot.estimates
    )
    return _rebuilt(snapshot, estimates)


def _row(snapshot: DashboardSnapshot, metric: str, method_role: str = "decision") -> Any:
    return next(
        r
        for r in snapshot.readout_rows
        if r["metric"] == metric and r["method_role"] == method_role
    )


# A row whose engine produced no point, no interval and no set.
_UNAVAILABLE = {"lift": None, "confidence_set": None, "binomial_set": None}


def _fieller(a: float, c: float, *, alternative: str = "two-sided", cov: float = 0.0):
    reference = JointContrastReference(a=a, c=c, var_a=1, var_c=1, cov_ac=cov)
    return relative_confidence_set(reference, alternative=alternative)


def test_decision_grade_binomial_set_discloses_numerical_qualification():
    from increment.dashboard._format import complete_confidence_set_text
    from increment.estimation.binomial_rr import nuisance_beta

    region = BinomialConfidenceSet(
        lower=-0.8,
        upper=1.0,
        alpha=0.05,
        level=0.95,
        decision_alpha=0.05,
        geometry="central",
        method=BINOMIAL_METHOD,
        numerical_qualification=BINOMIAL_NUMERICAL_QUALIFICATION,
        x_c=10,
        n_c=20,
        x_t=12,
        n_t=20,
        nuisance_beta=nuisance_beta(0.05),
    )
    text = complete_confidence_set_text(
        {"binomial_set": region, "confidence_set": None, "relative_confidence_set": None}
    )
    assert text is not None
    assert "conditional on deployed SciPy/Boost" in text


def test_point_backed_binomial_qualification_keeps_displayed_interval_endpoints():
    from increment.dashboard._format import interval_html
    from increment.estimation.binomial_rr import nuisance_beta

    bounded = BinomialConfidenceSet(
        lower=-0.8,
        upper=1.0,
        alpha=0.05,
        level=0.95,
        decision_alpha=0.05,
        geometry="central",
        method=BINOMIAL_METHOD,
        numerical_qualification=BINOMIAL_NUMERICAL_QUALIFICATION,
        x_c=10,
        n_c=20,
        x_t=12,
        n_t=20,
        nuisance_beta=nuisance_beta(0.05),
    )
    row = {
        "lift": 0.15,
        "lower": 0.10,
        "higher": 0.20,
        "value_scale": "relative",
        "relative_confidence_set": None,
        "binomial_set": bounded,
    }
    for display_text in (headline_interval(row), interval_html(row)):
        assert "10.0%" in display_text
        assert "20.0%" in display_text
        assert "conditional on deployed SciPy/Boost" in display_text


# Complete set text


@pytest.mark.slow
def test_point_backed_one_sided_fieller_set_keeps_its_direction_and_own_coverage(
    storefront: DashboardSnapshot,
) -> None:
    region = relative_confidence_set(
        JointContrastReference(a=2, c=3.0, var_a=0.04, var_c=0.09, cov_ac=0.01),
        alternative="greater",
    )
    bounds = _joint_additive_bounds(2.0, 0.2, 0.05, "greater", None)
    snapshot = _with(
        storefront,
        PRIMARY,
        alternative="greater",
        lift=region.estimate(),
        relative_confidence_set=region,
        abs_diff=2.0,
        abs_se=0.2,
        abs_lb=bounds[0],
        abs_ub=bounds[1],
        abs_reference_kind="normal",
        binomial_set=None,
        confidence_set=None,
    )
    row = _row(snapshot, PRIMARY)
    assert region.geometry == "one_sided"
    central_lower, central_upper = row["lower"], row["higher"]
    assert row["level"] == pytest.approx(0.9)

    # The concise headline is the central interval at its own 90% level, unchanged.
    assert confidence_set_text(row) is None
    headline = headline_interval(row)
    assert f"{central_lower:+.1%}" in headline and f"{central_upper:+.1%}" in headline

    # The full set is one-sided (open above) at the set's own 95% coverage.
    complete = complete_confidence_set_text(row)
    assert complete is not None
    ((lower, upper),) = region.intervals
    assert upper is None
    assert f"{lower:+.1%}" in complete and "+∞" in complete
    stated = [float(value) / 100 for value in re.findall(r"(\d+(?:\.\d+)?)%", complete)]
    assert any(abs(value - (1 - region.alpha)) < 5e-5 for value in stated)
    assert f"{central_upper:+.1%}" not in complete


@pytest.mark.parametrize(
    ("a", "c", "cov", "geometry"),
    [(10, 0, 0.0, "disconnected"), (1, 0, 0.0, "all_real"), (1, 1, -0.9, "disconnected")],
)
@pytest.mark.slow
def test_complete_set_text_keeps_every_endpoint_of_point_free_sets(
    storefront: DashboardSnapshot, a: float, c: float, cov: float, geometry: str
) -> None:
    region = _fieller(a, c, cov=cov)
    assert region.geometry == geometry
    row = _row(_with(storefront, PRIMARY, lift=None, relative_confidence_set=region), PRIMARY)
    complete = complete_confidence_set_text(row)
    assert complete is not None
    for interval in region.intervals:
        for endpoint in interval:
            assert endpoint is None or f"{endpoint:+.1%}" in complete
    assert complete.count("∞") == sum(
        endpoint is None for interval in region.intervals for endpoint in interval
    )


# Per-row outcomes


@pytest.mark.parametrize("rejects", [True, False])
@pytest.mark.parametrize("discovery", [True, False, None])
@pytest.mark.slow
def test_verdict_and_selection_are_independent(
    storefront: DashboardSnapshot, rejects: bool, discovery: bool | None
) -> None:
    interval = (
        Estimate(value=0.3, lb=0.1, ub=0.5, level=0.95)
        if rejects
        else Estimate(value=0.05, lb=-0.1, ub=0.2, level=0.95)
    )
    row = _row(_with(storefront, SECONDARY, lift=interval, discovery=discovery), SECONDARY)
    outcome = result_outcome(row)
    assert outcome.verdict == ("reject" if rejects else "not_reject")
    assert (
        outcome.selection
        == {True: "selected", False: "not_selected", None: "not_applied"}[discovery]
    )
    assert outcome.unavailable_reason is None


@pytest.mark.parametrize(
    ("a", "c", "cov", "expected"),
    [
        (1, 0, 0.0, "not_reject"),  # whole line: valid, nothing excluded, no point
        (10, 0, 0.0, "reject"),  # disconnected, null in the gap
        (1, 1, -0.9, "not_reject"),  # disconnected, null inside a component
    ],
)
def test_valid_sets_without_a_bounded_interval_are_not_unavailable(
    storefront: DashboardSnapshot, a: float, c: float, cov: float, expected: str
) -> None:
    region = _fieller(a, c, cov=cov)
    snapshot = _with(storefront, PRIMARY, lift=region.estimate(), relative_confidence_set=region)
    outcome = result_outcome(_row(snapshot, PRIMARY))
    assert outcome.verdict == expected
    assert outcome.unavailable_reason is None


def test_true_unavailability_keeps_its_engine_reason_and_is_not_a_non_rejection(
    storefront: DashboardSnapshot,
) -> None:
    reason = "joint_covariance_indefinite"
    snapshot = _with(storefront, PRIMARY, **_UNAVAILABLE, relative_unavailable_reason=reason)
    row = _row(snapshot, PRIMARY)
    assert row["stat_sig"] is False
    outcome = result_outcome(row)
    assert outcome.verdict == "unavailable"
    assert outcome.unavailable_reason == reason
    assert reason in result_outcome_text(row)


def test_missing_verdict_is_unavailable_not_a_default_non_rejection(
    storefront: DashboardSnapshot,
) -> None:
    row = {k: v for k, v in _row(storefront, PRIMARY).items() if k != "stat_sig"}
    assert result_outcome(row).verdict == "unavailable"


def test_text_qualifies_the_method_role_and_arm(storefront: DashboardSnapshot) -> None:
    row = {**_row(storefront, PRIMARY), "method": "cuped", "method_role": "sensitivity"}
    text = result_outcome_text(row)
    assert "cuped" in text and "sensitivity" in text
    assert str(row["group_id"]) not in text
    assert str(row["group_id"]) in result_outcome_text(row, arm=True)


# Levels


def test_interval_levels_keep_two_decimals_in_every_disclosure(
    storefront: DashboardSnapshot,
) -> None:
    level = 1 - 0.05 / 3
    snapshot = _with(
        storefront,
        PRIMARY,
        lift=Estimate(value=0.2, lb=-0.1, ub=0.5, level=level),
    )
    surfaces = (
        results_notes(snapshot),
        render_details(snapshot).text,
        render_metric_details(snapshot, metric=PRIMARY).text,
    )
    for text in surfaces:
        stated = [float(p) / 100 for p in re.findall(r"(\d+(?:\.\d+)?)%", text)]
        assert any(abs(value - level) < 5e-5 for value in stated)
        assert "98.3%" not in text


# Adverse warnings
