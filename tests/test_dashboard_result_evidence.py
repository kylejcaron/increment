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
    outcome_label,
)
from increment.dashboard._html import results_notes
from increment.estimation.inference import _joint_additive_bounds
from increment.estimation.results import (
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


def test_binomial_qualification_is_in_the_metric_inspector_interval_fold(storefront):
    from increment.dashboard._html import render_metric_details

    row = _row(storefront, "checkout_conversion")
    html = render_metric_details(storefront, metric=row["metric"]).text
    assert "How to read these intervals" in html
    assert "conditional on deployed SciPy/Boost" in html


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


@pytest.mark.parametrize("stat_sig", [True, False, None])
@pytest.mark.parametrize("discovery", [True, False, None])
def test_outcome_label_formats_captured_verdict_and_selection(
    storefront: DashboardSnapshot, stat_sig: bool | None, discovery: bool | None
) -> None:
    row = {
        **_row(storefront, SECONDARY),
        "stat_sig": stat_sig,
        "discovery": discovery,
    }
    label = outcome_label(row)
    assert label
    assert outcome_label({**row, "stat_sig": not stat_sig}) != label
    assert outcome_label({**row, "discovery": not discovery}) != label


def test_valid_sets_without_a_bounded_interval_keep_captured_verdict(
    storefront: DashboardSnapshot,
) -> None:
    region = _fieller(1, 0)
    snapshot = _with(storefront, PRIMARY, lift=region.estimate(), relative_confidence_set=region)
    row = _row(snapshot, PRIMARY)
    assert outcome_label(row) == outcome_label({**row, "relative_confidence_set": None})


def test_missing_verdict_is_not_converted_to_non_rejection(
    storefront: DashboardSnapshot,
) -> None:
    row = _row(storefront, PRIMARY)
    assert outcome_label({key: value for key, value in row.items() if key != "stat_sig"}) != (
        outcome_label({**row, "stat_sig": False})
    )


def test_readout_qualifications_distinguish_multiple_methods_for_one_metric(
    storefront: DashboardSnapshot,
) -> None:
    from increment.breakout.estimates import LiftEstimates
    from increment.dashboard._data import readout_interpretation_text

    decision = _decision(storefront, PRIMARY)
    sensitivity = decision.model_copy(update={"method": "cuped", "method_role": "sensitivity"})
    snapshot = _rebuilt(storefront, (*storefront.estimates, sensitivity))
    collection = LiftEstimates(list(snapshot.estimates))

    qualifications = readout_interpretation_text(snapshot, collection=collection)

    assert "method unadjusted" in qualifications
    assert "method role decision" in qualifications
    assert "method cuped" in qualifications
    assert "method role sensitivity" in qualifications


def test_metric_details_restore_metric_descriptions(storefront: DashboardSnapshot) -> None:
    declared = render_metric_details(storefront, metric=PRIMARY).text

    assert "Did the user check out within 14 days of exposure?" in declared


def test_outcome_label_does_not_add_row_identity_to_layout_text(
    storefront: DashboardSnapshot,
) -> None:
    row = {**_row(storefront, PRIMARY), "method": "cuped", "method_role": "sensitivity"}
    label = outcome_label(row)
    assert label
    assert outcome_label({**row, "group_id": "different-arm"}) == label


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
