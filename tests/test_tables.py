"""Tests for increment.tables' estimate adapters and CoefTable renderers.

installed in the default dev environment. ``estimates_to_readout`` and
``_liftestimate_to_row`` are dependency-free and must work without the extra.
Tests that exercise rendering are guarded with
``pytest.importorskip("coeftable")`` so they skip cleanly without it.
"""

from __future__ import annotations

import re
import warnings
from datetime import date
from fractions import Fraction
from typing import TypedDict

import pytest

from increment import AlwaysValid, estimate_sequential
from increment.breakout.estimates import BreakoutEstimate, DailyLiftEstimate
from increment.breakout.projection import to_frame
from increment.estimation.results import Estimate, LiftEstimate
from increment.semantics.unit_cycle import UnitCycleTApproximation
from increment.tables import (
    contrast_results_to_readout,
    estimates_to_readout,
    readout_table,
)
from tests.sequential_cases import capture, records, registration


def _certified_row(alpha=0.05):
    reg = registration(alpha=Fraction(alpha).limit_denominator())
    snapshot = capture(
        reg,
        records([0, 0, 0, 1] * 24, [0, 1, 1, 1] * 24),
    )
    return estimate_sequential(snapshot, AlwaysValid(registration=reg)).results[0]


def _estimate(metric, group_id, method, value, lb=None, ub=None, estimand="itt"):
    level = 0.95 if lb is not None else None
    return LiftEstimate(
        metric=metric,
        group_id=group_id,
        method=method,
        method_role="decision",
        lift=Estimate(value=value, lb=lb, ub=ub, level=level),
        estimand=estimand,
    )


def _breakout_estimate(
    metric,
    group_id,
    method,
    dimension,
    dimension_value,
    value,
    lb=None,
    ub=None,
    source=None,
    excluded=None,
):
    level = 0.95 if lb is not None else None
    return BreakoutEstimate(
        metric=metric,
        group_id=group_id,
        method=method,
        method_role="decision",
        dimension=dimension,
        dimension_value=dimension_value,
        source=source,
        lift=None if excluded is not None else Estimate(value=value, lb=lb, ub=ub, level=level),
        excluded=excluded,
    )


def _daily_estimate(
    metric,
    group_id,
    method,
    ds,
    value,
    lb=None,
    ub=None,
    dimension=None,
    dimension_value=None,
    source=None,
    ds_basis="calendar",
    estimand="itt",
):
    level = 0.95 if lb is not None else None
    unavailable = "few_units" if value != value else None
    return DailyLiftEstimate(
        metric=metric,
        group_id=group_id,
        method=method,
        method_role="decision",
        ds=ds,
        sampling_available=True,
        lift=None if unavailable is not None else Estimate(value=value, lb=lb, ub=ub, level=level),
        unavailable=unavailable,
        dimension=dimension,
        dimension_value=dimension_value,
        source=source,
        ds_basis=ds_basis,
        estimand=estimand,
    )


def _normalize_table_id(html):
    """Strip great_tables' per-``.gt()``-call random table id from *html*.

    Each render gets a fresh id in both an ``id="..."`` attribute and every
    CSS selector - call-count noise, not render content, so equality
    comparisons between two separately-rendered HTML strings normalize it out first.
    """
    match = re.search(r'id="([a-zA-Z0-9]+)"', html)
    if match is None:
        return html
    return html.replace(match.group(1), "TABLEID")


def _group_headers(html):
    """Extract visible coeftable group-heading cell text."""
    cells = re.findall(
        r"<(?:td|th)[^>]*\bgt_group_heading\b[^>]*>(.*?)</(?:td|th)>",
        html,
        flags=re.DOTALL,
    )
    return [re.sub(r"<[^>]+>", "", cell).strip() for cell in cells]


def test_liftestimate_to_row_maps_fields():
    """value/lb/ub/metric map to lift/lower/higher/method per the brief's adapter spec."""
    est = _estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22)
    row = estimates_to_readout([est])[0]
    assert row["metric"] == "revenue"
    assert row["method"] == "unadjusted"
    assert row["lift"] == 0.12
    assert row["lower"] == 0.02
    assert row["higher"] == 0.22
    assert row["group_id"] == "T"


def test_liftestimate_to_row_preserves_analysis_population():
    assigned = _estimate("revenue", "T", "unadjusted", 0.12)
    triggered = assigned.model_copy(update={"analysis_population": "triggered"})

    assert [row["analysis_population"] for row in estimates_to_readout([assigned, triggered])] == [
        "assigned",
        "triggered",
    ]


def test_readout_table_disambiguates_assigned_and_triggered_rows():
    pytest.importorskip("coeftable")
    assigned = _estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22)
    triggered = assigned.model_copy(
        update={
            "analysis_population": "triggered",
            "lift": Estimate(value=0.50, lb=0.30, ub=0.70, level=0.95),
        }
    )
    html = readout_table(estimates_to_readout([assigned, triggered])).gt().as_raw_html()

    assert "(assigned)" in html
    assert "(triggered)" in html


def test_readout_table_population_labels_align_trend_keys():
    pytest.importorskip("coeftable")
    import pandas as pd

    assigned = _estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22)
    triggered = assigned.model_copy(
        update={
            "analysis_population": "triggered",
            "lift": Estimate(value=0.50, lb=0.30, ub=0.70, level=0.95),
        }
    )
    assigned_trend = [
        _daily_estimate(
            "revenue",
            "T",
            "unadjusted",
            date(2024, 1, day),
            value,
            lb=value - 0.05,
            ub=value + 0.05,
        )
        for day, value in ((1, 0.08), (2, 0.12))
    ]
    triggered_trend = [
        _daily_estimate(
            "revenue",
            "T",
            "unadjusted",
            date(2024, 1, day),
            value + 0.30,
            lb=value + 0.25,
            ub=value + 0.35,
        )
        for day, value in ((1, 0.08), (2, 0.12))
    ]
    assigned_trend_frame = to_frame(assigned_trend, model=DailyLiftEstimate)
    triggered_trend_frame = to_frame(triggered_trend, model=DailyLiftEstimate)
    assert isinstance(assigned_trend_frame, pd.DataFrame)
    assert isinstance(triggered_trend_frame, pd.DataFrame)
    assert assigned_trend_frame["analysis_population"].eq("assigned").all()
    triggered_trend_frame["analysis_population"] = "triggered"
    trend_frame = pd.concat(
        [assigned_trend_frame, triggered_trend_frame],
        ignore_index=True,
    )

    html = (
        readout_table(
            estimates_to_readout([assigned, triggered]),
            trend=trend_frame,
        )
        .gt()
        .as_raw_html()
    )

    assert "revenue (assigned)" in html
    assert "revenue (triggered)" in html


def test_readout_table_rejects_duplicate_rows_without_population_identity():
    pytest.importorskip("coeftable")
    row = estimates_to_readout([_estimate("revenue", "T", "unadjusted", 0.12)])[0]
    duplicate = dict(row)
    duplicate.pop("analysis_population")

    from increment.errors import InvalidRequestError

    with pytest.raises(InvalidRequestError) as exc_info:
        readout_table([duplicate, duplicate])
    assert exc_info.value.code == "tables.readout_table_ambiguous"


def test_liftestimate_to_row_stat_sig_excludes_zero():
    """stat_sig is True iff the interval excludes 0."""
    sig = _estimate("m", "T", "unadjusted", 0.1, lb=0.02, ub=0.18)
    not_sig = _estimate("m", "T", "unadjusted", 0.1, lb=-0.05, ub=0.25)
    assert estimates_to_readout([sig])[0]["stat_sig"] is True
    assert estimates_to_readout([not_sig])[0]["stat_sig"] is False


def test_liftestimate_to_row_stat_sig_one_sided_greater_checks_only_lower_tail():
    """The bug excludes() has for one-sided results: an interval entirely
    BELOW 0 'excludes' it (both tails clear), but a 'greater' test must
    fail - the metric moved the wrong way."""
    from increment.estimation.inference import infer_lift

    # Deliberately construct a harm case: treatment worse than control,
    # tested one-sided "greater" (asking "did it improve?").
    est = infer_lift(
        metric="revenue",
        group_id="T",
        method="unadjusted",
        method_role="decision",
        log_rr=0.05 - 0.15,
        se_t=0.03,
        se_c=0.04,
        alternative="greater",
    )
    lift = est.require_lift()
    assert lift.ub is not None and lift.ub < 0.0  # interval entirely below 0
    assert est.require_lift().excludes(0.0) is True  # the old (wrong) check would say "significant"
    assert estimates_to_readout([est])[0]["stat_sig"] is False  # the fixed check says no


@pytest.mark.parametrize(
    "estimate",
    [
        _breakout_estimate(
            "revenue", "T", "unadjusted", "country", "US", 0.2, lb=0.1, ub=0.3
        ).model_copy(update={"prior_shrunk": True}),
        _daily_estimate(
            "revenue", "T", "unadjusted", date(2025, 1, 1), 0.2, lb=0.1, ub=0.3
        ).model_copy(update={"sampling_available": None}),
    ],
)
def test_legacy_prior_bound_breakout_and_unknown_daily_sampling_refuse(estimate):
    from increment.errors import InvalidRequestError

    with pytest.raises(InvalidRequestError) as raised:
        estimates_to_readout([estimate])
    assert raised.value.code == "readout.legacy.sampling_unreconstructible"
    assert raised.value.context["remedy"] == "recompute_from_source"


def test_prior_free_legacy_breakout_sampling_remains_readable():
    estimate = _breakout_estimate(
        "revenue", "T", "unadjusted", "country", "US", 0.2, lb=0.1, ub=0.3
    ).model_copy(update={"sampling_available": None, "prior_shrunk": False})
    (row,) = estimates_to_readout([estimate])
    assert row["stat_sig"] is True


def test_liftestimate_to_row_stat_sig_one_sided_less_checks_only_upper_tail():
    from increment.estimation.inference import infer_lift

    est = infer_lift(
        metric="latency",
        group_id="T",
        method="unadjusted",
        method_role="decision",
        log_rr=0.15 - 0.05,
        se_t=0.03,
        se_c=0.04,
        alternative="less",
    )
    lift = est.require_lift()
    assert lift.lb is not None and lift.lb > 0.0  # interval entirely above 0
    assert est.require_lift().excludes(0.0) is True
    assert estimates_to_readout([est])[0]["stat_sig"] is False


def test_liftestimate_to_row_null_lift_shifts_the_decision():
    """A shifted null_lift moves stat_sig even though the interval itself
    is unchanged - the guardrail case: not significant vs 0 two-sided,
    but confidently non-inferior vs -1% one-sided greater."""
    from increment.estimation.inference import infer_lift

    two_sided = infer_lift(
        metric="revenue",
        group_id="T",
        method="unadjusted",
        method_role="decision",
        log_rr=0.02 - 0.0,
        se_t=0.01,
        se_c=0.01,
    )
    assert estimates_to_readout([two_sided])[0]["stat_sig"] is False

    guardrail = infer_lift(
        metric="revenue",
        group_id="T",
        method="unadjusted",
        method_role="decision",
        log_rr=0.02 - 0.0,
        se_t=0.01,
        se_c=0.01,
        alternative="greater",
        null_lift=-0.01,
    )
    row = estimates_to_readout([guardrail])[0]
    assert row["null_lift"] == -0.01
    assert row["stat_sig"] is True


def test_liftestimate_to_row_null_abs_decision_uses_abs_endpoints():
    """When null_abs is set, stat_sig reads the ADDITIVE interval against
    it - never the relative one: a relative interval comfortably above 0
    still fails when the abs interval crosses the absolute boundary."""
    from increment.estimation.inference import infer_lift

    crossing = infer_lift(
        metric="revenue",
        group_id="T",
        method="unadjusted",
        method_role="decision",
        log_rr=0.15 - 0.05,
        se_t=0.01,
        se_c=0.01,
        alternative="greater",
        abs_diff=-0.5,
        abs_se=0.1,
        null_abs=-0.01,
    )
    # The relative branch would say True - proving the abs branch decides.
    lift = crossing.require_lift()
    assert lift.lb is not None and lift.lb > 0.0
    assert crossing.abs_lb is not None and crossing.abs_lb < -0.01
    row = estimates_to_readout([crossing])[0]
    assert row["null_abs"] == -0.01
    assert row["stat_sig"] is False

    clear = infer_lift(
        metric="revenue",
        group_id="T",
        method="unadjusted",
        method_role="decision",
        log_rr=0.15 - 0.05,
        se_t=0.01,
        se_c=0.01,
        alternative="greater",
        abs_diff=0.02,
        abs_se=0.005,
        null_abs=-0.01,
    )
    assert clear.abs_lb is not None and clear.abs_lb > -0.01
    assert estimates_to_readout([clear])[0]["stat_sig"] is True


def test_stat_sig_abs_unavailable_when_abs_se_none_is_false_not_crash():
    """D4: no abs_se means the additive decision is UNAVAILABLE - stat_sig
    is False, never a crash and never a silent fallback to the relative
    interval (which here would read True)."""
    from increment.estimation.inference import infer_lift

    est = infer_lift(
        metric="revenue",
        group_id="T",
        method="unadjusted",
        method_role="decision",
        log_rr=0.15 - 0.05,
        se_t=0.01,
        se_c=0.01,
        alternative="greater",
        null_abs=-0.01,
    )
    assert est.abs_lb is None and est.abs_ub is None
    lift = est.require_lift()
    assert lift.lb is not None and lift.lb > -0.01
    row = estimates_to_readout([est])[0]
    assert row["null_abs"] == -0.01
    assert row["stat_sig"] is False


def test_liftestimate_to_row_null_lift_defaults_to_zero_for_breakout_estimate():
    """BreakoutEstimate never carries null_lift/alternative (flat model,
    one-sided testing not built for that path yet) - defaults preserve
    today's two-sided-vs-0 behavior."""
    est = _breakout_estimate("revenue", "T", "unadjusted", "country", "US", 0.12, lb=0.02, ub=0.22)
    row = estimates_to_readout([est])[0]
    assert row["null_lift"] == 0.0
    assert row["stat_sig"] is True


def test_liftestimate_to_row_point_only_gives_null_bounds():
    """A point-only Estimate (no lb/ub) maps to nullable bounds."""
    est = _estimate("m", "T", "unadjusted", 0.1)
    row = estimates_to_readout([est])[0]
    assert row["lower"] is None
    assert row["higher"] is None


def _binomial_conversion_row(n, successes, *, group_id, metric="conv", experiment_id="exp1"):
    """A centered group_summary row for a genuinely binary (0/1)
    conversion arm, ``successes`` out of ``n``."""
    from increment.estimation.armstats import centered_row_from_raw_sums

    return centered_row_from_raw_sums(
        {
            "experiment_id": experiment_id,
            "metric": metric,
            "group_id": group_id,
            "n": n,
            "sum_y": float(successes),
            "sum_y2": float(successes),  # y in {0, 1}: y**2 == y
            "successes": successes,
            "sum_x": None,
            "sum_x2": None,
            "sum_xy": None,
            "sum_den": None,
            "sum_den2": None,
            "sum_yden": None,
        }
    )


def _binomial_lift_estimate(x_c, n_c, x_t, n_t, **kwargs):
    """A real ``LiftEstimate`` from the exact binomial risk-ratio
    method, via ``estimate_lift`` on a genuinely binary conversion
    metric - never hand-assembled, so its ``binomial_set`` is exactly
    what production code persists."""
    from increment.estimation.engine import estimate_lift
    from increment.semantics.models import ConversionMetric

    rows = [
        _binomial_conversion_row(n_c, x_c, group_id="control"),
        _binomial_conversion_row(n_t, x_t, group_id="treatment"),
    ]
    computation = estimate_lift(
        metrics=[ConversionMetric(name="conv", entity="user", fact="conv")],
        summary=rows,
        control_group="control",
        **kwargs,
    )
    assert not computation.failures, computation.failures
    return computation.results[0]


class TestLiftEstimateToRowBinomialSetOnly:
    """A set-only exact-binomial row (``lift=None``, an observed
    zero-control-count cell) must carry its persisted confidence set's
    bounds, level and significance verdict, not flatten to
    lower=higher=level=None, stat_sig=False - that discards a real
    confidence set that may well exclude the null."""

    def test_zero_control_count_emits_the_sets_own_bounds_and_level(self):
        est = _binomial_lift_estimate(x_c=0, n_c=200, x_t=100, n_t=200)
        assert est.lift is None
        assert est.binomial_set is not None
        row = estimates_to_readout([est])[0]
        assert row["lift"] is None
        assert row["lower"] == pytest.approx(est.binomial_set.lower)
        assert row["higher"] == est.binomial_set.upper
        assert row["level"] == pytest.approx(est.binomial_set.level)

    def test_readout_table_discloses_binomial_numerical_qualification(self):
        pytest.importorskip("coeftable")
        est = _binomial_lift_estimate(x_c=0, n_c=200, x_t=100, n_t=200)

        html = readout_table(estimates_to_readout([est])).gt().as_raw_html()

        assert "Numerical qualification" in html
        assert "conditional on deployed SciPy/Boost special-function error model" in html

    def test_zero_control_count_confident_separation_is_stat_sig(self):
        """Control never converted (0/200); treatment converted at 50%
        (100/200) - the confidence set clears the null even with no
        finite point. This used to hard-code stat_sig=False."""
        est = _binomial_lift_estimate(x_c=0, n_c=200, x_t=100, n_t=200)
        assert est.stat_sig() is True
        assert estimates_to_readout([est])[0]["stat_sig"] is True

    def test_zero_control_count_thin_data_is_not_stat_sig(self):
        """A thin zero-control-count cell (n=5 each arm) has a confidence
        set wide enough to include the null - stat_sig stays False, not
        flipped to True just because a set exists."""
        est = _binomial_lift_estimate(x_c=0, n_c=5, x_t=1, n_t=5)
        assert est.stat_sig() is False
        assert estimates_to_readout([est])[0]["stat_sig"] is False

    def test_breakout_estimate_set_only_row_matches_the_liftestimate_verdict(self):
        """A BreakoutEstimate has no stat_sig() method of its own - the
        table adapter must derive the same binomial-aware verdict from
        its persisted binomial_set directly."""
        lift_est = _binomial_lift_estimate(x_c=0, n_c=200, x_t=100, n_t=200)
        breakout_est = BreakoutEstimate(
            metric=lift_est.metric,
            group_id=lift_est.group_id,
            method=lift_est.method,
            method_role=lift_est.method_role,
            dimension="country",
            dimension_value="US",
            reference_kind="binomial",
            lift=None,
            binomial_set=lift_est.binomial_set,
        )
        row = estimates_to_readout([breakout_est])[0]
        assert row["stat_sig"] is True
        assert row["lower"] == pytest.approx(lift_est.binomial_set.lower)
        assert row["higher"] == lift_est.binomial_set.upper
        assert row["level"] == pytest.approx(lift_est.binomial_set.level)

    def test_a_set_admitted_at_a_structural_null_refuses_another_null_in_every_adapter(self):
        """At a billion units per arm and ``alpha = 1e-7`` the float margin dominates the tail
        level; a control arm of all successes still rejects a null ratio of two on its
        Clopper-Pearson bound alone, so production cuts and persists that row. Re-read at the
        unshifted null -- a ``LiftEstimate`` copy, or a ``BreakoutEstimate`` twin carrying the
        same set -- every adapter raises the refusal a fresh inversion at that null raises,
        rather than reporting the margin-floored non-rejection its tails would read."""
        from increment.estimation.binomial_rr import FINITE_SAMPLE_MAX_ARM_SIZE, BinomialDataError

        n = FINITE_SAMPLE_MAX_ARM_SIZE
        est = _binomial_lift_estimate(
            x_c=n, n_c=n, x_t=5, n_t=n, alpha=1e-7, alternative="less", null_lift=1.0
        )
        assert est.binomial_set is not None and est.binomial_set.decision_alpha == 1e-7
        assert est.stat_sig() is True
        assert estimates_to_readout([est])[0]["stat_sig"] is True

        retested = est.model_copy(update={"null_lift": 0.0})
        twin = BreakoutEstimate(
            metric=est.metric,
            group_id=est.group_id,
            method=est.method,
            method_role=est.method_role,
            alternative="less",
            dimension="country",
            dimension_value="US",
            reference_kind="binomial",
            lift=est.lift,
            binomial_set=est.binomial_set,
        )
        codes = set()
        for consumer in (retested.stat_sig, retested.p_value):
            with pytest.raises(BinomialDataError) as exc_info:
                consumer()
            codes.add(exc_info.value.code)
        for row in (retested, twin):
            with pytest.raises(BinomialDataError) as exc_info:
                estimates_to_readout([row])
            codes.add(exc_info.value.code)
        assert codes == {"estimation.binomial.tail_unrepresentable"}


class TestNullAbsPrecedesBinomialSet:
    """A row can carry BOTH a persisted ``binomial_set`` (exact-binomial
    reference) and a ``null_abs`` guardrail (an absolute-margin
    secondary estimated on a binomial-eligible metric) -- production
    populates both on the same ``DailyLiftEstimate``
    (``breakout/estimates.py``'s ``_daily_lift_rows_for_slice``, from
    ``policy.null_abs`` and ``estimate.binomial_set``). ``LiftEstimate.
    stat_sig()`` decides such a row on ``null_abs`` first and never
    reaches the binomial branch; the table adapter must reach the exact
    same verdict for the flat ``DailyLiftEstimate``/``BreakoutEstimate``
    twin of that row, not flip it by testing ``binomial_set`` first."""

    def test_daily_lift_estimate_matches_liftestimate_null_abs_verdict(self):
        """20/200 vs 60/200 with null_abs=0.2: the additive interval
        [0.124, 0.276] straddles 0.2, so the null_abs guardrail reports
        not-significant even though the relative binomial set clears its
        own (unshifted) null comfortably -- the two branches disagree,
        so getting the precedence right is the only way to pass."""
        lift_est = _binomial_lift_estimate(x_c=20, n_c=200, x_t=60, n_t=200, null_abs=0.2)
        assert lift_est.null_abs is not None
        assert lift_est.binomial_set is not None
        assert lift_est.stat_sig() is False

        daily_est = DailyLiftEstimate(
            metric=lift_est.metric,
            group_id=lift_est.group_id,
            method=lift_est.method,
            method_role=lift_est.method_role,
            ds=date(2024, 1, 1),
            lift=lift_est.lift,
            binomial_set=lift_est.binomial_set,
            null_abs=lift_est.null_abs,
            abs_lb=lift_est.abs_lb,
            abs_ub=lift_est.abs_ub,
            abs_diff=lift_est.abs_diff,
            abs_se=lift_est.abs_se,
            alternative=lift_est.alternative,
            null_lift=lift_est.null_lift,
            reference_kind=lift_est.reference_kind,
            sampling_available=True,
        )
        assert estimates_to_readout([daily_est])[0]["stat_sig"] is lift_est.stat_sig()


def test_estimates_to_readout_preserves_order_and_count():
    """estimates_to_readout maps 1:1, preserving order."""
    ests = [
        _estimate("a", "T", "unadjusted", 0.1, lb=0.0, ub=0.2),
        _estimate("b", "T", "unadjusted", -0.05, lb=-0.1, ub=0.0),
    ]
    rows = estimates_to_readout(ests)
    assert len(rows) == 2


def test_estimates_to_readout_preserves_lift_family_metadata():
    est = _estimate("revenue", "T", "unadjusted", 0.1, lb=0.0, ub=0.2).model_copy(
        update={
            "role": "secondary",
            "discovery": True,
            "family_axes": ("metric", "arm"),
            "family_q": 0.1,
            "family_threshold": 0.05,
        }
    )
    (row,) = estimates_to_readout([est])
    assert row["role"] == "secondary"
    assert row["discovery"] is True
    assert row["family_axes"] == ("metric", "arm")
    assert row["family_q"] == pytest.approx(0.1)
    assert row["family_threshold"] == pytest.approx(0.05)


def test_estimates_to_readout_preserves_breakout_family_metadata():
    est = _breakout_estimate(
        "revenue", "T", "unadjusted", "country", "US", 0.1, lb=0.0, ub=0.2
    ).model_copy(
        update={
            "role": "exploratory",
            "discovery": False,
            "family_axes": ("metric", "arm", "segment"),
            "family_q": 0.05,
            "family_threshold": 0.025,
        }
    )
    (row,) = estimates_to_readout([est])
    assert row["role"] == "exploratory"
    assert row["discovery"] is False
    assert row["family_axes"] == ("metric", "arm", "segment")
    assert row["family_q"] == pytest.approx(0.05)
    assert row["family_threshold"] == pytest.approx(0.025)


def _valid_estimate(
    metric="revenue", group_id="T", prior=None, preferred_direction=None, null_lift=0.0
):
    """A well-formed Normal estimate for posterior-qualified readout tests."""
    from increment.estimation.inference import infer_lift

    return infer_lift(
        metric=metric,
        group_id=group_id,
        method="unadjusted",
        method_role="decision",
        log_rr=_VALID_LOG_T - _VALID_LOG_C,
        se_t=_VALID_SE_T,
        se_c=_VALID_SE_C,
        prior=prior,
        preferred_direction=preferred_direction,
        null_lift=null_lift,
    )


_VALID_LOG_T = 0.15
_VALID_SE_T = 0.03
_VALID_LOG_C = 0.05
_VALID_SE_C = 0.04


def test_estimates_to_readout_qualifies_only_stored_posterior_values():
    from increment.estimation.inference import Normal

    no_prior = _valid_estimate()
    (no_prior_row,) = estimates_to_readout([no_prior])
    assert no_prior_row["posterior_chance_to_beat"] is None
    assert no_prior_row["posterior_risk_if_shipped"] is None
    assert no_prior_row["posterior_prob_favorable"] is None
    assert no_prior_row["posterior_components"] is None
    assert not {"chance_to_beat", "risk_if_shipped", "prob_favorable"} & no_prior_row.keys()

    posterior = _valid_estimate(prior=Normal(mu=0.0, sigma=0.1))
    (row,) = estimates_to_readout([posterior])
    assert row["posterior_chance_to_beat"] == pytest.approx(posterior.chance_to_beat())
    assert row["posterior_risk_if_shipped"] == pytest.approx(posterior.risk_if_shipped())
    assert "chance_to_beat" not in row
    assert "risk_if_shipped" not in row


def test_readout_uses_stored_posterior_for_clustered_and_binomial_rows():
    from increment.estimation.engine import Method
    from increment.estimation.inference import Normal
    from increment.estimation.results import LiftEstimate

    posterior = _valid_estimate(
        metric="conv",
        prior=Normal(mu=0.0, sigma=0.1),
    )
    clustered = LiftEstimate.model_validate(
        {**posterior.model_dump(mode="python"), "n_clusters": 3}
    )
    exact = _binomial_lift_estimate(
        x_c=10,
        n_c=100,
        x_t=20,
        n_t=100,
        methods=[Method(name="unadjusted", conversion_inference="finite_sample")],
    )
    assert exact.binomial_set is not None
    payload = posterior.model_dump(mode="python")
    payload.update(
        reference_kind="binomial",
        lift=exact.lift,
        binomial_set=exact.binomial_set,
    )
    binomial = LiftEstimate.model_validate(payload)

    (clustered_row, binomial_row) = estimates_to_readout([clustered, binomial])
    for estimate, rendered in (
        (clustered, clustered_row),
        (binomial, binomial_row),
    ):
        assert estimate.posterior_available is True
        assert rendered["posterior_chance_to_beat"] == pytest.approx(estimate.chance_to_beat())
        assert rendered["posterior_risk_if_shipped"] == pytest.approx(estimate.risk_if_shipped())


def test_estimates_to_readout_preserves_direction_aware_stored_posterior():
    from increment.estimation.inference import Normal, infer_lift

    est = infer_lift(
        metric="latency",
        group_id="T",
        method="unadjusted",
        method_role="decision",
        log_rr=-0.10,
        se_t=0.03,
        se_c=0.04,
        alternative="less",
        preferred_direction="decrease",
        prior=Normal(mu=0.0, sigma=0.1),
    )
    (row,) = estimates_to_readout([est])
    assert row["posterior_chance_to_beat"] == pytest.approx(est.chance_to_beat_favorable())
    assert row["posterior_risk_if_shipped"] == pytest.approx(est.risk_if_shipped_favorable())
    assert row["posterior_prob_favorable"] == pytest.approx(est.prob_favorable())


def test_estimates_to_readout_preserves_persisted_favorable_probability_at_shifted_null():
    from increment.estimation.inference import Normal, infer_lift

    est = infer_lift(
        metric="revenue",
        group_id="T",
        method="unadjusted",
        method_role="decision",
        log_rr=0.10,
        se_t=0.04,
        se_c=0.03,
        null_lift=0.08,
        preferred_direction="increase",
        prior=Normal(mu=0.0, sigma=0.1),
    )
    assert est.posterior_prob_favorable == pytest.approx(est.prob_favorable())
    assert est.posterior_prob_favorable != pytest.approx(est.chance_to_beat())
    (row,) = estimates_to_readout([est])
    assert row["posterior_prob_favorable"] == est.posterior_prob_favorable


def test_estimates_to_readout_preserves_breakout_and_daily_posterior_probability():
    from increment.breakout.estimates import _copy_common_fields
    from increment.estimation.inference import Normal, infer_lift

    lift = infer_lift(
        metric="revenue",
        group_id="T",
        method="unadjusted",
        method_role="decision",
        log_rr=0.10,
        se_t=0.04,
        se_c=0.03,
        preferred_direction="increase",
        prior=Normal(mu=0.0, sigma=0.1),
    )
    breakout = BreakoutEstimate(
        **_copy_common_fields(lift),
        dimension="country",
        dimension_value="US",
        source="observed",
    )
    daily = DailyLiftEstimate(
        **_copy_common_fields(lift),
        ds=date(2025, 1, 1),
    )
    for row_model in (breakout, daily):
        (row,) = estimates_to_readout([row_model])
        assert row["posterior_prob_favorable"] == lift.posterior_prob_favorable


def test_readout_rows_keep_canonical_mixture_components_for_every_result_view():
    from increment._canonical import canonical_json_bytes
    from increment.breakout.estimates import _copy_common_fields
    from increment.estimation.inference import infer_lift
    from increment.estimation.priors import MixturePrior

    lift = infer_lift(
        metric="revenue",
        group_id="T",
        method="unadjusted",
        method_role="decision",
        log_rr=0.1,
        se_t=0.04,
        se_c=0.03,
        preferred_direction="increase",
        prior=MixturePrior(weights=(0.4, 0.6), means=(0.0, 0.1), sigmas=(0.02, 0.15)),
    )
    assert lift.posterior_components is not None
    expected = canonical_json_bytes(lift.posterior_components.model_dump(mode="json")).decode()
    breakout = BreakoutEstimate(
        **_copy_common_fields(lift), dimension="country", dimension_value="US", source="observed"
    )
    daily = DailyLiftEstimate(**_copy_common_fields(lift), ds=date(2025, 1, 1))
    for estimate in (lift, breakout, daily):
        (row,) = estimates_to_readout([estimate])
        assert row["posterior_components"] == expected


def test_estimates_to_readout_keeps_posterior_values_missing_without_direction():
    (row,) = estimates_to_readout([_valid_estimate()])
    assert row["posterior_prob_favorable"] is None


def test_estimates_to_readout_leaves_posterior_values_missing_for_breakout_and_sequential():
    breakout = _breakout_estimate(
        "revenue", "T", "unadjusted", "country", "US", 0.12, lb=0.02, ub=0.22
    )
    (breakout_row,) = estimates_to_readout([breakout])
    assert breakout_row["posterior_chance_to_beat"] is None
    sequential = _certified_row()
    (sequential_row,) = estimates_to_readout([sequential])
    assert sequential_row["posterior_chance_to_beat"] is None
    assert sequential_row["posterior_risk_if_shipped"] is None


def test_tables_module_imports_without_tables_extra():
    """The module imports without coeftable; only rendering requires the extra."""
    import importlib

    import increment.tables as readout_module

    importlib.reload(readout_module)  # confirm a fresh import succeeds too
    assert hasattr(readout_module, "estimates_to_readout")


def test_readout_table_renders_two_metrics_one_method():
    """A two-metric, one-method frame renders and shows both metric labels."""
    pytest.importorskip("coeftable")
    ests = [
        _estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22),
        _estimate("signups", "T", "unadjusted", -0.05, lb=-0.12, ub=0.03),
    ]
    rows = estimates_to_readout(ests)
    table = readout_table(rows)
    html = table.gt().as_raw_html()
    assert "revenue" in html
    assert "signups" in html
    assert "Lift %" in html


def test_readout_table_marks_known_multiplicity_when_other_status_is_missing():
    pytest.importorskip("coeftable")
    known, missing = estimates_to_readout(
        [
            _estimate("m", "T", "unadjusted", 0.12, lb=0.02, ub=0.22),
            _estimate("other", "T", "unadjusted", 0.12, lb=0.02, ub=0.22),
        ]
    )
    known["multiplicity_status"] = "exploratory_unadjusted"
    missing["multiplicity_status"] = None

    html = readout_table([known, missing]).gt().as_raw_html()

    assert "m¹" in html
    assert "¹ Exploratory, unadjusted for multiplicity (1 of 2 rows)" in html
    assert "other¹" not in html


def test_readout_table_disambiguates_metric_colliding_with_status_marker():
    pytest.importorskip("coeftable")
    exploratory, declared = estimates_to_readout(
        [
            _estimate("m", "T", "unadjusted", 0.12, lb=0.02, ub=0.22),
            _estimate("m¹", "T", "unadjusted", 0.12, lb=0.02, ub=0.22),
        ]
    )
    exploratory["multiplicity_status"] = "exploratory_unadjusted"
    declared["multiplicity_status"] = "declared_plan"

    html = readout_table([exploratory, declared]).gt().as_raw_html()

    assert "m¹ (Exploratory, unadjusted for multiplicity)" in html
    assert "m¹ (Declared plan)" in html


def test_readout_table_marks_declared_plan_when_other_status_is_missing():
    pytest.importorskip("coeftable")
    declared, missing = estimates_to_readout(
        [
            _estimate("planned", "T", "unadjusted", 0.12, lb=0.02, ub=0.22),
            _estimate("unknown", "T", "unadjusted", 0.12, lb=0.02, ub=0.22),
        ]
    )
    declared["multiplicity_status"] = "declared_plan"
    missing["multiplicity_status"] = None

    html = readout_table([declared, missing]).gt().as_raw_html()

    assert "planned¹" in html
    assert "¹ Declared plan (1 of 2 rows)" in html
    assert "unknown¹" not in html


def test_readout_table_disambiguates_metric_matching_generated_suffix():
    pytest.importorskip("coeftable")
    rows = estimates_to_readout(
        [
            _estimate("m", "T", "unadjusted", 0.12, lb=0.02, ub=0.22),
            _estimate("m¹", "T", "unadjusted", 0.12, lb=0.02, ub=0.22),
            _estimate(
                "m¹ (Unknown multiplicity status)",
                "T",
                "unadjusted",
                0.12,
                lb=0.02,
                ub=0.22,
            ),
        ]
    )
    rows[0]["multiplicity_status"] = "exploratory_unadjusted"
    rows[1]["multiplicity_status"] = None
    rows[2]["multiplicity_status"] = None

    html = readout_table(rows).gt().as_raw_html()

    assert "m¹ (Exploratory, unadjusted for multiplicity)" in html
    assert "m¹ (Unknown multiplicity status)" in html
    assert "m¹ (Unknown multiplicity status) [row 2]" in html


def test_readout_table_splits_columns_for_two_methods():
    """Two distinct method values produce a split-column table."""
    pytest.importorskip("coeftable")
    ests = [
        _estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22),
        _estimate("revenue", "T", "adjusted", 0.10, lb=0.03, ub=0.17),
    ]
    rows = estimates_to_readout(ests)
    table = readout_table(rows)
    html = table.gt().as_raw_html()
    assert "unadjusted" in html
    assert "adjusted" in html


def test_readout_table_nest_by_method_stacks_methods():
    """nest_by='method' stacks methods as row values instead of splitting
    them into side-by-side columns - the inverse layout of the default
    'arm' behavior exercised above, for a single-arm, two-method frame."""
    pytest.importorskip("coeftable")
    ests = [
        _estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22),
        _estimate("revenue", "T", "adjusted", 0.10, lb=0.03, ub=0.17),
    ]
    rows = estimates_to_readout(ests)
    table = readout_table(rows, nest_by="method")
    html = table.gt().as_raw_html()
    # Stacked as row content, not a column-spanner header - the inverse of
    # test_readout_table_splits_columns_for_two_methods above.
    assert 'gt_row gt_center">unadjusted' in html
    assert 'gt_row gt_center">adjusted' in html
    assert 'gt_column_spanner">unadjusted' not in html
    assert 'gt_column_spanner">adjusted' not in html


def test_readout_table_handles_missing_ci_bounds():
    """A row with no CI (NaN lower/higher) renders without raising."""
    pytest.importorskip("coeftable")
    ests = [
        _estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22),
        _estimate("signups", "T", "unadjusted", 0.05),  # point-only -> NaN bounds
    ]
    rows = estimates_to_readout(ests)
    table = readout_table(rows)
    html = table.gt().as_raw_html()
    assert "signups" in html


def test_readout_table_list_of_dicts_raises_no_warning():
    """The primary list[dict] path (as produced by estimates_to_readout) must
    render without triggering great_tables' pyarrow-experimental warning or
    any other UserWarning - it builds a pandas frame internally."""
    pytest.importorskip("coeftable")
    ests = [_estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22)]
    rows = estimates_to_readout(ests)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        table = readout_table(rows)
        html = table.gt().as_raw_html()
    assert "revenue" in html


def test_readout_table_accepts_a_prebuilt_pandas_frame():
    """A caller-supplied pandas DataFrame is used as-is, no pandas import in
    the module itself, and rendering it raises no warning."""
    pytest.importorskip("coeftable")
    pd = pytest.importorskip("pandas")
    ests = [_estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22)]
    frame = pd.DataFrame(estimates_to_readout(ests))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        table = readout_table(frame)
        html = table.gt().as_raw_html()
    assert "revenue" in html


def test_readout_table_accepts_a_prebuilt_pyarrow_table():
    """A caller-supplied pyarrow Table is passed through narwhals as-is.
    great_tables warns unconditionally on pyarrow input; that warning is
    upstream and expected here, so it's asserted explicitly."""
    pytest.importorskip("coeftable")
    pa = pytest.importorskip("pyarrow")
    ests = [_estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22)]
    frame = pa.Table.from_pylist(estimates_to_readout(ests))
    table = readout_table(frame)
    with pytest.warns(UserWarning, match="experimental"):
        html = table.gt().as_raw_html()
    assert "revenue" in html


def test_estimates_to_readout_with_breakout_estimate_adds_segment_columns():
    """A BreakoutEstimate row carries `segment` (dimension_value) and
    `dimension` alongside the same keys `_liftestimate_to_row` produces -
    it has every field that function reads, so it flows through the same
    duck-typed path with two extra keys layered on top."""
    est = _breakout_estimate(
        "purchase_rate", "T", "unadjusted", "country", "US", 0.20, lb=0.05, ub=0.35
    )
    rows = estimates_to_readout([est])
    assert len(rows) == 1
    assert rows[0]["segment"] == "US"
    assert rows[0]["dimension"] == "country"
    assert rows[0]["metric"] == "purchase_rate"
    assert rows[0]["method"] == "unadjusted"
    assert rows[0]["lift"] == 0.20
    assert rows[0]["lower"] == 0.05
    assert rows[0]["higher"] == 0.35
    assert rows[0]["group_id"] == "T"
    assert "source" not in rows[0]  # BreakoutEstimate.source defaults to None


def test_estimates_to_readout_with_breakout_estimate_source_column():
    """A BreakoutEstimate carrying a `source` (e.g. from
    Analysis.run_breakout, which always resolves one) adds a
    `source` column too, disambiguating two breakouts that share a
    `dimension`/`dimension_value` but resolve to different FactSources."""
    est = _breakout_estimate(
        "purchase_rate", "T", "unadjusted", "plan_tier", "pro", 0.20, source="billing_events"
    )
    rows = estimates_to_readout([est])
    assert rows[0]["source"] == "billing_events"


def test_estimates_to_readout_with_breakout_estimate_excluded_none_for_a_live_row():
    """A real, live BreakoutEstimate (from a normal estimated cell) gets an
    `excluded` key set to `None`, meaningful here (the "not excluded" signal),
    so the key is always present, never omitted."""
    est = _breakout_estimate("purchase_rate", "T", "unadjusted", "country", "US", 0.20)
    rows = estimates_to_readout([est])
    assert rows[0]["excluded"] is None


def test_estimates_to_readout_with_breakout_estimate_excluded_reason_column():
    """A dense NaN row `run_breakout` emits for an excluded cell carries
    its `ExclusionReason` through to the readout row; the readout layer
    used to drop it entirely."""
    est = _breakout_estimate(
        "purchase_rate",
        "T",
        "unadjusted",
        "country",
        "MX",
        float("nan"),
        excluded="no_control_arm",
    )
    rows = estimates_to_readout([est])
    assert rows[0]["excluded"] == "no_control_arm"


def test_estimates_to_readout_lift_estimate_has_no_excluded_key():
    """A plain `LiftEstimate` row has no `excluded` concept at all: the
    key is absent, not `None`, same as `segment`/`dimension`/`source`."""
    est = _estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22)
    rows = estimates_to_readout([est])
    assert "excluded" not in rows[0]


def test_estimates_to_readout_daily_estimate_carries_ds():
    """A DailyLiftEstimate row keeps the type's defining axis: each row
    carries its own `ds`, so a 2-day list stays distinguishable by day
    rather than only by order."""
    ests = [
        _daily_estimate("revenue", "T", "unadjusted", date(2024, 1, 1), 0.05, lb=-0.01, ub=0.11),
        _daily_estimate("revenue", "T", "unadjusted", date(2024, 1, 2), 0.08, lb=0.01, ub=0.15),
    ]
    rows = estimates_to_readout(ests)
    assert [row["ds"] for row in rows] == [date(2024, 1, 1), date(2024, 1, 2)]


def test_estimates_to_readout_daily_estimate_has_no_excluded_key():
    """DailyLiftEstimate has no `excluded` field, so its row omits the key
    entirely - a defaulted `None` would falsely read as "real estimate" for
    a NaN daily row, breaking the documented `excluded` contract."""
    est = _daily_estimate("revenue", "T", "unadjusted", date(2024, 1, 1), 0.05, lb=-0.01, ub=0.11)
    rows = estimates_to_readout([est])
    assert "excluded" not in rows[0]


def test_estimates_to_readout_mixed_list_of_lift_and_breakout_estimates():
    """A mixed list of LiftEstimate and BreakoutEstimate converts correctly:
    the LiftEstimate row has no segment/dimension keys (absent, not None),
    while the BreakoutEstimate row does - detected per-row by duck-typing
    on `dimension_value` (the two share no base class to `isinstance` on)."""
    ests = [
        _estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22),
        _breakout_estimate("revenue", "T", "unadjusted", "country", "US", 0.20, lb=0.05, ub=0.35),
    ]
    rows = estimates_to_readout(ests)
    assert len(rows) == 2
    assert "segment" not in rows[0]
    assert "dimension" not in rows[0]
    assert rows[1]["segment"] == "US"
    assert rows[1]["dimension"] == "country"


def test_estimates_to_readout_mixed_list_frame_has_nan_for_missing_segment():
    """Downstream, `pd.DataFrame(data)` - exactly what readout_table's
    list-input branch builds internally - turns the LiftEstimate row's
    absent segment/dimension keys into NaN instead of raising."""
    pd = pytest.importorskip("pandas")
    ests = [
        _estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22),
        _breakout_estimate("revenue", "T", "unadjusted", "country", "US", 0.20, lb=0.05, ub=0.35),
    ]
    frame = pd.DataFrame(estimates_to_readout(ests))
    assert frame.loc[0, "segment"] != frame.loc[0, "segment"]  # NaN != NaN
    assert frame.loc[1, "segment"] == "US"


def test_readout_table_nest_by_segment_builds_for_single_method():
    """nest_by='segment' nests by the `segment` column and splits by
    `group_id` (arm) - the per-segment lift breakdown for one method."""
    pytest.importorskip("coeftable")
    ests = [
        _breakout_estimate(
            "purchase_rate", "T", "unadjusted", "country", "US", 0.12, lb=0.02, ub=0.22
        ),
        _breakout_estimate(
            "purchase_rate", "T", "unadjusted", "country", "GB", 0.05, lb=-0.02, ub=0.12
        ),
    ]
    rows = estimates_to_readout(ests)
    table = readout_table(rows, nest_by="segment")
    html = table.gt().as_raw_html()
    assert "US" in html
    assert "GB" in html


def test_readout_table_nest_by_segment_with_trend_renders_per_segment_sparklines():
    """The headline use case (`examples/breakout.py`): `nest_by='segment'`
    plus a dimensioned trend renders one sparkline per segment row - the
    success-path counterpart to
    `test_readout_table_trend_dimensioned_frame_raises_for_default_nest_by`,
    which only exercises the raise branch."""
    pytest.importorskip("coeftable")
    ests = [
        _breakout_estimate(
            "purchase_rate", "T", "unadjusted", "country", "US", 0.12, lb=0.02, ub=0.22
        ),
        _breakout_estimate(
            "purchase_rate", "T", "unadjusted", "country", "GB", 0.05, lb=-0.02, ub=0.12
        ),
    ]
    rows = estimates_to_readout(ests)
    trend_ests = [
        _daily_estimate(
            "purchase_rate",
            "T",
            "unadjusted",
            date(2024, 1, 1),
            0.10,
            lb=0.00,
            ub=0.20,
            dimension="country",
            dimension_value="US",
        ),
        _daily_estimate(
            "purchase_rate",
            "T",
            "unadjusted",
            date(2024, 1, 2),
            0.12,
            lb=0.02,
            ub=0.22,
            dimension="country",
            dimension_value="US",
        ),
        _daily_estimate(
            "purchase_rate",
            "T",
            "unadjusted",
            date(2024, 1, 1),
            0.03,
            lb=-0.05,
            ub=0.11,
            dimension="country",
            dimension_value="GB",
        ),
        _daily_estimate(
            "purchase_rate",
            "T",
            "unadjusted",
            date(2024, 1, 2),
            0.05,
            lb=-0.02,
            ub=0.12,
            dimension="country",
            dimension_value="GB",
        ),
    ]
    table = readout_table(rows, nest_by="segment", trend=trend_ests, trend_label="Trend")
    html = table.gt().as_raw_html()
    assert "US" in html
    assert "GB" in html
    assert "Trend" in html


def test_readout_table_nest_by_segment_raises_for_multiple_methods():
    """nest_by='segment' has nowhere to put a method axis - CoefTable is
    inherently 2-axis, and segment/arm already occupy both slots. `data`
    spanning more than one method must raise loudly."""
    pytest.importorskip("coeftable")
    ests = [
        _breakout_estimate(
            "purchase_rate", "T", "unadjusted", "country", "US", 0.12, lb=0.02, ub=0.22
        ),
        _breakout_estimate("purchase_rate", "T", "cuped", "country", "US", 0.10, lb=0.03, ub=0.17),
    ]
    rows = estimates_to_readout(ests)
    from increment.errors import InvalidRequestError

    with pytest.raises(InvalidRequestError) as exc_info:
        readout_table(rows, nest_by="segment")
    assert exc_info.value.code == "tables.readout_table_nest"


def test_readout_table_trend_renders_sparkline_column():
    """A breakout frame plus a matching trend frame renders a sparkline
    column labelled `trend_label`, containing at least one inline SVG."""
    pytest.importorskip("coeftable")
    ests = [_estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22)]
    rows = estimates_to_readout(ests)
    trend_ests = [
        _daily_estimate("revenue", "T", "unadjusted", date(2024, 1, 1), 0.05, lb=-0.01, ub=0.11),
        _daily_estimate("revenue", "T", "unadjusted", date(2024, 1, 2), 0.08, lb=0.01, ub=0.15),
    ]
    table = readout_table(rows, trend=trend_ests)
    html = table.gt().as_raw_html()
    assert "Trend" in html
    assert "<svg" in html


def test_readout_table_trend_none_matches_omitting_the_argument():
    """`trend=None` (explicit) renders byte-identical HTML to a call that
    omits `trend` entirely - both build off the same shared rows list.
    great_tables assigns a fresh random id per `.gt()` call, so that
    token is normalized out before comparing (see `_normalize_table_id`)."""
    pytest.importorskip("coeftable")
    ests = [_estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22)]
    rows = estimates_to_readout(ests)
    html_default = _normalize_table_id(readout_table(rows).gt().as_raw_html())
    html_explicit_none = _normalize_table_id(readout_table(rows, trend=None).gt().as_raw_html())
    assert html_default == html_explicit_none


def test_readout_table_trend_accepts_daily_lift_estimates_sequence():
    """A raw `Sequence[DailyLiftEstimate]` and its `to_frame()` equivalent
    produce identical HTML, modulo great_tables' per-call random id (see
    `_normalize_table_id`)."""
    pytest.importorskip("coeftable")
    ests = [_estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22)]
    rows = estimates_to_readout(ests)
    trend_ests = [
        _daily_estimate("revenue", "T", "unadjusted", date(2024, 1, 1), 0.05, lb=-0.01, ub=0.11),
        _daily_estimate("revenue", "T", "unadjusted", date(2024, 1, 2), 0.08, lb=0.01, ub=0.15),
    ]
    html_from_list = _normalize_table_id(readout_table(rows, trend=trend_ests).gt().as_raw_html())
    trend_frame = to_frame(trend_ests, model=DailyLiftEstimate)
    html_from_frame = _normalize_table_id(readout_table(rows, trend=trend_frame).gt().as_raw_html())
    assert html_from_list == html_from_frame


def test_readout_table_trend_missing_key_column_raises():
    """A trend frame missing a required key column (here `group_id`, the
    default nest_by='arm' nest column) raises ValueError naming it."""
    pytest.importorskip("coeftable")
    pd = pytest.importorskip("pandas")
    ests = [_estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22)]
    rows = estimates_to_readout(ests)
    trend_frame = pd.DataFrame(
        {
            "metric": ["revenue"],
            "ds": [date(2024, 1, 1)],
            "lift": [0.05],
            "lb": [-0.01],
            "ub": [0.11],
        }
    )
    from increment.errors import InvalidRequestError

    with pytest.raises(InvalidRequestError) as exc_info:
        readout_table(rows, trend=trend_frame)
    assert exc_info.value.code == "tables.readout_table_trend_missing_key_columns"
    assert exc_info.value.context["missing"] == ("group_id",)


def test_readout_table_trend_dimensioned_frame_raises_for_default_nest_by():
    """A dimensioned trend frame (carrying `segment`) passed with the
    default nest_by='arm' raises: `segment` isn't a nest_by='arm' grouping
    key, so per-day segment rows would collapse onto the same
    (metric, group_id, ds) key - the highest-value misuse case."""
    pytest.importorskip("coeftable")
    ests = [_estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22)]
    rows = estimates_to_readout(ests)
    trend_ests = [
        _daily_estimate(
            "revenue",
            "T",
            "unadjusted",
            date(2024, 1, 1),
            0.05,
            lb=-0.01,
            ub=0.11,
            dimension="country",
            dimension_value="US",
        ),
        _daily_estimate(
            "revenue",
            "T",
            "unadjusted",
            date(2024, 1, 1),
            0.03,
            lb=-0.02,
            ub=0.08,
            dimension="country",
            dimension_value="GB",
        ),
    ]
    from increment.errors import InvalidRequestError

    with pytest.raises(InvalidRequestError) as exc_info:
        readout_table(rows, trend=trend_ests)
    assert exc_info.value.code == "tables.readout_table_trend_duplicate_rows"
    assert "segment" in exc_info.value.context["varying"]  # ty: ignore[unsupported-operator]


def test_readout_table_trend_collapsed_split_column_raises():
    """`data`'s split column (`method`) has one distinct value so
    `split_columns=None`, but the trend frame carries two - those merge
    under the reduced key set. The message names `method`, the column
    that actually varies, not a hardcoded label."""
    pytest.importorskip("coeftable")
    ests = [_estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22)]
    rows = estimates_to_readout(ests)
    trend_ests = [
        _daily_estimate("revenue", "T", "unadjusted", date(2024, 1, 1), 0.05, lb=-0.01, ub=0.11),
        _daily_estimate("revenue", "T", "adjusted", date(2024, 1, 1), 0.07, lb=0.01, ub=0.13),
    ]
    from increment.errors import InvalidRequestError

    with pytest.raises(InvalidRequestError) as exc_info:
        readout_table(rows, trend=trend_ests)
    assert exc_info.value.code == "tables.readout_table_trend_duplicate_rows"
    assert "method" in exc_info.value.context["varying"]  # ty: ignore[unsupported-operator]


def test_readout_table_trend_duplicate_rows_same_key_raises():
    """Two trend rows sharing every active grouping key (metric, group_id,
    method, ds) raise ValueError, independent of any dimension/collapsed-split cause."""
    pytest.importorskip("coeftable")
    ests = [
        _estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22),
        _estimate("revenue", "T", "adjusted", 0.10, lb=0.03, ub=0.17),
    ]
    rows = estimates_to_readout(ests)
    trend_ests = [
        _daily_estimate("revenue", "T", "unadjusted", date(2024, 1, 1), 0.05, lb=-0.01, ub=0.11),
        _daily_estimate("revenue", "T", "unadjusted", date(2024, 1, 1), 0.06, lb=-0.02, ub=0.12),
        _daily_estimate("revenue", "T", "adjusted", date(2024, 1, 1), 0.04, lb=-0.03, ub=0.10),
    ]
    from increment.errors import InvalidRequestError

    with pytest.raises(InvalidRequestError) as exc_info:
        readout_table(rows, trend=trend_ests)
    assert exc_info.value.code == "tables.readout_table_trend_duplicate_rows"
    assert exc_info.value.context["key"]["metric"] == "revenue"  # ty: ignore[not-subscriptable]


def test_readout_table_trend_missing_metric_renders_blank_without_raising():
    """A metric present in `data` but absent from `trend` (e.g. a newly
    added `d7_retention` metric with no daily history yet) renders a
    blank sparkline cell for that row instead of raising."""
    pytest.importorskip("coeftable")
    ests = [
        _estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22),
        _estimate("d7_retention", "T", "unadjusted", 0.05, lb=-0.01, ub=0.11),
    ]
    rows = estimates_to_readout(ests)
    trend_ests = [
        _daily_estimate("revenue", "T", "unadjusted", date(2024, 1, 1), 0.05, lb=-0.01, ub=0.11),
        _daily_estimate("revenue", "T", "unadjusted", date(2024, 1, 2), 0.08, lb=0.01, ub=0.15),
    ]
    html = readout_table(rows, trend=trend_ests).gt().as_raw_html()
    assert "d7_retention" in html


def test_readout_table_trend_nan_point_renders_without_raising():
    """A NaN point in the middle of an otherwise-present series renders as
    a gap, not a crash - not asserted pixel-exact, just no raise and the
    metric still shows up in the output."""
    pytest.importorskip("coeftable")
    ests = [_estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22)]
    rows = estimates_to_readout(ests)
    trend_ests = [
        _daily_estimate("revenue", "T", "unadjusted", date(2024, 1, 1), 0.05, lb=-0.01, ub=0.11),
        _daily_estimate(
            "revenue",
            "T",
            "unadjusted",
            date(2024, 1, 2),
            float("nan"),
            lb=float("nan"),
            ub=float("nan"),
        ),
        _daily_estimate("revenue", "T", "unadjusted", date(2024, 1, 3), 0.09, lb=0.02, ub=0.16),
    ]
    html = readout_table(rows, trend=trend_ests).gt().as_raw_html()
    assert "revenue" in html


def test_readout_table_trend_mixed_ds_basis_raises():
    """A trend frame mixing a calendar-indexed and a cohort-indexed series
    raises, naming the mixed bases and each basis's metrics: a shared
    'ds' tick would name an observation date for one series and a unit's
    exposure date for the other."""
    pytest.importorskip("coeftable")
    ests = [
        _estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22),
        _estimate("d7_retention", "T", "unadjusted", 0.05, lb=-0.01, ub=0.11),
    ]
    rows = estimates_to_readout(ests)
    trend_ests = [
        _daily_estimate("revenue", "T", "unadjusted", date(2024, 1, 1), 0.05, lb=-0.01, ub=0.11),
        _daily_estimate(
            "d7_retention",
            "T",
            "unadjusted",
            date(2024, 1, 1),
            0.03,
            lb=-0.02,
            ub=0.08,
            ds_basis="cohort",
        ),
    ]
    from increment.errors import InvalidRequestError

    with pytest.raises(InvalidRequestError) as exc_info:
        readout_table(rows, trend=trend_ests)
    assert exc_info.value.code == "tables.readout_table_trend_mixed_ds_basis"
    assert exc_info.value.context["bases"] == ("calendar", "cohort")
    affected = exc_info.value.context["affected"]
    assert "d7_retention" in affected and "revenue" in affected  # ty: ignore[unsupported-operator]


def test_readout_table_trend_all_cohort_ds_basis_passes():
    """Every row 'cohort' is a SINGLE basis - the guard rejects mixing,
    not the cohort basis itself. A pure cohort-view trend renders."""
    pytest.importorskip("coeftable")
    ests = [_estimate("d7_retention", "T", "unadjusted", 0.05, lb=-0.01, ub=0.11)]
    rows = estimates_to_readout(ests)
    trend_ests = [
        _daily_estimate(
            "d7_retention",
            "T",
            "unadjusted",
            date(2024, 1, 1),
            0.03,
            lb=-0.02,
            ub=0.08,
            ds_basis="cohort",
        ),
        _daily_estimate(
            "d7_retention",
            "T",
            "unadjusted",
            date(2024, 1, 2),
            0.06,
            lb=0.01,
            ub=0.11,
            ds_basis="cohort",
        ),
    ]
    html = readout_table(rows, trend=trend_ests).gt().as_raw_html()
    assert "d7_retention" in html
    assert "<svg" in html


def test_readout_table_trend_without_ds_basis_column_passes():
    """A trend frame that never carried ds_basis at all (e.g. a hand-built
    frame) passes the guard unchanged - the column is optional."""
    pytest.importorskip("coeftable")
    pd = pytest.importorskip("pandas")
    ests = [_estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22)]
    rows = estimates_to_readout(ests)
    trend_frame = pd.DataFrame(
        {
            "metric": ["revenue", "revenue"],
            "group_id": ["T", "T"],
            "ds": [date(2024, 1, 1), date(2024, 1, 2)],
            "lift": [0.05, 0.08],
            "lb": [-0.01, 0.01],
            "ub": [0.11, 0.15],
        }
    )
    html = readout_table(rows, trend=trend_frame).gt().as_raw_html()
    assert "revenue" in html
    assert "<svg" in html


def test_trend_table_renders_html():
    pytest.importorskip("coeftable")
    import datetime as dt

    import pandas as pd

    from increment.tables import trend_table

    frame = pd.DataFrame(
        {
            "metric": ["purchase_rate"] * 3,
            "grain": ["week"] * 3,
            "period": [dt.date(2026, 1, 5), dt.date(2026, 1, 12), dt.date(2026, 1, 19)],
            "period_complete": [True, True, False],
            "n": [100, 110, 40],
            "value": [0.10, 0.12, 0.11],
            "ci_lb": [0.08, 0.10, 0.06],
            "ci_ub": [0.12, 0.14, 0.16],
        }
    )
    table = trend_table(frame)
    html = table.gt().as_raw_html()
    assert "purchase_rate" in html


def test_trend_table_rejects_missing_columns():
    pytest.importorskip("coeftable")
    import pandas as pd

    from increment.errors import InvalidRequestError
    from increment.tables import trend_table

    with pytest.raises(InvalidRequestError) as exc_info:
        trend_table(pd.DataFrame({"metric": ["m"], "period": [None], "value": [1.0]}))
    assert exc_info.value.code == "tables.trend_table_missing"
    assert exc_info.value.context["missing"] == ("period_complete",)


def test_trend_table_headline_falls_back_per_metric_not_globally():
    # Metric A has a complete period; metric B has none. The fallback-to-
    # latest-period rule applies per metric, so B must not vanish from the headline.
    pytest.importorskip("coeftable")
    import pandas as pd

    from increment.tables import trend_table

    frame = pd.DataFrame(
        {
            "metric": ["metric_a", "metric_a", "metric_b", "metric_b"],
            "grain": ["week"] * 4,
            "period": [
                date(2026, 1, 5),
                date(2026, 1, 12),
                date(2026, 1, 5),
                date(2026, 1, 12),
            ],
            "period_complete": [True, False, False, False],
            "n": [100, 90, 50, 55],
            "value": [0.10, 0.11, 0.20, 0.22],
            "ci_lb": [0.08, 0.09, 0.18, 0.20],
            "ci_ub": [0.12, 0.13, 0.22, 0.24],
        }
    )
    html = trend_table(frame).gt().as_raw_html()
    assert "metric_a" in html
    assert "metric_b" in html


def test_trend_table_nests_by_dimension_column():
    pytest.importorskip("coeftable")
    import pandas as pd

    from increment.tables import trend_table

    frame = pd.DataFrame(
        {
            "metric": ["purchase_rate"] * 4,
            "grain": ["week"] * 4,
            "segment": ["US", "US", "DE", "DE"],
            "period": [
                date(2026, 1, 5),
                date(2026, 1, 12),
                date(2026, 1, 5),
                date(2026, 1, 12),
            ],
            "period_complete": [True, False, True, False],
            "n": [50, 45, 30, 28],
            "value": [0.15, 0.16, 0.25, 0.26],
            "ci_lb": [0.12, 0.13, 0.20, 0.21],
            "ci_ub": [0.18, 0.19, 0.30, 0.31],
        }
    )
    html = trend_table(frame).gt().as_raw_html()
    assert "US" in html
    assert "DE" in html


# Decision-stat polarity: dispatch keys on preferred_direction, not the
# test's tail. Constants named so expected values derive from the numbers.

# A decisive latency improvement: log lift -0.10 with posterior sd 0.02
# (z = -5), monitored under the DEFAULT two-sided test.
_LATENCY_LOG_LIFT = -0.10
_LATENCY_SE = 0.02


def _latency_estimate():
    """A decrease-preferred LiftEstimate under the default two-sided test,
    built via infer_lift so the posterior is recoverable. The SE is split
    evenly across arms (a zero-SE arm is refused by the degenerate-arm
    guard) so the combined log-scale SE is exactly _LATENCY_SE."""
    import math

    from increment.estimation.inference import infer_lift

    per_arm_se = _LATENCY_SE / math.sqrt(2.0)
    est = infer_lift(
        metric="latency_p50",
        group_id="T",
        method="unadjusted",
        method_role="decision",
        log_rr=_LATENCY_LOG_LIFT - 0.0,
        se_t=per_arm_se,
        se_c=per_arm_se,
    )
    return est.model_copy(update={"preferred_direction": "decrease"})


def test_estimates_to_readout_does_not_recommend_unavailable_posterior():
    (row,) = estimates_to_readout([_latency_estimate()])
    assert row["posterior_chance_to_beat"] is None
    assert row["posterior_risk_if_shipped"] is None


# Absolute-scale rows (value_scale="absolute": LATE, compliance) must not
# render as percent under "Lift %".


def _late_estimate(value=2.5, lb=1.2, ub=3.8):
    return LiftEstimate(
        metric="revenue",
        group_id="T",
        method="unadjusted",
        method_role="decision",
        estimand="late",
        value_scale="absolute",
        scale="linear",
        lift=Estimate(value=value, lb=lb, ub=ub, level=0.95),
    )


def test_liftestimate_to_row_carries_estimand_and_value_scale():
    row = estimates_to_readout([_late_estimate()])[0]
    assert row["estimand"] == "late"
    assert row["value_scale"] == "absolute"
    # the metric column stays the raw name - render-side labels only
    assert row["metric"] == "revenue"


def test_liftestimate_to_row_estimand_defaults_for_breakout_estimate():
    row = estimates_to_readout(
        [_breakout_estimate("revenue", "T", "unadjusted", "country", "US", 0.12)]
    )[0]
    assert row["estimand"] == "itt"
    assert row["value_scale"] == "relative"


def test_readout_table_absolute_row_not_rendered_as_percent():
    """A $2.50 LATE must render as an absolute quantity, never '+250.0%'."""
    pytest.importorskip("coeftable")
    rows = estimates_to_readout([_late_estimate()])
    html = readout_table(rows).gt().as_raw_html()
    assert "+250.0%" not in html
    assert "+120.0%" not in html
    assert "2.50" in html
    assert "1.20" in html
    assert "3.80" in html


def test_readout_table_mixed_scales_render_each_row_on_its_own_scale():
    """A relative itt row and an absolute late row in one table: the itt
    row renders as percent, the late row as an absolute quantity."""
    pytest.importorskip("coeftable")
    itt = _estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22)
    rows = estimates_to_readout([itt, _late_estimate()])
    html = readout_table(rows).gt().as_raw_html()
    assert "+12.0%" in html
    assert "+250.0%" not in html
    assert "2.50" in html


# Encouragement pipeline: run() -> estimates_to_readout -> readout_table
# must not collide itt/compliance/late rows.


def test_readout_table_encouragement_pipeline_renders_three_distinct_rows():
    """The documented pipeline on a real encouragement run(): itt,
    compliance, and late rows must land as three distinct table rows
    instead of crashing coeftable with a duplicate-row SpecError."""
    pytest.importorskip("coeftable")
    import pyarrow as pa

    from increment import readouts
    from increment.frame import from_unit_summary
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    units = [
        ("u01", "control", 10.0, 0),
        ("u02", "control", 12.0, 0),
        ("u03", "control", 9.0, 0),
        ("u04", "control", 20.0, 1),
        ("u05", "treatment", 25.0, 1),
        ("u06", "treatment", 30.0, 1),
        ("u07", "treatment", 28.0, 1),
        ("u08", "treatment", 15.0, 0),
    ]
    cols = list(zip(*units, strict=True))
    table = pa.table(dict(zip(["user_id", "variant", "revenue", "clicked"], cols, strict=True)))
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment only moves revenue via uptake"
        ),
        min_first_stage_z=0.5,
    )
    src = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        uptake="clicked",
        design=design,
    )
    results = readouts.run(src)
    assert {r.estimand for r in results} == {"itt", "compliance", "late"}
    rows = estimates_to_readout(results)
    html = readout_table(rows).gt().as_raw_html()
    assert "revenue (late)" in html
    assert "uptake (compliance)" in html
    assert "revenue" in html


def test_readout_table_disambiguates_same_estimand_on_both_scales():
    """An estimand emitted on BOTH scales for the same metric (late
    additive + late relative) must still land as two distinct rows."""
    pytest.importorskip("coeftable")
    late_abs = _late_estimate()
    late_rel = LiftEstimate(
        metric="revenue",
        group_id="T",
        method="unadjusted",
        method_role="decision",
        estimand="late",
        value_scale="relative",
        lift=Estimate(value=0.25, lb=0.12, ub=0.38, level=0.95),
    )
    rows = estimates_to_readout([late_abs, late_rel])
    html = readout_table(rows).gt().as_raw_html()
    assert "revenue (late)" in html
    assert "+25.0%" in html
    assert "2.50" in html


# Forest coloring follows the metric's declared preferred_direction.


def test_readout_table_direction_follows_preferred_direction():
    """A decrease-preferred metric resolves lower_is_better; an
    increase-preferred one keeps the default higher_is_better."""
    pytest.importorskip("coeftable")
    latency = _latency_estimate()
    revenue = _valid_estimate().model_copy(update={"preferred_direction": "increase"})
    table = readout_table(estimates_to_readout([latency, revenue]))
    assert table.direction_for("latency_p50") == "lower_is_better"
    assert table.direction_for("revenue") == "higher_is_better"


def test_readout_table_decrease_preferred_improvement_colored_favorable():
    """A significant latency drop must render in the favorable color, not
    the unfavorable one."""
    pytest.importorskip("coeftable")
    from coeftable.theme import DEFAULT

    html = readout_table(estimates_to_readout([_latency_estimate()])).gt().as_raw_html()
    assert DEFAULT.favorable in html
    assert DEFAULT.unfavorable not in html


def test_readout_table_without_preferred_direction_keeps_default_direction():
    """Rows lacking preferred_direction (None) leave coeftable's default
    higher_is_better untouched."""
    pytest.importorskip("coeftable")
    rows = estimates_to_readout([_estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22)])
    table = readout_table(rows)
    assert table.direction_for("revenue") == "higher_is_better"


# Rendered decision layer, interval levels, and row metadata.


def _inferred_estimate(
    metric,
    value,
    se,
    *,
    level=0.95,
    alternative="two-sided",
    null_lift=0.0,
    preferred_direction="increase",
    note=None,
    inference="fixed",
    ds=None,
):
    """A LiftEstimate whose interval IS the closed-form Normal quantile
    infer_lift would produce - so decision-stat methods work and rendered
    numbers can be checked against the methods' own output exactly."""
    import math

    from scipy.stats import norm

    log_mean = math.log1p(value)
    z = norm.ppf(0.5 + level / 2.0)
    return LiftEstimate(
        metric=metric,
        group_id="T",
        method="unadjusted",
        method_role="decision",
        lift=Estimate(
            value=value,
            lb=math.expm1(log_mean - z * se),
            ub=math.expm1(log_mean + z * se),
            level=level,
            log_mean=log_mean,
            log_se=se,
        ),
        alternative=alternative,
        null_lift=null_lift,
        preferred_direction=preferred_direction,
        note=note,
        inference=inference,
        ds=ds,
    )


class TestAdapterCarriesLevelNoteInference:
    def test_row_carries_interval_level(self):
        """One-sided/margin rows carry 90% intervals next to default 95%
        ones; dropping `level` made a mixed table render both identically."""
        row = estimates_to_readout([_inferred_estimate("m", 0.03, 0.02, level=0.90)])[0]
        assert row["level"] == pytest.approx(0.90)

    def test_point_only_row_has_none_level(self):
        row = estimates_to_readout([_estimate("m", "T", "unadjusted", 0.12)])[0]
        assert row["level"] is None

    def test_row_carries_note_and_inference(self):
        est = _inferred_estimate("m", 0.03, 0.02, note="guardrail caveat")
        row = estimates_to_readout([est])[0]
        assert row["note"] == "guardrail caveat"
        assert row["inference"] == "fixed"

    def test_ds_emitted_for_plain_liftestimate(self):
        """readouts.asof_lift returns plain LiftEstimates each stamped with
        ds; the adapter previously dropped it (the emission was nested in
        the dimension branch), collapsing an as-of series onto one key."""
        est = _inferred_estimate("m", 0.03, 0.02, ds=date(2025, 1, 3))
        rows = estimates_to_readout([est])
        assert rows[0]["ds"] == date(2025, 1, 3)

    def test_ds_absent_for_undated_liftestimate(self):
        rows = estimates_to_readout([_estimate("m", "T", "unadjusted", 0.12)])
        assert "ds" not in rows[0]


_SHIFTED_NULL_COLOR = "#E17C05"


def _forest(table):
    (forest,) = [column for column in table.columns if type(column).__name__ == "Forest"]
    return forest


def _orange_rule_count(table) -> int:
    return table.gt().as_raw_html().count(f'stroke="{_SHIFTED_NULL_COLOR}"')


class TestRenderedDecisionStats:
    def test_stored_posterior_statistics_are_opt_in_and_explicitly_qualified(self):
        pytest.importorskip("coeftable")
        from increment.estimation.inference import Normal

        est = _valid_estimate(
            prior=Normal(mu=0.0, sigma=0.1),
            preferred_direction="increase",
            null_lift=-0.01,
        )
        rows = estimates_to_readout([est])
        default = readout_table(rows).gt().as_raw_html()
        assert "Posterior chance to beat" not in default
        assert "Posterior risk if shipped" not in default
        assert "Posterior P(favorable)" not in default

        opted_in = readout_table(rows, advisory=True).gt().as_raw_html()
        assert "Posterior chance to beat" in opted_in
        assert "Posterior risk if shipped" in opted_in
        assert "Posterior P(favorable)" in opted_in
        assert f"{est.chance_to_beat() * 100:.1f}%" in opted_in
        assert f"{est.risk_if_shipped() * 100:.1f}%" in opted_in

    def test_no_per_metric_verdict_column(self):
        pytest.importorskip("coeftable")
        est = _inferred_estimate(
            "retention", 0.001, 0.005, level=0.90, alternative="greater", null_lift=-0.05
        )
        html = readout_table(estimates_to_readout([est])).gt().as_raw_html()
        assert "Decision" not in html
        assert "pass vs" not in html

    def test_margin_met_while_interval_sits_wholly_below_zero(self):
        """A guardrail down 2% has its whole interval below the gray zero
        reference yet above the amber-orange -5% tested-null boundary. It reads
        as a negative effect while still meeting its declared margin."""
        pytest.importorskip("coeftable")
        guardrail = _inferred_estimate(
            "retention", -0.02, 0.004, level=0.90, alternative="greater", null_lift=-0.05
        )
        assert (
            guardrail.require_lift().ub is not None and guardrail.require_lift().ub < 0.0
        )  # wholly below 0
        rows = estimates_to_readout(
            # Contrast the gray zero-tested row with the guardrail's amber boundary.
            [guardrail, _inferred_estimate("conversion", 0.03, 0.004)]
        )
        assert rows[0]["stat_sig"] is True  # margin still met
        table = readout_table(rows)
        (forest,) = [c for c in table.columns if type(c).__name__ == "Forest"]
        assert getattr(forest, "ref", None) == 0.0

    def test_unstored_probability_is_not_rendered(self):
        pytest.importorskip("coeftable")
        plain = _inferred_estimate("revenue", 0.03, 0.02)
        html = readout_table(estimates_to_readout([plain]), advisory=True).gt().as_raw_html()
        assert "Posterior P(favorable)" not in html

    def test_shifted_null_adds_default_orange_rule_and_keeps_zero_ref(self):
        ct = pytest.importorskip("coeftable")
        table = readout_table(
            estimates_to_readout(
                [
                    _inferred_estimate(
                        "retention",
                        -0.02,
                        0.004,
                        level=0.90,
                        alternative="greater",
                        null_lift=-0.05,
                    )
                ]
            )
        )
        forest = _forest(table)
        assert forest.ref == 0.0
        assert forest.annotations == (
            ct.Rule(
                at="__shifted_null_lift",
                axis="x",
                color=_SHIFTED_NULL_COLOR,
                width=2.0,
                dash="dashed",
            ),
        )
        assert _orange_rule_count(table) == 1

    def test_out_of_range_shifted_null_expands_domain_and_renders_rule_in_viewport(self):
        pytest.importorskip("coeftable")
        table = readout_table(
            estimates_to_readout(
                [
                    _inferred_estimate(
                        "retention",
                        0.02,
                        0.004,
                        level=0.90,
                        alternative="greater",
                        null_lift=-0.5,
                    )
                ]
            )
        )
        html = table.gt().as_raw_html()
        assert _orange_rule_count(table) == 1
        (orange_svg,) = [
            svg
            for svg in re.findall(r"<svg\b.*?</svg>", html, flags=re.DOTALL)
            if f'stroke="{_SHIFTED_NULL_COLOR}"' in svg
        ]
        view_box = re.search(r'viewBox="0 0 ([0-9.]+) [0-9.]+"', orange_svg)
        rule = re.search(
            rf'<line x1="([0-9.]+)"[^>]+stroke="{_SHIFTED_NULL_COLOR}"',
            orange_svg,
        )
        assert view_box is not None and rule is not None
        assert 0.0 < float(rule.group(1)) < float(view_box.group(1))

    def test_mixed_nulls_annotate_only_shifted_row(self):
        pytest.importorskip("coeftable")
        rows = estimates_to_readout(
            [
                _inferred_estimate(
                    "retention",
                    -0.02,
                    0.004,
                    level=0.90,
                    alternative="greater",
                    null_lift=-0.05,
                ),
                _inferred_estimate("conversion", 0.03, 0.004),
            ]
        )
        table = readout_table(rows)
        assert _forest(table).ref == 0.0
        assert _orange_rule_count(table) == 1

    def test_distinct_shifted_nulls_annotate_each_row(self):
        pytest.importorskip("coeftable")
        rows = estimates_to_readout(
            [
                _inferred_estimate(
                    "retention",
                    -0.02,
                    0.004,
                    level=0.90,
                    alternative="greater",
                    null_lift=-0.05,
                ),
                _inferred_estimate(
                    "latency",
                    -0.01,
                    0.004,
                    level=0.90,
                    alternative="greater",
                    null_lift=-0.03,
                ),
            ]
        )
        assert _orange_rule_count(readout_table(rows)) == 2

    def test_zero_and_nonfinite_nulls_add_no_orange_rule(self):
        pytest.importorskip("coeftable")
        rows = estimates_to_readout(
            [
                _inferred_estimate("revenue", 0.03, 0.02),
                _inferred_estimate("signups", 0.02, 0.02),
            ]
        )
        rows[1]["null_lift"] = float("inf")
        table = readout_table(rows)
        assert len(_forest(table).annotations) == 1
        assert _orange_rule_count(table) == 0


class TestAsOfTrendPath:
    """readouts.asof_lift returns plain LiftEstimates with ds - previously
    unrenderable both ways (data= collapsed dates onto one key; trend=
    refused the model type)."""

    def test_whole_window_list_with_null_ds_refused_as_trend(self):
        pytest.importorskip("coeftable")
        headline = estimates_to_readout([_inferred_estimate("revenue", 0.05, 0.02)])
        from increment.errors import InvalidRequestError

        with pytest.raises(InvalidRequestError) as exc_info:
            readout_table(headline, trend=[_inferred_estimate("revenue", 0.05, 0.02)])
        assert exc_info.value.code == "tables.readout_table_trend_ds_all_null"


# Role-based grouping: `metric_group` is renamed `method`; `role`/`discovery`
# are adapter columns; `readout_table` groups rows by concise role labels via
# coeftable's `groups=`/`collapsible_groups=`.


def test_liftestimate_to_row_renames_metric_group_to_method():
    """`metric_group` is gone; `method` carries the same value. `role`/
    `discovery` default to None for a plain LiftEstimate."""
    est = _estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22)
    row = estimates_to_readout([est])[0]
    assert row["method"] == "unadjusted"
    assert "metric_group" not in row
    assert row["role"] is None
    assert row["discovery"] is None


def test_liftestimate_to_row_carries_role_and_discovery():
    """A LiftEstimate stamped with a declared-plan role/discovery verdict
    carries both straight through to the row."""
    est = _estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22).model_copy(
        update={"role": "secondary", "discovery": True}
    )
    row = estimates_to_readout([est])[0]
    assert row["role"] == "secondary"
    assert row["discovery"] is True


def test_liftestimate_to_row_breakout_estimate_role_and_discovery():
    est = _breakout_estimate(
        "purchase_rate", "T", "unadjusted", "country", "US", 0.12, lb=0.02, ub=0.22
    ).model_copy(update={"role": "exploratory", "discovery": False})
    row = estimates_to_readout([est])[0]
    assert row["role"] == "exploratory"
    assert row["discovery"] is False


def test_estimates_to_readout_column_set_has_method_not_metric_group():
    ests = [_estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22)]
    (row,) = estimates_to_readout(ests)
    assert "method" in row
    assert "metric_group" not in row
    assert "role" in row
    assert "discovery" in row


def test_readout_table_nest_by_method_still_works_after_rename():
    """nest_by='method' dispatches on the renamed `method` column, not the
    old `metric_group` name."""
    pytest.importorskip("coeftable")
    ests = [
        _estimate("revenue", "T", "unadjusted", 0.12, lb=0.02, ub=0.22),
        _estimate("revenue", "T", "adjusted", 0.10, lb=0.03, ub=0.17),
    ]
    rows = estimates_to_readout(ests)
    table = readout_table(rows, nest_by="method")
    html = table.gt().as_raw_html()
    assert 'gt_row gt_center">unadjusted' in html
    assert 'gt_row gt_center">adjusted' in html


def test_readout_table_nest_by_segment_raises_names_method_not_metric_group():
    """The nest_by='segment' single-method guard now names `method`."""
    pytest.importorskip("coeftable")
    ests = [
        _breakout_estimate(
            "purchase_rate", "T", "unadjusted", "country", "US", 0.12, lb=0.02, ub=0.22
        ),
        _breakout_estimate(
            "purchase_rate", "T", "adjusted", "country", "US", 0.11, lb=0.01, ub=0.21
        ),
    ]
    rows = estimates_to_readout(ests)
    from increment.errors import InvalidRequestError

    with pytest.raises(InvalidRequestError) as exc_info:
        readout_table(rows, nest_by="segment")
    assert exc_info.value.code == "tables.readout_table_nest"


def _role_row(
    metric,
    role,
    *,
    discovery=None,
    level=0.95,
    inference="fixed",
    value=0.1,
    family_threshold=None,
):
    return {
        "metric": metric,
        "group_id": "T",
        "method": "unadjusted",
        "lift": value,
        "lower": value - 0.05,
        "higher": value + 0.05,
        "level": level,
        "stat_sig": True,
        "role": role,
        "discovery": discovery,
        "inference": inference,
        "family_threshold": family_threshold,
    }


class TestRoleGroupedDisclosure:
    """Role labels render as concise section headings."""

    def test_all_none_role_renders_ungrouped(self):
        pytest.importorskip("coeftable")
        rows = [_role_row("revenue", None), _role_row("signups", None)]
        headers = _group_headers(readout_table(rows).gt().as_raw_html())
        assert headers == []

    @pytest.mark.parametrize(
        ("role", "expected"),
        [
            ("primary", "Primary"),
            ("secondary", "Secondaries"),
            ("guardrail", "Guardrails"),
            ("unassigned", "Unassigned"),
            ("exploratory", "Exploratory"),
            (None, "Design-level"),
        ],
    )
    def test_group_heading_is_exact_concise_role_label(self, role, expected):
        pytest.importorskip("coeftable")
        rows = [_role_row("metric", role)]
        if role is None:
            rows.append(_role_row("planned", "primary"))
        headers = _group_headers(readout_table(rows).gt().as_raw_html())
        assert expected in headers
        assert all(" — " not in header for header in headers)
        assert all(
            not any(prose in header for prose in ("q=", "FCR", "BH", "discovery correction"))
            for header in headers
        )

    def test_mixed_table_no_declared_plan_row_gets_design_level_heading(self):
        pytest.importorskip("coeftable")
        rows = [_role_row("revenue", "primary"), _role_row("uptake", None)]
        headers = _group_headers(readout_table(rows).gt().as_raw_html())
        assert headers == ["Primary", "Design-level"]


def test_readout_table_discovery_renders_as_its_own_column_not_stat_sig():
    """`discovery` is a distinct family-verdict column, never conflated
    with `stat_sig` - both True/False values must render distinctly."""
    pytest.importorskip("coeftable")
    rows = [
        _role_row("m1", "secondary", discovery=True),
        _role_row("m2", "secondary", discovery=False),
    ]
    html = readout_table(rows).gt().as_raw_html()
    assert "Discovery" in html


@pytest.mark.parametrize("adapter", [contrast_results_to_readout, estimates_to_readout])
@pytest.mark.parametrize(
    (
        "randomization_law",
        "independence_grain",
        "n_blocks",
        "method",
        "reference",
        "carryover_order",
    ),
    [
        (
            "independent_bernoulli_order",
            "unit_cycle",
            None,
            "switchback_unit_t_approximation",
            "unit_t_approximation",
            0,
        ),
        ("shared_schedule", "shared_block", 3, "switchback_block_t", "block_t", 2),
    ],
)
def test_estimates_to_readout_adapts_switchback_contrast_results(
    adapter, carryover_order, randomization_law, independence_grain, n_blocks, method, reference
):
    """Contrast evidence uses the same public table-row adapter as arm evidence."""
    from increment.estimation.contrast_results import ContrastResult, ContrastResults

    result = ContrastResult(
        metric="revenue",
        control_group="control",
        treatment_group="treatment",
        estimand="retained_window_total_difference",
        aggregation="sum",
        probability_ct=0.5,
        randomization_law=randomization_law,
        independence_grain=independence_grain,
        carryover_order=carryover_order,
        observation_steps=3,
        retained_steps=3 - carryover_order,
        method=method,
        reference=reference,
        reference_spec=UnitCycleTApproximation() if n_blocks is None else None,
        estimate=Estimate(value=1.0, lb=0.5, ub=1.5, level=0.95),
        standard_error=0.25,
        alternative="two-sided",
        null_abs=0.0,
        alpha=0.05,
        n_units=2,
        n_cycles=2 if n_blocks is None else 2 * n_blocks,
        n_blocks=n_blocks,
        ct_cycles=1,
        tc_cycles=1 if n_blocks is None else 2,
        dof=1.0 if n_blocks is None else n_blocks - 1.0,
    )
    rows = adapter(ContrastResults([result]))
    assert rows[0]["metric"] == "revenue"
    assert rows[0]["lift"] == 1.0
    assert rows[0]["lower"] == 0.5
    assert rows[0]["higher"] == 1.5
    assert rows[0]["randomization_law"] == randomization_law
    assert rows[0]["independence_grain"] == independence_grain
    assert rows[0]["carryover_order"] == carryover_order
    assert type(rows[0]["carryover_order"]) is int
    assert rows[0]["observation_steps"] == 3
    assert rows[0]["retained_steps"] == 3 - carryover_order
    assert rows[0]["ct_cycles"] == result.ct_cycles == 1
    assert rows[0]["tc_cycles"] == result.tc_cycles
    assert rows[0]["n_blocks"] == n_blocks
    assert type(rows[0]["n_blocks"]) is type(n_blocks)


@pytest.mark.parametrize("model", [BreakoutEstimate, DailyLiftEstimate])
@pytest.mark.parametrize("control_mean", [0.0, 1.0])
def test_flat_joint_rows_preserve_disconnected_set_significance(model, control_mean):
    from datetime import date

    from increment.estimation.inference import _joint_additive_bounds
    from increment.estimation.results import JointContrastReference, relative_confidence_set

    region = relative_confidence_set(
        JointContrastReference(a=10, c=control_mean, var_a=1, var_c=1, cov_ac=0)
    )
    # Built by the projection the row validator checks against, so only the
    # significance round trip is under test.
    bounds = _joint_additive_bounds(10.0, 1.0, 0.05, "two-sided", None)
    extra = (
        {"dimension": "country", "dimension_value": "US"}
        if model is BreakoutEstimate
        else {"ds": date(2026, 1, 1)}
    )
    estimate = model(
        metric="m",
        group_id="T",
        method="unadjusted",
        method_role="decision",
        lift=region.estimate(),
        relative_confidence_set=region,
        abs_diff=10.0,
        abs_se=1.0,
        abs_lb=bounds[0],
        abs_ub=bounds[1],
        abs_reference_kind="normal",
        sampling_available=True,
        **extra,
    )
    restored = model.model_validate_json(estimate.model_dump_json())
    assert region.geometry == "disconnected"
    assert estimates_to_readout([restored])[0]["stat_sig"] is True
    assert estimates_to_readout([restored])[0]["relative_confidence_set"].contains(0) is False


@pytest.mark.parametrize(
    ("control_mean", "expected"),
    [
        (3.0, (2 / 3, 0.5377650827935979, 0.8264657908535663, 0.9)),
        (0.0, (None, None, None, None)),
    ],
)
def test_directional_fieller_readout_retains_central_display_interval(control_mean, expected):
    from increment.estimation.inference import _joint_additive_bounds
    from increment.estimation.results import JointContrastReference, relative_confidence_set

    region = relative_confidence_set(
        JointContrastReference(a=2, c=control_mean, var_a=0.04, var_c=0.09, cov_ac=0.01),
        alternative="greater",
    )
    bounds = _joint_additive_bounds(2.0, 0.2, 0.05, "greater", None)
    estimate = LiftEstimate(
        metric="ratio",
        group_id="T",
        method="joint",
        method_role="decision",
        scale="linear",
        alternative="greater",
        lift=region.estimate(),
        relative_confidence_set=region,
        abs_diff=2.0,
        abs_se=0.2,
        abs_lb=bounds[0],
        abs_ub=bounds[1],
        abs_reference_kind="normal",
    )
    restored = LiftEstimate.model_validate_json(estimate.model_dump_json())
    row = estimates_to_readout([restored])[0]
    assert (row["lift"], row["lower"], row["higher"], row["level"]) == pytest.approx(
        expected, abs=1e-12
    )
    assert row["open_side"] is None
    assert row["relative_confidence_set"].contains(1.0)
    assert row["relative_confidence_set"].contains(0.0) is (control_mean == 0.0)
    assert row["stat_sig"] is (control_mean != 0.0)


@pytest.mark.parametrize("model", [BreakoutEstimate, DailyLiftEstimate])
def test_flat_partial_joint_rows_preserve_additive_margin_decision(model):
    from datetime import date

    extra = (
        {"dimension": "country", "dimension_value": "US"}
        if model is BreakoutEstimate
        else {"ds": date(2026, 1, 1)}
    )
    estimate = model(
        metric="m",
        group_id="T",
        method="unadjusted",
        method_role="decision",
        lift=None,
        relative_unavailable_reason="joint_covariance_indefinite",
        abs_diff=10,
        abs_se=1,
        abs_lb=8,
        abs_ub=12,
        null_abs=0,
        sampling_available=True,
        **extra,
    )
    restored = model.model_validate_json(estimate.model_dump_json())
    assert estimates_to_readout([restored])[0]["stat_sig"] is True
    assert estimates_to_readout([restored])[0]["lift"] is None
    assert (
        estimates_to_readout([restored])[0]["relative_unavailable_reason"]
        == "joint_covariance_indefinite"
    )


_HOSTILE = "<img src=x onerror=alert(1)>"


# Parse browser-visible label output rather than pinning HTML source escapes.
# Label axes include definition/warehouse values and hand-built role strings.

_EXECUTABLE_SCHEMES = frozenset({"javascript", "data", "vbscript"})
_URI_ATTRIBUTES = frozenset(
    {"href", "src", "xlink:href", "action", "formaction", "poster", "data", "background"}
)


class _DomText(TypedDict):
    href: str | None
    text: list[str]


class _Dom:
    """Structure of a rendered table: destinations, anchors, tags and text."""

    def __init__(self, html):
        from collections import Counter
        from html.parser import HTMLParser

        self.tags: Counter[str] = Counter()
        self.destinations: list[tuple[str, str, str]] = []
        self.handlers: list[tuple[str, str]] = []
        self.anchors: list[_DomText] = []
        self.sups: list[_DomText] = []
        self.alts: list[str] = []
        self._text: list[str] = []
        dom = self

        class Parser(HTMLParser):
            def __init__(self):
                super().__init__(convert_charrefs=True)
                self.skip = 0
                self.open: list[tuple[str, _DomText]] = []

            def handle_starttag(self, tag, attrs):
                dom.tags[tag] += 1
                if tag in ("style", "script"):
                    self.skip += 1
                values: dict[str, str] = {}
                for name, value in attrs:
                    name = name.lower()
                    values[name] = value or ""
                    if name in _URI_ATTRIBUTES:
                        dom.destinations.append((tag, name, value or ""))
                    if name.startswith("on"):
                        dom.handlers.append((tag, name))
                    if name == "alt":
                        dom.alts.append(value or "")
                if tag in ("a", "sup"):
                    bucket: _DomText = {"href": values.get("href"), "text": []}
                    (dom.anchors if tag == "a" else dom.sups).append(bucket)
                    self.open.append((tag, bucket))

            def handle_endtag(self, tag):
                if tag in ("style", "script") and self.skip:
                    self.skip -= 1
                for position in range(len(self.open) - 1, -1, -1):
                    if self.open[position][0] == tag:
                        del self.open[position]
                        break

            def handle_data(self, data):
                if self.skip:
                    return
                dom._text.append(data)
                for _tag, bucket in self.open:
                    bucket["text"].append(data)

        parser = Parser()
        parser.feed(html)
        parser.close()

    @property
    def text(self):
        return "".join(self._text)

    @property
    def readable(self):
        return self.text + " " + " ".join(self.alts)

    def executable_destinations(self):
        """Destinations a browser would treat as script/data after stripping
        the control characters and whitespace URL parsing ignores."""
        found = []
        for tag, name, value in self.destinations:
            compact = re.sub(r"[\x00-\x20\x7f-\x9f]+", "", value).lower()
            match = re.match(r"([a-z][a-z0-9+.\-]*):", compact)
            if match and match.group(1) in _EXECUTABLE_SCHEMES:
                found.append((tag, name, value))
        return found

    def percent_tokens(self):
        from collections import Counter

        return Counter(re.findall(r"[+-]?\d+\.\d%", self.text))


_LINK_TEXT = "marker"
_DANGEROUS_LABELS = [
    pytest.param("[marker](javascript:void%280%29)", id="javascript-percent"),
    pytest.param("[marker](javascript:alert(1))", id="javascript"),
    pytest.param("[marker](JaVaScRiPt:alert(1))", id="javascript-mixed-case"),
    pytest.param("[marker](&#106;avascript:alert(1))", id="javascript-decimal-entity"),
    pytest.param("[marker](jav&#x09;ascript:alert(1))", id="javascript-tab-entity"),
    pytest.param("[marker](jav&#x0A;ascript:alert(1))", id="javascript-newline-entity"),
    pytest.param("[marker](&#x6A;&#x61;vascript:alert(1))", id="javascript-hex-entity"),
    pytest.param("[marker](data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==)", id="data"),
    pytest.param("[marker](DaTa:text/html,<b>x</b>)", id="data-mixed-case"),
    pytest.param("[marker](vbscript:msgbox(1))", id="vbscript"),
    pytest.param("[marker](VbScript:msgbox(1))", id="vbscript-mixed-case"),
    pytest.param("![marker](javascript:alert(1))", id="javascript-image"),
]
_SAFE_LINKS = [
    pytest.param("https://example.com/docs?a=1", id="https"),
    pytest.param("http://example.com/docs", id="http"),
    pytest.param("/docs/readout", id="absolute-path"),
    pytest.param("../docs/readout.html", id="relative-path"),
    pytest.param("mailto:team@example.com", id="mailto"),
]
_RAW_MARKUP = [
    pytest.param(_HOSTILE, id="img-onerror"),
    pytest.param("<script>alert(1)</script>", id="script"),
    pytest.param('<a href="javascript:alert(1)">marker</a>', id="raw-anchor"),
    pytest.param("<svg onload=alert(1)>", id="svg-onload"),
]

# Surfaces that render the label as Markdown (row / nest cells) vs. as a
# column-spanner or group heading.
_MARKDOWN_SURFACES = ["metric", "arm", "segment", "method_nest"]
_SPLIT_SURFACES = ["split_method", "split_arm"]
_READOUT_SURFACES = [*_MARKDOWN_SURFACES, *_SPLIT_SURFACES, "group_heading"]
_ESTIMATE_A = (0.12, 0.02, 0.22)
_ESTIMATE_B = (0.05, -0.02, 0.12)


def _trend_for(keys):
    """Two-day DailyLiftEstimate series for each (metric, arm, method, dim, value)."""
    return [
        _daily_estimate(
            metric,
            arm,
            method,
            date(2024, 1, day),
            value,
            lb=lb,
            ub=ub,
            dimension=dimension,
            dimension_value=dimension_value,
        )
        for metric, arm, method, dimension, dimension_value in keys
        for day, (value, lb, ub) in zip((1, 2), (_ESTIMATE_B, _ESTIMATE_A), strict=True)
    ]


def _readout_case(surface, label):
    """``(rows, readout_table kwargs)`` placing *label* on one axis; the other
    row carries a fixed benign label and a different estimate."""
    a, b = _ESTIMATE_A, _ESTIMATE_B
    if surface == "metric":
        ests = [
            _estimate(label, "T", "unadjusted", *a),
            _estimate("other", "T", "unadjusted", *b),
        ]
        trend = _trend_for(
            [(label, "T", "unadjusted", None, None), ("other", "T", "unadjusted", None, None)]
        )
        return estimates_to_readout(ests), {"trend": trend}
    if surface == "arm":
        ests = [_estimate("m", label, "unadjusted", *a), _estimate("m", "C", "unadjusted", *b)]
        trend = _trend_for(
            [("m", label, "unadjusted", None, None), ("m", "C", "unadjusted", None, None)]
        )
        return estimates_to_readout(ests), {"trend": trend}
    if surface == "segment":
        ests = [
            _breakout_estimate("m", "T", "unadjusted", "country", label, *a),
            _breakout_estimate("m", "T", "unadjusted", "country", "GB", *b),
        ]
        trend = _trend_for(
            [("m", "T", "unadjusted", "country", label), ("m", "T", "unadjusted", "country", "GB")]
        )
        return estimates_to_readout(ests), {"nest_by": "segment", "trend": trend}
    if surface == "method_nest":
        ests = [_estimate("m", "T", label, *a), _estimate("m", "T", "other", *b)]
        return estimates_to_readout(ests), {"nest_by": "method"}
    if surface == "split_method":
        ests = [_estimate("m", "T", label, *a), _estimate("m", "T", "other", *b)]
        return estimates_to_readout(ests), {}
    if surface == "split_arm":
        ests = [_estimate("m", label, "u", *a), _estimate("m", "C", "u", *b)]
        return estimates_to_readout(ests), {"nest_by": "method"}
    if surface == "group_heading":
        return [
            _role_row("m1", label, value=a[0]),
            _role_row("m2", "primary", value=b[0]),
        ], {}
    raise AssertionError(surface)


def _render_readout(surface, label):
    rows, kwargs = _readout_case(surface, label)
    return rows, _Dom(readout_table(rows, **kwargs).gt().as_raw_html())


def _assert_label_inert(dom, control):
    """No executable destination, script, or handler beyond the benign render."""
    assert dom.executable_destinations() == []
    assert dom.tags["script"] == control.tags["script"]
    assert dom.tags["iframe"] == control.tags["iframe"]
    assert len(dom.handlers) == len(control.handlers)


@pytest.mark.parametrize("surface", _READOUT_SURFACES)
@pytest.mark.parametrize("label", _DANGEROUS_LABELS)
def test_readout_table_dangerous_markdown_label_is_inert(surface, label):
    """Dangerous link destinations in any label axis never reach a browser
    as a live destination, while the source identity, the readable label and
    the rendered estimates/intervals and sparkline graphics are untouched."""
    pytest.importorskip("coeftable")
    import copy

    rows, kwargs = _readout_case(surface, label)
    source = copy.deepcopy(rows)
    dom = _Dom(readout_table(rows, **kwargs).gt().as_raw_html())
    control_rows, control_kwargs = _readout_case(surface, _LINK_TEXT)
    control = _Dom(readout_table(control_rows, **control_kwargs).gt().as_raw_html())

    _assert_label_inert(dom, control)
    assert _LINK_TEXT in dom.readable
    assert rows == source
    assert label in {value for row in rows for value in row.values()}
    assert dom.percent_tokens() == control.percent_tokens()
    assert dom.tags["svg"] == control.tags["svg"]


@pytest.mark.parametrize("surface", _SPLIT_SURFACES)
@pytest.mark.parametrize("label", _DANGEROUS_LABELS)
def test_readout_table_units_wrapped_split_label_is_inert(surface, label):
    """A split value is rendered through great_tables' units notation, which
    reparses ``{{...}}`` content as Markdown; wrapping a payload in it must
    not reintroduce a live destination."""
    pytest.importorskip("coeftable")

    wrapped = "{{" + label + "}}"
    rows, dom = _render_readout(surface, wrapped)
    _, control = _render_readout(surface, _LINK_TEXT)

    _assert_label_inert(dom, control)
    assert _LINK_TEXT in dom.readable
    assert wrapped in {value for row in rows for value in row.values()}
    assert dom.percent_tokens() == control.percent_tokens()


@pytest.mark.parametrize("surface", _MARKDOWN_SURFACES)
@pytest.mark.parametrize("href", _SAFE_LINKS)
def test_readout_table_keeps_safe_markdown_links_and_emphasis(surface, href):
    """Neutralizing dangerous destinations must not disable Markdown: safe
    links keep their exact destination and text, emphasis still renders."""
    pytest.importorskip("coeftable")

    _, dom = _render_readout(surface, f"**bold** _soft_ [docs]({href})")

    assert [a["href"] for a in dom.anchors if "".join(a["text"]) == "docs"] == [href]
    assert dom.tags["strong"] >= 1
    assert dom.tags["em"] >= 1
    assert "bold" in dom.text
    assert dom.executable_destinations() == []


@pytest.mark.parametrize("surface", _READOUT_SURFACES)
@pytest.mark.parametrize("raw", _RAW_MARKUP)
def test_readout_table_raw_markup_in_label_renders_as_literal_text(surface, raw):
    pytest.importorskip("coeftable")

    label = "lbl" + raw
    rows, dom = _render_readout(surface, label)
    _, control = _render_readout(surface, "lbl")

    assert label in dom.text
    assert dom.tags["img"] == control.tags["img"]
    assert dom.tags["script"] == control.tags["script"]
    assert len(dom.handlers) == len(control.handlers)
    assert len(dom.anchors) == len(control.anchors)
    assert dom.executable_destinations() == []
    assert label in {value for row in rows for value in row.values()}


@pytest.mark.parametrize("surface", _READOUT_SURFACES)
def test_readout_table_benign_units_label_stays_readable_and_graphics_untouched(surface):
    pytest.importorskip("coeftable")

    _, dom = _render_readout(surface, "{{m/s^2}}")
    _, control = _render_readout(surface, _LINK_TEXT)

    assert "m/s" in dom.text
    assert dom.executable_destinations() == []
    assert dom.tags["svg"] == control.tags["svg"]
    assert dom.percent_tokens() == control.percent_tokens()


def test_readout_table_leaves_pandas_source_frame_unchanged():
    """Rendering hostile identities never rewrites the caller's frame."""
    pytest.importorskip("coeftable")
    import pandas as pd

    hostile = "[marker](javascript:alert(1))"
    rows = estimates_to_readout(
        [
            _breakout_estimate(hostile, hostile, hostile, "country", hostile, *_ESTIMATE_A),
            _breakout_estimate("m", "C", hostile, "country", "GB", *_ESTIMATE_B),
        ]
    )
    frame = pd.DataFrame(rows)
    before = frame.copy(deep=True)
    for nest_by in ("arm", "method"):
        readout_table(frame, nest_by=nest_by).gt().as_raw_html()
    pd.testing.assert_frame_equal(frame, before)
    assert (
        estimates_to_readout(
            [_breakout_estimate(hostile, hostile, hostile, "country", hostile, *_ESTIMATE_A)]
        )[0]["metric"]
        == hostile
    )


def _trend_frame(metric, segment=None):
    import pandas as pd

    data = {
        "metric": [metric] * 2,
        "grain": ["week"] * 2,
        "period": [date(2026, 1, 5), date(2026, 1, 12)],
        "period_complete": [True, True],
        "n": [100, 110],
        "value": [0.10, 0.20],
        "ci_lb": [0.05, 0.10],
        "ci_ub": [0.15, 0.30],
    }
    if segment is not None:
        data["segment"] = [segment] * 2
    return pd.DataFrame(data)


def _trend_value_tokens(dom):
    from collections import Counter

    return Counter(re.findall(r"\d+\.\d\d", dom.text))


@pytest.mark.parametrize("surface", ["metric", "segment"])
@pytest.mark.parametrize("label", _DANGEROUS_LABELS)
def test_trend_table_dangerous_markdown_label_is_inert(surface, label):
    pytest.importorskip("coeftable")
    import pandas as pd

    from increment.tables import trend_table

    def frame(value):
        return _trend_frame(value) if surface == "metric" else _trend_frame("m", value)

    source = frame(label)
    before = source.copy(deep=True)
    dom = _Dom(trend_table(source).gt().as_raw_html())
    control = _Dom(trend_table(frame(_LINK_TEXT)).gt().as_raw_html())

    _assert_label_inert(dom, control)
    assert _LINK_TEXT in dom.readable
    pd.testing.assert_frame_equal(source, before)
    assert _trend_value_tokens(dom) == _trend_value_tokens(control)
    assert dom.tags["svg"] == control.tags["svg"]


@pytest.mark.parametrize("surface", ["metric", "segment"])
@pytest.mark.parametrize("href", _SAFE_LINKS)
def test_trend_table_keeps_safe_markdown_links_and_emphasis(surface, href):
    pytest.importorskip("coeftable")

    from increment.tables import trend_table

    label = f"**bold** _soft_ [docs]({href})"
    frame = _trend_frame(label) if surface == "metric" else _trend_frame("m", label)
    dom = _Dom(trend_table(frame).gt().as_raw_html())

    assert [a["href"] for a in dom.anchors if "".join(a["text"]) == "docs"] == [href]
    assert dom.tags["strong"] >= 1
    assert dom.tags["em"] >= 1
    assert dom.executable_destinations() == []


@pytest.mark.parametrize("surface", ["metric", "segment"])
@pytest.mark.parametrize("raw", _RAW_MARKUP)
def test_trend_table_raw_markup_in_label_renders_as_literal_text(surface, raw):
    pytest.importorskip("coeftable")

    from increment.tables import trend_table

    def render(value):
        frame = _trend_frame(value) if surface == "metric" else _trend_frame("m", value)
        return _Dom(trend_table(frame).gt().as_raw_html())

    dom, control = render("lbl" + raw), render("lbl")

    assert "lbl" + raw in dom.text
    assert dom.tags["img"] == control.tags["img"]
    assert dom.tags["script"] == control.tags["script"]
    assert len(dom.handlers) == len(control.handlers)
    assert len(dom.anchors) == len(control.anchors)
    assert dom.executable_destinations() == []


# Non-HTML renderers: the same label axes must stay readable. LaTeX escaping
# differs across great_tables releases, so text is compared after undoing it.

_SPECIAL_LABEL = "a&b_c #d 50%"


def _latex_text(latex):
    text = latex.replace("\\textless{}", "<").replace("\\textgreater{}", ">")
    text = text.replace("{[}", "[").replace("{]}", "]")
    return re.sub(r"\\([&%$#_{}])", r"\1", text)


def _as_latex(table):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Styles are not yet supported in LaTeX output")
        return table.gt().as_latex()


@pytest.mark.parametrize("surface", _READOUT_SURFACES)
def test_readout_table_special_character_labels_are_readable_in_html_and_latex(surface):
    pytest.importorskip("coeftable")

    rows, kwargs = _readout_case(surface, _SPECIAL_LABEL)
    table = readout_table(rows, **kwargs)

    assert _SPECIAL_LABEL in _Dom(table.gt().as_raw_html()).text
    assert _SPECIAL_LABEL in _latex_text(_as_latex(table))


@pytest.mark.parametrize("surface", ["metric", "segment"])
def test_trend_table_special_character_labels_are_readable_in_html_and_latex(surface):
    pytest.importorskip("coeftable")

    from increment.tables import trend_table

    frame = (
        _trend_frame(_SPECIAL_LABEL) if surface == "metric" else _trend_frame("m", _SPECIAL_LABEL)
    )
    table = trend_table(frame)

    assert _SPECIAL_LABEL in _Dom(table.gt().as_raw_html()).text
    assert _SPECIAL_LABEL in _latex_text(_as_latex(table))


@pytest.mark.parametrize("surface", _MARKDOWN_SURFACES)
def test_readout_table_markdown_label_text_is_readable_in_latex(surface):
    pytest.importorskip("coeftable")

    rows, kwargs = _readout_case(surface, "[marker](https://example.com/docs)")

    assert _LINK_TEXT in _latex_text(_as_latex(readout_table(rows, **kwargs)))
