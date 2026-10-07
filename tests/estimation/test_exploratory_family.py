"""One exploratory Benjamini-Hochberg family over whole-window and segment rows."""

from __future__ import annotations

import copy
import json
import math
import pickle
import warnings
from datetime import date
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import pytest

from increment.breakout.estimates import BreakoutEstimate, run_breakout
from increment.decision import ArmHypothesisKey, PValueEvidence
from increment.errors import CapabilityError, CodedError, IncrementWarning
from increment.estimation.armstats import ScoreStats, centered_row_from_raw_sums
from increment.estimation.engine import Method, estimate_lift
from increment.estimation.family import (
    bh_select,
    exploratory_family_exclusion,
    select_exploratory_family,
)
from increment.estimation.inference import Normal, infer_ate, infer_lift
from increment.estimation.results import (
    LiftEstimate,
    _fcr_alpha_for,
    open_bound_from_two_sided_at_target,
)
from increment.semantics.models import ConversionMetric, MeanMetric

NOMINAL_ALPHA = 0.05


def _wald_row(
    metric: str,
    z: float,
    *,
    se: float = 0.05,
    alpha: float = NOMINAL_ALPHA,
    alternative: str = "two-sided",
    group_id: str = "treatment",
    method_role: Literal["decision", "sensitivity"] = "decision",
    **extra,
) -> LiftEstimate:
    """A whole-window row whose log-ratio sits ``z`` standard errors from zero."""
    half = se / math.sqrt(2.0)
    return infer_lift(
        metric,
        group_id,
        "unadjusted",
        z * se,
        half,
        half,
        alpha=alpha,
        alternative=alternative,
        method_role=method_role,
        **extra,
    )


def _mean_arm(n, mean, var, *, metric, group_id, country=None, x_mean=None, xy_cov=0.0):
    sum_y = float(n) * mean
    sum_x = None if x_mean is None else float(n) * x_mean
    raw = {
        "experiment_id": "exp1",
        "metric": metric,
        "group_id": group_id,
        "n": float(n),
        "sum_y": sum_y,
        "sum_y2": var * (n - 1) + sum_y**2 / float(n),
        "sum_x": sum_x,
        "sum_x2": None if sum_x is None else (n - 1) + sum_x**2 / float(n),
        "sum_xy": None if sum_x is None else xy_cov * (n - 1) + sum_x * sum_y / float(n),
        "sum_den": None,
        "sum_den2": None,
        "sum_yden": None,
    }
    if country is not None:
        raw["country"] = country
    return centered_row_from_raw_sums(raw)


def _binary_arm(n, successes, *, metric, group_id, country=None):
    raw = {
        "experiment_id": "exp1",
        "metric": metric,
        "group_id": group_id,
        "n": float(n),
        "successes": successes,
        "sum_y": float(successes),
        "sum_y2": float(successes),
        "sum_x": None,
        "sum_x2": None,
        "sum_xy": None,
        "sum_den": None,
        "sum_den2": None,
        "sum_yden": None,
    }
    if country is not None:
        raw["country"] = country
    return centered_row_from_raw_sums(raw)


def _mean_metric(name):
    return MeanMetric(name=name, entity="user", fact=name)


#: (metric, country) -> (control mean, treatment mean) for four mean metrics, plus a conversion
#: metric (exact binomial) and a metric whose arms are all one value (zero variance).
_MEANS = {
    ("m_a", "US"): (10.0, 12.0),
    ("m_a", "CA"): (10.0, 11.4),
    ("m_b", "US"): (10.0, 10.05),
    ("m_b", "CA"): (10.0, 10.0),
    ("m_c", "US"): (10.0, 10.9),
    ("m_c", "CA"): (10.0, 10.05),
}
_CONVERSIONS = {"US": (60, 96), "CA": (80, 88), "BR": (0, 25)}


def _segment_summary(*, with_conversion=True) -> pd.DataFrame:
    rows = []
    for (metric, country), (control, treatment) in _MEANS.items():
        rows.append(
            _mean_arm(400, control, 4.0, metric=metric, group_id="control", country=country)
        )
        rows.append(
            _mean_arm(400, treatment, 4.0, metric=metric, group_id="treatment", country=country)
        )
    if with_conversion:
        for country, (control, treatment) in _CONVERSIONS.items():
            rows.append(
                _binary_arm(300, control, metric="conv", group_id="control", country=country)
            )
            rows.append(
                _binary_arm(300, treatment, metric="conv", group_id="treatment", country=country)
            )
    frame = pd.DataFrame(rows)
    if with_conversion:
        frame["successes"] = pd.array([row.get("successes") for row in rows], dtype="Int64")
    return frame


def _metrics(*, with_conversion=True):
    metrics = [_mean_metric(name) for name in ("m_a", "m_b", "m_c")]
    if with_conversion:
        metrics.append(ConversionMetric(name="conv", entity="user", fact="conv"))
    return metrics


def _close(left, right, path="row"):
    """Structural equality with a relative tolerance on every float."""
    if isinstance(left, float) and isinstance(right, float):
        assert math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15), (path, left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys(), path
        for key in left:
            _close(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, (list, tuple)) and len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _close(a, b, f"{path}[{index}]")
    else:
        assert left == right, (path, left, right)


def _breakout(alternative="two-sided", *, summary=None, metrics=None, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IncrementWarning)
        return run_breakout(
            summary=_segment_summary() if summary is None else summary,
            metrics=_metrics() if metrics is None else metrics,
            control_group="control",
            dimension="country",
            alternative=alternative,
            **kwargs,
        )


def _evidence_p_values(metrics, summary) -> dict[object, float]:
    """Typed p-values straight from the estimator, independent of the family code."""
    computation = estimate_lift(metrics, summary, control_group="control")
    return {
        key: evidence.p_value
        for key, evidence in computation.evidence.items()
        if isinstance(evidence, PValueEvidence)
    }


# --- selection --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("zs", "q"),
    [
        pytest.param([4.5, 3.9, 2.5, 2.0, 0.3, 0.1], 0.1, id="mixed"),
        pytest.param([3.5, 2.4, 2.4, 2.4, 0.2], 0.2, id="tie-block-selected-together"),
        pytest.param([2.4, 2.4, 2.4, 0.2, 0.1], 0.05, id="tie-block-selected-together-or-not"),
        pytest.param([1.0, 0.5, 0.2], 0.05, id="nothing-selected"),
        pytest.param([6.0, 5.0, 4.0], 0.05, id="everything-selected"),
        pytest.param([4.0], 0.05, id="family-of-one-selected"),
        pytest.param([1.0], 0.05, id="family-of-one-kept"),
    ],
)
def test_discovery_equals_bh_select_on_the_rows_p_values(zs, q):
    rows = [_wald_row(f"m{i}", z) for i, z in enumerate(zs)]
    selected, threshold = bh_select([row.p_value() for row in rows], q)

    corrected = select_exploratory_family(rows, q=q)

    assert [row.metric for row in corrected] == [row.metric for row in rows]
    assert [row.discovery for row in corrected] == [i in selected for i in range(len(rows))]
    for row, original in zip(corrected, rows, strict=True):
        assert row.family_axes == ("metric", "arm")
        assert row.family_q == q
        assert row.family_size == len(rows)
        assert row.family_threshold == (threshold if selected else None)
        if not row.discovery:
            assert row.lift == original.lift


def test_empty_family_returns_nothing():
    assert select_exploratory_family([], q=0.1) == ()


@pytest.mark.parametrize("q", [0.0, -0.1, 1.5, float("nan"), float("inf")])
@pytest.mark.parametrize("rows", [[], [_wald_row("m", 4.0)]])
def test_invalid_q_is_refused_even_for_an_empty_family(q, rows):
    with pytest.raises(CodedError) as raised:
        select_exploratory_family(rows, q=q)
    assert raised.value.code == "estimation.family.bh_select_q_finite"


def test_unselected_rows_keep_their_nominal_interval_and_selected_rows_are_widened():
    rows = [_wald_row("hit", 6.0), _wald_row("miss", 0.2), _wald_row("near", 0.4)]
    hit, miss, near = select_exploratory_family(rows, q=0.05)

    assert (hit.discovery, miss.discovery, near.discovery) == (True, False, False)
    assert (miss.require_lift(), near.require_lift()) == (
        rows[1].require_lift(),
        rows[2].require_lift(),
    )
    assert hit.family_threshold == pytest.approx(0.05 / 3)
    widened, nominal = hit.require_lift(), rows[0].require_lift()
    assert widened.alpha == pytest.approx(0.05 / 3)
    assert widened.lb is not None and nominal.lb is not None
    assert widened.ub is not None and nominal.ub is not None
    assert widened.lb < nominal.lb and widened.ub > nominal.ub
    assert widened.value == nominal.value


def _segments(rows) -> list[BreakoutEstimate]:
    return [row for row in rows if isinstance(row, BreakoutEstimate)]


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
@pytest.mark.parametrize("reference", ["normal", "welch", "cluster"])
def test_wald_rows_are_reissued_as_their_estimator_builds_them(alternative, reference):
    """Each reference (Normal, Welch t, cluster t) and its additive sidecar reissue exactly."""
    variants: dict[str, dict[str, Any]] = {
        "normal": {},
        "welch": {"arm_ns": (40, 45), "abs_dof": 80.0},
        "cluster": {"dof": 18.0, "n_clusters": 20, "abs_dof": 36.0},
    }
    reference_kwargs = variants[reference]
    sign = -1.0 if alternative == "less" else 1.0
    half, zs, q = 0.05 / math.sqrt(2.0), [4.5, 3.4, 0.2], 0.05

    def build(index, alpha):
        return infer_lift(
            f"m{index}",
            "treatment",
            "unadjusted",
            sign * zs[index] * 0.05,
            half,
            half,
            alpha=alpha,
            alternative=alternative,
            abs_diff=1.0 + index,
            abs_se=0.3,
            method_role="decision",
            **reference_kwargs,
        )

    nominal = [build(i, NOMINAL_ALPHA) for i in range(len(zs))]
    assert all(row.abs_lb is not None for row in nominal)
    selected, threshold = bh_select([row.p_value() for row in nominal], q)
    assert selected == [0, 1]

    corrected = select_exploratory_family(nominal, q=q)

    fcr = min(threshold, NOMINAL_ALPHA)
    for index, row in enumerate(corrected):
        if index not in selected:
            assert row.require_lift() == nominal[index].require_lift()
            assert (row.abs_lb, row.abs_ub) == (nominal[index].abs_lb, nominal[index].abs_ub)
            continue
        expected = open_bound_from_two_sided_at_target(
            build(index, _fcr_alpha_for(alternative, fcr))
        )
        _close(row.require_lift().model_dump(), expected.require_lift().model_dump())
        _close((row.abs_lb, row.abs_ub), (expected.abs_lb, expected.abs_ub))
        assert expected.abs_alpha is not None and row.abs_alpha == expected.abs_alpha
        assert row.abs_lb is not None and nominal[index].abs_lb is not None
        assert row.abs_lb < nominal[index].abs_lb
        assert row.reference_kind == nominal[index].reference_kind


# --- equivalence with run_breakout(correction="bh") -------------------------------------


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
@pytest.mark.parametrize("q", [0.05, 0.3])
def test_selected_intervals_equal_run_breakout_bh(alternative, q):
    nominal = list(_breakout(alternative))
    reference = list(_breakout(alternative, correction="bh", q=q))

    corrected = select_exploratory_family(nominal, q=q)

    assert {row.reference_kind for row in corrected} >= {"binomial", "t"}
    assert len(corrected) == len(reference)
    tested = sum(row.excluded is None for row in reference)
    assert tested < len(reference)  # BR carries no mean-metric cells: excluded by design
    for ours, theirs in zip(corrected, reference, strict=True):
        _close(ours.model_dump(exclude={"family_size"}), theirs.model_dump(exclude={"family_size"}))
        assert ours.family_size == (tested if theirs.excluded is None else None)
    selected = [ours for ours in corrected if ours.discovery]
    assert bool(selected) == (alternative != "less")
    if q == 0.05 and alternative == "two-sided":
        widened = [
            (ours.require_lift(), before.require_lift())
            for ours, before in zip(corrected, nominal, strict=True)
            if ours.discovery and ours.reference_kind == "t"
        ]
        assert widened
        assert all(
            ours.lb is not None and before.lb is not None and ours.lb < before.lb
            for ours, before in widened
        )


@pytest.mark.parametrize("alternative", ["two-sided", "greater"])
@pytest.mark.parametrize("row_kind", ["whole", "breakout"])
def test_binomial_fcr_replaces_nominal_precision_disclosure(alternative, row_kind):
    from increment.estimation.binomial_rr import PRECISION_NOTE_PREFIX

    if row_kind == "breakout":
        nominal = list(_breakout(alternative))
    else:
        nominal = list(
            estimate_lift(
                [ConversionMetric(name="conv", entity="user", fact="conv")],
                [
                    _binary_arm(300, 60, metric="conv", group_id="control"),
                    _binary_arm(300, 96, metric="conv", group_id="treatment"),
                ],
                control_group="control",
                alternative=alternative,
            ).results
        )
    reference = select_exploratory_family(nominal, q=0.05)
    stale = f"source advisory | {PRECISION_NOTE_PREFIX}: nominal search"
    corrected = select_exploratory_family(
        [row.model_copy(update={"note": stale}) for row in nominal], q=0.05
    )
    selected = False
    for row, expected in zip(corrected, reference, strict=True):
        if row.discovery and row.reference_kind == "binomial":
            selected = True
            assert row.note == " | ".join(filter(None, ("source advisory", expected.note)))
        else:
            assert row.note == stale
    assert selected


def test_cuped_decision_rows_equal_run_breakout_bh():
    rows = []
    for country, treatment_mean in (("US", 12.0), ("CA", 10.05), ("MX", 11.0)):
        for group_id, mean in (("control", 10.0), ("treatment", treatment_mean)):
            rows.append(
                _mean_arm(
                    500,
                    mean,
                    4.0,
                    metric="m_a",
                    group_id=group_id,
                    country=country,
                    x_mean=0.0,
                    xy_cov=1.0,
                )
            )
    kwargs = {
        "summary": pd.DataFrame(rows),
        "metrics": [_mean_metric("m_a")],
        "methods": [Method(name="cuped", variance_reduction="cuped")],
    }
    nominal = list(_breakout(**kwargs))
    reference = list(_breakout(**kwargs, correction="bh", q=0.05))

    corrected = select_exploratory_family(nominal, q=0.05)

    assert {row.method for row in corrected} == {"cuped"}
    assert any(row.discovery for row in corrected) and not all(row.discovery for row in corrected)
    for ours, theirs in zip(corrected, reference, strict=True):
        _close(ours.model_dump(exclude={"family_size"}), theirs.model_dump(exclude={"family_size"}))


def test_a_cell_whose_estimation_failed_refuses_the_family_as_run_breakout_does():
    failed = BreakoutEstimate(
        metric="m_a",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        dimension="country",
        dimension_value="MX",
        lift=None,
        excluded="estimation_failed",
    )
    with pytest.raises(CapabilityError) as raised:
        select_exploratory_family([_wald_row("ok", 4.0), failed], q=0.1)
    assert raised.value.code == "family.evidence.incomplete"
    for clone in (copy.deepcopy(raised.value), pickle.loads(pickle.dumps(raised.value))):
        assert clone.code == raised.value.code
        assert str(clone) == str(raised.value)


def test_segment_family_reproduces_run_breakout_for_a_conservative_nonrejection():
    """A zero-variance cell is a non-rejection that still counts toward ``m``."""
    rows = [
        _mean_arm(400, 10.0, 4.0, metric="m_a", group_id="control", country="US"),
        _mean_arm(400, 12.0, 4.0, metric="m_a", group_id="treatment", country="US"),
        _mean_arm(400, 5.0, 0.0, metric="m_a", group_id="control", country="CA"),
        _mean_arm(400, 5.0, 0.0, metric="m_a", group_id="treatment", country="CA"),
    ]
    summary = pd.DataFrame(rows)
    metrics = [_mean_metric("m_a")]
    nominal = list(_breakout(summary=summary, metrics=metrics))
    reference = list(_breakout(summary=summary, metrics=metrics, correction="bh", q=0.05))

    corrected = _segments(select_exploratory_family(nominal, q=0.05))

    by_segment = {row.dimension_value: row for row in corrected}
    assert by_segment["CA"].excluded == "zero_variance"
    assert by_segment["CA"].discovery is False
    assert by_segment["US"].discovery is True
    assert {row.family_size for row in corrected} == {2}
    assert by_segment["US"].family_threshold == pytest.approx(0.025)
    ours = by_segment["US"].require_lift()
    theirs = next(row for row in reference if row.dimension_value == "US").require_lift()
    _close(ours.model_dump(), theirs.model_dump())


def test_a_cell_excluded_by_design_is_no_hypothesis_and_is_returned_unchanged():
    rows = [
        _mean_arm(400, 10.0, 4.0, metric="m_a", group_id="control", country="US"),
        _mean_arm(400, 12.0, 4.0, metric="m_a", group_id="treatment", country="US"),
        _mean_arm(400, 12.0, 4.0, metric="m_a", group_id="treatment", country="MX"),
    ]
    nominal = list(_breakout(summary=pd.DataFrame(rows), metrics=[_mean_metric("m_a")]))
    mexico = next(row for row in nominal if row.dimension_value == "MX")
    assert mexico.excluded == "no_control_arm"

    corrected = _segments(select_exploratory_family(nominal, q=0.1))

    assert next(row for row in corrected if row.dimension_value == "MX") == mexico
    (live,) = [row for row in corrected if row.dimension_value == "US"]
    assert live.family_size == 1 and live.discovery is True


def test_segment_cells_from_different_fact_sources_are_separate_hypotheses():
    """The same segment value read from two fact sources is two cells of one family."""
    segments = _segment_summary(with_conversion=False)
    segments = segments[segments["metric"].isin(["m_a", "m_c"])]
    metrics = [_mean_metric("m_a"), _mean_metric("m_c")]
    by_source = {
        source: list(_breakout(summary=segments, metrics=metrics, source=source))
        for source in ("event_log", "profiles")
    }
    cells = [*by_source["event_log"], *by_source["profiles"]]
    assert {row.source for row in cells} == {"event_log", "profiles"}
    q = 0.05

    p_values = []
    for cell in cells:
        slice_ = segments[segments["country"] == cell.dimension_value].drop(columns="country")
        key = ArmHypothesisKey(cell.metric, cell.group_id, "itt")
        p_values.append(_evidence_p_values(metrics, slice_)[key])
    selected, threshold = bh_select(p_values, q)
    assert selected and len(selected) < len(cells)

    corrected = select_exploratory_family(cells, q=q)

    assert [row.discovery for row in corrected] == [i in selected for i in range(len(cells))]
    assert {row.family_size for row in corrected} == {len(cells)}
    assert {row.family_threshold for row in corrected} == {threshold}
    assert [row.source for row in _segments(corrected)] == [row.source for row in cells]
    # Rows that differ only by source get the same verdict and the same interval.
    half = len(cells) // 2
    for left, right in zip(corrected[:half], corrected[half:], strict=True):
        assert left.discovery == right.discovery
        assert left.lift == right.lift

    twins = [row.model_copy(update={"estimand": "late"}) for row in (cells[0], cells[half])]
    with pytest.raises(CodedError) as raised:
        select_exploratory_family(twins, q=q)
    rows = cast("tuple[str, ...]", raised.value.context["rows"])
    assert len(rows) == 2 and len(set(rows)) == 2


def test_whole_window_and_segment_rows_are_one_family():
    pooled = pd.DataFrame(
        [
            _mean_arm(800, control, 4.0, metric=name, group_id="control")
            for name, control in (("m_a", 10.0), ("m_b", 10.0), ("m_c", 10.0))
        ]
        + [
            _mean_arm(800, treatment, 4.0, metric=name, group_id="treatment")
            for name, treatment in (("m_a", 11.7), ("m_b", 10.02), ("m_c", 10.5))
        ]
    )
    metrics = [_mean_metric(name) for name in ("m_a", "m_b", "m_c")]
    segments = _segment_summary(with_conversion=False)
    whole = estimate_lift(metrics, pooled, control_group="control").results
    cells = list(_breakout(summary=segments, metrics=metrics))
    q = 0.05

    # Independent p-values: the estimator's own typed evidence per whole-window and segment slice.
    pooled_p = _evidence_p_values(metrics, pooled)
    p_values = [pooled_p[ArmHypothesisKey(row.metric, row.group_id, "itt")] for row in whole]
    for cell in cells:
        slice_ = segments[segments["country"] == cell.dimension_value].drop(columns="country")
        key = ArmHypothesisKey(cell.metric, cell.group_id, "itt")
        p_values.append(_evidence_p_values(metrics, slice_)[key])
    selected, threshold = bh_select(p_values, q)

    corrected = select_exploratory_family([*whole, *cells], q=q)

    assert [row.discovery for row in corrected] == [i in selected for i in range(len(p_values))]
    assert {row.family_size for row in corrected} == {len(p_values)}
    assert {row.family_axes for row in corrected} == {("metric", "arm", "segment")}
    assert {row.family_threshold for row in corrected} == {threshold}
    assert any(row.discovery for row in corrected[: len(whole)])
    assert any(not row.discovery for row in corrected[len(whole) :])
    assert [type(row) for row in corrected] == [type(row) for row in [*whole, *cells]]

    # A selected whole-window row is the estimator's own second pass at the family level.
    fcr = min(threshold, NOMINAL_ALPHA)
    for original, row in zip(whole, corrected[: len(whole)], strict=True):
        if not row.discovery:
            continue
        again = estimate_lift(
            metrics, pooled, control_group="control", alpha=_fcr_alpha_for("two-sided", fcr)
        ).results
        expected = open_bound_from_two_sided_at_target(
            next(r for r in again if r.metric == original.metric)
        )
        _close(row.require_lift().model_dump(), expected.require_lift().model_dump())


def _cluster_row(group, totals):
    raw = {
        "experiment_id": "e",
        "metric": "m",
        "group_id": group,
        "n": len(totals),
        "sum_y": sum(totals),
        "sum_y2": sum(g * g for g in totals),
        "sum_x": None,
        "sum_x2": None,
        "sum_xy": None,
        "sum_den": float(len(totals)),
        "sum_den2": float(len(totals)),
        "sum_yden": sum(totals),
    }
    return centered_row_from_raw_sums(raw)


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_fieller_rows_are_reissued_from_their_joint_reference(alternative):
    metric = MeanMetric(name="m", entity="u", fact="f", aggregation="sum")

    sign = -1.0 if alternative == "less" else 1.0

    def totals(center):
        return [center + 0.1 * ((i % 5) - 2) for i in range(60)]

    summary = [_cluster_row("C", totals(5.0))] + [
        _cluster_row(group, totals(5.0 + sign * shift))
        for group, shift in (("T1", 0.6), ("T2", 0.3), ("T3", 0.01))
    ]

    def second_pass(alpha):
        return estimate_lift(
            [metric],
            summary,
            control_group="C",
            cluster="store",
            alpha=alpha,
            alternative=alternative,
        ).results

    nominal = second_pass(NOMINAL_ALPHA)
    assert all(row.relative_confidence_set is not None for row in nominal)
    q = 0.02
    selected, threshold = bh_select([row.p_value() for row in nominal], q)
    assert selected and len(selected) < len(nominal)

    corrected = select_exploratory_family(nominal, q=q)

    assert [row.discovery for row in corrected] == [i in selected for i in range(len(nominal))]
    fcr = min(threshold, NOMINAL_ALPHA)
    reissued = {row.group_id: row for row in second_pass(_fcr_alpha_for(alternative, fcr))}
    for index, (row, before) in enumerate(zip(corrected, nominal, strict=True)):
        if index not in selected:
            assert row.require_lift() == before.require_lift()
            assert row.relative_confidence_set == before.relative_confidence_set
            continue
        expected = open_bound_from_two_sided_at_target(reissued[row.group_id])
        _close(row.require_lift().model_dump(), expected.require_lift().model_dump())
        assert row.relative_confidence_set is not None
        assert expected.relative_confidence_set is not None
        _close(
            row.relative_confidence_set.model_dump(), expected.relative_confidence_set.model_dump()
        )
        _close((row.abs_lb, row.abs_ub), (expected.abs_lb, expected.abs_ub))
        assert expected.abs_alpha is not None and row.abs_alpha == expected.abs_alpha
        assert row.relative_confidence_set.alternative == alternative


@pytest.mark.parametrize(
    ("alternative", "null_abs"), [("two-sided", -0.5), ("less", -0.5), ("greater", -1.5)]
)
@pytest.mark.parametrize("q", [0.05, 0.4])
def test_a_margin_selected_row_with_an_unavailable_relative_interval_is_reissued(
    alternative, null_abs, q
):
    """A zero treatment mean leaves only the additive interval; the margin selects the row by it."""
    metric = _mean_metric("refunds")
    summary = pd.DataFrame(
        [
            _mean_arm(200, 1.0, 0.04, metric="refunds", group_id="control"),
            _mean_arm(200, 0.0, 0.0, metric="refunds", group_id="treatment"),
        ]
    )

    def second_pass(alpha):
        (row,) = estimate_lift(
            [metric],
            summary,
            control_group="control",
            alpha=alpha,
            alternative=alternative,
            null_abs=null_abs,
        ).results
        return row

    nominal = second_pass(NOMINAL_ALPHA)
    assert nominal.lift is None
    assert nominal.relative_unavailable_reason == "nonpositive_arm_mean"
    assert exploratory_family_exclusion(nominal) is None
    fillers = [_wald_row(f"m{i}", 0.1, alternative=alternative) for i in range(3)]

    corrected = select_exploratory_family([nominal, *fillers], q=q)

    row, *rest = corrected
    assert row.discovery is True
    assert [other.discovery for other in rest] == [False] * 3
    assert row.family_threshold == pytest.approx(q / 4)
    assert [other.require_lift() for other in rest] == [f.require_lift() for f in fillers]
    fcr = min(q / 4, NOMINAL_ALPHA)
    expected = second_pass(_fcr_alpha_for(alternative, fcr))
    _close((row.abs_lb, row.abs_ub), (expected.abs_lb, expected.abs_ub))
    assert (row.abs_diff, row.abs_se, row.lift) == (nominal.abs_diff, nominal.abs_se, None)
    assert nominal.abs_lb is not None and nominal.abs_ub is not None
    assert row.abs_lb is not None and row.abs_ub is not None
    if q / 4 < NOMINAL_ALPHA:
        assert row.abs_lb < nominal.abs_lb and row.abs_ub > nominal.abs_ub


def _offset_row(difference, variance, *, alternative="two-sided", alpha=NOMINAL_ALPHA):
    """An estimator row at a large offset: the arm means straddle zero, so only the additive
    interval exists, and it is cut around ``difference`` at float resolution."""

    def arm(group_id, mean):
        return {
            "experiment_id": "exp1",
            "metric": "refunds",
            "group_id": group_id,
            "n": 200.0,
            "ref_y": mean,
            "cy1": 0.0,
            "cy2": variance * 199.0,
        }

    (row,) = estimate_lift(
        [_mean_metric("refunds")],
        [arm("control", -difference / 2.0), arm("treatment", difference / 2.0)],
        control_group="control",
        alpha=alpha,
        alternative=alternative,
        null_abs=0.0,
    ).results
    assert row.relative_unavailable_reason == "nonpositive_arm_mean"
    return row


def _without_abs_alpha(row):
    """The row as a payload serialized before ``abs_alpha`` was persisted loads it."""
    payload = json.loads(row.model_dump_json())
    payload.pop("abs_alpha", None)
    return type(row).model_validate(payload)


@pytest.mark.parametrize("alternative", ["two-sided", "greater"])
@pytest.mark.parametrize(
    ("difference", "variance"),
    [
        (2.0, 4.0),
        (1.0e6, 4.0),
        (2.0**29 - 0.09, 1.0),
        (2.0**30 - 0.18, 4.0),
        (2.0**31 - 0.09, 1.0),
    ],
)
@pytest.mark.parametrize("q", [0.05, 0.4])
def test_a_selected_margin_row_at_a_large_offset_is_reissued_without_narrowing(
    alternative, difference, variance, q
):
    """However large the offset, a selected row's additive interval equals the estimator's at
    ``min(R*q/m, alpha)`` and is never narrower than the nominal interval."""
    nominal = _offset_row(difference, variance, alternative=alternative)
    assert exploratory_family_exclusion(nominal) is None
    fillers = [_wald_row(f"m{i}", 0.1, alternative=alternative) for i in range(3)]

    row, *rest = select_exploratory_family([nominal, *fillers], q=q)

    assert row.discovery is True and row.family_threshold == pytest.approx(q / 4)
    assert [other.discovery for other in rest] == [False] * 3
    fcr = min(q / 4, NOMINAL_ALPHA)
    expected = _offset_row(
        difference, variance, alternative=alternative, alpha=_fcr_alpha_for(alternative, fcr)
    )
    assert (row.abs_lb, row.abs_ub) == (expected.abs_lb, expected.abs_ub)
    assert nominal.abs_lb is not None and nominal.abs_ub is not None
    assert row.abs_lb is not None and row.abs_ub is not None
    assert row.abs_lb <= nominal.abs_lb and row.abs_ub >= nominal.abs_ub


@pytest.mark.parametrize("q", [0.05, 0.4])
def test_a_margin_row_with_bounds_near_the_float_limit_is_reissued_without_overflow(q):
    """Both bounds are finite while their difference is not. No estimator emits an ITT row with a
    standard error this large, so the joint row from ``infer_ate`` is labelled ITT to reach the
    family."""

    def build(alpha):
        return infer_ate(
            "m",
            "treatment",
            "unadjusted",
            point=None,
            scores=ScoreStats(metric="m", contrast="t", n=64, sum_psi=0.0, sum_psi2=2.56),
            alpha=alpha,
            abs_diff=5e307,
            abs_se=4.8e307,
            null_abs=-1.2e308,
            relative_unavailable_reason="joint_covariance_indefinite",
            method_role="decision",
        ).model_copy(update={"estimand": "itt"})

    nominal = build(NOMINAL_ALPHA)
    assert nominal.abs_lb is not None and nominal.abs_ub is not None
    assert math.isinf(nominal.abs_ub - nominal.abs_lb)
    fillers = [_wald_row(f"m{i}", 0.1) for i in range(3)]

    row, *_ = select_exploratory_family([nominal, *fillers], q=q)

    assert row.discovery is True and row.family_threshold == pytest.approx(q / 4)
    expected = build(min(q / 4, NOMINAL_ALPHA))
    assert (row.abs_lb, row.abs_ub) == (expected.abs_lb, expected.abs_ub)
    assert row.abs_lb is not None and row.abs_ub is not None
    assert row.abs_lb <= nominal.abs_lb and row.abs_ub >= nominal.abs_ub


def test_a_margin_row_without_its_persisted_alpha_is_left_out_of_the_family():
    """A row serialized before ``abs_alpha`` existed has an additive interval whose level is not
    recorded, and a level read back from the endpoints would be a guess."""
    legacy = _without_abs_alpha(_offset_row(2.0, 4.0))
    fillers = [_wald_row(f"m{i}", 0.1) for i in range(3)]

    assert legacy.abs_lb is not None and legacy.abs_ub is not None
    assert exploratory_family_exclusion(legacy)
    with pytest.raises(CodedError) as raised:
        select_exploratory_family([legacy, *fillers], q=0.4)
    assert raised.value.code == "estimation.family.exploratory_construction"
    assert [other.discovery for other in select_exploratory_family(fillers, q=0.4)] == [False] * 3


@pytest.mark.parametrize("alternative", ["two-sided", "greater"])
def test_a_legacy_wald_row_keeps_its_relative_level_and_gains_the_alpha_it_was_reissued_at(
    alternative,
):
    """Only an additive-only row depends on ``abs_alpha``: a Wald row's cap is its relative alpha."""
    half = 0.05 / math.sqrt(2.0)

    def build(alpha):
        return infer_lift(
            "m",
            "treatment",
            "unadjusted",
            4.5 * 0.05,
            half,
            half,
            alpha=alpha,
            alternative=alternative,
            abs_diff=1.0,
            abs_se=0.3,
            abs_dof=40.0,
            method_role="decision",
        )

    legacy = _without_abs_alpha(build(NOMINAL_ALPHA))
    fillers = [_wald_row(f"m{i}", 0.1, alternative=alternative) for i in range(3)]
    assert legacy.abs_alpha is None and exploratory_family_exclusion(legacy) is None

    row, *_ = select_exploratory_family([legacy, *fillers], q=0.05)

    expected = build(_fcr_alpha_for(alternative, 0.05 / 4))
    assert (row.abs_lb, row.abs_ub) == (expected.abs_lb, expected.abs_ub)
    assert expected.abs_alpha is not None and row.abs_alpha == expected.abs_alpha


# --- refusals ---------------------------------------------------------------------------


def _sequential_row():
    from increment import SequentialCell, estimate_sequential
    from increment.estimation.sequential import AlwaysValid
    from tests.sequential_cases import capture, records, registration

    reg = registration(cells=(SequentialCell(metric="outcome", group_id="treatment", family=True),))
    bundle = estimate_sequential(
        capture(reg, records([0, 0, 0, 1] * 24, [0, 1, 1, 1] * 24)),
        AlwaysValid(registration=reg),
    )
    return bundle.results[0]


def _winsor_row():
    from increment.estimation.winsor import estimate_winsor_lift
    from increment.winsor import RawArm, WinsorInferenceSpec, WinsorRawState, WinsorSupport

    raw = WinsorRawState(
        metric="revenue",
        study_id="e",
        population="assigned",
        missingness="error",
        quantile=0.75,
        inference=WinsorInferenceSpec(method="joint-rank-projection-v1"),
        support=WinsorSupport(lower=0, upper=None, provenance="External finite oracle support"),
        arms=(
            RawArm(group_id="C", values=(1, 2, 3, 9)),
            RawArm(group_id="T", values=(1, 2, 3, 5)),
        ),
    )
    return estimate_winsor_lift(raw, "C", "T")


def _additive_row():
    return infer_ate(
        "m",
        "treatment",
        "unadjusted",
        point=1.0,
        scores=ScoreStats(metric="m", contrast="t", n=64, sum_psi=0.0, sum_psi2=0.04 * 64),
        value_scale="absolute",
        method_role="decision",
    )


def _plain():
    return _wald_row("plain", 4.0)


def _prior_breakout_rows():
    summary = _segment_summary(with_conversion=False).query("metric == 'm_a'")
    return list(
        _breakout(summary=summary, metrics=[_mean_metric("m_a")], prior=Normal(mu=0.0, sigma=0.1))
    )


#: refusal code -> one builder per input shape that must raise it; ``plain`` is never to blame.
_REFUSALS = {
    "estimation.family.exploratory_row": {
        "day-axis": lambda: [
            _plain(),
            _plain().model_copy(update={"metric": "daily", "ds": date(2025, 1, 1)}),
        ],
        "not-a-row": lambda: [_plain(), "not a row"],
    },
    "estimation.family.exploratory_pre_corrected": {
        "run_breakout-bh": lambda: [_plain(), *_breakout(correction="bh", q=0.05)],
        "exploratory-output": lambda: [select_exploratory_family([_wald_row("a", 4.0)], q=0.1)[0]],
    },
    "estimation.family.exploratory_non_decision": {
        "sensitivity": lambda: [_plain(), _wald_row("companion", 4.0, method_role="sensitivity")],
    },
    "estimation.family.exploratory_sequential": {
        "always-valid": lambda: [_plain(), _sequential_row()],
    },
    "breakout.run_breakout_bh_excludes_prior": {
        "segment-prior": lambda: [_plain(), *_prior_breakout_rows()],
        "whole-window-prior": lambda: [
            _plain(),
            _wald_row("shrunk", 3.0, prior=Normal(mu=0.0, sigma=0.1)),
        ],
    },
    "estimation.family.exploratory_construction": {
        "quantile": lambda: [
            _plain(),
            _plain().model_copy(update={"metric": "q", "quantile_p_value": 0.01}),
        ],
        "winsor-confidence-set": lambda: [_plain(), _winsor_row()],
        "additive-ate": lambda: [_plain(), _additive_row()],
    },
}
_REFUSAL_PARAMS = [
    pytest.param(code, build, id=f"{code}:{name}")
    for code, builds in _REFUSALS.items()
    for name, build in builds.items()
]


@pytest.mark.parametrize(("code", "build"), _REFUSAL_PARAMS)
def test_inadmissible_rows_are_refused_with_a_stable_code(code, build):
    with pytest.raises(CodedError) as raised:
        select_exploratory_family(build(), q=0.1)
    assert raised.value.code == code
    named = cast("tuple[str, ...]", raised.value.context["rows"])
    assert named and all(isinstance(label, str) for label in named)
    assert "plain/treatment" not in named


def test_every_offending_row_is_named_and_the_first_hazard_wins():
    clean = _wald_row("clean", 4.0)
    sensitivity = _wald_row("companion", 3.0, method_role="sensitivity")
    sequential = _sequential_row()
    corrected = select_exploratory_family([_wald_row("done", 4.0)], q=0.1)[0]

    with pytest.raises(CodedError) as raised:
        select_exploratory_family([clean, sequential, sensitivity, corrected, sensitivity], q=0.1)
    assert raised.value.code == "estimation.family.exploratory_pre_corrected"
    assert raised.value.context["rows"] == ("done/treatment",)

    with pytest.raises(CodedError) as raised:
        select_exploratory_family([clean, sequential, sensitivity, sensitivity], q=0.1)
    assert raised.value.code == "estimation.family.exploratory_non_decision"
    assert raised.value.context["rows"] == ("companion/treatment", "companion/treatment")


def test_a_segment_prior_row_is_marked_so_it_cannot_enter_a_family():
    rows = _prior_breakout_rows()
    assert rows and all(isinstance(row, BreakoutEstimate) and row.prior_shrunk for row in rows)
    summary = _segment_summary(with_conversion=False).query("metric == 'm_a'")
    plain = list(_breakout(summary=summary, metrics=[_mean_metric("m_a")]))
    assert not any(row.prior_shrunk for row in plain)


def test_construction_refusal_names_each_row_and_why():
    quantile = _wald_row("q", 4.0).model_copy(update={"quantile_p_value": 0.01})
    with pytest.raises(CodedError) as raised:
        select_exploratory_family([_wald_row("ok", 4.0), quantile, _winsor_row()], q=0.1)
    assert raised.value.code == "estimation.family.exploratory_construction"
    rows = cast("tuple[str, ...]", raised.value.context["rows"])
    reasons = cast("tuple[str, ...]", raised.value.context["constructions"])
    assert rows == ("q/treatment", "revenue/T")
    assert len(reasons) == 2 and all(reasons)


@pytest.mark.parametrize(
    ("code", "build"),
    [
        pytest.param(code, next(iter(builds.values())), id=code)
        for code, builds in _REFUSALS.items()
    ],
)
def test_refusals_survive_pickle_and_deepcopy(code, build):
    with pytest.raises(CodedError) as raised:
        select_exploratory_family(build(), q=0.1)
    refusal = raised.value
    assert refusal.code == code
    for clone in (copy.deepcopy(refusal), pickle.loads(pickle.dumps(refusal))):
        assert type(clone) is type(refusal)
        assert clone.code == refusal.code
        assert dict(clone.context) == dict(refusal.context)
        assert str(clone) == str(refusal)


@pytest.mark.parametrize(("code", "build"), _REFUSAL_PARAMS)
def test_exclusion_reasons_match_the_refusal_and_dropping_them_admits_the_rest(code, build):
    rows = build()
    with pytest.raises(CodedError) as raised:
        select_exploratory_family(rows, q=0.1)
    context = raised.value.context
    expected = cast("tuple[str, ...]", context.get("reasons", context.get("constructions")))

    excluded = [row for row in rows if exploratory_family_exclusion(row) is not None]
    assert tuple(exploratory_family_exclusion(row) for row in excluded) == expected
    assert all(isinstance(reason, str) and reason for reason in expected)
    kept = [row for row in rows if exploratory_family_exclusion(row) is None]
    assert len(select_exploratory_family(kept, q=0.1)) == len(kept)


def test_admissible_rows_have_no_exclusion_reason():
    rows = [*_breakout(), _wald_row("plain", 3.0), _wald_row("greater", 3.0, alternative="greater")]
    assert any(row.excluded for row in _segments(rows))  # design-excluded cells stay admissible
    assert [exploratory_family_exclusion(row) for row in rows] == [None] * len(rows)
    assert select_exploratory_family(rows, q=0.1)


# --- false coverage rate ----------------------------------------------------------------


def _false_coverage_rate(reps, *, seed=20261004, m=8, q=0.2, se=0.05):
    """Mean false-coverage proportion of selected intervals, corrected and nominal."""
    rng = np.random.default_rng(seed)
    signal = np.array([0.0] * 4 + [2.0, 3.0, 4.0, 6.0]) * se
    corrected_rates, nominal_rates, widened = [], [], True
    for _ in range(reps):
        estimates = signal + se * rng.standard_normal(m)
        rows = [_wald_row(f"m{i}", estimates[i] / se, se=se, alpha=0.5) for i in range(m)]
        out = select_exploratory_family(rows, q=q)
        wrong_corrected = wrong_nominal = selected = 0
        for i, (row, before) in enumerate(zip(out, rows, strict=True)):
            if not row.discovery:
                continue
            selected += 1
            truth = math.expm1(signal[i])
            interval, nominal = row.require_lift(), before.require_lift()
            assert interval.lb is not None and interval.ub is not None
            assert nominal.lb is not None and nominal.ub is not None
            wrong_corrected += not (interval.lb <= truth <= interval.ub)
            wrong_nominal += not (nominal.lb <= truth <= nominal.ub)
            widened &= interval.lb <= nominal.lb and interval.ub >= nominal.ub
        corrected_rates.append(wrong_corrected / selected if selected else 0.0)
        nominal_rates.append(wrong_nominal / selected if selected else 0.0)
    return np.array(corrected_rates), np.array(nominal_rates), widened


def test_false_coverage_smoke_selected_intervals_contain_their_nominal_intervals():
    corrected, nominal, widened = _false_coverage_rate(40)
    assert widened
    assert corrected.mean() <= nominal.mean()


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_selected_interval_false_coverage_rate_is_controlled_near_q():
    q = 0.2
    corrected, nominal, widened = _false_coverage_rate(2000, q=q)
    mcse = corrected.std(ddof=1) / math.sqrt(len(corrected))
    assert widened
    assert corrected.mean() <= q + 4 * mcse
    assert nominal.mean() > q + 4 * mcse
    assert corrected.mean() > 0.0
