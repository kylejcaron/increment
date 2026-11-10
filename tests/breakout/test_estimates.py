"""Tests for `increment.breakout.estimates` - per-segment lift estimation. `run_breakout`'s
control-arm lookup is a dict keyed ONLY by metric name; one call across two segments would
silently drop a control arm."""

from __future__ import annotations

import math
import warnings
from datetime import date
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pytest

from increment.breakout.estimates import (
    BreakoutEstimate,
    BreakoutEstimates,
    DailyLiftEstimate,
    DailyLiftEstimates,
    DailyMetricValue,
    DailyMetricValues,
    EstimateList,
    LiftEstimates,
    run_breakout,
)
from increment.breakout.projection import to_frame
from increment.errors import CapabilityError, IncrementWarning, InvalidRequestError
from increment.estimation.armstats import centered_row_from_raw_sums
from increment.estimation.binomial_rr import FINITE_SAMPLE_MAX_ARM_SIZE
from increment.estimation.conversion_route import dense_min_count
from increment.estimation.engine import Method, estimate_lift
from increment.estimation.exploratory_family import select_exploratory_family
from increment.estimation.family import bh_select, e_bh_select
from increment.estimation.inference import Normal
from increment.estimation.results import Estimate, LiftEstimate
from increment.estimation.sequential import AlwaysValid
from increment.semantics.models import ConversionMetric, MeanMetric, Measure, RatioMetric
from tests.estimation._binomial_endpoint_reference import assert_set_contains_finer_reference
from tests.warning_codes import allow_warning_codes, warning_codes

# Helpers mirroring test_engine.py's arm/summary/metric fixture
# conventions, plus a breakout dimension column.


def _make_arm_row(
    n: int,
    mean: float,
    var: float,
    *,
    country: str,
    experiment_id: str = "exp1",
    metric: str = "rev",
    group_id: str = "control",
    sum_den: float | None = None,
    x_mean: float | None = None,
    x_var: float = 1.0,
    xy_cov: float = 0.0,
) -> dict:
    """Build one centered `group_summary` row with a `country` breakout column, like
    test_engine.py's `_make_arm_stats`. `sum_den` is optional; `cden2`/`cyden` land as 0.0
    (degenerate but COMPLETE) since a partial family is refused at the estimation seam."""
    sum_y = float(n) * mean
    sum_y2 = var * (n - 1) + sum_y**2 / float(n)
    x_sum = None if x_mean is None else float(n) * x_mean
    sum_x = x_sum
    sum_x2 = None if x_sum is None else x_var * (n - 1) + x_sum**2 / float(n)
    sum_xy = None if x_sum is None else xy_cov * (n - 1) + x_sum * sum_y / float(n)
    row = centered_row_from_raw_sums(
        {
            "experiment_id": experiment_id,
            "metric": metric,
            "group_id": group_id,
            "country": country,
            "n": n,
            "successes": None,
            "sum_y": sum_y,
            "sum_y2": sum_y2,
            "sum_x": sum_x,
            "sum_x2": sum_x2,
            "sum_xy": sum_xy,
            "sum_den": None,
            "sum_den2": None,
            "sum_yden": None,
        }
    )
    if sum_den is not None:
        row["ref_den"] = sum_den / float(n)
        row["cden1"] = 0.0
        row["cden2"] = 0.0
        row["cyden"] = 0.0
    return row


def _mean_metric(name: str = "rev") -> MeanMetric:
    """Build a MeanMetric fixture for tests using mean metrics."""
    return MeanMetric(name=name, entity="user", fact=name)


def _windowed_mean_metric(name: str = "rev") -> MeanMetric:
    """Closed-horizon MeanMetric (`window_days` set): avoids the unrelated open-ended-metric
    warning in tests exercising the uncorrected-multiplicity warning specifically."""
    return MeanMetric(name=name, entity="user", fact=name, window_days=30)


def _ratio_metric(name: str = "ratio_rev") -> RatioMetric:
    """Build a RatioMetric fixture for tests using ratio metrics."""
    return RatioMetric(
        name=name,
        entity="user",
        numerator=Measure(fact="num"),
        denominator=Measure(fact="den"),
    )


def _conversion_metric(name: str = "conv") -> ConversionMetric:
    """Build a ConversionMetric fixture for tests using conversion metrics."""
    return ConversionMetric(name=name, entity="user", fact=name)


def _conversion_arm_row(
    n: int,
    successes: int,
    *,
    country: str,
    group_id: str,
    metric: str = "conv",
    with_covariate: bool = False,
) -> dict:
    """A centered `group_summary` row for a genuinely binary (0/1)
    conversion arm, `successes` out of `n` - the exact binomial risk-ratio method
    admits any (n, successes) pair, including n=1 and successes=0."""
    return centered_row_from_raw_sums(
        {
            "experiment_id": "exp1",
            "metric": metric,
            "group_id": group_id,
            "country": country,
            "n": n,
            "sum_y": float(successes),
            "sum_y2": float(successes),  # y in {0, 1}: y**2 == y
            "successes": successes,
            "sum_x": 0.0 if with_covariate else None,
            "sum_x2": 0.0 if with_covariate else None,
            "sum_xy": 0.0 if with_covariate else None,
            "sum_den": None,
            "sum_den2": None,
            "sum_yden": None,
        }
    )


def test_segment_rekey_accepts_sensitivity_and_decision_rows_with_advisories():
    rows = [
        _make_arm_row(
            50,
            10.0 if group == "control" else 12.0,
            4.0,
            country="US",
            group_id=group,
            metric="ratio_rev",
            sum_den=50.0,
        )
        for group in ("control", "treatment")
    ]
    for row in rows:
        row["cden2"] = 220.5
    results = run_breakout(
        rows,
        [_ratio_metric("ratio_rev")],
        control_group="control",
        dimension="country",
        methods=[Method(name="unadjusted"), Method(name="sensitivity")],
        method_roles={"unadjusted": "decision", "sensitivity": "sensitivity"},
    )
    assert {row.method_role for row in results} == {"decision", "sensitivity"}
    assert all(row.dimension == "country" and row.dimension_value == "US" for row in results)


def _planned_breakout(
    rows,
    metrics,
    *,
    hypotheses,
    methods=None,
    method_roles=None,
    correction="none",
    q=0.1,
):
    from increment import readouts
    from increment.decision import AbsoluteArmDecisionProcedure, RelativeArmDecisionProcedure
    from increment.plan import compile_decision_plan
    from increment.semantics.design import Randomized
    from increment.sources import BreakoutMomentsSource

    methods = methods or [Method(name="unadjusted")]
    decision = next(
        (method for method in methods if (method_roles or {}).get(method.name) == "decision"),
        methods[0],
    )
    procedures = {}
    for metric in metrics:
        alpha, alternative, null_lift, null_abs, role = hypotheses[metric.name]
        common: dict[str, Any] = {
            "metric": metric.name,
            "alpha": alpha,
            "alternative": alternative,
            "role": role,
            "decision_method": decision,
            "sensitivity_methods": tuple(method for method in methods if method != decision),
        }
        procedures[metric.name] = (
            RelativeArmDecisionProcedure(**common, null_lift=null_lift)
            if null_abs is None
            else AbsoluteArmDecisionProcedure(**common, null_abs=null_abs)
        )
    plan = compile_decision_plan(None, metrics).model_copy(
        update={"declared": True, "alpha": alpha, "q": q, "procedures": procedures}
    )
    source = BreakoutMomentsSource(
        rows,
        dimension="country",
        metrics=metrics,
        study_id="exp1",
        design=Randomized(control_group="control"),
        plan=plan,
    )
    return readouts.breakout(source, "country", correction=correction, q=q)


def _compiled_mean_breakout(
    *,
    alternative="two-sided",
    margin=None,
    margin_abs=None,
    direction="increase",
    control_mean=10.0,
    treatment_mean=10.0,
    alpha=0.05,
    correction="none",
):
    from increment import (
        AnalysisPlan,
        ExperimentMetric,
        MetricSpec,
        MultiplicitySpec,
        View,
        from_frame,
        run,
    )

    frame = pd.DataFrame(
        [
            {
                "unit": f"{arm}-{i}",
                "arm": arm,
                "country": "US",
                "rev": mean + offset,
            }
            for arm, mean in (("control", control_mean), ("treatment", treatment_mean))
            for i, offset in enumerate([-1.0, 1.0] * 50)
        ]
    )
    experiment, reader = from_frame(
        frame,
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        breakouts=["country"],
        metrics=[
            MetricSpec(
                name="rev",
                type="mean",
                preferred_direction=direction,
            )
        ],
        plan=AnalysisPlan(
            primary=ExperimentMetric(metric="rev", margin=margin, margin_abs=margin_abs),
            alpha=alpha,
            alternative=alternative,
            view_multiplicity=MultiplicitySpec(correction=correction),
        ),
    )
    view = View(view="breakout", grain="total")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IncrementWarning)
            return run(experiment.with_views(view), reader).readout(view)
    finally:
        reader.close()


@pytest.mark.parametrize(
    ("alternative", "null", "direction"),
    [("greater", -0.02, "increase"), ("less", 0.05, "decrease")],
)
def test_compiled_segment_relative_margin_uses_declared_welch_tail(alternative, null, direction):
    from scipy.stats import t

    for delta in (-0.01, 0.0, 0.01):
        mean_t = 10.0 * (1.0 + null + delta)
        (row,) = _compiled_mean_breakout(
            margin=abs(null), direction=direction, treatment_mean=mean_t
        )
        variance = 100.0 / 99.0
        v_c, v_t = variance / (100 * 10.0**2), variance / (100 * mean_t**2)
        df = (v_c + v_t) ** 2 / ((v_c**2 + v_t**2) / 99)
        z = (math.log(mean_t / 10.0) - math.log1p(null)) / math.sqrt(v_c + v_t)
        expected = t.sf(z, df) if alternative == "greater" else t.cdf(z, df)
        assert (row.alternative, row.null_lift, row.null_abs) == (alternative, null, None)
        assert row.p_value() == pytest.approx(expected, rel=1e-10, abs=1e-13)
        assert row.require_lift().alpha == pytest.approx(0.1)
        assert row.role == "exploratory"
        reference_rows = [
            _make_arm_row(100, mean, variance, country="US", group_id=arm)
            for arm, mean in (("control", 10.0), ("treatment", mean_t))
        ]
        (reference,) = estimate_lift(
            [_mean_metric()],
            reference_rows,
            control_group="control",
            alpha=0.05,
            alternative=alternative,
            null_lift=null,
        ).results
        assert (row.require_lift().lb, row.require_lift().ub) == pytest.approx(
            (reference.require_lift().lb, reference.require_lift().ub)
        )


@pytest.mark.parametrize(
    ("control_mean", "treatment_mean"),
    [(10.0, 10.25), (0.0, 0.25), (-1.0, -0.75), (0.25, 0.0)],
)
def test_compiled_segment_absolute_margin_uses_metric_units_and_additive_reference(
    control_mean, treatment_mean
):
    from scipy.stats import t

    (row,) = _compiled_mean_breakout(
        margin_abs=0.5,
        direction="decrease",
        control_mean=control_mean,
        treatment_mean=treatment_mean,
    )
    difference = treatment_mean - control_mean
    se = math.sqrt(2.0 / 99.0)
    assert row.excluded is None
    assert (row.alternative, row.null_lift, row.null_abs) == ("less", 0.0, 0.5)
    assert row.p_value() == pytest.approx(t.cdf((difference - 0.5) / se, 198))
    assert row.abs_diff == pytest.approx(difference)
    assert row.abs_se == pytest.approx(se)
    assert row.abs_ub == pytest.approx(difference + t.isf(0.05, 198) * se)
    assert row.abs_reference_kind == "t"
    assert row.abs_reference_df == pytest.approx(198)
    assert row.abs_alpha == pytest.approx(0.1)
    assert row.stat_sig() is True
    if min(control_mean, treatment_mean) <= 0.0:
        assert row.lift is None
        assert row.relative_confidence_set is None
        assert row.relative_unavailable_reason == "nonpositive_arm_mean"


def test_absolute_segment_nonpositive_means_keep_bh_selection_and_fcr_on_replay():
    from scipy.stats import t

    differences = {"A": 0.2, "B": -0.5, "C": 0.5}
    rows = [
        _make_arm_row(
            100,
            mean,
            100.0 / 99.0,
            country=segment,
            group_id=arm,
        )
        for segment, difference in differences.items()
        for arm, mean in (("control", -1.0), ("treatment", -1.0 + difference))
    ]
    policy = (0.05, "less", 0.0, 0.5, "guardrail")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IncrementWarning)
        result = _planned_breakout(
            rows,
            [_mean_metric()],
            hypotheses={"rev": policy},
            correction="bh",
            q=0.03,
        )
    se = math.sqrt(2.0 / 99.0)
    for original in result:
        row = BreakoutEstimate.model_validate_json(original.model_dump_json())
        difference = differences[row.dimension_value]
        assert row.excluded is None
        assert row.lift is None
        assert row.relative_unavailable_reason == "nonpositive_arm_mean"
        assert (row.alternative, row.null_lift, row.null_abs) == ("less", 0.0, 0.5)
        assert row.p_value() == pytest.approx(t.cdf((difference - 0.5) / se, 198))
        assert row.discovery is (row.dimension_value != "C")
        assert row.family_size == 3
        assert row.family_threshold == pytest.approx(0.02)
        if row.discovery:
            assert row.family_nominal_alpha == 0.05
        assert row.abs_alpha == pytest.approx(0.02 if row.discovery else 0.1)
        tail = 0.01 if row.discovery else 0.05
        assert row.abs_ub == pytest.approx(difference + t.isf(tail, 198) * se)


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("zero_control_n", "missing_metric_observations"),
        ("zero_treatment_n", "missing_metric_observations"),
        ("one_control_n", "no_control_arm"),
        ("one_treatment_n", "few_units"),
        ("absent_control", "no_control_arm"),
        ("ratio_missing_denominator", "nonpositive_mean"),
        ("ratio_zero_denominator", "nonpositive_mean"),
        ("ratio_scalar_cuped", "nonpositive_mean"),
        ("relative_null", "nonpositive_mean"),
    ],
)
def test_absolute_segment_admission_preserves_count_and_relative_domain_gates(case, expected):

    ratio = case.startswith("ratio")
    metric = _ratio_metric("rev") if ratio else _mean_metric()
    control_mean = 1.0 if case.endswith("_n") or case == "absent_control" else -1.0
    rows = [
        _make_arm_row(
            100,
            mean,
            1.0,
            country="US",
            group_id=arm,
            sum_den=100.0 if ratio else None,
            x_mean=0.0 if case == "ratio_scalar_cuped" else None,
        )
        for arm, mean in (("control", control_mean), ("treatment", control_mean + 0.2))
    ]
    if case == "absent_control":
        rows.pop(0)
    elif case.endswith("_n"):
        rows[0 if "control" in case else 1]["n"] = 0 if case.startswith("zero") else 1
    elif case == "ratio_missing_denominator":
        rows[1]["ref_den"] = None
    elif case == "ratio_zero_denominator":
        rows[1]["ref_den"] = 0.0
    policy = (
        0.05,
        "less",
        0.05 if case == "relative_null" else 0.0,
        None if case == "relative_null" else 0.5,
        "guardrail",
    )
    methods = (
        [Method(name="cuped", variance_reduction="cuped")] if case == "ratio_scalar_cuped" else None
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IncrementWarning)
        result = _planned_breakout(
            rows,
            [metric],
            hypotheses={"rev": policy},
            methods=methods,
            correction="bh",
            q=0.1,
        )
    assert all(row.excluded == expected for row in result)
    assert all(row.p_value() is None for row in result)
    for row in result:
        assert (row.alternative, row.null_lift, row.null_abs) == policy[1:4]


@pytest.mark.parametrize(
    "kind",
    ["cuped_mean", "ratio", "ratio_mixed_unadjusted_decision", "ratio_mixed_cuped_decision"],
)
def test_absolute_segment_nonpositive_mean_reaches_existing_additive_method(kind):
    from scipy.stats import t

    metric = _mean_metric() if kind == "cuped_mean" else _ratio_metric("rev")
    methods = [Method(name="cuped", variance_reduction="cuped")] if kind == "cuped_mean" else None
    mixed = kind.startswith("ratio_mixed")
    if mixed:
        methods = [Method(name="unadjusted"), Method(name="cuped", variance_reduction="cuped")]
    decision = "cuped" if kind == "ratio_mixed_cuped_decision" else "unadjusted"
    rows = [
        _make_arm_row(
            100,
            mean,
            1.0,
            country="US",
            group_id=arm,
            x_mean=0.0 if kind == "cuped_mean" or mixed else None,
            sum_den=100.0 if kind.startswith("ratio") else None,
        )
        for arm, mean in (("control", -1.0), ("treatment", -0.8))
    ]
    policy = (0.05, "less", 0.0, 0.5, "guardrail")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IncrementWarning)
        result = _planned_breakout(
            rows,
            [metric],
            hypotheses={"rev": policy},
            methods=methods,
            method_roles=(
                {
                    name: "decision" if name == decision else "sensitivity"
                    for name in ("unadjusted", "cuped")
                }
                if mixed
                else None
            ),
            correction="bh" if mixed else "none",
            q=0.1,
        )
    row = next(
        row for row in result if row.method == ("cuped" if kind == "cuped_mean" else "unadjusted")
    )
    assert row.excluded is None
    assert row.relative_unavailable_reason == "nonpositive_arm_mean"
    assert row.abs_diff == pytest.approx(0.2)
    assert row.abs_se == pytest.approx(math.sqrt(0.02))
    assert row.p_value() == pytest.approx(t.cdf(-0.3 / math.sqrt(0.02), 198))
    if mixed:
        cuped = next(row for row in result if row.method == "cuped")
        assert cuped.excluded == "nonpositive_mean"
        decision_row = next(row for row in result if row.method_role == "decision")
        assert decision_row.family_size == 1
        assert decision_row.discovery is (decision == "unadjusted")


def test_absolute_ratio_cuped_fcr_keeps_nonpositive_sibling_out_of_method_input():
    from scipy.stats import t

    rows = [
        _make_arm_row(200, mean, 1.0, country="US", group_id=arm, sum_den=200.0, x_mean=0.0)
        for arm, mean in (("control", 10.0), ("good", 11.0), ("bad", -1.0))
    ]
    for row in rows:
        row["cxden"] = 0.0
    policy = (0.05, "greater", 0.0, 0.5, "guardrail")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IncrementWarning)
        result = _planned_breakout(
            rows,
            [_ratio_metric("rev")],
            hypotheses={"rev": policy},
            methods=[
                Method(name="cuped", variance_reduction="cuped"),
                Method(name="unadjusted"),
            ],
            method_roles={"cuped": "decision", "unadjusted": "sensitivity"},
            correction="bh",
            q=0.04,
        )
    by_cell = {(row.group_id, row.method): row for row in result}
    selected = by_cell[("good", "cuped")]
    assert selected.discovery is True
    assert selected.family_size == 2
    assert selected.abs_alpha == pytest.approx(0.02)
    assert selected.abs_lb == pytest.approx(1.0 - t.isf(0.01, 398) * 0.1)
    excluded = by_cell[("bad", "cuped")]
    assert excluded.excluded == "nonpositive_mean"
    assert excluded.discovery is False
    assert excluded.family_size == 2
    assert by_cell[("bad", "unadjusted")].excluded is None


def test_absolute_constant_nonpositive_segment_keeps_valid_bh_sibling():

    rows = [
        _make_arm_row(100, mean, variance, country=segment, group_id=arm)
        for segment, variance, treatment in (("constant", 0.0, -0.8), ("variable", 1.0, -1.5))
        for arm, mean in (("control", -1.0), ("treatment", treatment))
    ]
    policy = (0.05, "less", 0.0, 0.5, "guardrail")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IncrementWarning)
        result = _planned_breakout(
            rows,
            [_mean_metric()],
            hypotheses={"rev": policy},
            correction="bh",
            q=0.04,
        )
    by_segment = {row.dimension_value: row for row in result}
    assert by_segment["variable"].discovery is True
    assert by_segment["variable"].family_size == 2
    assert by_segment["constant"].discovery is False
    assert by_segment["constant"].family_size == 2
    assert by_segment["constant"].excluded == "zero_variance"
    assert by_segment["constant"].p_value() is None
    assert (by_segment["constant"].alternative, by_segment["constant"].null_abs) == ("less", 0.5)


def test_absolute_mixed_binary_cuped_keeps_additive_row_and_count_only_bonferroni():
    from scipy.stats import t

    rows = [
        _conversion_arm_row(
            n,
            successes,
            country=segment,
            group_id=arm,
            with_covariate=True,
        )
        for segment, n, successes_t in (("A", 100, 0), ("B", 1, 0), ("C", 100, 50))
        for arm, successes in (("control", max(n // 2, 1)), ("treatment", successes_t))
    ]
    for row in rows:
        row["cx2"] = float(row["n"])
    policy = (0.05, "less", 0.0, -0.1, "guardrail")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IncrementWarning)
        result = _planned_breakout(
            rows,
            [_conversion_metric()],
            hypotheses={"conv": policy},
            methods=[
                Method(name="cuped", variance_reduction="cuped"),
                Method(name="unadjusted"),
            ],
            method_roles={"cuped": "decision", "unadjusted": "sensitivity"},
            correction="bonferroni",
            q=0.1,
        )
    by_cell = {(row.dimension_value, row.method): row for row in result}
    additive = by_cell[("A", "cuped")]
    se = math.sqrt(0.25 / 99.0)
    assert additive.excluded is None
    assert additive.relative_unavailable_reason == "nonpositive_arm_mean"
    assert additive.p_value() == pytest.approx(t.cdf(-0.4 / se, 99))
    assert additive.abs_alpha == pytest.approx(0.05)
    assert additive.abs_ub == pytest.approx(-0.5 + t.isf(0.025, 99) * se)
    assert by_cell[("C", "cuped")].abs_alpha == pytest.approx(0.05)
    assert by_cell[("B", "cuped")].excluded == "few_units"
    assert by_cell[("B", "unadjusted")].excluded is None
    assert by_cell[("B", "unadjusted")].reference_kind == "binomial"


def test_compiled_segment_zero_null_keeps_one_sided_alternative_at_tiny_alpha():
    from scipy.stats import t

    (row,) = _compiled_mean_breakout(alternative="greater", treatment_mean=12.0, alpha=1e-20)
    variance = 100.0 / 99.0
    vc, vt = variance / 10000.0, variance / 14400.0
    df = (vc + vt) ** 2 / ((vc**2 + vt**2) / 99)
    assert row.alternative == "greater"
    assert row.null_lift == 0.0
    assert row.require_lift().alpha == 2e-20
    expected_lower = math.expm1(math.log(1.2) - t.isf(1e-20, df) * math.sqrt(vc + vt))
    assert row.require_lift().lb == pytest.approx(expected_lower, rel=1e-10)


def test_segment_mixed_hypotheses_share_one_bh_family_and_fcr_cap():
    from scipy.stats import t

    policies = {
        "rel": (0.05, "greater", -0.02, None, "primary"),
        "abs": (0.05, "less", 0.0, 0.5, "guardrail"),
        "zero": (0.05, "two-sided", 0.0, None, "secondary"),
    }
    means = {"rel": (10.3, 9.7), "abs": (9.5, 11.0), "zero": (11.0, 10.0)}
    rows = [
        _make_arm_row(200, mean, 1.0, country=segment, metric=metric, group_id=arm)
        for metric, treatment_means in means.items()
        for segment, mean_t in zip(("A", "B"), treatment_means, strict=True)
        for arm, mean in (("control", 10.0), ("treatment", mean_t))
    ]
    p_values = {}
    for metric, treatment_means in means.items():
        policy = policies[metric]
        for segment, mean_t in zip(("A", "B"), treatment_means, strict=True):
            vc, vt = 1.0 / 20000.0, 1.0 / (200 * mean_t**2)
            df = (vc + vt) ** 2 / ((vc**2 + vt**2) / 199)
            z = (math.log(mean_t / 10.0) - math.log1p(policy[2])) / math.sqrt(vc + vt)
            p_values[(metric, segment)] = (
                t.cdf((mean_t - 10.0 - policy[3]) / 0.1, 398)
                if policy[3] is not None
                else t.sf(z, df)
                if policy[1] == "greater"
                else 2 * t.sf(abs(z), df)
            )
    ordered = sorted(p_values, key=p_values.__getitem__)
    rank = max((i for i, key in enumerate(ordered, 1) if p_values[key] <= 0.04 * i / 6), default=0)
    selected = set(ordered[:rank])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IncrementWarning)
        result = _planned_breakout(
            rows,
            [_mean_metric(name) for name in policies],
            hypotheses=policies,
            correction="bh",
            q=0.04,
        )
    assert selected == {("rel", "A"), ("abs", "A"), ("zero", "A")}
    assert {(row.metric, row.dimension_value) for row in result if row.discovery} == selected
    for row in result:
        policy = policies[row.metric]
        assert (row.alternative, row.null_lift, row.null_abs) == (
            policy[1],
            policy[2],
            policy[3],
        )
        assert row.p_value() == pytest.approx(p_values[(row.metric, row.dimension_value)], rel=1e-9)
        assert row.family_size == 6
        assert row.role == "exploratory"
        if row.discovery:
            assert row.require_lift().alpha == pytest.approx(min(0.04 * rank / 6, 0.05))
            assert row.family_nominal_alpha == 0.05
    absolute = next(row for row in result if row.metric == "abs" and row.discovery)
    assert absolute.abs_ub == pytest.approx(-0.5 + t.isf(0.01, 398) * 0.1)
    assert absolute.abs_lb == pytest.approx(-0.5 - t.isf(0.01, 398) * 0.1)


@pytest.mark.parametrize("kind", ["cuped", "ratio"])
def test_shifted_segment_method_and_unavailable_rows_keep_complete_hypothesis(kind):

    metric = _mean_metric() if kind == "cuped" else _ratio_metric("rev")
    methods = [Method(name="cuped", variance_reduction="cuped")] if kind == "cuped" else None
    rows = [
        _make_arm_row(
            200 if segment == "A" else 1,
            mean,
            4.0,
            country=segment,
            group_id=arm,
            x_mean=1.0 if kind == "cuped" else None,
            sum_den=400.0 if kind == "ratio" else None,
        )
        for segment in ("A", "B")
        for arm, mean in (("control", 10.0), ("treatment", 10.0))
    ]
    policy = (0.05, "less", 0.05, None, "guardrail")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IncrementWarning)
        result = _planned_breakout(
            rows,
            [metric],
            hypotheses={"rev": policy},
            methods=methods,
            correction="bh",
            q=0.01,
        )
        (reference,) = estimate_lift(
            [metric],
            rows[:2],
            control_group="control",
            methods=methods,
            alpha=0.01,
            alternative="less",
            null_lift=0.05,
        ).results
    by_segment = {row.dimension_value: row for row in result}
    selected, unavailable = by_segment["A"], by_segment["B"]
    assert selected.discovery is True
    assert selected.family_size == 1
    assert selected.require_lift().ub == pytest.approx(reference.require_lift().ub)
    assert unavailable.excluded == "few_units"
    for row in result:
        assert (row.alternative, row.null_lift, row.null_abs, row.policy_name) == (
            "less",
            0.05,
            None,
            "compiled_plan",
        )


def test_conversion_segment_absolute_margin_uses_restricted_score_tail():
    from scipy.optimize import brentq
    from scipy.stats import norm

    from increment import (
        AnalysisPlan,
        ExperimentMetric,
        MetricSpec,
        MultiplicitySpec,
        View,
        from_frame,
        run,
    )

    for successes_t in (10, 20, 30):
        frame = pd.DataFrame(
            [
                {"unit": f"{arm}-{i}", "arm": arm, "country": "US", "conv": int(i < successes)}
                for arm, successes in (("control", 10), ("treatment", successes_t))
                for i in range(100)
            ]
        )
        experiment, reader = from_frame(
            frame,
            shape="summary",
            unit="unit",
            group="arm",
            control="control",
            metrics=[MetricSpec(name="conv", type="conversion", preferred_direction="decrease")],
            breakouts=["country"],
            plan=AnalysisPlan(
                primary=ExperimentMetric(metric="conv", margin_abs=0.1),
                view_multiplicity=MultiplicitySpec(correction="bh", q=0.1),
            ),
        )
        view = View(view="breakout", grain="total")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", IncrementWarning)
                (row,) = run(experiment.with_views(view), reader).readout(view)
        finally:
            reader.close()
        assert isinstance(row, BreakoutEstimate)

        def derivative(pc, successes_t=successes_t):
            if pc == 0.0:
                return math.inf
            if pc == 0.9:
                return -math.inf
            pt = pc + 0.1
            return 10 / pc - 90 / (1 - pc) + successes_t / pt - (100 - successes_t) / (1 - pt)

        pc = brentq(derivative, 0.0, 0.9)
        pt = pc + 0.1
        se = math.sqrt((pc * (1 - pc) + pt * (1 - pt)) / 100 * 200 / 199)
        expected = norm.cdf((successes_t / 100 - 0.1 - 0.1) / se)
        assert (row.alternative, row.null_abs, row.abs_reference_kind) == ("less", 0.1, "score")
        assert row.p_value() == pytest.approx(expected, rel=1e-8, abs=1e-13)


# CRITICAL: never call estimate_lift on a multi-segment summary at once.


class TestPreferredDirectionReachesTheLiftEstimate:
    def test_declared_preferred_direction_reaches_the_lift_estimate(self):
        """A metric declaring preferred_direction="decrease" must reach its
        LiftEstimate row, not just the plumbing that leaves it None -
        `prob_favorable()` raises without it."""
        from increment import readouts
        from increment.decision_wire import compiled_plan_to_json
        from increment.plan import compile_decision_plan
        from increment.semantics.design import Randomized
        from increment.sources import MomentsSource

        control = _make_arm_row(n=500, mean=10.0, var=4.0, country="US", group_id="control")
        treatment = _make_arm_row(n=500, mean=8.0, var=4.0, country="US", group_id="treatment")
        declared = MeanMetric(name="rev", entity="user", fact="rev", preferred_direction="decrease")
        undeclared = _mean_metric("orders")
        control2 = _make_arm_row(
            n=500, mean=10.0, var=4.0, country="US", metric="orders", group_id="control"
        )
        treatment2 = _make_arm_row(
            n=500, mean=8.0, var=4.0, country="US", metric="orders", group_id="treatment"
        )
        # A moments cube needs its format stamp and the plan it was reduced under.
        stamp = {
            "moments_format": 11,
            "winsor_lower_percentile": None,
            "winsor_upper_percentile": None,
            "winsor_lower_bound": None,
            "winsor_upper_bound": None,
            "winsor_n": None,
            "winsor_n_lower": None,
            "winsor_n_upper": None,
            "decision_plan": compiled_plan_to_json(
                compile_decision_plan(None, [declared, undeclared])
            ),
        }
        rows = [{**row, **stamp} for row in (control, treatment, control2, treatment2)]

        estimates = readouts.run(
            MomentsSource(
                rows,
                metrics=[declared, undeclared],
                study_id="exp1",
                design=Randomized(control_group="control"),
            )
        )
        by_metric = {e.metric: e for e in estimates}
        assert by_metric["rev"].preferred_direction == "decrease"
        assert by_metric["orders"].preferred_direction is None


class TestRunBreakoutSegmentIsolation:
    def test_two_segments_with_different_control_means_both_correct(self):
        """US: control=10, treatment=12 -> positive lift. CA: control=20, treatment=19 -> negative
        lift. A `{metric: control}` dict-key collision (one estimate_lift call across both)
        would keep only ONE segment's control arm, producing equal or mismatched results."""
        us_control = _make_arm_row(n=2000, mean=10.0, var=4.0, country="US", group_id="control")
        us_treatment = _make_arm_row(n=2000, mean=12.0, var=4.0, country="US", group_id="treatment")
        ca_control = _make_arm_row(n=1500, mean=20.0, var=4.0, country="CA", group_id="control")
        ca_treatment = _make_arm_row(n=1500, mean=19.0, var=4.0, country="CA", group_id="treatment")
        summary = pd.DataFrame([us_control, us_treatment, ca_control, ca_treatment])

        results = run_breakout(
            summary=summary,
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
        )

        assert len(results) == 2
        for r in results:
            assert isinstance(r, BreakoutEstimate)
            assert r.dimension == "country"
            assert r.metric == "rev"
            assert r.group_id == "treatment"
            assert r.policy_name == "default_exploratory"

        by_segment = {r.dimension_value: r for r in results}
        assert set(by_segment) == {"US", "CA"}

        # Ground truth: estimate_lift called directly on JUST that
        # segment's own two rows - what run_breakout MUST reproduce.
        us_expected = estimate_lift(
            metrics=[_mean_metric()],
            summary=pd.DataFrame([us_control, us_treatment]),
            control_group="control",
        ).results[0]
        ca_expected = estimate_lift(
            metrics=[_mean_metric()],
            summary=pd.DataFrame([ca_control, ca_treatment]),
            control_group="control",
        ).results[0]

        us_lift = by_segment["US"].lift
        ca_lift = by_segment["CA"].lift
        us_expected_lift = us_expected.lift
        ca_expected_lift = ca_expected.lift
        assert us_lift is not None and us_expected_lift is not None
        assert ca_lift is not None and ca_expected_lift is not None
        assert us_lift.value == us_expected_lift.value
        assert us_lift.lb == us_expected_lift.lb
        assert us_lift.ub == us_expected_lift.ub
        assert ca_lift.value == ca_expected_lift.value
        assert ca_lift.lb == ca_expected_lift.lb
        assert ca_lift.ub == ca_expected_lift.ub

        # They really do differ - opposite-signed lift, not a shared
        # value a buggy single-call implementation would inherit.
        assert us_lift.value > 0, "US treatment (12) > control (10)"
        assert ca_lift.value < 0, "CA treatment (19) < control (20)"

    def test_single_segment_verdict_matches_the_whole_window_row(self):
        """A segment holding every unit is the whole-window contrast: the cell's own
        ``p_value()``/``stat_sig()`` read the same persisted evidence ``estimate_lift``'s
        row reads, rather than a readout-side reconstruction."""
        control = _make_arm_row(n=2000, mean=10.0, var=4.0, country="ALL", group_id="control")
        treatment = _make_arm_row(n=2000, mean=10.3, var=4.0, country="ALL", group_id="treatment")
        summary = pd.DataFrame([control, treatment])
        (cell,) = run_breakout(
            summary=summary, metrics=[_mean_metric()], control_group="control", dimension="country"
        )
        (whole,) = estimate_lift(
            metrics=[_mean_metric()], summary=summary, control_group="control"
        ).results
        whole_p = whole.p_value()
        cell_p = cell.p_value()
        assert whole_p is not None and cell_p is not None
        assert cell_p == pytest.approx(whole_p, rel=1e-12, abs=0.0)
        assert cell.stat_sig() is whole.stat_sig()
        for null_lift in (0.01, 0.05):
            shifted_cell = cell.model_copy(update={"null_lift": null_lift})
            shifted_whole = whole.model_copy(update={"null_lift": null_lift})
            assert shifted_cell.p_value() == pytest.approx(
                shifted_whole.p_value(), rel=1e-12, abs=0.0
            )
            assert shifted_cell.stat_sig() is shifted_whole.stat_sig()

    def test_set_only_binomial_cell_decides_from_its_counts(self):
        """A zero-control-count conversion cell has no finite point; its verdict and
        p-value come from the exact set's persisted counts, as the whole-window row's do."""
        from tests.test_tables import _binomial_lift_estimate

        whole = _binomial_lift_estimate(x_c=0, n_c=200, x_t=100, n_t=200)
        assert whole.lift is None and whole.binomial_set is not None
        cell = BreakoutEstimate(
            metric=whole.metric,
            group_id=whole.group_id,
            method=whole.method,
            method_role=whole.method_role,
            dimension="country",
            dimension_value="US",
            reference_kind="binomial",
            lift=None,
            binomial_set=whole.binomial_set,
        )
        assert cell.stat_sig() is True
        assert cell.stat_sig() is whole.stat_sig()
        assert cell.p_value() == whole.p_value()
        assert cell.p_value() == whole.binomial_set.null_p_value(0.0, "two-sided")
        thin = _binomial_lift_estimate(x_c=0, n_c=5, x_t=1, n_t=5)
        thin_cell = cell.model_copy(update={"binomial_set": thin.binomial_set})
        assert thin_cell.stat_sig() is False
        assert thin_cell.p_value() == thin.p_value()

    def test_excluded_and_low_reliability_cells(self):
        """An excluded cell has no evidence to decide on; a low-reliability cell decides from its
        own interval, so its verdict agrees with its p-value."""
        excluded = BreakoutEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            dimension="country",
            dimension_value="US",
            lift=None,
            excluded="few_units",
        )
        assert excluded.stat_sig() is None
        assert excluded.p_value() is None
        control = _make_arm_row(n=20, mean=10.0, var=1.0, country="US", group_id="control")
        treatment = _make_arm_row(n=20, mean=12.0, var=1.0, country="US", group_id="treatment")
        (thin,) = run_breakout(
            summary=pd.DataFrame([control, treatment]),
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
            reliability_floor=50,
        )
        assert thin.low_reliability is True
        assert thin.stat_sig() is True
        thin_p = thin.p_value()
        assert thin_p is not None and thin_p < 0.05

    def test_daily_cell_with_a_margin_matches_the_whole_window_row(self):
        from tests.test_tables import _binomial_lift_estimate

        whole = _binomial_lift_estimate(x_c=20, n_c=200, x_t=60, n_t=200, null_abs=0.2)
        daily = DailyLiftEstimate(
            metric=whole.metric,
            group_id=whole.group_id,
            method=whole.method,
            method_role=whole.method_role,
            ds=date(2024, 1, 1),
            lift=whole.lift,
            binomial_set=whole.binomial_set,
            null_abs=whole.null_abs,
            abs_lb=whole.abs_lb,
            abs_ub=whole.abs_ub,
            abs_diff=whole.abs_diff,
            abs_se=whole.abs_se,
            alternative=whole.alternative,
            null_lift=whole.null_lift,
            reference_kind=whole.reference_kind,
            sampling_available=True,
        )
        assert daily.stat_sig() is whole.stat_sig() is False
        assert daily.p_value() == whole.p_value()
        assert daily.model_copy(update={"sampling_available": False}).stat_sig() is None

    def test_daily_nonpositive_cell_with_a_relative_margin_keeps_its_declared_null(self):
        """A daily CUPED cell whose adjusted means are nonpositive, carrying a declared
        relative margin and no absolute margin, never substitutes an additive-zero test
        for the hypothesis it retains."""
        from increment.estimation.results import row_decision_scale

        base: dict[str, Any] = {
            "metric": "rev",
            "group_id": "treatment",
            "method": "cuped",
            "method_role": "decision",
            "ds": date(2024, 1, 1),
            "lift": None,
            "relative_unavailable_reason": "nonpositive_arm_mean",
            "abs_diff": -0.9,
            "abs_se": 0.05,
            "abs_lb": -1.0,
            "abs_ub": -0.8,
            "abs_reference_kind": "t",
            "abs_reference_df": 1998.0,
            "abs_alpha": 0.1,
            "reference_kind": "normal",
            "alternative": "less",
            "sampling_available": True,
        }
        margin = DailyLiftEstimate(**base, null_lift=-0.05)
        assert margin.p_value() == 1.0
        assert margin.stat_sig() is False
        assert row_decision_scale(margin) == "relative"
        unshifted = DailyLiftEstimate(**base, null_lift=0.0)
        assert unshifted.stat_sig() is True
        assert row_decision_scale(unshifted) == "absolute"

    def test_absolute_scale_cell_with_a_nonzero_null_matches_the_whole_window_row(self):
        """An additive cell's Wald statistics are linear, so a shifted null is not
        log-transformed: the cell's p-value equals the ``LiftEstimate``'s and the
        hand Normal tail on the additive scale."""
        from scipy.stats import norm

        from increment.estimation.armstats import ScoreStats
        from increment.estimation.inference import infer_ate

        whole = infer_ate(
            metric="m",
            group_id="t",
            method="unadjusted",
            method_role="decision",
            alpha=0.05,
            alternative="two-sided",
            point=2.0,
            scores=ScoreStats(metric="m", contrast="t", n=100, sum_psi=0.0, sum_psi2=100.0),
            value_scale="absolute",
        ).model_copy(update={"null_lift": 1.5})
        assert whole.scale == "linear" and whole.value_scale == "absolute"
        cell = BreakoutEstimate(
            metric=whole.metric,
            group_id=whole.group_id,
            method=whole.method,
            method_role=whole.method_role,
            dimension="country",
            dimension_value="US",
            lift=whole.lift,
            value_scale="absolute",
            null_lift=whole.null_lift,
            reference_kind=cast("Any", whole.reference_kind),
            reference_df=whole.reference_df,
        )
        lift = whole.require_lift()
        assert lift.log_mean is not None and lift.log_se is not None
        expected = 2.0 * norm.sf(abs((lift.log_mean - 1.5) / lift.log_se))
        whole_p, cell_p = whole.p_value(), cell.p_value()
        assert whole_p is not None and cell_p is not None
        assert whole_p == pytest.approx(expected, rel=1e-12, abs=0.0)
        assert cell_p == pytest.approx(expected, rel=1e-12, abs=0.0)
        assert cell.stat_sig() is whole.stat_sig()

    def test_segment_row_order_does_not_change_either_answer(self):
        """Row arrival order (CA-then-US vs US-then-CA) must not change either segment's answer - a
        `{metric: control}` dict collision would make whichever segment is iterated LAST win."""
        us_control = _make_arm_row(n=2000, mean=10.0, var=4.0, country="US", group_id="control")
        us_treatment = _make_arm_row(n=2000, mean=12.0, var=4.0, country="US", group_id="treatment")
        ca_control = _make_arm_row(n=1500, mean=20.0, var=4.0, country="CA", group_id="control")
        ca_treatment = _make_arm_row(n=1500, mean=19.0, var=4.0, country="CA", group_id="treatment")

        forward = run_breakout(
            summary=pd.DataFrame([us_control, us_treatment, ca_control, ca_treatment]),
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
        )
        backward = run_breakout(
            summary=pd.DataFrame([ca_control, ca_treatment, us_control, us_treatment]),
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
        )

        forward_by_segment = {}
        for r in forward:
            lift = r.lift
            assert lift is not None
            forward_by_segment[r.dimension_value] = lift.value
        backward_by_segment = {}
        for r in backward:
            lift = r.lift
            assert lift is not None
            backward_by_segment[r.dimension_value] = lift.value
        assert forward_by_segment == backward_by_segment

    def test_segment_output_order_is_sorted_and_stable_across_calls(self):
        """`run_breakout` partitions rows into a dict keyed by dimension value; iteration order
        used to be raw `rows` insertion order (unordered from a DuckDB/arrow fetch) - must now
        be sorted and stable."""
        us_control = _make_arm_row(n=2000, mean=10.0, var=4.0, country="US", group_id="control")
        us_treatment = _make_arm_row(n=2000, mean=12.0, var=4.0, country="US", group_id="treatment")
        ca_control = _make_arm_row(n=1500, mean=20.0, var=4.0, country="CA", group_id="control")
        ca_treatment = _make_arm_row(n=1500, mean=19.0, var=4.0, country="CA", group_id="treatment")
        mx_control = _make_arm_row(n=1200, mean=8.0, var=4.0, country="MX", group_id="control")
        mx_treatment = _make_arm_row(n=1200, mean=9.0, var=4.0, country="MX", group_id="treatment")

        # Deliberately reverse-alphabetical row arrival order.
        shuffled_rows = [
            mx_control,
            mx_treatment,
            us_control,
            us_treatment,
            ca_control,
            ca_treatment,
        ]

        first_call = run_breakout(
            summary=pd.DataFrame(shuffled_rows),
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
        )
        # A *different* permutation, so the cross-call assertion below
        # exercises order stability rather than an identical rerun.
        second_call = run_breakout(
            summary=pd.DataFrame(list(reversed(shuffled_rows))),
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
        )

        first_order = [r.dimension_value for r in first_call]
        second_order = [r.dimension_value for r in second_call]

        assert first_order == sorted(first_order)
        assert first_order == second_order


# Shape / count acceptance.


class TestRunBreakoutShape:
    def test_two_segments_return_2x_single_segment_estimate_count(self):
        """2 segments -> 2x the estimates of an unsegmented estimate_lift
        call on one segment's worth of the same-shaped data."""
        us_control = _make_arm_row(n=500, mean=10.0, var=4.0, country="US", group_id="control")
        us_treatment = _make_arm_row(n=500, mean=11.0, var=4.0, country="US", group_id="treatment")
        ca_control = _make_arm_row(n=500, mean=10.0, var=4.0, country="CA", group_id="control")
        ca_treatment = _make_arm_row(n=500, mean=11.0, var=4.0, country="CA", group_id="treatment")
        summary = pd.DataFrame([us_control, us_treatment, ca_control, ca_treatment])

        result = run_breakout(
            summary=summary,
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
        )
        single_segment = estimate_lift(
            metrics=[_mean_metric()],
            summary=pd.DataFrame([us_control, us_treatment]),
            control_group="control",
        ).results

        assert len(result) == 2 * len(single_segment)

    def test_methods_axis_multiplies_across_segments(self):
        """methods=[...] forwards through to estimate_lift per segment:
        2 segments x 2 methods -> 4 results."""
        us_control = _make_arm_row(n=500, mean=10.0, var=4.0, country="US", group_id="control")
        us_treatment = _make_arm_row(n=500, mean=11.0, var=4.0, country="US", group_id="treatment")
        ca_control = _make_arm_row(n=500, mean=10.0, var=4.0, country="CA", group_id="control")
        ca_treatment = _make_arm_row(n=500, mean=11.0, var=4.0, country="CA", group_id="treatment")
        summary = pd.DataFrame([us_control, us_treatment, ca_control, ca_treatment])

        result = run_breakout(
            summary=summary,
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
            methods=[Method(name="unadjusted"), Method(name="m2")],
        )

        assert len(result) == 4
        assert {r.method for r in result} == {"unadjusted", "m2"}
        assert {r.dimension_value for r in result} == {"US", "CA"}

    def test_duplicate_method_names_refuse_before_breakout_partition(self):
        class ExplodingSummary:
            def __iter__(self):
                raise AssertionError("breakout summary was consumed before validation")

        with pytest.raises(InvalidRequestError) as exc_info:
            run_breakout(
                summary=ExplodingSummary(),
                metrics=[_mean_metric()],
                control_group="control",
                dimension="country",
                methods=[
                    Method(name="same"),
                    Method(name="same", variance_reduction="cuped"),
                ],
            )
        assert exc_info.value.code == "estimation.engine.method_names_unique"
        assert exc_info.value.context["duplicates"] == ("same",)

    def test_source_defaults_to_none_and_is_stamped_when_given(self):
        """`source` is `None` unless the caller passes one, then it's stamped onto every result -
        lets a breakout view disambiguate breakouts sharing a dimension but different
        sources."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 11.0, 4.0, country="US", group_id="treatment"),
        ]

        no_source = run_breakout(
            rows, [_mean_metric()], control_group="control", dimension="country"
        )
        assert all(r.source is None for r in no_source)

        with_source = run_breakout(
            rows,
            [_mean_metric()],
            control_group="control",
            dimension="country",
            source="events",
        )
        assert all(r.source == "events" for r in with_source)


# Confidence intervals reflect per-segment sample size.


class TestRunBreakoutConfidenceIntervals:
    def test_ci_present_and_narrower_for_larger_segment(self):
        """Same underlying lift/variance, different n - the larger
        segment's CI must be narrower, reflecting its bigger sample."""
        big_control = _make_arm_row(n=20000, mean=10.0, var=4.0, country="BIG", group_id="control")
        big_treatment = _make_arm_row(
            n=20000, mean=11.0, var=4.0, country="BIG", group_id="treatment"
        )
        small_control = _make_arm_row(
            n=200, mean=10.0, var=4.0, country="SMALL", group_id="control"
        )
        small_treatment = _make_arm_row(
            n=200, mean=11.0, var=4.0, country="SMALL", group_id="treatment"
        )
        summary = pd.DataFrame([big_control, big_treatment, small_control, small_treatment])

        result = run_breakout(
            summary=summary,
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
        )
        by_segment = {r.dimension_value: r for r in result}
        big = by_segment["BIG"]
        small = by_segment["SMALL"]

        for r in result:
            lift = r.lift
            assert lift is not None
            assert lift.lb is not None
            assert lift.ub is not None
            assert lift.lb < lift.value < lift.ub

        big_lift = big.lift
        small_lift = small.lift
        assert big_lift is not None and small_lift is not None
        assert big_lift.lb is not None
        assert big_lift.ub is not None
        assert small_lift.lb is not None
        assert small_lift.ub is not None
        big_width = big_lift.ub - big_lift.lb
        small_width = small_lift.ub - small_lift.lb
        assert big_width < small_width


# Bonferroni multiplicity correction across segments.


def _four_segment_summary() -> pd.DataFrame:
    """4 countries, each with a control + treatment row for one metric."""
    rows = []
    for i, country in enumerate(["US", "CA", "GB", "DE"]):
        rows.append(
            _make_arm_row(n=500, mean=10.0 + i, var=4.0, country=country, group_id="control")
        )
        rows.append(
            _make_arm_row(n=500, mean=11.0 + i, var=4.0, country=country, group_id="treatment")
        )
    return pd.DataFrame(rows)


class TestRunBreakoutBonferroniCorrection:
    def test_default_correction_is_none_and_level_unchanged(self):
        """No `correction=` passed - level stays exactly 1 - alpha (0.95),
        zero behavior change for any existing caller."""
        result = run_breakout(
            summary=_four_segment_summary(),
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
        )
        assert len(result) == 4
        for r in result:
            lift = r.lift
            assert lift is not None
            assert lift.level == pytest.approx(0.95)

    def test_bonferroni_divides_alpha_by_segment_count(self):
        """4 distinct segments -> alpha/4 per segment -> level ==
        1 - 0.05/4, every interval strictly wider than uncorrected."""
        summary = _four_segment_summary()

        corrected = run_breakout(
            summary=summary,
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
            correction="bonferroni",
        )
        uncorrected = run_breakout(
            summary=summary,
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
            correction="none",
        )

        assert len(corrected) == 4
        for r in corrected:
            lift = r.lift
            assert lift is not None
            assert lift.level == pytest.approx(1 - 0.05 / 4)

        by_segment_corrected = {r.dimension_value: r for r in corrected}
        by_segment_uncorrected = {r.dimension_value: r for r in uncorrected}
        for value in by_segment_corrected:
            c = by_segment_corrected[value]
            u = by_segment_uncorrected[value]
            c_lift = c.lift
            u_lift = u.lift
            assert c_lift is not None and u_lift is not None
            assert c_lift.lb is not None and c_lift.ub is not None
            assert u_lift.lb is not None and u_lift.ub is not None
            assert (c_lift.ub - c_lift.lb) > (u_lift.ub - u_lift.lb)

    def test_single_segment_bonferroni_equals_none(self):
        """K == 1 - Bonferroni correction is a no-op, same level and
        same interval as `correction="none"`."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 11.0, 4.0, country="US", group_id="treatment"),
        ]

        corrected = run_breakout(
            rows,
            [_mean_metric()],
            control_group="control",
            dimension="country",
            correction="bonferroni",
        )
        uncorrected = run_breakout(
            rows,
            [_mean_metric()],
            control_group="control",
            dimension="country",
            correction="none",
        )

        assert len(corrected) == len(uncorrected) == 1
        corrected_lift = corrected[0].lift
        uncorrected_lift = uncorrected[0].lift
        assert corrected_lift is not None and uncorrected_lift is not None
        assert corrected_lift.level == pytest.approx(0.95)
        assert corrected_lift.level == pytest.approx(uncorrected_lift.level)
        assert corrected_lift.lb == pytest.approx(uncorrected_lift.lb)
        assert corrected_lift.ub == pytest.approx(uncorrected_lift.ub)

    def test_empty_summary_bonferroni_does_not_divide_by_zero(self):
        """K == 0 (empty summary) must not crash: `correction="bonferroni"` degrades to the no-op
        `alpha_seg == alpha` instead of raising `ZeroDivisionError` on `alpha / 0`."""
        result = run_breakout(
            [],
            [_mean_metric()],
            control_group="control",
            dimension="country",
            correction="bonferroni",
        )
        assert isinstance(result, BreakoutEstimates)
        assert len(result.to_frame()) == 0

    def test_bonferroni_k_counts_count_estimable_cells(self):
        """Bonferroni divides alpha, per metric, by the (treatment arm, segment) cells its
        counts can estimate: a one-unit segment T is an unavailable row outside the family,
        and m2, one unit per arm in C too, is divided by one fewer."""
        rows = [
            _make_arm_row(
                n=1 if segment == "T" or (segment, metric) == ("C", "m2") else 200,
                mean=10.0 if arm == "control" else 11.0,
                var=4.0,
                country=segment,
                metric=metric,
                group_id=arm,
            )
            for segment in ("A", "B", "C", "T")
            for metric in ("m1", "m2")
            for arm in ("control", "treatment")
        ]
        with pytest.warns(IncrementWarning):
            result = run_breakout(
                summary=rows,
                metrics=[_mean_metric("m1"), _mean_metric("m2")],
                control_group="control",
                dimension="country",
                correction="bonferroni",
            )

        cells = {(r.metric, r.dimension_value): r for r in result}
        excluded = {key for key, r in cells.items() if r.excluded is not None}
        assert excluded == {("m1", "T"), ("m2", "T"), ("m2", "C")}
        assert all(cells[key].excluded == "few_units" for key in excluded)
        levels = {key: r.require_lift().level for key, r in cells.items() if key not in excluded}
        assert levels == {
            ("m1", "A"): pytest.approx(1 - 0.05 / 3),
            ("m1", "B"): pytest.approx(1 - 0.05 / 3),
            ("m1", "C"): pytest.approx(1 - 0.05 / 3),
            ("m2", "A"): pytest.approx(1 - 0.05 / 2),
            ("m2", "B"): pytest.approx(1 - 0.05 / 2),
        }
        members = {key for key, r in cells.items() if r.multiplicity_status == "exploratory_family"}
        assert members == set(levels)

    def test_a_sensitivity_method_leaves_the_decision_k_unchanged(self):
        """k follows the decision method's count rule: a scalar-CUPED conversion decision
        excludes the one-unit cell (B, t2), so its three estimable cells are read at
        1 - 0.05/3, with the same intervals, whether or not an unadjusted sensitivity
        method that admits that cell is also requested."""
        rows = []
        for segment in ("A", "B"):
            for arm in ("control", "t1", "t2"):
                n = 1 if (segment, arm) == ("B", "t2") else 1000
                row = _conversion_arm_row(
                    n, max(n // 10, 1), country=segment, group_id=arm, with_covariate=True
                )
                row["cx2"] = float(n)
                rows.append(row)
        cuped = Method(name="cuped", variance_reduction="cuped")
        decisions = []
        for methods in ([cuped], [cuped, Method(name="unadjusted")]):
            with pytest.warns(IncrementWarning):
                result = run_breakout(
                    rows,
                    [_conversion_metric()],
                    control_group="control",
                    dimension="country",
                    correction="bonferroni",
                    methods=methods,
                    method_roles={
                        m.name: "decision" if m is cuped else "sensitivity" for m in methods
                    },
                )
            decisions.append(
                {(r.dimension_value, r.group_id): r for r in result if r.method == "cuped"}
            )
        alone, beside = decisions
        assert alone.pop(("B", "t2")).excluded == beside.pop(("B", "t2")).excluded == "few_units"
        assert {key: r.require_lift().level for key, r in alone.items()} == dict.fromkeys(
            [("A", "t1"), ("A", "t2"), ("B", "t1")], pytest.approx(1 - 0.05 / 3)
        )
        assert {key: (r.require_lift().lb, r.require_lift().ub) for key, r in beside.items()} == {
            key: pytest.approx((r.require_lift().lb, r.require_lift().ub))
            for key, r in alone.items()
        }

    def test_invalid_correction_value_rejected(self):
        """A `correction` value outside {"none", "bonferroni"} raises
        ValueError rather than silently being accepted."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 11.0, 4.0, country="US", group_id="treatment"),
        ]

        with pytest.raises(InvalidRequestError) as exc_info:
            run_breakout(
                rows,
                [_mean_metric()],
                control_group="control",
                dimension="country",
                correction="holm",  # ty: ignore[invalid-argument-type]
            )
        assert exc_info.value.code == "breakout.run_breakout_correction"
        assert exc_info.value.context["correction"] == "holm"


class TestRunBreakoutRoleMetadata:
    def test_role_stamped_exploratory_regardless_of_correction(self):
        """Every run_breakout row is exploratory outside the BH path."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 11.0, 4.0, country="US", group_id="treatment"),
        ]
        for correction in ("none", "bonferroni"):
            result = run_breakout(
                rows,
                [_mean_metric()],
                control_group="control",
                dimension="country",
                correction=correction,
            )
            assert result, correction
            assert all(r.role == "exploratory" for r in result)
            assert all(r.discovery is None for r in result)


# BH multiplicity correction: one flat family across every (metric, arm,
# segment) cell this call produces.


def _bh_family_summary() -> pd.DataFrame:
    """3 metrics x 2 countries x (control, treatment): m_a is a strong lift
    in both countries, m_b is null in both, m_c is strong in US only -- a
    flat 6-cell family (metric x arm x segment) with a clean 3-selected /
    3-not-selected split, engineered so the SAME metric (m_c) has a
    different verdict per segment (proving the family spans segments, not
    just metrics)."""
    specs = [
        ("m_a", "US", 10.0, 12.0),
        ("m_a", "CA", 10.0, 12.0),
        ("m_b", "US", 10.0, 10.05),
        ("m_b", "CA", 10.0, 10.05),
        ("m_c", "US", 10.0, 10.9),
        ("m_c", "CA", 10.0, 10.05),
    ]
    rows = []
    for metric, country, control_mean, treatment_mean in specs:
        rows.append(
            _make_arm_row(
                n=500,
                mean=control_mean,
                var=4.0,
                country=country,
                metric=metric,
                group_id="control",
            )
        )
        rows.append(
            _make_arm_row(
                n=500,
                mean=treatment_mean,
                var=4.0,
                country=country,
                metric=metric,
                group_id="treatment",
            )
        )
    return pd.DataFrame(rows)


def _p_value_from_estimate(lift: Estimate) -> float:
    """An independent two-sided z-test from an Estimate's log-scale sufficient
    statistics: the hand oracle the BH selection below is checked against,
    deliberately not the row's own ``p_value()``."""
    from scipy.stats import norm as _norm_dist

    z = cast(float, lift.log_mean) / cast(float, lift.log_se)
    return float(2.0 * _norm_dist.sf(abs(z)))


class TestRunBreakoutBHCorrection:
    """``correction="bh"`` consumes prior-free sampling evidence and keeps
    declared members in the same family when a posterior is also requested."""

    def test_bh_keeps_prior_bound_members_in_sampling_family(self):
        summary = _bh_family_summary()
        metrics = [_mean_metric(name) for name in ("m_a", "m_b", "m_c")]
        baseline = run_breakout(
            summary,
            metrics,
            control_group="control",
            dimension="country",
            correction="bh",
        )
        posterior = run_breakout(
            summary,
            metrics,
            control_group="control",
            dimension="country",
            prior=Normal(mu=0.0, sigma=0.1),
            correction="bh",
        )
        plain_rows = [row for row in baseline if row.excluded is None]
        prior_rows = [row for row in posterior if row.excluded is None]
        assert len(prior_rows) == len(plain_rows) == 6
        for plain, prior_row in zip(plain_rows, prior_rows, strict=True):
            assert prior_row.lift == plain.lift
            assert (prior_row.reference_kind, prior_row.reference_df) == (
                plain.reference_kind,
                plain.reference_df,
            )
            assert (prior_row.family_size, prior_row.discovery) == (
                plain.family_size,
                plain.discovery,
            )
            assert prior_row.lift is not None
            assert plain.lift is not None
            assert prior_row.p_value() == pytest.approx(plain.p_value())
            assert prior_row.posterior_available is True
            family_view = prior_row._family_view()
            assert family_view is not None
            assert family_view.sampling_available is True
            assert family_view.posterior_available is True
            assert family_view.posterior_estimate == prior_row.posterior_estimate

    def test_bh_discovery_matches_hand_bh_across_metrics_and_segments(self):
        summary = _bh_family_summary()
        metrics = [_mean_metric("m_a"), _mean_metric("m_b"), _mean_metric("m_c")]
        q = 0.05

        result = run_breakout(
            summary=summary,
            metrics=metrics,
            control_group="control",
            dimension="country",
            correction="bh",
            q=q,
        )
        real = [r for r in result if r.excluded is None]
        assert len(real) == 6

        p_values = []
        for r in real:
            lift = r.lift
            assert lift is not None
            p_values.append(_p_value_from_estimate(lift))
        selected_idx, expected_selected_alpha = bh_select(p_values, q)
        selected_set = set(selected_idx)
        assert [r.discovery for r in real] == [i in selected_set for i in range(len(real))]

        selected_cells = {(real[i].metric, real[i].dimension_value) for i in selected_set}
        assert selected_cells == {("m_a", "US"), ("m_a", "CA"), ("m_c", "US")}

        for i, r in enumerate(real):
            assert r.role == "exploratory"
            lift = r.lift
            assert lift is not None
            if i in selected_set:
                assert lift.alpha == expected_selected_alpha
            else:
                assert lift.alpha == pytest.approx(0.05)  # nominal, default alpha=0.05

    def test_bh_family_leaves_count_excluded_segments_out(self):
        """A one-unit segment and a segment without a treatment row are unavailable rows
        outside the family: BH runs over the three estimable segments, so m = 3."""
        rows = []
        for country, treatment_mean in (("A", 10.45), ("B", 10.17), ("C", 10.0)):
            rows.append(_make_arm_row(1000, 10.0, 4.0, country=country, group_id="control"))
            rows.append(
                _make_arm_row(1000, treatment_mean, 4.0, country=country, group_id="treatment")
            )
        rows.append(_make_arm_row(1, 10.0, 0.0, country="T", group_id="control"))
        rows.append(_make_arm_row(1, 11.0, 0.0, country="T", group_id="treatment"))
        rows.append(_make_arm_row(50, 10.0, 4.0, country="O", group_id="control"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IncrementWarning)
            result = run_breakout(
                rows,
                [_mean_metric()],
                control_group="control",
                dimension="country",
                correction="bh",
            )

        by_segment = {row.dimension_value: row for row in result}
        assert (by_segment["T"].excluded, by_segment["O"].excluded) == (
            "few_units",
            "no_treatment_arm",
        )
        for segment in ("T", "O"):
            row = by_segment[segment]
            assert (row.lift, row.discovery, row.family_size) == (None, None, None)
            assert row.multiplicity_status == "exploratory_unadjusted"
        family = [by_segment[segment] for segment in ("A", "B", "C")]
        assert [row.family_size for row in family] == [3, 3, 3]
        # p = 5e-7, 0.057, 1: p(2) <= 2(0.10)/3, while under m = 5 it exceeds 2(0.10)/5.
        assert [row.discovery for row in family] == [True, True, False]
        assert all(row.family_threshold == pytest.approx(2 * 0.10 / 3) for row in family)
        assert all(row.require_lift().alpha == pytest.approx(0.05) for row in family[:2])

    def test_bh_family_counts_a_control_lost_to_a_nonpositive_mean(self):
        rows = [
            _make_arm_row(100, 10.0, 4.0, country="A", group_id="control"),
            _make_arm_row(100, 11.0, 4.0, country="A", group_id="treatment"),
            _make_arm_row(100, -1.0, 4.0, country="N", group_id="control"),
            _make_arm_row(100, 1.0, 4.0, country="N", group_id="treatment"),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IncrementWarning)
            result = run_breakout(
                rows,
                [_mean_metric()],
                control_group="control",
                dimension="country",
                correction="bh",
            )

        by_segment = {row.dimension_value: row for row in result}
        assert by_segment["N"].excluded == "nonpositive_mean"
        assert by_segment["N"].discovery is False
        assert by_segment["A"].family_size == by_segment["N"].family_size == 2

    def test_bh_family_count_exclusion_wins_over_an_outcome_reason(self):
        """A segment with no control row is excluded by count even when its treatment mean is
        non-positive, so the family is the one without that segment."""
        rows = [
            _make_arm_row(100, 10.0, 4.0, country="A", group_id="control"),
            _make_arm_row(100, 11.0, 4.0, country="A", group_id="treatment"),
            _make_arm_row(100, 10.0, 4.0, country="B", group_id="control"),
            _make_arm_row(100, 10.2, 4.0, country="B", group_id="treatment"),
        ]
        overlap = [*rows, _make_arm_row(100, -1.0, 4.0, country="X", group_id="treatment")]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IncrementWarning)
            with_x, without_x = [
                run_breakout(
                    summary,
                    [_mean_metric()],
                    control_group="control",
                    dimension="country",
                    correction="bh",
                )
                for summary in (overlap, rows)
            ]

        excluded = next(row for row in with_x if row.dimension_value == "X")
        assert (excluded.excluded, excluded.family_size, excluded.discovery) == (
            "no_control_arm",
            None,
            None,
        )
        assert excluded.multiplicity_status == "exploratory_unadjusted"
        family = [row for row in with_x if row.dimension_value != "X"]
        assert [row.family_size for row in family] == [2, 2]
        assert [(row.family_threshold, row.discovery) for row in family] == [
            (row.family_threshold, row.discovery) for row in without_x
        ]

    def test_bh_family_never_selects_an_outcome_exclusion_at_q_one(self):
        """At q = 1 every p-value passes the k = m step; a control lost to a non-positive mean
        still counts in m but is never selected, by the view or by select_exploratory_family."""
        rows = [
            _make_arm_row(100, 10.0, 4.0, country="A", group_id="control"),
            _make_arm_row(100, 11.0, 4.0, country="A", group_id="treatment"),
            _make_arm_row(100, -1.0, 4.0, country="N", group_id="control"),
            _make_arm_row(100, 1.0, 4.0, country="N", group_id="treatment"),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IncrementWarning)
            view, nominal = [
                run_breakout(
                    rows,
                    [_mean_metric()],
                    control_group="control",
                    dimension="country",
                    correction=correction,
                    q=1.0,
                )
                for correction in ("bh", "none")
            ]
        exploratory = cast(
            "list[BreakoutEstimate]", list(select_exploratory_family(list(nominal), q=1.0))
        )

        for result in (view, exploratory):
            by_segment = {row.dimension_value: row for row in result}
            assert (by_segment["N"].excluded, by_segment["N"].lift) == ("nonpositive_mean", None)
            assert by_segment["N"].discovery is False
            assert by_segment["A"].discovery is True
            assert by_segment["A"].family_size == by_segment["N"].family_size == 2

    # The covariate means differ by arm by construction, so the CUPED cells are flagged.
    @allow_warning_codes("estimation.cuped.covariate_imbalance")
    def test_bh_family_dedups_by_method(self):
        """2 methods x 1 metric x 2 segments: family size must count
        (metric, arm, segment) cells (2), not (metric, method, arm,
        segment) rows (4). The canonical unadjusted US row is significant,
        while contradictory CUPED evidence must remain uncredited."""
        rows = []
        for country, control_mean, treatment_mean in (("US", 10.0, 12.0), ("CA", 10.0, 10.05)):
            rows.append(
                _make_arm_row(
                    500,
                    control_mean,
                    4.0,
                    country=country,
                    group_id="control",
                    x_mean=0.0,
                    xy_cov=1.0,
                )
            )
            rows.append(
                _make_arm_row(
                    500,
                    treatment_mean,
                    4.0,
                    country=country,
                    group_id="treatment",
                    x_mean=2.0,
                    xy_cov=1.0,
                )
            )

        result = run_breakout(
            summary=pd.DataFrame(rows),
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
            methods=[Method(name="unadjusted"), Method(name="cuped", variance_reduction="cuped")],
            correction="bh",
            q=0.05,
        )
        real = [r for r in result if r.excluded is None]
        assert len(real) == 4
        assert {r.method for r in real} == {"unadjusted", "cuped"}

        by_segment: dict[str, list] = {}
        for r in real:
            by_segment.setdefault(r.dimension_value, []).append(r)

        assert by_segment["US"][0].discovery is True
        assert next(r for r in by_segment["US"] if r.method == "cuped").discovery is None
        assert by_segment["CA"][0].discovery is False

        decision_us = next(r for r in by_segment["US"] if r.method == "unadjusted")
        sensitivity_us = next(r for r in by_segment["US"] if r.method == "cuped")
        # BH re-inverts a selected cell at q * rank / family size over the two
        # (metric, arm, segment) cells -- a 4-row family would give 0.0125.
        assert decision_us.require_lift().alpha == pytest.approx(0.05 * 1 / 2)
        assert sensitivity_us.require_lift().alpha == pytest.approx(0.05)
        for r in by_segment["CA"]:
            assert r.require_lift().alpha == pytest.approx(0.05)

    def test_bh_conversion_with_cuped_first_reestimates_unadjusted_decision(self):
        rows = [
            _conversion_arm_row(
                1000, successes, country="US", group_id=group_id, with_covariate=True
            )
            for group_id, successes in (("control", 100), ("treatment", 300))
        ]
        for row in rows:
            row["cx2"] = 1000.0

        result = run_breakout(
            rows,
            [_conversion_metric()],
            control_group="control",
            dimension="country",
            methods=[
                Method(name="cuped", variance_reduction="cuped"),
                Method(name="unadjusted"),
            ],
            correction="bh",
            q=0.05,
        )

        by_method = {row.method: row for row in result}
        assert by_method["unadjusted"].method_role == "decision"
        assert by_method["unadjusted"].discovery is True
        assert by_method["unadjusted"].reference_kind == "binomial"
        assert by_method["cuped"].method_role == "sensitivity"
        assert by_method["cuped"].discovery is None
        assert by_method["unadjusted"].family_procedure == "bh"
        assert by_method["cuped"].family_procedure is None
        # The sensitivity row keeps its own moment-based reference rather than
        # borrowing the decision row's exact-binomial one; CUPED estimates its
        # variance from the arms, so that reference is Welch t.
        assert by_method["cuped"].reference_kind == "t"
        assert by_method["cuped"].reference_df is not None

    @pytest.mark.slow
    def test_bh_always_valid_uses_real_retained_evidence_and_selected_inversion(self):
        from fractions import Fraction

        from increment.estimators import SequentialCell, SequentialRegistration
        from increment.frame import MetricSpec
        from increment.semantics.models import ConversionMetric
        from tests.sequential_cases import capture, registration

        specs = [MetricSpec(name=name, type="conversion") for name in ("m_a", "m_b", "m_c")]
        cells = tuple(
            SequentialCell(
                metric=spec.name,
                group_id="treatment",
                segment=(("country", country),),
                family=True,
            )
            for spec in specs
            for country in ("US", "CA")
        )
        base = registration("bernoulli")
        reg = SequentialRegistration.model_validate(
            {
                **base.model_dump(),
                "models": tuple(
                    base.models[0].model_copy(update={"metric": spec.name}) for spec in specs
                ),
                "roster": cells,
                "q": Fraction(0.05),
            }
        )
        rows = []
        for country in ("US", "CA"):
            for i in range(500):
                for arm in ("control", "treatment"):
                    base_value = i % 2
                    rows.append(
                        {
                            "unit_id": f"{country}-{i:04d}-{arm}",
                            "group_id": arm,
                            "segments": {"country": country},
                            "values": {
                                spec.name: 1
                                if arm == "treatment" and spec.name == "m_a"
                                else base_value
                                for spec in specs
                            },
                        }
                    )
        result = run_breakout(
            capture(reg, rows),
            [ConversionMetric(name=spec.name, entity="user", fact="events") for spec in specs],
            control_group="control",
            dimension="country",
            correction="bh",
            q=0.05,
            inference=AlwaysValid(registration=reg),
        )
        assert len(result) == 6
        from increment.estimation.sequential_result import SequentialInferenceResult

        logs = []
        for row in result:
            assert isinstance(row.sequential_result, SequentialInferenceResult)
            logs.append(row.sequential_result.log_e)
        selected = set(e_bh_select(logs, 0.05))
        assert selected and len(selected) < len(result)
        for i, row in enumerate(result):
            assert row.sequential_result is not None
            assert row.discovery == (i in selected)
            expected = (
                min(Fraction(0.05) * len(selected) / 6, Fraction(1, 20))
                if i in selected
                else Fraction(1, 20)
            )
            assert row.sequential_result.bounds.alpha == expected
            assert row.role == "exploratory"
            assert row.family_guarantee == "finite_sample"
            assert row.family_nominal_alpha == pytest.approx(0.05)
        reinverted = next(row for i, row in enumerate(result) if i in selected)
        with pytest.raises(CapabilityError) as raised:
            BreakoutEstimate.model_validate({**reinverted.model_dump(), "discovery": False})
        assert raised.value.code == "sequential.source.invalid"

    def test_bh_zero_selection_all_discovery_false_no_reestimation(self):
        """Every cell null -> R=0: no crash, no re-estimation call, every
        real row discovery=False at the nominal level."""
        rows = []
        for country in ("US", "CA"):
            rows.append(_make_arm_row(500, 10.0, 4.0, country=country, group_id="control"))
            rows.append(_make_arm_row(500, 10.01, 4.0, country=country, group_id="treatment"))

        result = run_breakout(
            summary=pd.DataFrame(rows),
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
            correction="bh",
            q=0.05,
        )
        real = [r for r in result if r.excluded is None]
        assert len(real) == 2
        assert all(r.discovery is False for r in real)
        for r in real:
            lift = r.lift
            assert lift is not None
            assert lift.level == pytest.approx(0.95)

    def test_bh_empty_summary_does_not_crash(self):
        result = run_breakout(
            [],
            [_mean_metric()],
            control_group="control",
            dimension="country",
            correction="bh",
        )
        assert isinstance(result, BreakoutEstimates)
        assert len(result.to_frame()) == 0

    def test_bh_preserves_segment_and_declaration_order(self):
        """Family selection must not reorder the result: rows still come
        back grouped by segment (alphabetical) then declaration order,
        matching every other correction mode's output shape."""
        summary = _bh_family_summary()
        metrics = [_mean_metric("m_a"), _mean_metric("m_b"), _mean_metric("m_c")]
        result = run_breakout(
            summary=summary,
            metrics=metrics,
            control_group="control",
            dimension="country",
            correction="bh",
            q=0.05,
        )
        real = [r for r in result if r.excluded is None]
        assert [(r.dimension_value, r.metric) for r in real] == [
            ("CA", "m_a"),
            ("CA", "m_b"),
            ("CA", "m_c"),
            ("US", "m_a"),
            ("US", "m_b"),
            ("US", "m_c"),
        ]

    def test_bh_disjoint_segment_coverage_marks_absent_metric_unavailable(self):
        """Two metrics with genuinely disjoint segment coverage: m_a is
        present in both US and CA, m_b only in US. Combining both
        metrics' moments into ONE run_breakout call (as breakout()'s
        "bh" path does) must represent (m_b, CA) as an excluded row
        (0jm6) rather than omitting it silently - a consumer computing
        segment_heterogeneity over this output needs to see that m_b's
        segment coverage is incomplete, not mistake two real rows for a
        complete family. BH family selection still scopes to the
        (metric, arm, segment) cells actually queried in each segment,
        so the real 3-cell (m_a x US/CA, m_b x US) family is unaffected
        and no CapabilityError fires."""
        rows = [
            _make_arm_row(500, 10.0, 4.0, country="US", metric="m_a", group_id="control"),
            _make_arm_row(500, 11.0, 4.0, country="US", metric="m_a", group_id="treatment"),
            _make_arm_row(500, 10.0, 4.0, country="CA", metric="m_a", group_id="control"),
            _make_arm_row(500, 10.5, 4.0, country="CA", metric="m_a", group_id="treatment"),
            _make_arm_row(500, 10.0, 4.0, country="US", metric="m_b", group_id="control"),
            _make_arm_row(500, 10.9, 4.0, country="US", metric="m_b", group_id="treatment"),
        ]
        metrics = [_mean_metric("m_a"), _mean_metric("m_b")]

        with pytest.warns(IncrementWarning) as rec:
            combined = run_breakout(
                summary=pd.DataFrame(rows),
                metrics=metrics,
                control_group="control",
                dimension="country",
                correction="bh",
                q=0.05,
            )
        assert "breakout.estimates.no_row_for_group_metric" in warning_codes(rec)
        cells = {(r.metric, r.dimension_value) for r in combined}
        assert cells == {("m_a", "US"), ("m_a", "CA"), ("m_b", "US"), ("m_b", "CA")}

        absent = next(r for r in combined if r.metric == "m_b" and r.dimension_value == "CA")
        assert absent.excluded == "no_control_arm"
        assert absent.lift is None
        assert absent.discovery is None  # not part of the BH family
        assert absent.role == "exploratory"

        real = [r for r in combined if r.excluded is None]
        assert len(real) == 3
        assert all(r.role == "exploratory" for r in real)
        assert all(r.discovery is not None for r in real)

    def test_bh_stamps_zero_variance_nonrejection_as_family_member(self):
        rows = [
            _make_arm_row(500, 10.0, 0.0, country="US", group_id="control"),
            _make_arm_row(500, 11.0, 0.0, country="US", group_id="treatment"),
            _make_arm_row(500, 10.0, 4.0, country="CA", group_id="control"),
            _make_arm_row(500, 12.0, 4.0, country="CA", group_id="treatment"),
        ]

        with pytest.warns(IncrementWarning):
            result = run_breakout(
                summary=pd.DataFrame(rows),
                metrics=[_mean_metric()],
                control_group="control",
                dimension="country",
                correction="bh",
                q=0.05,
            )

        zero_variance = next(row for row in result if row.dimension_value == "US")
        selected = next(row for row in result if row.dimension_value == "CA")
        assert zero_variance.excluded == "zero_variance"
        assert zero_variance.lift is None
        assert zero_variance.discovery is False
        assert zero_variance.family_threshold == pytest.approx(0.025)
        assert selected.discovery is True
        assert selected.family_threshold == zero_variance.family_threshold


class TestRunBreakoutBHRoutesEachCellAtItsOwnLevel:
    """A flat BH family reads each cell's p-value at a threshold as small as ``q / m`` (``m``
    the family's hypothesis count). A conversion cell is still routed and cut at its own level;
    a cell dense there but not at ``q / m`` keeps the delta route and carries its counts, so
    the composite p-value the family reads is valid at ``q / m`` too."""

    @staticmethod
    def _segment_rows(
        country: str, smallest: int, *, metric: str = "conv", lift: int = 5
    ) -> list[dict[str, Any]]:
        """A 5% conversion contrast whose sparsest of the four counts is ``smallest``, the
        treatment arm ``lift`` conversions up."""
        return [
            _conversion_arm_row(
                20 * smallest, successes, country=country, group_id=group, metric=metric
            )
            for group, successes in (("control", smallest), ("treatment", smallest + lift))
        ]

    @staticmethod
    def _kinds(estimates: BreakoutEstimates, metric: str = "conv") -> dict[str, str | None]:
        return {
            row.dimension_value: row.reference_kind for row in estimates if row.metric == metric
        }

    @pytest.mark.parametrize(
        ("alternative", "alpha", "q"), [("two-sided", 0.2, 0.2), ("greater", 0.1, 0.1)]
    )
    def test_a_member_of_a_two_cell_family_keeps_its_own_route(self, alternative, alpha, q):
        # Both nominal levels are a 0.1 tail; q / 2 is a 0.05 tail, whose threshold the "GB"
        # cell clears and the "US" cell, dense at the nominal tail, does not.
        nominal, family = dense_min_count(0.1), dense_min_count(0.05)
        between = (nominal + family) // 2
        assert nominal < between < family
        estimates = run_breakout(
            [*self._segment_rows("US", between), *self._segment_rows("GB", 2 * family)],
            [_conversion_metric()],
            control_group="control",
            dimension="country",
            alpha=alpha,
            alternative=alternative,
            correction="bh",
            q=q,
        )
        assert self._kinds(estimates) == {"US": "t", "GB": "t"}
        us = next(row for row in estimates if row.dimension_value == "US")
        assert us.delta_evidence is not None
        assert us.delta_evidence.counts == (between, 20 * between, between + 5, 20 * between)
        # Dense at the nominal level (alpha two-sided, the single tail directional), not at the
        # level q / 2 the family reads it at: the level lies strictly between the two.
        assert q / 2 < us.delta_evidence.dense_level <= alpha

    @pytest.mark.parametrize("correction", ["none", "bonferroni"])
    def test_the_same_counts_without_a_bh_family_are_routed_at_their_own_level(self, correction):
        nominal, family = dense_min_count(0.1), dense_min_count(0.05)
        between = (nominal + family) // 2
        estimates = run_breakout(
            [*self._segment_rows("US", between), *self._segment_rows("GB", 2 * family)],
            [_conversion_metric()],
            control_group="control",
            dimension="country",
            alpha=0.2,
            correction=correction,
            q=0.2,
        )
        # Bonferroni divides alpha by the two segments: a 0.05 tail is the cell's own level.
        expected = "binomial" if correction == "bonferroni" else "t"
        assert self._kinds(estimates) == {"US": expected, "GB": "t"}

    def test_an_exploratory_family_over_uncorrected_cells_reads_the_same_evidence(self):
        """The rows of ``correction="none"`` corrected after the fact by
        ``select_exploratory_family`` reach the p-values, discoveries and intervals of
        ``correction="bh"`` for cells dense at the nominal level but not at ``q / m``, selected
        and not: both read the composite p-value from the same persisted counts, and both
        reissue a selected cell by its counts at the corrected level."""
        from increment.estimation.exploratory_family import select_exploratory_family

        nominal, family = dense_min_count(0.1), dense_min_count(0.05)
        between = (nominal + family) // 2
        rows = [
            *self._segment_rows("US", between, lift=120),
            *self._segment_rows("GB", between + 40, lift=100),
            *self._segment_rows("FR", 2 * family, lift=0),
            *self._segment_rows("DE", between + 20, lift=2),
        ]
        q = 0.2
        corrected = run_breakout(
            rows,
            [_conversion_metric()],
            control_group="control",
            dimension="country",
            alpha=0.2,
            correction="bh",
            q=q,
        )
        uncorrected = run_breakout(
            rows, [_conversion_metric()], control_group="control", dimension="country", alpha=0.2
        )
        assert all(row.delta_evidence is not None for row in uncorrected)
        exploratory = select_exploratory_family(list(uncorrected), q=q)
        by_segment = {row.dimension_value: row for row in corrected}
        assert {row.discovery for row in corrected} == {True, False}
        selected_below_level = [
            row
            for row in corrected
            if row.discovery and row.family_threshold is not None and row.binomial_set is not None
        ]
        assert selected_below_level, "a selected cell was reissued below its dense level"
        for row in exploratory:
            assert isinstance(row, BreakoutEstimate)
            twin = by_segment[row.dimension_value]
            assert row.p_value() == twin.p_value()
            assert row.discovery == twin.discovery
            assert row.family_threshold == twin.family_threshold
            assert row.reference_kind == twin.reference_kind
            assert row.delta_evidence == twin.delta_evidence
            assert (row.require_lift().lb, row.require_lift().ub) == (
                twin.require_lift().lb,
                twin.require_lift().ub,
            )
            assert (row.abs_lb, row.abs_ub) == (twin.abs_lb, twin.abs_ub)
            if row.discovery:
                assert row.stat_sig() is twin.stat_sig()

    def test_a_selected_directional_cell_is_reissued_at_the_tail_it_reports(self):
        """A ``greater`` family at ``q = 0.06`` over two cells selects one at ``fcr = 0.03``:
        its counts are dense at that tail but not at half of it, so the reissued cell keeps
        the delta route with its bound opened at ``0.03`` -- on the moment path and on the
        exploratory path alike."""
        from increment.estimation.exploratory_family import select_exploratory_family

        rows = [
            *(
                _conversion_arm_row(20_000, successes, country="US", group_id=group)
                for group, successes in (("control", 2_600), ("treatment", 2_900))
            ),
            *(
                _conversion_arm_row(20_000, successes, country="GB", group_id=group)
                for group, successes in (("control", 2_600), ("treatment", 2_600))
            ),
        ]
        corrected = run_breakout(
            rows,
            [_conversion_metric()],
            control_group="control",
            dimension="country",
            alpha=0.05,
            alternative="greater",
            correction="bh",
            q=0.06,
        )
        us = next(row for row in corrected if row.dimension_value == "US")
        assert us.discovery is True and us.family_threshold == 0.03
        assert us.reference_kind == "t"
        assert us.require_lift().alpha == 0.03 and us.require_lift().open_side == "upper"
        assert us.delta_evidence is not None
        assert 0.015 < us.delta_evidence.dense_level <= 0.03
        exploratory = select_exploratory_family(
            list(
                run_breakout(
                    rows,
                    [_conversion_metric()],
                    control_group="control",
                    dimension="country",
                    alpha=0.05,
                    alternative="greater",
                )
            ),
            q=0.06,
        )
        twin = next(
            row
            for row in exploratory
            if isinstance(row, BreakoutEstimate) and row.dimension_value == "US"
        )
        assert twin.reference_kind == "t"
        assert twin.require_lift().lb == us.require_lift().lb
        assert twin.p_value() == us.p_value()
        assert twin.delta_evidence == us.delta_evidence

    def test_the_family_size_counts_only_the_cells_present(self):
        """``other`` is carried by one segment only: the family is three cells, and every cell
        keeps the route its own level gives it whatever the family's size."""
        family, deeper = dense_min_count(0.05), dense_min_count(0.0375)
        assert family < deeper
        between_levels = (family + deeper) // 2
        nominal = dense_min_count(0.1)
        with pytest.warns(IncrementWarning) as rec:
            estimates = run_breakout(
                [
                    *self._segment_rows("US", (nominal + family) // 2),
                    *self._segment_rows("US", 4 * family, metric="other"),
                    *self._segment_rows("GB", between_levels),
                ],
                [_conversion_metric(), _conversion_metric("other")],
                control_group="control",
                dimension="country",
                alpha=0.2,
                correction="bh",
                q=0.3,
            )
        assert "breakout.estimates.no_row_for_group_metric" in warning_codes(rec)
        assert self._kinds(estimates) == {"US": "t", "GB": "t"}


# alternative= forwarding: one-sided per-segment testing.


class TestRunBreakoutAlternative:
    def test_default_alternative_is_two_sided_and_unchanged(self):
        """Omitting `alternative=` reproduces the exact interval `run_breakout` always produced,
        and every row is labeled "two-sided"."""
        rows = [
            _make_arm_row(500, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(500, 11.0, 4.0, country="US", group_id="treatment"),
        ]

        implicit = run_breakout(
            rows, [_mean_metric()], control_group="control", dimension="country"
        )
        explicit = run_breakout(
            rows,
            [_mean_metric()],
            control_group="control",
            dimension="country",
            alternative="two-sided",
        )

        assert [r.alternative for r in implicit] == ["two-sided"]
        for a, b in zip(implicit, explicit, strict=True):
            a_lift = a.lift
            b_lift = b.lift
            assert a_lift is not None and b_lift is not None
            assert a_lift.value == b_lift.value
            assert a_lift.lb == b_lift.lb
            assert a_lift.ub == b_lift.ub
            assert a_lift.level == b_lift.level

    def test_one_sided_greater_guardrail_matches_doubled_alpha_two_sided_interval(self):
        """One-sided `alternative="greater"` at `alpha=0.05` reproduces the SAME interval as
        two-sided at `alpha=0.10` - the alpha-doubling identity `infer_lift` relies on, just
        relabeled."""
        rows = [
            _make_arm_row(2000, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(2000, 12.0, 4.0, country="US", group_id="treatment"),
        ]

        one_sided = run_breakout(
            rows,
            [_mean_metric()],
            control_group="control",
            dimension="country",
            alpha=0.05,
            alternative="greater",
        )
        two_sided = run_breakout(
            rows, [_mean_metric()], control_group="control", dimension="country", alpha=0.10
        )

        assert one_sided[0].alternative == "greater"
        assert two_sided[0].alternative == "two-sided"
        one_sided_lift = one_sided[0].lift
        two_sided_lift = two_sided[0].lift
        assert one_sided_lift is not None and two_sided_lift is not None
        assert one_sided_lift.level == pytest.approx(0.90)
        assert one_sided_lift.lb == pytest.approx(two_sided_lift.lb)
        assert one_sided_lift.ub == pytest.approx(two_sided_lift.ub)

    def test_unestimable_segment_nan_row_carries_requested_alternative(self):
        """A segment excluded before `estimate_lift` runs (no control arm) still reports the
        REQUESTED `alternative` on its dense NaN row, matching a real row's label."""
        rows = [_make_arm_row(50, 10.0, 4.0, country="US", group_id="treatment")]

        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows,
                [_mean_metric()],
                control_group="control",
                dimension="country",
                alternative="less",
            )
        assert "breakout.estimates.slice_no_control_arm" in warning_codes(rec)

        assert len(result) == 1
        assert result[0].excluded == "no_control_arm"
        assert result[0].lift is None
        assert result[0].alternative == "less"


class TestRunBreakoutAlwaysValidInference:
    @pytest.mark.parametrize("correction", ["none", "bonferroni", "bh"])
    def test_registered_gaussian_segments_refuse_before_public_breakout(self, correction):
        from increment.errors import CapabilityError
        from tests.sequential_cases import raw_gaussian

        segments = tuple((("country", country),) for country in ("US", "CA", "GB", "DE"))
        snapshot, metrics, policy = raw_gaussian(
            n=4,
            segments=segments,
            alpha=0.05 / 4 if correction == "bonferroni" else 0.05,
            family=correction == "bh",
        )
        with pytest.raises(CapabilityError) as raised:
            run_breakout(
                snapshot,
                metrics,
                control_group="control",
                dimension="country",
                inference=policy,
                correction=correction,
            )
        assert raised.value.code == "sequential.route.unsupported"

    def test_moments_cannot_be_reinterpreted_as_raw_likelihood(self):
        from increment.errors import CapabilityError
        from tests.sequential_cases import raw_gaussian

        _, metrics, policy = raw_gaussian()
        with pytest.raises(CapabilityError) as raised:
            run_breakout(
                _four_segment_summary(),
                metrics,
                control_group="control",
                dimension="country",
                inference=policy,
            )
        assert raised.value.code == "sequential.source.invalid"


# Input format equivalence + defensive errors (mirrors
# test_engine.py's TestInputFormats).


class TestRunBreakoutInputFormats:
    def test_pandas_pyarrow_and_dict_rows_agree(self):
        """run_breakout gives numerically identical results for a pandas DataFrame, a pyarrow
        Table, and a plain list[dict] - mirrors estimate_lift's own format-equivalence contract."""
        us_control = _make_arm_row(n=2000, mean=10.0, var=4.0, country="US", group_id="control")
        us_treatment = _make_arm_row(n=2000, mean=11.0, var=4.0, country="US", group_id="treatment")
        ca_control = _make_arm_row(n=1500, mean=9.0, var=4.0, country="CA", group_id="control")
        ca_treatment = _make_arm_row(n=1500, mean=10.5, var=4.0, country="CA", group_id="treatment")
        rows = [us_control, us_treatment, ca_control, ca_treatment]

        pandas_result = run_breakout(
            summary=pd.DataFrame(rows),
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
        )
        pyarrow_result = run_breakout(
            summary=pa.Table.from_pylist(rows),
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
        )
        dict_result = run_breakout(
            summary=rows,
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
        )

        def _by_segment(results: list[BreakoutEstimate]) -> dict:
            out = {}
            for r in results:
                lift = r.lift
                assert lift is not None
                out[r.dimension_value] = (lift.value, lift.lb, lift.ub)
            return out

        assert len(pandas_result) == len(pyarrow_result) == len(dict_result) == 2
        assert _by_segment(pandas_result) == _by_segment(pyarrow_result) == _by_segment(dict_result)

    def test_dimension_column_missing_from_dataframe_raises(self):
        """A DataFrame missing the requested dimension column raises a
        clear ValueError, not a KeyError deep in the grouping loop."""
        row = _make_arm_row(n=100, mean=10.0, var=4.0, country="US", group_id="control")
        df = pd.DataFrame([row]).drop(columns=["country"])

        with pytest.raises(InvalidRequestError) as exc_info:
            run_breakout(
                summary=df,
                metrics=[_mean_metric()],
                control_group="control",
                dimension="country",
            )
        assert exc_info.value.code == "breakout.run_breakout_dimension"
        assert exc_info.value.context["dimension"] == "country"

    def test_dict_rows_missing_dimension_key_raises(self):
        """Mirrors estimate_lift's `test_dict_rows_missing_column_raises`:
        no schema to pre-validate, so it fails with a KeyError at first access."""
        row = _make_arm_row(n=100, mean=10.0, var=4.0, country="US", group_id="control")
        del row["country"]

        with pytest.raises(KeyError):
            run_breakout(
                summary=[row],
                metrics=[_mean_metric()],
                control_group="control",
                dimension="country",
            )

    def test_missing_ref_y_column_raises_value_error_not_key_error(self):
        """A `summary` DataFrame missing `ref_y` must fail with a
        descriptive ValueError naming the column, not a bare KeyError."""
        row = _make_arm_row(n=100, mean=10.0, var=4.0, country="US", group_id="control")
        df = pd.DataFrame([row]).drop(columns=["ref_y"])

        with pytest.raises(InvalidRequestError) as exc_info:
            run_breakout(
                summary=df,
                metrics=[_mean_metric()],
                control_group="control",
                dimension="country",
            )
        assert exc_info.value.code == "breakout.run_breakout_group"
        assert exc_info.value.context["missing"] == ("ref_y",)

    def test_methods_empty_list_raises_value_error(self):
        """`methods=[]` (distinct from `None`, which resolves to `[Method(name="unadjusted")]`)
        used to silently produce zero rows plus a factually wrong "no data" warning. Now raises
        up front."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment"),
        ]
        with pytest.raises(InvalidRequestError) as exc_info:
            run_breakout(
                rows,
                [_mean_metric("rev")],
                control_group="control",
                dimension="country",
                methods=[],
            )
        assert exc_info.value.code == "breakout.run_breakout_methods"

    def test_methods_none_still_uses_the_unadjusted_default(self):
        """`methods=None` is NOT the same as `methods=[]` - it resolves
        to the documented default, not "zero methods."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment"),
        ]
        result = run_breakout(
            rows, [_mean_metric("rev")], control_group="control", dimension="country", methods=None
        )
        assert [r.method for r in result] == ["unadjusted"]

    def test_empty_methods_by_metric_does_not_emit_unestimable_rows(self):
        """An explicitly disabled metric must not regain the default unadjusted method on an
        unestimable cell."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 0.0, 0.0, country="US", group_id="treatment"),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows,
                [_mean_metric("rev")],
                control_group="control",
                dimension="country",
                methods_by_metric={"rev": []},
            )
        assert "breakout.estimates.row_nonpositive_mean" in warning_codes(rec)
        assert result == []

    def test_duplicate_segment_metric_arm_row_raises(self):
        """Two rows for the same (segment, metric, arm) cell must refuse
        rather than silently discard one row's units or double-count the
        segment downstream."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment"),
            _make_arm_row(50, 13.0, 4.0, country="US", group_id="treatment"),
        ]
        with pytest.raises(InvalidRequestError) as exc_info:
            run_breakout(rows, [_mean_metric("rev")], control_group="control", dimension="country")
        assert exc_info.value.code == "breakout.run_breakout_duplicate"
        assert exc_info.value.context["dimension"] == "country"
        assert exc_info.value.context["value"] == "US"
        assert exc_info.value.context["metric"] == "rev"
        assert exc_info.value.context["group_id"] == "treatment"

    def test_undeclared_metric_row_raises(self):
        """A row naming a metric absent from the declared `metrics=` list
        must refuse rather than silently estimating an undeclared metric."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", metric="rev", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="US", metric="rev", group_id="treatment"),
            _make_arm_row(50, 12.0, 4.0, country="US", metric="unknown", group_id="treatment"),
        ]
        with pytest.raises(InvalidRequestError) as exc_info:
            run_breakout(rows, [_mean_metric("rev")], control_group="control", dimension="country")
        assert exc_info.value.code == "breakout.metric_found_group"
        assert exc_info.value.context["unknown_metrics"] == ("unknown",)
        assert exc_info.value.context["declared_names"] == ("rev",)


# BreakoutEstimate model (frozen-model style, mirrors
# test_armstats.py/test_inference.py).


class TestBreakoutEstimate:
    def test_frozen(self):
        est = BreakoutEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            dimension="country",
            dimension_value="US",
            lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
        )
        with pytest.raises((TypeError, ValueError)):
            est.dimension_value = "CA"  # ty: ignore[invalid-assignment]  # proving frozen at runtime

    def test_basic_construction(self):
        lift = Estimate(value=0.2, lb=0.1, ub=0.3, level=0.95)
        est = BreakoutEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            dimension="country",
            dimension_value="US",
            lift=lift,
        )
        assert est.metric == "rev"
        assert est.group_id == "treatment"
        assert est.method == "unadjusted"
        assert est.dimension == "country"
        assert est.dimension_value == "US"
        assert est.lift is lift

    def test_str_preserves_disconnected_asymptotic_sequential_components(self):
        from fractions import Fraction

        from increment.estimation.asymptotic_mean import AsymptoticMeanSet, MeanSetComponent
        from increment.estimation.sequential_result import AsymptoticSequentialResult

        bounds = AsymptoticMeanSet(
            components=(
                MeanSetComponent(lower=None, upper=Fraction(-1)),
                MeanSetComponent(lower=Fraction(1), upper=None),
            ),
            alpha=Fraction(1, 20),
            alternative="two-sided",
            count=5,
            k=Fraction(1),
            available=True,
            estimator_contrast=Fraction(0),
            estimator_variance=Fraction(1),
        )
        result = AsymptoticSequentialResult.model_construct(
            bounds=bounds, point_reason="ratio_point_unavailable"
        )
        row = BreakoutEstimate.model_construct(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            dimension="country",
            dimension_value="US",
            lift=None,
            sequential_result=result,
            low_reliability=False,
        )

        rendered = str(row)

        assert "ratio_interval=" in rendered
        assert "∪" in rendered and "geometry='disconnected'" in rendered
        assert "point_reason='ratio_point_unavailable'" in rendered

    def test_str_preserves_vacuous_unavailable_sequential_bounds(self):
        from fractions import Fraction

        from increment.estimation.asymptotic_mean import AsymptoticMeanSet, MeanSetComponent
        from increment.estimation.sequential_result import AsymptoticSequentialResult

        bounds = AsymptoticMeanSet(
            components=(MeanSetComponent(lower=None, upper=None),),
            alpha=Fraction(1, 20),
            alternative="two-sided",
            count=5,
            k=None,
            available=False,
            reason="variance_unavailable",
        )
        result = AsymptoticSequentialResult.model_construct(
            bounds=bounds, point_reason="point_unavailable"
        )
        row = BreakoutEstimate.model_construct(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            dimension="country",
            dimension_value="US",
            lift=None,
            sequential_result=result,
            low_reliability=False,
        )

        rendered = str(row)

        assert "geometry='unavailable'" in rendered
        assert "bounds_reason='variance_unavailable'" in rendered
        assert "point_reason='point_unavailable'" in rendered

    def test_str_preserves_relative_and_additive_unavailability_reasons(self):
        from increment.estimation.results import JointContrastReference, RelativeConfidenceSet

        relative_set = RelativeConfidenceSet.model_construct(
            reference=JointContrastReference.model_construct(c=0.0),
            alpha=0.05,
            geometry="unavailable",
            reason="relative_interval_unavailable",
            intervals=(),
        )
        row = BreakoutEstimate.model_construct(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            dimension="country",
            dimension_value="US",
            lift=None,
            relative_unavailable_reason="joint_covariance_unrepresentable",
            relative_confidence_set=relative_set,
            abs_diff=2.5,
            abs_lb=2.0,
            abs_ub=3.0,
            low_reliability=False,
        )

        rendered = str(row)

        assert "joint_covariance_unrepresentable" in rendered
        assert "relative_interval_unavailable" in rendered
        assert "zero_denominator" in rendered
        assert "abs_diff=+2.5" in rendered and "abs_interval=(+2, +3)" in rendered

    def test_abs_diff_abs_se_default_none(self):
        """abs_diff/abs_se default to None until a constructor populates them."""
        est = BreakoutEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            dimension="country",
            dimension_value="US",
            lift=Estimate(value=0.2, lb=0.1, ub=0.3, level=0.95),
        )
        assert est.abs_diff is None
        assert est.abs_se is None

    def test_abs_diff_abs_se_settable(self):
        est = BreakoutEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            dimension="country",
            dimension_value="US",
            lift=Estimate(value=0.2, lb=0.1, ub=0.3, level=0.95),
            abs_diff=1.5,
            abs_se=0.3,
        )
        assert est.abs_diff == 1.5
        assert est.abs_se == 0.3

    @pytest.mark.parametrize("field", ["abs_diff", "abs_se", "null_abs", "null_lift"])
    def test_nonfinite_scalar_fields_are_rejected(self, field: str) -> None:
        kwargs: dict[str, Any] = {field: math.inf}
        with pytest.raises(InvalidRequestError) as raised:
            BreakoutEstimate(
                metric="rev",
                group_id="treatment",
                method="unadjusted",
                method_role="decision",
                dimension="country",
                dimension_value="US",
                lift=Estimate(value=0.2),
                **kwargs,
            )
        assert raised.value.code == "model.field.nonfinite"
        assert raised.value.context["field"] == field

    def test_missing_required_field_raises(self):
        with pytest.raises(InvalidRequestError) as raised:
            BreakoutEstimate(
                group_id="treatment",
                method="unadjusted",
                dimension="country",
                dimension_value="US",
                lift=Estimate(value=0.1),
            )  # ty: ignore[missing-argument]  # proving `metric` is required
        assert raised.value.code == "model.field.missing"

    def test_neither_lift_nor_excluded_set_raises_coded_error(self):
        """The buried coded refusal must surface with its own `.code`, not a
        bare pydantic `ValidationError`."""
        with pytest.raises(InvalidRequestError) as exc_info:
            BreakoutEstimate(
                metric="rev",
                group_id="treatment",
                method="unadjusted",
                method_role="decision",
                dimension="country",
                dimension_value="US",
                lift=None,
            )
        assert exc_info.value.code == "breakout.breakout.exactly_one_lift"

    def test_round_trips_on_a_real_run_breakout_result(self):
        """model_dump_json() round-trips on a REAL run_breakout output - Estimate's dropped
        `posterior` field carries through the whole breakout wrapper, not just LiftEstimate in
        isolation."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment"),
        ]
        results = run_breakout(
            rows, [_mean_metric("rev")], control_group="control", dimension="country"
        )
        assert len(results) == 1
        est = results[0]
        round_tripped = BreakoutEstimate.model_validate_json(est.model_dump_json())
        assert round_tripped == est

    def test_breakout_estimate_row_preserves_a_welch_reference_through_round_trip(self):
        # Unequal arm counts give the unadjusted lift a Welch t reference
        # whose df must survive serialization, not be reconstructed as Normal.
        rows = [
            _make_arm_row(n=5, mean=10.0, var=4.0, country="US", group_id="control"),
            _make_arm_row(n=4, mean=11.0, var=4.0, country="US", group_id="treatment"),
        ]
        (row,) = run_breakout(
            rows, [_mean_metric("rev")], control_group="control", dimension="country"
        )
        assert row.dof is None
        assert row.reference_kind == "t"
        assert row.reference_df is not None
        round_tripped = BreakoutEstimate.model_validate(row.model_dump())

        assert round_tripped.dof is None
        assert round_tripped.reference_kind == "t"
        assert round_tripped.reference_df == pytest.approx(row.reference_df)

    def test_binomial_segment_keeps_its_score_reference_through_frame_and_round_trip(self):
        rows = [
            _conversion_arm_row(1000, 10, country="US", group_id="control"),
            _conversion_arm_row(1000, 4, country="US", group_id="treatment"),
        ]
        result = run_breakout(
            rows, [_conversion_metric()], control_group="control", dimension="country"
        )
        (row,) = result
        assert row.reference_kind == "binomial"
        assert row.abs_reference_kind == "score"
        frame = cast(pd.DataFrame, result.to_frame(backend="pandas"))
        assert frame["abs_reference_kind"].tolist() == ["score"]
        round_tripped = BreakoutEstimate.model_validate_json(row.model_dump_json())
        assert round_tripped.abs_reference_kind == "score"
        assert (round_tripped.abs_lb, round_tripped.abs_ub) == (row.abs_lb, row.abs_ub)


class TestBreakoutEstimateReferenceValidation:
    @staticmethod
    def _sequential_kwargs(**overrides: Any) -> dict[str, Any]:
        from increment.estimators import estimate_sequential
        from tests.sequential_cases import registered_bernoulli

        snapshot, policy = registered_bernoulli(n=4, segments=((("country", "US"),),))
        row = estimate_sequential(snapshot, policy).results[0]
        row = BreakoutEstimate.model_validate(
            {**row.model_dump(), "dimension": "country", "dimension_value": "US"}
        )
        return {**row.model_dump(), **overrides}

    @staticmethod
    def _kwargs(**overrides: Any) -> dict[str, Any]:
        values: dict[str, Any] = {
            "metric": "rev",
            "group_id": "treatment",
            "method": "unadjusted",
            "method_role": "decision",
            "dimension": "country",
            "dimension_value": "US",
            "lift": Estimate(value=0.1),
        }
        values.update(overrides)
        return values

    def test_normal_reference_is_the_default(self):
        row = BreakoutEstimate(**self._kwargs())
        assert row.reference_kind == "normal"
        assert row.reference_df is None

    def test_t_reference_round_trips_through_model_validate(self):
        row = BreakoutEstimate(**self._kwargs(dof=5.0, reference_kind="t", reference_df=5.0))
        again = BreakoutEstimate.model_validate(row.model_dump())
        assert again.reference_kind == "t"
        assert again.reference_df == 5.0

    def test_model_validate_infers_t_reference_from_legacy_dof(self):
        row = BreakoutEstimate.model_validate(self._kwargs(dof=8.0))
        assert row.reference_kind == "t"
        assert row.reference_df == 8.0

    def test_explicit_normal_reference_with_dof_is_not_legacy_inferred(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            BreakoutEstimate.model_validate(self._kwargs(dof=8.0, reference_kind="normal"))
        assert exc_info.value.code == "estimation.results.lift.dof_reference_df_mismatch"

    def test_explicit_none_reference_df_with_dof_is_not_legacy_inferred(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            BreakoutEstimate.model_validate(self._kwargs(dof=8.0, reference_df=None))
        assert exc_info.value.code == "estimation.results.lift.dof_reference_df_mismatch"

    def test_constructor_rejects_t_reference_without_df(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            BreakoutEstimate(**self._kwargs(reference_kind="t", reference_df=None))
        assert exc_info.value.code == "estimation.results.lift.reference_df_required_for_t"

    def test_constructor_rejects_reference_df_on_normal_reference(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            BreakoutEstimate(**self._kwargs(reference_kind="normal", reference_df=5.0))
        assert exc_info.value.code == "estimation.results.lift.reference_df_set_for_normal"

    def test_model_validate_rejects_dof_reference_df_mismatch(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            BreakoutEstimate.model_validate(
                self._kwargs(dof=5.0, reference_kind="t", reference_df=8.0)
            )
        assert exc_info.value.code == "estimation.results.lift.dof_reference_df_mismatch"

    def test_legacy_sequential_payload_refuses_without_raw_checkpoint(self):
        from increment.errors import CapabilityError

        with pytest.raises(CapabilityError) as raised:
            BreakoutEstimate.model_validate(self._kwargs(inference="always_valid"))
        assert raised.value.code == "sequential.continuation.legacy"

    def test_public_replay_rejects_gaussian_checkpoint(self):
        from increment.errors import CapabilityError
        from increment.estimation.sequential_runtime import (
            _evaluate_sequential_diagnostic,
            display_estimate,
        )
        from tests.sequential_cases import raw_gaussian

        # Only the diagnostic evaluator builds a non-admitted gaussian
        # checkpoint; estimate_sequential refuses the law before evaluating it.
        snapshot, _, policy = raw_gaussian(segments=((("country", "US"),),), n=4)
        result = _evaluate_sequential_diagnostic(snapshot, policy)[0]
        with pytest.raises(CapabilityError) as raised:
            BreakoutEstimate.model_validate(
                self._kwargs(
                    inference="always_valid",
                    reference_kind="sequential",
                    sequential_result=result,
                    lift=display_estimate(result),
                )
            )
        assert raised.value.code == "sequential.route.unsupported"

    @pytest.mark.parametrize("relabel", [False, True])
    def test_public_replay_preserves_bernoulli_reference_label(self, relabel):
        from increment.errors import CapabilityError

        row = BreakoutEstimate.model_validate(self._sequential_kwargs())
        if not relabel:
            assert BreakoutEstimate.model_validate_json(row.model_dump_json()) == row
            return
        payload = row.model_dump()
        payload.update(inference="fixed", reference_kind="normal")
        with pytest.raises(CapabilityError) as caught:
            BreakoutEstimate.model_validate(payload)
        assert caught.value.code == "sequential.source.invalid"

    def test_constructor_rejects_normal_reference_for_sequential_inference(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            BreakoutEstimate(**self._kwargs(inference="always_valid", reference_kind="normal"))
        assert exc_info.value.code == ("estimation.results.lift.inference_reference_kind_mismatch")

    def test_constructor_rejects_sequential_reference_for_fixed_inference(self):
        kwargs = self._sequential_kwargs(inference="fixed")
        with pytest.raises(InvalidRequestError) as exc_info:
            BreakoutEstimate(**kwargs)
        assert exc_info.value.code == ("estimation.results.lift.inference_reference_kind_mismatch")

    def test_constructor_rejects_reference_df_on_sequential_reference(self):
        kwargs = self._sequential_kwargs(reference_df=5.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            BreakoutEstimate(**kwargs)
        assert exc_info.value.code == "estimation.results.lift.reference_df_set_for_normal"

    @pytest.mark.parametrize(
        "field",
        ["relative_confidence_set", "relative_unavailable_reason", "abs_diff", "null_abs"],
    )
    def test_sequential_checkpoint_rejects_mixed_authoritative_payload(self, field):
        """A checkpoint-backed sequential view cannot carry fixed/joint sidecars."""
        from increment.errors import CapabilityError
        from increment.estimation.results import JointContrastReference, RelativeConfidenceSet

        payload = self._sequential_kwargs()
        if field == "relative_confidence_set":
            payload[field] = RelativeConfidenceSet(
                reference=JointContrastReference(a=1.0, c=1.0, var_a=1.0, var_c=1.0, cov_ac=0.0),
                alpha=0.05,
            )
        elif field == "relative_unavailable_reason":
            payload[field] = "joint_covariance_indefinite"
        else:
            payload[field] = 0.0 if field == "null_abs" else 1.0

        with pytest.raises(CapabilityError) as raised:
            BreakoutEstimate.model_validate(payload)
        assert raised.value.code == "sequential.source.invalid"


class TestRunBreakoutReliabilityFloor:
    """`reliability_floor` (default 50): a real estimate whose control or treatment arm has fewer
    units than the floor is still returned, not omitted or excluded, but flagged
    `low_reliability=True`."""

    def _mixed_n_summary(self, *, small_n: int, large_n: int) -> pd.DataFrame:
        """Two segments: 'thin' has `small_n` units per arm, 'thick' has `large_n` - both above the
        few_units gate (n>=2) and estimable, differing only in whether they clear the
        reliability floor."""
        rows = [
            _make_arm_row(n=small_n, mean=10.0, var=4.0, country="thin", group_id="control"),
            _make_arm_row(n=small_n, mean=11.0, var=4.0, country="thin", group_id="treatment"),
            _make_arm_row(n=large_n, mean=10.0, var=4.0, country="thick", group_id="control"),
            _make_arm_row(n=large_n, mean=11.0, var=4.0, country="thick", group_id="treatment"),
        ]
        return pd.DataFrame(rows)

    def test_below_floor_segment_flagged_above_floor_not(self):
        result = run_breakout(
            summary=self._mixed_n_summary(small_n=20, large_n=200),
            metrics=[_windowed_mean_metric()],
            control_group="control",
            dimension="country",
        )
        by_segment = {r.dimension_value: r for r in result}
        assert by_segment["thin"].low_reliability is True
        assert by_segment["thick"].low_reliability is False
        # Both are real, estimated rows - the floor does not exclude.
        assert by_segment["thin"].excluded is None
        assert by_segment["thin"].lift is not None

    def test_floor_is_a_parameter_not_a_literal(self):
        """The same n=20 segment is NOT flagged once the caller lowers
        the floor - proves it's a real parameter, not a hardcoded constant."""
        result = run_breakout(
            summary=self._mixed_n_summary(small_n=20, large_n=200),
            metrics=[_windowed_mean_metric()],
            control_group="control",
            dimension="country",
            reliability_floor=10,
        )
        by_segment = {r.dimension_value: r for r in result}
        assert by_segment["thin"].low_reliability is False

    def test_default_floor_is_50(self):
        """n=49 flags, n=50 does not: pins the documented default via the boundary itself, since
        `reliability_floor` is a function parameter with no field to read a default off of."""
        just_below = run_breakout(
            summary=self._mixed_n_summary(small_n=49, large_n=200),
            metrics=[_windowed_mean_metric()],
            control_group="control",
            dimension="country",
        )
        just_at = run_breakout(
            summary=self._mixed_n_summary(small_n=50, large_n=200),
            metrics=[_windowed_mean_metric()],
            control_group="control",
            dimension="country",
        )
        assert {r.dimension_value: r for r in just_below}["thin"].low_reliability is True
        assert {r.dimension_value: r for r in just_at}["thin"].low_reliability is False

    def test_asymmetric_arm_below_floor_flags_even_when_other_arm_is_large(self):
        """The floor check is an OR across control and treatment arm sizes: a LARGE control with a
        thin treatment (or vice versa) must still flag, not just when both arms are thin
        together."""
        control_thin_rows = [
            _make_arm_row(n=20, mean=10.0, var=4.0, country="control_thin", group_id="control"),
            _make_arm_row(n=200, mean=11.0, var=4.0, country="control_thin", group_id="treatment"),
        ]
        treatment_thin_rows = [
            _make_arm_row(n=200, mean=10.0, var=4.0, country="treatment_thin", group_id="control"),
            _make_arm_row(n=20, mean=11.0, var=4.0, country="treatment_thin", group_id="treatment"),
        ]
        result = run_breakout(
            summary=pd.DataFrame(control_thin_rows + treatment_thin_rows),
            metrics=[_windowed_mean_metric()],
            control_group="control",
            dimension="country",
        )
        by_segment = {r.dimension_value: r for r in result}
        assert by_segment["control_thin"].low_reliability is True
        assert by_segment["treatment_thin"].low_reliability is True


class TestRunBreakoutArmCounts:
    """`n_treat`/`n_control`: per-(segment, metric) arm unit counts,
    populated on every real estimate, `None` on an excluded row."""

    def test_real_row_carries_matching_arm_sizes(self):
        us_control = _make_arm_row(n=2000, mean=10.0, var=4.0, country="US", group_id="control")
        us_treatment = _make_arm_row(n=1500, mean=12.0, var=4.0, country="US", group_id="treatment")
        result = run_breakout(
            summary=pd.DataFrame([us_control, us_treatment]),
            metrics=[_mean_metric("rev")],
            control_group="control",
            dimension="country",
        )
        row = result[0]
        assert row.excluded is None
        assert row.n_treat == pytest.approx(1500.0)
        assert row.n_control == pytest.approx(2000.0)

    def test_excluded_row_has_none_arm_counts(self):
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment"),
            # MX: treatment only, no control row at all - excluded.
            _make_arm_row(50, 15.0, 4.0, country="MX", group_id="treatment"),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                summary=pd.DataFrame(rows),
                metrics=[_mean_metric("rev")],
                control_group="control",
                dimension="country",
            )
        assert "breakout.estimates.slice_no_control_arm" in warning_codes(rec)
        by_segment = {r.dimension_value: r for r in result}
        assert by_segment["MX"].excluded is not None
        assert by_segment["MX"].n_treat is None
        assert by_segment["MX"].n_control is None
        assert by_segment["US"].n_treat == pytest.approx(50.0)
        assert by_segment["US"].n_control == pytest.approx(50.0)

    def test_to_frame_carries_arm_counts_as_float_columns(self):
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment"),
        ]
        result = run_breakout(
            summary=pd.DataFrame(rows),
            metrics=[_mean_metric("rev")],
            control_group="control",
            dimension="country",
        )
        frame = cast(pd.DataFrame, result.to_frame(backend="pandas"))
        assert "n_treat" in frame.columns
        assert "n_control" in frame.columns
        assert frame.loc[0, "n_treat"] == pytest.approx(50.0)
        assert frame.loc[0, "n_control"] == pytest.approx(50.0)


# Exception handling: only a missing control arm, a non-positive-mean row, or
# estimate_lift's Degenerate-data/Log-delta guards are skipped; all else propagates.


class TestRunBreakoutErrorPropagation:
    def test_missing_control_arm_segment_is_skipped_with_warning(self):
        """The one condition run_breakout swallows into a NaN row: a segment
        with no control-group rows at all for the dimension value."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment"),
            # MX: treatment only, no control row at all.
            _make_arm_row(50, 15.0, 4.0, country="MX", group_id="treatment"),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows, [_mean_metric("rev")], control_group="control", dimension="country"
            )
        assert "breakout.estimates.slice_no_control_arm" in warning_codes(rec)
        assert {r.dimension_value for r in result} == {"US", "MX"}
        by_segment = {r.dimension_value: r for r in result}
        assert by_segment["US"].excluded is None
        assert by_segment["MX"].excluded == "no_control_arm"
        assert by_segment["MX"].lift is None

    def test_observational_method_name_refused_on_an_unestimable_segment(self):
        """The mislabel refusal is an invariant: a segment with no control arm (which run_breakout
        swallows into a NaN row) must refuse an observational Method just as loudly as an
        estimable one."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment"),
            # MX: treatment only, no control row at all - unestimable.
            _make_arm_row(50, 15.0, 4.0, country="MX", group_id="treatment"),
        ]
        with pytest.raises(InvalidRequestError) as exc_info:
            run_breakout(
                rows,
                [_mean_metric("rev")],
                control_group="control",
                dimension="country",
                methods=[Method(name="iptw")],
            )
        assert exc_info.value.code == "estimation.engine.method_name_observational"
        assert exc_info.value.context["m"] == "iptw"

    def test_unrelated_error_propagates_not_swallowed(self):
        """A valid segment that raises ValueError for an unrelated reason (an undeclared metric)
        must raise, not be swallowed - regression for a version that caught `except ValueError`
        unconditionally."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control", metric="undeclared"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment", metric="undeclared"),
        ]
        with pytest.raises(InvalidRequestError) as exc_info:
            run_breakout(rows, [_mean_metric("rev")], control_group="control", dimension="country")
        assert exc_info.value.code == "breakout.metric_found_group"
        assert "undeclared" in exc_info.value.context["unknown_metrics"]  # ty: ignore[unsupported-operator]

    def test_degenerate_metric_does_not_exclude_sibling_metric(self):
        """A metric whose both arms are degenerate ("Degenerate data: both arms have zero
        variance") is excluded ALONE - a healthy sibling metric in the SAME segment keeps its
        real estimate."""
        rows = [
            _make_arm_row(50, 5.0, 0.0, country="US", group_id="control", metric="rev"),
            _make_arm_row(50, 5.0, 0.0, country="US", group_id="treatment", metric="rev"),
            _make_arm_row(50, 8.0, 4.0, country="US", group_id="control", metric="orders"),
            _make_arm_row(50, 9.0, 4.0, country="US", group_id="treatment", metric="orders"),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows,
                [_mean_metric("rev"), _mean_metric("orders")],
                control_group="control",
                dimension="country",
            )
        assert "breakout.estimates.metric_guarded_excluded" in warning_codes(rec)
        by_metric = {r.metric: r for r in result}
        assert by_metric["rev"].excluded == "zero_variance"
        assert by_metric["rev"].lift is None
        assert by_metric["orders"].excluded is None
        orders_lift = by_metric["orders"].lift
        assert orders_lift is not None
        assert math.isfinite(orders_lift.value)
        assert orders_lift.value == pytest.approx(0.125, rel=0.05)

    def test_delta_method_guarded_metric_keeps_its_additive_row(self):
        """A control mean close to zero relative to its standard error ("Log delta method
        unreliable") blocks only that metric's relative lift: it keeps a live additive row and
        the sibling keeps its estimate."""
        rows = [
            _make_arm_row(50, 0.5, 4.0, country="US", group_id="control", metric="rev"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment", metric="rev"),
            _make_arm_row(50, 8.0, 4.0, country="US", group_id="control", metric="orders"),
            _make_arm_row(50, 9.0, 4.0, country="US", group_id="treatment", metric="orders"),
        ]
        result = run_breakout(
            rows,
            [_mean_metric("rev"), _mean_metric("orders")],
            control_group="control",
            dimension="country",
        )
        by_metric = {r.metric: r for r in result}
        assert by_metric["rev"].excluded is None
        assert by_metric["rev"].relative_unavailable_reason == "delta_method_unreliable"
        assert by_metric["rev"].abs_diff == pytest.approx(11.5)
        assert by_metric["orders"].excluded is None
        orders_lift = by_metric["orders"].lift
        assert orders_lift is not None
        assert math.isfinite(orders_lift.value)

    def test_duplicate_treatment_row_raises(self):
        """Two rows for the same (segment, metric, arm) cell used to double-count that segment
        downstream (k inflated, wrong Q df). Moments must arrive as ONE pre-aggregated row per
        cell."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment"),
        ]
        with pytest.raises(InvalidRequestError) as exc_info:
            run_breakout(rows, [_mean_metric("rev")], control_group="control", dimension="country")
        assert exc_info.value.code == "breakout.run_breakout_duplicate"
        assert exc_info.value.context["group_id"] == "treatment"

    def test_duplicate_control_row_raises(self):
        """A duplicate control row used to win silently by last-row-wins,
        discarding the first control row's 2000 units and flipping the
        lift (+0.132 -> -0.591 on this fixture) - must raise instead."""
        rows = [
            _make_arm_row(2000, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 30.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment"),
        ]
        with pytest.raises(InvalidRequestError) as exc_info:
            run_breakout(rows, [_mean_metric("rev")], control_group="control", dimension="country")
        assert exc_info.value.code == "breakout.run_breakout_duplicate"
        assert exc_info.value.context["group_id"] == "control"

    def test_zero_mean_treatment_row_is_dropped_other_estimates_still_returned(self):
        """A row with sum_y == 0 is dropped before estimate_lift runs -
        log(mean) is undefined at zero. The dropped row leaves a NaN,
        typed-excluded cell; a sibling metric still produces a real estimate."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control", metric="rev"),
            _make_arm_row(50, 0.0, 0.0, country="US", group_id="treatment", metric="rev"),
            _make_arm_row(50, 8.0, 4.0, country="US", group_id="control", metric="sessions"),
            _make_arm_row(50, 9.0, 4.0, country="US", group_id="treatment", metric="sessions"),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows,
                [_mean_metric("rev"), _mean_metric("sessions")],
                control_group="control",
                dimension="country",
            )
        assert "breakout.estimates.row_nonpositive_mean" in warning_codes(rec)
        assert {r.metric for r in result} == {"rev", "sessions"}
        by_metric = {r.metric: r for r in result}
        assert by_metric["rev"].excluded == "nonpositive_mean"
        assert by_metric["rev"].lift is None
        assert by_metric["sessions"].excluded is None

    def test_negative_mean_treatment_row_is_dropped_not_raising(self):
        """A negative mean (not just exactly zero) is caught by the same non-positive-mean guard,
        so ``math.log`` never raises ``ValueError: math domain error``."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, -2.0, 4.0, country="US", group_id="treatment"),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows, [_mean_metric("rev")], control_group="control", dimension="country"
            )
        assert "breakout.estimates.row_nonpositive_mean" in warning_codes(rec)
        assert len(result) == 1
        assert result[0].excluded == "nonpositive_mean"
        assert result[0].lift is None

    def test_one_metrics_lost_control_arm_only_drops_that_metrics_estimate(self):
        """A segment where ONLY 'rev's control row has a non-positive
        mean must NOT be a whole-segment skip - estimate_lift runs and
        silently drops 'rev's row; the warning must name that metric."""
        rows = [
            _make_arm_row(50, 0.0, 0.0, country="US", group_id="control", metric="rev"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment", metric="rev"),
            _make_arm_row(50, 8.0, 4.0, country="US", group_id="control", metric="sessions"),
            _make_arm_row(50, 9.0, 4.0, country="US", group_id="treatment", metric="sessions"),
        ]
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = run_breakout(
                rows,
                [_mean_metric("rev"), _mean_metric("sessions")],
                control_group="control",
                dimension="country",
            )
        assert sorted(warning_codes(record)) == sorted(
            [
                "breakout.estimates.row_nonpositive_mean",
                "breakout.estimates.lost_control_metric",
            ]
        )
        assert {r.metric for r in result} == {"rev", "sessions"}
        by_metric = {r.metric: r for r in result}
        assert by_metric["rev"].excluded == "nonpositive_mean"
        assert by_metric["rev"].lift is None
        assert by_metric["sessions"].excluded is None

    def test_zero_mean_control_row_skips_the_segment_with_its_own_reason(self):
        """Dropping the segment's ONLY control row skips the segment with the control's
        non-positive-mean reason, distinct from a genuinely missing arm - the
        per-metric "lost its control arm" warning must NOT fire here."""
        rows = [
            _make_arm_row(50, 0.0, 0.0, country="US", group_id="control"),
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="treatment"),
            _make_arm_row(50, 10.0, 4.0, country="CA", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="CA", group_id="treatment"),
        ]
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = run_breakout(
                rows, [_mean_metric("rev")], control_group="control", dimension="country"
            )
        codes = warning_codes(record)
        assert sorted(codes) == sorted(
            [
                "breakout.estimates.row_nonpositive_mean",
                "breakout.estimates.slice_no_live_control_dropped",
            ]
        )
        assert "breakout.estimates.lost_control_metric" not in codes
        assert {r.dimension_value for r in result} == {"US", "CA"}
        by_segment = {r.dimension_value: r for r in result}
        assert by_segment["US"].excluded == "nonpositive_mean"
        assert by_segment["US"].lift is None
        assert by_segment["CA"].excluded is None

    def test_degenerate_zero_variance_both_arms_segment_is_skipped_with_warning(self):
        """A segment where BOTH arms have zero within-segment variance
        makes estimate_lift's combined-SE guard raise "Degenerate data" -
        caught and skipped whole; an unaffected segment still estimates."""
        rows = [
            # US: both arms perfectly homogeneous - degenerate.
            _make_arm_row(10, 5.0, 0.0, country="US", group_id="control"),
            _make_arm_row(10, 6.0, 0.0, country="US", group_id="treatment"),
            # CA: ordinary segment, both arms have real spread.
            _make_arm_row(50, 10.0, 4.0, country="CA", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="CA", group_id="treatment"),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows, [_mean_metric("rev")], control_group="control", dimension="country"
            )
        assert "breakout.estimates.metric_guarded_excluded" in warning_codes(rec)
        assert {r.dimension_value for r in result} == {"US", "CA"}
        by_segment = {r.dimension_value: r for r in result}
        assert by_segment["US"].excluded == "zero_variance"
        assert by_segment["US"].lift is None
        assert by_segment["CA"].excluded is None

    def test_constant_control_segment_keeps_its_additive_row(self):
        """One constant arm blocks only the segment's relative lift: the segment keeps a
        live row whose additive interval is the one-sample t on the other arm."""
        from scipy.stats import t

        rows = [
            # US: a constant control arm beside a treatment arm with real spread.
            _make_arm_row(50, 5.0, 0.0, country="US", group_id="control"),
            _make_arm_row(50, 6.0, 4.0, country="US", group_id="treatment"),
            # CA: ordinary segment, real spread, precise arms.
            _make_arm_row(50, 10.0, 4.0, country="CA", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="CA", group_id="treatment"),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IncrementWarning)
            result = run_breakout(
                rows, [_mean_metric("rev")], control_group="control", dimension="country"
            )
        by_segment = {r.dimension_value: r for r in result}
        us = by_segment["US"]
        half_width = float(t.isf(0.025, 49)) * math.sqrt(4.0 / 50)
        assert us.excluded is None
        assert us.relative_unavailable_reason == "zero_variance"
        assert us.abs_lb == pytest.approx(1.0 - half_width)
        assert us.abs_ub == pytest.approx(1.0 + half_width)


# Dense breakout rows: every (segment, metric, method, arm) cell appears
# exactly once; a single-unit segment no longer aborts the whole call.


class TestRunBreakoutDenseGrid:
    def test_single_unit_segment_no_longer_aborts(self):
        """A segment with n=1 for one arm used to blow up ddof=1
        variance (ZeroDivisionError), aborting the WHOLE run_breakout
        call. It now comes back as a typed-excluded NaN row instead."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment"),
            # MX: only 1 unit in treatment - ddof=1 variance is undefined.
            _make_arm_row(1, 15.0, 0.0, country="MX", group_id="control"),
            _make_arm_row(1, 20.0, 0.0, country="MX", group_id="treatment"),
        ]
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = run_breakout(
                rows, [_mean_metric("rev")], control_group="control", dimension="country"
            )
        assert sorted(warning_codes(record)) == sorted(
            [
                "breakout.estimates.row_few_units",
                "breakout.estimates.row_few_units",
                "breakout.estimates.slice_no_live_control_dropped",
            ]
        )
        by_segment = {r.dimension_value: r for r in result}
        assert set(by_segment) == {"US", "MX"}
        assert by_segment["US"].excluded is None
        us_lift = by_segment["US"].lift
        assert us_lift is not None
        assert us_lift.value > 0
        assert by_segment["MX"].excluded == "few_units"
        assert by_segment["MX"].lift is None

    def test_dense_grid_cardinality_matches_segments_times_metrics_times_arms(self):
        """2 segments x 2 metrics x 2 non-control arms, every combination present with real,
        estimable data -> exactly 8 rows (the exact acceptance-criterion fixture)."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country=c, group_id=g, metric=m)
            for c in ("US", "CA")
            for m in ("rev", "sessions")
            for g in ("control", "treatment_a", "treatment_b")
        ]
        result = run_breakout(
            rows,
            [_mean_metric("rev"), _mean_metric("sessions")],
            control_group="control",
            dimension="country",
        )
        assert len(result) == 8
        cells = {(r.dimension_value, r.metric, r.group_id) for r in result}
        assert cells == {
            (c, m, g)
            for c in ("US", "CA")
            for m in ("rev", "sessions")
            for g in ("treatment_a", "treatment_b")
        }
        assert all(r.excluded is None for r in result)

    def test_dense_grid_covers_a_pair_missing_from_one_segment_entirely(self):
        """A (metric, arm) pair present in one segment but entirely absent from another still gets
        a dense NaN row - not silently dropped from the grid."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control", metric="rev"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment_a", metric="rev"),
            # US has an extra arm, treatment_b, only for 'rev'.
            _make_arm_row(50, 14.0, 4.0, country="US", group_id="treatment_b", metric="rev"),
            # CA never had treatment_b at all.
            _make_arm_row(50, 11.0, 4.0, country="CA", group_id="control", metric="rev"),
            _make_arm_row(50, 13.0, 4.0, country="CA", group_id="treatment_a", metric="rev"),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows, [_mean_metric("rev")], control_group="control", dimension="country"
            )
        assert "breakout.estimates.no_row_for_group_metric" in warning_codes(rec)
        # Union pairs: (rev, treatment_a), (rev, treatment_b) x 2 segments = 4 rows.
        assert len(result) == 4
        by_cell = {(r.dimension_value, r.group_id): r for r in result}
        assert by_cell[("CA", "treatment_b")].excluded == "no_treatment_arm"
        assert by_cell[("CA", "treatment_b")].lift is None
        assert by_cell[("US", "treatment_a")].excluded is None
        assert by_cell[("US", "treatment_b")].excluded is None
        assert by_cell[("CA", "treatment_a")].excluded is None

    def test_dense_grid_covers_a_metric_missing_from_one_segment_entirely(self):
        """A declared metric with no summary row at all in a segment (not just a missing arm
        within an otherwise-present metric) must still get a dense unavailable row for that
        segment (0jm6) - dropping it entirely would let a consumer like
        segment_heterogeneity mistake the survivors for a complete family."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="A", group_id="control", metric="rev"),
            _make_arm_row(50, 12.0, 4.0, country="A", group_id="treatment", metric="rev"),
            # segment B never has a 'rev' row at all -- not merely a missing arm.
            _make_arm_row(50, 11.0, 4.0, country="B", group_id="control", metric="orders"),
            _make_arm_row(50, 13.0, 4.0, country="B", group_id="treatment", metric="orders"),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows,
                [_mean_metric("rev"), _mean_metric("orders")],
                control_group="control",
                dimension="country",
            )
        assert "breakout.estimates.no_row_for_group_metric" in warning_codes(rec)
        assert len(result) == 4
        by_cell = {(r.dimension_value, r.metric): r for r in result}
        assert by_cell[("A", "rev")].excluded is None
        assert by_cell[("B", "orders")].excluded is None
        assert by_cell[("B", "rev")].excluded == "no_control_arm"
        assert by_cell[("B", "rev")].lift is None
        assert by_cell[("A", "orders")].excluded == "no_control_arm"
        assert by_cell[("A", "orders")].lift is None

    def test_control_less_segment_stays_silent_after_whole_segment_warning(self):
        """A control-less segment must not emit per-pair warnings after its whole-segment
        no-control warning, including for global pairs absent from that segment."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="A", group_id="control", metric=metric)
            for metric in ("rev", "sessions")
        ]
        rows.extend(
            _make_arm_row(50, 12.0, 4.0, country="A", group_id=group_id, metric=metric)
            for metric in ("rev", "sessions")
            for group_id in ("T1", "T2")
        )
        rows.extend(
            _make_arm_row(50, 15.0, 4.0, country="B", group_id="T1", metric=metric)
            for metric in ("rev", "sessions")
        )

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = run_breakout(
                rows,
                [_mean_metric("rev"), _mean_metric("sessions")],
                control_group="control",
                dimension="country",
            )

        assert warning_codes(record) == ["breakout.estimates.slice_no_control_arm"]

        by_cell = {(row.dimension_value, row.metric, row.group_id): row for row in result}
        assert {
            (metric, group_id) for metric in ("rev", "sessions") for group_id in ("T1", "T2")
        } == {
            (metric, group_id)
            for (segment, metric, group_id), row in by_cell.items()
            if segment == "B" and row.excluded == "no_control_arm"
        }

    def test_dense_grid_fans_out_over_methods(self):
        """A cell excluded in a segment still gets one row PER declared
        method - run_breakout's own row cardinality convention."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment"),
            # MX: no control arm at all.
            _make_arm_row(50, 15.0, 4.0, country="MX", group_id="treatment"),
        ]
        with pytest.warns(UserWarning):
            result = run_breakout(
                rows,
                [_mean_metric("rev")],
                control_group="control",
                dimension="country",
                methods=[Method(name="unadjusted"), Method(name="other")],
            )
        mx_rows = [r for r in result if r.dimension_value == "MX"]
        assert {r.method for r in mx_rows} == {"unadjusted", "other"}
        assert all(r.excluded == "no_control_arm" for r in mx_rows)

    def test_ratio_metric_nonpositive_denominator_excluded(self):
        """A ratio metric with a non-positive denominator mean is excluded even with a positive
        numerator mean - the family-dependent gate."""
        rows = [
            _make_arm_row(
                50, 10.0, 4.0, country="US", group_id="control", metric="ratio_rev", sum_den=100.0
            ),
            _make_arm_row(
                50, 12.0, 4.0, country="US", group_id="treatment", metric="ratio_rev", sum_den=0.0
            ),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows, [_ratio_metric("ratio_rev")], control_group="control", dimension="country"
            )
        assert "breakout.estimates.row_nonpositive_denominator_mean" in warning_codes(rec)
        assert len(result) == 1
        assert result[0].excluded == "nonpositive_mean"
        assert result[0].lift is None

    def test_ratio_metric_denominator_key_absent_warns_not_crashes(self):
        """A ratio row with `ref_den` entirely absent (not merely `None`) and a POSITIVE outcome
        mean must still warn gracefully, via a different `mean_den is None` path."""
        treatment_row = _make_arm_row(
            50, 12.0, 4.0, country="US", group_id="treatment", metric="ratio_rev"
        )
        assert treatment_row["ref_den"] is None
        del treatment_row["ref_den"]  # exercise the "key entirely absent" path
        rows = [
            _make_arm_row(
                50, 10.0, 4.0, country="US", group_id="control", metric="ratio_rev", sum_den=100.0
            ),
            treatment_row,
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows, [_ratio_metric("ratio_rev")], control_group="control", dimension="country"
            )
        assert "breakout.estimates.row_nonpositive_denominator_mean" in warning_codes(rec)
        assert len(result) == 1
        assert result[0].excluded == "nonpositive_mean"
        assert result[0].lift is None

    def test_ratio_metric_denominator_explicit_none_warns_not_crashes(self):
        """The sibling of the key-absent case: `ref_den` explicitly set to `None` rather than
        deleted - same `mean_den is None` guard, different code path."""
        rows = [
            _make_arm_row(
                50, 10.0, 4.0, country="US", group_id="control", metric="ratio_rev", sum_den=100.0
            ),
            _make_arm_row(
                50, 12.0, 4.0, country="US", group_id="treatment", metric="ratio_rev", sum_den=None
            ),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows, [_ratio_metric("ratio_rev")], control_group="control", dimension="country"
            )
        assert "breakout.estimates.row_nonpositive_denominator_mean" in warning_codes(rec)
        assert len(result) == 1
        assert result[0].excluded == "nonpositive_mean"
        assert result[0].lift is None

    def test_few_units_takes_precedence_over_nonpositive_mean(self):
        """A row that is BOTH n<2 AND mean(y)<=0 is classified 'few_units' - there's no ddof=1
        variance to even ask if the mean is positive."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(1, 0.0, 0.0, country="US", group_id="treatment"),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows, [_mean_metric("rev")], control_group="control", dimension="country"
            )
        assert "breakout.estimates.row_few_units" in warning_codes(rec)
        assert len(result) == 1
        assert result[0].excluded == "few_units"

    def test_explicit_empty_methods_by_metric_skips_unestimable_cells(self):
        """`methods_by_metric={name: []}` means "skip this metric" on the live
        path; an unestimable cell must honor the same contract instead of
        resurfacing as a default-unadjusted unavailable row."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", metric="rev", group_id="control"),
            _make_arm_row(1, 0.0, 0.0, country="US", metric="rev", group_id="treatment"),
            _make_arm_row(50, 10.0, 4.0, country="US", metric="aov", group_id="control"),
            _make_arm_row(1, 0.0, 0.0, country="US", metric="aov", group_id="treatment"),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows,
                [_mean_metric("rev"), _mean_metric("aov")],
                control_group="control",
                dimension="country",
                methods_by_metric={"rev": [Method(name="unadjusted")], "aov": []},
            )
        assert "breakout.estimates.row_few_units" in warning_codes(rec)
        assert {r.metric for r in result} == {"rev"}
        assert result[0].excluded == "few_units"
        assert result[0].method == "unadjusted"

    def test_abs_diff_abs_se_populated_from_a_real_non_excluded_estimate(self):
        """`run_breakout` copies `LiftEstimate.abs_diff`/`abs_se` onto the `BreakoutEstimate` for
        a real, live cell too, so the absolute scale is usable straight off a result."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment"),
        ]
        result = run_breakout(
            rows, [_mean_metric("rev")], control_group="control", dimension="country"
        )
        assert len(result) == 1
        assert result[0].excluded is None
        assert result[0].abs_diff == pytest.approx(2.0)
        assert result[0].abs_se is not None

    def test_metric_missing_control_row_entirely_warns_even_when_another_metric_has_one(self):
        """A live row whose metric never had a control row in this segment (not merely lost to
        the estimability gate) must warn like every other route, naming the metric and arm.
        Previously this pair silently became a NaN row with no warning at all."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control", metric="rev"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment", metric="rev"),
            # sessions never had a control row in this segment - never dropped, just absent.
            _make_arm_row(50, 9.0, 4.0, country="US", group_id="treatment", metric="sessions"),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows,
                [_mean_metric("rev"), _mean_metric("sessions")],
                control_group="control",
                dimension="country",
            )
        assert "breakout.estimates.route2_no_live_control_row" in warning_codes(rec)
        by_metric = {r.metric: r for r in result}
        assert by_metric["rev"].excluded is None
        assert by_metric["sessions"].excluded == "no_control_arm"
        assert by_metric["sessions"].lift is None


class TestRunBreakoutBinomialGateExemption:
    """The shared few_units/nonpositive_mean prefilter gates only rows bound
    for the log-Normal delta method, which needs a ddof=1 variance (n>=2)
    and positive means. A conversion/retention metric's n=1 or zero-event
    rows stay eligible for the exact binomial risk-ratio method (see
    binomial_rr.py), which admits both. Only a metric requiring the
    log-Normal delta method (a CUPED-adjusted conversion/retention metric)
    keeps the full gate."""

    def test_conversion_metric_n_equals_one_reaches_exact_binomial(self):
        """n=1 for both arms used to be dropped as 'few_units' before
        even reaching estimate_lift; the exact binomial method is
        admissible at n=1."""
        rows = [
            _conversion_arm_row(1, 1, country="US", group_id="control"),
            _conversion_arm_row(1, 1, country="US", group_id="treatment"),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = run_breakout(
                rows, [_conversion_metric()], control_group="control", dimension="country"
            )
        assert len(result) == 1
        row = result[0]
        assert row.excluded is None
        assert row.reference_kind == "binomial"
        assert row.lift is not None

    def test_auto_conversion_prior_keeps_sparse_binomial_rows_eligible(self):
        from increment.estimation.inference import Normal

        rows = [
            _conversion_arm_row(1, 1, country="US", group_id="control"),
            _conversion_arm_row(1, 1, country="US", group_id="treatment"),
        ]
        methods = [Method(name="unadjusted", conversion_inference="auto")]
        baseline = run_breakout(
            rows,
            [_conversion_metric()],
            control_group="control",
            dimension="country",
            methods=methods,
        )[0]
        informed = run_breakout(
            rows,
            [_conversion_metric()],
            control_group="control",
            dimension="country",
            methods=methods,
            prior=Normal(mu=0.0, sigma=0.1),
        )[0]

        assert informed.excluded is None
        assert informed.reference_kind == baseline.reference_kind == "binomial"
        assert informed.sampling_available is baseline.sampling_available is True
        assert informed.lift == baseline.lift
        assert informed.posterior_available is False

    def test_explicit_finite_sample_route_still_refuses_an_informative_prior(self):
        from increment.errors import InvalidRequestError
        from increment.estimation.inference import Normal

        rows = [
            _conversion_arm_row(2, 1, country="US", group_id="control"),
            _conversion_arm_row(2, 1, country="US", group_id="treatment"),
        ]
        with pytest.raises(InvalidRequestError) as raised:
            run_breakout(
                rows,
                [_conversion_metric()],
                control_group="control",
                dimension="country",
                methods=[Method(name="unadjusted", conversion_inference="finite_sample")],
                prior=Normal(mu=0.0, sigma=0.1),
            )
        assert raised.value.code == "estimation.binomial.finite_sample_unavailable"

    def test_finite_sample_failure_is_not_labelled_a_lift_guard_exclusion(self):
        """An arm above the finite-sample method's size ceiling fails with a binomial
        code, not a lift guard, so its row must not claim an extreme ratio."""
        big = FINITE_SAMPLE_MAX_ARM_SIZE + 1
        rows = [
            _conversion_arm_row(big, big // 10, country="GB", group_id="control"),
            _conversion_arm_row(big, big // 9, country="GB", group_id="treatment"),
            _conversion_arm_row(1000, 100, country="US", group_id="control"),
            _conversion_arm_row(1000, 130, country="US", group_id="treatment"),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_breakout(
                rows,
                [_conversion_metric()],
                control_group="control",
                dimension="country",
                methods=[Method(name="unadjusted", conversion_inference="finite_sample")],
            )
        by_segment = {row.dimension_value: row for row in result}
        assert by_segment["GB"].excluded == "estimation_failed"
        assert by_segment["US"].excluded is None

    def test_a_dense_segment_above_the_ceiling_is_estimated_by_default(self):
        """Under ``auto`` a dense segment never reaches the finite-sample ceiling: it takes
        the delta-method route beside a sparse segment on the finite-sample route."""
        big = FINITE_SAMPLE_MAX_ARM_SIZE + 1
        rows = [
            _conversion_arm_row(big, big // 10, country="GB", group_id="control"),
            _conversion_arm_row(big, big // 9, country="GB", group_id="treatment"),
            _conversion_arm_row(1000, 100, country="US", group_id="control"),
            _conversion_arm_row(1000, 130, country="US", group_id="treatment"),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_breakout(
                rows, [_conversion_metric()], control_group="control", dimension="country"
            )
        by_segment = {row.dimension_value: row for row in result}
        assert by_segment["GB"].excluded is None
        assert by_segment["GB"].reference_kind == "t"
        assert by_segment["US"].excluded is None
        assert by_segment["US"].reference_kind == "binomial"

    def test_one_arm_binomial_failure_beside_an_estimated_arm(self):
        """Only t2 exceeds the exact method's ceiling: t2 is an estimation
        failure, t1 still estimates, and a BH family refuses as run() does."""
        big = FINITE_SAMPLE_MAX_ARM_SIZE + 1
        rows = [
            _conversion_arm_row(1000, 100, country="US", group_id="control"),
            _conversion_arm_row(1000, 130, country="US", group_id="t1"),
            _conversion_arm_row(big, big // 9, country="US", group_id="t2"),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_breakout(
                rows, [_conversion_metric()], control_group="control", dimension="country"
            )
            by_arm = {row.group_id: row for row in result}
            assert by_arm["t2"].excluded == "estimation_failed"
            assert by_arm["t1"].excluded is None
            with pytest.raises(CapabilityError) as raised:
                run_breakout(
                    rows,
                    [_conversion_metric()],
                    control_group="control",
                    dimension="country",
                    correction="bh",
                )
        assert raised.value.code == "family.evidence.incomplete"

    def test_one_guarded_arm_beside_an_estimated_arm_stays_in_a_bh_family(self):
        """t2's arm mean is close to zero relative to its standard error, so only its relative
        lift is unavailable: its additive row joins the BH family beside t1."""
        rows = []
        for country in ("US", "CA"):
            rows += [
                _make_arm_row(500, 10.0, 4.0, country=country, group_id="control"),
                _make_arm_row(500, 11.0, 4.0, country=country, group_id="t1"),
                _make_arm_row(3, 1.0, 400.0, country=country, group_id="t2"),
            ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_breakout(
                rows,
                [_mean_metric("rev")],
                control_group="control",
                dimension="country",
                correction="bh",
            )
        by_cell = {(row.dimension_value, row.group_id): row for row in result}
        assert {by_cell[(c, "t2")].excluded for c in ("US", "CA")} == {None}
        assert {by_cell[(c, "t2")].relative_unavailable_reason for c in ("US", "CA")} == {
            "delta_method_unreliable"
        }
        assert all(by_cell[(c, "t1")].family_q == 0.10 for c in ("US", "CA"))
        assert all(by_cell[(c, g)].family_size == 4 for c in ("US", "CA") for g in ("t1", "t2"))

    def test_mixed_method_arm_with_two_exclusions_has_one_row_per_method(self):
        """t2 fails the exact method (too large) and CUPED's preparation
        (zero conversions); each method keeps exactly one row and reason."""
        big = FINITE_SAMPLE_MAX_ARM_SIZE + 1
        rows = [
            _conversion_arm_row(1000, 100, country="US", group_id="control", with_covariate=True),
            _conversion_arm_row(1000, 130, country="US", group_id="t1", with_covariate=True),
            _conversion_arm_row(big, 0, country="US", group_id="t2", with_covariate=True),
        ]
        for row in rows:
            row["cx2"] = float(row["n"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_breakout(
                rows,
                [_conversion_metric()],
                control_group="control",
                dimension="country",
                methods=[
                    Method(name="unadjusted"),
                    Method(name="cuped", variance_reduction="cuped"),
                ],
            )
        t2 = sorted((row.method, row.excluded) for row in result if row.group_id == "t2")
        assert t2 == [("cuped", "nonpositive_mean"), ("unadjusted", "estimation_failed")]

    def test_conversion_metric_zero_control_count_reaches_set_only_binomial(self):
        """Control never converted (0/50) - used to be dropped as
        'nonpositive_mean'; the exact binomial method admits it directly
        as a set-only row (no finite point, a real confidence set)."""
        rows = [
            _conversion_arm_row(50, 0, country="US", group_id="control"),
            _conversion_arm_row(50, 25, country="US", group_id="treatment"),
        ]
        result = run_breakout(
            rows, [_conversion_metric()], control_group="control", dimension="country"
        )
        assert len(result) == 1
        row = result[0]
        assert row.excluded is None
        assert row.lift is None
        assert row.binomial_set is not None
        assert row.binomial_set.point_available is False
        import narwhals as nw

        frame = nw.from_native(result.to_frame(), eager_only=True)
        assert "set_numerical_qualification" in frame.columns
        assert frame["set_numerical_qualification"].to_list() == [
            "scipy_special_function_error_model_conditional_v1"
        ]
        summary = nw.from_native(result.to_frame(columns="summary"), eager_only=True)
        assert summary["lb"].to_list() == [row.binomial_set.lower]
        assert summary["ub"].is_null().to_list() == [True]
        assert summary["set_upper"].is_null().to_list() == [True]
        assert summary["set_level"].to_list() == [row.binomial_set.level]

    def test_ancillary_uptake_preserves_exact_binary_breakout_sets(self):
        metric = _conversion_metric()
        bare = [
            _conversion_arm_row(50, 0, country="US", group_id="control"),
            _conversion_arm_row(50, 25, country="US", group_id="treatment"),
        ]
        rows = [
            centered_row_from_raw_sums(
                {
                    "experiment_id": "exp1",
                    "metric": metric.name,
                    "group_id": row["group_id"],
                    "country": "US",
                    "n": 50,
                    "sum_y": count,
                    "sum_y2": count,
                    "sum_d": count,
                    "sum_yd": count,
                    "sum_y2d": count,
                    "successes": row["successes"],
                }
            )
            for row, count in zip(bare, (0.0, 25.0), strict=True)
        ]
        expected = estimate_lift([metric], bare, control_group="control").results[0]
        engine = estimate_lift([metric], rows, control_group="control").results[0]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", IncrementWarning)
            (actual,) = run_breakout(rows, [metric], control_group="control", dimension="country")
        assert actual.excluded is None
        assert actual.lift is None
        assert actual.reference_kind == "binomial"
        reference = expected.binomial_set
        assert reference is not None
        assert_set_contains_finer_reference(reference)
        for result in (expected, engine, actual):
            region = result.binomial_set
            assert region is not None
            assert (region.lower, region.upper) == (reference.lower, reference.upper)
            assert region.upper is None
            assert region.point_available is False
            assert (region.x_c, region.n_c, region.x_t, region.n_t) == (0, 50, 25, 50)

    def test_nan_optional_families_are_absent_for_exact_gate(self):
        rows = [
            _conversion_arm_row(1, 0, country="US", group_id="control"),
            _conversion_arm_row(1, 1, country="US", group_id="treatment"),
        ]
        for row in rows:
            row["ref_den"] = math.nan
            row["sum_d"] = math.nan

        result = run_breakout(
            pd.DataFrame(rows),
            [_conversion_metric()],
            control_group="control",
            dimension="country",
        )

        assert result[0].excluded is None
        assert result[0].reference_kind == "binomial"

    @pytest.mark.parametrize("x_role", ["cluster_size", "uptake_total"])
    def test_sparse_non_covariate_x_role_does_not_bypass_legacy_gate(self, x_role):
        rows = [
            _conversion_arm_row(1, 1, country="US", group_id="control", with_covariate=True),
            _conversion_arm_row(1, 1, country="US", group_id="treatment", with_covariate=True),
        ]
        for row in rows:
            row["x_role"] = x_role

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = run_breakout(
                rows,
                [_conversion_metric()],
                control_group="control",
                dimension="country",
            )

        assert "breakout.estimates.row_few_units" in warning_codes(caught)
        assert result[0].excluded == "few_units"

    def test_conversion_metric_with_cuped_method_keeps_the_legacy_gate(self):
        """A conversion metric requesting a CUPED method still needs the
        log-Normal delta method's ddof=1 variance - n=1 stays 'few_units',
        since the same retained row would otherwise feed the CUPED
        method's own log-Normal path too."""
        rows = [
            _conversion_arm_row(1, 1, country="US", group_id="control"),
            _conversion_arm_row(1, 1, country="US", group_id="treatment"),
        ]
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = run_breakout(
                rows,
                [_conversion_metric()],
                control_group="control",
                dimension="country",
                methods=[Method(name="cuped", variance_reduction="cuped")],
            )
        assert "breakout.estimates.row_few_units" in warning_codes(record)
        assert len(result) == 1
        assert result[0].excluded == "few_units"

    @pytest.mark.parametrize(("n", "successes"), [(1, 1), (50, 0)])
    def test_mixed_methods_keep_exact_result_and_exclude_only_cuped(self, n, successes):
        rows = [
            _conversion_arm_row(
                n, successes, country="US", group_id="control", with_covariate=True
            ),
            _conversion_arm_row(
                n, successes, country="US", group_id="treatment", with_covariate=True
            ),
        ]
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = run_breakout(
                rows,
                [_conversion_metric()],
                control_group="control",
                dimension="country",
                methods=[
                    Method(name="unadjusted"),
                    Method(name="cuped", variance_reduction="cuped"),
                ],
            )
        expected_code = (
            "breakout.estimates.row_few_units"
            if n == 1
            else "breakout.estimates.row_nonpositive_mean"
        )
        assert expected_code in warning_codes(record)
        by_method = {row.method: row for row in result}
        assert by_method["unadjusted"].reference_kind == "binomial"
        assert by_method["unadjusted"].binomial_set is not None
        assert by_method["unadjusted"].excluded is None
        assert by_method["cuped"].lift is None
        assert by_method["cuped"].excluded == ("few_units" if n == 1 else "nonpositive_mean")
        assert by_method["unadjusted"].method_role == "decision"
        assert by_method["cuped"].method_role == "sensitivity"

    def test_ratio_metric_n_equals_one_still_gated(self):
        """A ratio metric is never binomial-eligible - n=1 stays
        'few_units' exactly as before. Both arms carry a positive
        denominator so only the ddof=1 gate is under test."""
        rows = [
            _make_arm_row(
                50, 10.0, 4.0, country="US", group_id="control", metric="ratio_rev", sum_den=500.0
            ),
            _make_arm_row(
                1, 10.0, 0.0, country="US", group_id="treatment", metric="ratio_rev", sum_den=10.0
            ),
        ]
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = run_breakout(
                rows, [_ratio_metric("ratio_rev")], control_group="control", dimension="country"
            )
        assert "breakout.estimates.row_few_units" in warning_codes(record)
        assert len(result) == 1
        assert result[0].excluded == "few_units"


class TestFlattenedBinomialCrossInvariants:
    @staticmethod
    def _payload(model: type[BreakoutEstimate] | type[DailyLiftEstimate]) -> dict[str, Any]:
        rows = [
            _conversion_arm_row(20, 3, country="US", group_id="control"),
            _conversion_arm_row(20, 5, country="US", group_id="treatment"),
        ]
        breakout = run_breakout(
            rows, [_conversion_metric()], control_group="control", dimension="country"
        )[0]
        payload = breakout.model_dump()
        if model is DailyLiftEstimate:
            payload.pop("excluded")
            payload.pop("role")
            payload["ds"] = date(2025, 1, 1)
        return payload

    @pytest.mark.parametrize("model", [BreakoutEstimate, DailyLiftEstimate])
    @pytest.mark.parametrize("mutation", ["bounds", "level", "counts", "reference"])
    def test_deserialized_binomial_row_rejects_cross_field_contradictions(self, model, mutation):
        payload = self._payload(model)
        if mutation == "bounds":
            payload["lift"]["lb"] += 0.1
        elif mutation == "level":
            payload["lift"]["alpha"] = 0.1
            payload["lift"]["level"] = 0.9
        elif mutation == "counts":
            payload["lift"]["value"] += 0.1
        else:
            payload["reference_kind"] = "normal"
        with pytest.raises(InvalidRequestError):
            model.model_validate(payload)


# run_breakout returns BreakoutEstimates; EstimateList/to_frame() mechanics
# (schema-from-class, slicing/concat, model= inference) span every result type.


class TestRunBreakoutReturnsBreakoutEstimates:
    def test_run_breakout_returns_breakout_estimates_with_rows(self):
        rows = [
            _make_arm_row(10, 5.0, 4.0, country="US", group_id="control"),
            _make_arm_row(10, 6.0, 4.0, country="US", group_id="treatment"),
        ]
        result = run_breakout(rows, [_mean_metric()], control_group="control", dimension="country")
        assert isinstance(result, BreakoutEstimates)
        assert isinstance(result, list)
        assert isinstance(result.to_frame(backend="pandas"), pd.DataFrame)

    def test_run_breakout_returns_breakout_estimates_when_empty(self):
        """No segments at all (empty *summary*) still returns the concrete collection type, not a
        bare `[]` - `.to_frame()` must work on the empty-result path too."""
        result = run_breakout([], [_mean_metric()], control_group="control", dimension="country")
        assert isinstance(result, BreakoutEstimates)
        assert len(result.to_frame()) == 0


class TestEstimateListSlicingAndConcatPreserveSubclass:
    """Slicing and `+` concatenation keep `.to_frame()` working: both
    override `__getitem__`/`__add__` to rebuild the concrete subclass
    rather than falling back to a bare `list`."""

    def _sample(self) -> BreakoutEstimates:
        return BreakoutEstimates(
            [
                BreakoutEstimate(
                    metric="rev",
                    group_id="T",
                    method="unadjusted",
                    method_role="decision",
                    dimension="country",
                    dimension_value="US",
                    lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
                ),
                BreakoutEstimate(
                    metric="rev",
                    group_id="T",
                    method="unadjusted",
                    method_role="decision",
                    dimension="country",
                    dimension_value="CA",
                    lift=Estimate(value=0.2, lb=0.1, ub=0.3, level=0.95),
                ),
            ]
        )

    def test_int_index_returns_the_plain_element(self):
        assert isinstance(self._sample()[0], BreakoutEstimate)

    def test_slice_preserves_the_concrete_subclass(self):
        sliced = self._sample()[0:1]
        assert isinstance(sliced, BreakoutEstimates)
        assert isinstance(sliced.to_frame(backend="pandas"), pd.DataFrame)
        assert len(sliced) == 1

    def test_add_preserves_the_concrete_subclass(self):
        combined = self._sample() + self._sample()
        assert isinstance(combined, BreakoutEstimates)
        assert len(combined) == 4
        assert isinstance(combined.to_frame(backend="pandas"), pd.DataFrame)

    def test_is_a_plain_list_throughout(self):
        """Every existing `list[BreakoutEstimate]` consumer - isinstance checks, iteration,
        indexing, `len()` - keeps working unchanged, since it really is a `list`."""
        sample = self._sample()
        assert isinstance(sample, list)
        assert len(sample) == 2
        assert [r.dimension_value for r in sample] == ["US", "CA"]


class TestToFrameFreeFunction:
    """`to_frame()` is what `EstimateList.to_frame` delegates to: the escape hatch for a plain list
    assembled some other way (tests, deserialized data, a comprehension that lost the subclass)."""

    def test_infers_model_from_a_non_empty_sequence(self):
        estimates = [
            LiftEstimate(
                metric="rev",
                group_id="T",
                method="unadjusted",
                method_role="decision",
                lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
            )
        ]
        frame = to_frame(estimates)
        assert isinstance(frame, pd.DataFrame)
        assert frame.loc[0, "metric"] == "rev"
        assert frame.loc[0, "group_id"] == "T"
        assert frame.loc[0, ["lift", "lb", "ub"]].to_list() == pytest.approx([0.1, 0.05, 0.15])

    def test_empty_sequence_without_model_raises(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            to_frame([])
        assert exc_info.value.code == "breakout.to_frame_infer"

    def test_empty_sequence_with_model_returns_correct_schema(self):
        frame = to_frame([], model=LiftEstimate)
        populated = to_frame(
            [
                LiftEstimate(
                    metric="rev",
                    group_id="T",
                    method="unadjusted",
                    method_role="decision",
                    lift=Estimate(value=0.1),
                    ds=date(2026, 1, 1),
                )
            ]
        )
        assert isinstance(frame, pd.DataFrame) and isinstance(populated, pd.DataFrame)
        assert frame.empty
        assert frame.dtypes.equals(populated.dtypes)

    @pytest.mark.parametrize(
        ("estimates", "column", "expected"),
        [
            (
                BreakoutEstimates(
                    [
                        BreakoutEstimate(
                            metric="rev",
                            group_id="T",
                            method="unadjusted",
                            method_role="decision",
                            dimension="country",
                            dimension_value="US",
                            lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
                        )
                    ]
                ),
                "lift",
                0.1,
            ),
            (
                DailyMetricValues(
                    [
                        DailyMetricValue(
                            ds=date(2025, 1, 15),
                            metric="rev",
                            group_id="T",
                            value=Estimate(value=0.2, lb=0.1, ub=0.3, level=0.95),
                            n=100,
                        )
                    ]
                ),
                "value",
                0.2,
            ),
            (
                DailyLiftEstimates(
                    [
                        DailyLiftEstimate(
                            ds=date(2025, 1, 15),
                            metric="rev",
                            group_id="T",
                            method="unadjusted",
                            method_role="decision",
                            lift=Estimate(value=0.3, lb=0.2, ub=0.4, level=0.95),
                        )
                    ]
                ),
                "lift",
                0.3,
            ),
        ],
    )
    def test_pandas_flattened_estimate_columns_remain_numeric(self, estimates, column, expected):
        frame = cast(pd.DataFrame, estimates.to_frame(backend="pandas"))

        assert pd.api.types.is_float_dtype(frame[column])
        assert frame.loc[0, column] == pytest.approx(expected)

    @pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
    def test_breakout_reference_columns_are_typed_ordered_and_keep_numeric_nulls(self, backend):
        import narwhals as nw

        estimates = BreakoutEstimates(
            [
                BreakoutEstimate(
                    **TestBreakoutEstimateReferenceValidation._kwargs(
                        dof=None,
                        reference_kind="t",
                        reference_df=5.5,
                    )
                ),
                BreakoutEstimate(
                    **TestBreakoutEstimateReferenceValidation._kwargs(dimension_value="CA")
                ),
            ]
        )
        frame = nw.from_native(to_frame(estimates, backend=backend))

        dof_index = frame.columns.index("dof")
        assert frame.columns[dof_index : dof_index + 3] == [
            "dof",
            "reference_kind",
            "reference_df",
        ]
        assert frame.schema["reference_kind"] == nw.String()
        assert frame.schema["reference_df"] == nw.Float64()
        assert frame["reference_kind"].to_list() == ["t", "normal"]
        assert frame["reference_df"][0] == 5.5
        assert frame["reference_df"].is_null().to_list() == [False, True]

    @pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
    def test_populated_prior_spec_exports_as_a_string_on_every_backend(self, backend):
        # prior_spec is the first model-typed field to reach to_frame: it
        # must serialize to its repr, not leak a pydantic object pyarrow refuses to coerce.
        import narwhals as nw

        from increment.estimation.priors import MixturePrior, StudentTPrior

        estimates = [
            LiftEstimate(
                metric="rev",
                group_id="T",
                method="unadjusted",
                method_role="decision",
                lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
                prior_spec=StudentTPrior(nu=4.0, scale=0.05),
            ),
            LiftEstimate(
                metric="rev",
                group_id="T2",
                method="unadjusted",
                method_role="decision",
                lift=Estimate(value=0.2, lb=0.1, ub=0.3, level=0.95),
                prior_spec=MixturePrior(weights=(0.5, 0.5), means=(0.0, 0.0), sigmas=(0.01, 0.08)),
            ),
            LiftEstimate(
                metric="rev",
                group_id="C",
                method="unadjusted",
                method_role="decision",
                lift=Estimate(value=0.0, lb=-0.1, ub=0.1, level=0.95),
            ),
        ]
        column = nw.from_native(to_frame(estimates, backend=backend))["prior_spec"].to_list()
        assert column[0] == "StudentTPrior(nu=4.0, scale=0.05, k=80)"
        assert column[1].startswith("MixturePrior(")
        assert bool(pd.isna(column[2]))

    @pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
    def test_optional_strings_keep_native_nulls_across_backends(self, backend):
        estimates = BreakoutEstimates(
            [
                BreakoutEstimate(
                    metric="rev",
                    group_id="T",
                    method="unadjusted",
                    method_role="decision",
                    dimension="country",
                    dimension_value="US",
                    lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
                ),
                BreakoutEstimate(
                    metric="rev",
                    group_id="T",
                    method="unadjusted",
                    method_role="decision",
                    dimension="country",
                    dimension_value="CA",
                    source="events",
                    note="guarded",
                    role="exploratory",
                    lift=Estimate(value=0.2, lb=0.1, ub=0.3, level=0.95),
                ),
            ]
        )
        frame = cast(Any, to_frame(estimates, backend=backend))

        if backend == "pandas":
            assert frame.loc[0, "source"] is pd.NA
            assert frame.loc[0, "note"] is pd.NA
            assert frame.loc[0, "role"] is pd.NA
            assert frame.loc[1, "source"] == "events"
        elif backend == "polars":
            assert frame["source"].to_list() == [None, "events"]
            assert frame["note"].to_list() == [None, "guarded"]
            assert frame["role"].to_list() == [None, "exploratory"]
        else:
            assert frame.column("source").to_pylist() == [None, "events"]
            assert frame.column("note").to_pylist() == [None, "guarded"]
            assert frame.column("role").to_pylist() == [None, "exploratory"]

    @pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
    def test_ds_stays_nullable_numeric_when_null_and_int_rows_mix(self, backend):
        """An as-of row's numeric `ds` (e.g. a sequential reveal cursor)
        sharing a frame with a null-`ds` total-grain row must not crash:
        the derived dtype has to tolerate the null cell, so an int day
        type maps to Float64 here exactly as every other optional int
        column already does."""
        import narwhals as nw

        rows = [
            LiftEstimate(
                metric="rev",
                group_id="T",
                method="unadjusted",
                method_role="decision",
                lift=Estimate(value=0.1),
                ds=None,
            ),
            LiftEstimate(
                metric="rev",
                group_id="T",
                method="unadjusted",
                method_role="decision",
                lift=Estimate(value=0.2),
                ds=3,
            ),
        ]
        frame = nw.from_native(to_frame(rows, backend=backend))
        assert frame.schema["ds"] == nw.Float64()
        assert frame["ds"].to_list()[1] == pytest.approx(3.0)
        assert frame["ds"].is_null().to_list() == [True, False]

    def test_mismatched_model_raises(self):
        """An explicit `model=` that disagrees with the actual runtime type of a non-empty sequence
        raises rather than silently building the wrong schema."""
        estimates = [
            LiftEstimate(
                metric="rev",
                group_id="T",
                method="unadjusted",
                method_role="decision",
                lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
            )
        ]
        with pytest.raises(InvalidRequestError) as exc_info:
            to_frame(estimates, model=BreakoutEstimate)
        assert exc_info.value.code == "breakout.to_frame_model"
        assert exc_info.value.context["model"] == "BreakoutEstimate"
        assert exc_info.value.context["inferred"] == "LiftEstimate"

    def test_works_directly_on_lift_estimates_collection(self):
        """LiftEstimates gets the exact same to_frame() as the day-axis types - the mixin logic is
        generic across every result model, not day-axis-specific despite the historical name."""
        estimates = LiftEstimates(
            [
                LiftEstimate(
                    metric="rev",
                    group_id="T",
                    method="unadjusted",
                    method_role="decision",
                    lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
                )
            ]
        )
        frame = cast(pd.DataFrame, estimates.to_frame(backend="pandas"))
        assert frame.loc[0, "lift"] == pytest.approx(0.1)

    def test_works_directly_on_daily_lift_estimates_collection(self):
        """DailyLiftEstimates (the day-axis type) via the same public
        entry point, for symmetry with the LiftEstimates case above."""
        estimates = DailyLiftEstimates(
            [
                DailyLiftEstimate(
                    metric="rev",
                    group_id="T",
                    method="unadjusted",
                    method_role="decision",
                    ds=date(2025, 1, 1),
                    lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
                )
            ]
        )
        frame = cast(pd.DataFrame, estimates.to_frame(backend="pandas"))
        assert frame.loc[0, "lift"] == pytest.approx(0.1)

    def test_estimate_list_is_a_generic_base_not_a_public_construction_type(self):
        """EstimateList itself has no `_model` set - concrete subclasses (BreakoutEstimates,
        LiftEstimates, DailyMetricValues, DailyLiftEstimates) are the only ones meant to be
        constructed directly."""
        assert not hasattr(EstimateList, "_model")


@pytest.mark.parametrize("rejects", [True, False])
def test_small_bernoulli_segment_uses_its_sequential_decision(rejects):
    from increment.estimators import SequentialCell
    from increment.tables import estimates_to_readout
    from tests.sequential_cases import capture, records, registration

    reg = registration(
        cells=(
            SequentialCell(metric="outcome", group_id="treatment", segment=(("country", "US"),)),
        )
    )
    control = [0] * 20 if rejects else [0, 1] * 10
    treatment = [1] * 20 if rejects else [0, 1] * 10
    snapshot = capture(reg, records(control, treatment, segments={"country": "US"}))
    result = run_breakout(
        snapshot,
        [ConversionMetric(name="outcome", entity="unit", fact="converted")],
        control_group="control",
        dimension="country",
        inference=AlwaysValid(registration=reg),
    )
    (row,) = result
    assert row.low_reliability is True
    assert row.sequential_result is not None and row.sequential_result.rejects() is rejects
    assert row.stat_sig() is rejects
    (displayed,) = estimates_to_readout(result)
    assert displayed["stat_sig"] is rejects
    assert displayed["decision_available"] is True
