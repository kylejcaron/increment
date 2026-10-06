"""Tests for `increment.breakout.estimates`'s day-axis additions:
`run_daily` (absolute per-arm daily means) and `run_daily_lift` (relative
lift per day).

`run_daily_lift` is `run_breakout` with `ds` as the partition key: it
inherits `estimate_lift`'s control-arm lookup hazard (a dict keyed only
by metric name in `increment.estimation.engine`) - calling it once
on a summary mixing two days' control rows for one metric would silently
drop a day's control. `TestRunDailyLiftDayIsolation` regression-tests
this, mirroring `TestRunBreakoutSegmentIsolation`.

`run_daily` never calls `estimate_lift` but routes each row through the
same `VARIANCE_MODELS` dispatch; `TestRunDailyRatioMetric` proves a
ratio metric's value is `sum_y / sum_den`, not `ArmStats.to_summary()`'s
numerator-only mean `sum_y / n`."""

from __future__ import annotations

import math
import warnings
from datetime import date
from fractions import Fraction
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from scipy.stats import norm

from increment.breakout.estimates import (
    DailyLiftEstimate,
    DailyLiftEstimates,
    DailyMetricValue,
    DailyMetricValues,
    run_daily,
    run_daily_lift,
    to_frame,
)
from increment.errors import IncrementWarning, InvalidRequestError, UnsupportedRequestError
from increment.estimation.armstats import ArmStats, centered_row_from_raw_sums
from increment.estimation.encouragement import estimate_encouragement
from increment.estimation.engine import Method, estimate_lift
from increment.estimation.results import Estimate
from increment.estimation.sequential import AlwaysValid
from increment.estimation.variance import RatioVarianceModel
from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec
from increment.semantics.models import (
    ConversionMetric,
    MeanMetric,
    Measure,
    RatioMetric,
    RetentionMetric,
)
from tests.sequential_cases import registration
from tests.warning_codes import warning_codes

# Helpers (ds column stands in for the breakout dimension)


def _alpha_split(alpha: float, divisor: int) -> float:
    """The greatest float at or below ``alpha / divisor`` - the readout's own
    conservative split of a plan alpha, which never rounds upward."""
    exact = Fraction(alpha) / divisor
    quotient = float(exact)
    return math.nextafter(quotient, 0.0) if Fraction(quotient) > exact else quotient


def _make_daily_row(
    n: int,
    mean: float,
    var: float,
    *,
    ds: date,
    experiment_id: str = "exp1",
    metric: str = "rev",
    group_id: str = "control",
) -> dict:
    """Build one centered `daily_group_summary` row from moments - same
    mean/var inputs as test_estimates.py's `_make_arm_row`, with a `ds`
    column."""
    sum_y = float(n) * mean
    sum_y2 = var * (n - 1) + sum_y**2 / float(n)
    return centered_row_from_raw_sums(
        {
            "ds": ds,
            "experiment_id": experiment_id,
            "metric": metric,
            "group_id": group_id,
            "n": float(n),
            "sum_y": sum_y,
            "sum_y2": sum_y2,
            "sum_x": None,
            "sum_x2": None,
            "sum_xy": None,
            "sum_den": None,
            "sum_den2": None,
            "sum_yden": None,
        }
    )


def _mean_metric(name: str = "rev") -> MeanMetric:
    """Build a MeanMetric fixture for tests using mean metrics."""
    return MeanMetric(name=name, entity="user", fact=name)


def _windowed_mean_metric(name: str = "rev") -> MeanMetric:
    """A closed-horizon MeanMetric (`window_days` set) - keeps the
    always-valid inference tests free of the unrelated open-ended-metric
    warning."""
    return MeanMetric(name=name, entity="user", fact=name, window_days=30)


def _ratio_metric(name: str = "rev_per_session") -> RatioMetric:
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


def _conversion_daily_row(
    n: int,
    successes: int,
    *,
    ds: date,
    group_id: str,
    metric: str = "conv",
    with_covariate: bool = False,
) -> dict:
    """A centered `daily_group_summary` row for a genuinely binary (0/1)
    conversion arm, `successes` out of `n` - the exact binomial risk-ratio method
    admits any (n, successes) pair, including n=1 and successes=0."""
    return centered_row_from_raw_sums(
        {
            "ds": ds,
            "experiment_id": "exp1",
            "metric": metric,
            "group_id": group_id,
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


def _retention_metric(name: str = "d7_retention") -> RetentionMetric:
    """Build a RetentionMetric fixture for tests using retention metrics."""
    return RetentionMetric(name=name, entity="user", fact="page_view", threshold_days=7)


def _bounded_retention_metric(name: str = "d7_retention") -> RetentionMetric:
    """Retention with a real observation band - reportable on a day axis."""
    return RetentionMetric(
        name=name,
        entity="user",
        fact="page_view",
        threshold_days=(7, 14),
    )


def _encouragement_design(**over) -> Encouragement:
    """Build an Encouragement fixture - mirrors
    tests/test_readouts_encouragement.py's `_design` helper."""
    base = {
        "control_group": "control",
        "uptake": UptakeSpec(fact="clicked"),
        "exclusion_restriction": ExclusionRestriction(
            acknowledged=True, justification="unclicked button assumed inert"
        ),
        "one_sided": True,
    }
    base.update(over)
    return Encouragement.model_validate(base)


def _encouragement_day_rows(
    *,
    ds: date,
    n: int,
    tau: float,
    compliance: float,
    seed: int,
    metric: str = "rev",
    n_control: int | None = None,
) -> list[dict]:
    """One as-of day's (control, treatment) `asof_group_summary` rows for a
    one-sided encouragement DGP, with `sum_d`/`sum_yd`/`sum_y2d` populated
    for `estimate_encouragement`'s first-stage/LATE reads. `n_control`
    builds asymmetric arms (defaults to `n`)."""
    rng = np.random.default_rng(seed)
    rows = []
    for group_id, encouraged in (("control", 0), ("treatment", 1)):
        arm_n = n if encouraged else (n_control if n_control is not None else n)
        d = rng.binomial(1, compliance, size=arm_n) if encouraged else np.zeros(arm_n)
        y = 10.0 + tau * d + rng.normal(0, 2.0, size=arm_n)
        yd = y * d
        rows.append(
            centered_row_from_raw_sums(
                {
                    "ds": ds,
                    "experiment_id": "s",
                    "metric": metric,
                    "group_id": group_id,
                    "n": float(arm_n),
                    "sum_y": float(y.sum()),
                    "sum_y2": float((y**2).sum()),
                    "sum_x": None,
                    "sum_x2": None,
                    "sum_xy": None,
                    "sum_den": None,
                    "sum_den2": None,
                    "sum_yden": None,
                    "sum_d": float(d.sum()),
                    "sum_yd": float(yd.sum()),
                    "sum_y2d": float((y**2 * d).sum()),
                }
            )
        )
    return rows


def _two_metric_encouragement_rows():
    metrics = [_mean_metric("rev"), _mean_metric("orders")]
    rows = [
        row
        for metric, seed in zip(metrics, (7, 8), strict=True)
        for row in _encouragement_day_rows(
            ds=date(2025, 1, 1),
            n=4000,
            tau=2.0,
            compliance=0.5,
            seed=seed,
            metric=metric.name,
        )
    ]
    return metrics, rows


# run_daily: correctness against hand-computed means/CIs


class TestRunDailyCorrectness:
    def test_two_day_two_arm_matches_hand_computed_mean_and_ci(self):
        """2 days x 2 arms -> 4 rows, CI checked against an independently
        computed log-scale Wald interval, not the implementation's own
        formula."""
        rows = [
            _make_daily_row(n=10, mean=5.0, var=4.0, ds=date(2025, 1, 1), group_id="control"),
            _make_daily_row(n=10, mean=6.0, var=4.0, ds=date(2025, 1, 1), group_id="treatment"),
            _make_daily_row(n=20, mean=5.5, var=9.0, ds=date(2025, 1, 2), group_id="control"),
            _make_daily_row(n=20, mean=7.0, var=9.0, ds=date(2025, 1, 2), group_id="treatment"),
        ]

        result = run_daily(rows, [_mean_metric()], alpha=0.05)

        assert len(result) == 4
        by_key = {(r.ds, r.group_id): r for r in result}
        assert set(by_key) == {
            (date(2025, 1, 1), "control"),
            (date(2025, 1, 1), "treatment"),
            (date(2025, 1, 2), "control"),
            (date(2025, 1, 2), "treatment"),
        }

        z = norm.ppf(0.975)
        expected = {
            (date(2025, 1, 1), "control"): (5.0, 4.0, 10),
            (date(2025, 1, 1), "treatment"): (6.0, 4.0, 10),
            (date(2025, 1, 2), "control"): (5.5, 9.0, 20),
            (date(2025, 1, 2), "treatment"): (7.0, 9.0, 20),
        }
        for key, (mean, var, n) in expected.items():
            r = by_key[key]
            value = r.value
            assert value is not None
            se_log = math.sqrt(var / (n * mean**2))
            assert r.metric == "rev"
            assert r.n == n
            assert value.value == pytest.approx(mean)
            assert value.lb == pytest.approx(math.exp(math.log(mean) - z * se_log))
            assert value.ub == pytest.approx(math.exp(math.log(mean) + z * se_log))
            assert value.level == pytest.approx(0.95)

    def test_alpha_widens_or_narrows_the_interval(self):
        """A smaller alpha (wider CI) must produce a wider interval -
        confirms `alpha` reaches the z-value, not decorative."""
        rows = [_make_daily_row(n=50, mean=10.0, var=4.0, ds=date(2025, 1, 1))]

        wide = run_daily(rows, [_mean_metric()], alpha=0.01)[0]  # 99% CI
        narrow = run_daily(rows, [_mean_metric()], alpha=0.20)[0]  # 80% CI

        wide_value = wide.value
        narrow_value = narrow.value
        assert wide_value is not None
        assert narrow_value is not None
        assert wide_value.lb is not None and wide_value.ub is not None
        assert narrow_value.lb is not None and narrow_value.ub is not None
        assert (wide_value.ub - wide_value.lb) > (narrow_value.ub - narrow_value.lb)


# run_daily: ratio metric value is sum_y/sum_den, not
# ArmStats.to_summary()'s numerator-only mean sum_y/n


class TestRunDailyRatioMetric:
    def test_ratio_metric_value_is_numerator_over_denominator(self):
        """Regression: a RatioMetric row's DailyMetricValue.value must be
        sum_y / sum_den (the ratio), NOT sum_y / n (the raw numerator
        mean the old ArmStats.to_summary() path silently returned)."""
        n = 3
        sum_y, sum_y2 = 60.0, 1400.0  # per-unit numerator: 10, 20, 30
        sum_den, sum_den2 = 12.0, 56.0  # per-unit denominator: 2, 4, 6
        sum_yden = 280.0  # 10*2 + 20*4 + 30*6
        row = centered_row_from_raw_sums(
            {
                "ds": date(2025, 1, 1),
                "experiment_id": "exp1",
                "metric": "rev_per_session",
                "group_id": "control",
                "n": n,
                "sum_y": sum_y,
                "sum_y2": sum_y2,
                "sum_x": None,
                "sum_x2": None,
                "sum_xy": None,
                "sum_den": sum_den,
                "sum_den2": sum_den2,
                "sum_yden": sum_yden,
            }
        )

        result = run_daily([row], [_ratio_metric()], alpha=0.05)

        assert len(result) == 1
        r = result[0]

        ratio = sum_y / sum_den  # correct "metric value" for a ratio metric
        numerator_mean = sum_y / n  # what the old ArmStats.to_summary() bug returned
        assert ratio != pytest.approx(numerator_mean), (
            "test is only meaningful if ratio and numerator-mean differ"
        )
        value = r.value
        assert value is not None
        assert value.value == pytest.approx(ratio)
        assert r.metric == "rev_per_session"
        assert r.n == n

        # Cross-checks the CI against RatioVarianceModel.log_mean_se on the
        # same ArmStats - run_daily calls the identical function estimate_lift uses.
        arm = ArmStats.from_raw_sums(
            study_id="exp1",
            metric="rev_per_session",
            group_id="control",
            n=n,
            sum_y=sum_y,
            sum_y2=sum_y2,
            sum_den=sum_den,
            sum_den2=sum_den2,
            sum_yden=sum_yden,
        )
        log_mean, se = RatioVarianceModel().log_mean_se(arm)
        z = norm.ppf(0.975)
        assert value.value == pytest.approx(math.exp(log_mean))
        assert value.lb == pytest.approx(math.exp(log_mean - z * se))
        assert value.ub == pytest.approx(math.exp(log_mean + z * se))
        assert value.level == pytest.approx(0.95)


# run_daily: conditions it may swallow (n < 2, or a non-positive mean -
# log(mean) is undefined for zero/negative values, both variance models)


class TestRunDailyUnavailableRows:
    """Unestimable daily cells retain their row identity and reason."""

    def test_n_equals_one_row_returns_nan_not_dropped(self):
        """A day with only 1 unit for some (metric, group_id) has no
        within-day spread for a ddof=1 variance - the row comes back with
        a NaN value and its real n, not omitted."""
        rows = [
            _make_daily_row(n=10, mean=5.0, var=4.0, ds=date(2025, 1, 1), group_id="control"),
            _make_daily_row(n=1, mean=8.0, var=0.0, ds=date(2025, 1, 1), group_id="treatment"),
        ]

        result = run_daily(rows, [_mean_metric()])

        assert len(result) == 2
        by_arm = {r.group_id: r for r in result}
        assert by_arm["control"].value is not None
        assert by_arm["treatment"].value is None
        assert by_arm["treatment"].unavailable == "few_units"
        assert by_arm["treatment"].n == 1, "real n is preserved on an unavailable row"

    def test_n_equals_one_row_emits_no_warning(self):
        """D2: the NaN is the whole signal - no UserWarning accompanies it."""
        rows = [
            _make_daily_row(n=1, mean=8.0, var=0.0, ds=date(2025, 1, 1), group_id="treatment"),
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any UserWarning becomes a failure
            result = run_daily(rows, [_mean_metric()])

        assert len(result) == 1
        assert result[0].value is None

    def test_n_equals_zero_row_returns_nan_row(self):
        """D5: n=0 is not special-cased. (A real daily_group_summary can't
        emit n=0 - it's a group_by(...).agg(n=count()) - so this is a
        synthetic row proving the guard is uniform.)"""
        row = _make_daily_row(n=1, mean=0.0, var=0.0, ds=date(2025, 1, 1), group_id="control")
        row["n"] = 0.0

        result = run_daily([row], [_mean_metric()])

        assert len(result) == 1
        assert result[0].value is None
        assert result[0].n == 0

    def test_zero_mean_row_returns_nan_not_dropped(self):
        """sum_y == 0 - e.g. a conversion metric on a day nobody in that
        arm converted - has an undefined log(mean). n=10 (not < 2)
        isolates this from the variance guard."""
        rows = [
            _make_daily_row(n=10, mean=5.0, var=4.0, ds=date(2025, 1, 1), group_id="control"),
            _make_daily_row(n=10, mean=0.0, var=0.0, ds=date(2025, 1, 1), group_id="treatment"),
        ]

        result = run_daily(rows, [_mean_metric()])

        assert len(result) == 2
        by_arm = {r.group_id: r for r in result}
        assert by_arm["treatment"].value is None
        assert by_arm["treatment"].n == 10, "n=10 proves this isn't the n<2 path"

    def test_negative_mean_row_returns_nan_not_raising(self):
        """sum_y < 0 (e.g. net revenue after refunds) is also outside
        log's domain - math.log is undefined on the whole non-positive
        half-line, not just zero, so this must not raise `ValueError:
        math domain error`."""
        rows = [
            _make_daily_row(n=10, mean=5.0, var=4.0, ds=date(2025, 1, 1), group_id="control"),
            _make_daily_row(n=10, mean=-3.0, var=4.0, ds=date(2025, 1, 1), group_id="treatment"),
        ]

        result = run_daily(rows, [_mean_metric()])

        assert len(result) == 2
        by_arm = {r.group_id: r for r in result}
        assert by_arm["treatment"].value is None

    def test_estimable_rows_are_completely_unaffected(self):
        """Anti-drift: a fully estimable input returns exactly what it did
        before NaN rows existed - same count, same values, no NaN."""
        rows = [
            _make_daily_row(n=100, mean=5.0, var=4.0, ds=date(2025, 1, 1), group_id="control"),
            _make_daily_row(n=100, mean=6.0, var=4.0, ds=date(2025, 1, 1), group_id="treatment"),
        ]

        result = run_daily(rows, [_mean_metric()])

        assert len(result) == 2
        for r in result:
            assert r.value is not None
            assert r.value.lb is not None and not math.isnan(r.value.lb)
            assert r.value.ub is not None and not math.isnan(r.value.ub)


# run_daily: input format equivalence + defensive errors (mirrors
# TestRunBreakoutInputFormats)


class TestRunDailyInputFormats:
    def test_pandas_pyarrow_and_dict_rows_agree(self):
        rows = [
            _make_daily_row(n=100, mean=10.0, var=4.0, ds=date(2025, 1, 1), group_id="control"),
            _make_daily_row(n=100, mean=11.0, var=4.0, ds=date(2025, 1, 1), group_id="treatment"),
        ]

        pandas_result = run_daily(pd.DataFrame(rows), [_mean_metric()])
        pyarrow_result = run_daily(pa.Table.from_pylist(rows), [_mean_metric()])
        dict_result = run_daily(rows, [_mean_metric()])

        def _by_key(results: list[DailyMetricValue]) -> dict:
            out = {}
            for r in results:
                value = r.value
                assert value is not None
                out[(r.ds, r.group_id)] = (value.value, value.lb, value.ub)
            return out

        assert len(pandas_result) == len(pyarrow_result) == len(dict_result) == 2
        assert _by_key(pandas_result) == _by_key(pyarrow_result) == _by_key(dict_result)

    def test_dataframe_missing_required_column_raises(self):
        row = _make_daily_row(n=100, mean=10.0, var=4.0, ds=date(2025, 1, 1))
        df = pd.DataFrame([row]).drop(columns=["ds"])

        with pytest.raises(InvalidRequestError) as exc_info:
            run_daily(df, [_mean_metric()])
        assert exc_info.value.code == "breakout.daily_group_summary"
        assert exc_info.value.context["missing"] == ("ds",)

    def test_dict_rows_missing_required_key_raises(self):
        row = _make_daily_row(n=100, mean=10.0, var=4.0, ds=date(2025, 1, 1))
        del row["ds"]

        with pytest.raises(KeyError):
            run_daily([row], [_mean_metric()])

    def test_unsupported_ds_type_raises_coded_error(self):
        row = _make_daily_row(n=10, mean=5.0, var=4.0, ds=date(2025, 1, 1))
        row["ds"] = 12345  # neither datetime, date, nor str

        with pytest.raises(InvalidRequestError) as exc_info:
            run_daily([row], [_mean_metric()])
        assert exc_info.value.code == "breakout.coerce_type_datetime"
        assert exc_info.value.context["v"] == 12345
        assert exc_info.value.context["type_name"] == "int"

    def test_alpha_outside_open_unit_interval_raises(self):
        row = _make_daily_row(n=10, mean=5.0, var=4.0, ds=date(2025, 1, 1))

        with pytest.raises(InvalidRequestError) as exc_info:
            run_daily([row], [_mean_metric()], alpha=1.5)
        assert exc_info.value.code == "estimation.diagnostics.alpha"
        assert exc_info.value.context["alpha"] == 1.5

    def test_alpha_that_underflows_on_halving_raises(self):
        row = _make_daily_row(n=10, mean=5.0, var=4.0, ds=date(2025, 1, 1))

        with pytest.raises(InvalidRequestError) as exc_info:
            run_daily([row], [_mean_metric()], alpha=5e-324)
        assert exc_info.value.code == "estimation.meta.alpha_too_small"


# run_daily: dimension/source stamping (mirrors run_breakout's own
# TestRunBreakoutShape/TestRunBreakoutInputFormats dimension tests)


class TestRunDailyDimensionStamping:
    def test_dimension_and_source_stamped_onto_every_result(self):
        rows = [
            _make_daily_row(n=10, mean=5.0, var=4.0, ds=date(2025, 1, 1), group_id="control"),
            _make_daily_row(n=10, mean=6.0, var=4.0, ds=date(2025, 1, 1), group_id="treatment"),
        ]
        for row in rows:
            row["country"] = "US"

        result = run_daily(rows, [_mean_metric()], dimension="country", source="events")

        assert len(result) == 2
        for r in result:
            assert r.dimension == "country"
            assert r.dimension_value == "US"
            assert r.source == "events"

    def test_dimension_value_is_string_coerced(self):
        """Mirrors run_breakout's `str(row[dimension])` convention: a
        non-string dimension value still comes back a str."""
        row = _make_daily_row(n=10, mean=5.0, var=4.0, ds=date(2025, 1, 1))
        row["plan_tier"] = 3  # int-typed property value

        result = run_daily([row], [_mean_metric()], dimension="plan_tier")

        assert result[0].dimension_value == "3"
        assert isinstance(result[0].dimension_value, str)

    def test_no_dimension_leaves_new_fields_none(self):
        """The un-dimensioned call (dimension=None, the default) must not
        populate dimension/dimension_value/source at all, whether the
        default is implicit or passed explicitly."""
        rows = [_make_daily_row(n=10, mean=5.0, var=4.0, ds=date(2025, 1, 1))]

        implicit = run_daily(rows, [_mean_metric()])
        explicit = run_daily(rows, [_mean_metric()], dimension=None)

        for result in (implicit, explicit):
            assert result[0].dimension is None
            assert result[0].dimension_value is None
            assert result[0].source is None

    def test_dimension_column_missing_from_dataframe_raises(self):
        """A DataFrame-shaped summary missing the requested dimension
        column raises a clear ValueError."""
        row = _make_daily_row(n=100, mean=10.0, var=4.0, ds=date(2025, 1, 1))
        df = pd.DataFrame([row])  # no "country" column at all

        with pytest.raises(InvalidRequestError) as exc_info:
            run_daily(df, [_mean_metric()], dimension="country")
        assert exc_info.value.code == "breakout.daily_group_summary"
        assert exc_info.value.context["missing"] == ("country",)

    def test_dict_rows_missing_dimension_key_raises(self):
        """The plain-mapping path can't pre-validate without a schema, so
        a missing dimension key fails at first row access with a KeyError."""
        row = _make_daily_row(n=100, mean=10.0, var=4.0, ds=date(2025, 1, 1))

        with pytest.raises(KeyError):
            run_daily([row], [_mean_metric()], dimension="country")


# DailyMetricValue model (mirrors TestBreakoutEstimate)


class TestDailyMetricValue:
    def test_frozen(self):
        val = DailyMetricValue(
            ds=date(2025, 1, 1),
            metric="rev",
            group_id="control",
            value=Estimate(value=10.0, lb=9.0, ub=11.0, level=0.95),
            n=100,
        )
        with pytest.raises((TypeError, ValueError)):
            val.n = 200  # ty: ignore[invalid-assignment]  # proving frozen at runtime

    def test_missing_required_field_raises(self):
        with pytest.raises(InvalidRequestError) as raised:
            DailyMetricValue(
                metric="rev",
                group_id="control",
                value=Estimate(value=10.0),
            )  # ty: ignore[missing-argument]  # proving `ds`/`n` are required
        assert raised.value.code == "model.field.missing"

    def test_dimension_fields_default_to_none(self):
        val = DailyMetricValue(
            ds=date(2025, 1, 1),
            metric="rev",
            group_id="control",
            value=Estimate(value=10.0, lb=9.0, ub=11.0, level=0.95),
            n=100,
        )
        assert val.dimension is None
        assert val.dimension_value is None
        assert val.source is None

    def test_dimension_fields_can_be_set(self):
        val = DailyMetricValue(
            ds=date(2025, 1, 1),
            metric="rev",
            group_id="control",
            value=Estimate(value=10.0, lb=9.0, ub=11.0, level=0.95),
            n=100,
            dimension="country",
            dimension_value="US",
            source="events",
        )
        assert val.dimension == "country"
        assert val.dimension_value == "US"
        assert val.source == "events"

    def test_neither_value_nor_unavailable_set_raises_coded_error(self):
        """The buried coded refusal must surface with its own `.code`, not a
        bare pydantic `ValidationError`."""
        with pytest.raises(InvalidRequestError) as exc_info:
            DailyMetricValue(
                ds=date(2025, 1, 1),
                metric="rev",
                group_id="control",
                value=None,
                n=100,
            )
        assert exc_info.value.code == "breakout.daily_metric.exactly_one_value"


# run_daily_lift: CRITICAL - never call estimate_lift on a multi-day
# summary at once (mirrors TestRunBreakoutSegmentIsolation)


class TestRunDailyLiftDayIsolation:
    def test_two_days_with_different_control_means_both_correct(self):
        """Day 1: control=10, treatment=12 (positive lift). Day 2:
        control=20, treatment=19 (negative lift). Opposite-signed lift with
        a 2x-different control mean catches a `{metric: control}` dict-key
        collision that would keep only one day's control arm."""
        d1_control = _make_daily_row(
            n=2000, mean=10.0, var=4.0, ds=date(2025, 1, 1), group_id="control"
        )
        d1_treatment = _make_daily_row(
            n=2000, mean=12.0, var=4.0, ds=date(2025, 1, 1), group_id="treatment"
        )
        d2_control = _make_daily_row(
            n=1500, mean=20.0, var=4.0, ds=date(2025, 1, 2), group_id="control"
        )
        d2_treatment = _make_daily_row(
            n=1500, mean=19.0, var=4.0, ds=date(2025, 1, 2), group_id="treatment"
        )
        summary = pd.DataFrame([d1_control, d1_treatment, d2_control, d2_treatment])

        results = run_daily_lift(summary, [_mean_metric()], control_group="control")

        assert len(results) == 2
        for r in results:
            assert isinstance(r, DailyLiftEstimate)
            assert r.metric == "rev"
            assert r.group_id == "treatment"

        by_day = {r.ds: r for r in results}
        assert set(by_day) == {date(2025, 1, 1), date(2025, 1, 2)}

        # Ground truth: estimate_lift on just that day's own two rows.
        d1_expected = estimate_lift(
            metrics=[_mean_metric()],
            summary=pd.DataFrame([d1_control, d1_treatment]),
            control_group="control",
        ).results[0]
        d2_expected = estimate_lift(
            metrics=[_mean_metric()],
            summary=pd.DataFrame([d2_control, d2_treatment]),
            control_group="control",
        ).results[0]

        d1_lift = by_day[date(2025, 1, 1)].lift
        d2_lift = by_day[date(2025, 1, 2)].lift
        d1_expected_lift = d1_expected.lift
        d2_expected_lift = d2_expected.lift
        assert d1_lift is not None and d1_expected_lift is not None
        assert d2_lift is not None and d2_expected_lift is not None
        assert d1_lift.value == d1_expected_lift.value
        assert d1_lift.lb == d1_expected_lift.lb
        assert d1_lift.ub == d1_expected_lift.ub
        assert d2_lift.value == d2_expected_lift.value
        assert d2_lift.lb == d2_expected_lift.lb
        assert d2_lift.ub == d2_expected_lift.ub

        assert d1_lift.value > 0, "day 1 treatment (12) > control (10)"
        assert d2_lift.value < 0, "day 2 treatment (19) < control (20)"

    def test_row_order_does_not_change_either_days_answer(self):
        """Regression guard: feeding day 2's rows before day 1's rows (or
        vice versa) must not change either day's answer."""
        d1_control = _make_daily_row(
            n=2000, mean=10.0, var=4.0, ds=date(2025, 1, 1), group_id="control"
        )
        d1_treatment = _make_daily_row(
            n=2000, mean=12.0, var=4.0, ds=date(2025, 1, 1), group_id="treatment"
        )
        d2_control = _make_daily_row(
            n=1500, mean=20.0, var=4.0, ds=date(2025, 1, 2), group_id="control"
        )
        d2_treatment = _make_daily_row(
            n=1500, mean=19.0, var=4.0, ds=date(2025, 1, 2), group_id="treatment"
        )

        forward = run_daily_lift(
            [d1_control, d1_treatment, d2_control, d2_treatment],
            [_mean_metric()],
            control_group="control",
        )
        backward = run_daily_lift(
            [d2_control, d2_treatment, d1_control, d1_treatment],
            [_mean_metric()],
            control_group="control",
        )

        forward_by_day = {}
        for r in forward:
            lift = r.lift
            assert lift is not None
            forward_by_day[r.ds] = lift.value
        backward_by_day = {}
        for r in backward:
            lift = r.lift
            assert lift is not None
            backward_by_day[r.ds] = lift.value
        assert forward_by_day == backward_by_day


class TestRunDailyLiftSegmentIsolation:
    """`run_daily_lift(dimension=...)` must partition by the (day, segment)
    pair, not by day alone: two segments' control rows for the same metric
    on the same day would otherwise land in one `estimate_lift` call whose
    `{metric: control}` dict keeps only whichever it iterated last,
    comparing both segments' treatment arms against one segment's control."""

    def test_two_segments_on_the_same_day_both_correct(self):
        """One day, two segments. US: control=10, treatment=12 (positive
        lift). CA: control=20, treatment=19 (negative lift) - a day-only
        partition would show up as a wrong SIGN, not a slightly-off number."""
        ds = date(2025, 1, 1)
        us_control = _make_daily_row(n=2000, mean=10.0, var=4.0, ds=ds, group_id="control")
        us_treatment = _make_daily_row(n=2000, mean=12.0, var=4.0, ds=ds, group_id="treatment")
        ca_control = _make_daily_row(n=1500, mean=20.0, var=4.0, ds=ds, group_id="control")
        ca_treatment = _make_daily_row(n=1500, mean=19.0, var=4.0, ds=ds, group_id="treatment")
        for row in (us_control, us_treatment):
            row["country"] = "US"
        for row in (ca_control, ca_treatment):
            row["country"] = "CA"
        summary = pd.DataFrame([us_control, us_treatment, ca_control, ca_treatment])

        results = run_daily_lift(
            summary, [_mean_metric()], control_group="control", dimension="country"
        )

        assert len(results) == 2
        by_segment = {r.dimension_value: r for r in results}
        assert set(by_segment) == {"US", "CA"}

        # Ground truth: estimate_lift on just that segment's own two rows.
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

        assert us_lift.value > 0, "US treatment (12) > control (10)"
        assert ca_lift.value < 0, "CA treatment (19) < control (20)"

    def test_day_and_segment_partition_jointly(self):
        """2 days x 2 segments = 4 independent slices, each estimated on
        its own 2 rows - neither axis alone is a sufficient partition."""
        rows = []
        expected: dict[tuple[date, str], float] = {}
        # (day, segment) -> (control mean, treatment mean), all four
        # deliberately different so no two slices can be confused.
        spec = {
            (date(2025, 1, 1), "US"): (10.0, 12.0),
            (date(2025, 1, 1), "CA"): (20.0, 19.0),
            (date(2025, 1, 2), "US"): (30.0, 36.0),
            (date(2025, 1, 2), "CA"): (40.0, 38.0),
        }
        for (ds, country), (c_mean, t_mean) in spec.items():
            control = _make_daily_row(n=2000, mean=c_mean, var=4.0, ds=ds, group_id="control")
            treatment = _make_daily_row(n=2000, mean=t_mean, var=4.0, ds=ds, group_id="treatment")
            control["country"] = country
            treatment["country"] = country
            rows.extend([control, treatment])
            expected_lift = (
                estimate_lift(
                    metrics=[_mean_metric()],
                    summary=pd.DataFrame([control, treatment]),
                    control_group="control",
                )
                .results[0]
                .lift
            )
            assert expected_lift is not None
            expected[(ds, country)] = expected_lift.value

        results = run_daily_lift(
            pd.DataFrame(rows), [_mean_metric()], control_group="control", dimension="country"
        )

        assert len(results) == 4
        actual = {}
        for r in results:
            lift = r.lift
            assert lift is not None
            actual[(r.ds, r.dimension_value)] = lift.value
        assert actual == expected

    def test_dimension_and_source_stamped_onto_every_result(self):
        ds = date(2025, 1, 1)
        rows = [
            _make_daily_row(n=2000, mean=10.0, var=4.0, ds=ds, group_id="control"),
            _make_daily_row(n=2000, mean=12.0, var=4.0, ds=ds, group_id="treatment"),
        ]
        for row in rows:
            row["country"] = "US"

        results = run_daily_lift(
            rows,
            [_mean_metric()],
            control_group="control",
            dimension="country",
            source="events",
        )

        assert len(results) == 1
        assert results[0].dimension == "country"
        assert results[0].dimension_value == "US"
        assert results[0].source == "events"

    def test_dimension_value_is_string_coerced(self):
        """Mirrors run_breakout/run_daily's `str(row[dimension])`
        convention: an int-typed property still comes back a str."""
        ds = date(2025, 1, 1)
        rows = [
            _make_daily_row(n=2000, mean=10.0, var=4.0, ds=ds, group_id="control"),
            _make_daily_row(n=2000, mean=12.0, var=4.0, ds=ds, group_id="treatment"),
        ]
        for row in rows:
            row["plan_tier"] = 3

        results = run_daily_lift(
            rows, [_mean_metric()], control_group="control", dimension="plan_tier"
        )

        assert results[0].dimension_value == "3"
        assert isinstance(results[0].dimension_value, str)

    def test_no_dimension_leaves_new_fields_none(self):
        """The un-dimensioned call (the only behavior that existed before
        `dimension` was added) must not populate the new fields, for both
        an implicit and an explicit `dimension=None`."""
        ds = date(2025, 1, 1)
        rows = [
            _make_daily_row(n=2000, mean=10.0, var=4.0, ds=ds, group_id="control"),
            _make_daily_row(n=2000, mean=12.0, var=4.0, ds=ds, group_id="treatment"),
        ]

        implicit = run_daily_lift(rows, [_mean_metric()], control_group="control")
        explicit = run_daily_lift(rows, [_mean_metric()], control_group="control", dimension=None)

        for results in (implicit, explicit):
            assert results[0].dimension is None
            assert results[0].dimension_value is None
            assert results[0].source is None
        implicit_lift = implicit[0].lift
        explicit_lift = explicit[0].lift
        assert implicit_lift is not None and explicit_lift is not None
        assert implicit_lift.value == explicit_lift.value

    def test_dimension_column_missing_from_dataframe_raises(self):
        ds = date(2025, 1, 1)
        rows = [
            _make_daily_row(n=2000, mean=10.0, var=4.0, ds=ds, group_id="control"),
            _make_daily_row(n=2000, mean=12.0, var=4.0, ds=ds, group_id="treatment"),
        ]
        df = pd.DataFrame(rows)  # no "country" column at all

        with pytest.raises(InvalidRequestError) as exc_info:
            run_daily_lift(df, [_mean_metric()], control_group="control", dimension="country")
        assert exc_info.value.code == "breakout.run_daily_lift_dimension_found"
        assert exc_info.value.context["dimension"] == "country"

    def test_dict_rows_missing_dimension_key_raises(self):
        """Mirrors run_daily's dict-rows-missing-key contract: the
        plain-mapping path can't pre-validate without a schema, so it
        fails at first row access with a KeyError."""
        ds = date(2025, 1, 1)
        rows = [
            _make_daily_row(n=2000, mean=10.0, var=4.0, ds=ds, group_id="control"),
            _make_daily_row(n=2000, mean=12.0, var=4.0, ds=ds, group_id="treatment"),
        ]

        with pytest.raises(KeyError):
            run_daily_lift(rows, [_mean_metric()], control_group="control", dimension="country")


# run_daily_lift: per-day CI reflects per-day sample size (mirrors
# TestRunBreakoutConfidenceIntervals)


class TestRunDailyLiftConfidenceIntervals:
    def test_ci_narrower_for_larger_day(self):
        big_control = _make_daily_row(
            n=20000, mean=10.0, var=4.0, ds=date(2025, 1, 1), group_id="control"
        )
        big_treatment = _make_daily_row(
            n=20000, mean=11.0, var=4.0, ds=date(2025, 1, 1), group_id="treatment"
        )
        small_control = _make_daily_row(
            n=200, mean=10.0, var=4.0, ds=date(2025, 1, 2), group_id="control"
        )
        small_treatment = _make_daily_row(
            n=200, mean=11.0, var=4.0, ds=date(2025, 1, 2), group_id="treatment"
        )
        summary = pd.DataFrame([big_control, big_treatment, small_control, small_treatment])

        result = run_daily_lift(summary, [_mean_metric()], control_group="control")
        by_day = {r.ds: r for r in result}
        big = by_day[date(2025, 1, 1)]
        small = by_day[date(2025, 1, 2)]

        big_lift = big.lift
        small_lift = small.lift
        assert big_lift is not None and small_lift is not None
        assert big_lift.lb is not None and big_lift.ub is not None
        assert small_lift.lb is not None and small_lift.ub is not None
        assert (big_lift.ub - big_lift.lb) < (small_lift.ub - small_lift.lb)


# run_daily_lift: reliability_floor - a thin day's estimate is returned
# but flagged low_reliability=True, and readout suppresses its stat_sig


DAY1 = date(2025, 1, 1)
DAY2 = date(2025, 1, 2)


class TestRunDailyLiftReliabilityFloor:
    """`reliability_floor` (default 50): a day whose control or treatment
    arm has fewer units than the floor still gets a real, flagged
    (`low_reliability=True`) estimate, not an omitted or NaN'd one."""

    def _daily_rows(
        self,
        *,
        day1_n: tuple[int, int],
        day2_n: tuple[int, int],
        var: float = 4.0,
    ) -> list[dict]:
        """Two days x two arms; only each day's `(control_n, treatment_n)`
        pair varies - both arms stay above the few_units gate (n>=2),
        differing only in whether they clear the reliability floor."""
        rows = []
        for ds, (control_n, treatment_n) in ((DAY1, day1_n), (DAY2, day2_n)):
            rows.append(_make_daily_row(control_n, 10.0, var, ds=ds, group_id="control"))
            rows.append(_make_daily_row(treatment_n, 11.0, var, ds=ds, group_id="treatment"))
        return rows

    def test_daily_lift_flags_low_reliability_below_floor(self):
        """A day whose treatment arm has n < floor carries
        low_reliability=True; a day clearing the floor does not. The
        flagged day is still a real, estimated row."""
        rows = self._daily_rows(day1_n=(100, 30), day2_n=(100, 100))
        out = run_daily_lift(
            rows, [_windowed_mean_metric()], control_group="control", reliability_floor=50
        )
        by_ds = {e.ds: e for e in out if e.group_id == "treatment"}
        assert by_ds[DAY1].low_reliability is True
        assert by_ds[DAY2].low_reliability is False
        # The floor flags, never excludes: the thin day's lift is real.
        assert by_ds[DAY1].lift is not None

    def test_daily_lift_floor_checks_control_arm_too(self):
        """A thin CONTROL arm also trips the flag: the check is an OR
        across both arms of the day's comparison."""
        rows = self._daily_rows(day1_n=(30, 200), day2_n=(200, 200))
        out = run_daily_lift(rows, [_windowed_mean_metric()], control_group="control")
        by_ds = {e.ds: e for e in out if e.group_id == "treatment"}
        assert by_ds[DAY1].low_reliability is True
        assert by_ds[DAY2].low_reliability is False

    def test_daily_lift_floor_zero_disables(self):
        """reliability_floor=0: every estimable day is unflagged (any live
        arm has n >= 2 > 0) - proves the threshold is a real parameter."""
        rows = self._daily_rows(day1_n=(5, 5), day2_n=(200, 200))
        out = run_daily_lift(
            rows, [_windowed_mean_metric()], control_group="control", reliability_floor=0
        )
        assert all(e.low_reliability is False for e in out)

    def test_readout_suppresses_stat_sig_for_low_reliability_daily_row(self):
        """tables.py's stat_sig is forced False for a flagged
        DailyLiftEstimate even when the interval excludes 0."""
        pytest.importorskip("coeftable")
        from increment.tables import estimates_to_readout

        out = run_daily_lift(
            self._daily_rows(day1_n=(20, 20), day2_n=(200, 200), var=0.04),
            [_windowed_mean_metric()],
            control_group="control",
        )
        by_ds = {est.ds: row for est, row in zip(out, estimates_to_readout(out), strict=True)}
        assert by_ds[DAY1]["low_reliability"] is True
        assert by_ds[DAY1]["stat_sig"] is False
        assert by_ds[DAY2]["low_reliability"] is False
        assert by_ds[DAY2]["stat_sig"] is True


# alternative= forwarding: one-sided per-day testing


class TestRunDailyLiftAlternative:
    def test_default_alternative_is_two_sided_and_unchanged(self):
        """Omitting `alternative=` must reproduce the exact interval
        `run_daily_lift` always produced, and every row is labeled
        "two-sided"."""
        control = _make_daily_row(
            n=500, mean=10.0, var=4.0, ds=date(2025, 1, 1), group_id="control"
        )
        treatment = _make_daily_row(
            n=500, mean=11.0, var=4.0, ds=date(2025, 1, 1), group_id="treatment"
        )
        summary = pd.DataFrame([control, treatment])

        implicit = run_daily_lift(summary, [_mean_metric()], control_group="control")
        explicit = run_daily_lift(
            summary, [_mean_metric()], control_group="control", alternative="two-sided"
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
        """A one-sided `alternative="greater"` guardrail at `alpha=0.05`
        reproduces the SAME interval numbers as the plain two-sided call at
        `alpha=0.10` - the alpha-doubling identity `TestRunBreakoutAlternative`
        exercises for the segment axis."""
        control = _make_daily_row(
            n=2000, mean=10.0, var=4.0, ds=date(2025, 1, 1), group_id="control"
        )
        treatment = _make_daily_row(
            n=2000, mean=12.0, var=4.0, ds=date(2025, 1, 1), group_id="treatment"
        )
        summary = pd.DataFrame([control, treatment])

        one_sided = run_daily_lift(
            summary,
            [_mean_metric()],
            control_group="control",
            alpha=0.05,
            alternative="greater",
        )
        two_sided = run_daily_lift(summary, [_mean_metric()], control_group="control", alpha=0.10)

        assert one_sided[0].alternative == "greater"
        assert two_sided[0].alternative == "two-sided"
        one_sided_lift = one_sided[0].lift
        two_sided_lift = two_sided[0].lift
        assert one_sided_lift is not None and two_sided_lift is not None
        assert one_sided_lift.level == pytest.approx(0.90)
        assert one_sided_lift.lb == pytest.approx(two_sided_lift.lb)
        assert one_sided_lift.ub == pytest.approx(two_sided_lift.ub)

    def test_unestimable_day_nan_row_carries_requested_alternative(self):
        """A day with no control arm never reaches `estimate_lift`; the
        NaN row `_nan_lift_rows` emits must still report the REQUESTED
        `alternative`, mirroring `run_breakout`'s NaN-row convention."""
        rows = [
            _make_daily_row(n=50, mean=10.0, var=4.0, ds=date(2025, 1, 1), group_id="treatment")
        ]

        result = run_daily_lift(rows, [_mean_metric()], control_group="control", alternative="less")

        assert len(result) == 1
        assert result[0].lift is None
        assert result[0].alternative == "less"


# run_daily_lift: snapshot-scoped Bonferroni correction for an as-of trend -
# each metric's family is every segment in the full summary, even one absent from an early day.


class TestRunDailyLiftBonferroniCorrection:
    def _dimensioned_asof_rows(self) -> list[dict]:
        def row(ds: date, country: str, group_id: str, mean: float) -> dict:
            return {
                **_make_daily_row(n=2000, mean=mean, var=4.0, ds=ds, group_id=group_id),
                "country": country,
            }

        return [
            row(date(2025, 1, 1), "US", "control", 10.0),
            row(date(2025, 1, 1), "US", "treatment", 11.0),
            row(date(2025, 1, 2), "US", "control", 10.0),
            row(date(2025, 1, 2), "US", "treatment", 11.0),
            row(date(2025, 1, 2), "CA", "control", 20.0),
            row(date(2025, 1, 2), "CA", "treatment", 22.0),
        ]

    def test_counts_later_segment_in_the_entire_input_series(self):
        rows = self._dimensioned_asof_rows()
        corrected = run_daily_lift(
            rows,
            [_windowed_mean_metric()],
            control_group="control",
            dimension="country",
            view="asof",
            alpha=0.05,
            correction="bonferroni",
        )
        uncorrected = run_daily_lift(
            rows,
            [_windowed_mean_metric()],
            control_group="control",
            dimension="country",
            view="asof",
            alpha=0.05,
            correction="none",
        )

        assert corrected
        for row in corrected:
            lift = row.lift
            assert lift is not None
            assert lift.level == pytest.approx(0.975)
        for row in uncorrected:
            lift = row.lift
            assert lift is not None
            assert lift.level == pytest.approx(0.95)
        by_key_corrected = {(row.ds, row.dimension_value): row for row in corrected}
        by_key_uncorrected = {(row.ds, row.dimension_value): row for row in uncorrected}
        corrected_us_lift = by_key_corrected[(date(2025, 1, 1), "US")].lift
        assert corrected_us_lift is not None
        assert corrected_us_lift.level == pytest.approx(0.975)
        for key, row in by_key_corrected.items():
            baseline = by_key_uncorrected[key]
            lift = row.lift
            baseline_lift = baseline.lift
            assert lift is not None and baseline_lift is not None
            assert lift.lb is not None and lift.ub is not None
            assert baseline_lift.lb is not None and baseline_lift.ub is not None
            assert lift.ub - lift.lb > baseline_lift.ub - baseline_lift.lb

    def test_counts_segments_per_metric_across_the_complete_input_series(self):
        def rows_for(
            metric: str, ds: date, country: str, control_mean: float, treatment_mean: float
        ) -> list[dict]:
            return [
                {
                    **_make_daily_row(
                        n=2000,
                        mean=mean,
                        var=4.0,
                        ds=ds,
                        metric=metric,
                        group_id=group_id,
                    ),
                    "country": country,
                }
                for group_id, mean in (("control", control_mean), ("treatment", treatment_mean))
            ]

        rows = [
            *rows_for("metric_a", date(2025, 1, 1), "US", 10.0, 11.0),
            *rows_for("metric_a", date(2025, 1, 2), "US", 10.0, 11.0),
            *rows_for("metric_b", date(2025, 1, 1), "US", 20.0, 22.0),
            *rows_for("metric_b", date(2025, 1, 2), "US", 20.0, 22.0),
            *rows_for("metric_b", date(2025, 1, 2), "CA", 30.0, 33.0),
        ]

        corrected = run_daily_lift(
            rows,
            [_windowed_mean_metric("metric_a"), _windowed_mean_metric("metric_b")],
            control_group="control",
            dimension="country",
            view="asof",
            alpha=0.05,
            correction="bonferroni",
        )

        for row in corrected:
            lift = row.lift
            assert lift is not None
            assert lift.level == pytest.approx({"metric_a": 0.95, "metric_b": 0.975}[row.metric])
        by_key = {(row.metric, row.ds, row.dimension_value): row for row in corrected}
        metric_b_lift = by_key[("metric_b", date(2025, 1, 1), "US")].lift
        assert metric_b_lift is not None
        assert metric_b_lift.level == pytest.approx(0.975)

    def test_corrected_dimension_rejects_unknown_raw_metric(self):
        rows = self._dimensioned_asof_rows()
        rows[0]["metric"] = "unknown"

        with pytest.raises(InvalidRequestError) as exc_info:
            run_daily_lift(
                rows,
                [_windowed_mean_metric()],
                control_group="control",
                dimension="country",
                correction="bonferroni",
            )
        assert exc_info.value.code == "breakout.run_daily_lift_metric_declared"
        assert exc_info.value.context["unknown"] == ("unknown",)
        assert exc_info.value.context["declared"] == ("rev",)

    def test_corrects_real_itt_and_late_rows_for_later_appearing_segment(self):
        design = _encouragement_design(min_first_stage_z=4.0)

        def country_rows(ds: date, country: str, seed: int) -> list[dict]:
            rows = _encouragement_day_rows(ds=ds, n=4000, tau=2.0, compliance=0.5, seed=seed)
            for row in rows:
                row["country"] = country
            return rows

        rows = [
            *country_rows(date(2025, 1, 1), "US", 1),
            *country_rows(date(2025, 1, 2), "US", 2),
            *country_rows(date(2025, 1, 2), "CA", 3),
        ]
        corrected = run_daily_lift(
            rows,
            [_windowed_mean_metric()],
            control_group="control",
            dimension="country",
            design=design,
            view="asof",
            correction="bonferroni",
        )

        for estimand in ("itt", "late"):
            live = [row for row in corrected if row.estimand == estimand and row.lift is not None]
            assert live
            for row in live:
                lift = row.lift
                assert lift is not None
                assert math.isfinite(lift.value)
                assert lift.level == pytest.approx(0.975)
        early_us = [
            row
            for row in corrected
            if row.ds == date(2025, 1, 1) and row.dimension_value == "US" and row.lift is not None
        ]
        assert {row.estimand for row in early_us} == {"itt", "compliance", "late"}
        for row in early_us:
            lift = row.lift
            assert lift is not None
            assert math.isfinite(lift.value)
            assert lift.level == pytest.approx(0.975)

    def test_rejects_unknown_correction(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            run_daily_lift(
                self._dimensioned_asof_rows(),
                [_windowed_mean_metric()],
                control_group="control",
                correction="holm",  # ty: ignore[invalid-argument-type]
            )
        assert exc_info.value.code == "breakout.run_daily_lift"
        assert exc_info.value.context["correction"] == "holm"

    def test_default_correction_preserves_existing_behavior(self):
        rows = self._dimensioned_asof_rows()

        # These rows are dimensioned by country, so the dimension must be
        # declared: without it each day carries two controls, which is now
        # refused as a malformed arm inventory rather than silently resolved.
        implicit = run_daily_lift(
            rows,
            [_windowed_mean_metric()],
            control_group="control",
            dimension="country",
            view="asof",
        )
        explicit = run_daily_lift(
            rows,
            [_windowed_mean_metric()],
            control_group="control",
            dimension="country",
            view="asof",
            alpha=0.05,
            correction="none",
        )

        assert implicit == explicit


class TestRunDailyLiftRequestValidation:
    def test_unsupported_estimand_raises_coded_error(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            run_daily_lift([], [_mean_metric("rev")], control_group="control", estimands=("bogus",))
        assert exc_info.value.code == "breakout.unknown_estimand_supported"
        assert exc_info.value.context["unknown_estimands"] == ("bogus",)

    def test_compliance_estimand_without_asof_encouragement_raises(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            run_daily_lift(
                [], [_mean_metric("rev")], control_group="control", estimands=("compliance",)
            )
        assert exc_info.value.code == "breakout.compliance_late_view"


# inference=AlwaysValid on the as-of trend: only asof's moments are
# cumulative through each day, forming a real confidence SEQUENCE; daily/cohort slices are disjoint, with no such interpretation.


class TestRunDailyLiftAlwaysValidInference:
    def test_gaussian_always_valid_readout_refuses_before_projection(self):
        from increment.errors import CapabilityError
        from tests.sequential_cases import raw_gaussian

        snapshot, metrics, policy = raw_gaussian(n=4, label=date(2025, 1, 1))
        with pytest.raises(CapabilityError) as raised:
            run_daily_lift(
                snapshot, metrics, control_group="control", view="asof", inference=policy
            )
        assert raised.value.code == "sequential.route.unsupported"

    @pytest.mark.parametrize("view", ["daily", "cohort"])
    def test_disjoint_slices_refuse_before_summary_access(self, view):
        class UnreadSummary:
            def __iter__(self):
                raise AssertionError("read before disjoint-slice refusal")

        with pytest.raises(UnsupportedRequestError) as raised:
            run_daily_lift(
                UnreadSummary(),
                [_windowed_mean_metric()],
                control_group="control",
                view=view,
                inference=AlwaysValid(registration=registration("gaussian")),
            )
        assert raised.value.code == "readout.inference.disjoint_slices"

    def test_asof_binary_uptake_late_is_explicitly_refused(self):
        from increment.errors import CapabilityError
        from tests.sequential_cases import raw_gaussian

        snapshot, metrics, policy = raw_gaussian(label=date(2025, 1, 1))
        with pytest.raises(CapabilityError) as raised:
            run_daily_lift(
                snapshot,
                metrics,
                control_group="control",
                view="asof",
                design=_encouragement_design(),
                inference=policy,
                estimands=("late",),
            )
        assert raised.value.code == "sequential.route.unsupported"


# DailyLiftEstimate model (mirrors TestBreakoutEstimate)


class TestDailyLiftEstimate:
    def test_frozen(self):
        est = DailyLiftEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            ds=date(2025, 1, 1),
            lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
        )
        with pytest.raises((TypeError, ValueError)):
            est.ds = date(2025, 1, 2)  # ty: ignore[invalid-assignment]  # proving frozen at runtime

    def test_basic_construction(self):
        lift = Estimate(value=0.2, lb=0.1, ub=0.3, level=0.95)
        est = DailyLiftEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            ds=date(2025, 1, 1),
            lift=lift,
        )
        assert est.metric == "rev"
        assert est.group_id == "treatment"
        assert est.method == "unadjusted"
        assert est.ds == date(2025, 1, 1)
        assert est.lift is lift

    @pytest.mark.parametrize("field", ["null_lift", "null_abs", "abs_diff", "abs_se"])
    def test_nonfinite_scalar_fields_are_rejected(self, field: str) -> None:
        kwargs: dict[str, Any] = {field: -math.inf}
        with pytest.raises(InvalidRequestError) as raised:
            DailyLiftEstimate(
                metric="rev",
                group_id="treatment",
                method="unadjusted",
                method_role="decision",
                ds=date(2025, 1, 1),
                lift=Estimate(value=0.2),
                **kwargs,
            )
        assert raised.value.code == "model.field.nonfinite"
        assert raised.value.context["field"] == field

    def test_neither_lift_nor_unavailable_set_raises_coded_error(self):
        """The buried coded refusal must surface with its own `.code`, not a
        bare pydantic `ValidationError`."""
        with pytest.raises(InvalidRequestError) as exc_info:
            DailyLiftEstimate(
                metric="rev",
                group_id="treatment",
                method="unadjusted",
                method_role="decision",
                ds=date(2025, 1, 1),
                lift=None,
            )
        assert exc_info.value.code == "breakout.daily_lift.exactly_one_unavailable"

    def test_require_lift_on_valid_unavailable_row_raises_availability_error(self):
        est = DailyLiftEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            ds=date(2025, 1, 1),
            lift=None,
            unavailable="few_units",
        )

        with pytest.raises(InvalidRequestError) as exc_info:
            est.require_lift()

        assert exc_info.value.code == "breakout.daily_lift.point_unavailable"

    def test_round_trips_on_a_real_run_daily_lift_result(self):
        """model_dump_json() round-trips on a REAL run_daily_lift output,
        the day-axis counterpart of TestBreakoutEstimate's round-trip test."""
        rows = [
            _make_daily_row(n=50, mean=10.0, var=4.0, ds=date(2025, 1, 1), group_id="control"),
            _make_daily_row(n=50, mean=12.0, var=4.0, ds=date(2025, 1, 1), group_id="treatment"),
        ]
        results = run_daily_lift(rows, [_mean_metric("rev")], control_group="control")
        assert len(results) == 1
        est = results[0]
        round_tripped = DailyLiftEstimate.model_validate_json(est.model_dump_json())
        assert round_tripped == est

    def test_public_replay_rejects_gaussian_checkpoint(self):
        from increment.errors import CapabilityError
        from increment.estimation.sequential_runtime import (
            _evaluate_sequential_diagnostic,
            display_estimate,
        )
        from tests.sequential_cases import raw_gaussian

        snapshot, _, policy = raw_gaussian(n=4, label=date(2025, 1, 1))
        result = _evaluate_sequential_diagnostic(snapshot, policy)[0]
        with pytest.raises(CapabilityError) as raised:
            DailyLiftEstimate(
                metric="rev",
                group_id="treatment",
                method="unadjusted",
                method_role="decision",
                inference="always_valid",
                reference_kind="sequential",
                ds=date(2025, 1, 1),
                lift=display_estimate(result),
                sequential_result=result,
            )
        assert raised.value.code == "sequential.route.unsupported"

    @pytest.mark.parametrize("relabel", [False, True])
    def test_public_replay_preserves_bernoulli_reference_label(self, relabel):
        from increment import estimate_sequential
        from increment.errors import CapabilityError
        from tests.sequential_cases import registered_bernoulli

        snapshot, policy = registered_bernoulli(n=4)
        result = estimate_sequential(snapshot, policy).results[0]
        row = DailyLiftEstimate(
            metric="revenue",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            inference="always_valid",
            reference_kind="sequential",
            ds=date(2025, 1, 1),
            lift=result.lift,
            sequential_result=result.sequential_result,
        )
        if not relabel:
            assert DailyLiftEstimate.model_validate_json(row.model_dump_json()) == row
            return
        payload = row.model_dump()
        payload.update(inference="fixed", reference_kind="normal")
        with pytest.raises(CapabilityError) as caught:
            DailyLiftEstimate.model_validate(payload)
        assert caught.value.code == "sequential.source.invalid"

    def test_fixed_daily_interval_cannot_be_relabeled_anytime(self):
        from increment.errors import InvalidRequestError

        row = DailyLiftEstimate(
            ds=date(2025, 1, 1),
            metric="m",
            group_id="T",
            method="unadjusted",
            method_role="decision",
            lift=Estimate(value=0.3, lb=0.25, ub=0.35, level=0.95),
        )
        payload = row.model_dump()
        payload["inference"] = "always_valid"
        with pytest.raises(InvalidRequestError) as caught:
            DailyLiftEstimate.model_validate(payload)
        assert caught.value.code == "estimation.results.lift.inference_reference_kind_mismatch"

    def test_daily_legacy_degrees_retain_student_reference(self):
        row = DailyLiftEstimate(
            ds=date(2025, 1, 1),
            metric="m",
            group_id="T",
            method="unadjusted",
            method_role="decision",
            lift=Estimate(value=0.3, lb=0.25, ub=0.35, level=0.95),
            dof=7.0,
        )
        assert row.reference_kind == "t"
        assert row.reference_df == 7.0

    @pytest.mark.parametrize(
        "field",
        ["relative_confidence_set", "relative_unavailable_reason", "abs_diff", "null_abs"],
    )
    def test_sequential_checkpoint_rejects_mixed_authoritative_payload(self, field):
        """A checkpoint-backed sequential view cannot carry fixed/joint sidecars."""
        from increment import estimate_sequential
        from increment.errors import CapabilityError
        from increment.estimation.results import JointContrastReference, RelativeConfidenceSet
        from tests.sequential_cases import registered_bernoulli

        snapshot, policy = registered_bernoulli(n=4)
        result = estimate_sequential(snapshot, policy).results[0]
        payload = {
            "metric": "revenue",
            "group_id": "treatment",
            "method": "unadjusted",
            "method_role": "decision",
            "inference": "always_valid",
            "reference_kind": "sequential",
            "ds": date(2025, 1, 1),
            "lift": result.lift,
            "sequential_result": result.sequential_result,
        }
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
            DailyLiftEstimate.model_validate(payload)
        assert raised.value.code == "sequential.source.invalid"


# run_daily_lift: narrow exception handling - an unestimable slice never
# raises; every OTHER error `estimate_lift` can raise still propagates unchanged.


class TestRunDailyLiftErrorPropagation:
    def test_unrelated_error_propagates_not_swallowed(self):
        """A day WITH a valid control arm that raises ValueError for an
        unrelated reason (an undeclared metric present in the data) must
        raise, not be silently swallowed as a skipped day."""
        rows = [
            _make_daily_row(
                50, 10.0, 4.0, ds=date(2025, 1, 1), group_id="control", metric="undeclared"
            ),
            _make_daily_row(
                50, 12.0, 4.0, ds=date(2025, 1, 1), group_id="treatment", metric="undeclared"
            ),
        ]
        with pytest.raises(InvalidRequestError) as exc_info:
            run_daily_lift(rows, [_mean_metric("rev")], control_group="control")
        assert exc_info.value.code == "breakout.run_daily_lift_metric_declared"
        assert "undeclared" in exc_info.value.context["unknown"]  # ty: ignore[unsupported-operator]

    def test_ds_column_missing_from_dataframe_raises(self):
        row = _make_daily_row(50, 10.0, 4.0, ds=date(2025, 1, 1))
        df = pd.DataFrame([row]).drop(columns=["ds"])

        with pytest.raises(InvalidRequestError) as exc_info:
            run_daily_lift(df, [_mean_metric("rev")], control_group="control")
        assert exc_info.value.code == "breakout.run_daily_lift_ds_column_found"


# run_daily_lift: every route to an unestimable (day, metric, arm) slice -
# non-positive mean, dropped control row, no live control arm, or estimate_lift's own degenerate-data guard - returns a NaN row instead of dropping it.


class TestRunDailyLiftNaNRowsForUnestimableSlices:
    """An unestimable (day, metric, arm) lift comes back NaN, not absent:
    every path that used to `continue` past a slice now emits one NaN
    DailyLiftEstimate instead. No UserWarning accompanies any of them."""

    def test_day_missing_control_arm_returns_nan_rows(self):
        """A day whose control arm is absent entirely can't be compared
        against anything - every non-control arm gets a NaN lift for that
        day rather than the day vanishing from the series."""
        d1_control = _make_daily_row(
            n=2000, mean=10.0, var=4.0, ds=date(2025, 1, 1), group_id="control"
        )
        d1_treatment = _make_daily_row(
            n=2000, mean=12.0, var=4.0, ds=date(2025, 1, 1), group_id="treatment"
        )
        d2_treatment = _make_daily_row(
            n=1500, mean=19.0, var=4.0, ds=date(2025, 1, 2), group_id="treatment"
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            results = run_daily_lift(
                [d1_control, d1_treatment, d2_treatment],
                [_mean_metric()],
                control_group="control",
            )

        by_day = {r.ds: r for r in results}
        assert set(by_day) == {date(2025, 1, 1), date(2025, 1, 2)}
        assert by_day[date(2025, 1, 1)].lift is not None
        assert by_day[date(2025, 1, 2)].lift is None
        assert by_day[date(2025, 1, 2)].unavailable == "no_control_arm"
        from increment.plan import compile_decision_plan
        from increment.semantics.models import AnalysisPlan

        plan = compile_decision_plan(
            AnalysisPlan(primary="rev"), [_mean_metric("rev")], path="warehouse"
        )
        planned = run_daily_lift(
            [d1_control, d1_treatment, d2_treatment],
            [_mean_metric("rev")],
            control_group="control",
            plan=plan,
        )
        planned_by_day = {row.ds: row for row in planned}
        assert planned_by_day[date(2025, 1, 2)].role == "primary"

    def test_zero_mean_treatment_row_returns_nan_for_that_arm_only(self):
        """One arm's non-positive mean makes only that arm's lift NaN;
        a second, healthy treatment arm on the same day is unaffected."""
        control = _make_daily_row(
            n=2000, mean=10.0, var=4.0, ds=date(2025, 1, 1), group_id="control"
        )
        good = _make_daily_row(n=2000, mean=12.0, var=4.0, ds=date(2025, 1, 1), group_id="t_good")
        zero = _make_daily_row(n=2000, mean=0.0, var=0.0, ds=date(2025, 1, 1), group_id="t_zero")

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            results = run_daily_lift(
                [control, good, zero], [_mean_metric()], control_group="control"
            )

        by_arm = {r.group_id: r for r in results}
        assert set(by_arm) == {"t_good", "t_zero"}
        assert by_arm["t_good"].lift is not None
        assert by_arm["t_zero"].lift is None
        assert by_arm["t_zero"].unavailable == "nonpositive_mean"

    def test_zero_mean_control_row_makes_that_metrics_arms_nan(self):
        """When the CONTROL row is the non-positive one, every non-control
        arm for that metric loses its comparison and goes NaN."""
        control = _make_daily_row(
            n=2000, mean=0.0, var=0.0, ds=date(2025, 1, 1), group_id="control"
        )
        treatment = _make_daily_row(
            n=2000, mean=12.0, var=4.0, ds=date(2025, 1, 1), group_id="treatment"
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            results = run_daily_lift(
                [control, treatment], [_mean_metric()], control_group="control"
            )

        assert len(results) == 1
        assert results[0].group_id == "treatment"
        assert results[0].lift is None
        assert results[0].unavailable == "no_control_arm"

    def test_one_metrics_lost_control_leaves_the_other_metric_estimable(self):
        """Two metrics, one day. Metric A's control is non-positive (its
        arms go NaN); metric B is healthy and estimates normally. The
        NaN is scoped to the metric that lost its control, not the day."""
        a_control = _make_daily_row(
            n=2000, mean=0.0, var=0.0, ds=date(2025, 1, 1), metric="a", group_id="control"
        )
        a_treatment = _make_daily_row(
            n=2000, mean=12.0, var=4.0, ds=date(2025, 1, 1), metric="a", group_id="treatment"
        )
        b_control = _make_daily_row(
            n=2000, mean=10.0, var=4.0, ds=date(2025, 1, 1), metric="b", group_id="control"
        )
        b_treatment = _make_daily_row(
            n=2000, mean=12.0, var=4.0, ds=date(2025, 1, 1), metric="b", group_id="treatment"
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            results = run_daily_lift(
                [a_control, a_treatment, b_control, b_treatment],
                [_mean_metric("a"), _mean_metric("b")],
                control_group="control",
            )

        by_metric = {r.metric: r for r in results}
        assert set(by_metric) == {"a", "b"}
        assert by_metric["a"].lift is None
        assert by_metric["a"].unavailable == "no_control_arm"
        assert by_metric["b"].lift is not None

    def test_degenerate_zero_variance_day_returns_nan_rows(self):
        """estimate_lift's own "both arms have zero variance" guard makes
        the slice unestimable - NaN rows, not an absent day."""
        control = _make_daily_row(
            n=2000, mean=10.0, var=0.0, ds=date(2025, 1, 1), group_id="control"
        )
        treatment = _make_daily_row(
            n=2000, mean=10.0, var=0.0, ds=date(2025, 1, 1), group_id="treatment"
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            results = run_daily_lift(
                [control, treatment], [_mean_metric()], control_group="control"
            )

        assert len(results) == 1
        assert results[0].group_id == "treatment"
        assert results[0].lift is None
        assert results[0].unavailable == "zero_variance"
        frame = cast(pd.DataFrame, results.to_frame())
        assert pd.isna(frame.loc[0, "lift"])
        assert frame.loc[0, "unavailable"] == "zero_variance"

    def test_daily_lift_partial_guard_stays_dense(self):
        control = _make_daily_row(
            n=2000, mean=10.0, var=4.0, ds=date(2025, 1, 1), group_id="control"
        )
        good = _make_daily_row(n=2000, mean=12.0, var=4.0, ds=date(2025, 1, 1), group_id="good")
        bad = _make_daily_row(n=2000, mean=10.0, var=0.0, ds=date(2025, 1, 1), group_id="bad")

        # Supply a non-degenerate covariate so CUPED remains estimable for
        # the guarded arm even though its unadjusted outcome variance is zero.
        for row, covariance in ((control, 1.0), (good, 1.0), (bad, 0.0)):
            n = int(row["n"])
            row.update(
                ref_x=0.0, cx1=0.0, cx2=float(n - 1), cxy=covariance * (n - 1), x_role="covariate"
            )

        methods = [
            Method(name="unadjusted"),
            Method(name="cuped", variance_reduction="cuped"),
        ]
        with pytest.warns(IncrementWarning) as rec:
            results = run_daily_lift(
                [control, good, bad], [_mean_metric()], control_group="control", methods=methods
            )
        assert "breakout.estimates.daily_partial_guarded_arms" in warning_codes(rec)

        by_key = {(row.group_id, row.method): row for row in results}
        assert by_key[("good", "unadjusted")].lift is not None
        assert by_key[("good", "cuped")].lift is not None
        assert by_key[("bad", "unadjusted")].lift is None
        assert by_key[("bad", "unadjusted")].unavailable == "zero_variance"
        assert by_key[("bad", "cuped")].lift is not None

    def test_nan_rows_carry_dimension_fields_when_dimensioned(self):
        """A NaN row is still fully identified: dimension/dimension_value/
        source are populated exactly as an estimable row's would be."""
        control = _make_daily_row(
            n=2000, mean=0.0, var=0.0, ds=date(2025, 1, 1), group_id="control"
        )
        treatment = _make_daily_row(
            n=2000, mean=12.0, var=4.0, ds=date(2025, 1, 1), group_id="treatment"
        )
        for row in (control, treatment):
            row["country"] = "US"

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            results = run_daily_lift(
                pd.DataFrame([control, treatment]),
                [_mean_metric()],
                control_group="control",
                dimension="country",
                source="events",
            )

        assert len(results) == 1
        assert results[0].lift is None
        assert results[0].unavailable == "no_control_arm"
        assert results[0].dimension == "country"
        assert results[0].dimension_value == "US"
        assert results[0].source == "events"

    def test_one_nan_row_per_requested_method(self):
        """With two methods requested, an unestimable slice yields one NaN
        row per method - the same cardinality an estimable slice has."""
        control = _make_daily_row(
            n=2000, mean=0.0, var=0.0, ds=date(2025, 1, 1), group_id="control"
        )
        treatment = _make_daily_row(
            n=2000, mean=12.0, var=4.0, ds=date(2025, 1, 1), group_id="treatment"
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            results = run_daily_lift(
                [control, treatment],
                [_mean_metric()],
                control_group="control",
                methods=[
                    Method(name="unadjusted"),
                    Method(name="cuped", variance_reduction="cuped"),
                ],
            )

        assert {r.method for r in results} == {"unadjusted", "cuped"}
        assert all(r.lift is None for r in results)
        assert all(r.unavailable == "no_control_arm" for r in results)

    def test_observational_method_name_refused_on_an_unestimable_day(self):
        """The mislabel refusal is an invariant, not a side effect of a day
        happening to be estimable: a fully degenerate day must refuse too."""
        control = _make_daily_row(
            n=2000, mean=0.0, var=0.0, ds=date(2025, 1, 1), group_id="control"
        )
        treatment = _make_daily_row(
            n=2000, mean=12.0, var=4.0, ds=date(2025, 1, 1), group_id="treatment"
        )

        with pytest.raises(InvalidRequestError) as exc_info:
            run_daily_lift(
                [control, treatment],
                [_mean_metric()],
                control_group="control",
                methods=[Method(name="iptw")],
            )
        assert exc_info.value.code == "estimation.engine.method_name_observational"

    def test_estimable_days_are_completely_unaffected(self):
        """Anti-drift: a fully estimable input returns exactly what it did
        before NaN rows existed."""
        d1_control = _make_daily_row(
            n=2000, mean=10.0, var=4.0, ds=date(2025, 1, 1), group_id="control"
        )
        d1_treatment = _make_daily_row(
            n=2000, mean=12.0, var=4.0, ds=date(2025, 1, 1), group_id="treatment"
        )
        expected = estimate_lift(
            metrics=[_mean_metric()],
            summary=pd.DataFrame([d1_control, d1_treatment]),
            control_group="control",
        ).results[0]

        results = run_daily_lift(
            [d1_control, d1_treatment], [_mean_metric()], control_group="control"
        )

        assert len(results) == 1
        result_lift = results[0].lift
        expected_lift = expected.lift
        assert result_lift is not None and expected_lift is not None
        assert result_lift.value == expected_lift.value

    def test_metric_with_no_control_row_at_all_returns_nan_without_dropping_sibling(self):
        """Metric "b" has no control row at all (never present, not
        dropped). Before the fix, `estimate_lift`'s `control_by_metric`
        lookup miss silently swallowed metric "b"'s treatment row; now it
        comes back as a NaN row like any other unestimable arm, and
        metric "a" (with a live control) is unaffected."""
        a_control = _make_daily_row(
            n=2000, mean=10.0, var=4.0, ds=date(2025, 1, 1), metric="a", group_id="control"
        )
        a_treatment = _make_daily_row(
            n=2000, mean=12.0, var=4.0, ds=date(2025, 1, 1), metric="a", group_id="treatment"
        )
        b_treatment = _make_daily_row(
            n=2000, mean=12.0, var=4.0, ds=date(2025, 1, 1), metric="b", group_id="treatment"
        )
        # No metric-"b"-control row anywhere in `rows` - never present,
        # not dropped after the fact.
        rows = [a_control, a_treatment, b_treatment]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            results = run_daily_lift(
                rows,
                [_mean_metric("a"), _mean_metric("b")],
                control_group="control",
            )

        assert len(results) == 2
        by_metric = {r.metric: r for r in results}
        assert set(by_metric) == {"a", "b"}
        assert by_metric["b"].group_id == "treatment"
        assert by_metric["b"].lift is None
        assert by_metric["b"].unavailable == "no_control_arm"
        assert by_metric["a"].lift is not None

    def test_delta_method_unreliable_returns_nan_rows(self):
        """`estimate_lift`'s "Log delta method unreliable" guard (combined
        log-scale SE >= 0.5) also makes the day unestimable, not absent.
        Thin, noisy near-zero-mean arms give per-arm log-scale SE
        0.577 / 0.481, combined 0.75 >= 0.5."""
        control = _make_daily_row(n=12, mean=1.0, var=4.0, ds=date(2025, 1, 1), group_id="control")
        treatment = _make_daily_row(
            n=12, mean=1.2, var=4.0, ds=date(2025, 1, 1), group_id="treatment"
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            results = run_daily_lift(
                [control, treatment], [_mean_metric()], control_group="control"
            )

        assert len(results) == 1
        assert results[0].group_id == "treatment"
        assert results[0].lift is None
        assert results[0].unavailable == "extreme_ratio"

    def test_negative_mean_treatment_row_returns_nan_for_that_arm_only(self):
        """A negative mean (not just zero) trips the same non-positive-mean
        guard, e.g. refund-heavy net revenue. Only that arm's lift goes
        NaN; a healthy sibling treatment arm is unaffected."""
        control = _make_daily_row(
            n=2000, mean=10.0, var=4.0, ds=date(2025, 1, 1), group_id="control"
        )
        good = _make_daily_row(n=2000, mean=12.0, var=4.0, ds=date(2025, 1, 1), group_id="t_good")
        negative = _make_daily_row(
            n=2000, mean=-5.0, var=4.0, ds=date(2025, 1, 1), group_id="t_negative"
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            results = run_daily_lift(
                [control, good, negative], [_mean_metric()], control_group="control"
            )

        by_arm = {r.group_id: r for r in results}
        assert set(by_arm) == {"t_good", "t_negative"}
        assert by_arm["t_good"].lift is not None
        assert by_arm["t_negative"].lift is None
        assert by_arm["t_negative"].unavailable == "nonpositive_mean"


class TestRunDailyLiftBinomialGateExemption:
    """The shared few_units/nonpositive_mean prefilter gates only rows bound
    for the log-Normal delta method, which needs a ddof=1 variance (n>=2)
    and positive means. A conversion/retention metric's n=1 or zero-event
    rows stay eligible for the exact binomial risk-ratio method (see
    binomial_rr.py), which admits both. Only a metric requiring the
    log-Normal delta method (a CUPED-adjusted conversion metric) keeps the
    full gate."""

    def test_conversion_metric_n_equals_one_reaches_exact_binomial(self):
        """n=1 for both arms used to be dropped as 'few_units' before
        even reaching estimate_lift; the exact binomial method is
        admissible at n=1."""
        rows = [
            _conversion_daily_row(1, 1, ds=date(2025, 1, 1), group_id="control"),
            _conversion_daily_row(1, 1, ds=date(2025, 1, 1), group_id="treatment"),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = run_daily_lift(rows, [_conversion_metric()], control_group="control")
        assert len(result) == 1
        row = result[0]
        assert row.unavailable is None
        assert row.lift is not None

    def test_conversion_metric_zero_control_count_reaches_set_only_binomial(self):
        """Control never converted (0/50) - used to be dropped as
        'nonpositive_mean'; the exact binomial method admits it directly
        as a set-only row (no finite point, a real confidence set)."""
        rows = [
            _conversion_daily_row(50, 0, ds=date(2025, 1, 1), group_id="control"),
            _conversion_daily_row(50, 25, ds=date(2025, 1, 1), group_id="treatment"),
        ]
        result = run_daily_lift(rows, [_conversion_metric()], control_group="control")
        assert len(result) == 1
        row = result[0]
        assert row.unavailable is None
        assert row.lift is None
        assert row.binomial_set is not None
        assert row.binomial_set.point_available is False

    def test_conversion_metric_with_cuped_method_keeps_the_legacy_gate(self):
        """A conversion metric requesting a CUPED method still needs the
        log-Normal delta method's ddof=1 variance - n=1 stays
        'few_units'."""
        rows = [
            _conversion_daily_row(1, 1, ds=date(2025, 1, 1), group_id="control"),
            _conversion_daily_row(1, 1, ds=date(2025, 1, 1), group_id="treatment"),
        ]
        result = run_daily_lift(
            rows,
            [_conversion_metric()],
            control_group="control",
            methods=[Method(name="cuped", variance_reduction="cuped")],
        )
        assert len(result) == 1
        assert result[0].unavailable == "few_units"

    @pytest.mark.parametrize(("n", "successes"), [(1, 1), (50, 0)])
    def test_mixed_methods_keep_exact_result_and_make_only_cuped_unavailable(self, n, successes):
        rows = [
            _conversion_daily_row(
                n,
                successes,
                ds=date(2025, 1, 1),
                group_id="control",
                with_covariate=True,
            ),
            _conversion_daily_row(
                n,
                successes,
                ds=date(2025, 1, 1),
                group_id="treatment",
                with_covariate=True,
            ),
        ]
        result = run_daily_lift(
            rows,
            [_conversion_metric()],
            control_group="control",
            methods=[Method(name="unadjusted"), Method(name="cuped", variance_reduction="cuped")],
        )
        by_method = {row.method: row for row in result}
        assert by_method["unadjusted"].reference_kind == "binomial"
        assert by_method["unadjusted"].binomial_set is not None
        assert by_method["unadjusted"].unavailable is None
        assert by_method["cuped"].lift is None
        assert by_method["cuped"].unavailable == ("few_units" if n == 1 else "nonpositive_mean")


# RetentionMetric has no valid independent-per-day snapshot - both
# functions must reject it up front, not silently report the raw daily occurrence rate mislabeled as retention.


class TestRejectRetentionMetrics:
    def test_run_daily_raises_naming_the_retention_metric(self):
        rows = [_make_daily_row(n=10, mean=1.0, var=0.2, ds=date(2025, 1, 1))]
        with pytest.raises(InvalidRequestError) as raised:
            run_daily(rows, [_bounded_retention_metric()])

        assert raised.value.code == "breakout.retention.daily"

    def test_run_daily_raises_before_any_other_work(self):
        """The guard fires even for a `summary` that would otherwise raise
        a DIFFERENT error (missing `ds` column) - proving it runs before
        any row processing, not interleaved with it."""
        row = _make_daily_row(n=10, mean=1.0, var=0.2, ds=date(2025, 1, 1))
        df = pd.DataFrame([row]).drop(columns=["ds"])
        with pytest.raises(InvalidRequestError) as raised:
            run_daily(df, [_bounded_retention_metric()])
        assert raised.value.code == "breakout.retention.daily"

    def test_run_daily_unaffected_by_mean_and_ratio_metrics(self):
        """No RetentionMetric in `metrics` -> the new guard is a no-op."""
        rows = [
            _make_daily_row(n=10, mean=1.0, var=0.2, ds=date(2025, 1, 1), group_id="control"),
            _make_daily_row(n=10, mean=1.5, var=0.2, ds=date(2025, 1, 1), group_id="treatment"),
        ]
        result = run_daily(rows, [_mean_metric(), _ratio_metric()])
        assert len(result) == 2

    def test_run_daily_lift_raises_naming_the_retention_metric(self):
        rows = [
            _make_daily_row(n=50, mean=1.0, var=0.2, ds=date(2025, 1, 1), group_id="control"),
            _make_daily_row(n=50, mean=1.5, var=0.2, ds=date(2025, 1, 1), group_id="treatment"),
        ]
        with pytest.raises(InvalidRequestError) as raised:
            run_daily_lift(rows, [_bounded_retention_metric()], control_group="control")
        assert raised.value.code == "breakout.retention.daily"

    def test_run_daily_lift_raises_before_partitioning_by_day(self):
        """The guard fires even for a `summary` missing the `ds` column
        entirely - proving it runs before day-partitioning, not after."""
        row = _make_daily_row(n=50, mean=1.0, var=0.2, ds=date(2025, 1, 1))
        df = pd.DataFrame([row]).drop(columns=["ds"])
        with pytest.raises(InvalidRequestError) as raised:
            run_daily_lift(df, [_bounded_retention_metric()], control_group="control")
        assert raised.value.code == "breakout.retention.daily"

    def test_run_daily_lift_unaffected_by_mean_and_ratio_metrics(self):
        """No RetentionMetric in `metrics` -> the new guard is a no-op."""
        rows = [
            _make_daily_row(n=50, mean=10.0, var=4.0, ds=date(2025, 1, 1), group_id="control"),
            _make_daily_row(n=50, mean=12.0, var=4.0, ds=date(2025, 1, 1), group_id="treatment"),
        ]
        result = run_daily_lift(rows, [_mean_metric()], control_group="control")
        assert len(result) == 1


def test_run_daily_lift_plan_splits_primary_alpha_and_stamps_role():
    """A compiled plan drives the per-cell display alpha: a primary's daily
    cell alpha is the plan alpha split across the primary count times its own
    non-control arms (the same composition run() uses), and the declared role
    is stamped on the row."""
    from increment.plan import compile_decision_plan
    from increment.semantics.models import AnalysisPlan

    metrics = [_mean_metric("rev"), _mean_metric("visits")]
    plan = compile_decision_plan(
        AnalysisPlan(alpha=0.1, primary=("rev", "visits")), metrics, path="warehouse"
    )
    ds = date(2025, 1, 1)
    rows = [
        _make_daily_row(200, mean, 4.0, ds=ds, metric=m, group_id=g)
        for m in ("rev", "visits")
        for g, mean in (("control", 10.0), ("t1", 12.0), ("t2", 11.0))
    ]
    results = run_daily_lift(rows, metrics, control_group="control", plan=plan)
    rev = [r for r in results if r.metric == "rev" and r.lift is not None]
    assert rev, "expected estimable primary cells"
    # 2 primaries x 2 non-control arms -> one combined conservative division.
    expected = _alpha_split(0.1, 2 * 2)
    assert all(r.lift is not None and r.require_lift().alpha == expected for r in rev)
    assert all(r.role == "primary" for r in rev)
    assert all(r.policy_name == "compiled_plan" for r in rev)


# view-aware retention guard: the same moment-shaped `summary` reaches
# run_daily from run_breakout/run_asof/a cohort reduction; the caller declares which one via `view=`.


def test_run_daily_lift_forwards_guardrail_tail_and_shifted_null():
    """Day-axis estimation consumes the complete compiled procedure, not just
    alpha+role: the adverse tail and shifted null are both carried into the row."""
    from increment.plan import compile_decision_plan
    from increment.semantics.models import AnalysisPlan, ExperimentMetric

    metrics = [
        _mean_metric("rev"),
        MeanMetric(name="cost", entity="user", fact="cost", preferred_direction="decrease"),
    ]
    plan = compile_decision_plan(
        AnalysisPlan(
            alpha=0.1,
            primary="rev",
            guardrails=[ExperimentMetric(metric="cost", margin=0.02)],
            alternative="greater",
        ),
        metrics,
        path="warehouse",
    )
    ds = date(2025, 1, 1)
    rows = [
        _make_daily_row(200, mean, 4.0, ds=ds, metric=metric, group_id=group)
        for metric in ("rev", "cost")
        for group, mean in (("control", 10.0), ("treatment", 12.0))
    ]

    results = run_daily_lift(rows, metrics, control_group="control", plan=plan)
    (cost,) = [row for row in results if row.metric == "cost"]
    assert cost.role == "guardrail"
    assert cost.alternative == "less"
    assert cost.null_lift == pytest.approx(0.02)
    assert cost.null_abs is None


def test_run_daily_lift_resolves_primary_arm_count_per_date_slice():
    """A late-arriving treatment arm must not spend an earlier date's alpha."""
    from increment.compatibility import _conservative_divide
    from increment.plan import compile_decision_plan
    from increment.semantics.models import AnalysisPlan

    metric = _mean_metric("rev")
    plan = compile_decision_plan(AnalysisPlan(alpha=0.1, primary="rev"), [metric], path="warehouse")
    day1, day2 = date(2025, 1, 1), date(2025, 1, 2)
    rows = [
        _make_daily_row(200, 10.0, 4.0, ds=day1, group_id="control"),
        _make_daily_row(200, 12.0, 4.0, ds=day1, group_id="t1"),
        _make_daily_row(200, 10.0, 4.0, ds=day2, group_id="control"),
        _make_daily_row(200, 12.0, 4.0, ds=day2, group_id="t1"),
        _make_daily_row(200, 11.0, 4.0, ds=day2, group_id="t2"),
    ]

    results = run_daily_lift(rows, [metric], control_group="control", plan=plan)
    alpha_by_day = {
        ds: {row.require_lift().alpha for row in results if row.ds == ds and row.lift is not None}
        for ds in (day1, day2)
    }
    assert alpha_by_day[day1] == {_conservative_divide(0.1, 1)}
    assert alpha_by_day[day2] == {_conservative_divide(0.1, 2)}


def test_run_daily_lift_refuses_plan_missing_a_requested_metric():
    """A nonempty partial policy map must not disable the fallback correction
    for a metric it silently skipped."""
    from increment.plan import compile_decision_plan
    from increment.semantics.models import AnalysisPlan

    rev, extra = _mean_metric("rev"), _mean_metric("extra")
    plan = compile_decision_plan(AnalysisPlan(primary="rev"), [rev], path="warehouse")
    ds = date(2025, 1, 1)
    rows = [
        _make_daily_row(200, mean, 4.0, ds=ds, metric=metric, group_id=group)
        for metric in ("rev", "extra")
        for group, mean in (("control", 10.0), ("treatment", 12.0))
    ]

    with pytest.raises(InvalidRequestError) as exc_info:
        run_daily_lift(rows, [rev, extra], control_group="control", plan=plan)
    assert exc_info.value.code == "breakout.run_daily_lift_plan_missing_procedures"
    assert exc_info.value.context["missing_procedures"] == ("extra",)


def test_run_daily_lift_preserves_absolute_guardrail_evidence():
    from increment.plan import compile_decision_plan
    from increment.semantics.models import AnalysisPlan, ExperimentMetric
    from increment.tables import estimates_to_readout

    metrics = [
        _mean_metric("rev"),
        MeanMetric(name="cost", entity="user", fact="cost", preferred_direction="decrease"),
    ]
    plan = compile_decision_plan(
        AnalysisPlan(
            primary="rev",
            guardrails=[ExperimentMetric(metric="cost", margin_abs=1.5)],
        ),
        metrics,
        path="warehouse",
    )
    ds = date(2025, 1, 1)
    rows = [
        _make_daily_row(2000, mean, 4.0, ds=ds, metric=metric, group_id=group)
        for metric in ("rev", "cost")
        for group, mean in (("control", 10.0), ("treatment", 10.5))
    ]

    results = run_daily_lift(rows, metrics, control_group="control", plan=plan)
    (cost,) = [row for row in results if row.metric == "cost"]
    assert cost.null_abs is not None
    assert cost.null_abs == pytest.approx(1.5)
    assert cost.abs_diff == pytest.approx(0.5)
    assert cost.abs_se is not None
    assert cost.abs_lb is not None and cost.abs_ub is not None
    assert cost.abs_ub < cost.null_abs
    (readout,) = estimates_to_readout([cost])
    assert readout["stat_sig"] is True


class TestViewAwareRetentionGuard:
    def test_daily_view_still_rejects_bounded_retention(self):
        """The independent-per-day view has no retention reading, bounded or not."""
        rows = [_make_daily_row(n=10, mean=1.0, var=0.2, ds=date(2025, 1, 1))]
        with pytest.raises(InvalidRequestError) as raised:
            run_daily(rows, [_bounded_retention_metric()], view="daily")
        assert raised.value.code == "breakout.retention.daily"

    def test_asof_view_accepts_bounded_retention(self):
        rows = [
            _make_daily_row(n=10, mean=1.0, var=0.2, ds=date(2025, 1, 1), metric="d7_retention")
        ]
        results = run_daily(rows, [_bounded_retention_metric()], view="asof")
        assert results
        assert all(r.ds_basis == "calendar" for r in results)

    def test_cohort_view_stamps_cohort_basis(self):
        rows = [
            _make_daily_row(n=10, mean=1.0, var=0.2, ds=date(2025, 1, 1), metric="d7_retention")
        ]
        results = run_daily(rows, [_bounded_retention_metric()], view="cohort")
        assert results
        assert all(r.ds_basis == "cohort" for r in results)

    def test_unbounded_retention_rejected_on_daily_and_cohort_views(self):
        rows = [
            _make_daily_row(n=10, mean=1.0, var=0.2, ds=date(2025, 1, 1), metric="d7_retention")
        ]
        for view in ("daily", "cohort"):
            with pytest.raises(InvalidRequestError) as raised:
                run_daily(rows, [_retention_metric()], view=view)
            assert raised.value.code == "breakout.retention.unbounded"
            assert raised.value.context["names"] == ("d7_retention",)
            assert raised.value.context["supported_view"] == "asof"

    def test_asof_view_accepts_unbounded_retention(self):
        """The calendar-axis cumulative series reports the unbounded
        ratchet honestly - no rejection on view='asof'."""
        rows = [
            _make_daily_row(n=10, mean=1.0, var=0.2, ds=date(2025, 1, 1), metric="d7_retention")
        ]
        results = run_daily(rows, [_retention_metric()], view="asof")
        assert results
        assert all(r.ds_basis == "calendar" for r in results)

    def test_default_view_is_daily_and_stamps_calendar_basis(self):
        """Every existing caller keeps today's behaviour."""
        rows = [_make_daily_row(n=10, mean=5.0, var=4.0, ds=date(2025, 1, 1))]
        results = run_daily(rows, [_mean_metric()])
        assert all(r.ds_basis == "calendar" for r in results)

    def test_lift_carries_the_same_contract(self):
        rows = [
            _conversion_daily_row(
                n=50,
                successes=20,
                ds=date(2025, 1, 1),
                group_id="control",
                metric="d7_retention",
            ),
            _conversion_daily_row(
                n=50,
                successes=30,
                ds=date(2025, 1, 1),
                group_id="treatment",
                metric="d7_retention",
            ),
        ]
        with pytest.raises(InvalidRequestError) as raised:
            run_daily_lift(
                rows, [_bounded_retention_metric()], control_group="control", view="daily"
            )
        assert raised.value.code == "breakout.retention.daily"
        estimates = run_daily_lift(
            rows, [_bounded_retention_metric()], control_group="control", view="cohort"
        )
        assert estimates
        assert all(e.ds_basis == "cohort" for e in estimates)


# DailyMetricValues.to_frame / DailyLiftEstimates.to_frame: the shared
# to_frame() mechanics, generic over each model's own `model_fields`.


class TestDailyMetricValueToFrame:
    def test_maps_fields_to_columns(self):
        """One row per DailyMetricValue, mapped directly from the model's
        fields. ``ds`` casts to a proper ``datetime64`` column (compared
        via ``.date()`` since pandas 2+ no longer treats a ``Timestamp``
        as equal to a bare ``date`` at midnight)."""
        values = DailyMetricValues(
            [
                DailyMetricValue(
                    ds=date(2025, 1, 15),
                    metric="purchase_rate",
                    group_id="T",
                    value=Estimate(value=0.30, lb=0.25, ub=0.35, level=0.95),
                    n=120,
                ),
                DailyMetricValue(
                    ds=date(2025, 1, 16),
                    metric="purchase_rate",
                    group_id="C",
                    value=Estimate(value=0.22, lb=0.18, ub=0.26, level=0.95),
                    n=118,
                ),
            ]
        )
        frame = cast(pd.DataFrame, values.to_frame())
        assert frame.loc[0, "ds"].date() == date(2025, 1, 15)
        assert frame.loc[0, "metric"] == "purchase_rate"
        assert frame.loc[0, "group_id"] == "T"
        assert frame.loc[0, "value"] == pytest.approx(0.30)
        assert frame.loc[0, "lb"] == pytest.approx(0.25)
        assert frame.loc[0, "ub"] == pytest.approx(0.35)
        assert frame.loc[0, "n"] == 120
        assert frame.loc[1, "ds"].date() == date(2025, 1, 16)
        assert frame.loc[1, "group_id"] == "C"
        assert frame.loc[1, "value"] == pytest.approx(0.22)
        assert frame.loc[1, "n"] == 118

    def test_ds_is_json_serializable_for_charting(self):
        """Regression: a plain object-dtype `datetime.date` column renders
        fine in pandas but chokes a charting library's JSON serializer
        (observed via Altair in a marimo notebook: 'Object of type date
        is not JSON serializable'). `ds` must be real `datetime64`."""
        values = DailyMetricValues(
            [
                DailyMetricValue(
                    ds=date(2025, 1, 15),
                    metric="m",
                    group_id="T",
                    value=Estimate(value=0.3, lb=0.25, ub=0.35, level=0.95),
                    n=120,
                ),
            ]
        )
        frame = cast(pd.DataFrame, values.to_frame())
        assert pd.api.types.is_datetime64_any_dtype(frame["ds"])

    def test_empty_list_has_correct_columns(self):
        """An empty ``DailyMetricValues`` is routine (a sparse day gets
        skipped) - it must still be a 0-row frame with the documented 10
        columns, not columnless, so ``frame["value"]`` gets an empty
        column instead of ``KeyError``."""
        frame = DailyMetricValues([]).to_frame()
        assert len(frame) == 0

    def test_carries_dimension_columns_when_present(self):
        """dimension/dimension_value/source populate when a
        DailyMetricValue carries them and stay None/NaN otherwise - the
        frame's column set is unconditional either way."""
        values = DailyMetricValues(
            [
                DailyMetricValue(
                    ds=date(2025, 1, 15),
                    metric="purchase_rate",
                    group_id="T",
                    value=Estimate(value=0.30, lb=0.25, ub=0.35, level=0.95),
                    n=120,
                    dimension="country",
                    dimension_value="US",
                    source="events",
                ),
                DailyMetricValue(
                    ds=date(2025, 1, 16),
                    metric="purchase_rate",
                    group_id="C",
                    value=Estimate(value=0.22, lb=0.18, ub=0.26, level=0.95),
                    n=118,
                ),
            ]
        )
        frame = cast(pd.DataFrame, values.to_frame())
        assert frame.loc[0, "dimension"] == "country"
        assert frame.loc[0, "dimension_value"] == "US"
        assert frame.loc[0, "source"] == "events"
        assert pd.isna(frame.loc[1, "dimension"])
        assert pd.isna(frame.loc[1, "dimension_value"])
        assert pd.isna(frame.loc[1, "source"])

    @pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
    def test_optional_strings_keep_native_nulls_across_backends(self, backend):
        values = DailyMetricValues(
            [
                DailyMetricValue(
                    ds=date(2025, 1, 15),
                    metric="purchase_rate",
                    group_id="T",
                    value=Estimate(value=0.30, lb=0.25, ub=0.35, level=0.95),
                    n=120,
                    dimension="country",
                    dimension_value="US",
                    source="events",
                ),
                DailyMetricValue(
                    ds=date(2025, 1, 16),
                    metric="purchase_rate",
                    group_id="T",
                    value=None,
                    unavailable="few_units",
                    n=1,
                ),
            ]
        )
        frame = cast(Any, values.to_frame(backend=backend))

        if backend == "pandas":
            assert frame.loc[0, "source"] == "events"
            assert frame.loc[1, "source"] is pd.NA
            assert frame.loc[1, "dimension"] is pd.NA
        elif backend == "polars":
            assert frame["source"].to_list() == ["events", None]
            assert frame["dimension"].to_list() == ["country", None]
        else:
            assert frame.column("source").to_pylist() == ["events", None]
            assert frame.column("dimension").to_pylist() == ["country", None]

    def test_backend_pyarrow_returns_a_pyarrow_table(self):
        """backend="pyarrow" builds a native pyarrow.Table via narwhals,
        proving the mixin is backend-agnostic. Also regression-tests the
        date->datetime cast: pyarrow's Datetime field rejects a raw
        `datetime.date` (`ArrowTypeError`) where pandas coerces silently."""
        values = DailyMetricValues(
            [
                DailyMetricValue(
                    ds=date(2025, 1, 15),
                    metric="m",
                    group_id="T",
                    value=Estimate(value=0.3, lb=0.25, ub=0.35, level=0.95),
                    n=120,
                ),
            ]
        )
        frame = values.to_frame(backend="pyarrow")
        assert isinstance(frame, pa.Table)
        assert frame.column("value")[0].as_py() == pytest.approx(0.3)
        assert frame.column("n")[0].as_py() == 120


class TestDailyLiftEstimateToFrame:
    def test_maps_fields_to_columns(self):
        """One row per DailyLiftEstimate; lift.value/.lb/.ub flatten onto
        lift/lb/ub, mirroring DailyMetricValues.to_frame's flattening.
        ``ds`` casts to a proper ``datetime64`` column."""
        values = DailyLiftEstimates(
            [
                DailyLiftEstimate(
                    ds=date(2025, 1, 15),
                    metric="purchase_rate",
                    group_id="T",
                    method="unadjusted",
                    method_role="decision",
                    lift=Estimate(value=0.30, lb=0.25, ub=0.35, level=0.95),
                ),
                DailyLiftEstimate(
                    ds=date(2025, 1, 16),
                    metric="purchase_rate",
                    group_id="T",
                    method="cuped",
                    method_role="decision",
                    lift=Estimate(value=0.22, lb=0.18, ub=0.26, level=0.95),
                ),
            ]
        )
        frame = cast(pd.DataFrame, values.to_frame())
        assert frame.loc[0, "ds"].date() == date(2025, 1, 15)
        assert frame.loc[0, "metric"] == "purchase_rate"
        assert frame.loc[0, "group_id"] == "T"
        assert frame.loc[0, "method"] == "unadjusted"
        assert frame.loc[0, "lift"] == pytest.approx(0.30)
        assert frame.loc[0, "lb"] == pytest.approx(0.25)
        assert frame.loc[0, "ub"] == pytest.approx(0.35)
        assert frame.loc[1, "ds"].date() == date(2025, 1, 16)
        assert frame.loc[1, "method"] == "cuped"
        assert frame.loc[1, "lift"] == pytest.approx(0.22)

    def test_ds_is_json_serializable_for_charting(self):
        """Same regression DailyMetricValues.to_frame guards: `ds` must be
        real `datetime64`, not `object` dtype."""
        values = DailyLiftEstimates(
            [
                DailyLiftEstimate(
                    ds=date(2025, 1, 15),
                    metric="m",
                    group_id="T",
                    method="unadjusted",
                    method_role="decision",
                    lift=Estimate(value=0.3, lb=0.25, ub=0.35, level=0.95),
                ),
            ]
        )
        frame = cast(pd.DataFrame, values.to_frame())
        assert pd.api.types.is_datetime64_any_dtype(frame["ds"])

    def test_ds_stays_datetime_when_every_row_has_a_null_ds(self):
        """A LiftEstimate row with ds=None (the default for every total-grain
        estimate) must not downgrade the column to string: the declared
        annotation already says datetime, and no row supplies a conflicting
        type."""
        from increment.estimation.results import Estimate, LiftEstimate

        frame = cast(
            pd.DataFrame,
            to_frame(
                [
                    LiftEstimate(
                        metric="rev",
                        group_id="T",
                        method="unadjusted",
                        method_role="decision",
                        lift=Estimate(value=0.1),
                    )
                ]
            ),
        )
        assert pd.api.types.is_datetime64_any_dtype(frame["ds"]), (
            "all-null ds must stay real datetime64, not object/string dtype"
        )

    def test_empty_list_columns_differ_from_daily_metric_value(self):
        """Each concrete collection carries its own ``_model``: an empty
        ``DailyMetricValues([])`` and ``DailyLiftEstimates([])`` must
        resolve to DIFFERENT, correctly-typed column sets even with no
        instance to introspect - the schema comes from the class, not
        from peeking at the (empty) list."""
        value_frame = DailyMetricValues([]).to_frame()
        lift_frame = DailyLiftEstimates([]).to_frame()
        assert "value" in value_frame.columns and "n" in value_frame.columns
        assert "lift" in lift_frame.columns and "method" in lift_frame.columns
        assert "n" not in lift_frame.columns
        assert "method" not in value_frame.columns

    def test_carries_dimension_columns_when_present(self):
        """dimension/dimension_value/source populate when a
        DailyLiftEstimate carries them and stay None/NaN otherwise - the
        frame's column set is unconditional either way."""
        values = DailyLiftEstimates(
            [
                DailyLiftEstimate(
                    ds=date(2025, 1, 15),
                    metric="purchase_rate",
                    group_id="T",
                    method="unadjusted",
                    method_role="decision",
                    lift=Estimate(value=0.30, lb=0.25, ub=0.35, level=0.95),
                    dimension="country",
                    dimension_value="US",
                    source="events",
                ),
                DailyLiftEstimate(
                    ds=date(2025, 1, 16),
                    metric="purchase_rate",
                    group_id="T",
                    method="unadjusted",
                    method_role="decision",
                    lift=Estimate(value=0.22, lb=0.18, ub=0.26, level=0.95),
                ),
            ]
        )
        frame = cast(pd.DataFrame, values.to_frame())
        assert frame.loc[0, "dimension"] == "country"
        assert frame.loc[0, "dimension_value"] == "US"
        assert frame.loc[0, "source"] == "events"
        assert pd.isna(frame.loc[1, "dimension"])
        assert pd.isna(frame.loc[1, "dimension_value"])
        assert pd.isna(frame.loc[1, "source"])

    def test_backend_pyarrow_returns_a_pyarrow_table(self):
        """backend="pyarrow" builds a native pyarrow.Table via narwhals -
        the same cross-backend proof for the sibling model."""
        values = DailyLiftEstimates(
            [
                DailyLiftEstimate(
                    ds=date(2025, 1, 15),
                    metric="m",
                    group_id="T",
                    method="unadjusted",
                    method_role="decision",
                    lift=Estimate(value=0.3, lb=0.25, ub=0.35, level=0.95),
                ),
            ]
        )
        frame = values.to_frame(backend="pyarrow")
        assert isinstance(frame, pa.Table)
        assert frame.column("lift")[0].as_py() == pytest.approx(0.3)
        assert frame.column("method")[0].as_py() == "unadjusted"


# run_daily / run_daily_lift return the EstimateList subclass, not a bare
# list, so `.to_frame()` works on a pipeline's own output unwrapped.


class TestRunDailyReturnsDailyMetricValues:
    def test_run_daily_returns_daily_metric_values_with_rows(self):
        rows = [_make_daily_row(n=10, mean=5.0, var=4.0, ds=date(2025, 1, 1))]
        result = run_daily(rows, [_mean_metric()])
        assert isinstance(result, DailyMetricValues)
        assert isinstance(result, list)
        assert isinstance(result.to_frame(), pd.DataFrame)

    def test_run_daily_returns_daily_metric_values_when_empty(self):
        """Every metric filtered to nothing (empty summary) still returns
        the concrete collection type, not a bare `[]`."""
        result = run_daily([], [_mean_metric()])
        assert isinstance(result, DailyMetricValues)
        assert len(result.to_frame()) == 0


class TestRunDailyLiftReturnsDailyLiftEstimates:
    def test_run_daily_lift_returns_daily_lift_estimates_with_rows(self):
        rows = [
            _make_daily_row(n=50, mean=10.0, var=4.0, ds=date(2025, 1, 1), group_id="control"),
            _make_daily_row(n=50, mean=12.0, var=4.0, ds=date(2025, 1, 1), group_id="treatment"),
        ]
        result = run_daily_lift(rows, [_mean_metric()], control_group="control")
        assert isinstance(result, DailyLiftEstimates)
        assert isinstance(result, list)
        assert {row.policy_name for row in result} == {"default_exploratory"}
        assert isinstance(result.to_frame(), pd.DataFrame)

    def test_run_daily_lift_returns_daily_lift_estimates_when_empty(self):
        result = run_daily_lift([], [_mean_metric()], control_group="control")
        assert isinstance(result, DailyLiftEstimates)
        assert len(result.to_frame()) == 0


# run_daily_lift(view="asof", design=Encouragement(...)): the additive
# `estimand="late"` dispatch, additional to the existing itt-family rows.


class TestRunDailyLiftAsofEncouragementLate:
    def test_weak_day_omits_late_and_keeps_itt_and_compliance(self):
        design = _encouragement_design(min_first_stage_z=4.0)
        rows = _encouragement_day_rows(
            ds=date(2025, 1, 1), n=2000, tau=2.0, compliance=0.005, seed=1
        )

        results = run_daily_lift(
            rows,
            [_mean_metric("rev")],
            control_group="control",
            design=design,
            view="asof",
        )

        assert {row.estimand for row in results} == {"itt", "compliance"}
        assert not [row for row in results if row.estimand == "late"]
        (compliance,) = [row for row in results if row.estimand == "compliance"]
        assert compliance.metric == f"{_mean_metric('rev').name}_uptake"
        assert "late suppressed" in (compliance.note or "")

    def test_late_only_weak_day_keeps_explanatory_compliance(self):
        rows = _encouragement_day_rows(
            ds=date(2025, 1, 1), n=2000, tau=2.0, compliance=0.005, seed=1
        )

        results = run_daily_lift(
            rows,
            [_mean_metric("rev")],
            control_group="control",
            design=_encouragement_design(min_first_stage_z=4.0),
            view="asof",
            estimands=("late",),
        )

        assert {row.estimand for row in results} == {"compliance"}
        assert results[0].note

    def test_itt_only_skips_encouragement_diagnostics(self, monkeypatch):
        rows = _encouragement_day_rows(ds=date(2025, 1, 1), n=4000, tau=2.0, compliance=0.5, seed=7)
        calls = 0

        def unexpected(*args, **kwargs):
            nonlocal calls
            calls += 1
            raise AssertionError("estimate_encouragement must not run")

        monkeypatch.setattr("increment.breakout.estimates.estimate_encouragement", unexpected)
        results = run_daily_lift(
            rows,
            [_mean_metric("rev")],
            control_group="control",
            design=_encouragement_design(),
            view="asof",
            estimands=("itt",),
        )

        assert calls == 0
        assert results and {row.estimand for row in results} == {"itt"}

    def test_asof_encouragement_scopes_compliance_to_each_metric(self):
        metrics, rows = _two_metric_encouragement_rows()
        results = run_daily_lift(
            rows,
            metrics,
            control_group="control",
            design=_encouragement_design(min_first_stage_z=0.1),
            view="asof",
        )
        assert {row.metric for row in results if row.estimand == "compliance"} == {
            f"{metric.name}_uptake" for metric in metrics
        }

    def test_asof_encouragement_dimension_correction_reaches_every_estimand(self):
        rows = [
            dict(row, country=country)
            for country, seed in (("US", 7), ("CA", 8))
            for row in _encouragement_day_rows(
                ds=date(2025, 1, 1), n=4000, tau=2.0, compliance=0.5, seed=seed
            )
        ]
        uncorrected = run_daily_lift(
            rows,
            [_mean_metric()],
            control_group="control",
            dimension="country",
            design=_encouragement_design(min_first_stage_z=0.1),
            view="asof",
        )
        corrected = run_daily_lift(
            rows,
            [_mean_metric()],
            control_group="control",
            dimension="country",
            design=_encouragement_design(min_first_stage_z=0.1),
            view="asof",
            correction="bonferroni",
        )
        assert {row.dimension_value for row in corrected} == {"US", "CA"}

        def by_key(rows):
            return {(row.dimension_value, row.estimand, row.value_scale): row for row in rows}

        plain, adjusted = by_key(uncorrected), by_key(corrected)
        for key in adjusted:
            assert (
                adjusted[key].require_lift().ub - adjusted[key].require_lift().lb
                >= plain[key].require_lift().ub - plain[key].require_lift().lb
            )

    def test_asof_encouragement_plan_reaches_compliance_and_late(self):
        from increment.plan import compile_decision_plan
        from increment.semantics.models import AnalysisPlan

        metric, other = _mean_metric("rev"), _mean_metric("other")
        design = _encouragement_design(min_first_stage_z=0.1)
        plan = compile_decision_plan(
            AnalysisPlan(alpha=0.05, primary=("rev", "other")),
            [metric, other],
            path="warehouse",
            design=design,
        )
        rows = _encouragement_day_rows(ds=date(2025, 1, 1), n=4000, tau=2.0, compliance=0.5, seed=7)

        results = run_daily_lift(
            rows,
            [metric],
            control_group="control",
            design=design,
            view="asof",
            plan=plan,
        )
        diagnostics = [
            row
            for row in results
            if row.estimand in ("compliance", "late") and row.lift is not None
        ]
        assert diagnostics
        # 2 declared primaries, one non-control arm -> one combined division.
        expected = _alpha_split(0.05, 2)
        assert all(row.role == "primary" for row in diagnostics)
        assert all(
            row.lift is not None and row.require_lift().alpha == expected for row in diagnostics
        )

    def test_one_sided_violation_raises_instead_of_becoming_nan(self):
        """A declared one-sided design whose control arm shows real
        uptake is an instrumentation bug, not per-slice estimability - it
        must raise a hard `ValueError`, never become a NaN `late` row."""
        design = _encouragement_design(one_sided=True, min_first_stage_z=4.0)
        rows = [
            _make_daily_row(n=100, mean=10.0, var=4.0, ds=date(2025, 1, 1), group_id="control")
            | {"sum_d": 5.0, "cyd": 10.0, "cy2d": 20.0},
            _make_daily_row(n=100, mean=12.0, var=4.0, ds=date(2025, 1, 1), group_id="treatment")
            | {"sum_d": 50.0, "cyd": 100.0, "cy2d": 200.0},
        ]

        with pytest.raises(InvalidRequestError) as raised:
            run_daily_lift(
                rows,
                [_mean_metric("rev")],
                control_group="control",
                design=design,
                view="asof",
            )
        assert raised.value.code == "estimation.encouragement.one_sided_encouragement"

    def test_strong_day_recovers_additive_late_matching_estimate_encouragement(self):
        """A strong first stage recovers the SAME additive `late` value
        `estimate_encouragement` itself computes on those exact moments -
        the free-function dispatch is a thin per-day wrapper, not a
        re-derivation."""
        design = _encouragement_design(min_first_stage_z=4.0)
        rows = _encouragement_day_rows(ds=date(2025, 1, 1), n=4000, tau=2.0, compliance=0.5, seed=7)
        expected = [
            r
            for r in estimate_encouragement(
                [_mean_metric("rev")], rows, design, estimands=("late",)
            ).results
            if r.estimand == "late" and r.value_scale == "absolute"
        ][0]

        results = run_daily_lift(
            rows,
            [_mean_metric("rev")],
            control_group="control",
            design=design,
            view="asof",
        )

        late = [r for r in results if r.estimand == "late" and r.value_scale == "absolute"]
        assert len(late) == 1
        late_lift = late[0].lift
        expected_lift = expected.lift
        assert late_lift is not None and expected_lift is not None
        assert late_lift.value == pytest.approx(expected_lift.value)
        assert late_lift.lb == pytest.approx(expected_lift.lb)
        assert late[0].ds == date(2025, 1, 1)
        # Sanity: the DGP's true additive LATE is tau=2.0, recovered here at
        # n=4000/arm - not a coincidental match against a mis-wired NaN.
        assert late_lift.value == pytest.approx(2.0, abs=0.5)

    def test_daily_view_with_encouragement_design_never_reaches_late_branch(self):
        """The branch requires `view == "asof"`; a plain `view="daily"`
        call with the same Encouragement design must behave exactly as
        before this parameter existed: itt rows only, no `late` rows."""
        design = _encouragement_design(min_first_stage_z=4.0)
        rows = _encouragement_day_rows(
            ds=date(2025, 1, 1), n=2000, tau=2.0, compliance=0.005, seed=1
        )

        results = run_daily_lift(
            rows, [_mean_metric("rev")], control_group="control", design=design, view="daily"
        )

        assert {r.estimand for r in results} == {"itt"}

    def test_asof_with_variance_reduction_method_refuses_missing_late_covariate(self):
        """CUPED is forwarded to LATE rather than silently falling back to
        unadjusted; these deliberately incomplete moment rows omit the
        uptake cross-moment ``cxd`` CUPED LATE needs, so it must refuse
        explicitly."""
        from increment.estimation.engine import Method

        design = _encouragement_design(min_first_stage_z=4.0)
        rows = _encouragement_day_rows(ds=date(2025, 1, 1), n=4000, tau=2.0, compliance=0.5, seed=7)
        # Supply the covariate moments CUPED uses for ITT, but deliberately
        # omit cxd to model an older/partial summary that lacks this slot.
        for row in rows:
            n = row["n"]
            row["ref_x"] = 5.0
            row["cx1"] = 0.0
            row["cx2"] = n * 1.0
            row["cxy"] = 0.0
            row["x_role"] = "covariate"

        with pytest.raises(InvalidRequestError) as raised:
            run_daily_lift(
                rows,
                [_mean_metric("rev")],
                control_group="control",
                design=design,
                view="asof",
                methods=[Method(name="cuped", variance_reduction="cuped")],
            )
        assert raised.value.code == "estimation.encouragement.cuped_adjusted_late"

    def test_asof_strong_ratio_first_stage_with_empty_methods_emits_no_late_rows(self):
        """An explicit empty methods list disables LATE before the ratio
        metric's unsupported-LATE guard runs."""
        n = 1000
        rows = [
            centered_row_from_raw_sums(
                {
                    "ds": date(2025, 1, 1),
                    "experiment_id": "exp1",
                    "metric": "rev_per_session",
                    "group_id": group_id,
                    "n": n,
                    "sum_y": sum_y,
                    "sum_y2": sum_y2,
                    "sum_x": None,
                    "sum_x2": None,
                    "sum_xy": None,
                    "sum_den": 2.0 * n,
                    "sum_den2": 4.0 * n,
                    "sum_yden": 2.0 * sum_y,
                    "sum_d": sum_d,
                    "sum_yd": sum_yd,
                    "sum_y2d": sum_y2d,
                }
            )
            for group_id, sum_y, sum_y2, sum_d, sum_yd, sum_y2d in (
                ("control", 10.0 * n, 104.0 * n, 0.0, 0.0, 0.0),
                ("treatment", 11.0 * n, 125.0 * n, 500.0, 5500.0, 62500.0),
            )
        ]

        results = run_daily_lift(
            rows,
            [_ratio_metric()],
            control_group="control",
            design=_encouragement_design(min_first_stage_z=4.0),
            view="asof",
            methods=[],
        )

        assert not [row for row in results if row.estimand == "late"]

    def test_thin_day_late_row_flagged_low_reliability_healthy_day_not(self):
        """`reliability_floor` covers the additive `late` dispatch too: a
        thin day's real LATE row carries `low_reliability=True` by the
        same either-arm-below-floor rule the itt rows use (the IV-ratio
        interval is more fragile at small n). A healthy day's LATE stays
        False."""
        design = _encouragement_design(min_first_stage_z=4.0)
        thin = _encouragement_day_rows(ds=date(2025, 1, 1), n=40, tau=2.0, compliance=0.5, seed=7)
        healthy = _encouragement_day_rows(
            ds=date(2025, 1, 2), n=4000, tau=2.0, compliance=0.5, seed=8
        )

        results = run_daily_lift(
            thin + healthy,
            [_mean_metric("rev")],
            control_group="control",
            design=design,
            view="asof",
        )

        late = {r.ds: r for r in results if r.estimand == "late" and r.value_scale == "absolute"}
        # Both days' LATE rows are real (n=40 per arm at compliance=0.5
        # clears min_first_stage_z=4) - the floor flags, never NaNs.
        assert late[date(2025, 1, 1)].lift is not None
        assert late[date(2025, 1, 1)].low_reliability is True
        assert late[date(2025, 1, 2)].low_reliability is False
        # The same thin day's itt row agrees - the two dispatches share
        # one both-arms bookkeeping rule, not two divergent ones.
        itt = {r.ds: r for r in results if r.estimand == "itt" and r.group_id == "treatment"}
        assert itt[date(2025, 1, 1)].low_reliability is True
        assert itt[date(2025, 1, 2)].low_reliability is False

    def test_late_floor_checks_control_arm_too(self):
        """Asymmetric arms pin the EITHER-arm rule on the late dispatch:
        thin control (n=40) beside healthy treatment (n=4000) must still
        flag - an `or`-to-`and` mutation would pass the symmetric-arms
        test but fail here. Mirrors
        `test_daily_lift_floor_checks_control_arm_too` for itt."""
        design = _encouragement_design(min_first_stage_z=4.0)
        rows = _encouragement_day_rows(
            ds=date(2025, 1, 1), n=4000, n_control=40, tau=2.0, compliance=0.5, seed=9
        )

        results = run_daily_lift(
            rows,
            [_mean_metric("rev")],
            control_group="control",
            design=design,
            view="asof",
        )

        late = [r for r in results if r.estimand == "late" and r.value_scale == "absolute"]
        assert len(late) == 1
        assert late[0].lift is not None
        assert late[0].low_reliability is True


# run_daily_lift: a degenerate metric's day-slice skip must not suppress a
# healthy sibling metric's estimate in the same day slice.


def _daily_row(n, mean, var, *, metric, group_id, ds):
    from increment.estimation.armstats import centered_row_from_raw_sums

    sum_y = n * mean
    row = centered_row_from_raw_sums(
        {
            "experiment_id": "exp1",
            "metric": metric,
            "group_id": group_id,
            "n": float(n),
            "sum_y": sum_y,
            "sum_y2": var * (n - 1) + sum_y**2 / n,
            "sum_x": None,
            "sum_x2": None,
            "sum_xy": None,
            "sum_den": None,
            "sum_den2": None,
            "sum_yden": None,
        }
    )
    row["ds"] = ds
    return row


def test_degenerate_metric_does_not_suppress_its_siblings_in_a_day_slice():
    """`flat` has zero variance in both arms and is legitimately unestimable;
    `revenue` in the same day slice is healthy and must keep its own lift,
    as run_breakout already guarantees per metric."""
    import datetime as dt

    ds = dt.date(2024, 1, 1)
    rows = [
        _daily_row(1000, 10.0, 4.0, metric="revenue", group_id="control", ds=ds),
        _daily_row(1000, 11.0, 4.0, metric="revenue", group_id="treatment", ds=ds),
        _daily_row(1000, 5.0, 0.0, metric="flat", group_id="control", ds=ds),
        _daily_row(1000, 5.0, 0.0, metric="flat", group_id="treatment", ds=ds),
    ]
    metrics = [
        MeanMetric(name="revenue", entity="user", fact="revenue", window_days=30),
        MeanMetric(name="flat", entity="user", fact="flat", window_days=30),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_daily_lift(rows, metrics, control_group="control")

    by_metric = {row.metric: row for row in results}
    assert by_metric["flat"].unavailable == "zero_variance"
    revenue = by_metric["revenue"]
    assert revenue.unavailable is None, f"suppressed by a degenerate sibling: {revenue.unavailable}"
    assert revenue.lift is not None
    assert revenue.require_lift().value == pytest.approx(0.1)
