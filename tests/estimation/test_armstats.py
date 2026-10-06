"""ArmStats / SummaryStats: the centered seam and its format-1 adapter."""

import math
import warnings
from fractions import Fraction

import numpy as np
import pytest

from increment.errors import CapabilityError, IncrementRuntimeWarning, InvalidRequestError
from increment.estimation.armstats import (
    ArmStats,
    BinomialDataError,
    ScoreStats,
    SummaryStats,
    binary_counts,
    canonical_bernoulli_arm,
    centered_row_from_raw_sums,
)
from increment.estimation.engine import _df_to_arms, estimate_lift
from increment.estimation.results import Estimate, LiftEstimate
from increment.estimation.variance import se_log_mean
from increment.semantics.models import MeanMetric
from tests.estimation._conversion_counts import producer_arm
from tests.warning_codes import warning_codes


def _mean_metric(name: str = "rev") -> MeanMetric:
    """A minimal mean metric, so an estimate can be driven from raw rows."""
    return MeanMetric(name=name, entity="user", fact=name)


def _arm(**overrides: object) -> ArmStats:
    """Build a minimal valid ArmStats, overriding any field."""
    base = ArmStats.from_raw_sums(
        study_id="exp1", metric="rev", group_id="A", n=10, sum_y=50.0, sum_y2=300.0
    )
    return base.model_copy(update=overrides)


def _centered(y: np.ndarray, **extra: float) -> dict[str, float]:
    """The v2 wire fields a producer would emit for *y*."""
    ref = float(np.mean(y))
    return {
        "n": len(y),
        "ref_y": ref,
        "cy1": float(np.sum(y - ref)),
        "cy2": float(np.sum((y - ref) ** 2)),
        **extra,
    }


def _native(y: np.ndarray, **extra: float) -> ArmStats:
    # Stamp the covariate role whenever a covariate mean is supplied, so the
    # built arm keeps the x_role-set-iff-ref_x-set invariant combine() relies on.
    fields = _centered(y, **extra)
    x_role = "covariate" if fields.get("ref_x") is not None else None
    return ArmStats.model_validate(
        {"study_id": "exp1", "metric": "rev", "group_id": "A", "x_role": x_role, **fields}
    )


class TestArmStats:
    def test_frozen(self):
        """ArmStats instances are immutable."""
        s = ArmStats(
            study_id="exp1",
            metric="revenue",
            group_id="A",
            n=100,
            ref_y=5.0,
            cy1=0.0,
            cy2=500.0,
        )
        with pytest.raises((TypeError, ValueError)):
            s.n = 200  # ty: ignore[invalid-assignment]  # proving frozen at runtime

    def test_to_summary_matches_numpy(self):
        """to_summary().var equals np.var(ddof=1) on the underlying unit-level data."""
        n = 100
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 10, size=n)
        summary = _native(data).to_summary()
        assert summary.n == n
        assert summary.mean == pytest.approx(data.mean(), rel=1e-12)
        assert summary.var == pytest.approx(np.var(data, ddof=1), rel=1e-12)

    def test_summary_stats_var_formula(self):
        """A format-1 record still reduces to (sum_y2 - sum_y**2/n) / (n-1)."""
        n, sum_y, sum_y2 = 50, 250.0, 1500.0
        stats = ArmStats.from_raw_sums(
            study_id="exp1", metric="rev", group_id="A", n=n, sum_y=sum_y, sum_y2=sum_y2
        )
        s = stats.to_summary()
        assert s.var == pytest.approx((sum_y2 - sum_y**2 / n) / (n - 1), rel=1e-12)
        assert s.mean == pytest.approx(sum_y / n, rel=1e-12)

    def test_var_y_matches_numpy(self):
        """var_y() equals np.var(ddof=1)."""
        rng = np.random.default_rng(99)
        data = rng.normal(5, 2, size=75)
        assert _native(data).var_y() == pytest.approx(np.var(data, ddof=1), rel=1e-12)

    def test_cov_yx_matches_numpy(self):
        """cov_yx() equals np.cov(y, x, ddof=1)[0,1]."""
        n = 60
        rng = np.random.default_rng(123)
        y = rng.normal(10, 3, size=n)
        x = rng.normal(5, 1, size=n) + 0.3 * y  # correlated
        rx = float(np.mean(x))
        stats = _native(
            y,
            ref_x=rx,
            cx1=float(np.sum(x - rx)),
            cx2=float(np.sum((x - rx) ** 2)),
            cxy=float(np.sum((x - rx) * (y - np.mean(y)))),
        )
        assert stats.cov_yx() == pytest.approx(np.cov(y, x, ddof=1)[0, 1], rel=1e-12)
        assert stats.var_x() == pytest.approx(np.var(x, ddof=1), rel=1e-12)
        assert stats.mean_x() == pytest.approx(x.mean(), rel=1e-12)

    def test_cov_yx_needs_covariate(self):
        """cov_yx() raises InvalidRequestError when the covariate family is absent."""
        with pytest.raises(InvalidRequestError) as exc_info:
            _arm().cov_yx()
        assert exc_info.value.code == "estimation.armstats.arm_stats.needs_covariate_cxy"

    def test_n_less_than_two_raises(self):
        """to_summary() raises InvalidRequestError when n < 2 (can't compute ddof=1 variance)."""
        stats = _arm(n=1)
        with pytest.raises(InvalidRequestError) as exc_info:
            stats.to_summary()
        assert exc_info.value.code == "estimation.armstats.arm_stats.least_compute_metric"

    def test_var_y_n_less_than_two_raises(self):
        """var_y() raises InvalidRequestError when n < 2."""
        with pytest.raises(InvalidRequestError) as exc_info:
            _arm(n=1).var_y()
        assert exc_info.value.code == "estimation.armstats.arm_stats.least_compute_metric"

    def test_nonpositive_n_refused_at_construction(self):
        """n=-3 used to construct silently (flipping the sign of every
        /(n-1) correction until a later guard happened to catch it) and
        n=0 died as a raw ZeroDivisionError in to_summary(). A
        group_summary row exists only for a non-empty arm, so n >= 1 is a
        construction-time invariant."""
        for bad_n in (0, -3):
            with pytest.raises(InvalidRequestError) as raised:
                ArmStats(
                    study_id="exp1",
                    metric="rev",
                    group_id="A",
                    n=bad_n,
                    ref_y=5.0,
                    cy1=0.0,
                    cy2=1.0,
                )
            assert raised.value.code == "model.field.range"

    def test_armstats_study_id_field(self):
        arm = _arm(study_id="exp1")
        assert arm.study_id == "exp1"
        with pytest.raises(InvalidRequestError) as raised:
            ArmStats(
                experiment_id="exp1",
                metric="m",
                group_id="T",
                n=10,
                ref_y=0.5,
                cy1=0.0,
                cy2=4.0,
            )  # ty: ignore[missing-argument]  # old name rejected (frozen schema, no alias)
        assert raised.value.code == "model.field.missing"

    @pytest.mark.parametrize("field", ["sum_w", "sum_w2"])
    @pytest.mark.parametrize("validate", [False, True], ids=["construct", "model_validate"])
    def test_direct_inputs_refuse_obsolete_weight_fields(self, field: str, validate: bool):
        payload = {
            "study_id": "exp1",
            "metric": "rev",
            "group_id": "A",
            "n": 10,
            "ref_y": 5.0,
            "cy1": 0.0,
            "cy2": 1.0,
            field: 4.0,
        }
        with pytest.raises(InvalidRequestError) as exc_info:
            if validate:
                ArmStats.model_validate(payload)
            else:
                ArmStats(**payload)  # ty: ignore[invalid-argument-type]
        assert exc_info.value.code == "estimation.armstats.arm_stats.obsolete_armstats_fields"

    def test_negative_cy2_is_refused_not_clamped(self):
        """A sum of squares cannot be negative; corrupt input must not be
        silently absorbed into a zero variance."""
        arm = _arm(cy2=-5.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            arm.var_y()
        assert exc_info.value.code == "estimation.armstats.arm_stats.centered_sum_squares"


@pytest.mark.parametrize("field", ["ref_y", "cy1", "cy2"])
def test_armstats_rejects_nonfinite_core_moments(field):
    with pytest.raises(InvalidRequestError) as exc_info:
        ArmStats.model_validate({**_arm().model_dump(), field: float("inf")})
    assert exc_info.value.code == "estimation.armstats.arm_stats.group_summary_moment"


@pytest.mark.parametrize(
    ("family", "fields"),
    [
        ("covariate", ("ref_x", "cx1", "cx2", "cxy")),
        ("denominator", ("ref_den", "cden1", "cden2", "cyden")),
        ("uptake", ("sum_d", "cyd", "cy2d")),
    ],
)
def test_armstats_rejects_partial_optional_families(family, fields):
    payload = _arm().model_dump()
    payload[fields[0]] = 1.0
    with pytest.raises(InvalidRequestError) as exc_info:
        ArmStats.model_validate(payload)
    assert exc_info.value.code == "estimation.armstats.arm_stats.partial_family_missing"


def test_armstats_requires_declared_x_role_for_materialized_x():
    payload = {
        **_arm().model_dump(),
        "ref_x": 1.0,
        "cx1": 0.0,
        "cx2": 2.0,
        "cxy": 0.5,
    }
    with pytest.raises(InvalidRequestError) as exc_info:
        ArmStats.model_validate(payload)
    assert exc_info.value.code == "estimation.armstats.arm_stats.x_role_declared"


@pytest.mark.parametrize(
    ("role", "code"),
    [
        ("invalid", "estimation.armstats.arm_stats.x_role_one"),
        (None, "estimation.armstats.arm_stats.x_role_declared"),
    ],
)
def test_armstats_rejects_invalid_x_role(role, code):
    payload = {**_arm().model_dump(), "x_role": role}
    if role is None:
        payload.update({"ref_x": 1.0, "cx1": 0.0, "cx2": 2.0, "cxy": 0.5})
    with pytest.raises(InvalidRequestError) as exc_info:
        ArmStats.model_validate(payload)
    assert exc_info.value.code == code


@pytest.mark.parametrize(
    ("field", "prerequisites", "code"),
    [
        (
            "cxden",
            {"ref_x": 1.0, "cx1": 0.0, "cx2": 2.0, "cxy": 0.5, "x_role": "covariate"},
            "estimation.armstats.arm_stats.cxden_complete_denominator",
        ),
        (
            "cxd",
            {"ref_x": 1.0, "cx1": 0.0, "cx2": 2.0, "cxy": 0.5, "x_role": "covariate"},
            "estimation.armstats.arm_stats.cxd_complete_uptake",
        ),
    ],
)
def test_armstats_rejects_cross_moments_without_prerequisite_families(field, prerequisites, code):
    with pytest.raises(InvalidRequestError) as exc_info:
        ArmStats.model_validate({**_arm().model_dump(), **prerequisites, field: 0.0})
    assert exc_info.value.code == code


class TestSummaryStats:
    def test_frozen(self):
        s = SummaryStats(n=100, mean=5.0, var=1.0)
        with pytest.raises((TypeError, ValueError)):
            s.mean = 6.0  # ty: ignore[invalid-assignment]  # proving frozen at runtime

    def test_basic_construction(self):
        s = SummaryStats(n=10, mean=3.5, var=0.25)
        assert s.n == 10
        assert s.mean == 3.5
        assert s.var == 0.25

    def test_zero_n_refused(self):
        """n=0 violates SummaryStats' declared lower bound."""
        with pytest.raises(InvalidRequestError) as raised:
            SummaryStats(n=0, mean=5.0, var=1.0)
        assert raised.value.code == "model.field.range"

    def test_field_range_violation_is_coded(self):
        with pytest.raises(InvalidRequestError) as raised:
            SummaryStats(n=0, mean=0.0, var=1.0)
        assert raised.value.code == "model.field.range"
        assert raised.value.context["model"] == "SummaryStats"
        assert raised.value.context["field"] == "n"

    def test_negative_var_refused(self):
        """var=-1 violates SummaryStats' declared lower bound."""
        with pytest.raises(InvalidRequestError) as raised:
            SummaryStats(n=10, mean=5.0, var=-1.0)
        assert raised.value.code == "model.field.range"

    def test_nan_var_refused(self):
        with pytest.raises(InvalidRequestError) as raised:
            SummaryStats(n=10, mean=5.0, var=math.nan)
        assert raised.value.code == "model.field.range"

    def test_infinite_var_refused_by_name(self):
        """``var`` fails pydantic's ``ge=0.0`` field constraint for NaN, but
        ``+inf`` passes it and must be caught by the custom finiteness check."""
        with pytest.raises(InvalidRequestError) as exc_info:
            SummaryStats(n=10, mean=5.0, var=math.inf)
        assert exc_info.value.code == "estimation.armstats.summary_stats.var_finite"


class TestDfToArmsIngress:
    def test_df_to_arms_maps_wire_experiment_id_to_study_id(self):
        rows = [
            {
                "experiment_id": "e",
                "metric": "m",
                "group_id": "T",
                "n": 5,
                "ref_y": 0.4,
                "cy1": 0.0,
                "cy2": 1.5,
            }
        ]
        arms = _df_to_arms(rows)
        assert arms[0].study_id == "e"

    @pytest.mark.parametrize("backend", ["rows", "arrow", "pandas"])
    @pytest.mark.parametrize("failures", [0, 513])
    def test_large_trial_counts_do_not_round_at_dataframe_ingress(self, backend, failures):
        import pandas as pd
        import pyarrow as pa

        n = 2**61 + 1
        successes = n - failures
        ref = successes / n
        rows = [
            {
                "experiment_id": "e",
                "metric": "conv",
                "group_id": "control",
                "n": n,
                "successes": successes,
                "ref_y": ref,
                "cy1": float(Fraction(successes) - n * Fraction(ref)),
                "cy2": float(Fraction(successes * failures, n)),
            }
        ]
        frame = (
            rows
            if backend == "rows"
            else (pa.Table.from_pylist(rows) if backend == "arrow" else pd.DataFrame(rows))
        )
        (arm,) = _df_to_arms(frame)
        assert binary_counts(arm, "conversion") == (successes, n)

    @pytest.mark.parametrize("nullable_integer", [False, True])
    def test_pandas_missing_successes_remain_absent_for_mean_rows(self, nullable_integer):
        import pandas as pd

        frame = pd.DataFrame(
            [
                {
                    "experiment_id": "e",
                    "metric": "m",
                    "group_id": "control",
                    "n": 5,
                    "successes": math.nan,
                    "ref_y": 0.4,
                    "cy1": 0.0,
                    "cy2": 1.5,
                }
            ]
        )
        if nullable_integer:
            frame["successes"] = frame["successes"].astype("Int64")
        (arm,) = _df_to_arms(frame)
        assert arm.successes is None
        assert arm.to_summary().mean == pytest.approx(0.4)
        assert arm.to_summary().var == pytest.approx(1.5 / 4)

    @pytest.mark.parametrize("field", ["sum_w", "sum_w2"])
    def test_from_raw_sums_refuses_deprecated_weight_field(self, field: str):
        with pytest.raises(TypeError):
            ArmStats.from_raw_sums(
                study_id="e",
                metric="m",
                group_id="T",
                n=5,
                sum_y=2.0,
                sum_y2=1.5,
                **{field: 4.0},  # ty: ignore[invalid-argument-type]  # deliberately invalid keyword
            )

    @pytest.mark.parametrize("field", ["sum_w", "sum_w2"])
    def test_df_to_arms_refuses_deprecated_weight_field(self, field: str):
        rows = [
            {
                "experiment_id": "e",
                "metric": "m",
                "group_id": "T",
                "n": 5,
                "ref_y": 0.4,
                "cy1": 0.0,
                "cy2": 1.5,
                field: 4.0,
            }
        ]
        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_lift([_mean_metric("m")], rows, control_group="T")
        assert exc_info.value.code == "estimation.engine.carries_unsupported_weighted"

    @pytest.mark.parametrize("field", ["sum_w", "sum_w2"])
    def test_centered_row_refuses_deprecated_weight_field(self, field: str):
        with pytest.raises(InvalidRequestError) as exc_info:
            centered_row_from_raw_sums({"n": 5, "sum_y": 2.0, "sum_y2": 1.5, field: 4.0})
        assert exc_info.value.code == "estimation.armstats.centered_raw_row"
        assert exc_info.value.context == {"unsupported": (field,)}

    def test_df_to_arms_refuses_format_one_rows_by_name(self):
        """Raw additive sums no longer parse. Detection is a REFUSAL, never a
        silent reinterpretation - and it must name the way in."""
        rows = [
            {
                "experiment_id": "e",
                "metric": "m",
                "group_id": "T",
                "n": 5,
                "sum_y": 2.0,
                "sum_y2": 1.5,
            }
        ]
        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_lift([_mean_metric("m")], rows, control_group="T")
        assert exc_info.value.code == "estimation.engine.carries_format_moments"


def _uptake_arm(**over: object) -> ArmStats:
    """Build a valid uptake-moment ArmStats, overriding any field.

    4 units: y = [1, 2, 3, 4], d = [0, 1, 1, 0]  ->  yd = [0, 2, 3, 0],
    y2d = [0, 4, 9, 0].
    """
    base = ArmStats.from_raw_sums(
        study_id="s",
        metric="m",
        group_id="t",
        n=4,
        sum_y=10.0,
        sum_y2=30.0,
        sum_d=2.0,
        sum_yd=5.0,
        sum_y2d=13.0,
    )
    return base.model_copy(update=over)


class TestUptakeReductions:
    def test_uptake_reductions_match_hand_computation(self):
        a = _uptake_arm()
        assert a.mean_d() == pytest.approx(0.5)
        # var_d = (2 - 4/4) / 3 = 1/3
        assert a.var_d() == pytest.approx(1 / 3)
        # cov_yd = (5 - 10*2/4) / 3 = 0
        assert a.cov_yd() == pytest.approx(0.0, abs=1e-15)
        # var_yd = (13 - 25/4) / 3 = 2.25
        assert a.var_yd() == pytest.approx(2.25)
        # cov_y_yd = (13 - 10*5/4) / 3 = 0.5 / 3 = 1/6
        assert a.cov_y_yd() == pytest.approx(1 / 6)
        # mean(y*d) = 5/4; mean(y*(1-d)) = 5/4
        assert a.mean_yd() == pytest.approx(1.25)
        assert a.mean_y_untaken() == pytest.approx(1.25)

    def test_uptake_reductions_refuse_missing_moments(self):
        a = _uptake_arm(sum_d=None, cyd=None, cy2d=None)
        codes = {
            "mean_d": "estimation.armstats.arm_stats.mean_d_needs_uptake_sum",
            "var_d": "estimation.armstats.arm_stats.mean_d_needs_uptake_sum",
            "cov_yd": "estimation.armstats.arm_stats.cov_yd_needs_uptake_sum_cyd",
            "var_yd": "estimation.armstats.arm_stats.needs_uptake_cyd",
            "cov_y_yd": "estimation.armstats.arm_stats.needs_uptake_cyd",
            "mean_yd": "estimation.armstats.arm_stats.needs_uptake_cyd",
        }
        for meth, code in codes.items():
            with pytest.raises(InvalidRequestError) as exc_info:
                getattr(a, meth)()
            assert exc_info.value.code == code


class TestLiftEstimateMetadata:
    def test_lift_estimate_estimand_defaults_keep_backcompat(self):
        le = LiftEstimate(
            metric="m",
            group_id="t",
            method="unadjusted",
            method_role="decision",
            lift=Estimate(value=0.1, lb=0.0, ub=0.2, level=0.95),
        )
        assert le.estimand == "itt"
        assert le.value_scale == "relative"
        assert le.note is None


def test_armstats_accepts_complete_winsorization_metadata():
    arm = ArmStats.model_validate(
        {
            **_arm().model_dump(),
            "winsor_upper_percentile": 0.99,
            "winsor_upper_bound": 100.0,
            "winsor_n": 10,
            "winsor_n_lower": 0,
            "winsor_n_upper": 2,
        }
    )
    assert arm.winsor_n_upper == 2


def test_armstats_rejects_partial_winsorization_metadata():
    with pytest.raises(InvalidRequestError) as exc_info:
        ArmStats.model_validate({**_arm().model_dump(), "winsor_upper_bound": 100.0})
    assert exc_info.value.code == "estimation.armstats.arm_stats.winsorization_metadata_winsor"


def test_armstats_rejects_overlapping_cap_counts():
    with pytest.raises(InvalidRequestError) as exc_info:
        ArmStats.model_validate(
            {
                **_arm().model_dump(),
                "winsor_upper_bound": 100.0,
                "winsor_n": 10,
                "winsor_n_lower": 6,
                "winsor_n_upper": 5,
            }
        )
    assert exc_info.value.code == "estimation.armstats.arm_stats.winsorization_cap_counts"


def test_armstats_rejects_nonfinite_winsorization_metadata():
    with pytest.raises(InvalidRequestError) as exc_info:
        ArmStats.model_validate(
            {
                **_arm().model_dump(),
                "winsor_lower_percentile": float("inf"),
                "winsor_n": 10,
                "winsor_n_lower": 1,
                "winsor_n_upper": 1,
            }
        )
    assert exc_info.value.code == "estimation.armstats.arm_stats.winsorization_metadata_finite"


def test_armstats_rejects_lower_percentile_without_resolved_bound():
    with pytest.raises(InvalidRequestError) as exc_info:
        ArmStats.model_validate(
            {
                **_arm().model_dump(),
                "winsor_lower_percentile": 0.01,
                "winsor_n": 10,
                "winsor_n_lower": 1,
                "winsor_n_upper": 1,
            }
        )
    assert exc_info.value.code == "estimation.armstats.arm_stats.winsorization_lower_percentile"


def test_armstats_rejects_upper_percentile_without_resolved_bound():
    with pytest.raises(InvalidRequestError) as exc_info:
        ArmStats.model_validate(
            {
                **_arm().model_dump(),
                "winsor_upper_percentile": 0.99,
                "winsor_n": 10,
                "winsor_n_lower": 1,
                "winsor_n_upper": 1,
            }
        )
    assert exc_info.value.code == "estimation.armstats.arm_stats.winsorization_upper_percentile"


def test_armstats_rejects_inverted_winsorization_bounds():
    with pytest.raises(InvalidRequestError) as exc_info:
        ArmStats.model_validate(
            {
                **_arm().model_dump(),
                "winsor_lower_bound": 100.0,
                "winsor_upper_bound": 1.0,
                "winsor_n": 10,
                "winsor_n_lower": 1,
                "winsor_n_upper": 1,
            }
        )
    assert exc_info.value.code == "estimation.armstats.arm_stats.winsorization_resolved_lower"


def test_winsorization_result_fields_pair_arms():
    """Each arm's own cap counts reach the result row without a swap."""
    result = estimate_lift(
        [_mean_metric()],
        [
            {
                "experiment_id": "exp1",
                "metric": "rev",
                "group_id": "A",
                "n": 10,
                "ref_y": 5.0,
                "cy1": 0.0,
                "cy2": 50.0,
                "winsor_n": 10,
                "winsor_n_lower": 0,
                "winsor_n_upper": 1,
            },
            {
                "experiment_id": "exp1",
                "metric": "rev",
                "group_id": "B",
                "n": 10,
                "ref_y": 5.0,
                "cy1": 0.0,
                "cy2": 50.0,
                "winsor_n": 12,
                "winsor_n_lower": 0,
                "winsor_n_upper": 2,
            },
        ],
        control_group="A",
    ).results[0]
    assert result.winsor_control_n_upper == 1
    assert result.winsor_treatment_n_upper == 2


class TestFormatOneConditioning:
    """Format-1 (raw additive sums) input keeps format-1's bands.

    When |mean| >> sd, the variance signal in ``sum_y2`` falls below the
    rounding already baked into the stored sums (the crossover is at a
    coefficient of variation near sqrt(machine epsilon), ~1e-8).  The
    adapter must never yield a negative variance, must not crash a
    downstream ``sqrt``, and must not stay silent when the value it keeps
    is smaller than its own floating-point noise floor.  The guard now
    fires where the data ENTERS - ``from_raw_sums`` - instead of deep
    inside a downstream estimator.
    """

    @staticmethod
    def _arm_from(y: np.ndarray) -> ArmStats:
        return ArmStats.from_raw_sums(
            study_id="exp1",
            metric="rev",
            group_id="A",
            n=len(y),
            sum_y=float(np.sum(y)),
            sum_y2=float(np.sum(y * y)),
        )

    _NOISE_CODES = frozenset(
        {
            "estimation.armstats.centered_sum_squares_clamped",
            "estimation.armstats.centered_sum_squares_noise_floor",
        }
    )

    @pytest.mark.parametrize(("seed", "mean"), [(1, 1e8), (2, 1e8), (0, 1e9)])
    def test_noise_dominated_sums_never_negative_or_silent(self, seed, mean):
        """mean >> sd: the true variance (~1) is below the rounding the stored
        sums carry, so whether the naive difference lands negative or positive
        is decided by the producing library's summation order. Either way the
        adapter must warn, keep the variance finite and non-negative, and hand
        a downstream ``sqrt`` a safe value."""
        y = np.random.default_rng(seed).normal(mean, 1.0, 10_000)
        with pytest.warns(IncrementRuntimeWarning) as rec:
            arm = self._arm_from(y)
        assert set(warning_codes(rec)) & self._NOISE_CODES
        assert math.isfinite(arm.var_y()) and arm.var_y() >= 0.0
        math.sqrt(arm.to_summary().var)  # must not raise

    def test_negative_within_rounding_is_clamped_to_zero(self):
        """``sum_y2`` one ulp below ``sum_y**2 / n`` is negative by rounding
        alone (-128 against a slack of thousands): clamp to zero and warn."""
        with pytest.warns(IncrementRuntimeWarning) as rec:
            arm = ArmStats.from_raw_sums(
                study_id="exp1",
                metric="rev",
                group_id="A",
                n=100,
                sum_y=1e10,
                sum_y2=math.nextafter(1e18, 0.0),
            )
        assert "estimation.armstats.centered_sum_squares_clamped" in warning_codes(rec)
        assert arm.var_y() == 0.0

    def test_positive_below_noise_floor_is_clamped_and_warns(self):
        """``sum_y2`` one ulp above ``sum_y**2 / n`` is positive (+128) but
        below the cancelled sums' own noise floor; silence would present
        rounding as a measurement, so the adapter warns and clamps."""
        with pytest.warns(IncrementRuntimeWarning) as rec:
            arm = ArmStats.from_raw_sums(
                study_id="exp1",
                metric="rev",
                group_id="A",
                n=100,
                sum_y=1e10,
                sum_y2=math.nextafter(1e18, math.inf),
            )
        assert "estimation.armstats.centered_sum_squares_noise_floor" in warning_codes(rec)
        assert arm.var_y() == 0.0

    def test_var_y_negative_beyond_noise_refused(self):
        """A centered sum of squares far below zero cannot be rounding;
        the sums are inconsistent.  Refuse loudly, naming the arm."""
        with pytest.raises(InvalidRequestError) as exc_info:
            ArmStats.from_raw_sums(
                study_id="exp1", metric="rev", group_id="A", n=100, sum_y=100.0, sum_y2=5.0
            )
        assert exc_info.value.code == "estimation.armstats.centered_sum_squares"

    def test_saturated_arm_zero_variance_stays_silent(self):
        """A genuinely constant arm (e.g. every unit converted, y=1) has an
        exactly-zero centered sum: no clamp, no warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            arm = ArmStats.from_raw_sums(
                study_id="exp1", metric="conv", group_id="A", n=100, sum_y=100.0, sum_y2=100.0
            )
            assert arm.var_y() == 0.0

    def test_var_y_moderate_offset_stays_accurate(self):
        """mean=1e6 (CV=1e-6) is inside the recoverable zone: the adapter
        must stay within ~1e-3 of numpy's two-pass variance, silently."""
        y = np.random.default_rng(0).normal(1e6, 1.0, 10_000)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            arm = self._arm_from(y)
            assert arm.var_y() == pytest.approx(np.var(y, ddof=1), rel=1e-3)


def _fraction_reductions(y: np.ndarray, d: np.ndarray) -> dict[str, float]:
    """Exact-rational (n-1)-normalised reductions, the only honest reference
    once the float ones start cancelling."""
    n = len(y)
    fy = [Fraction(v) for v in y]
    fd = [Fraction(v) for v in d]
    fyd = [a * b for a, b in zip(fd, fy, strict=True)]
    my = sum(fy) / n
    myd = sum(fyd) / n
    return {
        "var_y": float(sum((v - my) ** 2 for v in fy) / (n - 1)),
        "cov_yd": float(
            sum((a - my) * (b - sum(fd) / n) for a, b in zip(fy, fd, strict=True)) / (n - 1)
        ),
        "var_yd": float(sum((v - myd) ** 2 for v in fyd) / (n - 1)),
        "cov_y_yd": float(
            sum((a - my) * (b - myd) for a, b in zip(fy, fyd, strict=True)) / (n - 1)
        ),
    }


def _native_uptake(y: np.ndarray, d: np.ndarray) -> ArmStats:
    r = float(np.mean(y))
    return ArmStats(
        study_id="s",
        metric="m",
        group_id="t",
        n=len(y),
        ref_y=r,
        cy1=float(np.sum(y - r)),
        cy2=float(np.sum((y - r) ** 2)),
        sum_d=float(np.sum(d)),
        cyd=float(np.sum(d * (y - r))),
        cy2d=float(np.sum(d * (y - r) ** 2)),
    )


class TestSubgroupExpansions:
    """The d-family reductions are analytic expansions, not raw recovery.

    ``cy2d/(n-1)`` is NOT ``Var(d*y)``: the subgroup fields carry only the
    WITHIN-subgroup dispersion, and the between-group component enters
    through ``ref_y**2 * sum_d * (1-p)``.  At mu ~ 1 that term is a
    rounding-scale nuisance and dropping it is invisible; the large-mu
    cases below are what make it load-bearing.
    """

    @pytest.mark.parametrize("mean", [1.0, 1e6, 1e12])
    def test_expansions_match_exact_rational_reference(self, mean: float):
        rng = np.random.default_rng(17)
        n = 1000
        y = rng.normal(mean, 1.0, n)
        d = (rng.random(n) < 0.4).astype(float)
        arm = _native_uptake(y, d)
        exact = _fraction_reductions(y, d)
        for name, want in exact.items():
            assert getattr(arm, name)() == pytest.approx(want, rel=1e-12), name

    def test_between_group_term_dominates_var_yd_at_large_mu(self):
        """Guards the corollary directly: forget the between term and
        Var(d*y) collapses by ~9 orders of magnitude here."""
        rng = np.random.default_rng(19)
        n, mean = 1000, 1e6
        y = rng.normal(mean, 1.0, n)
        d = (rng.random(n) < 0.4).astype(float)
        arm = _native_uptake(y, d)
        assert arm.cy2d is not None
        within_only = arm.cy2d / (n - 1)
        assert arm.var_yd() == pytest.approx(float(np.var(y * d, ddof=1)), rel=1e-12)
        assert arm.var_yd() > 1e8 * within_only


class TestPartitionCombination:
    """Combining partitions must equal the single-pass record.

    Adversarial partitions (single-row chunks, lopsided sizes) and chunk
    scales spread across many orders of magnitude: every centered field is
    a quadratic expansion in the between-partition mean spread, so this is
    associative and order-independent to fp roundoff.
    """

    @staticmethod
    def _arm_over(
        y: np.ndarray, x: np.ndarray, den: np.ndarray, d: np.ndarray, gid: str = "A"
    ) -> ArmStats:
        ry, rx, rd = float(np.mean(y)), float(np.mean(x)), float(np.mean(den))
        return ArmStats(
            study_id="exp1",
            metric="rev",
            group_id=gid,
            n=len(y),
            ref_y=ry,
            cy1=float(np.sum(y - ry)),
            cy2=float(np.sum((y - ry) ** 2)),
            ref_x=rx,
            x_role="covariate",
            cx1=float(np.sum(x - rx)),
            cx2=float(np.sum((x - rx) ** 2)),
            cxy=float(np.sum((x - rx) * (y - ry))),
            ref_den=rd,
            cden1=float(np.sum(den - rd)),
            cden2=float(np.sum((den - rd) ** 2)),
            cyden=float(np.sum((y - ry) * (den - rd))),
            sum_d=float(np.sum(d)),
            cyd=float(np.sum(d * (y - ry))),
            cy2d=float(np.sum(d * (y - ry) ** 2)),
            cxd=float(np.sum(d * (x - rx))),
        )

    REDUCTIONS = (
        "mean_y",
        "mean_x",
        "mean_den",
        "var_y",
        "var_x",
        "var_den",
        "cov_yx",
        "cov_yden",
        "mean_d",
        "var_d",
        "cov_yd",
        "cov_xd",
        "var_yd",
        "cov_y_yd",
        "mean_yd",
        "mean_y_untaken",
    )

    def test_partition_combination_matches_single_pass(self):
        rng = np.random.default_rng(7)
        n = 10_007
        y = rng.lognormal(0.0, 1.0, n)
        x = rng.lognormal(0.5, 0.8, n)
        den = rng.lognormal(0.2, 0.6, n)
        d = (rng.random(n) < 0.3).astype(float)
        # Extreme scale spread aligned with the partition: one chunk ~1e12,
        # one ~1e-6, the rest ~O(1) - the deltas the expansion must absorb.
        scale = np.ones(n)
        scale[1:5000] = 1e12
        scale[5001:] = 1e-6
        y, x, den = y * scale, x * scale, den * scale

        bounds = [0, 2, 5000, 5002, n]
        parts = [
            self._arm_over(y[a:b], x[a:b], den[a:b], d[a:b])
            for a, b in zip(bounds, bounds[1:], strict=False)
        ]
        combined = ArmStats.combine(parts)
        whole = self._arm_over(y, x, den, d)

        assert combined.n == n
        for name in self.REDUCTIONS:
            got, want = getattr(combined, name)(), getattr(whole, name)()
            assert got == pytest.approx(want, rel=1e-12), name

    def test_combine_is_order_independent(self):
        rng = np.random.default_rng(23)
        parts = []
        for i in range(5):
            k = 200 + 37 * i
            y = rng.normal(1e9 + 500.0 * i, 2.0, k)
            x = rng.normal(3.0, 1.0, k)
            den = rng.normal(7.0, 1.0, k)
            d = (rng.random(k) < 0.5).astype(float)
            parts.append(self._arm_over(y, x, den, d))
        forward = ArmStats.combine(parts)
        backward = ArmStats.combine(list(reversed(parts)))
        for name in self.REDUCTIONS:
            assert getattr(forward, name)() == pytest.approx(getattr(backward, name)(), rel=1e-11)

    def test_combine_refuses_mixed_families(self):
        rng = np.random.default_rng(3)
        y = rng.normal(1.0, 1.0, 50)
        full = self._arm_over(y, y + 1, y + 2, (rng.random(50) < 0.5).astype(float))
        bare = _native(y)
        bare = bare.model_copy(update={"metric": "rev", "study_id": "exp1", "group_id": "A"})
        with pytest.raises(InvalidRequestError) as exc_info:
            ArmStats.combine([full, bare])
        assert exc_info.value.code == "estimation.armstats.arm_stats.combine_family_some"

    def test_combine_refuses_unlabelled_cross_arm_merge(self):
        rng = np.random.default_rng(4)
        y = rng.normal(1.0, 1.0, 40)
        a = _native(y).model_copy(update={"group_id": "A"})
        b = _native(y + 1).model_copy(update={"group_id": "B"})
        with pytest.raises(InvalidRequestError) as exc_info:
            ArmStats.combine([a, b])
        assert exc_info.value.code == "estimation.armstats.arm_stats.combine_partitions_from"
        pooled = ArmStats.combine([a, b], group_id="(pooled)")
        assert pooled.group_id == "(pooled)"
        assert pooled.n == 80

    def test_combine_carries_x_role_and_preserves_moments(self):
        """Partitions declaring a non-covariate x role (here cluster_size)
        still combine on the declaration, not the row shape: the pooled arm
        keeps the declared role and its x-family moments equal the
        single-pass record over the concatenated data - the arithmetic is
        unchanged by reading x_role instead of ref_x."""
        rng = np.random.default_rng(101)
        y = rng.normal(1e6, 3.0, 400)
        x = rng.normal(5.0, 1.0, 400)
        den = rng.normal(7.0, 1.0, 400)
        d = (rng.random(400) < 0.5).astype(float)
        bounds = [0, 150, 400]
        parts = [
            self._arm_over(y[a:b], x[a:b], den[a:b], d[a:b]).model_copy(
                update={"x_role": "cluster_size"}
            )
            for a, b in zip(bounds, bounds[1:], strict=False)
        ]
        combined = ArmStats.combine(parts)
        whole = self._arm_over(y, x, den, d).model_copy(update={"x_role": "cluster_size"})
        assert combined.x_role == "cluster_size"
        for name in ("mean_x", "var_x", "cov_yx"):
            assert getattr(combined, name)() == pytest.approx(getattr(whole, name)(), rel=1e-12), (
                name
            )

    def test_combine_refuses_mismatched_x_role(self):
        """Partitions of one population must agree on what the x family
        carries: a covariate arm and a cluster-size arm cannot be pooled
        even though both materialise the x family."""
        rng = np.random.default_rng(102)
        y = rng.normal(1.0, 1.0, 50)
        x = rng.normal(3.0, 1.0, 50)
        den = rng.normal(7.0, 1.0, 50)
        d = (rng.random(50) < 0.5).astype(float)
        covariate = self._arm_over(y, x, den, d)  # x_role="covariate"
        size = self._arm_over(y + 1, x, den, d).model_copy(update={"x_role": "cluster_size"})
        with pytest.raises(InvalidRequestError) as exc_info:
            ArmStats.combine([covariate, size], group_id="(pooled)")
        assert exc_info.value.code == "estimation.armstats.arm_stats.combine_partitions_declaring"

    def test_combine_refuses_an_empty_sequence(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            ArmStats.combine([])
        assert exc_info.value.code == "estimation.armstats.arm_stats.combine_needs_least"

    def test_combine_refuses_mismatched_study_or_metric(self):
        a = _native(np.array([1.0, 2.0, 3.0]))
        b = _native(np.array([1.0, 2.0, 3.0])).model_copy(update={"metric": "other"})
        with pytest.raises(InvalidRequestError) as exc_info:
            ArmStats.combine([a, b], group_id="A")
        assert exc_info.value.code == "estimation.armstats.arm_stats.combine_needs_one"

    def test_combine_refuses_uptake_family_present_on_some_partitions(self):
        with_uptake = _uptake_arm()
        without_uptake = _uptake_arm().model_copy(update={"sum_d": None, "cyd": None, "cy2d": None})
        with pytest.raises(InvalidRequestError) as exc_info:
            ArmStats.combine([with_uptake, without_uptake], group_id="A")
        assert exc_info.value.code == "estimation.armstats.arm_stats.combine_uptake_family"

    def test_combine_refuses_inconsistent_winsorization_metadata(self):
        base = _native(np.array([1.0, 2.0, 3.0, 4.0])).model_copy(
            update={
                "winsor_upper_percentile": 0.99,
                "winsor_upper_bound": 100.0,
                "winsor_n": 4,
                "winsor_n_lower": 0,
                "winsor_n_upper": 1,
            }
        )
        mismatched = base.model_copy(update={"winsor_upper_bound": 200.0})
        with pytest.raises(InvalidRequestError) as exc_info:
            ArmStats.combine([base, mismatched], group_id="A")
        assert (
            exc_info.value.code
            == "estimation.armstats.arm_stats.combine_inconsistent_winsorization"
        )


class TestFormatOneAdapterParity:
    """A format-1 record and the format-2 record of the SAME data must
    reduce to the same numbers at normal scales - that is the whole
    backwards-compatibility contract."""

    def test_v1_loaded_matches_v2_native(self):
        rng = np.random.default_rng(31)
        n = 2000
        y = rng.lognormal(0.0, 0.8, n)
        x = rng.lognormal(0.2, 0.6, n)
        den = rng.lognormal(0.1, 0.5, n)
        d = (rng.random(n) < 0.45).astype(float)

        v1 = ArmStats.from_raw_sums(
            study_id="exp1",
            metric="rev",
            group_id="A",
            n=n,
            sum_y=float(y.sum()),
            sum_y2=float((y * y).sum()),
            sum_x=float(x.sum()),
            sum_x2=float((x * x).sum()),
            sum_xy=float((x * y).sum()),
            sum_den=float(den.sum()),
            sum_den2=float((den * den).sum()),
            sum_yden=float((y * den).sum()),
            sum_d=float(d.sum()),
            sum_yd=float((y * d).sum()),
            sum_y2d=float((y * y * d).sum()),
            sum_xd=float((x * d).sum()),
        )
        v2 = TestPartitionCombination._arm_over(y, x, den, d)
        for name in TestPartitionCombination.REDUCTIONS:
            assert getattr(v1, name)() == pytest.approx(getattr(v2, name)(), rel=1e-9), name

    def test_first_moments_are_exactly_recoverable(self):
        """``sum(y) == n*ref_y + cy1`` with no cancellation: that is why the
        residual first moment is on the wire at all."""
        rng = np.random.default_rng(37)
        y = rng.normal(1e9, 1.0, 5000)
        arm = _native(y)
        recovered = arm.n * Fraction(arm.ref_y) + Fraction(arm.cy1)
        exact = sum(Fraction(v) for v in y)
        assert float(recovered) == pytest.approx(float(exact), rel=1e-15)


class TestUptakeRangeValidation:
    """``sum_d`` is a count of takers and must sit in ``[0, n]``: a negative
    or over-``n`` value cannot arise from a real uptake fact and signals
    corrupt or mis-aggregated ingress."""

    def test_negative_uptake_count_is_rejected(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            ArmStats(
                study_id="s",
                metric="m",
                group_id="t",
                n=4,
                ref_y=2.5,
                cy1=0.0,
                cy2=5.0,
                sum_d=-1.0,
                cyd=0.0,
                cy2d=1.0,
            )
        assert exc_info.value.code == "estimation.armstats.arm_stats.sum_d_outside"

    def test_uptake_count_exceeding_n_is_rejected(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            ArmStats(
                study_id="s",
                metric="m",
                group_id="t",
                n=4,
                ref_y=2.5,
                cy1=0.0,
                cy2=5.0,
                sum_d=5.0,
                cyd=0.0,
                cy2d=1.0,
            )
        assert exc_info.value.code == "estimation.armstats.arm_stats.sum_d_outside"

    def test_uptake_count_at_boundaries_is_accepted(self):
        # The masked moments must match the mask at each boundary: with no
        # taker-up every masked moment is zero, and with all four the masked
        # moments equal the unmasked ones.
        ArmStats(
            study_id="s",
            metric="m",
            group_id="t",
            n=4,
            ref_y=2.5,
            cy1=0.0,
            cy2=5.0,
            sum_d=0.0,
            cyd=0.0,
            cy2d=0.0,
        )
        ArmStats(
            study_id="s",
            metric="m",
            group_id="t",
            n=4,
            ref_y=2.5,
            cy1=0.0,
            cy2=5.0,
            sum_d=4.0,
            cyd=0.0,
            cy2d=5.0,
        )


class TestVarYdRefusesInconsistentMoments:
    """``var_yd`` used to floor any negative analytic-expansion value to
    0.0 unconditionally, hiding infeasible (corrupted / mis-aggregated)
    moments behind a plausible-looking zero variance."""

    def test_grossly_inconsistent_moments_are_refused_not_zeroed(self):
        a = _uptake_arm()
        assert a.var_yd() == pytest.approx(2.25)
        # cy2d far too small for the rest of the uptake family: the
        # analytic expansion goes deeply negative, well beyond the
        # floating-point noise floor -- not merely cancelled.
        corrupted = a.model_copy(update={"cy2d": -1000.0})
        with pytest.raises(InvalidRequestError) as exc_info:
            corrupted.var_yd()
        assert exc_info.value.code == "estimation.armstats.centered_sum_squares"

    def test_ordinary_data_is_unaffected(self):
        """The common (non-adversarial) case is an exact no-op against
        the pre-fix formula -- the policy only changes behaviour once a
        deficit exceeds the floating-point noise floor."""
        a = _uptake_arm()
        assert a.var_yd() == pytest.approx(2.25)


class TestScoreStatsSeCentering:
    """``ScoreStats.se()`` must center via exact (Fraction-based)
    arithmetic, not a raw ``sum_psi2 - sum_psi**2/n`` subtraction that
    catastrophically cancels for scores offset far from zero."""

    def test_refuses_grossly_inconsistent_scores_instead_of_zeroing(self):
        """A ``sum_psi2`` far too small for the declared ``sum_psi``/``n``
        used to silently clamp to se()==0.0 via ``max(negative, 0.0)``;
        it must now be refused as corrupt."""
        n, sum_psi = 1000, 1e8
        sum_psi2 = sum_psi**2 / n - 1e6  # ~1e4 of fp noise at this scale
        s = ScoreStats(metric="m", contrast="T", n=n, sum_psi=sum_psi, sum_psi2=sum_psi2)
        with pytest.raises(InvalidRequestError) as exc_info:
            s.se()
        assert exc_info.value.code == "estimation.armstats.centered_sum_squares"

    def test_recovers_precision_a_naive_subtraction_loses(self):
        """At a large score offset, the naive ``sum_psi2 - sum_psi**2/n``
        subtraction measurably degrades the recovered variance; the
        exact-Fraction centering this now routes through does not."""
        n, offset = 1000, 1e7
        rng = np.random.default_rng(129)
        psi = offset + rng.normal(0.0, 1.0, n)
        sum_psi = float(np.sum(psi))
        sum_psi2 = float(np.sum(psi * psi))
        naive = max(sum_psi2 - sum_psi**2 / n, 0.0)
        s = ScoreStats(metric="m", contrast="T", n=n, sum_psi=sum_psi, sum_psi2=sum_psi2)
        exact = s.se() ** 2 * n * n  # back out the centered sum of squares
        true_signal = float(np.sum((psi - psi.mean()) ** 2))
        assert abs(exact - true_signal) < abs(naive - true_signal)

    def test_mean_zero_scores_are_unaffected(self):
        """The common case (Hajek-normalized IPTW scores, sum_psi == 0)
        must be an exact no-op, matching the pre-fix behaviour."""
        s = ScoreStats(metric="m", contrast="T", n=6, sum_psi=0.0, sum_psi2=11.9212)
        assert s.se() == pytest.approx(math.sqrt(11.9212) / 6, rel=1e-12)

    def test_se_refuses_a_nonpositive_normalizer(self):
        s = ScoreStats(metric="m", contrast="T", n=0, sum_psi=0.0, sum_psi2=0.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            s.se()
        assert exc_info.value.code == "estimation.armstats.score_stats.se_needs_positive"


class TestScoreStatsClusterFieldValidation:
    """``cluster_variance``/``n_clusters`` are the cluster-robust seam and
    must travel together (both set or both None), with ``n_clusters >= 1``
    when set -- both now CodedModel refusals, not a bare ValidationError."""

    def test_cluster_variance_without_n_clusters_refuses(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            ScoreStats(
                metric="m", contrast="T", n=10, sum_psi=0.0, sum_psi2=1.0, cluster_variance=5.0
            )
        assert exc_info.value.code == "estimation.armstats.score_stats.cluster_variance"

    def test_n_clusters_without_cluster_variance_refuses(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            ScoreStats(metric="m", contrast="T", n=10, sum_psi=0.0, sum_psi2=1.0, n_clusters=5)
        assert exc_info.value.code == "estimation.armstats.score_stats.cluster_variance"

    def test_n_clusters_below_one_refuses(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            ScoreStats(
                metric="m",
                contrast="T",
                n=10,
                sum_psi=0.0,
                sum_psi2=1.0,
                cluster_variance=5.0,
                n_clusters=0,
            )
        assert exc_info.value.code == "estimation.armstats.score_stats.n_clusters_clustered"


class TestVarYOverflowSafe:
    """``_ddof1_var`` computed ``c1 * c1 / n`` directly: a legitimate but
    huge reference offset overflowed that intermediate to ``inf`` before
    the subtraction, silently reporting ``var_y() == 0.0`` for a
    genuinely huge but representable variance."""

    def test_large_offset_no_longer_overflows_to_zero(self):
        a = ArmStats(
            study_id="s", metric="m", group_id="g", n=100, ref_y=1e308, cy1=1e155, cy2=1.1e308
        )
        assert a.var_y() == pytest.approx(1.0101010101010098e305, rel=1e-9)

    def test_neighbouring_float_input_stays_finite_and_close(self):
        """One ulp away from the pinned reproduction: still finite and of
        the same order of magnitude, not a discontinuous jump back to 0."""
        cy1 = math.nextafter(1e155, math.inf)
        a = ArmStats(
            study_id="s", metric="m", group_id="g", n=100, ref_y=1e308, cy1=cy1, cy2=1.1e308
        )
        assert math.isfinite(a.var_y())
        assert a.var_y() == pytest.approx(1.0101010101010098e305, rel=1e-6)

    def test_matches_direct_computation_at_ordinary_scale(self):
        """Ordinary scale: the overflow-safe scaled form is a no-op
        against numpy's own ddof=1 variance."""
        rng = np.random.default_rng(3)
        y = rng.normal(5.0, 2.0, 50)
        ref = float(np.mean(y))
        cy1 = float(np.sum(y - ref))
        cy2 = float(np.sum((y - ref) ** 2))
        a = ArmStats(study_id="s", metric="m", group_id="g", n=50, ref_y=ref, cy1=cy1, cy2=cy2)
        assert a.var_y() == pytest.approx(float(np.var(y, ddof=1)), rel=1e-9)

    def test_covariance_reductions_also_stay_finite_at_the_same_scale(self):
        """``cov_yx``/``cov_yden``/``cov_yd`` share the identical
        ``a * b / n`` correction term; pin one representative."""
        a = ArmStats(
            study_id="s",
            metric="m",
            group_id="g",
            n=100,
            ref_y=1e308,
            cy1=1e155,
            cy2=1.1e308,
            ref_x=1e308,
            cx1=1e155,
            cx2=1.1e308,
            cxy=1e155,
            x_role="covariate",
        )
        assert math.isfinite(a.cov_yx())


class TestScoreCenteringFeasibility:
    """``cy1`` (the y-family's residual first moment, "~0 but exact" per
    the class docstring) and its own second moment ``cy2`` must satisfy
    Cauchy-Schwarz (``cy1**2 <= n*cy2``) for ANY real sample sharing
    these ``n`` values, before any rounding at all: if ``ref_y`` is not
    actually this arm's own mean, that bound is violated outright,
    signalling ingress fed the wrong reference (mismatched partitions,
    corruption, or a bug upstream) rather than a real arm."""

    def test_grossly_inconsistent_reference_is_refused(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            ArmStats(study_id="s", metric="m", group_id="g", n=10, ref_y=0.0, cy1=100.0, cy2=1.0)
        assert exc_info.value.code == "estimation.armstats.cross_moment_violates"

    def test_covariate_family_reference_is_also_checked(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            ArmStats(
                study_id="s",
                metric="m",
                group_id="g",
                n=10,
                ref_y=0.0,
                cy1=0.0,
                cy2=1.0,
                ref_x=0.0,
                cx1=100.0,
                cx2=1.0,
                cxy=0.0,
                x_role="covariate",
            )
        assert exc_info.value.code == "estimation.armstats.cross_moment_violates"

    def test_boundary_equality_at_n_equals_one_is_accepted(self):
        """n=1: cy2 == cy1**2 exactly (a single unit collapses the two
        moments into the same number) -- equality, not a violation."""
        ArmStats(study_id="s", metric="m", group_id="g", n=1, ref_y=5.0, cy1=3.0, cy2=9.0)

    def test_zero_variance_family_does_not_force_a_bit_exact_reference(self):
        """A perfectly constant y (cy2 == 0.0 exactly) must not force
        cy1 to be bit-exact zero too -- a denormal-scale residual from
        an upstream reduction is still legitimate at that boundary."""
        ArmStats(study_id="s", metric="m", group_id="g", n=10, ref_y=5.0, cy1=-1e-300, cy2=0.0)

    def test_near_boundary_rounding_noise_is_accepted(self):
        """cy1 just inside the Cauchy-Schwarz boundary (not exactly 0,
        not exactly at sqrt(n*cy2)) must not be refused as corrupt."""
        n, cy2 = 10_000, 5.0
        cy1 = math.sqrt(n * cy2) * (1.0 - 1e-10)
        ArmStats(study_id="s", metric="m", group_id="g", n=n, ref_y=1.0, cy1=cy1, cy2=cy2)


class TestZeroVarianceCrossMoment:
    """With an exactly zero variance the Cauchy-Schwarz bound is zero, so the
    tolerance is absolute: a denormal-scale artifact is rounding, a materially
    nonzero cross moment is impossible for any real sample."""

    def test_materially_nonzero_cross_against_zero_variance_refuses(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            ArmStats(study_id="s", metric="m", group_id="g", n=10, ref_y=0.0, cy1=100.0, cy2=0.0)
        assert exc_info.value.code == "estimation.armstats.cross_moment_materially"

    def test_denormal_scale_residual_is_still_accepted(self):
        arm = ArmStats(study_id="s", metric="m", group_id="g", n=10, ref_y=0.0, cy1=1e-300, cy2=0.0)
        assert arm.var_y() == 0.0

    def test_exactly_zero_moments_are_accepted(self):
        arm = ArmStats(study_id="s", metric="m", group_id="g", n=10, ref_y=5.0, cy1=0.0, cy2=0.0)
        assert arm.var_y() == 0.0


class TestUptakeBoundaryIdentities:
    """The uptake-masked moments sum over taking-up units only, so the mask and
    the moments must agree at the boundary."""

    def test_no_takeup_with_a_nonzero_masked_moment_refuses(self):
        # Previously accepted, and var_yd() then reported a plausible nonzero
        # variance for an arm where nobody took up.
        with pytest.raises(InvalidRequestError) as exc_info:
            ArmStats(
                study_id="s",
                metric="m",
                group_id="t",
                n=10,
                ref_y=1.0,
                cy1=0.0,
                cy2=5.0,
                sum_d=0.0,
                cyd=0.0,
                cy2d=1.0,
            )
        assert exc_info.value.code == "estimation.armstats.arm_stats.sum_d_zero_but_masked_nonzero"

    def test_no_takeup_with_all_masked_moments_zero_is_accepted(self):
        arm = ArmStats(
            study_id="s",
            metric="m",
            group_id="t",
            n=10,
            ref_y=1.0,
            cy1=0.0,
            cy2=5.0,
            sum_d=0.0,
            cyd=0.0,
            cy2d=0.0,
        )
        assert arm.var_yd() == 0.0

    def test_a_fractional_uptake_total_is_still_accepted(self):
        """Analytic fixtures use n*p directly; that must keep working."""
        arm = ArmStats(
            study_id="s",
            metric="m",
            group_id="t",
            n=400,
            ref_y=10.0,
            cy1=0.0,
            cy2=100.0,
            sum_d=19.2,
            cyd=1.0,
            cy2d=2.0,
        )
        assert arm.sum_d == 19.2


def _gamma(terms: int) -> float:
    """Higham's ``k u / (1 - k u)``, the largest relative error of summing *terms* nonnegative
    values in any order, with ``u = 2**-53``."""
    product = terms * 2.0**-53
    return product / (1.0 - product)


class TestBinaryCountsBernoulliConsistency:
    """Exact counts and stored second moments must agree for declared binary data.
    A constant non-binary arm can have the same mean but the wrong variance.
    """

    def test_rejects_constant_arm_with_bernoulli_looking_mean(self):
        # Ten constant y=0.5 outcomes: mean 0.5 looks like "5 successes
        # out of 10", but cy2 (centered sum of squares) is exactly 0.0,
        # not the Bernoulli-consistent 2.5.
        arm = ArmStats.from_raw_sums(
            study_id="e",
            metric="conv",
            group_id="control",
            n=10,
            sum_y=5.0,
            sum_y2=2.5,
            successes=5,
        )
        with pytest.raises(BinomialDataError) as exc_info:
            binary_counts(arm, "conversion")
        assert exc_info.value.code == "estimation.binomial.inconsistent_bernoulli_variance"

    def test_accepts_genuine_bernoulli_arm(self):
        # Five ones, five zeros: sum_y2 == sum_y for genuine 0/1 data.
        arm = ArmStats.from_raw_sums(
            study_id="e",
            metric="conv",
            group_id="control",
            n=10,
            sum_y=5.0,
            sum_y2=5.0,
            successes=5,
        )
        assert binary_counts(arm, "conversion") == (5, 10)

    def test_accepts_all_zero_and_all_success_boundaries(self):
        zero_arm = ArmStats.from_raw_sums(
            study_id="e", metric="conv", group_id="control", n=8, sum_y=0.0, sum_y2=0.0, successes=0
        )
        assert binary_counts(zero_arm, "conversion") == (0, 8)
        full_arm = ArmStats.from_raw_sums(
            study_id="e", metric="conv", group_id="control", n=8, sum_y=8.0, sum_y2=8.0, successes=8
        )
        assert binary_counts(full_arm, "conversion") == (8, 8)

    def test_tolerance_scales_with_n_not_a_fixed_epsilon(self):
        # A genuine 0/1 reconstruction through floating summation over a
        # large n accumulates real (not corrupt) rounding; the
        # scale-derived tolerance (variance_slack) must still accept it.
        n = 200_000
        successes = 60_000
        y = np.zeros(n)
        y[:successes] = 1.0
        rng = np.random.default_rng(0)
        rng.shuffle(y)
        arm = ArmStats.from_raw_sums(
            study_id="e",
            metric="conv",
            group_id="control",
            n=n,
            successes=successes,
            sum_y=float(np.sum(y)),
            sum_y2=float(np.sum(y * y)),
        )
        assert binary_counts(arm, "conversion") == (successes, n)

    @pytest.mark.parametrize(
        ("n", "successes"),
        [
            (10_000_000, 20_000),
            (100_000_000, 200_000),
            (1_000_000_000, 2_000_000),
            (1_000_000_000, 500_000_000),
        ],
    )
    def test_a_summation_drift_within_the_rounding_bound_of_n_terms_is_accepted(self, n, successes):
        """A centered sum of squares adds ``n`` nonnegative terms, which any summation order
        evaluates within ``gamma_(n+8)`` of itself, ``k u / (1 - k u)`` with ``u = 2**-53``.
        Sequential accumulation of equal tiny residuals onto a growing total drifts that far at
        the largest arm sizes, so a genuine arm whose second moment errs by 90% of the bound is
        not corrupt."""
        expected = successes * (n - successes) / n
        drift = 0.9 * _gamma(n + 8) * expected
        arm = ArmStats(
            study_id="e",
            metric="conv",
            group_id="control",
            n=n,
            successes=successes,
            ref_y=successes / n,
            cy1=0.0,
            cy2=expected + drift,
        )
        assert binary_counts(arm, "conversion") == (successes, n)

    def test_a_drift_between_the_first_order_bound_and_the_compounded_one_is_accepted(self):
        """At a billion terms the compounding of the roundings, ``(k u)**2``, is 1.2e-14 of the
        sum, fourteen times the eight units of headroom a first-order bound ``k u`` leaves: a
        genuine second moment that has drifted the full worst case is not corrupt."""
        n, successes = 1_000_000_000, 2_000_000
        expected = successes * (n - successes) / n
        first_order = (n + 8) * 2.0**-53
        compounded = _gamma(n + 8)
        assert compounded > first_order
        arm = ArmStats(
            study_id="e",
            metric="conv",
            group_id="control",
            n=n,
            successes=successes,
            ref_y=successes / n,
            cy1=0.0,
            cy2=expected * (1.0 + 0.5 * (first_order + compounded)),
        )
        assert binary_counts(arm, "conversion") == (successes, n)

    @pytest.mark.parametrize("n", [10_000_000, 100_000_000, 1_000_000_000])
    def test_a_second_moment_beyond_the_rounding_bound_is_refused(self, n):
        successes = n // 500
        expected = successes * (n - successes) / n
        arm = ArmStats(
            study_id="e",
            metric="conv",
            group_id="control",
            n=n,
            successes=successes,
            ref_y=successes / n,
            cy1=0.0,
            cy2=expected * (1.0 + 4.0 * (n + 8) * 2.0**-53),
        )
        with pytest.raises(BinomialDataError) as exc_info:
            binary_counts(arm, "conversion")
        assert exc_info.value.code == "estimation.binomial.inconsistent_bernoulli_variance"

    @staticmethod
    def _stored_arm(n: int, successes: int, *, residual_shift: float = 0.0) -> ArmStats:
        """The arm as a producer that centers on the correctly rounded rate stores it: that rate
        as the reference, the exact residual of the integer sum from it, and the exact centered
        sum of squares, with ``residual_shift`` of aggregation error on the residual."""
        ref_y = successes / n
        return ArmStats(
            study_id="e",
            metric="conv",
            group_id="control",
            n=n,
            successes=successes,
            ref_y=ref_y,
            cy1=float(Fraction(successes) - n * Fraction(ref_y)) + residual_shift,
            cy2=float(Fraction(successes * (n - successes), n)),
        )

    @pytest.mark.parametrize("n", [2**54 + 2, 2**58, 2**61, 10**18])
    @pytest.mark.parametrize("rare", [513, 514, 641, 777, 1027, 2561, 3001])
    @pytest.mark.parametrize("rare_side", ["failures", "successes"])
    def test_integer_counts_survive_beyond_float_spacing(self, n, rare, rare_side):
        """A count such as 513 failures must not become 512 through a float total."""
        successes = rare if rare_side == "successes" else n - rare
        arm = self._stored_arm(n, successes)
        assert binary_counts(arm, "conversion") == (successes, n)

    @pytest.mark.parametrize("shift", [5e-7, -5e-7])
    def test_the_aggregation_tolerance_on_the_first_moment_holds_at_any_scale(self, shift):
        n = 2**61
        arm = self._stored_arm(n, n - 513, residual_shift=shift)
        assert binary_counts(arm, "conversion") == (n - 513, n)

    @pytest.mark.parametrize("shift", [1e-3, -1e-3, 0.4, -0.4])
    def test_corrupt_first_moment_is_refused_when_rounding_cannot_explain_it(self, shift):
        with pytest.raises(BinomialDataError) as exc_info:
            binary_counts(self._stored_arm(1000, 300, residual_shift=shift), "conversion")
        assert exc_info.value.code == "estimation.binomial.reconstructed_counts_not_binary"

    @pytest.mark.parametrize(
        ("n", "successes"),
        [(10**11, 3 * 10**10), (10 * 2**54, 3 * 2**54), (3 * 2**60, 2**60)],
    )
    def test_counts_are_independent_of_rounded_producer_residuals(self, n, successes):
        ref = successes / n
        residual = float(successes * Fraction(1.0 - ref) + (n - successes) * Fraction(-ref))
        arm = self._stored_arm(n, successes).model_copy(update={"cy1": residual})
        assert binary_counts(arm, "conversion") == (successes, n)

    def test_missing_exact_counts_refuse_instead_of_guessing_from_moments(self):
        arm = self._stored_arm(3 * 2**60, 2**60).model_copy(update={"successes": None})
        with pytest.raises(CapabilityError) as exc_info:
            binary_counts(arm, "conversion")
        assert exc_info.value.code == "estimation.binomial.exact_counts_required"

    @pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1), (1, 2, 0)])
    def test_partition_merges_preserve_rare_failures_beyond_float_spacing(self, order):
        n = 2**60
        parts = [self._stored_arm(n, n - failures) for failures in (513, 777, 1500)]
        merged = ArmStats.combine([parts[i] for i in order])
        total, failures = 3 * n, 2790
        assert binary_counts(merged, "conversion") == (total - failures, total)
        expected = float(Fraction((total - failures) * failures, total * (total - 1)))
        assert merged.var_y() == pytest.approx(expected, rel=1e-12, abs=0.0)

    def test_missing_partition_count_cannot_be_reconstructed_after_merge(self):
        complete = self._stored_arm(2**60, 2**60 - 513)
        missing = self._stored_arm(2**60, 2**60 - 777).model_copy(update={"successes": None})
        with pytest.raises(CapabilityError) as exc_info:
            binary_counts(ArmStats.combine([complete, missing]), "conversion")
        assert exc_info.value.code == "estimation.binomial.exact_counts_required"


class TestCanonicalBernoulliArm:
    """An arm `binary_counts` accepts, re-formed from its counts: the y family is the exact
    one of a 0/1 arm, and the rest of the record is the arm's own."""

    N, SUCCESSES = 200_000, 60_000

    def _drifted(self) -> ArmStats:
        """The arm with the largest second-moment drift `binary_counts` still admits."""
        expected = self.SUCCESSES * (self.N - self.SUCCESSES) / self.N
        admitted = None
        for step in (2.0**-k for k in range(60, 20, -1)):
            arm = ArmStats(
                study_id="e",
                metric="conv",
                group_id="control",
                n=self.N,
                successes=self.SUCCESSES,
                ref_y=self.SUCCESSES / self.N,
                cy1=4e-7,
                cy2=expected * (1.0 + step),
            )
            try:
                binary_counts(arm, "conversion")
            except BinomialDataError:
                break
            admitted = arm
        assert admitted is not None and admitted.cy2 != expected
        return admitted

    def test_the_y_family_is_the_exact_one_of_a_0_1_arm(self):
        arm = canonical_bernoulli_arm(self._drifted(), self.SUCCESSES)
        summary = arm.to_summary()
        n, x = self.N, self.SUCCESSES
        assert summary.mean == pytest.approx(x / n, rel=1e-15, abs=0.0)
        assert summary.var == pytest.approx(x * (n - x) / (n * (n - 1)), rel=1e-15, abs=0.0)

    def test_the_counts_do_not_move(self):
        drifted = self._drifted()
        assert binary_counts(drifted, "conversion") == (self.SUCCESSES, self.N)
        canonical = canonical_bernoulli_arm(drifted, self.SUCCESSES)
        assert binary_counts(canonical, "conversion") == (self.SUCCESSES, self.N)
        assert canonical_bernoulli_arm(canonical, self.SUCCESSES) == canonical

    def test_it_is_the_arm_the_counts_alone_give(self):
        exact = ArmStats.from_raw_sums(
            study_id="e",
            metric="conv",
            group_id="control",
            n=self.N,
            successes=self.SUCCESSES,
            sum_y=float(self.SUCCESSES),
            sum_y2=float(self.SUCCESSES),
        )
        assert canonical_bernoulli_arm(self._drifted(), self.SUCCESSES) == exact

    def test_every_other_field_of_the_record_is_kept(self):
        arm = ArmStats(
            study_id="exp",
            metric="conv",
            group_id="treatment",
            n=1_000,
            successes=300,
            ref_y=0.3,
            cy1=0.0,
            cy2=210.0 * (1.0 + 1e-12),
            ref_x=0.5,
            cx1=0.0,
            cx2=250.0,
            cxy=3.0,
            x_role="covariate",
            sum_d=400.0,
            cyd=1.5,
            cy2d=80.0,
            winsor_n=1_000,
            winsor_n_lower=0,
            winsor_n_upper=0,
        )
        kept = canonical_bernoulli_arm(arm, 300).model_dump()
        original = arm.model_dump()
        for field in ("ref_y", "cy1", "cy2"):
            kept.pop(field), original.pop(field)
        assert kept == original

    @pytest.mark.parametrize(
        ("n", "failures"),
        [(2**61, 2560), (2**61, 512), (2**58, 416), (2**55, 60)],
    )
    def test_an_arm_with_many_units_and_few_failures_keeps_their_variance(self, n, failures):
        """The failures' centered sum of squares is ``failures`` to within ``failures / n``: no
        rounding of raw sums is involved, so an arm of this many units whose variance a raw-sum
        centering would clamp to zero (its noise floor is ``8 * eps * n``, above ``failures``)
        keeps it, and the standard error of its log mean stays the closed form
        ``sqrt(failures / (successes * (n - 1)))``."""
        template = ArmStats(
            study_id="e", metric="conv", group_id="control", n=n, ref_y=0.0, cy1=0.0, cy2=0.0
        )
        successes = n - failures
        arm = canonical_bernoulli_arm(template, successes)
        centered = Fraction(successes * failures, n)
        assert arm.cy2 == pytest.approx(float(centered), rel=1e-15, abs=0.0)
        assert arm.var_y() == pytest.approx(float(centered / (n - 1)), rel=1e-12, abs=0.0)
        assert arm.mean_y() == pytest.approx(successes / n, rel=2.0**-52, abs=0.0)
        summary = arm.to_summary()
        assert se_log_mean(summary.var, summary.mean, summary.n) == pytest.approx(
            math.sqrt(float(Fraction(failures, successes * (n - 1)))), rel=1e-12, abs=0.0
        )
        assert binary_counts(arm, "conversion") == (successes, n)

    @pytest.mark.parametrize("successes", [-1, 1_001])
    def test_counts_that_are_not_a_count_of_the_arm_are_refused(self, successes):
        template = ArmStats(
            study_id="e", metric="conv", group_id="control", n=1_000, ref_y=0.0, cy1=0.0, cy2=0.0
        )
        with pytest.raises(BinomialDataError) as exc_info:
            canonical_bernoulli_arm(template, successes)
        assert exc_info.value.code == "estimation.binomial.reconstructed_counts_not_binary"


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestBinaryCountsProducerPathTolerance:
    """`binary_counts` against the REAL two-phase producer path --
    `increment.query.builders.group_summary`'s DuckDB
    window(``AVG``)-then-``SUM((y-ref_y)**2)`` aggregation, the exact SQL
    shape that produces a conversion arm's ``ref_y``/``cy1``/``cy2`` in
    production -- not a hand-reconstructed moment (``ArmStats.
    from_raw_sums`` uses near-exact ``Fraction`` arithmetic and so never
    exercises this two-phase floating error).

    ``binary_counts`` must accept every genuinely Bernoulli cell this real path produces and
    must still refuse a genuinely corrupted cell at the same scale. The cells below are the
    measured sizes up to 16M units; 64M, 100M and 1B units, where one DuckDB aggregation takes
    minutes, are measured by ``scripts/measure_binomial_ceiling.py recovery``.
    """

    @pytest.mark.parametrize(
        ("n", "success_frac"),
        [
            (200_000, 0.3),
            (1_000_000, 0.3),
            (1_000_000, 0.0001),
            (2_000_000, 0.001),
            (4_000_000, 0.002),
            (4_000_000, 0.5),
            # The measured worst cells at 16M and 100M units sat near 335x and 275x the base
            # `variance_slack`: the cells that set `_BERNOULLI_CONSISTENCY_SLACK`.
            (16_000_000, 0.002),
            (16_000_000, 0.0001),
            (16_000_000, 0.5),
        ],
    )
    def test_producer_path_tolerance_grid(self, n, success_frac):
        successes = max(1, round(n * success_frac))
        assert binary_counts(producer_arm(n, successes), "conversion") == (successes, n)

    @pytest.mark.parametrize("n", [4_000_000, 16_000_000])
    def test_producer_path_corrupted_input_still_refuses_at_scale(self, n):
        """A fractional outcome has no exact binary count, even when its mean is 50%."""
        with pytest.raises(CapabilityError) as exc_info:
            binary_counts(producer_arm(n, None), "conversion")
        assert exc_info.value.code == "estimation.binomial.exact_counts_required"


@pytest.mark.slow
def test_a_unit_frame_and_its_exported_moments_decide_alike_beyond_the_former_arm_ceiling():
    """Frame and exported-moments decisions agree with one arm above four million units.
    The parity harness's ``exact_binomial_beyond_the_former_arm_ceiling`` case also
    checks definitions, unit-day artifacts and unit panels at this scale."""
    import tempfile
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq

    from increment import AnalysisPlan, MetricSpec
    from increment.analysis import Analysis
    from tests.analysis_factory import lift_rows

    n_c, n_t, x_c, x_t = 4_000_100, 4_000, 4_000, 80
    units = np.arange(n_c + n_t, dtype=np.int64)
    converted = np.zeros(units.size, np.int8)
    converted[:x_c] = 1
    converted[n_c : n_c + x_t] = 1
    frame = pa.table(
        {
            "user_id": units,
            "group_id": pa.array(np.where(units < n_c, "control", "treatment")),
            "conversion": converted,
        }
    )
    metric = MetricSpec(name="conversion", type="conversion")
    plan = AnalysisPlan(secondaries=["conversion"])
    with Analysis.from_unit_summary(
        frame, unit="user_id", group="group_id", control="control", metrics=[metric], plan=plan
    ) as summary:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "moments.parquet"
            summary.export(path)
            rows = pq.read_table(path).to_pylist()
        from_frame = list(lift_rows(summary.run()))
    with Analysis.from_moments(rows, control="control", metrics=[metric]) as replay:
        from_cube = list(lift_rows(replay.run()))
    for results in (from_frame, from_cube):
        (row,) = results
        assert row.reference_kind == "binomial" and row.binomial_set is not None
        counts = (row.binomial_set.x_c, row.binomial_set.n_c, row.binomial_set.x_t)
        assert counts == (x_c, n_c, x_t) and row.binomial_set.n_t == n_t
    assert from_frame[0].binomial_set == from_cube[0].binomial_set
