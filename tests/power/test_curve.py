from __future__ import annotations

import math
from itertools import product
from typing import Any, Literal

import pytest

from increment import power_curve
from increment.errors import InvalidRequestError
from increment.estimation.sequential import GaussianScoreMixture
from increment.power import Baseline, PowerDesign, achieved_power, minimum_detectable_effect
from increment.power._validation import _require_finite, _require_relative_domain
from increment.results import PowerCurve, PowerCurvePoint
from tests.power._procedures import make_procedure

EXPECTED_COLUMNS = [
    "solve_for",
    "n_per_arm",
    "n_total",
    "relative_lift",
    "target_power",
    "alpha",
    "power",
    "power_basis",
    "numerical_qualification",
    "mde_relative",
    "mde_unavailable_reason",
    "effective_var",
    "n_clusters_per_arm",
    "n_clusters_total",
    "n_triggered_per_arm",
    "n_triggered_total",
    "duration_days",
    "expected_n_total",
    "expected_duration_days",
]

BASELINE = Baseline.from_proportion(0.10)
PROCEDURE = make_procedure()


def _assert_power_result_fields(point: PowerCurvePoint, result: Any) -> None:
    assert point.n_per_arm == result.n_per_arm
    assert point.n_total == result.n_total
    assert point.power == result.power
    assert point.power_basis == result.power_basis
    assert point.mde_relative == result.mde_relative
    assert point.mde_unavailable_reason == result.mde_unavailable_reason
    assert point.effective_var == result.effective_var
    assert point.n_clusters_per_arm == result.n_clusters_per_arm
    assert point.n_clusters_total == result.n_clusters_total
    assert point.n_triggered_per_arm == result.n_triggered_per_arm
    assert point.n_triggered_total == result.n_triggered_total
    assert point.expected_n_total == result.expected_n_total


class TestFixedPowerCurve:
    def test_power_mode_matches_repeated_achieved_power(self):
        sample_sizes = [500, 1_000]
        relative_lifts = [0.02, 0.05]
        procedures = [make_procedure(alpha=a) for a in [0.01, 0.05]]

        curve = power_curve(
            n_per_arm=sample_sizes,
            relative_lift=relative_lifts,
            procedure=procedures,
            baseline=BASELINE,
        )

        expected_dimensions = list(product(sample_sizes, relative_lifts, procedures))
        assert [(p.n_per_arm, p.relative_lift, p.alpha) for p in curve] == [
            (n, lift, proc.compiled_alpha) for n, lift, proc in expected_dimensions
        ]
        for point, (n, lift, proc) in zip(curve, expected_dimensions, strict=True):
            direct = achieved_power(
                n_per_arm=n,
                relative_lift=lift,
                baseline=BASELINE,
                procedure=proc,
            )
            assert point.solve_for == "power"
            assert point.target_power == 0.8
            _assert_power_result_fields(point, direct)

    def test_mde_mode_matches_repeated_minimum_detectable_effect(self):
        sample_sizes = [500, 1_000]
        target_powers = [0.8, 0.9]
        procedures = [make_procedure(alpha=a) for a in [0.01, 0.05]]

        curve = power_curve(
            n_per_arm=sample_sizes,
            target_power=target_powers,
            procedure=procedures,
            baseline=BASELINE,
        )

        expected_dimensions = list(product(sample_sizes, target_powers, procedures))
        assert [(p.n_per_arm, p.target_power, p.alpha) for p in curve] == [
            (n, target, proc.compiled_alpha) for n, target, proc in expected_dimensions
        ]
        for point, (n, target, proc) in zip(curve, expected_dimensions, strict=True):
            direct = minimum_detectable_effect(
                n_per_arm=n,
                baseline=BASELINE,
                procedure=proc,
                design=PowerDesign(power=target),
            )
            assert point.solve_for == "mde"
            assert point.relative_lift is None
            _assert_power_result_fields(point, direct)

    @pytest.mark.parametrize("solve_for", ["power", "mde"])
    @pytest.mark.parametrize("reverse", [False, True])
    def test_mixed_decisions_keep_independent_cuped_credit(self, solve_for, reverse):
        from increment.estimation.arm_contract import ArmPlanningProcedure
        from increment.semantics.models import MethodSpec

        plain = ArmPlanningProcedure.standard("mean")
        cuped = MethodSpec(name="cuped", variance_reduction="cuped")
        sensitivity = plain.model_copy(update={"sensitivity_methods": (cuped,)})
        adjusted = plain.model_copy(update={"decision_method": cuped})
        procedures = (adjusted, sensitivity) if reverse else (sensitivity, adjusted)
        baseline = Baseline(mean=1.0, var=1.0, cuped_rho=0.9)
        curve = power_curve(
            n_per_arm=1_000,
            baseline=baseline,
            procedure=procedures,
            relative_lift=0.1 if solve_for == "power" else None,
            target_power=0.8 if solve_for == "mde" else None,
        )
        for point, proc in zip(curve, procedures, strict=True):
            expected = (
                achieved_power(1_000, 0.1, baseline, proc)
                if solve_for == "power"
                else minimum_detectable_effect(1_000, baseline, proc)
            )
            assert expected.power is not None
            assert expected.mde_relative is not None
            assert point.power == pytest.approx(expected.power)
            assert point.mde_relative == pytest.approx(expected.mde_relative)
            assert point.effective_var == pytest.approx(expected.effective_var)

    def test_power_mode_keeps_zero_companion_at_low_target(self):
        curve = power_curve(
            n_per_arm=100,
            relative_lift=0.1,
            procedure=make_procedure(),
            baseline=Baseline(mean=1.0, var=2.0),
            design=PowerDesign(power=0.01),
        )

        assert len(curve) == 1
        assert curve[0].power > 0.01
        assert curve[0].mde_relative == 0.0
        assert curve[0].mde_unavailable_reason is None

    def test_scalar_inputs_and_duplicate_order_are_preserved(self):
        curve = power_curve(
            n_per_arm=[1_000, 1_000],
            relative_lift=0.05,
            procedure=PROCEDURE,
            baseline=BASELINE,
        )

        assert [(p.n_per_arm, p.relative_lift, p.alpha) for p in curve] == [
            (1_000, 0.05, 0.05),
            (1_000, 0.05, 0.05),
        ]

    def test_procedure_and_target_power_do_not_mutate_design(self):
        design = PowerDesign(power=0.8, allocation=0.25)
        procedures = [make_procedure(alpha=a) for a in [0.01, 0.10]]

        curve = power_curve(
            n_per_arm=1_000,
            target_power=[0.85, 0.9],
            procedure=procedures,
            baseline=BASELINE,
            design=design,
        )

        assert design == PowerDesign(power=0.8, allocation=0.25)
        assert {(p.alpha, p.target_power) for p in curve} == {
            (0.01, 0.85),
            (0.10, 0.85),
            (0.01, 0.9),
            (0.10, 0.9),
        }

    def test_family_procedure_uses_tiny_tail_without_cancellation(self):
        procedure = make_procedure(alpha=math.ldexp(1.0, -1022), family_size=3)

        curve = power_curve(
            n_per_arm=5_000,
            relative_lift=0.05,
            baseline=BASELINE,
            procedure=procedure,
        )

        assert curve[0].alpha == procedure.compiled_alpha
        assert math.isfinite(curve[0].power)

    def test_duration_uses_total_units_without_changing_statistics(self):
        kwargs: dict[str, Any] = {
            "n_per_arm": 1_000,
            "relative_lift": 0.05,
            "baseline": BASELINE,
            "procedure": PROCEDURE,
            "design": PowerDesign(allocation=0.25),
        }

        plain = power_curve(**kwargs)[0]
        timed = power_curve(**kwargs, units_per_week=10_000)[0]

        assert timed.duration_days == math.ceil(7 * timed.n_total / 10_000)
        assert plain.duration_days is None
        assert timed.model_dump(exclude={"duration_days"}) == plain.model_dump(
            exclude={"duration_days"}
        )

    def test_expected_duration_uses_expected_n_total(self):
        procedure = make_procedure(inference=GaussianScoreMixture(), population="assigned")
        kwargs: dict[str, Any] = {
            "n_per_arm": [1_000, 2_000],
            "relative_lift": 0.05,
            "baseline": BASELINE,
            "procedure": procedure,
            "planned_looks": 2,
            "max_workers": 1,
        }
        plain = power_curve(**kwargs)[0]
        timed = power_curve(**kwargs, units_per_week=10_000)[0]

        assert plain.expected_duration_days is None
        assert timed.expected_n_total is not None
        assert timed.expected_duration_days == math.ceil(7 * timed.expected_n_total / 10_000)
        assert timed.duration_days is not None
        assert timed.expected_duration_days <= timed.duration_days


class TestPowerCurveTriggeredCounts:
    """``PowerResult.n_triggered_per_arm``/``n_triggered_total`` propagate
    through ``PowerCurvePoint``, its model dump, and every frame backend;
    a curve with no declared trigger rate carries a numeric null instead."""

    def test_declared_trigger_rate_carries_triggered_counts(self):
        baseline = Baseline(mean=10.0, var=25.0, trigger_rate=0.2)
        pc = power_curve(
            n_per_arm=[500],
            relative_lift=[0.1],
            baseline=baseline,
            procedure=make_procedure(alternative="two-sided"),
            design=PowerDesign(power=0.8),
        )
        point = pc[0]
        assert point.n_triggered_per_arm == 100
        assert point.n_triggered_total == 200
        dumped = point.model_dump()
        assert dumped["n_triggered_per_arm"] == 100
        assert dumped["n_triggered_total"] == 200

    @pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
    def test_declared_trigger_rate_carries_through_frame(
        self, backend: Literal["pandas", "polars", "pyarrow"]
    ):
        import narwhals as nw

        baseline = Baseline(mean=10.0, var=25.0, trigger_rate=0.2)
        pc = power_curve(
            n_per_arm=[500],
            relative_lift=[0.1],
            baseline=baseline,
            procedure=make_procedure(alternative="two-sided"),
            design=PowerDesign(power=0.8),
        )
        frame = nw.from_native(pc.to_frame(backend=backend), eager_only=True)
        assert frame["n_triggered_per_arm"].to_list() == [100]
        assert frame["n_triggered_total"].to_list() == [200]

    def test_undeclared_trigger_rate_is_a_numeric_null(self):
        pc = power_curve(
            n_per_arm=[500],
            relative_lift=[0.1],
            baseline=BASELINE,
            procedure=PROCEDURE,
        )
        point = pc[0]
        assert point.n_triggered_per_arm is None
        assert point.n_triggered_total is None
        dumped = point.model_dump()
        assert dumped["n_triggered_per_arm"] is None
        assert dumped["n_triggered_total"] is None

    @pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
    def test_undeclared_trigger_rate_stays_numeric_null_in_frame(
        self, backend: Literal["pandas", "polars", "pyarrow"]
    ):
        import narwhals as nw

        pc = power_curve(
            n_per_arm=[500],
            relative_lift=[0.1],
            baseline=BASELINE,
            procedure=PROCEDURE,
        )
        frame = nw.from_native(pc.to_frame(backend=backend), eager_only=True)
        assert frame.schema["n_triggered_per_arm"] == nw.Float64
        assert frame.schema["n_triggered_total"] == nw.Float64
        assert frame["n_triggered_per_arm"].is_null().to_list() == [True]
        assert frame["n_triggered_total"].is_null().to_list() == [True]


class TestPowerCurveValidation:
    @pytest.mark.parametrize(
        ("kwargs", "code", "name"),
        [
            ({}, "power.exactly_one_relative", None),
            ({"relative_lift": 0.05, "target_power": 0.8}, "power.exactly_one_relative", None),
            ({"relative_lift": []}, "power.empty", "relative_lift"),
            ({"relative_lift": 0.05, "n_per_arm": []}, "power.empty", "n_per_arm"),
            ({"relative_lift": 0.05, "procedure": []}, "power.empty", "procedure"),
        ],
    )
    def test_grid_shape_validation(self, kwargs: dict[str, Any], code: str, name: str | None):
        call_kwargs: dict[str, Any] = {
            "n_per_arm": 1_000,
            "baseline": BASELINE,
            "procedure": PROCEDURE,
        }
        call_kwargs.update(kwargs)
        with pytest.raises(InvalidRequestError) as raised:
            power_curve(**call_kwargs)
        assert raised.value.code == code
        if name is not None:
            assert raised.value.context["name"] == name

    @pytest.mark.parametrize(
        ("planned_looks", "code"),
        [
            pytest.param(True, "power.planned_looks_positive", id="bool"),
            pytest.param(2.5, "power.planned_looks_positive", id="float"),
            pytest.param(0, "power.planned_looks", id="zero"),
            pytest.param(-1, "power.planned_looks", id="negative"),
        ],
    )
    def test_invalid_planned_looks_precedes_grid_materialization(
        self, planned_looks: Any, code: str
    ):
        with pytest.raises(InvalidRequestError) as raised:
            power_curve(
                n_per_arm=[],
                relative_lift=0.05,
                baseline=BASELINE,
                procedure=PROCEDURE,
                planned_looks=planned_looks,
            )
        assert raised.value.code == code

    @pytest.mark.parametrize(
        ("n_per_arm", "code"),
        [
            pytest.param(True, "power.curve.n_per_arm_int", id="bool"),
            pytest.param(1, "power.curve.n_per_arm_min", id="too-small"),
            pytest.param(2.5, "power.curve.n_per_arm_int", id="float"),
        ],
    )
    def test_invalid_sample_size(self, n_per_arm: Any, code: str):
        with pytest.raises(InvalidRequestError) as raised:
            power_curve(
                n_per_arm=n_per_arm,
                relative_lift=0.05,
                baseline=BASELINE,
                procedure=PROCEDURE,
            )
        assert raised.value.code == code

    @pytest.mark.parametrize(
        ("units_per_week", "code"),
        [
            pytest.param(True, "power.units_per_week_finite_positive", id="bool"),
            pytest.param(0.0, "power.units_per_week_finite_positive", id="zero"),
            pytest.param(-1.0, "power.units_per_week_finite_positive", id="negative"),
            pytest.param(float("inf"), "power.units_per_week_finite_positive", id="inf"),
            pytest.param(float("nan"), "power.units_per_week_finite_positive", id="nan"),
        ],
    )
    def test_invalid_units_per_week(self, units_per_week: Any, code: str):
        with pytest.raises(InvalidRequestError) as raised:
            power_curve(
                n_per_arm=1_000,
                relative_lift=0.05,
                baseline=BASELINE,
                procedure=PROCEDURE,
                units_per_week=units_per_week,
            )
        assert raised.value.code == code

    def test_units_per_week_retired_code_maps_to_canonical(self):
        from increment.errors import RETIRED_CODES

        assert RETIRED_CODES["power.units_per_week"] == "power.units_per_week_finite_positive"

    @pytest.mark.parametrize(
        ("max_workers", "code"),
        [
            pytest.param(True, "power.max_workers_positive", id="bool"),
            pytest.param(2.5, "power.max_workers_positive", id="float"),
            pytest.param(0, "power.max_workers", id="zero"),
        ],
    )
    def test_invalid_max_workers(self, max_workers: Any, code: str):
        with pytest.raises(InvalidRequestError) as raised:
            power_curve(
                n_per_arm=1_000,
                relative_lift=0.05,
                baseline=BASELINE,
                procedure=PROCEDURE,
                max_workers=max_workers,
            )
        assert raised.value.code == code

    def test_string_sample_size_is_rejected_as_a_scalar(self):
        with pytest.raises(InvalidRequestError) as raised:
            power_curve(
                n_per_arm="1000",  # ty: ignore[invalid-argument-type]
                relative_lift=0.05,
                baseline=BASELINE,
                procedure=PROCEDURE,
            )
        assert raised.value.code == "power.scalar_numeric_sequence"
        assert raised.value.context["name"] == "n_per_arm"

    @pytest.mark.parametrize("value", [0.0, 1.0])
    def test_target_power_uses_coded_field_validation(self, value: float):
        with pytest.raises(InvalidRequestError) as raised:
            power_curve(
                n_per_arm=1_000,
                baseline=BASELINE,
                procedure=PROCEDURE,
                target_power=value,
            )
        assert raised.value.code == "model.field.range"


class TestSharedValidationHelpers:
    """``increment.power._validation``'s finiteness and relative-domain
    guards; shared by every power module but not reachable through a
    public entry point of their own."""

    @pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
    def test_require_finite_rejects_non_finite_values(self, value: float):
        with pytest.raises(InvalidRequestError) as raised:
            _require_finite("mean", value)
        assert raised.value.code == "power.finite"
        assert raised.value.context["name"] == "mean"

    @pytest.mark.parametrize("value", [-1.0, -2.5])
    def test_require_relative_domain_rejects_values_at_or_below_the_floor(self, value: float):
        with pytest.raises(InvalidRequestError) as raised:
            _require_relative_domain("relative_lift", value)
        assert raised.value.code == "power.require_relative_domain"
        assert raised.value.context["value"] == value


@pytest.mark.slow
@pytest.mark.parametrize("solve_for", ["power", "mde"])
def test_sequential_rows_match_direct_scalar_calls(solve_for: str):
    procedure = make_procedure(inference=GaussianScoreMixture(), population="assigned")
    kwargs: dict[str, Any]
    if solve_for == "power":
        kwargs = {"relative_lift": 0.05}
    else:
        kwargs = {"target_power": 0.85}

    curve = power_curve(
        n_per_arm=[1_000, 2_000],
        baseline=BASELINE,
        procedure=procedure,
        planned_looks=2,
        max_workers=1,
        **kwargs,
    )

    for point in curve:
        if solve_for == "power":
            assert point.relative_lift is not None
            direct = achieved_power(
                n_per_arm=point.n_per_arm,
                relative_lift=point.relative_lift,
                baseline=BASELINE,
                procedure=procedure,
                planned_looks=2,
            )
        else:
            direct = minimum_detectable_effect(
                n_per_arm=point.n_per_arm,
                baseline=BASELINE,
                procedure=procedure,
                design=PowerDesign(power=point.target_power),
                planned_looks=2,
            )
        _assert_power_result_fields(point, direct)


@pytest.mark.slow
def test_parallel_results_exactly_match_serial_order():
    procedure = make_procedure(inference=GaussianScoreMixture(), population="assigned")
    kwargs: dict[str, Any] = {
        "n_per_arm": [1_000, 2_000],
        "relative_lift": [0.03, 0.05],
        "baseline": BASELINE,
        "procedure": procedure,
        "planned_looks": 2,
    }

    serial = power_curve(**kwargs, max_workers=1)
    parallel = power_curve(**kwargs, max_workers=4)

    assert parallel == serial
    assert parallel.to_dicts() == serial.to_dicts()


def _point(*, n_per_arm: int = 1_000) -> PowerCurvePoint:
    return PowerCurvePoint(
        solve_for="power",
        n_per_arm=n_per_arm,
        n_total=2 * n_per_arm,
        relative_lift=0.05,
        target_power=0.8,
        alpha=0.05,
        power=0.7,
        power_basis="exact",
        mde_relative=0.06,
        effective_var=0.09,
    )


class TestPowerCurveCollection:
    def test_to_dicts_returns_fresh_plain_rows(self):
        curve = PowerCurve([_point()])

        rows = curve.to_dicts()
        rows[0]["power"] = 0.0

        assert rows[0]["n_per_arm"] == 1_000
        assert curve[0].power == 0.7

    def test_slice_returns_independent_power_curve(self):
        curve = PowerCurve([_point(n_per_arm=1_000), _point(n_per_arm=2_000)])

        sliced = curve[:1]
        sliced.append(_point(n_per_arm=3_000))

        assert isinstance(sliced, PowerCurve)
        assert [row.n_per_arm for row in sliced] == [1_000, 3_000]
        assert [row.n_per_arm for row in curve] == [1_000, 2_000]
        assert callable(sliced.to_dicts)
        assert callable(sliced.to_frame)

    def test_add_and_in_place_add_preserve_power_curve(self):
        left = PowerCurve([_point(n_per_arm=1_000)])
        right = PowerCurve([_point(n_per_arm=2_000)])

        combined = left + right
        left += right

        assert isinstance(combined, PowerCurve)
        assert isinstance(left, PowerCurve)
        assert [row.n_per_arm for row in combined] == [1_000, 2_000]
        assert [row.n_per_arm for row in left] == [1_000, 2_000]

    def test_approximate_diagnostic_has_no_runtime_power_claim(self):
        point = _point().model_copy(update={"power_basis": "approximate"})

        assert point.numerical_qualification == "unclaimed_approximation_diagnostic_v1"
        assert PowerCurve([point]).to_dicts()[0]["numerical_qualification"] == (
            "unclaimed_approximation_diagnostic_v1"
        )

    @pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
    def test_to_frame_has_stable_columns(self, backend: Literal["pandas", "polars", "pyarrow"]):
        frame: Any = PowerCurve([_point()]).to_frame(backend=backend)

        if backend == "pyarrow":
            assert frame.column_names == EXPECTED_COLUMNS
        else:
            assert list(frame.columns) == EXPECTED_COLUMNS

    @pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
    def test_empty_slice_to_frame_preserves_schema(
        self, backend: Literal["pandas", "polars", "pyarrow"]
    ):
        empty = PowerCurve([_point()])[:0]

        frame: Any = empty.to_frame(backend=backend)

        assert isinstance(empty, PowerCurve)
        if backend == "pyarrow":
            assert frame.column_names == EXPECTED_COLUMNS
            assert frame.num_rows == 0
        else:
            assert list(frame.columns) == EXPECTED_COLUMNS
            assert len(frame) == 0


class TestUnavailableCompanionMde:
    """A supplied-effect row whose target has no minimum detectable effect
    keeps its power and carries a numeric-null companion with its reason
    through the curve, JSON, and every frame backend."""

    # n=50 in the decreasing direction: the H1 noncentrality peaks below
    # the 0.8 target, so the companion MDE does not exist there.
    _BASELINE = Baseline(mean=1.0, var=1.0)
    _PROCEDURE = make_procedure(
        alternative="less",
        identification="randomized",
        population="assigned",
        variance_adjustment="none",
    )

    def _curve(self) -> PowerCurve:
        return power_curve(
            n_per_arm=[50, 400],
            relative_lift=-0.5,
            procedure=self._PROCEDURE,
            baseline=self._BASELINE,
        )

    def test_rows_match_direct_calls_and_pair_null_with_reason(self):
        unavailable, available = self._curve()
        for point, n in ((unavailable, 50), (available, 400)):
            direct = achieved_power(n, -0.5, self._BASELINE, self._PROCEDURE)
            _assert_power_result_fields(point, direct)
        assert unavailable.mde_relative is None
        assert unavailable.mde_unavailable_reason == "unattainable"
        assert 0.0 < unavailable.power < 1.0
        assert available.mde_relative is not None
        assert available.mde_unavailable_reason is None

    def test_json_round_trip_preserves_null_and_reason(self):
        curve = self._curve()
        for point in curve:
            restored = PowerCurvePoint.model_validate_json(point.model_dump_json())
            assert restored == point
            assert restored.numerical_qualification == point.numerical_qualification
        dumped = curve.to_dicts()
        assert dumped[0]["mde_relative"] is None
        assert dumped[0]["mde_unavailable_reason"] == "unattainable"
        assert dumped[1]["mde_unavailable_reason"] is None

    @pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
    def test_frames_keep_a_numeric_null(self, backend: Literal["pandas", "polars", "pyarrow"]):
        import narwhals as nw

        frame = nw.from_native(self._curve().to_frame(backend=backend), eager_only=True)
        assert frame.schema["mde_relative"] == nw.Float64
        assert frame.schema["mde_unavailable_reason"] == nw.String
        assert frame["mde_relative"].is_null().to_list() == [True, False]
        assert frame["mde_unavailable_reason"].is_null().to_list() == [False, True]
        assert frame["numerical_qualification"].to_list() == [
            "closed_form_model_only_v1",
            "closed_form_model_only_v1",
        ]
        assert frame["mde_unavailable_reason"][0] == "unattainable"

    @pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
    def test_all_null_companion_column_stays_numeric(
        self, backend: Literal["pandas", "polars", "pyarrow"]
    ):
        import narwhals as nw

        only_unavailable = self._curve()[:1]
        frame = nw.from_native(only_unavailable.to_frame(backend=backend), eager_only=True)
        assert frame.schema["mde_relative"] == nw.Float64
        assert frame["mde_relative"].is_null().to_list() == [True]

    @pytest.mark.parametrize(
        ("mde_relative", "reason"),
        [(None, None), (0.06, "unattainable"), (None, "not_a_reason")],
    )
    def test_invalid_value_reason_pairs_are_rejected(self, mde_relative, reason):
        with pytest.raises(ValueError):
            PowerCurvePoint(
                solve_for="power",
                n_per_arm=100,
                n_total=200,
                relative_lift=0.05,
                target_power=0.8,
                alpha=0.05,
                power=0.7,
                power_basis="asymptotic",
                mde_relative=mde_relative,
                mde_unavailable_reason=reason,
                effective_var=0.09,
            )
