"""Conditional precision uses the original frozen cohort and per-unit scale."""

import json
import math
import sys
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

import pytest
from scipy.stats import nct, t

from increment._holdout_records import HoldoutStudy, HoldoutWindow
from increment.errors import InvalidRequestError
from increment.power.holdout import plan_holdout


@pytest.fixture(scope="module")
def comparison() -> tuple[HoldoutStudy, HoldoutWindow]:
    fixture = Path(__file__).parents[1] / "fixtures/holdouts/shopping_v1.json"
    records = json.loads(fixture.read_text())
    return HoldoutStudy.model_validate(records["study"]), HoldoutWindow.model_validate(
        records["window"]
    )


def _registered_rate(
    comparison: tuple[HoldoutStudy, HoldoutWindow], p0: str
) -> tuple[HoldoutStudy, HoldoutWindow]:
    study, window = (record.model_dump() for record in comparison)
    study_id = f"shopping-holdout-{p0}"
    registration_ref = f"registration/{study_id}/2025-01-01"
    study.update(study_id=study_id, registration_ref=registration_ref, holdout_probability=p0)
    window.update(
        study_id=study_id,
        registration_ref=registration_ref,
        lock_ref=f"locks/{study_id}/mar-orders",
    )
    return HoldoutStudy.model_validate(study), HoldoutWindow.model_validate(window)


@pytest.mark.parametrize(
    ("mature", "counts", "se", "df", "width", "power"),
    [
        (1000, (100, 900), 0.210818511, 122.0562824, 0.8346687, 0.6528669),
        (500, (50, 450), 0.298142397, 60.4124334, 1.1925797, 0.3785984),
    ],
)
def test_unequal_allocation_at_long_horizon(
    comparison: tuple[HoldoutStudy, HoldoutWindow],
    mature: int,
    counts: tuple[int, int],
    se: float,
    df: float,
    width: float,
    power: float,
) -> None:
    result = plan_holdout(
        *comparison,
        projected_assignments=1000,
        projected_by_freeze=mature,
        variance_eligible=4,
        variance_held_out=4,
        effect=0.5,
    )

    assert (result.n_assigned_held_out, result.n_assigned_eligible) == (100, 900)
    assert (result.n_mature_held_out, result.n_mature_eligible) == counts
    assert (result.standard_error, result.df, result.interval_width, result.power) == pytest.approx(
        (se, df, width, power), abs=5e-7, rel=0
    )
    assert result.reason_code is result.reason is None
    expected = {
        "window_id": "mar-orders",
        "start": "2025-03-10T00:00:00Z",
        "end": "2025-04-10T00:00:00Z",
        "assignment_unit": "subscriber",
        "outcome_unit": "orders per subscriber",
        "p0": "0.10",
        "projected_assignments": 1000,
        "projected_by_freeze": mature,
        "variance_eligible": 4,
        "variance_held_out": 4,
        "effect": 0.5,
        "alpha": 0.05,
        "basis": "conditional_welch_approximation",
    }
    assert json.loads(result.model_dump_json(include=set(expected))) == expected


def test_separately_registered_balanced_plan(
    comparison: tuple[HoldoutStudy, HoldoutWindow],
) -> None:
    result = plan_holdout(
        *_registered_rate(comparison, "0.50"),
        projected_assignments=1000,
        projected_by_freeze=1000,
        variance_eligible=4,
        variance_held_out=4,
        effect=0.5,
    )
    assert result.p0 == Decimal("0.50")
    assert (result.n_mature_held_out, result.n_mature_eligible) == (500, 500)
    assert (result.standard_error, result.df, result.interval_width, result.power) == pytest.approx(
        (0.126491106, 998, 0.4964381, 0.9766536), abs=5e-7, rel=0
    )


def test_planned_width_at_matched_three_per_arm(
    comparison: tuple[HoldoutStudy, HoldoutWindow],
) -> None:
    result = plan_holdout(
        *_registered_rate(comparison, "0.50"),
        projected_assignments=6,
        projected_by_freeze=6,
        variance_eligible=4,
        variance_held_out=4,
        effect=2,
    )
    assert (result.n_mature_held_out, result.n_mature_eligible) == (3, 3)
    assert result.standard_error == pytest.approx(math.sqrt(8 / 3), rel=1e-12)
    assert result.interval_width == pytest.approx(6.533915871 - (-2.533915871), abs=5e-9)


@pytest.mark.parametrize("effect", [-0.5, 0.0, 0.5])
def test_unequal_variances_use_both_noncentral_t_tails(
    comparison: tuple[HoldoutStudy, HoldoutWindow], effect: float
) -> None:
    result = plan_holdout(
        *comparison,
        projected_assignments=1000,
        projected_by_freeze=1000,
        variance_eligible=4,
        variance_held_out=9,
        effect=effect,
    )
    a, b = 4 / 900, 9 / 100
    se = math.sqrt(a + b)
    df = (a + b) ** 2 / (a**2 / 899 + b**2 / 99)
    critical = t.isf(0.025, df)
    expected = nct.cdf(-critical, df, effect / se) + nct.sf(critical, df, effect / se)
    assert (result.standard_error, result.df, result.interval_width, result.power) == pytest.approx(
        (se, df, 2 * critical * se, expected), rel=1e-10
    )
    if effect == 0:
        assert result.power == pytest.approx(0.05, rel=1e-12)


def test_projection_floors_exact_probability_regardless_of_decimal_context(
    comparison: tuple[HoldoutStudy, HoldoutWindow],
) -> None:
    with localcontext(prec=2):
        result = plan_holdout(
            *comparison,
            projected_assignments=10**18 + 9,
            projected_by_freeze=999,
            variance_eligible=4,
            variance_held_out=4,
            effect=0.5,
        )
    assert (result.n_assigned_held_out, result.n_assigned_eligible) == (10**17, 9 * 10**17 + 9)
    assert (result.n_mature_held_out, result.n_mature_eligible) == (99, 900)


@pytest.mark.parametrize("variance", [math.ulp(0.0), sys.float_info.max])
def test_planned_standard_error_survives_variance_scale_extremes(
    comparison: tuple[HoldoutStudy, HoldoutWindow], variance: float
) -> None:
    result = plan_holdout(
        *_registered_rate(comparison, "0.50"),
        projected_assignments=6,
        projected_by_freeze=6,
        variance_eligible=variance,
        variance_held_out=variance,
        effect=0,
    )
    expected_se = math.sqrt(variance) * math.sqrt(2 / 3)
    assert result.standard_error == pytest.approx(expected_se, rel=1e-12, abs=0)
    assert result.df == pytest.approx(4.0, rel=1e-12)
    assert result.interval_width == pytest.approx(2 * 2.776445105 * expected_se, rel=1e-9, abs=0)


def test_single_positive_variance_uses_its_own_degrees_of_freedom(
    comparison: tuple[HoldoutStudy, HoldoutWindow],
) -> None:
    result = plan_holdout(
        *comparison,
        projected_assignments=1000,
        projected_by_freeze=1000,
        variance_eligible=0,
        variance_held_out=9,
        effect=0.5,
    )
    critical = t.isf(0.025, 99)
    expected = nct.cdf(-critical, 99, 5 / 3) + nct.sf(critical, 99, 5 / 3)
    assert (result.standard_error, result.df, result.interval_width, result.power) == pytest.approx(
        (0.3, 99, 0.6 * critical, expected), rel=1e-10
    )


@pytest.mark.parametrize(
    ("mature", "variance_eligible", "variance_held_out", "counts", "reason"),
    [(19, 4, 4, (1, 18), "arm_too_small"), (1000, 0, 0, (100, 900), "zero_variance")],
)
def test_unavailable_precision_keeps_counts_and_null_numbers(
    comparison: tuple[HoldoutStudy, HoldoutWindow],
    mature: int,
    variance_eligible: float,
    variance_held_out: float,
    counts: tuple[int, int],
    reason: str,
) -> None:
    result = plan_holdout(
        *comparison,
        projected_assignments=1000,
        projected_by_freeze=mature,
        variance_eligible=variance_eligible,
        variance_held_out=variance_held_out,
        effect=0.5,
    )
    assert (result.n_mature_held_out, result.n_mature_eligible) == counts
    assert result.reason_code == "holdout.precision_unavailable"
    assert result.reason == reason
    payload = json.loads(result.model_dump_json())
    assert all(payload[key] is None for key in ("standard_error", "df", "interval_width", "power"))


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("projected_assignments", -1, "model.field.range"),
        ("projected_by_freeze", 1001, "model.field.range"),
        ("projected_by_freeze", True, "model.field.type"),
        ("variance_eligible", -1.0, "model.field.range"),
        ("variance_held_out", True, "model.field.type"),
        ("variance_held_out", float("inf"), "model.field.nonfinite"),
        ("effect", float("nan"), "model.field.nonfinite"),
    ],
)
def test_invalid_projection_is_refused(
    comparison: tuple[HoldoutStudy, HoldoutWindow], field: str, value: object, code: str
) -> None:
    kwargs: dict[str, Any] = {
        "projected_assignments": 1000,
        "projected_by_freeze": 1000,
        "variance_eligible": 4,
        "variance_held_out": 4,
        "effect": 0.5,
    }
    kwargs[field] = value
    with pytest.raises(InvalidRequestError) as error:
        plan_holdout(*comparison, **kwargs)
    assert error.value.code == code
    assert error.value.context["field"] == field


def test_planning_requires_original_registered_terms(
    comparison: tuple[HoldoutStudy, HoldoutWindow],
) -> None:
    study, window = comparison
    study = study.model_copy(update={"registered_window": None})
    with pytest.raises(InvalidRequestError) as error:
        plan_holdout(
            study,
            window,
            projected_assignments=1000,
            projected_by_freeze=1000,
            variance_eligible=4,
            variance_held_out=4,
            effect=0.5,
        )
    assert error.value.code == "holdout.window_not_registered"
    assert error.value.context == {
        "study_id": "shopping-25",
        "window_id": "mar-orders",
        "fields": ("registered_window",),
    }


def test_planning_cannot_offer_iid_precision_for_a_dependent_law(
    comparison: tuple[HoldoutStudy, HoldoutWindow],
) -> None:
    study, window = comparison
    study = HoldoutStudy.model_validate(study.model_dump() | {"allocation_scheme": "clustered"})
    with pytest.raises(ValueError):
        plan_holdout(
            study,
            window,
            projected_assignments=1000,
            projected_by_freeze=1000,
            variance_eligible=4,
            variance_held_out=4,
            effect=0.5,
        )


def test_unrepresentable_mature_counts_are_refused(
    comparison: tuple[HoldoutStudy, HoldoutWindow],
) -> None:
    with pytest.raises(InvalidRequestError) as error:
        plan_holdout(
            *comparison,
            projected_assignments=10**310,
            projected_by_freeze=10**310,
            variance_eligible=4,
            variance_held_out=4,
            effect=0.5,
        )
    assert error.value.code == "holdout.precision_unrepresentable"
    assert error.value.context == {"window_id": "mar-orders"}
