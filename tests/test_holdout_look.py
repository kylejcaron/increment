"""A descriptive request never admits the fixed-horizon inference path."""

import json
import pickle
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from increment._holdout_look import check_design, check_look
from increment._holdout_records import HoldoutStudy, HoldoutWindow
from increment.errors import InvalidRequestError, UnsupportedRequestError


@pytest.fixture(scope="module")
def comparison() -> tuple[HoldoutStudy, HoldoutWindow]:
    fixture = Path(__file__).parent / "fixtures/holdouts/shopping_v1.json"
    records = json.loads(fixture.read_text())
    return HoldoutStudy.model_validate(records["study"]), HoldoutWindow.model_validate(
        records["window"]
    )


@pytest.mark.parametrize(
    "as_of", [datetime(2025, 3, 25, tzinfo=UTC), datetime(2025, 4, 13, tzinfo=UTC)]
)
def test_interim_cannot_issue_fixed_horizon_interval(
    comparison: tuple[HoldoutStudy, HoldoutWindow], as_of: datetime
) -> None:
    assert check_look(*comparison, as_of=as_of, request="describe") == "descriptive"
    with pytest.raises(InvalidRequestError) as error:
        check_look(*comparison, as_of=as_of, request="fixed_horizon")

    restored = pickle.loads(pickle.dumps(error.value))
    assert isinstance(restored, InvalidRequestError)
    assert restored.code == "holdout.not_final_cut"
    assert restored.context == {
        "window_id": "mar-orders",
        "as_of": as_of,
        "final_as_of": datetime(2025, 4, 12, tzinfo=UTC),
    }
    with pytest.raises(TypeError):
        restored.context["as_of"] = datetime(2025, 4, 12, tzinfo=UTC)


def test_replaying_the_exact_cut_does_not_turn_description_into_inference(
    comparison: tuple[HoldoutStudy, HoldoutWindow],
) -> None:
    cut = datetime(2025, 4, 12, tzinfo=UTC)
    assert check_look(*comparison, as_of=cut, request="fixed_horizon") == "final"
    replayed_cut = cut.astimezone(timezone(timedelta(hours=-7)))
    assert check_look(*comparison, as_of=replayed_cut, request="fixed_horizon") == "final"
    assert check_look(*comparison, as_of=cut, request="describe") == "descriptive"


def test_final_cut_may_equal_window_end(
    comparison: tuple[HoldoutStudy, HoldoutWindow],
) -> None:
    study, window = (record.model_dump() for record in comparison)
    cut = datetime(2025, 4, 10, tzinfo=UTC)
    study["registered_window"]["final_as_of"] = cut
    window["terms"]["final_as_of"] = cut
    assert (
        check_look(
            HoldoutStudy.model_validate(study),
            HoldoutWindow.model_validate(window),
            as_of=cut,
            request="fixed_horizon",
        )
        == "final"
    )


@pytest.mark.parametrize(
    "as_of", [datetime(2025, 3, 25, tzinfo=UTC), datetime(2025, 4, 12, tzinfo=UTC)]
)
def test_sequential_request_is_unsupported_even_at_the_final_cut(
    comparison: tuple[HoldoutStudy, HoldoutWindow], as_of: datetime
) -> None:
    with pytest.raises(UnsupportedRequestError) as error:
        check_look(*comparison, as_of=as_of, request="sequential")
    assert error.value.code == "holdout.sequential_unsupported"
    assert error.value.context == {"window_id": "mar-orders"}


def test_exploratory_window_cannot_become_fixed_horizon_at_its_claimed_cut(
    comparison: tuple[HoldoutStudy, HoldoutWindow],
) -> None:
    study, window = comparison
    study = study.model_copy(update={"registered_window": None})
    cut = datetime(2025, 4, 12, tzinfo=UTC)
    assert check_look(study, window, as_of=cut, request="describe") == "descriptive"
    with pytest.raises(InvalidRequestError) as error:
        check_look(study, window, as_of=cut, request="fixed_horizon")
    assert error.value.code == "holdout.window_not_registered"
    assert error.value.context == {
        "study_id": "shopping-25",
        "window_id": "mar-orders",
        "fields": ("registered_window",),
    }


def test_descriptive_look_still_refuses_foreign_registration(
    comparison: tuple[HoldoutStudy, HoldoutWindow],
) -> None:
    study, window = comparison
    foreign_window = HoldoutWindow.model_validate(
        window.model_dump() | {"registration_ref": "registration/another-study/1"}
    )
    with pytest.raises(InvalidRequestError) as error:
        check_look(study, foreign_window, as_of=window.terms.final_as_of, request="describe")
    assert error.value.code == "holdout.registration_mismatch"
    assert error.value.context == {"study_id": "shopping-25", "field": "registration_ref"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("allocation_scheme", "clustered"),
        ("entry_basis", "post_assignment_filter"),
        ("no_interference", False),
        ("no_interference", None),
        ("design_ref", None),
    ],
)
def test_design_gate_names_the_canonical_reason_without_observation_rows(
    comparison: tuple[HoldoutStudy, HoldoutWindow], field: str, value: object
) -> None:
    study, _ = comparison
    assert check_design(study) is None
    changed = HoldoutStudy.model_validate(study.model_dump() | {field: value})
    finding = check_design(changed)
    assert finding is not None
    assert finding.code == "portfolio.impact.unsupported_design"
    assert finding.context == {}
