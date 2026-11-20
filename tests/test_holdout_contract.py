"""The original registration, later lock and separately exported evidence bind."""

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from increment._holdout_records import (
    ConfigRevision,
    HoldoutSnapshot,
    HoldoutStudy,
    HoldoutWindow,
    check_config_binding,
    check_registration,
    check_snapshot_binding,
    index_configs,
    parse_sdk_archive,
    require_registration,
)
from increment.errors import InvalidRequestError


@pytest.fixture(scope="module")
def shopping() -> dict[str, Any]:
    return json.loads((Path(__file__).parent / "fixtures/holdouts/shopping_v1.json").read_text())


@pytest.fixture
def comparison(shopping: dict[str, Any]) -> tuple[HoldoutStudy, HoldoutWindow]:
    return HoldoutStudy.model_validate(shopping["study"]), HoldoutWindow.model_validate(
        shopping["window"]
    )


def test_lock_cannot_change_the_january_comparison(
    comparison: tuple[HoldoutStudy, HoldoutWindow],
) -> None:
    study, window = comparison
    lock = window.model_dump(mode="json")
    lock["terms"]["end"] = "2025-06-10T00:00:00Z"

    with pytest.raises(InvalidRequestError) as error:
        require_registration(study, HoldoutWindow.model_validate(lock))

    assert error.value.code == "holdout.window_not_registered"
    assert error.value.context == {
        "study_id": "shopping-25",
        "window_id": "mar-orders",
        "fields": ("end",),
    }


def test_absent_original_slot_is_a_descriptive_finding(
    comparison: tuple[HoldoutStudy, HoldoutWindow],
) -> None:
    study, window = comparison
    finding = check_registration(study.model_copy(update={"registered_window": None}), window)

    assert finding is not None
    assert finding.code == "holdout.window_not_registered"
    assert finding.context == {
        "study_id": "shopping-25",
        "window_id": "mar-orders",
        "fields": ("registered_window",),
    }
    with pytest.raises(TypeError):
        finding.context["fields"] = ()  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    ("locked_at", "accepted"),
    [
        ("2024-12-31T23:59:59Z", False),
        ("2025-01-01T00:00:00Z", True),
        ("2025-03-01T00:00:00Z", True),
        ("2025-03-10T00:00:00Z", False),
    ],
)
def test_confirmation_lock_may_follow_first_assignment_but_not_freeze(
    comparison: tuple[HoldoutStudy, HoldoutWindow],
    locked_at: str,
    accepted: bool,
) -> None:
    study, window = comparison
    lock = HoldoutWindow.model_validate(window.model_dump() | {"locked_at": locked_at})

    if accepted:
        assert check_registration(study, lock) is None
    else:
        with pytest.raises(InvalidRequestError) as error:
            check_registration(study, lock)
        assert error.value.code == "holdout.registration_mismatch"
        assert error.value.context == {"study_id": "shopping-25", "field": "locked_at"}


@pytest.mark.parametrize(
    ("field", "date", "accepted"),
    [("final_as_of", "2025-04-10T00:00:00Z", True), ("end", "2025-03-10T00:00:00Z", False)],
)
def test_registered_window_requires_a_positive_follow_up_not_a_reporting_lag(
    comparison: tuple[HoldoutStudy, HoldoutWindow],
    field: str,
    date: str,
    accepted: bool,
) -> None:
    study, window = comparison
    terms = window.terms.model_dump() | {field: date}
    study = HoldoutStudy.model_validate(study.model_dump() | {"registered_window": terms})
    window = HoldoutWindow.model_validate(window.model_dump() | {"terms": terms})

    if accepted:
        assert check_registration(study, window) is None
    else:
        with pytest.raises(InvalidRequestError) as error:
            check_registration(study, window)
        assert error.value.code == "holdout.registration_mismatch"
        assert error.value.context == {"study_id": "shopping-25", "field": "end"}


@pytest.mark.parametrize("probability", ["0", "1", "0.00000000000000000001", True])
def test_probability_must_define_two_representable_arms(
    shopping: dict[str, Any], probability: object
) -> None:
    with pytest.raises(InvalidRequestError) as error:
        HoldoutStudy.model_validate(shopping["study"] | {"holdout_probability": probability})

    assert error.value.code == ("model.field.type" if probability is True else "model.field.range")
    assert error.value.context["field"] == "holdout_probability"


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("entered_at", "2025-01-02T00:00:00", "model.field.type"),
        ("unit_id", 2, "model.field.type"),
        ("eligible", 1, "model.field.type"),
        ("extra", "unexpected", "model.field.unknown"),
    ],
)
def test_nested_entry_fields_are_not_coerced(
    shopping: dict[str, Any], field: str, value: object, code: str
) -> None:
    snapshot = shopping["snapshot"]
    entry = snapshot["entries"][0] | {field: value}
    with pytest.raises(InvalidRequestError) as error:
        HoldoutSnapshot.model_validate(snapshot | {"entries": [entry]})

    assert error.value.code == code
    assert error.value.context["field"] == field


def test_transport_retains_opaque_ids_exact_decimals_and_utc(shopping: dict[str, Any]) -> None:
    raw = shopping["snapshot"]
    entry = raw["entries"][0] | {
        "unit_id": " É:01 ",
        "entered_at": "2025-01-01T19:00:00-05:00",
    }
    entries = [entry, *raw["entries"][1:]]
    snapshot = HoldoutSnapshot.model_validate(raw | {"entries": entries})
    reordered = HoldoutSnapshot.model_validate(raw | {"entries": entries[::-1]})
    encoded = snapshot.model_dump_json()
    study = HoldoutStudy.model_validate(shopping["study"])

    assert encoded == reordered.model_dump_json()
    restored = HoldoutSnapshot.model_validate_json(encoded)
    assert restored.model_dump_json() == encoded
    restored_entry = next(row for row in restored.entries if row.entry_id == "entry-1")
    assert restored_entry.unit_id == " É:01 "
    assert restored_entry.entered_at == datetime(2025, 1, 2, tzinfo=UTC)
    assert "2025-01-02T00:00:00Z" in encoded
    assert isinstance(restored.entries, tuple)
    assert HoldoutStudy.model_validate_json(study.model_dump_json()).holdout_probability == Decimal(
        "0.10"
    )
    with pytest.raises(ValidationError):
        restored_entry.unit_id = "renamed"  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    ("record", "field"),
    [("window", "registration_ref"), ("snapshot", "study_id"), ("snapshot", "window_id")],
)
def test_cannot_relabel_foreign_headers(
    shopping: dict[str, Any],
    comparison: tuple[HoldoutStudy, HoldoutWindow],
    record: str,
    field: str,
) -> None:
    study, window = comparison
    altered = shopping[record] | {field: "foreign/1"}
    with pytest.raises(InvalidRequestError) as error:
        if record == "window":
            check_registration(study, HoldoutWindow.model_validate(altered))
        else:
            check_snapshot_binding(study, window, HoldoutSnapshot.model_validate(altered))

    assert error.value.code == "holdout.registration_mismatch"
    assert error.value.context == {"study_id": "shopping-25", "field": field}


def test_sdk_and_external_records_preserve_distinct_assignment_proofs(
    shopping: dict[str, Any], comparison: tuple[HoldoutStudy, HoldoutWindow]
) -> None:
    study, window = comparison
    sdk = HoldoutSnapshot.model_validate(shopping["snapshot"])
    external = HoldoutSnapshot.model_validate(shopping["external_assignment"])
    check_snapshot_binding(study, window, sdk)
    check_snapshot_binding(study, window, external)
    check_config_binding(study, sdk, sdk.configs[0])
    check_config_binding(study, external, external.configs[0])
    sdk_payloads = {row.entry_id: row.payload.model_dump(mode="json") for row in sdk.memberships}
    ext_payloads = {
        row.entry_id: row.payload.model_dump(mode="json") for row in external.memberships
    }
    assert sdk_payloads["entry-2"] == {
        "protocol": 1,
        "revision": "r1",
        "holdout": "shopping",
        "group": "held_out",
        "group_bps": 1000,
    }
    assert ext_payloads["entry-2"] == {
        "holdout_key": "shopping",
        "revision": "r1",
        "group": "held_out",
        "holdout_probability": "0.10",
        "assignment_ref": "ext/r1/entry-2",
    }
    assert (sdk_payloads["entry-1"]["group_bps"], ext_payloads["entry-1"]["group"]) == (
        9000,
        "eligible",
    )


def test_launch_label_is_not_an_sdk_archive(
    shopping: dict[str, Any], comparison: tuple[HoldoutStudy, HoldoutWindow]
) -> None:
    raw = shopping["snapshot"]["configs"][0]
    config = ConfigRevision.model_validate(raw | {"sdk_json": '"A/2"'})
    with pytest.raises(InvalidRequestError) as error:
        parse_sdk_archive(config)
    assert error.value.code == "allocation.invalid_config"


@pytest.mark.parametrize(
    ("field", "altered", "binding"),
    [("law_id", "different-law/1", "law_id"), ("targets", ["shopping-a"], "components")],
)
def test_a_new_archive_cannot_change_registered_assignment_law_or_scope(
    shopping: dict[str, Any],
    comparison: tuple[HoldoutStudy, HoldoutWindow],
    field: str,
    altered: object,
    binding: str,
) -> None:
    study, _ = comparison
    snapshot = HoldoutSnapshot.model_validate(shopping["external_assignment"])
    config = snapshot.configs[0].model_dump()
    config["external_policy"][field] = altered

    with pytest.raises(InvalidRequestError) as error:
        check_config_binding(study, snapshot, ConfigRevision.model_validate(config))

    assert error.value.code == "holdout.registration_mismatch"
    assert error.value.context == {"study_id": "shopping-25", "field": binding}


def test_same_archive_ref_cannot_attest_different_policy_content(
    shopping: dict[str, Any], comparison: tuple[HoldoutStudy, HoldoutWindow]
) -> None:
    study, _ = comparison
    raw = shopping["external_assignment"]
    original = raw["configs"][0]
    changed = original | {
        "external_policy": original["external_policy"] | {"law_id": "different-law/1"}
    }
    snapshot = HoldoutSnapshot.model_validate(raw | {"configs": [original, changed]})

    with pytest.raises(InvalidRequestError) as error:
        index_configs(study, snapshot)

    assert error.value.code == "holdout.registration_mismatch"
    assert error.value.context == {"study_id": "shopping-25", "field": "configs"}
