"""The original registration, later lock and separately exported evidence bind."""

import json
import pickle
from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from increment._holdout_records import (
    ArmCounts,
    ConfigRevision,
    HoldoutReason,
    HoldoutSnapshot,
    HoldoutStudy,
    HoldoutWindow,
    Outcome,
    RosterEvidence,
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


def test_changed_terms_cannot_hide_a_lock_that_predates_registration(
    comparison: tuple[HoldoutStudy, HoldoutWindow],
) -> None:
    study, window = comparison
    lock = window.model_dump(mode="json")
    lock["locked_at"] = "2024-12-31T23:59:59Z"
    lock["terms"]["outcome_id"] = "revenue"

    with pytest.raises(InvalidRequestError) as error:
        check_registration(study, HoldoutWindow.model_validate(lock))

    assert error.value.code == "holdout.registration_mismatch"
    assert error.value.context == {"study_id": "shopping-25", "field": "locked_at"}


@pytest.mark.parametrize("fractional_field", ["registered_at", "locked_at"])
def test_a_lock_earlier_in_the_same_microsecond_cannot_be_rounded_to_registration(
    shopping: dict[str, Any], fractional_field: str
) -> None:
    raw_study = shopping["study"] | {
        "registered_at": (
            "2025-01-01T00:00:00.0000011Z"
            if fractional_field == "registered_at"
            else "2025-01-01T00:00:00.0000010Z"
        )
    }
    raw_window = shopping["window"] | {
        "locked_at": (
            "2025-01-01T00:00:00.0000010Z"
            if fractional_field == "registered_at"
            else "2025-01-01T00:00:00.0000009Z"
        )
    }
    with pytest.raises(InvalidRequestError) as error:
        study = HoldoutStudy.model_validate(raw_study)
        window = HoldoutWindow.model_validate(raw_window)
        check_registration(study, window)

    assert error.value.code == "model.field.type"
    assert error.value.context["field"] == fractional_field


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
    assert error.value.context["field"] == (
        "<unknown field>" if code == "model.field.unknown" else field
    )


@pytest.mark.parametrize("invalid_field", ["unit_id", "entered_at"])
def test_schema_refusals_do_not_echo_unit_identifiers(
    shopping: dict[str, Any], invalid_field: str
) -> None:
    raw = shopping["snapshot"]
    secret = "private-account-7e39"
    entry = raw["entries"][0] | {"unit_id": secret}
    if invalid_field == "unit_id":
        entry["unit_id"] = [secret]
        expected_code = "model.field.type"
    else:
        del entry["entered_at"]
        expected_code = "model.field.missing"

    with pytest.raises(InvalidRequestError) as error:
        HoldoutSnapshot.model_validate_json(json.dumps(raw | {"entries": [entry]}))

    assert error.value.code == expected_code
    assert error.value.context["field"] == invalid_field
    assert secret not in repr(error.value.context)
    assert secret not in str(error.value)
    assert secret not in str(error.value.__cause__)


@pytest.mark.parametrize(
    ("case", "loc", "code"),
    [
        ("missing_header", ("snapshot_ref",), "model.field.missing"),
        (
            "sdk_payload_extra",
            ("memberships", 0, "payload", "<unknown field>"),
            "model.field.unknown",
        ),
        ("unknown_header", ("<unknown field>",), "model.field.unknown"),
    ],
    ids=["missing_header", "sdk_payload_extra", "unknown_header"],
)
def test_snapshot_schema_refusals_redact_enclosing_and_embedded_values(
    shopping: dict[str, Any], case: str, loc: tuple[str | int, ...], code: str
) -> None:
    raw = shopping["snapshot"]
    secret = "private-account-7e39"
    if case == "missing_header":
        entry = raw["entries"][0] | {"unit_id": secret}
        snapshot = {"entries": [entry]} | {
            key: value for key, value in raw.items() if key not in {"entries", "snapshot_ref"}
        }
    elif case == "sdk_payload_extra":
        row = raw["memberships"][0]
        payload = row["payload"] | {"unit_id": secret}
        snapshot = raw | {"memberships": [row | {"payload": payload}]}
    else:
        snapshot = raw | {secret: None}

    with pytest.raises(InvalidRequestError) as error:
        HoldoutSnapshot.model_validate_json(json.dumps(snapshot))

    assert error.value.code == code
    assert error.value.context["loc"] == loc
    assert error.value.context["field"] == loc[-1]
    assert error.value.context["value"] is None
    assert secret not in str(error.value)
    cause = error.value.__cause__
    if isinstance(cause, ValidationError):
        assert secret not in repr(cause.errors())


def test_truncated_snapshot_json_does_not_echo_submitted_evidence() -> None:
    unit = "private-account-7e39"
    value = "76392107.4375"
    truncated = f'{{"outcomes":[{{"unit_id":"{unit}","value":{value}}}]'

    with pytest.raises(ValidationError) as error:
        HoldoutSnapshot.model_validate_json(truncated, strict=True, context={"label": "import"})

    failure = error.value
    details = failure.errors()
    assert [(detail["type"], detail["loc"], detail.get("input")) for detail in details] == [
        ("json_invalid", (), None),
    ]
    assert failure.__cause__ is None
    assert failure.__context__ is None
    for diagnostic in (str(failure), repr(details), failure.json()):
        assert unit not in diagnostic
        assert value not in diagnostic


def test_direct_outcome_cross_field_refusal_redacts_its_cause(shopping: dict[str, Any]) -> None:
    secret = "private-account-7e39"
    outcome = shopping["snapshot"]["outcomes"][0] | {
        "unit_id": secret,
        "state": "measured",
        "value": None,
    }

    with pytest.raises(InvalidRequestError) as error:
        Outcome.model_validate(outcome)

    assert error.value.code == "model.field.type"
    assert error.value.context["loc"] == ("value",)
    assert error.value.context["value"] is None
    assert secret not in str(error.value)
    cause = error.value.__cause__
    if isinstance(cause, ValidationError):
        assert secret not in repr(cause.errors())


def test_invalid_declaration_field_keeps_analyst_value(shopping: dict[str, Any]) -> None:
    with pytest.raises(InvalidRequestError) as error:
        HoldoutStudy.model_validate(shopping["study"] | {"scope_kind": "feature-family"})

    assert error.value.code == "model.field.literal"
    assert error.value.context["value"] == "feature-family"


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
    restored = HoldoutSnapshot.model_validate_json(
        encoded, extra="forbid", context={"label": "import"}, by_name=True
    )
    header = {
        key: value for key, value in json.loads(encoded).items() if not isinstance(value, list)
    }
    strict_header = HoldoutSnapshot.model_validate_json(
        json.dumps(header),
        strict=True,
        context={"label": "import"},
        by_alias=False,
        by_name=True,
    )
    assert strict_header.snapshot_ref == snapshot.snapshot_ref
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


@pytest.mark.parametrize("field", ["entries", "components"])
def test_positional_json_filters_select_before_canonical_ordering(
    shopping: dict[str, Any], field: str
) -> None:
    model: HoldoutSnapshot | HoldoutStudy
    if field == "entries":
        raw = shopping["snapshot"]
        model = HoldoutSnapshot.model_validate(raw | {field: raw[field][:2][::-1]})
        key, requested, first = "entry_id", "entry-2", "entry-1"
    else:
        raw = shopping["study"]
        model = HoldoutStudy.model_validate(raw | {field: raw[field][::-1]})
        key, requested, first = "launch_id", "B", "A"
    include = {field: {0: {key}}}
    exclude = {field: {-1: True}}

    assert json.loads(model.model_dump_json())[field][0][key] == first
    assert model.model_dump(include=include)[field] == ({key: requested},)
    assert model.model_dump(mode="json", include=include)[field] == [{key: requested}]
    assert json.loads(model.model_dump_json(exclude=exclude))[field][0][key] == requested


def test_revalidating_a_copied_snapshot_freezes_rows_again(shopping: dict[str, Any]) -> None:
    snapshot = HoldoutSnapshot.model_validate(shopping["snapshot"])
    entries = list(snapshot.entries[:1])
    copied = snapshot.model_copy(update={"entries": entries})

    validated = HoldoutSnapshot.model_validate(copied)
    entries.clear()

    assert isinstance(validated.entries, tuple)
    assert validated.entries[0].entry_id == "entry-1"


def test_revalidating_a_snapshot_checks_copied_nested_rows(shopping: dict[str, Any]) -> None:
    snapshot = HoldoutSnapshot.model_validate(shopping["snapshot"])
    malformed = snapshot.entries[0].model_copy(update={"eligible": 1})
    with pytest.raises(InvalidRequestError) as error:
        HoldoutSnapshot.model_validate(snapshot.model_copy(update={"entries": (malformed,)}))

    assert error.value.code == "model.field.type"
    assert error.value.context["field"] == "eligible"


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


def _layered_archive(
    snapshot: HoldoutSnapshot, raw_sdk: str, experiments: list[dict[str, object]]
) -> ConfigRevision:
    sdk = json.loads(raw_sdk)
    sdk["experiments"] = experiments
    sdk["holdouts"][0]["targets"] = ["shopping-layer"]
    sdk["layers"] = [
        {
            "key": "shopping-layer",
            "ranges": [
                {"experiment": experiment["key"], "start_bps": i * 1000, "end_bps": (i + 1) * 1000}
                for i, experiment in enumerate(experiments)
            ],
        }
    ]
    return ConfigRevision.model_validate(
        snapshot.configs[0].model_dump() | {"sdk_json": json.dumps(sdk)}
    )


def test_layer_target_alone_does_not_register_even_a_single_experiment(
    shopping: dict[str, Any],
) -> None:
    study = HoldoutStudy.model_validate(
        shopping["study"]
        | {"scope_kind": "feature", "components": [{"launch_id": "A", "target": "shopping-layer"}]}
    )
    snapshot = HoldoutSnapshot.model_validate(shopping["snapshot"])
    sdk = shopping["snapshot"]["configs"][0]["sdk_json"]
    config = _layered_archive(snapshot, sdk, json.loads(sdk)["experiments"][:1])
    with pytest.raises(InvalidRequestError) as error:
        check_config_binding(study, snapshot, config)
    assert error.value.code == "holdout.registration_mismatch"
    assert error.value.context == {"study_id": "shopping-25", "field": "components"}


@pytest.mark.parametrize("expansion", [True, False], ids=["expansion", "replacement"])
def test_registered_sdk_layer_rejects_a_new_experiment_under_the_same_target(
    shopping: dict[str, Any], expansion: bool
) -> None:
    study = HoldoutStudy.model_validate(
        shopping["study"]
        | {
            "components": [
                {"launch_id": "A", "target": "shopping-layer", "experiment_key": "shopping-a"},
                {"launch_id": "B", "target": "shopping-layer", "experiment_key": "shopping-b"},
            ]
        }
    )
    snapshot = HoldoutSnapshot.model_validate(shopping["snapshot"])
    sdk = shopping["snapshot"]["configs"][0]["sdk_json"]
    original = json.loads(sdk)["experiments"]
    check_config_binding(study, snapshot, _layered_archive(snapshot, sdk, original))
    new = original[1] | {"key": "shopping-c"}
    changed = [*(original if expansion else original[:1]), new]

    with pytest.raises(InvalidRequestError) as error:
        check_config_binding(study, snapshot, _layered_archive(snapshot, sdk, changed))
    assert error.value.code == "holdout.registration_mismatch"
    assert error.value.context == {"study_id": "shopping-25", "field": "components"}


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


def test_identical_archive_can_reactivate_without_filling_the_gap(
    shopping: dict[str, Any], comparison: tuple[HoldoutStudy, HoldoutWindow]
) -> None:
    study, _ = comparison
    raw = shopping["snapshot"]
    original = raw["configs"][0]
    first = original | {"effective_to": "2025-02-01T00:00:00Z"}
    second = original | {"effective_from": "2025-03-01T00:00:00Z"}
    snapshot = HoldoutSnapshot.model_validate(raw | {"configs": [second, first, first]})

    index = index_configs(study, snapshot)
    assert len(index.activations["r1"]) == 2
    assert index.at("r1", datetime(2025, 1, 31, tzinfo=UTC)) == snapshot.configs[1]
    assert index.at("r1", datetime(2025, 2, 1, tzinfo=UTC)) is None
    assert index.at("r1", datetime(2025, 3, 1, tzinfo=UTC)) == snapshot.configs[0]
    assert index.at("r1", datetime(2025, 4, 12, tzinfo=UTC)) is None


def test_nonidentical_overlapping_activations_are_ambiguous(
    shopping: dict[str, Any], comparison: tuple[HoldoutStudy, HoldoutWindow]
) -> None:
    study, _ = comparison
    raw = shopping["snapshot"]
    original = raw["configs"][0]
    overlap = original | {"effective_to": "2025-02-01T00:00:00Z"}
    snapshot = HoldoutSnapshot.model_validate(raw | {"configs": [original, overlap]})

    with pytest.raises(InvalidRequestError) as error:
        index_configs(study, snapshot)
    assert error.value.code == "holdout.registration_mismatch"
    assert error.value.context == {"study_id": "shopping-25", "field": "configs"}


def test_nested_reason_context_survives_copy_and_pickle() -> None:
    context = {"fields": ["end"]}
    evidence = RosterEvidence(
        members=(),
        identity=(),
        values=(),
        held_out=ArmCounts(),
        eligible=ArmCounts(),
        n_eligible_entries=0,
        n_ineligible_entries=0,
        n_unclassified=0,
        n_after_freeze=0,
        reasons=(
            HoldoutReason.model_validate(
                {"code": "holdout.window_not_registered", "context": context}
            ),
        ),
    )
    context["fields"].append("outcome_id")

    for restored in (
        evidence.model_copy(deep=True),
        deepcopy(evidence),
        pickle.loads(pickle.dumps(evidence)),
    ):
        assert restored == evidence
        assert restored.reasons[0].context == {"fields": ("end",)}
