"""Portable fixed assignment and independent integrity outcomes."""

import json
import pickle
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from increment.assignment import (
    AssignmentRecord,
    AssignmentShare,
    AssignmentSpec,
    assign,
    assignment_bucket,
    audit_assignments,
    validate_assignment_specs,
)
from increment.errors import InvalidRequestError
from increment.semantics.models import AnalysisPlan, Experiment

VECTORS = json.loads((Path(__file__).parent / "fixtures/assignment_v1.json").read_text())


def spec(**changes):
    return AssignmentSpec.model_validate({**VECTORS["spec"], **changes})


def update(model, **changes):
    return type(model).model_validate({**model.model_dump(mode="json"), **changes})


def test_frozen_portable_vectors_and_canonical_round_trip():
    contract = spec()
    assert contract.canonical_json() == VECTORS["canonical_spec"]
    assert contract.fingerprint == VECTORS["fingerprint"]
    for vector in VECTORS["vectors"]:
        record = assign(contract, vector["unit_id"], eligible=vector["eligible"])
        assert record.model_dump(mode="json") == vector["record"]
        ns, layer = contract.layer.namespace, contract.layer
        identity = (ns.name, ns.version, ns.salt, ns.identity_key, vector["unit_id"])
        if vector["eligible"]:
            assert assignment_bucket("holdout", *identity) == vector["holdout_bucket"]
            assert (
                assignment_bucket("layer", *identity, layer.name, layer.version, layer.salt)
                == vector["layer_bucket"]
            )
            assert (
                assignment_bucket(
                    "arm",
                    *identity,
                    layer.name,
                    layer.version,
                    layer.salt,
                    contract.experiment,
                    contract.version,
                    contract.salt,
                    contract.eligibility_version,
                )
                == vector["arm_bucket"]
            )
    restored = AssignmentSpec.model_validate_json(contract.canonical_json())
    assert restored.fingerprint == contract.fingerprint
    assert assign(restored, "用户") == assign(contract, "用户")
    reordered = spec(arms=list(reversed(VECTORS["spec"]["arms"])))
    assert reordered.fingerprint == contract.fingerprint
    assert assign(reordered, "0") == assign(contract, "0")
    assert assignment_bucket("test", "a", "bc") == 1739
    assert assignment_bucket("test", "ab", "c") == 7907
    assert assignment_bucket("test", "é") != assignment_bucket("test", "e\u0301")


@pytest.mark.parametrize(
    "needle,replacement",
    [
        ('"schema_version":1', '"schema_version":1,"schema_version":1'),
        ('"identity_key":"user_id"', '"identity_key":"user_id","identity_key":"other"'),
    ],
)
def test_portable_json_rejects_duplicate_object_keys(needle, replacement):
    with pytest.raises(InvalidRequestError) as error:
        AssignmentSpec.model_validate_json(spec().canonical_json().replace(needle, replacement))
    assert error.value.code == "assignment.contract.invalid"


@pytest.mark.parametrize("payload", [b"\xff", 42, "{"])
def test_portable_json_keeps_standard_malformed_input_validation(payload):
    with pytest.raises(ValidationError):
        AssignmentSpec.model_validate_json(payload)


@pytest.mark.parametrize("unit_id", [1, True, b"1", "", None, "\ud800"])
def test_identity_is_never_coerced(unit_id):
    with pytest.raises(InvalidRequestError) as error:
        assign(spec(), unit_id)
    assert error.value.code == "assignment.input.identity"
    assert pickle.loads(pickle.dumps(error.value)).code == error.value.code
    assert deepcopy(error.value).context == error.value.context


@pytest.mark.parametrize("eligible", [1, None, "true"])
def test_eligibility_is_a_preassignment_boolean(eligible):
    with pytest.raises(InvalidRequestError, match="eligible"):
        assign(spec(), "0", eligible=eligible)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("algorithm", "other"),
        ("version", 1),
        ("salt", "\ud800"),
    ],
)
def test_invalid_portable_spec_refuses_with_code(field, value):
    with pytest.raises(InvalidRequestError):
        spec(**{field: value})


@pytest.mark.parametrize(
    "shares",
    [
        [{"name": "control", "buckets": 8000}, {"name": "control", "buckets": 2000}],
        [{"name": "control", "buckets": 8000}, {"name": "treatment", "buckets": 1000}],
        [{"name": "control", "buckets": 8000}, {"name": "treatment", "buckets": True}],
        [{"name": "control", "buckets": 8000}, {"name": "treatment", "buckets": 2000.0}],
    ],
)
def test_invalid_bucket_allocations(shares):
    with pytest.raises(InvalidRequestError):
        spec(arms=shares)


@pytest.mark.parametrize("label", ["(unassigned)", "(mixed assignment)"])
def test_accounting_labels_cannot_be_assignment_arms(label):
    with pytest.raises(InvalidRequestError) as error:
        spec(arms=[{"name": "control", "buckets": 8000}, {"name": label, "buckets": 2000}])
    assert error.value.code == "assignment.contract.invalid"


@pytest.mark.parametrize(
    "bucket,arm", [(0, "control"), (7999, "control"), (8000, "treatment"), (9999, "treatment")]
)
def test_half_open_arm_boundaries(monkeypatch, bucket, arm):
    def fixed(domain, *parts):
        return {"holdout": 1000, "layer": 0, "arm": bucket}[domain]

    monkeypatch.setattr("increment.assignment.assignment_bucket", fixed)
    assert assign(spec(), "id").arm == arm


@pytest.mark.parametrize(
    "holdout,layer,status",
    [
        (999, 0, "holdout"),
        (1000, 5999, "assigned"),
        (1000, 6000, "outside_layer"),
        (1000, 8999, "outside_layer"),
        (1000, 9000, "outside_layer"),
        (1000, 9999, "outside_layer"),
    ],
)
def test_holdout_and_layer_boundaries(monkeypatch, holdout, layer, status):
    monkeypatch.setattr(
        "increment.assignment.assignment_bucket",
        lambda domain, *parts: {"holdout": holdout, "layer": layer, "arm": 0}[domain],
    )
    assert assign(spec(), "id").status == status


def test_shared_layers_exclude_and_independent_layers_have_distinct_draws():
    checkout = spec()
    search = spec(experiment="search", salt="search-salt")
    other = spec(
        experiment="checkout",
        layer=update(
            checkout.layer,
            name="independent",
            experiments=(AssignmentShare(name="checkout", buckets=10000),),
        ),
    )
    validate_assignment_specs((checkout, search))
    checkout_arms, other_arms = [], []
    held_out = 0
    for i in range(300):
        a, b, c = (assign(s, str(i)) for s in (checkout, search, other))
        assert not (a.status == b.status == "assigned")
        assert (a.status == "holdout") == (b.status == "holdout") == (c.status == "holdout")
        held_out += a.status == "holdout"
        if a.arm is not None and c.arm is not None:
            checkout_arms.append(a.arm)
            other_arms.append(c.arm)
    assert held_out > 0
    assert checkout_arms != other_arms
    assert set(checkout_arms) == set(other_arms) == {"control", "treatment"}


@pytest.mark.parametrize("scope", ["namespace", "layer", "experiment"])
def test_conflicting_shared_declarations_refuse(scope):
    first = spec()
    if scope == "namespace":
        second = spec(
            layer=update(first.layer, namespace=update(first.layer.namespace, salt="changed"))
        )
    elif scope == "layer":
        second = spec(layer=update(first.layer, salt="changed"))
    else:
        second = spec(salt="changed")
    with pytest.raises(InvalidRequestError) as error:
        validate_assignment_specs((first, second))
    assert error.value.code == "assignment.contract.conflict"
    assert error.value.context["scope"] == scope


def test_versioned_stickiness_requires_new_rules_and_preserves_old_enrollments():
    old = spec()
    new = spec(
        version="2",
        arms=[{"name": "control", "buckets": 2000}, {"name": "treatment", "buckets": 8000}],
    )
    old.validate_successor(new)
    logged = assign(old, "0")
    assert audit_assignments(old, [logged]).replay == "matched"
    wrong = audit_assignments(new, [logged])
    assert wrong.replay == "failed"
    assert wrong.allocation.status == "not_checked_missing_counts"
    assert "specification_version" in wrong.mismatches[0].reasons
    for changed in (spec(salt="changed"), spec(eligibility_version="changed"), spec(arms=new.arms)):
        with pytest.raises(InvalidRequestError) as error:
            old.validate_successor(changed)
        assert error.value.code == "assignment.contract.version_reused"
    changed_ns = update(old.layer.namespace, holdout_buckets=2000)
    with pytest.raises(InvalidRequestError) as error:
        old.validate_successor(spec(version="2", layer=update(old.layer, namespace=changed_ns)))
    assert error.value.context["scope"] == "namespace"
    with pytest.raises(InvalidRequestError) as error:
        old.validate_successor(spec(version="2", layer=update(old.layer, salt="changed")))
    assert error.value.context["scope"] == "layer"


def test_audit_separates_hash_match_exclusion_and_observed_balance():
    contract = spec()
    records = [assign(contract, str(i)) for i in range(1000)]
    report = audit_assignments(contract, records)
    assert report.replay == "matched"
    assert report.exclusions == "not_violated"
    assert report.allocation.status == "not_rejected"
    assert report.allocation.construction == "always_valid"
    changed = [update(r, arm="treatment") if r.status == "assigned" else r for r in records]
    report = audit_assignments(contract, changed)
    assert report.replay == "failed"
    assert report.exclusions == "not_violated"
    assert report.allocation.status == "failed"
    held_out = assign(contract, "4")
    violation = audit_assignments(contract, [update(held_out, status="assigned", arm="control")])
    assert violation.exclusions == "failed"
    assert violation.exclusion_units == ("4",)
    assert violation.replay == "failed"


@pytest.mark.parametrize(
    "changes",
    [
        {"arm": "unexpected"},
        {"arm": None},
        {"status": "holdout"},
        {"status": "outside_layer", "arm": None},
        {"contract_fingerprint": "wrong"},
        {"specification_version": "old"},
    ],
)
def test_invalid_logs_do_not_publish_partial_balance(changes):
    report = audit_assignments(spec(), [update(assign(spec(), "0"), **changes)])
    assert report.replay == "failed"
    assert report.allocation.status == "not_checked_missing_counts"
    assert report.allocation.context["reason"] == "assignment_log_invalid"


def test_empty_duplicates_and_cluster_grain():
    contract = spec()
    empty = audit_assignments(contract, [])
    assert empty.replay == empty.exclusions == "not_checked"
    assert empty.allocation.status == "not_checked_missing_counts"
    record = assign(contract, "0")
    with pytest.raises(InvalidRequestError) as error:
        audit_assignments(contract, [record, record])
    assert error.value.code == "assignment.log.duplicate"
    report = audit_assignments(contract, [record], randomization_grain="cluster")
    assert report.allocation.randomization_grain == "cluster"
    for restored in (
        type(report).model_validate_json(report.model_dump_json()),
        deepcopy(report),
        pickle.loads(pickle.dumps(report)),
    ):
        assert restored == report
        assert restored.replay == "matched"
    restored_record = AssignmentRecord.model_validate_json(record.model_dump_json())
    assert audit_assignments(contract, [restored_record]).replay == "matched"


def experiment(**changes):
    return Experiment(
        name="checkout",
        exposure="assignment",
        unit="user_id",
        start="2025-01-01",
        control_group="control",
        plan=AnalysisPlan(primary="revenue"),
        **changes,
    )


def test_analysis_law_derives_from_contract_and_survives_serialization():
    exp = experiment(assignment_contract=spec())
    assert exp.resolved_design() == spec().design()
    assert exp.allocation == {"control": 0.8, "treatment": 0.2}
    assert exp.allocation_scheme == "independent"
    for restored in (Experiment.model_validate_json(exp.model_dump_json()),):
        assert restored.resolved_design() == exp.resolved_design()
        assert restored.assignment_contract is not None
        assert restored.assignment_contract.fingerprint == spec().fingerprint
    assert (
        experiment(
            assignment_contract=spec(), allocation={"control": 8e307, "treatment": 2e307}
        ).resolved_design()
        == spec().design()
    )
    external = experiment(allocation={"control": 0.8, "treatment": 0.2})
    assert external.assignment_contract is None
    assert external.resolved_design().allocation_scheme is None
    assert "assignment_contract" not in external.model_dump(mode="json")


@pytest.mark.parametrize(
    "changes",
    [
        {"allocation": {"control": 0.5, "treatment": 0.5}},
        {"allocation_scheme": "blocked"},
        {"cluster": "household"},
        {"design": {"mechanism": "observational", "covariates": [{"property": "x"}]}},
    ],
)
def test_contract_analysis_conflicts_refuse_before_data_loading(changes):
    with pytest.raises(InvalidRequestError) as error:
        experiment(assignment_contract=spec(), **changes)
    assert error.value.code == "assignment.contract.design_mismatch"
