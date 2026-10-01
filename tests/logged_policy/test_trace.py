"""Trace admission: the spec's Fixture 3 refusals plus every structural gate.

Each refusal is asserted by ``.code`` and its immutable ``.context``; the
rendered message is never matched.
"""

from __future__ import annotations

import copy
import json
import math
import pickle
from collections.abc import Mapping, MutableMapping
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import numpy as np
import pytest

from increment.errors import CapabilityError, CodedError, InvalidRequestError
from increment.logged_policy import (
    PROPENSITY_FLOOR,
    REFERENCE_POLICY_V1,
    TARGET_POLICY_V1,
    DecisionRecord,
    LoggedTrace,
    PolicyRegistry,
    TabularPolicy,
    estimate_policy_contrast,
)

T0 = datetime(2026, 1, 3, tzinfo=UTC)
HOUR = timedelta(hours=1)

# The registry the spec's Fixture 3 names: epsilon-greedy/v3 plays A with 0.10 at x=0.
EG_V3 = TabularPolicy(
    policy_id="epsilon-greedy",
    version="v3",
    probabilities={0: {"A": 0.1, "B": 0.9}, 1: {"A": 0.9, "B": 0.1}},
)
FIXED = TabularPolicy(policy_id="fixed-random", version="v1", default={"A": 0.5, "B": 0.5})
REGISTRY = PolicyRegistry([EG_V3, FIXED])


def record(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "decision_time": T0,
        "unit_id": "u07",
        "decision_index": 1,
        "candidate_actions": ("A", "B"),
        "chosen_action": "A",
        "propensity": 0.1,
        "logging_policy_id": "epsilon-greedy",
        "logging_policy_version": "v3",
        "pre_decision_context": {"x": 0, "prior": None},
        "update_batch": "eg-b003",
        "reward_observation_boundary": T0 + HOUR,
        "reward": 0.2,
    }
    base.update(overrides)
    return base


def test_fixture3_stale_propensity_is_refused_with_its_coordinates():
    with pytest.raises(InvalidRequestError) as raised:
        LoggedTrace.from_records([record(propensity=0.9)], registry=REGISTRY)
    assert raised.value.code == "logged_policy.trace.propensity_stale"
    context = raised.value.context
    assert context["unit_id"] == "u07"
    assert context["decision_index"] == 1
    assert context["logging_policy_id"] == "epsilon-greedy"
    assert context["logging_policy_version"] == "v3"
    assert context["chosen_action"] == "A"
    assert context["logged"] == 0.9
    assert context["registered"] == 0.1


def test_fixture3_missing_propensity_is_refused_distinctly():
    row = record(unit_id="u08", chosen_action="B", propensity=None, reward=0.5)
    row["pre_decision_context"] = {"x": 1, "prior": None}
    with pytest.raises(InvalidRequestError) as raised:
        LoggedTrace.from_records([row], registry=REGISTRY)
    assert raised.value.code == "logged_policy.trace.propensity_missing"
    assert raised.value.context["unit_id"] == "u08"
    assert raised.value.context["decision_index"] == 1


def test_fixture3_both_rows_refuse_at_the_record_gate_before_the_registry_audit():
    stale = record(propensity=0.9)
    missing = record(unit_id="u08", chosen_action="B", propensity=None, reward=0.5)
    with pytest.raises(InvalidRequestError) as raised:
        LoggedTrace.from_records([stale, missing], registry=REGISTRY)
    # Record-level gates run while records are built, in input order, so the
    # missing propensity (a record defect) fires before the stale audit.
    assert raised.value.code == "logged_policy.trace.propensity_missing"


def test_refusal_context_is_immutable():
    with pytest.raises(CodedError) as raised:
        LoggedTrace.from_records([record(propensity=0.9)], registry=REGISTRY)
    with pytest.raises(TypeError):
        cast("MutableMapping[str, object]", raised.value.context)["logged"] = 0.1


@pytest.mark.parametrize(
    ("propensity", "code"),
    [
        (math.nan, "logged_policy.trace.propensity_nonfinite"),
        (math.inf, "logged_policy.trace.propensity_nonfinite"),
        (0.0, "logged_policy.trace.propensity_out_of_range"),
        (1.0, "logged_policy.trace.propensity_out_of_range"),
        (-0.2, "logged_policy.trace.propensity_out_of_range"),
        (0.04, "logged_policy.trace.propensity_below_floor"),
    ],
)
def test_propensity_gates_refuse_before_the_registry_is_consulted(propensity, code):
    with pytest.raises(InvalidRequestError) as raised:
        DecisionRecord(**record(propensity=propensity))
    assert raised.value.code == code
    assert raised.value.context["propensity"] is propensity or math.isnan(propensity)
    if code == "logged_policy.trace.propensity_below_floor":
        assert raised.value.context["floor"] == PROPENSITY_FLOOR == 0.05


def test_propensity_exactly_at_the_floor_is_admitted():
    policy = TabularPolicy(
        policy_id="edge", version="v1", probabilities={0: {"A": 0.05, "B": 0.95}}
    )
    row = record(propensity=0.05, logging_policy_id="edge", logging_policy_version="v1")
    trace = LoggedTrace.from_records([row], registry=PolicyRegistry([policy]))
    assert trace.records[0].propensity == 0.05


def test_unregistered_logging_version_is_refused():
    row = record(logging_policy_version="v9")
    with pytest.raises(InvalidRequestError) as raised:
        LoggedTrace.from_records([row], registry=REGISTRY)
    assert raised.value.code == "logged_policy.trace.logging_policy_unregistered"
    assert raised.value.context["registered"] == (
        ("epsilon-greedy", "v3"),
        ("fixed-random", "v1"),
    )


def test_open_reward_fails_complete_horizon_admission():
    with pytest.raises(InvalidRequestError) as raised:
        LoggedTrace.from_records([record(reward=None)], registry=REGISTRY)
    assert raised.value.code == "logged_policy.trace.open_reward"
    context = raised.value.context
    assert context["unit_id"] == "u07"
    assert context["decision_index"] == 1
    assert context["reward_observation_boundary"] == T0 + HOUR


def test_unit_missing_a_decision_is_refused_for_the_declared_horizon():
    first = record(propensity=0.1)
    with pytest.raises(InvalidRequestError) as raised:
        LoggedTrace.from_records([first], registry=REGISTRY, horizon=2)
    assert raised.value.code == "logged_policy.trace.incomplete_horizon"
    assert raised.value.context["horizon"] == 2
    assert raised.value.context["observed_horizon"] == 1


def test_shorter_unit_is_refused_when_horizon_is_inferred_from_the_longest_unit():
    rows = [
        record(unit_id="a"),
        record(
            unit_id="a",
            decision_index=2,
            decision_time=T0 + 2 * HOUR,
            reward_observation_boundary=T0 + 3 * HOUR,
        ),
        record(unit_id="b"),
    ]
    with pytest.raises(InvalidRequestError) as raised:
        LoggedTrace.from_records(rows, registry=REGISTRY)
    assert raised.value.code == "logged_policy.trace.incomplete_horizon"
    assert raised.value.context == {"unit_id": "b", "observed_horizon": 1, "horizon": 2}


def test_gapped_or_time_reversed_indices_are_unordered():
    gapped = [
        record(),
        record(
            decision_index=3, decision_time=T0 + 2 * HOUR, reward_observation_boundary=T0 + 3 * HOUR
        ),
    ]
    with pytest.raises(InvalidRequestError) as raised:
        LoggedTrace.from_records(gapped, registry=REGISTRY)
    assert raised.value.code == "logged_policy.trace.unordered"
    assert raised.value.context["decision_indices"] == (1, 3)
    reversed_time = [
        record(),
        record(decision_index=2, decision_time=T0 - HOUR, reward_observation_boundary=T0),
    ]
    with pytest.raises(InvalidRequestError) as raised:
        LoggedTrace.from_records(reversed_time, registry=REGISTRY)
    assert raised.value.code == "logged_policy.trace.unordered"


def test_input_order_does_not_matter_but_the_admitted_trace_is_sorted():
    rows = [
        record(unit_id="b"),
        record(
            unit_id="a",
            decision_index=2,
            decision_time=T0 + 2 * HOUR,
            reward_observation_boundary=T0 + 3 * HOUR,
        ),
        record(unit_id="a"),
        record(
            unit_id="b",
            decision_index=2,
            decision_time=T0 + 2 * HOUR,
            reward_observation_boundary=T0 + 3 * HOUR,
        ),
    ]
    trace = LoggedTrace.from_records(rows, registry=REGISTRY)
    assert [(r.unit_id, r.decision_index) for r in trace.records] == [
        ("a", 1),
        ("a", 2),
        ("b", 1),
        ("b", 2),
    ]
    assert trace.horizon == 2 and trace.n_units == 2 and trace.unit_ids == ("a", "b")


def test_slash_colliding_policy_identities_are_distinct_logging_laws():
    import pyarrow as pa

    cases = (
        (("p1", "v1"), ("p2", "v1")),
        (("a/b", "c"), ("a", "b/c")),
    )
    for first, second in cases:
        policies = [
            TabularPolicy(
                policy_id=policy_id,
                version=version,
                default={"A": propensity, "B": 1 - propensity},
            )
            for (policy_id, version), propensity in zip((first, second), (0.5, 0.8), strict=True)
        ]
        rows = [
            record(
                unit_id=f"u{index}",
                logging_policy_id=policy_id,
                logging_policy_version=version,
                propensity=propensity,
            )
            for index, ((policy_id, version), propensity) in enumerate(
                zip((first, second), (0.5, 0.8), strict=True)
            )
        ]
        registry = PolicyRegistry(policies)
        for trace in (
            LoggedTrace.from_records(rows, registry=registry),
            LoggedTrace.from_frame(
                pa.Table.from_pylist(rows), registry=registry, context_columns=()
            ),
        ):
            with pytest.raises(CapabilityError) as raised:
                estimate_policy_contrast(trace, TARGET_POLICY_V1, REFERENCE_POLICY_V1)
            assert raised.value.code == "logged_policy.inference.adaptive_logging_unsupported"
            assert raised.value.context["logging_policy_keys"] == (first, second)
            assert raised.value.context["logging_policy_versions"] == tuple(
                f"{policy_id}/{version}" for policy_id, version in (first, second)
            )

    fixed = ("a/b", "c")
    trace = LoggedTrace.from_records(
        [record(logging_policy_id=fixed[0], logging_policy_version=fixed[1], propensity=0.5)],
        registry=PolicyRegistry(
            [TabularPolicy(policy_id=fixed[0], version=fixed[1], default={"A": 0.5, "B": 0.5})]
        ),
    )
    assert trace.logging_policy_versions == ("a/b/c",)


def test_candidate_set_must_be_fixed_and_contain_the_chosen_action():
    with pytest.raises(InvalidRequestError) as raised:
        DecisionRecord(**record(chosen_action="C"))
    assert raised.value.code == "logged_policy.trace.candidate_set_mismatch"
    assert raised.value.context["chosen_action"] == "C"
    rows = [
        record(unit_id="a"),
        record(unit_id="b", candidate_actions=("B", "A"), chosen_action="B", propensity=0.9),
    ]
    with pytest.raises(InvalidRequestError) as raised:
        LoggedTrace.from_records(rows, registry=REGISTRY)
    assert raised.value.code == "logged_policy.trace.candidate_set_mismatch"
    assert raised.value.context["expected_candidate_actions"] == ("A", "B")
    assert raised.value.context["candidate_actions"] == ("B", "A")


def test_boundary_before_decision_and_nonfinite_reward_are_refused():
    with pytest.raises(InvalidRequestError) as raised:
        DecisionRecord(**record(reward_observation_boundary=T0 - HOUR))
    assert raised.value.code == "logged_policy.trace.boundary_before_decision"
    with pytest.raises(InvalidRequestError) as raised:
        DecisionRecord(**record(reward=math.nan))
    assert raised.value.code == "logged_policy.trace.reward_nonfinite"


def test_empty_input_is_refused():
    with pytest.raises(InvalidRequestError) as raised:
        LoggedTrace.from_records([], registry=REGISTRY)
    assert raised.value.code == "logged_policy.trace.empty"


def test_direct_construction_cannot_supply_fabricated_registry_evidence():
    rec = DecisionRecord(**record(propensity=0.9))
    with pytest.raises(InvalidRequestError) as raised:
        LoggedTrace(records=(rec,), horizon=1, logging_distributions=((0.9, 0.1),))
    assert raised.value.code == "logged_policy.trace.registry_audit_required"


def test_foreign_validation_context_cannot_bypass_logging_distribution_admission():
    class EqualToAnyToken:
        def __eq__(self, other):
            return True

    with pytest.raises(InvalidRequestError) as raised:
        LoggedTrace.model_validate(
            {
                "records": (DecisionRecord(**record(propensity=0.9)),),
                "horizon": 1,
                "logging_distributions": ((0.9, 0.1),),
            },
            context={"audit": EqualToAnyToken()},
        )
    assert raised.value.code == "logged_policy.trace.registry_audit_required"


def test_from_frame_reads_context_columns_and_comma_separated_candidates():
    import polars as pl

    frame = pl.DataFrame(
        {
            "decision_time": [T0, T0 + 2 * HOUR],
            "unit_id": ["u07", "u07"],
            "decision_index": [1, 2],
            "candidate_actions": ["A,B", "A, B"],
            "chosen_action": ["A", "B"],
            "propensity": [0.1, 0.1],
            "logging_policy_id": ["epsilon-greedy", "epsilon-greedy"],
            "logging_policy_version": ["v3", "v3"],
            "update_batch": ["eg-b003", "eg-b003"],
            "reward_observation_boundary": [T0 + HOUR, T0 + 3 * HOUR],
            "reward": [0.2, 0.7],
            "x": [0, 1],
        }
    )
    trace = LoggedTrace.from_frame(frame, registry=REGISTRY)
    assert trace.horizon == 2
    assert trace.records[1].pre_decision_context == {"x": 1}
    assert trace.logging_distributions == ((0.1, 0.9), (0.9, 0.1))
    with pytest.raises(InvalidRequestError) as raised:
        LoggedTrace.from_frame(frame.drop("reward"), registry=REGISTRY)
    assert raised.value.code == "logged_policy.trace.missing_columns"
    assert raised.value.context == {"missing": ("reward",)}


def test_registry_refuses_duplicate_versions_and_tabular_policy_refuses_bad_tables():
    with pytest.raises(InvalidRequestError) as raised:
        PolicyRegistry([FIXED, FIXED])
    assert raised.value.code == "logged_policy.registry.duplicate"
    with pytest.raises(CapabilityError) as raised:
        TabularPolicy(policy_id="p", version="v1", probabilities={0: {"A": 0.6, "B": 0.6}})
    assert raised.value.code == "logged_policy.support.policy_not_normalized"
    with pytest.raises(CapabilityError) as raised:
        TARGET_POLICY_V1.probability("A", {"x": 2})
    assert raised.value.code == "logged_policy.support.context_unsupported"


def test_spec_policies_reproduce_the_declared_probabilities():
    assert TARGET_POLICY_V1.probability("B", {"x": 0}) == 0.8
    assert TARGET_POLICY_V1.probability("B", {"x": 1}) == 0.2
    assert TARGET_POLICY_V1.probability("A", {"x": 1}) == 0.8
    assert REFERENCE_POLICY_V1.probability("B", {"x": 0, "prior": 0.4}) == 0.5
    assert REFERENCE_POLICY_V1.probability("A", {}) == 0.5
    assert TARGET_POLICY_V1.probability("C", {"x": 0}) == 0.0


def test_policy_tables_and_record_context_are_recursively_immutable_copies():
    when = datetime(2026, 1, 1, tzinfo=UTC)
    probabilities: dict[datetime | None, dict[str, float]] = {
        when: {"A": 0.1, "B": 0.9},
        None: {"A": 0.3, "B": 0.7},
    }
    context: dict[str, Any] = {
        "x": when,
        "payload": b"immutable",
        "history": {"rewards": [0.2]},
    }
    policy = TabularPolicy(policy_id="copied", version="v1", probabilities=probabilities)
    row = record(
        propensity=0.1,
        logging_policy_id="copied",
        logging_policy_version="v1",
        pre_decision_context=context,
    )
    trace = LoggedTrace.from_records([row], registry=PolicyRegistry([policy]))

    probabilities[when]["A"] = 0.9
    context["history"]["rewards"].append(0.8)
    assert policy.probability("B", {"x": None}) == 0.7
    assert policy.probability("A", {"x": when}) == 0.1
    assert trace.records[0].pre_decision_context["x"] == when
    assert trace.records[0].pre_decision_context["payload"] == b"immutable"
    assert trace.records[0].pre_decision_context["history"]["rewards"] == (0.2,)
    with pytest.raises(TypeError):
        cast("MutableMapping[str, object]", policy.probabilities[when])["A"] = 0.9
    with pytest.raises(TypeError):
        cast("MutableMapping[str, object]", trace.records[0].pre_decision_context)["x"] = 1


def test_decision_record_context_dump_is_detached_and_roundtrips_in_python():
    context = {"x": 0, "history": {"rewards": [0.2]}, "tags": {"a", "b"}}
    rec = DecisionRecord(**record(pre_decision_context=context))

    payload = rec.model_dump(warnings="error")
    assert DecisionRecord.model_validate(payload) == rec

    dumped_context = cast("MutableMapping[str, object]", payload["pre_decision_context"])
    cast("MutableMapping[str, object]", dumped_context["history"])["rewards"] = ()
    context["history"]["rewards"].append(0.9)
    assert rec.pre_decision_context["history"]["rewards"] == (0.2,)
    with pytest.raises(TypeError):
        cast("MutableMapping[str, object]", rec.pre_decision_context)["history"] = {}


def test_decision_record_context_json_roundtrip_and_logged_trace_reconstruction():
    import json

    context = {"x": 0, "history": {"rewards": [0.2]}, "tags": ["a", "b"]}
    rec = DecisionRecord(**record(pre_decision_context=context))

    encoded_record = rec.model_dump_json(warnings="error")
    assert DecisionRecord.model_validate_json(encoded_record) == rec

    trace = LoggedTrace.from_records([rec], registry=REGISTRY)
    payload = json.loads(trace.model_dump_json(warnings="error"))
    decoded = LoggedTrace.from_records(
        payload["records"], registry=REGISTRY, horizon=payload["horizon"]
    )
    assert decoded == trace
    with pytest.raises(InvalidRequestError) as raised:
        LoggedTrace.model_validate_json(trace.model_dump_json(warnings="error"))
    assert raised.value.code == "logged_policy.trace.registry_audit_required"


def test_context_normalizes_numpy_scalars_and_rejects_unsupported_values():
    accepted = DecisionRecord(**record(pre_decision_context={"x": np.int64(7)}))
    assert accepted.pre_decision_context["x"] == 7
    assert type(accepted.pre_decision_context["x"]) is int

    for unsupported in (bytearray(b"x"), np.longdouble("1.25")):
        with pytest.raises(CodedError) as exc:
            DecisionRecord(**record(pre_decision_context={"unsupported": unsupported}))
        assert exc.value.code == "logged_policy.trace.context_value_unsupported"
        assert exc.value.context == {"value_type": type(unsupported).__name__}

    with pytest.raises(CodedError) as exc:
        TabularPolicy(
            policy_id="invalid",
            version="v1",
            probabilities={np.longdouble("1.25"): {"A": 0.5, "B": 0.5}},
        )
    assert exc.value.code == "logged_policy.trace.context_value_unsupported"
    assert exc.value.context == {"value_type": "longdouble"}


class _ReportingPolicy:
    """A policy outside the tabular family that reports whatever the test dictates."""

    policy_id = "reported"
    version = "v1"

    def __init__(self, reported: dict[str, Any]) -> None:
        self._reported = reported

    def probability(self, action: str, context: Any) -> Any:
        return self._reported[action]


@pytest.mark.parametrize(
    "reported",
    [
        {"A": 0.1, "B": math.nan},
        {"A": 0.1, "B": math.inf},
        {"A": -math.inf, "B": 0.1},
        {"A": math.inf, "B": -math.inf},
        {"A": 10**400, "B": 0.9},
        {"A": -0.1, "B": 1.1},
        {"A": 0.1, "B": 0.1},
        {"A": 0.1, "B": None},
        {"A": 0.1, "B": "0.9"},
    ],
)
def test_malformed_reconstructed_logging_distribution_is_refused(reported):
    row = record(logging_policy_id="reported", logging_policy_version="v1")
    with pytest.raises(CapabilityError) as raised:
        LoggedTrace.from_records([row], registry=PolicyRegistry([_ReportingPolicy(reported)]))
    assert raised.value.code == "logged_policy.support.policy_not_normalized"
    assert raised.value.context["policy_id"] == "reported"
    assert raised.value.context["version"] == "v1"
    assert raised.value.context["context"] == {"x": 0, "prior": None}
    probabilities = cast("Mapping[str, object]", raised.value.context["probabilities"])
    assert tuple(probabilities) == ("A", "B")
    assert probabilities["A"] == reported["A"]


@pytest.mark.parametrize(
    "table", [{"A": math.inf, "B": -math.inf}, {"A": math.nan, "B": 1.0}, {"A": math.inf, "B": 0.0}]
)
def test_tabular_policy_refuses_non_finite_tables_as_not_normalized(table):
    with pytest.raises(CapabilityError) as raised:
        TabularPolicy(policy_id="p", version="v1", probabilities={0: table})
    assert raised.value.code == "logged_policy.support.policy_not_normalized"
    assert raised.value.context["context"] == {"x": 0}
    with pytest.raises(CapabilityError) as raised:
        TabularPolicy(policy_id="p", version="v1", default=table)
    assert raised.value.code == "logged_policy.support.policy_not_normalized"
    assert raised.value.context["context"] == "default"


def test_candidate_set_that_omits_a_logged_action_is_refused():
    # epsilon-greedy/v3 plays B with 0.9 at x=0, so a candidate set of only A
    # cannot carry the full logging distribution.
    with pytest.raises(CapabilityError) as raised:
        LoggedTrace.from_records([record(candidate_actions=("A",))], registry=REGISTRY)
    assert raised.value.code == "logged_policy.support.policy_not_normalized"
    assert raised.value.context["probabilities"] == {"A": 0.1}


def test_candidate_never_played_by_the_logging_policy_is_admitted_with_probability_zero():
    row = record(
        candidate_actions=("A", "B", "C"),
        propensity=0.5,
        logging_policy_id="fixed-random",
        logging_policy_version="v1",
    )
    trace = LoggedTrace.from_records([row], registry=REGISTRY)
    assert trace.logging_distributions == ((0.5, 0.5, 0.0),)


@pytest.mark.parametrize("constructor", ["records", "frame", "replay"])
def test_logging_floor_covers_positive_unchosen_support(constructor):
    import json

    import pyarrow as pa

    logger = TabularPolicy(policy_id="biased", version="v1", default={"A": 0.99, "B": 0.01})
    row = record(
        propensity=0.99,
        logging_policy_id="biased",
        logging_policy_version="v1",
    )
    registry = PolicyRegistry([logger])
    with pytest.raises(InvalidRequestError) as raised:
        if constructor == "frame":
            frame = pa.Table.from_pylist([{**row, "x": 0, "prior": None}])
            LoggedTrace.from_frame(frame, registry=registry, context_columns=("x", "prior"))
        else:
            rows = json.loads(json.dumps([row], default=str)) if constructor == "replay" else [row]
            LoggedTrace.from_records(rows, registry=registry)
    assert raised.value.code == "logged_policy.trace.propensity_below_floor"
    assert raised.value.context["action"] == "B"
    assert raised.value.context["propensity"] == 0.01
    assert raised.value.context["logging_policy_id"] == "biased"
    assert raised.value.context["logging_policy_version"] == "v1"
    assert raised.value.context["context"] == {"x": 0, "prior": None}


@pytest.mark.parametrize("probabilities", [{"A": 0.95, "B": 0.05}, {"A": 0.5, "B": 0.5, "C": 0.0}])
def test_logging_floor_admits_boundary_and_irrelevant_zero_support(probabilities):
    import pyarrow as pa

    logger = TabularPolicy(policy_id="fixed", version="v1", default=probabilities)
    rows = [
        record(
            unit_id=f"u{i}",
            candidate_actions=tuple(probabilities),
            propensity=probabilities["A"],
            logging_policy_id="fixed",
            logging_policy_version="v1",
            reward=float(i),
        )
        for i in range(40)
    ]
    registry = PolicyRegistry([logger])
    for trace in (
        LoggedTrace.from_records(rows, registry=registry),
        LoggedTrace.from_frame(pa.Table.from_pylist(rows), registry=registry, context_columns=()),
    ):
        result = estimate_policy_contrast(trace, logger, logger)
        assert (result.estimate, result.lb, result.ub) == (0.0, 0.0, 0.0)
        if "C" in probabilities:
            unsupported = TabularPolicy(
                policy_id="unsupported", version="v1", default={"A": 0.45, "B": 0.45, "C": 0.1}
            )
            with pytest.raises(CapabilityError) as raised:
                estimate_policy_contrast(trace, unsupported, logger)
            assert raised.value.code == "logged_policy.support.action_unsupported"


def test_default_only_policy_ignores_an_unhashable_supported_context_value():
    nested = {"x": {"history": [0.2]}}
    assert REFERENCE_POLICY_V1.probability("B", nested) == 0.5
    row = record(
        logging_policy_id="reference-policy",
        logging_policy_version="v1",
        propensity=0.5,
        pre_decision_context=nested,
    )
    trace = LoggedTrace.from_records([row], registry=PolicyRegistry([REFERENCE_POLICY_V1]))
    assert trace.logging_distributions == ((0.5, 0.5),)
    # A tabulated policy without a default has nothing for that value.
    with pytest.raises(CapabilityError) as raised:
        TARGET_POLICY_V1.probability("B", nested)
    assert raised.value.code == "logged_policy.support.context_unsupported"
    # Values outside the context domain still refuse, default or not.
    with pytest.raises(InvalidRequestError) as raised:
        REFERENCE_POLICY_V1.probability("B", {"x": bytearray(b"x")})
    assert raised.value.code == "logged_policy.trace.context_value_unsupported"


def _two_row_columns(unit_ids: list[Any]) -> dict[str, list[Any]]:
    return {
        "decision_time": [T0, T0],
        "unit_id": unit_ids,
        "decision_index": [1, 1],
        "candidate_actions": ["A,B", "A,B"],
        "chosen_action": ["A", "A"],
        "propensity": [0.1, 0.1],
        "logging_policy_id": ["epsilon-greedy", "epsilon-greedy"],
        "logging_policy_version": ["v3", "v3"],
        "update_batch": ["eg-b003", "eg-b003"],
        "reward_observation_boundary": [T0 + HOUR, T0 + HOUR],
        "reward": [0.2, 0.2],
        "x": [0, 0],
    }


def test_from_frame_refuses_distinct_native_ids_that_share_a_canonical_string():
    pd = pytest.importorskip("pandas")

    columns = _two_row_columns([1, "1"])
    columns["decision_index"] = [1, 2]
    columns["decision_time"][1] = T0 + 2 * HOUR
    columns["reward_observation_boundary"][1] = T0 + 3 * HOUR
    frame = pd.DataFrame(columns, dtype=object)
    with pytest.raises(InvalidRequestError) as raised:
        LoggedTrace.from_frame(frame, registry=REGISTRY)
    assert raised.value.code == "estimation.crossfit.identity_collision"
    assert raised.value.context == {
        "what": "trace frame unit_id",
        "canonical": "1",
        "first": "1",
        "second": "'1'",
    }
    with pytest.raises(TypeError):
        cast("MutableMapping[str, object]", raised.value.context)["canonical"] = "other"


@pytest.mark.parametrize("numeric_ids", [False, True])
def test_from_frame_preserves_native_id_contrasts(numeric_ids):
    import polars as pl

    rows = [
        record(
            unit_id=i if numeric_ids else str(i),
            chosen_action="A" if i % 3 == 0 else "B",
            propensity=0.5,
            logging_policy_id="fixed-random",
            logging_policy_version="v1",
            pre_decision_context={"x": i % 2},
            reward=(i % 5) / 5,
        )
        for i in range(128)
    ]
    frame_rows = [
        {**{k: v for k, v in row.items() if k != "pre_decision_context"}, "x": i % 2}
        for i, row in enumerate(rows)
    ]
    expected = LoggedTrace.from_records(
        [{**row, "unit_id": str(row["unit_id"])} for row in rows], registry=REGISTRY
    )
    actual = LoggedTrace.from_frame(pl.DataFrame(frame_rows), registry=REGISTRY)
    assert actual == expected
    assert estimate_policy_contrast(
        actual, TARGET_POLICY_V1, REFERENCE_POLICY_V1
    ) == estimate_policy_contrast(expected, TARGET_POLICY_V1, REFERENCE_POLICY_V1)


def test_from_frame_keeps_equal_numpy_and_python_id_representations_together():
    pd = pytest.importorskip("pandas")

    columns = _two_row_columns([np.int64(7), 7])
    columns["decision_index"] = [1, 2]
    columns["decision_time"][1] = T0 + 2 * HOUR
    columns["reward_observation_boundary"][1] = T0 + 3 * HOUR
    trace = LoggedTrace.from_frame(pd.DataFrame(columns, dtype=object), registry=REGISTRY)
    assert trace.unit_ids == ("7",)
    assert trace.horizon == 2


@pytest.mark.parametrize(
    ("unit_ids", "row"), [(["u07", None], 1), ([7.0, math.nan], 1), ([7.0, math.inf], 1)]
)
def test_from_frame_refuses_null_and_nonfinite_unit_ids(unit_ids, row):
    import polars as pl

    with pytest.raises(InvalidRequestError) as raised:
        LoggedTrace.from_frame(pl.DataFrame(_two_row_columns(unit_ids)), registry=REGISTRY)
    assert raised.value.code == "logged_policy.trace.unit_id_missing"
    assert raised.value.context["row"] == row
    missing = raised.value.context["unit_id"]
    assert missing is None or (isinstance(missing, float) and not math.isfinite(missing))


def test_from_frame_refuses_arrow_null_unit_ids():
    import pyarrow as pa

    with pytest.raises(InvalidRequestError) as raised:
        LoggedTrace.from_frame(pa.table(_two_row_columns(["u07", None])), registry=REGISTRY)
    assert raised.value.code == "logged_policy.trace.unit_id_missing"
    assert raised.value.context["row"] == 1
    assert raised.value.context["unit_id"] is None


def test_from_frame_refuses_nullable_pandas_unit_ids():
    pd = pytest.importorskip("pandas")

    frame = pd.DataFrame(_two_row_columns(["u07", "u08"]))
    frame["unit_id"] = pd.array(["u07", pd.NA], dtype="string")
    with pytest.raises(InvalidRequestError) as raised:
        LoggedTrace.from_frame(frame, registry=REGISTRY)
    assert raised.value.code == "logged_policy.trace.unit_id_missing"
    assert raised.value.context["row"] == 1
    assert raised.value.context["unit_id"] is pd.NA


def test_mixed_naive_and_aware_timestamps_are_refused():
    naive = T0.replace(tzinfo=None)
    with pytest.raises(InvalidRequestError) as raised:
        DecisionRecord(**record(reward_observation_boundary=naive + HOUR))
    assert raised.value.code == "logged_policy.trace.boundary_awareness_mixed"
    assert raised.value.context["unit_id"] == "u07"
    assert raised.value.context["decision_index"] == 1
    across_units = [
        record(unit_id="a"),
        record(unit_id="b", decision_time=naive, reward_observation_boundary=naive + HOUR),
    ]
    with pytest.raises(InvalidRequestError) as raised:
        LoggedTrace.from_records(across_units, registry=REGISTRY)
    assert raised.value.code == "logged_policy.trace.decision_time_awareness_mixed"
    context = raised.value.context
    assert context["unit_id"] == "b"
    assert context["decision_index"] == 1
    assert context["reference_unit_id"] == "a"
    within_unit = [
        record(),
        record(
            decision_index=2,
            decision_time=naive + 2 * HOUR,
            reward_observation_boundary=naive + 3 * HOUR,
        ),
    ]
    with pytest.raises(InvalidRequestError) as raised:
        LoggedTrace.from_records(within_unit, registry=REGISTRY)
    assert raised.value.code == "logged_policy.trace.decision_time_awareness_mixed"


def test_uniformly_naive_timestamps_are_admitted_and_ordered():
    naive = T0.replace(tzinfo=None)
    rows = [
        record(decision_time=naive, reward_observation_boundary=naive + HOUR),
        record(
            unit_id="u08",
            decision_time=naive + HOUR,
            reward_observation_boundary=naive + 2 * HOUR,
            logging_policy_id="fixed-random",
            logging_policy_version="v1",
            propensity=0.5,
        ),
    ]
    trace = LoggedTrace.from_records(rows, registry=REGISTRY)
    assert trace.horizon == 1
    assert trace.logging_policy_versions == ("epsilon-greedy/v3", "fixed-random/v1")
    assert trace.update_batches == ("eg-b003",)


def test_recorded_route_pairs_full_laws_before_sorting_and_snapshots_inputs():
    rows = [
        record(unit_id="b", candidate_actions=("A", "B", "C"), chosen_action="A", propensity=0.2),
        record(
            unit_id="a",
            candidate_actions=("A", "B", "C"),
            chosen_action="A",
            propensity=0.2,
            pre_decision_context={"x": 1, "history": [{"score": 3.0}]},
        ),
    ]
    laws = [{"A": 0.2, "B": 0.3, "C": 0.5}, {"A": 0.2, "B": 0.7, "C": 0.1}]
    trace = LoggedTrace.from_records(rows, logging_distributions=laws)
    laws[0]["A"] = 0.99
    rows[0]["unit_id"] = "mutated"
    assert trace.unit_ids == ("a", "b")
    assert trace.logging_distributions == ((0.2, 0.7, 0.1), (0.2, 0.3, 0.5))
    payload = json.loads(trace.model_dump_json(warnings="error"))
    read_back = LoggedTrace.from_records(
        payload["records"],
        logging_distributions=[
            dict(zip(row["candidate_actions"], law, strict=True))
            for row, law in zip(payload["records"], payload["logging_distributions"], strict=True)
        ],
        horizon=payload["horizon"],
    )
    for restored in (read_back, pickle.loads(pickle.dumps(trace)), copy.deepcopy(trace)):
        assert restored == trace
        with pytest.raises(TypeError):
            restored.records[0].pre_decision_context["history"][0]["score"] = 4.0
        with pytest.raises(TypeError):
            cast(Any, restored.logging_distributions[0])[0] = 0.99


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({}, "logged_policy.trace.admission_route"),
        (
            {"registry": REGISTRY, "logging_distributions": [{"A": 0.5, "B": 0.5}]},
            "logged_policy.trace.admission_route",
        ),
        ({"logging_distributions": []}, "logged_policy.trace.logging_distribution_misaligned"),
        (
            {"logging_distributions": [{"A": 0.5, "B": 0.5, "C": 0.0}]},
            "logged_policy.trace.logging_distribution_keys",
        ),
        (
            {"logging_distributions": [{"A": 0.4, "B": 0.4}]},
            "logged_policy.support.policy_not_normalized",
        ),
    ],
)
def test_recorded_route_rejects_wrong_route_length_keys_and_probabilities(kwargs, code):
    with pytest.raises((InvalidRequestError, CapabilityError)) as raised:
        LoggedTrace.from_records([record()], **kwargs)
    assert raised.value.code == code


def test_recorded_route_rejects_floor_and_chosen_propensity_mismatch():
    with pytest.raises(InvalidRequestError) as raised:
        LoggedTrace.from_records(
            [record(propensity=0.5)], logging_distributions=[{"A": 0.01, "B": 0.99}]
        )
    assert raised.value.code == "logged_policy.trace.propensity_below_floor"
    with pytest.raises(InvalidRequestError) as raised:
        LoggedTrace.from_records(
            [record(propensity=0.6)], logging_distributions=[{"A": 0.5, "B": 0.5}]
        )
    assert raised.value.code == "logged_policy.trace.propensity_stale"


def test_recorded_frame_route_pairs_distribution_column_before_sorting():
    import polars as pl

    first = record(unit_id="b", candidate_actions=("A", "B", "C"), propensity=0.2)
    second = record(unit_id="a", candidate_actions=("A", "B", "C"), propensity=0.2)
    first["logging_distribution"] = {"A": 0.2, "B": 0.3, "C": 0.5}
    second["logging_distribution"] = {"A": 0.2, "B": 0.7, "C": 0.1}
    for row in (first, second):
        row["x"] = row.pop("pre_decision_context")["x"]
    trace = LoggedTrace.from_frame(
        pl.DataFrame([first, second]),
        logging_distribution_column="logging_distribution",
    )
    assert trace.unit_ids == ("a", "b")
    assert trace.logging_distributions == ((0.2, 0.7, 0.1), (0.2, 0.3, 0.5))


@pytest.mark.parametrize("both", [False, True])
def test_frame_admission_route_error_names_the_frame_inputs(both):
    import polars as pl

    kwargs = {"registry": REGISTRY, "logging_distribution_column": "law"} if both else {}
    with pytest.raises(InvalidRequestError) as raised:
        LoggedTrace.from_frame(pl.DataFrame(), **kwargs)
    assert raised.value.code == "logged_policy.trace.admission_route"
    assert raised.value.context["alternatives"] == ("registry", "logging_distribution_column")
    assert raised.value.context["supplied"] == (
        ("registry", "logging_distribution_column") if both else ()
    )


def test_tabular_policy_json_roundtrip_preserves_action_law():
    policy = TabularPolicy(
        policy_id="saved",
        version="v1",
        context_key="country",
        probabilities={"US": {"A": 0.2, "B": 0.8}},
        default={"A": 0.6, "B": 0.4},
    )
    payload = policy.model_dump(mode="json", warnings="error")
    restored = TabularPolicy.model_validate_json(policy.model_dump_json(warnings="error"))
    assert json.loads(policy.model_dump_json(warnings="error")) == payload
    for context, expected in (({"country": "US"}, 0.8), ({"country": "CA"}, 0.4), ({}, 0.4)):
        assert restored.probability("B", context) == expected
    payload["probabilities"]["US"]["B"] = 0.0
    assert policy.probability("B", {"country": "US"}) == 0.8


@pytest.mark.parametrize(
    ("kwargs", "context", "expected"),
    [
        ({"default": {"A": 0.5, "B": 0.5}}, {"x": 9}, 0.5),
        ({"probabilities": {"a": {"A": 0.3, "B": 0.7}}}, {"x": "a"}, 0.7),
    ],
    ids=["default-only", "absent-default"],
)
def test_tabular_policy_json_roundtrip_keeps_default_presence(kwargs, context, expected):
    policy = TabularPolicy(policy_id="p", version="v1", **kwargs)
    restored = TabularPolicy.model_validate_json(policy.model_dump_json(warnings="error"))
    assert restored.probability("B", context) == expected
    assert (restored.default is None) == (policy.default is None)
    if policy.default is None:
        with pytest.raises(CapabilityError) as raised:
            restored.probability("B", {"x": "other"})
        assert raised.value.code == "logged_policy.support.context_unsupported"


def _typed_key_policy() -> TabularPolicy:
    when = datetime(2026, 1, 2, tzinfo=UTC)
    return TabularPolicy(
        policy_id="typed",
        version="v1",
        probabilities={
            1: {"A": 0.1, "B": 0.9},
            "1": {"A": 0.2, "B": 0.8},
            ("a", 2): {"A": 0.3, "B": 0.7},
            when: {"A": 0.4, "B": 0.6},
        },
        default={"A": 0.5, "B": 0.5},
    )


def _typed_key_law(policy: TabularPolicy) -> list[float]:
    when = datetime(2026, 1, 2, tzinfo=UTC)
    return [policy.probability("B", {"x": key}) for key in (1, "1", ("a", 2), when, "unseen", 2)]


def test_tabular_policy_python_persistence_keeps_typed_keys_and_isolation():
    policy = _typed_key_policy()
    expected = [0.9, 0.8, 0.7, 0.6, 0.5, 0.5]
    assert _typed_key_law(policy) == expected

    dumped = policy.model_dump(mode="python", warnings="error")
    assert {type(key) for key in dumped["probabilities"]} >= {int, str, tuple, datetime}
    restored = TabularPolicy.model_validate(dumped)
    dumped["probabilities"][1]["B"] = 0.0
    dumped["default"]["B"] = 0.0
    assert _typed_key_law(restored) == expected
    assert _typed_key_law(policy) == expected

    assert _typed_key_law(copy.deepcopy(policy)) == expected
    assert _typed_key_law(pickle.loads(pickle.dumps(policy))) == expected
    with pytest.raises(TypeError):
        cast("MutableMapping[str, float]", policy.distribution({"x": 1}))["B"] = 0.0


def test_tabular_policy_json_refuses_non_string_keys_directly():
    for policy in (_typed_key_policy(), TARGET_POLICY_V1):
        for dump in (
            lambda p: p.model_dump(mode="json"),
            lambda p: p.model_dump_json(),
        ):
            with pytest.raises(CapabilityError) as raised:
                dump(policy)
            assert raised.value.code == "logged_policy.policy.json_context_key"
            assert raised.value.context["policy_id"] == policy.policy_id
            key_types = cast("tuple[str, ...]", raised.value.context["key_types"])
            assert key_types == tuple(sorted(key_types))
            assert "str" not in key_types
    error = raised.value
    assert pickle.loads(pickle.dumps(error)).code == error.code
    assert copy.deepcopy(error).context["key_types"] == error.context["key_types"]
    # Python persistence is unaffected, and construction never applies the JSON rule.
    assert TARGET_POLICY_V1.model_dump(mode="python")["probabilities"][0] == {"A": 0.2, "B": 0.8}


def test_tabular_policy_constructor_refusal_precedes_json_key_refusal():
    with pytest.raises(CapabilityError) as raised:
        TabularPolicy(policy_id="p", version="v1", probabilities={1: {"A": 0.6, "B": 0.6}})
    assert raised.value.code == "logged_policy.support.policy_not_normalized"
