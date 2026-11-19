"""Public v1 allocation records, compiled snapshots and refusals."""

import json
import pickle
from copy import deepcopy
from pathlib import Path

import pytest

from increment.assignment import AllocationEvaluator, Decision, Membership, compile_allocation
from increment.errors import InvalidRequestError


@pytest.fixture
def vectors():
    path = Path(__file__).resolve().parents[1] / "docs/reference/assignment-v1-vectors.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _decide(evaluator: AllocationEvaluator, case) -> Decision:
    return evaluator.decide(case["target"], case["unit_id"])


def _run_case(config: object, case) -> Decision | Membership | None:
    evaluator = compile_allocation(config)
    match case["op"]:
        case "decide":
            return _decide(evaluator, case)
        case "membership":
            return evaluator.holdout_membership(case["target"], case["unit_id"])
        case "compile":
            return None
        case other:
            raise AssertionError(f"Unknown vector operation: {other}")


@pytest.mark.integration
def test_allocation_matches_published_vectors(vectors) -> None:
    for case in vectors["cases"]:
        config = vectors["configs"][case["config"]]
        if "error" in case:
            with pytest.raises(InvalidRequestError) as caught:
                _run_case(config, case)
            assert caught.value.code == case["error"], case["id"]
        else:
            assert _run_case(config, case) == case["expected"], case["id"]


@pytest.mark.integration
def test_compiled_allocation_is_detached_from_inputs_and_outputs(vectors) -> None:
    cases = {case["id"]: case for case in vectors["cases"]}
    original = cases["range-start-inclusive"]
    original_search = cases["eligible-miss-other-layer"]
    config = deepcopy(vectors["configs"]["program"])
    edited = vectors["configs"]["mutated-program"]
    evaluator = compile_allocation(config)
    returned = _decide(evaluator, original)
    assert returned == original["expected"]
    nested_membership = returned["membership"]
    assert nested_membership is not None
    standalone_membership = evaluator.holdout_membership("shopping", original["unit_id"])
    assert standalone_membership == nested_membership
    standalone_membership["group_bps"] = 0
    assert nested_membership == original["expected"]["membership"]
    nested_membership["revision"] = "edited-output"
    returned["arm"] = None

    config["revision"] = edited["revision"]
    for arm, edited_arm in zip(
        config["experiments"][0]["arms"], edited["experiments"][0]["arms"], strict=True
    ):
        arm["weight_bps"] = edited_arm["weight_bps"]
    config["layers"][0]["ranges"][0].update(edited["layers"][0]["ranges"][0])
    holdout = config["holdouts"][0]
    holdout["held_out_bps"] = edited["holdouts"][0]["held_out_bps"]
    holdout["targets"][:] = edited["holdouts"][0]["targets"]
    assert config == edited

    assert _decide(evaluator, original) == original["expected"]
    assert _decide(evaluator, original_search) == original_search["expected"]
    assert (
        evaluator.holdout_membership("shopping", original["unit_id"])
        == original["expected"]["membership"]
    )
    fresh = compile_allocation(config)
    for case_id in ("fresh-compile-of-edited-program", "fresh-scope-removes-membership"):
        case = cases[case_id]
        assert _decide(fresh, case) == case["expected"], case_id


def test_nonfinite_basis_points_are_refused() -> None:
    for nonfinite in (float("nan"), float("inf"), -float("inf")):
        config = {
            "protocol": 1,
            "revision": "r1",
            "experiments": [
                {
                    "key": "cart",
                    "enrollment_bps": nonfinite,
                    "arms": [
                        {"key": "control", "weight_bps": 5000},
                        {"key": "treatment", "weight_bps": 5000},
                    ],
                }
            ],
            "layers": [],
        }
        with pytest.raises(InvalidRequestError) as caught:
            compile_allocation(config)
        assert caught.value.code == "allocation.invalid_config"
        assert caught.value.context["field"] == "experiments[0].enrollment_bps"


def test_invalid_unit_context_is_private_immutable_and_survives_transport() -> None:
    evaluator = compile_allocation(
        {
            "protocol": 1,
            "revision": "r1",
            "experiments": [],
            "layers": [{"key": "checkout", "ranges": []}],
            "holdouts": [{"key": "shopping", "held_out_bps": 1000, "targets": ["checkout"]}],
        }
    )
    for operation, key in (
        (evaluator.decide, "checkout"),
        (evaluator.holdout_membership, "shopping"),
    ):
        with pytest.raises(InvalidRequestError) as caught:
            operation(key, "private-unit-\ud800")
        error = caught.value
        serialized = pickle.dumps(error)
        assert b"private-unit-" not in serialized
        for restored in (error, deepcopy(error), pickle.loads(serialized)):
            assert isinstance(restored, InvalidRequestError)
            assert restored.code == "allocation.invalid_unit"
            assert restored.context == {
                "field": "unit_id",
                "reason": "must be a nonempty string of Unicode scalar values",
            }
            assert "private-unit-" not in str(restored)
            assert restored.__cause__ is None and restored.__context__ is None
            with pytest.raises(TypeError):
                restored.context["field"] = "other"  # ty: ignore[invalid-assignment]
