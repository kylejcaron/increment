"""Bounded rational admission on the sequential wire.

A checkpoint, registration or compiled plan spells every exact rational as
``Fraction`` prints it, and JSON carries one only as that string or as an
integer literal. Reading one back must cost work proportional to the
encoding, so exponents, oversized components, zero denominators, booleans
and JSON floating literals refuse with one structured code before any
arithmetic, while supported captures replay unchanged.
"""

from __future__ import annotations

import copy
import json
import pickle
import sys
from fractions import Fraction
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from increment import Analysis
from increment.decision_wire import (
    WireCompiledDecisionPlan,
    compiled_plan_from_dict,
    compiled_plan_from_json,
)
from increment.errors import CodedError, CodedModel, InvalidRequestError
from increment.semantics.models import AnalysisPlan, InferenceSpec
from increment.semantics.rational import PORTABLE_RATIONAL_DIGITS, PortableRational
from increment.semantics.sequential import (
    PredeclaredAdjustment,
    PredictivePrior,
    ScalarMeanModel,
    SequentialCell,
    SequentialCompliancePolicy,
    SequentialRegistration,
)
from increment.sequential_state import SequentialArmState, SequentialSnapshot, snapshot_from_json
from tests.sequential_cases import registration

CODE = "sequential.wire.rational_invalid"
LIMIT = PORTABLE_RATIONAL_DIGITS
LONG = "9" * (LIMIT + 1)
HOSTILE = ["1e50000", "-1e-50000", LONG, "1/" + LONG, "1/0"]
_SCALAR_MODEL = {
    "metric": "m",
    "law": "scalar_mean",
    "rho": "1/10",
    "start_count": 2,
    "assignment": "iid_fixed_bernoulli_randomization",
    "treatment_probability": "1/2",
    "consistency_and_no_interference": True,
    "segment_membership": "pre_assignment",
    "unit_model": "iid_stationary_potential_outcomes",
    "moments": "finite_2_plus_delta",
    "positive_limiting_variance": True,
    "positive_population_control": True,
}
_ARM = {"metric": "m", "group_id": "c", "law": "scalar_mean", "n": 2}


def _hostile_snapshot(value: str) -> dict:
    return {
        "version": 2,
        "registration": {},
        "registration_id": "x",
        "prefix_id": "x",
        "records": [],
        "finalized": True,
        "states": [{**_ARM, "mean": [value], "scatter": [["0"]]}],
    }


def _hostile_row(value: str) -> dict:
    return {
        "moments_format": 9,
        "record_kind": "sequential_checkpoint",
        "experiment_id": "untrusted",
        "decision_plan": "{}",
        "sequential_snapshot": json.dumps(_hostile_snapshot(value)),
    }


def _assert_refusal(exc: CodedError, field: str, hostile: str | None = None) -> None:
    assert isinstance(exc, InvalidRequestError)
    assert exc.code == CODE
    assert set(exc.context) == {"field", "reason", "limit_digits", "route_forward"}
    assert exc.context["field"] == field
    assert exc.context["limit_digits"] == LIMIT
    assert isinstance(exc.context["route_forward"], str) and exc.context["route_forward"]
    reason = exc.context["reason"]
    assert isinstance(reason, str) and reason
    # The hostile encoding is described, never copied, so the refusal stays small.
    if hostile is not None:
        assert hostile not in str(exc) and hostile not in reason


def test_hostile_checkpoint_row_refuses_before_rational_expansion():
    with pytest.raises(InvalidRequestError) as exc:
        Analysis.from_moments([_hostile_row("1e50000")], metrics={"m": "mean"}, control="c")
    _assert_refusal(exc.value, "SequentialArmState.mean", "1e50000")


@pytest.mark.parametrize("hostile", HOSTILE)
def test_snapshot_json_refuses_hostile_state_mean(hostile):
    with pytest.raises(InvalidRequestError) as exc:
        snapshot_from_json(json.dumps(_hostile_snapshot(hostile)))
    _assert_refusal(exc.value, "SequentialArmState.mean", hostile)


def test_refusal_survives_pickle_and_deepcopy():
    with pytest.raises(InvalidRequestError) as exc:
        snapshot_from_json(json.dumps(_hostile_snapshot("1e50000")))
    original = exc.value
    for clone in (pickle.loads(pickle.dumps(original)), copy.deepcopy(original)):
        assert type(clone) is InvalidRequestError
        assert clone.code == CODE
        assert dict(clone.context) == dict(original.context)
        assert str(clone) == str(original)


def _registration_payload(**overrides) -> dict:
    return {**registration("bernoulli").model_dump(), **overrides}


_FIELDS = [
    pytest.param(
        SequentialArmState,
        lambda h: {**_ARM, "mean": [h], "scatter": [["0"]]},
        "SequentialArmState.mean",
        id="state-mean",
    ),
    pytest.param(
        SequentialArmState,
        lambda h: {**_ARM, "mean": ["0"], "scatter": [[h]]},
        "SequentialArmState.scatter",
        id="state-scatter",
    ),
    pytest.param(
        PredictivePrior,
        lambda h: {"kind": "beta", "a": h, "b": "1"},
        "PredictivePrior.a",
        id="prior-beta-a",
    ),
    pytest.param(
        PredictivePrior,
        lambda h: {"kind": "nig", "kappa": "1", "nu": "2", "mean": ["0"], "scale": [[h]]},
        "PredictivePrior.scale",
        id="prior-nig-scale",
    ),
    pytest.param(
        PredictivePrior,
        lambda h: {"kind": "nig", "kappa": h, "nu": "2", "mean": ["0"], "scale": [["2"]]},
        "PredictivePrior.kappa",
        id="prior-nig-kappa",
    ),
    pytest.param(
        PredeclaredAdjustment,
        lambda h: {"coefficient": h, "center": "0"},
        "PredeclaredAdjustment.coefficient",
        id="adjustment-coefficient",
    ),
    pytest.param(
        PredeclaredAdjustment,
        lambda h: {"coefficient": "1/2", "center": h},
        "PredeclaredAdjustment.center",
        id="adjustment-center",
    ),
    pytest.param(
        ScalarMeanModel, lambda h: {**_SCALAR_MODEL, "rho": h}, "ScalarMeanModel.rho", id="rho"
    ),
    pytest.param(
        ScalarMeanModel,
        lambda h: {**_SCALAR_MODEL, "treatment_probability": h},
        "ScalarMeanModel.treatment_probability",
        id="treatment-probability",
    ),
    pytest.param(
        SequentialCompliancePolicy,
        lambda h: {"alpha": h},
        "SequentialCompliancePolicy.alpha",
        id="compliance-alpha",
    ),
    pytest.param(
        SequentialCompliancePolicy,
        lambda h: {"alpha": "1/20", "null_lift": h},
        "SequentialCompliancePolicy.null_lift",
        id="compliance-null-lift",
    ),
    pytest.param(
        SequentialCell,
        lambda h: {"metric": "m", "group_id": "t", "alpha": h},
        "SequentialCell.alpha",
        id="cell-alpha",
    ),
    pytest.param(
        SequentialCell,
        lambda h: {"metric": "m", "group_id": "t", "null_lift": h},
        "SequentialCell.null_lift",
        id="cell-null-lift",
    ),
    pytest.param(
        SequentialRegistration,
        lambda h: _registration_payload(q=h),
        "SequentialRegistration.q",
        id="registration-q",
    ),
    pytest.param(
        InferenceSpec,
        lambda h: {"kind": "always_valid", "baseline_rate": h},
        "InferenceSpec.baseline_rate",
        id="inference-baseline-rate",
    ),
    pytest.param(
        AnalysisPlan,
        lambda h: {
            "primary": "m",
            "inference": {
                "kind": "asymptotic_mean",
                "adjustments": {"m": {"coefficient": "1", "center": h}},
            },
        },
        "PredeclaredAdjustment.center",
        id="plan-nested-adjustment",
    ),
    pytest.param(
        SequentialSnapshot,
        lambda h: {**_hostile_snapshot("0"), "registration": _registration_payload(q=h)},
        "SequentialRegistration.q",
        id="snapshot-nested-registration",
    ),
]


@pytest.mark.parametrize("model,payload,field", _FIELDS)
@pytest.mark.parametrize("hostile", HOSTILE)
def test_every_serialized_rational_field_refuses_hostile_encodings(model, payload, field, hostile):
    with pytest.raises(InvalidRequestError) as via_json:
        model.model_validate_json(json.dumps(payload(hostile)))
    _assert_refusal(via_json.value, field, hostile)
    with pytest.raises(InvalidRequestError) as via_python:
        model.model_validate(payload(hostile))
    assert via_python.value.code == CODE
    assert dict(via_python.value.context) == dict(via_json.value.context)


def _golden_plan() -> dict:
    root = Path(__file__).parent / "goldens" / "decision_wire"
    return json.loads((root / "asymptotic_mean_mixed_family.json").read_text())


_PLAN_RATIONALS = [
    (lambda p, h: p["compliance"].update(alpha=h), "SequentialCompliancePolicy.alpha"),
    (lambda p, h: p["inference"]["registration"].update(q=h), "SequentialRegistration.q"),
    (
        lambda p, h: p["inference"]["registration"]["models"][0].update(rho=h),
        "ScalarMeanModel.rho",
    ),
    (
        lambda p, h: p["inference"]["registration"]["roster"][0].update(alpha=h),
        "SequentialCell.alpha",
    ),
]


@pytest.mark.parametrize("mutate,field", _PLAN_RATIONALS)
def test_compiled_plan_wire_refuses_nested_hostile_rationals(mutate, field):
    payload = _golden_plan()
    mutate(payload, "1e50000")
    with pytest.raises(InvalidRequestError) as exc:
        WireCompiledDecisionPlan.model_validate(payload)
    _assert_refusal(exc.value, field, "1e50000")
    with pytest.raises(InvalidRequestError) as decoded:
        compiled_plan_from_json(json.dumps(payload))
    _assert_refusal(decoded.value, field, "1e50000")


class _Probe(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    value: PortableRational


def _probe(encoding: object) -> Fraction:
    return _Probe.model_validate_json(json.dumps({"value": encoding})).value


@pytest.mark.parametrize(
    "encoding,expected",
    [
        ("1/20", Fraction(1, 20)),
        ("-3/4", Fraction(-3, 4)),
        ("+5", Fraction(5)),
        ("-0", Fraction(0)),
        ("0.05", Fraction(1, 20)),
        ("-.5", Fraction(-1, 2)),
        ("5.", Fraction(5)),
        ("2/4", Fraction(1, 2)),
        (7, Fraction(7)),
        (-12, Fraction(-12)),
        ("9" * LIMIT, Fraction(int("9" * LIMIT))),
        ("1/" + "9" * LIMIT, Fraction(1, int("9" * LIMIT))),
        ("-" + "9" * LIMIT + "/" + "7" * LIMIT, Fraction(-int("9" * LIMIT), int("7" * LIMIT))),
    ],
)
def test_bounded_spellings_are_admitted_exactly(encoding, expected):
    value = _probe(encoding)
    assert value == expected and type(value) is Fraction
    # Canonical spelling survives a JSON round trip; a reduced quotient reads back reduced.
    assert _Probe.model_validate_json(_Probe(value=value).model_dump_json()).value == expected
    assert _Probe(value=value).model_dump() == {"value": str(expected)}


@pytest.mark.parametrize(
    "encoding",
    [
        "1e5",
        "1E5",
        "1.5e3",
        "1e+5",
        "1e-5",
        " 1/2",
        "1/2 ",
        "1/2\n",
        "1_000",
        "\u0661\u0662\u0663",
        "0x10",
        "1/-2",
        "1/+2",
        "--1",
        "",
        "-",
        ".",
        "1/",
        "/2",
        "1.2.3",
        "1/2/3",
        "1.5/2",
        "nan",
        "inf",
        "-inf",
        "0/0",
        "1/00",
        "0" * (LIMIT + 1),
        "1." + "0" * (LIMIT + 1),
        "1" * (2 * LIMIT + 3),
    ],
)
def test_unbounded_or_foreign_spellings_refuse_before_conversion(encoding):
    with pytest.raises(InvalidRequestError) as exc:
        _probe(encoding)
    _assert_refusal(exc.value, "_Probe.value")
    with pytest.raises(InvalidRequestError) as via_python:
        _Probe(value=encoding)
    assert via_python.value.code == CODE


def test_fixed_decimal_with_an_unportable_reduced_denominator_refuses():
    # 4096 fractional digits ending in 1 pass lexically but reduce to a 4097-digit denominator.
    encoding = "." + "0" * (LIMIT - 1) + "1"
    with pytest.raises(InvalidRequestError) as exc:
        _Probe(value=encoding)
    _assert_refusal(exc.value, "_Probe.value", encoding)


def test_in_process_numbers_keep_their_exact_meaning():
    trusted = Fraction(1, 20)
    assert SequentialCell(metric="m", group_id="t", alpha=trusted).alpha is trusted
    # A plain float binds its exact binary64 value; the compliance policy binds the decimal.
    assert SequentialCell(metric="m", group_id="t", alpha=0.05).alpha == Fraction(0.05)
    assert SequentialCompliancePolicy(alpha=0.05).alpha == Fraction(1, 20)
    assert InferenceSpec(kind="always_valid", baseline_rate=0.3).baseline_rate == Fraction(3, 10)
    state = SequentialArmState.model_validate(
        {**_ARM, "mean": (5e-324,), "scatter": ((sys.float_info.max,),)}
    )
    assert state.mean == (Fraction(1, 2**1074),)
    assert state.scatter == ((Fraction(sys.float_info.max),),)
    assert SequentialArmState.model_validate_json(state.model_dump_json()) == state


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_floats_refuse_with_the_shared_code(value):
    with pytest.raises(InvalidRequestError) as exc:
        SequentialCell(metric="m", group_id="t", alpha=value)
    _assert_refusal(exc.value, "SequentialCell.alpha")


_JSON_NUMBERS = ["0.05", "1e-5", "1E+5", "1e50000", "NaN", "-Infinity"]


def _json_text(payload: dict, literal: str) -> str:
    # Place a raw JSON token where the rational is spelled; json.dumps cannot emit one.
    return json.dumps(payload).replace('"__RAW__"', literal, 1)


@pytest.mark.parametrize("model,payload,field", _FIELDS)
@pytest.mark.parametrize("literal", _JSON_NUMBERS)
def test_json_floating_literals_refuse_at_every_rational_field(model, payload, field, literal):
    # The JSON reader has already rounded the literal to binary64 (or to infinity), so
    # neither the spelling nor the exact value can be recovered; only a string carries it.
    with pytest.raises(InvalidRequestError) as exc:
        model.model_validate_json(_json_text(payload("__RAW__"), literal))
    _assert_refusal(exc.value, field, literal)


@pytest.mark.parametrize("model,payload,field", _FIELDS)
@pytest.mark.parametrize("value", [True, False])
def test_booleans_refuse_at_every_rational_field_on_both_paths(model, payload, field, value):
    with pytest.raises(InvalidRequestError) as via_json:
        model.model_validate_json(json.dumps(payload(value)))
    _assert_refusal(via_json.value, field)
    with pytest.raises(InvalidRequestError) as via_python:
        model.model_validate(payload(value))
    assert dict(via_python.value.context) == dict(via_json.value.context)


def test_checkpoint_row_json_floating_literal_refuses_at_its_owning_field():
    # The public format-9 reader validates the checkpoint text as JSON, not a decoded dict.
    row = _hostile_row("__RAW__")
    row["sequential_snapshot"] = _json_text(_hostile_snapshot("__RAW__"), "0.05")
    with pytest.raises(InvalidRequestError) as exc:
        Analysis.from_moments([row], metrics={"m": "mean"}, control="c")
    _assert_refusal(exc.value, "SequentialArmState.mean", "0.05")


@pytest.mark.parametrize("mutate,field", _PLAN_RATIONALS)
@pytest.mark.parametrize("literal", ["0.05", "1e-5", "1e50000", "true"])
def test_compiled_plan_json_text_refuses_numeric_rational_literals(mutate, field, literal):
    payload = _golden_plan()
    mutate(payload, "__RAW__")
    text = _json_text(payload, literal)
    with pytest.raises(InvalidRequestError) as decoded:
        compiled_plan_from_json(text)
    _assert_refusal(decoded.value, field, literal)
    with pytest.raises(InvalidRequestError) as typed:
        WireCompiledDecisionPlan.model_validate_json(text)
    assert dict(typed.value.context) == dict(decoded.value.context)


def test_trusted_plan_mapping_keeps_python_float_semantics():
    # A Python caller's float is trusted: the compliance policy binds the decimal it
    # typed and a cell binds the exact binary64, so both equal the golden's spellings.
    payload = _golden_plan()
    payload["compliance"]["alpha"] = 0.05
    payload["inference"]["registration"]["roster"][0]["alpha"] = 0.05
    plan = compiled_plan_from_dict(payload)
    assert plan.compliance is not None and plan.compliance.alpha == Fraction(1, 20)
    assert plan == compiled_plan_from_json(json.dumps(_golden_plan()))


def test_json_integer_literals_and_canonical_strings_replay_declared_decimals():
    policy = SequentialCompliancePolicy(alpha=0.05, null_lift=0.5)
    assert (policy.alpha, policy.null_lift) == (Fraction(1, 20), Fraction(1, 2))
    assert SequentialCompliancePolicy.model_validate_json(policy.model_dump_json()) == policy
    replayed = SequentialCompliancePolicy.model_validate_json('{"alpha": "1/20", "null_lift": 1}')
    assert replayed.null_lift == Fraction(1) and type(replayed.null_lift) is Fraction
    spec = InferenceSpec(kind="always_valid", baseline_rate=0.3)
    assert InferenceSpec.model_validate_json(spec.model_dump_json()) == spec
    text = '{"kind": "always_valid", "baseline_rate": "0.3"}'
    assert InferenceSpec.model_validate_json(text).baseline_rate == Fraction(3, 10)


def test_oversized_json_integer_refuses_at_its_owning_field():
    payload = json.dumps({**_ARM, "mean": [10**LIMIT], "scatter": [["0"]]})
    with pytest.raises(InvalidRequestError) as exc:
        SequentialArmState.model_validate_json(payload)
    _assert_refusal(exc.value, "SequentialArmState.mean")


@pytest.mark.parametrize(
    "constructor,field",
    [
        (
            lambda value: SequentialCompliancePolicy(alpha=value),
            "SequentialCompliancePolicy.alpha",
        ),
        (
            lambda value: SequentialCompliancePolicy(alpha=0.05, null_lift=value),
            "SequentialCompliancePolicy.null_lift",
        ),
        (
            lambda value: InferenceSpec(kind="always_valid", baseline_rate=value),
            "InferenceSpec.baseline_rate",
        ),
    ],
)
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_decimal_policy_nonfinite_input_preserves_rational_refusal(constructor, field, value):
    with pytest.raises(InvalidRequestError) as exc:
        constructor(value)
    _assert_refusal(exc.value, field)


@pytest.mark.parametrize("value", [None, [1, 2], {"n": 1}, b"1/2", object()])
def test_foreign_types_refuse_instead_of_crashing(value):
    with pytest.raises(InvalidRequestError) as exc:
        SequentialCell(metric="m", group_id="t", alpha=value)
    assert exc.value.code == CODE
    assert exc.value.context["field"] == "SequentialCell.alpha"


def test_export_outside_the_portable_limit_refuses_rather_than_writing_the_checkpoint():
    huge = Fraction(10 ** (LIMIT + 1))
    # Trusted in-process rationals are admitted unchanged ...
    state = SequentialArmState.model_validate(
        {**_ARM, "mean": (huge,), "scatter": ((Fraction(0),),)}
    )
    assert state.mean[0] is huge
    # ... but cannot be written as a portable checkpoint.
    for export in (state.model_dump_json, state.model_dump, lambda: state.model_dump(mode="json")):
        with pytest.raises(InvalidRequestError) as exc:
            export()
        _assert_refusal(exc.value, "export", str(huge))
    tiny = Fraction(1, 10 ** (LIMIT + 1))
    with pytest.raises(InvalidRequestError) as exc:
        PredictivePrior(kind="beta", a=tiny, b=1).model_dump_json()
    _assert_refusal(exc.value, "export", str(tiny.denominator))


def test_snapshot_payload_that_is_not_json_refuses_with_a_code():
    with pytest.raises(CodedError) as exc:
        snapshot_from_json('{"version": 2, "states": [{"n": 1' + "0" * 5000 + "}]}")
    assert exc.value.code == "sequential.source.invalid"
    with pytest.raises(CodedError) as malformed:
        snapshot_from_json("{not json")
    assert malformed.value.code == "sequential.source.invalid"
