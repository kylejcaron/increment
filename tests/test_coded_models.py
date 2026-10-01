from __future__ import annotations

import pickle
from datetime import date
from typing import Any, cast

import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    field_validator,
)

from increment.errors import (
    CodedModel,
    DefinitionError,
    InvalidRequestError,
    RefusalSpec,
    refuse,
    unwrap_coded,
)
from increment.estimation.arm_contract import PlanningFamilyExpansion
from increment.semantics.models import Definitions

_SPEC = RefusalSpec(
    "test.coded_model.negative",
    InvalidRequestError,
    lambda *, value: f"value must be >= 0, got {value}",
)


class _Widget(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")
    value: int = Field(default=0)

    @field_validator("value")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            refuse(_SPEC, value=v)
        return v


@pytest.mark.parametrize(
    "construct",
    [
        lambda: _Widget(value=-1),
        lambda: _Widget.model_validate({"value": -1}),
        lambda: _Widget.model_validate_json('{"value": -1}'),
        lambda: _Widget.model_validate_strings({"value": "-1"}),
    ],
)
def test_coded_model_surfaces_code_on_every_direct_model_entry_path(construct):
    with pytest.raises(InvalidRequestError) as exc_info:
        construct()
    assert exc_info.value.code == "test.coded_model.negative"
    assert exc_info.value.context["value"] == -1


def test_coded_model_nested_revalidation_surfaces_code():
    class _Outer(CodedModel, BaseModel):
        model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")
        widget: _Widget

    outer = _Outer(widget=_Widget(value=1))
    assert outer.widget.value == 1
    with pytest.raises(InvalidRequestError) as exc_info:
        _Outer.model_validate({"widget": {"value": -1}})
    assert exc_info.value.code == "test.coded_model.negative"


def test_coded_model_inherited_once_on_root_surfaces_code_through_subclass():
    """CodedModel is applied once at the root of a two-level hierarchy (decision.py's actual
    shape: `_ArmDecisionProcedure(CodedModel, BaseModel)` -> `RelativeArmDecisionProcedure(_ArmDecisionProcedure)`),
    never re-inserted on the subclass -- re-inserting it fails C3 linearization."""

    class _BaseProcedure(CodedModel, BaseModel):
        model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)
        metric: str = Field(min_length=1)

    class _RelativeProcedure(_BaseProcedure):
        axis: str = "relative"

        @field_validator("axis")
        @classmethod
        def _known_axis(cls, v: str) -> str:
            if v not in ("relative", "absolute"):
                refuse(_SPEC, value=v)
            return v

    assert _RelativeProcedure(metric="m", axis="relative").axis == "relative"
    with pytest.raises(InvalidRequestError) as exc_info:
        _RelativeProcedure(metric="m", axis="bogus")
    assert exc_info.value.code == "test.coded_model.negative"
    assert exc_info.value.context["value"] == "bogus"


def test_coded_model_declared_type_failure_is_coded():
    with pytest.raises(InvalidRequestError) as raised:
        _Widget(value="not-an-int")
    assert raised.value.code == "model.field.type"


@pytest.mark.parametrize(
    "construct",
    [
        lambda: PlanningFamilyExpansion.model_validate({"family_size": "3"}, strict=True),
        lambda: PlanningFamilyExpansion.model_validate_json('{"family_size": "3"}', strict=True),
    ],
)
def test_coded_model_preserves_strict_validation(construct):
    with pytest.raises(InvalidRequestError) as raised:
        construct()
    assert raised.value.code == "model.field.type"


def test_coded_model_preserves_model_validate_options():
    class _Aliased(CodedModel, BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: int = Field(alias="wireValue")

    model = _Aliased.model_validate(
        {"value": 3, "ignored": True},
        extra="ignore",
        by_alias=False,
        by_name=True,
    )
    assert model.value == 3


def test_coded_model_preserves_context_json_mode_defaults_and_nested_types():
    class _Envelope(CodedModel, BaseModel):
        when: date
        widget: _Widget
        label: str = "default"

        @field_validator("when")
        @classmethod
        def _require_json_context(cls, value: date, info: ValidationInfo) -> date:
            if info.mode != "json" or info.context != {"source": "wire"}:
                raise ValueError("validation options were not preserved")
            return value

    model = _Envelope.model_validate_json(
        '{"when": "2026-09-14", "widget": {"value": 2}}',
        strict=True,
        context={"source": "wire"},
    )

    assert model.when == date(2026, 9, 14)
    assert model.widget == _Widget(value=2)
    assert model.label == "default"


@pytest.mark.parametrize(
    "construct",
    [
        lambda: Definitions(day_boundary="EST"),
        lambda: Definitions.model_validate({"day_boundary": "EST"}),
        lambda: Definitions.model_validate_json('{"day_boundary": "EST"}'),
        lambda: Definitions.model_validate_strings({"day_boundary": "EST"}),
    ],
)
def test_coded_model_direct_boundaries_surface_definition_error(construct):
    with pytest.raises(DefinitionError) as exc_info:
        construct()
    assert exc_info.value.code == "definition.day_boundary_accepts"
    assert exc_info.value.context["v"] == "EST"


@pytest.mark.parametrize(
    "validate",
    [
        lambda: TypeAdapter(Definitions).validate_python({"day_boundary": "EST"}),
        lambda: TypeAdapter(Definitions).validate_json('{"day_boundary": "EST"}'),
    ],
)
def test_type_adapter_boundary_supports_explicit_coded_error_recovery(validate):
    with pytest.raises(ValidationError) as exc_info:
        validate()

    with pytest.raises(DefinitionError) as coded:
        unwrap_coded(exc_info.value)
    assert coded.value.code == "definition.day_boundary_accepts"
    assert coded.value.context["v"] == "EST"


def test_ordinary_model_boundary_supports_explicit_coded_error_recovery():
    class _Envelope(BaseModel):
        definitions: Definitions

    with pytest.raises(ValidationError) as exc_info:
        _Envelope(definitions={"day_boundary": "EST"})

    with pytest.raises(DefinitionError) as coded:
        unwrap_coded(exc_info.value)
    assert coded.value.code == "definition.day_boundary_accepts"
    assert coded.value.context["v"] == "EST"


def test_field_translation_walks_aliases_unknown_keys_and_scalar_items():
    class _Fields(CodedModel, BaseModel):
        model_config = ConfigDict(extra="forbid")
        count: int = Field(alias="wireCount", ge=1)
        values: list[int]
        by_index: dict[int, int] = {}

    with pytest.raises(InvalidRequestError) as raised:
        _Fields.model_validate({"wireCount": 0, "values": [1]})
    assert raised.value.code == "model.field.range"

    with pytest.raises(InvalidRequestError) as raised:
        _Fields.model_validate({"wireCount": 1, "values": ["bad"]})
    assert raised.value.code == "model.field.type"

    with pytest.raises(InvalidRequestError) as raised:
        _Fields.model_validate({"wireCount": 1, "values": [1], "extra": True})
    assert raised.value.code == "model.field.unknown"

    with pytest.raises(InvalidRequestError) as raised:
        _Fields.model_validate({"wireCount": 1, "values": [1], "by_index": {"one": 1}})
    assert raised.value.code == "model.field.type"
    assert raised.value.context["field"] == "by_index"


def test_nested_coded_model_field_translation_is_preserved():
    class _Child(CodedModel, BaseModel):
        count: int = Field(ge=1)

    class _Parent(CodedModel, BaseModel):
        child: _Child

    with pytest.raises(InvalidRequestError) as raised:
        _Parent.model_validate({"child": {"count": 0}})
    assert raised.value.code == "model.field.range"

    class _Collection(CodedModel, BaseModel):
        children: list[_Child]

    with pytest.raises(InvalidRequestError) as raised:
        _Collection.model_validate({"children": [{"count": 0}]})
    assert raised.value.code == "model.field.range"


def test_uncoded_validator_error_keeps_mixed_validation_error_atomic():
    class _Mixed(CodedModel, BaseModel):
        bounded: int = Field(ge=1)
        raw: int

        @field_validator("raw")
        @classmethod
        def _reject(cls, value: int) -> int:
            raise ValueError("uncoded")

    with pytest.raises(ValidationError) as raised:
        _Mixed(bounded=0, raw=1)
    assert {error["type"] for error in raised.value.errors()} >= {
        "greater_than_equal",
        "value_error",
    }


def test_invalid_input_context_is_bounded_and_picklable():
    class _Unpicklable:
        def __reduce__(self):
            raise TypeError("no pickle")

    with pytest.raises(InvalidRequestError) as raised:
        _Widget(value=cast(Any, _Unpicklable()))
    assert isinstance(raised.value.context["value"], str)
    assert len(raised.value.context["value"]) <= 256
    assert pickle.loads(pickle.dumps(raised.value)).code == "model.field.type"


def test_invalid_input_with_a_failing_repr_still_refuses_with_its_code():
    class _BadRepr:
        def __repr__(self) -> str:
            raise RuntimeError("repr failed")

    with pytest.raises(InvalidRequestError) as raised:
        _Widget(value=cast(Any, _BadRepr()))
    assert raised.value.code == "model.field.type"
    assert pickle.loads(pickle.dumps(raised.value)).code == "model.field.type"


def test_large_integer_context_is_bounded():
    class _Bounded(CodedModel, BaseModel):
        value: int = Field(le=1)

    # 10**5000 exceeds Python's int-to-str digit limit, so it must never be stringified.
    for value in (10**300, 10**5000):
        with pytest.raises(InvalidRequestError) as raised:
            _Bounded(value=value)
        assert isinstance(raised.value.context["value"], str)
        assert len(raised.value.context["value"]) <= 256


def test_untagged_union_failure_is_not_attributed_to_a_guessed_member():
    class _Left(CodedModel, BaseModel):
        model_config = ConfigDict(extra="forbid")
        left: int = Field(ge=1)

    class _Right(CodedModel, BaseModel):
        model_config = ConfigDict(extra="forbid")
        right: int = Field(ge=1)

    class _Holder(CodedModel, BaseModel):
        item: _Left | _Right

    # Every member fails, so the intended member is unknown: stay uncoded.
    with pytest.raises(ValidationError):
        _Holder.model_validate({"item": {"right": 0}})

    class _Single(CodedModel, BaseModel):
        item: _Left | int

    with pytest.raises(ValidationError):
        _Single.model_validate({"item": {"left": 0}})


def test_nested_untagged_union_failure_stays_uncoded():
    class _Left(CodedModel, BaseModel):
        model_config = ConfigDict(extra="forbid")
        left: int = Field(ge=1)

    class _Right(CodedModel, BaseModel):
        model_config = ConfigDict(extra="forbid")
        right: int = Field(ge=1)

    class _Child(CodedModel, BaseModel):
        item: _Left | _Right

    class _Holder(CodedModel, BaseModel):
        child: _Child

    # The union sits two models deep; its member labels must still not be guessed at.
    with pytest.raises(ValidationError):
        _Holder.model_validate({"child": {"item": {"right": 0}}})
