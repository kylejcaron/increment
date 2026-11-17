"""Independent, immutable projections of retained posterior queries."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Literal, Protocol, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from increment._immutable import _FrozenMapping
from increment._literals import PreferredDirection
from increment.errors import CodedModel
from increment.estimation._relative_posterior import (
    _invalid,
    _PosteriorNumericalError,
    _unavailable,
)
from increment.estimation.priors import LiftPosterior
from increment.semantics.models import favorable_alternative

_OPERATIONS = frozenset(
    {
        "chance_to_beat",
        "chance_to_beat_favorable",
        "risk_if_shipped",
        "risk_if_shipped_favorable",
        "quantile",
    }
)


class _PosteriorQueryRow(Protocol):
    @property
    def preferred_direction(self) -> PreferredDirection | None: ...

    @property
    def posterior_scale(self) -> Literal["linear", "log"] | None: ...

    def _posterior(self) -> LiftPosterior | None: ...


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return _FrozenMapping({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _validate_parameters(operation: str, parameters: Mapping[str, object]) -> None:
    expected = ("probability",) if operation == "quantile" else ()
    if set(parameters) != set(expected):
        _invalid(
            operation,
            "unexpected parameters",
            expected_parameters=expected,
            parameters=parameters,
        )
    if operation == "quantile":
        probability = parameters["probability"]
        if not isinstance(probability, (int, float)) or isinstance(probability, bool):
            _invalid(operation, "probability must be numeric", probability=probability)
        if not 0 < probability < 1:
            _invalid(operation, "probability must be in (0, 1)", probability=probability)


class _PosteriorQueryProjection(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    value: float | None
    operation: str
    parameters: Mapping[str, object]
    code: str | None = None
    context: Mapping[str, object] = Field(default_factory=dict, validate_default=True)

    @field_validator("operation")
    @classmethod
    def _supported_operation(cls, value: str) -> str:
        if value not in _OPERATIONS:
            _invalid(value, "unsupported operation")
        return value

    @model_validator(mode="after")
    def _valid_parameters(self) -> _PosteriorQueryProjection:
        _validate_parameters(self.operation, self.parameters)
        return self

    @field_validator("parameters", "context", mode="after")
    @classmethod
    def _freeze_mapping(cls, value: Mapping[str, object]) -> Mapping[str, object]:
        return _FrozenMapping({key: _freeze(item) for key, item in value.items()})

    @field_serializer("parameters", "context")
    def _serialize_mapping(self, value: Mapping[str, object]) -> object:
        return _plain(value)


def _posterior_chance_to_beat(row: _PosteriorQueryRow, *, favorable: bool) -> float | None:
    side = favorable_alternative(row.preferred_direction) if favorable else "greater"
    if side is None:
        return None
    if hasattr(row, "_refuse_selected_sequential_probability"):
        cast(Callable[[], None], row._refuse_selected_sequential_probability)()
    posterior = row._posterior()
    if posterior is None:
        return None
    # Zero lift is zero on the log scale as well: log1p(0) == 0.
    return posterior.survival(0.0) if side == "greater" else posterior.cdf(0.0)


def _posterior_risk_if_shipped(row: _PosteriorQueryRow, *, favorable: bool) -> float | None:
    posterior = row._posterior()
    if posterior is None or row.posterior_scale is None:
        return None
    if favorable:
        if row.preferred_direction is None:
            return None
        if row.preferred_direction == "decrease":
            return posterior.expected_positive_part(scale=row.posterior_scale)
    return posterior.expected_negative_part(scale=row.posterior_scale)


def _project_posterior_query(
    row: _PosteriorQueryRow, *, operation: str, parameters: Mapping[str, object]
) -> _PosteriorQueryProjection:
    """Refuse invalid requests; retain numerical failures without changing density."""
    if operation not in _OPERATIONS:
        _invalid(operation, "unsupported operation")
    _validate_parameters(operation, parameters)
    try:
        if operation in {"chance_to_beat", "chance_to_beat_favorable"}:
            value = _posterior_chance_to_beat(row, favorable=operation.endswith("_favorable"))
        elif operation in {"risk_if_shipped", "risk_if_shipped_favorable"}:
            value = _posterior_risk_if_shipped(row, favorable=operation.endswith("_favorable"))
        else:
            posterior = row._posterior()
            probability = cast(float, parameters["probability"])
            value = None if posterior is None else posterior.quantile(probability)
        if value is not None and not math.isfinite(value):
            _unavailable(operation, "result_unrepresentable", **parameters)
    except _PosteriorNumericalError as error:
        return _PosteriorQueryProjection(
            value=None,
            operation=operation,
            parameters=parameters,
            code=error.code,
            context=error.context,
        )
    return _PosteriorQueryProjection(value=value, operation=operation, parameters=parameters)
