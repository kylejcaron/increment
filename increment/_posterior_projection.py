"""Independent, immutable projections of retained posterior queries."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from increment._immutable import _FrozenMapping
from increment._literals import PreferredDirection
from increment.estimation._relative_posterior import _PosteriorNumericalError
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


class _PosteriorQueryProjection(BaseModel):
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
            raise ValueError("Unsupported posterior query operation")
        return value

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
    """Evaluate one supported query without changing retained density availability."""
    if operation not in _OPERATIONS:
        raise ValueError("Unsupported posterior query operation")
    expected = {"probability"} if operation == "quantile" else set()
    if set(parameters) != expected:
        raise ValueError("Unexpected posterior query parameters")
    try:
        if operation in {"chance_to_beat", "chance_to_beat_favorable"}:
            value = _posterior_chance_to_beat(row, favorable=operation.endswith("_favorable"))
        elif operation in {"risk_if_shipped", "risk_if_shipped_favorable"}:
            value = _posterior_risk_if_shipped(row, favorable=operation.endswith("_favorable"))
        else:
            posterior = row._posterior()
            probability = parameters["probability"]
            if not isinstance(probability, (int, float)) or isinstance(probability, bool):
                raise ValueError("Posterior quantile probability must be numeric")
            value = None if posterior is None else posterior.quantile(probability)
    except _PosteriorNumericalError as error:
        return _PosteriorQueryProjection(
            value=None,
            operation=operation,
            parameters=parameters,
            code=error.code,
            context=error.context,
        )
    return _PosteriorQueryProjection(value=value, operation=operation, parameters=parameters)
