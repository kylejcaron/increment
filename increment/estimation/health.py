"""Receive-only models for source-backed experiment health diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator, model_validator

from increment.estimation.diagnostics import SRMResult

type HealthKind = Literal[
    "assigned_allocation",
    "triggered_balance",
    "trigger_rates",
    "trigger_maturity",
    "feed_completeness",
    "trigger_assumptions",
]
type AnalysisPopulation = Literal["assigned", "triggered"]
type HealthAvailability = Literal["available", "not_applicable", "unavailable"]
type HealthConclusion = Literal["not_rejected", "warning", "descriptive"]
type HealthValue = str | int | float | bool | None | tuple[str, ...] | Mapping[str, object]


class HealthCheck(BaseModel):
    """One identity-preserving source-health observation, never a ship verdict."""

    model_config = ConfigDict(frozen=True, validate_default=True)

    check: HealthKind
    analysis_population: AnalysisPopulation
    metric: str | None = None
    feed: str | None = None
    availability: HealthAvailability
    conclusion: HealthConclusion | None = None
    code: str | None = None
    context: Mapping[str, object]
    srm: SRMResult | None = None
    rates: Mapping[str, float] | None = None

    @field_validator("context", mode="before")
    @classmethod
    def _own_context(cls, value: Mapping[str, object]) -> Mapping[str, object]:
        return value

    @field_validator("rates", mode="before")
    @classmethod
    def _own_rates(cls, value: Mapping[str, float] | None) -> Mapping[str, float] | None:
        return None if value is None else cast(Mapping[str, float], _freeze_health(value))

    @field_serializer("context", "rates")
    def _serialize_mappings(self, value: Mapping[str, object] | None) -> dict[str, object] | None:
        if value is None:
            return None
        return {key: _thaw(item) for key, item in value.items()}

    @model_validator(mode="after")
    def _validate_availability(self) -> HealthCheck:
        object.__setattr__(
            self, "context", cast(Mapping[str, object], _freeze_health(self.context))
        )
        if self.rates is not None:
            object.__setattr__(self, "rates", cast(Mapping[str, float], _freeze_health(self.rates)))
        if self.availability != "available":
            if self.conclusion is not None:
                raise ValueError("unavailable and not-applicable checks cannot have a conclusion")
            if not self.context:
                raise ValueError(
                    "unavailable and not-applicable checks require a reason in context"
                )
        if self.availability == "available" and self.conclusion is None:
            raise ValueError("available checks require a diagnostic conclusion")
        return self


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _freeze_health(value: object) -> object:
    from increment._immutable import _FrozenMapping

    if isinstance(value, Mapping):
        return _FrozenMapping({key: _freeze_health(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_health(item) for item in value)
    return value


class ExperimentHealth(BaseModel):
    """Frozen collection of health checks; deliberately has no aggregate verdict."""

    model_config = ConfigDict(frozen=True, validate_default=True)

    checks: tuple[HealthCheck, ...]
    source_snapshot_id: str | None = None
    source_cutoff: str | None = None

    @field_validator("checks", mode="before")
    @classmethod
    def _own_checks(cls, value: object) -> tuple[HealthCheck, ...]:
        if not isinstance(value, (tuple, list)):
            raise TypeError("checks must be a tuple or list")
        return tuple(cast(HealthCheck, item) for item in value)

    @model_validator(mode="after")
    def _stable_check_order(self) -> ExperimentHealth:
        order = {
            name: index
            for index, name in enumerate(
                (
                    "assigned_allocation",
                    "triggered_balance",
                    "trigger_rates",
                    "trigger_maturity",
                    "feed_completeness",
                    "trigger_assumptions",
                )
            )
        }
        keys = [
            (order[check.check], check.analysis_population, check.metric or "", check.feed or "")
            for check in self.checks
        ]
        if keys != sorted(keys):
            raise ValueError("health checks must use stable check/population/metric/feed order")
        return self

    def __repr__(self) -> str:
        return f"ExperimentHealth(checks={len(self.checks)})"

    def __str__(self) -> str:
        return repr(self)

    def __reduce__(self):
        return _restore_experiment_health, (self.model_dump(mode="python"),)

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        copied = type(self).model_validate(self.model_dump(mode="python"))
        if memo is not None:
            memo[id(self)] = copied
        return copied


__all__ = ["ExperimentHealth", "HealthCheck"]


def _restore_experiment_health(values: dict[str, object]) -> ExperimentHealth:
    return ExperimentHealth.model_validate(values)
