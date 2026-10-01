"""Stochastic policies and the immutable (policy_id, version) registry."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, Self, cast, runtime_checkable
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FieldSerializationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from increment.errors import CodedModel, _thaw
from increment.logged_policy._refusals import _raise

_NORMALIZATION_TOLERANCE = 1e-12
_MISSING_CONTEXT = object()
type ContextScalar = (
    None
    | bool
    | int
    | float
    | str
    | bytes
    | Decimal
    | date
    | datetime
    | time
    | timedelta
    | UUID
    | Enum
)
type ContextKey = ContextScalar | tuple[Any, ...] | frozenset[Any]
type ContextValue = (
    ContextScalar
    | Mapping[ContextKey, Any]
    | list[Any]
    | tuple[Any, ...]
    | set[Any]
    | frozenset[Any]
)


def _is_numpy_scalar(value: Any) -> bool:
    """Recognize numpy scalar MROs without making numpy a runtime dependency."""
    return any(
        base.__name__ == "generic" and base.__module__.partition(".")[0] == "numpy"
        for base in type(value).__mro__
    )


def _freeze(value: Any) -> ContextValue:
    """Normalize the shared context domain and recursively freeze its containers."""
    if _is_numpy_scalar(value):
        normalized = value.item()
        if _is_numpy_scalar(normalized):
            _raise("logged_policy.trace.context_value_unsupported", value_type=type(value).__name__)
        return _freeze(normalized)
    if isinstance(value, Mapping):
        return MappingProxyType({_freeze_key(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    if value is None or isinstance(
        value, (bool, int, float, str, bytes, Decimal, date, datetime, time, timedelta, UUID, Enum)
    ):
        return value
    _raise("logged_policy.trace.context_value_unsupported", value_type=type(value).__name__)


def _freeze_key(value: Any) -> ContextKey:
    frozen = _freeze(value)
    try:
        hash(frozen)
    except TypeError:
        _raise("logged_policy.trace.context_value_unsupported", value_type=type(value).__name__)
    return cast("ContextKey", frozen)


@runtime_checkable
class StochasticPolicy(Protocol):
    """A pre-registered policy that reports its exact action probability at a history.

    ``policy_id`` and ``version`` are read-only so frozen models satisfy the
    protocol; plain class or instance attributes satisfy it too.
    """

    @property
    def policy_id(self) -> str: ...

    @property
    def version(self) -> str: ...

    def probability(self, action: str, context: Mapping[str, Any]) -> float: ...


def _validate_distribution(
    probabilities: Mapping[str, float], *, policy_id: str, version: str, context: object
) -> None:
    values = tuple(probabilities.values())
    # Finiteness first: fsum raises on a mix of +inf and -inf.
    if not all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in values) or not math.isclose(
        math.fsum(values), 1.0, rel_tol=0.0, abs_tol=_NORMALIZATION_TOLERANCE
    ):
        _raise(
            "logged_policy.support.policy_not_normalized",
            policy_id=policy_id,
            version=version,
            context=context,
            probabilities=dict(probabilities),
        )


class TabularPolicy(CodedModel, BaseModel):
    """A policy whose action distribution is a table over one context value.

    ``probabilities`` maps the value of ``context[context_key]`` to an
    action distribution; ``default`` applies when the key is absent or its
    value is not tabulated. A policy with an empty table and a ``default``
    ignores the history entirely (the spec's ``reference-policy``).

    Persistence: Python-mode dumps, ``copy.deepcopy`` and (trusted) pickle
    keep every key type and rebuild an equivalent policy. JSON dumps
    roundtrip only when every table key is a string; any other key type
    refuses at serialization with ``logged_policy.policy.json_context_key``
    rather than becoming a different key. An absent ``default`` stays absent.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    context_key: str = "x"
    probabilities: Mapping[ContextKey, Mapping[str, float]] = Field(default_factory=dict)
    default: Mapping[str, float] | None = None

    @field_validator("probabilities", mode="before")
    @classmethod
    def _raw_probability_keys_in_domain(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            _raise(
                "logged_policy.trace.context_value_unsupported",
                value_type=type(value).__name__,
            )
        return _freeze(value)

    @model_validator(mode="after")
    def _normalized(self) -> TabularPolicy:
        for value, distribution in self.probabilities.items():
            _validate_distribution(
                distribution,
                policy_id=self.policy_id,
                version=self.version,
                context={self.context_key: value},
            )
        if self.default is not None:
            _validate_distribution(
                self.default, policy_id=self.policy_id, version=self.version, context="default"
            )
        object.__setattr__(self, "probabilities", _freeze(self.probabilities))
        object.__setattr__(self, "default", None if self.default is None else _freeze(self.default))
        return self

    def _require_json_keys(self) -> None:
        bad = {type(key).__name__ for key in self.probabilities if not isinstance(key, str)}
        if bad:
            _raise(
                "logged_policy.policy.json_context_key",
                policy_id=self.policy_id,
                version=self.version,
                key_types=tuple(sorted(bad)),
                route=(
                    "persist with model_dump(mode='python') or pickle from a trusted source, "
                    "or use string context keys"
                ),
            )

    @field_serializer("probabilities", "default")
    def _serialize_distribution_mapping(self, value: Any, info: FieldSerializationInfo) -> Any:
        if info.mode_is_json():
            self._require_json_keys()
        return _thaw(value)

    def model_dump(self, *, mode: str = "python", **kwargs: Any) -> dict[str, Any]:
        if mode == "json":
            self._require_json_keys()
        return super().model_dump(mode=mode, **kwargs)

    def model_dump_json(self, **kwargs: Any) -> str:
        self._require_json_keys()
        return super().model_dump_json(**kwargs)

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        return type(self).model_validate, (self.model_dump(mode="python"),)

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        copied = type(self).model_validate(self.model_dump(mode="python"))
        if memo is not None:
            memo[id(self)] = copied
        return copied

    def _tabulated(self, value: Any) -> Mapping[str, float] | None:
        """The distribution tabulated for a supported context value, if any.

        Table keys are hashable, so an unhashable value (a nested mapping,
        say) is never tabulated and resolves to ``default`` like any other
        untabulated value; only values outside the context domain refuse.
        """
        frozen = _freeze(value)
        try:
            hash(frozen)
        except TypeError:
            return None
        return self.probabilities.get(cast("ContextKey", frozen))

    def distribution(self, context: Mapping[str, Any]) -> Mapping[str, float]:
        """The full action distribution at ``context`` (actions outside it have probability 0)."""
        value = context.get(self.context_key, _MISSING_CONTEXT)
        table = None if value is _MISSING_CONTEXT else self._tabulated(value)
        if table is None:
            table = self.default
        if table is None:
            _raise(
                "logged_policy.support.context_unsupported",
                policy_id=self.policy_id,
                version=self.version,
                context_key=self.context_key,
                context=dict(context),
            )
        return table

    def probability(self, action: str, context: Mapping[str, Any]) -> float:
        return float(self.distribution(context).get(action, 0.0))


class PolicyRegistry:
    """Immutable lookup of registered policies by ``(policy_id, version)``."""

    __slots__ = ("_policies",)

    def __init__(self, policies: Iterable[StochasticPolicy]) -> None:
        table: dict[tuple[str, str], StochasticPolicy] = {}
        for policy in policies:
            key = (policy.policy_id, policy.version)
            if key in table:
                _raise("logged_policy.registry.duplicate", policy_id=key[0], version=key[1])
            table[key] = policy
        self._policies: Mapping[tuple[str, str], StochasticPolicy] = MappingProxyType(table)

    @property
    def versions(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._policies)

    def __contains__(self, key: object) -> bool:
        return key in self._policies

    def get(self, policy_id: str, version: str) -> StochasticPolicy | None:
        return self._policies.get((policy_id, version))


TARGET_POLICY_V1 = TabularPolicy(
    policy_id="target-policy",
    version="v1",
    probabilities={0: {"A": 0.2, "B": 0.8}, 1: {"A": 0.8, "B": 0.2}},
)
REFERENCE_POLICY_V1 = TabularPolicy(
    policy_id="reference-policy", version="v1", default={"A": 0.5, "B": 0.5}
)
