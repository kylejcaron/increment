"""Portable, immutable estimator advisories retained with readout evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_serializer, model_validator

from increment._canonical import canonical_digest_bytes
from increment._immutable import _FrozenMapping


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _FrozenMapping({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    return value


class ReadoutAdvisory(BaseModel):
    """An essential estimator qualification attached to its affected cell."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    code: str
    context: Mapping[str, object]
    message: str
    severity: Literal["info", "warning"] = "warning"

    @model_validator(mode="after")
    def _freeze_context(self):
        frozen = _freeze(self.context)
        canonical_digest_bytes(_thaw(frozen))
        object.__setattr__(self, "context", frozen)
        return self

    @field_serializer("context", when_used="json")
    def _serialize_context(self, value):
        return _thaw(value)
