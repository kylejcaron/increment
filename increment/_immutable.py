"""Immutable containers shared by query-free package layers."""

from collections.abc import Mapping
from typing import Any


class _FrozenMapping(Mapping[str, Any]):
    """Small immutable mapping that remains deepcopy- and pickle-compatible."""

    def __init__(self, value: Mapping[str, Any]) -> None:
        self._value = dict(value)

    def __getitem__(self, key: str) -> Any:
        return self._value[key]

    def __iter__(self):
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and self._value == other

    def __reduce__(self):
        return type(self), (self._value,)

    def __deepcopy__(self, memo: dict[int, Any]):
        from copy import deepcopy

        return type(self)(deepcopy(self._value, memo))
