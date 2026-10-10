"""Immutable containers shared by query-free package layers."""

from collections.abc import Iterator, Mapping
from typing import Any


class _FrozenMapping[Key, Value](Mapping[Key, Value]):
    """Small immutable mapping that remains deepcopy- and pickle-compatible."""

    def __init__(self, value: Mapping[Key, Value]) -> None:
        self._value: dict[Key, Value] = dict(value)

    def __getitem__(self, key: Key) -> Value:
        return self._value[key]

    def __iter__(self) -> Iterator[Key]:
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)

    def __repr__(self) -> str:
        from increment._display import format_mapping

        return format_mapping(self._value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and self._value == other

    def __reduce__(self):
        return type(self), (self._value,)

    def __deepcopy__(self, memo: dict[int, Any]):
        from copy import deepcopy

        return type(self)(deepcopy(self._value, memo))
