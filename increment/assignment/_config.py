"""Validate and copy the fields used to make allocation decisions."""

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import NoReturn, TypeGuard, cast

from increment.errors import InvalidRequestError, raiser, refusals

from ._hash import _FULL, _hash_bytes, _prefix

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "allocation.invalid_config": "Invalid allocation configuration: {field} {reason}.",
        "allocation.invalid_target": "Invalid allocation target: {field} {reason}.",
        "allocation.invalid_unit": "Invalid allocation unit: {field} {reason}.",
    },
)
_raise = raiser(_REFUSALS)
_SURROGATE = re.compile(r"[\ud800-\udfff]")
_SCALAR_REASON = "must be a nonempty string of Unicode scalar values"
_LENGTH_REASON = "must fit in a 32-bit UTF-8 byte length"


@dataclass(frozen=True, slots=True)
class _Arm:
    start: int
    key: str
    weight_bps: int


@dataclass(frozen=True, slots=True)
class _Experiment:
    key: str
    enrollment_bps: int
    arms: tuple[_Arm, ...]
    enroll_prefix: bytes
    arm_prefix: bytes


@dataclass(frozen=True, slots=True)
class _Range:
    start: int
    end: int
    experiment: _Experiment


@dataclass(frozen=True, slots=True)
class _Layer:
    prefix: bytes
    ranges: tuple[_Range, ...]


@dataclass(frozen=True, slots=True)
class _Holdout:
    key: str
    prefix: bytes
    held_out_bps: int


@dataclass(frozen=True, slots=True)
class _Allocation:
    revision: str
    targets: Mapping[str, _Experiment | _Layer]
    holdouts: Mapping[str, _Holdout]
    covered: Mapping[str, _Holdout]


def _invalid_config(field: str, reason: str) -> NoReturn:
    _raise("allocation.invalid_config", field=field, reason=reason)


def _invalid_target(field: str, reason: str) -> NoReturn:
    _raise("allocation.invalid_target", field=field, reason=reason)


def _is_scalar(value: object) -> TypeGuard[str]:
    return type(value) is str and bool(value) and _SURROGATE.search(value) is None


def _check_target(value: object, field: str) -> None:
    if not _is_scalar(value):
        _invalid_target(field, _SCALAR_REASON)


def _unit_bytes(value: object) -> bytes:
    if not _is_scalar(value):
        _raise("allocation.invalid_unit", field="unit_id", reason=_SCALAR_REASON)
    encoded = _hash_bytes(value)
    if encoded is None:
        _raise("allocation.invalid_unit", field="unit_id", reason=_LENGTH_REASON)
    return encoded


def _record(
    value: object, field: str, *required: str, optional: str | None = None
) -> Mapping[str, object]:
    if type(value) is not dict:
        _invalid_config(field, "must be an object")
    for name in required:
        if name not in value:
            _invalid_config(field, f"is missing the {name} field")
    if any(type(name) is not str or (name not in required and name != optional) for name in value):
        _invalid_config(field, "contains an unrecognized field")
    return cast("Mapping[str, object]", value)


def _array(value: object, field: str) -> Sequence[object]:
    if type(value) is not list:
        _invalid_config(field, "must be an array")
    return cast("Sequence[object]", value)


def _string(value: object, field: str) -> str:
    if not _is_scalar(value):
        _invalid_config(field, _SCALAR_REASON)
    return value


def _key_bytes(key: str, field: str) -> bytes:
    encoded = _hash_bytes(key)
    if encoded is None:
        _invalid_config(field, _LENGTH_REASON)
    return encoded


def _integer(value: object, field: str, lower: int, upper: int) -> int:
    if not (type(value) is int or (type(value) is float and value.is_integer())) or not (
        lower <= value <= upper
    ):
        _invalid_config(field, f"must be an integer from {lower} through {upper}")
    return int(value)


def _parse_arms(value: object, field: str) -> tuple[_Arm, ...]:
    records = _array(value, field)
    if len(records) < 2:
        _invalid_config(field, "must contain at least two arms")
    arms: dict[bytes, tuple[str, int]] = {}
    for index, raw in enumerate(records):
        path = f"{field}[{index}]"
        arm = _record(raw, path, "key", "weight_bps")
        key = _string(arm["key"], f"{path}.key")
        encoded = key.encode("utf-8")
        if encoded in arms:
            _invalid_config(f"{path}.key", "duplicates an arm key")
        arms[encoded] = (key, _integer(arm["weight_bps"], f"{path}.weight_bps", 1, _FULL))

    start = 0
    ranges = []
    for encoded in sorted(arms):
        key, weight = arms[encoded]
        ranges.append(_Arm(start, key, weight))
        start += weight
    if start != _FULL:
        _invalid_config(field, "must have weights totaling 10000 basis points")
    return tuple(ranges)


def _parse_experiments(value: object) -> dict[str, _Experiment]:
    experiments: dict[str, _Experiment] = {}
    for index, raw in enumerate(_array(value, "experiments")):
        path = f"experiments[{index}]"
        experiment = _record(raw, path, "key", "enrollment_bps", "arms")
        key = _string(experiment["key"], f"{path}.key")
        if key in experiments:
            _invalid_config(f"{path}.key", "duplicates an experiment key")
        encoded = _key_bytes(key, f"{path}.key")
        experiments[key] = _Experiment(
            key,
            _integer(experiment["enrollment_bps"], f"{path}.enrollment_bps", 0, _FULL),
            _parse_arms(experiment["arms"], f"{path}.arms"),
            _prefix(b"enroll", encoded),
            _prefix(b"arm", encoded),
        )
    return experiments


def _parse_ranges(
    value: object, field: str, experiments: dict[str, _Experiment], layered: set[str]
) -> tuple[_Range, ...]:
    ranges: list[_Range] = []
    for index, raw in enumerate(_array(value, field)):
        path = f"{field}[{index}]"
        interval = _record(raw, path, "experiment", "start_bps", "end_bps")
        key = _string(interval["experiment"], f"{path}.experiment")
        if key not in experiments:
            _invalid_config(f"{path}.experiment", "must name a configured experiment")
        if key in layered:
            _invalid_config(f"{path}.experiment", "already belongs to a layer range")
        start = _integer(interval["start_bps"], f"{path}.start_bps", 0, _FULL)
        end = _integer(interval["end_bps"], f"{path}.end_bps", 0, _FULL)
        if start >= end:
            _invalid_config(path, "must start below its end")
        layered.add(key)
        ranges.append(_Range(start, end, experiments[key]))

    ranges.sort(key=lambda interval: interval.start)
    previous_end = 0
    for interval in ranges:
        if interval.start < previous_end:
            _invalid_config(field, "must not contain overlapping ranges")
        previous_end = interval.end
    return tuple(ranges)


def _parse_layers(
    value: object, experiments: dict[str, _Experiment]
) -> tuple[dict[str, _Layer], set[str]]:
    layers: dict[str, _Layer] = {}
    layered: set[str] = set()
    for index, raw in enumerate(_array(value, "layers")):
        path = f"layers[{index}]"
        layer = _record(raw, path, "key", "ranges")
        key = _string(layer["key"], f"{path}.key")
        if key in layers or key in experiments:
            _invalid_config(f"{path}.key", "duplicates a layer or experiment key")
        layers[key] = _Layer(
            _prefix(b"layer", _key_bytes(key, f"{path}.key")),
            _parse_ranges(layer["ranges"], f"{path}.ranges", experiments, layered),
        )
    return layers, layered


def _parse_holdouts(
    value: object, targets: Collection[str]
) -> tuple[dict[str, _Holdout], dict[str, _Holdout]]:
    holdouts: dict[str, _Holdout] = {}
    covered: dict[str, _Holdout] = {}
    for index, raw in enumerate(_array(value, "holdouts")):
        path = f"holdouts[{index}]"
        holdout = _record(raw, path, "key", "held_out_bps", "targets")
        key = _string(holdout["key"], f"{path}.key")
        if key in holdouts or key in targets:
            _invalid_config(f"{path}.key", "duplicates a holdout or decision target key")
        scope = _array(holdout["targets"], f"{path}.targets")
        if not scope:
            _invalid_config(f"{path}.targets", "must contain at least one decision target")
        compiled = _Holdout(
            key,
            _prefix(b"holdout", _key_bytes(key, f"{path}.key")),
            _integer(holdout["held_out_bps"], f"{path}.held_out_bps", 0, _FULL),
        )
        for target_index, raw_target in enumerate(scope):
            field = f"{path}.targets[{target_index}]"
            target = _string(raw_target, field)
            if target not in targets:
                _invalid_config(field, "must name a layer or a standalone experiment")
            if target in covered:
                _invalid_config(field, "already belongs to a holdout")
            covered[target] = compiled
        holdouts[key] = compiled
    return holdouts, covered


def _compile(config: object) -> _Allocation:
    root = _record(
        config, "config", "protocol", "revision", "experiments", "layers", optional="holdouts"
    )
    _integer(root["protocol"], "protocol", 1, 1)
    revision = _string(root["revision"], "revision")
    experiments = _parse_experiments(root["experiments"])
    layers, layered = _parse_layers(root["layers"], experiments)
    targets: dict[str, _Experiment | _Layer] = dict(layers)
    targets.update(
        (key, experiment) for key, experiment in experiments.items() if key not in layered
    )
    holdouts, covered = _parse_holdouts(root.get("holdouts", []), targets)
    return _Allocation(
        revision,
        MappingProxyType(targets),
        MappingProxyType(holdouts),
        MappingProxyType(covered),
    )
