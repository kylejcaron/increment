"""Canonical JSON encoding shared by query-free package layers."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import cast
from uuid import UUID


class CanonicalJSONError(ValueError):
    """A value cannot be represented in the canonical JSON vocabulary."""

    def __init__(self, message: str, *, code: str = "artifact.digest.json") -> None:
        super().__init__(message)
        self.code = code


def _json_value(value: object) -> object:
    mapping = _mapping(value)
    if mapping is not None:
        out: dict[str, object] = {}
        for key, item in mapping.items():
            if not isinstance(key, str):
                raise CanonicalJSONError("JSON keys must be strings")
            if key in out:
                raise CanonicalJSONError("duplicate JSON key")
            out[key] = _json_value(item)
        return out
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        if not -(2**53) < value < 2**53:
            raise CanonicalJSONError("JSON integer exceeds exact range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalJSONError(
                "canonical JSON numbers must be finite", code="artifact.digest.nonfinite"
            )
        return 0 if value == 0.0 else value
    if isinstance(value, UUID):
        return str(value).lower()
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise CanonicalJSONError(
                "canonical JSON datetime must be timezone-aware",
                code="artifact.digest.timestamp",
            )
        normalized = value.astimezone(UTC)
        return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_json_value(x) for x in value]
    if isinstance(value, (set, frozenset)):
        raise CanonicalJSONError("unordered sets are not canonical JSON")
    raise CanonicalJSONError(f"unsupported JSON value {type(value).__name__}")


def _mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        result = dump(mode="python", by_alias=True)
        if isinstance(result, Mapping):
            return cast(Mapping[str, object], result)
    return None


def canonical_model_value(value: object) -> object:
    """Normalize values using the artifact contract's legacy model semantics."""
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="python", by_alias=True, exclude_none=False)
    if isinstance(value, Mapping):
        return {str(key): canonical_model_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [canonical_model_value(item) for item in value]
    if isinstance(value, (datetime, date, UUID)):
        return value
    return value


def _jcs_float(value: float) -> bytes:
    if not math.isfinite(value):
        raise CanonicalJSONError(
            "canonical JSON numbers must be finite", code="artifact.digest.nonfinite"
        )
    if value == 0.0:
        return b"0"
    text = repr(value).lower()
    magnitude = abs(value)
    if "e" not in text and not (magnitude >= 1e21 or magnitude < 1e-6):
        if text.endswith(".0"):
            text = text[:-2]
        return text.encode()
    coefficient, exponent_text = text.split("e", 1) if "e" in text else (text, "0")
    exponent = int(exponent_text)
    if 1e-6 <= magnitude < 1e21:
        from decimal import Decimal

        expanded = format(Decimal(text), "f")
        if "." in expanded:
            expanded = expanded.rstrip("0").rstrip(".")
        return expanded.encode()
    sign = ""
    if coefficient.startswith("-"):
        sign, coefficient = "-", coefficient[1:]
    digits = coefficient.replace(".", "").rstrip("0") or "0"
    point = coefficient.find(".")
    if point < 0:
        point = len(coefficient)
    exponent += point - 1
    mantissa = digits[0] + (("." + digits[1:]) if len(digits) > 1 else "")
    return f"{sign}{mantissa}e{exponent:+d}".encode()


def _render(value: object) -> bytes:
    if value is None:
        return b"null"
    if value is True:
        return b"true"
    if value is False:
        return b"false"
    if isinstance(value, str):
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        except UnicodeEncodeError as exc:
            raise CanonicalJSONError(f"string is not valid UTF-8: {exc}") from exc
    if isinstance(value, int):
        return str(value).encode()
    if isinstance(value, float):
        return _jcs_float(value)
    if isinstance(value, list):
        return b"[" + b",".join(_render(x) for x in value) + b"]"
    if isinstance(value, dict):
        try:
            pairs = sorted(
                cast(dict[str, object], value).items(),
                key=lambda item: item[0].encode("utf-16-be", "strict"),
            )
            encoded_pairs = [
                json.dumps(k, ensure_ascii=False).encode("utf-8") + b":" + _render(v)
                for k, v in pairs
            ]
        except UnicodeEncodeError as exc:
            raise CanonicalJSONError(f"object key is not valid UTF-8: {exc}") from exc
        return b"{" + b",".join(encoded_pairs) + b"}"
    raise CanonicalJSONError("internal noncanonical JSON value")


def canonical_json_bytes(value: object) -> bytes:
    """Encode *value* as deterministic, format-1 canonical JSON bytes."""
    return _render(_json_value(value))


def canonical_json_loads(payload: str) -> object:
    """Parse canonical JSON while preserving the writer's numeric domain."""

    def parse_int(token: str) -> int | float:
        value = int(token)
        if -(2**53) < value < 2**53:
            return value
        parsed = float(token)
        if not math.isfinite(parsed) or _jcs_float(parsed).decode() != token:
            raise CanonicalJSONError("JSON integer exceeds exact range")
        return parsed

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in items:
            if key in result:
                raise CanonicalJSONError(
                    "duplicate JSON key", code="artifact.digest.recanonicalize"
                )
            result[key] = item
        return result

    return json.loads(payload, parse_int=parse_int, object_pairs_hook=pairs)


__all__ = ["CanonicalJSONError", "canonical_json_bytes", "canonical_json_loads"]
