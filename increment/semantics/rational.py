"""Portable exact rationals on the sequential wire.

A registration, checkpoint or compiled plan spells every rational the way
``fractions.Fraction`` prints it: an optionally signed integer, or an
integer quotient with a positive denominator. Reading that spelling back
must cost work proportional to its length, so admission is lexical first
(type and length, then grammar, then a digit ceiling per integer component)
and only then a bounded ``Fraction`` conversion. ``Fraction("1e5000000")``
would otherwise allocate a five-million-digit integer from a nine-byte
payload, so scientific notation is refused before any arithmetic.

The ceiling is ``PORTABLE_RATIONAL_DIGITS`` decimal digits per component.
A finite binary64 is ``m * 2**e`` with ``|m| < 2**53`` and
``-1074 <= e <= 971``, so its exact rational has at most 309 numerator
digits and a power-of-two denominator of at most 1,075 bits (324 digits).
A retained arm state sums such values over ``n`` units: the mean's
denominator divides ``n * 2**1074``, and a centered scatter entry's
denominator divides ``n**2 * 2**2148`` (647 digits plus twice the digits
of ``n``) with a numerator below ``n**3 * 2**4198`` (1,264 digits plus
three times the digits of ``n``). Both components fit the ceiling for any
sample count below ``10**944``. The golden registered fixtures use fewer
than four digits per component; the extreme binary64 replay fixture stays
below 1,300. The ceiling leaves ample headroom while bounding hostile
portable input.

In-process values are trusted: a ``Fraction`` and a finite ``Decimal``
retain their exact meaning, a finite ``float`` binds its exact binary64
rational and a bounded ``int`` binds itself; ``bool`` is not a rational.
``DeclaredRational`` is the same wire type for declaration fields, where a
trusted finite ``float`` binds the decimal it was typed as instead.
Serialization spells ``str(Fraction)`` and enforces the ceiling, refusing
rather than writing a checkpoint no reader can admit.

JSON numeric literals never reach this parser as text. The JSON reader
caps an integer literal at the interpreter's own digit limit and hands the
rest to the integer branch, but it turns every floating literal -- fixed
or exponent, finite or not -- into a binary64 before validation, so both
the spelling and the exact value are already gone. A floating literal
therefore refuses in JSON validation mode; a JSON document spells a
rational as a string or an integer literal.
"""

from __future__ import annotations

import math
import re
from decimal import Decimal
from fractions import Fraction
from types import MappingProxyType
from typing import Annotated, NoReturn

from pydantic import BeforeValidator, PlainSerializer, ValidationInfo

from increment.errors import InvalidRequestError, RefusalSpec, refuse

PORTABLE_RATIONAL_DIGITS = 4096
"""Decimal digits admitted per integer component of a serialized rational."""

_CEILING = 10**PORTABLE_RATIONAL_DIGITS
# Sign, numerator, solidus, denominator: the longest admissible encoding.
_MAX_ENCODING_LENGTH = 2 * PORTABLE_RATIONAL_DIGITS + 2
_JSON_RATIONAL_CONTEXT = MappingProxyType({"rational_json": True})
_INTEGER = re.compile(r"[+-]?([0-9]+)")
_QUOTIENT = re.compile(r"[+-]?([0-9]+)/([0-9]+)")
_FIXED_DECIMAL = re.compile(r"[+-]?(?:[0-9]+\.[0-9]*|\.[0-9]+)")
_ROUTE_FORWARD = (
    "use bounded exact rationals from supported captures; "
    "scientific notation is not a portable rational encoding"
)


def _render_rational_invalid(
    *, field: str, reason: str, limit_digits: int, route_forward: str
) -> str:
    return (
        f"{field} is not a portable rational: {reason}; a portable rational is a signed "
        "integer, an integer quotient with a positive denominator or a fixed decimal, "
        f"each component at most {limit_digits} digits; {route_forward}"
    )


RATIONAL_INVALID = RefusalSpec(
    "sequential.wire.rational_invalid", InvalidRequestError, _render_rational_invalid
)


def _refuse(field: str, reason: str) -> NoReturn:
    refuse(
        RATIONAL_INVALID,
        field=field,
        reason=reason,
        limit_digits=PORTABLE_RATIONAL_DIGITS,
        route_forward=_ROUTE_FORWARD,
    )


def _portable(value: Fraction | int) -> bool:
    """Whether both reduced components fit the ceiling; two comparisons, no text."""
    return -_CEILING < value.numerator < _CEILING and value.denominator < _CEILING


def _admit(encoding: str, field: str) -> Fraction:
    """Convert a lexically admitted spelling; every step is bounded by the ceiling."""
    if len(encoding) > _MAX_ENCODING_LENGTH:
        _refuse(
            field,
            f"encoding of {len(encoding)} characters exceeds the "
            f"{_MAX_ENCODING_LENGTH}-character portable limit",
        )
    if match := _QUOTIENT.fullmatch(encoding):
        numerator, denominator = match.groups()
        for part, digits in (("numerator", numerator), ("denominator", denominator)):
            if len(digits) > PORTABLE_RATIONAL_DIGITS:
                _refuse(field, f"{part} has {len(digits)} digits")
        if not denominator.strip("0"):
            _refuse(field, "denominator is zero")
        value = Fraction(encoding)
    elif match := _INTEGER.fullmatch(encoding):
        if len(match.group(1)) > PORTABLE_RATIONAL_DIGITS:
            _refuse(field, f"integer has {len(match.group(1))} digits")
        value = Fraction(encoding)
    elif _FIXED_DECIMAL.fullmatch(encoding):
        sign = 1 if encoding[0] in "+-" else 0
        digits = len(encoding) - sign - 1
        if digits > PORTABLE_RATIONAL_DIGITS:
            _refuse(field, f"decimal has {digits} digits")
        value = Fraction(encoding)
    else:
        _refuse(
            field,
            "not a signed integer, integer quotient or fixed decimal spelled with ASCII digits",
        )
    if not _portable(value):
        _refuse(field, "the reduced denominator exceeds the ceiling")
    return value


def portable_rational(value: object, *, field: str = "value") -> Fraction:
    """Admit one serialized or in-process rational for *field*.

    Strings are admitted lexically before any conversion. A ``Fraction`` is
    returned as the same object; bounded ``int``, finite ``float`` and
    finite ``Decimal`` bind their exact value. ``bool`` is refused: ``True``
    is not a spelling of one.
    """
    if isinstance(value, Fraction):
        return value
    if isinstance(value, str):
        return _admit(value, field)
    if isinstance(value, bool):
        _refuse(field, "bool is not a rational encoding")
    if isinstance(value, int):
        if not _portable(value):
            _refuse(field, "integer exceeds the ceiling")
        return Fraction(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            _refuse(field, "binary64 must be finite")
        return Fraction(value)
    if isinstance(value, Decimal):
        if not value.is_finite():
            _refuse(field, "decimal must be finite")
        return Fraction(value)
    _refuse(field, f"{type(value).__name__} is not a rational encoding")


def _is_json_input(info: ValidationInfo) -> bool:
    return info.mode == "json" or info.context is _JSON_RATIONAL_CONTEXT


def _field(info: ValidationInfo) -> str:
    name = info.field_name
    if name is None:
        return "value"
    title = (info.config or {}).get("title")
    return f"{title}.{name}" if title else name


def _validate(value: object, info: ValidationInfo) -> Fraction:
    field = _field(info)
    if isinstance(value, float) and _is_json_input(info):
        _refuse(field, "a JSON floating literal was rounded to binary64; spell it as a string")
    return portable_rational(value, field=field)


def _declared_decimal(value: object, info: ValidationInfo) -> object:
    # Trusted declaration floats bind their shortest decimal spelling.
    # JSON and non-finite floats reach the shared coded refusal unchanged.
    if isinstance(value, float) and not _is_json_input(info) and math.isfinite(value):
        return Fraction(str(value))
    return value


def _serialize(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (Fraction, int)):
        _refuse("export", f"{type(value).__name__} is not a rational")
    if not _portable(value):
        _refuse("export", "a reduced component exceeds the ceiling")
    return str(value)


PortableRational = Annotated[
    Fraction,
    BeforeValidator(_validate),
    PlainSerializer(_serialize, return_type=str, when_used="always"),
]
"""Exact rational bounded on the wire; spelled by ``str(Fraction)`` in every dump mode."""

DeclaredRational = Annotated[PortableRational, BeforeValidator(_declared_decimal)]
"""``PortableRational`` for a declaration field: a trusted finite ``float`` binds the
decimal it was typed as (``0.05`` is ``1/20``), not its binary64 rational."""

__all__ = [
    "PORTABLE_RATIONAL_DIGITS",
    "RATIONAL_INVALID",
    "PortableRational",
    "portable_rational",
]
