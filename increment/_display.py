"""Small formatting helpers for concise public result representations."""

import math
from collections.abc import Mapping
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def format_mapping(value: Mapping[Any, Any]) -> str:
    """Render mapping contents recursively using ordinary mapping syntax."""
    return repr(_plain(value))


def _format_number(value: Fraction | int | float | Decimal, digits: int) -> str:
    if isinstance(value, float):
        return format(value, f".{digits}g")
    with localcontext() as context:
        context.prec = digits
        if isinstance(value, Fraction):
            number = Decimal(value.numerator) / Decimal(value.denominator)
        else:
            number = Decimal(value)
            as_float = float(number)
            if math.isfinite(as_float):
                return format(as_float, f".{digits}g")
        return format(number.normalize(context), f".{digits}g")


def _effect_number(value: float, *, relative: bool) -> float | Decimal:
    """Scale a lift for display without overflowing finite float effects."""
    return Decimal(value).scaleb(2) if relative else value


def format_effect(value: float, *, relative: bool, digits: int = 4) -> str:
    """Format a lift as a percent, or an absolute effect in its original units."""
    number = _effect_number(value, relative=relative)
    return f"{_format_number(number, digits)}%" if relative else _format_number(number, digits)


def format_effect_interval(
    lower: float | None,
    upper: float | None,
    *,
    relative: bool,
    digits: int = 4,
) -> str:
    """Format lift interval endpoints using the row's declared effect scale."""
    suffix = "%" if relative else ""
    return interval_text(
        None
        if lower is None
        else f"{_format_number(_effect_number(lower, relative=relative), digits)}{suffix}",
        None
        if upper is None
        else f"{_format_number(_effect_number(upper, relative=relative), digits)}{suffix}",
    )


def interval_text(lower: str | None, upper: str | None) -> str:
    """Bracket formatted endpoints; an infinite end (``None``) is open: ``[a, ∞)``, ``(−∞, b]``."""
    opening = "(−∞" if lower is None else f"[{lower}"
    closing = "∞)" if upper is None else f"{upper}]"
    return f"{opening}, {closing}"


def format_interval(
    lower: Fraction | int | float | None,
    upper: Fraction | int | float | None,
    *,
    digits: int = 4,
    empty: bool = False,
) -> str:
    """Format interval endpoints compactly; ``None`` denotes an infinite end."""
    if empty:
        return "∅"
    return interval_text(
        None if lower is None else _format_number(lower, digits),
        None if upper is None else _format_number(upper, digits),
    )
