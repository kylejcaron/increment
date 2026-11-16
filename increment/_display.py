"""Small formatting helpers for concise public result representations."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
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


def format_coverage(level: float, alpha: float | None = None) -> str:
    """Keep the retained error budget visible when coverage rounds to zero or one."""
    percent = level * 100
    text = f"{percent:.2f}".rstrip("0").rstrip(".") + "%"
    if 0 < level < 1 and text in ("0%", "100%"):
        text = f"{percent:.15g}%"
        if text in ("0%", "100%"):
            text = f"{float(percent)!r}%"
    if level == 1 and alpha is not None and alpha > 0:
        text += f" (α={alpha!r})"
    return text


@dataclass(frozen=True)
class _ReadoutInterval:
    lower: float | None
    upper: float | None
    level: float | None
    alpha: float | None
    open_side: str | None
    one_sided: bool = False


def _readout_interval(row: Mapping[str, Any], *, absolute: bool = False) -> _ReadoutInterval:
    """Project a fixed guardrail's tested bound without changing its retained evidence."""

    def number(key: str) -> float | None:
        value = row.get(key)
        return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None

    def present(value: Any) -> bool:
        if value is None:
            return False
        try:
            return not bool(value != value)
        except TypeError:
            return False

    lower = number("abs_lb" if absolute else "lower")
    upper = number("abs_ub" if absolute else "higher")
    alpha = number("abs_alpha" if absolute else "coverage_alpha")
    if alpha is None and not absolute:
        alpha = number("alpha")
    level = math.fsum((1.0, -alpha)) if absolute and alpha is not None else number("level")
    raw_open = row.get("open_side") if not absolute else None
    open_side = raw_open if isinstance(raw_open, str) else None
    interval = _ReadoutInterval(lower, upper, level, alpha, open_side)
    alternative = row.get("alternative")
    if (
        not isinstance(row.get("role"), str)
        or row["role"] != "guardrail"
        or not isinstance(alternative, str)
        or alternative not in ("greater", "less")
        or not isinstance(row.get("inference"), str)
        or row["inference"] != "fixed"
    ):
        return interval
    greater = alternative == "greater"
    if (lower if greater else upper) is None:
        return interval
    if absolute:
        reference = row.get("abs_reference_kind")
        if (
            not isinstance(reference, str)
            or reference not in ("normal", "t", "score")
            or lower is None
            or upper is None
        ):
            return interval
        tail_alpha = None if alpha is None else alpha / 2.0
    else:
        region = row.get("relative_confidence_set")
        binomial = row.get("binomial_set")
        if region is not None and present(region):
            if (
                getattr(region, "geometry", None) != "one_sided"
                or getattr(region, "alternative", None) != alternative
                or len(getattr(region, "intervals", ())) != 1
                or region.intervals[0][1 if greater else 0] is not None
            ):
                return interval
            tail_alpha = region.alpha
        elif binomial is not None and present(binomial):
            if getattr(binomial, "geometry", None) != ("lower_bound" if greater else "upper_bound"):
                return interval
            tail_alpha = binomial.decision_alpha
        else:
            reference = row.get("reference_kind")
            if (
                present(row.get("confidence_set"))
                or present(row.get("relative_unavailable_reason"))
                or (isinstance(reference, str) and reference not in ("normal", "t"))
            ):
                return interval
            if alpha is None and level is not None:
                alpha = math.fsum((1.0, -level))
            # A closed fixed interval spends two tails; an already-open bound spends one.
            if open_side == ("upper" if greater else "lower"):
                tail_alpha = alpha
            elif open_side is None and lower is not None and upper is not None:
                tail_alpha = None if alpha is None else alpha / 2.0
            else:
                return interval
    if tail_alpha is None or not 0 < tail_alpha < 1:
        return interval
    return _ReadoutInterval(
        lower if greater else None,
        None if greater else upper,
        math.fsum((1.0, -tail_alpha)),
        tail_alpha,
        "upper" if greater else "lower",
        one_sided=True,
    )
