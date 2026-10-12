"""Chronological ordering for string day-axis labels."""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterable
from typing import Any

from increment.errors import CapabilityError, RefusalSpec, refuse

_ISO_DATE_LABEL = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PREFIXED_INTEGER_LABEL = re.compile(r"^(?P<prefix>[^+\-0-9]*)(?P<sign>[+-]?)(?P<digits>\d+)$")

_DAY_AXIS_LABEL_UNORDERABLE = RefusalSpec(
    "frame.asof.day_axis_unorderable",
    CapabilityError,
    template="cannot order day-axis label(s) {labels!r} chronologically -- string day-axis labels are ordered only when every distinct label is an unambiguous ISO-8601 date ('YYYY-MM-DD', lexicographic order is chronological) or every label shares one consistent '<prefix><signed integer>' shape (e.g. 'd1', 'd-1'), with no two distinct labels parsing to the same integer. Declare an explicit date or numeric day axis instead of a free-form string label.",
)


def _is_iso_date_label(value: str) -> bool:
    if not _ISO_DATE_LABEL.match(value):
        return False
    try:
        dt.date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _day_axis_label_order(labels: Iterable[Any]) -> dict[Any, tuple[Any, ...]]:
    """A chronologically-justified rank for every distinct day-axis label.

    This is a COLLECTION-level decision - it depends on the full set of
    distinct labels, not any one value in isolation - because the only
    orders this accepts are ones the ENTIRE set jointly justifies. Guessing
    at anything else (raw string order misreads ``"01/01/2026"`` as before
    ``"12/31/2025"``; naive natural-sort digit-chunking misreads ``"d0"``
    as after ``"d-1"``) is exactly the class of bug this function exists
    to prevent, so it refuses instead of guessing.

    Date/Datetime/numeric labels already compare correctly on their own.
    A string label orders correctly only when the whole set is either:

    - every label an unambiguous ISO-8601 date (``"YYYY-MM-DD"``, where
      lexicographic order IS chronological), or
    - every label one consistent ``<prefix><signed integer>`` shape (e.g.
      ``"d1"``, ``"d-1"``, ``"day_10"``), ranked by the parsed integer -
      REFUSED if two distinct labels parse to the same integer (``"d1"``
      and ``"d01"``), since their cumulative order would then silently
      depend on backend/input row order.

    Any other shape refuses, naming the offending label(s).
    """
    # Reject bytes-like labels before deduplication: they cannot justify a
    # chronological order, and bytearray/memoryview values would otherwise
    # fail with an unhelpful hashing error.
    materialized = list(labels)
    if any(isinstance(v, bytes | bytearray | memoryview) for v in materialized):
        refuse(_DAY_AXIS_LABEL_UNORDERABLE, labels=tuple(sorted(map(str, materialized))))
    distinct = list(dict.fromkeys(materialized))
    strings = [v for v in distinct if isinstance(v, str)]
    if not strings:
        return {v: (v,) for v in distinct}
    if len(strings) != len(distinct):
        refuse(_DAY_AXIS_LABEL_UNORDERABLE, labels=tuple(sorted(map(str, distinct))))

    if all(_is_iso_date_label(v) for v in strings):
        return {v: (v,) for v in distinct}

    matches = {v: _PREFIXED_INTEGER_LABEL.match(v) for v in strings}
    if all(matches.values()):
        prefixes = {m.group("prefix") for m in matches.values() if m is not None}
        if len(prefixes) == 1:
            parsed = {
                v: (-1 if m.group("sign") == "-" else 1) * int(m.group("digits"))
                for v, m in matches.items()
                if m is not None
            }
            by_value: dict[int, list[str]] = {}
            for v, n in parsed.items():
                by_value.setdefault(n, []).append(v)
            collisions = [vs for vs in by_value.values() if len(vs) > 1]
            if not collisions:
                return {v: (parsed[v],) for v in distinct}
            offending = tuple(sorted(v for vs in collisions for v in vs))
            refuse(_DAY_AXIS_LABEL_UNORDERABLE, labels=offending)

    refuse(_DAY_AXIS_LABEL_UNORDERABLE, labels=tuple(sorted(strings)))
