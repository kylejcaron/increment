"""Shared Literal aliases for value domains used across many modules; import
these rather than respelling the values inline."""

from __future__ import annotations

from typing import Literal, get_args

Role = Literal["primary", "secondary", "guardrail", "unassigned"]
# A result row's role: a compiled plan role, or "exploratory" for a metric added after the
# plan was fixed (never a procedure role, so it never enters a compiled family).
RowRole = Literal["primary", "secondary", "guardrail", "unassigned", "exploratory"]
Alternative = Literal["two-sided", "greater", "less"]
ValueScale = Literal["relative", "absolute"]
PreferredDirection = Literal["increase", "decrease", "neutral"]
MultiplicityCorrection = Literal["none", "bh", "bonferroni", "e_bh"]
Correction = Literal["none", "bh", "bonferroni"]
ConversionInference = Literal["auto", "finite_sample"]

ALTERNATIVE_VALUES = frozenset(get_args(Alternative))
VALUE_SCALE_VALUES = frozenset(get_args(ValueScale))
