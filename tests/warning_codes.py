"""Shared helper for asserting on coded library warnings in tests.

`pytest`'s `WarningMessage.message` is typed `Warning | str`, so a bare
`w.message.code` doesn't type-check; this narrows via `isinstance` once here
instead of scattering that narrowing (or a `# ty: ignore`) across every test
that asserts on a warning's code.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping

from increment.errors import IncrementWarning


def warning_codes(record: Iterable[warnings.WarningMessage]) -> list[str]:
    """Stable `.code` of every coded `IncrementWarning` in *record*, in order."""
    return [w.message.code for w in record if isinstance(w.message, IncrementWarning)]


def warning_context(record: Iterable[warnings.WarningMessage], code: str) -> Mapping[str, object]:
    """`.context` of the single warning in *record* carrying *code*."""
    contexts = [
        w.message.context
        for w in record
        if isinstance(w.message, IncrementWarning) and w.message.code == code
    ]
    assert len(contexts) == 1, f"expected exactly one {code!r} warning, found {len(contexts)}"
    return contexts[0]
