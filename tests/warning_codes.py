"""Shared helpers for asserting on, and allowing, coded library warnings in tests.

`pytest`'s `WarningMessage.message` is typed `Warning | str`, so a bare
`w.message.code` doesn't type-check; this narrows via `isinstance` once here
instead of scattering that narrowing (or a `# ty: ignore`) across every test
that asserts on a warning's code.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Mapping
from functools import wraps
from typing import Any

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


def allow_warning_codes(*codes: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Allow only coded warnings carrying one of *codes* during a test.

    Other warnings are re-emitted with their original category and location,
    so the repository's warning-as-error policy still applies.
    """

    def decorate(test: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(test)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            records: list[warnings.WarningMessage] = []
            try:
                with warnings.catch_warnings(record=True) as caught:
                    records = caught
                    warnings.simplefilter("always", IncrementWarning)
                    result = test(*args, **kwargs)
            finally:
                for record in records:
                    if getattr(record.message, "code", None) not in codes:
                        warnings.warn_explicit(
                            record.message,
                            record.category,
                            record.filename,
                            record.lineno,
                        )
            return result

        return wrapped

    return decorate
