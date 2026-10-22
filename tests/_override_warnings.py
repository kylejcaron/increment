"""Test helpers for intentional call-time overrides of declared analysis methods."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import wraps
from typing import Any

from increment.errors import IncrementWarning

_OVERRIDE_CODE = "analysis.run.override_replaces_declared"


def allow_declared_overrides(test: Callable[..., Any]) -> Callable[..., Any]:
    """Allow only declared-override warnings during a comparison test.

    Other warnings are re-emitted with their original category and location,
    so the repository's warning-as-error policy still applies.
    """

    @wraps(test)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        records: list[warnings.WarningMessage]
        try:
            with warnings.catch_warnings(record=True) as caught:
                records = caught
                warnings.simplefilter("always", IncrementWarning)
                result = test(*args, **kwargs)
        finally:
            for record in records:
                if getattr(record.message, "code", None) != _OVERRIDE_CODE:
                    warnings.warn_explicit(
                        record.message,
                        record.category,
                        record.filename,
                        record.lineno,
                    )
        return result

    return wrapped
