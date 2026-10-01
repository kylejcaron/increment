from __future__ import annotations

from increment.power.core import PowerResult


def available_mde(result: PowerResult) -> float:
    """The result's minimum detectable effect, which the caller expects to exist."""
    assert result.mde_unavailable_reason is None, result.mde_unavailable_reason
    assert result.mde_relative is not None
    return result.mde_relative
