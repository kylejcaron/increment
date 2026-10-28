"""Trigger coverage validation shared across evidence and query adapters."""

from __future__ import annotations

from increment.errors import InvalidRequestError, RefusalSpec, refuse

_TRIGGER_ARM_MISSING = RefusalSpec(
    "query.integrity.trigger_arm_missing",
    InvalidRequestError,
    template="trigger {trigger!r} never fires in arm(s) {silent!r} (observed rates: {rates!r}); {route}",
)


def validate_trigger_fires_in_every_arm(rates: dict[str, float], trigger: str | None) -> None:
    """Refuse a declared trigger that never fires in a declared arm.

    A trigger that fires in one arm only is the treatment, not an eligibility
    rule - narrowing on it conditions the population on assignment and the
    comparison is no longer randomized.
    """
    silent = sorted(arm for arm, rate in rates.items() if rate == 0.0)
    if silent:
        refuse(
            _TRIGGER_ARM_MISSING,
            trigger=trigger,
            silent=tuple(silent),
            rates={arm: round(rate, 4) for arm, rate in rates.items()},
            route=(
                "analyze the assigned population, or declare an eligibility fact observable in every arm"
            ),
        )
