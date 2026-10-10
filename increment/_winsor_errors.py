"""Shared refusal contract for winsor state and numerical kernels."""

from typing import NoReturn

from increment.errors import CapabilityError, InvalidRequestError, RefusalSpec, refuse

_REFUSALS = {
    name: RefusalSpec(
        f"estimation.winsor.{name}",
        CapabilityError
        if name
        in {"raw_state_required", "support_required", "design_unsupported", "rank_size_unsupported"}
        else InvalidRequestError,
        lambda *, reason: reason,
    )
    for name in (
        "raw_state_required",
        "rank_size_unsupported",
        "support_required",
        "design_unsupported",
        "pool_mismatch",
        "support_violation",
        "invalid_state",
        "posterior_unavailable",
        "reinversion_required",
        "density_required",
        "pilot_negative_outcome",
        "cutoff_in_zero_atom",
        "pilot_degenerate",
        "density_unresolved",
        "studentization_degenerate",
        "endpoint_unrepresentable",
    )
}


def winsor_refuse(code: str, reason: str) -> NoReturn:
    refuse(_REFUSALS[code], reason=reason)
