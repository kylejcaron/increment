"""Declaration-only gates for a persistent holdout comparison."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, assert_never

from increment._holdout_records import HoldoutReason, check_registration
from increment.errors import InvalidRequestError, RefusalSpec, UnsupportedRequestError, refuse

if TYPE_CHECKING:
    from datetime import datetime

    from increment._holdout_records import HoldoutStudy, HoldoutWindow, LookRequest


_NOT_FINAL_CUT = RefusalSpec(
    "holdout.not_final_cut",
    InvalidRequestError,
    template=(
        "Window {window_id!r} has a registered data cut of {final_as_of!s}, not {as_of!s}."
        " Use the registered cut for inference or request a description."
    ),
)
_SEQUENTIAL_UNSUPPORTED = RefusalSpec(
    "holdout.sequential_unsupported",
    UnsupportedRequestError,
    template=(
        "Window {window_id!r} supports only one fixed-horizon decision."
        " Request a description or use the registered final data cut."
    ),
)


def check_design(study: HoldoutStudy) -> HoldoutReason | None:
    """Diagnose whether the original assignment supports an independent contrast."""
    if (
        study.allocation_scheme != "iid_bernoulli"
        or study.entry_basis != "pretreatment_first_entry"
        or study.no_interference is not True
        or study.design_ref is None
    ):
        return HoldoutReason(code="portfolio.impact.unsupported_design")
    return None


def check_look(
    study: HoldoutStudy, window: HoldoutWindow, *, as_of: datetime, request: LookRequest
) -> Literal["descriptive", "final"]:
    """Admit inference only on the original slot's exact evidence cut."""
    finding = check_registration(study, window)
    if request == "describe":
        return "descriptive"
    if finding is not None:
        finding.raise_error()
    if request == "sequential":
        refuse(_SEQUENTIAL_UNSUPPORTED, window_id=window.window_id)
    if request != "fixed_horizon":
        assert_never(request)
    final_as_of = window.terms.final_as_of
    if as_of != final_as_of:
        refuse(_NOT_FINAL_CUT, window_id=window.window_id, as_of=as_of, final_as_of=final_as_of)
    return "final"
