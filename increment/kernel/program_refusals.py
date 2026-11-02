from __future__ import annotations

from collections.abc import Sequence
from typing import NoReturn

from increment.errors import InvalidRequestError, RefusalSpec, refusals, refuse


def _render(message: str, **_: object) -> str:
    return message


_CODES = (
    "store.run.identity_incomplete",
    "store.run.timestamp_invalid",
    "store.run.roster_mismatch",
    "store.run.identity_mismatch",
    "store.run.provenance_mismatch",
    "store.run.scope_manifest_invalid",
    "store.run.scope_mapping_invalid",
    "store.run.metadata_mismatch",
    "store.run.empty",
    "store.run.duplicate",
    "store.run.corrupt",
    "store.run.not_found",
    "store.cell.not_found",
    "store.history.ambiguous",
    "store.decision.timestamp_invalid",
    "store.decision.reference_invalid",
    "store.decision.log_corrupt",
    "store.decision.event_invalid",
    "store.decision.outcome_mismatch",
    "store.decision.schema_unsupported",
)
_REFUSALS = refusals(
    InvalidRequestError,
    {code: RefusalSpec(code, InvalidRequestError, render=_render) for code in _CODES},
)


def _render_manifest_unsupported(*, found: Sequence[str], supported: Sequence[str]) -> str:
    return (
        f"This saved run uses format {', '.join(found)}, which this version of Increment "
        f"cannot read; it reads {', '.join(supported)}. Upgrade Increment to open the run."
    )


_MANIFEST_UNSUPPORTED = RefusalSpec(
    "store.run.manifest_unsupported", InvalidRequestError, render=_render_manifest_unsupported
)


def refuse_store(code: str, message: str, **context: object) -> NoReturn:
    refuse(_REFUSALS[code], message=message, **context)


def refuse_unsupported_run_manifest(found: Sequence[str], supported: Sequence[str]) -> NoReturn:
    refuse(_MANIFEST_UNSUPPORTED, found=list(found), supported=list(supported))
