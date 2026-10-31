from __future__ import annotations

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


def refuse_store(code: str, message: str, **context: object) -> NoReturn:
    refuse(_REFUSALS[code], message=message, **context)
