"""Shared owner for a single evidence execution-context lifetime."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from typing import Any, Protocol

_ACTIVE_EXECUTION_CONTEXT: ContextVar[Any | None] = ContextVar(
    "_ACTIVE_EXECUTION_CONTEXT", default=None
)


class _EvidenceContext(Protocol):
    def evidence_scope(self) -> AbstractContextManager[object]: ...


@contextmanager
def execution_context_scope(context: Any) -> Iterator[None]:
    """Make ``context`` the ambient run context for operations entered inside it."""
    token = _ACTIVE_EXECUTION_CONTEXT.set(context)
    try:
        yield
    finally:
        _ACTIVE_EXECUTION_CONTEXT.reset(token)


def current_execution_context() -> Any | None:
    """The run context an enclosing shared capture made ambient, if any."""
    return _ACTIVE_EXECUTION_CONTEXT.get()


@contextmanager
def owned_execution_context[ContextT: _EvidenceContext](
    make_context: Callable[[], ContextT],
    *,
    execution_context_scope: Callable[[ContextT], AbstractContextManager[object]],
) -> Iterator[ContextT]:
    """Own the lifetime and ambient binding for one evidence context."""
    context = make_context()
    with context.evidence_scope(), execution_context_scope(context):
        yield context
