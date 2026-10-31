"""Shared owner for a single evidence execution-context lifetime."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Protocol


class _EvidenceContext(Protocol):
    def evidence_scope(self) -> AbstractContextManager[object]: ...


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
