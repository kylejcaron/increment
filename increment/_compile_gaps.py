"""Report a query a warehouse's SQL dialect cannot express with one coded error.

A translation failure is recognised by where it was raised, never by backend
name: an Ibis SQL compiler had no rule for an operation, or SQLGlot could not
write the compiled query in the backend's dialect. Ibis and SQLGlot are read
from ``sys.modules`` and never imported here, so a run that did not load them
keeps its original exception.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from increment.errors import CapabilityError, raiser, refusals

_REFUSALS = refusals(
    CapabilityError,
    {
        "query.backend.operation_unsupported": (
            "the {backend} SQL dialect cannot express this analysis: {detail}; {route}"
        ),
    },
)
_raise = raiser(_REFUSALS)

_ROUTE = (
    "analyze from a dataframe with from_frame, or use a warehouse whose SQL dialect "
    "supports this operation"
)


def _innermost_self(exc: BaseException, owner: type) -> Any:
    """The ``self`` of the innermost traceback frame that is an ``owner`` instance."""
    found = None
    tb = exc.__traceback__
    while tb is not None:
        candidate = tb.tb_frame.f_locals.get("self")
        if isinstance(candidate, owner):
            found = candidate
        tb = tb.tb_next
    return found


def compile_gap_backend(exc: BaseException) -> str | None:
    """Name the backend whose SQL dialect could not translate ``exc``'s query.

    Returns ``None`` when ``exc`` is not a translation failure.
    """
    ibis_errors = sys.modules.get("ibis.common.exceptions")
    compilers = sys.modules.get("ibis.backends.sql.compilers.base")
    if ibis_errors is None or compilers is None:
        return None
    unsupported = ibis_errors.UnsupportedOperationError
    if isinstance(exc, (ibis_errors.OperationNotDefinedError, unsupported, NotImplementedError)):
        compiler = _innermost_self(exc, compilers.SQLGlotCompiler)
        if compiler is not None:
            return type(compiler).__module__.rpartition(".")[2]
    sqlglot_errors = sys.modules.get("sqlglot.errors")
    sql_backends = sys.modules.get("ibis.backends.sql")
    if (
        sqlglot_errors is not None
        and sql_backends is not None
        and isinstance(exc, unsupported)
        and isinstance(exc.__cause__, sqlglot_errors.UnsupportedError)
    ):
        backend = _innermost_self(exc, sql_backends.SQLBackend)
        if backend is not None:
            return str(backend.name)
    return None


def raise_if_compile_gap(exc: BaseException) -> None:
    """Raise ``query.backend.operation_unsupported`` from ``exc`` when it is a translation failure."""
    backend = compile_gap_backend(exc)
    if backend is None:
        return
    try:
        _raise(
            "query.backend.operation_unsupported",
            backend=backend,
            detail=str(exc),
            route=_ROUTE,
        )
    except CapabilityError as error:
        raise error from exc


@contextmanager
def translating_compile_gaps() -> Iterator[None]:
    """Report a translation failure raised inside the block with the coded refusal."""
    try:
        yield
    except Exception as exc:
        raise_if_compile_gap(exc)
        raise
