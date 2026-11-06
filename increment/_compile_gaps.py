"""Report a query a warehouse's SQL dialect cannot express with one coded error.

A translation failure is recognised by where it was raised, never by backend
name: an Ibis SQL compiler had no rule for an operation, or SQLGlot could not
write the compiled query in the backend's dialect. The classifier reads Ibis and
SQLGlot from ``sys.modules`` and never imports them, so a run that did not load
them keeps its original exception. Only :func:`load_dialect_compiler`, called
before compiling for a named dialect, imports an Ibis module.
"""

from __future__ import annotations

import importlib
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
        "query.backend.compiler_unavailable": (
            "the installed Ibis cannot write SQL for {backend!r}; {route}"
        ),
    },
)
_raise = raiser(_REFUSALS)

_ROUTE = (
    "analyze from a dataframe with from_frame, or see 'Other warehouses' in the data-model guide"
)
_COMPILER_ROUTE = (
    "pass backend= the name of an Ibis SQL backend, such as 'trino', "
    "or install an Ibis release that supports this one"
)


def load_dialect_compiler(dialect: str) -> None:
    """Load Ibis's SQL compiler for ``dialect``, as connecting to that backend does.

    Ibis compiles for a dialect by name only once its compiler module is imported.
    """
    package = "ibis.backends.sql.compilers"
    try:
        module = importlib.import_module(f"{package}.{dialect}")
    except ModuleNotFoundError as exc:
        if not (exc.name or "").startswith(f"{package}."):
            raise
        module = None
    if getattr(module, "compiler", None) is None:
        _raise("query.backend.compiler_unavailable", backend=dialect, route=_COMPILER_ROUTE)


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
