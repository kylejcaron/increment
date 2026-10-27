"""Enforce the no-database, no-process, no-network unit-test boundary."""

from __future__ import annotations

import multiprocessing.process
import socket
import subprocess
from collections.abc import Callable
from typing import Any, Literal, cast

import pytest

UNIT_EXCLUDING_MARKERS = frozenset(
    {
        "slow",
        "integration",
        "parameter_recovery",
        "numerical_certification",
        "examples",
        "stress",
        "benchmark",
    }
)

_current_unit_nodeid: str | None = None
_guards_installed = False
_BACKEND_FIXTURE_ATTRIBUTE = "__increment_backend_fixture__"
_CONNECT_ATTRIBUTE = "connect"
_CONNECT_EX_ATTRIBUTE = "connect_ex"


def is_unit_selected(item: pytest.Item) -> bool:
    return not any(item.get_closest_marker(marker) is not None for marker in UNIT_EXCLUDING_MARKERS)


def boundary_message(nodeid: str, action: str) -> str:
    return (
        f"{nodeid} {action}; unit tests run without databases, subprocesses or network. "
        "Mark the test with @pytest.mark.integration (still in the fast tier and PR CI), "
        "or test the pure function on in-memory data."
    )


def backend_fixture(
    *,
    scope: Literal["function", "class", "module", "package", "session"] = "function",
    **kwargs: Any,
) -> Callable[[Callable[..., Any]], Any]:
    """Declare a fixture whose setup opens a real backend connection."""

    def decorate(function: Callable[..., Any]) -> Any:
        setattr(function, _BACKEND_FIXTURE_ATTRIBUTE, True)
        fixture_factory = cast(Callable[..., Any], pytest.fixture)
        return fixture_factory(scope=scope, **kwargs)(function)

    return decorate


def _refuse(action: str) -> None:
    if _current_unit_nodeid is not None:
        pytest.fail(boundary_message(_current_unit_nodeid, action), pytrace=False)


def _is_internet_socket(sock: socket.socket) -> bool:
    return sock.family in (socket.AF_INET, socket.AF_INET6)


def install_guards() -> None:
    """Patch process-wide boundary operations; outside unit tests calls are unchanged."""
    import duckdb
    from ibis.backends.duckdb import Backend

    global _guards_installed
    if _guards_installed:
        return
    _guards_installed = True

    original_duckdb_connect = duckdb.connect

    def guarded_duckdb_connect(*args: Any, **kwargs: Any) -> Any:
        _refuse("opened a DuckDB connection")
        return original_duckdb_connect(*args, **kwargs)

    setattr(duckdb, _CONNECT_ATTRIBUTE, guarded_duckdb_connect)

    original_do_connect = Backend.do_connect

    def guarded_do_connect(self: Any, *args: Any, **kwargs: Any) -> Any:
        _refuse("opened a DuckDB connection")
        return original_do_connect(self, *args, **kwargs)

    Backend.do_connect = guarded_do_connect

    original_popen_init = subprocess.Popen.__init__

    def guarded_popen_init(self: subprocess.Popen[Any], *args: Any, **kwargs: Any) -> None:
        _refuse("started a subprocess")
        original_popen_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = guarded_popen_init

    original_process_start = multiprocessing.process.BaseProcess.start

    def guarded_process_start(self: Any, *args: Any, **kwargs: Any) -> Any:
        _refuse("started a process")
        return original_process_start(self, *args, **kwargs)

    multiprocessing.process.BaseProcess.start = guarded_process_start

    original_socket_connect = socket.socket.connect
    original_socket_connect_ex = socket.socket.connect_ex

    def guarded_socket_connect(self: socket.socket, address: Any) -> Any:
        if _is_internet_socket(self):
            _refuse("opened a network connection")
        return original_socket_connect(self, address)

    def guarded_socket_connect_ex(self: socket.socket, address: Any) -> Any:
        if _is_internet_socket(self):
            _refuse("opened a network connection")
        return original_socket_connect_ex(self, address)

    setattr(socket.socket, _CONNECT_ATTRIBUTE, guarded_socket_connect)
    setattr(socket.socket, _CONNECT_EX_ATTRIBUTE, guarded_socket_connect_ex)


def check_backend_fixtures(items: list[pytest.Item]) -> None:
    offenders: list[str] = []
    for item in items:
        if not is_unit_selected(item):
            continue
        fixtureinfo = getattr(item, "_fixtureinfo", None)
        if fixtureinfo is None:
            continue
        for name, definitions in fixtureinfo.name2fixturedefs.items():
            if definitions and getattr(
                definitions[-1].func, "__increment_backend_fixture__", False
            ):
                offenders.append(f"{item.nodeid}: backend fixture {name}")
    if offenders:
        details = "\n".join(f"  {offender}" for offender in sorted(offenders))
        raise pytest.UsageError(
            "Unit tests may not request backend fixtures:\n"
            f"{details}\n"
            "Unit tests run without databases, subprocesses or network. Mark the test with "
            "@pytest.mark.integration (still in the fast tier and PR CI), or test the pure "
            "function on in-memory data."
        )


def set_current_unit(item: pytest.Item | None) -> None:
    global _current_unit_nodeid
    _current_unit_nodeid = item.nodeid if item is not None and is_unit_selected(item) else None
