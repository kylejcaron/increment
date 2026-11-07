"""Unit-day publications the parity matrix's artifact ingress shares between adoptions.

A publication is fixed by the definitions it is published from, the warehouse rows and the
extension kinds it carries (``publication_key``). Inside ``shared()``, the first adoption of a
key builds its warehouse, declares a native binding over it and publishes; that warehouse and
generation stay open until the block ends. Every adoption, the first included, opens its own
resolver through ``from_unit_day_artifact`` against the expected context of a native binding it
declares itself, so declaration warnings and refusals and the adoption checks run on each one.
A shared publication is read-only: when an adopted pair is released (or its adoption refuses),
every table and view of the warehouse must be unchanged since publication, or the release fails
and the publication is discarded. Outside ``shared()``, each adoption publishes its own.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import duckdb

from increment import builders
from increment.artifacts import artifact_context
from increment.query.session import WarehouseArtifactStore

from .cases import (
    Bound,
    _close_parity_binding,
    _owned,
    _publish,
    _publish_and_adopt,
    _track_cleanup,
    _track_connection,
)

if TYPE_CHECKING:
    from increment.query.artifact_contract import ArtifactContext
    from increment.semantics.artifact import UnitDayArtifactRef
    from increment.semantics.models import Definitions

Fingerprint = dict[str, tuple[Any, ...]]


@dataclass(frozen=True)
class _Publication:
    connection: Any
    ref: UnitDayArtifactRef
    fingerprint: Fingerprint


_SHARED: dict[str, _Publication] | None = None


def publication_key(
    definitions: Definitions, rows: list[dict[str, Any]], kinds: tuple[str, ...]
) -> str:
    """Everything a publication is built from: definitions, warehouse rows, extension kinds."""
    return hashlib.sha256(repr((definitions.model_dump_json(), rows, kinds)).encode()).hexdigest()


@contextmanager
def shared() -> Iterator[None]:
    """Publish each key once for the adoptions made inside this block."""
    global _SHARED
    _SHARED = {}
    try:
        yield
    finally:
        publications, _SHARED = _SHARED, None
        for publication in publications.values():
            publication.connection.disconnect()


def adopt(
    key: str,
    *,
    connect: Callable[[], Any],
    declare: Callable[[Any], Bound],
    kinds: tuple[str, ...],
) -> Bound:
    """Adopt the publication of *key*, publishing it from ``declare(connect())`` on first use.

    *declare* builds the native binding over a connection; *kinds* names the extensions."""
    if _SHARED is None:
        con = connect()
        try:
            native = declare(con)
        except BaseException:
            con.disconnect()
            raise
        return _publish_and_adopt(con, native, kinds)
    publication = _SHARED.get(key)
    if publication is None:
        publication, context = _publish_shared(connect, declare, kinds)
        _SHARED[key] = publication
    else:
        native = declare(publication.connection)
        try:
            context = artifact_context(*native)
        finally:
            _close_parity_binding(native)
    store = WarehouseArtifactStore(publication.connection, schema_name="artifacts")
    try:
        adopted = builders.from_unit_day_artifact(store, publication.ref, expected_context=context)
    except BaseException:
        _check_unchanged(key, publication)
        raise
    _owned(adopted).publication = (context, publication.ref)
    return _track_cleanup(adopted, partial(_check_unchanged, key, publication))


def _publish_shared(
    connect: Callable[[], Any], declare: Callable[[Any], Bound], kinds: tuple[str, ...]
) -> tuple[_Publication, ArtifactContext]:
    con = connect()
    try:
        native = declare(con)
    except BaseException:
        con.disconnect()
        raise
    try:
        context, _store, ref = _publish(con, native, kinds)
    except BaseException:
        _close_parity_binding(_track_connection(native, con))
        raise
    _close_parity_binding(native)
    return _Publication(con, ref, _fingerprint(con)), context


def _fingerprint(con: Any) -> Fingerprint:
    """Row count and order-independent content hash of every table, and every view's SQL."""
    views = con.raw_sql(
        "SELECT database_name, schema_name, view_name, sql FROM duckdb_views() WHERE NOT internal"
    ).fetchall()
    state: Fingerprint = {f"view {d}.{s}.{name}": (sql,) for d, s, name, sql in views}
    tables = con.raw_sql(
        "SELECT database_name, schema_name, table_name FROM duckdb_tables()"
    ).fetchall()
    for d, s, name in tables:
        state[f"table {d}.{s}.{name}"] = con.raw_sql(
            f'SELECT count(*), sum(hash(t)) FROM "{d}"."{s}"."{name}" AS t'
        ).fetchone()
    return state


def _check_unchanged(key: str, publication: _Publication) -> None:
    """Fail, and discard the publication, when its warehouse changed since publication."""
    try:
        current = _fingerprint(publication.connection)
    except duckdb.Error as error:
        current = {"connection": (f"unreadable: {error}",)}
    before = publication.fingerprint
    changed = sorted(
        name for name in before.keys() | current.keys() if before.get(name) != current.get(name)
    )
    if not changed:
        return
    if _SHARED is not None and _SHARED.get(key) is publication:
        del _SHARED[key]
    publication.connection.disconnect()
    raise AssertionError(
        f"a parity leg changed a shared unit-day publication ({', '.join(changed)}); "
        "shared publications are read-only"
    )
