"""WarehouseSession: one ibis connection's fact-table cache and TEMP-table
mechanics, shared by every warehouse-backed facade (`Analysis`, `Report`).
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, cast

import ibis
import ibis.expr.types as ir
import sqlglot
from ibis import Table
from ibis.backends.sql import SQLBackend

from increment.errors import (
    CapabilityError,
    CodedError,
    IncrementWarning,
    InvalidRequestError,
    RefusalSpec,
    WarningSpec,
    raiser,
    refusals,
    refuse,
    warn,
)
from increment.query.artifact_contract import (
    ImmutableArtifactSnapshot,
    validate_artifact_context,
    validate_snapshot_identity,
    validate_trusted_ref,
)
from increment.query.artifact_digest import (
    _SQL_DIGEST_BACKENDS,
    BUCKETED_DIGEST_FORMAT,
    SQL_DIGEST_FORMAT,
    digest_relation_bucketed,
    digest_relation_sql_v2,
    digest_relation_stream,
    stream_canonical_rows,
)
from increment.query.artifact_digest import REFUSALS as DIGEST_REFUSALS
from increment.query.fact_resolution import _dim_joined_fact_table
from increment.query.schemas import (
    ARTIFACT_BASE_RELATION_ROLES,
    ARTIFACT_RELATION_PRIMARY_KEYS,
    ARTIFACT_RELATION_SCHEMAS,
)
from increment.semantics.artifact import (
    ArtifactContext,
    ArtifactRelationRef,
    RelationLocator,
    UnitDayArtifactManifest,
    UnitDayArtifactRef,
)
from increment.semantics.models import Definitions, FactSource

_ARTIFACT_STORE_NAMESPACE = RefusalSpec(
    "artifact.store.namespace",
    CapabilityError,
    lambda *, message, **_: message,
)


Renderer = Callable[..., str]


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "query.session.snapshot.source_missing": RefusalSpec(
            "query.session.snapshot.source_missing",
            CapabilityError,
            template="source was not included in the pinned execution; a live fallback is forbidden",
        ),
        "query.session.snapshot.materialization_failed": RefusalSpec(
            "query.session.snapshot.materialization_failed",
            CapabilityError,
            template="source snapshot requires warehouse TEMP tables; live fallback is forbidden",
        ),
        "query.session.materialization_namespace": RefusalSpec(
            "query.session.materialization_namespace",
            CapabilityError,
            template="TEMP namespace could not be resolved for {backend}",
        ),
        "query.session.artifact_publication.handle_closed": "publication handle is closed",
        "query.session.warehouse_artifact.unknown_relation_role": "unknown artifact relation role {role!r}",
        "query.session.warehouse_artifact.publication_handle_sealed": "publication handle is sealed",
        "query.session.warehouse_artifact.manifest_bound_publication": "manifest is not bound to this publication",
        "query.session.warehouse_artifact.manifest_context_differs": "manifest context differs from publication context",
        "query.session.warehouse_artifact.manifest_digest_does": "manifest digest does not match manifest body",
        "query.session.warehouse_artifact.manifest_relation_locators": "manifest relation locators must be unique",
        "query.session.warehouse_artifact.manifest_references_unwritten": "manifest references an unwritten relation",
        "query.session.warehouse_artifact.publication_state_unknown": RefusalSpec(
            "query.session.warehouse_artifact.publication_state_unknown",
            InvalidRequestError,
            template=(
                "publication ended without a manifest and its cleanup did not complete; "
                "the retained relations and the recovery route are in the refusal context"
            ),
            keys=frozenset({"relations", "artifact_id", "generation_id", "tombstoned", "route"}),
        ),
        "query.session.warehouse_artifact.publication_cleanup_incomplete": RefusalSpec(
            "query.session.warehouse_artifact.publication_cleanup_incomplete",
            InvalidRequestError,
            template=(
                "publication ended without a manifest and some relations could not be "
                "dropped; the manifest insert never began, so drop the listed relations "
                "by name"
            ),
            keys=frozenset({"relations", "artifact_id", "generation_id", "route"}),
        ),
        "query.session.warehouse_artifact.unknown_digest_format": "unknown artifact relation digest format {digest_format!r}",
    },
)
_raise = raiser(_REFUSALS)

_WARNINGS: dict[str, WarningSpec] = {}


def _register_warning(code: str, warning_type: type[IncrementWarning], render: Renderer) -> None:
    _WARNINGS[code] = WarningSpec(code, warning_type, render)


def _warn(code: str, /, *, stacklevel: int = 2, **context: object) -> None:
    warn(_WARNINGS[code], stacklevel=stacklevel + 1, context=context)


_register_warning(
    "query.session.materialization_degraded",
    IncrementWarning,
    lambda *, name: (
        f"could not materialize {name!r} (CREATE TEMP failed or its physical "
        "namespace could not be resolved); this session will re-scan raw "
        "events instead. Grant CREATE TEMP TABLE and namespace access, "
        "or pass store='none' to disable materialization."
    ),
)


def _validate_artifact_schema(table: ir.Table, role: str) -> None:
    type_names = {
        "STRING": "string",
        "INT64": "int64",
        "FLOAT64": "float64",
        "DATE": "date",
        "TIMESTAMP_UTC_US": "timestamp('UTC')",
        "BOOLEAN": "boolean",
    }
    expected_schema = ibis.schema(
        {
            field: type_names[type_tag]
            for field, type_tag, _nullable in ARTIFACT_RELATION_SCHEMAS[role]
        }
    ).to_pyarrow()
    actual_schema = table.schema().to_pyarrow()
    for field in expected_schema:
        if field.name not in actual_schema.names:
            stored = None
        elif actual_schema.field(field.name).type != field.type:
            stored = str(actual_schema.field(field.name).type)
        else:
            continue
        refuse(
            DIGEST_REFUSALS["artifact.digest.type"],
            field=field.name,
            constraint="stored columns need their canonical physical type",
            value={"expected": str(field.type), "stored": stored},
            route="republish the relation with the canonical artifact column types",
            role=role,
        )


@dataclass(frozen=True, slots=True)
class _StoredRelation:
    ref: ArtifactRelationRef


@dataclass(slots=True)
class _StoredGeneration:
    artifact_id: uuid.UUID
    generation_id: uuid.UUID
    context: ArtifactContext
    relations: dict[str, _StoredRelation]
    manifest: UnitDayArtifactManifest | None = None
    ref: UnitDayArtifactRef | None = None
    manifest_insert_started: bool = False
    # Relations whose create may have landed but that never became registered
    # (create/read/digest failed and the immediate drop failed too): abort still owns them.
    unregistered: dict[str, RelationLocator] = dataclass_field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _AbortOutcome:
    """What an aborted, never-published generation left behind.

    `retained` holds catalog/schema-qualified names still in the warehouse."""

    tombstoned: bool
    retained: tuple[str, ...]
    tombstone_error: BaseException | None
    erase_error: BaseException | None = None


def _create_table_namespace(
    con: Any,
    catalog: str | None,
    schema: str | None,
    database: str | tuple[str, str] | None,
) -> str | tuple[str, str] | None:
    """The `database=` value for `create_table` on this backend.

    Snowflake needs a quoted catalog-plus-schema string. BigQuery needs the bare
    dataset: Ibis 12 otherwise reuses ``project.dataset`` as the dataset in DDL.
    The BigQuery table name carries the explicit project instead.
    """
    if catalog is None or schema is None:
        return database
    backend = getattr(con, "name", None)
    if backend == "snowflake":
        return ".".join(
            sqlglot.exp.to_identifier(part, quoted=True).sql("snowflake")
            for part in (catalog, schema)
        )
    if backend == "bigquery":
        return schema
    return database


class _ArtifactPublication:
    def __init__(self, store: WarehouseArtifactStore, generation: _StoredGeneration) -> None:
        self._store = store
        self._generation = generation
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            _raise("query.session.artifact_publication.handle_closed")

    @property
    def artifact_id(self) -> uuid.UUID:
        self._ensure_open()
        return self._generation.artifact_id

    @property
    def generation_id(self) -> uuid.UUID:
        self._ensure_open()
        return self._generation.generation_id

    def write_relation(self, role: str, table: ir.Table) -> ArtifactRelationRef:
        self._ensure_open()
        return self._store._write_relation(self._generation, role, table)

    def publish_manifest(self, manifest: UnitDayArtifactManifest) -> UnitDayArtifactRef:
        self._ensure_open()
        return self._store._publish_manifest(self._generation, manifest)


class WarehouseArtifactStore:
    """An immutable artifact store backed by one SQL connection.

    Publication persists generation-owned warehouse relations. A snapshot pins
    the manifest and relation references, then lazily creates private temporary
    copies during verification. Verified copies isolate reads from later changes
    and are dropped when the snapshot closes. A declared schema is created on
    demand before store-owned tables are written.
    """

    _INDEX_TABLE_COLUMNS = (
        "artifact_id",
        "generation_id",
        "manifest_name",
        "manifest_sha256",
        "manifest_json",
    )
    _DROPPED_TABLE_COLUMNS = ("artifact_id", "generation_id", "dropped_at")

    def __init__(
        self,
        con: SQLBackend,
        *,
        catalog: str | None = None,
        schema_name: str | None = None,
        namespace: str | None = None,
    ) -> None:
        required = (
            "to_pyarrow",
            "execute",
            "create_table",
            "table",
            "drop_table",
            "insert",
        )
        if any(not callable(getattr(con, name, None)) for name in required):
            raise CapabilityError(
                "artifact store requires a SQL backend with immutable read support",
                code="artifact.store.atomic",
                context={"backend": type(con).__name__},
            )
        self._con = con
        self._catalog = catalog
        self._schema = schema_name or namespace
        self._database: str | tuple[str, str] | None = (
            (self._catalog, self._schema)
            if self._catalog is not None and self._schema is not None
            else self._schema
        )
        self._create_database = _create_table_namespace(
            self._con, self._catalog, self._schema, self._database
        )
        self._meta_name = "ud_manifest_index"
        self._dropped_name = "ud_manifest_dropped"
        self._generations: dict[tuple[uuid.UUID, uuid.UUID], _StoredGeneration] = {}
        self._validate_namespace()
        self._ensure_metadata_table()
        self._ensure_dropped_table()

    def _validate_namespace(self) -> None:
        try:
            RelationLocator(
                catalog=self._catalog,
                schema=self._schema,
                name=self._meta_name,
            )
        except ValueError as exc:
            refuse(
                _ARTIFACT_STORE_NAMESPACE,
                message=(
                    f"artifact store namespace {self._database!r} is not a valid "
                    f"artifact identifier: {exc}"
                ),
                catalog=self._catalog,
                schema=self._schema,
            )

    def _create_table_name(self, name: str) -> str:
        if self._con.name == "bigquery" and self._catalog is not None and self._schema is not None:
            return sqlglot.table(name, db=self._schema, catalog=self._catalog, quoted=True).sql(
                "bigquery"
            )
        return name

    def _ensure_namespace(self) -> None:
        if self._schema is None:
            return
        create_database = getattr(self._con, "create_database", None)
        if not callable(create_database):
            return
        try:
            create_database(self._schema, catalog=self._catalog, force=False)
        except Exception:
            # Backends commonly report an existing schema as an error. Probe
            # it before allowing metadata DDL to surface a real namespace error.
            list_tables = getattr(self._con, "list_tables", None)
            if not callable(list_tables):
                return
            try:
                list_tables(database=self._database)
            except Exception:
                refuse(
                    _ARTIFACT_STORE_NAMESPACE,
                    message=f"artifact store namespace {self._database!r} does not exist",
                    catalog=self._catalog,
                    schema=self._schema,
                )

    def _ensure_metadata_table(self) -> None:
        self._ensure_namespace()
        try:
            table = self._con.table(self._meta_name, database=self._database)
        except Exception:
            try:
                self._con.create_table(
                    self._create_table_name(self._meta_name),
                    schema=ibis.schema(dict.fromkeys(self._INDEX_TABLE_COLUMNS, "string")),
                    database=cast("Any", self._create_database),
                    overwrite=False,
                )
            except Exception:
                # A concurrent store may have created the table; re-read and validate.
                # A table that is still unreadable fails closed.
                try:
                    table = self._con.table(self._meta_name, database=self._database)
                except Exception:
                    refuse(
                        _ARTIFACT_STORE_NAMESPACE,
                        message=(
                            f"artifact store manifest index {self._meta_name!r} does not "
                            f"exist in namespace {self._database!r} and could not be "
                            "created; grant CREATE on the namespace or create the table"
                        ),
                        catalog=self._catalog,
                        schema=self._schema,
                        table=self._meta_name,
                    )
            else:
                # Re-read so the backend-assigned column order is recorded.
                table = self._con.table(self._meta_name, database=self._database)
        self._meta_columns: tuple[str, ...] = self._validated_columns(
            table, self._meta_name, self._INDEX_TABLE_COLUMNS
        )

    def _ensure_dropped_table(self) -> None:
        try:
            table = self._con.table(self._dropped_name, database=self._database)
        except Exception:
            try:
                self._con.create_table(
                    self._create_table_name(self._dropped_name),
                    schema=ibis.schema(
                        {
                            "artifact_id": "string",
                            "generation_id": "string",
                            "dropped_at": "string",
                        }
                    ),
                    database=cast("Any", self._create_database),
                    overwrite=False,
                )
            except Exception:
                # A concurrent store may have created the table, or the one-time
                # migration has not run. Re-read and validate; a missing table fails
                # closed so readers never ignore tombstones they cannot see.
                try:
                    table = self._con.table(self._dropped_name, database=self._database)
                except Exception:
                    refuse(
                        _ARTIFACT_STORE_NAMESPACE,
                        message=(
                            f"artifact store tombstone table {self._dropped_name!r} does "
                            f"not exist in namespace {self._database!r}; an administrator "
                            "must run the one-time migration that creates it before any "
                            "store constructs against this namespace"
                        ),
                        catalog=self._catalog,
                        schema=self._schema,
                    )
            else:
                # Re-read the freshly created table so its actual (backend
                # -assigned) column order is recorded, not just assumed.
                table = self._con.table(self._dropped_name, database=self._database)
        self._dropped_columns: tuple[str, ...] = self._validated_columns(
            table, self._dropped_name, self._DROPPED_TABLE_COLUMNS
        )

    def _validated_columns(
        self, table: ir.Table, name: str, expected_columns: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Return the physical column order of an all-string store table, or refuse."""
        actual = dict(table.schema().items())
        expected = set(expected_columns)
        missing = sorted(expected - set(actual))
        extra = sorted(set(actual) - expected)
        wrong_type = sorted(col for col in expected & set(actual) if not actual[col].is_string())
        if missing or extra or wrong_type:
            refuse(
                _ARTIFACT_STORE_NAMESPACE,
                message=(
                    f"artifact store table {name!r} in namespace {self._database!r} has "
                    "an unexpected schema; recreate or migrate it to the all-string "
                    f"columns {expected_columns!r}"
                ),
                catalog=self._catalog,
                schema=self._schema,
                table=name,
                missing_columns=tuple(missing),
                extra_columns=tuple(extra),
                wrong_type_columns=tuple(wrong_type),
            )
        return tuple(actual)

    def _is_dropped(self, artifact_id: uuid.UUID, generation_id: uuid.UUID) -> bool:
        dropped = self._con.table(self._dropped_name, database=self._database)
        matched = dropped.filter(
            (dropped.artifact_id == str(artifact_id))
            & (dropped.generation_id == str(generation_id))
        ).limit(1)
        return bool(self._con.to_pyarrow(matched).to_pylist())

    def _refuse_if_dropped(self, artifact_id: uuid.UUID, generation_id: uuid.UUID) -> None:
        if not self._is_dropped(artifact_id, generation_id):
            return
        from increment.query.artifact_contract import ArtifactContractError

        raise ArtifactContractError(
            "artifact.generation.dropped",
            "artifact generation has been durably dropped",
            artifact_id=str(artifact_id),
            generation_id=str(generation_id),
        )

    @property
    def visible_manifests(self) -> tuple[UnitDayArtifactRef, ...]:
        from increment.query.artifact_contract import ArtifactContractError

        metadata = self._con.table(self._meta_name, database=self._database)
        dropped = self._con.table(self._dropped_name, database=self._database)
        visible = metadata.anti_join(dropped, ["artifact_id", "generation_id"])
        rows = self._con.to_pyarrow(visible).to_pylist()
        seen: set[tuple[str, str]] = set()
        refs: list[UnitDayArtifactRef] = []
        for row in rows:
            key = (row["artifact_id"], row["generation_id"])
            if key in seen:
                raise ArtifactContractError(
                    "artifact.manifest.invalid",
                    "manifest index has duplicate rows for one generation",
                    artifact_id=row["artifact_id"],
                    generation_id=row["generation_id"],
                )
            seen.add(key)
            refs.append(
                UnitDayArtifactRef(
                    artifact_id=uuid.UUID(row["artifact_id"]),
                    generation_id=uuid.UUID(row["generation_id"]),
                    manifest=RelationLocator(
                        catalog=self._catalog, schema=self._schema, name=row["manifest_name"]
                    ),
                    manifest_sha256=row["manifest_sha256"],
                )
            )
        refs.sort(key=lambda ref: (ref.artifact_id.int, ref.generation_id.int))
        return tuple(refs)

    def validate_locator(
        self,
        locator: RelationLocator,
        *,
        artifact_id: uuid.UUID,
        generation_id: uuid.UUID,
        role: str,
    ) -> None:
        if locator.catalog != self._catalog or locator.schema_name != self._schema:
            from increment.query.artifact_contract import ArtifactContractError

            raise ArtifactContractError(
                "artifact.identifier.unsafe",
                "artifact relation is outside the store namespace",
                context={
                    "namespace": self._schema,
                    "expected_catalog": self._catalog,
                    "rejected_catalog": locator.catalog,
                    "rejected_schema": locator.schema_name,
                },
            )
        prefix = f"ud_{artifact_id.hex[:12]}_{generation_id.hex[:12]}_"
        if not locator.name.startswith(prefix):
            from increment.query.artifact_contract import ArtifactContractError

            raise ArtifactContractError(
                "artifact.refresh.invalid_ref",
                "artifact relation is outside the pinned generation",
                context={"name": locator.name},
            )
        expected_suffix = "manifest" if role == "manifest" else role
        remainder = locator.name[len(prefix) :]
        valid_suffix = (
            remainder == expected_suffix
            if role == "manifest"
            else remainder == expected_suffix
            or (
                remainder.startswith(f"{expected_suffix}_")
                and remainder[len(expected_suffix) + 1 :].isdigit()
            )
        )
        if not valid_suffix:
            from increment.query.artifact_contract import ArtifactContractError

            raise ArtifactContractError(
                "artifact.refresh.invalid_ref",
                "artifact relation role does not match locator",
                context={"role": role},
            )

    def begin_publication(
        self,
        *,
        expected_context: ArtifactContext,
        refresh_of: UnitDayArtifactRef | None = None,
    ):
        validate_artifact_context(expected_context)
        if refresh_of is not None:
            validate_trusted_ref(refresh_of)
            with self.open_snapshot(refresh_of) as snapshot:
                prior = snapshot.read_manifest(
                    refresh_of.manifest, expected_sha256=refresh_of.manifest_sha256
                )
                if prior.context.sha256 != expected_context.sha256:
                    from increment.query.artifact_contract import ArtifactContractError

                    raise ArtifactContractError(
                        "artifact.refresh.context_mismatch",
                        "refresh context differs from fixed prior generation",
                        context={"artifact_id": str(refresh_of.artifact_id)},
                    )
            artifact_id = refresh_of.artifact_id
        else:
            artifact_id = uuid.uuid4()
        generation_id = uuid.uuid4()
        generation = _StoredGeneration(artifact_id, generation_id, expected_context, {}, None, None)
        self._generations[(artifact_id, generation_id)] = generation
        return self._publication_context(generation)

    def _write_tombstone(self, artifact_id: uuid.UUID, generation_id: uuid.UUID) -> None:
        """Durably hide a generation: any manifest row, present or landing later,
        is excluded from `visible_manifests` and refused on read."""
        if self._is_dropped(artifact_id, generation_id):
            return
        row = {
            "artifact_id": str(artifact_id),
            "generation_id": str(generation_id),
            "dropped_at": datetime.now(UTC).isoformat(),
        }
        # Build the row in the tombstone table's actual column order: the
        # inserted expression binds by position, not name, and physical order
        # may differ from _DROPPED_TABLE_COLUMNS.
        self._con.insert(
            self._dropped_name,
            ibis.memtable([{name: row[name] for name in self._dropped_columns}]),
            database=cast("Any", self._database),
        )

    def _abort_unpublished(self, generation: _StoredGeneration, exc: BaseException | None):
        """Clean up after a publication whose manifest handle was never set.

        Attaches notes to `exc` (when given) and never raises. The outcome says whether
        the invalidation tombstone was written and which relations remain."""
        key = (generation.artifact_id, generation.generation_id)
        locators = {name: stored.ref.relation for name, stored in generation.relations.items()}
        locators.update(generation.unregistered)
        qualified = {
            name: ".".join(
                part
                for part in (locator.catalog, locator.schema_name, locator.name)
                if part is not None
            )
            for name, locator in locators.items()
        }
        if not generation.manifest_insert_started:
            # The insert never began, so no row can appear: dropping by name is safe.
            dropped_failed: list[str] = []
            first_error: BaseException | None = None
            for name in tuple(locators):
                try:
                    self._con.drop_table(name, database=self._database, force=True)
                except BaseException as erase:  # attempt every relation; never replace exc
                    dropped_failed.append(qualified[name])
                    first_error = first_error or erase
            if dropped_failed and exc is not None:
                exc.add_note(
                    "abort erasure incomplete; the manifest insert never began, so drop "
                    f"relations [{', '.join(dropped_failed)}] (artifact_id={key[0]}, "
                    f"generation_id={key[1]}) by name: {first_error!r}"
                )
            return _AbortOutcome(
                tombstoned=True,
                retained=tuple(dropped_failed),
                tombstone_error=None,
                erase_error=first_error,
            )
        # The index insert was submitted: it may commit at any later time, even
        # after this abort. Only a tombstone written first makes erasing safe.
        listing = ", ".join(qualified.values())
        try:
            self._write_tombstone(*key)
        except BaseException as tombstone:  # never replace the body's exception
            if exc is not None:
                exc.add_note(
                    "manifest publication state unknown and the invalidation tombstone could "
                    f"not be written; relations preserved (artifact_id={key[0]}, "
                    f"generation_id={key[1]}, relations=[{listing}]). From a fresh process "
                    "call store.abandon_generation(artifact_id, generation_id) first "
                    "(durable, safe to repeat), then drop the listed relations by name. "
                    f"Tombstone error: {tombstone!r}"
                )
            return _AbortOutcome(
                tombstoned=False,
                retained=tuple(qualified.values()),
                tombstone_error=tombstone,
            )
        retained: list[str] = []
        first_erase_error: BaseException | None = None
        for name in tuple(locators):
            try:
                self._con.drop_table(name, database=self._database, force=True)
            except BaseException as erase:  # attempt every relation; never replace exc
                retained.append(qualified[name])
                first_erase_error = first_erase_error or erase
        if retained and exc is not None:
            exc.add_note(
                "abort erasure incomplete; the generation is tombstoned and hidden. Drop "
                f"relations [{', '.join(retained)}] (artifact_id={key[0]}, "
                f"generation_id={key[1]}) by name: {first_erase_error!r}"
            )
        return _AbortOutcome(
            tombstoned=True,
            retained=tuple(retained),
            tombstone_error=None,
            erase_error=first_erase_error,
        )

    @contextmanager
    def _publication_context(self, generation: _StoredGeneration) -> Iterator[_ArtifactPublication]:
        publication = _ArtifactPublication(self, generation)
        try:
            yield publication
        except BaseException as exc:
            key = (generation.artifact_id, generation.generation_id)
            if generation.ref is not None:
                # The manifest is already visible: invalidate through the
                # tombstone so the abort never leaves a dangling generation.
                try:
                    self.drop_generation(*key)
                except BaseException as cleanup:  # never replace the body's exception
                    exc.add_note(
                        "abort cleanup incomplete; finish with drop_generation("
                        f"{generation.artifact_id}, {generation.generation_id}): {cleanup!r}"
                    )
            else:
                self._abort_unpublished(generation, exc)
            self._generations.pop(key, None)
            raise
        finally:
            publication._closed = True
        if generation.ref is None:
            # The body ended without a manifest. If a manifest insert was attempted and
            # the caller handled its failure, the same tombstone-first rule applies.
            outcome = self._abort_unpublished(generation, None)
            self._generations.pop((generation.artifact_id, generation.generation_id), None)
            if not generation.manifest_insert_started:
                if outcome.retained:
                    try:
                        _raise(
                            "query.session.warehouse_artifact.publication_cleanup_incomplete",
                            relations=outcome.retained,
                            artifact_id=generation.artifact_id,
                            generation_id=generation.generation_id,
                            route="drop the listed relations by name",
                        )
                    except CodedError as refusal:
                        raise refusal from outcome.erase_error
                return
            if not outcome.tombstoned or outcome.retained:
                cause = outcome.tombstone_error or outcome.erase_error
                if outcome.tombstoned:
                    route = (
                        "the generation is already tombstoned and hidden; drop the "
                        "retained relations by name"
                    )
                else:
                    route = (
                        "from a fresh process call store.abandon_generation(artifact_id, "
                        "generation_id) first (durable, safe to repeat), then drop the "
                        "retained relations by name; drop_generation also works once a "
                        "manifest row exists"
                    )
                try:
                    _raise(
                        "query.session.warehouse_artifact.publication_state_unknown",
                        relations=outcome.retained,
                        artifact_id=generation.artifact_id,
                        generation_id=generation.generation_id,
                        tombstoned=outcome.tombstoned,
                        route=route,
                    )
                except CodedError as refusal:
                    raise refusal from cause

    def _digest_table(self, table: ir.Table, role: str, *, digest_format: int):
        """Recompute a relation's digest with the exact codec `digest_format`
        names -- never re-derived from the connected backend, since
        `SQL_DIGEST_FORMAT` and `BUCKETED_DIGEST_FORMAT` hash different row
        encodings and a stored format id must fully determine which one a
        later verify recomputes."""
        schema = ARTIFACT_RELATION_SCHEMAS[role]
        primary_key = ARTIFACT_RELATION_PRIMARY_KEYS[role]
        if digest_format == 1:
            rows, row_count, tmpdir = stream_canonical_rows(
                table, schema, self._con, primary_key=primary_key
            )
            try:
                with closing(cast(Any, rows)):
                    return digest_relation_stream(
                        role, schema, rows, primary_key=primary_key, row_count=row_count
                    )
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        if digest_format == SQL_DIGEST_FORMAT:
            return digest_relation_sql_v2(self._con, table, role, schema, primary_key=primary_key)
        if digest_format == BUCKETED_DIGEST_FORMAT:
            return digest_relation_bucketed(self._con, table, role, schema, primary_key=primary_key)
        _raise(
            "query.session.warehouse_artifact.unknown_digest_format", digest_format=digest_format
        )

    def _digest_table_for_write(self, table: ir.Table, role: str):
        """Pick the fastest digest this connection can compute for *role*
        and stamp it with the exact format id that codec produced."""
        if role not in ARTIFACT_BASE_RELATION_ROLES:
            return self._digest_table(table, role, digest_format=1)
        if self._con.name in _SQL_DIGEST_BACKENDS:
            return self._digest_table(table, role, digest_format=SQL_DIGEST_FORMAT)
        return self._digest_table(table, role, digest_format=BUCKETED_DIGEST_FORMAT)

    def _write_relation(
        self, generation: _StoredGeneration, role: str, table: ir.Table
    ) -> ArtifactRelationRef:
        if role not in ARTIFACT_RELATION_SCHEMAS:
            _raise("query.session.warehouse_artifact.unknown_relation_role", role=role)
        if generation.manifest is not None:
            _raise("query.session.warehouse_artifact.publication_handle_sealed")

        relation_suffix = role
        if any(stored.ref.role == role for stored in generation.relations.values()):
            relation_suffix = f"{role}_{sum(stored.ref.role == role for stored in generation.relations.values()) + 1}"
        locator = RelationLocator(
            catalog=self._catalog,
            schema=self._schema,
            name=(
                f"ud_{generation.artifact_id.hex[:12]}_{generation.generation_id.hex[:12]}_{relation_suffix}"
            ),
        )
        self.validate_locator(
            locator,
            artifact_id=generation.artifact_id,
            generation_id=generation.generation_id,
            role=cast("Any", role),
        )
        # The create may land even when it raises (a lost response), so the locator is
        # tracked from before the create until it is registered or provably dropped.
        generation.unregistered[locator.name] = locator
        try:
            self._con.create_table(
                self._create_table_name(locator.name),
                table,
                database=cast("Any", self._create_database),
                overwrite=False,
            )
            persisted = self._con.table(locator.name, database=cast("Any", self._database))
            # Only ARTIFACT_BASE_RELATION_ROLES use the fast digest (format 2 or 3,
            # as the backend supports). Extension roles stay on format 1 because
            # validate_extension always recomputes content_sha256(); a fast digest
            # would make every later read refuse with artifact.extension.invalid.
            digest = self._digest_table_for_write(persisted, role)
        except BaseException:
            try:
                self._con.drop_table(locator.name, database=cast("Any", self._database), force=True)
            except BaseException:  # never replace the propagating error; abort retries
                pass
            else:
                del generation.unregistered[locator.name]
            raise
        del generation.unregistered[locator.name]
        ref = ArtifactRelationRef(
            digest_format=digest.digest_format,
            artifact_id=generation.artifact_id,
            generation_id=generation.generation_id,
            role=cast("Any", role),
            relation=locator,
            schema_sha256=digest.schema_sha256,
            content_sha256=digest.content_sha256,
            row_count=digest.row_count,
            primary_key=ARTIFACT_RELATION_PRIMARY_KEYS[role],
        )
        generation.relations[locator.name] = _StoredRelation(ref)
        return ref

    def _publish_manifest(
        self, generation: _StoredGeneration, manifest: UnitDayArtifactManifest
    ) -> UnitDayArtifactRef:
        from increment.query.artifact_digest import manifest_sha256

        if generation.manifest is not None:
            _raise("query.session.warehouse_artifact.publication_handle_sealed")
        if (
            manifest.artifact_id != generation.artifact_id
            or manifest.generation_id != generation.generation_id
        ):
            _raise("query.session.warehouse_artifact.manifest_bound_publication")
        if manifest.context.sha256 != generation.context.sha256:
            _raise("query.session.warehouse_artifact.manifest_context_differs")
        if manifest.manifest_sha256 != manifest_sha256(manifest):
            _raise("query.session.warehouse_artifact.manifest_digest_does")
        refs = [manifest.base.exposures, manifest.base.measure_stats]
        refs.extend(extension.relation for extension in manifest.extensions)
        if len({ref.relation.name for ref in refs}) != len(refs):
            _raise("query.session.warehouse_artifact.manifest_relation_locators")
        if any(
            ref.relation.name not in generation.relations
            or generation.relations[ref.relation.name].ref != ref
            for ref in refs
        ):
            _raise("query.session.warehouse_artifact.manifest_references_unwritten")
        # A relation whose write failed and whose cleanup could not finish may still
        # exist: a manifest must not be published over it, so abort cleanup retries it.
        unreferenced = sorted(
            (set(generation.relations) | set(generation.unregistered))
            - {ref.relation.name for ref in refs}
        )
        if unreferenced:
            from increment.query.artifact_contract import ArtifactContractError

            raise ArtifactContractError(
                "artifact.manifest.unreferenced_relation",
                "publication wrote a relation that the manifest does not reference",
                relation_names=tuple(unreferenced),
            )
        manifest_locator = RelationLocator(
            catalog=self._catalog,
            schema=self._schema,
            name=(
                f"ud_{generation.artifact_id.hex[:12]}_{generation.generation_id.hex[:12]}_manifest"
            ),
        )
        self.validate_locator(
            manifest_locator,
            artifact_id=generation.artifact_id,
            generation_id=generation.generation_id,
            role="manifest",
        )
        ref = UnitDayArtifactRef(
            artifact_id=generation.artifact_id,
            generation_id=generation.generation_id,
            manifest=manifest_locator,
            manifest_sha256=manifest.manifest_sha256,
        )
        row = {
            "artifact_id": str(ref.artifact_id),
            "generation_id": str(ref.generation_id),
            "manifest_name": ref.manifest.name,
            "manifest_sha256": ref.manifest_sha256,
            "manifest_json": json.dumps(manifest.model_dump(mode="json"), separators=(",", ":")),
        }
        # Align by observed physical order: the inserted expression binds by position.
        aligned = {name: row[name] for name in self._meta_columns}
        # From here the insert may commit at any later time, even if this call raises.
        generation.manifest_insert_started = True
        self._con.insert(
            self._meta_name,
            ibis.memtable([aligned]),
            database=cast("Any", self._database),
        )
        generation.manifest = copy.deepcopy(manifest)
        generation.ref = ref
        return ref

    def _generation_for_ref(self, ref: UnitDayArtifactRef) -> _StoredGeneration | None:
        generation = self._generations.get((ref.artifact_id, ref.generation_id))
        if generation is None:
            metadata = self._con.table(self._meta_name, database=self._database)
            persisted = metadata.filter(
                (metadata.artifact_id == str(ref.artifact_id))
                & (metadata.generation_id == str(ref.generation_id))
            )
            rows = self._con.to_pyarrow(persisted).to_pylist()
            if rows and rows[0]["manifest_sha256"] == ref.manifest_sha256:
                from increment.query.artifact_digest import manifest_sha256

                manifest = UnitDayArtifactManifest.model_validate(
                    json.loads(rows[0]["manifest_json"])
                )
                if manifest_sha256(manifest) != rows[0]["manifest_sha256"]:
                    from increment.query.artifact_contract import ArtifactContractError

                    raise ArtifactContractError(
                        "artifact.manifest.invalid",
                        "retained manifest digest does not match its stored content",
                        artifact_id=str(ref.artifact_id),
                        generation_id=str(ref.generation_id),
                    )
                relation_refs = {
                    manifest.base.exposures.relation.name: manifest.base.exposures,
                    manifest.base.measure_stats.relation.name: manifest.base.measure_stats,
                    **{
                        extension.relation.relation.name: extension.relation
                        for extension in manifest.extensions
                    },
                }
                generation = _StoredGeneration(
                    ref.artifact_id,
                    ref.generation_id,
                    manifest.context,
                    {
                        name: _StoredRelation(relation_ref)
                        for name, relation_ref in relation_refs.items()
                    },
                    manifest,
                    ref,
                )
        return generation

    @contextmanager
    def open_snapshot(self, ref: UnitDayArtifactRef):
        validate_trusted_ref(ref)
        self._refuse_if_dropped(ref.artifact_id, ref.generation_id)
        generation = self._generation_for_ref(ref)
        if generation is None or generation.ref != ref or generation.manifest is None:
            from increment.query.artifact_contract import ArtifactContractError

            raise ArtifactContractError(
                "artifact.refresh.invalid_ref",
                "artifact generation is not published",
                context={"artifact_id": str(ref.artifact_id)},
            )
        relations = copy.deepcopy(generation.relations)
        manifest = copy.deepcopy(generation.manifest)
        created_snapshots: list[tuple[str, Any]] = []

        def drop_snapshot(name: str, database: Any) -> None:
            try:
                self._con.drop_table(name, database=database, force=True)
            except Exception:
                pass

        def read_manifest(locator: RelationLocator, expected_sha256: str):
            self.validate_locator(
                locator,
                artifact_id=ref.artifact_id,
                generation_id=ref.generation_id,
                role="manifest",
            )
            if locator != ref.manifest or expected_sha256 != ref.manifest_sha256:
                from increment.query.artifact_contract import ArtifactContractError

                raise ArtifactContractError(
                    "artifact.refresh.invalid_ref",
                    "manifest locator or digest does not match reference",
                    context={"name": locator.name},
                )
            return copy.deepcopy(manifest)

        def verify_relation(relation_ref: ArtifactRelationRef, expected_role: str):
            if (
                relation_ref.artifact_id != ref.artifact_id
                or relation_ref.generation_id != ref.generation_id
                or relation_ref.role != expected_role
            ):
                from increment.query.artifact_contract import ArtifactContractError

                raise ArtifactContractError(
                    "artifact.snapshot.mixed",
                    "relation is not bound to this snapshot",
                    context={"role": expected_role},
                )
            stored = relations.get(relation_ref.relation.name)
            if stored is None or stored.ref != relation_ref:
                from increment.query.artifact_contract import ArtifactContractError

                raise ArtifactContractError(
                    "artifact.snapshot.mixed",
                    "relation reference does not match snapshot",
                    context={"role": expected_role},
                )
            self._refuse_if_dropped(ref.artifact_id, ref.generation_id)
            snapshot_name = f"ud_snap_{uuid.uuid4().hex}"
            # DuckDB/Postgres TEMP tables live outside the publication schema.
            database = self._create_database if self._con.name == "snowflake" else None
            try:
                live = self._con.table(relation_ref.relation.name, database=self._database)
                snapshot_table = self._con.create_table(
                    snapshot_name,
                    live,
                    database=cast("Any", database),
                    temp=True,
                    overwrite=False,
                )
                op = snapshot_table.op()
                if self._con.name == "postgres":
                    database = "pg_temp"
                elif op.namespace.catalog and op.namespace.database:
                    database = (op.namespace.catalog, op.namespace.database)
                elif op.namespace.database:
                    database = op.namespace.database
                _validate_artifact_schema(snapshot_table, expected_role)
                digest = self._digest_table(
                    snapshot_table, expected_role, digest_format=relation_ref.digest_format
                )
                self._refuse_if_dropped(ref.artifact_id, ref.generation_id)
                if (
                    digest.content_sha256 != relation_ref.content_sha256
                    or digest.schema_sha256 != relation_ref.schema_sha256
                    or digest.row_count != relation_ref.row_count
                ):
                    from increment.query.artifact_contract import ArtifactContractError

                    raise ArtifactContractError(
                        "artifact.snapshot.mixed",
                        "relation changed after snapshot verification",
                        context={"role": expected_role},
                    )
            except BaseException:
                drop_snapshot(snapshot_name, database)
                self._refuse_if_dropped(ref.artifact_id, ref.generation_id)
                raise
            created_snapshots.append((snapshot_name, database))
            return snapshot_table

        from increment.query.sequential_capture import record_batches

        snapshot = ImmutableArtifactSnapshot(
            ref.artifact_id,
            ref.generation_id,
            read_manifest,
            verify_relation,
            lambda expression: self._con.to_pyarrow(expression),
            lambda expression: record_batches(self._con, expression),
        )
        validate_snapshot_identity(snapshot, ref)
        try:
            yield snapshot
        finally:
            for name, database in created_snapshots:
                drop_snapshot(name, database)

    def abandon_generation(self, artifact_id: uuid.UUID, generation_id: uuid.UUID) -> None:
        """Durably hide a generation that may or may not have a manifest row yet.

        Writes only the tombstone: a row already present, or one that commits
        later, is excluded from `visible_manifests` and refused on read. Use it
        from a fresh process when a publication's outcome is unknown, then drop
        the retained relations by name. Idempotent and safe to repeat.
        """
        from increment.query.artifact_contract import ArtifactContractError

        arguments = (("artifact_id", artifact_id), ("generation_id", generation_id))
        rejected = [(name, value) for name, value in arguments if not isinstance(value, uuid.UUID)]
        if rejected:
            raise ArtifactContractError(
                "artifact.refresh.invalid_ref",
                "artifact and generation IDs are required",
                rejected=tuple(name for name, _ in rejected),
                route="pass uuid.UUID instances for artifact_id and generation_id",
                **{f"{name}_type": type(value).__name__ for name, value in rejected},
            )
        self._write_tombstone(artifact_id, generation_id)

    def drop_generation(self, artifact_id: uuid.UUID, generation_id: uuid.UUID) -> None:
        """Durably invalidate a generation, then erase its relations.

        Inserting the tombstone is the invalidation point: it happens before
        any relation is touched, so a failure partway through physical
        erasure still leaves the generation hidden and refused everywhere,
        safe to retry from a fresh process using the retained manifest-index
        receipt. A never-published pair returns immediately with no effect.
        An already-dropped pair is not a true no-op: it revalidates the
        retained receipt and retries erasing every named relation. Data
        already read out of the warehouse before invalidation is not revoked.
        """
        from increment.query.artifact_contract import ArtifactContractError
        from increment.query.artifact_digest import manifest_sha256

        metadata = self._con.table(self._meta_name, database=self._database)
        persisted = metadata.filter(
            (metadata.artifact_id == str(artifact_id))
            & (metadata.generation_id == str(generation_id))
        )
        rows = self._con.to_pyarrow(persisted).to_pylist()
        if not rows:
            return
        if len(rows) > 1:
            raise ArtifactContractError(
                "artifact.manifest.invalid",
                "manifest index has duplicate rows for one generation",
                artifact_id=str(artifact_id),
                generation_id=str(generation_id),
            )
        row = rows[0]
        try:
            manifest = UnitDayArtifactManifest.model_validate(json.loads(row["manifest_json"]))
        except (TypeError, ValueError) as exc:
            raise ArtifactContractError(
                "artifact.manifest.invalid",
                "retained manifest metadata failed validation",
                artifact_id=str(artifact_id),
                generation_id=str(generation_id),
            ) from exc
        if manifest_sha256(manifest) != row["manifest_sha256"]:
            raise ArtifactContractError(
                "artifact.manifest.invalid",
                "retained manifest digest does not match its stored content",
                artifact_id=str(artifact_id),
                generation_id=str(generation_id),
            )
        if manifest.artifact_id != artifact_id or manifest.generation_id != generation_id:
            raise ArtifactContractError(
                "artifact.refresh.invalid_ref",
                "retained manifest identity does not match the requested generation",
                artifact_id=str(artifact_id),
                generation_id=str(generation_id),
            )
        relation_refs = [manifest.base.exposures, manifest.base.measure_stats]
        relation_refs.extend(extension.relation for extension in manifest.extensions)
        for relation_ref in relation_refs:
            self.validate_locator(
                relation_ref.relation,
                artifact_id=artifact_id,
                generation_id=generation_id,
                role=cast("Any", relation_ref.role),
            )
        key = (artifact_id, generation_id)
        names = sorted({relation_ref.relation.name for relation_ref in relation_refs})

        self._write_tombstone(artifact_id, generation_id)

        for name in names:
            self._con.drop_table(name, database=self._database, force=True)
        # Evict only after every erasure succeeds, so a failure partway
        # through leaves the generation retrievable for a retry.
        self._generations.pop(key, None)


_MAX_IDENTIFIER_LEN = 63  # Postgres's limit - the tightest among supported backends


@dataclass(frozen=True, slots=True)
class SourceScope:
    """Predicates `pin_sources` pushes into each fact/dim branch before the
    union, so a scoped snapshot (percentile winsorization, artifact publish)
    copies only the rows the readout can reference. No upper time bound
    exists: freshness reads the same fact source and must see events past
    every unit's window, so only a lower bound is ever pushed."""

    enrolled_units: ir.Table  # distinct (unit_id, experiment_id) columns
    window_start: datetime | None


@dataclass(frozen=True, slots=True)
class _MaterializedRelation:
    requested_name: str
    name: str
    catalog: str | None
    schema: str | None


class WarehouseSession:
    """One ibis connection's fact-table cache and TEMP-table mechanics.

    Knows Definitions, FactSource, and Table - never arms, triggers,
    or metrics. Policy (when to materialize) belongs to callers.
    """

    def __init__(self, con: SQLBackend, defs: Definitions) -> None:
        self._con = con
        self._defs = defs
        self._fact_tables: dict[tuple[str, str], Table] = {}
        self._source_tables: dict[str, Table] | None = None
        self._dim_tables: dict[str, Table] = {}
        self._materialized: list[_MaterializedRelation] = []
        # Suffix every temp-table name so sessions sharing one `con` never collide on
        # CREATE TEMP TABLE, where a forced overwrite can drop an unrelated permanent table.
        self._suffix = uuid.uuid4().hex[:12]

    @property
    def con(self) -> SQLBackend:
        return self._con

    @property
    def defs(self) -> Definitions:
        return self._defs

    def source_sql(self, sql: str) -> Table:
        """Resolve source SQL against this operation's captured inputs, if any."""
        if self._source_tables is not None:
            if sql not in self._source_tables:
                _raise("query.session.snapshot.source_missing")
            return self._source_tables[sql]
        return self._con.sql(sql, dialect=self._defs.dialect)

    def pin_sources(
        self,
        queries: tuple[str, ...],
        *,
        column_hints: Mapping[str, tuple[str | None, str | None]] = MappingProxyType({}),
        scope: SourceScope | None = None,
    ) -> WarehouseSession:
        """Capture all requested streams in one execution, preserving their types.

        `column_hints` maps a query's exact SQL text (the same key `source_sql`
        uses) to `(unit_column, timestamp_column)`: either or both may be `None`
        when that predicate does not apply to this stream (e.g. a dimension
        source has no timestamp axis; an exposure/trigger stream is deliberately
        never scoped -- see `_pinned_source`'s own module docs). With
        `scope=None` (the default), no predicate is pushed and behavior is
        unchanged from before this parameter existed.

        Each stream occupies distinct typed columns of a tagged UNION ALL.
        The tagged relation stays in a warehouse TEMP table. Subsequent
        validation and reductions read only that table; raw rows never cross
        the client boundary. The returned session owns its cleanup.
        """
        queries = tuple(dict.fromkeys(queries))
        tables = []
        for sql in queries:
            table = self.source_sql(sql)
            if scope is not None:
                unit_col, ts_col = column_hints.get(sql, (None, None))
                if ts_col is not None and scope.window_start is not None:
                    table = table.filter(table[ts_col] >= scope.window_start)
                if unit_col is not None:
                    table = table.semi_join(
                        scope.enrolled_units, table[unit_col] == scope.enrolled_units["unit_id"]
                    )
            tables.append(table)
        fields = [
            (index, name, f"s{index}_c{column}", dtype)
            for index, table in enumerate(tables)
            for column, (name, dtype) in enumerate(table.schema().items())
        ]
        branches = [
            table.select(
                _stream=ibis.literal(index, type="int64"),
                **{
                    alias: table[name] if owner == index else ibis.null().cast(dtype)
                    for owner, name, alias, dtype in fields
                },
            )
            for index, table in enumerate(tables)
        ]
        pinned = WarehouseSession(self._con, self._defs)
        pinned._source_tables = {}
        if not branches:
            return pinned
        expression = ibis.union(*branches, distinct=False) if len(branches) > 1 else branches[0]
        # One CTAS statement fixes every stream at the same execution snapshot.
        captured = pinned.materialize_table(
            pinned.temp_name("source_snapshot"), expression, required=True
        )
        for index, sql in enumerate(queries):
            stream = captured.filter(captured["_stream"] == index)
            pinned._source_tables[sql] = stream.select(
                **{name: stream[alias] for owner, name, alias, _dtype in fields if owner == index}
            )
        return pinned

    def fact_table(self, fact_source: FactSource, unit: str) -> Table:
        """Memoized fact-source SQL -> declared dim joins -> builder renames.

        Keyed by ``(fact_source.name, unit)``: two metrics sharing a fact
        source at the same unit reuse one ``con.sql(...)`` call, and
        `Report` can resolve a source at a unit other than the
        experiment's own.
        """
        key = (fact_source.name, unit)
        cached = self._fact_tables.get(key)
        if cached is None:
            cached = _dim_joined_fact_table(
                self._con,
                self._defs,
                fact_source,
                unit,
                self._dim_tables,
                resolve_sql=self.source_sql,
            )
            self._fact_tables[key] = cached
        return cached

    def temp_name(self, *parts: str) -> str:
        """Build a temp-table name capped at every supported backend's identifier limit.

        Postgres silently truncates identifiers past 63 bytes, which
        could collide two differently-named tables. Falls back to a
        hash of the full name when it would exceed the limit, trading
        debuggability for guaranteed uniqueness.
        """
        raw = "_".join((*parts, self._suffix))
        if len(raw.encode()) <= _MAX_IDENTIFIER_LEN:
            return raw
        return f"cr_{hashlib.sha256(raw.encode()).hexdigest()[:24]}"

    def _materialized_relation(
        self, requested_name: str, relation: ir.Table
    ) -> _MaterializedRelation | None:
        op = relation.op()
        catalog, schema = op.namespace.catalog, op.namespace.database
        if self._con.name == "postgres":
            rows = self._con.to_pyarrow(
                self._con.sql(
                    "SELECT nspname FROM pg_catalog.pg_namespace WHERE oid = pg_my_temp_schema()"
                )
            ).to_pylist()
            schema = rows[0]["nspname"] if rows else None
            if not isinstance(schema, str) or not schema.startswith("pg_temp_"):
                return None
            catalog = None
        elif self._con.name == "snowflake":
            catalog = catalog or getattr(self._con, "current_catalog", None)
            schema = schema or getattr(self._con, "current_database", None)
            if not catalog or not schema:
                return None
        return _MaterializedRelation(requested_name, op.name, catalog, schema)

    def materialize_table(self, name: str, expr: ir.Table, *, required: bool = False) -> ir.Table:
        """``con.create_table(temp=True)``, degrading to unmaterialized on failure.

        Optional materialization warns and falls back on failure. Required
        snapshots refuse instead: a second live read would change the evidence.
        Never falls back to ``create_view`` either - an auto-created
        view is a persistent object this class can't clean up, and on
        backends with owner's-rights views could leak a lower-privileged
        reader's access to data it couldn't otherwise read.
        """
        created = None
        cleanup = None
        try:
            if self._con.name == "snowflake":
                import sqlglot as sg

                catalog = getattr(self._con, "current_catalog", None)
                schema = getattr(self._con, "current_database", None)
                if not catalog or not schema:
                    _raise("query.session.materialization_namespace", backend=self._con.name)
                # Pin creation so cleanup never depends on a later namespace lookup.
                database = sg.table(schema, db=catalog, quoted=True).sql(dialect="snowflake")
                cleanup = _MaterializedRelation(name, name, catalog, schema)
                created = self._con.create_table(name, expr, temp=True, database=database)
            else:
                created = self._con.create_table(name, expr, temp=True)
            op = created.op()
            if self._con.name == "postgres":
                # pg_temp addresses only this connection's TEMP schema.
                cleanup = _MaterializedRelation(name, op.name, None, "pg_temp")
            elif op.namespace.catalog and op.namespace.database:
                cleanup = _MaterializedRelation(
                    name, op.name, op.namespace.catalog, op.namespace.database
                )
            elif cleanup is not None:
                cleanup = _MaterializedRelation(name, op.name, cleanup.catalog, cleanup.schema)
            physical = self._materialized_relation(name, created)
            if physical is not None:
                self._materialized.append(physical)
                return created
        except Exception:  # degrade, never propagate a privilege error
            pass
        if created is not None and cleanup is not None:
            try:
                self._drop_materialized_relation(cleanup)
            except Exception:  # retain a qualified locator for close() to retry
                self._materialized.append(cleanup)
        if required:
            _raise("query.session.snapshot.materialization_failed")
        _warn("query.session.materialization_degraded", name=name, stacklevel=4)
        return expr

    def _drop_materialized_relation(self, relation: _MaterializedRelation) -> None:
        database = (
            (relation.catalog, relation.schema)
            if relation.catalog is not None and relation.schema is not None
            else relation.schema or relation.catalog
        )
        self._con.drop_table(relation.name, database=database, force=True)

    def drop_materialized(self) -> None:
        """Drop every TEMP table this session materialized (best-effort) and draw a fresh name suffix.

        Tolerates a backend that already reclaimed a name, or a closed
        `con`. Safe to call more than once; the fresh suffix ensures a
        later materialization never reuses a name this drop failed to
        remove.
        """
        for relation in self._materialized:
            try:
                self._drop_materialized_relation(relation)
            except Exception:  # best-effort cleanup only
                pass
        self._materialized = []
        self._suffix = uuid.uuid4().hex[:12]

    @property
    def materialized_names(self) -> list[str]:
        return [relation.requested_name for relation in self._materialized]
