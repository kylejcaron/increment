"""Resolve semantic-layer fact/breakout declarations to ibis tables.

Bridges the semantic layer (``increment.semantics.models``, which
describes facts/dims/entities by their declared vocabulary) to the
builders in ``increment.query.builders`` (which expect the canonical
column set defined in ``increment.query.schemas``). Shared by
``Analysis`` and ``Report`` so both facades build fact tables
identically.
"""

from __future__ import annotations

from collections.abc import Callable

import ibis
from ibis import Table
from ibis.backends.sql import SQLBackend

from increment.errors import InvalidRequestError, raiser, refusals
from increment.query.dims import join_dims
from increment.semantics.models import Breakout, Definitions, Experiment, Fact, Factor, FactSource

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "query.fact_resolution.fact_declared_any": "Fact '{fact_name}' is not declared in any fact source in the definitions. Available facts: {available}",
        "query.fact_resolution.pass_exactly_one": "{constructor}: pass exactly one of control= or design= -- control= is shorthand for design=Randomized(control_group=...); got control={control!r}, design={design!r}",
        "query.fact_resolution.breakout_property_references": "breakout property '{property}' references unknown source '{source}' -- this should have been caught at definitions-load time",
        "query.fact_resolution.breakout_property_could": "breakout property '{property}' could not be resolved to any fact source with unit '{unit}' as an entity -- this should have been caught at definitions-load time",
        "query.fact_resolution.fact_source_property": "fact source '{name}': property '{prop_name}' renames onto the canonical/value column '{prop_name}', which would silently overwrite it; rename the property",
        "query.fact_resolution.fact_source_property_reads_canonical_column": "fact source '{name}': property '{prop_name}' reads the canonical source column '{prop_column}', which the builders rename onto ts/unit_id; point the property at a different column",
        "query.fact_resolution.fact_source_property_deletes_value_column": "fact source '{name}': property '{prop_name}' renames away the canonical/value column '{prop_column}', which would silently delete it; point the property at a different column",
    },
)
_raise = raiser(_REFUSALS)


def _find_fact_source(defs: Definitions, fact_name: str) -> tuple[FactSource, Fact]:
    """Locate the FactSource and Fact for a fact name."""
    for fs in defs.fact_sources:
        for fact in fs.facts:
            if fact.name == fact_name:
                return fs, fact
    _raise(
        "query.fact_resolution.fact_declared_any",
        fact_name=fact_name,
        available=[f.name for fs in defs.fact_sources for f in fs.facts],
    )


def _require_design(control: str | None, design: object | None, *, constructor: str) -> None:
    """Enforce that a seam constructor receives exactly one of control=/design=."""
    if (control is None) == (design is None):
        _raise(
            "query.fact_resolution.pass_exactly_one",
            constructor=constructor,
            control=control,
            design=design,
        )


def _resolve_breakout_fact_source(
    defs: Definitions, experiment: Experiment, breakout: Breakout | Factor
) -> FactSource:
    """Resolve the FactSource backing *breakout*.

    Resolves by breakout/factor property (and an optional explicit
    source name), not by fact name - ``Factor`` shares ``Breakout``'s
    ``property``/``source`` shape, so one resolver serves both.

    ``Breakout.source`` can stay ``None`` on the model even after the
    loader validates a source exists, so this re-resolves
    independently at query time using the loader's own rule: first
    source with the property and the experiment's unit as an entity.
    """
    if breakout.source is not None:
        for fs in defs.fact_sources:
            if fs.name == breakout.source:
                return fs
        _raise(
            "query.fact_resolution.breakout_property_references",
            property=breakout.property,
            source=breakout.source,
        )

    for fs in defs.fact_sources:
        if experiment.unit in fs.entities and any(
            p.name == breakout.property for p in defs.properties_of(fs)
        ):
            return fs

    _raise(
        "query.fact_resolution.breakout_property_could",
        property=breakout.property,
        unit=experiment.unit,
    )


#: Canonical builder columns a non-identity property rename must never
#: touch, in addition to this source's own declared fact value columns -
#: renaming onto or away from one silently clobbers that column's data.
_BUILDER_RESERVED_COLUMNS = frozenset({"ts", "unit_id", "event", "experiment_id"})


def _resolve_entity_source(fs: FactSource, unit: str) -> str | None:
    """The column this source's ``unit_id`` will be renamed FROM, or None.

    Resolved once so the collision guard and the rename cannot disagree: when
    *unit* is not one of this source's entities (a denominator-only fact source,
    say) the rename falls back to the first declared entity, and a property
    reading THAT column is renamed away just the same.
    """
    if unit in fs.entities:
        return unit
    for entity in fs.entities:
        if entity != "unit_id":
            return entity
    return None


def _reject_property_rename_collisions(fs: FactSource, entity_source: str | None = None) -> None:
    """Refuse a property that would clobber a canonical/value column.

    A canonical BUILDER column collides on the property name alone: this
    function's caller renames the declared entity and timestamp onto
    ``unit_id``/``ts``, so an identity declaration named ``ts`` is no safer
    than a rename onto it -- one of the two values silently wins, leaving
    every window computation reading a property as a timestamp.

    A property may also read a column the caller is about to rename INTO a
    canonical name, which loses it whether or not the property moves it, so
    that is checked for identity declarations too. The timestamp always
    qualifies; among the declared entities exactly one does -- the column
    :func:`_resolve_entity_source` picked, which is the analysed unit when this
    source declares it and the first declared entity otherwise. Passing that
    resolved value in, rather than the requested unit, keeps this check and the
    rename from disagreeing about which entity is consumed.

    A fact's own VALUE column is only lost by a genuine rename
    (``prop.column != prop.name``), in both directions: the destination can
    overwrite it and the source can rename it away (e.g.
    ``Property(name="country", column="amount")``). An identity declaration
    moves nothing there and stays allowed.
    """
    value_columns = {f.column for f in fs.facts if f.column is not None}
    renamed_sources = {fs.timestamp_column}
    if entity_source is not None:
        renamed_sources.add(entity_source)
    for prop in fs.properties:
        if prop.name in _BUILDER_RESERVED_COLUMNS:
            _raise(
                "query.fact_resolution.fact_source_property",
                name=fs.name,
                prop_name=prop.name,
            )
        if prop.column in renamed_sources:
            _raise(
                "query.fact_resolution.fact_source_property_reads_canonical_column",
                name=fs.name,
                prop_name=prop.name,
                prop_column=prop.column,
            )
        if prop.column == prop.name:
            continue
        if prop.name in value_columns:
            _raise(
                "query.fact_resolution.fact_source_property",
                name=fs.name,
                prop_name=prop.name,
            )
        if prop.column in _BUILDER_RESERVED_COLUMNS | value_columns:
            _raise(
                "query.fact_resolution.fact_source_property_deletes_value_column",
                name=fs.name,
                prop_name=prop.name,
                prop_column=prop.column,
            )


def _rename_to_builder_cols(tbl: Table, fs: FactSource, unit: str) -> Table:
    """Rename entity->unit_id and timestamp->ts so the builders can consume.

    The semantic layer names entities/timestamps by their physical
    column names; the builders expect the canonical column set in
    ``increment.query.schemas``.

    ``unit`` (``Experiment.unit``) drives which declared entity maps to
    ``unit_id`` - a fact source can declare several (e.g.
    ``[session_id, user_id]``), and picking the wrong one silently
    aggregates at the wrong grain with no error.
    """
    entity_source = _resolve_entity_source(fs, unit)
    _reject_property_rename_collisions(fs, entity_source)
    kw = {}
    if entity_source is not None and entity_source != "unit_id":
        kw["unit_id"] = entity_source
    if fs.timestamp_column and fs.timestamp_column != "ts":
        kw["ts"] = fs.timestamp_column
    # Rename property columns from physical names to logical names so
    # that _apply_filter (which uses flt.property) can find them.
    for prop in fs.properties:
        if prop.column != prop.name:
            kw[prop.name] = prop.column
    if "experiment_id" not in tbl.columns:
        tbl = tbl.mutate(experiment_id=ibis.null().cast("string"))
    if kw:
        tbl = tbl.rename(**kw)
    return tbl


def _dim_joined_fact_table(
    con: SQLBackend,
    defs: Definitions,
    fs: FactSource,
    unit: str,
    dim_cache: dict[str, Table],
    *,
    resolve_sql: Callable[[str], Table] | None = None,
) -> Table:
    """Fact-source SQL -> declared dim joins -> builder column renames.

    Shared by ``Analysis`` and ``Report``, the one seam that keeps both
    facades building fact tables identically. *dim_cache* is
    caller-owned, keyed by dim name, so two fact sources sharing a dim
    reuse one scan.
    """

    def default_resolver(sql: str) -> Table:
        return con.sql(sql, dialect=defs.dialect)

    resolve_sql = resolve_sql or default_resolver
    tbl = resolve_sql(fs.sql)
    if fs.dims:
        dim_by_name = {d.name: d for d in defs.dim_sources}
        pairs = []
        for name in fs.dims:
            dim = dim_by_name[name]
            cached = dim_cache.get(name)
            if cached is None:
                cached = resolve_sql(dim.sql)
                dim_cache[name] = cached
            pairs.append((dim, cached))
        tbl = join_dims(tbl, ts_column=fs.timestamp_column, dims=pairs, execute=con.execute)
    return _rename_to_builder_cols(tbl, fs, unit)


def _resolve_value_column(fs: FactSource, fact: Fact) -> str | None:
    """Return the value column for a fact, or None for occurrence-only."""
    if fact.column is not None:
        return fact.column
    return None
