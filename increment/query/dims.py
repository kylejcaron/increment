"""Generate dim joins for fact sources declared with ``dims:``.

One join strategy per physical shape (see ``DimValidity``): a plain
dim LEFT-joins on the entity column; a validity-range dim LEFT-joins
on entity plus ``ts >= valid_from AND ts < valid_to``; a changelog dim
is windowed into ranges first, then joined the same way.

``valid_to`` is normalized to a sentinel before the join - a
NULL-tolerant predicate in the join itself defeats range-join
optimization. Joins are LEFT so an unmatched fact row keeps NULL
properties instead of vanishing. Each dim must have one row per entity
(plain) or non-overlapping validity per entity (versioned); violating
either would silently fan out fact rows, so it is refused before the
join runs.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import ibis

from increment.errors import InvalidRequestError, raiser, refusals

if TYPE_CHECKING:
    from collections.abc import Sequence

    import ibis.expr.types as ir

    from increment.semantics.models import DimSource

#: Far-future close for open validity ranges. Matches the convention the
#: data-model guide asks warehouse tables to follow.
VALID_TO_SENTINEL = dt.datetime(9999, 12, 31)


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "query.dims.dim_source_multiple": "dim source '{name}': multiple rows for the same '{entity}' value would fan out the fact table; deduplicate the dim source or declare a validity range",
        "query.dims.dim_source_two": "dim source '{name}': two rows share one validity boundary for the same '{entity}' value; the tie would resolve by physical row order rather than the declared history",
        "query.dims.dim_source_overlapping": "dim source '{name}': overlapping validity ranges for the same '{entity}' value; fix the changelog/validity source",
        "query.dims.dim_source_property": "dim source '{name}': property '{column}' collides with an existing column on the fact source; rename the property or drop the fact-side column from the source SQL",
        "query.dims.dim_source_fact": "dim source '{name}': the fact source projects a 'valid_from'/'valid_to' column, which the generated range join reserves; alias it in the fact source SQL",
    },
)
_raise = raiser(_REFUSALS)


def _as_ranges(dim: DimSource, tbl: ir.Table) -> ir.Table:
    """Project *tbl* to ``(entity, valid_from, valid_to, <property columns>)``.

    Normalizes all three physical shapes to one ranged, sentinel-closed
    form, renaming property columns so the join step never touches raw
    column names.
    """
    assert dim.validity is not None
    props = {p.name: tbl[p.column] for p in dim.properties}
    if dim.validity.changed_at is not None:
        changed = tbl[dim.validity.changed_at]
        window = ibis.window(group_by=tbl[dim.entity], order_by=changed)
        return tbl.select(
            tbl[dim.entity],
            valid_from=changed,
            valid_to=changed.lead().over(window).coalesce(ibis.literal(VALID_TO_SENTINEL)),
            **props,
        )
    valid_from, valid_to = dim.validity.valid_from, dim.validity.valid_to
    assert valid_from is not None and valid_to is not None  # enforced by DimValidity
    return tbl.select(
        tbl[dim.entity],
        valid_from=tbl[valid_from],
        valid_to=tbl[valid_to].coalesce(ibis.literal(VALID_TO_SENTINEL)),
        **props,
    )


def _reject_duplicate_entities(
    dim: DimSource, tbl: ir.Table, execute: Callable[[ir.Scalar], object]
) -> None:
    """Refuse a plain dim with more than one row per entity value.

    An unenforced duplicate silently fans the fact table out through the
    equi-join, doubling every additive metric downstream.
    """
    counts = tbl.group_by(dim.entity).agg(n_rows=tbl.count())
    n_duplicated = int(cast("int", execute(counts.filter(counts.n_rows > 1).count())))
    if n_duplicated:
        _raise("query.dims.dim_source_multiple", entity=dim.entity, name=dim.name)


def _reject_overlapping_ranges(
    dim: DimSource, ranges: ir.Table, execute: Callable[[ir.Scalar], object]
) -> None:
    """Refuse validity ranges that overlap - a tie included.

    Two rows sharing one boundary (e.g. a changelog with two updates at
    the same ``changed_at``) resolve which value wins by physical row
    order rather than the declared history, so a tied ``valid_from`` is
    rejected outright first: the general overlap check below orders by
    ``valid_from`` and is itself unreliable across a tie.
    """
    entity = ranges[dim.entity]
    tied = ranges.group_by([entity, ranges.valid_from]).agg(n_rows=ranges.count())
    n_tied = int(cast("int", execute(tied.filter(tied.n_rows > 1).count())))
    if n_tied:
        _raise("query.dims.dim_source_two", entity=dim.entity, name=dim.name)
    window = ibis.window(group_by=entity, order_by=ranges.valid_from)
    previous_valid_to = ranges.valid_to.lag().over(window)
    overlapping = ranges.filter(
        previous_valid_to.notnull() & (ranges.valid_from < previous_valid_to)
    )
    n_overlapping = int(cast("int", execute(overlapping.count())))
    if n_overlapping:
        _raise("query.dims.dim_source_overlapping", entity=dim.entity, name=dim.name)


def join_dims(
    fact: ir.Table,
    *,
    ts_column: str,
    dims: Sequence[tuple[DimSource, ir.Table]],
    execute: Callable[[ir.Scalar], object] | None = None,
) -> ir.Table:
    """LEFT-join every declared dim into *fact*, one property column each.

    *ts_column* keys the range predicate for versioned dims (unused for
    plain dims). Raises ``ValueError`` when a dim property name collides
    with an existing fact column - silent suffixing would break every
    downstream property lookup - or when the dim source violates the
    one-row-per-entity/non-overlapping-range cardinality the join assumes.
    """

    def default_execute(expression: ir.Scalar) -> object:
        return expression.execute()

    execute = execute or default_execute
    out = fact
    for dim, tbl in dims:
        collisions = [p.name for p in dim.properties if p.name in out.columns]
        if collisions:
            _raise("query.dims.dim_source_property", column=collisions[0], name=dim.name)
        if dim.validity is not None and not {"valid_from", "valid_to"}.isdisjoint(out.columns):
            # Range joins reserve these names; a same-named fact column
            # would be silently suffixed, pinning the final select to the wrong side.
            _raise("query.dims.dim_source_fact", name=dim.name)
        if dim.validity is None:
            _reject_duplicate_entities(dim, tbl, execute)
            proj = tbl.select(
                tbl[dim.entity],
                **{p.name: tbl[p.column] for p in dim.properties},
            )
            joined = out.left_join(proj, out[dim.entity] == proj[dim.entity])
        else:
            proj = _as_ranges(dim, tbl)
            _reject_overlapping_ranges(dim, proj, execute)
            joined = out.left_join(
                proj,
                [
                    out[dim.entity] == proj[dim.entity],
                    out[ts_column] >= proj.valid_from,
                    out[ts_column] < proj.valid_to,
                ],
            )
        out = joined.select(
            *[joined[c] for c in out.columns],
            *[joined[p.name] for p in dim.properties],
        )
    return out
