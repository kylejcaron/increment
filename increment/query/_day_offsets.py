"""Per-unit day ranges with backend-native relational expansion."""

from __future__ import annotations

from typing import Any, cast

import ibis
import ibis.expr.operations as ops
import ibis.expr.types as ir
import sqlglot.expressions as sge


class _DayOffsets(ops.TableUnnest):
    """Expand a unit's day range, retaining a null offset when it is empty."""


def _register_day_offsets() -> None:
    from ibis.backends.sql.compilers.base import SQLGlotCompiler
    from ibis.backends.sql.compilers.snowflake import SnowflakeCompiler

    def portable(compiler: Any, op: _DayOffsets, **kwargs: Any) -> Any:
        return compiler.visit_TableUnnest(op, **kwargs)

    def snowflake(
        compiler: Any,
        op: _DayOffsets,
        *,
        parent: Any,
        column: Any,
        column_name: str,
        **_: Any,
    ) -> Any:
        quoted = compiler.quoted
        alias = sge.to_identifier("__panel_day", quoted=quoted)
        flatten = sge.Lateral(
            this=sge.Anonymous(
                this="FLATTEN",
                expressions=[sge.Kwarg(this=sge.Var(this="INPUT"), expression=column)],
            ),
            alias=sge.TableAlias(this=alias),
        )
        # FLATTEN's output names are fixed; derived-column aliases do not rename VALUE.
        offset = compiler.cast(
            sge.Column(this=sge.to_identifier("VALUE", quoted=quoted), table=alias.copy()),
            op.column.dtype.value_type,
        ).as_(column_name, quoted=quoted)
        columns = sge.Column(
            this=sge.Star(),
            table=sge.to_identifier(parent.alias_or_name, quoted=quoted),
        )
        return sge.select(columns, offset).from_(parent).join(flatten, join_type="CROSS")

    pending = [SQLGlotCompiler]
    while pending:
        compiler = pending.pop()
        pending.extend(compiler.__subclasses__())
        visit = snowflake if issubclass(compiler, SnowflakeCompiler) else portable
        compiler.visit__DayOffsets = visit  # ty: ignore[unresolved-attribute]


_register_day_offsets()


def expand_day_offsets(keys: ir.Table, day_count: ir.IntegerValue) -> ir.Table:
    """Append offsets ``0..day_count-1``; empty ranges retain one null offset."""
    name = "__panel_day_offsets"
    while name in keys.columns:
        name += "_"
    day_range = ibis.range(0, day_count, 1)
    # A negative sentinel preserves empty ranges without an outer unnest.
    # Real offsets are nonnegative; the sentinel is nulled before date arithmetic.
    has_days = cast("Any", day_count) > 0
    day_range = has_days.ifelse(day_range, ibis.array([-1]).cast(day_range.type()))
    # Projection scopes scalar horizon reductions before the lateral expansion.
    ranges = keys.mutate(**{name: day_range})
    expanded = (
        _DayOffsets(
            parent=ranges.op(),
            column=ranges[name].op(),
            column_name="day_offset",
            offset=None,
            keep_empty=False,
        )
        .to_expr()
        .drop(name)
    )
    offset = expanded.day_offset
    return expanded.mutate(day_offset=(offset >= 0).ifelse(offset, ibis.null().cast(offset.type())))
