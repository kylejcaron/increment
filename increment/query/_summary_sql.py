"""Name existing summary relations without changing their construction or sharing."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import cast

import ibis
import ibis.common.exceptions as ibis_errors
import ibis.expr.operations as ops
import ibis.expr.types as ir
import sqlglot
from ibis.common.graph import Graph, Node
from sqlglot import exp

from increment._compile_gaps import load_dialect_compiler

_REFERENCE_NODES = (ops.Field, ops.CountStar, ops.CountDistinctStar)


def _descendants(graph: Graph, root: Node) -> set[Node]:
    found: set[Node] = set()
    pending = [root]
    while pending:
        node = pending.pop()
        if node not in found:
            found.add(node)
            pending.extend(graph.get(node, ()))
    return found


def _cte_candidates(
    root: ops.Relation, marked: Mapping[ops.Relation, str]
) -> tuple[ops.Relation, ...]:
    graph = Graph.from_bfs(root, filter=lambda node: not isinstance(node, _REFERENCE_NODES))
    ordered, incoming = cast(tuple[Graph, Graph], graph.toposort())
    occurrences: Counter[Node] = Counter({root: 1})
    for node in reversed(tuple(ordered)):
        for child in graph[node]:
            occurrences[child] = min(2, occurrences[child] + occurrences[node])
    scoped: set[Node] = set()
    for node in graph:
        if isinstance(node, ops.Subquery):
            scoped.update(_descendants(graph, node))
    candidates = []
    for node in graph:
        if (
            not isinstance(node, ops.Relation)
            or node not in marked
            or occurrences[node] != 1
            or node in scoped
        ):
            continue
        cone = _descendants(graph, node)
        if all(
            all(parent in cone for parent in incoming[child])
            for child in cone
            if child is not node and isinstance(child, ops.Relation)
        ):
            candidates.append(node)
    return tuple(candidates)


def _occupied_names(root: ops.Relation, dialect: str | None) -> set[str]:
    import ibis.backends.sql.compilers as compilers

    compiler = getattr(compilers, dialect).compiler if dialect is not None else None
    sql_dialect = compiler.dialect if compiler is not None else None
    names: set[str] = set()
    for node in Graph.from_bfs(root):
        name = getattr(node, "name", None)
        if isinstance(name, str):
            names.add(name.casefold())
        query = getattr(node, "query", None)
        if isinstance(query, str):
            for statement in sqlglot.parse(query, read=sql_dialect):
                if statement is not None:
                    names.update(
                        identifier.name.casefold()
                        for identifier in statement.find_all(exp.Identifier)
                    )
    return names


class SummarySqlStages:
    """Collect semantic boundaries for an independently rendered preview graph."""

    def __init__(self) -> None:
        self._marked: dict[ops.Relation, str] = {}

    def mark(self, role: str, relation: ir.Table) -> ir.Table:
        """Remember a downstream role while preserving the exact expression."""
        self._marked[relation.op()] = role
        return relation

    def to_sql(self, relation: ir.Table, *, dialect: str | None = None) -> str:
        """Compile safe named boundaries without mutating the original graph."""
        if dialect is None:
            try:
                dialect = relation._find_backend(use_default=True).name
            except ibis_errors.IbisError:
                dialect = "duckdb"
        load_dialect_compiler(dialect)
        root = relation.op()
        candidates = _cte_candidates(root, self._marked)
        occupied = _occupied_names(root, dialect)
        names: dict[ops.Relation, str] = {}
        for node in candidates:
            base = f"inc_{self._marked[node]}"
            name = base
            suffix = 2
            while name.casefold() in occupied:
                name = f"{base}_{suffix}"
                suffix += 1
            occupied.add(name.casefold())
            names[node] = name

        def annotate(node, children):
            rewritten = node.__recreate__(children) if children else node
            if node in names:
                return rewritten.to_expr().alias(names[node]).op()
            return rewritten

        annotated = root.replace(annotate).to_expr()
        return ibis.to_sql(annotated, dialect=dialect)
