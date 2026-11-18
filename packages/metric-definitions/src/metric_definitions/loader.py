"""Load a catalog and locate invalid declarations in the input file."""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

import yaml
from pydantic import ValidationError
from yaml.composer import ComposerError
from yaml.error import Mark
from yaml.events import AliasEvent
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from .models import Catalog, Location


class CatalogError(ValueError):
    """Invalid catalog content, with a code and one-based file location."""

    def __init__(self, code: str, path: Path, line: int, column: int, detail: str) -> None:
        self.code = code
        self.path = path
        self.line = line
        self.column = column
        self.detail = detail
        super().__init__(f"{path}:{line}:{column}: {detail}")


_STR_TAG = "tag:yaml.org,2002:str"
_BOOL_TAG = "tag:yaml.org,2002:bool"


class _Loader(yaml.SafeLoader):
    # Values such as "on" and ISO dates are strings unless their declaration says otherwise.
    yaml_implicit_resolvers = {
        char: [
            (tag, pattern)
            for tag, pattern in resolvers
            if tag not in (_BOOL_TAG, "tag:yaml.org,2002:timestamp")
        ]
        for char, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }

    def compose_node(self, parent: Node | None, index: int | Node | None) -> Node | None:
        event = self.peek_event()
        if isinstance(event, AliasEvent):
            raise ComposerError(None, None, "YAML aliases are not supported.", event.start_mark)
        return super().compose_node(parent, cast(int, index))  # PyYAML's stub narrows index.


_Loader.add_implicit_resolver(_BOOL_TAG, re.compile(r"^(?:true|false)$"), list("tf"))


def _at(path: Path, mark: Mark | None, code: str, detail: str) -> CatalogError:
    return CatalogError(
        code, path, mark.line + 1 if mark else 1, mark.column + 1 if mark else 1, detail
    )


def _index(node: Node, path: Path) -> dict[Location, Mark]:
    marks: dict[Location, Mark] = {}
    pending: list[tuple[Node, Location]] = [(node, ())]
    while pending:
        child, location = pending.pop()
        marks[location] = child.start_mark
        if isinstance(child, MappingNode):
            names: set[str] = set()
            for key, item in child.value:
                if not isinstance(key, ScalarNode) or key.tag != _STR_TAG:
                    raise _at(
                        path, key.start_mark, "invalid_key", "Use a string as the mapping key."
                    )
                if key.value in names:
                    raise _at(
                        path, key.start_mark, "duplicate_key", f"Duplicate key {key.value!r}."
                    )
                names.add(key.value)
                pending.append((item, (*location, key.value)))
        elif isinstance(child, SequenceNode):
            pending.extend((item, (*location, i)) for i, item in enumerate(child.value))
    return marks


def _mark_for(location: Location, marks: dict[Location, Mark]) -> Mark:
    if len(location) >= 4 and location[0] == "metrics" and location not in marks:
        # Pydantic inserts the selected metric variant ahead of its real YAML field.
        location = (*location[:2], *location[3:])
    while location not in marks:
        location = location[:-1]
    return marks[location]


def load(path: str | Path) -> Catalog:
    """Read and validate a versioned YAML catalog from disk."""
    path = Path(path)
    try:
        loader = _Loader(path.read_text(encoding="utf-8"))
    except UnicodeError as exc:
        raise _at(path, None, "invalid_yaml", str(exc)) from exc
    try:
        node = loader.get_single_node()
        if node is None:
            raise _at(path, None, "invalid_definition", "Supply a versioned mapping.")
        marks = _index(node, path)
        data = loader.construct_document(node)
    except yaml.MarkedYAMLError as exc:
        raise _at(path, exc.problem_mark, "invalid_yaml", exc.problem or str(exc)) from exc
    except (yaml.YAMLError, ValueError, RecursionError) as exc:
        if isinstance(exc, CatalogError):
            raise
        raise _at(path, None, "invalid_yaml", str(exc)) from exc
    finally:
        loader.dispose()
    try:
        return Catalog.model_validate(data)
    except ValidationError as exc:
        error = exc.errors(include_url=False)[0]
        code = str(error["type"])
        if code not in (
            "invalid_formula",
            "unknown_reference",
            "formula_cycle",
            "invalid_population",
            "wrong_filter_source",
            "invalid_filter_value",
            "invalid_filter_operator",
            "invalid_reference_name",
            "duplicate_name",
            "invalid_conversion",
            "wrong_population_entity",
        ):
            code = "invalid_definition"
        raise _at(path, _mark_for(error["loc"], marks), code, error["msg"]) from exc
