from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from scripts.check_public_typing import main, public_any

pytestmark = pytest.mark.tooling


def _package(root: Path, **files: str) -> Path:
    for name, source in files.items():
        path = root / "sample" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dedent(source))
    return root


def test_counts_any_in_function_signatures(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": """
                import typing as t
                from typing import Any as A, Annotated, Literal
                __all__ = ["convert", "clean", "Alias", "Legacy", "label", "marker"]
                Legacy = A
                label = "not an annotation"
                class _Kind:
                    ITEM = 1
                marker = _Kind.ITEM
                type Alias = dict[str, A]
                async def convert(x: "list[A]", /, *args: t.Any,
                                  flag: Annotated[A, "Any"], **kw: A) -> "t.Any":
                    local: A = x
                    def nested(value: A) -> A: return value
                    return local
                def clean(x: Literal["Any"], y: Annotated[str, "Any"]) -> str:
                    local: A = x
                    return local
            """,
        },
    )

    assert public_any(root, ["sample"]) == {
        "sample.convert": 5,
        "sample.clean": 0,
        "sample.label": 0,
        "sample.marker": 0,
        "sample.Alias": 1,
        "sample.Legacy": 1,
    }


def test_overloads_keep_independent_annotation_allowances(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": """
                from typing import Any, overload as variant
                __all__ = ["convert"]
                @variant
                def convert(value: Any) -> str: ...
                @variant
                def convert(value: str) -> Any: ...
                def convert(value: object) -> object: ...
            """,
        },
    )
    assert public_any(root, ["sample"]) == {"sample.convert": 2}
    baseline = root / "baseline.txt"
    args = ["--root", str(root), "--namespace", "sample", "--baseline", str(baseline)]
    assert main([*args, "--write-baseline"]) == 0
    path = root / "sample/__init__.py"
    path.write_text(path.read_text().replace("value: Any) -> str", "value: str) -> Any"))
    assert main(args) == 1


def test_annotation_aliases_unwrap_declared_and_quoted_types(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": """
                from os import environ
                from typing import Any, TypeAlias
                from builtins import list as L
                from .bridge import Box, Mapping as Map
                from .types import (
                    Declared,
                    Modern,
                    Quoted,
                    MomentRows,
                    OptionalMomentRows,
                    ExternalRows,
                    ExternalBoxRows,
                    QuotedExternalRows,
                    LiteralRows,
                    AnnotatedRows,
                    BuiltinRows,
                )
                __all__ = [
                    "convert",
                    "ExternalBoxRows",
                    "QuotedExternalRows",
                    "LiteralRows",
                    "AnnotatedRows",
                    "ExternalRows",
                    "MapRows",
                    "MomentRows",
                    "OptionalMomentRows",
                    "Rows",
                    "BuiltinRows",
                    "ImplicitRows",
                    "environment",
                    "environment_any",
                    "environment_arithmetic",
                    "payload",
                    "union_payload",
                ]
                def convert(value: Declared, extra: "Modern") -> Quoted: ...
                environment = environ["APP CONFIG"]
                environment_any = environ["Any"]
                environment_arithmetic = environ["Any + 1"]
                payload = {"Any": 1}["Any"]
                union_payload = {"Any": 1} | {"other": 2}
                Rows = Box[Any]
                BuiltinRows = L["Any"]
                MapRows = Map[str, Any]
                ImplicitRows = list[str]
            """,
            "types.py": """
                from collections.abc import Mapping
                from datetime import date
                from third_party import ExternalBox
                from typing import Any, Literal, Annotated, TypeAlias as Alias
                class Box[T]: ...
                Declared: Alias = Any
                type Modern = list[Declared]
                Quoted: Alias = "Modern"
                MomentRows = tuple[Mapping[str, Any], ...]
                OptionalMomentRows = tuple[Mapping[str, Any], ...] | None
                ExternalRows = tuple[Mapping[str, Any], ...] | date
                ExternalBoxRows = ExternalBox[Any]
                QuotedExternalRows: Alias = ExternalBox["str"]
                LiteralRows = ExternalBox[Literal["Any"]]
                AnnotatedRows = ExternalBox[Annotated[str, Any]]
            """,
            "bridge.py": """
                from collections.abc import Mapping
                from .types import Box
            """,
        },
    )
    with pytest.raises(ValueError):
        public_any(root, ["sample"])
    path = root / "sample/__init__.py"
    path.write_text(
        path.read_text().replace(
            'environment_any = environ["Any"]',
            'environment_any: str = environ["Any"]',
        )
    )
    counts = public_any(root, ["sample"])
    assert counts["sample.environment_any"] == 0
    assert counts["sample.environment_arithmetic"] == 0
    assert counts == {
        "sample.convert": 3,
        "sample.ExternalBoxRows": 1,
        "sample.QuotedExternalRows": 0,
        "sample.LiteralRows": 0,
        "sample.AnnotatedRows": 0,
        "sample.ExternalRows": 1,
        "sample.MapRows": 1,
        "sample.MomentRows": 1,
        "sample.OptionalMomentRows": 1,
        "sample.Rows": 1,
        "sample.BuiltinRows": 1,
        "sample.ImplicitRows": 0,
        "sample.environment": 0,
        "sample.environment_any": 0,
        "sample.environment_arithmetic": 0,
        "sample.payload": 0,
        "sample.union_payload": 0,
    }
    baseline = root / "baseline.txt"
    args = ["--root", str(root), "--namespace", "sample", "--baseline", str(baseline)]
    assert main([*args, "--write-baseline"]) == 0
    init_path = root / "sample/__init__.py"
    init_path.write_text(
        init_path.read_text().replace("ImplicitRows = list[str]", "ImplicitRows = list[Any]")
    )
    baseline_before_implicit_any = baseline.read_text()
    assert main(args) == 1
    assert baseline.read_text() == baseline_before_implicit_any
    assert public_any(root, ["sample"])["sample.ImplicitRows"] == 1
    init_path.write_text(
        init_path.read_text().replace("ImplicitRows = list[Any]", "ImplicitRows = list[str]")
    )
    type_path = root / "sample/types.py"
    type_path.write_text(
        type_path.read_text().replace(
            'QuotedExternalRows: Alias = ExternalBox["str"]',
            'QuotedExternalRows = ExternalBox["Any"]',
        )
    )
    with pytest.raises(ValueError):
        public_any(root, ["sample"])
    baseline_before_ambiguity = baseline.read_text()
    assert main(args) == 1
    assert main([*args, "--write-baseline"]) == 1
    assert baseline.read_text() == baseline_before_ambiguity
    type_path.write_text(
        type_path.read_text().replace(
            'QuotedExternalRows = ExternalBox["Any"]',
            'QuotedExternalRows: Alias = ExternalBox["Any"]',
        )
    )
    assert public_any(root, ["sample"])["sample.QuotedExternalRows"] == 1
    assert main(args) == 1


def test_valid_typevar_generic_does_not_resolve_its_name_as_an_annotation(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": """
                from typing import TypeVar
                __all__ = ["identity", "named"]
                T = TypeVar("T")
                named = TypeVar("Any")
                def identity(value: T) -> T: ...
            """,
        },
    )

    assert public_any(root, ["sample"]) == {
        "sample.identity": 0,
        "sample.named": 0,
    }


def test_typevar_bounds_constraints_and_defaults_count_any_annotations(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": """
                from typing import Any, TypeVar
                __all__ = ["bounded", "constrained", "defaulted"]
                Bounded = TypeVar("Bound", bound=Any)
                Constrained = TypeVar("Choice", Any, str)
                Defaulted = TypeVar("Defaulted", default=Any)
                def bounded(value: Bounded) -> Bounded: ...
                def constrained(value: Constrained) -> None: ...
                def defaulted(value: Defaulted) -> None: ...
            """,
        },
    )

    assert public_any(root, ["sample"]) == {
        "sample.bounded": 2,
        "sample.constrained": 1,
        "sample.defaulted": 1,
    }


@pytest.mark.parametrize(
    ("alias", "declaration", "function", "signature"),
    [
        (
            "P",
            'P = ParamSpec("P")',
            "apply",
            "def apply(callback: Callable[P, int]) -> Callable[P, int]: ...",
        ),
        (
            "Ts",
            'Ts = TypeVarTuple("Ts")',
            "pack",
            "def pack(values: tuple[*Ts]) -> tuple[*Ts]: ...",
        ),
        (
            "UserId",
            'UserId = NewType("UserId", int)',
            "lookup",
            "def lookup(value: UserId) -> UserId: ...",
        ),
        (
            "Alias",
            'Alias = TypeAliasType("Alias", int)',
            "convert",
            "def convert(value: Alias) -> Alias: ...",
        ),
        (
            "Record",
            'Record = NamedTuple("Record", [("field", str)])',
            "read_record",
            "def read_record(value: Record) -> Record: ...",
        ),
        (
            "Record",
            'Record = TypedDict("Record", {"field": str})',
            "read_record",
            "def read_record(value: Record) -> Record: ...",
        ),
    ],
    ids=["ParamSpec", "TypeVarTuple", "NewType", "TypeAliasType", "NamedTuple", "TypedDict"],
)
def test_standard_typing_factories_do_not_resolve_names_as_annotations(
    tmp_path: Path, alias: str, declaration: str, function: str, signature: str
) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": f"""
                from typing import Any, Callable, ParamSpec, TypeVarTuple, NewType
                from typing import TypeAliasType, NamedTuple, TypedDict
                __all__ = ["{alias}", "{function}"]
                {declaration}
                {signature}
            """,
        },
    )

    assert public_any(root, ["sample"]) == {
        f"sample.{alias}": 0,
        f"sample.{function}": 0,
    }


def test_factory_type_positions_count_any_but_ignore_names_and_flags(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": """
                from typing import Any, NamedTuple, TypedDict, TypeVar, ParamSpec, TypeVarTuple
                __all__ = [
                    "plain_tuple", "typed_tuple", "plain_dict", "typed_dict",
                    "bounded", "defaulted_param", "defaulted_tuple", "defaulted_var",
                ]
                PlainTuple = NamedTuple("Any", [("Any", str)])
                TypedTuple = NamedTuple("Typed", [("Any", Any)])
                PlainDict = TypedDict("Any", {"Any": str}, total=False, closed=True)
                TypedDictValue = TypedDict("Schema", {"Any": Any}, total=False, closed=True)
                Bounded = TypeVar("Bounded", bound=Any)
                DefaultedParam = ParamSpec("P", default=Any)
                DefaultedTuple = TypeVarTuple("Ts", default=Any)
                DefaultedVar = TypeVar("DefaultedVar", default=Any)
                def plain_tuple(value: PlainTuple) -> None: ...
                def typed_tuple(value: TypedTuple) -> None: ...
                def plain_dict(value: PlainDict) -> None: ...
                def typed_dict(value: TypedDictValue) -> None: ...
                def bounded(value: Bounded) -> None: ...
                def defaulted_param(value: DefaultedParam) -> None: ...
                def defaulted_tuple(value: DefaultedTuple) -> None: ...
                def defaulted_var(value: DefaultedVar) -> None: ...
            """,
        },
    )

    assert public_any(root, ["sample"]) == {
        "sample.plain_tuple": 0,
        "sample.typed_tuple": 1,
        "sample.plain_dict": 0,
        "sample.typed_dict": 1,
        "sample.bounded": 1,
        "sample.defaulted_param": 1,
        "sample.defaulted_tuple": 1,
        "sample.defaulted_var": 1,
    }


def test_annotation_only_type_field_does_not_shadow_builtin(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": """
                __all__ = ["Record", "InitializedRecord"]
                class Record:
                    type: type
                class InitializedRecord:
                    type: type = int
            """,
        },
    )

    assert public_any(root, ["sample"]) == {
        "sample.Record": 0,
        "sample.InitializedRecord": 0,
    }


def test_namedtuple_field_entry_alias_counts_any_and_baseline_growth(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": """
                from typing import Any, NamedTuple
                __all__ = ["Record", "read"]
                _field = ("Any", str)
                Record = NamedTuple("Row", [_field])
                def read(value: Record) -> None: ...
            """,
        },
    )
    assert public_any(root, ["sample"]) == {"sample.Record": 0, "sample.read": 0}
    baseline = root / "baseline.txt"
    args = ["--root", str(root), "--namespace", "sample", "--baseline", str(baseline)]
    assert main([*args, "--write-baseline"]) == 0
    path = root / "sample/__init__.py"
    path.write_text(path.read_text().replace('_field = ("Any", str)', '_field = ("Any", Any)'))
    baseline_before_growth = baseline.read_text()
    observed = (
        public_any(root, ["sample"]),
        main(args),
        baseline.read_text(),
    )
    assert observed == (
        {"sample.Record": 1, "sample.read": 1},
        1,
        baseline_before_growth,
    )


@pytest.mark.parametrize(
    ("name", "declaration", "before", "after"),
    [
        (
            "Rows",
            'Rows = TypeAliasType("Any", list[str])',
            "list[str]",
            "list[Any]",
        ),
        (
            "Record",
            'Record = NamedTuple("Any", [("Any", str)])',
            '("Any", str)',
            '("Any", Any)',
        ),
        (
            "Record",
            'Record = TypedDict("Any", {"Any": str}, total=False, closed=True)',
            '"Any": str',
            '"Any": Any',
        ),
    ],
    ids=["TypeAliasType", "NamedTuple", "TypedDict"],
)
def test_exported_factory_type_values_count_any(
    tmp_path: Path, name: str, declaration: str, before: str, after: str
) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": f"""
                from typing import Any, TypeAliasType, NamedTuple, TypedDict
                __all__ = ["{name}"]
                {declaration}
            """,
        },
    )
    assert public_any(root, ["sample"]) == {f"sample.{name}": 0}
    baseline = root / "baseline.txt"
    args = ["--root", str(root), "--namespace", "sample", "--baseline", str(baseline)]
    assert main([*args, "--write-baseline"]) == 0
    path = root / "sample/__init__.py"
    path.write_text(path.read_text().replace(before, after))
    baseline_before_growth = baseline.read_text()
    observed = (
        public_any(root, ["sample"]),
        main(args),
        baseline.read_text(),
    )
    assert observed == (
        {f"sample.{name}": 1},
        1,
        baseline_before_growth,
    )


def test_initialized_self_named_imported_any_field_counts_growth(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": """
                from builtins import str as payload
                __all__ = ["Record"]
                class Record:
                    payload: payload = payload
            """,
        },
    )
    assert public_any(root, ["sample"]) == {"sample.Record": 0}
    baseline = root / "baseline.txt"
    args = ["--root", str(root), "--namespace", "sample", "--baseline", str(baseline)]
    assert main([*args, "--write-baseline"]) == 0
    path = root / "sample/__init__.py"
    path.write_text(
        path.read_text().replace(
            "from builtins import str as payload",
            "from typing import Any as payload",
        )
    )
    baseline_before_growth = baseline.read_text()
    observed = (public_any(root, ["sample"]), main(args), baseline.read_text())
    assert observed == ({"sample.Record": 1}, 1, baseline_before_growth)


def test_value_annotation_excludes_internal_factory_initializer_any(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": """
                from typing import Any, TypeAliasType
                __all__ = ["payload"]
                payload: object = TypeAliasType("Internal", str)
            """,
        },
    )
    assert public_any(root, ["sample"]) == {"sample.payload": 0}
    baseline = root / "baseline.txt"
    args = ["--root", str(root), "--namespace", "sample", "--baseline", str(baseline)]
    assert main([*args, "--write-baseline"]) == 0
    path = root / "sample/__init__.py"
    path.write_text(
        path.read_text().replace('TypeAliasType("Internal", str)', 'TypeAliasType("Internal", Any)')
    )
    baseline_before_initializer_change = baseline.read_text()
    observed = (public_any(root, ["sample"]), main(args), baseline.read_text())
    assert observed == ({"sample.payload": 0}, 0, baseline_before_initializer_change)


def test_typed_runtime_mapping_key_any_is_not_annotation(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": """
                from os import environ
                from typing import Any, Mapping
                __all__ = ["environment"]
                config: Mapping[str, str] = environ
                environment = config["str"]
            """,
        },
    )
    assert public_any(root, ["sample"]) == {"sample.environment": 0}
    baseline = root / "baseline.txt"
    args = ["--root", str(root), "--namespace", "sample", "--baseline", str(baseline)]
    assert main([*args, "--write-baseline"]) == 0
    path = root / "sample/__init__.py"
    path.write_text(path.read_text().replace('config["str"]', 'config["Any"]'))
    baseline_before_key_change = baseline.read_text()
    observed = (public_any(root, ["sample"]), main(args), baseline.read_text())
    assert observed == ({"sample.environment": 0}, 0, baseline_before_key_change)


def test_eager_method_annotation_uses_predefinition_class_bindings(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": """
                from builtins import str as read
                __all__ = ["Record"]
                class Record:
                    def read(self) -> read: ...
            """,
        },
    )
    assert public_any(root, ["sample"]) == {"sample.Record": 0}
    baseline = root / "baseline.txt"
    args = ["--root", str(root), "--namespace", "sample", "--baseline", str(baseline)]
    assert main([*args, "--write-baseline"]) == 0
    path = root / "sample/__init__.py"
    path.write_text(
        path.read_text().replace(
            "from builtins import str as read", "from typing import Any as read"
        )
    )
    baseline_before_growth = baseline.read_text()
    observed = (public_any(root, ["sample"]), main(args), baseline.read_text())
    assert observed == ({"sample.Record": 1}, 1, baseline_before_growth)


def test_typed_dict_union_fields_merge_rightmost_and_count_growth(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": """
                from typing import Any, TypedDict
                __all__ = ["Record", "RightWins"]
                Record = TypedDict("Record", {"x": Any} | {"y": str})
                RightWins = TypedDict("RightWins", {"x": Any} | {"x": str})
            """,
        },
    )
    baseline = root / "baseline.txt"
    args = ["--root", str(root), "--namespace", "sample", "--baseline", str(baseline)]
    initial = public_any(root, ["sample"])
    assert main([*args, "--write-baseline"]) == 0
    path = root / "sample/__init__.py"
    path.write_text(path.read_text().replace('{"y": str}', '{"y": Any}'))
    baseline_before_growth = baseline.read_text()
    observed = (
        initial,
        public_any(root, ["sample"]),
        main(args),
        baseline.read_text(),
    )
    assert observed == (
        {"sample.Record": 1, "sample.RightWins": 0},
        {"sample.Record": 2, "sample.RightWins": 0},
        1,
        baseline_before_growth,
    )


def test_typed_dict_unresolved_field_container_fails_closed(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": """
                from typing import TypedDict
                __all__ = ["Record"]
                Record = TypedDict("Record", computed_fields())
            """,
        },
    )

    with pytest.raises(ValueError):
        public_any(root, ["sample"])


def test_pep695_bounds_and_constraints_are_public_annotation_sites(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": """
                from typing import Any
                __all__ = ["produce", "constrained"]
                def produce[T: str]() -> int: ...
                def constrained[T: (str, Any)](value: T) -> None: ...
            """,
        },
    )
    baseline = root / "baseline.txt"
    args = ["--root", str(root), "--namespace", "sample", "--baseline", str(baseline)]
    initial = public_any(root, ["sample"])
    assert main([*args, "--write-baseline"]) == 0
    path = root / "sample/__init__.py"
    path.write_text(path.read_text().replace("def produce[T: str]", "def produce[T: Any]"))
    baseline_before_growth = baseline.read_text()
    observed = (
        initial,
        public_any(root, ["sample"]),
        main(args),
        baseline.read_text(),
    )
    assert observed == (
        {"sample.produce": 0, "sample.constrained": 1},
        {"sample.produce": 1, "sample.constrained": 1},
        1,
        baseline_before_growth,
    )


def test_type_parameter_bounds_use_enclosing_scope_not_class_body(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": """
                from typing import Any, TypeAlias
                __all__ = ["Record", "AnyBoundRecord"]
                _Bound = str
                _AnyBound = Any
                class Record[T: _Bound]:
                    _Bound: TypeAlias = Any
                class AnyBoundRecord[T: _AnyBound]:
                    _AnyBound: TypeAlias = str
            """,
        },
    )
    baseline = root / "baseline.txt"
    args = ["--root", str(root), "--namespace", "sample", "--baseline", str(baseline)]
    initial = public_any(root, ["sample"])
    assert main([*args, "--write-baseline"]) == 0
    path = root / "sample/__init__.py"
    path.write_text(path.read_text().replace("_Bound = str", "_Bound = Any"))
    baseline_before_growth = baseline.read_text()
    observed = (
        initial,
        public_any(root, ["sample"]),
        main(args),
        baseline.read_text(),
    )
    assert observed == (
        {"sample.Record": 0, "sample.AnyBoundRecord": 1},
        {"sample.Record": 1, "sample.AnyBoundRecord": 1},
        1,
        baseline_before_growth,
    )


def test_pep695_bound_scope_is_lazy_but_eager_parameter_stays_any(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": """
                from typing import Any as Bound, Any
                __all__ = ["transform"]
                def transform[T: Bound](value: Bound) -> T: ...
                Bound = str
            """,
        },
    )
    baseline = root / "baseline.txt"
    args = ["--root", str(root), "--namespace", "sample", "--baseline", str(baseline)]
    initial = public_any(root, ["sample"])
    assert main([*args, "--write-baseline"]) == 0
    path = root / "sample/__init__.py"
    path.write_text(path.read_text().replace("Bound = str", "Bound = Any"))
    baseline_before_growth = baseline.read_text()
    observed = (
        initial,
        public_any(root, ["sample"]),
        main(args),
        baseline.read_text(),
    )
    assert observed == (
        {"sample.transform": 1},
        {"sample.transform": 2},
        1,
        baseline_before_growth,
    )


def test_qualified_private_class_alias_counts_signature_growth(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": """
                from typing import Any, TypeAlias
                __all__ = ["read"]
                class _Types:
                    Payload: TypeAlias = str
                def read() -> _Types.Payload: ...
            """,
        },
    )
    assert public_any(root, ["sample"]) == {"sample.read": 0}
    baseline = root / "baseline.txt"
    args = ["--root", str(root), "--namespace", "sample", "--baseline", str(baseline)]
    assert main([*args, "--write-baseline"]) == 0
    path = root / "sample/__init__.py"
    path.write_text(
        path.read_text().replace("Payload: TypeAlias = str", "Payload: TypeAlias = Any")
    )
    baseline_before_growth = baseline.read_text()
    observed = (public_any(root, ["sample"]), main(args), baseline.read_text())
    assert observed == ({"sample.read": 1}, 1, baseline_before_growth)


def test_recursive_pep695_alias_counts_any_finitely_and_rejects_growth(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": """
                from typing import Any
                __all__ = ["Tree"]
                type Tree = int | list[Tree]
            """,
        },
    )
    baseline = root / "baseline.txt"
    args = ["--root", str(root), "--namespace", "sample", "--baseline", str(baseline)]
    initial = public_any(root, ["sample"])
    assert main([*args, "--write-baseline"]) == 0
    path = root / "sample/__init__.py"
    path.write_text(
        path.read_text().replace("type Tree = int | list[Tree]", "type Tree = Any | list[Tree]")
    )
    baseline_before_growth = baseline.read_text()
    observed = (
        initial,
        public_any(root, ["sample"]),
        main(args),
        baseline.read_text(),
    )
    assert observed == (
        {"sample.Tree": 0},
        {"sample.Tree": 1},
        1,
        baseline_before_growth,
    )


def test_typing_star_import_recognizes_generic_aliases(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": """
                from typing import *
                __all__ = ["Rows", "Handler"]
                Rows = Mapping[str, str]
                Handler = Callable[[str], str]
            """,
        },
    )
    baseline = root / "baseline.txt"
    args = ["--root", str(root), "--namespace", "sample", "--baseline", str(baseline)]
    initial = public_any(root, ["sample"])
    assert main([*args, "--write-baseline"]) == 0
    path = root / "sample/__init__.py"
    path.write_text(
        path.read_text()
        .replace("Mapping[str, str]", "Mapping[str, Any]")
        .replace("Callable[[str], str]", "Callable[[Any], str]")
    )
    baseline_before_growth = baseline.read_text()
    observed = (
        initial,
        public_any(root, ["sample"]),
        main(args),
        baseline.read_text(),
    )
    assert observed == (
        {"sample.Rows": 0, "sample.Handler": 0},
        {"sample.Rows": 1, "sample.Handler": 1},
        1,
        baseline_before_growth,
    )


def test_reexported_external_union_and_generic_preserve_any(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": """
                from .types import ExternalRows, ExternalBoxRows
                __all__ = ["ExternalRows", "ExternalBoxRows"]
            """,
            "types.py": """
                from typing import Any, Mapping
                from .bridge import Day, Box
                ExternalRows = tuple[Mapping[str, Any], ...] | Day
                ExternalBoxRows = Box[Any]
            """,
            "bridge.py": """
                from datetime import date as Day
                from third_party import ExternalBox as Box
            """,
        },
    )

    assert public_any(root, ["sample"]) == {
        "sample.ExternalRows": 1,
        "sample.ExternalBoxRows": 1,
    }


def test_classmethod_overload_allowances_are_independent_per_site(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": """
                from typing import Any, overload
                __all__ = ["Builder"]
                class Builder:
                    @classmethod
                    @overload
                    def build(cls, value: Any) -> str: ...
                    @classmethod
                    @overload
                    def build(cls, value: str) -> str: ...
                    @classmethod
                    def build(cls, value: object) -> Any: ...
            """,
        },
    )
    assert public_any(root, ["sample"]) == {"sample.Builder": 2}
    baseline = root / "baseline.txt"
    args = ["--root", str(root), "--namespace", "sample", "--baseline", str(baseline)]
    assert main([*args, "--write-baseline"]) == 0
    path = root / "sample/__init__.py"
    path.write_text(
        path.read_text().replace(
            "    def build(cls, value: Any) -> str: ...\n"
            "    @classmethod\n"
            "    @overload\n"
            "    def build(cls, value: str) -> str: ...",
            "    def build(cls, value: str) -> str: ...\n"
            "    @classmethod\n"
            "    @overload\n"
            "    def build(cls, value: Any) -> str: ...",
        )
    )
    assert public_any(root, ["sample"]) == {"sample.Builder": 2}
    assert main(args) == 1


def test_class_local_aliases_resolve_lexically_across_sibling_classes(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": """
                from typing import Any, TypeAlias
                __all__ = ["Plain", "Dynamic"]
                _Payload: TypeAlias = str
                class Plain:
                    _Payload: TypeAlias = str
                    def read(self) -> _Payload: ...
                class Dynamic:
                    _Payload: TypeAlias = Any
                    def read(self) -> _Payload: ...
            """,
        },
    )

    assert public_any(root, ["sample"]) == {
        "sample.Plain": 0,
        "sample.Dynamic": 1,
    }


def test_star_imports_resolve_typing_and_explicit_local_exports(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": """
                from typing import *
                from .bridge import *
                __all__ = ["convert", "forward"]
                def convert(value: Any) -> Any: ...
            """,
            "bridge.py": """
                from .types import *
                __all__ = ["forward"]
                def forward(value: Dynamic) -> Dynamic: ...
            """,
            "types.py": """
                from typing_extensions import Any as Dynamic
                __all__ = ["Dynamic"]
            """,
        },
    )
    assert public_any(root, ["sample"]) == {"sample.convert": 2, "sample.forward": 2}


def test_counts_any_in_public_fields_not_private_ones(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": """
                from typing import Any, TYPE_CHECKING
                __all__ = ["Record"]
                class _Base:
                    inherited: Any
                    def read(self, key: Any) -> Any: ...
                if TYPE_CHECKING:
                    _Parent = _Base
                else:
                    _Parent = object
                class Record(_Parent, ValueError):
                    value: Any
                    _hidden: Any
                    def __init__(self, value: Any) -> None:
                        self.saved: Any = value
                        self.value: str = ""
                        self._secret: Any = value
                        local: Any = value
                        self._populate(value)
                    def read(self, key: str) -> str: ...
                    def save(self) -> None:
                        self.later: Any = None
                    @property
                    def payload(self) -> Any: ...
                    def _populate(self, key: Any) -> Any:
                        self.populated: Any = key
            """,
        },
    )

    assert public_any(root, ["sample"]) == {"sample.Record": 6}


def test_receiver_fields_in_compound_bodies_exclude_nested_scopes(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": """
                from typing import Any
                __all__ = ["Record"]
                class Record:
                    def __init__(self) -> None:
                        with context():
                            self.opened: Any = None
                        for item in ():
                            self.iterated: Any = item
                        else:
                            self.exhausted: Any = None
                        while condition():
                            self.repeated: Any = None
                        match value():
                            case _:
                                self.matched: Any = None
                        try:
                            self.tried: Any = None
                        except Exception:
                            self.caught: Any = None
                        finally:
                            self.finished: Any = None
                        def nested():
                            self.hidden: Any = None
                        class Nested:
                            self.hidden_too: Any = None
            """,
        },
    )
    assert public_any(root, ["sample"]) == {"sample.Record": 8}


def test_follows_lazy_exports_and_reexports(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": """
                raise RuntimeError("must not import this package")
                _EXTRA = {"fresh": ("sample.bridge", "forward")}
                _EXPORTS = {"renamed": ("sample.bridge", "forward"), **_EXTRA}
                _EXPORTS |= {"tools": ("sample.tools", None)}
                __all__ = list(_EXPORTS)
                def __getattr__(name): ...
            """,
            "bridge.py": "from .impl import original as forward\n",
            "impl.py": """
                from .types import Dynamic
                def original(value: Dynamic) -> Dynamic: ...
            """,
            "types.py": "from typing_extensions import Any as Dynamic",
            "tools.py": """
                from .impl import original
                __all__ = ["original"]
            """,
        },
    )

    assert public_any(root, ["sample"]) == {
        "sample.renamed": 2,
        "sample.fresh": 2,
        "sample.tools": 0,
        "sample.tools.original": 2,
    }


@pytest.mark.parametrize(
    "files",
    [
        {"__init__.py": '__all__ = ["missing"]'},
        {
            "__init__.py": 'from .cycle import missing\n__all__ = ["missing"]',
            "cycle.py": "from . import missing",
        },
        {"__init__.py": "__all__ = compute_exports()"},
        {
            "__init__.py": """
                from typing import TypeAlias
                __all__ = ["convert"]
                First: TypeAlias = "Second"
                type Second = First
                def convert(value: First) -> None: ...
            """,
        },
        {"__init__.py": '__all__ = ["value"]\n__all__ *= 2\nvalue = 1'},
    ],
    ids=["missing-export", "cyclic-reexport", "dynamic-exports", "cyclic-alias", "mutation"],
)
def test_unresolved_exports_fail_closed(tmp_path: Path, files: dict[str, str]) -> None:
    root = _package(tmp_path, **files)

    with pytest.raises(ValueError):
        public_any(root, ["sample"])


def test_fails_when_a_count_grows(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": """
                from typing import Any
                __all__ = ["convert"]
                def convert(value: Any) -> str: ...
            """,
        },
    )
    baseline = root / "baseline.txt"
    args = ["--root", str(root), "--namespace", "sample", "--baseline", str(baseline)]
    assert main([*args, "--write-baseline"]) == 0
    assert main(args) == 0
    (root / "sample/__init__.py").write_text(
        'from typing import Any\n__all__ = ["convert"]\ndef convert(value: Any) -> Any: ...\n'
    )

    assert main(args) == 1


def test_existing_counts_cannot_hide_new_names_or_annotations(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": """
                from typing import Any
                __all__ = ["convert"]
                def convert(value: Any) -> str: ...
            """,
        },
    )
    baseline = root / "baseline.txt"
    args = ["--root", str(root), "--namespace", "sample", "--baseline", str(baseline)]
    assert main([*args, "--write-baseline"]) == 0
    path = root / "sample/__init__.py"
    path.write_text(
        'from typing import Any\n__all__ = ["convert"]\ndef convert(value: str) -> Any: ...\n'
    )
    assert main(args) == 1
    path.write_text(
        '__all__ = ["convert"]\n__all__.append("new")\n'
        "def convert(value: str) -> str: ...\nnew = 1\n"
    )
    assert main(args) == 1
    path.write_text('__all__ = ["convert"]\ndef convert(value: str) -> str: ...\n')
    assert main(args) == 0
    path.write_text(
        '_EXPORTS = {"convert": ("sample.impl", "convert")}\n'
        '_EXPORTS |= {"new": ("sample.impl", "new")}\n__all__ = list(_EXPORTS)\n'
    )
    (root / "sample/impl.py").write_text("def convert(value: str) -> str: ...\nnew = 1\n")
    assert main(args) == 1


def test_default_namespaces_use_manifest_headers_and_source_names(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        **{
            "__init__.py": '__all__ = ["value"]\nvalue: int = 1',
            "other.py": '__all__ = ["field"]\nfield: int = 2',
            "_private.py": '__all__ = ["hidden"]\nhidden: int = 3',
            "internal.py": '__all__ = ["unresolved"]',
        },
    )
    (root / "tests").mkdir()
    (root / "tests/public_exports.txt").write_text(
        "[sample]\nstale_name: not an export list\n[sample.other]\n"
    )
    baseline = root / "baseline.txt"
    args = ["--root", str(root), "--package", "sample", "--baseline", str(baseline)]
    assert main([*args, "--write-baseline"]) == 0
    (root / "sample/other.py").write_text('__all__ = ["field", "new"]\nfield = 2\nnew = 3')

    assert main(args) == 1
