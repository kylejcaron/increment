from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from scripts.check_public_typing import main, public_any


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
                from .types import Declared, Modern, Quoted
                __all__ = ["convert"]
                def convert(value: Declared, extra: "Modern") -> Quoted: ...
            """,
            "types.py": """
                from typing import Any, TypeAlias as Alias
                Declared: Alias = Any
                type Modern = list[Declared]
                Quoted: Alias = "Modern"
            """,
        },
    )
    assert public_any(root, ["sample"]) == {"sample.convert": 3}


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
