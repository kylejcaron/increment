import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


SCRIPT = Path(__file__).parents[1] / "scripts" / "check_deferred_imports.py"


def _package(tmp_path: Path) -> Path:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "a.py").write_text("")
    (package / "b.py").write_text("")
    return package


def _run(package: Path, baseline: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(package), "--baseline", str(baseline)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_new_deferred_edge_fails_with_its_module_pair(tmp_path: Path) -> None:
    package = _package(tmp_path)
    (package / "a.py").write_text("def load():\n    import pkg.b\n")
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("# deferred\n# cycle\n")

    result = _run(package, baseline)

    assert result.returncode == 1
    assert "pkg.a -> pkg.b" in result.stdout


@pytest.mark.parametrize(
    "guard",
    [
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:",
        "import typing\nif typing.TYPE_CHECKING:",
    ],
)
def test_type_checking_import_is_excluded_from_runtime_cycle(tmp_path: Path, guard: str) -> None:
    package = _package(tmp_path)
    (package / "a.py").write_text(f"{guard}\n    import pkg.b\n")
    (package / "b.py").write_text("import pkg.a\n")
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("# deferred\n# cycle\n")

    result = _run(package, baseline)

    assert result.returncode == 0


def test_deferred_cycle_is_rejected_until_its_members_are_baselined(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    (package / "a.py").write_text("def load():\n    import pkg.b\n")
    (package / "b.py").write_text("import pkg.a\n")
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("# deferred\npkg.a -> pkg.b\n# cycle\n")

    result = _run(package, baseline)

    assert result.returncode == 1
    assert "runtime cycle: pkg.a, pkg.b" in result.stdout

    baseline.write_text("# deferred\npkg.a -> pkg.b\n# cycle\npkg.a\npkg.b\n")
    result = _run(package, baseline)

    assert result.returncode == 0


def test_mixed_from_import_keeps_base_module_cycle(tmp_path: Path) -> None:
    package = _package(tmp_path)
    (package / "p").mkdir()
    (package / "p" / "__init__.py").write_text("VALUE = 1\nfrom pkg.a import marker\n")
    (package / "p" / "child.py").write_text("")
    (package / "a.py").write_text("from pkg.p import VALUE, child\n")
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("# deferred\n# cycle\n")

    result = _run(package, baseline)

    assert result.returncode == 1
    assert "runtime cycle: pkg.a, pkg.p" in result.stdout


def test_from_import_of_only_submodule_keeps_package_cycle(tmp_path: Path) -> None:
    package = _package(tmp_path)
    (package / "p").mkdir()
    (package / "p" / "__init__.py").write_text("from pkg.a import marker\n")
    (package / "p" / "child.py").write_text("")
    (package / "a.py").write_text("from pkg.p import child\n")
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("# deferred\n# cycle\n")

    result = _run(package, baseline)

    assert result.returncode == 1
    assert "runtime cycle: pkg.a, pkg.p" in result.stdout


def test_package_reexport_edges_do_not_form_runtime_cycle(tmp_path: Path) -> None:
    package = _package(tmp_path)
    (package / "__init__.py").write_text("VALUE = 1\nfrom pkg import child\n")
    (package / "child.py").write_text("from pkg import VALUE\n")
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("# deferred\n# cycle\n")

    result = _run(package, baseline)

    assert result.returncode == 0


def test_dotted_child_import_still_detects_ancestor_package_cycle(tmp_path: Path) -> None:
    package = _package(tmp_path)
    (package / "p").mkdir()
    (package / "p" / "__init__.py").write_text("import pkg.a\n")
    (package / "p" / "child.py").write_text("")
    (package / "a.py").write_text("import pkg.p.child\n")
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("# deferred\n# cycle\n")

    result = _run(package, baseline)

    assert result.returncode == 1
    assert "runtime cycle: pkg.a, pkg.p" in result.stdout


@pytest.mark.parametrize(
    "source",
    [
        "import importlib\ndef load():\n    importlib.import_module('pkg.b')",
        "import importlib as loader\ndef load():\n    loader.import_module('pkg.b')",
        "from importlib import import_module as load_module\ndef load():\n    load_module('pkg.b')",
        "from importlib import import_module\ndef load():\n    import_module('.b', package='pkg')",
        "from importlib import import_module\ndef load():\n    import_module(name='pkg.b')",
        "from importlib import import_module\ndef load():\n    import_module('.b', 'pkg')",
    ],
)
def test_literal_import_module_call_closes_cycle(tmp_path: Path, source: str) -> None:
    package = _package(tmp_path)
    (package / "a.py").write_text(source)
    (package / "b.py").write_text("import pkg.a\n")
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("# deferred\npkg.a -> pkg.b\n# cycle\n")

    result = _run(package, baseline)

    assert result.returncode == 1
    assert "runtime cycle: pkg.a, pkg.b" in result.stdout


def test_importlib_alias_shadowed_by_function_parameter_is_not_dynamic_import(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    (package / "a.py").write_text(
        "import importlib as loader\ndef load(loader):\n    loader.import_module('pkg.b')\n"
    )
    (package / "b.py").write_text("import pkg.a\n")
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("# deferred\n# cycle\n")

    result = _run(package, baseline)

    assert result.returncode == 0


def test_function_local_importlib_alias_does_not_leak_to_another_function(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    (package / "a.py").write_text(
        "def bind_locally():\n    import importlib as loader\n"
        "def load():\n    loader.import_module('pkg.b')\n"
    )
    (package / "b.py").write_text("import pkg.a\n")
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("# deferred\n# cycle\n")

    result = _run(package, baseline)

    assert result.returncode == 0


@pytest.mark.parametrize(
    "source",
    [
        "import pkg.p.child",
        "from pkg.p.child import marker",
        "from importlib import import_module\nimport_module('pkg.p.child')",
    ],
)
def test_nested_import_records_ancestor_package_cycle(tmp_path: Path, source: str) -> None:
    package = _package(tmp_path)
    (package / "p").mkdir()
    (package / "p" / "__init__.py").write_text("import pkg.a\n")
    (package / "p" / "child.py").write_text("marker = 1\n")
    (package / "a.py").write_text(source)
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("# deferred\n# cycle\n")

    result = _run(package, baseline)

    assert result.returncode == 1
    assert "runtime cycle: pkg.a, pkg.p" in result.stdout


@pytest.mark.parametrize(
    "definition",
    [
        "@import_module('pkg.b')\ndef load(value=import_module('pkg.b')):\n    pass",
        "@import_module('pkg.b')\nasync def load(value=import_module('pkg.b')):\n    pass",
    ],
)
def test_function_default_import_runs_at_definition_time(tmp_path: Path, definition: str) -> None:
    package = _package(tmp_path)
    (package / "a.py").write_text(f"from importlib import import_module\n{definition}\n")
    (package / "b.py").write_text("import pkg.a\n")
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("# deferred\n# cycle\n")

    result = _run(package, baseline)

    assert result.returncode == 1
    assert "runtime cycle: pkg.a, pkg.b" in result.stdout


def test_removing_an_edge_from_the_baseline_is_allowed(tmp_path: Path) -> None:
    package = _package(tmp_path)
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("# deferred\npkg.a -> pkg.b\n# cycle\n")

    result = _run(package, baseline)

    assert result.returncode == 0


def test_external_deferred_import_is_ignored_and_relative_internal_import_is_tracked(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    (package / "__init__.py").write_text("def load():\n    from . import b\n")
    (package / "a.py").write_text("def load():\n    import ibis\n    from . import b\n")
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("# deferred\n# cycle\n")

    result = _run(package, baseline)

    assert result.returncode == 1
    assert "pkg.a -> pkg.b" in result.stdout
    assert "pkg -> pkg.b" in result.stdout
    assert "ibis" not in result.stdout
