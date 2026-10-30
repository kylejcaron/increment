from __future__ import annotations

import fnmatch
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import cast

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXTENSION_LAYERS = {"designs", "observational", "program", "artifacts"}


def _config() -> tuple[dict[str, object], dict[str, list[str]]]:
    tach = tomllib.loads((ROOT / "tach.toml").read_text())
    layer_map = tomllib.loads((ROOT / "tests/layer_map.toml").read_text())
    return tach, layer_map


def _module_layers(config: dict[str, object], layer_map: dict[str, list[str]]) -> dict[str, str]:
    raw_modules = config["modules"]
    assert isinstance(raw_modules, list)
    modules = cast(list[dict[str, object]], raw_modules)
    result: dict[str, str] = {}
    for module in modules:
        path = cast(str, module["path"])
        matches = [
            (len(pattern), layer)
            for layer, patterns in layer_map.items()
            for pattern in patterns
            if fnmatch.fnmatchcase(path, pattern)
        ]
        assert matches, f"{path} has no layer"
        longest = max(length for length, _layer in matches)
        owners = {layer for length, layer in matches if length == longest}
        assert len(owners) == 1, f"{path} has ambiguous layer: {owners}"
        expected = owners.pop()
        assert module.get("layer") == expected, (
            f"{path} tach layer disagrees with tests/layer_map.toml"
        )
        result[path] = expected
    return result


def _assert_layer_dependencies_are_valid(config: dict[str, object], layers: dict[str, str]) -> None:
    raw_modules = config["modules"]
    assert isinstance(raw_modules, list)
    modules = cast(list[dict[str, object]], raw_modules)
    violations = [
        (path, dependency)
        for module in modules
        for path in [cast(str, module["path"])]
        for dependency in cast(list[str], module.get("depends_on", []))
        if (layers[path] == "core" and layers.get(dependency) in EXTENSION_LAYERS)
        or (
            layers[path] in EXTENSION_LAYERS
            and layers.get(dependency) in EXTENSION_LAYERS
            and layers[path] != layers[dependency]
        )
    ]
    assert not violations, violations


def test_extension_layers_depend_on_core_only() -> None:
    config, layer_map = _config()
    layers = _module_layers(config, layer_map)
    _assert_layer_dependencies_are_valid(config, layers)


def test_core_layer_does_not_import_extensions() -> None:
    config, layer_map = _config()
    layers = _module_layers(config, layer_map)
    _assert_layer_dependencies_are_valid(config, layers)


def test_query_artifact_and_ledger_paths_stay_in_core_exception_boundary() -> None:
    config, layer_map = _config()
    layers = _module_layers(config, layer_map)
    assert layers["increment.query"] == "core"
    core_patterns = layer_map["core"]
    extension_patterns = [
        pattern
        for layer, patterns in layer_map.items()
        if layer in EXTENSION_LAYERS
        for pattern in patterns
    ]
    query_paths = (
        "increment.query.artifact_contract",
        "increment.query.artifact_digest",
        "increment.query.artifact_extensions",
        "increment.query.artifact_publish",
        "increment.query.artifact_reader",
        "increment.query.schemas",
        "increment.query.sequential_capture",
        "increment.query.ledger",
    )
    assert all(
        any(fnmatch.fnmatchcase(path, pattern) for pattern in core_patterns)
        and not any(fnmatch.fnmatchcase(path, pattern) for pattern in extension_patterns)
        for path in query_paths
    )


@pytest.mark.integration
def test_cross_extension_import_fails_tach_check(tmp_path: Path) -> None:
    tach = shutil.which("tach")
    assert tach is not None
    (tmp_path / "increment" / "core").mkdir(parents=True)
    (tmp_path / "increment" / "designs").mkdir()
    (tmp_path / "increment" / "observational").mkdir()
    for module in ("core", "designs", "observational"):
        (tmp_path / "increment" / module / "__init__.py").write_text("")
    (tmp_path / "increment" / "designs" / "__init__.py").write_text(
        "from increment.observational import value\n"
    )
    (tmp_path / "increment" / "observational" / "__init__.py").write_text("value = 1\n")
    (tmp_path / "increment" / "core" / "__init__.py").write_text("value = 1\n")
    (tmp_path / "tach.toml").write_text(
        'source_roots = ["."]\n'
        'root_module = "ignore"\n'
        'layers = ["observational", "designs", "core"]\n'
        '[[modules]]\npath = "increment.designs"\n'
        'layer = "designs"\ndepends_on = ["increment.core"]\n'
        '[[modules]]\npath = "increment.observational"\n'
        'layer = "observational"\ndepends_on = ["increment.core"]\n'
        '[[modules]]\npath = "increment.core"\n'
        'layer = "core"\ndepends_on = []\n'
    )
    result = subprocess.run(
        [tach, "check"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0


@pytest.mark.integration
def test_artifact_public_ingresses_remain_available() -> None:
    from increment.query.artifact_contract import ArtifactStore
    from increment.query.artifact_publish import ArtifactPublisher
    from increment.query.artifact_reader import ArtifactMomentSource as ReaderSource
    from increment.query.source import ArtifactMomentSource as FacadeSource

    assert all(
        isinstance(public_type, type)
        for public_type in (ArtifactStore, ArtifactPublisher, ReaderSource, FacadeSource)
    )
