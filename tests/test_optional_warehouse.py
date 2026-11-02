from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from increment import artifacts, from_definitions, from_unit_day_artifact, query, workflows
from increment.errors import CodedError
from increment.estimators import MetricTrends
from increment.semantics.loader import load

ROOT = Path(__file__).resolve().parents[1]
MISSING = "install.warehouse_extra_missing"
INSTALL = 'pip install "increment[warehouse]"'
_ANY = cast(Any, None)

# Each call needs the warehouse install; its arguments are never read.
WAREHOUSE_CALLS: dict[str, Callable[[], object]] = {
    "from_unit_day_artifact": lambda: from_unit_day_artifact(_ANY, _ANY, expected_context=_ANY),
    "build_panel_for_metric": lambda: query.build_panel_for_metric(_ANY, _ANY, _ANY, _ANY),
    "panel_sql": lambda: query.panel_sql(_ANY, _ANY),
    "summary_sql": lambda: query.summary_sql(_ANY, _ANY),
    "artifact_context": lambda: artifacts.artifact_context(_ANY, _ANY),
    "publish_unit_day_artifact": lambda: artifacts.publish_unit_day_artifact(_ANY, _ANY, _ANY),
    "materialize": lambda: workflows.materialize(_ANY, _ANY),
    "experiment_definition": lambda: workflows.experiment_definition(_ANY, _ANY),
    "breakout_summaries": lambda: workflows.breakout_summaries(_ANY, _ANY),
    "factor_summaries": lambda: workflows.factor_summaries(_ANY, _ANY),
    "trigger_rates": lambda: workflows.trigger_rates(_ANY, _ANY),
    "available_metrics": lambda: workflows.available_metrics(_ANY, _ANY),
    "MetricTrends": lambda: MetricTrends(_ANY, _ANY),
}

PUBLIC_NAMESPACES = (
    "increment",
    "increment.results",
    "increment.power",
    "increment.artifacts",
    "increment.digest",
    "increment.estimators",
    "increment.evidence",
    "increment.workflows",
)

# Resolves public names and documented direct imports in a fresh interpreter.
_PUBLIC_NAMES_PROBE = """
import importlib, importlib.abc, importlib.util, json, sys
from increment.errors import CodedError

dependency, missing, install, *namespaces = sys.argv[1:]
if dependency == "broken":
    class BrokenWarehouseImport(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "ibis":
                return importlib.util.spec_from_loader(fullname, self)
        def create_module(self, spec):
            return None
        def exec_module(self, module):
            raise ModuleNotFoundError(name="warehouse_dependency_failure")
    sys.meta_path.insert(0, BrokenWarehouseImport())
else:
    sys.modules[dependency] = None

failures = {}
for namespace, name in (
    ("increment.query.session", "WarehouseArtifactStore"),
    ("increment.query.artifact_digest", "ArtifactDigestError"),
):
    route = f"{namespace}.{name}"
    try:
        getattr(importlib.import_module(namespace), name)
    except CodedError as exc:
        expected = {"feature": namespace, "install": install}
        if dependency == "broken" or exc.code != missing or dict(exc.context) != expected:
            failures[route] = [exc.code, dict(exc.context)]
    except ModuleNotFoundError as exc:
        if dependency != "broken" or exc.name != "warehouse_dependency_failure":
            failures[route] = [type(exc).__name__, exc.name]
    except Exception as exc:
        failures[route] = type(exc).__name__
    else:
        failures[route] = "missing refusal"

for namespace, name in (
    ("increment.query.artifact_contract", "open_trusted_manifest_snapshot"),
    ("increment.query.artifact_ports", "ArtifactQueryPorts"),
):
    getattr(importlib.import_module(namespace), name)

for namespace in namespaces if dependency != "broken" else ():
    module = importlib.import_module(namespace)
    for name in module.__all__:
        try:
            getattr(module, name)
        except CodedError as exc:
            expected = {"feature": f"{namespace}.{name}", "install": install}
            if exc.code != missing or dict(exc.context) != expected:
                failures[f"{namespace}.{name}"] = [exc.code, dict(exc.context)]
        except Exception as exc:
            failures[f"{namespace}.{name}"] = type(exc).__name__
print(json.dumps(failures))
"""


@pytest.mark.parametrize("module", ["ibis", "sqlglot"])
def test_definitions_without_the_warehouse_extra_refuses_with_the_install_route(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, module: str
) -> None:
    monkeypatch.setitem(sys.modules, module, None)

    with pytest.raises(CodedError) as refused:
        from_definitions(tmp_path)

    assert refused.value.code == MISSING
    assert refused.value.context == {"feature": "from_definitions", "install": INSTALL}


@pytest.mark.grid("Each warehouse-only function independently checks its install before binding")
@pytest.mark.parametrize("feature", sorted(WAREHOUSE_CALLS))
def test_warehouse_entry_points_refuse_before_reading_their_arguments(
    monkeypatch: pytest.MonkeyPatch, feature: str
) -> None:
    monkeypatch.setitem(sys.modules, "ibis", None)

    with pytest.raises(CodedError) as refused:
        WAREHOUSE_CALLS[feature]()

    assert refused.value.code == MISSING
    assert refused.value.context == {"feature": feature, "install": INSTALL}


def test_definition_sql_without_the_warehouse_extra_refuses_with_the_install_route(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "sqlglot", None)
    definitions = tmp_path / "definitions.yaml"
    definitions.write_text("fact_sources:\n  - name: orders\n    sql: SELECT 1 AS unit_id\n")

    with pytest.raises(CodedError) as refused:
        load(definitions)

    assert refused.value.code == MISSING
    assert refused.value.context["install"] == INSTALL


@pytest.mark.integration
def test_public_names_resolve_or_refuse_with_the_install_route_without_the_warehouse_extra() -> (
    None
):
    for dependency in ("ibis", "sqlglot", "broken"):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                _PUBLIC_NAMES_PROBE,
                dependency,
                MISSING,
                INSTALL,
                *PUBLIC_NAMESPACES,
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == {}, dependency


@pytest.mark.integration
def test_dataframe_run_never_imports_ibis_or_sqlglot() -> None:
    readme = (ROOT / "README.md").read_text()
    quickstart = readme.split("## Quickstart", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    match = re.search(r"```python\s*\n(.*?)\n```", quickstart, flags=re.DOTALL)
    assert match is not None
    probe = (
        match.group(1)
        + "\nimport json, sys\n"
        + "print(json.dumps(sorted({'ibis', 'sqlglot'} & set(sys.modules))))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=ROOT, check=False
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == []
