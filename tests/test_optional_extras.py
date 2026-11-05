from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from increment import tables
from increment.errors import CodedError

ROOT = Path(__file__).resolve().parents[1]

# Imports the dashboard in a fresh interpreter with one dependency blocked.
_DASHBOARD_PROBE = """
import json, sys
sys.modules[sys.argv[1]] = None
from increment.errors import CodedError
try:
    import increment.dashboard
except CodedError as exc:
    refusal = {
        "code": exc.code,
        "context": dict(exc.context),
        "import_error": isinstance(exc, ImportError),
    }
else:
    refusal = None
print(json.dumps(refusal))
"""


@pytest.mark.integration
@pytest.mark.parametrize("module", ["marimo", "anywidget", "coeftable", "pandas"])
def test_dashboard_without_its_extra_refuses_with_the_install_route(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _DASHBOARD_PROBE, module],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "code": "install.dashboard_extra_missing",
        "context": {
            "feature": "increment.dashboard",
            "install": 'pip install "increment[dashboard]"',
        },
        "import_error": True,
    }


def test_readout_table_without_the_tables_extra_refuses_with_the_install_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "coeftable", None)

    with pytest.raises(CodedError) as refused:
        tables.readout_table([])

    assert refused.value.code == "install.tables_extra_missing"
    assert refused.value.context == {
        "feature": "readout_table",
        "install": 'pip install "increment[tables]"',
    }
