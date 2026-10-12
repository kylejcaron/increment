"""Collection refuses expensive-purpose tests without an effective cost mark."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run_test_tier import _clean_pytest_environment

pytestmark = pytest.mark.integration


def _collect_and_run(tmp_path: Path, source: str) -> subprocess.CompletedProcess[str]:
    config = tmp_path / "pytest.ini"
    config.write_text(
        "[pytest]\nmarkers = slow: expensive\n parameter_recovery: calibration\n"
        " examples: notebook\n stress: exhaustive\n numerical_certification: exact inferential certification\n"
    )
    project = Path(__file__).resolve().parents[1]
    (tmp_path / "conftest.py").write_text((project / "conftest.py").read_text())
    (tmp_path / "test_consumer.py").write_text(source)
    environment = _clean_pytest_environment(os.environ)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["GITHUB_ACTIONS"] = "true"
    environment["PYTHONPATH"] = str(project)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(config),
            "--confcutdir",
            str(tmp_path),
            "-q",
            str(tmp_path / "test_consumer.py"),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )


@pytest.mark.parametrize(
    "category", ["parameter_recovery", "numerical_certification", "examples", "stress"]
)
def test_expensive_category_refuses_before_running_a_consumer(tmp_path: Path, category: str):
    result = _collect_and_run(
        tmp_path,
        f"import pytest\nfrom pathlib import Path\n@pytest.mark.{category}\n"
        "def test_consumer():\n    Path('executed').write_text('unexpected')\n",
    )
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, result.stdout + result.stderr
    assert not (tmp_path / "executed").exists()


@pytest.mark.parametrize("scope", ["module", "class", "parameter"])
def test_inherited_slow_mark_allows_the_consumer(tmp_path: Path, scope: str):
    sources = {
        "module": "pytestmark = pytest.mark.slow\n@pytest.mark.parameter_recovery\n"
        "def test_consumer():\n    Path('executed').write_text('5')\n",
        "class": "@pytest.mark.slow\nclass TestConsumer:\n    @pytest.mark.examples\n"
        "    def test_consumer(self):\n        Path('executed').write_text('5')\n",
        "parameter": "@pytest.mark.stress\n@pytest.mark.parametrize('value', "
        "[pytest.param(5, marks=pytest.mark.slow)])\n"
        "def test_consumer(value):\n    Path('executed').write_text(str(value))\n",
    }
    result = _collect_and_run(
        tmp_path, "import pytest\nfrom pathlib import Path\n" + sources[scope]
    )
    assert result.returncode == pytest.ExitCode.OK, result.stdout + result.stderr
    assert (tmp_path / "executed").read_text() == "5"


@pytest.mark.parametrize("scope", ["module", "class", "parameter"])
def test_numerical_certification_inherits_slow_cost_mark(tmp_path: Path, scope: str):
    sources = {
        "module": "pytestmark = pytest.mark.slow\n@pytest.mark.numerical_certification\n"
        "def test_consumer():\n    Path('executed').write_text('5')\n",
        "class": "@pytest.mark.slow\nclass TestConsumer:\n"
        "    @pytest.mark.numerical_certification\n"
        "    def test_consumer(self):\n        Path('executed').write_text('5')\n",
        "parameter": "@pytest.mark.stress\n@pytest.mark.parametrize('value', "
        "[pytest.param(5, marks=[pytest.mark.slow, pytest.mark.numerical_certification])])\n"
        "def test_consumer(value):\n    Path('executed').write_text(str(value))\n",
    }
    result = _collect_and_run(
        tmp_path, "import pytest\nfrom pathlib import Path\n" + sources[scope]
    )
    assert result.returncode == pytest.ExitCode.OK, result.stdout + result.stderr
    assert (tmp_path / "executed").read_text() == "5"
