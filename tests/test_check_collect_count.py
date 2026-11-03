from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from scripts.check_collect_count import Collection, _check_fanout, collect, main, report

pytestmark = pytest.mark.tooling

ROOT = Path(__file__).resolve().parents[1]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _repo(tmp_path: Path, files: dict[str, str]) -> Path:
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    for name, text in files.items():
        (repo / name).write_text(text)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def test_actual_class_method_nodeid_honors_reasoned_grid_marker(tmp_path: Path) -> None:
    tests_root = tmp_path / "tests"
    tests_root.mkdir()
    (tests_root / "test_grid.py").write_text(
        "import pytest\n"
        "class TestGrid:\n"
        "    @pytest.mark.grid('large sweep')\n"
        "    def test_marked(self): pass\n"
    )
    relpath = os.path.relpath(tests_root / "test_grid.py", ROOT)
    nodeids = "".join(f"{relpath}::TestGrid::test_marked[{i}]\n" for i in range(13))

    assert _check_fanout(nodeids, tests_root) == []


@pytest.mark.integration
class TestCollection:
    def test_each_tier_counts_items_by_their_markers(self, tmp_path: Path) -> None:
        repo = _repo(
            tmp_path,
            {
                "tests/test_mix.py": (
                    "import pytest\n"
                    "def test_plain(): pass\n"
                    "@pytest.mark.integration\n"
                    "def test_boundary(): pass\n"
                    "@pytest.mark.slow\n"
                    "def test_heavy(): pass\n"
                    "@pytest.mark.slow\n"
                    "@pytest.mark.parameter_recovery\n"
                    "def test_recovery(): pass\n"
                )
            },
        )

        counts = collect(repo).tiers

        assert counts == {"unit": 1, "fast": 2, "slow": 1, "tooling": 0, "all": 4}

    def test_check_items_report_exact_tooling_node_ids(self) -> None:
        tiers = ("unit", "fast", "slow", "tooling", "all")
        unit_fast = "tests/test_tooling.py::test_fast"
        slow = "tests/test_tooling.py::test_slow"
        base_nodes = {
            "unit": (unit_fast,),
            "fast": (unit_fast,),
            "slow": (slow,),
            "tooling": (),
            "all": (unit_fast, slow),
        }
        head_nodes = {
            "unit": (),
            "fast": (),
            "slow": (),
            "tooling": (unit_fast, slow),
            "all": (unit_fast, slow),
        }
        base = Collection(
            tiers={tier: len(base_nodes[tier]) for tier in tiers},
            nodeids=tuple(sorted({node for values in base_nodes.values() for node in values})),
            tier_nodeids=base_nodes,
        )
        head = Collection(
            tiers={tier: len(head_nodes[tier]) for tier in tiers},
            nodeids=base.nodeids,
            tier_nodeids=head_nodes,
        )

        changes = report(
            base.tiers,
            head.tiers,
            base_nodeids=base.tier_nodeids,
            head_nodeids=head.tier_nodeids,
        )

        assert "unit removed node IDs:\n  tests/test_tooling.py::test_fast\n" in changes
        assert "fast removed node IDs:\n  tests/test_tooling.py::test_fast\n" in changes
        assert "slow removed node IDs:\n  tests/test_tooling.py::test_slow\n" in changes
        assert (
            "tooling added node IDs:\n"
            "  tests/test_tooling.py::test_fast\n"
            "  tests/test_tooling.py::test_slow\n"
        ) in changes

    def test_working_tree_delta_and_unmarked_grid_are_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = _repo(tmp_path, {"tests/test_one.py": "def test_one(): pass\n"})
        (repo / "tests/test_grid.py").write_text(
            "import pytest\n"
            "@pytest.mark.parametrize('value', range(13))\n"
            "def test_unmarked(value): pass\n"
            "@pytest.mark.parametrize('value', range(13))\n"
            "@pytest.mark.grid('large sweep')\n"
            "def test_marked(value): pass\n"
        )
        monkeypatch.chdir(repo)

        assert main(["--base", "main", "--repo", str(repo)]) == 0

        out = capsys.readouterr().out
        assert "unit 1 -> 27 (+26)" in out
        assert "test_unmarked: 13 collected items" in out
        assert "test_marked: 13 collected items" not in out

    def test_make_check_items_reports_the_working_tree_delta(self, tmp_path: Path) -> None:
        repo = _repo(tmp_path, {"tests/test_one.py": "def test_one(): pass\n"})
        (repo / "tests/test_two.py").write_text("def test_two(): pass\n")

        result = subprocess.run(
            ["make", "check-items", f"CHECK_ITEMS_REPO={repo}", "BASE=main"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr + result.stdout
        assert "unit 1 -> 2 (+1)" in result.stdout

    def test_plain_make_defaults_to_install(self) -> None:
        result = subprocess.run(
            ["make", "--dry-run"], cwd=ROOT, text=True, capture_output=True, check=False
        )
        assert result.returncode == 0
        assert "uv sync --group dev --extra demo --extra tables --extra dashboard" in result.stdout
