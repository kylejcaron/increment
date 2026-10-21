"""Exercise worker exit statuses with real mutmut and fresh pytest processes."""

import os
import signal
from collections import Counter
from pathlib import Path

import pytest

from scripts.mutation_campaign import COLLECTION_ERROR, RUNNER_ERROR, _rates, classify, execute


@pytest.fixture
def worker_project(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    (tmp_path / "sample").mkdir()
    (tmp_path / "sample/__init__.py").write_text("")
    (tmp_path / "sample/value.py").write_text("def value():\n    return 1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_value.py").write_text(
        "from sample.value import value\n"
        "def test_value():\n    assert value() == 1\n"
        "def test_z_redundant():\n    assert value() == 1\n"
        "def test_zz_no_kill():\n    assert value() in (1, 2)\n"
    )
    # Its own pytest section makes the sample the rootdir, so conftest discovery
    # stops here even when the temporary directory sits inside this repository.
    (tmp_path / "pyproject.toml").write_text(
        '[tool.mutmut]\nsource_paths = ["sample"]\n'
        'only_mutate = ["sample/value.py"]\n'
        'pytest_add_cli_args_test_selection = ["tests/test_value.py"]\n'
        "\n[tool.pytest.ini_options]\n"
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTEST_ADDOPTS", None)
    env.update(MUTANT_UNDER_TEST="", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    log = tmp_path / "generate.log"
    code, _ = execute("generate", "sample/value.py", [], tmp_path, env, log, 30)
    assert code == 0, log.read_text()
    return tmp_path, env


@pytest.mark.slow
@pytest.mark.parametrize(
    ("scenario", "expected_code", "expected_status"),
    [
        ("clean", 0, "survived"),
        ("assertion", 1, "killed"),
        ("forced_fail", 1, "killed"),
        ("crash", -signal.SIGKILL, "crashed"),
        ("wrong_path", RUNNER_ERROR, "runner error"),
        ("forced_fail_wrong_path", RUNNER_ERROR, "runner error"),
        ("import_error", RUNNER_ERROR, "runner error"),
        ("system_exit", RUNNER_ERROR, "runner error"),
        ("generation_error", RUNNER_ERROR, "runner error"),
        ("evidence_error", RUNNER_ERROR, "runner error"),
        ("collection_error", COLLECTION_ERROR, "collection error"),
        ("forced_fail_collection_error", COLLECTION_ERROR, "collection error"),
        ("usage_error", 4, "runner error"),
        ("no_tests", 5, "no tests"),
    ],
)
def test_worker_outcomes(worker_project, scenario, expected_code, expected_status):
    work, env = worker_project
    source = "sample/value.py"
    stage = "test"
    tests = []
    test_file = work / "mutants/tests/test_value.py"
    if scenario.startswith("forced_fail"):
        env["MUTANT_UNDER_TEST"] = "fail"
    if scenario.endswith("wrong_path"):
        source = "json.py"
    elif scenario == "assertion":
        test_file.write_text("def test_value():\n    assert False\n")
    elif scenario == "import_error":
        source = "sample/missing.py"
    elif scenario == "system_exit":
        (work / "mutants/sample/value.py").write_text("raise SystemExit(1)\n")
    elif scenario == "crash":
        test_file.write_text(
            "import os, signal\n\ndef test_value():\n    os.kill(os.getpid(), signal.SIGKILL)\n"
        )
    elif scenario == "generation_error":
        stage = "generate"
    elif scenario == "evidence_error":
        stage = "stats"
        (work / "associations.json").mkdir()
    elif scenario.endswith("collection_error"):
        test_file.write_text("raise TypeError('collection failed')\n")
        tests = ["tests/test_value.py::test_value"]
    elif scenario == "usage_error":
        tests = ["tests/test_value.py::missing"]
    elif scenario == "no_tests":
        test_file.write_text("")

    log = work / "worker.log"
    code, _ = execute(stage, source, tests, work, env, log, 30)
    output = log.read_text()
    assert code == expected_code, output
    # The worker exit-code contract: only a genuine harness failure (RUNNER_ERROR,
    # never a signal-terminated crash) tells campaign() to abort the run.
    assert classify(code) == expected_status
    if expected_code == 1:
        assert "1 failed" in output
        assert "PYTEST_EXIT 1" in output
    elif expected_code == RUNNER_ERROR:
        assert "Traceback" in output
    elif expected_code == COLLECTION_ERROR:
        assert "ERROR collecting" in output
    elif expected_status == "crashed":
        assert classify(code) != "runner error"


@pytest.mark.parametrize("args", [["--worker"], ["--worker", "bad-stage", "sample/value.py"]])
@pytest.mark.slow
def test_malformed_worker_arguments(tmp_path: Path, args: list[str]):
    import subprocess
    import sys

    from scripts.mutation_campaign import SCRIPT

    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == RUNNER_ERROR, result.stderr
    assert classify(result.returncode) == "runner error"


def test_rates_excludes_crashed_from_both_numerators():
    # A crashed mutant is a detected fault, but a crash can also mean the
    # harness itself died. Neither rate should credit it as a kill.
    counts = Counter(killed=2, survived=1, crashed=1)
    rates = _rates(counts)
    assert rates["executed_kill_rate"] == pytest.approx(2 / 3)
    assert rates["score"] == pytest.approx(2 / 4)


def test_rates_excludes_unexecuted_statuses_from_executed_kill_rate():
    counts = Counter(killed=1, survived=1, **{"no tests": 1, "timeout": 1, "collection error": 1})
    rates = _rates(counts)
    assert rates["executed_kill_rate"] == pytest.approx(1 / 2)
    assert rates["score"] == pytest.approx(1 / 5)


def test_rates_empty_counts_is_none():
    rates = _rates(Counter())
    assert rates == {"executed_kill_rate": None, "score": None}


def test_greedy_cover_prefers_largest_unique_gain_and_lists_rest():
    from scripts.mutation_campaign import greedy_cover

    kills = {
        "test_a": {"m1", "m2"},
        "test_b": {"m2", "m3", "m4"},
        "test_c": {"m1"},
    }
    cover, candidates = greedy_cover(kills)
    assert cover == ["test_b", "test_a"]
    assert candidates == {"test_c": 1}


def test_failing_tests_parses_xunit1_file_and_name(tmp_path: Path):
    from scripts.mutation_campaign import failing_tests

    junit = tmp_path / "junit.xml"
    junit.write_text(
        '<testsuite><testcase classname="tests.test_value" file="tests/test_value.py" name="test_bad">'
        '<failure message="failed"/></testcase><testcase classname="tests.test_value" '
        'file="tests/test_value.py" name="test_ok"/></testsuite>'
    )
    assert failing_tests(junit) == {"tests/test_value.py::test_bad"}


def test_failing_tests_preserves_class_in_pytest_nodeid(tmp_path: Path):
    from scripts.mutation_campaign import failing_tests

    junit = tmp_path / "junit.xml"
    junit.write_text(
        '<testsuite><testcase classname="tests.test_value.TestContainer" '
        'file="tests/test_value.py" name="test_bad"><failure/></testcase></testsuite>'
    )
    assert failing_tests(junit) == {"tests/test_value.py::TestContainer::test_bad"}


def test_string_literal_kills_do_not_determine_cover():
    from scripts.mutation_campaign import effective_kills

    kills = {"test_literal": {"m_literal"}, "test_behavior": {"m_literal", "m_behavior"}}
    assert effective_kills(kills, {"m_literal"}) == {
        "test_literal": set(),
        "test_behavior": {"m_behavior"},
    }


def test_exit_first_flags_are_removed_from_combined_pytest_args():
    from scripts.mutation_campaign import without_exit_first

    args = ["-x", "-q", "--maxfail=1", "--maxfail", "1", "--exitfirst"]
    assert without_exit_first(args) == ["-q"]


def test_string_literal_mutants_are_flagged():
    from scripts.mutation_campaign import string_literal_mutants

    metadata = {
        "mutants": {
            "sample.value.x_value__mutmut_1": "--- a\n+++ b\n@@\n- return 'a'\n+ return 'b'\n",
            "sample.value.x_value__mutmut_2": "--- a\n+++ b\n@@\n- return 1\n+ return 2\n",
        }
    }
    assert string_literal_mutants(metadata) == {"sample.value.x_value__mutmut_1"}


@pytest.mark.integration
class TestKillMatrix:
    def test_kill_matrix_attributes_every_killed_mutant(
        self, worker_project, tmp_path: Path, monkeypatch
    ):
        import json

        from scripts import mutation_campaign

        work, _ = worker_project
        pyproject = work / "pyproject.toml"
        config = pyproject.read_text()
        pyproject.write_text(
            config.replace(
                'source_paths = ["sample"]',
                'source_paths = ["sample"]\npytest_add_cli_args = ["-x", "-q"]',
            )
        )
        (work / "tests/test_value.py").write_text(
            "from sample.value import value\n"
            "def test_value():\n    assert value() == 1\n"
            "def test_z_redundant():\n    assert value() == 1\n"
            "def test_zz_no_kill():\n    assert value() in (1, 2)\n"
        )
        monkeypatch.setattr(mutation_campaign, "ROOT", work)
        monkeypatch.setattr(
            mutation_campaign,
            "CAMPAIGNS",
            {"sample": ("sample/value.py", "tests/test_value.py", ["value"])},
        )
        output = tmp_path / "campaign"
        output.mkdir()
        mutation_campaign.campaign("sample", output, kill_matrix=True)
        report = json.loads((output / "sample.json").read_text())
        assert report["score"] == 1.0
        matrix = report["kill_matrix"]
        assert matrix["cover"] == ["tests/test_value.py::test_value"]
        assert matrix["candidates_for_removal"] == {"tests/test_value.py::test_z_redundant": 1}
        assert matrix["killed_nothing"] == ["tests/test_value.py::test_zz_no_kill"]
        baseline_output = tmp_path / "baseline"
        baseline_output.mkdir()
        mutation_campaign.campaign("sample", baseline_output)
        baseline_report = json.loads((baseline_output / "sample.json").read_text())
        assert "kill_matrix" not in baseline_report
        assert baseline_report["schema_version"] == 3

    def test_statistics_worker_returns_without_running_baseline_tests(
        self, worker_project, monkeypatch
    ):
        from contextlib import chdir

        from scripts import mutation_campaign

        work, _ = worker_project

        def unexpected_baseline(*args, **kwargs):
            raise AssertionError("stats stage must not invoke the baseline test runner")

        monkeypatch.setattr(mutation_campaign, "run_tests", unexpected_baseline)
        with chdir(work):
            assert mutation_campaign.worker("stats", "sample/value.py", []) == 0
        assert (work / "associations.json").exists()
