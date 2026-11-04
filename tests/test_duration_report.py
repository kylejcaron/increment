import json
from pathlib import Path

import pytest

from scripts.duration_report import (
    SlowItem,
    layer_over_budget,
    main,
    over_budget,
    over_budget_count_errors,
)

pytestmark = pytest.mark.tooling


def test_lists_items_over_budget_slowest_first(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    junit = tmp_path / "junit.xml"
    junit.write_text(
        '<testsuites><testsuite><testcase classname="tests.test_example" '
        'file="tests/test_example.py" name="test_medium" time="6"/>'
        '<testcase classname="tests.test_example.TestBoundary" '
        'file="tests/test_example.py" name="test_slow[value]" time="12"/>'
        '<testcase classname="tests.test_example" '
        'name="test_fallback" time="8"/></testsuite></testsuites>'
    )

    assert over_budget(junit) == [
        SlowItem("tests/test_example.py::TestBoundary::test_slow[value]", 12.0),
        SlowItem("tests/test_example.py::test_fallback", 8.0),
        SlowItem("tests/test_example.py::test_medium", 6.0),
    ]
    assert main([str(junit)]) == 0
    summary = capsys.readouterr().out
    assert "tests/test_example.py::TestBoundary::test_slow[value]" in summary
    assert summary.index("test_slow[value]") < summary.index("test_fallback")


def test_reports_nothing_when_all_items_fit(tmp_path: Path):
    junit = tmp_path / "junit.xml"
    junit.write_text(
        '<testsuite><testcase classname="tests.test_example" name="test_at_budget" '
        'time="2"/><testcase classname="tests.test_example" name="test_below" '
        'time="1.9"/></testsuite>'
    )

    assert over_budget(junit, budget_s=2.0) == []


def test_skipped_items_are_ignored(tmp_path: Path):
    junit = tmp_path / "junit.xml"
    junit.write_text(
        '<testsuite><testcase classname="tests.test_example" name="test_skipped" '
        'time="20"><skipped/></testcase></testsuite>'
    )

    assert over_budget(junit) == []


def test_reports_tests_over_their_layer_budget(tmp_path: Path):
    junit = tmp_path / "junit.xml"
    junit.write_text(
        "<testsuite>"
        '<testcase classname="tests.test_unit" file="tests/test_unit.py" '
        'name="test_expensive" time="0.11"/>'
        '<testcase classname="tests.test_integration" file="tests/test_integration.py" '
        'name="test_expensive" time="1.1"/>'
        '<testcase classname="tests.test_slow" file="tests/test_slow.py" '
        'name="test_expensive" time="30.1"/>'
        "</testsuite>"
    )
    collection = tmp_path / "collection.json"
    collection.write_text(
        '{"tier_nodeids":{"unit":["tests/test_unit.py::test_expensive"],'
        '"integration":["tests/test_integration.py::test_expensive"],'
        '"slow":["tests/test_slow.py::test_expensive"]}}'
    )

    result = layer_over_budget(junit, collection)

    assert [(item.layer, item.nodeid) for item in result] == [
        ("slow", "tests/test_slow.py::test_expensive"),
        ("integration", "tests/test_integration.py::test_expensive"),
        ("unit", "tests/test_unit.py::test_expensive"),
    ]


def test_over_budget_count_may_fall_not_grow(tmp_path: Path):
    junit = tmp_path / "junit.xml"
    junit.write_text(
        '<testsuite><testcase classname="tests.test_unit" file="tests/test_unit.py" '
        'name="test_over" time="0.11"/></testsuite>'
    )
    collection = tmp_path / "collection.json"
    collection.write_text('{"tier_nodeids":{"unit":["tests/test_unit.py::test_over"]}}')
    baseline = tmp_path / "baseline.json"
    baseline.write_text('{"budgets_s":{"unit":0.1},"max_over_budget":{"unit":2}}')

    assert over_budget_count_errors(junit, collection, baseline) == []
    baseline.write_text('{"budgets_s":{"unit":0.1},"max_over_budget":{"unit":0}}')
    assert over_budget_count_errors(junit, collection, baseline) == ["unit"]


def test_aggregate_deduplicates_nodes_and_obeys_baseline_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    artifacts = tmp_path / "artifacts"
    nodeid = "tests/test_shared.py::test_expensive"
    for shard in ("python-3.12", "python-3.14"):
        report_dir = artifacts / shard
        report_dir.mkdir(parents=True)
        (report_dir / "junit.xml").write_text(
            '<testsuite><testcase classname="tests.test_shared" file="tests/test_shared.py" '
            'name="test_expensive" time="1.1"/></testsuite>'
        )
        (report_dir / "junit-collection.json").write_text(
            f'{{"shard_id":"{shard}","tier_nodeids":{{"integration":["{nodeid}"]}}}}'
        )
    baseline = tmp_path / "baseline.json"
    baseline.write_text('{"mode":"report","max_over_budget":{"unit":0,"integration":0,"slow":0}}')
    arguments = [
        "--artifacts-dir",
        str(artifacts),
        "--baseline",
        str(baseline),
        "--expected-shards",
        "python-3.12,python-3.14",
    ]

    assert main(arguments) == 0
    report_output = capsys.readouterr().out
    assert "- integration: 1 item(s) over 1 s" in report_output
    assert "::warning::" in report_output

    baseline.write_text('{"mode":"enforce","max_over_budget":{"unit":0,"integration":0,"slow":0}}')
    assert main(arguments) == 1
    enforce_output = capsys.readouterr().out
    assert "- integration: 1 item(s) over 1 s" in enforce_output
    assert "::error::" in enforce_output


@pytest.mark.parametrize("scenario", ["empty", "unpaired", "missing-shard"])
@pytest.mark.parametrize("mode", ["report", "enforce"])
def test_incomplete_artifacts_warn_or_fail_by_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], scenario: str, mode: str
):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    if scenario == "unpaired":
        shard = artifacts / "fast-1"
        shard.mkdir()
        (shard / "junit.xml").write_text("<testsuite/>")
    if scenario == "missing-shard":
        shard = artifacts / "fast-1"
        shard.mkdir()
        (shard / "junit.xml").write_text("<testsuite/>")
        (shard / "junit-collection.json").write_text(
            '{"shard_id":"fast-1","tier_nodeids":{"unit":[]}}'
        )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"mode": mode, "max_over_budget": {"unit": 0, "integration": 0, "slow": 0}})
    )
    result = main(
        [
            "--artifacts-dir",
            str(artifacts),
            "--baseline",
            str(baseline),
            "--expected-shards",
            "fast-1,slow-1",
        ]
    )

    output = capsys.readouterr().out
    assert ("::warning::Incomplete duration evidence" in output) == (mode == "report")
    assert ("::error::Incomplete duration evidence" in output) == (mode == "enforce")
    assert result == (0 if mode == "report" else 1)
