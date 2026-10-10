from __future__ import annotations

from pathlib import Path

import pytest

from tests.doc_evidence import (
    EvidenceReference,
    check_collected,
    check_static,
    collect_nodeids,
    extract_references,
    selector_files,
)


def test_extracts_typed_references_and_normalizes_campaign_kinds() -> None:
    refs = extract_references(
        "`selector:tests/test_sample.py::TestSample::test_value` "
        "`helper:tests/support.py` `campaign-module:calibration.run --dry-run` "
        '`campaign-script:"calibration/run R.R" --seed 1` `evidence-id:EV-42`',
        source="guide.md",
    )
    assert [(ref.kind, ref.raw) for ref in refs] == [
        ("selector", "tests/test_sample.py::TestSample::test_value"),
        ("helper", "tests/support.py"),
        ("campaign", "campaign-module:calibration.run --dry-run"),
        ("campaign", 'campaign-script:"calibration/run R.R" --seed 1'),
        ("evidence_id", "EV-42"),
    ]
    assert all(ref.source == "guide.md:1" for ref in refs)


def test_selector_shorthand_is_scoped_to_each_evidence_cell() -> None:
    refs = extract_references(
        "| Evidence | Test selector |\n|---|---|\n"
        "| EV-1 | `tests/test_one.py::test_a`; `::test_b` |\n"
        "| EV-2 | `tests/test_two.py::test_c`; `::test_d` |",
        source="validation.md",
    )
    assert [ref.raw for ref in refs] == [
        "tests/test_one.py::test_a",
        "tests/test_one.py::test_b",
        "tests/test_two.py::test_c",
        "tests/test_two.py::test_d",
    ]
    journey_refs = extract_references(
        "| Journey | Existing evidence | Limit |\n|---|---|---|\n"
        "| A | `tests/estimation/test_one.py::test_a` | measured evidence in `left|right` |\n"
        "| B | `tests/estimation/test_two.py::test_c`; `::test_d` | Scope |",
        source="validation.md",
    )
    assert [ref.raw for ref in journey_refs] == [
        "tests/estimation/test_one.py::test_a",
        "tests/estimation/test_two.py::test_c",
        "tests/estimation/test_two.py::test_d",
    ]


def test_ambiguous_untyped_helper_fails_with_location() -> None:
    with pytest.raises(ValueError, match=r"guide.md:1:.*helper:"):
        extract_references("`tests/support.py`", source="guide.md")


def test_untyped_artifact_filename_requires_helper_reference(tmp_path: Path) -> None:
    fixture = tmp_path / "fixtures" / "sample.json"
    fixture.parent.mkdir()
    fixture.write_text("{}")
    with pytest.raises(ValueError, match=r"ambiguous untyped reference"):
        extract_references("`sample.json`", source="guide.md")
    refs = extract_references("`helper:fixtures/sample.json`", source="guide.md")
    assert check_static(tmp_path, refs).errors == ()


def test_inline_arithmetic_is_not_an_untyped_path() -> None:
    refs = extract_references("The ratio is `5/4`.", source="guide.md")
    assert refs == ()


def test_untyped_selector_and_wrong_class_resolution(tmp_path: Path) -> None:
    sample = tmp_path / "tests" / "test_sample.py"
    sample.parent.mkdir()
    sample.write_text("class C:\n    def test_ok(self):\n        pass\n")
    refs = extract_references("`tests/test_sample.py::C::test_ok`", source="guide.md")
    assert check_static(tmp_path, refs).errors == ()
    wrong = extract_references("`tests/test_sample.py::Wrong::test_ok`", source="guide.md")
    wrong_errors = check_static(tmp_path, wrong).errors
    assert wrong_errors and all(error.startswith("guide.md:1:") for error in wrong_errors)
    renamed = extract_references("`tests/test_sample.py::test_renamed`", source="guide.md")
    renamed_errors = check_static(tmp_path, renamed).errors
    assert renamed_errors and all("test_renamed" in error for error in renamed_errors)


def test_static_checks_evidence_id_campaign_module_script_and_helpers(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_x.py").write_text("def test_x():\n    pass\n")
    (tmp_path / "calibration").mkdir()
    (tmp_path / "calibration/__init__.py").write_text("")
    (tmp_path / "calibration/run.py").write_text("")
    (tmp_path / "run R.R").write_text("")
    refs = (
        EvidenceReference("selector", "tests/test_x.py::test_x", "doc.md"),
        EvidenceReference("helper", "tests/test_x.py", "doc.md"),
        EvidenceReference("campaign", "campaign-module:calibration.run --seed 4", "doc.md"),
        EvidenceReference("campaign", 'campaign-script:"run R.R" --seed 4', "doc.md"),
        EvidenceReference("evidence_id", "EV-MISSING", "doc.md"),
    )
    report = check_static(tmp_path, refs, evidence_ids={"EV-OK"})
    assert any("EV-MISSING" in error for error in report.errors)


def test_pending_parameterized_selector_requires_exact_collected_id(tmp_path: Path) -> None:
    sample = tmp_path / "tests" / "test_sample.py"
    sample.parent.mkdir()
    sample.write_text(
        "class Base:\n    def test_case(self):\n        pass\nclass Child(Base):\n    pass\n"
    )
    refs = extract_references("`tests/test_sample.py::Child::test_case[param]`", source="doc.md")
    report = check_static(tmp_path, refs)
    assert report.pending_nodeids == ("tests/test_sample.py::Child::test_case[param]",)
    assert check_collected(report, {"tests/test_sample.py::Child::test_case[param]"}) == ()
    assert check_collected(report, {"tests/test_sample.py::Base::test_case[param]"})


def test_existing_but_uncollected_selector_fails_collection_check(tmp_path: Path) -> None:
    sample = tmp_path / "tests" / "test_sample.py"
    sample.parent.mkdir()
    sample.write_text("def test_case():\n    pass\n")
    refs = extract_references("`tests/test_sample.py::test_case`", source="doc.md")

    report = check_static(tmp_path, refs)

    assert report.errors == ()
    assert check_collected(report, set()) == (
        "selector did not collect exact node id: tests/test_sample.py::test_case",
    )


def test_file_and_function_selectors_require_collected_descendants(tmp_path: Path) -> None:
    sample = tmp_path / "tests" / "test_sample.py"
    sample.parent.mkdir()
    sample.write_text("def test_case():\n    pass\n")
    refs = extract_references(
        "`tests/test_sample.py` `tests/test_sample.py::test_case`",
        source="doc.md",
    )
    report = check_static(tmp_path, refs)

    assert check_collected(report, {"tests/test_sample.py::test_case"}) == ()
    assert check_collected(report, set()) == (
        "file selector collected no descendant tests: tests/test_sample.py",
        "selector did not collect exact node id: tests/test_sample.py::test_case",
    )


def test_bare_parameterized_function_accepts_descendant_but_explicit_id_is_exact(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "tests" / "test_sample.py"
    sample.parent.mkdir()
    sample.write_text("def test_case():\n    pass\n")
    refs = extract_references(
        "`tests/test_sample.py::test_case` `tests/test_sample.py::test_case[chosen]`",
        source="doc.md",
    )
    report = check_static(tmp_path, refs)

    assert check_collected(report, {"tests/test_sample.py::test_case[chosen]"}) == ()
    assert check_collected(report, {"tests/test_sample.py::test_case[other]"}) == (
        "selector did not collect exact node id: tests/test_sample.py::test_case[chosen]",
    )


def test_class_selector_requires_collected_descendant_tests(tmp_path: Path) -> None:
    sample = tmp_path / "tests" / "test_sample.py"
    sample.parent.mkdir()
    sample.write_text(
        "class TestWithTests:\n    def test_case(self):\n        pass\nclass TestEmpty:\n    pass\n"
    )
    refs = extract_references(
        "`tests/test_sample.py::TestWithTests` `tests/test_sample.py::TestEmpty`",
        source="doc.md",
    )

    report = check_static(tmp_path, refs)

    assert report.errors == ()
    assert report.pending_nodeids == (
        "tests/test_sample.py::TestEmpty",
        "tests/test_sample.py::TestWithTests",
    )
    assert report.class_selectors == report.pending_nodeids
    errors = check_collected(report, {"tests/test_sample.py::TestWithTests::test_case"})
    assert any("tests/test_sample.py::TestEmpty" in error for error in errors)
    assert all("TestWithTests" not in error for error in errors)


def test_collection_loads_configured_plugins_without_executing_cited_or_other_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\nfilterwarnings = ["error"]\n'
    )
    (tests / "test_selected.py").write_text(
        "import pytest\nfrom pathlib import Path\n"
        "@pytest.mark.timeout(1)\n"
        "def test_selected():\n"
        "    Path('EXECUTED').write_text('unexpected execution')\n"
        "    raise RuntimeError('collection must not execute tests')\n"
    )
    for name in ("test_helper.py", "test_campaign.py"):
        (tests / name).write_text("raise RuntimeError('uncited file was imported')\n")
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    refs = (
        EvidenceReference("selector", "tests/test_selected.py::test_selected", "doc.md"),
        EvidenceReference("helper", "tests/test_helper.py", "doc.md"),
        EvidenceReference("campaign", "campaign-module:tests.test_campaign", "doc.md"),
    )
    nodeids = collect_nodeids(tmp_path, selector_files(refs))
    assert nodeids == frozenset({"tests/test_selected.py::test_selected"})
    assert not (tmp_path / "EXECUTED").exists()


@pytest.mark.parametrize("through_local_base", [False, True])
def test_module_qualified_inherited_selector_is_resolved_by_collection(
    tmp_path: Path, through_local_base: bool
) -> None:
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "conftest.py").write_text(
        "raise RuntimeError('an ancestor conftest is outside the cited repository')\n"
    )
    root = tmp_path / "cited-repository"
    tests = root / "tests"
    tests.mkdir(parents=True)
    (tests / "__init__.py").write_text("")
    (tests / "support.py").write_text(
        "class Base:\n"
        "    def test_case(self):\n"
        "        raise RuntimeError('collection must not execute inherited tests')\n"
    )
    inheritance = (
        "class Intermediate(support.Base):\n    pass\nclass TestChild(Intermediate):\n    pass\n"
        if through_local_base
        else "class TestChild(support.Base):\n    pass\n"
    )
    (tests / "test_sample.py").write_text("from tests import support\n" + inheritance)
    selector = "tests/test_sample.py::TestChild::test_case"
    refs = extract_references(f"`{selector}`", source="doc.md")
    report = check_static(root, refs)
    assert report.errors == ()
    assert check_collected(report, collect_nodeids(root, selector_files(refs))) == ()


def test_campaign_arguments_are_not_executed(tmp_path: Path) -> None:
    refs = extract_references(
        '`campaign-script:"tests/test_x.py" $(touch SHOULD_NOT_EXIST)`', source="doc.md"
    )
    report = check_static(tmp_path, refs)
    assert report.errors
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()


@pytest.mark.parametrize(
    ("document_name", "target", "expected_exit"),
    [
        ("llms.txt", "guides/quickstart.md", 1),
        (
            "llms.txt",
            "https://raw.githubusercontent.com/kylejcaron/increment/main/docs/guides/quickstart.md",
            0,
        ),
        ("limitations.md", "guides/quickstart.md", 0),
    ],
)
def test_curated_links_distinguish_copied_llms_index_from_markdown_pages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    document_name: str,
    target: str,
    expected_exit: int,
) -> None:
    from scripts import check_doc_references

    guide = tmp_path / "docs/guides/quickstart.md"
    guide.parent.mkdir(parents=True)
    guide.write_text("# Quickstart\n")
    (tmp_path / "docs" / document_name).write_text(f"[Quickstart]({target})\n")
    monkeypatch.setattr(
        check_doc_references, "__file__", str(tmp_path / "scripts/check_doc_references.py")
    )

    assert check_doc_references.main(["--check"]) == expected_exit
    if expected_exit:
        assert target in capsys.readouterr().err
