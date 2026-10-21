from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import check_test_weakening as check


def findings(before: str, after: str):
    return check.compare(
        check.collect({"tests/test_sample.py": before}),
        check.collect({"tests/test_sample.py": after}),
    )


def test_deleted_test_is_reported():
    result = findings("def test_kept():\n    assert True\n", "")
    assert [(item.kind, item.path) for item in result] == [
        ("test-count-decreased", "tests"),
        ("assertion-count-decreased", "tests"),
    ]


def test_removed_assertion_is_reported():
    result = findings("def test_value():\n    assert value == 3\n", "def test_value():\n    pass\n")
    assert "assertion-count-decreased" in {item.kind for item in result}


def test_test_moved_between_files_is_neutral():
    before = check.collect({"tests/test_old.py": "def test_kept():\n    assert True\n"})
    after = check.collect({"tests/test_new.py": "def test_kept():\n    assert True\n"})
    assert check.compare(before, after) == []


def test_loosened_tolerance_is_reported_and_tightened_is_not():
    before = "def test_value():\n    assert value == pytest.approx(1, rel=1e-6)\n"
    loose = findings(before, "def test_value():\n    assert value == pytest.approx(1, rel=1e-3)\n")
    tight = findings(before, "def test_value():\n    assert value == pytest.approx(1, rel=1e-9)\n")
    assert "tolerance-loosened" in {item.kind for item in loose}
    assert tight == []


def test_masked_tolerance_increase_is_reported():
    before = "import numpy as np\n\ndef test_value():\n    np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-12)\n"
    after = "import numpy as np\n\ndef test_value():\n    np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-6)\n"
    assert "tolerance-loosened" in {item.kind for item in findings(before, after)}


def test_loosening_after_inserted_stricter_assertion_is_reported():
    before = "def test_value():\n    assert x == pytest.approx(1, rel=1e-6)\n"
    after = "def test_value():\n    assert x == pytest.approx(1, rel=1e-12)\n    assert x == pytest.approx(1, rel=1e-3)\n"
    assert "tolerance-loosened" in {item.kind for item in findings(before, after)}


def test_added_loose_tolerance_keyword_is_reported():
    before = "def test_value():\n    assert x == pytest.approx(1)\n"
    after = "def test_value():\n    assert x == pytest.approx(1, abs=1e-3)\n"
    assert "tolerance-loosened" in {item.kind for item in findings(before, after)}


def test_duplicate_tolerance_calls_are_compared_as_multisets():
    source = (
        "def test_value():\n"
        "    assert x == pytest.approx(1, rel=1e-9)\n"
        "    assert x == pytest.approx(1, rel=1e-3)\n"
    )
    assert findings(source, source) == []


def test_tighter_explicit_tolerance_is_not_weakening():
    before = "def test_value():\n    assert x == pytest.approx(1)\n"
    assert findings(before, "def test_value():\n    assert x == pytest.approx(1, abs=0.0)\n") == []
    assert findings(before, "def test_value():\n    assert x == pytest.approx(1, rel=1e-9)\n") == []


def test_nonliteral_tolerance_changes_are_reported_only_when_expression_changes():
    same = "def test_value():\n    assert x == pytest.approx(1, rel=TOL)\n"
    changed = "def test_value():\n    assert x == pytest.approx(1, rel=NEW_TOL)\n"
    assert findings(same, same) == []
    assert "tolerance-loosened" in {item.kind for item in findings(same, changed)}


def test_added_skip_is_reported():
    before = "def test_value():\n    assert True\n"
    after = "@pytest.mark.skip\ndef test_value():\n    assert True\n"
    assert "skip-added" in {item.kind for item in findings(before, after)}


def test_added_xfail_mark_is_reported_as_xfail():
    before = "def test_value():\n    assert True\n"
    after = "@pytest.mark.xfail\ndef test_value():\n    assert True\n"
    assert "xfail-added" in {item.kind for item in findings(before, after)}


def test_reduced_max_examples_is_reported():
    before = "from hypothesis import settings\n\n@settings(max_examples=200)\ndef test_value():\n    assert True\n"
    after = "from hypothesis import settings\n\n@settings(max_examples=50)\ndef test_value():\n    assert True\n"
    assert "hypothesis-examples-reduced" in {item.kind for item in findings(before, after)}


def test_default_hypothesis_examples_reductions_are_reported():
    before = "from hypothesis import given\n\n@given(value=int)\ndef test_value(value):\n    assert True\n"
    reduced = (
        "from hypothesis import given, settings\n\n"
        "@settings(max_examples=1)\n@given(value=int)\ndef test_value(value):\n    assert True\n"
    )
    increased = (
        "from hypothesis import given, settings\n\n"
        "@settings(max_examples=200)\n@given(value=int)\ndef test_value(value):\n    assert True\n"
    )
    assert "hypothesis-examples-reduced" in {item.kind for item in findings(before, reduced)}
    assert findings(before, increased) == []


def test_helper_tolerance_loosenings_are_reported_at_helper_location():
    before = "def _assert_neighboring_ratio_rows(value):\n    assert value == pytest.approx(1, rel=1e-12)\n\ndef test_value():\n    pass\n"
    after = "def _assert_neighboring_ratio_rows(value):\n    assert value == pytest.approx(1, rel=1e-3)\n\ndef test_value():\n    pass\n"
    result = check.compare(
        check.collect({"tests/test_old.py": before}),
        check.collect({"tests/test_old.py": after}),
    )
    assert any(
        item.kind == "tolerance-loosened" and item.path == "tests/test_old.py" and item.line == 1
        for item in result
    )


def test_tolerance_loosened_inside_another_call_is_reported():
    before = "def test_value(ratios):\n    assert all(r == pytest.approx(2.0, abs=1e-12) for r in ratios)\n"
    after = "def test_value(ratios):\n    assert all(r == pytest.approx(2.0, abs=1e-3) for r in ratios)\n"
    assert any(item.kind == "tolerance-loosened" for item in findings(before, after))


def test_timeout_marker_without_limit_inherits_default():
    before = "@pytest.mark.timeout(300)\ndef test_value():\n    assert True\n"
    after = "@pytest.mark.timeout(method='thread')\ndef test_value():\n    assert True\n"
    assert findings(before, after) == []


def test_raised_timeout_is_reported():
    before = "@pytest.mark.timeout(30)\ndef test_value():\n    assert True\n"
    after = "@pytest.mark.timeout(300)\ndef test_value():\n    assert True\n"
    assert "timeout-raised" in {item.kind for item in findings(before, after)}


def test_assertions_in_helpers_count_toward_suite_total():
    before = "def helper():\n    assert valid\n\ndef test_value():\n    helper()\n"
    after = "def helper():\n    pass\n\ndef test_value():\n    helper()\n"
    assert "assertion-count-decreased" in {item.kind for item in findings(before, after)}


def test_imported_math_isclose_alias_tracks_tolerance_changes():
    before = "from math import isclose as close\n\ndef test_value():\n    assert close(a, b, rel_tol=1e-9)\n"
    after = "from math import isclose as close\n\ndef test_value():\n    assert close(a, b, rel_tol=1e-3)\n"
    assert "tolerance-loosened" in {item.kind for item in findings(before, after)}


def test_numpy_testing_alias_counts_assertion_removals():
    before = "import numpy.testing as npt\n\ndef test_value():\n    npt.assert_equal(a, b)\n"
    after = "import numpy.testing as npt\n\ndef test_value():\n    pass\n"
    assert "assertion-count-decreased" in {item.kind for item in findings(before, after)}


def test_direct_polars_frame_assert_uses_polars_tolerance_names():
    before = "from polars.testing import assert_frame_equal\n\ndef test_value():\n    assert_frame_equal(a, b, rel_tol=1e-5)\n"
    after = "from polars.testing import assert_frame_equal\n\ndef test_value():\n    assert_frame_equal(a, b, rel_tol=1e-7)\n"
    assert findings(before, after) == []


def test_relocated_duplicate_qualname_does_not_steal_exact_match():
    before = check.collect(
        {
            "tests/a.py": "def test_x():\n    assert True\n",
            "tests/b.py": "@pytest.mark.skip\ndef test_x():\n    assert True\n",
        }
    )
    after = check.collect(
        {
            "tests/b.py": "@pytest.mark.skip\ndef test_x():\n    assert True\n",
            "tests/c.py": "def test_x():\n    assert True\n",
        }
    )
    assert check.compare(before, after) == []


def test_nested_test_definition_is_not_collected_as_a_test():
    before = "def test_a():\n    assert True\n\ndef test_b():\n    assert True\n"
    after = "def test_b():\n    assert True\n    def test_a():\n        assert True\n"
    assert "test-count-decreased" in {item.kind for item in findings(before, after)}


def test_module_pytestmark_list_skips_each_test():
    before = "def test_value():\n    assert True\n"
    after = "pytestmark = [pytest.mark.skip]\n\ndef test_value():\n    assert True\n"
    assert "skip-added" in {item.kind for item in findings(before, after)}


def test_class_pytestmark_xfails_each_test():
    before = "class TestGroup:\n    def test_value(self):\n        assert True\n"
    after = "class TestGroup:\n    pytestmark = pytest.mark.xfail\n    def test_value(self):\n        assert True\n"
    assert "xfail-added" in {item.kind for item in findings(before, after)}


def test_module_importorskip_skips_tests():
    before = "def test_value():\n    assert True\n"
    after = "pytest.importorskip('optional_package')\n\ndef test_value():\n    assert True\n"
    assert "skip-added" in {item.kind for item in findings(before, after)}


def test_reordered_equal_sum_tolerance_tuples_are_neutral():
    before = (
        "def test_value():\n"
        "    assert x == pytest.approx(1, rel=1, abs=0)\n"
        "    assert x == pytest.approx(1, rel=0, abs=1)\n"
    )
    after = (
        "def test_value():\n"
        "    assert x == pytest.approx(1, rel=0, abs=1)\n"
        "    assert x == pytest.approx(1, rel=1, abs=0)\n"
    )
    assert findings(before, after) == []


def test_added_tolerance_call_is_neutral():
    before = "def test_value():\n    assert x == pytest.approx(1)\n"
    after = (
        "def test_value():\n    assert x == pytest.approx(1)\n    assert y == pytest.approx(2)\n"
    )
    assert findings(before, after) == []


def test_skipif_condition_change_is_reported():
    before = "@pytest.mark.skipif(False, reason='condition')\ndef test_value():\n    assert True\n"
    after = "@pytest.mark.skipif(True, reason='condition')\ndef test_value():\n    assert True\n"
    assert "skip-added" in {item.kind for item in findings(before, after)}


def test_xfail_behavior_keyword_change_is_reported():
    before = "@pytest.mark.xfail(run=False, strict=True)\ndef test_value():\n    assert True\n"
    after = "@pytest.mark.xfail(run=True, strict=True)\ndef test_value():\n    assert True\n"
    assert "xfail-added" in {item.kind for item in findings(before, after)}


def test_adding_relative_tolerance_to_abs_only_approx_is_reported():
    before = "def test_value():\n    assert x == pytest.approx(1, abs=1e-9)\n"
    after = "def test_value():\n    assert x == pytest.approx(1, abs=1e-9, rel=1e-8)\n"
    assert "tolerance-loosened" in {item.kind for item in findings(before, after)}


def test_module_settings_binding_reductions_are_reported():
    before = (
        "from hypothesis import settings\n"
        "_BASE = settings(max_examples=200)\n"
        "_SETTINGS = settings(_BASE)\n"
        "@_SETTINGS\n"
        "def test_value():\n    assert True\n"
    )
    after = (
        "from hypothesis import settings\n"
        "_BASE = settings(max_examples=50)\n"
        "_SETTINGS = settings(_BASE)\n"
        "@_SETTINGS\n"
        "def test_value():\n    assert True\n"
    )
    assert "hypothesis-examples-reduced" in {item.kind for item in findings(before, after)}


def test_effective_timeout_increases_from_disabled_and_project_default_are_reported():
    bounded = "@pytest.mark.timeout(30)\ndef test_value():\n    assert True\n"
    disabled = "@pytest.mark.timeout(0)\ndef test_value():\n    assert True\n"
    default = "def test_value():\n    assert True\n"
    extended = "@pytest.mark.timeout(600)\ndef test_value():\n    assert True\n"
    assert "timeout-raised" in {item.kind for item in findings(bounded, disabled)}
    assert "timeout-raised" in {item.kind for item in findings(bounded, default)}
    assert "timeout-raised" in {item.kind for item in findings(default, extended)}


def test_positional_numpy_isclose_tolerance_can_be_tightened():
    before = "import numpy as np\n\ndef test_value():\n    assert np.isclose(a, b, 1e-3)\n"
    after = "import numpy as np\n\ndef test_value():\n    assert np.isclose(a, b, 1e-9)\n"
    assert findings(before, after) == []


def test_decorator_assertion_calls_are_counted():
    before = (
        "@pytest.mark.parametrize('context', [pytest.raises(ValueError)])\n"
        "def test_value(context):\n    assert True\n"
    )
    after = (
        "@pytest.mark.parametrize('context', [contextlib.nullcontext()])\n"
        "def test_value(context):\n    assert True\n"
    )
    assert "assertion-count-decreased" in {item.kind for item in findings(before, after)}


def test_import_aliases_are_scoped_to_their_function():
    before = (
        "import math as check\n"
        "def test_value():\n    assert check.isclose(a, b, rel_tol=1e-9)\n"
        "def helper():\n    from pathlib import Path as check\n    return check('example')\n"
    )
    after = before.replace("rel_tol=1e-9", "rel_tol=1e-3")
    assert "tolerance-loosened" in {item.kind for item in findings(before, after)}


def test_test_in_nested_test_class_is_collected():
    before = "class TestOuter:\n    class TestInner:\n        def test_value(self):\n            assert True\n"
    after = "class TestOuter:\n    class TestInner:\n        pass\n"
    assert "test-count-decreased" in {item.kind for item in findings(before, after)}


def test_positional_approx_relative_tolerance_survives_abs_keyword():
    before = "def test_value():\n    assert value == pytest.approx(1, 1e-9, abs=1e-12)\n"
    after = "def test_value():\n    assert value == pytest.approx(1, 1e-3, abs=1e-12)\n"
    assert "tolerance-loosened" in {item.kind for item in findings(before, after)}


def test_inline_parent_hypothesis_settings_are_resolved():
    before = (
        "from hypothesis import settings\n"
        "@settings(settings(max_examples=200))\n"
        "def test_value():\n    assert True\n"
    )
    after = before.replace("max_examples=200", "max_examples=50")
    assert "hypothesis-examples-reduced" in {item.kind for item in findings(before, after)}


def test_skipif_reason_only_change_is_neutral():
    before = "@pytest.mark.skipif(True, reason='before')\ndef test_value():\n    assert True\n"
    after = "@pytest.mark.skipif(True, reason='after')\ndef test_value():\n    assert True\n"
    assert findings(before, after) == []


def test_xfail_reason_only_change_is_neutral():
    before = "@pytest.mark.xfail(reason='before')\ndef test_value():\n    assert True\n"
    after = "@pytest.mark.xfail(reason='after')\ndef test_value():\n    assert True\n"
    assert findings(before, after) == []


def test_skip_and_xfail_reason_only_changes_are_neutral():
    skip_reason_changes = (
        (
            "@pytest.mark.skip('old reason')\ndef test_value():\n    assert True\n",
            "@pytest.mark.skip('new reason')\ndef test_value():\n    assert True\n",
        ),
        (
            "@pytest.mark.skip\ndef test_value():\n    assert True\n",
            "@pytest.mark.skip(reason='explained')\ndef test_value():\n    assert True\n",
        ),
        (
            "@pytest.mark.skip('x')\ndef test_value():\n    assert True\n",
            "@pytest.mark.skip(reason='x')\ndef test_value():\n    assert True\n",
        ),
        (
            "@pytest.mark.xfail(reason='old reason')\ndef test_value():\n    assert True\n",
            "@pytest.mark.xfail(reason='new reason')\ndef test_value():\n    assert True\n",
        ),
    )
    assert all(findings(before, after) == [] for before, after in skip_reason_changes)


def test_approx_positional_absolute_only_to_relative_tolerance_is_reported():
    before = "def test_value():\n    assert value == pytest.approx(1, None, 1e-9)\n"
    after = "def test_value():\n    assert value == pytest.approx(1, 1e-8, 1e-9)\n"
    assert "tolerance-loosened" in {item.kind for item in findings(before, after)}


def test_added_looser_tolerance_preserving_base_multiset_is_neutral():
    before = "def test_value():\n    assert value == pytest.approx(1, rel=1e-6)\n"
    after = (
        "def test_value():\n"
        "    assert value == pytest.approx(1, rel=1e-6)\n"
        "    assert value == pytest.approx(1, rel=1e-3)\n"
    )
    assert findings(before, after) == []


def test_assigned_importorskip_alias_skips_module_tests():
    before = "def test_value():\n    assert True\n"
    after = (
        "from pytest import importorskip as optional_import\n"
        "optional = optional_import('optional_package')\n"
        "def test_value():\n    assert True\n"
    )
    assert "skip-added" in {item.kind for item in findings(before, after)}


def test_approx_abs_none_removing_relative_tolerance_is_reported():
    before = "def test_value():\n    assert value == pytest.approx(1, rel=1e-9, abs=None)\n"
    after = "def test_value():\n    assert value == pytest.approx(1, abs=None)\n"
    assert "tolerance-loosened" in {item.kind for item in findings(before, after)}


def test_literal_skipif_restoring_execution_is_neutral():
    before = "@pytest.mark.skipif(True, reason='disabled')\ndef test_value():\n    assert True\n"
    after = "@pytest.mark.skipif(False, reason='disabled')\ndef test_value():\n    assert True\n"
    assert findings(before, after) == []


def test_literal_xfail_restoring_execution_is_neutral():
    before = "@pytest.mark.xfail(True, reason='disabled')\ndef test_value():\n    assert True\n"
    after = "@pytest.mark.xfail(False, reason='disabled')\ndef test_value():\n    assert True\n"
    assert findings(before, after) == []


def test_first_module_timeout_marker_is_effective():
    before = (
        "pytestmark = [pytest.mark.timeout(30), pytest.mark.timeout(300)]\n"
        "def test_value():\n    assert True\n"
    )
    after = (
        "pytestmark = [pytest.mark.timeout(0), pytest.mark.timeout(300)]\n"
        "def test_value():\n    assert True\n"
    )
    assert "timeout-raised" in {item.kind for item in findings(before, after)}


def test_stacked_function_and_class_timeout_markers_follow_pytest_precedence():
    before = (
        "@pytest.mark.timeout(300)\n"
        "@pytest.mark.timeout(30)\n"
        "class TestGroup:\n"
        "    @pytest.mark.timeout(300)\n"
        "    @pytest.mark.timeout(30)\n"
        "    def test_function(self):\n        assert True\n"
        "    def test_class(self):\n        assert True\n"
    )
    after = before.replace("@pytest.mark.timeout(30)", "@pytest.mark.timeout(600)")
    assert sum(item.kind == "timeout-raised" for item in findings(before, after)) == 2


def test_keyword_skipif_condition_restoring_execution_is_neutral():
    before = "@pytest.mark.skipif(condition=True, reason='disabled')\ndef test_value():\n    assert True\n"
    after = "@pytest.mark.skipif(condition=False, reason='disabled')\ndef test_value():\n    assert True\n"
    assert findings(before, after) == []


def test_multiple_skipif_conditions_are_only_neutral_when_all_false():
    before = (
        "@pytest.mark.skipif(False, False, reason='disabled')\ndef test_value():\n    assert True\n"
    )
    after = (
        "@pytest.mark.skipif(False, True, reason='disabled')\ndef test_value():\n    assert True\n"
    )
    assert "skip-added" in {item.kind for item in findings(before, after)}


def test_tolerance_multiset_containment_keeps_comparison_subject_identity():
    before = "def test_value():\n    assert x == pytest.approx(1, rel=1e-6)\n"
    after = (
        "def test_value():\n"
        "    assert x == pytest.approx(1, rel=1e-3)\n"
        "    assert y == pytest.approx(1, rel=1e-6)\n"
    )
    assert "tolerance-loosened" in {item.kind for item in findings(before, after)}


def test_bare_skipif_and_xfail_markers_are_unconditional():
    before = "def test_value():\n    assert True\n"
    skip = "@pytest.mark.skipif\ndef test_value():\n    assert True\n"
    xfail = "@pytest.mark.xfail\ndef test_value():\n    assert True\n"
    assert "skip-added" in {item.kind for item in findings(before, skip)}
    assert "xfail-added" in {item.kind for item in findings(before, xfail)}


def test_explicit_none_approx_tolerances_match_pytest_defaults():
    explicit_none = "def test_value():\n    assert value == pytest.approx(1, rel=None, abs=None)\n"
    defaulted = "def test_value():\n    assert value == pytest.approx(1)\n"
    assert findings(explicit_none, defaulted) == []
    assert findings(defaulted, explicit_none) == []


@pytest.mark.integration
class TestGitHistory:
    def repo(self, tmp_path: Path) -> Path:
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True
        )
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
        (repo / "tests").mkdir()
        (repo / "tests/test_sample.py").write_text(
            "def test_value():\n    assert x == pytest.approx(1, rel=1e-6)\n"
        )
        self.commit(repo, "initial")
        return repo

    @staticmethod
    def commit(repo: Path, message: str) -> str:
        subprocess.run(["git", "-C", str(repo), "add", "tests"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", message], check=True)
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()

    def test_ci_mode_ignores_trailers(self, tmp_path: Path, capsys):
        repo = self.repo(tmp_path)
        (repo / "tests/test_sample.py").write_text("def test_value():\n    pass\n")
        self.commit(repo, "approved removal\n\nTest-Reduction-Approved: owner approved")
        assert check.main(["--ci", "--base", "HEAD~1", "--repo", str(repo)]) == 1
        assert "assertion-count-decreased" in capsys.readouterr().out

    def test_ci_mode_approved_flag_passes(self, tmp_path: Path, capsys):
        repo = self.repo(tmp_path)
        (repo / "tests/test_sample.py").write_text("def test_value():\n    pass\n")
        self.commit(repo, "approved removal\n\nTest-Reduction-Approved: owner approved")
        assert check.main(["--ci", "--approved", "--base", "HEAD~1", "--repo", str(repo)]) == 0
        assert "assertion-count-decreased" in capsys.readouterr().out

    def test_trailer_approves_only_its_commit(self, tmp_path: Path, capsys):
        repo = self.repo(tmp_path)
        (repo / "tests/test_sample.py").write_text("def test_value():\n    pass\n")
        self.commit(repo, "remove assertion\n\nTest-Reduction-Approved: yes")
        (repo / "tests/test_sample.py").write_text(
            "def test_value():\n    assert x == pytest.approx(1, rel=1e-3)\n"
        )
        self.commit(repo, "approved tolerance change\n\nTest-Reduction-Approved: yes")
        (repo / "tests/test_sample.py").write_text(
            "def test_value():\n    assert x == pytest.approx(1, rel=1e-1)\n"
        )
        later = self.commit(repo, "loosen tolerance")
        assert check.main(["--base", "HEAD~3", "--repo", str(repo)]) == 1
        assert later in capsys.readouterr().out

    def test_working_tree_weakening_is_never_approved(self, tmp_path: Path):
        repo = self.repo(tmp_path)
        (repo / "tests/test_sample.py").write_text("def test_value():\n    pass\n")
        assert check.main(["--base", "HEAD", "--repo", str(repo)]) == 1

    def test_clean_merge_of_approved_reduction_is_not_reported_again(self, tmp_path: Path):
        repo = self.repo(tmp_path)
        base = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        subprocess.run(["git", "-C", str(repo), "checkout", "-q", "-b", "other"], check=True)
        (repo / "tests/test_sample.py").write_text("def test_value():\n    pass\n")
        self.commit(repo, "approved removal\n\nTest-Reduction-Approved: yes")
        subprocess.run(["git", "-C", str(repo), "checkout", "-q", "main"], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "merge", "-q", "--no-ff", "other", "-m", "merge"], check=True
        )
        assert check.main(["--base", base, "--repo", str(repo)]) == 0

    def test_merge_resolution_weakening_is_reported(self, tmp_path: Path):
        repo = self.repo(tmp_path)
        base = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        subprocess.run(["git", "checkout", "-q", "-b", "other"], cwd=repo, check=True)
        (repo / "tests/test_other.py").write_text("def test_extra():\n    assert True\n")
        self.commit(repo, "side test")
        subprocess.run(["git", "checkout", "-q", "main"], cwd=repo, check=True)
        (repo / "tests/test_sample.py").write_text(
            "def test_value():\n    assert x == pytest.approx(1, rel=1e-3)\n"
        )
        self.commit(repo, "main tolerance change")
        subprocess.run(
            ["git", "merge", "-q", "--no-ff", "other", "-m", "merge"], cwd=repo, check=True
        )
        (repo / "tests/test_sample.py").write_text("def test_value():\n    pass\n")
        self.commit(repo, "merge resolution")
        assert check.main(["--base", base, "--repo", str(repo)]) == 1

    def test_merge_commit_reduction_intersects_different_parent_totals(
        self, tmp_path: Path, capsys
    ):
        repo = self.repo(tmp_path)
        base = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        subprocess.run(["git", "checkout", "-q", "-b", "other"], cwd=repo, check=True)
        (repo / "tests/test_other.py").write_text(
            "def test_b():\n    assert True\n"
            "def test_c():\n    assert True\n"
            "def test_d():\n    assert True\n"
        )
        self.commit(repo, "add three tests")
        subprocess.run(["git", "checkout", "-q", "main"], cwd=repo, check=True)
        (repo / "tests/test_main.py").write_text(
            "def test_b():\n    assert True\ndef test_c():\n    assert True\n"
        )
        self.commit(repo, "add two tests")
        subprocess.run(["git", "merge", "--no-ff", "--no-commit", "other"], cwd=repo, check=True)
        (repo / "tests/test_other.py").unlink()
        (repo / "tests/test_main.py").unlink()
        (repo / "tests/test_sample.py").write_text("def test_value():\n    assert True\n")
        self.commit(repo, "resolve merge with one test")
        merge = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        assert check.main(["--base", base, "--repo", str(repo)]) == 1
        output = capsys.readouterr().out.splitlines()
        finding_lines = [line for line in output if line.startswith(f"{merge} ")]
        assert finding_lines
        assert any("test-count-decreased" in line for line in finding_lines)
        assert any("assertion-count-decreased" in line for line in finding_lines)

    def test_merge_finding_intersects_timeout_and_hypothesis_changes(self, tmp_path: Path, capsys):
        repo = self.repo(tmp_path)
        sample = repo / "tests/test_sample.py"
        sample.write_text(
            "@pytest.mark.timeout(10)\n"
            "def test_value():\n    assert x == pytest.approx(1, rel=1e-6)\n"
        )
        hypothesis = repo / "tests/test_hypothesis.py"
        hypothesis.write_text(
            "from hypothesis import settings\n"
            "@settings(max_examples=300)\n"
            "def test_generated():\n    assert True\n"
        )
        self.commit(repo, "configure effective limits")
        base = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        subprocess.run(["git", "checkout", "-q", "-b", "other"], cwd=repo, check=True)
        hypothesis.write_text(
            "from hypothesis import settings\n"
            "@settings(max_examples=200)\n"
            "def test_generated():\n    assert True\n"
        )
        self.commit(repo, "lower hypothesis examples")
        subprocess.run(["git", "checkout", "-q", "main"], cwd=repo, check=True)
        sample.write_text(
            "@pytest.mark.timeout(20)\n"
            "def test_value():\n    assert x == pytest.approx(1, rel=1e-6)\n"
        )
        self.commit(repo, "raise timeout")
        subprocess.run(
            ["git", "merge", "-q", "--no-ff", "--no-commit", "other"],
            cwd=repo,
            check=True,
        )
        sample.write_text(
            "@pytest.mark.timeout(600)\n"
            "def test_value():\n    assert x == pytest.approx(1, rel=1e-6)\n"
        )
        hypothesis.write_text(
            "from hypothesis import settings\n"
            "@settings(max_examples=100)\n"
            "def test_generated():\n    assert True\n"
        )
        self.commit(repo, "resolve effective limits")
        merge = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        assert check.main(["--base", base, "--repo", str(repo)]) == 1
        findings = [
            line for line in capsys.readouterr().out.splitlines() if line.startswith(f"{merge} ")
        ]
        assert any("timeout-raised" in line for line in findings)
        assert any("hypothesis-examples-reduced" in line for line in findings)

    def test_unknown_base_exits_2(self, tmp_path: Path):
        repo = self.repo(tmp_path)
        assert check.main(["--base", "not-a-revision", "--repo", str(repo)]) == 2


def test_installed_tolerance_defaults_match_the_comparison_contract():
    import inspect
    import math

    import numpy as np
    import pytest as pytest_module
    from numpy.testing import assert_allclose

    approx_type = type(pytest_module.approx(1))
    relative_name = "DEFAULT_RELATIVE_TOLERANCE"
    absolute_name = "DEFAULT_ABSOLUTE_TOLERANCE"
    assert getattr(approx_type, relative_name) == 1e-6
    assert getattr(approx_type, absolute_name) == 1e-12
    assert inspect.signature(math.isclose).parameters["rel_tol"].default == 1e-9
    assert inspect.signature(math.isclose).parameters["abs_tol"].default == 0.0
    assert inspect.signature(np.isclose).parameters["rtol"].default == 1e-5
    assert inspect.signature(np.isclose).parameters["atol"].default == 1e-8
    assert inspect.signature(np.allclose).parameters["rtol"].default == 1e-5
    assert inspect.signature(np.allclose).parameters["atol"].default == 1e-8
    assert inspect.signature(assert_allclose).parameters["rtol"].default == 1e-7
    assert inspect.signature(assert_allclose).parameters["atol"].default == 0
