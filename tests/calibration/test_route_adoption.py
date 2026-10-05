"""Audited adoption of saved bound enumerations (``calibration.route_adoption``).

The audit compares a decision path's code at two revisions; the tests run it on small source
trees, where the expected verdict is known. Adoption is run on real small designs: the records a
saved checkpoint would hold are produced by the current enumeration, so a record the current
runtime reproduces is adopted and any record it does not is refused.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import pytest

from calibration import conversion_route as cr
from calibration import route_adoption as ra

# --- the audit ---------------------------------------------------------------------------


def _sources(files: dict[str, str]) -> ra.Sources:
    return files.get


def _path(entries, files_then, files_now):
    return ra.audit(_sources(files_then), _sources(files_now), entries)


MODULE = "pkg/core.py"
ENTRY = [("pkg.core", "decide")]


def _core(decide_body: str = "return helper(x) > LIMIT", helper: str = "x * 2", limit: str = "3"):
    return {
        MODULE: f"LIMIT = {limit}\n\n\ndef helper(x):\n    return {helper}\n\n\n"
        f"def decide(x):\n    {decide_body}\n\n\ndef unrelated(x):\n    return x - 1\n"
    }


class TestAuditComparesWhatAPathReaches:
    def test_identical_code_has_no_difference(self):
        assert _path(ENTRY, _core(), _core()).differences == ()

    @pytest.mark.parametrize(
        "changed",
        [
            {"decide_body": "return helper(x) >= LIMIT"},
            {"helper": "x * 3"},
            {"limit": "4"},
        ],
        ids=["the entry", "what it calls", "a constant it reads"],
    )
    def test_a_change_in_anything_the_path_reaches_is_a_difference(self, changed):
        compared = _path(ENTRY, _core(), _core(**changed))
        assert len(compared.differences) == 1

    def test_a_change_the_path_does_not_reach_is_not_a_difference(self):
        now = _core()
        now[MODULE] = now[MODULE].replace("return x - 1", "return x - 2")
        assert _path(ENTRY, _core(), now).differences == ()

    def test_comments_docstrings_and_annotations_are_not_differences(self):
        then = {MODULE: "def decide(x):\n    return x > 3\n"}
        now = {
            MODULE: 'def decide(x: float) -> bool:\n    """Whether x is large."""\n'
            "    # compared against three\n    return x > 3\n"
        }
        assert _path(ENTRY, then, now).differences == ()

    def test_a_default_is_part_of_the_code(self):
        then = {MODULE: "def decide(x, limit=3):\n    return x > limit\n"}
        now = {MODULE: "def decide(x, limit=4):\n    return x > limit\n"}
        assert len(_path(ENTRY, then, now).differences) == 1

    def test_a_definition_added_or_removed_on_the_path_is_a_difference(self):
        then = {MODULE: "def decide(x):\n    return x > 3\n"}
        now = {MODULE: "def decide(x):\n    return later(x)\n\n\ndef later(x):\n    return x > 3\n"}
        compared = _path(ENTRY, then, now)
        assert {
            d.definition: (d.source is None, d.destination is None) for d in compared.differences
        } == {
            ("pkg.core", "decide"): (False, False),
            ("pkg.core", "later"): (True, False),
        }
        removed = _path(ENTRY, now, then)
        assert {
            d.definition: (d.source is None, d.destination is None) for d in removed.differences
        } == {
            ("pkg.core", "decide"): (False, False),
            ("pkg.core", "later"): (False, True),
        }

    def test_imports_are_followed_across_modules_by_name_alias_and_locally(self):
        def tree(limit: str) -> dict[str, str]:
            return {
                "pkg/__init__.py": "",
                "pkg/core.py": (
                    "from pkg import rules\n"
                    "from pkg.bounds import LIMIT as TOP\n\n\n"
                    "def decide(x):\n"
                    "    from pkg.extra import adjust\n"
                    "    return adjust(rules.score(x)) > TOP\n"
                ),
                "pkg/rules.py": "def score(x):\n    return x\n",
                "pkg/bounds.py": f"LIMIT = {limit}\n",
                "pkg/extra.py": "def adjust(x):\n    return x\n",
            }

        assert _path(ENTRY, tree("3"), tree("3")).differences == ()
        changed = _path(ENTRY, tree("3"), tree("4"))
        assert {d.definition for d in changed.differences} == {("pkg.bounds", "LIMIT")}
        moved = tree("3")
        moved["pkg/extra.py"] = "def adjust(x):\n    return x + 1\n"
        assert {d.definition for d in _path(ENTRY, tree("3"), moved).differences} == {
            ("pkg.extra", "adjust")
        }

    def test_a_method_is_reached_once_its_class_is(self):
        def tree(body: str, other: str) -> dict[str, str]:
            return {
                "pkg/core.py": (
                    "class Rule:\n"
                    "    def __init__(self, limit):\n        self.limit = limit\n\n"
                    f"    def test(self, x):\n        {body}\n\n"
                    f"    def other(self, x):\n        {other}\n\n\n"
                    "def decide(x):\n    return Rule(3).test(x)\n"
                )
            }

        then = tree("return x > self.limit", "return 0")
        assert _path(ENTRY, then, tree("return x > self.limit", "return 1")).differences == ()
        changed = _path(ENTRY, then, tree("return x >= self.limit", "return 0"))
        assert {d.definition for d in changed.differences} == {("pkg.core", "Rule.test")}

    def test_a_method_of_a_class_nothing_names_is_not_reached(self):
        def tree(body: str) -> dict[str, str]:
            return {
                "pkg/core.py": f"class Rule:\n    def test(self, x):\n        return {body}\n\n\n"
                "def decide(rule, x):\n    return rule.test(x)\n"
            }

        assert _path(ENTRY, tree("x > 3"), tree("x > 4")).differences == ()

    def test_a_review_covers_exactly_the_digests_it_was_read_at(self):
        compared = _path(ENTRY, _core(helper="x * 2"), _core(helper="x * 3"))
        (difference,) = compared.differences
        reviewed = {
            difference.definition: ra.Review(
                frozenset({difference.source, "another"}), difference.destination, "why"
            )
        }
        assert compared.unreviewed(reviewed) == ()
        further = _path(ENTRY, _core(helper="x * 2"), _core(helper="x * 4"))
        assert further.unreviewed(reviewed) == further.differences

    def test_a_missing_entry_is_an_error_not_an_empty_path(self):
        with pytest.raises(LookupError):
            ra.closure(ra.Tree(_sources(_core())), [("pkg.core", "absent")])


class TestStatements:
    def test_the_expressions_a_target_is_assigned_are_keyed_by_target(self):
        tree = ra.Tree(
            _sources(
                {
                    MODULE: "def f(n):\n    total = 0.0\n    for a, b in pairs(n):\n"
                    "        w = a * b\n        total += float(w)\n    return total\n"
                }
            )
        )
        found = ra.statements(tree, "pkg.core", ["f"])
        assert set(found) == {"total", "for a, b", "w", "total +="}

    def test_the_same_statement_in_two_functions_is_one_digest(self):
        tree = ra.Tree(
            _sources(
                {
                    MODULE: "def f(n):\n    w = n * 2\n\n\ndef g(n):\n    w = n * 2\n    v = 1\n",
                }
            )
        )
        merged = ra.statements(tree, "pkg.core", ["f", "g"])
        assert len(merged["w"]) == 1
        assert ra.statements(tree, "pkg.core", ["f"])["w"] == merged["w"]


# --- adoption ----------------------------------------------------------------------------

DESIGNS = (
    cr.MirrorCell("bound", 300, 0.3, 0.1, 0.2, "two-sided"),
    cr.MirrorCell("bound", 240, 0.05, 0.4, 0.1, "greater"),
)


@pytest.fixture
def runtime(monkeypatch):
    """The routing floor the designs straddle, and an audit that finds nothing to review."""
    monkeypatch.setattr(cr, "dense_min_count", lambda tail: 14)
    monkeypatch.setattr(ra, "destination_fingerprint", lambda: "tree-digest")

    def passing(repo, revision):
        compared = ra.Audit({}, {})
        return ra.RevisionAudit(f"full-{revision}", compared, (), (), True)

    monkeypatch.setattr(ra, "audit_revision", passing)


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """A checkpoint in the layout earlier runs saved, from the current enumeration."""
    path = tmp_path_factory.mktemp("saved") / "saved.jsonl"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cr, "dense_min_count", lambda tail: 14)
        enumerations = [cr.enumerate_design(cell) for cell in DESIGNS]
    lines = []
    for cell, e in zip(DESIGNS, enumerations, strict=True):
        power = {
            "route": "borderline",
            "planned": 0.3,
            "basis": "exact",
            "asymptotic_share": e.asymptotic_share,
            "asymptotic": e.asymptotic,
            "asymptotic_part": e.asymptotic_part,
            "finite_part": e.finite_part,
            "omitted": e.omitted / (1.0 + 1e-9),
        }
        lines.append(json.dumps({"design": cr._design_key(cell), "power": power}))
    path.write_text("\n".join(lines) + "\n")
    return path, enumerations


def _adopt(source, out, **overrides):
    from increment.estimation.results import BINOMIAL_METHOD

    arguments: dict[str, Any] = {
        "revisions": ["abc"],
        "construction": BINOMIAL_METHOD,
        "law": ra.route_law(),
        "evidence": "launched after the commit",
    }
    return ra.adopt(source, out, **(arguments | overrides))


class TestAdopt:
    def test_saved_records_are_adopted_under_a_manifest_and_read_back(
        self, runtime, saved, tmp_path
    ):
        source, fresh = saved
        out = tmp_path / "adopted.jsonl"
        manifest = _adopt(source, out)
        assert manifest["source"]["records"] == 2
        assert manifest["campaign"]["evidence"] == "launched after the commit"
        adopted = cr.read_checkpoint(out)
        for cell, expected in zip(DESIGNS, fresh, strict=True):
            power = adopted[json.dumps(cr._design_key(cell))]
            assert power.enumeration == expected
            assert power.plan.model == "borderline_minimum"
            assert not power.plan.certified
            assert power.adopted == ra.manifest_digest(manifest)

    def test_the_manifest_records_what_was_audited_and_how_each_record_was_fingerprinted(
        self, runtime, saved, tmp_path
    ):
        source, _ = saved
        manifest = _adopt(source, tmp_path / "adopted.jsonl")
        assert manifest["source"]["sha256"]
        assert manifest["destination"]["closure"] == "tree-digest"
        assert manifest["audit"][0]["arithmetic"] == list(ra.ARITHMETIC_KEYS)
        assert set(manifest["records"]) == {json.dumps(cr._design_key(c)) for c in DESIGNS}
        assert all(len(sha) == 64 for sha in manifest["records"].values())

    def test_a_resumed_run_keeps_the_enumeration_plans_again_and_stays_adopted(
        self, runtime, saved, tmp_path, monkeypatch
    ):
        source, fresh = saved
        out = tmp_path / "adopted.jsonl"
        manifest = _adopt(source, out)
        planned = []

        def plan(cell):
            from increment.power.core import BINOMIAL_PLANNING_MODEL

            planned.append(cell)
            return cr.Plan(
                BINOMIAL_PLANNING_MODEL, "dense", "exact", 0.3, 0.3, 0.3, 0.0, False, True
            )

        def refuse(cell):
            raise AssertionError("an adopted enumeration must not be summed again")

        monkeypatch.setattr(cr, "bound_cells", lambda grid: DESIGNS)
        monkeypatch.setattr(cr, "plan_design", plan)
        monkeypatch.setattr(cr, "enumerate_design", refuse)
        cr.bound(workers=1, out=out)
        assert planned == list(DESIGNS)
        resumed = cr.read_checkpoint(out)
        for cell, expected in zip(DESIGNS, fresh, strict=True):
            power = resumed[json.dumps(cr._design_key(cell))]
            assert power.enumeration == expected
            assert power.plan.model != "borderline_minimum"
            assert power.adopted == ra.manifest_digest(manifest)
        planned.clear()
        cr.bound(workers=1, out=out)
        assert planned == []

    def test_a_design_indexed_by_position_is_mapped_through_the_grid_it_names(
        self, runtime, saved, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(cr, "bound_cells", lambda grid: DESIGNS)
        records = [json.loads(line) for line in saved[0].read_text().splitlines()]
        indexed = tmp_path / "indexed.jsonl"
        indexed.write_text(
            "".join(
                json.dumps({"design": index, "power": record["power"]}) + "\n"
                for index, record in enumerate(records)
            )
        )
        manifest = _adopt(indexed, tmp_path / "adopted.jsonl", grid="original")
        assert (manifest["source"]["layout"], manifest["source"]["grid"]) == ("indexed", "original")
        adopted = cr.read_checkpoint(tmp_path / "adopted.jsonl")
        assert set(adopted) == {json.dumps(cr._design_key(cell)) for cell in DESIGNS}

    def test_adopted_files_concatenate_into_one_checkpoint(self, runtime, saved, tmp_path):
        first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
        one = _adopt(saved[0], first, evidence="first campaign")
        two = _adopt(saved[0], second, evidence="second campaign")
        assert ra.manifest_digest(one) != ra.manifest_digest(two)
        joined = tmp_path / "joined.jsonl"
        joined.write_text(first.read_text() + second.read_text())
        assert len(cr.read_checkpoint(joined)) == len(DESIGNS)

    @pytest.mark.parametrize(
        ("name", "overrides", "message"),
        [
            ("construction", {"construction": "binomial_bb_difference_v2"}, "runtime construction"),
            ("law", {"law": "max(100, ceil(10 z^4))"}, "routing law"),
            ("revisions", {"revisions": []}, "name the revision"),
        ],
    )
    def test_a_construction_or_law_other_than_the_current_one_is_refused(
        self, runtime, saved, tmp_path, name, overrides, message
    ):
        out = tmp_path / "adopted.jsonl"
        with pytest.raises(ra.AdoptionError, match=message):
            _adopt(saved[0], out, **overrides)
        assert not out.exists()

    def test_an_existing_destination_is_not_overwritten(self, runtime, saved, tmp_path):
        out = tmp_path / "adopted.jsonl"
        out.write_text("kept\n")
        with pytest.raises(ra.AdoptionError, match="exists"):
            _adopt(saved[0], out)
        assert out.read_text() == "kept\n"

    def test_a_decision_path_that_differs_is_refused_and_nothing_is_written(
        self, runtime, saved, tmp_path, monkeypatch
    ):
        difference = ra.Difference(("pkg.core", "decide"), "a", "b")

        def failing(repo, revision):
            return ra.RevisionAudit(
                revision, ra.Audit({}, {}, (difference,)), (difference,), (), True
            )

        monkeypatch.setattr(ra, "audit_revision", failing)
        out = tmp_path / "adopted.jsonl"
        with pytest.raises(ra.AdoptionError, match=r"pkg\.core\.decide"):
            _adopt(saved[0], out)
        assert not out.exists()

    @pytest.mark.parametrize("field", ["asymptotic_part", "asymptotic_share", "asymptotic"])
    def test_a_record_the_current_runtime_does_not_reproduce_is_refused(
        self, runtime, saved, tmp_path, field
    ):
        lines = saved[0].read_text().splitlines()
        record = json.loads(lines[0])
        record["power"][field] += 1e-9
        altered = tmp_path / "altered.jsonl"
        altered.write_text("\n".join([json.dumps(record), *lines[1:]]) + "\n")
        out = tmp_path / "adopted.jsonl"
        with pytest.raises(ra.AdoptionError, match=field):
            _adopt(altered, out)
        assert not out.exists()

    def test_a_window_mass_the_design_does_not_have_is_refused(self, runtime, saved, tmp_path):
        record = json.loads(saved[0].read_text().splitlines()[0])
        record["power"]["omitted"] *= 3.0
        altered = tmp_path / "altered.jsonl"
        altered.write_text(json.dumps(record) + "\n")
        with pytest.raises(ra.AdoptionError, match="window mass"):
            _adopt(altered, tmp_path / "adopted.jsonl")

    def test_a_decision_the_runtime_refuses_is_refused(self, runtime, saved, tmp_path, monkeypatch):
        import increment.power._binomial as binomial

        monkeypatch.setattr(binomial, "refused", lambda decision: True)
        with pytest.raises(ra.AdoptionError, match="refuses"):
            _adopt(saved[0], tmp_path / "adopted.jsonl")

    def test_only_the_saved_layout_is_read(self, runtime, saved, tmp_path):
        record = json.loads(saved[0].read_text().splitlines()[0])
        record["power"]["finite_sample"] = 0.1
        other = tmp_path / "other.jsonl"
        other.write_text(json.dumps(record) + "\n")
        with pytest.raises(ra.AdoptionError, match="8 fields"):
            _adopt(other, tmp_path / "adopted.jsonl")

    def test_a_design_indexed_by_position_needs_the_grid_it_indexes(self, runtime, saved, tmp_path):
        record = json.loads(saved[0].read_text().splitlines()[0])
        indexed = tmp_path / "indexed.jsonl"
        indexed.write_text(json.dumps({"design": 0, "power": record["power"]}) + "\n")
        with pytest.raises(ra.AdoptionError, match="--grid original"):
            _adopt(indexed, tmp_path / "adopted.jsonl")


class TestAdoptedRecordsAreReadOnlyWhileTheirManifestHolds:
    @pytest.fixture
    def adopted(self, runtime, saved, tmp_path):
        out = tmp_path / "adopted.jsonl"
        _adopt(saved[0], out)
        return out

    @staticmethod
    def _lines(path):
        return path.read_text().splitlines()

    def test_an_enumeration_altered_after_adoption_is_refused(self, adopted):
        lines = self._lines(adopted)
        record = json.loads(lines[1])
        record["enumeration"]["finite_part"] += 1e-6
        adopted.write_text("\n".join([lines[0], json.dumps(record), *lines[2:]]) + "\n")
        with pytest.raises(cr.CheckpointError, match="not the one its manifest adopted"):
            cr.read_checkpoint(adopted)

    def test_a_manifest_that_is_altered_or_missing_leaves_its_records_uncited(self, adopted):
        lines = self._lines(adopted)
        manifest = json.loads(lines[0])
        manifest["adoption"]["campaign"]["evidence"] = "something else"
        adopted.write_text("\n".join([json.dumps(manifest), *lines[1:]]) + "\n")
        with pytest.raises(cr.CheckpointError, match="manifest this file does not hold"):
            cr.read_checkpoint(adopted)
        adopted.write_text("\n".join(lines[1:]) + "\n")
        with pytest.raises(cr.CheckpointError, match="manifest this file does not hold"):
            cr.read_checkpoint(adopted)

    def test_a_decision_path_that_has_changed_since_voids_the_adoption(self, adopted, monkeypatch):
        monkeypatch.setattr(ra, "destination_fingerprint", lambda: "another-tree")
        with pytest.raises(cr.CheckpointError, match="decision path has changed"):
            cr.read_checkpoint(adopted)

    def test_another_runtime_or_routing_law_voids_the_adoption(self, adopted, monkeypatch):
        monkeypatch.setattr(ra, "route_law", lambda: "max(1, ceil(1 z^4))")
        with pytest.raises(cr.CheckpointError, match="routing law"):
            cr.read_checkpoint(adopted)

    def test_a_record_a_manifest_does_not_list_is_refused(self, adopted):
        lines = self._lines(adopted)
        manifest = json.loads(lines[0])
        del manifest["adoption"]["records"][json.dumps(cr._design_key(DESIGNS[0]))]
        # The altered manifest has a new digest; the record still cites the old one.
        adopted.write_text("\n".join([json.dumps(manifest), *lines[1:]]) + "\n")
        with pytest.raises(cr.CheckpointError):
            cr.read_checkpoint(adopted)

    def test_a_record_that_cites_nothing_is_read_as_an_ordinary_one(self, adopted):
        lines = self._lines(adopted)
        record = json.loads(lines[1])
        del record["adopted"]
        adopted.write_text("\n".join([lines[0], json.dumps(record), *lines[2:]]) + "\n")
        power = cr.read_checkpoint(adopted)[json.dumps(cr._design_key(DESIGNS[0]))]
        assert power.adopted is None
        assert asdict(power.enumeration)["construction"]


class TestAdoptionCommand:
    def test_the_command_adopts_and_reports(self, runtime, saved, tmp_path, capsys):
        out = tmp_path / "adopted.jsonl"
        from increment.estimation.results import BINOMIAL_METHOD

        status = cr.main(
            [
                "adopt",
                str(saved[0]),
                "--out",
                str(out),
                "--revision",
                "abc",
                "--construction",
                BINOMIAL_METHOD,
                "--route-law",
                ra.route_law(),
            ]
        )
        assert status == 0
        assert "adopted 2 records" in capsys.readouterr().out
        assert cr.read_checkpoint(out)

    def test_the_command_refuses_with_a_status_and_the_reason(
        self, runtime, saved, tmp_path, capsys
    ):
        out = tmp_path / "adopted.jsonl"
        status = cr.main(
            [
                "adopt",
                str(saved[0]),
                "--out",
                str(out),
                "--revision",
                "abc",
                "--construction",
                "binomial_bb_difference_v2",
                "--route-law",
                ra.route_law(),
            ]
        )
        assert status == 2
        assert "runtime construction" in capsys.readouterr().err
        assert not out.exists()


class TestAnOrdinaryResumeNeverAdopts:
    def test_an_earlier_layout_is_refused_on_resume_whatever_it_holds(
        self, runtime, saved, tmp_path
    ):
        with pytest.raises(cr.CheckpointError, match="adopt"):
            cr.read_checkpoint(saved[0])
