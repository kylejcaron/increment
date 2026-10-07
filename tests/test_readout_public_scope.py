"""Public Analysis.run readouts keep required-cell failures and completeness."""

import copy
import pickle

import narwhals as nw
import numpy as np
import pandas as pd
import pytest

from increment import Analysis, AnalysisPlan, MetricSpec
from increment.breakout.estimates import LiftEstimates
from increment.errors import IncrementWarning, InvalidRequestError, UnsupportedRequestError
from increment.estimation.readout_types import CellKey, ReadoutResults
from increment.semantics.design import Randomized
from tests.warning_codes import warning_codes, warning_context


def _frame(*, guard_constant=True, drop_guard_treatment=False):
    rng = np.random.default_rng(7)
    rows = []
    for group, shift in (("control", 0.0), ("treatment", 0.3)):
        for i in range(200):
            row = {
                "unit": f"{group}{i}",
                "arm": group,
                "rev": float(rng.normal(1.0 + shift, 0.5)),
                "guard": 1.0 if guard_constant else float(rng.normal(1.0, 0.5)),
            }
            if drop_guard_treatment and group == "treatment":
                row["guard"] = None
            rows.append(row)
    return pd.DataFrame(rows)


def _analysis(frame, *, drop_missing_guard=False):
    return Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        control="control",
        metrics=[
            MetricSpec(name="rev", type="mean", preferred_direction="increase"),
            MetricSpec(
                name="guard",
                type="mean",
                preferred_direction="decrease",
                missing="drop" if drop_missing_guard else "error",
            ),
        ],
        plan=AnalysisPlan(primary="rev", guardrails=["guard"]),
    )


@pytest.mark.parametrize(
    ("scheme", "status", "construction", "alpha"),
    [
        ("independent", "failed", "always_valid", 0.001),
        (None, "not_checked_missing_declaration", "none", None),
    ],
)
def test_analysis_run_surfaces_assignment_integrity_without_changing_rows(
    scheme, status, construction, alpha
):
    frame = pd.DataFrame(
        [{"unit": f"c{i}", "arm": "control", "rev": float(i % 7 + 1)} for i in range(900)]
        + [{"unit": f"t{i}", "arm": "treatment", "rev": float(i % 7 + 2)} for i in range(100)]
    )

    def run(design_scheme):
        design = Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme=design_scheme,
        )
        return Analysis.from_unit_summary(
            frame,
            unit="unit",
            group="arm",
            design=design,
            metrics=[MetricSpec(name="rev", type="mean", preferred_direction="increase")],
            plan=AnalysisPlan(primary="rev"),
        ).run()

    results = run(scheme)
    baseline = run(None)
    (integrity,) = next(iter(results.metadata.scope.by_source.values())).integrity
    assert (integrity.status, integrity.construction, integrity.alpha) == (
        status,
        construction,
        alpha,
    )
    assert integrity.observed == {"control": 900, "treatment": 100}
    restored = ReadoutResults.model_validate_json(results.model_dump_json())
    assert next(iter(restored.metadata.scope.by_source.values())).integrity == (integrity,)
    assert [(row.metric, row.group_id, row.lift) for row in results] == [
        (row.metric, row.group_id, row.lift) for row in baseline
    ]
    assert len(results) == 1 and results[0].lift is not None



def _imbalanced_assignment_frame():
    return pd.DataFrame(
        [
            {
                "unit": f"{group[0]}{i}",
                "arm": group,
                "ds": pd.Timestamp("2025-01-01"),
                "rev": float(i % 7 + 1),
                "took": int(group == "treatment" and i % 2 == 0),
            }
            for group, count in (("control", 900), ("treatment", 100))
            for i in range(count)
        ]
    )


def _independent_design(design_type=Randomized):
    if design_type is Randomized:
        return design_type(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        )
    from increment.semantics.design import Encouragement, UptakeSpec

    return Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="took"),
        allocation={"control": 0.5, "treatment": 0.5},
        allocation_scheme="independent",
    )


def _assert_failed_assigned_integrity(results):
    (integrity,) = next(iter(results.metadata.scope.by_source.values())).integrity
    assert integrity.status == "failed"
    assert integrity.construction == "always_valid"
    assert integrity.alpha == 0.001
    assert integrity.analysis_population == "assigned"
    assert integrity.observed == {"control": 900, "treatment": 100}


def test_unit_panel_run_checks_declared_assignment_counts():
    frame = _imbalanced_assignment_frame()
    results = Analysis.from_unit_panel(
        frame,
        unit="unit",
        group="arm",
        date="ds",
        design=_independent_design(),
        metrics=[MetricSpec(name="rev", type="mean")],
    ).run()

    _assert_failed_assigned_integrity(results)


def test_moments_replay_run_checks_declared_assignment_counts(tmp_path):
    import pyarrow.parquet as pq

    from increment.semantics.design import Randomized

    design = _independent_design(Randomized)
    summary = Analysis.from_unit_summary(
        _imbalanced_assignment_frame(),
        unit="unit",
        group="arm",
        design=design,
        metrics=[MetricSpec(name="rev", type="mean")],
    )
    path = tmp_path / "assignment-counts.parquet"
    summary.export(path)
    replay = Analysis.from_moments(
        pq.read_table(path).to_pylist(),
        metrics=[MetricSpec(name="rev", type="mean")],
        design=design,
    )

    _assert_failed_assigned_integrity(replay.run())


def test_encouragement_run_checks_assignment_not_uptake_counts():
    from increment.semantics.design import Encouragement

    frame = _imbalanced_assignment_frame()
    results = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        design=_independent_design(Encouragement),
        metrics=[MetricSpec(name="rev", type="mean")],
    ).run(estimands=["itt", "compliance"])

    _assert_failed_assigned_integrity(results)

def _registered_integrity_analysis(*, n_control=900, n_treatment=100):
    from tests.binary_sequential_cases import ROUTES, unit_rows

    rows = [
        row
        for row in unit_rows(seed=17, n=n_control)
        if row["variant"] == "control" or int(row["user_id"].split("-")[1]) < n_treatment
    ]
    return Analysis.from_unit_summary(
        pd.DataFrame(rows),
        unit="user_id",
        group="variant",
        metrics=[MetricSpec(name="purchase", type="conversion")],
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        plan=AnalysisPlan(primary="purchase", inference=ROUTES["always_valid"].inference),
        experiment_id="exp",
        exposure_date="enrollment",
    )


def test_registered_sequential_run_checks_source_assigned_counts():
    results = _registered_integrity_analysis().run()

    (integrity,) = next(iter(results.metadata.scope.by_source.values())).integrity
    assert integrity.status == "failed"
    assert integrity.construction == "always_valid"
    assert integrity.alpha == 0.001
    assert integrity.observed == {"control": 900, "treatment": 100}
    assert any(
        component["kind"] == "assignment_counts" for component in results.source["components"]
    )


def test_registered_sequential_run_marks_unavailable_source_counts(monkeypatch):


    from increment.errors import CapabilityError

    analysis = _registered_integrity_analysis(n_control=200, n_treatment=200)

    def unavailable_counts(self):
        raise CapabilityError(
            "assignment counts are unavailable",
            code="test.source.assignment_counts_unavailable",
            context={},
        )

    monkeypatch.setattr(type(analysis._src), "unit_counts", unavailable_counts)
    results = analysis.run()

    (integrity,) = next(iter(results.metadata.scope.by_source.values())).integrity
    assert integrity.status == "not_checked_missing_counts"
    assert integrity.code == "integrity.counts_missing"
    assert integrity.observed is None

def test_sequential_trigger_augmentation_reuses_finite_snapshot_integrity(monkeypatch):
    from increment.readouts import _sequential_scope
    from increment.readouts._sequential_scope import scope_sequential_results

    analysis = _registered_integrity_analysis()
    source_type = type(analysis._src)
    original_counts = source_type.unit_counts
    count_calls = []

    def counted_counts(self):
        count_calls.append(True)
        return original_counts(self)

    original_integrity = _sequential_scope.assignment_integrity
    integrity_calls = []

    def counted_integrity(*args, **kwargs):
        integrity_calls.append(True)
        return original_integrity(*args, **kwargs)

    monkeypatch.setattr(source_type, "unit_counts", counted_counts)
    monkeypatch.setattr(_sequential_scope, "assignment_integrity", counted_integrity)
    assigned = analysis.run()
    augmented = scope_sequential_results(
        analysis._src,
        assigned,
        assigned.sequential_snapshot,
        metrics=["purchase"],
        estimands=None,
        triggered_declared=True,
    )

    (integrity,) = next(iter(augmented.metadata.scope.by_source.values())).integrity
    assert len(count_calls) == 1
    assert len(integrity_calls) == 1
    assert integrity == next(iter(assigned.metadata.scope.by_source.values())).integrity[0]
    assert any(
        row.analysis_population == "triggered"
        and row.failure_code == "readout.cell.unsupported_request"
        for row in augmented
    )

def test_constant_guardrail_failure_survives_collection_views_and_saved_results():
    with pytest.warns(IncrementWarning) as captured:
        results = _analysis(_frame()).run()
    assert warning_codes(captured) == ["readouts.run.cell_refused"]
    context = warning_context(captured, "readouts.run.cell_refused")
    assert (context["metric_name"], context["group_id"], context["method_name"]) == (
        "guard",
        "treatment",
        "unadjusted",
    )
    assert isinstance(results, LiftEstimates)
    assert results.metadata is not None
    failed = [row for row in results if row.failure_code is not None]
    assert [(row.metric, row.failure_context["reason"]) for row in failed] == [
        ("guard", "zero_variance")
    ]
    assert failed[0].lift is None
    assert all(row.decision_scope_complete is False for row in results)
    assert results.metadata.scope.decision_complete("assigned") is False

    sliced = results[:1]
    assert sliced.metadata.partial is True
    assert sliced.metadata.scope == results.metadata.scope
    filtered = results.filter(lambda row: row.metric == "rev")
    assert filtered.metadata.partial is True
    assert "view_partial" in filtered.to_frame().columns

    restored = ReadoutResults.model_validate_json(results.model_dump_json())
    assert restored.metadata == results.metadata
    assert [row.failure_code for row in restored] == [row.failure_code for row in results]
    assert [CellKey.from_row(row) for row in restored] == [CellKey.from_row(row) for row in results]


def test_missing_arm_for_one_metric_is_a_typed_cell_failure_while_another_observes_it():
    with pytest.warns(IncrementWarning) as captured:
        analysis = _analysis(
            _frame(guard_constant=False, drop_guard_treatment=True), drop_missing_guard=True
        )
    assert warning_codes(captured) == ["frame.validation.metric_missing_drop"]
    assert warning_context(captured, "frame.validation.metric_missing_drop")["name"] == "guard"
    results = analysis.run()
    (failed,) = [row for row in results if row.failure_code is not None]
    assert failed.metric == "guard"
    assert failed.failure_code == "readout.cell.missing_arm"
    assert failed.failure_context["group_id"] == "treatment"
    assert failed.lift is None
    (surviving,) = [row for row in results if row.failure_code is None]
    assert surviving.metric == "rev" and surviving.lift is not None
    assert results.metadata.scope.decision_complete("assigned") is False


def test_complete_readout_reports_complete_scope():
    results = _analysis(_frame(guard_constant=False)).run()
    assert [row.failure_code for row in results] == [None, None]
    assert results.metadata.scope.decision_complete("assigned") is True
    assert all(row.decision_scope_complete is True for row in results)


@pytest.mark.parametrize("operation", ["metadata_set", "metadata_delete"])
def test_scoped_metadata_cannot_be_rebound_or_removed(operation):
    results = _analysis(_frame(guard_constant=False)).run()
    metadata = results.metadata
    with pytest.raises(InvalidRequestError) as raised:
        if operation == "metadata_set":
            results.metadata = None
        else:
            del results.metadata
    assert raised.value.code == "readout.collection.mutation_unsupported"
    assert dict(raised.value.context) == {"operation": operation, "model": "LiftEstimates"}
    assert results.metadata == metadata
    assert results.metadata.scope.decision_complete("assigned") is True
    assert results.to_frame()["view_partial"].eq(False).all()


@pytest.mark.parametrize("operation", ["deepcopy", "pickle"])
def test_collection_copy_protocols_preserve_failed_scope_and_provenance(operation):
    with pytest.warns(IncrementWarning) as captured:
        results = _analysis(_frame()).run()
    assert warning_codes(captured) == ["readouts.run.cell_refused"]
    copied = (
        copy.deepcopy(results) if operation == "deepcopy" else pickle.loads(pickle.dumps(results))
    )
    assert copied.metadata == results.metadata
    assert copied.model_dump_json() == results.model_dump_json()
    frame = copied.to_frame()
    assert frame.loc[frame["metric"] == "guard", "lift"].isna().all()
    assert frame["decision_scope_complete"].eq(False).all()


def test_concatenating_a_slice_with_its_complement_restores_the_whole_scope():
    results = _analysis(_frame(guard_constant=False)).run()
    left = results.filter(lambda row: row.metric == "rev")
    right = results.filter(lambda row: row.metric == "guard")
    joined = left.concat(right)
    assert joined.metadata.partial is False
    assert joined.metadata.scope == results.metadata.scope


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
def test_frames_carry_failure_and_scope_columns_on_every_backend(backend):
    with pytest.warns(IncrementWarning) as captured:
        frame = _analysis(_frame()).run().to_frame(backend=backend)
    assert warning_codes(captured) == ["readouts.run.cell_refused"]
    wrapped = nw.from_native(frame, eager_only=True)
    assert {"failure_code", "decision_scope_complete", "view_partial"} <= set(wrapped.columns)


def test_readout_table_renders_failed_scope_and_partial_collection_view():
    pytest.importorskip("coeftable")
    from increment.tables import estimates_to_readout, readout_table

    with pytest.warns(IncrementWarning) as captured:
        results = _analysis(_frame()).run()
    assert warning_codes(captured) == ["readouts.run.cell_refused"]
    full_rows = estimates_to_readout(results)
    failed_row = next(row for row in full_rows if row["failure_code"] is not None)
    assert failed_row["stat_sig"] is None
    full_html = readout_table(full_rows).gt().as_raw_html()
    assert "estimation.engine.lift_guard" in full_html
    assert "zero_variance" in full_html
    assert "Incomplete" in full_html

    filtered = results.filter(lambda row: row.metric == "rev")
    rows = estimates_to_readout(filtered)
    assert all(row["view_partial"] is True for row in rows)
    partial_html = readout_table(rows).gt().as_raw_html()
    assert "Partial view" in partial_html
    assert "Incomplete" in partial_html


def test_constant_primary_is_a_failed_decision_cell_beside_a_surviving_guardrail():
    frame = _frame(guard_constant=False)
    frame["rev"] = 1.0
    with pytest.warns(IncrementWarning) as captured:
        results = _analysis(frame).run()
    assert warning_codes(captured) == ["readouts.run.cell_refused"]
    (failed,) = [row for row in results if row.failure_code is not None]
    assert (failed.metric, failed.failure_context["reason"]) == ("rev", "zero_variance")
    assert failed.lift is None
    (surviving,) = [row for row in results if row.failure_code is None]
    assert surviving.metric == "guard" and surviving.lift is not None
    assert results.metadata.scope.decision_complete("assigned") is False
    assert all(row.decision_scope_complete is False for row in results)


def test_sensitivity_survives_when_every_decision_method_fails():
    from increment import Method

    rng = np.random.default_rng(18)
    n = 200
    rows = []
    for group, shift in (("control", 0.0), ("treatment", 0.2)):
        x = rng.normal(1.0, np.sqrt(38.0), n)
        y = x + rng.normal(0.0, np.sqrt(2.0), n) + shift
        rows.extend(
            {"unit": f"{group}{i}", "arm": group, "rev": float(y[i]), "x": float(x[i])}
            for i in range(n)
        )
    analysis = Analysis.from_unit_summary(
        pd.DataFrame(rows),
        unit="unit",
        group="arm",
        control="control",
        metrics=[
            MetricSpec(
                name="rev",
                type="mean",
                covariate="x",
                decision_method=Method(name="unadjusted"),
                sensitivity_methods=(Method(name="cuped", variance_reduction="cuped"),),
            )
        ],
    )
    with pytest.warns(IncrementWarning) as captured:
        results = analysis.run()
    assert warning_codes(captured) == ["readouts.run.cell_refused"]
    (sensitivity,) = [row for row in results if row.method_role == "sensitivity"]
    (failed_decision,) = [row for row in results if row.method_role == "decision"]
    assert sensitivity.method == "cuped" and sensitivity.lift is not None
    assert failed_decision.failure_code is not None and failed_decision.lift is None
    assert failed_decision.decision_scope_complete is False
    assert results.metadata.scope.decision_complete("assigned") is False
    frame = results.to_frame()
    assert len(frame) == len(results)
    assert frame["failure_code"].notna().sum() == 1
    restored = ReadoutResults.model_validate_json(results.model_dump_json())
    assert restored.metadata == results.metadata
    assert [row.failure_code for row in restored] == [row.failure_code for row in results]


def test_every_cell_failing_remains_an_explicit_refusal():
    frame = _frame()
    frame["rev"] = 1.0
    with pytest.warns(IncrementWarning) as captured:
        with pytest.raises(UnsupportedRequestError) as raised:
            _analysis(frame).run()
    assert warning_codes(captured) == ["readouts.run.cell_refused", "readouts.run.cell_refused"]
    assert raised.value.code == "readout.estimate_lift_every"
    assert {
        (
            failure["metric"],
            failure["group_id"],
            failure["method"],
            failure["code"],
            failure["context"]["reason"],
        )
        for failure in raised.value.context["failures"]
    } == {
        ("rev", "treatment", "unadjusted", "estimation.engine.lift_guard", "zero_variance"),
        ("guard", "treatment", "unadjusted", "estimation.engine.lift_guard", "zero_variance"),
    }
    assert {
        (warning.message.context["metric_name"], warning.message.context["group_id"])
        for warning in captured
    } == {("rev", "treatment"), ("guard", "treatment")}
