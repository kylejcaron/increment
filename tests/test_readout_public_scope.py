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
from increment.estimation.results import LiftEstimate
from increment.semantics.design import Randomized
from tests.readout_journeys import (
    assert_failed_assigned_integrity as check_failed_assigned_integrity,
)
from tests.readout_journeys import (
    assert_filtered_family_scope_unchanged,
    assert_prior_free_sampling_family,
)
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
        ("blocked", "unsupported_assignment", "none", None),
        ("adaptive", "unsupported_assignment", "none", None),
        ("quota", "unsupported_assignment", "none", None),
        ("fixed_counts", "unsupported_assignment", "none", None),
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


def test_observational_run_marks_assignment_integrity_not_applicable():
    from increment.semantics.design import AdjustmentSet, Observational

    frame = pd.DataFrame(
        [
            {
                "unit": f"{group}{i}",
                "arm": group,
                "x": float(i % 11),
                "rev": float(i % 7 + 1),
            }
            for group in ("control", "treatment")
            for i in range(40)
        ]
    )
    results = _readouts(
        Analysis.from_unit_summary(
            frame,
            unit="unit",
            group="arm",
            design=Observational(
                control_group="control", adjustment=AdjustmentSet(covariates=("x",))
            ),
            metrics=[MetricSpec(name="rev", type="mean")],
        ).run()
    )

    assert results.metadata is not None
    (integrity,) = next(iter(results.metadata.scope.by_source.values())).integrity
    assert integrity.status == "not_applicable"
    assert integrity.construction == "none"
    assert integrity.observed is None


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


def _assert_failed_assigned_integrity(results: object) -> None:
    assert isinstance(results, LiftEstimates)
    check_failed_assigned_integrity(results)


def test_informative_prior_keeps_sampling_family_evidence_prior_free():
    from increment.estimation.inference import Normal

    rows = [
        {"unit": f"{group}-{index}", "arm": group, "y": value}
        for group in ("control", "treatment")
        for index, value in enumerate([9.0, 11.0] * 50)
    ]
    analysis = Analysis.from_unit_summary(
        pd.DataFrame(rows),
        unit="unit",
        group="arm",
        control="control",
        metrics=[
            MetricSpec(
                name="y",
                type="mean",
                prior=Normal(mu=0.10, sigma=0.01),
                preferred_direction="increase",
            )
        ],
        plan=AnalysisPlan(primary="y"),
    )
    results = analysis.run()
    assert isinstance(results, LiftEstimates)
    assert_prior_free_sampling_family(results)


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


def test_unit_summary_preserves_literal_none_arm_but_excludes_unassigned_roster():
    from increment.sources import UNASSIGNED_LABEL

    rows: list[dict[str, str | int | float | None]] = [
        {"unit": f"{arm}-{index}", "arm": arm, "rev": float(index + 1)}
        for arm in ("control", "treatment", "None")
        for index in range(4)
    ]
    rows.extend(
        {"unit": f"unassigned-{index}", "arm": None, "rev": float(index + 1)} for index in range(2)
    )
    results = _readouts(
        Analysis.from_unit_summary(
            pd.DataFrame(rows),
            unit="unit",
            group="arm",
            design=Randomized(
                control_group="control",
                allocation={"control": 0.3, "treatment": 0.3, "None": 0.4},
                allocation_scheme="blocked",
            ),
            on_unassigned="exclude",
            metrics=[MetricSpec(name="rev", type="mean")],
        ).run()
    )

    assert {row.group_id for row in _lift_rows(results)} == {"treatment", "None"}
    assert results.metadata is not None
    (integrity,) = next(iter(results.metadata.scope.by_source.values())).integrity
    assert integrity.status == "unsupported_assignment"
    assert integrity.observed == {
        "control": 4,
        "treatment": 4,
        "None": 4,
        UNASSIGNED_LABEL: 2,
    }
    assert results.source is not None
    assert any(
        component["kind"] == "assignment_counts" for component in results.source["components"]
    )


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
    from datetime import date

    from increment.readouts import _sequential_scope
    from tests.test_sequential_public_sources import _native_fixture

    _connection, _definitions, analysis = _native_fixture("bernoulli", triggered=True)
    analysis.capture_sequential(finalized=True, as_of=date(2025, 1, 16))
    source_type = type(analysis._src)
    original_counts = source_type.assignment_counts
    count_calls = []

    def counted_counts(self, *, population="assigned"):
        count_calls.append(population)
        return original_counts(self, population=population)

    original_integrity = _sequential_scope.assignment_integrity
    integrity_calls = []

    def counted_integrity(*args, **kwargs):
        integrity_calls.append(True)
        return original_integrity(*args, **kwargs)

    monkeypatch.setattr(source_type, "assignment_counts", counted_counts)
    monkeypatch.setattr(_sequential_scope, "assignment_integrity", counted_integrity)
    try:
        results = analysis.run()
    finally:
        analysis.close()

    (integrity,) = next(iter(results.metadata.scope.by_source.values())).integrity
    assert count_calls == ["assigned"]
    assert len(integrity_calls) == 1
    populations = {row.analysis_population for row in results}
    assert populations == {"assigned", "triggered"}
    assigned = next(row for row in results if row.analysis_population == "assigned")
    triggered = next(row for row in results if row.analysis_population == "triggered")
    assert (triggered.estimand, triggered.value_scale, triggered.alternative) == (
        assigned.estimand,
        assigned.value_scale,
        assigned.alternative,
    )
    assert triggered.failure_code == "readout.cell.unsupported_request"


def test_sequential_narrowed_view_keeps_the_registered_family_membership():
    from tests.binary_sequential_cases import ROUTES, frame_analysis, unit_rows

    analysis = frame_analysis(
        unit_rows(seed=29, n=180, secondaries=2),
        ROUTES["always_valid"].inference,
        secondaries=2,
    )

    def memberships(results):
        return {
            family.name: tuple(
                sorted(
                    (cell.metric, cell.group_id, cell.estimand, cell.analysis_population)
                    for cell in family.members
                )
            )
            for family in results.metadata.scope.families
        }

    full = analysis.run()
    narrowed = analysis.run(metrics=["purchase"], estimands=["itt"])
    assert {row.metric for row in narrowed} == {"purchase"}
    assert memberships(narrowed) == memberships(full)
    assert {member[0] for members in memberships(narrowed).values() for member in members} >= {
        "purchase",
        "secondary_0",
        "secondary_1",
    }


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
    assert failed[0].role == "guardrail"
    assert failed[0].multiplicity_status == "declared_plan"
    guardrail_family = next(
        family
        for family in results.metadata.scope.families
        if family.family_id == failed[0].family_id
    )
    assert any(
        cell.metric == "guard" and cell.method_role == "decision"
        for cell in guardrail_family.members
    )
    assert all(row.decision_scope_complete is False for row in results)
    assert results.metadata.scope.decision_complete("assigned") is False

    sliced = results[:1]
    assert sliced.metadata.partial is True
    assert sliced.metadata.scope == results.metadata.scope
    filtered = results.filter(lambda row: row.metric == "rev")
    assert_filtered_family_scope_unchanged(results, filtered)
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
    assert failed.failure_code == "readout.cell.missing_metric_observations"
    assert failed.failure_context["group_id"] == "treatment"
    assert failed.lift is None
    (surviving,) = [row for row in results if row.failure_code is None]
    assert surviving.metric == "rev" and surviving.lift is not None
    failed_family = next(
        family for family in results.metadata.scope.families if family.family_id == failed.family_id
    )
    assert any(
        cell.metric == "guard" and cell.group_id == failed.group_id
        for cell in failed_family.members
    )


def test_complete_readout_reports_complete_scope():
    results = _analysis(_frame(guard_constant=False)).run()
    assert [row.failure_code for row in results] == [None, None]
    assert results.metadata.scope.decision_complete("assigned") is True
    assert all(row.decision_scope_complete is True for row in results)


@pytest.mark.parametrize("operation", ["metadata_set", "metadata_delete"])
def test_scoped_metadata_cannot_be_rebound_or_removed(operation):
    results = _readouts(_analysis(_frame(guard_constant=False)).run())
    metadata = results.metadata
    assert metadata is not None
    with pytest.raises(InvalidRequestError) as raised:
        if operation == "metadata_set":
            results.metadata = None
        else:
            del results.metadata
    assert raised.value.code == "readout.collection.mutation_unsupported"
    assert dict(raised.value.context) == {"operation": operation, "model": "LiftEstimates"}
    assert results.metadata is not None
    assert results.metadata == metadata
    assert results.metadata.scope.decision_complete("assigned") is True
    assert nw.from_native(results.to_frame(), eager_only=True)["view_partial"].to_list() == [
        False,
        False,
    ]


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
    assert failed.role == "primary"
    assert failed.multiplicity_status == "declared_plan"
    assert failed.family_id is not None
    primary_family = next(
        family for family in results.metadata.scope.families if family.family_id == failed.family_id
    )
    assert any(
        cell.metric == "rev" and cell.group_id == failed.group_id for cell in primary_family.members
    )
    (surviving,) = [row for row in results if row.failure_code is None]
    assert surviving.metric == "guard" and surviving.lift is not None
    assert results.metadata.scope.decision_complete("assigned") is False
    assert all(row.decision_scope_complete is False for row in results)


def test_failed_multiarm_secondary_keeps_full_family_and_sensitivity_provenance():
    from increment import Method
    from increment.tables import estimates_to_readout

    rng = np.random.default_rng(23)
    frame_rows = []
    for arm, shift in (("control", 0.0), ("treatment_a", 0.25), ("treatment_b", 0.5)):
        for index in range(100):
            x = float(rng.normal())
            frame_rows.append(
                {
                    "unit": f"{arm}-{index}",
                    "arm": arm,
                    "primary": 2.0 + shift + x,
                    "secondary": None
                    if arm == "treatment_b"
                    else 3.0 + shift + x + float(rng.normal(scale=0.2)),
                    "x": x,
                }
            )
    with pytest.warns(IncrementWarning) as captured:
        analysis = Analysis.from_unit_summary(
            pd.DataFrame(frame_rows),
            unit="unit",
            group="arm",
            control="control",
            metrics=[
                MetricSpec(name="primary", type="mean"),
                MetricSpec(
                    name="secondary",
                    type="mean",
                    missing="drop",
                    covariate="x",
                    decision_method=Method(name="unadjusted"),
                    sensitivity_methods=(Method(name="cuped", variance_reduction="cuped"),),
                ),
            ],
            plan=AnalysisPlan(primary="primary", secondaries=["secondary"], q=0.2),
        )
    assert warning_codes(captured) == ["frame.validation.metric_missing_drop"]
    results = _readouts(analysis.run())
    rows = _lift_rows(results)
    secondary_rows = [row for row in rows if row.metric == "secondary"]
    decision_rows = [row for row in secondary_rows if row.method_role == "decision"]
    sensitivity_rows = [row for row in secondary_rows if row.method_role == "sensitivity"]
    assert results.metadata is not None
    assert {row.group_id for row in decision_rows} == {"treatment_a", "treatment_b"}
    failed = next(row for row in decision_rows if row.group_id == "treatment_b")
    successful = next(row for row in decision_rows if row.group_id == "treatment_a")
    assert failed.failure_code == "readout.cell.missing_metric_observations"
    assert failed.role == successful.role == "secondary"
    assert failed.multiplicity_status == successful.multiplicity_status == "declared_plan"
    assert failed.family_id == successful.family_id
    assert successful.discovery is not None
    assert failed.discovery is None
    (family,) = [
        family for family in results.metadata.scope.families if family.family_id == failed.family_id
    ]
    assert {cell.group_id for cell in family.members if cell.metric == "secondary"} == {
        "treatment_a",
        "treatment_b",
    }
    assert sensitivity_rows
    assert all(row.multiplicity_status == "declared_plan" for row in sensitivity_rows)
    assert all(row.family_id is None and row.discovery is None for row in sensitivity_rows)

    frame = nw.from_native(results.to_frame(), eager_only=True)
    assert set(frame.filter(nw.col("metric") == "secondary")["multiplicity_status"].to_list()) == {
        "declared_plan"
    }
    readout = estimates_to_readout(results)
    assert all(
        row["multiplicity_status"] == "declared_plan"
        for row in readout
        if row["metric"] == "secondary"
    )
    failed_readout = next(
        row for row in readout if row["metric"] == "secondary" and row["group_id"] == "treatment_b"
    )
    assert "chance_to_beat (advisory)" not in failed_readout
    assert failed_readout["posterior_chance_to_beat"] is None
    assert failed_readout["posterior_risk_if_shipped"] is None
    assert failed_readout["posterior_prob_favorable"] is None
    unavailable_row = successful.model_copy(update={"sampling_available": False})
    (unavailable_readout,) = estimates_to_readout([unavailable_row])
    assert unavailable_readout["posterior_chance_to_beat"] is None
    assert unavailable_readout["posterior_risk_if_shipped"] is None
    assert unavailable_readout["posterior_prob_favorable"] is None
    from increment.tables import readout_table

    html = readout_table(readout).gt().as_raw_html()
    assert "Declared plan" in html
    restored = ReadoutResults.model_validate_json(results.model_dump_json())
    assert restored.metadata == results.metadata
    restored_rows = []
    for row in restored:
        assert isinstance(row, LiftEstimate)
        restored_rows.append(row)
    assert [row.family_id for row in restored_rows] == [row.family_id for row in rows]
    filtered = results.filter(lambda row: row.metric == "secondary")
    assert filtered.metadata is not None and results.metadata is not None
    assert filtered.metadata.scope.families == results.metadata.scope.families
    assert {row.family_id for row in _lift_rows(filtered) if row.method_role == "decision"} == {
        failed.family_id
    }


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
        results = _readouts(analysis.run())
    assert warning_codes(captured) == ["readouts.run.cell_refused"]
    rows = _lift_rows(results)
    (sensitivity,) = [row for row in rows if row.method_role == "sensitivity"]
    (failed_decision,) = [row for row in rows if row.method_role == "decision"]
    assert sensitivity.method == "cuped" and sensitivity.lift is not None
    assert failed_decision.failure_code is not None and failed_decision.lift is None
    assert failed_decision.decision_scope_complete is False
    assert results.metadata is not None
    assert results.metadata.scope.decision_complete("assigned") is False
    frame = nw.from_native(results.to_frame(), eager_only=True)
    assert len(frame) == len(results)
    assert frame["failure_code"].is_null().sum() == len(frame) - 1
    restored = ReadoutResults.model_validate_json(results.model_dump_json())
    assert restored.metadata == results.metadata
    restored_rows = []
    for row in restored:
        assert isinstance(row, LiftEstimate)
        restored_rows.append(row)
    assert [row.failure_code for row in restored_rows] == [row.failure_code for row in rows]


def test_every_cell_failing_remains_an_explicit_refusal():
    frame = _frame()
    frame["rev"] = 1.0
    with pytest.warns(IncrementWarning) as captured:
        with pytest.raises(UnsupportedRequestError) as raised:
            _analysis(frame).run()
    assert warning_codes(captured) == ["readouts.run.cell_refused", "readouts.run.cell_refused"]
    assert raised.value.code == "readout.estimate_lift_every"
    from typing import Any

    from pydantic import TypeAdapter

    failures = TypeAdapter(list[dict[str, Any]]).validate_python(raised.value.context["failures"])
    failure_rows = set()
    for failure in failures:
        context = failure["context"]
        failure_rows.add(
            (
                failure["metric"],
                failure["group_id"],
                failure["method"],
                failure["code"],
                context["reason"],
            )
        )
    assert failure_rows == {
        ("rev", "treatment", "unadjusted", "estimation.engine.lift_guard", "zero_variance"),
        ("guard", "treatment", "unadjusted", "estimation.engine.lift_guard", "zero_variance"),
    }
    warning_rows = set()
    for warning in captured:
        assert isinstance(warning.message, IncrementWarning)
        warning_rows.add(
            (warning.message.context["metric_name"], warning.message.context["group_id"])
        )
    assert warning_rows == {("rev", "treatment"), ("guard", "treatment")}


def _readouts(value) -> LiftEstimates:
    assert isinstance(value, LiftEstimates)
    assert value.metadata is not None
    return value


def _lift_rows(results: LiftEstimates) -> list[LiftEstimate]:
    rows = []
    for row in results:
        assert isinstance(row, LiftEstimate)
        rows.append(row)
    return rows
