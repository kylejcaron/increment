"""Observational and encouragement readouts enumerate their expected cells before estimation."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from increment import Analysis, MetricSpec
from increment.breakout.estimates import LiftEstimates
from increment.errors import IncrementWarning
from increment.estimation.readout_types import CellKey, ReadoutResults
from increment.semantics.design import (
    AdjustmentSet,
    Encouragement,
    ExclusionRestriction,
    Observational,
    UptakeSpec,
)
from increment.readouts._design_scope import resolve_roster
from tests.warning_codes import warning_codes, warning_context


def test_count_roster_excludes_reserved_unassigned_sentinel_not_literal_none():
    class Source:
        context = SimpleNamespace(cluster=None)

        def assignment_counts(self, *, population):
            assert population == "assigned"
            return {"(unassigned)": 2, "None": 3, "treatment": 4}

    arms, roster_source, complete, counts = resolve_roster(
        Source(),
        SimpleNamespace(allocation=None),
        "assigned",
        {"revenue": {"None", "treatment"}},
    )

    assert arms == ("None", "treatment")
    assert roster_source == "experiment_counts"
    assert complete is True
    assert counts == {"(unassigned)": 2, "None": 3, "treatment": 4}


def test_artifact_assignment_counts_skip_null_before_string_conversion():
    from increment.query.source import ArtifactMomentSource

    class Reader:
        _population_units = None
        _artifact_experiment = SimpleNamespace(trigger=None)

        def _extension(self, request):
            assert request == {"kind": "assignment_counts", "populations": ("assigned",)}
            return SimpleNamespace(kind="assignment_counts", populations=("assigned",))

        def _read_extension(self, extension, *, request):
            assert request["kind"] == "assignment_counts"
            return [
                {"population": "assigned", "group_id": None, "n_units": 2},
                {"population": "assigned", "group_id": "None", "n_units": 3},
            ]

    assert ArtifactMomentSource.unit_counts(Reader()) == {"None": 3}


def _encouragement_frame(n=300, seed=3):
    rng = np.random.default_rng(seed)
    uptake = np.concatenate([np.zeros(n), rng.binomial(1, 0.5, n)])
    outcome = rng.normal(1.0, 0.5, 2 * n) + 0.4 * uptake
    return pd.DataFrame(
        {
            "unit_id": range(2 * n),
            "group_id": ["control"] * n + ["treatment"] * n,
            "uptake": uptake,
            "rev": outcome,
        }
    )


def test_artifact_triggered_counts_skip_null_before_string_conversion():
    from increment.query.source import ArtifactMomentSource

    class Reader:
        context = SimpleNamespace(cluster=None)

        def _extension(self, request):
            return SimpleNamespace(
                kind="assignment_counts", populations=request["populations"]
            )

        def _read_extension(self, extension, *, request):
            assert request["populations"] == ("assigned", "triggered")
            return [
                {
                    "population": "triggered",
                    "group_id": None,
                    "n_randomization_units": 2,
                    "n_units": 2,
                },
                {
                    "population": "triggered",
                    "group_id": "None",
                    "n_randomization_units": 3,
                    "n_units": 3,
                },
            ]

    assert ArtifactMomentSource.triggered_counts(Reader()) == (
        "unit",
        {"None": 3},
        {},
    )


def test_artifact_cluster_counts_skip_null_before_string_conversion():
    from increment.query.source import ArtifactMomentSource

    class Reader:
        context = SimpleNamespace(cluster="store")

        def _extension(self, request):
            assert request == {"kind": "cluster_identity", "cluster_name": "store"}
            return SimpleNamespace(kind="cluster_identity", cluster_name="store")

        def _read_extension(self, extension, *, request):
            return [
                {"unit_id": "null-group", "cluster_id": "c1"},
                {"unit_id": "literal-none-group", "cluster_id": "c2"},
            ]

        def _exposure_rows(self):
            return [
                {"unit_id": "null-group", "group_id": None},
                {"unit_id": "literal-none-group", "group_id": "None"},
            ]

    assert ArtifactMomentSource.cluster_counts(Reader()) == {"None": 1}


def _encouragement(frame):
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="uptake"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="uptake does not gate the outcome"
        ),
    )
    return Analysis.from_unit_summary(
        frame,
        unit="unit_id",
        group="group_id",
        metrics=[MetricSpec(name="rev", type="mean")],
        design=design,
        uptake="uptake",
    )


def test_encouragement_enumerates_itt_late_and_compliance_cells_with_compliance_evidence():
    results = _encouragement(_encouragement_frame()).run()
    assert isinstance(results, LiftEstimates)
    metadata = results.metadata
    assert metadata is not None
    assert all(row.failure_code is None for row in results)
    assert {"itt", "late", "compliance"} <= {cell.estimand for cell in metadata.scope.cells}
    kinds = {component["kind"] for component in results.source["components"]}
    assert {"moments_rows", "compliance_summary"} <= kinds
    restored = ReadoutResults.model_validate_json(results.model_dump_json())
    assert restored.metadata == metadata
    assert [CellKey.from_row(row) for row in restored] == [CellKey.from_row(row) for row in results]


def test_compliance_only_encouragement_readout_records_compliance_evidence():
    results = _encouragement(_encouragement_frame()).run(estimands=["compliance"])
    assert {cell.estimand for cell in results.metadata.scope.cells} == {"compliance"}
    assert "compliance_summary" in {c["kind"] for c in results.source["components"]}
    assert all(row.failure_code is None for row in results)


def _observational_frame(n=400, seed=5, drop_second_metric_in="T2"):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=3 * n)
    x2 = rng.normal(size=3 * n)
    group = np.repeat(["C", "T1", "T2"], n)
    rev = 1.0 + 0.1 * x1 + rng.normal(0, 0.3, 3 * n) + 0.1 * (group == "T1") + 0.1 * (group == "T2")
    other = 1.0 + 0.1 * x2 + rng.normal(0, 0.3, 3 * n)
    frame = pd.DataFrame(
        {"unit": range(3 * n), "arm": group, "x1": x1, "x2": x2, "rev": rev, "other": other}
    )
    if drop_second_metric_in is not None:
        frame.loc[frame["arm"] == drop_second_metric_in, "other"] = None
    return frame


def _observational(frame):
    design = Observational(control_group="C", adjustment=AdjustmentSet(covariates=("x1", "x2")))
    return Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        metrics=[
            MetricSpec(name="rev", type="mean"),
            MetricSpec(name="other", type="mean", missing="drop"),
        ],
        design=design,
    )


def test_observational_metric_lacking_an_arm_is_a_typed_failed_cell_beside_a_complete_metric():
    with pytest.warns(IncrementWarning) as captured:
        analysis = _observational(_observational_frame())
    assert warning_codes(captured) == ["frame.validation.metric_missing_drop"]
    assert warning_context(captured, "frame.validation.metric_missing_drop")["name"] == "other"
    results = analysis.run()
    assert results.metadata is not None
    failed = [row for row in results if row.failure_code is not None]
    assert {(row.metric, row.group_id, row.failure_code) for row in failed} == {
        ("other", "T2", "readout.cell.missing_arm")
    }
    assert all(row.lift is None for row in failed)
    healthy = {(row.metric, row.group_id) for row in results if row.failure_code is None}
    assert {("rev", "T1"), ("rev", "T2"), ("other", "T1")} <= healthy
    assert results.metadata.scope.decision_complete("assigned") is False
    assert all(row.decision_scope_complete is False for row in results)

    restored = ReadoutResults.model_validate_json(results.model_dump_json())
    assert restored.metadata == results.metadata
    assert [row.failure_code for row in restored] == [row.failure_code for row in results]


def test_observational_with_every_arm_observed_has_no_failed_cells():
    results = _observational(_observational_frame(drop_second_metric_in=None)).run()
    assert all(row.failure_code is None for row in results)
    assert {(cell.metric, cell.group_id) for cell in results.metadata.scope.cells} >= {
        ("rev", "T1"),
        ("rev", "T2"),
        ("other", "T1"),
        ("other", "T2"),
    }


def test_roster_omits_none_arm_ids_from_counts_and_observed_rows():
    from types import SimpleNamespace

    from increment.readouts._design_scope import resolve_roster

    source = SimpleNamespace(
        context=SimpleNamespace(cluster=None),
        assignment_counts=lambda *, population: {"control": 4, "treatment": 4, None: 1},
    )
    design = SimpleNamespace(allocation=None)

    arms, roster_source, complete, counts = resolve_roster(
        source,
        design,
        "assigned",
        {"metric": {"control", "treatment", None}},
    )

    assert arms == ("control", "treatment")
    assert roster_source == "experiment_counts"
    assert complete is True
    assert counts == {"control": 4, "treatment": 4}


def test_metric_row_inventory_omits_none_from_observed_arms():
    from types import SimpleNamespace

    from increment.readouts._metric_rows import _load_metric_rows

    source = SimpleNamespace(
        moments=lambda metric, *, grain, by: [
            {"group_id": None, "metric": metric.name},
            {"group_id": "control", "metric": metric.name},
            {"group_id": "treatment", "metric": metric.name},
        ]
    )
    metric = SimpleNamespace(name="metric", type="mean", winsorization=None)

    captured = _load_metric_rows(source, metric, by=(), control_group="control")

    assert captured.observed_arms == frozenset({"control", "treatment"})
    assert captured.evidence_count == 3
