"""Observational and encouragement readouts enumerate their expected cells before estimation."""

import numpy as np
import pandas as pd
import pytest

from increment import Analysis, MetricSpec
from increment.breakout.estimates import LiftEstimates
from increment.errors import IncrementWarning
from increment.estimation.readout_types import CellKey, ReadoutResults
from increment.estimation.results import LiftEstimate
from increment.semantics.design import (
    AdjustmentSet,
    Encouragement,
    ExclusionRestriction,
    Observational,
    UptakeSpec,
)
from tests.warning_codes import warning_codes, warning_context


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
    assert {row.multiplicity_status for row in results} == {"undeclared_plan"}
    kinds = {component["kind"] for component in results.source["components"]}
    assert {"moments_rows", "compliance_summary"} <= kinds
    restored = ReadoutResults.model_validate_json(results.model_dump_json())
    assert restored.metadata == metadata
    assert [CellKey.from_row(row) for row in restored] == [CellKey.from_row(row) for row in results]


def test_encouragement_scope_matches_itt_only_family_selection():
    from increment.semantics.models import AnalysisPlan

    frame = _encouragement_frame()
    frame["other"] = frame["rev"] + np.linspace(-0.1, 0.1, len(frame))
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="uptake"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="uptake does not gate the outcome"
        ),
    )
    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit_id",
        group="group_id",
        metrics=[MetricSpec(name="rev", type="mean"), MetricSpec(name="other", type="mean")],
        design=design,
        uptake="uptake",
        plan=AnalysisPlan(primary="rev", secondaries=["other"]),
    )
    results = analysis.run()
    assert results.metadata is not None
    family = next(
        family for family in results.metadata.scope.families if family.name == "secondary"
    )
    assert {(cell.metric, cell.estimand) for cell in family.members} == {("other", "itt")}
    itt = next(row for row in results if row.metric == "other" and row.estimand == "itt")
    late = next(row for row in results if row.metric == "other" and row.estimand == "late")
    assert isinstance(itt, LiftEstimate)
    assert isinstance(late, LiftEstimate)
    assert itt.role == late.role == "secondary"
    assert itt.family_id == family.family_id
    assert late.family_id is None


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


@pytest.mark.parametrize("mechanism", ["observational", "encouragement"])
@pytest.mark.parametrize("with_prior", [False, True])
def test_design_scoped_cell_records_retain_declared_posterior(mechanism, with_prior):
    from increment.estimation.inference import Normal

    if mechanism == "observational":
        analysis = _observational(_observational_frame(drop_second_metric_in=None))
    else:
        analysis = _encouragement(_encouragement_frame())
    prior = Normal(mu=0.0, sigma=0.1) if with_prior else None
    results = analysis.run(**({"prior": prior} if with_prior else {}))

    rows_by_cell = {CellKey.from_row(row): row for row in results}
    records = results.metadata.cells
    assert records
    for record in records:
        if record.cell.metric == "uptake":
            assert record.posterior is None
            continue
        if prior is None:
            assert record.posterior is None
            continue
        row = rows_by_cell[record.cell]
        assert record.posterior is not None
        assert record.posterior.prior == prior
        assert record.posterior.available == row.posterior_available
        assert record.posterior.model == row.posterior_model
        assert record.posterior.scale == row.posterior_scale
        assert record.posterior.reason_code == row.posterior_reason_code
        assert record.posterior.reason_context == row.posterior_reason_context


def test_observational_metric_lacking_an_arm_is_a_typed_failed_cell_beside_a_complete_metric():
    with pytest.warns(IncrementWarning) as captured:
        analysis = _observational(_observational_frame())
    assert warning_codes(captured) == ["frame.validation.metric_missing_drop"]
    assert warning_context(captured, "frame.validation.metric_missing_drop")["name"] == "other"
    results = analysis.run()
    assert results.metadata is not None
    assert {row.multiplicity_status for row in results} == {"undeclared_plan"}
    failed = [row for row in results if row.failure_code is not None]
    assert {(row.metric, row.group_id, row.failure_code) for row in failed} == {
        ("other", "T2", "readout.cell.missing_metric_observations")
    }
    assert all(row.lift is None for row in failed)
    healthy = {(row.metric, row.group_id) for row in results if row.failure_code is None}
    assert {("rev", "T1"), ("rev", "T2"), ("other", "T1")} <= healthy
    assert results.metadata.scope.decision_complete("assigned") is False
    assert all(row.decision_scope_complete is False for row in results)
    copied_failure = failed[0].model_copy()
    with pytest.raises(TypeError):
        copied_failure.decision_scope_reason_context["missing_cells"][0]["metric"] = "changed"

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
