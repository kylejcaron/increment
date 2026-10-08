"""Consumer checks stay effective under bounded semantic regressions."""

from dataclasses import replace

import pandas as pd
import pytest

from increment import Analysis, AnalysisPlan, MetricSpec
from increment.breakout.estimates import LiftEstimates
from increment.errors import WireFormatError
from increment.estimation.decision_types import PValueEvidence
from increment.estimation.inference import Normal
from increment.estimation.readout_types import CellKey
from increment.semantics.design import Randomized
from tests.readout_journeys import (
    assert_failed_assigned_integrity,
    assert_filtered_family_scope_unchanged,
    assert_prior_free_sampling_family,
)
from tests.test_readout_public_scope import _analysis, _frame


def _failed_cell_readout() -> LiftEstimates:
    with pytest.warns():
        return _analysis(_frame()).run()


def _assert_required_failed_cell_is_visible(results: LiftEstimates) -> None:
    assert "estimation.engine.lift_guard" in {
        row.failure_code for row in results if row.failure_code is not None
    }
    assert any(row.metric == "guard" and row.lift is None for row in results)


def test_journey_three_rejects_a_missing_required_failed_cell():
    results = _failed_cell_readout()
    _assert_required_failed_cell_is_visible(results)

    with pytest.raises(WireFormatError) as raised:
        LiftEstimates(
            [row for row in results if row.failure_code is None], metadata=results.metadata
        )
    assert raised.value.code == "readout.serialization.identity_mismatch"
    assert raised.value.context["reason"] == "collection rows do not cover complete scope"


def _integrity_readout() -> LiftEstimates:
    frame = pd.DataFrame(
        [
            {
                "unit": f"{group[0]}{index}",
                "arm": group,
                "ds": pd.Timestamp("2025-01-01"),
                "rev": float(index % 7 + 1),
            }
            for group, count in (("control", 900), ("treatment", 100))
            for index in range(count)
        ]
    )
    results = Analysis.from_unit_panel(
        frame,
        unit="unit",
        group="arm",
        date="ds",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics=[MetricSpec(name="rev", type="mean")],
    ).run()
    assert isinstance(results, LiftEstimates)
    return results


def test_journey_six_rejects_hidden_failed_assignment_integrity():
    results = _integrity_readout()
    assert_failed_assigned_integrity(results)
    assert results.metadata is not None

    sources = {
        source_id: source.model_copy(update={"integrity": ()})
        for source_id, source in results.metadata.scope.by_source.items()
    }
    scope = results.metadata.scope.model_copy(update={"by_source": sources})
    metadata = results.metadata.model_copy(update={"scope": scope})
    perturbed = LiftEstimates(results, metadata=metadata)

    with pytest.raises(AssertionError):
        assert_failed_assigned_integrity(perturbed)


def _prior_readout() -> LiftEstimates:
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
    return results


def test_journey_four_rejects_prior_sensitive_evidence_in_sampling_family():
    results = _prior_readout()
    assert_prior_free_sampling_family(results)
    row = results[0]
    assert row.posterior_prob_favorable is not None
    posterior_probability = float(row.posterior_prob_favorable)
    p_value = row.p_value()
    assert p_value is not None
    assert abs(p_value - (1.0 - posterior_probability)) > 1e-6
    assert results.metadata is not None

    cell = CellKey.from_row(row)
    records = []
    for record in results.metadata.cells:
        if record.cell == cell:
            assert record.sampling is not None
            evidence = record.sampling.evidence
            assert isinstance(evidence, PValueEvidence)
            sampling = record.sampling.model_copy(
                update={"evidence": replace(evidence, p_value=1.0 - posterior_probability)}
            )
            record = record.model_copy(update={"sampling": sampling})
        records.append(record)
    metadata = results.metadata.model_copy(update={"cells": tuple(records)})
    perturbed = LiftEstimates(results, metadata=metadata)

    with pytest.raises(AssertionError):
        assert_prior_free_sampling_family(perturbed)


def test_journey_three_rejects_filtering_that_shrinks_family_scope():
    original = _analysis(_frame(guard_constant=False)).run()
    filtered = original.filter(lambda row: row.metric == "rev")
    assert_filtered_family_scope_unchanged(original, filtered)
    assert filtered.metadata is not None

    visible = {CellKey.from_row(row) for row in filtered}
    reduced_families = tuple(
        family.model_copy(
            update={"members": tuple(cell for cell in family.members if cell in visible)}
        )
        for family in filtered.metadata.scope.families
    )
    scope = filtered.metadata.scope.model_copy(update={"families": reduced_families})
    metadata = filtered.metadata.model_copy(update={"scope": scope})
    perturbed = LiftEstimates(filtered, metadata=metadata)

    with pytest.raises(AssertionError):
        assert_filtered_family_scope_unchanged(original, perturbed)
