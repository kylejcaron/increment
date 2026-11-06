"""Readouts enumerate their expected cells before estimation and return each as a row."""

import narwhals as nw
import numpy as np
import pandas as pd
import pytest

from increment import MetricSpec, from_frame, run, runnable
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
from tests.analysis_factory import view_readout
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
            justification="uptake does not gate the outcome"
        ),
    )
    return from_frame(
        frame,
        shape="summary",
        unit="unit_id",
        group="group_id",
        metrics=[MetricSpec(name="rev", type="mean")],
        design=design,
        uptake="uptake",
    )


def test_encouragement_enumerates_itt_late_and_compliance_cells_with_compliance_evidence():
    results = view_readout(*_encouragement(_encouragement_frame()))
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
            justification="uptake does not gate the outcome"
        ),
    )
    experiment, resolver = from_frame(
        frame,
        shape="summary",
        unit="unit_id",
        group="group_id",
        metrics=[MetricSpec(name="rev", type="mean"), MetricSpec(name="other", type="mean")],
        design=design,
        uptake="uptake",
        plan=AnalysisPlan(primary="rev", secondaries=["other"]),
    )
    results = view_readout(experiment, resolver)
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
    results = view_readout(*_encouragement(_encouragement_frame()), estimands=["compliance"])
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


_OBSERVATIONAL_METRICS = (
    MetricSpec(name="rev", type="mean"),
    MetricSpec(name="other", type="mean", missing="skip"),
)


def _observational(frame, metrics=_OBSERVATIONAL_METRICS):
    design = Observational(control_group="C", adjustment=AdjustmentSet(covariates=("x1", "x2")))
    return from_frame(
        frame,
        shape="summary",
        unit="unit",
        group="arm",
        metrics=list(metrics),
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
    results = view_readout(*analysis, **({"prior": prior} if with_prior else {}))

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
    # Missing 0/400 C, 0/400 T1 and 400/400 T2 values of `other`: chi-square 1200 on 2 df.
    with pytest.warns(IncrementWarning) as captured:
        experiment, resolver = _observational(_observational_frame())
    assert warning_codes(captured) == ["frame.validation.metric_missing_differs_by_arm"]
    assert warning_context(captured, "frame.validation.metric_missing_differs_by_arm")[
        "metric"
    ] == ("other")
    results = view_readout(experiment, resolver)
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
    results = view_readout(*_observational(_observational_frame(drop_second_metric_in=None)))
    assert all(row.failure_code is None for row in results)
    assert {(cell.metric, cell.group_id) for cell in results.metadata.scope.cells} >= {
        ("rev", "T1"),
        ("rev", "T2"),
        ("other", "T1"),
        ("other", "T2"),
    }


def test_observational_run_with_every_cell_present_is_scope_complete():
    results = view_readout(*_observational(_observational_frame(drop_second_metric_in=None)))
    assert {row.decision_scope_complete for row in results} == {True}
    assert {row.decision_scope_reason_code for row in results} == {None}
    (source_scope,) = results.metadata.scope.by_source.values()
    (roster,) = source_scope.rosters
    assert roster.source == "experiment_counts"
    # The counts name the arms only; assignment integrity stays unchecked.
    (integrity,) = source_scope.integrity
    assert integrity.status == "not_applicable"
    assert integrity.observed is None


def test_observational_counted_arm_without_outcome_is_incomplete():
    other = MetricSpec(name="other", type="mean", missing="skip")
    with pytest.warns(IncrementWarning):
        experiment, reader = _observational(_observational_frame(), metrics=(other,))
    results = view_readout(experiment, reader)
    assert {row.decision_scope_complete for row in results} == {False}
    missing = {
        (cell["metric"], cell["group_id"])
        for row in results
        for cell in row.decision_scope_reason_context["missing_cells"]
    }
    assert missing == {("other", "T2")}


def _secondary_family_readout(design):
    """`a` has no values in T3 and `b` none in T2 or T3. `b`'s T1 contrast has sd 0.5 in both
    arms and a 0.1 shift at n = 150, so t = sqrt(3) and p is about 0.084."""
    from increment import Method
    from increment.semantics.models import AnalysisPlan

    rng = np.random.default_rng(17)
    n = 150
    frames = []
    for arm, shift_a, shift_b in (
        ("C", 0.0, 0.0),
        ("T1", 2.0, 0.1),
        ("T2", 3.0, 0.0),
        ("T3", 0.0, 0.0),
    ):
        noise = rng.normal(size=n)
        frames.append(
            pd.DataFrame(
                {
                    "unit": [f"{arm}{i}" for i in range(n)],
                    "arm": arm,
                    "x": rng.normal(size=n),
                    "a": np.nan if arm == "T3" else 1.0 + shift_a + rng.normal(0, 0.5, n),
                    "b": np.nan
                    if arm in ("T2", "T3")
                    else 2.0 + shift_b + (noise - noise.mean()) / noise.std(ddof=1) * 0.5,
                }
            )
        )
    with pytest.warns(IncrementWarning):
        experiment, reader = from_frame(
            pd.concat(frames, ignore_index=True),
            shape="summary",
            unit="unit",
            group="arm",
            metrics=[
                MetricSpec(name="a", type="mean", missing="skip"),
                MetricSpec(name="b", type="mean", missing="skip"),
            ],
            design=design,
            plan=AnalysisPlan(secondaries=["a", "b"], q=0.1),
        )
    return view_readout(experiment, reader, decision_method=Method(name="unadjusted"))


def test_observational_family_counts_every_scoped_member():
    from increment.semantics.design import Randomized

    results = _secondary_family_readout(
        Observational(control_group="C", adjustment=AdjustmentSet(covariates=("x",)))
    )
    (family,) = results.metadata.scope.families
    assert len(family.members) == 6
    rows = {(row.metric, row.group_id): row for row in results}
    # BH at q = 0.1 over six members, the four unobserved ones at p = 1: the third-smallest p,
    # b in T1, is above 3 * 0.1 / 6, so only the two `a` contrasts are discoveries.
    assert rows[("a", "T1")].family_threshold == pytest.approx(2 * 0.1 / len(family.members))
    assert rows[("b", "T1")].discovery is False
    randomized = _secondary_family_readout(Randomized(control_group="C"))
    assert next(r for r in randomized if (r.metric, r.group_id) == ("b", "T1")).discovery is False


def test_single_refused_metric_returns_unavailable_row():
    code = "estimation.variance.ratio_moments_nonpositive_denominator_mean"
    frame = pd.DataFrame(
        {
            "unit": [f"{arm}{i}" for arm in ("control", "treatment") for i in range(4)],
            "arm": ["control"] * 4 + ["treatment"] * 4,
            "revenue": [1.0, 2.0, 3.0, 4.0] * 2,
            "sessions": [0.0] * 4 + [1.0] * 4,
        }
    )
    experiment, reader = from_frame(
        frame,
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics=[MetricSpec(name="r", type="ratio", numerator="revenue", denominator="sessions")],
    )
    with pytest.warns(IncrementWarning):
        result = run(experiment, reader)
    (row,) = result.readout()
    assert isinstance(row, LiftEstimate)
    assert (row.metric, row.group_id, row.failure_code) == ("r", "treatment", code)
    assert row.stat_sig() is None
    projected = nw.from_native(result.to_frame(), eager_only=True)
    assert projected.select("metric", "group_id", "failure_code").rows() == [
        ("r", "treatment", code)
    ]
    static_codes = {
        refusal.code
        for status in runnable(experiment, "r").values()
        for refusal in status.static_refusals
    }
    assert "readout.estimate_lift_every" not in static_codes
