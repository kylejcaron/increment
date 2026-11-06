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


# Each arm's (a, b) shifts from control. `b`'s T1 contrast has sd 0.5 in both arms and a 0.1
# shift at n = 150, so t = sqrt(3) and p is about 0.084; `a`'s contrasts have p near 0.
_FAMILY_SHIFTS = {"C": (0.0, 0.0), "T1": (2.0, 0.1), "T2": (3.0, 0.0), "T3": (0.0, 0.0)}


def _family_frame(arms, *, unobserved=None):
    """150 units in each of *arms*; *unobserved* maps a metric to the arms it has no values in."""
    unobserved = unobserved or {}
    rng = np.random.default_rng(17)
    n = 150
    frames = []
    for arm in arms:
        shift_a, shift_b = _FAMILY_SHIFTS[arm]
        noise = rng.normal(size=n)
        frames.append(
            pd.DataFrame(
                {
                    "unit": [f"{arm}{i}" for i in range(n)],
                    "arm": arm,
                    "x": rng.normal(size=n),
                    "a": np.nan
                    if arm in unobserved.get("a", ())
                    else 1.0 + shift_a + rng.normal(0, 0.5, n),
                    "b": np.nan
                    if arm in unobserved.get("b", ())
                    else 2.0 + shift_b + (noise - noise.mean()) / noise.std(ddof=1) * 0.5,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _family_design(mechanism):
    from increment.semantics.design import Randomized

    if mechanism == "randomized":
        return Randomized(control_group="C")
    return Observational(control_group="C", adjustment=AdjustmentSet(covariates=("x",)))


def _family_analysis(frame, design, plan=None, **options):
    from increment.semantics.models import AnalysisPlan

    return from_frame(
        frame,
        shape="summary",
        unit="unit",
        group="arm",
        metrics=[
            MetricSpec(name="a", type="mean", missing="skip"),
            MetricSpec(name="b", type="mean", missing="skip"),
        ],
        design=design,
        plan=AnalysisPlan(secondaries=["a", "b"], q=0.1) if plan is None else plan,
        **options,
    )


def _family_readout(experiment, reader, scope=None):
    from increment import Method

    return view_readout(experiment, reader, scope, decision_method=Method(name="unadjusted"))


def _secondary_family_readout(design, *, unobserved=None):
    with pytest.warns(IncrementWarning):
        analysis = _family_analysis(
            _family_frame(("C", "T1", "T2", "T3"), unobserved=unobserved), design
        )
    return _family_readout(*analysis)


def _late_arm_readout(mechanism, plan=None):
    """The family's first moments read of each metric holds C and T1; the counts, and every
    later read, hold the late arm T2 too."""
    from tests.analysis_factory import drifting_analysis

    early = _family_analysis(_family_frame(("C", "T1")), _family_design(mechanism), plan)
    late = _family_analysis(_family_frame(("C", "T1", "T2")), _family_design(mechanism), plan)
    (experiment, reader), _source = drifting_analysis(early, late)
    try:
        return _family_readout(experiment, reader)
    finally:
        for _, analysis_reader in (early, late):
            analysis_reader.close()


def test_observational_family_counts_every_scoped_member():
    results = _secondary_family_readout(
        _family_design("observational"), unobserved={"a": ("T3",), "b": ("T2", "T3")}
    )
    (family,) = results.metadata.scope.families
    assert len(family.members) == 6
    rows = {(row.metric, row.group_id): row for row in results}
    # BH at q = 0.1 over six members, the four unobserved ones at p = 1: the third-smallest p,
    # b in T1, is above 3 * 0.1 / 6, so only the two `a` contrasts are discoveries.
    assert rows[("a", "T1")].family_threshold == pytest.approx(2 * 0.1 / len(family.members))
    assert rows[("b", "T1")].discovery is False
    assert {rows[cell].family_size for cell in (("a", "T1"), ("a", "T2"), ("b", "T1"))} == {
        len(family.members)
    }
    # Only the selected cells are reissued at the FCR level, from the nominal 0.05.
    assert rows[("a", "T1")].family_nominal_alpha == 0.05
    assert rows[("b", "T1")].family_nominal_alpha is None
    randomized = _secondary_family_readout(
        _family_design("randomized"), unobserved={"a": ("T3",), "b": ("T2", "T3")}
    )
    assert next(r for r in randomized if (r.metric, r.group_id) == ("b", "T1")).discovery is False


def test_observational_family_keeps_a_member_without_control_values():
    """`b` has no control values, so each of its three cells is a p = 1 member, as under a
    randomized design: BH at q = 0.1 over six members selects the two `a` contrasts."""
    by_mechanism = {}
    for mechanism in ("observational", "randomized"):
        results = _secondary_family_readout(
            _family_design(mechanism), unobserved={"a": ("T3",), "b": ("C",)}
        )
        (family,) = results.metadata.scope.families
        assert len(family.members) == 6
        rows = {(row.metric, row.group_id): row for row in results}
        for arm in ("T1", "T2", "T3"):
            assert rows[("b", arm)].failure_code == "readout.cell.missing_metric_observations"
            assert (rows[("b", arm)].failure_context or {})["unobserved_arm"] == "C"
        by_mechanism[mechanism] = {
            (cell, rows[cell].family_size, rows[cell].family_threshold, rows[cell].discovery)
            for cell in (("a", "T1"), ("a", "T2"))
        }
    assert by_mechanism["observational"] == by_mechanism["randomized"]
    assert {size for _cell, size, _thr, _disc in by_mechanism["observational"]} == {6}
    for _cell, _size, threshold, discovery in by_mechanism["observational"]:
        assert threshold == pytest.approx(2 * 0.1 / 6)
        assert discovery is True


def test_randomized_family_counts_a_late_arm_as_a_non_rejection():
    results = _late_arm_readout("randomized")
    (family,) = results.metadata.scope.families
    assert len(family.members) == 4
    rows = {(row.metric, row.group_id): row for row in results}
    for metric in ("a", "b"):
        assert rows[(metric, "T2")].failure_code == "readout.cell.missing_metric_observations"
    # BH at q = 0.1 over {~0, 0.084, 1, 1}: at k = 2 the cutoff 2 * 0.1 / 4 is below 0.084, so
    # one cell is selected and the threshold is 1 * 0.1 / 4.
    for cell in (("a", "T1"), ("b", "T1")):
        assert rows[cell].family_size == 4
        assert rows[cell].family_threshold == pytest.approx(0.1 * 1 / 4)
    assert rows[("a", "T1")].discovery is True
    assert rows[("b", "T1")].discovery is False


@pytest.mark.parametrize("mechanism", ["randomized", "observational"])
def test_primary_alpha_split_counts_a_late_arm(mechanism):
    from increment.semantics.models import AnalysisPlan

    results = _late_arm_readout(mechanism, AnalysisPlan(primary="a", secondaries=["b"], q=0.1))
    (primary,) = [family for family in results.metadata.scope.families if family.name == "primary"]
    assert {cell.group_id for cell in primary.members} == {"T1", "T2"}
    (row,) = [row for row in results if (row.metric, row.group_id) == ("a", "T1")]
    # One primary split across its two counted treatment arms: 0.05 / 2.
    assert row.require_lift().alpha == pytest.approx(0.05 / 2)


def test_primary_alpha_split_counts_an_arm_that_lands_after_the_counts():
    """A randomized run reads its counts before its moments. T2 lands between the two reads, so
    only the moments read holds it; the split still counts every arm the primary's scope lists."""
    from increment.semantics.design import Randomized
    from increment.semantics.models import AnalysisPlan
    from tests.analysis_factory import drifting_analysis

    design = Randomized(control_group="C", allocation={"C": 0.34, "T1": 0.33, "T2": 0.33})
    counted = _family_analysis(_family_frame(("C", "T1")), design, AnalysisPlan(primary="a"))
    landed = _family_analysis(_family_frame(("C", "T1", "T2")), design, AnalysisPlan(primary="a"))
    # The counts are counted's; each metric's first moments read is landed's.
    (experiment, reader), _source = drifting_analysis(landed, counted)
    try:
        results = _family_readout(experiment, reader)
    finally:
        for _, analysis_reader in (counted, landed):
            analysis_reader.close()
    (primary,) = [family for family in results.metadata.scope.families if family.name == "primary"]
    assert {cell.group_id for cell in primary.members} == {"T1", "T2"}
    for arm in ("T1", "T2"):
        (row,) = [row for row in results if (row.metric, row.group_id) == ("a", arm)]
        assert row.require_lift().alpha == pytest.approx(0.05 / 2)


@pytest.mark.parametrize("mechanism", ["randomized", "observational"])
def test_primary_alpha_split_without_counts_follows_its_own_arms(mechanism, tmp_path):
    """Without enrollment counts a primary's alpha is split across the arms its own moments
    read: `b` reads T2 too, but `a` has no T2 values, so `a`'s T1 keeps the whole 0.05."""
    import pyarrow.parquet as pq

    from increment import from_moments
    from increment.semantics.models import AnalysisPlan
    from increment.workflows import export_moments

    plan = AnalysisPlan(primary="a", secondaries=["b"], q=0.1)
    frame = _family_frame(("C", "T1", "T2"), unobserved={"a": ("T2",)})
    with pytest.warns(IncrementWarning):
        experiment, reader = _family_analysis(frame, _family_design(mechanism), plan)
    path = tmp_path / "moments.parquet"
    try:
        export_moments(experiment, reader, path)
    finally:
        reader.close()
    rows = [
        {key: value for key, value in row.items() if key != "assignment_counts"}
        for row in pq.read_table(path).to_pylist()
    ]
    experiment, reader = from_moments(
        rows,
        metrics=[
            MetricSpec(name="a", type="mean", missing="skip"),
            MetricSpec(name="b", type="mean", missing="skip"),
        ],
        design=_family_design(mechanism),
        plan=plan,
    )
    results = _family_readout(experiment, reader)
    (row,) = [row for row in results if (row.metric, row.group_id) == ("a", "T1")]
    assert row.require_lift().alpha == pytest.approx(0.05)


@pytest.mark.parametrize("mechanism", ["randomized", "observational"])
def test_constant_family_member_keeps_its_reason_and_its_slot(mechanism):
    """`b` is constant in every arm, so neither scale of its contrasts has an uncertainty: each
    `b` cell is refused, an unavailable row carrying the lift guard, and a p = 1 member. BH at
    q = 0.1 over {~0, ~0, 1, 1} selects the two `a` contrasts at the threshold 2 * 0.1 / 4."""
    frame = _family_frame(("C", "T1", "T2")).assign(b=2.0)
    with pytest.warns(IncrementWarning) as captured:
        results = _family_readout(*_family_analysis(frame, _family_design(mechanism)))
    assert set(warning_codes(captured)) == {"readouts.run.cell_refused"}
    (family,) = results.metadata.scope.families
    assert len(family.members) == 4
    rows = {(row.metric, row.group_id): row for row in results}
    for arm in ("T1", "T2"):
        assert rows[("b", arm)].failure_code == "estimation.engine.lift_guard"
        assert (rows[("b", arm)].failure_context or {})["reason"] == "zero_variance"
        assert rows[("a", arm)].family_size == len(family.members)
        assert rows[("a", arm)].family_threshold == pytest.approx(2 * 0.1 / 4)
        assert rows[("a", arm)].discovery is True


def test_never_enrolled_arm_leaves_the_family_scope():
    """A declared arm without an enrolled unit is a count exclusion, not a hypothesis: its cells
    stay unavailable rows, outside the family, and the family is the two T1 contrasts."""
    from increment.semantics.design import Randomized

    design = Randomized(control_group="C", allocation={"C": 0.34, "T1": 0.33, "T2": 0.33})
    results = _family_readout(*_family_analysis(_family_frame(("C", "T1")), design))
    (family,) = results.metadata.scope.families
    assert {(cell.metric, cell.group_id) for cell in family.members} == {("a", "T1"), ("b", "T1")}
    rows = {(row.metric, row.group_id): row for row in results}
    for metric in ("a", "b"):
        assert rows[(metric, "T2")].failure_code == "readout.cell.missing_arm"
        assert rows[(metric, "T2")].family_id is None
        # BH at q = 0.1 over {~0, 0.084}: both are selected.
        assert rows[(metric, "T1")].family_size == len(family.members)
        assert rows[(metric, "T1")].family_threshold == pytest.approx(0.1 * 2 / 2)


def test_breakout_family_counts_a_late_arm_per_segment():
    """The late arm T2 has enrolled in segment A only: its segment B cells are count
    exclusions, and the family is every other segment's cells, as many as its size."""
    from increment import View

    frame = pd.concat(
        [
            _family_frame(arms).assign(seg=seg, unit=lambda f, seg=seg: seg + f["unit"])
            for seg, arms in (("A", ("C", "T1", "T2")), ("B", ("C", "T1")))
        ],
        ignore_index=True,
    )
    experiment, reader = _family_analysis(frame, _family_design("randomized"), breakouts=["seg"])
    with pytest.warns(IncrementWarning) as captured:
        results = _family_readout(experiment, reader, View(view="breakout", grain="total"))
    assert set(warning_codes(captured)) == {"breakout.estimates.no_row_for_group_metric"}
    (family,) = results.metadata.scope.families
    assert {(cell.metric, cell.group_id, cell.dimension_value) for cell in family.members} == {
        (metric, arm, seg)
        for metric in ("a", "b")
        for arm, seg in (("T1", "A"), ("T2", "A"), ("T1", "B"))
    }
    rows = {(row.metric, row.group_id, row.dimension_value): row for row in results}
    for cell in family.members:
        key = (cell.metric, cell.group_id, cell.dimension_value)
        assert rows[key].family_size == len(family.members)
    for metric in ("a", "b"):
        excluded = rows[(metric, "T2", "B")]
        assert (excluded.excluded, excluded.family_id, excluded.family_size) == (
            "no_treatment_arm",
            None,
            None,
        )


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
