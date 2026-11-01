"""Public Analysis.run readouts keep required-cell failures and completeness."""

import copy
import pickle
import subprocess
import sys
from datetime import date
from typing import Any, cast

import narwhals as nw
import numpy as np
import pandas as pd
import pytest

from increment import Analysis, AnalysisPlan, CupedCovariate, Method, MetricSpec
from increment.breakout.estimates import (
    BreakoutEstimate,
    DailyLiftEstimate,
    DailyLiftEstimates,
    DailyMetricValue,
    DailyMetricValues,
    LiftEstimates,
)
from increment.errors import IncrementWarning, InvalidRequestError, UnsupportedRequestError
from increment.estimation.readout_types import CellKey, ReadoutResults
from increment.estimation.results import Estimate, LiftEstimate
from increment.semantics.design import Randomized
from increment.tables import readout_table
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


def _analysis(frame, *, drop_missing_guard=False, design=None):
    return Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        control=None if design is not None else "control",
        design=design,
        metrics=[
            MetricSpec(name="rev", type="mean", preferred_direction="increase"),
            MetricSpec(
                name="guard",
                type="mean",
                preferred_direction="decrease",
                missing="drop" if drop_missing_guard else "error",
            ),
            MetricSpec(name="secondary", type="mean", preferred_direction="increase"),
        ]
        if "secondary" in frame
        else [
            MetricSpec(name="rev", type="mean", preferred_direction="increase"),
            MetricSpec(
                name="guard",
                type="mean",
                preferred_direction="decrease",
                missing="drop" if drop_missing_guard else "error",
            ),
        ],
        plan=(
            AnalysisPlan(primary="rev", guardrails=["guard"])
            if "secondary" not in frame
            else AnalysisPlan(primary="rev", guardrails=["guard"], secondaries=["secondary"])
        ),
    )


def test_default_repr_qualifies_comparisons_families_and_interval_meaning():
    frame = _frame(guard_constant=False)
    frame["secondary"] = frame["rev"] * 0.8
    results = _analysis(
        frame,
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
        ),
    ).run()
    rendered = repr(results)
    assert "treatment vs control" in rendered
    assert "control identity not retained" not in rendered
    assert rendered.index("rev (primary") < rendered.index("guard (guardrail")
    assert rendered.index("guard (guardrail") < rendered.index("secondary (secondary")
    assert rendered.count("Multiplicity (primary)") == 1
    assert rendered.count("Multiplicity (secondary)") == 1
    assert "family" in rendered
    assert "does not reject; inconclusive" in rendered
    assert "direction supported" in rendered
    assert "not evidence of no effect" in rendered
    assert any(line.startswith("secondary selection:") for line in rendered.splitlines())


def test_readout_table_uses_qualification_bullets_and_keeps_interpretation_per_row():
    results = _analysis(_frame(guard_constant=False)).run(metrics=["rev"])
    html = readout_table(results).gt().as_raw_html()
    subtitle = html.split('class="gt_heading gt_subtitle', 1)[1].split("</td>", 1)[0]
    assert "<ul " in subtitle
    assert subtitle.count("<li>") == 6
    assert "Coherence: captured." in subtitle
    assert "Effective evidence modes:" in subtitle
    assert "Integrity: not_checked" in subtitle
    assert subtitle.count("sha256:") == 1
    assert "Interpretation:" not in subtitle and "Interpretation:" not in html
    assert "Interval interpretation" in html
    assert 'th[id="Interval-interpretation"]{width:18em' in html
    assert "Excludes zero — increase established" in html
    assert results[0].source_snapshot_id not in subtitle


def test_row_only_table_discloses_unknown_scope_and_integrity():
    from increment.tables import estimates_to_readout

    results = _analysis(_frame(guard_constant=False)).run(metrics=["rev"])
    # A received row carries no collection evidence; the table must not look vouched for.
    html = readout_table(estimates_to_readout([results[0]])).gt().as_raw_html()
    subtitle = html.split('class="gt_heading gt_subtitle', 1)[1].split("</td>", 1)[0]
    assert "Scope and integrity unknown." in subtitle
    scoped = readout_table(results).gt().as_raw_html()
    assert "Scope and integrity unknown." not in scoped


def test_readout_table_escapes_user_text_in_qualification_list():
    frame = _frame(guard_constant=False)
    frame["arm"] = frame["arm"].replace("treatment", "<img src=x>")
    results = _analysis(frame).run(metrics=["rev"])
    html = readout_table(results).gt().as_raw_html()
    subtitle = html.split('class="gt_heading gt_subtitle', 1)[1].split("</td>", 1)[0]

    assert "<img" not in subtitle
    assert "&lt;img src=x&gt;" in subtitle


def test_readout_table_preserves_title_and_subtitle_without_notes():
    table = readout_table(
        [
            {
                "metric": "revenue",
                "method": "unadjusted",
                "lift": 0.1,
                "lower": 0.0,
                "higher": 0.2,
                "stat_sig": False,
                "group_id": "treatment",
            }
        ],
        title="Experiment A",
        subtitle="Week 1",
    )

    html = table.gt().as_raw_html()

    assert "Experiment A" in html
    assert "Week 1" in html


def test_absolute_margin_repr_matches_exported_additive_interpretation():
    from increment.semantics.models import ExperimentMetric
    from increment.tables import estimates_to_readout

    records = []
    for arm in ("control", "treatment"):
        for index in range(200):
            value = 9.0 if index % 2 == 0 else 11.0
            records.append(
                {
                    "unit": f"{arm}-{index}",
                    "arm": arm,
                    "revenue": value,
                    "latency": value,
                }
            )
    analysis = Analysis.from_unit_summary(
        pd.DataFrame(records),
        unit="unit",
        group="arm",
        control="control",
        metrics=[
            MetricSpec(name="revenue", type="mean", preferred_direction="increase"),
            MetricSpec(name="latency", type="mean", preferred_direction="decrease"),
        ],
        plan=AnalysisPlan(
            primary="revenue",
            guardrails=[ExperimentMetric(metric="latency", margin_abs=0.5)],
        ),
    )
    with analysis:
        results = analysis.run()

    latency = next(row for row in estimates_to_readout(results) if row["metric"] == "latency")
    rendered = repr(results)

    assert "effect direction not established" in latency["effect_interpretation"]
    assert "effect direction supported" not in latency["effect_interpretation"]
    exported = latency["effect_interpretation"].removeprefix("Interpretation: ")
    assert f"latency interpretation: {exported}" in rendered.splitlines()


def test_absolute_margin_uses_additive_interval_not_relative_interval():
    from increment.tables import estimates_to_readout

    estimate = LiftEstimate(
        metric="revenue",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        lift=Estimate(value=0.2, lb=0.1, ub=0.3, level=0.95),
        null_lift=0.0,
        null_abs=-0.01,
        abs_diff=-0.5,
        abs_lb=-0.66,
        abs_ub=-0.34,
    )
    (row,) = estimates_to_readout([estimate])

    assert "direction supported (decrease)" in row["effect_interpretation"]
    assert "increase supported" not in row["effect_interpretation"]


def test_margin_success_below_zero_does_not_claim_increase():
    from increment.tables import estimates_to_readout

    estimate = LiftEstimate(
        metric="latency",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        lift=Estimate(value=-0.03, lb=-0.04, ub=-0.02, level=0.95),
        null_lift=0.0,
        null_abs=-0.05,
        abs_diff=-0.03,
        abs_lb=-0.04,
        abs_ub=-0.02,
        alternative="greater",
    )
    (row,) = estimates_to_readout([estimate])

    assert "rejects the null" in row["effect_interpretation"]
    assert "margin met" in row["effect_interpretation"]
    assert "effect direction supported (decrease)" in row["effect_interpretation"]
    assert "increase" not in row["effect_interpretation"]


def test_absolute_margin_without_additive_interval_does_not_fall_back_to_relative():
    from increment.tables import estimates_to_readout

    estimate = LiftEstimate(
        metric="revenue",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        lift=Estimate(value=0.2, lb=0.1, ub=0.3, level=0.95),
        null_lift=0.0,
        null_abs=-0.01,
    )
    (row,) = estimates_to_readout([estimate])

    assert "unavailable" in row["effect_interpretation"].lower()
    assert "increase supported" not in row["effect_interpretation"]


def test_estimate_frames_preserve_unknown_nullable_evidence_and_daily_values():
    estimate = LiftEstimate(
        metric="revenue",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        lift=Estimate(value=0.2, lb=0.1, ub=0.3, level=0.95),
    )
    unscoped = cast(pd.DataFrame, LiftEstimates([estimate]).to_frame(backend="pandas"))
    assert str(unscoped["decision_scope_complete"].dtype) == "boolean"
    assert unscoped["decision_scope_complete"].isna().all()
    assert str(unscoped["posterior_available"].dtype) == "boolean"
    assert unscoped["posterior_available"].isna().all()

    daily = cast(
        pd.DataFrame,
        DailyMetricValues(
            [
                DailyMetricValue(
                    ds=date(2024, 1, 1),
                    metric="revenue",
                    group_id="treatment",
                    value=Estimate(value=10, lb=9, ub=11, level=0.95),
                    n=100,
                )
            ]
        ).to_frame(backend="pandas"),
    )
    assert daily["point_available"].tolist() == [True]
    assert daily["sampling_available"].tolist() == [True]
    assert daily["unavailable_cell_count"].tolist() == [0]


def test_daily_mean_repr_shows_values_without_treatment_control_contrast():
    values = DailyMetricValues(
        [
            DailyMetricValue(
                ds=date(2024, 1, 1),
                metric="revenue",
                group_id="control",
                value=Estimate(value=10.0, lb=9.0, ub=11.0, level=0.95),
                n=100,
            ),
            DailyMetricValue(
                ds=date(2024, 1, 1),
                metric="revenue",
                group_id="treatment",
                value=Estimate(value=12.0, lb=11.0, ub=13.0, level=0.95),
                n=100,
            ),
        ]
    )

    from increment._interpretation import interpret_readout, render_readout_text

    rendered = render_readout_text(interpret_readout(values, None))

    assert "absolute mean" in rendered
    assert "10" in rendered and "12" in rendered
    assert "arm control" in rendered and "arm treatment" in rendered
    assert "treatment vs control" not in rendered
    assert "Interpretation:" not in rendered


def test_daily_unavailable_reason_is_preserved_in_readout():
    values = DailyMetricValues(
        [
            DailyMetricValue(
                ds=date(2024, 1, 1),
                metric="revenue",
                group_id="treatment",
                value=None,
                unavailable="few_units",
                n=1,
            )
        ]
    )

    assert "few_units" in repr(values)


def test_filtering_all_daily_means_keeps_arm_mean_identity():
    from increment.estimation.readout_types import (
        CellKey,
        CellRecord,
        PopulationRoster,
        ReadoutMetadata,
        ReadoutScope,
        SourceReadoutScope,
    )

    source_id = f"sha256:{'a' * 64}"
    cell = CellKey(
        kind="arm",
        metric="rev",
        method=None,
        method_role=None,
        estimand=None,
        analysis_population="assigned",
        group_id="treatment",
        ds=date(2024, 1, 1),
        ds_basis="calendar",
    )
    source = SourceReadoutScope(
        source_snapshot_id=source_id,
        cells=(cell,),
        decision_cells=(cell,),
        rosters=(
            PopulationRoster(
                analysis_population="assigned",
                arms=("control", "treatment"),
                source="experiment_counts",
                complete=True,
                control_group="control",
            ),
        ),
        decision_complete_by_population={"assigned": True},
    )
    metadata = ReadoutMetadata(
        scope=ReadoutScope(
            snapshot_id=source_id,
            cells=(cell,),
            decision_cells=(cell,),
            populations=("assigned",),
            families=(),
            by_source={source_id: source},
        ),
        cells=(CellRecord(cell=cell, source_snapshot_id=source_id),),
    )
    values = DailyMetricValues(
        [
            DailyMetricValue(
                ds=date(2024, 1, 1),
                metric="rev",
                group_id="treatment",
                source_snapshot_id=source_id,
                decision_scope_complete=True,
                value=Estimate(value=2.0, lb=1.5, ub=2.5, level=0.95),
                n=10,
            )
        ],
        metadata=metadata,
    )

    rendered = repr(values.filter(lambda _: False))

    assert "arm treatment" in rendered
    assert "treatment vs control" not in rendered


def test_row_only_readout_table_retains_roster_and_family_qualifications():
    from increment.tables import estimates_to_readout

    results = _analysis(_frame(guard_constant=False)).run(metrics=["rev"])
    rows = estimates_to_readout(results)
    direct_html = readout_table(results).gt().as_raw_html()
    row_only_html = readout_table(rows).gt().as_raw_html()

    assert "family" in direct_html.lower()
    assert "roster" in direct_html.lower()
    assert "family" in row_only_html.lower()
    assert "roster" in row_only_html.lower()


def test_empty_breakout_table_keeps_segment_schema():
    pytest.importorskip("coeftable")
    from increment.breakout.estimates import BreakoutEstimates

    html = readout_table(BreakoutEstimates([]), nest_by="segment").gt().as_raw_html()

    assert "Scope and integrity unknown" in html


def test_interpretation_labels_distinct_source_snapshots():
    from increment._interpretation import interpret_readout, render_readout_text

    estimate = LiftEstimate(
        metric="revenue",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        source_snapshot_id=f"sha256:{'1' * 64}",
        lift=Estimate(value=0.2, lb=0.1, ub=0.3, level=0.95),
    )
    second = estimate.model_copy(update={"source_snapshot_id": f"sha256:{'2' * 64}"})
    rendered = render_readout_text(interpret_readout([estimate, second], None))

    assert rendered.count("sha256:") == 2


def test_failure_passthrough_is_escaped_in_complete_html_table():
    payload = "<img src=x onerror=alert(1)>"
    frame = _frame(guard_constant=False).rename(columns={"guard": payload})
    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        control="control",
        metrics=[
            MetricSpec(name="rev", type="mean"),
            MetricSpec(
                name=payload,
                type="mean",
                missing="drop",
                preferred_direction="decrease",
            ),
        ],
        plan=AnalysisPlan(primary="rev", guardrails=[payload]),
    )
    with analysis:
        results = analysis.run()
    html = readout_table(cast(Any, results)).gt().as_raw_html()
    assert "<img" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html


def test_saved_results_decode_when_only_the_boundary_module_is_imported():
    analysis = _analysis(_frame())
    try:
        payload = analysis.run(metrics=["rev"]).model_dump_json()
    finally:
        analysis.close()

    code = (
        "import sys; "
        "from increment.estimation.readout_types import ReadoutResults; "
        "assert 'increment.breakout.estimates' not in sys.modules; "
        "results = ReadoutResults.model_validate_json(sys.argv[1]); "
        "assert results[0].metric == 'rev'"
    )
    child = subprocess.run([sys.executable, "-c", code, payload], capture_output=True, text=True)
    assert child.returncode == 0, child.stderr


def test_ordinary_run_keeps_each_mixed_family_row_provenance():
    analysis = Analysis.from_unit_summary(
        _frame(guard_constant=False),
        unit="unit",
        group="arm",
        control="control",
        metrics=[
            MetricSpec(name="rev", type="mean", preferred_direction="increase"),
            MetricSpec(name="guard", type="mean", preferred_direction="decrease"),
        ],
        plan=AnalysisPlan(primary="rev", secondaries=["guard"]),
    )
    results = _readouts(analysis.run())
    primary = [row for row in results if row.metric == "rev"]
    secondary = [row for row in results if row.metric == "guard"]
    assert primary and secondary
    assert results.metadata is not None
    families = results.metadata.scope.families
    primary_family = next(
        family for family in families if any(member.metric == "rev" for member in family.members)
    )
    secondary_family = next(
        family for family in families if any(member.metric == "guard" for member in family.members)
    )
    assert primary_family.family_id != secondary_family.family_id
    assert {row.role for row in primary} == {"primary"}
    assert {row.family_id for row in primary} == {primary_family.family_id}
    assert {row.family_axes for row in primary} == {None}
    assert secondary_family.family is not None
    assert {row.family_axes for row in secondary} == {secondary_family.family.axes}


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
    """Restored analyses retain assignment diagnostics as well as results."""
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
    health = replay.health_check()
    assigned = next(check for check in health.checks if check.check == "assigned_allocation")
    assert assigned.availability == "available"
    assert assigned.conclusion == "warning"
    assert assigned.srm is not None and assigned.srm.is_srm

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


def test_registered_sequential_capture_retains_unassigned_audit_counts():
    from increment.sources import UNASSIGNED_LABEL
    from tests.binary_sequential_cases import ROUTES, unit_rows

    rows = unit_rows(seed=17, n=200)
    rows.append(
        {
            "user_id": "unassigned-1",
            "variant": None,
            "purchase": 0,
            "enrollment": rows[0]["enrollment"],
        }
    )
    analysis = Analysis.from_unit_summary(
        pd.DataFrame(rows),
        unit="user_id",
        group="variant",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        on_unassigned="exclude",
        metrics=[MetricSpec(name="purchase", type="conversion")],
        plan=AnalysisPlan(primary="purchase", inference=ROUTES["always_valid"].inference),
        experiment_id="exp",
        exposure_date="enrollment",
    )

    snapshot = analysis.sequential_snapshot()
    assert snapshot.assignment_counts[UNASSIGNED_LABEL] == 1
    assert UNASSIGNED_LABEL not in {cell.group_id for cell in snapshot.registration.roster}
    results = analysis.run()
    metadata = results.metadata
    assert metadata is not None
    (integrity,) = next(iter(metadata.scope.by_source.values())).integrity
    assert integrity.context["unassigned_units"] == 1


def test_registered_sequential_run_marks_legacy_missing_source_counts(tmp_path):
    import pyarrow.parquet as pq

    from increment.estimators import capture_sequential_snapshot
    from increment.sources import ASSIGNMENT_COUNTS_FIELD
    from tests.binary_sequential_cases import unit_rows

    rows = unit_rows(seed=17, n=200)
    analysis = _registered_integrity_analysis(n_control=200, n_treatment=200)
    snapshot = analysis.sequential_snapshot()
    path = tmp_path / "registered-checkpoint.parquet"
    analysis.export(path)
    payload = pq.read_table(path).to_pylist()
    legacy_snapshot = capture_sequential_snapshot(
        snapshot.registration,
        [
            {
                "unit_id": row["user_id"],
                "group_id": row["variant"],
                "values": {"purchase": row["purchase"]},
                "segments": {},
            }
            for row in rows
        ],
        source_id=snapshot.registration.source_id,
        definitions_id=snapshot.registration.definitions_id,
        finalized=True,
        reveal_cursor=snapshot.reveal_cursor,
    )
    payload[0]["sequential_snapshot"] = legacy_snapshot.model_dump_json()
    payload[0].pop(ASSIGNMENT_COUNTS_FIELD, None)
    replay = Analysis.from_moments(
        payload,
        metrics=[MetricSpec(name="purchase", type="conversion")],
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
    )
    results = replay.run()

    assert results.metadata is not None
    assert results.metadata.scope is not None
    (integrity,) = next(iter(results.metadata.scope.by_source.values())).integrity
    assert integrity.status == "not_checked_missing_counts"
    assert integrity.code == "integrity.counts_missing"
    assert integrity.observed is None


def test_sequential_trigger_augmentation_reuses_captured_snapshot_integrity(monkeypatch):
    from datetime import date

    import ibis

    from increment.readouts import _sequential_scope
    from tests.test_sequential_public_sources import _native_fixture

    query_calls: list[str] = []
    armed = False
    connect = ibis.duckdb.connect

    def guarded_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        for method in ("execute", "to_pyarrow"):
            original = getattr(connection, method, None)
            if original is None:
                continue

            def guard_query(*query_args, _method=method, _original=original, **query_kwargs):
                if armed:
                    query_calls.append(_method)
                    raise AssertionError("run() must not query after sequential capture")
                return _original(*query_args, **query_kwargs)

            monkeypatch.setattr(connection, method, guard_query)
        return connection

    monkeypatch.setattr(ibis.duckdb, "connect", guarded_connect)

    _connection, _definitions, analysis = _native_fixture("bernoulli", triggered=True)
    snapshot = analysis.capture_sequential(finalized=True, as_of=date(2025, 1, 16))
    expected_counts = snapshot.assignment_counts
    assert expected_counts is not None
    integrity_calls = []
    original_integrity = _sequential_scope.assignment_integrity

    def counted_integrity(*args, **kwargs):
        integrity_calls.append(kwargs.get("counts", args[1] if len(args) > 1 else None))
        return original_integrity(*args, **kwargs)

    monkeypatch.setattr(_sequential_scope, "assignment_integrity", counted_integrity)
    armed = True
    try:
        results = analysis.run()
    finally:
        armed = False
        analysis.close()

    (integrity,) = next(iter(results.metadata.scope.by_source.values())).integrity
    assert query_calls == []
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
    assert triggered.failure_code is None
    assert triggered.require_sequential_result().checkpoint.population == "triggered"
    assert triggered.require_sequential_result().checkpoint.prefix_id != (
        assigned.require_sequential_result().checkpoint.prefix_id
    )


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


def test_readout_results_json_replay_preserves_daily_date_axis():
    from datetime import date

    row = DailyLiftEstimate(
        metric="revenue",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        ds=date(2025, 1, 2),
        lift=None,
        unavailable="no_treatment_arm",
    )
    results = DailyLiftEstimates([row])

    restored = ReadoutResults.model_validate_json(results.model_dump_json())

    assert restored[0].ds == date(2025, 1, 2)
    assert isinstance(restored[0].ds, date)


@pytest.mark.parametrize(
    "estimate_type",
    [LiftEstimate, DailyLiftEstimate, BreakoutEstimate],
)
def test_unavailable_estimate_repr_keeps_identity_and_failure_reason(estimate_type):
    fields = {
        "metric": "guard",
        "group_id": "treatment",
        "method": "unadjusted",
        "method_role": "decision",
        "inference": "fixed",
        "alternative": "two-sided",
        "null_lift": 0.0,
        "lift": None,
        "failure_code": "readout.cell.missing_metric_observations",
        "failure_context": {
            "reason": "missing_metric_observations",
            "metric": "guard",
            "group_id": "treatment",
        },
    }
    if estimate_type is DailyLiftEstimate:
        fields.update(
            ds="2025-01-01",
            analysis_population="assigned",
            estimand="itt",
            value_scale="relative",
            dimension=None,
            dimension_value=None,
            sequential_result=None,
            relative_confidence_set=None,
            binomial_set=None,
            unavailable=None,
            relative_unavailable_reason=None,
            sampling_available=False,
            low_reliability=False,
        )
    elif estimate_type is BreakoutEstimate:
        fields.update(
            dimension="country",
            dimension_value="US",
            excluded="no_data",
        )
    row = estimate_type.model_construct(**fields)

    rendered = str(row)
    assert "guard" in rendered
    assert "treatment" in rendered
    assert "readout.cell.missing_metric_observations" in rendered
    assert "missing_metric_observations" in rendered
    from increment._interpretation import interpret_readout, render_readout_text

    collection_text = render_readout_text(interpret_readout([row], None))
    if estimate_type is BreakoutEstimate:
        assert "country US" in collection_text
    elif estimate_type is DailyLiftEstimate:
        assert "date 2025-01-01" in collection_text


def test_scoped_repr_keeps_relative_unavailability_reason():
    results = _analysis(_frame(guard_constant=False)).run(metrics=["rev"])
    row = results[0].model_copy(
        update={"lift": None, "relative_unavailable_reason": "zero_relative_variance"}
    )
    collection = LiftEstimates([row], metadata=results.metadata)

    rendered = str(collection)

    assert "relative unavailable" in rendered.lower()
    assert "zero_relative_variance" in rendered


def test_cell_reason_preserves_persisted_failure():
    row = LiftEstimate.model_construct(
        metric="guard",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        lift=None,
        failure_code="readout.cell.missing_metric_observations",
        failure_context={"reason": "missing_metric_observations"},
    )
    restored_json = LiftEstimate.model_validate_json(row.model_dump_json())
    restored_pickle = pickle.loads(pickle.dumps(row))

    assert row.reason == "missing_metric_observations"
    assert restored_json.reason == restored_pickle.reason == row.reason
    for restored in (restored_json, restored_pickle):
        assert restored.failure_code == row.failure_code
        assert restored.failure_context == row.failure_context
        assert restored.lift is None
    assert "reason" not in LiftEstimate.model_fields
    assert "reason" not in row.model_dump()

    code_only = LiftEstimate.model_construct(
        metric="guard",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        lift=None,
        failure_code="readout.cell.failure",
        failure_context={},
    )
    available = LiftEstimate.model_construct(
        metric="guard",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        lift=None,
        failure_code=None,
        failure_context=None,
    )
    assert code_only.reason == "readout.cell.failure"
    assert available.reason is None


def test_missing_arm_for_one_metric_is_a_typed_cell_failure_while_another_observes_it(tmp_path):
    import pyarrow.parquet as pq

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
    rendered = str(failed)
    assert "guard" in rendered
    assert "treatment" in rendered
    assert "readout.cell.missing_metric_observations" in rendered
    assert "reason=" in rendered

    from increment.tables import readout_table

    failed_repr = repr(results)
    assert "unavailable guard:" in failed_repr.lower()
    failed_table = readout_table(results).gt().as_raw_html()
    assert "guard" in failed_table
    corrected = _analysis(_frame(guard_constant=False)).run()
    assert corrected.metadata.scope.decision_complete("assigned") is True
    assert "guard" in repr(corrected)
    corrected_table = readout_table(corrected).gt().as_raw_html()
    assert "guard" in corrected_table
    assert failed.lift is None

    path = tmp_path / "partial-required-cell.parquet"
    analysis.export(path)
    replay = Analysis.from_moments(
        pq.read_table(path).to_pylist(),
        metrics=[MetricSpec(name="rev", type="mean"), MetricSpec(name="guard", type="mean")],
        control="control",
    )
    restored = _readouts(replay.run())
    restored_rows = _lift_rows(restored)
    assert restored.metadata is not None
    (restored_failed,) = [row for row in restored_rows if row.failure_code is not None]
    (restored_surviving,) = [row for row in restored_rows if row.failure_code is None]
    assert restored_failed.failure_code == failed.failure_code
    assert restored_failed.failure_context == failed.failure_context
    assert restored_failed.lift is None
    assert restored_failed.decision_scope_complete is False
    assert restored.metadata.scope.decision_complete("assigned") is False
    assert restored_surviving.metric == "rev" and restored_surviving.lift is not None

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
    frame = copied.to_frame(backend="pandas")
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
    assert "Decision scope is incomplete" in full_html

    filtered = results.filter(lambda row: row.metric == "rev")
    rows = estimates_to_readout(filtered)
    assert all(row["view_partial"] is True for row in rows)
    partial_html = readout_table(rows).gt().as_raw_html()
    assert "Partial view" in partial_html
    assert "Decision scope is incomplete" in partial_html


def test_qualified_collection_repr_frames_and_empty_table_keep_scope():
    pytest.importorskip("coeftable")
    from increment import from_frame, run
    from increment.tables import estimates_to_readout, readout_table
    from tests.test_tables import _as_latex, _latex_text

    experiment, resolver = from_frame(
        _frame(guard_constant=False),
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics=[
            MetricSpec(name="rev", type="mean", preferred_direction="increase"),
            MetricSpec(name="guard", type="mean", preferred_direction="decrease"),
        ],
        plan=AnalysisPlan(primary="rev", guardrails=["guard"]),
    )
    result = run(experiment, resolver)
    results = result.readout()
    rendered = repr(results)
    assert "SRM not checked" in rendered
    assert not any(
        word in rendered.lower()
        for word in ("not success", "failed", "incomplete", "unknown", "unavailable")
    )
    assert "declare allocation" in rendered
    lines = rendered.splitlines()
    metric_headline = next(i for i, line in enumerate(lines) if "rev (primary" in line)
    assert metric_headline < 5
    assert metric_headline < next(i for i, line in enumerate(lines) if "SRM not checked" in line)
    qualifications = [
        "metric",
        "analysis_population",
        "source_snapshot_id",
        "integrity_status",
        "required_cell_count",
        "unavailable_cells",
        "hidden_cells",
        "advisories",
        "view_partial",
    ]
    for backend in ("pandas", "polars", "pyarrow"):
        frame = nw.from_native(results.to_frame(backend=backend), eager_only=True)
        direct = nw.from_native(result.to_frame(backend=backend), eager_only=True)
        assert set(qualifications) <= set(frame.columns)
        assert direct.columns == ["scope_view", "scope_values", *frame.columns]
        assert direct.select(qualifications).rows() == frame.select(qualifications).rows()
    rows = estimates_to_readout(results)
    assert all(row["integrity_status"] == "not_checked_missing_declaration" for row in rows)
    assert all(row["analysis_population"] == "assigned" for row in rows)
    for table in (readout_table(results), readout_table(result)):
        assert "not_checked_missing_declaration" in table.gt().as_raw_html()
    assert "not_checked_missing_declaration" in _latex_text(_as_latex(readout_table(results)))

    empty = results.filter(lambda row: False)
    assert "View: partial" in repr(empty)
    empty_frame = nw.from_native(empty.to_frame(), eager_only=True)
    assert len(empty_frame) == 0 and "integrity_status" in empty_frame.columns
    assert "Scope and integrity unknown" not in readout_table(empty).gt().as_raw_html()

    unselected = run(experiment, resolver, metrics=[])
    live_frame = nw.from_native(unselected.to_frame(), eager_only=True)
    unselected_frame = nw.from_native(
        pickle.loads(pickle.dumps(unselected)).to_frame(), eager_only=True
    )
    assert len(unselected_frame) == 0
    assert {"scope_view", "scope_values", "metric", "lift"} <= set(unselected_frame.columns)
    assert "integrity_status" not in unselected_frame.columns
    assert unselected_frame.schema == live_frame.schema
    assert "Scope and integrity unknown" in readout_table(unselected).gt().as_raw_html()


def test_real_assignment_integrity_failure_still_says_failed():
    results = Analysis.from_unit_panel(
        _imbalanced_assignment_frame(),
        unit="unit",
        group="arm",
        date="ds",
        design=_independent_design(),
        metrics=[MetricSpec(name="rev", type="mean")],
    ).run()

    rendered = repr(results)
    assert "Integrity: failed" in rendered
    lines = rendered.splitlines()
    assert next(i for i, line in enumerate(lines) if "rev (" in line) < next(
        i for i, line in enumerate(lines) if line.startswith("Integrity:")
    )


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


def _assert_selection_repr_disclosures(results, selected_row, sensitivity_row):
    from increment.estimation.results import _selection_qualification

    selection = _selection_qualification(selected_row)
    assert selection is not None
    rendered = repr(results)
    assert selection in rendered
    assert f"method role {sensitivity_row.method_role}" in rendered


def test_failed_multiarm_secondary_keeps_full_family_and_sensitivity_provenance():
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
                    decision_method=Method(name="unadjusted"),
                    sensitivity_methods=(
                        Method.cuped(covariates=CupedCovariate(name="x", kind="continuous")),
                    ),
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
    assert successful.discovery is True
    _assert_selection_repr_disclosures(results, successful, sensitivity_rows[0])
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
    assert "treatment_a vs control" in html
    assert "treatment_b vs control" in html
    assert "required cells" in html.lower()
    assert "Unadjusted for multiplicity" not in html
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
                decision_method=Method(name="unadjusted"),
                sensitivity_methods=(
                    Method.cuped(covariates=CupedCovariate(name="x", kind="continuous")),
                ),
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


pytestmark = pytest.mark.integration
