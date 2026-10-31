"""Portable decision-cell behavior on the seeded public readout paths."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_stored_margin_comparison_survives_json_and_arrow():
    import json

    from increment.errors import CodedError
    from increment.result_envelope import Operation, ResultCell, compare_margin

    _data, _design = _relevel_test_data()
    row, _withheld = _relevel_test_t_rows(_data, _design)
    cell = ResultCell.from_row(row)
    restored = (
        ResultCell.from_payload(json.loads(json.dumps(cell.to_payload()))),
        ResultCell.from_arrow(ResultCell.to_arrow([cell]))[0],
    )

    for candidate in restored:
        assert Operation.COMPARE_MARGIN in candidate.supports["sampling"]
        assert compare_margin(candidate, margin=row.null_lift, scale="relative") == row.stat_sig()
        margin = row.null_lift + 0.1
        expected = row.model_copy(update={"null_lift": margin, "null_abs": None}).stat_sig()
        assert compare_margin(candidate, margin=margin, scale="relative") == expected
        absolute_margin = row.null_abs or 0.0
        absolute_expected = row.model_copy(
            update={"null_lift": 0.0, "null_abs": absolute_margin}
        ).stat_sig()
        assert (
            compare_margin(candidate, margin=absolute_margin, scale="absolute") == absolute_expected
        )

    unavailable = ResultCell.from_report_payload(
        row_kind="sensitivity",
        identity=cell.identity,
        label=cell.label,
        row=row,
    )
    with pytest.raises(CodedError) as raised:
        compare_margin(unavailable, margin=0.0, scale="relative")
    assert raised.value.code == "results.operation.unsupported"


def test_native_absolute_margin_matches_public_decision_after_codecs():
    from increment.breakout.estimates import BreakoutEstimates
    from increment.errors import CodedError
    from increment.result_envelope import ResultCell, compare_margin
    from tests.breakout.test_heterogeneity import _absolute_breakout_estimate
    from tests.estimation.test_decision_stats import _absolute_linear_est

    rows = (
        _absolute_linear_est(mu=1.0, sigma=0.2),
        _absolute_breakout_estimate("A", value=1.0, se=0.2),
    )
    for row in rows:
        cell = (
            BreakoutEstimates([row]).as_cells()[0]
            if isinstance(row, type(rows[1]))
            else ResultCell.from_row(row)
        )
        restored = (
            ResultCell.from_payload(json.loads(json.dumps(cell.to_payload()))),
            ResultCell.from_arrow(ResultCell.to_arrow([cell]))[0],
        )
        expected = row.model_copy(update={"null_lift": 0.0, "null_abs": None}).stat_sig()
        assert expected is True
        for candidate in restored:
            assert compare_margin(candidate, margin=0.0, scale="absolute") == expected
            with pytest.raises(CodedError) as raised:
                compare_margin(candidate, margin=0.0, scale="relative")
            assert raised.value.code == "results.operation.unsupported"


def test_absolute_margin_refuses_when_retained_additive_bounds_are_absent():
    from increment.errors import CodedError
    from increment.result_envelope import ResultCell, compare_margin

    row, _ = _relevel_test_t_rows(*_relevel_test_data())
    row = row.model_copy(update={"abs_lb": None, "abs_ub": None, "abs_alpha": None})
    cell = ResultCell.from_row(row)
    with pytest.raises(CodedError) as raised:
        compare_margin(cell, margin=0.0, scale="absolute")
    assert raised.value.code == "results.operation.unsupported"


def test_sequential_bonferroni_recomputes_at_requested_level():
    import math

    from increment.decision import AsymptoticSequentialEvidence
    from increment.estimators import SequentialCell
    from increment.frame import MetricSpec
    from increment.result_envelope import ResultCell, combine_family_evidence, family_evidence
    from tests.test_asymptotic_public_sources import _analysis, _frame, _plan

    metric = MetricSpec(name="outcome", type="mean")
    registered_cell = SequentialCell(
        metric="outcome", group_id="treatment", family=True, alpha=0.05
    )
    rows = _analysis(
        _frame([1, 2, 3] * 20, [1.49, 2.49, 3.49] * 20),
        [metric],
        _plan([metric], cells=[registered_cell], secondary=True),
    ).run()
    cell = ResultCell.from_row(rows[0])
    retained = (
        ResultCell.from_payload(json.loads(json.dumps(cell.to_payload()))),
        ResultCell.from_arrow(ResultCell.to_arrow([cell]))[0],
    )
    for candidate in retained:
        item = family_evidence(candidate)
        assert isinstance(item.evidence, AsymptoticSequentialEvidence)
        result = item.evidence.result
        from fractions import Fraction

        for level in (0.05, 0.01):
            alpha = Fraction(str(level))
            expected = result._denominator_resolved(alpha) and result.log_e >= math.log(1 / level)
            assert (
                item.identity_hash
                in combine_family_evidence((item,), procedure="bonferroni", level=level)
            ) is expected


def test_registered_likelihood_family_evidence_survives_codecs():
    import math
    from fractions import Fraction

    from increment.decision import AsymptoticSequentialEvidence, EValueEvidence
    from increment.estimation.family import e_bh_select
    from increment.estimators import AlwaysValid, SequentialCell
    from increment.result_envelope import ResultCell, combine_family_evidence, family_evidence
    from tests.sequential_cases import capture, estimate_sequential, records, registration

    registered_cells = (
        SequentialCell(metric="outcome", group_id="treatment", alpha=Fraction(1, 20), family=True),
        SequentialCell(
            metric="outcome", group_id="never-enrolled", alpha=Fraction(1, 20), family=True
        ),
        SequentialCell(
            metric="outcome", group_id="also-never-enrolled", alpha=Fraction(1, 20), family=True
        ),
    )
    registered = registration(cells=registered_cells)
    bundle = estimate_sequential(
        capture(registered, records([0] * 6, [1, 1, 1, 0, 0, 0])),
        AlwaysValid(registration=registered),
    )
    likelihood_cells = tuple(ResultCell.from_row(row) for row in bundle.results)
    from increment.frame import MetricSpec
    from tests.test_asymptotic_public_sources import _analysis, _frame, _plan

    metric = MetricSpec(name="outcome", type="mean")
    sequential_cell = SequentialCell(
        metric="outcome", group_id="treatment", family=True, alpha=0.05
    )
    asymptotic_rows = _analysis(
        _frame([1, 2, 3] * 30, [7, 10, 13] * 30),
        [metric],
        _plan([metric], cells=[sequential_cell], secondary=True),
    ).run()
    cells = (*likelihood_cells, ResultCell.from_row(asymptotic_rows[0]))
    retained = (
        tuple(ResultCell.from_payload(json.loads(json.dumps(item.to_payload()))) for item in cells),
        ResultCell.from_arrow(ResultCell.to_arrow(cells)),
    )
    for restored in retained:
        evidence = tuple(family_evidence(candidate) for candidate in restored)
        likelihood_evidence = [
            item for item in evidence if isinstance(item.evidence, EValueEvidence)
        ]
        assert len(likelihood_evidence) == 3
        missing = next(
            item
            for item in likelihood_evidence
            if isinstance(item.evidence, EValueEvidence)
            and item.evidence.checkpoint.status == "missing"
        )
        assert isinstance(missing.evidence, EValueEvidence)
        assert missing.evidence.log_e == float("-inf")
        assert len(evidence) == 4
        assert any(isinstance(item.evidence, AsymptoticSequentialEvidence) for item in evidence)
        logs = []
        for item in evidence:
            if isinstance(item.evidence, EValueEvidence):
                logs.append(item.evidence.log_e)
            elif isinstance(item.evidence, AsymptoticSequentialEvidence):
                logs.append(item.evidence.result.log_e)
            else:
                raise AssertionError("expected retained sequential evidence")
        expected_e_bh = frozenset(
            evidence[index].identity_hash for index in e_bh_select(logs, 0.05)
        )
        assert combine_family_evidence(evidence, procedure="e_bh", level=0.05) == expected_e_bh
        threshold = Fraction(1, 20) / len(evidence)
        log_threshold = math.log(1 / float(threshold))
        expected_bonferroni = set()
        for item in evidence:
            if isinstance(item.evidence, EValueEvidence):
                rejects = item.evidence.log_e >= log_threshold
            elif isinstance(item.evidence, AsymptoticSequentialEvidence):
                result = item.evidence.result
                rejects = result._denominator_resolved(threshold) and result.log_e >= log_threshold
            else:
                raise AssertionError("expected retained sequential evidence")
            if rejects:
                expected_bonferroni.add(item.identity_hash)
        assert combine_family_evidence(evidence, procedure="bonferroni", level=0.05) == frozenset(
            expected_bonferroni
        )


def test_typed_family_evidence_recomputes_bh_after_both_codecs():
    import json

    from increment.estimation.decision_types import PValueEvidence
    from increment.result_envelope import ResultCell, combine_family_evidence, family_evidence

    selected, unselected = _relevel_test_family_rows(*_relevel_test_data())
    cells = [ResultCell.from_row(row) for row in (selected, unselected)]
    expected = {
        cell.identity.hash()
        for cell, row in zip(cells, (selected, unselected), strict=True)
        if row.discovery
    }
    round_trips = (
        [ResultCell.from_payload(json.loads(json.dumps(cell.to_payload()))) for cell in cells],
        ResultCell.from_arrow(ResultCell.to_arrow(cells)),
    )
    for restored in round_trips:
        evidence = tuple(family_evidence(cell) for cell in restored)
        assert all(isinstance(item.evidence, PValueEvidence) for item in evidence)
        assert combine_family_evidence(evidence, procedure="bh", level=0.05) == expected
        bonferroni = frozenset(
            item.identity_hash
            for item in evidence
            if isinstance(item.evidence, PValueEvidence) and item.evidence.p_value <= 0.025
        )
        assert combine_family_evidence(evidence, procedure="bonferroni", level=0.05) == bonferroni


def test_set_only_winsor_does_not_support_margin_comparison_after_codecs():
    from increment.errors import CodedError
    from increment.result_envelope import Operation, ResultCell, compare_margin

    data, design = _relevel_test_data()
    row = _relevel_test_winsor_row(data, design)
    cell = ResultCell.from_row(row)
    restored = (
        ResultCell.from_payload(json.loads(json.dumps(cell.to_payload()))),
        ResultCell.from_arrow(ResultCell.to_arrow([cell]))[0],
    )
    for candidate in restored:
        assert Operation.COMPARE_MARGIN not in candidate.supports["sampling"]
        with pytest.raises(CodedError) as raised:
            compare_margin(candidate, margin=row.null_lift + 0.05, scale="relative")
        assert raised.value.code == "results.operation.unsupported"


@pytest.mark.integration
def test_seeded_lift_cells_round_trip_retained_row_and_legacy_interval(seeded_con, seeded_defs):
    import increment.result_envelope as result_envelope
    from increment import Analysis, Method
    from increment.breakout.estimates import LiftEstimates
    from increment.estimation.results import LiftEstimate
    from increment.result_envelope import Operation, ResultCell, relevel

    rows = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con).run(
        decision_method=Method(name="unadjusted")
    )
    assert isinstance(rows, LiftEstimates)
    for candidate in rows:
        assert (
            LiftEstimate.model_validate(json.loads(json.dumps(candidate.model_dump(mode="json"))))
            == candidate
        )
    row = next(r for r in rows if r.metric == "avg_session_duration")
    assert isinstance(row, LiftEstimate)

    cell = row.as_cells(metadata=rows.metadata)[0]
    assert not hasattr(result_envelope, "reinterval")
    assert not hasattr(Operation, "REINTERVAL")
    expected = row.reintervalize(0.02)
    restored = ResultCell.from_payload(json.loads(json.dumps(cell.to_payload())))
    actual = relevel(restored, alpha=0.02)
    assert actual.summary is not None
    assert expected.lift is not None
    assert actual.summary == expected.as_cells()[0].summary
    assert actual.summary.value_scale == row.value_scale == "relative"
    reference = json.loads(
        (Path(__file__).parent / "fixtures" / "result_envelope_reference.json").read_text()
    )["reference"]["mean/avg_session_duration"]
    assert expected.lift.lb == pytest.approx(reference["lower"])
    assert expected.lift.ub == pytest.approx(reference["upper"])
    assert Operation.RELEVEL in actual.supports["sampling"]


@pytest.mark.integration
def test_exploratory_run_stamping_preserves_source_content_provenance(seeded_con, seeded_defs):
    import warnings

    from increment import Analysis
    from increment.errors import IncrementWarning

    analysis = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con)
    try:
        (metric,) = analysis.available_metrics[:1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.warns(IncrementWarning):
                rows = analysis.run(metrics=[], exploratory_metrics=[metric.name])
        assert rows.metadata is not None
        assert rows.metadata.source_content_digest is not None
        (cell,) = rows.as_cells()
        assert cell.provenance.source_content_digest == rows.metadata.source_content_digest
        assert cell.identity.source_content_digest == rows.metadata.source_content_digest
        payload = cell.to_payload()
        assert "source_content_digest" not in payload["identity"]
        restored = type(cell).from_payload(payload)
        assert restored.identity.source_content_digest == cell.identity.source_content_digest
    finally:
        analysis.close()


@pytest.mark.integration
def test_ratio_and_cuped_rows_survive_detached_json_arrow_operations(
    seeded_con, seeded_pre_period_con, seeded_defs
):
    import gc

    from increment import Analysis, Method
    from increment.breakout.estimates import LiftEstimates
    from increment.estimation.results import LiftEstimate
    from increment.result_envelope import Operation, ResultCell, relevel
    from increment.semantics.loader import load
    from tests.analysis_factory import make_analysis_like

    reference = json.loads(
        (Path(__file__).parent / "fixtures" / "result_envelope_reference.json").read_text()
    )["reference"]
    analysis = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_pre_period_con)
    ratio_analysis = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con)
    ratio = next(
        metric for metric in load(seeded_defs).metrics if metric.name == "revenue_per_session"
    )
    ratio_analysis = make_analysis_like(ratio_analysis, [*ratio_analysis.metrics, ratio])
    ratio_rows = ratio_analysis.run(decision_method=Method(name="unadjusted"))
    assert isinstance(ratio_rows, LiftEstimates)
    ratio_row = next(row for row in ratio_rows if row.metric == "revenue_per_session")
    assert isinstance(ratio_row, LiftEstimate)
    cuped_rows = analysis.run(decision_method=Method(name="cuped", variance_reduction="cuped"))
    assert isinstance(cuped_rows, LiftEstimates)
    cuped_row = next(row for row in cuped_rows if row.metric == "avg_session_duration")
    assert isinstance(cuped_row, LiftEstimate)

    cases = []
    for row, key in (
        (ratio_row, "ratio/revenue_per_session/treatment"),
        (cuped_row, "cuped/avg_session_duration"),
    ):
        cell = ResultCell.from_row(row)
        cases.append(
            (
                row,
                reference[key],
                row.reintervalize(0.02),
                ResultCell.from_payload(json.loads(json.dumps(cell.to_payload()))),
                ResultCell.from_arrow(ResultCell.to_arrow([cell]))[0],
            )
        )

    del analysis, ratio_analysis, ratio_rows, cuped_rows
    gc.collect()

    for row, _expected, public_reinterval, *restored_cells in cases:
        assert row.lift is not None
        assert public_reinterval.lift is not None
        for restored in restored_cells:
            assert Operation.RELEVEL in restored.supports["sampling"]
            leveled = relevel(restored, alpha=0.02)
            assert leveled.summary is not None
            assert leveled.summary == public_reinterval.as_cells()[0].summary
            assert leveled.summary.value_scale == row.value_scale == "relative"
            reintervalled = relevel(restored, alpha=0.02).retained_row()
            assert reintervalled.lift is not None
            assert reintervalled.lift.alpha == pytest.approx(public_reinterval.lift.alpha)
            assert (reintervalled.lift.lb, reintervalled.lift.ub) == pytest.approx(
                (public_reinterval.lift.lb, public_reinterval.lift.ub)
            )


@pytest.mark.integration
def test_family_reissue_caps_threshold_at_each_rows_allocated_alpha(seeded_con, seeded_defs):
    from increment import Analysis, Method
    from increment.breakout.estimates import LiftEstimates
    from increment.estimation.family import select_exploratory_family
    from increment.estimation.results import LiftEstimate

    rows = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con).run(
        decision_method=Method(name="unadjusted")
    )
    assert isinstance(rows, LiftEstimates)
    row = next(item for item in rows if item.metric == "avg_session_duration")
    assert isinstance(row, LiftEstimate)
    assert row.lift is not None
    low_budget = row.reintervalize(0.01)
    assert low_budget.lift is not None
    clear_family = {
        "discovery": None,
        "family_id": None,
        "family_q": None,
        "family_size": None,
        "family_threshold": None,
        "family_axes": None,
        "family_guarantee": None,
        "family_procedure": None,
        "family_nominal_alpha": None,
        "multiplicity_status": "exploratory_unadjusted",
        "role": "exploratory",
    }
    corrected = [
        select_exploratory_family([item.model_copy(update=clear_family)], q=0.03)[0]
        for item in (row, low_budget)
    ]
    assert corrected[0].lift is not None
    assert corrected[1].lift is not None
    assert corrected[0].lift.alpha == pytest.approx(0.03)
    assert corrected[1].lift.alpha == pytest.approx(0.01)
    high_budget = row.reintervalize(0.03)
    low_budget_expected = low_budget.reintervalize(0.01)
    assert isinstance(high_budget, LiftEstimate)
    assert isinstance(low_budget_expected, LiftEstimate)
    assert high_budget.lift is not None
    assert low_budget_expected.lift is not None
    from increment.errors import CodedError

    assert isinstance(corrected[0], LiftEstimate)
    high_budget_row = corrected[0]
    low_budget_row = corrected[1]
    assert isinstance(low_budget_row, LiftEstimate)
    with pytest.raises(CodedError) as raised:
        high_budget_row.reintervalize(0.031)
    assert raised.value.code == "results.operation.alpha_above_allocation"
    with pytest.raises(CodedError) as low_budget_refusal:
        low_budget_row.reintervalize(0.02)
    assert low_budget_refusal.value.code == "results.operation.alpha_above_allocation"
    assert high_budget_row.lift is not None and high_budget.lift is not None
    assert (high_budget_row.lift.lb, high_budget_row.lift.ub) == pytest.approx(
        (high_budget.lift.lb, high_budget.lift.ub)
    )
    assert low_budget_expected.lift is not None and low_budget_row.lift is not None
    assert (low_budget_row.lift.lb, low_budget_row.lift.ub) == pytest.approx(
        (low_budget_expected.lift.lb, low_budget_expected.lift.ub)
    )


@pytest.mark.integration
def test_additive_only_primary_reintervalizes_from_its_absolute_reference():
    import pandas as pd

    from increment import Analysis, AnalysisPlan, MetricSpec, Randomized
    from increment.estimation.results import LiftEstimate

    data = pd.DataFrame(
        [
            {"unit": f"{group}-{index}", "arm": group, "value": value}
            for group, values in (
                ("control", [1.0, 2.0, 3.0, 4.0, 5.0]),
                ("treatment", [-4.0, -3.0, -2.0, -1.0, 0.0]),
            )
            for index, value in enumerate(values)
        ]
    )
    rows = Analysis.from_unit_summary(
        data,
        unit="unit",
        group="arm",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics=[MetricSpec(name="value", type="mean")],
        plan=AnalysisPlan(alpha=0.05, primary="value"),
    ).run()
    row = next(
        candidate
        for candidate in rows
        if isinstance(candidate, LiftEstimate) and candidate.group_id == "treatment"
    )

    assert row.lift is None
    assert row.abs_reference_kind == "t"
    assert row.abs_lb is not None and row.abs_ub is not None
    reissued = row.reintervalize(0.01)
    assert reissued.lift is None
    assert reissued.abs_alpha == pytest.approx(0.01)
    assert reissued.abs_lb is not None and reissued.abs_ub is not None


@pytest.mark.integration
def test_withheld_t_row_relevels_from_retained_statistics_after_arrow():
    from typing import cast

    import pandas as pd

    from increment import Analysis, AnalysisPlan, MetricSpec
    from increment.errors import CodedError
    from increment.estimation.results import LiftEstimate
    from increment.result_envelope import Operation, ResultCell, relevel
    from increment.semantics.design import Randomized

    data = pd.DataFrame(
        [
            {"unit": f"{group}-{value}", "arm": group, "value": float(value)}
            for group, values in (("control", range(1, 21)), ("treatment", range(4, 24)))
            for value in values
        ]
    )
    rows = cast(
        list[LiftEstimate],
        list(
            Analysis.from_unit_summary(
                data,
                unit="unit",
                group="arm",
                design=Randomized(
                    control_group="control",
                    allocation={"control": 0.5, "treatment": 0.5},
                    allocation_scheme="independent",
                ),
                metrics=[MetricSpec(name="value", type="mean")],
                plan=AnalysisPlan(alpha=0.05, primary="value"),
            ).run()
        ),
    )
    row = next(candidate for candidate in rows if candidate.group_id == "treatment")
    assert isinstance(row, LiftEstimate)
    withheld = row.reintervalize(1e-100)
    assert withheld.lift is not None and withheld.lift.alpha is None
    replayed = ResultCell.from_arrow(ResultCell.to_arrow(withheld.as_cells()))[0]
    assert Operation.RELEVEL in replayed.supports["sampling"]
    with pytest.raises(CodedError) as raised:
        relevel(replayed, alpha=0.04)
    assert raised.value.code == "results.operation.alpha_above_allocation"
    assert relevel(replayed, alpha=1e-110).summary is not None
    with pytest.raises(CodedError) as raised:
        relevel(replayed, alpha=0.06)
    assert raised.value.code == "results.operation.alpha_above_allocation"


def _relevel_test_data():
    import pandas as pd

    from increment.semantics.design import Randomized

    data = pd.DataFrame(
        [
            {
                "unit": f"{group}-{index}",
                "arm": group,
                "value": float(index % 11 + (group == "treatment") * 5),
                "null_value": float(index % 9),
                "winsor": float(index % 17 + (group == "treatment")),
            }
            for group in ("control", "treatment")
            for index in range(40)
        ]
    )
    design = Randomized(
        control_group="control",
        allocation={"control": 0.5, "treatment": 0.5},
        allocation_scheme="independent",
    )
    return data, design


def _relevel_test_t_rows(data, design):
    from typing import cast

    from increment import Analysis, AnalysisPlan, MetricSpec
    from increment.estimation.results import LiftEstimate

    rows = cast(
        list[LiftEstimate],
        list(
            Analysis.from_unit_summary(
                data,
                unit="unit",
                group="arm",
                design=design,
                metrics=[MetricSpec(name="value", type="mean")],
                plan=AnalysisPlan(alpha=0.05, primary="value"),
            ).run()
        ),
    )
    declared = next(row for row in rows if row.group_id == "treatment")
    return declared, declared.reintervalize(1e-100)


def _relevel_test_family_rows(data, design):
    from typing import cast

    from increment import Analysis, MetricSpec
    from increment.estimation.family import select_exploratory_family
    from increment.estimation.results import LiftEstimate

    rows = cast(
        list[LiftEstimate],
        list(
            Analysis.from_unit_summary(
                data,
                unit="unit",
                group="arm",
                design=design,
                metrics=[
                    MetricSpec(name="value", type="mean"),
                    MetricSpec(name="null_value", type="mean"),
                ],
            ).run()
        ),
    )
    family_rows = select_exploratory_family(rows, q=0.05)
    selected = next(row for row in family_rows if row.discovery is True)
    unselected = next(row for row in family_rows if row.discovery is False)
    return selected, unselected


def _relevel_test_winsor_row(data, design):
    from typing import cast

    from increment import Analysis, MetricSpec
    from increment.estimation.results import LiftEstimate

    rows = cast(
        list[LiftEstimate],
        list(
            Analysis.from_unit_summary(
                data,
                unit="unit",
                group="arm",
                design=design,
                metrics=[
                    MetricSpec(
                        name="winsor",
                        type="mean",
                        winsorization={
                            "upper_percentile": 0.75,
                            "inference": {"method": "influence-normal-v1"},
                        },
                    )
                ],
            ).run()
        ),
    )
    row = next(candidate for candidate in rows if candidate.group_id == "treatment")
    assert row.confidence_set is not None
    return row


def _relevel_test_joint_row():
    from scipy.stats import norm

    from increment.estimation.results import (
        JointContrastReference,
        LiftEstimate,
        relative_confidence_set,
    )

    joint_set = relative_confidence_set(
        JointContrastReference(a=10.0, c=1.0, var_a=1.0, var_c=1.0, cov_ac=0.0)
    )
    half_width = float(norm.isf(0.025))
    return LiftEstimate(
        metric="joint",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        inference="fixed",
        reference_kind="normal",
        value_scale="relative",
        scale="linear",
        lift=joint_set.estimate(),
        relative_confidence_set=joint_set,
        abs_diff=10.0,
        abs_se=1.0,
        abs_lb=10.0 - half_width,
        abs_ub=10.0 + half_width,
        abs_reference_kind="normal",
    )


def _retained_relevel_alpha(row):
    alphas = [
        alpha
        for alpha in (
            None if row.lift is None else row.lift.alpha,
            row.abs_alpha,
        )
        if alpha is not None
    ]
    if row.confidence_set is not None:
        alphas.append(row.confidence_set.alpha)
    if row.relative_confidence_set is not None:
        alphas.append(row.relative_confidence_set.alpha)
    if not alphas:
        return None
    retained = min(alphas)
    return retained if row.alternative == "two-sided" else retained / 2.0


def _relevel_test_cells():
    from increment.estimation.results import LiftEstimate
    from increment.result_envelope import ResultCell

    data, design = _relevel_test_data()
    declared, withheld = _relevel_test_t_rows(data, design)
    selected, unselected = _relevel_test_family_rows(data, design)
    winsor = _relevel_test_winsor_row(data, design)
    joint = _relevel_test_joint_row()
    assert isinstance(declared, LiftEstimate)
    assert isinstance(withheld, LiftEstimate)
    assert isinstance(selected, LiftEstimate)
    assert isinstance(unselected, LiftEstimate)
    assert isinstance(winsor, LiftEstimate)
    return tuple(
        (name, row, ResultCell.from_row(row))
        for name, row in (
            ("t", declared),
            ("withheld_t", withheld),
            ("selected_family", selected),
            ("unselected_family", unselected),
            ("winsor_set", winsor),
            ("joint_reference", joint),
        )
    )


def _assert_interval_contains(outer, inner):
    assert outer is not None and inner is not None
    assert outer.value_scale == inner.value_scale
    if inner.lower is None:
        assert outer.lower is None
    elif outer.lower is not None:
        assert outer.lower <= inner.lower or outer.lower == pytest.approx(inner.lower)
    if inner.upper is None:
        assert outer.upper is None
    elif outer.upper is not None:
        assert outer.upper >= inner.upper or outer.upper == pytest.approx(inner.upper)


@pytest.mark.integration
def test_directional_bh_interval_cannot_be_releveled_after_arrow():
    from typing import cast

    import pandas as pd

    from increment import Analysis, AnalysisPlan, MetricSpec
    from increment.errors import CodedError
    from increment.estimation.results import LiftEstimate
    from increment.result_envelope import Operation, ResultCell, relevel
    from increment.semantics.design import Randomized

    data = pd.DataFrame(
        [
            {
                "unit": f"{group}-{index}",
                "arm": group,
                "v": float(index % 11 + (1 if group == "control" else 11)),
            }
            for group in ("control", "treatment")
            for index in range(100)
        ]
    )
    rows = cast(
        list[LiftEstimate],
        list(
            Analysis.from_unit_summary(
                data,
                unit="unit",
                group="arm",
                design=Randomized(
                    control_group="control",
                    allocation={"control": 0.5, "treatment": 0.5},
                    allocation_scheme="independent",
                ),
                metrics=[MetricSpec(name="v", type="mean")],
                plan=AnalysisPlan(alternative="greater", secondaries=["v"], q=0.05),
            ).run()
        ),
    )
    row = next(candidate for candidate in rows if candidate.group_id == "treatment")
    assert row.discovery is True
    assert row.lift is not None and row.lift.open_side == "upper"
    cell = ResultCell.from_arrow(ResultCell.to_arrow(row.as_cells()))[0]
    with pytest.raises(CodedError) as raised:
        relevel(cell, alpha=0.01)
    assert raised.value.code == "results.operation.unsupported"
    assert Operation.RELEVEL not in cell.supports["sampling"]


@pytest.mark.integration
def test_relevel_is_conservative_across_pilot_cell_kinds():
    from hypothesis import given, settings
    from hypothesis import strategies as st

    from increment.errors import CodedError
    from increment.result_envelope import Operation, ResultCell, relevel

    cells = _relevel_test_cells()

    @settings(max_examples=12, deadline=None)
    @given(
        requests=st.lists(
            st.sampled_from([0.001, 0.01, 0.025, 0.04, 0.05, 0.06, 0.1]),
            min_size=1,
            max_size=5,
        )
    )
    def check_sequence(requests):
        for name, row, original in cells:
            initial_alpha = _retained_relevel_alpha(row)
            restored = ResultCell.from_payload(json.loads(json.dumps(original.to_payload())))
            arrow_restored = ResultCell.from_arrow(ResultCell.to_arrow([original]))[0]
            for cell in (restored, arrow_restored):
                assert cell.identity == original.identity
                assert cell.family == original.family
                if name in {"winsor_set", "joint_reference"}:
                    assert Operation.RELEVEL not in cell.supports["sampling"]
                    assert cell.components["sampling"][0] == original.components["sampling"][0]
                    continue
                assert Operation.RELEVEL in cell.supports["sampling"]
                assert initial_alpha is not None
                current = cell
                current_alpha = initial_alpha
                original_status = cell.components["sampling"][0]
                original_row = cell.retained_row()
                family_fields = (
                    "multiplicity_status",
                    "discovery",
                    "family_id",
                    "family_procedure",
                    "family_threshold",
                    "family_guarantee",
                    "family_nominal_alpha",
                    "family_size",
                )
                for alpha in requests:
                    if alpha > current_alpha:
                        with pytest.raises(CodedError) as raised:
                            relevel(current, alpha=alpha)
                        assert raised.value.code == "results.operation.alpha_above_allocation"
                        continue
                    widened = relevel(current, alpha=alpha)
                    assert widened.identity == original.identity
                    assert widened.family == original.family
                    assert widened.provenance == original.provenance
                    assert widened.components["sampling"][0] == original_status
                    _assert_interval_contains(widened.summary, original.summary)
                    reissued = widened.retained_row()
                    assert all(
                        getattr(reissued, field) == getattr(original_row, field)
                        for field in family_fields
                    )
                    next_alpha = _retained_relevel_alpha(reissued)
                    assert next_alpha is not None
                    current = widened
                    current_alpha = next_alpha

    check_sequence()


@pytest.mark.integration
def test_unavailable_winsor_projection_retains_hypothesis_and_plan_metadata():
    import pandas as pd

    from increment import Analysis, AnalysisPlan, MetricSpec
    from increment.estimation.results import LiftEstimate
    from increment.result_envelope import AnalysisEnvelope, ResultCell, render
    from increment.semantics.design import Randomized
    from increment.tables import readout_table

    data = pd.DataFrame(
        [
            {
                "unit": f"{group}-{index}",
                "arm": group,
                "plain": float(index + (group == "treatment")),
                "winsor": float(index * 2 + 1 + (group == "treatment")),
            }
            for group in ("control", "treatment")
            for index in range(1, 41)
        ]
    )
    rows = Analysis.from_unit_summary(
        data,
        unit="unit",
        group="arm",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics=[
            MetricSpec(name="plain", type="mean"),
            MetricSpec(
                name="winsor",
                type="mean",
                winsorization={
                    "upper_percentile": 0.75,
                    "inference": {"method": "influence-normal-v1"},
                },
            ),
        ],
        plan=AnalysisPlan(alpha=0.05, primary=("plain", "winsor")),
    ).run()
    winsor = next(item for item in rows if item.metric == "winsor")
    assert isinstance(winsor, LiftEstimate)
    assert winsor.role == "primary"
    winsor = winsor.model_copy(update={"null_lift": 0.2, "null_abs": 1.0})
    envelope = AnalysisEnvelope(
        (ResultCell.from_row(winsor, rows.metadata),), metadata=rows.metadata
    )
    restored = AnalysisEnvelope.from_arrow(envelope.to_arrow())

    projected = restored.cells[0]._projection_row()
    assert projected.null_lift == pytest.approx(0.2)
    assert projected.null_abs == pytest.approx(1.0)
    assert projected.role == "primary"
    assert projected.multiplicity_status == "declared_plan"
    table = readout_table(restored)._repr_html_()
    assert "Primary" in table
    assert "declared_plan" in render(restored, "full")


@pytest.mark.integration
def test_result_envelope_arrow_round_trip_preserves_cell_and_readout_metadata(
    seeded_con, seeded_defs
):
    from increment import Analysis, Method
    from increment.breakout.estimates import LiftEstimates
    from increment.result_envelope import AnalysisEnvelope, ResultCell

    rows = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con).run(
        decision_method=Method(name="unadjusted")
    )
    assert isinstance(rows, LiftEstimates)
    cells = [cell for row in rows for cell in row.as_cells(metadata=rows.metadata)]
    envelope = AnalysisEnvelope(cells=tuple(cells), metadata=rows.metadata)
    from dataclasses import replace

    from increment.result_envelope import ComponentStatus

    statuses = dict(cells[0].components)
    statuses["sequential"] = (
        ComponentStatus(
            "unavailable",
            code="sequential.not_registered",
            reason="registration missing",
            context={"reason": "alpha unavailable"},
        ),
        None,
    )
    amended = replace(cells[0], components=statuses)
    unavailable = ResultCell.from_arrow(ResultCell.to_arrow([amended]))[0]
    assert unavailable.components["sequential"][0] == amended.components["sequential"][0]
    arrow = envelope.to_arrow()
    restored = AnalysisEnvelope.from_arrow(arrow)
    assert restored == AnalysisEnvelope.from_payload(
        json.loads(json.dumps(envelope.to_payload(include_capture_metadata=True)))
    )
    metadata = rows.metadata
    assert metadata is not None
    live_metadata = type(metadata).model_validate(
        {
            "scope": metadata.scope,
            "cells": metadata.cells,
            "partial": metadata.partial,
            "partial_reason": metadata.partial_reason,
            "coherence": "live",
            "effective_modes": {"fact": "live"},
        }
    )
    live_envelope = AnalysisEnvelope(cells=tuple(cells), metadata=live_metadata)
    restored_live = AnalysisEnvelope.from_arrow(live_envelope.to_arrow())
    restored_metadata = restored_live.metadata
    assert restored_metadata is not None
    assert restored_metadata.coherence == "live"
    assert restored_metadata.effective_modes == {"fact": "live"}
    from increment.result_envelope import render

    assert "Coherence: live" in render(restored_live, "brief")

    assert render(restored, "brief") == render(envelope, "brief")
    unscoped = AnalysisEnvelope(cells=tuple(cells), metadata=None)
    assert "integrity.unknown" in render(unscoped, "brief")


@pytest.mark.integration
def test_arrow_decoder_validates_construction_tag_and_row_consistency(seeded_con, seeded_defs):
    import pyarrow as pa

    from increment import Analysis, Method
    from increment.breakout.estimates import LiftEstimates
    from increment.errors import CodedError
    from increment.result_envelope import ResultCell

    rows = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con).run(
        decision_method=Method(name="unadjusted")
    )
    assert isinstance(rows, LiftEstimates)
    row = next(item for item in rows if item.metric == "avg_session_duration")
    cell = row.as_cells(metadata=rows.metadata)[0]
    source = ResultCell.to_arrow([cell])
    for replacement in ("unknown-construction-v9", "binomial"):
        entries = source.to_pylist()
        entries[0]["sampling"]["construction"] = replacement
        altered = pa.Table.from_pylist(entries, schema=source.schema)
        with pytest.raises(CodedError) as raised:
            ResultCell.from_arrow(altered)
        assert raised.value.code == "results.cell.schema_unsupported"


def _assert_sequential_family_evidence_survives_codecs(cell, row):
    from increment.estimation.decision_types import AsymptoticSequentialEvidence
    from increment.estimation.family import e_bh_select
    from increment.result_envelope import ResultCell, combine_family_evidence, family_evidence

    evidence_cells = (
        ResultCell.from_payload(json.loads(json.dumps(cell.to_payload()))),
        ResultCell.from_arrow(ResultCell.to_arrow([cell]))[0],
    )
    expected = set(e_bh_select([row.sequential_result.log_e], 0.05))
    for restored_cell in evidence_cells:
        (stored_evidence,) = (family_evidence(restored_cell),)
        assert isinstance(stored_evidence.evidence, AsymptoticSequentialEvidence)
        selected = combine_family_evidence((stored_evidence,), procedure="e_bh", level=0.05)
        assert bool(selected) is bool(expected)


def test_asymptotic_sequential_component_retains_checkpoint_and_registration_alpha():
    from increment.estimation.sequential_result import AsymptoticSequentialResult
    from increment.frame import MetricSpec
    from increment.result_envelope import AnalysisEnvelope, Operation, ResultCell, render
    from tests.test_asymptotic_public_sources import _analysis, _frame, _plan

    metric = MetricSpec(name="outcome", type="mean")
    row = _analysis(
        _frame([1, 2, 3, 4] * 20, [6, 7, 8, 9] * 20),
        [metric],
        _plan([metric]),
    ).run(estimands=("itt",))[0]
    assert isinstance(row.sequential_result, AsymptoticSequentialResult)
    cell = row.as_cells()[0]
    restored = ResultCell.from_payload(json.loads(json.dumps(cell.to_payload())))
    sequential_state = restored.components["sequential"][1]
    assert sequential_state is not None
    decoded = ResultCell.from_payload(json.loads(json.dumps(restored.to_payload())))
    decoded_row = decoded.retained_row()
    assert decoded_row.sequential_result is not None
    assert row.sequential_result is not None
    assert decoded_row.sequential_result.checkpoint == row.sequential_result.checkpoint
    assert decoded_row.sequential_result.decision_alpha == row.sequential_result.decision_alpha
    _assert_sequential_family_evidence_survives_codecs(cell, row)
    assert decoded.supports["sequential"] == frozenset()
    assert Operation.RELEVEL not in decoded.supports["sequential"]
    assert decoded.summary is not None
    reference = json.loads(
        (Path(__file__).parent / "fixtures" / "result_envelope_reference.json").read_text()
    )["reference"]["sequential/outcome"]
    assert decoded.summary.lower == pytest.approx(reference["lower"])
    assert decoded.summary.upper == pytest.approx(reference["upper"])
    assert decoded.summary.lower == pytest.approx(float(row.sequential_result.bounds.lower) - 1)
    assert decoded.summary.upper == pytest.approx(float(row.sequential_result.bounds.upper) - 1)
    assert decoded.summary.alpha == pytest.approx(reference["decision_alpha"])
    assert "asymptotic sequential" in render(AnalysisEnvelope((decoded,), metadata=None), "brief")
    from increment.estimation.results import LiftEstimate

    unavailable = LiftEstimate(
        metric=row.metric,
        group_id=row.group_id,
        method=row.method,
        method_role=row.method_role,
        estimand=row.estimand,
        analysis_population="triggered",
        value_scale=row.value_scale,
        alternative=row.alternative,
        reference_kind="sequential",
        scale="linear",
        inference="asymptotic_mean",
        role=row.role,
        lift=None,
        source_snapshot_id=row.source_snapshot_id,
        failure_code="readout.cell.unsupported_request",
        failure_context={"reason": "triggered_chain_uncommitted"},
        sampling_available=False,
        sampling_reason_code="readout.cell.unsupported_request",
        sampling_reason_context={"reason": "triggered_chain_uncommitted"},
        decision_scope_complete=False,
    )
    unavailable_cell = ResultCell.from_row(unavailable)
    from increment.result_envelope import family_evidence

    triggered_round_trips = (
        ResultCell.from_payload(json.loads(json.dumps(unavailable_cell.to_payload()))),
        ResultCell.from_arrow(ResultCell.to_arrow([unavailable_cell]))[0],
    )
    for retained in triggered_round_trips:
        evidence = family_evidence(retained)
        assert evidence.evidence is None
        assert evidence.reason_code == "readout.cell.unsupported_request"
    output = render(AnalysisEnvelope((unavailable_cell,), metadata=None), "full")
    assert "triggered_chain_uncommitted" in output
    from fractions import Fraction

    from tests.estimation.test_asymptotic_mean import _evaluate

    disconnected, _ = _evaluate([-2, 1, 2] * 40, [10, 12] * 40, alpha=Fraction(1, 10))
    disconnected_cell = disconnected.as_cells()[0]
    assert disconnected.sequential_result is not None
    assert disconnected.sequential_result.bounds.status == "disconnected"
    assert disconnected_cell.summary is not None
    assert disconnected_cell.summary.lower is None
    assert disconnected_cell.summary.upper is None
    assert "disconnected" in (disconnected_cell.summary.reason or "")
    from increment.result_envelope import AnalysisEnvelope, render

    rendered = render(AnalysisEnvelope((disconnected_cell,), metadata=None))
    assert "disconnected" in rendered
    import re

    from increment.breakout.estimates import LiftEstimates
    from increment.result_envelope import AnalysisEnvelope
    from increment.tables import readout_table

    legacy_rows = LiftEstimates([row])
    envelope = AnalysisEnvelope(tuple(row.as_cells()), metadata=None)
    normalized_envelope = re.sub(
        r'(id="|#|ct-)[a-z]{10}', r"\1<generated-id>", readout_table(envelope)._repr_html_()
    )
    normalized_rows = re.sub(
        r'(id="|#|ct-)[a-z]{10}', r"\1<generated-id>", readout_table(legacy_rows)._repr_html_()
    )
    assert normalized_envelope == normalized_rows


@pytest.mark.integration
def test_envelope_table_projection_matches_legacy_readout(seeded_con, seeded_defs):
    from increment import Analysis, Method
    from increment.breakout.estimates import LiftEstimates
    from increment.result_envelope import AnalysisEnvelope
    from increment.tables import readout_table

    rows = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con).run(
        decision_method=Method(name="unadjusted")
    )
    assert isinstance(rows, LiftEstimates)
    cells = rows.as_cells()
    envelope = AnalysisEnvelope(cells=tuple(cells), metadata=rows.metadata)
    import re

    actual = re.sub(
        r'(id="|#|ct-)[a-z]{10}', r"\1<generated-id>", readout_table(envelope)._repr_html_()
    )
    legacy = re.sub(
        r'(id="|#|ct-)[a-z]{10}', r"\1<generated-id>", readout_table(rows)._repr_html_()
    )
    assert actual == legacy


@pytest.mark.integration
def test_binomial_set_only_table_projection_matches_legacy(seeded_con):
    import re

    import pandas as pd

    from increment import Analysis, AnalysisPlan, MetricSpec
    from increment.breakout.estimates import LiftEstimates
    from increment.estimation.results import LiftEstimate
    from increment.result_envelope import AnalysisEnvelope
    from increment.semantics.design import Randomized
    from increment.tables import readout_table

    data = pd.DataFrame(
        [
            {"unit": f"{group}-{index}", "arm": group, "value": float(success)}
            for group, outcomes in (
                ("control", [0] * 40),
                ("treatment", [0] * 20 + [1] * 20),
            )
            for index, success in enumerate(outcomes)
        ]
    )
    results = Analysis.from_unit_summary(
        data,
        unit="unit",
        group="arm",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics=[MetricSpec(name="value", type="conversion")],
        plan=AnalysisPlan(primary="value"),
    ).run()
    assert isinstance(results, LiftEstimates)
    row = results[0]
    assert isinstance(row, LiftEstimate)
    assert row.lift is None and row.binomial_set is not None
    assert not row.binomial_set.point_available
    envelope = AnalysisEnvelope(tuple(results.as_cells()), results.metadata)
    actual = re.sub(
        r'(id="|#|ct-)[a-z]{10}', r"\1<generated-id>", readout_table(envelope)._repr_html_()
    )
    legacy = re.sub(
        r'(id="|#|ct-)[a-z]{10}', r"\1<generated-id>", readout_table(results)._repr_html_()
    )
    assert actual == legacy


@pytest.mark.integration
def test_collection_and_envelope_headlines_share_relative_lift_value():
    import pandas as pd

    from increment import Analysis, MetricSpec
    from increment.breakout.estimates import LiftEstimates
    from increment.estimation.results import LiftEstimate
    from increment.result_envelope import AnalysisEnvelope, ResultCell, render
    from increment.semantics.design import Randomized
    from increment.tables import readout_table

    data = pd.DataFrame(
        [
            {"unit": f"{group}-{index}", "arm": group, "value": float(index + offset)}
            for group, offset in (("control", 0), ("treatment", 2))
            for index in range(8)
        ]
    )
    rows = Analysis.from_unit_summary(
        data,
        unit="unit",
        group="arm",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics=[MetricSpec(name="value", type="mean")],
    ).run()
    assert isinstance(rows, LiftEstimates)
    row = rows[0]
    assert isinstance(row, LiftEstimate)
    assert row.lift is not None
    collection = readout_table(rows)._repr_html_()
    cell = ResultCell.from_arrow(ResultCell.to_arrow(row.as_cells(metadata=rows.metadata)))[0]
    envelope = render(AnalysisEnvelope((cell,), rows.metadata), "brief")
    assert cell.summary is not None
    assert cell.summary.value_scale == row.value_scale == "relative"
    assert cell.summary.point == pytest.approx(row.lift.value)
    assert f"{row.lift.value * 100:.1f}%" in collection
    assert str(row.lift.value) in envelope


@pytest.mark.integration
def test_conversion_point_and_binomial_interval_survive_envelope_arrow_round_trip():
    import pandas as pd

    from increment import Analysis, AnalysisPlan, MetricSpec
    from increment.breakout.estimates import LiftEstimates
    from increment.estimation.results import LiftEstimate
    from increment.result_envelope import ResultCell
    from increment.semantics.design import Randomized

    data = pd.DataFrame(
        [
            {"unit": f"{group}-{index}", "arm": group, "converted": 1.0}
            for group in ("control", "treatment")
            for index in range(40)
        ]
    )
    rows = Analysis.from_unit_summary(
        data,
        unit="unit",
        group="arm",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics=[MetricSpec(name="converted", type="conversion")],
        plan=AnalysisPlan(primary="converted"),
    ).run()
    assert isinstance(rows, LiftEstimates)
    row = rows[0]
    assert isinstance(row, LiftEstimate)
    cell = ResultCell.from_arrow(ResultCell.to_arrow(row.as_cells(metadata=rows.metadata)))[0]
    assert cell.summary is not None
    assert cell.summary.point == pytest.approx(0.0)
    assert cell.summary.lower == pytest.approx(-0.3230467396656616)
    assert cell.summary.upper == pytest.approx(0.26740011633364064)
    assert cell.summary.value_scale == "relative"


@pytest.mark.integration
def test_additive_only_unavailable_table_projection_matches_legacy():
    import re

    import pandas as pd

    from increment import Analysis, AnalysisPlan, MetricSpec
    from increment.breakout.estimates import LiftEstimates
    from increment.estimation.results import LiftEstimate
    from increment.result_envelope import AnalysisEnvelope
    from increment.semantics.design import Randomized
    from increment.tables import readout_table

    data = pd.DataFrame(
        [
            {"unit": f"{group}-{index}", "arm": group, "value": value}
            for group, values in (
                ("control", [-3.0, -2.0, -1.0, -4.0]),
                ("treatment", [-2.0, -1.0, 0.0, -3.0]),
            )
            for index, value in enumerate(values)
        ]
    )
    results = Analysis.from_unit_summary(
        data,
        unit="unit",
        group="arm",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics=[MetricSpec(name="value", type="mean")],
        plan=AnalysisPlan(primary="value"),
    ).run()
    assert isinstance(results, LiftEstimates)
    row = results[0]
    assert isinstance(row, LiftEstimate)
    assert row.lift is None and row.abs_diff is not None and row.abs_lb is not None
    envelope = AnalysisEnvelope(tuple(results.as_cells()), results.metadata)
    assert envelope.cells[0].summary is not None
    assert envelope.cells[0].summary.value_scale == "absolute"
    assert envelope.cells[0].summary.qualification is not None
    assert "requested relative scale unavailable" in envelope.cells[0].summary.qualification
    from increment.result_envelope import render

    rendered = render(envelope, "full")
    assert "absolute" in rendered
    assert "confidence" in rendered
    actual = re.sub(
        r'(id="|#|ct-)[a-z]{10}', r"\1<generated-id>", readout_table(envelope)._repr_html_()
    )
    legacy = re.sub(
        r'(id="|#|ct-)[a-z]{10}', r"\1<generated-id>", readout_table(results)._repr_html_()
    )
    assert actual == legacy


def test_result_cell_schema_is_versioned_and_structured():
    from increment.result_envelope import ResultCell

    schema = ResultCell.to_arrow([]).schema
    assert schema.names == [
        "schema_version",
        "row_kind",
        "identity",
        "label",
        "family",
        "source_snapshot_id",
        "provenance",
        "summary",
        "point",
        "lower",
        "upper",
        "alpha",
        "se",
        "df",
        "reason_code",
        "retained_row",
        "row_payload",
        "projection_fields",
        "sampling",
        "posterior",
        "sequential",
    ]
    assert str(schema.field("schema_version").type) == "int32"
    assert schema.field("sampling").type.names == [
        "status",
        "code",
        "reason",
        "context",
        "advisories",
        "construction",
        "kind",
        "state_reason",
        "supports",
    ]
    golden = json.loads(
        (
            Path(__file__).parent / "goldens" / "decision_wire" / "result_envelope_v3.schema.json"
        ).read_text()
    )
    actual = {field.name: {"type": str(field.type), "nullable": field.nullable} for field in schema}
    assert actual == golden


@pytest.mark.integration
def test_relative_log_statistics_survive_json_and_arrow_round_trips(seeded_con, seeded_defs):
    from increment import Analysis, Method
    from increment.breakout.estimates import LiftEstimates
    from increment.result_envelope import ResultCell, relevel

    rows = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con).run(
        decision_method=Method(name="unadjusted")
    )

    assert isinstance(rows, LiftEstimates)
    row = next(item for item in rows if item.metric == "avg_session_duration")
    assert row.lift is not None
    assert row.lift.log_mean is not None and row.lift.log_se is not None
    changed_lift = row.lift.model_copy(
        update={
            "log_mean": row.lift.log_mean + 0.2,
            "log_se": row.lift.log_se * 1.5,
        }
    )
    changed = row.model_copy(update={"lift": changed_lift})
    originals = [
        row.as_cells(metadata=rows.metadata)[0],
        changed.as_cells(metadata=rows.metadata)[0],
    ]
    assert originals[0].retained_row().abs_diff == originals[1].retained_row().abs_diff

    json_cells = [ResultCell.from_payload(cell.to_payload()) for cell in originals]
    arrow_cells = ResultCell.from_arrow(ResultCell.to_arrow(originals))
    for cells in (json_cells, arrow_cells):
        replayed = [relevel(cell, alpha=0.02) for cell in cells]
        replayed_rows = [item.retained_row() for item in replayed]
        assert all(
            item.summary == replayed_row.as_cells()[0].summary
            for item, replayed_row in zip(replayed, replayed_rows, strict=True)
        )
        assert replayed_rows[0].lift is not None and replayed_rows[1].lift is not None
        assert replayed_rows[0].lift.log_mean != replayed_rows[1].lift.log_mean
        assert replayed_rows[0].lift.log_se != replayed_rows[1].lift.log_se


@pytest.mark.integration
def test_posterior_qualified_table_projection_matches_legacy(seeded_con, seeded_defs):
    import re
    import warnings

    from increment import Analysis, Method
    from increment.breakout.estimates import LiftEstimates
    from increment.estimation.inference import Normal
    from increment.result_envelope import AnalysisEnvelope
    from increment.tables import readout_table

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con).run(
            decision_method=Method(name="unadjusted"), prior=Normal(mu=0.0, sigma=0.1)
        )
    assert isinstance(rows, LiftEstimates)
    assert all(item.posterior_available is True for item in rows)
    from dataclasses import replace

    from increment.result_envelope import ComponentStatus, ResultCell

    row = next(item for item in rows if item.metric == "avg_session_duration")
    cell = row.as_cells(metadata=rows.metadata)[0]
    posterior_state = cell.components["posterior"][1]
    assert posterior_state is not None
    components = dict(cell.components)
    components["posterior"] = (
        ComponentStatus(
            "advisory",
            code="posterior.retained_advisory",
            reason="retained posterior was reviewed",
            context={"metric": "avg_session_duration"},
            advisories=("posterior.retained",),
        ),
        posterior_state,
    )
    advisory_cell = replace(cell, components=components)
    restored_cell = ResultCell.from_arrow(ResultCell.to_arrow([advisory_cell]))[0]
    assert restored_cell.components["posterior"][0] == advisory_cell.components["posterior"][0]
    assert restored_cell.components["posterior"][1] == posterior_state
    envelope = AnalysisEnvelope(tuple(rows.as_cells()), rows.metadata)
    actual = re.sub(
        r'(id="|#|ct-)[a-z]{10}',
        r"\1<generated-id>",
        readout_table(envelope, advisory=True)._repr_html_(),
    )
    legacy = re.sub(
        r'(id="|#|ct-)[a-z]{10}',
        r"\1<generated-id>",
        readout_table(rows, advisory=True)._repr_html_(),
    )
    assert actual == legacy


@pytest.mark.integration
def test_directional_family_selection_retains_effective_alpha_and_advisory(seeded_con, seeded_defs):
    from dataclasses import replace

    from increment import Analysis, Method
    from increment.breakout.estimates import LiftEstimates
    from increment.estimation.family import select_exploratory_family
    from increment.estimation.results import LiftEstimate
    from increment.result_envelope import ComponentStatus, Operation, ResultCell

    rows = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con).run(
        decision_method=Method(name="unadjusted")
    )
    assert isinstance(rows, LiftEstimates)
    row = next(item for item in rows if item.metric == "avg_session_duration")
    assert row.lift is not None
    clear_family = {
        "discovery": None,
        "family_id": None,
        "family_q": None,
        "family_size": None,
        "family_threshold": None,
        "family_axes": None,
        "family_guarantee": None,
        "family_procedure": None,
        "family_nominal_alpha": None,
        "multiplicity_status": "exploratory_unadjusted",
        "role": "exploratory",
    }
    directional = row.model_copy(
        update={
            **clear_family,
            "alternative": "greater",
            "lift": row.lift.model_copy(update={"alpha": 0.05}),
        }
    )
    corrected = select_exploratory_family([directional], q=0.03)[0]
    assert isinstance(corrected, LiftEstimate)
    cell = ResultCell.from_row(corrected)
    components = dict(cell.components)
    components["sampling"] = (
        ComponentStatus(
            "advisory",
            code="sampling.retained_advisory",
            reason="reviewed estimate",
            context={"source": "review"},
            advisories=("sampling.retained_advisory",),
        ),
        components["sampling"][1],
    )
    cell = replace(cell, components=components)
    corrected_cell = ResultCell.from_payload(json.loads(json.dumps(cell.to_payload())))
    corrected = corrected_cell.retained_row()
    assert corrected_cell.summary is not None
    assert corrected_cell.components["sampling"][0].state == "advisory"
    assert Operation.RELEVEL not in corrected_cell.supports["sampling"]
    assert corrected_cell.components["sampling"][0].advisories == ("sampling.retained_advisory",)
    assert corrected.lift is not None
    assert corrected.lift.open_side is not None
    assert corrected.lift.alpha == pytest.approx(0.025)


@pytest.mark.integration
def test_unsupported_sampling_constructions_refuse_without_losing_reason(seeded_con, seeded_defs):
    import copy
    import pickle

    from increment import Analysis, Method
    from increment.breakout.estimates import LiftEstimates
    from increment.errors import CodedError
    from increment.estimation.armstats import ArmStats
    from increment.estimation.inference import infer_independent_mean
    from increment.estimation.winsor import estimate_winsor_lift
    from increment.result_envelope import ResultCell, compare_margin, relevel
    from increment.winsor import RawArm, WinsorInferenceSpec, WinsorRawState

    rows = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con).run(
        decision_method=Method(name="unadjusted")
    )
    assert isinstance(rows, LiftEstimates)
    original = next(item for item in rows if item.metric == "avg_session_duration")
    quantile = original.model_copy(update={"quantile_p_value": 0.01})
    independent_mean = infer_independent_mean(
        ArmStats(
            study_id="independent_study",
            metric="independent_metric",
            group_id="control",
            n=3,
            ref_y=1.0,
            cy1=0.0,
            cy2=2.0,
        ),
        ArmStats(
            study_id="independent_study",
            metric="independent_metric",
            group_id="treatment",
            n=3,
            ref_y=2.0,
            cy1=0.0,
            cy2=2.0,
        ),
    )
    raw = WinsorRawState(
        metric="winsor_metric",
        study_id="winsor_study",
        missingness="none",
        quantile=0.5,
        inference=WinsorInferenceSpec(method="influence-normal-v1"),
        arms=(
            RawArm(group_id="control", values=tuple(float(i) for i in range(1, 25))),
            RawArm(group_id="treatment", values=tuple(float(i + 1) for i in range(1, 25))),
        ),
    )
    winsor = estimate_winsor_lift(raw, "control", "treatment")

    for row, reason in (
        (quantile, "quantile interval (its standard error depends on alpha)"),
        (winsor, "percentile-winsorized confidence set (a single-level construction)"),
        (independent_mean, "estimand 'ate'"),
    ):
        cell = ResultCell.from_row(row)
        assert cell.supports["sampling"] == frozenset()
        with pytest.raises(CodedError) as raised:
            compare_margin(cell, margin=0.0, scale="relative")
        assert raised.value.code == "results.operation.unsupported"
        restored = ResultCell.from_payload(cell.to_payload())
        status = restored.components["sampling"][0]
        if row is winsor:
            assert restored.components["sampling"][1] is None
            assert status.state == "unavailable"
            assert status.reason == reason
        else:
            assert restored.components["sampling"][1] is not None
            assert status.reason == reason
        for retained in (copy.deepcopy(cell), pickle.loads(pickle.dumps(cell))):
            with pytest.raises(CodedError) as raised:
                relevel(retained, alpha=0.03)
            assert raised.value.code == "results.operation.unsupported"
            assert raised.value.context["reason"] == reason
    winsor_cell = ResultCell.from_row(winsor)
    assert winsor_cell.components["sampling"][0].state == "unavailable"
    assert winsor_cell.components["sampling"][0].reason is not None
    assert winsor_cell.components["sampling"][1] is None
    payload = winsor_cell.to_payload()
    assert "raw" not in json.dumps(payload)
    arrow = ResultCell.to_arrow([winsor_cell])
    arrow_row = arrow.to_pylist()[0]
    assert arrow_row["retained_row"] is None


@pytest.mark.integration
def test_dated_unavailable_winsor_projection_matches_arrow_readout_scope():
    from datetime import date

    import pandas as pd

    from increment import Analysis, MetricSpec
    from increment.declarations import CellKey
    from increment.estimation.readout_types import cell_order
    from increment.estimation.results import LiftEstimate
    from increment.result_envelope import AnalysisEnvelope, ResultCell, render
    from increment.semantics.design import Randomized

    data = pd.DataFrame(
        [
            {
                "unit": f"{group}-{index}",
                "arm": group,
                "plain": float(index + (group == "treatment")),
                "winsor": float(index * 2 + 1 + (group == "treatment")),
            }
            for group in ("control", "treatment")
            for index in range(1, 41)
        ]
    )
    rows = Analysis.from_unit_summary(
        data,
        unit="unit",
        group="arm",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics=[
            MetricSpec(name="plain", type="mean"),
            MetricSpec(
                name="winsor",
                type="mean",
                winsorization={
                    "upper_percentile": 0.5,
                    "inference": {"method": "influence-normal-v1"},
                },
            ),
        ],
    ).run()
    assert rows.metadata is not None
    dated_rows = {
        row.metric: row.model_copy(update={"ds": date(2026, 10, 26)})
        for row in rows
        if isinstance(row, LiftEstimate)
    }
    keys = {
        CellKey.from_row(row): CellKey.from_row(dated_rows[row.metric])
        for row in rows
        if isinstance(row, LiftEstimate)
    }

    def dated_key(key):
        return keys.get(key, key)

    scope = rows.metadata.scope
    by_source = {
        source_id: source.model_copy(
            update={
                "cells": tuple(sorted(map(dated_key, source.cells), key=cell_order)),
                "decision_cells": tuple(
                    sorted(map(dated_key, source.decision_cells), key=cell_order)
                ),
            }
        )
        for source_id, source in scope.by_source.items()
    }
    metadata = rows.metadata.model_copy(
        update={
            "scope": scope.model_copy(
                update={
                    "cells": tuple(sorted(map(dated_key, scope.cells), key=cell_order)),
                    "decision_cells": tuple(
                        sorted(map(dated_key, scope.decision_cells), key=cell_order)
                    ),
                    "by_source": by_source,
                }
            ),
            "cells": tuple(
                record.model_copy(update={"cell": dated_key(record.cell)})
                for record in rows.metadata.cells
            ),
        }
    )
    envelope = AnalysisEnvelope(
        tuple(ResultCell.from_row(dated_rows[row.metric], metadata) for row in rows),
        metadata=metadata,
    )
    restored = AnalysisEnvelope.from_arrow(envelope.to_arrow())
    projected = next(
        cell._projection_row() for cell in restored.cells if cell.identity.key.metric == "winsor"
    )
    assert projected.ds == date(2026, 10, 26)
    text = render(restored, "full")
    assert "Hidden by filter winsor" not in text
    assert "percentile-winsorized confidence set" in text


@pytest.mark.integration
def test_envelope_consumers_render_unavailable_winsor_with_valid_sibling(seeded_con, seeded_defs):
    from increment import Analysis, Method
    from increment.breakout.estimates import LiftEstimates
    from increment.estimation.winsor import estimate_winsor_lift
    from increment.result_envelope import AnalysisEnvelope, ResultCell, render
    from increment.tables import readout_table
    from increment.winsor import RawArm, WinsorInferenceSpec, WinsorRawState

    rows = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con).run(
        decision_method=Method(name="unadjusted")
    )
    assert isinstance(rows, LiftEstimates)
    sibling = next(row for row in rows if row.metric == "avg_session_duration")
    target = next(row for row in rows if row.metric != sibling.metric)
    winsor = estimate_winsor_lift(
        WinsorRawState(
            metric="winsor_metric",
            study_id="winsor_study",
            missingness="none",
            quantile=0.5,
            inference=WinsorInferenceSpec(method="influence-normal-v1"),
            arms=(
                RawArm(group_id="control", values=tuple(float(i) for i in range(1, 25))),
                RawArm(group_id="treatment", values=tuple(float(i + 1) for i in range(1, 25))),
            ),
        ),
        "control",
        "treatment",
    )
    identity_fields = (
        "metric",
        "group_id",
        "method",
        "method_role",
        "inference",
        "alternative",
        "estimand",
        "analysis_population",
        "value_scale",
        "source_snapshot_id",
    )
    winsor = winsor.model_copy(update={name: getattr(target, name) for name in identity_fields})
    envelope = AnalysisEnvelope(
        (ResultCell.from_row(sibling), ResultCell.from_row(winsor, rows.metadata)),
        metadata=rows.metadata,
    )
    restored = AnalysisEnvelope.from_arrow(envelope.to_arrow())

    table = readout_table(restored)._repr_html_()
    brief = render(restored, "brief")
    text = render(restored, "full")
    reason = "percentile-winsorized confidence set (a single-level construction)"
    for output in (table, brief, text):
        assert target.metric in output
        assert f"Hidden by filter {target.metric}" not in output
    assert "percentile-winsorized" in table
    assert "status=unavailable" in brief
    assert reason in text


@pytest.mark.integration
def test_posterior_reason_context_survives_metadata_free_cell_creation(seeded_con, seeded_defs):
    from increment import Analysis, Method
    from increment.breakout.estimates import LiftEstimates

    rows = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con).run(
        decision_method=Method(name="unadjusted")
    )
    assert isinstance(rows, LiftEstimates)
    original = next(item for item in rows if item.metric == "avg_session_duration")
    unavailable = original.model_copy(
        update={
            "posterior_components": None,
            "posterior_latent_mean": None,
            "posterior_reason_code": "posterior.fixture_unavailable",
            "posterior_reason_context": {"reason": "fixture absence", "metric": original.metric},
        }
    )
    cell = unavailable.as_cells()[0]
    status, state = cell.components["posterior"]
    assert state is None
    assert status.code == "posterior.fixture_unavailable"
    assert status.context == {"reason": "fixture absence", "metric": original.metric}


@pytest.mark.integration
def test_date_valued_row_fields_survive_json_and_arrow_round_trips(seeded_con, seeded_defs):
    from datetime import date

    from increment import Analysis, Method
    from increment.breakout.estimates import LiftEstimates
    from increment.result_envelope import ResultCell

    rows = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con).run(
        decision_method=Method(name="unadjusted")
    )
    assert isinstance(rows, LiftEstimates)
    row = next(item for item in rows if item.metric == "avg_session_duration")
    dated = row.model_copy(update={"ds": date(2026, 10, 26)})
    cell = dated.as_cells()[0]
    from_json = ResultCell.from_payload(json.loads(json.dumps(cell.to_payload())))
    from_arrow = ResultCell.from_arrow(ResultCell.to_arrow([cell]))[0]
    assert from_json.retained_row().ds == date(2026, 10, 26)
    assert from_arrow.retained_row().ds == date(2026, 10, 26)


@pytest.mark.integration
def test_directional_reinterval_with_posterior_round_trips_through_arrow(seeded_con, seeded_defs):
    import warnings

    from increment import Analysis, Method
    from increment.breakout.estimates import LiftEstimates
    from increment.estimation.inference import Normal
    from increment.result_envelope import ResultCell, relevel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con).run(
            decision_method=Method(name="unadjusted"), prior=Normal(mu=0.0, sigma=0.1)
        )
    assert isinstance(rows, LiftEstimates)
    row = next(item for item in rows if item.metric == "avg_session_duration")
    directional = row.model_copy(update={"alternative": "greater"})
    corrected = relevel(ResultCell.from_row(directional), alpha=0.02)
    arrow = ResultCell.to_arrow([corrected])
    restored = ResultCell.from_arrow(arrow)[0]
    assert restored.retained_row().alternative == "greater"
    assert restored.components["posterior"][1] is not None
    assert restored.components["posterior"][1] == corrected.components["posterior"][1]


@pytest.mark.integration
def test_arrow_ipc_bytes_are_stable_across_fresh_processes(seeded_con, seeded_defs):
    import subprocess
    import sys

    from increment import Analysis, Method
    from increment.breakout.estimates import LiftEstimates
    from increment.result_envelope import AnalysisEnvelope

    analysis = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con)
    rows = analysis.run(decision_method=Method(name="unadjusted"))
    assert isinstance(rows, LiftEstimates)
    breakout_rows = analysis.run_breakout()
    envelope = AnalysisEnvelope(
        cells=(*rows.as_cells(), *breakout_rows.as_cells()),
        metadata=None,
    )
    source = """
import json, sys
import pyarrow as pa
from increment.result_envelope import AnalysisEnvelope
envelope = AnalysisEnvelope.from_payload(json.load(sys.stdin))
table = envelope.to_arrow()
sink = pa.BufferOutputStream()
with pa.ipc.new_stream(sink, table.schema) as writer:
    writer.write_table(table)
sys.stdout.buffer.write(sink.getvalue().to_pybytes())
"""
    payload = json.dumps(envelope.to_payload()).encode()
    outputs = [
        subprocess.run(
            [sys.executable, "-c", source],
            input=payload,
            capture_output=True,
            check=True,
            cwd=Path(__file__).resolve().parents[1],
        ).stdout
        for _ in range(2)
    ]
    assert outputs[0] == outputs[1]


def test_prechange_public_interval_reference_is_frozen():
    reference = json.loads(
        (Path(__file__).parent / "fixtures" / "result_envelope_reference.json").read_text()
    )
    assert reference["provenance"].startswith(
        "Captured 2026-10-26 from pre-existing public Analysis.run()"
    )
    assert reference["reference"]["mean/avg_session_duration"]["alpha"] == 0.02


@pytest.mark.integration
def test_family_threshold_reinterval_matches_frozen_public_reference(seeded_con, seeded_defs):
    from increment import Analysis, Method
    from increment.breakout.estimates import LiftEstimates
    from increment.estimation.family import select_exploratory_family
    from increment.result_envelope import ResultCell

    rows = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con).run(
        decision_method=Method(name="unadjusted")
    )
    assert isinstance(rows, LiftEstimates)
    reference = json.loads(
        (Path(__file__).parent / "fixtures" / "result_envelope_reference.json").read_text()
    )["reference"]
    clear_family = {
        "discovery": None,
        "family_id": None,
        "family_q": None,
        "family_size": None,
        "family_threshold": None,
        "family_axes": None,
        "family_guarantee": None,
        "family_procedure": None,
        "family_nominal_alpha": None,
        "multiplicity_status": "exploratory_unadjusted",
        "role": "exploratory",
    }
    for metric, key in (
        ("avg_session_duration", "family_threshold/mean/avg_session_duration"),
        ("purchase_rate", "family_threshold/binomial/purchase_rate"),
    ):
        row = next(item for item in rows if item.metric == metric)
        legacy = select_exploratory_family([row.model_copy(update=clear_family)], q=0.03)[0]
        expected = reference[key]
        assert legacy.lift is not None
        assert legacy.lift.alpha == pytest.approx(expected["alpha"])
        assert legacy.lift.lb == pytest.approx(expected["lower"])
        assert legacy.lift.ub == pytest.approx(expected["upper"])
        restored = ResultCell.from_payload(
            json.loads(json.dumps(ResultCell.from_row(row, rows.metadata).to_payload()))
        )
        actual = select_exploratory_family(
            [restored.retained_row().model_copy(update=clear_family)], q=0.03
        )[0]
        assert actual.lift is not None
        assert actual.lift.alpha == pytest.approx(expected["alpha"])
        assert actual.lift.lb == pytest.approx(expected["lower"])
        assert actual.lift.ub == pytest.approx(expected["upper"])


@pytest.mark.integration
def test_composite_readout_metadata_survives_arrow_and_brief_render():
    import warnings

    import pandas as pd

    from increment import Analysis, AnalysisPlan, MetricSpec
    from increment.result_envelope import AnalysisEnvelope, render
    from increment.semantics.design import Randomized

    def frame(n_control: int, n_treatment: int, *, constant_guard: bool = False):
        return pd.DataFrame(
            [
                {
                    "unit": f"{group}-{index}",
                    "arm": group,
                    "value": float(index % 11 + (0.2 if group == "treatment" else 0.0)),
                    "guard": 1.0 if constant_guard else float(index % 7 + 1),
                }
                for group, count in (("control", n_control), ("treatment", n_treatment))
                for index in range(count)
            ]
        )

    def run(
        data,
        *,
        allocation: bool = True,
        guardrail: bool = False,
    ):
        design = Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5} if allocation else None,
            allocation_scheme="independent" if allocation else None,
        )
        metrics = [MetricSpec(name="value", type="mean")]
        plan = AnalysisPlan(primary="value", guardrails=["guard"] if guardrail else [])
        if guardrail:
            metrics.append(MetricSpec(name="guard", type="mean", preferred_direction="decrease"))
        return Analysis.from_unit_summary(
            data, unit="unit", group="arm", design=design, metrics=metrics, plan=plan
        ).run()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        guarded = run(frame(100, 100, constant_guard=True), guardrail=True).filter(
            lambda row: row.metric == "value"
        )
    failed_srm = run(frame(900, 100))
    unchecked = run(frame(100, 100), allocation=False)
    combined = guarded.concat(failed_srm).concat(unchecked)
    envelope = AnalysisEnvelope(tuple(combined.as_cells()), combined.metadata)
    restored = AnalysisEnvelope.from_arrow(envelope.to_arrow())
    assert restored.metadata == envelope.metadata
    assert render(restored, "brief") == render(envelope, "brief")
    output = render(restored, "brief")
    assert "integrity.failed" in output
    assert envelope.metadata is not None
    source_ids = tuple(envelope.metadata.scope.by_source)
    headlines = [line for line in output.splitlines() if line.startswith("value ")]
    assert len(headlines) == len(envelope.cells)
    assert all(any(source_id[:12] in line for line in headlines) for source_id in source_ids)
    failed_source_id = next(
        source_id
        for source_id, source in envelope.metadata.scope.by_source.items()
        if any(item.status == "failed" for item in source.integrity)
    )
    assert any(
        "integrity.failed" in line and failed_source_id[:12] in line for line in output.splitlines()
    )
    visible_primary = run(frame(100, 100), guardrail=True).filter(lambda row: row.metric == "value")
    primary_envelope = AnalysisEnvelope(tuple(visible_primary.as_cells()), visible_primary.metadata)
    for audience in ("brief", "full"):
        output = render(primary_envelope, audience)
        assert output.count("Warning: readout.hidden_required_cell:guard") == 1
        assert "Warning: readout.partial_view" in output


@pytest.mark.integration
def test_render_labels_disambiguate_same_metric_cells(seeded_con, seeded_defs):
    from dataclasses import replace

    from increment import Analysis, Method
    from increment.breakout.estimates import LiftEstimates
    from increment.result_envelope import AnalysisEnvelope, render

    rows = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con).run(
        decision_method=Method(name="unadjusted")
    )
    assert isinstance(rows, LiftEstimates)
    row = next(item for item in rows if item.metric == "avg_session_duration")
    unadjusted_cell = row.as_cells()[0]
    cuped_key = unadjusted_cell.identity.key.model_copy(update={"method": "cuped"})
    cuped_cell = replace(
        unadjusted_cell,
        identity=unadjusted_cell.identity.model_copy(update={"key": cuped_key}),
    )
    output = render(AnalysisEnvelope((unadjusted_cell, cuped_cell), None))
    assert "arm=treatment" in output
    assert "method=unadjusted" in output
    assert "method=cuped" in output
    assert "role=decision" in output


def test_render_keeps_retained_comparison_identity_after_arrow_round_trip():
    import pandas as pd

    from increment import Analysis, MetricSpec
    from increment.breakout.estimates import LiftEstimates
    from increment.result_envelope import AnalysisEnvelope, render
    from increment.semantics.design import Randomized

    data = pd.DataFrame(
        [
            {"unit": f"{group}-{index}", "arm": group, "value": float(index + offset)}
            for group, offset in (("holdout", 0), ("treatment", 2))
            for index in range(6)
        ]
    )
    rows = Analysis.from_unit_summary(
        data,
        unit="unit",
        group="arm",
        design=Randomized(
            control_group="holdout",
            allocation={"holdout": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics=[MetricSpec(name="value", type="mean")],
    ).run()
    assert isinstance(rows, LiftEstimates)
    envelope = AnalysisEnvelope(tuple(rows.as_cells()), rows.metadata)
    restored = AnalysisEnvelope.from_arrow(envelope.to_arrow())
    headline = render(restored).splitlines()[0]
    assert "holdout" in headline
    assert "control identity unknown" not in headline


def test_relevel_uses_available_absolute_interval_when_relative_tails_unresolve():
    import pandas as pd

    from increment import Analysis, AnalysisPlan, MetricSpec
    from increment.breakout.estimates import LiftEstimates
    from increment.estimation.results import LiftEstimate
    from increment.result_envelope import ResultCell, relevel
    from increment.semantics.design import Randomized
    from increment.semantics.models import ExperimentMetric

    data = pd.DataFrame(
        [
            {"unit": f"{group}-{index}", "arm": group, "value": float(index + offset)}
            for group, offset in (("control", 0), ("treatment", 3))
            for index in range(1, 7)
        ]
    )
    rows = Analysis.from_unit_summary(
        data,
        unit="unit",
        group="arm",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics=[MetricSpec(name="value", type="mean", preferred_direction="increase")],
        plan=AnalysisPlan(guardrails=[ExperimentMetric(metric="value", margin=0.2)]),
    ).run()
    assert isinstance(rows, LiftEstimates)
    assert isinstance(rows[0], LiftEstimate)
    cell = ResultCell.from_row(rows[0])
    json_cell = ResultCell.from_payload(json.loads(json.dumps(cell.to_payload())))
    arrow_cell = ResultCell.from_arrow(ResultCell.to_arrow([cell]))[0]
    for restored in (json_cell, arrow_cell):
        refused = relevel(restored, alpha=1e-100)
        assert refused.summary is not None
        assert refused.summary.point is not None
        assert refused.summary.value_scale == "absolute"
        assert refused.summary.lower is not None and refused.summary.upper is not None
        from increment.result_envelope import AnalysisEnvelope, render

        rendered = render(AnalysisEnvelope((refused,), metadata=None), "full")
        assert "null (relative lift)" in rendered


def test_brief_render_has_one_headline_per_metric_and_full_keeps_it():
    import pandas as pd

    from increment import Analysis, MetricSpec
    from increment.breakout.estimates import LiftEstimates
    from increment.result_envelope import AnalysisEnvelope, render
    from increment.semantics.design import Randomized

    data = pd.DataFrame(
        [
            {
                "unit": f"{group}-{index}",
                "arm": group,
                "value": float(index + offset),
                "other": float(index + offset + 1),
            }
            for group, offset in (("control", 0), ("treatment", 2))
            for index in range(8)
        ]
    )
    rows = Analysis.from_unit_summary(
        data,
        unit="unit",
        group="arm",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics=[
            MetricSpec(name="value", type="mean"),
            MetricSpec(name="other", type="mean"),
        ],
    ).run()
    assert isinstance(rows, LiftEstimates)
    envelope = AnalysisEnvelope(tuple(rows.as_cells()), metadata=None)
    brief_lines = render(envelope, "brief").splitlines()
    headline_lines = [line for line in brief_lines if not line.startswith("Warning: ")]
    warning_lines = [line for line in brief_lines if line.startswith("Warning: ")]
    assert len(headline_lines) == len({cell.identity.key.metric for cell in envelope.cells})
    assert warning_lines
    full_lines = set(render(envelope, "full").splitlines())
    assert set(brief_lines) <= full_lines


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_joint_confidence_set_summary_preserves_disconnected_geometry():
    from increment.estimation.results import LiftEstimate
    from increment.result_envelope import ResultCell
    from tests.estimation.test_joint_relative_reference import _shared_joint_from_values

    row = _shared_joint_from_values([(1, 1), (2, -1)], [(1, 2), (2, 0)])
    assert isinstance(row, LiftEstimate)
    confidence_set = row.relative_confidence_set
    assert confidence_set is not None
    cell = ResultCell.from_row(row)
    assert cell.summary is not None
    assert cell.summary.point is None
    assert cell.summary.value_scale == "relative"
    assert cell.summary.alpha == pytest.approx(confidence_set.alpha)
    assert cell.summary.geometry == confidence_set.geometry
    assert cell.summary.intervals == confidence_set.intervals
    assert cell.summary.reason is not None
    payload = json.loads(json.dumps(cell.to_payload()))
    restored = ResultCell.from_payload(payload)
    restored = ResultCell.from_arrow(ResultCell.to_arrow([restored]))[0]
    from increment.result_envelope import AnalysisEnvelope, render

    brief = render(AnalysisEnvelope((restored,), metadata=None), "brief")
    full = render(AnalysisEnvelope((restored,), metadata=None), "full")
    assert "disconnected set" in brief
    assert "zero_denominator" in brief
    assert set(brief.splitlines()) <= set(full.splitlines())


@pytest.mark.filterwarnings("ignore::increment.errors.IncrementWarning")
def test_brief_projection_keeps_selection_status_failure_reason_and_roundtrip_identity():
    from dataclasses import replace

    import pandas as pd

    from increment import Analysis, MetricSpec
    from increment.estimation.family import select_exploratory_family
    from increment.estimation.results import LiftEstimate
    from increment.result_envelope import AnalysisEnvelope, ResultCell, render
    from increment.semantics.design import Randomized

    data = pd.DataFrame(
        [
            {
                "unit": f"{group}-{index}",
                "arm": group,
                "shift": float(index % 11 + (group == "treatment") * 5),
                "steady": float(index % 11),
            }
            for group in ("control", "treatment")
            for index in range(100)
        ]
    )
    rows = Analysis.from_unit_summary(
        data,
        unit="unit",
        group="arm",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics=[MetricSpec(name=name, type="mean") for name in ("shift", "steady")],
    ).run()
    from increment.breakout.estimates import LiftEstimates

    assert isinstance(rows, LiftEstimates)
    lift_rows = [row for row in rows if isinstance(row, LiftEstimate)]
    selected = select_exploratory_family(lift_rows, q=0.05)
    cells = [cell for row in selected if isinstance(row, LiftEstimate) for cell in row.as_cells()]
    dated_key = cells[0].identity.key.model_copy(update={"ds": "2026-10-26"})
    identity = cells[0].identity.model_copy(update={"key": dated_key})
    cells.append(replace(cells[0], identity=identity))
    from increment.estimation.encouragement import estimate_encouragement
    from tests.estimation.test_cuped_late import METRIC, _design, _rows

    encouragement_rows = estimate_encouragement(
        [METRIC], _rows(), _design(), estimands=("itt", "late")
    ).results
    cells.extend(row.as_cells()[0] for row in encouragement_rows if row.estimand in {"itt", "late"})
    unavailable_rows = Analysis.from_unit_summary(
        pd.DataFrame(
            [
                {
                    "unit": f"{group}-{index}",
                    "arm": group,
                    "flat": 1.0,
                    "valid": float(index),
                }
                for group in ("control", "treatment")
                for index in range(20)
            ]
        ),
        unit="unit",
        group="arm",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics=[MetricSpec(name=name, type="mean") for name in ("flat", "valid")],
    ).run()
    assert isinstance(unavailable_rows, LiftEstimates)
    cells.append(
        next(row for row in unavailable_rows if row.metric == "flat").as_cells(
            metadata=unavailable_rows.metadata
        )[0]
    )
    restored = ResultCell.from_arrow(
        ResultCell.to_arrow([ResultCell.from_payload(cell.to_payload()) for cell in cells])
    )
    envelope = AnalysisEnvelope(tuple(restored), metadata=rows.metadata)

    brief = render(envelope, "brief")
    full = render(envelope, "full")
    assert "shift" in brief and "steady" in brief
    assert "selected" in brief and "not selected" in brief
    assert "FCR-adjusted" in brief
    assert "exploratory, not confirmatory" in brief
    assert "flat" in brief and "zero_variance" in brief
    assert "status=available" in brief and "status=unavailable" in brief
    assert set(brief.splitlines()) <= set(full.splitlines())
    assert "estimand=itt" in brief and "estimand=late" in brief
    assert "date=2026-10-26" in brief
    assert brief.count("source=sha256:") > 1


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_brief_joint_set_uses_displayed_level_and_preserves_available_point():
    from increment.result_envelope import AnalysisEnvelope, ResultCell, render
    from tests.estimation.test_joint_relative_reference import _inferred_joint_row

    row = _inferred_joint_row(alternative="greater")
    cell = ResultCell.from_row(row)
    assert cell.summary is not None
    assert cell.summary.point is not None
    assert cell.summary.reason is None
    restored = ResultCell.from_payload(cell.to_payload())
    restored = ResultCell.from_arrow(ResultCell.to_arrow([restored]))[0]

    brief = render(AnalysisEnvelope((restored,), metadata=None), "brief")
    assert "point unavailable" not in brief
    assert "95.0% confidence" in brief
    assert "90.0% confidence" not in brief


def test_cell_rejects_conflicting_source_snapshot_identity():
    import copy

    import pandas as pd

    from increment import Analysis, MetricSpec
    from increment.errors import CodedError
    from increment.result_envelope import ResultCell
    from increment.semantics.design import Randomized

    rows = Analysis.from_unit_summary(
        pd.DataFrame(
            [
                {"unit": f"{group}-{index}", "arm": group, "value": float(index)}
                for group in ("control", "treatment")
                for index in range(8)
            ]
        ),
        unit="unit",
        group="arm",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics=[MetricSpec(name="value", type="mean")],
    ).run()
    cell = rows.as_cells()[0]
    identity = cell.identity.model_copy(update={"source_snapshot_id": "identity-source"})
    with pytest.raises(CodedError) as raised:
        ResultCell.from_report_payload(
            row_kind="sensitivity",
            identity=identity,
            label=cell.label,
            row=rows[0],
            source_snapshot_id="payload-source",
        )
    assert raised.value.code == "results.cell.schema_unsupported"

    payload = copy.deepcopy(cell.to_payload())
    payload["identity"]["source_snapshot_id"] = "identity-source"
    payload["source_snapshot_id"] = "payload-source"
    with pytest.raises(CodedError) as raised:
        ResultCell.from_payload(payload)
    assert raised.value.code == "results.cell.schema_unsupported"


def test_payload_summary_is_rederived_from_retained_lift_row():
    import copy

    import pandas as pd

    from increment import Analysis, MetricSpec
    from increment.result_envelope import ResultCell
    from increment.semantics.design import Randomized

    rows = Analysis.from_unit_summary(
        pd.DataFrame(
            [
                {
                    "unit": f"{group}-{index}",
                    "arm": group,
                    "value": float(index + (group == "treatment")),
                }
                for group in ("control", "treatment")
                for index in range(8)
            ]
        ),
        unit="unit",
        group="arm",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics=[MetricSpec(name="value", type="mean")],
    ).run()
    cell = rows.as_cells()[0]
    assert cell.summary is not None and cell.summary.point is not None
    payload = copy.deepcopy(cell.to_payload())
    payload["summary"]["point"] += 10.0

    restored = ResultCell.from_payload(payload)

    assert restored.summary == cell.summary


def test_prior_bound_unknown_sampling_keeps_summary_unavailable_after_roundtrips():
    import copy
    import json
    import pickle

    import pandas as pd

    from increment import Analysis, MetricSpec
    from increment.estimation.results import LiftEstimate
    from increment.result_envelope import ResultCell
    from increment.semantics.design import Randomized

    rows = Analysis.from_unit_summary(
        pd.DataFrame(
            [
                {
                    "unit": f"{group}-{index}",
                    "arm": group,
                    "value": float(index + (group == "treatment")),
                }
                for group in ("control", "treatment")
                for index in range(8)
            ]
        ),
        unit="unit",
        group="arm",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics=[MetricSpec(name="value", type="mean")],
    ).run()
    row = rows[0]
    assert isinstance(row, LiftEstimate)
    row = row.model_copy(update={"prior_shrunk": True, "sampling_available": None})
    cell = ResultCell.from_row(row)
    assert cell.summary is None
    restored = (
        ResultCell.from_payload(json.loads(json.dumps(cell.to_payload()))),
        ResultCell.from_arrow(ResultCell.to_arrow([cell]))[0],
        pickle.loads(pickle.dumps(cell)),
        copy.deepcopy(cell),
    )

    assert all(item.summary is None for item in restored)


def test_switchback_payload_summary_keeps_display_alpha_not_procedure_alpha():
    import copy

    from increment.decision import ContrastDecisionProcedure
    from increment.estimation.contrast import ContrastStats, estimate_contrast
    from increment.estimation.contrast_results import ContrastResults
    from increment.result_envelope import ResultCell
    from increment.semantics.unit_cycle import UnitCycleTApproximation

    stats = ContrastStats(
        metric="orders",
        aggregation="any",
        probability_ct=0.8,
        randomization_law="independent_bernoulli_order",
        independence_grain="unit_cycle",
        carryover_order=0,
        observation_steps=3,
        retained_steps=3,
        control_group="control",
        treatment_group="treatment",
        n_units=3,
        n_cycles=9,
        ct_cycles=7,
        tc_cycles=2,
        reference_delta=0.2,
        mean_residual=0.1,
        m2_delta=0.03,
    )
    procedure = ContrastDecisionProcedure(
        reference=UnitCycleTApproximation(),
        metric="orders",
        role="secondary",
        alternative="greater",
        null_abs=0.0,
        alpha=0.025,
    )
    (result,) = estimate_contrast(stats, procedure).results
    (cell,) = ContrastResults([result]).as_cells()
    assert result.estimate.alpha == 0.05
    assert cell.summary is not None and cell.summary.alpha == result.estimate.alpha
    payload = copy.deepcopy(cell.to_payload())
    payload["summary"]["alpha"] = procedure.alpha

    restored = ResultCell.from_payload(payload)

    assert restored.summary is not None
    assert restored.summary.alpha == result.estimate.alpha


def test_provenance_requires_timezone_aware_creation_time():
    from datetime import datetime

    from increment.errors import CodedError
    from increment.result_envelope import Provenance

    with pytest.raises(CodedError) as raised:
        Provenance(None, None, "0+unknown", datetime(2025, 1, 1))
    assert raised.value.code == "results.cell.provenance_timestamp_invalid"


def test_render_rejects_unknown_audience_with_coded_error():
    from typing import Literal, cast

    from increment.errors import CodedError
    from increment.result_envelope import AnalysisEnvelope, render

    with pytest.raises(CodedError) as raised:
        render(AnalysisEnvelope((), None), cast(Literal["brief", "full"], "detailed"))
    assert raised.value.code == "results.render.audience_invalid"
