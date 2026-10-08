from __future__ import annotations

import pandas as pd
import pytest

from increment._evidence_dispatch import ContrastReadoutRequest
from increment.analysis import Analysis
from increment.decision import ContrastDecisionProcedure
from increment.errors import CapabilityError, InvalidRequestError
from increment.semantics.assignment import (
    IndependentBernoulliOrder,
    SharedScheduleOrder,
    SwitchbackAssignment,
    SwitchbackWindow,
)
from increment.semantics.design import Randomized
from increment.semantics.models import (
    AnalysisPlan,
    ExperimentMetric,
    MethodSpec,
    MultiplicitySpec,
)
from increment.semantics.unit_cycle import UnitCycleTApproximation
from increment.switchback import from_switchback_panel
from tests.sequential_cases import registered_spec


def _assignment() -> SwitchbackAssignment:
    return SwitchbackAssignment(
        sequence=IndependentBernoulliOrder(probability_ct=0.5),
        window=SwitchbackWindow(washout_steps=1, observation_steps=1),
    )


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for unit, order in (("u1", ("control", "treatment")), ("u2", ("treatment", "control"))):
        for cycle in range(2):
            for period, group in enumerate(order):
                for step in range(2):
                    rows.append(
                        {
                            "unit": unit,
                            "cycle": cycle,
                            "period": period,
                            "step": step,
                            "group": group,
                            "value": float(cycle + period + step + (group == "treatment")),
                        }
                    )
    return pd.DataFrame(rows)


def _source(reference=None):
    return from_switchback_panel(
        _frame(),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"value": "mean"},
        identification=Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        ),
        assignment=_assignment(),
        contrast_references={"value": reference or UnitCycleTApproximation()},
    )


def test_contrast_request_defensively_freezes_procedures():
    procedure = _procedure()
    procedures = {"value": procedure}
    request = ContrastReadoutRequest(metrics=tuple(_source().metrics), procedures=procedures)
    procedures.clear()
    assert request.procedures["value"] is procedure
    with pytest.raises(TypeError):
        request.procedures["other"] = procedure  # type: ignore[index]  # ty: ignore[invalid-assignment]


def _procedure(metric: str = "value") -> ContrastDecisionProcedure:
    return ContrastDecisionProcedure(
        reference=UnitCycleTApproximation(),
        metric=metric,
        role="primary",
        alternative="two-sided",
        null_abs=0.0,
        alpha=0.05,
    )


@pytest.mark.parametrize(
    ("procedures", "expected"),
    [
        ({}, {"missing": ("value",), "extra": (), "mismatches": ()}),
        (
            {"other": _procedure("other")},
            {"missing": ("value",), "extra": ("other",), "mismatches": ()},
        ),
        # A mismatch is reported without pinning how its entry is rendered.
        ({"value": _procedure("other")}, {"missing": (), "extra": (), "mismatches": None}),
    ],
)
def test_contrast_request_rejects_non_exact_procedure_mapping_before_source_access(
    procedures: dict[str, ContrastDecisionProcedure], expected: dict[str, object]
):
    metric = _source().metrics[0]
    with pytest.raises(InvalidRequestError) as exc:
        ContrastReadoutRequest(metrics=(metric,), procedures=procedures)
    assert exc.value.code == "readout.contrast.request"
    context = exc.value.context
    for name, value in expected.items():
        if value is None:
            assert context[name]
        else:
            assert context[name] == value


def test_analysis_run_matches_arm_configs_by_metric_name():
    # A caller may select an equal-name Metric with changed metadata. The
    # existing source config must still be selected for the stable run path.
    from increment.estimation.engine import Method
    from increment.frame import MetricSpec, synthesise_metric

    frame = pd.DataFrame(
        {
            "unit": ["u1", "u2", "u3", "u4"],
            "group": ["control", "control", "treatment", "treatment"],
            "outcome": [1.0, 2.0, 3.0, 4.0],
        }
    )
    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="group",
        metrics=[MetricSpec(name="outcome", type="mean", decision_method=Method(name="declared"))],
        control="control",
    )
    variant = synthesise_metric(
        MetricSpec(name="outcome", type="mean", winsorization={"lower_value": 0.0})
    )
    results = analysis.run(metrics=[variant])
    assert results and results[0].metric == "outcome"
    assert results[0].method == "declared"


def test_analysis_switchback_run_returns_one_result_per_metric_and_diagnostic():
    analysis = Analysis.from_switchback_panel(
        _frame(),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"value": "mean"},
        identification=Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        ),
        assignment=_assignment(),
        contrast_references={"value": UnitCycleTApproximation()},
    )
    results = analysis.run()
    assert len(results) == 1
    assert results[0].metric == "value"
    diagnostic = analysis.assignment_diagnostic()
    assert diagnostic.n_units == 2
    assert diagnostic.observation_rows > 0


@pytest.mark.parametrize(
    "method",
    ["run_daily", "run_daily_lift", "run_asof", "run_asof_lift", "run_breakout", "srm", "sitewide"],
)
def test_switchback_refuses_parallel_readouts_with_coded_capability(method: str):
    analysis = Analysis.from_switchback_panel(
        _frame(),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"value": "mean"},
        identification=Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        ),
        assignment=_assignment(),
        contrast_references={"value": UnitCycleTApproximation()},
    )
    with pytest.raises(CapabilityError) as exc:
        if method == "sitewide":
            analysis.sitewide("value")
        else:
            getattr(analysis, method)()
    assert exc.value.code == "facade.analysis.contrast_unavailable"


def test_contrast_results_convert_to_readout_rows():
    analysis = Analysis.from_switchback_panel(
        _frame(),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"value": "mean"},
        identification=Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        ),
        assignment=_assignment(),
        contrast_references={"value": UnitCycleTApproximation()},
    )
    from increment.tables import estimates_to_readout

    row = estimates_to_readout(analysis.run())[0]
    assert row["value_scale"] == "absolute"
    assert row["estimand"] == "retained_window_total_difference"
    assert row["n_units"] == 2
    assert row["method"] == "switchback_unit_t_approximation"
    assert row["reference"] == "unit_t_approximation"
    assert row["n_blocks"] is None
    assert (row["ct_cycles"], row["tc_cycles"]) == (2, 2)


def test_shared_schedule_contrast_results_convert_to_readout_rows_with_block_counts():
    """A shared-block readout row distinguishes roster size (``n_units``)
    from the inferential block count (``n_blocks``), and reports the
    block-level method/reference rather than the unit-cycle ones."""
    rows: list[dict[str, object]] = []
    roster = ("u1", "u2", "u3")
    block_order = {0: ("control", "treatment"), 1: ("treatment", "control")}
    for unit in roster:
        for cycle, order in block_order.items():
            for period, group in enumerate(order):
                for step in range(2):
                    value = 999.0 if step == 0 else 10.0 + 6.0 * (group == "treatment")
                    rows.append(
                        {
                            "unit": unit,
                            "cycle": cycle,
                            "period": period,
                            "step": step,
                            "group": group,
                            "value": value,
                        }
                    )
    analysis = Analysis.from_switchback_panel(
        pd.DataFrame(rows),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"value": "mean"},
        identification=Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        ),
        assignment=SwitchbackAssignment(
            sequence=SharedScheduleOrder(probability_ct=0.5),
            window=SwitchbackWindow(washout_steps=1, observation_steps=1),
        ),
    )
    from increment.tables import estimates_to_readout

    row = estimates_to_readout(analysis.run())[0]
    assert row["n_units"] == 3
    assert row["n_blocks"] == 2
    assert (row["ct_cycles"], row["tc_cycles"]) == (1, 1)
    assert row["method"] == "switchback_block_t"
    assert row["reference"] == "block_t"
    assert row["randomization_law"] == "shared_schedule"
    assert row["independence_grain"] == "shared_block"


@pytest.mark.parametrize("carryover_order", [0, 1, 2])
@pytest.mark.parametrize("shared", [False, True])
def test_conversion_readout_uses_retained_window_estimand(shared, carryover_order):
    from increment.estimation.contrast_results import ContrastResult
    from increment.tables import estimates_to_readout
    from tests.analysis_factory import contrast_rows

    rows = []
    for unit in ("u1", "u2"):
        for cycle in range(4):
            order = ("control", "treatment") if cycle < 2 else ("treatment", "control")
            for period, group in enumerate(order):
                for step in range(4):
                    converted = 1 <= step < 1 + carryover_order or (
                        step >= 1 + carryover_order and group == "treatment" and cycle % 2 == 0
                    )
                    rows.append(
                        {
                            "unit": unit,
                            "cycle": cycle,
                            "period": period,
                            "step": step,
                            "group": group,
                            "converted": converted,
                        }
                    )
    analysis = Analysis.from_switchback_panel(
        pd.DataFrame(rows),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"converted": "conversion"},
        contrast_references=None if shared else {"converted": UnitCycleTApproximation()},
        identification=Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        ),
        assignment=SwitchbackAssignment(
            sequence=(
                SharedScheduleOrder(probability_ct=0.5) if shared else IndependentBernoulliOrder()
            ),
            window=SwitchbackWindow(
                washout_steps=1, observation_steps=3, carryover_order=carryover_order
            ),
        ),
    )
    results = analysis.run()
    result = contrast_rows(results)[0]
    assert result.estimate.value == pytest.approx(0.5)
    assert result.aggregation == "any"
    assert ContrastResult.model_validate_json(result.model_dump_json()) == result
    from increment.estimation.readout_types import ReadoutResults

    restored_collection = ReadoutResults.model_validate_json(results.model_dump_json())
    assert restored_collection.metadata == results.metadata
    assert restored_collection.source == results.source
    assert restored_collection[0].aggregation == "any"
    row = estimates_to_readout(restored_collection)[0]
    assert row["estimand"] == "retained_window_conversion_difference"
    assert row["observation_steps"] == 3
    assert row["retained_steps"] == 3 - carryover_order
    assert row["lift"] == pytest.approx(0.5)
    expected_counts = (2, 2) if shared else (4, 4)
    assert (result.ct_cycles, result.tc_cycles) == expected_counts
    assert (row["ct_cycles"], row["tc_cycles"]) == expected_counts


def test_switchback_preferred_direction_reaches_result_and_readout():
    from increment.frame import MetricSpec
    from increment.tables import estimates_to_readout

    analysis = Analysis.from_switchback_panel(
        _frame(),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics=[
            MetricSpec(
                name="value",
                type="mean",
                value_column="value",
                preferred_direction="decrease",
            )
        ],
        identification=Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        ),
        assignment=_assignment(),
        contrast_references={"value": UnitCycleTApproximation()},
    )
    result = analysis.run()[0]
    row = estimates_to_readout([result])[0]
    assert result.preferred_direction == "decrease"
    assert row["preferred_direction"] == "decrease"


@pytest.mark.parametrize(
    ("plan", "reason"),
    [
        (
            AnalysisPlan(
                primary=ExperimentMetric(
                    metric="value",
                    decision_method=MethodSpec(name="cuped", variance_reduction="cuped"),
                )
            ),
            "unsupported_plan_override",
        ),
        (
            AnalysisPlan(primary=ExperimentMetric(metric="value", margin=0.1)),
            "relative_margin",
        ),
        (
            AnalysisPlan(
                primary="value",
                inference=registered_spec(),
            ),
            "unsupported_inference",
        ),
        (
            AnalysisPlan(secondaries=["value"]),
            "family_inference",
        ),
    ],
)
def test_switchback_plan_refusals_are_coded_before_frame_access(monkeypatch, plan, reason):
    def fail(*_args, **_kwargs):
        raise AssertionError("frame access occurred before plan validation")

    monkeypatch.setattr("increment.switchback.nw.from_native", fail)
    with pytest.raises(CapabilityError) as exc:
        from_switchback_panel(
            object(),  # ty: ignore[invalid-argument-type]
            unit="unit",
            cycle="cycle",
            period="period",
            step="step",
            group="group",
            metrics={"value": "mean"},
            identification=Randomized(
                control_group="control", allocation={"control": 0.5, "treatment": 0.5}
            ),
            assignment=_assignment(),
            contrast_references={"value": UnitCycleTApproximation()},
            plan=plan,
        )
    assert exc.value.code == "source.frame.switchback.plan"
    assert exc.value.context["reason"] == reason


@pytest.mark.parametrize(
    ("plan", "context"),
    [
        (
            AnalysisPlan(q=0.2, primary="value"),
            {
                "message": "switchback contrasts do not support an explicitly supplied plan q",
                "reason": "multiplicity_q",
            },
        ),
        (
            AnalysisPlan(
                primary="value",
                view_multiplicity=MultiplicitySpec(correction="bonferroni"),
            ),
            {
                "message": "switchback contrasts do not support view multiplicity",
                "reason": "view_multiplicity",
            },
        ),
    ],
)
def test_switchback_plan_multiplicity_refusals_precede_frame_access(monkeypatch, plan, context):
    def fail(*_args, **_kwargs):
        raise AssertionError("frame access occurred before multiplicity validation")

    monkeypatch.setattr("increment.switchback.nw.from_native", fail)
    with pytest.raises(CapabilityError) as exc:
        Analysis.from_switchback_panel(
            object(),  # ty: ignore[invalid-argument-type]
            unit="unit",
            cycle="cycle",
            period="period",
            step="step",
            group="group",
            metrics={"value": "mean"},
            identification=Randomized(
                control_group="control", allocation={"control": 0.5, "treatment": 0.5}
            ),
            assignment=_assignment(),
            contrast_references={"value": UnitCycleTApproximation()},
            plan=plan,
        )
    assert exc.value.code == "source.frame.switchback.plan"
    assert exc.value.context == context


def test_switchback_default_q_survives_plan_round_trip():
    plan = AnalysisPlan.model_validate(AnalysisPlan(primary="value").model_dump())

    analysis = Analysis.from_switchback_panel(
        _frame(),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"value": "mean"},
        identification=Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        ),
        assignment=_assignment(),
        contrast_references={"value": UnitCycleTApproximation()},
        plan=plan,
    )

    assert len(analysis.run()) == 1


def _switchback_analysis(reference=None):
    import pandas as pd

    from increment.analysis import Analysis
    from increment.semantics.assignment import (
        IndependentBernoulliOrder,
        SwitchbackAssignment,
        SwitchbackWindow,
    )
    from increment.semantics.design import Randomized

    rows = []
    for unit, order in (("u1", ("control", "treatment")), ("u2", ("treatment", "control"))):
        for cycle in range(2):
            for period, group in enumerate(order):
                for step in range(2):
                    rows.append(
                        {
                            "unit": unit,
                            "cycle": cycle,
                            "period": period,
                            "step": step,
                            "group": group,
                            "value": float(cycle + period + step + (group == "treatment")),
                        }
                    )
    return Analysis.from_switchback_panel(
        pd.DataFrame(rows),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"value": "mean"},
        identification=Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        ),
        contrast_references={"value": reference or UnitCycleTApproximation()},
        assignment=SwitchbackAssignment(
            sequence=IndependentBernoulliOrder(probability_ct=0.5),
            window=SwitchbackWindow(washout_steps=1, observation_steps=1),
        ),
    )


def test_sum_contrast_saved_collection_replay_preserves_identity_and_consumers():
    from increment.estimation.readout_types import ReadoutResults
    from increment.tables import estimates_to_readout

    results = _switchback_analysis().run()
    restored = ReadoutResults.model_validate_json(results.model_dump_json())

    assert restored.metadata == results.metadata
    assert restored.source == results.source
    assert restored[0].aggregation == "sum"
    assert restored.to_frame()["aggregation"].iloc[0] == "sum"
    assert estimates_to_readout(restored)[0]["aggregation"] == "sum"


def test_non_partial_contrast_collection_requires_every_scoped_row():
    import json

    from increment.errors import WireFormatError
    from increment.estimation.readout_types import ReadoutResults

    results = _switchback_analysis().run()
    payload = json.loads(results.model_dump_json())
    payload["rows"] = []
    with pytest.raises(WireFormatError) as raised:
        ReadoutResults.model_validate_json(json.dumps(payload))

    assert raised.value.code == "readout.serialization.identity_mismatch"


def test_contrast_concat_refuses_unknown_or_conflicting_provenance():
    from increment.estimation.contrast_results import ContrastResults

    results = _switchback_analysis().run()
    with pytest.raises(InvalidRequestError) as raised:
        results.concat(ContrastResults())
    assert raised.value.code == "readout.collection.concat_metadata_mismatch"

    changed_source = type(results)(
        results,
        metadata=results.metadata,
        source="another-source",
        sequential_snapshot=results.sequential_snapshot,
    )
    with pytest.raises(InvalidRequestError) as raised:
        results.concat(changed_source)
    assert raised.value.code == "readout.collection.concat_metadata_mismatch"


def test_cross_snapshot_contrast_concat_preserves_coverage_and_refuses_sequential_state():
    from increment.estimation.contrast_results import ContrastResults
    from increment.estimation.readout_types import (
        CellKey,
        CellRecord,
        ReadoutMetadata,
        ReadoutResults,
        ReadoutScope,
        SourceReadoutScope,
    )
    from increment.tables import estimates_to_readout

    row = _switchback_analysis().run()[0]

    def scoped(source_id, snapshot_id, sequential_snapshot=None, source="switchback"):
        source_row = row.model_copy(update={"source_snapshot_id": source_id})
        cell = CellKey.from_row(source_row)
        source_scope = SourceReadoutScope(
            source_snapshot_id=source_id,
            cells=(cell,),
            decision_cells=(cell,),
            rosters=(),
            decision_complete_by_population={"assigned": source_row.decision_scope_complete},
        )
        scope = ReadoutScope(
            snapshot_id=snapshot_id,
            cells=(cell,),
            decision_cells=(cell,),
            populations=("assigned",),
            by_source={source_id: source_scope},
        )
        metadata = ReadoutMetadata(
            scope=scope,
            cells=(CellRecord(cell=cell, source_snapshot_id=source_id),),
        )
        return ContrastResults(
            [source_row],
            metadata=metadata,
            source=source,
            sequential_snapshot=sequential_snapshot,
        )

    declared_source = {"kind": "declared", "sha256": "a" * 64}
    exploratory_source = {"kind": "exploratory", "sha256": "b" * 64}
    combined = scoped("snapshot-a", "scope-a", source=declared_source).concat(
        scoped("snapshot-b", "scope-b", source=exploratory_source)
    )
    assert combined.metadata.partial is False
    assert set(combined.metadata.scope.by_source) == {"snapshot-a", "snapshot-b"}
    assert combined.metadata.scope.by_source["snapshot-a"].source == declared_source
    assert combined.metadata.scope.by_source["snapshot-b"].source == exploratory_source
    assert not combined.to_frame()["view_partial"].any()
    restored = ReadoutResults.model_validate_json(combined.model_dump_json())
    assert restored.metadata == combined.metadata
    assert restored.source is None
    assert set(restored.metadata.scope.by_source) == {"snapshot-a", "snapshot-b"}
    assert restored.metadata.scope.by_source["snapshot-a"].source == declared_source
    assert restored.metadata.scope.by_source["snapshot-b"].source == exploratory_source
    partial = combined.filter(lambda item: item.source_snapshot_id == "snapshot-a")
    assert partial.metadata.partial is True
    assert partial.to_frame()["view_partial"].all()
    assert estimates_to_readout(partial)[0]["view_partial"] is True

    checkpoint = object()
    with pytest.raises(InvalidRequestError) as raised:
        scoped("snapshot-a", "scope-a", checkpoint, source="declared").concat(
            scoped("snapshot-b", "scope-b", checkpoint, source="exploratory")
        )
    assert raised.value.code == "readout.collection.concat_metadata_mismatch"


def test_streaming_digest_preserves_full_width_integer_identity():
    from increment._canonical import canonical_digest_bytes
    from increment.estimation.readout_types import StreamingDigest

    assert canonical_digest_bytes({"sample_size": 2**54}) != canonical_digest_bytes(
        {"sample_size": 2**54 + 1}
    )
    values = (2**53, 2**53 + 1, 2**54, 2**54 + 1)
    digests = []
    for value in values:
        digest = StreamingDigest()
        digest.update({"sample_size": value})
        digests.append(digest.hexdigest())

    assert len(set(digests)) == len(values)


def test_readout_wire_rejects_duplicate_object_keys_with_portable_refusal():
    import pickle
    from copy import deepcopy

    from increment.errors import WireFormatError
    from increment.estimation.readout_types import ReadoutResults

    results = _switchback_analysis().run()
    payload = results.model_dump_json().replace(
        '"kind": "increment.readout"',
        '"kind": "increment.readout", "kind": "increment.readout"',
    )
    with pytest.raises(WireFormatError) as raised:
        ReadoutResults.model_validate_json(payload)

    assert raised.value.code == "readout.serialization.duplicate_object_key"
    assert deepcopy(raised.value).code == raised.value.code
    assert pickle.loads(pickle.dumps(raised.value)).code == raised.value.code


def test_readout_wire_requires_snapshot_for_checkpoint_bearing_results():
    import json
    import pickle
    from copy import deepcopy
    from datetime import date

    from increment.errors import WireFormatError
    from increment.estimation.readout_types import ReadoutResults
    from tests.test_sequential_public_sources import _native_fixture

    _connection, _definitions, analysis = _native_fixture("bernoulli")
    try:
        analysis.capture_sequential(finalized=True, as_of=date(2025, 1, 16))
        results = analysis.run()
        assert results.sequential_snapshot is not None
        assert any(row.sequential_result is not None for row in results)
        payload = json.loads(results.model_dump_json())
        payload["sequential_snapshot"] = None

        with pytest.raises(WireFormatError) as raised:
            ReadoutResults.model_validate_json(json.dumps(payload))

        assert raised.value.code == "readout.serialization.sequential_snapshot_required"
        assert deepcopy(raised.value).code == raised.value.code
        assert pickle.loads(pickle.dumps(raised.value)).code == raised.value.code
    finally:
        analysis.close()


@pytest.mark.parametrize(
    "method", ["run_daily", "run_asof", "breakout_summaries", "factor_summaries"]
)
def test_switchback_day_axis_and_breakout_routes_refuse(method: str) -> None:
    from increment.errors import CapabilityError

    analysis = _switchback_analysis()
    with pytest.raises(CapabilityError) as refusal:
        getattr(analysis, method)()
    assert refusal.value.code == "facade.analysis.contrast_unavailable"


def test_switchback_dashboard_breakout_reads_refuses_with_contrast_code() -> None:
    from increment.errors import CapabilityError
    from increment.semantics.models import Breakout

    with pytest.raises(CapabilityError) as refusal:
        _switchback_analysis().dashboard_breakout_reads(Breakout(property="geo"))
    assert refusal.value.code == "facade.analysis.contrast_unavailable"
    assert refusal.value.context["method"] == "dashboard_breakout_reads"


def test_assignment_diagnostic_refuses_arm_evidence_with_its_own_code() -> None:
    from increment.errors import CapabilityError

    frame = pd.DataFrame(
        {"unit": ["u1", "u2"], "group": ["control", "treatment"], "outcome": [1.0, 2.0]}
    )
    analysis = Analysis.from_unit_summary(
        frame, unit="unit", group="group", metrics={"outcome": "mean"}, control="control"
    )
    with pytest.raises(CapabilityError) as refusal:
        analysis.assignment_diagnostic()
    assert refusal.value.code == "facade.analysis.switchback_only"
    assert refusal.value.context["method"] == "assignment_diagnostic"


def test_switchback_sitewide_refuses_non_native_source() -> None:
    from increment.errors import CapabilityError

    analysis = _switchback_analysis()
    with pytest.raises(CapabilityError) as refusal:
        analysis.sitewide("value")
    assert refusal.value.code == "facade.analysis.contrast_unavailable"


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("run_daily", {}),
        ("run_daily", {"dimension": "geo"}),
        ("run_daily_lift", {}),
        ("run_daily_lift", {"dimension": "geo"}),
        ("run_asof", {}),
        ("run_asof", {"dimension": "geo"}),
        ("run_asof_lift", {}),
        ("run_asof_lift", {"dimension": "geo"}),
        ("materialize", {}),
        ("summary_sql", {}),
        ("publish_unit_day_artifact", {"store": None}),
    ],
)
def test_envelope_refuses_arm_native_artifact_and_all_day_axis_routes(method, kwargs):
    from tests.estimation.test_unit_cycle_envelope import envelope

    reference = envelope(p=0.5, cycles=2).model_copy(update={"assignment": _assignment()})
    analysis = _switchback_analysis(reference)
    result = analysis.run()[0]
    assert result.method == "switchback_unit_variance_envelope"
    with pytest.raises(CapabilityError) as caught:
        getattr(analysis, method)(**kwargs)
    assert caught.value.code == "facade.analysis.contrast_unavailable"


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
def test_public_envelope_plan_result_frame_and_readout_roundtrip(backend):
    import narwhals as nw

    from increment.decision_wire import compiled_plan_from_json, compiled_plan_to_json
    from increment.estimation.contrast_results import ContrastResult
    from increment.tables import estimates_to_readout
    from tests.estimation.test_unit_cycle_envelope import envelope

    reference = envelope(p=0.5, cycles=2).model_copy(update={"assignment": _assignment()})
    analysis = _switchback_analysis(reference)
    source = _source(reference)
    results = analysis.run()
    result = results[0]
    context = source.context
    restored_context = type(context).model_validate_json(context.model_dump_json())
    assert restored_context == context
    assert restored_context.procedures["value"].reference == reference
    restored = ContrastResult.model_validate_json(result.model_dump_json())
    assert restored == result
    from increment.decision import (
        CompiledDecisionPlan,
        CompiledViewPolicies,
        FixedInference,
        MultiplicityFamily,
    )
    from increment.estimation.contrast import estimate_contrast

    policy = MultiplicityFamily(name="none")
    compiled = CompiledDecisionPlan(
        declared=True,
        alpha=result.alpha,
        q=0.1,
        path="frame/contrast",
        inference=FixedInference(),
        procedures=context.procedures,
        view_policies=CompiledViewPolicies(
            asof=policy, randomized_breakout=policy, encouragement_breakout=policy
        ),
    )
    plan = compiled_plan_from_json(compiled_plan_to_json(compiled))
    assert plan == compiled
    restored_procedure = plan.procedures["value"]
    assert isinstance(restored_procedure, ContrastDecisionProcedure)
    assert restored_procedure.reference == reference
    replay = estimate_contrast(source.contrast_stats(source.metrics[0]), restored_procedure)
    replay_row = replay.results[0]
    scope_fields = {
        "source_snapshot_id",
        "decision_scope_complete",
        "decision_scope_reason_code",
        "decision_scope_reason_context",
    }
    assert {
        name: getattr(replay_row, name)
        for name in ContrastResult.model_fields
        if name not in scope_fields
    } == {
        name: getattr(result, name)
        for name in ContrastResult.model_fields
        if name not in scope_fields
    }
    assert result.analysis_population == "assigned"
    assert result.source_snapshot_id is not None
    assert result.decision_scope_complete is True
    assert result.decision_scope_reason_code is None
    from increment.estimation.readout_types import ReadoutResults

    restored_collection = ReadoutResults.model_validate_json(results.model_dump_json())
    assert restored_collection.metadata == results.metadata
    assert restored_collection.source == results.source
    frame = nw.from_native(results.to_frame(backend=backend), eager_only=True)
    restored = results[0]
    row = next(frame.iter_rows(named=True))
    readout = estimates_to_readout(results)[0]
    for name in ("method", "reference", "mean_slope", "effective_alpha", "residual_p_value"):
        assert row[name] == readout[name] == getattr(result, name)
    assert row["source_snapshot_id"] == result.source_snapshot_id
    assert row["analysis_population"] == "assigned"
    assert row["decision_scope_complete"] is True
    assert row["view_partial"] is False
    assert readout["source_snapshot_id"] == result.source_snapshot_id
    assert readout["analysis_population"] == "assigned"
    assert readout["decision_scope_complete"] is True
    assert readout["view_partial"] is False
    import json

    assert (
        json.loads(row["reference_spec"])
        == readout["reference_spec"]
        == reference.model_dump(mode="json")
    )
    assert (
        json.loads(row["provenance"])
        == readout["provenance"]
        == reference.provenance.model_dump(mode="json")
    )
    assert frame["dof"].is_null().to_list() == [True]
    assert readout["dof"] is None and row["mean_slope"] > 0
    assert readout["stat_sig"] == (result.residual_p_value < result.alpha)
