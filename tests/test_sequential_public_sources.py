"""Bernoulli decisions, portable replay, and immutable source continuation."""

from datetime import date, timedelta
from fractions import Fraction

import pytest

from increment import Analysis, SequentialCompliancePolicy, SequentialRegistration
from increment.breakout.estimates import LiftEstimates
from increment.errors import CapabilityError, InvalidRequestError
from increment.frame import MetricSpec, synthesise_metric
from increment.results import LiftEstimate
from increment.semantics.design import Randomized
from increment.semantics.models import AnalysisPlan, InferenceSpec
from increment.sequential_source import (
    frame_observation_mapping,
    native_observation_mapping,
    sequential_definition_id,
)
from tests.analysis_factory import _native_source
from tests.sequential_cases import registration


def _assert_cross_source_readouts_equal(left, right):
    identity_fields = {"source_snapshot_id", "family_id"}
    assert [row.model_dump(exclude=identity_fields) for row in left] == [
        row.model_dump(exclude=identity_fields) for row in right
    ]
    assert [row.source_snapshot_id for row in left] != [row.source_snapshot_id for row in right]


def _plan(specs, law, *, cells=None, date=None, exposure_date="exposure"):
    design = Randomized(control_group="control")
    base = registration(law, cells=cells)
    reg = SequentialRegistration.model_validate(
        {
            **base.model_dump(),
            "definitions_id": sequential_definition_id(
                [synthesise_metric(spec) for spec in specs],
                design,
                transformations=specs,
                source_mapping=frame_observation_mapping(
                    unit="unit", group="arm", date=date, exposure_date=exposure_date
                ),
            ),
        }
    )
    return AnalysisPlan(
        primary="outcome", inference=InferenceSpec(kind="always_valid", registration=reg)
    )


def gaussian_plan(
    specs,
    *,
    law="gaussian",
    cells=None,
    source_id="frame",
    alpha=0.05,
    q=0.10,
    secondaries=None,
    unit="user_id",
    group="variant",
    date=None,
    exposure_date="exposure",
):
    """Registered test declarations, independent of all observed values."""
    from fractions import Fraction

    from increment import SequentialCell

    design = Randomized(control_group="control")
    base = registration(law)
    models = tuple(base.models[0].model_copy(update={"metric": spec.name}) for spec in specs)
    reg = SequentialRegistration.model_validate(
        {
            **base.model_dump(),
            "source_id": source_id,
            "models": models,
            "q": Fraction(q),
            "definitions_id": sequential_definition_id(
                [synthesise_metric(spec) for spec in specs],
                design,
                transformations=specs,
                source_mapping=frame_observation_mapping(
                    unit=unit, group=group, date=date, exposure_date=exposure_date
                ),
            ),
            "roster": cells
            if cells is not None
            else tuple(
                SequentialCell(
                    metric=spec.name,
                    group_id="treatment",
                    family=True,
                    alpha=Fraction(alpha),
                )
                for spec in specs
            ),
        }
    )
    return AnalysisPlan(
        alpha=alpha,
        q=q,
        secondaries=secondaries or [spec.name for spec in specs],
        inference=InferenceSpec(kind="always_valid", registration=reg),
    )


def _frame(c, t, *, offset=0):
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "unit": f"{offset + i:08d}-{arm}",
                "arm": arm,
                "outcome": value,
                "exposure": offset + i,
            }
            for i, pair in enumerate(zip(c, t, strict=True))
            for arm, value in zip(("control", "treatment"), pair, strict=True)
        ]
    )


def _analysis(frame, specs, plan):
    return Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        control="control",
        metrics=specs,
        experiment_id="experiment",
        plan=plan,
        exposure_date="exposure",
    )


def _row(analysis):
    rows = analysis.run()
    assert isinstance(rows, LiftEstimates)
    assert {row.multiplicity_status for row in rows} == {"declared_plan"}
    return rows[0]


def test_registered_whole_window_family_scope_reports_registration_q():
    specs = [MetricSpec(name="outcome", type="conversion")]
    plan = gaussian_plan(
        specs, law="bernoulli", q=0.2, source_id="experiment", unit="unit", group="arm"
    )
    analysis = _analysis(_frame([0, 1] * 20, [1, 1] * 20), specs, plan)

    rows = analysis.run()
    assert rows.metadata.scope.families
    registration_q = float(plan.inference.registration.q)
    assert all(row.family_q == registration_q for row in rows)
    assert all(
        scope.family.q == registration_q
        for scope in rows.metadata.scope.families
        if scope.family is not None
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "law,c,t",
    [
        ("bernoulli", [0, 0, 0, 1] * 24, [0, 1, 1, 1] * 24),
    ],
)
def test_frame_current_wire_and_moments_replay_same_actual_decision(tmp_path, law, c, t):
    import pandas as pd
    import pyarrow.parquet as pq

    specs = [MetricSpec(name="outcome", type="conversion")]
    plan = _plan(specs, law)
    frame = _frame(c, t)
    analysis = _analysis(frame, specs, plan)
    original = _row(analysis)
    assert original.stat_sig()
    assert original.require_lift().value > 0
    lower = original.require_sequential_result().bounds.lower
    assert lower is not None and lower > 1

    # Mutating caller data cannot rewrite the captured exact observations.
    frame.loc[:, "outcome"] = 0
    assert _row(analysis).require_sequential_result() == original.require_sequential_result()
    snapshot = analysis.sequential_snapshot()
    assert analysis.capture_sequential(finalized=True, previous=snapshot) == snapshot

    path = tmp_path / "checkpoint.parquet"
    analysis.export(path)
    payload = pq.read_table(path).to_pylist()
    assert payload[0]["moments_format"] == 10
    assert '"wire_version":3' in payload[0]["decision_plan"]
    legacy_payload = [dict(payload[0], moments_format=8)]
    with pytest.raises(CapabilityError) as legacy:
        Analysis.from_moments(legacy_payload, metrics=specs, control="control")
    assert legacy.value.code == "sequential.continuation.legacy"
    replay = Analysis.from_moments(payload, metrics=specs, control="control")
    assert replay.sequential_snapshot() == snapshot
    replayed = _row(replay)
    assert replayed.require_sequential_result() == original.require_sequential_result()
    assert replayed.source_snapshot_id == original.source_snapshot_id
    assert replayed.family_id == original.family_id
    previous_format = [dict(payload[0], moments_format=9)]
    previous_format[0].pop("source_identity")
    previous_replay = Analysis.from_moments(previous_format, metrics=specs, control="control")
    try:
        previous_snapshot = previous_replay.sequential_snapshot()
        assert previous_snapshot == snapshot
        previous_result = _row(previous_replay)
        assert previous_result.require_sequential_result() == original.require_sequential_result()
        assert previous_result.source_snapshot_id != original.source_snapshot_id
    finally:
        previous_replay.close()
    projected = replay.run().to_frame(backend="pandas")
    assert isinstance(projected, pd.DataFrame)
    assert projected["sequential_log_e"].iloc[0] == str(
        original.require_exact_sequential_result().log_e
    )
    assert projected["sequential_lower"].iloc[0] > 0
    assert projected["lift"].iloc[0] > 0


def test_checkpoint_export_preserves_captured_assignment_counts_and_omits_legacy_counts(
    tmp_path,
):
    import json

    import pyarrow.parquet as pq

    from increment import capture_sequential_snapshot
    from increment.semantics.design import Randomized
    from increment.sources import ASSIGNMENT_COUNTS_FIELD, MomentsSource, export_source_moments

    specs = [MetricSpec(name="outcome", type="conversion")]
    design = Randomized(
        control_group="control",
        allocation={"control": 0.5, "treatment": 0.5},
        allocation_scheme="independent",
    )
    frame = _frame([0, 1] * 24, [1, 1] * 24)
    analysis = _analysis(frame, specs, _plan(specs, "bernoulli"))
    _row(analysis)
    snapshot = analysis.sequential_snapshot()
    assert snapshot.assignment_counts == {"control": 48, "treatment": 48}

    path = tmp_path / "checkpoint-counts.parquet"
    analysis.export(path)
    payload = pq.read_table(path).to_pylist()
    assert json.loads(payload[0][ASSIGNMENT_COUNTS_FIELD]) == {
        "control": 48,
        "treatment": 48,
    }
    replay = Analysis.from_moments(payload, metrics=specs, design=design)
    assert replay.sequential_snapshot().assignment_counts == snapshot.assignment_counts

    legacy_snapshot = capture_sequential_snapshot(
        snapshot.registration,
        [
            {
                "unit_id": row["unit"],
                "group_id": row["arm"],
                "values": {"outcome": row["outcome"]},
                "segments": {},
            }
            for row in frame.to_dict(orient="records")
        ],
        source_id=snapshot.registration.source_id,
        definitions_id=snapshot.registration.definitions_id,
        finalized=True,
        reveal_cursor=snapshot.reveal_cursor,
    )
    payload[0]["sequential_snapshot"] = legacy_snapshot.model_dump_json()
    payload[0].pop(ASSIGNMENT_COUNTS_FIELD, None)
    legacy_source = MomentsSource(
        payload,
        metrics=analysis.metrics,
        study_id=str(payload[0]["experiment_id"]),
        design=design,
    )
    legacy_path = tmp_path / "legacy-checkpoint-counts.parquet"
    export_source_moments(legacy_source, legacy_path, observational_refusal=lambda _metric: None)
    legacy_wire = pq.read_table(legacy_path).to_pylist()
    assert ASSIGNMENT_COUNTS_FIELD not in legacy_wire[0]

    legacy_replay = Analysis.from_moments(legacy_wire, metrics=specs, design=design)
    legacy_results = legacy_replay.run()
    assert legacy_results.metadata is not None
    integrity = next(iter(legacy_results.metadata.scope.by_source.values())).integrity[0]
    assert integrity.status == "not_checked_missing_counts"


def test_checkpoint_assignment_counts_are_immutable_after_validation():
    from typing import cast

    specs = [MetricSpec(name="outcome", type="conversion")]
    analysis = _analysis(
        _frame([0, 1] * 24, [1, 1] * 24),
        specs,
        _plan(specs, "bernoulli"),
    )
    _row(analysis)
    snapshot = analysis.sequential_snapshot()
    with pytest.raises(TypeError):
        cast(dict[str, int], snapshot.assignment_counts)["control"] = 0
    restored = type(snapshot).model_validate_json(snapshot.model_dump_json())
    assert restored.assignment_counts == {"control": 48, "treatment": 48}


def test_continuation_without_new_capture_counts_does_not_reuse_parent_totals():
    from typing import cast

    from increment import capture_sequential_snapshot
    from increment.estimation.assignment_integrity import assignment_integrity
    from tests.sequential_cases import records

    declaration = registration("bernoulli")
    previous = capture_sequential_snapshot(
        declaration,
        records([0], [1]),
        source_id=declaration.source_id,
        definitions_id=declaration.definitions_id,
        finalized=True,
        assignment_counts={"control": 100, "treatment": 100},
    )
    continued = capture_sequential_snapshot(
        declaration,
        records([0, 0], [1, 1]),
        source_id=declaration.source_id,
        definitions_id=declaration.definitions_id,
        finalized=True,
        previous=previous,
    )
    assert continued.ancestors[-1].assignment_counts == {
        "control": 100,
        "treatment": 100,
    }
    with pytest.raises(TypeError):
        cast(dict[str, int], continued.ancestors[-1].assignment_counts)["control"] = 0
    restored = type(continued).model_validate_json(continued.model_dump_json())
    assert restored.assignment_counts is None
    assert restored.ancestors[-1].assignment_counts == {
        "control": 100,
        "treatment": 100,
    }
    assert continued.assignment_counts is None
    integrity = assignment_integrity(
        Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        continued.assignment_counts,
    )
    assert integrity.status == "not_checked_missing_counts"
    assert integrity.observed is None


@pytest.mark.slow
def test_frame_zero_event_prefix_append_and_rewrite_do_not_reset_process():
    import pandas as pd

    specs = [MetricSpec(name="outcome", type="conversion")]
    plan = _plan(specs, "bernoulli")
    early = _frame([0] * 4, [0] * 4)
    first = _analysis(early, specs, plan)
    first_row = _row(first)
    assert first_row.lift is None
    assert first_row.require_sequential_result().point_reason
    assert not first_row.stat_sig()
    from increment.tables import estimates_to_readout

    displayed = estimates_to_readout([first_row])[0]
    assert displayed["lift"] is None
    assert displayed["sequential_point_reason"]
    assert displayed["sequential_log_e"] == str(first_row.require_exact_sequential_result().log_e)
    assert displayed["posterior_prob_favorable"] is None
    assert displayed["stat_sig"] is False
    snapshot = first.sequential_snapshot()
    later = pd.concat(
        [early, _frame([0, 0, 0, 1] * 24, [0, 1, 1, 1] * 24, offset=4)], ignore_index=True
    )
    continued = _analysis(later, specs, plan)
    linked = continued.sequential_snapshot(previous=snapshot)
    assert linked.parent_id == snapshot.prefix_id
    assert linked.arm("outcome", "control").n == 100
    assert _row(continued).stat_sig()
    rewritten = later.copy()
    rewritten.loc[0, "outcome"] = 1
    changed = _analysis(rewritten, specs, plan)
    with pytest.raises(CapabilityError) as raised:
        changed.sequential_snapshot(previous=snapshot)
    assert raised.value.code == "sequential.continuation.rewrite"


def test_registration_rejected_before_dataframe_protocol_is_touched():
    specs = [MetricSpec(name="outcome", type="conversion")]
    plan = _plan(specs, "bernoulli")
    reg = plan.inference.registration
    wrong = SequentialRegistration.model_validate({**reg.model_dump(), "source_id": "other"})
    wrong_plan = AnalysisPlan(
        primary="outcome", inference=InferenceSpec(kind="always_valid", registration=wrong)
    )

    class UnreadFrame:
        def __getattribute__(self, name):
            raise AssertionError(f"dataframe was accessed before registration rejection: {name}")

    with pytest.raises(CapabilityError) as raised:
        _analysis(UnreadFrame(), specs, wrong_plan)
    assert raised.value.code == "sequential.source.invalid"


def test_gaussian_frame_registration_refuses_before_dataframe_access():
    specs = [MetricSpec(name="outcome", type="mean")]
    plan = _plan(specs, "gaussian")

    class UnreadFrame:
        def __getattribute__(self, name):
            raise AssertionError(f"dataframe accessed before Gaussian refusal: {name}")

    with pytest.raises(CapabilityError) as raised:
        _analysis(UnreadFrame(), specs, plan)
    assert raised.value.code == "sequential.route.unsupported"


def test_summary_reveal_order_follows_exposure_not_row_order():
    """Rows sorted by outcome must reveal the same sequence as rows in exposure order."""
    specs = [MetricSpec(name="outcome", type="conversion")]
    plan = _plan(specs, "bernoulli")
    frame = _frame([0, 0, 0, 1] * 24, [0, 1, 1, 1] * 24)
    by_outcome = frame.sort_values(["outcome", "unit"], ascending=[False, True], ignore_index=True)
    in_order = _analysis(frame, specs, plan)
    outcome_sorted = _analysis(by_outcome, specs, plan)
    assert outcome_sorted.sequential_snapshot() == in_order.sequential_snapshot()
    assert (
        _row(outcome_sorted).require_sequential_result()
        == _row(in_order).require_sequential_result()
    )


def test_summary_reveal_order_ranks_day_labels_numerically():
    """``d10`` follows ``d9``; lexicographic label order would reveal it after ``d1``."""
    specs = [MetricSpec(name="outcome", type="conversion")]
    plan = _plan(specs, "bernoulli")
    frame = _frame([0, 0, 0, 1] * 24, [0, 1, 1, 1] * 24)
    labelled = frame.assign(exposure=[f"d{day}" for day in frame["exposure"]])
    assert (
        _analysis(labelled, specs, plan).sequential_snapshot()
        == _analysis(frame, specs, plan).sequential_snapshot()
    )


def test_sparse_panel_sequential_cohort_and_continuation_match_dense() -> None:
    """Sparse zero events preserve finalized cohorts and continuation state."""
    import pandas as pd

    specs = [MetricSpec(name="outcome", type="conversion", window_days=2)]
    plan = _plan(specs, "bernoulli", date="day", exposure_date="exposed")
    registration = plan.inference.registration
    assert registration is not None
    reveal = registration.reveal.model_copy(update={"longest_window_days": 2})
    plan = plan.model_copy(
        update={
            "inference": InferenceSpec(
                kind="always_valid",
                registration=registration.model_copy(update={"reveal": reveal}),
            )
        }
    )
    days = [date(2026, 1, day) for day in range(1, 5)]
    sparse_rows = []
    dense_rows = []
    for unit_index in range(8):
        exposure = days[0] if unit_index % 2 == 0 else days[2]
        event = float(unit_index % 3 == 0)
        for day in days:
            row = {
                "unit": f"u{unit_index}",
                "arm": "control" if unit_index < 4 else "treatment",
                "day": day,
                "exposed": exposure,
                "outcome": event if day == exposure else 0.0,
            }
            dense_rows.append(row)
            if day == exposure:
                sparse_rows.append(row)

    def source(rows):
        return Analysis.from_unit_panel(
            pd.DataFrame(rows),
            unit="unit",
            group="arm",
            date="day",
            exposure_date="exposed",
            control="control",
            metrics=specs,
            experiment_id="experiment",
            observation_end=days[-1],
            plan=plan,
        )

    sparse = source(sparse_rows)
    dense = source(dense_rows)
    early = [
        analysis.capture_sequential(finalized=True, as_of=days[1]) for analysis in (sparse, dense)
    ]
    assert early[0] == early[1]
    assert len(early[0].records) == 4
    continued = [
        analysis.capture_sequential(finalized=True, as_of=days[3], previous=snapshot)
        for analysis, snapshot in zip((sparse, dense), early, strict=True)
    ]
    assert continued[0] == continued[1]
    assert continued[0].parent_id == early[0].prefix_id
    assert len(continued[0].records) == 8
    assert continued[0].states == continued[1].states


@pytest.mark.parametrize("scratch_budget", [0, 64 * 1024 * 1024])
def test_sparse_panel_sequential_state_matches_dense_and_streamed_kernels(
    scratch_budget: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sequential capture and estimates agree across daily panel kernels."""
    import pandas as pd

    import increment.frame as frame_module

    monkeypatch.setattr(frame_module, "_DAY_PANEL_SCRATCH_BUDGET_BYTES", scratch_budget)
    specs = [MetricSpec(name="outcome", type="conversion", window_days=2)]
    plan = _plan(specs, "bernoulli", date="day", exposure_date="exposed")
    registration = plan.inference.registration
    assert registration is not None
    reveal = registration.reveal.model_copy(update={"longest_window_days": 2})
    plan = plan.model_copy(
        update={
            "inference": InferenceSpec(
                kind="always_valid",
                registration=registration.model_copy(update={"reveal": reveal}),
            )
        }
    )
    days = [date(2026, 1, day) for day in range(1, 5)]
    rows = [
        {
            "unit": f"u{unit_index}",
            "arm": "control" if unit_index < 4 else "treatment",
            "day": days[unit_index % len(days)],
            "exposed": days[0] if unit_index % 2 == 0 else days[2],
            "outcome": float(unit_index % 3 == 0) if unit_index % len(days) == 0 else 0.0,
        }
        for unit_index in range(8)
    ]

    def analyze(budget: int):
        monkeypatch.setattr(frame_module, "_DAY_PANEL_SCRATCH_BUDGET_BYTES", budget)
        return Analysis.from_unit_panel(
            pd.DataFrame(rows),
            unit="unit",
            group="arm",
            date="day",
            exposure_date="exposed",
            control="control",
            metrics=specs,
            experiment_id="experiment",
            observation_end=days[-1],
            plan=plan,
        )

    source = analyze(scratch_budget)
    daily = source.run_daily()
    snapshot = source.capture_sequential(finalized=True, as_of=days[-1])
    result = source.run()[0]
    other = analyze(64 * 1024 * 1024 if scratch_budget == 0 else 0)
    other_daily = other.run_daily()
    other_snapshot = other.capture_sequential(finalized=True, as_of=days[-1])

    assert sorted(daily, key=lambda row: (row.ds, row.group_id)) == sorted(
        other_daily, key=lambda row: (row.ds, row.group_id)
    )
    assert snapshot.records == other_snapshot.records
    assert snapshot.states == other_snapshot.states
    assert snapshot.prefix_id == other_snapshot.prefix_id
    assert result == other.run()[0]
    assert len(snapshot.records) == 8


def test_sequential_summary_without_exposure_refuses_before_reading_frame():
    from tests.sequential_cases import UnreadFrame

    specs = [MetricSpec(name="outcome", type="conversion")]
    plan = _plan(specs, "bernoulli")

    with pytest.raises(InvalidRequestError) as raised:
        Analysis.from_unit_summary(
            UnreadFrame(),
            unit="unit",
            group="arm",
            control="control",
            metrics=specs,
            experiment_id="experiment",
            plan=plan,
            exposure_date=None,
        )
    assert raised.value.code == "source.frame.sequential_exposure_date"
    assert raised.value.context["constructor"] == "from_unit_summary"


@pytest.mark.parametrize(
    "law,metric_type",
    [("gaussian", "mean"), ("gaussian_ratio", "ratio")],
)
def test_non_bernoulli_public_compilation_refuses_before_source_access(law, metric_type):
    from increment.plan import compile_decision_plan

    specs = [
        MetricSpec(
            name="outcome",
            type=metric_type,
            numerator="num" if metric_type == "ratio" else None,
            denominator="den" if metric_type == "ratio" else None,
        )
    ]
    plan = _plan(specs, law)
    metrics = [synthesise_metric(specs[0])]
    with pytest.raises(CapabilityError) as raised:
        compile_decision_plan(
            plan,
            metrics,
            path="frame",
            design=Randomized(control_group="control"),
        )
    assert raised.value.code == "sequential.route.unsupported"


@pytest.mark.slow
@pytest.mark.parametrize(
    "axis", ["date", "datetime", "numeric", "numeric_float", "iso", "structured"]
)
def test_panel_common_window_only_reveals_finalized_units_and_current_asof(axis, tmp_path):
    import pyarrow.parquet as pq

    specs = [MetricSpec(name="outcome", type="conversion", window_days=2)]
    plan = _plan(specs, "bernoulli", date="day", exposure_date="exposed")
    frame = _frame([0, 0, 0, 1] * 24, [0, 1, 1, 1] * 24)
    from datetime import datetime

    def label(day):
        if axis == "numeric":
            return day
        if axis == "numeric_float":
            return float(day)
        if axis == "structured":
            return f"d{day}"
        value = date(2025, 1, 1) + timedelta(days=day)
        if axis == "iso":
            return value.isoformat()
        if axis == "datetime":
            return datetime.combine(value, datetime.min.time())
        return value

    start = label(0)
    frame["day"] = start
    frame["exposed"] = start
    analysis = Analysis.from_unit_panel(
        frame,
        unit="unit",
        group="arm",
        date="day",
        exposure_date="exposed",
        control="control",
        metrics=specs,
        experiment_id="experiment",
        plan=plan,
        observation_end=label(20),
    )
    with pytest.raises(CapabilityError) as missing_horizon:
        analysis.capture_sequential(finalized=True)
    assert missing_horizon.value.code == "sequential.source.invalid"
    empty = analysis.capture_sequential(finalized=True, as_of=start)
    assert not empty.records
    assert not _row(analysis).stat_sig()
    full = analysis.capture_sequential(finalized=True, as_of=label(14))
    assert len(full.records) == 192
    assert full.parent_id == empty.prefix_id
    assert _row(analysis).stat_sig()
    daily = analysis.run_asof_lift(completed_windows_only=True)
    assert len(daily) == 1
    assert daily[0].multiplicity_status == "declared_plan"
    assert daily[0].sequential_result == _row(analysis).require_sequential_result()
    assert daily[0].n_control == 96
    assert daily[0].ds == label(14)
    assert type(daily[0].ds) is type(label(14))
    from increment.breakout.estimates import DailyLiftEstimate
    from increment.sequential_state import snapshot_from_json

    restored = snapshot_from_json(full.model_dump_json())
    assert restored == full
    assert type(restored.reveal_cursor) is type(label(14))
    from increment.estimation.readout_types import ReadoutResults

    scoped_daily = type(daily)(daily, metadata=daily.metadata, sequential_snapshot=full)
    restored_daily = ReadoutResults.model_validate_json(scoped_daily.model_dump_json())
    assert restored_daily.metadata is not None
    assert restored_daily.metadata == daily.metadata
    if axis in ("date", "datetime", "numeric", "numeric_float", "structured"):
        assert type(restored_daily.metadata.scope.families[0].identity_look) is type(label(14))
    assert DailyLiftEstimate.model_validate_json(daily[0].model_dump_json()) == daily[0]
    path = tmp_path / "panel-checkpoint.parquet"
    analysis.export(path)
    replay = Analysis.from_moments(
        pq.read_table(path).to_pylist(), metrics=specs, control="control"
    )
    assert replay.sequential_snapshot() == full
    assert _row(replay).sequential_result == _row(analysis).sequential_result
    replayed_daily = replay.run_asof_lift(completed_windows_only=True)
    assert list(replayed_daily) == list(daily)
    assert type(replayed_daily[0].ds) is type(label(14))


@pytest.mark.slow
def test_registered_asof_secondary_family_is_joint_per_look():
    specs = [
        MetricSpec(name=name, type="conversion", window_days=2)
        for name in ("outcome", "other", "third")
    ]
    from increment import SequentialCell

    cells = (
        SequentialCell(metric="outcome", group_id="treatment", family=False, alpha=Fraction(1, 20)),
        SequentialCell(metric="other", group_id="treatment", family=True, alpha=Fraction(1, 40)),
        SequentialCell(metric="third", group_id="treatment", family=True, alpha=Fraction(1, 40)),
    )
    plan = gaussian_plan(
        specs,
        cells=cells,
        law="bernoulli",
        secondaries=["other", "third"],
        source_id="experiment",
        unit="unit",
        group="arm",
        date="day",
        exposure_date="exposed",
    ).model_copy(update={"primary": "outcome"})
    frame = _frame([0, 0, 0, 1] * 24, [0, 1, 1, 1] * 24)
    frame["other"] = frame["outcome"]
    frame["third"] = frame["outcome"]
    start = date(2025, 1, 1)
    frame["day"] = start
    frame["exposed"] = start
    analysis = Analysis.from_unit_panel(
        frame,
        unit="unit",
        group="arm",
        date="day",
        exposure_date="exposed",
        control="control",
        metrics=specs,
        experiment_id="experiment",
        plan=plan,
        observation_end=start + timedelta(days=20),
    )
    analysis.capture_sequential(finalized=True, as_of=start + timedelta(days=14))

    results = analysis.run_asof_lift(completed_windows_only=True)

    assert {row.metric for row in results} == {"outcome", "other", "third"}
    assert results.metadata is not None
    families = results.metadata.scope.families
    assert len(families) == 2
    secondary_family = next(
        family for family in families if any(member.metric == "other" for member in family.members)
    )
    assert secondary_family.complete
    assert {member.metric for member in secondary_family.members} == {"other", "third"}
    assert {row.family_id for row in results if row.metric in {"other", "third"}} == {
        secondary_family.family_id
    }


@pytest.mark.slow
def test_current_envelope_rejects_relabeling_duplicate_rows_and_legacy(tmp_path):
    import pyarrow.parquet as pq

    specs = [MetricSpec(name="outcome", type="conversion")]
    analysis = _analysis(_frame([0, 1] * 8, [1, 1] * 8), specs, _plan(specs, "bernoulli"))
    path = tmp_path / "checkpoint.parquet"
    analysis.export(path)
    payload = pq.read_table(path).to_pylist()
    for mutation in (
        [{**payload[0], "experiment_id": "other"}],
        [payload[0], payload[0]],
        [{**payload[0], "moments_format": 7}],
    ):
        with pytest.raises(CapabilityError):
            Analysis.from_moments(mutation, metrics=specs, control="control")


_SUBNORMAL = 5e-324


@pytest.mark.slow
@pytest.mark.parametrize(
    "c,t,decisive",
    [
        pytest.param(
            [1.0, 2.5, -0.75, 3.0] * 24 + [_SUBNORMAL, 2.0],
            [6.0, 10.5, 14.25, 18.0] * 24 + [3 * _SUBNORMAL, 9.0],
            True,
            id="decisive-signed-subnormal",
        ),
        pytest.param(
            [1.0, 2.5, -0.75, 3.0] * 24 + [_SUBNORMAL, 1e300],
            [6.0, 10.5, 14.25, 18.0] * 24 + [3 * _SUBNORMAL, -1e300],
            False,
            id="extreme-magnitude",
        ),
    ],
)
def test_exported_checkpoint_replays_exact_rationals_and_refuses_mutated_ones(
    tmp_path, c, t, decisive
):
    """A genuinely valid export replays state, hashes, decisions and bounds exactly;
    the same envelope with one rational rewritten in scientific notation refuses
    with the shared wire code, so a malformed-registration error cannot stand in."""
    import json
    import re

    import pyarrow.parquet as pq

    from increment.sequential_state import snapshot_from_json

    specs = [MetricSpec(name="outcome", type="mean")]
    plan = AnalysisPlan(primary="outcome", inference=InferenceSpec(kind="asymptotic_mean"))
    design = Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5})
    analysis = Analysis.from_unit_summary(
        _frame(c, t),
        unit="unit",
        group="arm",
        design=design,
        metrics=specs,
        experiment_id="experiment",
        plan=plan,
        exposure_date="exposure",
    )
    row = _row(analysis)
    result = row.require_asymptotic_sequential_result()
    snapshot = analysis.sequential_snapshot()
    wire = snapshot.model_dump_json()
    components = [
        part.lstrip("+-")
        for spelled in re.findall(r'"[+-]?[0-9]+(?:/[0-9]+)?"', wire)
        for part in spelled.strip('"').split("/")
    ]
    # A subnormal's 2**1074 denominator and 300-digit numerators travel as plain
    # integer quotients; the extreme frame also carries a signed mean.
    assert any(state.mean[0].denominator % 2**1074 == 0 for state in snapshot.states)
    assert max(map(len, components)) > 300
    assert row.stat_sig() is decisive
    if decisive:
        lower = result.bounds.components[0].lower
        assert lower is not None and lower > 1
    else:
        lowest = min(state.mean[0] for state in snapshot.states)
        assert lowest < 0 and f'"{lowest}"' in wire

    path = tmp_path / "checkpoint.parquet"
    analysis.export(path)
    payload = pq.read_table(path).to_pylist()
    assert payload[0]["moments_format"] == 10
    replay = Analysis.from_moments(payload, metrics=specs, control="control")
    replayed = replay.sequential_snapshot()
    assert replayed == snapshot
    assert replayed.states == snapshot.states
    assert replayed.registration == snapshot.registration
    assert (replayed.registration_id, replayed.prefix_id) == (
        snapshot.registration_id,
        snapshot.prefix_id,
    )
    assert snapshot_from_json(replayed.model_dump_json()) == snapshot
    replayed_row = _row(replay)
    assert replayed_row.require_asymptotic_sequential_result() == result
    assert replayed_row.require_lift() == row.require_lift()
    assert replayed_row.stat_sig() is row.stat_sig()
    assert analysis.capture_sequential(finalized=True, previous=replayed) == snapshot

    envelope = json.loads(payload[0]["sequential_snapshot"])
    rewrites = {
        "SequentialArmState.mean": lambda s: s["states"][0]["mean"].__setitem__(0, "1e50000"),
        "SequentialArmState.scatter": lambda s: s["states"][1]["scatter"][0].__setitem__(
            0, "1e50000"
        ),
        "SequentialRegistration.q": lambda s: s["registration"].__setitem__("q", "1e50000"),
        "ScalarMeanModel.rho": lambda s: s["registration"]["models"][0].__setitem__(
            "rho", "1e50000"
        ),
    }
    for field, rewrite in rewrites.items():
        mutated = json.loads(json.dumps(envelope))
        rewrite(mutated)
        with pytest.raises(InvalidRequestError) as refused:
            Analysis.from_moments(
                [{**payload[0], "sequential_snapshot": json.dumps(mutated)}],
                metrics=specs,
                control="control",
            )
        assert refused.value.code == "sequential.wire.rational_invalid"
        assert refused.value.context["field"] == field
        assert "1e50000" not in str(refused.value)
    # The compiled plan carries the same registration and must retain the same refusal.
    plan_payload = json.loads(payload[0]["decision_plan"])
    plan_payload["inference"]["registration"]["q"] = "1e50000"
    with pytest.raises(InvalidRequestError) as wrapped:
        Analysis.from_moments(
            [{**payload[0], "decision_plan": json.dumps(plan_payload)}],
            metrics=specs,
            control="control",
        )
    assert wrapped.value.code == "sequential.wire.rational_invalid"
    assert wrapped.value.context["field"] == "SequentialRegistration.q"
    assert "1e50000" not in str(wrapped.value)


def _declare_fixture_trigger(definition) -> None:
    """A fact-based trigger whose feed the fixture's evidence certifies."""
    definition["fact_sources"][0]["facts"].append({"name": "triggered_event", "column": None})
    definition["exposures"].append({"name": "triggered", "fact": "triggered_event"})
    definition["experiments"][0]["trigger"] = "triggered"


def _fixture_trigger_event(unit: str) -> dict:
    """Units below index 12 trigger half an hour after assignment, before any outcome."""
    from datetime import UTC, datetime

    return {
        "unit_id": unit,
        "event": "triggered_event",
        "value": 0.0,
        "ts": datetime(2025, 1, 1, 0, 30, tzinfo=UTC),
    }


def _certified_fixture_evidence():
    from datetime import UTC, datetime

    from increment import SourceSnapshotEvidence

    certified = datetime(2025, 1, 31, tzinfo=UTC)
    return SourceSnapshotEvidence(certified, {"events": certified, "uptake_events": certified})


def _native_fixture(
    law, *, uptake_only=False, unbounded_retention=False, triggered=False, registration_q=None
):
    from datetime import UTC, datetime

    # All declarations precede construction or reading of either source table.
    from typing import Any

    import ibis
    import pandas as pd

    from increment.semantics.models import Definitions
    from tests.analysis_factory import make_analysis

    fact_sources: list[dict[str, Any]] = [
        {
            "name": "events",
            "sql": "SELECT * FROM events",
            "timestamp_column": "ts",
            "entities": ["unit_id"],
            "facts": [{"name": "outcome_event", "column": "value"}],
        }
    ]
    definition: dict[str, Any] = {
        "dialect": "duckdb",
        "fact_sources": fact_sources,
        "exposures": [
            {
                "name": "assigned",
                "sql": "SELECT unit_id, ts, group_id FROM enrolled",
            }
        ],
        "metrics": [
            {
                "name": "outcome",
                "type": "conversion" if law == "bernoulli" else "mean",
                "entity": "unit_id",
                "fact": "outcome_event",
                "window_days": 2,
                "preferred_direction": "increase",
                **({"aggregation": "sum"} if law in ("gaussian", "scalar_mean") else {}),
            }
        ],
        "experiments": [
            {
                "name": "experiment",
                "exposure": "assigned",
                "unit": "unit_id",
                "control_group": "control",
                "start": "2025-01-01",
                "end": "2025-01-20",
                "plan": {"primary": "outcome"},
            }
        ],
    }
    if unbounded_retention:
        definition["metrics"].append(
            {
                "name": "stay",
                "type": "retention",
                "entity": "unit_id",
                "fact": "outcome_event",
                "threshold_days": 1,
                "preferred_direction": "increase",
            }
        )
    if triggered:
        _declare_fixture_trigger(definition)
    from increment import SequentialCell, SequentialModel
    from increment.semantics.design import Encouragement

    design: Randomized | Encouragement = Randomized(control_group="control")
    if law == "gaussian_ratio":
        definition["metrics"] = [
            {
                "name": "outcome",
                "type": "ratio",
                "entity": "unit_id",
                "numerator": {"fact": "outcome_event", "aggregation": "sum", "window_days": 2},
                "denominator": {
                    "fact": "denominator_event",
                    "aggregation": "sum",
                    "window_days": 2,
                },
                "preferred_direction": "increase",
            }
        ]
        definition["fact_sources"][0]["facts"].append(
            {"name": "denominator_event", "column": "value"}
        )
    if uptake_only:
        design = Encouragement.model_validate(
            {
                "control_group": "control",
                "one_sided": False,
                "uptake": {"fact": "clicked", "window_days": 2},
                "exclusion_restriction": {
                    "acknowledged": True,
                    "justification": "inert encouragement",
                },
            }
        )
        fact_sources.append(
            {
                "name": "uptake_events",
                "sql": "SELECT * FROM uptake_events",
                "timestamp_column": "ts",
                "entities": ["unit_id"],
                "facts": [{"name": "clicked", "column": None}],
            }
        )
    defs = Definitions.model_validate(definition)
    if law == "scalar_mean":
        from tests.asymptotic_cases import mean_registration

        base = mean_registration()
    else:
        base = registration(law)
    base = (
        base.model_copy(update={"q": Fraction(str(registration_q))})
        if registration_q is not None
        else base
    )
    if uptake_only:
        model = SequentialModel.model_validate(
            {**base.models[0].model_dump(), "metric": "uptake", "observable": "uptake"}
        )
        base = SequentialRegistration.model_validate(
            {
                **base.model_dump(),
                "models": (model,),
                "roster": (
                    SequentialCell(metric="uptake", group_id="treatment", estimand="compliance"),
                ),
            }
        )
    reg = SequentialRegistration.model_validate(
        {
            **base.model_dump(),
            "definitions_id": sequential_definition_id(
                defs.metrics,
                design,
                source_mapping=native_observation_mapping(defs, defs.experiments[0]),
            ),
        }
    )
    plan = AnalysisPlan(
        primary="outcome",
        inference=InferenceSpec(
            kind="asymptotic_mean" if law == "scalar_mean" else "always_valid", registration=reg
        ),
        compliance=SequentialCompliancePolicy(alpha=reg.roster[0].alpha) if uptake_only else None,
    )
    experiment = defs.experiments[0].model_copy(update={"plan": plan})
    defs = defs.model_copy(update={"experiments": (experiment,)})
    if law == "bernoulli":
        control, treatment = [0, 0, 0, 1] * 24, [0, 1, 1, 1] * 24
    elif law == "gaussian_ratio":
        control, treatment = _paired_ratio_values()
    else:
        control, treatment = [1, 2, 3, 4] * 24, [6, 10, 14, 18] * 24
    enrolled, events = [], []
    for i, pair in enumerate(zip(control, treatment, strict=True)):
        for arm, value in zip(("control", "treatment"), pair, strict=True):
            unit = f"{i:08d}-{arm}"
            enrolled.append(
                {"unit_id": unit, "group_id": arm, "ts": datetime(2025, 1, 1, tzinfo=UTC)}
            )
            values = value if isinstance(value, tuple) else (value,)
            for event, component in zip(
                ("outcome_event", "denominator_event"), values, strict=False
            ):
                if component:
                    events.append(
                        {
                            "unit_id": unit,
                            "event": event,
                            "value": float(component),
                            "ts": datetime(2025, 1, 1, 1, tzinfo=UTC),
                        }
                    )
            if triggered and i < 12:
                events.append(_fixture_trigger_event(unit))
    connection = ibis.duckdb.connect()
    connection.create_table("enrolled", pd.DataFrame(enrolled))
    if uptake_only:
        connection.create_table(
            "uptake_events", pd.DataFrame([{**row, "event": "clicked"} for row in events])
        )
    else:
        connection.create_table("events", pd.DataFrame(events))
    try:
        analysis = make_analysis(
            connection,
            defs,
            experiment=experiment,
            _design=design,
            metrics=list(defs.metrics) if unbounded_retention else None,
            source_snapshot_evidence=_certified_fixture_evidence() if triggered else None,
        )
    except BaseException:
        connection.disconnect()
        raise
    return connection, defs, analysis


@pytest.mark.slow
def test_from_definitions_auto_binding_preserves_omitted_q(tmp_path):
    import json

    import yaml

    from increment.semantics.models import InferenceSpec

    connection, defs, seeded_analysis = _native_fixture("bernoulli")
    try:
        experiment = defs.experiments[0]
        plan = AnalysisPlan(primary="outcome", inference=InferenceSpec(kind="always_valid"))
        auto_experiment = experiment.model_copy(
            update={
                "plan": plan,
                "allocation": {"control": 0.5, "treatment": 0.5},
                "allocation_scheme": "independent",
            }
        )
        auto_defs = defs.model_copy(update={"experiments": (auto_experiment,)})
        payload = auto_defs.model_dump(mode="json")
        payload["experiments"][0]["plan"].pop("q", None)
        definitions_path = tmp_path / "automatic.yaml"
        definitions_path.write_text(yaml.safe_dump(payload))

        automatic = Analysis.from_definitions(
            "experiment", definitions_path, connection, store="none"
        )
        try:
            bound_plan = automatic.experiment.plan
            assert bound_plan.inference is not None
            assert bound_plan.inference.registration is not None
            assert "q" not in bound_plan.model_fields_set
            from increment import compile_unit_day_artifact_context

            context = compile_unit_day_artifact_context("experiment", auto_defs)
            assert "plan_q_explicit" not in json.loads(context.canonical_json)
            automatic.capture_sequential(finalized=True, as_of=date(2025, 1, 16))
            assert len(automatic.run()) == 1
        finally:
            automatic.close()
    finally:
        seeded_analysis.close()


@pytest.mark.slow
def test_omitted_plan_q_survives_native_and_artifact_replay():
    import json
    from datetime import date
    from typing import cast

    from increment.query.artifact_digest import canonical_json
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.artifact import ArtifactContext, _artifact_context_digest
    from increment.sources import MomentSource

    connection, defs, native = _native_fixture("bernoulli", registration_q=0.20)
    try:
        native.capture_sequential(finalized=True, as_of=date(2025, 1, 16))
        native_row = _row(native)
        context = artifact_context(defs, defs.experiments[0], "error")
        context_payload = json.loads(context.canonical_json)
        assert "plan_q_explicit" not in context_payload
        assert context_payload["experiment"]["plan"]["q"] == 0.1

        # Older contexts serialized the default q without recording explicitness.
        legacy_payload = json.loads(context.canonical_json)
        legacy_canonical = canonical_json(legacy_payload)
        legacy_context = ArtifactContext(
            context_format=context.context_format,
            canonical_json=legacy_canonical,
            sha256=_artifact_context_digest(legacy_canonical),
        )
        from increment.query.source import _artifact_source_context

        _restored_experiment, legacy_source_context = _artifact_source_context(legacy_context)
        assert legacy_source_context.plan.q_explicit is False

        # The new explicit-default marker must survive context replay and keep
        # the registration-q refusal active before the source can read evidence.
        explicit_experiment = defs.experiments[0].model_copy(
            update={"plan": defs.experiments[0].plan.model_copy(update={"q": 0.1})}
        )
        explicit_defs = defs.model_copy(update={"experiments": (explicit_experiment,)})
        explicit_context = artifact_context(explicit_defs, explicit_experiment, "error")
        explicit_payload = json.loads(explicit_context.canonical_json)
        assert explicit_payload["plan_q_explicit"] is True
        assert explicit_payload["experiment"]["plan"]["q"] == 0.1
        _restored_experiment, explicit_source_context = _artifact_source_context(explicit_context)
        assert explicit_source_context.plan.q_explicit is True

        from increment import readouts

        class UnreadSource:
            capabilities = frozenset({"total"})
            breakouts = ()
            shape = None

            def __init__(self):
                self.context = explicit_source_context
                self.moment_calls = 0

            def moments(self, *args, **kwargs):
                self.moment_calls += 1
                raise AssertionError("mismatched explicit artifact q must refuse before reads")

        unread = UnreadSource()
        with pytest.raises(CapabilityError) as mismatched:
            readouts.run(cast(MomentSource, unread))
        assert mismatched.value.code == "sequential.source.invalid"
        assert unread.moment_calls == 0

        store = WarehouseArtifactStore(connection, schema_name="artifacts")
        reference = native.publish_unit_day_artifact(store)
        with Analysis.from_unit_day_artifact(store, reference, expected_context=context) as adopted:
            adopted.capture_sequential(finalized=True, as_of=date(2025, 1, 16))
            artifact_row = _row(adopted)
            assert (
                artifact_row.require_sequential_result() == native_row.require_sequential_result()
            )
            assert artifact_row.family_q == native_row.family_q
    finally:
        native.close()


@pytest.mark.slow
@pytest.mark.parametrize("law", ["bernoulli", "scalar_mean"])
def test_native_artifact_and_current_wire_replay_actual_likelihood(tmp_path, law):
    import pyarrow.parquet as pq

    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore

    connection, defs, native = _native_fixture(law)
    try:
        with pytest.raises(CapabilityError):
            native.run()
        with pytest.raises(CapabilityError) as missing_horizon:
            native.capture_sequential(finalized=True)
        assert missing_horizon.value.code == "sequential.source.invalid"
        first = native.capture_sequential(finalized=True, as_of=date(2025, 1, 16))
        native_row = _row(native)
        assert native_row.stat_sig()
        assert native.capture_sequential(finalized=True, as_of=date(2025, 1, 16)) == first
        context = artifact_context(defs, defs.experiments[0], "error")
        store = WarehouseArtifactStore(connection, schema_name="artifacts")
        reference = native.publish_unit_day_artifact(store)
        with Analysis.from_unit_day_artifact(store, reference, expected_context=context) as adopted:
            with pytest.raises(CapabilityError) as missing_horizon:
                adopted.capture_sequential(finalized=True)
            assert missing_horizon.value.code == "sequential.source.invalid"
            captured = adopted.capture_sequential(finalized=True, as_of=date(2025, 1, 16))
            assert captured.states == first.states
            assert captured.prefix_id == first.prefix_id
            assert (
                _row(adopted).require_sequential_result() == native_row.require_sequential_result()
            )
            assert _row(adopted).stat_sig()
            native_daily = native.run_asof_lift(completed_windows_only=True)
            adopted_daily = adopted.run_asof_lift(completed_windows_only=True)
            _assert_cross_source_readouts_equal(adopted_daily, native_daily)
            path = tmp_path / "artifact-checkpoint.parquet"
            adopted.export(path)
            specs = [
                MetricSpec(name="outcome", type="mean" if law == "scalar_mean" else "conversion")
            ]
            replayed = Analysis.from_moments(
                pq.read_table(path).to_pylist(), metrics=specs, control="control"
            )
            assert (
                _row(replayed).require_sequential_result() == native_row.require_sequential_result()
            )
            replayed_daily = replayed.run_asof_lift(completed_windows_only=True)
            _assert_cross_source_readouts_equal(replayed_daily, native_daily)
        # New units enter after every previously finalized unit in reveal order.
        connection.raw_sql(
            "INSERT INTO enrolled VALUES "
            "('00000096-control', 'control', TIMESTAMPTZ '2025-01-02 00:00:00+00'), "
            "('00000096-treatment', 'treatment', TIMESTAMPTZ '2025-01-02 00:00:00+00')"
        )
        treatment_value = 1
        connection.raw_sql(
            "INSERT INTO events VALUES ('00000096-treatment', 'outcome_event', "
            f"{treatment_value}, TIMESTAMPTZ '2025-01-02 01:00:00+00')"
        )
        appended = native.capture_sequential(finalized=True, as_of=date(2025, 1, 17))
        assert appended.parent_id == first.prefix_id
        assert appended.records[: len(first.records)] == first.records
        assert len(appended.records) == len(first.records) + 2
        next_reference = native.publish_unit_day_artifact(store)
        with Analysis.from_unit_day_artifact(
            store, next_reference, expected_context=context
        ) as next_artifact:
            continued = next_artifact.capture_sequential(
                finalized=True,
                as_of=date(2025, 1, 17),
                previous=first,
            )
            assert continued == appended
            assert (
                _row(next_artifact).require_sequential_result()
                == _row(native).require_sequential_result()
            )
        # A source correction cannot use an omitted previous= to reset its retained prefix.
        connection.raw_sql(
            "UPDATE enrolled SET group_id = 'treatment' WHERE unit_id = '00000000-control'"
        )
        with pytest.raises(CapabilityError) as raised:
            native.capture_sequential(finalized=True, as_of=date(2025, 1, 16))
        assert raised.value.code == "sequential.continuation.rewrite"
    finally:
        native.close()


@pytest.mark.slow
def test_native_gaussian_capture_refuses_before_public_replay():
    with pytest.raises(CapabilityError) as raised:
        _native_fixture("gaussian")
    assert raised.value.code == "sequential.route.unsupported"


@pytest.mark.parametrize("publication", [False, True])
def test_assignment_validation_cannot_split_the_source_snapshot(monkeypatch, publication):
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore

    connection, defs, native = _native_fixture("bernoulli")
    try:
        expected = native.capture_sequential(finalized=True, as_of=date(2025, 1, 16))
        expected_result = _row(native).require_sequential_result()

        source = _native_source(native)
        source_type = type(source)
        validate = source_type._validate_mixed_assignments

        def rewrite_live_inputs_after_assignment_validation(pinned):
            validate(pinned)
            connection.raw_sql("UPDATE events SET value = value + 100, ts = ts + INTERVAL 365 DAY")
            connection.raw_sql(
                "UPDATE enrolled SET group_id = 'treatment' WHERE unit_id = '00000000-control'"
            )

        monkeypatch.setattr(
            source_type,
            "_validate_mixed_assignments",
            rewrite_live_inputs_after_assignment_validation,
        )
        if publication:
            context = artifact_context(defs, defs.experiments[0], "error")
            store = WarehouseArtifactStore(connection, schema_name="artifacts")
            reference = native.publish_unit_day_artifact(store)
            with Analysis.from_unit_day_artifact(
                store, reference, expected_context=context
            ) as adopted:
                snapshot = adopted.capture_sequential(finalized=True, as_of=date(2025, 1, 16))
                assert snapshot.prefix_id == expected.prefix_id
                assert _row(adopted).require_sequential_result() == expected_result
        else:
            snapshot = native.capture_sequential(finalized=True, as_of=date(2025, 1, 16))
            assert snapshot == expected
            assert _row(native).require_sequential_result() == expected_result
        # The live rewrite occurred; equality above comes from the pinned inputs.
        changed = connection.table("events").to_pyarrow().to_pylist()
        assert all(row["value"] > 100 for row in changed)
    finally:
        native.close()


@pytest.mark.parametrize("observable", ["uptake", "outcome"])
def test_metric_free_sequential_frames_require_uptake_only_registration(tmp_path, observable):
    import pyarrow.parquet as pq

    from increment import SequentialCell
    from increment.errors import CodedError
    from increment.semantics.design import Encouragement

    design = Encouragement.model_validate(
        {
            "control_group": "control",
            "one_sided": False,
            "uptake": {"fact": "clicked"},
            "exclusion_restriction": {"acknowledged": True, "justification": "uptake only"},
        }
    )
    base = registration("bernoulli")
    model = base.models[0].model_copy(update={"metric": observable, "observable": observable})
    reg = SequentialRegistration.model_validate(
        {
            **base.model_dump(),
            "models": (model,),
            "roster": (
                SequentialCell(
                    metric=observable,
                    group_id="treatment",
                    estimand="compliance" if observable == "uptake" else "itt",
                ),
            ),
            "definitions_id": sequential_definition_id(
                [],
                design,
                transformations=[],
                source_mapping=frame_observation_mapping(
                    unit="unit", group="arm", uptake="clicked", exposure_date="exposure"
                ),
            ),
        }
    )
    plan = AnalysisPlan(
        inference=InferenceSpec(kind="always_valid", registration=reg),
        compliance=SequentialCompliancePolicy(alpha=reg.roster[0].alpha),
    )
    frame = _frame([0, 0, 0, 1] * 24, [0, 1, 1, 1] * 24).rename(columns={"outcome": "clicked"})

    def build():
        return Analysis.from_unit_summary(
            frame,
            unit="unit",
            group="arm",
            metrics=[],
            design=design,
            plan=plan,
            experiment_id=reg.source_id,
            exposure_date="exposure",
        )

    if observable == "outcome":
        with pytest.raises(CodedError) as rejected:
            build()
        assert rejected.value.code == "sequential.source.invalid"
        return
    analysis = build()
    snapshot = analysis.capture_sequential(finalized=True)
    rows = analysis.run(estimands=("compliance",))
    assert isinstance(rows, LiftEstimates) and len(rows) == 1
    checkpoint = rows[0].require_sequential_result().checkpoint
    assert checkpoint.control.successes == 24 and checkpoint.treatment.successes == 72
    path = tmp_path / "metric-free-checkpoint.parquet"
    analysis.export(path)
    for replay_plan in (None, plan):
        replay = Analysis.from_moments(
            pq.read_table(path).to_pylist(), metrics=[], design=design, plan=replay_plan
        )
        assert replay.sequential_snapshot() == snapshot
        assert list(replay.run(estimands=("compliance",))) == list(rows)


@pytest.mark.parametrize(
    "kind,family,baseline_rate",
    [
        ("always_valid", False, None),
        ("always_valid", True, None),
        ("always_valid", None, None),
        ("asymptotic_mean", False, None),
        ("asymptotic_mean", True, None),
        ("asymptotic_mean", None, None),
        ("always_valid", False, 0.25),
        ("always_valid", True, 0.25),
    ],
)
def test_automatic_metric_free_sequential_compliance(kind, family, baseline_rate):
    from increment.errors import CodedError
    from increment.semantics.design import Encouragement

    design = Encouragement.model_validate(
        {
            "control_group": "control",
            "allocation": {"control": 0.5, "treatment": 0.5},
            "uptake": {"fact": "clicked"},
            "exclusion_restriction": {"acknowledged": True, "justification": "uptake only"},
        }
    )
    frame = _frame([0, 0, 0, 1] * 24, [0, 1, 1, 1] * 24).rename(columns={"outcome": "clicked"})
    plan = AnalysisPlan(
        inference=InferenceSpec(kind=kind, baseline_rate=baseline_rate),
        compliance=None
        if family is None
        else SequentialCompliancePolicy(alpha=0.05, family=family),
    )

    def build():
        return Analysis.from_unit_summary(
            frame,
            unit="unit",
            group="arm",
            metrics=[],
            design=design,
            uptake="clicked",
            exposure_date="exposure",
            plan=plan,
        )

    if family is None or kind == "asymptotic_mean":
        with pytest.raises(CodedError) as rejected:
            build()
        assert rejected.value.code == "sequential.source.invalid"
        return
    analysis = build()
    analysis.capture_sequential(finalized=True)
    (row,) = analysis.run(estimands=("compliance",))
    checkpoint = row.require_sequential_result().checkpoint
    assert checkpoint.control.n == checkpoint.treatment.n == 96
    assert checkpoint.control.successes == 24 and checkpoint.treatment.successes == 72
    from math import log

    from scipy.special import betaln

    from increment.sequential_source import DEFAULT_BERNOULLI_PRIOR_WEIGHT

    a, b = (
        (1.0, 1.0)
        if baseline_rate is None
        else (
            DEFAULT_BERNOULLI_PRIOR_WEIGHT * baseline_rate,
            DEFAULT_BERNOULLI_PRIOR_WEIGHT * (1 - baseline_rate),
        )
    )
    expected_log_e = (
        betaln(a + 24, b + 72) + betaln(a + 72, b + 24) - 2 * betaln(a, b) + 192 * log(2)
    )
    assert float(row.require_exact_sequential_result().log_e) == pytest.approx(
        expected_log_e, abs=1e-10
    )


@pytest.mark.slow
def test_direct_artifact_source_captures_encouragement_sequential_state():
    from datetime import UTC, datetime

    import pandas as pd

    from increment.query.artifact_contract import unit_day_artifact_extension_catalog
    from increment.query.artifact_reader import ArtifactMomentSource
    from increment.query.session import WarehouseArtifactStore

    connection, defs, native = _native_fixture("bernoulli", uptake_only=True, triggered=True)
    try:
        events = []
        for arm in ("control", "treatment"):
            for index in range(96):
                unit_id = f"{index:08d}-{arm}"
                events.append(
                    {
                        "unit_id": unit_id,
                        "event": "outcome_event",
                        "value": float(index % 2),
                        "ts": datetime(2025, 1, 1, 1, tzinfo=UTC),
                    }
                )
                if index < 12:
                    events.append(
                        {
                            "unit_id": unit_id,
                            "event": "triggered_event",
                            "value": None,
                            "ts": datetime(2025, 1, 2, tzinfo=UTC),
                        }
                    )
        connection.create_table("events", pd.DataFrame(events))
        context = native.artifact_context
        store = WarehouseArtifactStore(connection, schema_name="direct_sequential")
        extensions = [
            entry.request
            for entry in unit_day_artifact_extension_catalog(context)
            if entry.request.kind
            in {"trigger_population", "trigger_measure_stats", "encouragement_uptake"}
        ]
        reference = native.publish_unit_day_artifact(store, extensions=extensions)
        with ArtifactMomentSource.open(
            store,
            reference,
            expected_context=context,
        ) as source:
            snapshot = source.capture_sequential(finalized=True, as_of=date(2025, 1, 16))

        assert snapshot.registration.models[0].observable == "uptake"
        assert snapshot.records
    finally:
        native.close()
        connection.disconnect()


@pytest.mark.slow
def test_native_uptake_only_capture_and_wire_never_need_the_outcome_table(tmp_path):
    import pyarrow.parquet as pq

    from increment.breakout.estimates import LiftEstimates

    connection, _defs, native = _native_fixture("bernoulli", uptake_only=True)
    try:
        assert "events" not in connection.list_tables()
        snapshot = native.capture_sequential(finalized=True, as_of=date(2025, 1, 16))
        rows = native.run(estimands=("compliance",))
        assert isinstance(rows, LiftEstimates)
        assert len(rows) == 1 and rows[0].stat_sig()
        checkpoint = rows[0].require_sequential_result().checkpoint
        assert checkpoint.control.n == checkpoint.treatment.n == 96
        assert checkpoint.control.successes == 24
        assert checkpoint.treatment.successes == 72
        daily = native.run_asof_lift(estimands=("compliance",), completed_windows_only=True)
        assert len(daily) == 1 and daily[0].ds == date(2025, 1, 16)
        path = tmp_path / "uptake-checkpoint.parquet"
        native.export(path)
        for specs in ([], [MetricSpec(name="outcome", type="conversion")]):
            replay = Analysis.from_moments(
                pq.read_table(path).to_pylist(),
                metrics=specs,
                design=_native_source(native).context.design,
            )
            assert replay.sequential_snapshot() == snapshot
            assert list(replay.run(estimands=("compliance",))) == list(rows)
            replayed_daily = replay.run_asof_lift(
                estimands=("compliance",), completed_windows_only=True
            )
            _assert_cross_source_readouts_equal(replayed_daily, daily)
    finally:
        native.close()


@pytest.mark.slow
def test_uptake_checkpoint_ignores_unbounded_outcome_retention_in_the_catalog(tmp_path):
    """A compliance-only checkpoint reads uptake, never the outcome table, so a
    retention metric in the source catalog neither makes the facade's completed
    windows contradictory nor trips its encouragement guard; any request that
    consumes outcomes keeps both refusals. The portable replay keeps that same
    catalog (an empty one would not exercise the exemption); the unit-day
    artifact route is the parity case
    `audit-compliance-only-sequential-unbounded-retention-catalog`."""
    import pyarrow.parquet as pq

    from increment.errors import CodedError

    connection, _, native = _native_fixture("bernoulli", uptake_only=True, unbounded_retention=True)
    try:
        assert "events" not in connection.list_tables()
        as_of = date(2025, 1, 16)
        snapshot = native.capture_sequential(finalized=True, as_of=as_of)
        (row,) = native.run_asof_lift(estimands=("compliance",), completed_windows_only=True)
        assert row.estimand == "compliance" and row.ds == as_of
        assert row.sequential_result is not None
        checkpoint = row.sequential_result.checkpoint
        assert checkpoint.control.n == checkpoint.treatment.n == 96
        assert checkpoint.control.successes == 24 and checkpoint.treatment.successes == 72
        daily = native.run_asof_lift(estimands=("compliance",), completed_windows_only=True)

        path = tmp_path / "uptake-retention-catalog.parquet"
        native.export(path)
        replay = Analysis.from_moments(
            pq.read_table(path).to_pylist(),
            metrics=[
                MetricSpec(name="outcome", type="conversion", window_days=2),
                MetricSpec(name="stay", type="retention", threshold_days=1),
            ],
            design=_native_source(native).context.design,
        )
        assert [metric.name for metric in replay.metrics] == ["outcome", "stay"]
        assert replay.sequential_snapshot() == snapshot
        replayed_daily = replay.run_asof_lift(
            estimands=("compliance",), completed_windows_only=True
        )
        _assert_cross_source_readouts_equal(replayed_daily, daily)

        # Every request that reads outcomes still meets the retention guards.
        for source in (native, replay):
            for estimands in (("itt",), ("itt", "compliance"), None):
                with pytest.raises(CodedError) as refused:
                    source.run_asof_lift(estimands=estimands, completed_windows_only=True)
                assert refused.value.code == "breakout.retention.encouragement", estimands
    finally:
        native.close()


def _paired_ratio_values():
    return ([(1, 1), (2, 1), (3, 2), (4, 3)] * 24, [(8, 1), (12, 2), (18, 3), (22, 2)] * 24)


@pytest.mark.slow
def test_frame_gaussian_ratio_refuses_before_public_capture(tmp_path):
    specs = [
        MetricSpec(name="outcome", type="ratio", numerator="outcome", denominator="denominator")
    ]
    plan = _plan(specs, "gaussian_ratio")
    control, treatment = _paired_ratio_values()
    frame = _frame([x for x, _ in control], [x for x, _ in treatment])
    frame["denominator"] = [
        component for pair in zip(control, treatment, strict=True) for _numerator, component in pair
    ]
    with pytest.raises(CapabilityError) as raised:
        _analysis(frame, specs, plan)
    assert raised.value.code == "sequential.route.unsupported"


@pytest.mark.slow
def test_scalar_mean_maturity_boundary_matches_frame_native_and_artifact():
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore

    connection, defs, native = _native_fixture("scalar_mean")
    try:
        plan = defs.experiments[0].plan
        assert plan is not None and plan.inference is not None
        reg = plan.inference.registration
        assert reg is not None
        start = date(2025, 1, 1)
        boundary = start + timedelta(days=reg.reveal.longest_window_days - 1)
        specs = [MetricSpec(name="outcome", type="mean", window_days=2)]
        frame_reg = SequentialRegistration.model_validate(
            {
                **reg.model_dump(),
                "definitions_id": sequential_definition_id(
                    [synthesise_metric(spec) for spec in specs],
                    Randomized(control_group="control"),
                    transformations=specs,
                    source_mapping=frame_observation_mapping(
                        unit="unit", group="arm", date="day", exposure_date="exposed"
                    ),
                ),
            }
        )
        frame = _frame([1, 2, 3, 4] * 24, [6, 10, 14, 18] * 24)
        frame["day"] = frame["exposed"] = start
        panel = Analysis.from_unit_panel(
            frame,
            unit="unit",
            group="arm",
            date="day",
            exposure_date="exposed",
            control="control",
            metrics=specs,
            experiment_id="experiment",
            observation_end=boundary,
            plan=plan.model_copy(
                update={"inference": InferenceSpec(kind="asymptotic_mean", registration=frame_reg)}
            ),
        )
        store = WarehouseArtifactStore(connection, schema_name="artifacts")
        reference = native.publish_unit_day_artifact(store)
        context = artifact_context(defs, defs.experiments[0], "error")
        with Analysis.from_unit_day_artifact(store, reference, expected_context=context) as adopted:
            sources = (panel, native, adopted)
            for source in sources:
                empty = source.capture_sequential(
                    finalized=True, as_of=boundary - timedelta(days=1)
                )
                assert not empty.records
            snapshots = [
                source.capture_sequential(finalized=True, as_of=boundary) for source in sources
            ]
            assert [len(snapshot.records) for snapshot in snapshots] == [192, 192, 192]
            assert snapshots[0].states == snapshots[1].states == snapshots[2].states
            assert (
                _row(panel).require_lift()
                == _row(native).require_lift()
                == _row(adopted).require_lift()
            )
    finally:
        native.close()


def _role_plan(specs, *, alpha, q):
    """Always-valid plan carrying a primary, two family secondaries and a guardrail."""
    from fractions import Fraction

    from increment import (
        JointReveal,
        PredictivePrior,
        SequentialCell,
        SequentialModel,
    )

    design = Randomized(control_group="control")
    prior = PredictivePrior(kind="beta", a=1, b=1)
    reg = SequentialRegistration(
        source_id="experiment",
        definitions_id=sequential_definition_id(
            [synthesise_metric(spec) for spec in specs],
            design,
            transformations=specs,
            source_mapping=frame_observation_mapping(
                unit="unit", group="arm", exposure_date="exposure"
            ),
        ),
        control_group="control",
        committed_before_data=True,
        reveal=JointReveal(
            filtration_id="joint-units-v1",
            independent_unit_vectors=True,
            simultaneous_metrics=True,
            outcome_independent_order=True,
            immutable_finalized_outcomes=True,
            longest_window_days=14,
        ),
        models=tuple(
            SequentialModel(
                metric=spec.name,
                law="bernoulli",
                control_prior=prior,
                treatment_prior=prior,
                positive_population_control=True,
            )
            for spec in specs
        ),
        roster=(
            SequentialCell(metric="checkout", group_id="treatment", alpha=Fraction(alpha)),
            SequentialCell(
                metric="signups", group_id="treatment", alpha=Fraction(alpha), family=True
            ),
            SequentialCell(
                metric="sessions", group_id="treatment", alpha=Fraction(alpha), family=True
            ),
            SequentialCell(
                metric="refunds", group_id="treatment", alpha=Fraction(alpha), alternative="less"
            ),
        ),
        q=Fraction(q),
    )
    return AnalysisPlan(
        alpha=alpha,
        q=q,
        primary="checkout",
        secondaries=["signups", "sessions"],
        guardrails=["refunds"],
        inference=InferenceSpec(kind="always_valid", registration=reg),
    )


def test_always_valid_plan_reports_every_role_at_its_declared_budget():
    """Contract: the primary at alpha, each guardrail one-sided at alpha unsplit,
    and the secondaries as an e-BH family at q -- every role reported, and only
    family members carry a discovery verdict."""
    from fractions import Fraction

    import pandas as pd

    alpha, q = 0.05, 0.20
    specs = [
        MetricSpec(name="checkout", type="conversion"),
        MetricSpec(name="signups", type="conversion"),
        MetricSpec(name="sessions", type="conversion"),
        MetricSpec(name="refunds", type="conversion", preferred_direction="decrease"),
    ]
    plan = _role_plan(specs, alpha=alpha, q=q)
    frame = pd.DataFrame(
        [
            {
                "unit": f"{i:08d}-{arm}",
                "arm": arm,
                "checkout": int(arm == "treatment" and i % 2 == 0 or i % 4 == 0),
                "signups": int(arm == "treatment" and i % 3 != 0 or i % 5 == 0),
                "sessions": int(i % 2 == 0),
                "refunds": int(i % 7 == 0),
                "exposure": i,
            }
            for i in range(100)
            for arm in ("control", "treatment")
        ]
    )
    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        control="control",
        metrics=specs,
        experiment_id="experiment",
        plan=plan,
        exposure_date="exposure",
    )
    observed = {}
    for row in analysis.run():
        # run() is typed as a union; these are arm rows, not contrasts.
        assert isinstance(row, LiftEstimate)
        assert row.multiplicity_status == "declared_plan"
        observed[row.metric] = (
            row.role,
            row.inference,
            row.alternative,
            row.require_sequential_result().decision_alpha,
            row.discovery,
            row.family_axes,
        )
    assert observed["checkout"] == (
        "primary",
        "always_valid",
        "two-sided",
        Fraction(alpha),
        None,
        None,
    )
    assert observed["refunds"] == ("guardrail", "always_valid", "less", Fraction(alpha), None, None)
    assert observed["signups"][:3] == ("secondary", "always_valid", "two-sided")
    assert observed["sessions"][:3] == ("secondary", "always_valid", "two-sided")
    assert observed["signups"][4] is True and observed["signups"][5] == ("metric", "arm")
    assert observed["sessions"][4] is False and observed["sessions"][5] == ("metric", "arm")
