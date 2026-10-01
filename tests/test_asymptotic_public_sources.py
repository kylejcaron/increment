"""Continuous public analysis, wire, capture and reporting contracts."""

from datetime import date, timedelta
from fractions import Fraction as F

import pytest

from increment import Analysis, AsymptoticMean, JointReveal, SequentialCell
from increment.breakout.estimates import DailyLiftEstimate, LiftEstimates
from increment.errors import CapabilityError
from increment.estimation.sequential_result import AsymptoticSequentialResult
from increment.frame import MetricSpec, synthesise_metric
from increment.semantics.design import Randomized
from increment.semantics.models import AnalysisPlan, InferenceSpec, MultiplicitySpec
from increment.sequential_source import frame_observation_mapping, sequential_definition_id
from increment.sequential_state import (
    SequentialSnapshot,
    declare_sequential_freeze,
    declare_sequential_freeze_cells,
    snapshot_from_json,
)
from tests.asymptotic_cases import mean_capture, mean_model, mean_records, mean_registration


def _plan(
    specs,
    *,
    cells=None,
    panel=False,
    secondary=False,
    start=2,
    uncorrected=False,
    law="scalar_mean",
):
    design = Randomized(control_group="control")
    registration = mean_registration(
        definitions_id=sequential_definition_id(
            [synthesise_metric(s) for s in specs],
            design,
            transformations=specs,
            source_mapping=frame_observation_mapping(
                unit="unit",
                group="arm",
                date="day" if panel else None,
                exposure_date="exposed" if panel else "exposure",
            ),
        ),
        models=tuple(mean_model(s.name, start_count=start, law=law) for s in specs),
        cells=cells,
    )
    return AnalysisPlan(
        primary=None if secondary else specs[0].name,
        secondaries=[s.name for s in specs] if secondary else (),
        view_multiplicity=MultiplicitySpec(correction="none" if uncorrected else "bonferroni")
        if cells and cells[0].segment
        else None,
        inference=InferenceSpec(kind="asymptotic_mean", registration=registration),
    )


def _frame(control, treatment, *, offset=0):
    import pandas as pd

    return pd.DataFrame(
        [
            {"unit": r["unit_id"], "arm": r["group_id"], "exposure": i, **r["values"]}
            for i, r in enumerate(mean_records(control, treatment, offset=offset))
        ]
    )


def _analysis(frame, specs, plan, *, design=None):
    return Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        control="control" if design is None else None,
        metrics=specs,
        experiment_id="experiment",
        plan=plan,
        design=design,
        exposure_date="exposure",
    )


def _row(analysis, **kwargs):
    rows = analysis.run(**kwargs)
    assert isinstance(rows, LiftEstimates)
    return rows[0]


@pytest.mark.slow
def test_analysis_wire_export_replay_and_truthful_numeric_columns(tmp_path):
    import pandas as pd
    import pyarrow.parquet as pq

    specs = [MetricSpec(name="outcome", type="mean")]
    plan = AnalysisPlan(primary="outcome", inference=InferenceSpec(kind="asymptotic_mean"))
    frame = _frame([1, 2, 3] * 30, [7, 10, 13] * 30)
    analysis = _analysis(
        frame,
        specs,
        plan,
        design=Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5}),
    )
    rows = analysis.run()
    assert isinstance(rows, LiftEstimates)
    row = rows[0]
    assert row.stat_sig()
    assert row.inference == "asymptotic_mean"
    snapshot = analysis.sequential_snapshot()
    frame.loc[:, "outcome"] = 0
    assert _row(analysis) == row
    assert analysis.capture_sequential(finalized=True, previous=snapshot) == snapshot
    path = tmp_path / "continuous.parquet"
    analysis.export(path)
    replay = Analysis.from_moments(
        pq.read_table(path).to_pylist(), metrics=specs, control="control"
    )
    assert replay.sequential_snapshot() == snapshot
    assert _row(replay) == row
    projected = replay.run().to_frame(backend="pandas")
    assert isinstance(projected, pd.DataFrame)
    assert projected["sequential_log_e"].isna().iloc[0]
    assert projected["sequential_validity_regime"].iloc[0] == "asymptotic_sequential"
    assert projected["sequential_lower"].iloc[0] > 0
    from increment.tables import estimates_to_readout

    displayed = estimates_to_readout([row])[0]
    assert displayed["sequential_log_e"] is None
    assert displayed["sequential_validity_regime"] == "asymptotic_sequential"
    assert displayed["prob_favorable"] is None


@pytest.mark.slow
def test_public_stopping_freezes_first_crossing_and_rejects_rewritten_prefix():
    import pandas as pd

    specs = [MetricSpec(name="outcome", type="mean")]
    plan = _plan(specs, start=4)
    records = _frame([1, 2] * 40, [5, 7] * 40)
    previous = None
    stopped_row = None
    stop_pairs = None
    for pairs in (2, 4, 8, 16, 32, 80):
        current = _analysis(records.iloc[: 2 * pairs].copy(), specs, plan)
        previous = current.sequential_snapshot(previous=previous)
        row = _row(current)
        if row.stat_sig():
            stopped_row = row.require_asymptotic_sequential_result()
            stop_pairs = pairs
            break
    assert stopped_row is not None
    assert stop_pairs is not None
    at_stop = _analysis(records.iloc[: 2 * stop_pairs].copy(), specs, plan)
    declared = at_stop.capture_sequential(finalized=True, previous=previous, freeze=["outcome"])
    assert declared.frozen[0].cell.metric == "outcome"
    assert declared.prefix_id != previous.prefix_id
    assert any(a.prefix_id == previous.prefix_id for a in declared.ancestors)

    final = _analysis(records, specs, plan)
    final.sequential_snapshot(previous=declared)
    frozen = _row(final).require_asymptotic_sequential_result()
    assert frozen.checkpoint.status == "frozen"
    assert frozen.bounds == stopped_row.bounds
    assert frozen.checkpoint.prefix_id == stopped_row.checkpoint.prefix_id

    changed = records.copy()
    changed.loc[0, "outcome"] = 99
    with pytest.raises(CapabilityError) as raised:
        _analysis(changed, specs, plan).sequential_snapshot(previous=declared)
    assert raised.value.code == "sequential.continuation.rewrite"
    assert isinstance(records, pd.DataFrame)


@pytest.mark.slow
def test_freeze_declared_next_look_and_later_look_all_agree():
    """The required declare -> next look -> later look sequence: a freeze
    declared at one look is visible unmodified two and three looks later,
    without ever re-declaring it."""
    specs = [MetricSpec(name="outcome", type="mean")]
    plan = _plan(specs, start=4)
    records = _frame([1, 2] * 40, [5, 7] * 40)
    previous = None
    declared_at = None
    for pairs in (2, 4, 8, 16, 32, 80):
        current = _analysis(records.iloc[: 2 * pairs].copy(), specs, plan)
        freeze = ["outcome"] if pairs == 4 else []
        previous = current.capture_sequential(finalized=True, previous=previous, freeze=freeze)
        if pairs == 4:
            declared_at = previous
        elif declared_at is not None:
            assert previous.frozen[0].cell.metric == "outcome"
            assert previous.frozen[0].prefix_id == declared_at.frozen[0].prefix_id
    assert previous is not None
    assert declared_at is not None
    assert previous.frozen == declared_at.frozen


def test_empty_freeze_roundtrips_and_can_continue():
    specs = [MetricSpec(name="outcome", type="mean")]
    plan = _plan(specs, start=4)
    records = _frame([1, 2] * 40, [5, 7] * 40)
    early = _analysis(records.iloc[:16].copy(), specs, plan)
    snapshot = early.sequential_snapshot()

    # Empty requests must preserve replayable state through both public routes.
    empty_named = declare_sequential_freeze(snapshot, [])
    empty_cells = declare_sequential_freeze_cells(snapshot, [])
    assert empty_named == snapshot
    assert empty_cells == snapshot

    later = _analysis(records.iloc[:32].copy(), specs, plan)
    expected = later.sequential_snapshot(previous=snapshot)
    for empty in (empty_named, empty_cells):
        reloaded = snapshot_from_json(empty.model_dump_json())
        assert later.sequential_snapshot(previous=reloaded) == expected


def test_continuation_refuses_dropping_a_previously_frozen_cell():
    """A malformed continuation that drops a recorded freeze is refused, not
    silently accepted as an ordinary append."""
    specs = [MetricSpec(name="outcome", type="mean")]
    plan = _plan(specs, start=4)
    records = _frame([1, 2] * 40, [5, 7] * 40)
    early = _analysis(records.iloc[:16].copy(), specs, plan)
    declared = early.capture_sequential(finalized=True, freeze=["outcome"])
    later = _analysis(records.iloc[:32].copy(), specs, plan).sequential_snapshot(previous=declared)
    dropped = later.model_copy(update={"frozen": ()})
    with pytest.raises(CapabilityError) as raised:
        dropped.verify_parent(declared)
    assert raised.value.code == "sequential.freeze.dropped"


def test_freezing_cells_out_of_name_order_keeps_every_earlier_freeze():
    """``declare_sequential_freeze_cells`` stores ``frozen`` sorted by
    (metric, group_id, estimand, segment) for a canonical digest, so freezing
    metric ``clicks`` after metric ``revenue`` re-sorts the tuple to
    (clicks, revenue) -- a different POSITION than the (revenue,) it replaces.
    The chain-walk and verify_parent checks must accept this as a legitimate
    same-count declaration step (set-wise superset, unchanged content for every
    carried-forward cell), not refuse it as though a prior freeze changed."""
    reg = mean_registration(
        models=(mean_model("revenue", start_count=4), mean_model("clicks", start_count=4)),
        cells=(
            SequentialCell(metric="revenue", group_id="treatment"),
            SequentialCell(metric="clicks", group_id="treatment"),
        ),
    )
    records = mean_records([1, 2] * 40, [5, 7] * 40, metrics=("revenue", "clicks"))
    early = mean_capture(reg, records[:16])
    declared_revenue = declare_sequential_freeze(early, ["revenue"])
    assert [c.cell.metric for c in declared_revenue.frozen] == ["revenue"]
    mid = mean_capture(reg, records[16:32], previous=declared_revenue, append=True)
    declared_both = declare_sequential_freeze(mid, ["clicks"])
    assert {c.cell.metric for c in declared_both.frozen} == {"revenue", "clicks"}
    final = mean_capture(reg, records[32:], previous=declared_both, append=True)
    assert {c.cell.metric for c in final.frozen} == {"revenue", "clicks"}
    reloaded = SequentialSnapshot.model_validate(final.model_dump())
    assert {c.cell.metric for c in reloaded.frozen} == {"revenue", "clicks"}
    assert reloaded.prefix_id == final.prefix_id


def _freeze_chain_fixture():
    """A genuine 8 -> declare -> 32 chain, plus a plain (never-frozen) 8 -> 32
    chain sharing the SAME early prefix, to splice against."""
    specs = [MetricSpec(name="outcome", type="mean")]
    plan = _plan(specs, start=4)
    records = _frame([1, 2] * 40, [5, 7] * 40)
    early = _analysis(records.iloc[:16].copy(), specs, plan)
    at_8 = early.sequential_snapshot()
    declared = declare_sequential_freeze(at_8, ["outcome"])
    settled = _analysis(records, specs, plan).sequential_snapshot(previous=declared)
    unfrozen_final = _analysis(records, specs, plan).sequential_snapshot(previous=at_8)
    return at_8, declared, settled, unfrozen_final


def test_a_frozen_checkpoint_introduced_without_any_declaration_step_is_refused():
    """Retroactive freeze: a checkpoint whose content is 100% authentic
    (real states, real prefix_id) appears in self.frozen, but the chain
    never actually grew to introduce it at any declaration step."""
    at_8, declared, settled, unfrozen_final = _freeze_chain_fixture()
    forged = unfrozen_final.model_copy(update={"frozen": (declared.frozen[0],)})
    with pytest.raises(CapabilityError) as raised:
        SequentialSnapshot.model_validate(forged.model_dump())
    assert raised.value.code == "sequential.source.invalid"


def test_a_duplicate_ancestor_entry_that_does_not_extend_the_freeze_is_refused():
    """A repeated ancestor at the same look, with an unchanged (not
    strictly-extended) frozen set, is refused -- only a genuine declaration
    step may repeat a look's n_records."""
    at_8, declared, settled, unfrozen_final = _freeze_chain_fixture()
    forged = settled.model_copy(
        update={"ancestors": (settled.ancestors[0], settled.ancestors[0], settled.ancestors[1])}
    )
    with pytest.raises(CapabilityError) as raised:
        SequentialSnapshot.model_validate(forged.model_dump())
    assert raised.value.code == "sequential.source.invalid"


def test_a_frozen_checkpoint_declared_at_the_wrong_ancestor_prefix_is_refused():
    """A newly-introduced frozen checkpoint whose own prefix_id does not
    match the immediately preceding chain entry is refused, even inside an
    otherwise well-formed same-count tie."""
    at_8, declared, settled, unfrozen_final = _freeze_chain_fixture()
    wrong_checkpoint = declared.frozen[0].model_copy(update={"prefix_id": settled.prefix_id})
    tie_ancestor = settled.ancestors[1].model_copy(update={"frozen": (wrong_checkpoint,)})
    forged = settled.model_copy(update={"ancestors": (settled.ancestors[0], tie_ancestor)})
    with pytest.raises(CapabilityError) as raised:
        SequentialSnapshot.model_validate(forged.model_dump())
    assert raised.value.code == "sequential.source.invalid"


def test_a_frozen_cell_that_disappears_then_reappears_along_the_chain_is_refused():
    """Frozen sets must grow monotonically; a chain entry that drops a
    previously-declared freeze, followed by one that carries it again, is
    refused even though every individual digest is internally consistent."""
    at_8, declared, settled, unfrozen_final = _freeze_chain_fixture()
    dropped = settled.ancestors[1].model_copy(update={"frozen": ()})
    forged = settled.model_copy(update={"ancestors": (settled.ancestors[0], dropped)})
    with pytest.raises(CapabilityError) as raised:
        SequentialSnapshot.model_validate(forged.model_dump())
    assert raised.value.code == "sequential.source.invalid"


def test_a_same_look_reread_after_a_freeze_returns_the_frozen_previous_unchanged():
    """Re-reading the SAME records (no new finalized units) after a freeze
    must not be mistaken for a rewrite."""
    specs = [MetricSpec(name="outcome", type="mean")]
    plan = _plan(specs, start=4)
    records = _frame([1, 2] * 40, [5, 7] * 40)
    early = _analysis(records.iloc[:16].copy(), specs, plan)
    declared = early.capture_sequential(finalized=True, freeze=["outcome"])
    same_look = _analysis(records.iloc[:16].copy(), specs, plan)
    relinked = same_look.sequential_snapshot(previous=declared)
    assert relinked.frozen == declared.frozen
    assert relinked.prefix_id == declared.prefix_id


@pytest.mark.parametrize("parent_pairs", [8, 16])
def test_linking_a_frozen_capture_to_an_earlier_unfrozen_parent_keeps_the_freeze(parent_pairs):
    """Proving continuation from a never-frozen capture -- an earlier look, or
    the same look before the freeze -- must not drop a freeze declared since:
    the source keeps evaluating the frozen evidence."""
    specs = [MetricSpec(name="outcome", type="mean")]
    plan = _plan(specs, start=4)
    records = _frame([1, 2] * 40, [5, 7] * 40)
    parent = _analysis(records.iloc[: 2 * parent_pairs].copy(), specs, plan).sequential_snapshot()
    later = _analysis(records.iloc[:32].copy(), specs, plan)
    declared = later.capture_sequential(finalized=True, freeze=["outcome"])
    linked = later.sequential_snapshot(previous=parent)
    assert linked.frozen == declared.frozen
    assert later.sequential_snapshot().frozen == declared.frozen
    assert _row(later).require_asymptotic_sequential_result().checkpoint.status == "frozen"


def test_raw_records_and_analysis_freeze_every_cell_of_a_segmented_metric_alike():
    """``declare_sequential_freeze`` on raw finalized records and
    ``Analysis.capture_sequential(freeze=)`` on the same frame produce the same
    snapshot: every observed segment of the named metric is frozen, and the
    segment with no data yet stays monitored."""
    from increment import capture_sequential_snapshot, declare_sequential_freeze

    specs = [MetricSpec(name="outcome", type="mean")]
    cells = tuple(
        SequentialCell(
            metric="outcome",
            group_id="treatment",
            family=True,
            segment=(("segment", s),),
            alpha=F(1, 40),
        )
        for s in ("a", "b", "absent")
    )
    plan = _plan(specs, cells=cells)
    frame = _frame([1, 2] * 30, [8, 10] * 30)
    frame["segment"] = ["a" if i % 4 < 2 else "b" for i in range(len(frame))]
    declared = _analysis(frame, specs, plan).capture_sequential(finalized=True, freeze=["outcome"])
    registration = plan.inference.registration
    records = [
        {
            "unit_id": row.unit,
            "group_id": row.arm,
            "source_identity": {
                "unit_column": "unit",
                "group_column": "arm",
                "uptake_column": None,
            },
            "values": {"outcome": row.outcome},
            "segments": {"segment": row.segment},
        }
        for row in frame.itertuples()
    ]
    raw = capture_sequential_snapshot(
        registration,
        records,
        source_id=registration.source_id,
        definitions_id=registration.definitions_id,
        finalized=True,
    )
    assert declare_sequential_freeze(raw, ["outcome"]) == declared
    assert {c.cell.segment for c in declared.frozen} == {(("segment", "a"),), (("segment", "b"),)}


@pytest.mark.parametrize("metric", ["unregistered", "outcome"])
def test_a_freeze_with_nothing_left_to_freeze_is_refused(metric):
    from increment import declare_sequential_freeze

    reg = mean_registration(models=(mean_model("outcome", start_count=4),))
    declared = declare_sequential_freeze(
        mean_capture(reg, mean_records([1, 2] * 8, [5, 7] * 8)), ["outcome"]
    )
    with pytest.raises(CapabilityError) as raised:
        declare_sequential_freeze(declared, [metric])
    assert raised.value.code == "sequential.freeze.invalid"


def test_a_frozen_family_member_is_reselected_at_every_look():
    """Freezing fixes a member's evidence, not its family verdict. Member ``a``
    is frozen with an e-value between m/(2q) and m/q, selected beside ``b``;
    once ``b``'s evidence falls, e-BH over the same frozen evidence no longer
    selects ``a``. Remembering the earlier verdict would break the
    self-consistency e-BH's false discovery rate control rests on."""
    from increment import declare_sequential_freeze
    from increment.estimation._certified import log_interval
    from increment.estimation.sequential import AsymptoticMean
    from increment.estimation.sequential_runtime import selected_snapshot_results

    q = F(1, 10)
    cells = tuple(
        SequentialCell(metric=m, group_id="treatment", family=True, alpha=q / 2) for m in "ab"
    )
    reg = mean_registration(models=(mean_model("a"), mean_model("b")), cells=cells, q=q)
    policy = AsymptoticMean(registration=reg)

    def records(n, shift_a, shift_b, offset=0):
        rows = []
        for i in range(n):
            value = float(1 + i % 4)
            for arm, shifts in (("control", (0.0, 0.0)), ("treatment", (shift_a, shift_b))):
                values = {"a": value + shifts[0], "b": value + shifts[1]}
                rows.append(
                    {"unit_id": f"{offset + i:06d}-{arm}", "group_id": arm, "values": values}
                )
        return rows

    def family(snapshot):
        return {
            r.metric: r for r in selected_snapshot_results(snapshot, policy, nominal_alpha=q / 2)
        }

    first = mean_capture(reg, records(100, 0.5, 1.0))
    before = family(first)
    log_e = before["a"].require_asymptotic_sequential_result().log_e
    assert (-log_interval(q / 2)).hi > log_e >= (-log_interval(q)).hi  # m/q > e >= m/(2q)
    assert before["a"].discovery and before["b"].discovery
    frozen = declare_sequential_freeze(first, ["a"])
    after = family(
        mean_capture(reg, records(500, 0.0, 0.0, offset=100), previous=frozen, append=True)
    )
    assert after["a"].require_asymptotic_sequential_result().log_e == log_e
    assert after["a"].require_sequential_result().checkpoint.status == "frozen"
    assert not after["b"].discovery
    assert not after["a"].discovery


def test_a_registration_stored_under_bonferroni_asymptotic_families_cannot_continue():
    """A stored registration without the recorded e-BH commitment is the
    Bonferroni-era identity: its snapshot and compiled plan refuse as legacy
    instead of resuming under the weaker false-discovery guarantee. Registrations
    whose selection rule never changed keep their identity."""
    import json

    from increment.decision_wire import compiled_plan_from_dict, compiled_plan_to_dict
    from increment.plan import compile_decision_plan
    from increment.sequential_state import snapshot_from_json

    specs = [MetricSpec(name=name, type="mean") for name in ("outcome", "other")]
    cells = tuple(
        SequentialCell(metric=s.name, group_id="treatment", family=True, alpha=F(1, 40))
        for s in specs
    )
    frame = _frame([1, 2] * 8, [5, 7] * 8)
    frame["other"] = frame["outcome"] * 2
    declared = _plan(specs, cells=cells, secondary=True)
    analysis = _analysis(frame, specs, declared)
    snapshot = analysis.sequential_snapshot()
    assert snapshot.registration.asymptotic_family == "e_bh"
    stored = json.loads(snapshot.model_dump_json())
    del stored["registration"]["asymptotic_family"]
    with pytest.raises(CapabilityError) as raised:
        snapshot_from_json(json.dumps(stored))
    assert raised.value.code == "sequential.continuation.legacy"
    compiled = compile_decision_plan(
        declared,
        [synthesise_metric(s) for s in specs],
        path="frame",
        design=Randomized(control_group="control"),
    )
    plan = json.loads(json.dumps(compiled_plan_to_dict(compiled)))
    del plan["inference"]["registration"]["asymptotic_family"]
    with pytest.raises(CapabilityError) as raised:
        compiled_plan_from_dict(plan)
    assert raised.value.code == "sequential.continuation.legacy"
    assert _plan(specs[:1]).inference.registration.asymptotic_family is None


@pytest.mark.slow
def test_frozen_snapshot_survives_export_and_from_moments_replay(tmp_path):
    specs = [MetricSpec(name="outcome", type="mean")]
    plan = _plan(specs, start=4)
    records = _frame([1, 2] * 40, [5, 7] * 40)
    early = _analysis(records.iloc[:16].copy(), specs, plan)
    declared = early.capture_sequential(finalized=True, freeze=["outcome"])
    final = _analysis(records, specs, plan)
    final.sequential_snapshot(previous=declared)
    path = tmp_path / "checkpoint.parquet"
    final.export(path)
    import pyarrow.parquet as pq

    replay = Analysis.from_moments(
        pq.read_table(path).to_pylist(), metrics=specs, control="control"
    )
    replayed_snapshot = replay.sequential_snapshot()
    assert replayed_snapshot.frozen[0].cell.metric == "outcome"
    assert replayed_snapshot.frozen[0].prefix_id == declared.frozen[0].prefix_id
    frozen_row = _row(replay).require_asymptotic_sequential_result()
    assert frozen_row.checkpoint.status == "frozen"


def test_public_encouragement_itt_uses_the_asymptotic_mean_law():
    """Intent-to-treat under a randomized encouragement design is the
    same functional and observation stream as a randomized mean contrast, so the
    asymptotic scalar-mean law admits it -- the refusal keyed on design.mechanism,
    not on any property of the observations. An Encouragement design always
    derives its uptake column from design.uptake.fact (Analysis.from_unit_summary's
    default), so the frame and the registered definitions_id both carry it even
    though this registration monitors ITT alone, with no compliance cell."""
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    specs = [MetricSpec(name="outcome", type="mean")]
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="encouragement affects revenue only through clicks"
        ),
    )
    registration = mean_registration(
        definitions_id=sequential_definition_id(
            [synthesise_metric(s) for s in specs],
            design,
            transformations=specs,
            source_mapping=frame_observation_mapping(
                unit="unit", group="arm", exposure_date="exposure", uptake="clicked"
            ),
        ),
        models=(mean_model("outcome", start_count=4),),
        cells=(SequentialCell(metric="outcome", group_id="treatment"),),
    )
    plan = AnalysisPlan(
        primary="outcome",
        inference=InferenceSpec(kind="asymptotic_mean", registration=registration),
    )
    import pandas as pd

    frame = pd.DataFrame(
        [
            {"unit": r["unit_id"], "arm": r["group_id"], "clicked": 0, "exposure": i, **r["values"]}
            for i, r in enumerate(mean_records([1, 2, 3, 4] * 20, [6, 7, 8, 9] * 20))
        ]
    )
    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        metrics=specs,
        experiment_id="experiment",
        plan=plan,
        design=design,
        exposure_date="exposure",
    )
    row = _row(analysis, estimands=("itt",))
    assert row.stat_sig()
    result = row.require_asymptotic_sequential_result()
    assert result.checkpoint.model.law == "scalar_mean"


def test_scalar_mean_under_observational_design_still_refuses():
    """The mechanism relaxation must not widen past encouragement: observational
    designs are not randomized on any axis, and the refusal must still name it."""
    from increment.semantics.design import AdjustmentSet, IdentificationGate, Observational

    specs = [MetricSpec(name="outcome", type="mean")]
    design = Observational(
        control_group="control",
        adjustment=AdjustmentSet(covariates=("age",)),
        gate=IdentificationGate(),
    )
    registration = mean_registration(
        definitions_id=sequential_definition_id(
            [synthesise_metric(s) for s in specs],
            design,
            transformations=specs,
            source_mapping=frame_observation_mapping(unit="unit", group="arm"),
        ),
        models=(mean_model("outcome", start_count=4),),
        cells=(SequentialCell(metric="outcome", group_id="treatment"),),
    )
    plan = AnalysisPlan(
        primary="outcome",
        inference=InferenceSpec(kind="asymptotic_mean", registration=registration),
    )
    import pandas as pd

    frame = pd.DataFrame(
        [
            {"unit": r["unit_id"], "arm": r["group_id"], **r["values"]}
            for r in mean_records([1, 2, 3, 4] * 20, [6, 7, 8, 9] * 20)
        ]
    )
    with pytest.raises(CapabilityError) as raised:
        Analysis.from_unit_summary(
            frame,
            unit="unit",
            group="arm",
            metrics=specs,
            experiment_id="experiment",
            plan=plan,
            design=design,
        ).run()
    assert raised.value.code == "sequential.route.unsupported"


@pytest.mark.slow
def test_continuous_secondary_family_is_asymptotic_e_bh():
    specs = [MetricSpec(name=name, type="mean") for name in ("outcome", "other")]
    cells = tuple(
        SequentialCell(metric=s.name, group_id="treatment", family=True, alpha=F(1, 40))
        for s in specs
    )
    plan = _plan(specs, cells=cells, secondary=True)
    frame = _frame([1, 2] * 30, [8, 10] * 30)
    frame["other"] = 0
    analysis = _analysis(frame, specs, plan)
    rows = {row.metric: row for row in analysis.run()}
    assert rows["outcome"].discovery is True
    assert rows["other"].discovery is False
    assert rows["other"].require_asymptotic_sequential_result().bounds.reason == "zero_arm_variance"
    # e-BH, not per-cell Bonferroni: a discovery reinverts at the family's
    # realized threshold (q*1/2 == nominal alpha, so fcr_alpha == family_threshold),
    # an unselected cell keeps its registered allocation, and all rows share the
    # family guarantee, threshold and nominal alpha.
    assert rows["outcome"].require_asymptotic_sequential_result().decision_alpha == F(1, 20)
    assert rows["other"].require_asymptotic_sequential_result().decision_alpha == F(1, 40)
    for row in rows.values():
        assert row.family_threshold == pytest.approx(0.05)
        assert row.family_guarantee == "asymptotic_sequential"
        assert row.family_nominal_alpha == pytest.approx(0.05)


@pytest.mark.slow
def test_public_breakout_retains_absent_segments():
    specs = [MetricSpec(name="outcome", type="mean")]
    cells = tuple(
        SequentialCell(
            metric="outcome",
            group_id="treatment",
            family=True,
            segment=(("segment", s),),
            alpha=F(1, 40),
        )
        for s in ("present", "absent")
    )
    plan = _plan(specs, cells=cells)
    frame = _frame([1, 2] * 30, [8, 10] * 30)
    frame["segment"] = "present"
    analysis = _analysis(frame, specs, plan)
    rows = {row.dimension_value: row for row in analysis.run_breakout()}
    assert rows["present"].discovery is True
    assert rows["absent"].discovery is False
    assert isinstance(rows["absent"].sequential_result, AsymptoticSequentialResult)
    assert rows["absent"].sequential_result.bounds.reason == "missing_arm"
    assert isinstance(rows["present"].sequential_result, AsymptoticSequentialResult)
    assert rows["present"].sequential_result.decision_alpha == F(1, 40)


@pytest.mark.slow
def test_public_ratio_breakout_uses_predeclared_bonferroni():
    """A ratio metric's segmented family is a continuous breakout end to end:
    the registered Bonferroni view passes both the registration and the
    request validators, each segment is judged at its own allocation, and no
    e-BH threshold or reinversion is reported."""
    import pandas as pd

    specs = [MetricSpec(name="outcome", type="ratio", numerator="num", denominator="den")]
    cells = tuple(
        SequentialCell(
            metric="outcome",
            group_id="treatment",
            family=True,
            segment=(("segment", s),),
            alpha=F(1, 40),
        )
        for s in ("present", "absent")
    )
    plan = _plan(specs, cells=cells, law="ratio_mean")
    frame = pd.DataFrame(
        [
            {
                "unit": f"{i:04d}-{arm}",
                "arm": arm,
                "exposure": 2 * i + (arm == "treatment"),
                "num": (1 + i % 4) * (4 if arm == "treatment" else 1),
                "den": 2 + i % 2,
                "segment": "present",
            }
            for i in range(60)
            for arm in ("control", "treatment")
        ]
    )
    rows = {row.dimension_value: row for row in _analysis(frame, specs, plan).run_breakout()}
    assert rows["present"].discovery is True
    assert rows["absent"].discovery is False
    for row in rows.values():
        assert row.family_threshold is None
    present = rows["present"].sequential_result
    assert isinstance(present, AsymptoticSequentialResult)
    assert present.checkpoint.model.law == "ratio_mean"
    assert present.decision_alpha == F(1, 40)


@pytest.mark.slow
def test_segmented_registration_as_of_refuses_like_run():
    """A segmented roster's rows carry segment identity only through
    breakout(): the finalized as-of checkpoint refuses exactly as the
    whole-window run does instead of reporting unlabeled primary rows."""
    specs = [MetricSpec(name="outcome", type="mean", window_days=2)]
    cells = tuple(
        SequentialCell(
            metric="outcome", group_id="treatment", segment=(("segment", s),), alpha=F(1, 20)
        )
        for s in ("a", "b")
    )
    plan = _plan(specs, cells=cells, panel=True, uncorrected=True)
    frame = _frame([1, 2] * 20, [8, 10] * 20)
    start = date(2025, 1, 1)
    frame["day"] = start
    frame["exposed"] = start
    frame["segment"] = ["a" if i % 4 < 2 else "b" for i in range(len(frame))]
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
    contexts = []
    for readout in (analysis.run, analysis.run_asof_lift):
        with pytest.raises(CapabilityError) as raised:
            readout()
        assert raised.value.code == "sequential.route.unsupported"
        contexts.append(raised.value.context)
    assert contexts[0] == contexts[1]


@pytest.mark.slow
def test_panel_finalization_daily_projection_and_replay(tmp_path):
    import pyarrow.parquet as pq

    specs = [MetricSpec(name="outcome", type="mean", window_days=2)]
    plan = AnalysisPlan(primary="outcome", inference=InferenceSpec(kind="asymptotic_mean"))
    frame = _frame([1, 2] * 20, [8, 10] * 20)
    start = date(2025, 1, 1)
    frame["day"] = start
    frame["exposed"] = start
    analysis = Analysis.from_unit_panel(
        frame,
        unit="unit",
        group="arm",
        date="day",
        exposure_date="exposed",
        design=Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5}),
        metrics=specs,
        experiment_id="experiment",
        plan=plan,
        observation_end=start + timedelta(days=20),
    )
    early = analysis.capture_sequential(finalized=True, as_of=start)
    assert not early.records
    assert not _row(analysis).stat_sig()
    full = analysis.capture_sequential(finalized=True, as_of=start + timedelta(days=14))
    assert len(full.records) == 80
    assert full.parent_id == early.prefix_id
    daily = analysis.run_asof_lift(completed_windows_only=True)
    assert len(daily) == 1 and daily[0].n_control == 40
    assert daily[0].sequential_result == _row(analysis).sequential_result
    assert DailyLiftEstimate.model_validate_json(daily[0].model_dump_json()) == daily[0]
    path = tmp_path / "panel.parquet"
    analysis.export(path)
    replay = Analysis.from_moments(
        pq.read_table(path).to_pylist(), metrics=specs, control="control"
    )
    assert list(replay.run_asof_lift(completed_windows_only=True)) == list(daily)


@pytest.mark.slow
def test_secondary_family_daily_decisions_preserve_allocations_and_replay(tmp_path):
    import pyarrow.parquet as pq

    specs = [MetricSpec(name=name, type="mean", window_days=2) for name in ("outcome", "other")]
    cells = tuple(
        SequentialCell(metric=spec.name, group_id="treatment", family=True, alpha=F(1, 40))
        for spec in specs
    )
    plan = _plan(specs, cells=cells, panel=True, secondary=True)
    frame = _frame([1, 2] * 20, [8, 10] * 20)
    start = date(2025, 1, 1)
    frame["other"] = 0
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
    total_rows = analysis.run()
    assert isinstance(total_rows, LiftEstimates)
    totals = {row.metric: row for row in total_rows}
    daily = analysis.run_asof_lift(completed_windows_only=True)
    assert {row.metric: row.discovery for row in daily} == {"outcome": True, "other": False}
    expected_decision_alpha = {"outcome": F(1, 20), "other": F(1, 40)}
    for row in daily:
        assert row.sequential_result == totals[row.metric].sequential_result
        assert isinstance(row.sequential_result, AsymptoticSequentialResult)
        assert row.sequential_result.decision_alpha == expected_decision_alpha[row.metric]
        assert row.family_guarantee == "asymptotic_sequential"
        assert row.family_nominal_alpha == pytest.approx(0.05)
    widened = next(row for row in daily if row.metric == "outcome")
    with pytest.raises(CapabilityError) as raised:
        DailyLiftEstimate.model_validate({**widened.model_dump(), "discovery": False})
    assert raised.value.code == "sequential.source.invalid"
    path = tmp_path / "secondary-panel.parquet"
    analysis.export(path)
    replay = Analysis.from_moments(
        pq.read_table(path).to_pylist(), metrics=specs, control="control"
    )
    assert list(replay.run_asof_lift(completed_windows_only=True)) == list(daily)


@pytest.mark.slow
def test_explicitly_uncorrected_continuous_breakout_has_no_family_discovery():
    specs = [MetricSpec(name="outcome", type="mean")]
    cells = tuple(
        SequentialCell(
            metric="outcome",
            group_id="treatment",
            family=False,
            segment=(("segment", segment),),
            alpha=F(1, 20),
        )
        for segment in ("present", "absent")
    )
    plan = _plan(specs, cells=cells, uncorrected=True)
    frame = _frame([1, 2] * 30, [8, 10] * 30)
    frame["segment"] = "present"
    analysis = _analysis(frame, specs, plan)
    rows = {row.dimension_value: row for row in analysis.run_breakout()}
    assert rows["present"].require_lift().value == pytest.approx(5.0)
    assert rows["present"].discovery is None
    assert rows["absent"].discovery is None
    assert isinstance(rows["present"].sequential_result, AsymptoticSequentialResult)
    assert rows["present"].sequential_result.decision_alpha == F(1, 20)


@pytest.mark.parametrize("mutation", ["rho", "start_count", "alpha", "null_lift", "q"])
def test_identity_changes_cannot_continue_a_public_prefix(mutation):
    reg = mean_registration()
    records = mean_records([1, 2], [4, 5])
    first = mean_capture(reg, records)
    payload = reg.model_dump()
    if mutation in ("rho", "start_count"):
        payload["models"] = ({**payload["models"][0], mutation: 3},)
    elif mutation in ("alpha", "null_lift"):
        payload["roster"] = ({**payload["roster"][0], mutation: F(1, 40)},)
    else:
        payload["q"] = F(1, 5)
    changed = type(reg).model_validate(payload)
    with pytest.raises(CapabilityError) as raised:
        mean_capture(changed, records, previous=first)
    assert raised.value.code == "sequential.continuation.rewrite"


def test_unsupported_ratio_is_refused_before_frame_access():
    specs = [MetricSpec(name="outcome", type="ratio", numerator="num", denominator="den")]
    plan = _plan(specs)

    class UnreadFrame:
        def __getattribute__(self, name):
            raise AssertionError(f"unsupported request read data: {name}")

    with pytest.raises(CapabilityError) as raised:
        _analysis(UnreadFrame(), specs, plan)
    assert raised.value.code == "sequential.route.unsupported"


def test_power_does_not_relabel_the_old_log_se_boundary():
    from typing import Any, cast

    from increment.power.sequential import sequential_power

    with pytest.raises(CapabilityError) as raised:
        sequential_power(
            cast(Any, AsymptoticMean(registration=mean_registration())),
            delta=0.1,
            se_full=0.1,
            alpha=0.05,
            planned_looks=2,
        )
    assert raised.value.code == "sequential.route.unsupported"


@pytest.mark.slow
@pytest.mark.parametrize("correction", ["bonferroni", "none"])
def test_automatic_asymptotic_segments_keep_bonferroni_and_match_the_manual_form(correction):
    """Predeclared segment metadata on an asymptotic plan fixes the same
    registration a caller writes by hand: one pre-assignment segment cell per
    metric and level, judged at a fixed per-cell allocation (an equal share of
    q across metrics, never above a metric's compiled level, split across the
    levels) under the route's Bonferroni family, or uncorrected when declared
    so. e-BH never enters, and an absent level stays a monitored empty cell."""
    import math

    from increment.estimation.asymptotic_mean import mixture_r_star
    from increment.sequential_state import canonical_id

    specs = [MetricSpec(name="outcome", type="mean"), MetricSpec(name="other", type="mean")]
    levels = ("present", "absent")
    design = Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5})
    expected_n = 100
    automatic = AnalysisPlan(
        primary="outcome",
        secondaries=["other"],
        view_multiplicity=MultiplicitySpec(correction=correction),
        inference=InferenceSpec(
            kind="asymptotic_mean",
            expected_decision_sample_size=expected_n,
            segments={"segment": levels},
        ),
    )
    family = correction == "bonferroni"
    alpha = min(F(0.05), F(0.1) / 2) / 2 if family else F(0.05)
    binding = sequential_definition_id(
        [synthesise_metric(s) for s in specs],
        design,
        transformations=specs,
        source_mapping=frame_observation_mapping(
            unit="unit", group="arm", exposure_date="exposure"
        ),
    )
    rho = F(math.sqrt(float(mixture_r_star(alpha) / expected_n)))
    manual_registration = mean_registration(
        definitions_id=binding,
        q=F(0.1),
        reveal=JointReveal(
            filtration_id=canonical_id({"binding": binding, "reveal": "joint_units_v1"}),
            independent_unit_vectors=True,
            simultaneous_metrics=True,
            outcome_independent_order=True,
            immutable_finalized_outcomes=True,
            longest_window_days=0,
        ),
        models=tuple(mean_model(s.name, rho=rho) for s in specs),
        cells=tuple(
            SequentialCell(
                metric=s.name,
                group_id="treatment",
                segment=(("segment", level),),
                family=family,
                alpha=alpha,
            )
            for s in specs
            for level in levels
        ),
    )
    manual = automatic.model_copy(
        update={
            "inference": InferenceSpec(kind="asymptotic_mean", registration=manual_registration)
        }
    )
    frame = _frame([1, 2] * 30, [8, 10] * 30)
    frame["other"] = frame["outcome"]
    frame["segment"] = "present"
    sources = [_analysis(frame, specs, plan, design=design) for plan in (automatic, manual)]
    assert sources[0].sequential_snapshot() == sources[1].sequential_snapshot()
    registration = sources[0].sequential_snapshot().registration
    assert all(m.segment_membership == "pre_assignment" for m in registration.models)
    readouts = [
        {(row.metric, row.dimension_value): row for row in source.run_breakout()}
        for source in sources
    ]
    assert {k: v.model_dump() for k, v in readouts[0].items()} == {
        k: v.model_dump() for k, v in readouts[1].items()
    }
    rows = readouts[0]
    assert set(rows) == {(s.name, level) for s in specs for level in levels}
    for (_, level), row in rows.items():
        result = row.sequential_result
        assert isinstance(result, AsymptoticSequentialResult)
        assert result.decision_alpha == alpha
        assert row.family_threshold is None
        if level == "absent":
            assert result.bounds.reason == "missing_arm"
            assert row.discovery is (False if family else None)
        else:
            assert row.discovery is (True if family else None)
    with pytest.raises(CapabilityError) as raised:
        sources[0].run()
    # The whole-window view keeps its segmented refusal: the secondary's compiled
    # family membership differs from its segment cells (a primary alone refuses the
    # segmented roster outright); either way the manual form refuses identically.
    assert raised.value.code in {"sequential.source.invalid", "sequential.route.unsupported"}
    with pytest.raises(CapabilityError) as same:
        sources[1].run()
    assert same.value.code == raised.value.code


def test_asymptotic_segments_refuse_bh_before_any_frame_is_read():
    specs = [MetricSpec(name="outcome", type="mean")]
    plan = AnalysisPlan(
        primary="outcome",
        view_multiplicity=MultiplicitySpec(correction="bh"),
        inference=InferenceSpec(kind="asymptotic_mean", segments={"segment": ("a", "b")}),
    )

    class UnreadFrame:
        def __getattribute__(self, name):
            raise AssertionError(f"unsupported request read data: {name}")

    with pytest.raises(CapabilityError) as raised:
        _analysis(
            UnreadFrame(),
            specs,
            plan,
            design=Randomized(control_group="control", allocation={"control": 1, "treatment": 1}),
        )
    assert raised.value.code == "sequential.route.unsupported"


@pytest.mark.parametrize("segments", [{}, {"segment": ("a", "b")}])
def test_automatic_encouragement_keeps_its_corrected_breakout_refusal(segments):
    from increment.errors import CodedError
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    design = Encouragement(
        control_group="control",
        allocation={"control": 1, "treatment": 1},
        uptake=UptakeSpec(fact="clicked"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="The encouragement only affects uptake."
        ),
    )
    plan = AnalysisPlan(
        primary="outcome",
        view_multiplicity=MultiplicitySpec(correction="bh"),
        inference=InferenceSpec(kind="asymptotic_mean", segments=segments),
    )
    with pytest.raises(CodedError) as raised:
        _analysis(None, [MetricSpec(name="outcome", type="mean")], plan, design=design)
    assert raised.value.code == "plan.corrected_encouragement_breakout"


_TYPED_KINDS = ("pandas", "polars", "arrow")


def _canonical_segment_frame(levels):
    frame = _frame([1, 2, 4] * 30, [8, 10, 9] * 30)
    frame["segment"] = [levels[(i // 3) % len(levels)] for i in range(len(frame))]
    return frame


@pytest.mark.parametrize("kind", _TYPED_KINDS)
@pytest.mark.parametrize("registered", ["automatic", "explicit"])
def test_boolean_and_null_segments_match_the_canonical_string_oracle(kind, registered):
    from tests.test_sequential_registration_contract import typed_segment_frame

    specs = [MetricSpec(name="outcome", type="mean")]
    levels = ("true", "false", "__null__", "absent")
    design = None
    if registered == "automatic":
        design = Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5})
        plan = AnalysisPlan(
            primary="outcome",
            inference=InferenceSpec(
                kind="asymptotic_mean",
                expected_decision_sample_size=100,
                segments={"segment": levels},
            ),
        )
    else:
        cells = tuple(
            SequentialCell(
                metric="outcome",
                group_id="treatment",
                family=True,
                segment=(("segment", level),),
                alpha=F(1, 80),
            )
            for level in levels
        )
        plan = _plan(specs, cells=cells)
    canonical = _canonical_segment_frame(levels[:3])
    oracle = _analysis(canonical, specs, plan, design=design)
    typed = _analysis(typed_segment_frame(canonical, kind), specs, plan, design=design)

    snapshot = typed.sequential_snapshot()
    assert snapshot == oracle.sequential_snapshot()
    assert {(s.segment[0][1], s.n > 0) for s in snapshot.states} == {
        (level, level != "absent") for level in levels
    }
    rows = {row.dimension_value: row for row in typed.run_breakout()}
    expected = {row.dimension_value: row for row in oracle.run_breakout()}
    assert {k: v.model_dump() for k, v in rows.items()} == {
        k: v.model_dump() for k, v in expected.items()
    }
    assert set(rows) == set(levels)
    for level, row in rows.items():
        assert isinstance(row.sequential_result, AsymptoticSequentialResult)
        assert row.sequential_result.bounds.available is (level != "absent")
        assert (row.sequential_result.bounds.reason == "missing_arm") is (level == "absent")


@pytest.mark.parametrize("kind", _TYPED_KINDS)
def test_boolean_covariate_also_declared_as_a_segment_keeps_the_adjusted_state(kind):
    """A pre-assignment Boolean covariate serving as its own segment feeds the adjustment
    as its numeric 0/1 value and labels cells as ``true``/``false``, exactly like a numeric
    covariate beside an independently prepared string segment."""
    import numpy as np
    import pandas as pd

    from increment import Method

    rng = np.random.default_rng(3)
    n = 240
    arms = ["control", "treatment"] * (n // 2)
    covariate = (rng.random(n) < 0.4).astype(int)
    outcome = 2.0 + 1.5 * covariate + rng.normal(0.0, 1.0, n) + [a == "treatment" for a in arms]
    frame = pd.DataFrame(
        {
            "unit": [f"u{i:04d}" for i in range(n)],
            "arm": arms,
            "exposure": range(n),
            "outcome": outcome,
            "covariate": covariate,
            "seg": ["true" if c else "false" for c in covariate],
        }
    )
    levels = ("true", "false", "absent")
    design = Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5})
    spec = MetricSpec(
        name="outcome",
        type="mean",
        covariate="covariate",
        decision_method=Method(name="cuped", variance_reduction="cuped"),
    )

    def plan(dimension):
        return AnalysisPlan(
            primary="outcome",
            inference=InferenceSpec(
                kind="asymptotic_mean",
                expected_decision_sample_size=100,
                segments={dimension: levels},
            ),
        )

    oracle = _analysis(frame, [spec], plan("seg"), design=design)
    bools = frame["covariate"].astype(bool)
    base = frame.drop(columns="covariate")
    if kind == "pandas":
        typed_frame = base.assign(covariate=bools.astype("boolean"))
    elif kind == "polars":
        import polars as pl

        typed_frame = pl.from_pandas(base).with_columns(
            pl.Series("covariate", bools, dtype=pl.Boolean)
        )
    else:
        import pyarrow as pa

        typed_frame = pa.Table.from_pandas(base, preserve_index=False).append_column(
            "covariate", pa.array(bools, pa.bool_())
        )
    typed = _analysis(typed_frame, [spec], plan("covariate"), design=design)

    def retained(analysis):
        return sorted(
            (s.group_id, s.segment[0][1], s.n, s.mean, s.scatter)
            for s in analysis.sequential_snapshot().states
        )

    assert retained(typed) == retained(oracle)
    assert {row[1] for row in retained(typed) if row[2] > 0} == {"true", "false"}
    rows = {row.dimension_value: row for row in typed.run_breakout()}
    expected = {row.dimension_value: row for row in oracle.run_breakout()}
    for level in levels:
        got, want = rows[level].sequential_result, expected[level].sequential_result
        assert (got.bounds, got.decision_alpha, got.point_reason) == (
            want.bounds,
            want.decision_alpha,
            want.point_reason,
        )
        assert rows[level].lift == expected[level].lift
