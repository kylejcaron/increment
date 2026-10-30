import pandas as pd
import pytest

from increment import Analysis, AnalysisPlan, RunResult, run
from increment.decision_wire import compiled_plan_from_dict
from increment.frame import MetricSpec, synthesise_metric
from increment.kernel.contracts import Registration
from increment.plan import compile_decision_plan


def test_top_level_run_surface_is_public():
    assert callable(run)
    assert RunResult.__name__ == "RunResult"


def test_top_level_run_returns_a_validated_in_memory_result():
    from dataclasses import replace

    import pytest

    from increment.errors import CodedError

    plan = AnalysisPlan(primary="value")
    metric = synthesise_metric(MetricSpec(name="value", type="mean"))
    compiled = compile_decision_plan(plan, [metric], path="frame")
    registration = Registration.from_plan(compiled)
    analysis = Analysis.from_unit_summary(
        pd.DataFrame(
            {
                "unit": ["c1", "c2", "c3", "c4", "t1", "t2", "t3", "t4"],
                "arm": ["control"] * 4 + ["treatment"] * 4,
                "value": [1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0],
            }
        ),
        unit="unit",
        group="arm",
        control="control",
        metrics={"value": "mean"},
        plan=plan,
        experiment_id="exp",
    )
    try:
        result = run(
            analysis,
            registration,
            compiled,
            experiment_id="exp",
            analysis_spec_hash="sha256:" + "a" * 64,
            execution_identity_hash="sha256:" + "b" * 64,
            run_id="run-1",
        )
    finally:
        analysis.close()

    assert isinstance(result, RunResult)
    assert result.cells == result.run.cells
    assert len(result.effects) == len(result.cells) == 1
    assert result.decisions.run_id == "run-1"
    assert result.run.decision_plan is not None
    assert registration.matches(compiled_plan_from_dict(result.run.decision_plan.plan_payload))
    assert all(
        cell.identity.scope is not None
        and cell.identity.scope.population == cell.identity.key.analysis_population
        for cell in result.cells
    )
    assert result.decisions.cells[0].membership == "member"
    assert result.decisions.cells[0].family_id is not None

    with pytest.raises(CodedError) as mismatch:
        replace(result.run, registration_hash="sha256:" + "0" * 64)
    assert mismatch.value.code == "store.run.identity_mismatch"
    assert result.decisions.cells[0].status == "reject"
    assert result.decisions.families[0].status == "reject"


def _guardrail_case_inputs():
    from increment import ExperimentMetric

    plan = AnalysisPlan(
        primary="value",
        guardrails=[ExperimentMetric(metric="guard", margin_abs=0.5)],
    )
    specs = [
        MetricSpec(name="value", type="mean"),
        MetricSpec(name="guard", type="mean", preferred_direction="decrease"),
    ]
    metrics = [synthesise_metric(spec) for spec in specs]
    compiled = compile_decision_plan(plan, metrics, path="frame")
    registration = Registration.from_plan(compiled)
    return plan, specs, compiled, registration


def _guardrail_case_frame():
    return pd.DataFrame(
        {
            "unit": [f"u{i}" for i in range(8)],
            "arm": ["control"] * 4 + ["treatment"] * 4,
            "value": [1.0, 1.01, 0.99, 1.0, 2.0, 2.01, 1.99, 2.0],
            "guard": [10.0, 10.01, 9.99, 10.0, 10.0, 10.01, 9.99, 10.0],
        }
    )


def _guardrail_case_analysis(plan, specs):
    from increment import Randomized

    return Analysis.from_unit_summary(
        _guardrail_case_frame(),
        unit="unit",
        group="arm",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics=specs,
        plan=plan,
        experiment_id="exp",
    )


def _guardrail_case_result():
    plan, specs, compiled, registration = _guardrail_case_inputs()
    analysis = _guardrail_case_analysis(plan, specs)
    try:
        result = run(
            analysis,
            registration,
            compiled,
            experiment_id="exp",
            analysis_spec_hash="sha256:" + "c" * 64,
            execution_identity_hash="sha256:" + "d" * 64,
            run_id="run-guardrail",
        )
    finally:
        analysis.close()
    return result, registration, compiled


def _unavailable_sampling(cells, metric):
    from dataclasses import replace

    from increment.result_envelope import ComponentStatus

    return tuple(
        replace(
            cell,
            components={
                **cell.components,
                "sampling": (
                    ComponentStatus(
                        "unavailable",
                        "decision.test.unavailable",
                        "test evidence intentionally unavailable",
                    ),
                    None,
                ),
            },
        )
        if cell.identity.key.metric == metric
        else cell
        for cell in cells
    )


def _nonreject_sampling(cells, metric):
    from dataclasses import replace

    from increment.result_envelope import RetainedState

    changed = []
    for cell in cells:
        if cell.identity.key.metric != metric:
            changed.append(cell)
            continue
        status, state = cell.components["sampling"]
        row = cell.retained_row()
        assert row.lift is not None
        assert state is not None
        interval = row.lift.model_copy(
            update={"lb": row.null_lift - 100.0, "ub": row.null_lift + 100.0}
        )
        modified_row = row.model_copy(
            update={
                "lift": interval,
                "abs_alpha": row.abs_alpha if row.null_abs is not None else None,
                "abs_lb": None if row.null_abs is None else row.null_abs - 100.0,
                "abs_ub": None if row.null_abs is None else row.null_abs + 100.0,
            }
        )
        changed.append(
            replace(
                cell,
                components={
                    **cell.components,
                    "sampling": (
                        status,
                        RetainedState(
                            state.construction,
                            state.reason,
                            state.kind,
                            modified_row.model_dump(mode="json"),
                        ),
                    ),
                },
            )
        )
    return tuple(changed)


def _integrity_metadata(metadata, integrity):
    scope = metadata.scope
    source_id, source = next(iter(scope.by_source.items()))
    checks = source.integrity
    if integrity == "unknown":
        checks = ()
    else:
        from increment.declarations import IntegrityStatus

        enum_status = IntegrityStatus(integrity)
        checks = tuple(item.model_copy(update={"status": enum_status}) for item in checks)
    updated_source = source.model_copy(update={"integrity": checks})
    return metadata.model_copy(
        update={"scope": scope.model_copy(update={"by_source": {source_id: updated_source}})}
    )


def _evaluate_guardrail_program(
    result, registration, compiled, *, unavailable=None, nonreject=None, integrity=None
):
    from dataclasses import replace

    from increment.decide import decide
    from increment.store.types import pinned_run

    cells = result.run.cells
    if unavailable is not None:
        cells = _unavailable_sampling(cells, unavailable)
    if nonreject is not None:
        cells = _nonreject_sampling(cells, nonreject)
    metadata = result.run.metadata
    assert metadata is not None
    if integrity is not None:
        metadata = _integrity_metadata(metadata, integrity)
    altered = replace(result.run, cells=cells, metadata=metadata)
    return decide(pinned_run(altered), registration, compiled).program


def test_public_run_requires_primary_and_safe_guardrail_to_pass():
    result, _, _ = _guardrail_case_result()
    assert {cell.role: cell.status for cell in result.decisions.cells} == {
        "primary": "reject",
        "guardrail": "reject",
    }
    assert result.decisions.program.status == "pass"


def test_unavailable_primary_or_guardrail_is_not_evaluated():
    result, registration, compiled = _guardrail_case_result()
    for metric, reason in (
        ("value", "decision.primary.not_evaluated"),
        ("guard", "decision.guardrail.not_evaluated"),
    ):
        program = _evaluate_guardrail_program(result, registration, compiled, unavailable=metric)
        assert program.status == "not_evaluated"
        assert reason in program.reason_codes


def test_primary_nonrejection_or_unsafe_guardrail_blocks_program():
    result, registration, compiled = _guardrail_case_result()
    for metric, reason in (
        ("value", "decision.primary.not_rejected"),
        ("guard", "decision.guardrail.unsafe"),
    ):
        program = _evaluate_guardrail_program(result, registration, compiled, nonreject=metric)
        assert program.status == "block"
        assert reason in program.reason_codes


def test_failed_or_unknown_integrity_blocks_program():
    result, registration, compiled = _guardrail_case_result()
    for integrity in ("failed", "not_checked_missing_counts", "unknown"):
        program = _evaluate_guardrail_program(result, registration, compiled, integrity=integrity)
        assert program.status == "block"
        assert f"integrity.{integrity}" in program.reason_codes


def test_public_run_returns_each_disjoint_family_from_its_complete_metadata_roster():
    plan = AnalysisPlan(primary="value", secondaries=["first", "second"], q=0.1)
    metrics = [
        synthesise_metric(MetricSpec(name=name, type="mean"))
        for name in ("value", "first", "second")
    ]
    compiled = compile_decision_plan(plan, metrics, path="frame")
    registration = Registration.from_plan(compiled)
    frame = pd.DataFrame(
        {
            "unit": [f"u{i}" for i in range(8)],
            "arm": ["control"] * 4 + ["treatment"] * 4,
            "value": [1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0, 11.0],
            "first": [1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0, 11.0],
            "second": [1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0, 11.0],
        }
    )
    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        control="control",
        metrics={"value": "mean", "first": "mean", "second": "mean"},
        plan=plan,
        experiment_id="family-roster",
    )
    try:
        result = run(
            analysis,
            registration,
            compiled,
            experiment_id="family-roster",
            analysis_spec_hash="sha256:" + "a" * 64,
            execution_identity_hash="sha256:" + "b" * 64,
            run_id="run-family-roster",
        )
    finally:
        analysis.close()

    metadata = result.run.metadata
    assert metadata is not None
    scope = metadata.scope
    declared = {(family.family_id, family.source_snapshot_id): family for family in scope.families}
    verdicts = {
        (family.family_id, family.source_snapshot_id): family
        for family in result.decisions.families
    }
    from dataclasses import replace

    from increment.decide import decide
    from increment.result_envelope import ComponentStatus
    from increment.store.types import pinned_run

    reversed_metadata = metadata.model_copy(
        update={"scope": scope.model_copy(update={"families": tuple(reversed(scope.families))})}
    )
    reordered_run = replace(result.run, metadata=reversed_metadata)
    reordered_decisions = decide(pinned_run(reordered_run), registration, compiled)
    assert [
        (family.family_id, family.source_snapshot_id) for family in reordered_decisions.families
    ] == sorted(verdicts)
    assert verdicts.keys() == declared.keys()
    assert len(verdicts) == 2
    family_members = [
        {member.identity_hash for member in family.members} for family in verdicts.values()
    ]
    assert family_members[0].isdisjoint(family_members[1])
    for identity in result.run.cells:
        expected = next(
            (
                family.family_id
                for family in scope.families
                if identity.identity.source_snapshot_id == family.source_snapshot_id
                and identity.identity.key in family.members
            ),
            None,
        )
        verdict = next(
            item
            for item in result.decisions.cells
            if item.identity_hash == identity.identity.hash()
        )
        assert (verdict.family_id, verdict.membership) == (
            (expected, "member") if expected is not None else (None, "nonmember")
        )

    hidden_cell = next(cell for cell in result.run.cells if cell.identity.key.metric == "second")
    hidden_cells = tuple(
        replace(
            cell,
            components={
                **cell.components,
                "sampling": (
                    ComponentStatus(
                        "unavailable",
                        "decision.test.hidden",
                        "registered evidence unavailable",
                    ),
                    None,
                ),
            },
        )
        if cell.identity.hash() == hidden_cell.identity.hash()
        else cell
        for cell in result.run.cells
    )
    hidden_run = replace(result.run, cells=hidden_cells)
    hidden_decisions = decide(pinned_run(hidden_run), registration, compiled)
    hidden_verdict = next(
        item for item in hidden_decisions.cells if item.identity_hash == hidden_cell.identity.hash()
    )
    assert hidden_verdict.membership == "member"
    assert hidden_verdict.status == "not_evaluated"
    hidden_family = next(
        family
        for family in hidden_decisions.families
        if hidden_verdict.family_id == family.family_id
    )
    declared_hidden_family = next(
        family
        for family in scope.families
        if family.family_id == hidden_family.family_id
        and family.source_snapshot_id == hidden_family.source_snapshot_id
    )
    assert len(hidden_family.members) == len(declared_hidden_family.members)


def test_scoped_family_permutations_keep_source_population_and_segment_identity():
    from dataclasses import replace

    from increment.decide import decide
    from increment.estimation.readout_types import FamilyScope, family_identity
    from increment.store.types import pinned_run

    result, registration, compiled = _guardrail_case_result()
    metadata = result.run.metadata
    assert metadata is not None
    template = metadata.scope.families[0]
    variants = (
        ("source-z", "assigned", "country"),
        ("source-a", "assigned", "country"),
        ("source-a", "triggered", "country"),
        ("source-a", "triggered", "device"),
    )
    families = []
    for source, population, dimension in variants:
        identity = family_identity(
            template.source_snapshot_id,
            population,
            template.view,
            dimension,
            source,
            f"{dimension}-{population}",
        )
        families.append(
            FamilyScope.model_validate(
                {
                    **template.model_dump(mode="python"),
                    "family_id": identity,
                    "analysis_population": population,
                    "dimension": dimension,
                    "source": source,
                    "name": f"{dimension}-{population}",
                }
            )
        )
    permuted = metadata.model_copy(
        update={"scope": metadata.scope.model_copy(update={"families": tuple(reversed(families))})}
    )
    decisions = decide(pinned_run(replace(result.run, metadata=permuted)), registration, compiled)

    expected = sorted(
        (family.source_snapshot_id, str(family.analysis_population), family.family_id)
        for family in families
    )
    actual = [
        (family.source_snapshot_id, family.analysis_population, family.family_id)
        for family in decisions.families
    ]
    assert actual == expected
    assert {item[2] for item in actual} == {family.family_id for family in families}


def test_selected_quantile_public_run_matches_legacy_estimate_and_verdict():
    from increment import MetricSpec
    from increment.estimation.results import LiftEstimate

    data = pd.DataFrame(
        [
            {"unit": f"{group}-{index}", "arm": group, "value": float(index + offset)}
            for group, offset in (("control", 0), ("treatment", 10))
            for index in range(30)
        ]
    )
    plan = AnalysisPlan(primary="median", q=0.1)
    metric = MetricSpec(name="median", value_column="value", type="quantile", quantile=0.5)
    compiled = compile_decision_plan(plan, [synthesise_metric(metric)], path="frame")
    registration = Registration.from_plan(compiled)
    analysis = Analysis.from_unit_summary(
        data,
        unit="unit",
        group="arm",
        control="control",
        metrics=[metric],
        plan=plan,
        experiment_id="selected-quantile",
    )
    try:
        legacy = analysis.run()
        result = run(
            analysis,
            registration,
            compiled,
            experiment_id="selected-quantile",
            analysis_spec_hash="sha256:" + "a" * 64,
            execution_identity_hash="sha256:" + "b" * 64,
            run_id="selected-quantile-run",
        )
    finally:
        analysis.close()

    legacy_row = legacy[0]
    retained = result.cells[0].retained_row()
    assert isinstance(legacy_row, LiftEstimate)
    assert retained.method == legacy_row.method
    assert retained.alternative == legacy_row.alternative
    assert retained.lift is not None and legacy_row.lift is not None
    assert retained.lift.value == pytest.approx(legacy_row.lift.value)
    assert retained.lift.lb == pytest.approx(legacy_row.lift.lb)
    assert retained.lift.ub == pytest.approx(legacy_row.lift.ub)
    assert result.effects[0] == result.cells[0].summary
    assert result.decisions.cells[0].status == ("reject" if legacy_row.stat_sig() else "retain")
    assert result.decisions.cells[0].discovery is legacy_row.discovery


def test_observational_public_run_matches_legacy_estimate_and_decision():
    from increment import (
        AdjustmentSet,
        IdentificationGate,
        MetricSpec,
        Observational,
    )
    from increment.estimation.results import LiftEstimate

    data = pd.DataFrame(
        [
            {
                "unit": f"u{index}",
                "arm": "treatment" if index % 2 else "control",
                "x": float(index % 9),
                "revenue": float(index % 9) + (1.5 if index % 2 else 0.0),
            }
            for index in range(90)
        ]
    )
    plan = AnalysisPlan(primary="revenue")
    metric = MetricSpec(name="revenue", type="mean")
    analysis = Analysis.from_unit_summary(
        data,
        unit="unit",
        group="arm",
        metrics=[metric],
        design=Observational(
            control_group="control",
            adjustment=AdjustmentSet(covariates=("x",)),
            gate=IdentificationGate(overlap="trim"),
        ),
        plan=plan,
        experiment_id="observational-run",
    )
    compiled = analysis.compiled_decision_plan
    registration = Registration.from_plan(compiled)
    try:
        legacy = analysis.run()[0]
        result = run(
            analysis,
            registration,
            compiled,
            experiment_id="observational-run",
            analysis_spec_hash="sha256:" + "e" * 64,
            execution_identity_hash="sha256:" + "f" * 64,
            run_id="observational-run",
        )
    finally:
        analysis.close()

    retained = result.cells[0].retained_row()
    assert isinstance(legacy, LiftEstimate)
    assert result.cells[0].identity.key.metric == legacy.metric
    assert result.cells[0].identity.key.method == legacy.method
    assert retained.lift is not None and legacy.lift is not None
    assert retained.lift.value == pytest.approx(legacy.lift.value)
    assert retained.stat_sig() is legacy.stat_sig()
    assert result.decisions.cells[0].status == ("reject" if legacy.stat_sig() else "retain")


def test_sequential_ebh_public_run_matches_legacy_row_verdicts():
    from increment import InferenceSpec, MetricSpec, Randomized
    from increment.estimation.results import LiftEstimate

    metrics = [MetricSpec(name=name, type="conversion") for name in ("checkout", "signup")]
    data = pd.DataFrame(
        [
            {
                "unit": f"{arm}-{index}",
                "arm": arm,
                "exposure": index,
                "checkout": float(index % 4 == 0 or arm == "treatment"),
                "signup": float(index % 3 == 0 or arm == "treatment" and index % 2 == 0),
            }
            for arm in ("control", "treatment")
            for index in range(60)
        ]
    )
    plan = AnalysisPlan(
        secondaries=("checkout", "signup"),
        q=0.1,
        inference=InferenceSpec(kind="always_valid"),
    )
    analysis = Analysis.from_unit_summary(
        data,
        unit="unit",
        group="arm",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics=metrics,
        plan=plan,
        exposure_date="exposure",
        experiment_id="sequential-ebh",
    )
    compiled = analysis.compiled_decision_plan
    registration = Registration.from_plan(compiled)
    try:
        legacy = analysis.run()
        result = run(
            analysis,
            registration,
            compiled,
            experiment_id="sequential-ebh",
            analysis_spec_hash="sha256:" + "1" * 64,
            execution_identity_hash="sha256:" + "2" * 64,
            run_id="sequential-ebh-run",
        )
    finally:
        analysis.close()

    assert {family.procedure for family in result.decisions.families} == {"e_bh"}
    for row in legacy:
        assert isinstance(row, LiftEstimate)
        verdict = next(
            item
            for item in result.decisions.cells
            if item.identity_hash
            == next(
                cell.identity.hash()
                for cell in result.cells
                if cell.identity.key.metric == row.metric
                and cell.identity.key.group_id == row.group_id
            )
        )
        assert verdict.status == ("reject" if row.stat_sig() else "retain")
        assert verdict.discovery is row.discovery


@pytest.mark.integration
def test_trigger_declared_public_run_matches_legacy_population_results(tmp_path):
    import datetime as dt

    import ibis
    import pyarrow as pa

    from increment import SourceSnapshotEvidence
    from increment.estimation.results import LiftEstimate

    definitions = tmp_path / "trigger.yml"
    definitions.write_text(
        """
dialect: duckdb
fact_sources:
  - name: events
    sql: SELECT * FROM events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: revenue
        column: value
      - name: enrolled
        column: null
      - name: saw_surface
        column: null
exposures:
  - name: assignment
    fact: enrolled
  - name: saw_surface
    fact: saw_surface
metrics:
  - name: revenue
    type: mean
    entity: user_id
    fact: revenue
    aggregation: sum
    window_days: 7
experiments:
  - name: exp
    exposure: assignment
    trigger: saw_surface
    unit: user_id
    start: 2024-01-01T00:00:00
    control_group: C
    plan:
      primary: revenue
"""
    )
    rows: list[dict[str, object]] = []
    enrolled_at = dt.datetime(2024, 1, 2)
    observed_at = dt.datetime(2024, 1, 8)
    for arm in ("C", "T"):
        for index in range(100):
            user = f"{arm}{index}"
            rows.append(
                {
                    "user_id": user,
                    "group_id": arm,
                    "ts": enrolled_at,
                    "event": "enrolled",
                    "value": 0.0,
                    "experiment_id": "exp",
                }
            )
            if index < 50:
                rows.append(
                    {
                        "user_id": user,
                        "group_id": arm,
                        "ts": enrolled_at,
                        "event": "saw_surface",
                        "value": 0.0,
                        "experiment_id": "exp",
                    }
                )
            rows.append(
                {
                    "user_id": user,
                    "group_id": arm,
                    "ts": observed_at,
                    "event": "revenue",
                    "value": float(index % 4 + (1 if arm == "T" and index < 50 else 0)),
                    "experiment_id": "exp",
                }
            )
    con = ibis.duckdb.connect()
    con.create_table("events", obj=pa.Table.from_pylist(rows))
    snapshot_at = dt.datetime(2024, 1, 31, tzinfo=dt.UTC)
    analysis = Analysis.from_definitions(
        "exp",
        definitions,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(snapshot_at, {"events": snapshot_at}),
    )
    compiled = analysis.compiled_decision_plan
    registration = Registration.from_plan(compiled)
    try:
        legacy = analysis.run()
        result = run(
            analysis,
            registration,
            compiled,
            experiment_id="exp",
            analysis_spec_hash="sha256:" + "7" * 64,
            execution_identity_hash="sha256:" + "8" * 64,
            run_id="triggered-run",
        )
    finally:
        analysis.close()
        con.disconnect()

    assert {row.analysis_population for row in legacy} == {"assigned", "triggered"}
    assert {cell.identity.key.analysis_population for cell in result.cells} == {
        "assigned",
        "triggered",
    }
    for row in legacy:
        assert isinstance(row, LiftEstimate)
        cell = next(
            candidate
            for candidate in result.cells
            if candidate.identity.key.metric == row.metric
            and candidate.identity.key.analysis_population == row.analysis_population
        )
        cell_identity_hash = cell.identity.hash()
        verdict = next(
            candidate
            for candidate in result.decisions.cells
            if candidate.identity_hash == cell_identity_hash
        )
        family = next(
            candidate
            for candidate in result.decisions.families
            if any(member.identity_hash == cell_identity_hash for member in candidate.members)
        )
        assert verdict.status == ("reject" if row.stat_sig() else "retain")
        assert verdict.family_id == family.family_id
        assert family.analysis_population == row.analysis_population


def test_breakout_declared_analysis_run_matches_legacy_whole_window_results():
    from increment.estimation.results import LiftEstimate

    data = pd.DataFrame(
        [
            {"unit": f"{segment}-{arm}-{index}", "arm": arm, "segment": segment, "value": value}
            for segment, values in (
                ("north", {"control": (1.0, 2.0, 3.0), "treatment": (2.0, 3.0, 4.0)}),
                ("south", {"control": (4.0, 5.0, 6.0), "treatment": (5.0, 6.0, 7.0)}),
            )
            for arm, observations in values.items()
            for index, value in enumerate(observations)
        ]
    )
    plan = AnalysisPlan(primary="value")
    analysis = Analysis.from_unit_summary(
        data,
        unit="unit",
        group="arm",
        control="control",
        metrics={"value": "mean"},
        breakouts=("segment",),
        plan=plan,
        experiment_id="breakout-run",
    )
    compiled = analysis.compiled_decision_plan
    registration = Registration.from_plan(compiled)
    try:
        legacy = analysis.run()
        breakout_rows = analysis.run_breakout()
        result = run(
            analysis,
            registration,
            compiled,
            experiment_id="breakout-run",
            analysis_spec_hash="sha256:" + "3" * 64,
            execution_identity_hash="sha256:" + "4" * 64,
            run_id="breakout-run",
        )
    finally:
        analysis.close()

    assert len(breakout_rows) == 2
    assert {row.dimension_value for row in breakout_rows} == {"north", "south"}
    assert len(result.cells) == len(legacy) == 1
    legacy_row = legacy[0]
    assert isinstance(legacy_row, LiftEstimate)
    assert result.cells[0].identity.key.analysis_population == legacy_row.analysis_population
    assert result.cells[0].identity.key.metric == legacy_row.metric
    assert result.cells[0].identity.key.dimension is None
    assert result.cells[0].identity.key.dimension_value is None
    assert result.cells[0].retained_row().stat_sig() is legacy_row.stat_sig()


def test_switchback_public_run_refuses_before_read_with_legacy_route():
    from increment import Randomized
    from increment.errors import CodedError
    from increment.semantics.assignment import (
        IndependentBernoulliOrder,
        SwitchbackAssignment,
        SwitchbackWindow,
    )

    frame = pd.DataFrame(
        [
            {
                "unit": unit,
                "cycle": cycle,
                "period": period,
                "step": step,
                "group": group,
                "value": float(cycle + period + step + (group == "treatment")),
            }
            for unit, order in (("u1", ("control", "treatment")), ("u2", ("treatment", "control")))
            for cycle in range(2)
            for period, group in enumerate(order)
            for step in range(2)
        ]
    )
    analysis = Analysis.from_switchback_panel(
        frame,
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
            sequence=IndependentBernoulliOrder(probability_ct=0.5),
            window=SwitchbackWindow(washout_steps=1, observation_steps=1),
        ),
        plan=AnalysisPlan(primary="value"),
        experiment_id="switchback-run",
    )
    metric = synthesise_metric(MetricSpec(name="value", type="mean"))
    compiled = compile_decision_plan(AnalysisPlan(primary="value"), [metric], path="frame")
    registration = Registration.from_plan(compiled)
    try:
        legacy = analysis.run()
        with pytest.raises(CodedError) as raised:
            run(
                analysis,
                registration,
                compiled,
                experiment_id="switchback-run",
                analysis_spec_hash="sha256:" + "5" * 64,
                execution_identity_hash="sha256:" + "6" * 64,
                run_id="switchback-run",
            )
    finally:
        analysis.close()

    assert len(legacy) == 1
    assert raised.value.code == "store.run.legacy_analysis_required"
    assert raised.value.context["route"] == "Analysis.run()"


@pytest.mark.integration
def test_run_result_preserves_set_only_projection_through_store_codecs_and_rendering(tmp_path):
    import json

    from increment import MetricSpec, Randomized
    from increment.result_envelope import AnalysisEnvelope, render
    from increment.store import ParquetResultsStore

    data = pd.DataFrame(
        [
            {"unit": f"{group}-{index}", "arm": group, "value": float(success)}
            for group, outcomes in (("control", [0] * 40), ("treatment", [0] * 20 + [1] * 20))
            for index, success in enumerate(outcomes)
        ]
    )
    plan = AnalysisPlan(primary="value")
    metric = MetricSpec(name="value", type="conversion")
    compiled = compile_decision_plan(plan, [synthesise_metric(metric)], path="frame")
    registration = Registration.from_plan(compiled)
    analysis = Analysis.from_unit_summary(
        data,
        unit="unit",
        group="arm",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics=[metric],
        plan=plan,
        experiment_id="set-only",
    )
    store = ParquetResultsStore(tmp_path / "runs")
    try:
        result = run(
            analysis,
            registration,
            compiled,
            experiment_id="set-only",
            analysis_spec_hash="sha256:" + "c" * 64,
            execution_identity_hash="sha256:" + "d" * 64,
            run_id="set-only-run",
            results_store=store,
        )
    finally:
        analysis.close()

    envelope = AnalysisEnvelope(result.cells, result.run.metadata)
    restored = (
        AnalysisEnvelope.from_payload(json.loads(json.dumps(envelope.to_payload()))),
        AnalysisEnvelope.from_arrow(envelope.to_arrow()),
        store.open_run("set-only", "sha256:" + "c" * 64, "set-only-run").envelope(),
    )
    effect = result.effects[0]
    assert effect is not None and effect.point is None
    assert effect.lower is not None and effect.upper is None
    assert result.decisions.cells[0].status == "reject"
    for candidate in restored:
        output = render(candidate, "full")
        assert candidate.cells[0].summary == effect
        assert f"{effect.point!r} [{effect.lower!r}, {effect.upper!r}]" in output


def test_public_run_rejects_registration_that_differs_from_analysis_plan():
    import pytest

    from increment.errors import CodedError

    analysis_plan = AnalysisPlan(primary="value", alpha=0.05)
    registered_plan = AnalysisPlan(primary="value", alpha=0.01)
    metric = synthesise_metric(MetricSpec(name="value", type="mean"))
    compiled = compile_decision_plan(registered_plan, [metric], path="frame")
    registration = Registration.from_plan(compiled)
    analysis = Analysis.from_unit_summary(
        pd.DataFrame(
            {
                "unit": ["c1", "c2", "c3", "c4", "t1", "t2", "t3", "t4"],
                "arm": ["control"] * 4 + ["treatment"] * 4,
                "value": [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0],
            }
        ),
        unit="unit",
        group="arm",
        control="control",
        metrics={"value": "mean"},
        plan=analysis_plan,
    )
    try:
        with pytest.raises(CodedError) as error:
            run(
                analysis,
                registration,
                compiled,
                experiment_id="exp",
                analysis_spec_hash="sha256:" + "e" * 64,
                execution_identity_hash="sha256:" + "f" * 64,
                run_id="run-mismatched-plan",
            )
    finally:
        analysis.close()

    assert error.value.code == "store.decision.reference_invalid"


def test_optional_store_and_event_log_replay_same_decisions(tmp_path):
    from increment.store import DecisionLog, ParquetResultsStore

    plan = AnalysisPlan(primary="value", secondaries=["secondary"], q=0.1)
    metrics = [
        synthesise_metric(MetricSpec(name="value", type="mean")),
        synthesise_metric(MetricSpec(name="secondary", type="mean")),
    ]
    compiled = compile_decision_plan(plan, metrics, path="frame")
    registration = Registration.from_plan(compiled)
    frame = pd.DataFrame(
        {
            "unit": ["c1", "c2", "c3", "c4", "t1", "t2", "t3", "t4"],
            "arm": ["control"] * 4 + ["treatment"] * 4,
            "value": [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0],
            "secondary": [4.0, 3.0, 2.0, 1.0, 5.0, 4.0, 3.0, 2.0],
        }
    )
    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        control="control",
        metrics={"value": "mean", "secondary": "mean"},
        plan=plan,
        experiment_id="exp",
    )
    import pytest

    from increment.errors import CodedError

    with pytest.raises(CodedError) as omitted:
        run(
            analysis,
            registration,
            compiled,
            experiment_id="exp",
            analysis_spec_hash="sha256:" + "a" * 64,
            execution_identity_hash="sha256:" + "b" * 64,
            run_id="run-incomplete-roster",
            analysis_options={"metrics": ["value"]},
        )
    assert omitted.value.code == "store.decision.reference_invalid"
    store = ParquetResultsStore(tmp_path / "runs")
    log_path = tmp_path / "decisions.jsonl"
    decision_log = DecisionLog(path=log_path)
    try:
        result = run(
            analysis,
            registration,
            compiled,
            experiment_id="exp",
            analysis_spec_hash="sha256:" + "a" * 64,
            execution_identity_hash="sha256:" + "b" * 64,
            run_id="run-1",
            results_store=store,
            decision_log=decision_log,
        )
    finally:
        analysis.close()

    restored_log = DecisionLog(path=log_path)
    replayed = restored_log.replay(
        lambda experiment, run_id: store.open_run(experiment, "sha256:" + "a" * 64, run_id),
        registration,
    )

    assert result.run.decision_plan is not None

    from increment.store import HistorySelector

    identity = result.run.cells[0].identity
    assert identity.metric_hash
    assert identity.estimator_id
    assert identity.method_config_hash
    assert identity.method_instance_id
    assert identity.scope is not None
    history = store.history(
        "exp",
        HistorySelector(
            metric_hash=identity.metric_hash,
            estimator_id=identity.estimator_id,
            method_config_hash=identity.method_config_hash,
            method_instance_id=identity.method_instance_id,
            scope_hash=identity.scope.hash(),
        ),
    )
    assert history.num_rows == 1
    assert restored_log.events
    assert replayed[0] == result.decisions
    assert result.decisions.cells[0].membership == "member"
    assert result.decisions.cells[1].membership == "member"
    from dataclasses import replace

    import pytest

    from increment.errors import CodedError

    altered = DecisionLog()
    altered.append(replace(restored_log.events[0], evidence_digest="sha256:" + "0" * 64))
    with pytest.raises(CodedError) as error:
        altered.replay(
            lambda experiment, run_id: store.open_run(experiment, "sha256:" + "a" * 64, run_id),
            registration,
        )
    assert error.value.code == "store.decision.reference_invalid"
    changed_reference = replace(
        restored_log.events[0].references[0],
        identity=result.run.cells[0].identity,
    )
    altered_roster = DecisionLog()
    altered_roster.append(
        replace(
            restored_log.events[0],
            references=(changed_reference,),
        )
    )
    with pytest.raises(CodedError) as roster_error:
        altered_roster.replay(
            lambda experiment, run_id: store.open_run(experiment, "sha256:" + "a" * 64, run_id),
            registration,
        )
    assert roster_error.value.code == "store.decision.reference_invalid"


def test_public_run_binds_design_level_encouragement_compliance_identity():
    from increment import Encouragement, ExclusionRestriction, UptakeSpec

    plan = AnalysisPlan(primary="value")
    metric = synthesise_metric(MetricSpec(name="value", type="mean"))
    compiled = compile_decision_plan(plan, [metric], path="frame")
    registration = Registration.from_plan(compiled)
    analysis = Analysis.from_unit_summary(
        pd.DataFrame(
            {
                "unit": [f"u{i}" for i in range(8)],
                "arm": ["control"] * 4 + ["treatment"] * 4,
                "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "clicked": [0, 0, 0, 0, 0, 1, 1, 1],
            }
        ),
        unit="unit",
        group="arm",
        design=Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="clicked"),
            exclusion_restriction=ExclusionRestriction(
                justification="Test-only encouragement identity"
            ),
            one_sided=True,
        ),
        uptake="clicked",
        metrics={"value": "mean"},
        plan=plan,
        experiment_id="encouragement",
    )
    try:
        result = run(
            analysis,
            registration,
            compiled,
            experiment_id="encouragement",
            analysis_spec_hash="sha256:" + "c" * 64,
            execution_identity_hash="sha256:" + "d" * 64,
            run_id="run-encouragement",
        )
    finally:
        analysis.close()

    compliance = next(cell for cell in result.cells if cell.identity.key.metric == "uptake")
    assert compliance.identity.metric_hash
    assert compliance.identity.estimator_id == "randomized.unadjusted"
    assert compliance.identity.method_config_hash
    assert compliance.identity.method_instance_id == "uptake:decision:unadjusted"


def test_unicode_daily_metadata_order_roundtrips_after_run():
    from dataclasses import replace

    from increment import from_frame
    from increment.kernel.contracts import Scope
    from increment.result_envelope import AnalysisEnvelope

    rows = []
    for day in ("2025-01-01", "2025-01-02"):
        for arm, shift in (("control", 0.0), ("z", 1.0), ("é", 2.0)):
            for unit in range(8):
                rows.append(
                    {
                        "unit": f"{arm}-{day}-{unit}",
                        "arm": arm,
                        "day": day,
                        "value": float(unit) + shift,
                    }
                )
    experiment, resolver = from_frame(
        pd.DataFrame(rows),
        shape="panel",
        unit="unit",
        group="arm",
        date="day",
        control="control",
        metrics={"value": "mean"},
    )
    scope = Scope(view="daily", grain="daily")
    requested = replace(
        experiment,
        scopes=(scope,),
        operations={scope.hash(): "run_daily"},
    )

    result = run(requested, resolver)
    envelope = AnalysisEnvelope(result.cells, result.run.metadata)

    restored = AnalysisEnvelope.from_payload(envelope.to_payload())

    assert {cell.identity.key.group_id for cell in restored.cells} == {"control", "z", "é"}
