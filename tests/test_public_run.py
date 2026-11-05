from typing import Any, cast

import pandas as pd
import pytest

from increment import (
    AnalysisPlan,
    RunResult,
    View,
    from_definitions,
    from_frame,
    run,
)
from increment.decision_wire import compiled_plan_from_dict
from increment.frame import MetricSpec, synthesise_metric
from increment.kernel.contracts import Registration
from increment.plan import compile_decision_plan
from tests.analysis_factory import view_readout


def test_top_level_run_surface_is_public():
    assert callable(run)
    assert RunResult.__name__ == "RunResult"


def test_top_level_run_returns_a_validated_in_memory_result():
    from dataclasses import replace

    import pytest

    from increment.errors import CodedError
    from increment.store import Run

    plan = AnalysisPlan(primary="value")
    metric = synthesise_metric(MetricSpec(name="value", type="mean"))
    compiled = compile_decision_plan(plan, [metric], path="frame")
    registration = Registration.from_plan(compiled)
    experiment, resolver = from_frame(
        pd.DataFrame(
            {
                "unit": ["c1", "c2", "c3", "c4", "t1", "t2", "t3", "t4"],
                "arm": ["control"] * 4 + ["treatment"] * 4,
                "value": [1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0],
            }
        ),
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics={"value": "mean"},
        plan=plan,
        experiment_id="exp",
    )
    try:
        result = run(experiment, resolver, run_id="run-1")
    finally:
        resolver.close()

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

    assert isinstance(result.run, Run)
    with pytest.raises(CodedError) as mismatch:
        replace(result.run, registration_hash="sha256:" + "0" * 64)
    assert mismatch.value.code == "store.run.identity_mismatch"
    assert result.decisions.cells[0].status == "reject"
    assert result.decisions.families[0].status == "reject"


_LOWER = [1.0, 1.01, 0.99, 1.0]
_HIGHER = [2.0, 2.01, 1.99, 2.0]


def _guardrail_case_inputs(direction="increase", alternative="two-sided"):
    from increment import ExperimentMetric

    plan = AnalysisPlan(
        primary="value",
        guardrails=[ExperimentMetric(metric="guard", margin_abs=0.5)],
        alternative=alternative,
    )
    value = (
        MetricSpec(name="value", type="mean")
        if direction is None
        else MetricSpec(name="value", type="mean", preferred_direction=direction)
    )
    specs = [value, MetricSpec(name="guard", type="mean", preferred_direction="decrease")]
    metrics = [synthesise_metric(spec) for spec in specs]
    compiled = compile_decision_plan(plan, metrics, path="frame")
    registration = Registration.from_plan(compiled)
    return plan, specs, compiled, registration


def _guardrail_case_frame(*, treatment_higher=True):
    control, treatment = (_LOWER, _HIGHER) if treatment_higher else (_HIGHER, _LOWER)
    return pd.DataFrame(
        {
            "unit": [f"u{i}" for i in range(8)],
            "arm": ["control"] * 4 + ["treatment"] * 4,
            "value": control + treatment,
            "guard": [10.0, 10.01, 9.99, 10.0, 10.0, 10.01, 9.99, 10.0],
        }
    )


def _guardrail_case_analysis(plan, specs, *, treatment_higher=True):
    from increment import Randomized

    return from_frame(
        _guardrail_case_frame(treatment_higher=treatment_higher),
        shape="summary",
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


def _guardrail_case_result(direction="increase", *, treatment_higher=True, alternative="two-sided"):
    plan, specs, compiled, registration = _guardrail_case_inputs(direction, alternative)
    experiment, resolver = _guardrail_case_analysis(plan, specs, treatment_higher=treatment_higher)
    try:
        result = run(experiment, resolver, run_id="run-guardrail")
    finally:
        resolver.close()
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


def test_primary_rejecting_on_adverse_side_blocks_program():
    # Means 2.0 against 1.0: lift -0.5 with relative SE about 0.0023, |t| about 217.
    result, _, _ = _guardrail_case_result("increase", treatment_higher=False)

    primary = next(cell for cell in result.decisions.cells if cell.role == "primary")
    assert (primary.status, primary.direction) == ("reject", "adverse")
    assert result.decisions.program.status == "block"
    assert "decision.primary.adverse" in result.decisions.program.reason_codes


@pytest.mark.parametrize(("treatment_higher", "status"), [(False, "pass"), (True, "block")])
def test_primary_direction_orients_program_verdict(treatment_higher, status):
    result, _, _ = _guardrail_case_result("decrease", treatment_higher=treatment_higher)

    program = result.decisions.program
    assert program.status == status
    assert ("decision.primary.adverse" in program.reason_codes) == (status == "block")


@pytest.mark.parametrize(
    ("alternative", "treatment_higher", "status", "reason"),
    [
        ("two-sided", True, "not_evaluated", "decision.primary.direction_unspecified"),
        ("greater", True, "pass", None),
        ("less", False, "pass", None),
    ],
)
def test_primary_without_direction_is_not_evaluated(alternative, treatment_higher, status, reason):
    """An undirected primary has no favorable side under a two-sided plan; a one-sided
    alternative names it, so a rejection on that side passes."""
    result, _, _ = _guardrail_case_result(
        None, treatment_higher=treatment_higher, alternative=alternative
    )

    program = result.decisions.program
    assert program.status == status
    assert program.reason_codes == (() if reason is None else (reason,))


def test_one_adverse_treatment_arm_blocks_program():
    from increment import Randomized

    frame = pd.DataFrame(
        {
            "unit": [f"u{i}" for i in range(16)],
            "arm": ["control"] * 8 + ["up"] * 4 + ["down"] * 4,
            "value": _HIGHER * 2 + [3.0, 3.01, 2.99, 3.0] + _LOWER,
        }
    )
    experiment, resolver = from_frame(
        frame,
        shape="summary",
        unit="unit",
        group="arm",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "up": 0.25, "down": 0.25},
            allocation_scheme="independent",
        ),
        metrics=[MetricSpec(name="value", type="mean", preferred_direction="increase")],
        plan=AnalysisPlan(primary="value"),
        experiment_id="exp",
    )
    try:
        result = run(experiment, resolver, run_id="run-arms")
    finally:
        resolver.close()

    primary = [cell for cell in result.decisions.cells if cell.role == "primary"]
    assert sorted((cell.status, cell.direction) for cell in primary) == [
        ("reject", "adverse"),
        ("reject", "favorable"),
    ]
    assert result.decisions.program.status == "block"
    assert "decision.primary.adverse" in result.decisions.program.reason_codes


@pytest.mark.parametrize(
    ("direction", "effect", "verdict", "status", "primary_code"),
    [
        ("increase", 10.0, "favorable", "block", None),
        ("increase", -10.0, "adverse", "block", "decision.primary.adverse"),
        (None, 10.0, "unspecified", "not_evaluated", "decision.primary.direction_unspecified"),
    ],
)
def test_switchback_primary_direction_orients_program(
    tmp_path, direction, effect, verdict, status, primary_code
):
    """A switchback primary's verdict reads the direction its contrast procedure binds, both
    in the live run and when the run is decided from its stored and reopened cells. Switchback
    assignment integrity is never checked, so a favorable primary blocks on that code alone."""
    from increment import from_switchback_frame
    from increment.store import ParquetResultsStore
    from tests.test_frame_switchback import _assignment, _identification

    rows = []
    for unit in range(12):
        for cycle in range(4):
            order = (
                ("control", "treatment") if (unit + cycle) % 2 == 0 else ("treatment", "control")
            )
            control_value = 10.0 + 0.3 * unit + 0.1 * cycle
            contrast = effect + (unit % 3 - 1) * 0.1 + (cycle % 2 * 2 - 1) * 0.02
            for period, group in enumerate(order):
                value = control_value + (contrast if group == "treatment" else 0.0)
                rows.extend(
                    {
                        "unit": f"u{unit}",
                        "cycle": cycle,
                        "period": period,
                        "step": step,
                        "group": group,
                        "value": 999.0 if step == 0 else value,
                    }
                    for step in range(3)
                )
    spec = (
        MetricSpec(name="value", type="mean")
        if direction is None
        else MetricSpec(name="value", type="mean", preferred_direction=direction)
    )
    experiment, resolver = from_switchback_frame(
        pd.DataFrame(rows),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics=[spec],
        identification=_identification(),
        assignment=_assignment(p_ct=0.5),
        plan=AnalysisPlan(primary="value"),
        experiment_id="switchback",
    )
    with resolver:
        live = run(experiment, resolver).decisions
        # With a results store, the run is decided from its reopened stored cells.
        restored = run(experiment, resolver, results_store=ParquetResultsStore(tmp_path)).decisions

    codes = {"integrity.unsupported_assignment"} | ({primary_code} if primary_code else set())
    for decisions in (live, restored):
        assert decisions.program.status == status
        assert decisions.program.reason_codes == tuple(sorted(codes))
        (primary,) = [cell for cell in decisions.cells if cell.role == "primary"]
        assert (primary.status, primary.direction) == ("reject", verdict)


def _sequential_harm_result(kind, *, direction="increase"):
    import numpy as np

    from increment import InferenceSpec, Randomized

    n = 3000
    arm = np.array(["control", "treatment"] * n)
    treated = arm == "treatment"
    if kind == "always_valid":
        # Conversion 5% -> 1%: 150 against 30 events per 3000 units.
        index = np.arange(2 * n) // 2
        values = np.where(treated, index % 100 == 0, index % 20 == 0).astype(int)
        spec = MetricSpec(name="value", type="conversion", preferred_direction=direction)
        inference = InferenceSpec(kind="always_valid", baseline_rate=0.05)
    else:
        # Mean 20 -> 18 (-10%) with unit sd 5.
        values = 20.0 + np.where(treated, -2.0, 0.0) + np.random.default_rng(3).normal(0, 5, 2 * n)
        spec = MetricSpec(name="value", type="mean", preferred_direction=direction)
        inference = InferenceSpec(kind="asymptotic_mean")
    experiment, resolver = from_frame(
        pd.DataFrame(
            {
                "unit": [f"u{i}" for i in range(2 * n)],
                "arm": arm,
                "exposed": np.arange(2 * n),
                "value": values,
            }
        ),
        shape="summary",
        unit="unit",
        group="arm",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics=[spec],
        plan=AnalysisPlan(primary="value", inference=inference),
        experiment_id="seq",
        exposure_date="exposed",
    )
    try:
        return run(experiment, resolver, run_id="run-sequential")
    finally:
        resolver.close()


@pytest.mark.parametrize("kind", ["always_valid", "asymptotic_mean"])
def test_sequential_primary_rejecting_on_adverse_side_blocks_program(kind):
    result = _sequential_harm_result(kind)

    assert result.readout()[0].preferred_direction == "increase"
    assert result.decisions.program.status == "block"
    assert "decision.primary.adverse" in result.decisions.program.reason_codes


@pytest.mark.parametrize("sequential", [False, True])
def test_row_and_procedure_direction_agree(sequential):
    result = (
        _sequential_harm_result("always_valid", direction="decrease")
        if sequential
        else _guardrail_case_result("decrease")[0]
    )

    assert result.run.decision_plan is not None
    bound = compiled_plan_from_dict(result.run.decision_plan.plan_payload)
    rows = [row for row in result.readout() if row.metric == "value"]
    assert rows and {row.preferred_direction for row in rows} == {"decrease"}
    assert cast(Any, bound.procedures["value"]).favorable_side == "less"


def test_version_1_bound_plan_reads_direction_unspecified(tmp_path):
    """A run saved before plans recorded favorable sides: the bound plan records the
    one-sided test (``alternative="less"``, the adverse side of a primary declared
    ``increase``) and no favorable side, and its rejection on that side must not pass."""
    from dataclasses import replace

    from increment import Randomized
    from increment.decide import decide
    from increment.kernel.contracts import CompiledPlanBinding
    from increment.store import ParquetResultsStore

    experiment, resolver = from_frame(
        _guardrail_case_frame(treatment_higher=False),
        shape="summary",
        unit="unit",
        group="arm",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics=[MetricSpec(name="value", type="mean")],
        plan=AnalysisPlan(primary="value", alternative="less"),
        experiment_id="exp",
    )
    try:
        result = run(experiment, resolver, run_id="run-v1")
    finally:
        resolver.close()
    assert result.run.decision_plan is not None
    payload = result.run.decision_plan.plan_payload
    for procedure in cast(dict[str, dict[str, object]], payload["procedures"]).values():
        procedure.pop("favorable_side", None)
    registration = Registration.from_plan(compiled_plan_from_dict(payload))
    saved = replace(
        cast(Any, result.run),
        registration_hash=registration.registration_hash,
        decision_plan=CompiledPlanBinding(registration.registration_hash, 1, payload),
    )
    store = ParquetResultsStore(tmp_path)
    store.put(saved)

    handle = store.open_run(saved.experiment_id, saved.analysis_spec_hash, saved.run_id)
    decisions = decide(handle, registration)

    primary = next(cell for cell in decisions.cells if cell.role == "primary")
    assert (primary.status, primary.direction) == ("reject", "unspecified")
    assert decisions.program.status == "not_evaluated"
    assert "decision.primary.direction_unspecified" in decisions.program.reason_codes


def test_undirected_two_sided_plan_identity_unchanged(tmp_path):
    """An experiment that declares no direction under a two-sided plan keeps the identity it
    had before plans recorded favorable sides, and its run binds a version-1 plan. Its moments
    export, whose source declaration holds the compiled plan, keeps its identity too."""
    import pyarrow.parquet as pq

    from increment import Randomized, from_moments
    from increment.workflows import export_moments

    specs = [MetricSpec(name="value", type="mean"), MetricSpec(name="guard", type="mean")]
    plan = AnalysisPlan(primary="value")
    experiment, resolver = from_frame(
        _guardrail_case_frame(),
        shape="summary",
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
    try:
        result = run(experiment, resolver, run_id="run-identity")
        export_moments(experiment, resolver, tmp_path / "moments.parquet")
    finally:
        resolver.close()
    moments, moments_resolver = from_moments(
        pq.read_table(tmp_path / "moments.parquet").to_pylist(),
        control="control",
        metrics=specs,
        plan=plan,
    )
    moments_resolver.close()

    assert experiment.hash() == "a6dfa22c9e36df96d652849f9d87867d5129af4f6cf70d3cc992a9c7dcf221da"
    assert experiment.registration.registration_hash == (
        "02ec970306c91a8236c653b22fb8aa78f96941ffc1d96eeef2fd588ccd41817c"
    )
    assert [cell.identity.hash() for cell in result.cells] == [
        "2b602368814ada3a5e1951ff047563430e45217cff5c9644c6643f1ad550845f",
        "90edcef83f55c9a5120d6ee3fcf3bf97af54a3d0d4822eac0e3252656d6f08af",
    ]
    assert result.run.decision_plan is not None and result.run.decision_plan.version == 1
    assert moments.hash() == "dec2c8f641e8e668c8b3bd752c2e40ad51fc00a131c6bbba7ce19b928d6971f1"


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
    experiment, resolver = from_frame(
        frame,
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics={"value": "mean", "first": "mean", "second": "mean"},
        plan=plan,
        experiment_id="family-roster",
    )
    try:
        result = run(experiment, resolver, run_id="run-family-roster")
    finally:
        resolver.close()

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
    from increment.store import Run
    from increment.store.types import pinned_run

    reversed_metadata = metadata.model_copy(
        update={"scope": scope.model_copy(update={"families": tuple(reversed(scope.families))})}
    )
    assert isinstance(result.run, Run)
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
    experiment, resolver = from_frame(
        data,
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics=[metric],
        plan=plan,
        experiment_id="selected-quantile",
    )
    try:
        legacy = view_readout(experiment, resolver)
        result = run(experiment, resolver, run_id="selected-quantile-run")
    finally:
        resolver.close()

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
    experiment, resolver = from_frame(
        data,
        shape="summary",
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
    try:
        legacy = view_readout(experiment, resolver)[0]
        result = run(experiment, resolver, run_id="observational-run")
    finally:
        resolver.close()

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
    experiment, resolver = from_frame(
        data,
        shape="summary",
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
    try:
        legacy = view_readout(experiment, resolver)
        result = run(experiment, resolver, run_id="sequential-ebh-run")
    finally:
        resolver.close()

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

    from increment.artifacts import SourceSnapshotEvidence
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
    experiment, resolver = from_definitions(
        definitions,
        con,
        experiment_name="exp",
        source_snapshot_evidence=SourceSnapshotEvidence(snapshot_at, {"events": snapshot_at}),
    )
    assigned = View(view="run", grain="total")
    triggered = View(view="run", grain="total", population="triggered")
    try:
        legacy = [
            *view_readout(experiment, resolver, assigned),
            *view_readout(experiment, resolver, triggered),
        ]
        result = run(experiment.with_views(triggered), resolver, run_id="triggered-run")
    finally:
        cast(Any, resolver).close()
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
    experiment, resolver = from_frame(
        data,
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics={"value": "mean"},
        breakouts=("segment",),
        plan=plan,
        experiment_id="breakout-run",
    )
    try:
        legacy = view_readout(experiment, resolver)
        breakout_rows = view_readout(experiment, resolver, View(view="breakout", grain="total"))
        result = run(experiment, resolver, run_id="breakout-run")
    finally:
        resolver.close()

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
    experiment, resolver = from_frame(
        data,
        shape="summary",
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
        result = run(experiment, resolver, run_id="set-only-run", results_store=store)
    finally:
        resolver.close()

    envelope = result.envelope
    restored = (
        AnalysisEnvelope.from_payload(json.loads(json.dumps(envelope.to_payload()))),
        AnalysisEnvelope.from_arrow(envelope.to_arrow()),
        store.open_run("set-only", result.run.analysis_spec_hash, "set-only-run").envelope(),
    )
    effect = result.effects[0]
    assert effect is not None and effect.point is None
    assert effect.lower is not None and effect.upper is None
    assert result.decisions.cells[0].status == "reject"
    for candidate in restored:
        output = render(candidate, "full")
        assert candidate.cells[0].summary == effect
        assert f"{effect.point!r} [{effect.lower!r}, {effect.upper!r}]" in output


def test_public_run_preserves_live_percentile_winsorized_result(tmp_path):
    import pickle

    from increment import MetricSpec, Randomized
    from increment.errors import WireFormatError
    from increment.estimation.readout_types import ReadoutResults
    from increment.estimation.results import LiftEstimate
    from increment.result_envelope import render
    from increment.store import DecisionLog, ParquetResultsStore
    from tests.test_kernel_review_regressions import _frame_read_calls

    data = pd.DataFrame(
        [
            {"unit": f"{group}-{index}", "arm": group, "value": float(index + offset)}
            for group, offset in (("control", 0), ("treatment", 8))
            for index in range(24)
        ]
    )
    metric = MetricSpec(
        name="value",
        type="mean",
        winsorization={"upper_percentile": 0.9},
    )
    experiment, resolver = from_frame(
        data,
        shape="summary",
        unit="unit",
        group="arm",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics=[metric],
        plan=AnalysisPlan(primary="value"),
        experiment_id="winsor-live",
    )
    result = run(experiment, resolver)
    live = result.live_rows[0]
    assert isinstance(live, LiftEstimate)
    assert live.lift is not None
    assert live.confidence_set is not None
    assert result.effects[0] is not None
    assert result.effects[0].point == pytest.approx(live.lift.value)
    assert result.effects[0].lower == pytest.approx(live.confidence_set.relative.lower.value)
    assert result.effects[0].upper == pytest.approx(live.confidence_set.relative.upper.value)
    output = render(result, "full")
    assert "confidence set" in output
    brief = render(result)
    assert f"{live.lift.value:.6g}" in brief
    assert "status=unavailable" in brief
    assert result.cells[0].summary is None
    restored = ReadoutResults.model_validate_json(result.readout().model_dump_json())
    assert restored[0].confidence_set == live.confidence_set
    with pytest.raises(WireFormatError) as pickled:
        pickle.dumps(result)
    assert pickled.value.code == "wire.result.raw_outcomes_unavailable"

    logged = run(experiment, resolver, decision_log=DecisionLog())
    assert logged.decisions.cells[0].status == "not_evaluated"

    with _frame_read_calls() as reads, pytest.raises(WireFormatError) as error:
        run(experiment, resolver, results_store=ParquetResultsStore(tmp_path / "runs"))
    assert error.value.code == "wire.result.raw_outcomes_unavailable"
    assert reads == []
    assert not tuple((tmp_path / "runs").rglob("*.parquet"))


def _daily_panel_run(tmp_path, scopes):
    from dataclasses import replace
    from datetime import date

    from increment import Randomized
    from increment.store import ParquetResultsStore

    days = (date(2025, 1, 1), date(2025, 1, 2))
    frame = pd.DataFrame(
        {
            "unit": [f"u{index}" for index in range(40)],
            "arm": ["control", "treatment"] * 20,
            "day": [days[index // 20] for index in range(40)],
            "value": [float(index % 7) for index in range(40)],
            "region": ["north", "north", "south", "south"] * 10,
        }
    )
    experiment, resolver = from_frame(
        frame,
        shape="panel",
        unit="unit",
        group="arm",
        date="day",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics={"value": "mean"},
        breakouts=["region"],
        plan=AnalysisPlan(primary="value"),
    )
    experiment = replace(experiment, views=scopes)
    store = ParquetResultsStore(tmp_path / "runs")
    result = run(experiment, resolver, results_store=store)
    restored = RunResult.from_handle(
        store.open_run(result.run.experiment_id, result.run.analysis_spec_hash, result.run.run_id),
        registration=experiment.registration,
    )
    return result, restored, days


def test_run_result_returns_each_scope_collection_and_refuses_ambiguous_selection(tmp_path):
    from increment.breakout.estimates import DailyMetricValues, LiftEstimates
    from increment.errors import CodedError

    whole = View(view="run", grain="total")
    values = View(view="daily", grain="daily", values=True)
    result, restored, days = _daily_panel_run(tmp_path, (whole, values))

    whole_rows = result.readout(whole)
    value_rows = result.readout(values)
    assert isinstance(whole_rows, LiftEstimates) and isinstance(value_rows, DailyMetricValues)
    assert [row.metric for row in whole_rows] == ["value"]
    assert {(row.group_id, row.ds) for row in value_rows} == {
        (arm, day) for arm in ("control", "treatment") for day in days
    }
    run_id = result.run.run_id
    lift = View(view="daily", grain="daily")
    for select, code, context in (
        (
            result.readout,
            "results.readout.scope_required",
            {"run_id": run_id, "scope_hashes": (whole.hash(), values.hash())},
        ),
        (
            lambda: result.readout(lift),
            "results.readout.scope_unknown",
            {"run_id": run_id, "scope_hash": lift.hash()},
        ),
        (lambda: restored.readout(values), "results.readout.not_retained", {"run_id": run_id}),
    ):
        with pytest.raises(CodedError) as raised:
            select()
        assert raised.value.code == code
        assert raised.value.context == context
        with pytest.raises(TypeError):
            raised.value.context["run_id"] = "another-run"  # ty: ignore[invalid-assignment]


def test_run_frame_keys_each_row_to_its_view_through_store_and_pickle(tmp_path):
    import pickle

    import narwhals as nw

    views = (
        View(view="run", grain="total"),
        View(view="breakout", grain="total"),
        View(view="daily", grain="daily"),
        View(view="daily", grain="daily", values=True),
        View(view="asof", grain="asof"),
        View(view="asof", grain="asof", values=True),
        View(view="asof", grain="asof", completion_policy=True),
    )
    result, restored, _ = _daily_panel_run(tmp_path, views)
    daily_lift, asof_lift = result.readout(views[2]), result.readout(views[4])
    assert {(row.metric, row.ds) for row in daily_lift} == {
        (row.metric, row.ds) for row in asof_lift
    }

    # No view declares a day, segment or population, so each row's key is its view's hash.
    identity = ("scope_view", "scope_values", "scope_key")
    fields = ("metric", "group_id", "dimension_value", "ds", "lift", "value")
    expected = sorted(
        (
            (scope.view, scope.values, scope.hash(), *(row.get(name) for name in fields))
            for scope in views
            for row in nw.from_native(
                result.readout(scope).to_frame(backend="polars"), eager_only=True
            ).rows(named=True)
        ),
        key=repr,
    )
    for projected in (result, restored, pickle.loads(pickle.dumps(result))):
        frame = nw.from_native(projected.to_frame(backend="polars"), eager_only=True)
        assert frame.columns[:3] == list(identity)
        assert sorted(frame.select(*identity, *fields).rows(), key=repr) == expected
    selected = nw.from_native(result.to_frame(columns=["lift"]), eager_only=True)
    assert selected.columns == [*identity, "lift", "view_partial"]


def _two_metric_run(*, skip_sessions=False):
    import numpy as np

    from increment import Randomized

    rng = np.random.default_rng(2)
    n = 400
    arm = np.array(["control"] * n + ["treatment"] * n)
    frame = pd.DataFrame(
        {
            "u": [f"u{i}" for i in range(2 * n)],
            "arm": arm,
            "revenue": 10 + (arm == "treatment") * 0.5 + rng.normal(0, 2, 2 * n),
            "sessions": 4 + rng.normal(0, 1, 2 * n),
        }
    )
    if skip_sessions:
        frame.loc[::10, "sessions"] = None
    experiment, reader = from_frame(
        frame,
        shape="summary",
        unit="u",
        group="arm",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics=[
            MetricSpec(name="revenue"),
            MetricSpec(name="sessions", missing="skip" if skip_sessions else "error"),
        ],
        plan=AnalysisPlan(primary="revenue", secondaries=["sessions"]),
    )
    with reader:
        return run(experiment, reader)


def _sequential_panel_run():
    import datetime as dt
    from dataclasses import replace

    from increment import InferenceSpec, Randomized
    from increment.evidence import SequentialCaptureRequest

    exposed, closed = dt.date(2025, 1, 1), dt.date(2025, 1, 10)
    frame = pd.DataFrame(
        [
            {
                "unit": f"u{index}",
                "arm": "control" if index % 2 else "treatment",
                "day": day,
                "exposed_on": exposed,
                "value": value,
            }
            for index in range(80)
            for day, value in ((exposed, 0.0), (dt.date(2025, 1, 2), float(index % 7 + index % 2)))
        ]
    )
    experiment, reader = from_frame(
        frame,
        shape="panel",
        unit="unit",
        group="arm",
        date="day",
        exposure_date="exposed_on",
        observation_end=closed,
        metrics=[MetricSpec(name="value", type="mean", window_days=2)],
        design=Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5}),
        plan=AnalysisPlan(primary="value", inference=InferenceSpec(kind="asymptotic_mean")),
    )
    views = (View(view="run", grain="total"), View(view="asof", grain="asof"))
    with reader:
        result = run(
            replace(experiment, views=views),
            reader,
            capture=SequentialCaptureRequest(finalized=True, as_of=closed),
        )
    return result, views


def test_run_result_repr_is_a_compact_readout(tmp_path):
    from random import Random

    from tests.analysis_factory import lift_rows

    result = _two_metric_run()
    text = repr(result)

    assert len(text.splitlines()) <= 25
    assert "sha256" not in text
    assert result.run.run_id in text
    for row in lift_rows(result.readout()):
        assert row.metric in text
        assert f"{row.require_lift().value * 100:.6g}%" in text

    # A daily row with an estimate prints as available.
    daily = View(view="daily", grain="daily")
    daily_run, _, _ = _daily_panel_run(tmp_path, (daily,))
    daily_lines = [line for line in repr(daily_run).splitlines() if "date=" in line]
    estimated = [row for row in cast(Any, daily_run.readout(daily)) if row.lift is not None]
    assert estimated and len(daily_lines) == len(daily_run.cells)
    for row in estimated:
        (line,) = (line for line in daily_lines if f"date={row.ds}" in line)
        assert "status=available" in line

    # A conversion with no control events keeps its exact set without a point.
    rng = Random(13)
    arms = ["treatment" if rng.random() < 0.5 else "control" for _ in range(200)]
    experiment, reader = from_frame(
        pd.DataFrame(
            {
                "unit": [f"u{i:03d}" for i in range(200)],
                "arm": arms,
                "converted": [int(arm == "treatment" and i % 5 < 2) for i, arm in enumerate(arms)],
            }
        ),
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics={"converted": "conversion"},
    )
    with reader:
        conversion = run(experiment, reader)
    (row,) = lift_rows(conversion.readout())
    assert row.lift is None and row.binomial_set is not None
    (line,) = (line for line in repr(conversion).splitlines() if line.startswith("converted"))
    assert f"{row.binomial_set.lower * 100:.6g}" in line
    assert row.binomial_set.geometry in line
    assert "status=available" in line


def test_run_result_summary_frame_matches_view_summaries(tmp_path):
    import narwhals as nw

    from tests.analysis_factory import lift_rows

    identity = ["scope_view", "scope_values", "scope_key"]
    result = _two_metric_run(skip_sessions=True)
    summary = nw.from_native(result.to_frame(backend="polars", columns="summary"), eager_only=True)
    view = nw.from_native(
        result.readout().to_frame(backend="polars", columns="summary"), eager_only=True
    )
    assert summary.columns == [*identity, *view.columns]
    assert summary.select(view.columns).rows() == view.rows()
    # Only the skipped metric's row carries its observed-population label.
    labels = dict(summary.select("metric", "population").rows())
    (skipped,) = (row for row in lift_rows(result.readout()) if row.metric == "sessions")
    assert labels == {"revenue": None, "sessions": skipped.population}
    assert skipped.population is not None

    scopes = (View(view="run", grain="total"), View(view="daily", grain="daily"))
    both, _, _ = _daily_panel_run(tmp_path, scopes)
    frame = nw.from_native(both.to_frame(backend="polars", columns="summary"), eager_only=True)
    views = {
        scope.view: nw.from_native(
            both.readout(scope).to_frame(backend="polars", columns="summary"), eager_only=True
        )
        for scope in scopes
    }
    assert frame.columns[:3] == identity
    assert set(frame.columns) == {*identity, *(name for v in views.values() for name in v.columns)}
    for name, view_frame in views.items():
        rows = frame.filter(nw.col("scope_view") == name).select(view_frame.columns)
        assert rows.rows() == view_frame.rows()


def test_run_result_frame_refuses_a_bare_column_string():
    from increment.errors import CodedError

    result = _two_metric_run()
    with pytest.raises(CodedError) as raised:
        result.to_frame(columns="lift")
    assert raised.value.code == "results.frame.columns_string"
    assert set(raised.value.context) == {"columns", "route"}
    assert raised.value.context["columns"] == "lift"


def test_run_result_summary_covers_every_view_kind(tmp_path):
    import narwhals as nw

    # A sequential plan governs every view of its run and refuses daily slices and
    # unregistered breakouts, so its whole-window and as-of cells come from a second run.
    fixed_views = (
        View(view="run", grain="total"),
        View(view="breakout", grain="total"),
        View(view="daily", grain="daily"),
    )
    fixed, _, _ = _daily_panel_run(tmp_path, fixed_views)
    sequential, sequential_views = _sequential_panel_run()
    assert {cell.identity.key.inference for cell in sequential.cells} == {"asymptotic_mean"}
    for result, views in ((fixed, fixed_views), (sequential, sequential_views)):
        text = repr(result)
        assert len(text.splitlines()) <= 25
        for cell, effect in zip(result.cells, result.effects, strict=True):
            key = cell.identity.key
            for coordinate in (key.metric, key.dimension_value, key.ds):
                if coordinate is not None:
                    assert str(coordinate) in text
            if effect is not None and effect.point is not None:
                factor = 100 if effect.value_scale == "relative" else 1
                assert f"{effect.point * factor:.6g}" in text
        frame = nw.from_native(
            result.to_frame(backend="polars", columns="summary"), eager_only=True
        )
        expected = {"scope_view", "scope_values", "scope_key"}
        for scope in views:
            expected.update(
                nw.from_native(
                    result.readout(scope).to_frame(backend="polars", columns="summary"),
                    eager_only=True,
                ).columns
            )
        assert set(frame.columns) == expected


def test_live_winsor_effect_preserves_unbounded_and_undefined_endpoints():
    from increment import MetricSpec, Randomized
    from increment.estimation.results import LiftEstimate
    from increment.winsor import WinsorInferenceSpec, WinsorSupport

    for upper, expected_status, control_values, treatment_values in (
        (None, "unbounded", (1, 2, 3, 9), (1, 2, 3, 5)),
        (0, "undefined", (0, 0), (0, 0)),
    ):
        plan = AnalysisPlan(primary="value")
        metric = MetricSpec(
            name="value",
            type="mean",
            winsorization={
                "upper_percentile": 0.99,
                "support": WinsorSupport(
                    lower=0, upper=upper, provenance="Declared outcome support"
                ),
                "inference": WinsorInferenceSpec(method="joint-rank-projection-v1"),
            },
        )
        experiment, resolver = from_frame(
            pd.DataFrame(
                [
                    {"unit": f"{group}-{index}", "arm": group, "value": float(value)}
                    for group, values in (
                        ("control", control_values),
                        ("treatment", treatment_values),
                    )
                    for index, value in enumerate(values)
                ]
            ),
            shape="summary",
            unit="unit",
            group="arm",
            design=Randomized(
                control_group="control",
                allocation={"control": 0.5, "treatment": 0.5},
                allocation_scheme="independent",
            ),
            metrics=[metric],
            plan=plan,
            experiment_id=f"rank-{expected_status}",
        )
        try:
            row = view_readout(experiment, resolver)[0]
            assert isinstance(row, LiftEstimate)
            result = run(experiment, resolver, run_id=f"rank-{expected_status}")
            effect = result.effects[0]
            assert effect is not None
            assert row.confidence_set is not None
            assert row.confidence_set.relative.upper.status == expected_status
            assert effect.upper is None
            assert effect.alpha == row.confidence_set.alpha
            assert effect.qualification == row.confidence_set.qualification
            assert effect.reason is not None
        finally:
            resolver.close()


def test_public_run_refuses_an_empty_run_id_before_reads():
    from increment.errors import CodedError
    from tests.test_kernel_review_regressions import _frame_read_calls

    experiment, resolver = from_frame(
        pd.DataFrame(
            {
                "unit": ["c1", "c2", "t1", "t2"],
                "arm": ["control", "control", "treatment", "treatment"],
                "value": [1.0, 2.0, 2.0, 3.0],
            }
        ),
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics={"value": "mean"},
    )
    with _frame_read_calls() as reads, pytest.raises(CodedError) as error:
        run(experiment, resolver, run_id="")

    assert (error.value.code, error.value.context["field"]) == (
        "store.run.identity_incomplete",
        "run_id",
    )
    assert reads == []


def _arm_experiment_registered_elsewhere():
    analysis_plan = AnalysisPlan(primary="value", alpha=0.05)
    registered_plan = AnalysisPlan(primary="value", alpha=0.01)
    metric = synthesise_metric(MetricSpec(name="value", type="mean"))
    compiled = compile_decision_plan(registered_plan, [metric], path="frame")
    experiment, resolver = from_frame(
        pd.DataFrame(
            {
                "unit": ["c1", "c2", "c3", "c4", "t1", "t2", "t3", "t4"],
                "arm": ["control"] * 4 + ["treatment"] * 4,
                "value": [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0],
            }
        ),
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics={"value": "mean"},
        plan=analysis_plan,
    )
    return experiment, resolver, Registration.from_plan(compiled)


def _switchback_experiment_registered_elsewhere():
    from increment import from_switchback_frame
    from tests.test_frame_switchback import _assignment, _identification, _rows

    def build(alpha: float):
        return from_switchback_frame(
            pd.DataFrame(_rows()),
            unit="unit",
            cycle="cycle",
            period="period",
            step="step",
            group="group",
            metrics={"mean": "mean"},
            identification=_identification(),
            assignment=_assignment(),
            plan=AnalysisPlan(primary="mean", alpha=alpha),
        )

    registered, other = build(0.01)
    other.close()
    experiment, resolver = build(0.05)
    return experiment, resolver, registered.registration


@pytest.mark.parametrize(
    "build", [_arm_experiment_registered_elsewhere, _switchback_experiment_registered_elsewhere]
)
def test_public_run_rejects_registration_that_differs_from_analysis_plan(tmp_path, build):
    from dataclasses import replace

    from increment.errors import CodedError
    from increment.store import DecisionLog, ParquetResultsStore
    from tests.test_kernel_review_regressions import _frame_read_calls

    experiment, resolver, registration = build()
    log_path = tmp_path / "decisions.jsonl"
    with resolver, _frame_read_calls() as reads, pytest.raises(CodedError) as error:
        run(
            replace(experiment, registration=registration),
            resolver,
            results_store=ParquetResultsStore(tmp_path / "runs"),
            decision_log=DecisionLog(path=log_path),
        )

    assert error.value.code == "store.decision.reference_invalid"
    assert reads == []
    assert not tuple((tmp_path / "runs").rglob("*.parquet"))
    assert not log_path.exists()


def test_optional_store_and_event_log_replay_same_decisions(tmp_path):
    from dataclasses import replace

    from increment.errors import CodedError
    from increment.store import DecisionLog, HistorySelector, ParquetResultsStore
    from tests.test_kernel_review_regressions import _frame_read_calls

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
    experiment, resolver = from_frame(
        frame,
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics={"value": "mean", "secondary": "mean"},
        plan=plan,
        experiment_id="exp",
    )
    store = ParquetResultsStore(tmp_path / "runs")
    log_path = tmp_path / "decisions.jsonl"
    decision_log = DecisionLog(path=log_path)

    with _frame_read_calls() as reads, pytest.raises(CodedError) as omitted:
        run(
            experiment,
            resolver,
            run_id="run-incomplete-roster",
            results_store=store,
            decision_log=decision_log,
            metrics=["value"],
        )
    assert omitted.value.code == "store.decision.reference_invalid"
    assert reads == []
    assert not tuple((tmp_path / "runs").rglob("*.parquet"))
    assert not log_path.exists()

    result = run(
        experiment, resolver, run_id="run-1", results_store=store, decision_log=decision_log
    )

    def open_run(experiment_id, run_id):
        return store.open_run(experiment_id, result.run.analysis_spec_hash, run_id)

    restored_log = DecisionLog(path=log_path)
    replayed = restored_log.replay(open_run, registration)

    assert result.run.run_id == "run-1"
    assert result.run.decision_plan is not None

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

    reopened = RunResult.from_handle(open_run("exp", "run-1"), registration=registration)
    assert reopened.cells == result.cells
    assert reopened.effects == result.effects
    assert reopened.decisions == result.decisions
    assert reopened.live_rows == (None,) * len(result.cells)
    other_plan = AnalysisPlan(primary="value", secondaries=["secondary"], q=0.05)
    wrong_registration = Registration.from_plan(
        compile_decision_plan(other_plan, metrics, path="frame")
    )
    with pytest.raises(CodedError) as wrong:
        RunResult.from_handle(open_run("exp", "run-1"), registration=wrong_registration)
    assert wrong.value.code == "store.decision.reference_invalid"

    altered = DecisionLog()
    altered.append(replace(restored_log.events[0], evidence_digest="sha256:" + "0" * 64))
    with pytest.raises(CodedError) as error:
        altered.replay(open_run, registration)
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
        altered_roster.replay(open_run, registration)
    assert roster_error.value.code == "store.decision.reference_invalid"


def test_event_log_replays_bonferroni_breakout_families(tmp_path):
    """A finite-sample Bonferroni family has no FDR level; its logged event still replays."""
    from increment import MultiplicitySpec
    from increment.store import DecisionLog, ParquetResultsStore

    frame = pd.DataFrame(
        {
            "unit": [f"u{index}" for index in range(16)],
            "arm": ["control"] * 8 + ["treatment"] * 8,
            "segment": ["north", "south"] * 8,
            "value": [3.0, 4.0, 5.0, 6.0, 3.5, 4.5, 5.5, 6.5]
            + [4.0, 5.0, 6.0, 7.0, 4.5, 5.5, 6.5, 7.5],
        }
    )
    experiment, resolver = from_frame(
        frame,
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics={"value": "mean"},
        breakouts=("segment",),
        plan=AnalysisPlan(
            primary="value", view_multiplicity=MultiplicitySpec(correction="bonferroni")
        ),
        experiment_id="exp",
    )
    experiment = experiment.with_views(View(view="breakout", grain="total"))
    store = ParquetResultsStore(tmp_path / "runs")
    log_path = tmp_path / "decisions.jsonl"
    result = run(experiment, resolver, results_store=store, decision_log=DecisionLog(path=log_path))

    restored_log = DecisionLog(path=log_path)
    replayed = restored_log.replay(
        lambda experiment_id, run_id: store.open_run(
            experiment_id, result.run.analysis_spec_hash, run_id
        ),
        experiment.registration,
    )

    assert [event.rule for event in restored_log.events] == ["bonferroni"]
    assert replayed == (result.decisions,)


def test_public_run_binds_design_level_encouragement_compliance_identity():
    from increment import Encouragement, ExclusionRestriction, UptakeSpec

    plan = AnalysisPlan(primary="value")
    experiment, resolver = from_frame(
        pd.DataFrame(
            {
                "unit": [f"u{i}" for i in range(8)],
                "arm": ["control"] * 4 + ["treatment"] * 4,
                "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "clicked": [0, 0, 0, 0, 0, 1, 1, 1],
            }
        ),
        shape="summary",
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
        result = run(experiment, resolver, run_id="run-encouragement")
    finally:
        resolver.close()

    compliance = next(cell for cell in result.cells if cell.identity.key.metric == "uptake")
    assert compliance.identity.metric_hash
    assert compliance.identity.estimator_id == "randomized.unadjusted"
    assert compliance.identity.method_config_hash
    assert compliance.identity.method_instance_id == "uptake:decision:unadjusted"


def test_unicode_daily_metadata_order_roundtrips_after_run():
    from dataclasses import replace

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
    scope = View(view="daily", grain="daily")
    requested = replace(
        experiment,
        views=(scope,),
        operations={scope.hash(): "run_daily"},
    )

    result = run(requested, resolver)
    restored = AnalysisEnvelope.from_payload(result.envelope.to_payload())

    assert {cell.identity.key.group_id for cell in restored.cells} == {"control", "z", "é"}


def _guarded_breakout_run():
    """A healthy segment A beside a one-unit segment T, a control with a negative mean (N), a
    segment with no variance in either arm (Z) and a null segment W, read as a BH breakout."""
    import warnings

    from increment.errors import IncrementWarning

    arms = {
        "A": {"control": (8.0, 12.0) * 50, "treatment": (9.0, 13.0) * 50},
        "T": {"control": (10.0,), "treatment": (12.0,)},
        "N": {"control": (-3.0, 1.0) * 50, "treatment": (-1.0, 3.0) * 50},
        "Z": {"control": (5.0,) * 4, "treatment": (5.0,) * 4},
        "W": {"control": (8.0, 12.0) * 50, "treatment": (8.0, 12.0) * 50},
    }
    frame = pd.DataFrame(
        [
            {"unit": f"{segment}-{arm}-{index}", "arm": arm, "segment": segment, "value": value}
            for segment, by_arm in arms.items()
            for arm, values in by_arm.items()
            for index, value in enumerate(values)
        ]
    )
    experiment, reader = from_frame(
        frame,
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics={"value": "mean"},
        breakouts=("segment",),
        plan=AnalysisPlan(primary="value"),
    )
    view = View(view="breakout", grain="total")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IncrementWarning)
            result = run(experiment.with_views(view), reader)
    finally:
        reader.close()
    return experiment, result, view


def _decide_restored(experiment, result, cells=None):
    """Decide *result*'s run again from its cells after a portable payload round trip."""
    from dataclasses import replace

    from increment.decide import decide
    from increment.result_envelope import AnalysisEnvelope
    from increment.store.types import pinned_run

    envelope = AnalysisEnvelope(result.cells if cells is None else cells, result.run.metadata)
    restored = AnalysisEnvelope.from_payload(envelope.to_payload())
    return decide(
        pinned_run(replace(result.run, cells=restored.cells, metadata=restored.metadata)),
        experiment.registration,
    )


def _bh_family(decisions):
    (family,) = [family for family in decisions.families if family.procedure == "bh"]
    return family


def test_restored_decisions_match_live_breakout_family():
    from dataclasses import replace

    experiment, result, view = _guarded_breakout_run()
    live = {row.dimension_value: row for row in result.readout(view)}
    segment_of = {cell.identity.hash(): cell.identity.key.dimension_value for cell in result.cells}

    family = _bh_family(result.decisions)
    members = {segment_of[member.identity_hash]: member for member in family.members}

    assert set(members) == {"A", "N", "W", "Z"}
    # BH over m = 4: A's p of about 4e-4 is below 0.10/4; N and Z enter as p = 1, W has p = 1.
    discoveries = [True, False, False, False]
    assert [members[segment].discovery for segment in ("A", "N", "W", "Z")] == discoveries
    assert all(member.discovery == live[segment].discovery for segment, member in members.items())
    assert "breakout.nonpositive_mean" in members["N"].reason_codes
    assert "estimation.engine.lift_guard" in members["Z"].reason_codes
    assert family.reason_codes == (
        "breakout.nonpositive_mean",
        "decision.not_evaluated.monitoring_view",
        "estimation.engine.lift_guard",
    )
    assert _bh_family(_decide_restored(experiment, result)) == family

    # A segment row whose relative interval has zero variance and no additive margin (a
    # clustered shape a dataframe breakout cannot read) has no p-value; the live family takes
    # it as p = 1, and so must the stored one.
    cells = []
    for cell in result.cells:
        if cell.identity.key.dimension_value == "W":
            payload = cast("dict[str, Any]", cell.row_payload)
            interval = ("lb", "ub", "level", "alpha", "open_side", "log_mean", "log_se")
            lift = {**payload["data"]["lift"], **dict.fromkeys(interval)}
            data = {
                **payload["data"],
                "lift": lift,
                "relative_unavailable_reason": "zero_relative_variance",
            }
            cell = replace(cell, row_payload={**payload, "data": data})
        cells.append(cell)
    zero = _bh_family(_decide_restored(experiment, result, tuple(cells)))
    zero_members = {segment_of[member.identity_hash]: member for member in zero.members}
    assert [zero_members[segment].discovery for segment in ("A", "N", "W", "Z")] == discoveries
    assert "evidence.p_value.unavailable" in zero_members["W"].reason_codes


def test_restored_family_with_genuinely_missing_evidence_is_not_evaluated():
    from dataclasses import replace

    from increment.decide import decide
    from increment.store.types import pinned_run

    experiment, result, _ = _guarded_breakout_run()
    code = "estimation.binomial.finite_sample_arm_ceiling_exceeded"
    cells = []
    for cell in result.cells:
        if cell.identity.key.dimension_value == "Z":
            payload = cast("dict[str, Any]", cell.row_payload)
            data = {
                **payload["data"],
                "excluded": "estimation_failed",
                "failure_code": code,
                "sampling_reason_code": code,
            }
            cell = replace(cell, row_payload={**payload, "data": data})
        cells.append(cell)

    live = decide(pinned_run(replace(result.run, cells=tuple(cells))), experiment.registration)
    restored = _decide_restored(experiment, result, tuple(cells))

    for family in (_bh_family(live), _bh_family(restored)):
        assert family.status == "not_evaluated"
        assert all(member.discovery is None for member in family.members)
        assert {"results.operation.unsupported", code} <= set(family.reason_codes)
    assert _bh_family(restored) == _bh_family(live)
