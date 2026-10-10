"""Pre-data observation identity, compliance policy and public runtime behavior."""

from datetime import date, timedelta
from fractions import Fraction

import pytest

from increment import Analysis, SequentialCell, SequentialCompliancePolicy, SequentialRegistration
from increment.errors import CapabilityError, InvalidRequestError
from increment.estimation.results import LiftEstimate
from increment.frame import MetricSpec
from tests.sequential_cases import UnreadFrame
from tests.test_sequential_public_sources import _analysis, _frame, _native_fixture, _plan, _row


def _uptake_design():
    """The encouragement identification the uptake fixtures are built with."""
    from increment.semantics.design import Encouragement

    return Encouragement.model_validate(
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


@pytest.mark.parametrize("role", ["unit", "group", "date", "exposure_date"])
def test_initial_panel_mapping_mismatch_refuses_before_frame_access(role):
    specs = [MetricSpec(name="outcome", type="conversion", window_days=2)]
    plan = _plan(specs, "bernoulli", date="day", exposure_date="exposed")
    mapping = {"unit": "unit", "group": "arm", "date": "day", "exposure_date": "exposed"}
    mapping[role] = "another_column"
    with pytest.raises(CapabilityError) as exc:
        Analysis.from_unit_panel(
            UnreadFrame(),
            unit=mapping["unit"],
            group=mapping["group"],
            date=mapping["date"],
            exposure_date=mapping["exposure_date"],
            control="control",
            metrics=specs,
            experiment_id="experiment",
            plan=plan,
        )
    assert exc.value.code == "sequential.source.invalid"


@pytest.mark.parametrize("role", ["unit", "group"])
def test_initial_summary_mapping_mismatch_refuses_before_frame_access(role):
    specs = [MetricSpec(name="outcome", type="conversion")]
    mapping = {"unit": "unit", "group": "arm"}
    mapping[role] = "another_column"
    with pytest.raises(CapabilityError) as exc:
        Analysis.from_unit_summary(
            UnreadFrame(),
            unit=mapping["unit"],
            group=mapping["group"],
            control="control",
            metrics=specs,
            experiment_id="experiment",
            plan=_plan(specs, "bernoulli"),
            exposure_date="exposure",
        )
    assert exc.value.code == "sequential.source.invalid"


@pytest.mark.parametrize("recipe", ["exposure", "outcome"])
def test_initial_native_mapping_mismatch_refuses_before_querying_new_recipe(recipe):
    from tests.analysis_factory import make_analysis

    connection, definitions, original = _native_fixture("bernoulli")
    try:
        field = "exposures" if recipe == "exposure" else "fact_sources"
        declarations = getattr(definitions, field)
        changed = declarations[0].model_copy(update={"sql": "SELECT * FROM must_never_be_read"})
        definitions = definitions.model_copy(update={field: (changed, *declarations[1:])})
        with pytest.raises(CapabilityError) as exc:
            source = make_analysis(connection, definitions, experiment=original.experiment)
            source.capture_sequential(finalized=True, as_of=date(2025, 1, 16))
        assert exc.value.code == "sequential.source.invalid"
    finally:
        connection.disconnect()


def test_negative_bernoulli_null_ratio_refuses_at_registration():
    from tests.sequential_cases import registration

    with pytest.raises(InvalidRequestError) as exc:
        registration(
            cells=(
                SequentialCell(
                    metric="outcome",
                    group_id="treatment",
                    null_lift=Fraction(-2),
                ),
            )
        )
    assert exc.value.code == "sequential.registration.invalid"


def test_public_frame_raw_checkpoint_smoke():
    specs = [MetricSpec(name="outcome", type="conversion")]
    analysis = _analysis(_frame([0, 1], [1, 1]), specs, _plan(specs, "bernoulli"))
    row = _row(analysis)
    result = row.require_sequential_result()
    assert result.checkpoint.control.successes == 1
    assert result.checkpoint.treatment.successes == 2
    assert result.bounds.lower is not None
    assert row.require_lift().value == 1
    assert type(row).model_validate_json(row.model_dump_json()) == row


@pytest.mark.slow
@pytest.mark.parametrize("uptake_only", [False, True])
def test_public_run_uses_captured_randomized_or_encouragement_state_without_live_moments(
    uptake_only,
):
    connection, _, source = _native_fixture("bernoulli", uptake_only=uptake_only)
    try:
        snapshot = source.capture_sequential(finalized=True, as_of=date(2025, 1, 16))
        estimands = ("compliance",) if uptake_only else ("itt",)
        expected = list(source.run(estimands=estimands))
        for name in connection.list_tables():
            connection.drop_table(name)
        actual = list(source.run(estimands=estimands))
        assert actual == expected
        assert len(actual) == 1 and actual[0].stat_sig()
        assert actual[0].require_sequential_result().checkpoint.prefix_id == snapshot.prefix_id
    finally:
        connection.disconnect()


@pytest.mark.parametrize(
    "field,value",
    [
        ("alpha", Fraction(1, 10)),
        ("alternative", "less"),
        ("null_lift", Fraction(1, 4)),
        ("family", True),
    ],
)
def test_uptake_registration_must_match_explicit_compiled_policy(field, value):
    from increment.plan import compile_decision_plan

    connection, _, source = _native_fixture("bernoulli", uptake_only=True)
    try:
        plan = source.experiment.plan
        assert plan is not None and plan.inference is not None
        reg = plan.inference.registration
        cell = reg.roster[0].model_copy(update={field: value})
        reg = SequentialRegistration.model_validate({**reg.model_dump(), "roster": (cell,)})
        policy = SequentialCompliancePolicy(alpha=Fraction(1, 20))
        changed = plan.model_copy(
            update={
                "inference": plan.inference.model_copy(update={"registration": reg}),
                "compliance": policy,
            }
        )
        with pytest.raises(CapabilityError) as exc:
            compile_decision_plan(changed, source.metrics, design=_uptake_design())
        assert exc.value.code == "sequential.source.invalid"
    finally:
        connection.disconnect()


def test_matching_compliance_family_q_compiles():
    from increment.plan import compile_decision_plan

    connection, _, source = _native_fixture("bernoulli", uptake_only=True)
    try:
        plan = source.experiment.plan
        assert plan is not None and plan.inference is not None
        registration = plan.inference.registration
        cell = registration.roster[0].model_copy(update={"family": True})
        registration = registration.model_copy(update={"roster": (cell,)})
        changed = plan.model_copy(
            update={
                "inference": plan.inference.model_copy(update={"registration": registration}),
                "compliance": SequentialCompliancePolicy(
                    alpha=cell.alpha,
                    family=True,
                ),
            }
        )
        compiled = compile_decision_plan(changed, source.metrics, design=_uptake_design())
        assert compiled.q == float(registration.q)
    finally:
        connection.disconnect()


@pytest.mark.parametrize("registered_q", [Fraction(1, 2), Fraction(0.1) + Fraction(1, 10**100)])
def test_compliance_family_q_must_match_compiled_plan(registered_q):
    from increment.plan import compile_decision_plan

    connection, _, source = _native_fixture("bernoulli", uptake_only=True)
    try:
        plan = source.experiment.plan
        assert plan is not None and plan.inference is not None
        registration = plan.inference.registration
        cell = registration.roster[0].model_copy(update={"family": True})
        registration = registration.model_copy(update={"q": registered_q, "roster": (cell,)})
        changed = plan.model_copy(
            update={
                "inference": plan.inference.model_copy(update={"registration": registration}),
                "compliance": SequentialCompliancePolicy(
                    alpha=cell.alpha,
                    family=True,
                ),
            }
        )
        with pytest.raises(CapabilityError) as exc:
            compile_decision_plan(changed, source.metrics, design=_uptake_design())
        assert exc.value.code == "sequential.source.invalid"
    finally:
        connection.disconnect()


def test_multiple_uptake_models_refuse_single_compliance_policy():
    from increment.plan import compile_decision_plan

    connection, _, source = _native_fixture("bernoulli", uptake_only=True)
    try:
        plan = source.experiment.plan
        assert plan is not None and plan.inference is not None
        registration = plan.inference.registration
        second_model = registration.models[0].model_copy(update={"metric": "uptake_2"})
        second_cell = registration.roster[0].model_copy(update={"metric": "uptake_2"})
        registration = registration.model_copy(
            update={
                "models": (*registration.models, second_model),
                "roster": (*registration.roster, second_cell),
            }
        )
        changed = plan.model_copy(
            update={"inference": plan.inference.model_copy(update={"registration": registration})}
        )
        with pytest.raises(CapabilityError) as exc:
            compile_decision_plan(changed, source.metrics, design=_uptake_design())
        assert exc.value.code == "sequential.source.invalid"
    finally:
        connection.disconnect()


def test_uptake_column_override_must_match_initial_registration():
    from increment.frame import synthesise_metric
    from increment.semantics.models import AnalysisPlan, InferenceSpec
    from increment.sequential_source import frame_observation_mapping, sequential_definition_id

    connection, _, native = _native_fixture("bernoulli", uptake_only=True)
    try:
        design = _uptake_design()
        specs = [MetricSpec(name="outcome", type="conversion", window_days=2)]
        registered = native.experiment.plan.inference.registration
        reg = SequentialRegistration.model_validate(
            {
                **registered.model_dump(),
                "definitions_id": sequential_definition_id(
                    [synthesise_metric(s) for s in specs],
                    design,
                    transformations=specs,
                    source_mapping=frame_observation_mapping(
                        unit="unit",
                        group="arm",
                        date="day",
                        exposure_date="exposed",
                        uptake="clicked",
                    ),
                ),
            }
        )
        plan = AnalysisPlan(
            inference=InferenceSpec(kind="always_valid", registration=reg),
            compliance=SequentialCompliancePolicy(alpha=reg.roster[0].alpha),
        )
        with pytest.raises(CapabilityError) as exc:
            Analysis.from_unit_panel(
                UnreadFrame(),
                unit="unit",
                group="arm",
                date="day",
                exposure_date="exposed",
                uptake="different_clicked_column",
                metrics=specs,
                design=design,
                experiment_id=reg.source_id,
                plan=plan,
            )
        assert exc.value.code == "sequential.source.invalid"
    finally:
        connection.disconnect()


def test_compliance_shortcut_without_uptake_refuses_before_checkpoint_or_outcomes():
    connection, _, source = _native_fixture("bernoulli")
    try:
        connection.drop_table("events")
        with pytest.raises(CapabilityError) as exc:
            source.run(estimands=("compliance",))
        assert exc.value.code == "sequential.source.invalid"
    finally:
        connection.disconnect()


def test_encouragement_asof_requires_completion_before_legacy_checkpoint():
    from increment.errors import InvalidRequestError

    connection, _, source = _native_fixture("bernoulli", uptake_only=True)
    try:
        with pytest.raises(InvalidRequestError) as exc:
            source.run_asof_lift(estimands=("compliance",))
        assert exc.value.code == "readout.encouragement.asof_completion"
        with pytest.raises(CapabilityError) as missing:
            source.run_asof_lift(estimands=("compliance",), completed_windows_only=True)
        assert missing.value.code == "sequential.continuation.legacy"
    finally:
        connection.disconnect()


def test_compliance_policy_wire_roundtrip_and_mismatch_rejection():
    from increment.decision_wire import compiled_plan_from_dict, compiled_plan_to_dict
    from increment.errors import WireFormatError
    from increment.plan import compile_decision_plan

    connection, _, source = _native_fixture("bernoulli", uptake_only=True)
    try:
        plan = compile_decision_plan(
            source.experiment.plan, source.metrics, design=_uptake_design()
        )
        payload = compiled_plan_to_dict(plan)
        assert compiled_plan_from_dict(payload) == plan
        bad = {**payload, "compliance": None}
        with pytest.raises(WireFormatError) as exc:
            compiled_plan_from_dict(bad)
        assert exc.value.code == "wire.compiled_plan.compliance_mismatch"
    finally:
        connection.disconnect()


@pytest.mark.parametrize("correction", ["bh", "bonferroni"])
def test_breakout_registered_q_mismatch_precedes_missing_checkpoint(correction):
    from increment import MultiplicitySpec
    from tests.test_sequential_public_sources import gaussian_plan

    specs = [MetricSpec(name="outcome", type="conversion", window_days=2)]
    cells = tuple(
        SequentialCell(
            metric="outcome",
            group_id="treatment",
            segment=(("segment", name),),
            family=correction == "bh",
            alpha=Fraction(1, 40),
        )
        for name in ("A", "B")
    )
    requested_q = 0.2
    plan = gaussian_plan(specs, law="bernoulli", cells=cells, date="day", exposure_date="exposed")
    plan = plan.model_copy(
        update={
            "q": requested_q,
            "view_multiplicity": MultiplicitySpec(
                correction=correction, q=requested_q if correction == "bh" else None
            ),
        }
    )
    import pyarrow as pa

    source = Analysis.from_unit_panel(
        pa.table(
            {
                "user_id": ["c", "t"],
                "variant": ["control", "treatment"],
                "day": [0, 0],
                "exposed": [0, 0],
                "segment": ["A", "A"],
                "outcome": [0, 1],
            }
        ),
        unit="user_id",
        group="variant",
        date="day",
        exposure_date="exposed",
        control="control",
        metrics=specs,
        plan=plan,
    )
    with pytest.raises(CapabilityError) as exc:
        source.run_breakout()
    assert exc.value.code == "sequential.source.invalid"


@pytest.mark.parametrize("arms", [("treatment",), ("t0", "t1")])
def test_breakout_bonferroni_allocation_precedes_missing_checkpoint(arms):
    import pyarrow as pa

    from increment import MultiplicitySpec
    from tests.test_sequential_public_sources import gaussian_plan

    specs = [MetricSpec(name="outcome", type="conversion", window_days=2)]
    cells = tuple(
        SequentialCell(
            metric="outcome",
            group_id=arm,
            segment=(("segment", name),),
            alpha=Fraction(1, 20 * len(arms)),
        )
        for arm in arms
        for name in ("A", "B")
    )
    plan = gaussian_plan(specs, law="bernoulli", cells=cells, date="day", exposure_date="exposed")
    plan = plan.model_copy(
        update={
            "primary": "outcome",
            "secondaries": (),
            "view_multiplicity": MultiplicitySpec(correction="bonferroni"),
        }
    )
    source = Analysis.from_unit_panel(
        pa.table(
            {
                "user_id": ["c", "t"],
                "variant": ["control", arms[0]],
                "day": [0, 0],
                "exposed": [0, 0],
                "segment": ["A", "A"],
                "outcome": [0, 1],
            }
        ),
        unit="user_id",
        group="variant",
        date="day",
        exposure_date="exposed",
        control="control",
        metrics=specs,
        plan=plan,
    )
    with pytest.raises(CapabilityError) as exc:
        source.run_breakout()
    assert exc.value.code == "sequential.source.invalid"


def test_corrected_compliance_breakout_retains_pre_read_scope_refusal():
    from increment import MultiplicitySpec
    from increment.errors import UnsupportedRequestError
    from increment.plan import compile_decision_plan

    connection, _, native = _native_fixture("bernoulli", uptake_only=True)
    try:
        plan = native.experiment.plan
        assert plan is not None
        plan = plan.model_copy(
            update={
                "view_multiplicity": MultiplicitySpec(correction="bonferroni"),
            }
        )
        with pytest.raises(UnsupportedRequestError) as exc:
            compile_decision_plan(
                plan,
                native.metrics,
                design=_uptake_design(),
                estimands=("compliance",),
            )
        assert exc.value.code == "plan.corrected_encouragement_breakout"
    finally:
        connection.disconnect()


@pytest.mark.slow
def test_artifact_initial_mapping_rejects_changed_saved_recipe_before_relation_access(
    monkeypatch,
):
    from contextlib import contextmanager

    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore

    connection, definitions, native = _native_fixture("bernoulli")
    relation_reads = []
    try:
        store = WarehouseArtifactStore(connection, schema_name="artifacts")
        ref = native.publish_unit_day_artifact(store)
        changed = definitions.fact_sources[0].model_copy(
            update={"sql": "SELECT * FROM must_never_be_read"}
        )
        definitions = definitions.model_copy(update={"fact_sources": (changed,)})
        wrong = artifact_context(definitions, definitions.experiments[0], "error")
        original_open_snapshot = store.open_snapshot

        class SnapshotWithChangedContext:
            def __init__(self, snapshot):
                self.snapshot = snapshot

            def __getattr__(self, name):
                return getattr(self.snapshot, name)

            def read_manifest(self, *args, **kwargs):
                manifest = self.snapshot.read_manifest(*args, **kwargs)
                return manifest.model_copy(update={"context": wrong})

            def verify_relation(self, relation, *, expected_role):
                relation_reads.append(expected_role)
                raise AssertionError("sequential recipe validation must precede relation reads")

        @contextmanager
        def open_snapshot_with_changed_context(reference):
            with original_open_snapshot(reference) as snapshot:
                yield SnapshotWithChangedContext(snapshot)

        monkeypatch.setattr(store, "open_snapshot", open_snapshot_with_changed_context)
        with pytest.raises(CapabilityError) as exc:
            Analysis.from_unit_day_artifact(store, ref, expected_context=wrong)
        assert exc.value.code == "sequential.source.invalid"
        assert relation_reads == []
    finally:
        connection.disconnect()


def test_automatic_registration_normalizes_weights_and_tunes_before_capture():
    import math

    from increment import AnalysisPlan, InferenceSpec, Randomized

    plan = AnalysisPlan(
        primary="outcome",
        inference=InferenceSpec(kind="asymptotic_mean", expected_decision_sample_size=321),
    )
    analysis = Analysis.from_unit_summary(
        _frame([1, 2, 3], [4, 6, 8]),
        unit="unit",
        group="arm",
        exposure_date="exposure",
        metrics={"outcome": "mean"},
        design=Randomized(control_group="control", allocation={"control": 0.6, "treatment": 0.1}),
        plan=plan,
    )
    registration = analysis.sequential_snapshot().registration
    model = registration.models[0]
    assert model.treatment_probability == Fraction(0.1) / (Fraction(0.6) + Fraction(0.1))
    tuned = 321 * float(model.rho) ** 2
    assert tuned - math.log1p(tuned) == pytest.approx(
        -2 * math.log(float(registration.roster[0].alpha)), rel=1e-12
    )


def test_automatic_family_allocation_preserves_subnormal_tails():
    from increment import AnalysisPlan, InferenceSpec, Randomized

    frame = _frame([1, 2, 3], [4, 6, 8])
    frame["other"] = frame["outcome"] + 1
    q = float.fromhex("0x0.0000000000001p-1022")
    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        exposure_date="exposure",
        metrics={"outcome": "mean", "other": "mean"},
        design=Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5}),
        plan=AnalysisPlan(
            secondaries=("outcome", "other"),
            q=q,
            inference=InferenceSpec(kind="asymptotic_mean"),
        ),
    )
    registration = analysis.sequential_snapshot().registration
    assert all(cell.alpha == Fraction(q) / 2 for cell in registration.roster)
    assert sum(cell.alpha for cell in registration.roster) == Fraction(q)


@pytest.mark.slow
def test_native_and_artifact_capture_include_the_final_window_day():
    from datetime import timedelta

    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore

    connection, definitions, native = _native_fixture("scalar_mean")
    try:
        registration = native.experiment.plan.inference.registration
        final_day = date(2025, 1, 1) + timedelta(days=registration.reveal.longest_window_days - 1)
        before = native.capture_sequential(finalized=True, as_of=final_day - timedelta(days=1))
        assert len(before.records) == 0
        completed = native.capture_sequential(finalized=True, as_of=final_day)
        assert len(completed.records) == 192
        context = artifact_context(definitions, definitions.experiments[0], "error")
        store = WarehouseArtifactStore(connection, schema_name="final_day_artifacts")
        reference = native.publish_unit_day_artifact(store)
        with Analysis.from_unit_day_artifact(store, reference, expected_context=context) as adopted:
            restored = adopted.capture_sequential(finalized=True, as_of=final_day)
            assert restored.records == completed.records
            assert restored.states == completed.states
            assert restored.prefix_id == completed.prefix_id
    finally:
        native.close()
        connection.disconnect()


@pytest.mark.slow
def test_native_registration_freeze_survives_artifact_publish_and_reopen():
    from datetime import timedelta

    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore

    connection, definitions, native = _native_fixture("scalar_mean")
    try:
        registration = native.experiment.plan.inference.registration
        final_day = date(2025, 1, 1) + timedelta(days=registration.reveal.longest_window_days - 1)
        completed = native.capture_sequential(finalized=True, as_of=final_day, freeze=["outcome"])
        assert completed.frozen[0].cell.metric == "outcome"
        context = artifact_context(definitions, definitions.experiments[0], "error")
        store = WarehouseArtifactStore(connection, schema_name="freeze_artifacts")
        reference = native.publish_unit_day_artifact(store)
        with Analysis.from_unit_day_artifact(store, reference, expected_context=context) as adopted:
            restored = adopted.capture_sequential(
                finalized=True, as_of=final_day, previous=completed
            )
            assert restored.frozen == completed.frozen
            assert restored.prefix_id == completed.prefix_id
    finally:
        native.close()
        connection.disconnect()


@pytest.mark.slow
def test_automatic_native_registration_survives_artifact_reopen(tmp_path):
    import yaml

    from increment import Definitions
    from increment.breakout.estimates import LiftEstimate
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore

    connection, definitions, _ = _native_fixture("scalar_mean")
    try:
        payload = definitions.model_dump(mode="json")
        payload["experiments"][0]["allocation"] = {"control": 0.5, "treatment": 0.5}
        payload["experiments"][0]["start"] = "2025-01-01T00:00:00Z"
        payload["experiments"][0]["end"] = "2025-01-20T00:00:00Z"
        payload["experiments"][0]["plan"] = {
            "primary": "outcome",
            "inference": {"kind": "asymptotic_mean"},
        }
        declared = Definitions.model_validate(payload)
        path = tmp_path / "automatic.yaml"
        path.write_text(yaml.safe_dump(declared.model_dump(mode="json")))
        context = artifact_context(declared, declared.experiments[0], "error")
        native = Analysis.from_definitions("experiment", path, con=connection)
        first = native.capture_sequential(finalized=True, as_of=date(2025, 1, 16))
        expected = list(native.run())
        assert isinstance(expected[0], LiftEstimate)
        assert expected[0].stat_sig()
        store = WarehouseArtifactStore(connection, schema_name="automatic_artifacts")
        reference = native.publish_unit_day_artifact(store)
        with Analysis.from_unit_day_artifact(store, reference, expected_context=context) as adopted:
            restored = adopted.capture_sequential(finalized=True, as_of=date(2025, 1, 16))
            assert restored == first
            actual = list(adopted.run())
            assert [
                row.model_dump(exclude={"source_snapshot_id", "family_id"}) for row in actual
            ] == [row.model_dump(exclude={"source_snapshot_id", "family_id"}) for row in expected]
            assert {row.source_snapshot_id for row in actual}.isdisjoint(
                row.source_snapshot_id for row in expected
            )
    finally:
        connection.disconnect()


@pytest.mark.slow
def test_registered_native_context_preserves_legacy_policy_digest():
    import hashlib
    import json
    from datetime import UTC, datetime

    from increment.query.artifact_contract import (
        ARTIFACT_DOMAIN_ROOT,
        compile_unit_day_artifact_context,
    )
    from increment.query.artifact_digest import canonical_json

    connection, definitions, _ = _native_fixture("scalar_mean")
    try:
        experiment = definitions.experiments[0].model_copy(
            update={
                "start": datetime(2025, 1, 1, tzinfo=UTC),
                "end": datetime(2025, 1, 20, tzinfo=UTC),
            }
        )
        definitions = definitions.model_copy(update={"experiments": (experiment,)})
        context = compile_unit_day_artifact_context("experiment", definitions)
        legacy = json.loads(context.canonical_json)
        legacy["experiment"]["plan"]["inference"].pop("expected_decision_sample_size", None)
        legacy_json = canonical_json(legacy)
        legacy_digest = hashlib.sha256(
            ARTIFACT_DOMAIN_ROOT + b"context\x00" + legacy_json.encode()
        ).hexdigest()
        assert context.sha256 == legacy_digest
    finally:
        connection.disconnect()


@pytest.mark.slow
@pytest.mark.parametrize("typed", [False, True])
def test_legacy_sequential_wire_rejected_before_payload_rehydration(typed):
    from increment.decision_wire import (
        WireCompiledDecisionPlan,
        compiled_plan_from_dict,
        compiled_plan_to_dict,
        compiled_plan_to_dto,
    )
    from increment.plan import compile_decision_plan

    connection, _, source = _native_fixture("bernoulli", uptake_only=True)
    try:
        plan = compile_decision_plan(
            source.experiment.plan, source.metrics, design=_uptake_design()
        )
        payload = compiled_plan_to_dict(plan)
        assert payload["wire_version"] == 3
        payload["wire_version"] = 2
        payload["procedures"] = None
        if typed:
            payload["inference"] = compiled_plan_to_dto(plan).inference
        with pytest.raises(CapabilityError) as exc:
            if typed:
                WireCompiledDecisionPlan.model_validate(payload)
            else:
                compiled_plan_from_dict(payload)
        assert exc.value.code == "sequential.continuation.legacy"
    finally:
        connection.disconnect()


@pytest.mark.slow
@pytest.mark.parametrize("cell_alpha", [Fraction(1, 40), Fraction(1, 20)])
def test_same_arm_segmented_compliance_cells_conserve_policy_alpha(cell_alpha):
    from increment import AlwaysValid
    from increment.plan import compile_decision_plan

    connection, _, source = _native_fixture("bernoulli", uptake_only=True)
    try:
        plan = source.experiment.plan
        assert plan is not None and plan.inference is not None
        assert plan.inference.registration is not None
        registration = plan.inference.registration
        cell = registration.roster[0]
        cells = (
            cell.model_copy(update={"segment": (("region", "north"),), "alpha": cell_alpha}),
            cell.model_copy(update={"segment": (("region", "south"),), "alpha": cell_alpha}),
        )
        registration = registration.model_copy(update={"roster": cells})
        changed = plan.model_copy(
            update={
                "inference": plan.inference.model_copy(update={"registration": registration}),
                "compliance": SequentialCompliancePolicy(alpha=Fraction(1, 20)),
            }
        )
        if cell_alpha == Fraction(1, 20):
            with pytest.raises(CapabilityError) as exc:
                compile_decision_plan(changed, source.metrics, design=_uptake_design())
            assert exc.value.code == "sequential.source.invalid"
            return
        compiled = compile_decision_plan(changed, source.metrics, design=_uptake_design())
        assert isinstance(compiled.inference, AlwaysValid)
        retained = compiled.inference.registration.roster
        assert len(retained) == 2
        assert sum((cell.alpha for cell in retained), Fraction(0)) == Fraction(1, 20)
        assert {cell.group_id for cell in retained} == {"treatment"}
    finally:
        connection.disconnect()


@pytest.mark.slow
def test_legacy_sequential_snapshot_rejected_before_record_replay():
    import json

    from increment.sequential_state import snapshot_from_json

    connection, _, source = _native_fixture("bernoulli")
    try:
        snapshot = source.capture_sequential(finalized=True, as_of=date(2025, 1, 16))
        payload = json.loads(snapshot.model_dump_json())
        payload["version"] = 1
        payload["records"] = "invalid legacy records must not be decoded"
        with pytest.raises(CapabilityError) as exc:
            snapshot_from_json(json.dumps(payload))
        assert exc.value.code == "sequential.continuation.legacy"
    finally:
        connection.disconnect()


@pytest.mark.parametrize("kind", ["registration", "snapshot", "checkpoint"])
def test_direct_legacy_declarations_refuse_before_field_validation(kind):
    from types import MappingProxyType

    from increment.estimation.sequential_result import SequentialCheckpoint
    from increment.sequential_state import SequentialSnapshot

    model = {
        "registration": SequentialRegistration,
        "snapshot": SequentialSnapshot,
        "checkpoint": SequentialCheckpoint,
    }[kind]
    with pytest.raises(CapabilityError) as exc:
        model.model_validate(MappingProxyType({"version": 1}))
    assert exc.value.code == "sequential.continuation.legacy"


@pytest.mark.parametrize("typed", [False, True])
def test_fixed_wire_refuses_sequential_version_before_payload_rehydration(typed):
    from increment.decision_wire import (
        WireCompiledDecisionPlan,
        compiled_plan_from_dict,
        compiled_plan_to_dict,
        compiled_plan_to_dto,
    )
    from increment.errors import CodedError
    from increment.plan import compile_decision_plan
    from increment.semantics.design import Randomized

    source = _analysis(
        _frame([0, 1], [0, 1]), [MetricSpec(name="outcome", type="conversion")], None
    )
    plan = compile_decision_plan(
        None, source.metrics, path="frame", design=Randomized(control_group="control")
    )
    payload = compiled_plan_to_dict(plan)
    assert compiled_plan_from_dict(payload) == plan
    payload["wire_version"] = 3
    payload["procedures"] = None
    if typed:
        payload["inference"] = compiled_plan_to_dto(plan).inference
    with pytest.raises(CodedError) as exc:
        if typed:
            WireCompiledDecisionPlan.model_validate(payload)
        else:
            compiled_plan_from_dict(payload)
    assert exc.value.code == "wire.payload.invalid"


@pytest.mark.slow
def test_fixed_moments_sequential_stamp_requires_sequential_envelope(tmp_path):
    import pyarrow.parquet as pq

    from increment import AnalysisPlan
    from tests.analysis_factory import lift_rows, make_analysis

    connection, definitions, _ = _native_fixture("bernoulli")
    try:
        experiment = definitions.experiments[0].model_copy(
            update={"plan": AnalysisPlan(primary="outcome")}
        )
        definitions = definitions.model_copy(
            update={
                "experiments": (experiment,),
                "metrics": tuple(
                    metric.model_copy(update={"window_days": None})
                    for metric in definitions.metrics
                ),
            }
        )
        source = make_analysis(connection, definitions, experiment=experiment)
        path = tmp_path / "fixed.parquet"
        source.export(path)
        payload = pq.read_table(path).to_pylist()
        assert payload and payload[0]["moments_format"] == 12
        supported = Analysis.from_moments(
            payload,
            metrics=[MetricSpec(name="outcome", type="conversion")],
            control="control",
        )
        assert (
            lift_rows(supported.run())[0].require_lift()
            == lift_rows(source.run())[0].require_lift()
        )
        with pytest.raises(CapabilityError) as exc:
            Analysis.from_moments(
                [
                    dict(row, record_kind="sequential_checkpoint", moments_format=9)
                    for row in payload
                ],
                metrics=[MetricSpec(name="outcome", type="conversion")],
                control="control",
            )
        assert exc.value.code == "sequential.source.invalid"
    finally:
        connection.disconnect()


@pytest.mark.slow
def test_sequential_moments_format8_refuses_before_replay(tmp_path):
    import pyarrow.parquet as pq

    connection, _, source = _native_fixture("bernoulli")
    try:
        source.capture_sequential(finalized=True, as_of=date(2025, 1, 16))
        path = tmp_path / "sequential.parquet"
        source.export(path)
        payload = pq.read_table(path).to_pylist()
        assert payload and payload[0]["moments_format"] == 10
        supported = Analysis.from_moments(
            payload,
            metrics=[MetricSpec(name="outcome", type="conversion")],
            control="control",
        )
        assert supported.sequential_snapshot() == source.sequential_snapshot()
        with pytest.raises(CapabilityError) as exc:
            Analysis.from_moments(
                [dict(row, moments_format=8) for row in payload],
                metrics=[MetricSpec(name="outcome", type="conversion")],
                control="control",
            )
        assert exc.value.code == "sequential.continuation.legacy"
    finally:
        connection.disconnect()


# -- Automatic exact multi-arm and predeclared frame segments: the manual
# registration is the oracle; the automatic plan must construct its form.


def _arm_rows(arms, *, n=60, offset=0, levels=None):
    rows = []
    for i in range(n):
        for k, arm in enumerate(arms):
            unit = offset + i
            rows.append(
                {
                    "unit": f"{unit:05d}-{arm}",
                    "arm": arm,
                    "outcome": int((unit * 7 + 3 * k) % 10 < 3 + 2 * k),
                    "other": int((unit * 5 + k) % 10 < 4 + k),
                    "exposure": unit,
                    "segment": levels[unit % len(levels)] if levels else "all",
                }
            )
    import pandas as pd

    return pd.DataFrame(rows)


def _summary(frame, specs, plan, design):
    return Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        exposure_date="exposure",
        metrics=specs,
        design=design,
        experiment_id="experiment",
        plan=plan,
    )


def _explicit_bernoulli(specs, roster, *, design, q=Fraction(0.1), date=None):
    """A caller-written exact registration over these cells, independent of the builder."""
    from increment import JointReveal, SequentialModel
    from increment.frame import synthesise_metric
    from increment.sequential_source import (
        bernoulli_prior,
        frame_observation_mapping,
        sequential_definition_id,
    )
    from increment.sequential_state import canonical_id

    prior = bernoulli_prior(None)
    binding = sequential_definition_id(
        [synthesise_metric(s) for s in specs],
        design,
        transformations=specs,
        source_mapping=frame_observation_mapping(
            unit="unit", group="arm", date=date, exposure_date="exposed" if date else "exposure"
        ),
    )
    return SequentialRegistration(
        source_id="experiment",
        definitions_id=binding,
        control_group="control",
        committed_before_data=True,
        reveal=JointReveal(
            filtration_id=canonical_id({"binding": binding, "reveal": "joint_units_v1"}),
            independent_unit_vectors=True,
            simultaneous_metrics=True,
            outcome_independent_order=True,
            immutable_finalized_outcomes=True,
            longest_window_days=max((s.window_days or 0 for s in specs), default=0),
        ),
        models=tuple(
            SequentialModel(
                metric=s.name,
                law="bernoulli",
                control_prior=prior,
                treatment_prior=prior,
                positive_population_control=True,
            )
            for s in specs
        ),
        roster=roster,
        q=q,
    )


def _rows(readout):
    return [row.model_dump() for row in readout]


@pytest.mark.slow
def test_automatic_exact_multi_arm_matches_the_manual_registration_end_to_end(tmp_path):
    import pyarrow.parquet as pq

    from increment.semantics.design import Randomized
    from increment.semantics.models import AnalysisPlan, InferenceSpec

    design = Randomized(control_group="control", allocation={"control": 0.5, "a": 0.25, "b": 0.25})
    specs = [
        MetricSpec(name="outcome", type="conversion"),
        MetricSpec(name="other", type="conversion"),
    ]
    automatic = AnalysisPlan(
        primary="outcome", secondaries=["other"], inference=InferenceSpec(kind="always_valid")
    )
    roster = tuple(
        SequentialCell(
            metric=metric,
            group_id=arm,
            alpha=Fraction(0.05) / 2 if metric == "outcome" else Fraction(0.05),
            family=metric == "other",
        )
        for metric in ("outcome", "other")
        for arm in ("a", "b")
    )
    manual = automatic.model_copy(
        update={
            "inference": InferenceSpec(
                kind="always_valid",
                registration=_explicit_bernoulli(specs, roster, design=design),
            )
        }
    )
    arms = ("control", "a", "b")
    first = _arm_rows(arms)
    auto_source = _summary(first, specs, automatic, design)
    manual_source = _summary(first, specs, manual, design)
    snapshot = auto_source.sequential_snapshot()
    assert snapshot == manual_source.sequential_snapshot()
    rows = auto_source.run()
    assert _rows(rows) == _rows(manual_source.run())
    assert {(r.metric, r.group_id, r.role) for r in rows} == {
        (metric, arm, role)
        for metric, role in (("outcome", "primary"), ("other", "secondary"))
        for arm in ("a", "b")
    }
    assert {r.discovery for r in rows if r.metric == "other"} <= {True, False}
    assert all(r.discovery is None for r in rows if r.metric == "outcome")
    assert all(r.require_sequential_result().checkpoint.status == "current" for r in rows)

    # Idempotent replay, then an appended prefix, agree with the manual form.
    assert auto_source.capture_sequential(finalized=True, previous=snapshot) == snapshot
    import pandas as pd

    later = pd.concat([first, _arm_rows(arms, offset=60)], ignore_index=True)
    auto_later = _summary(later, specs, automatic, design)
    manual_later = _summary(later, specs, manual, design)
    appended = auto_later.capture_sequential(finalized=True, previous=snapshot)
    assert appended.parent_id == snapshot.prefix_id
    assert appended == manual_later.capture_sequential(finalized=True, previous=snapshot)
    assert _rows(auto_later.run()) == _rows(manual_later.run())

    # Export/import keeps every arm's cell and state.
    path = tmp_path / "multi-arm.parquet"
    auto_later.export(path)
    replay = Analysis.from_moments(
        pq.read_table(path).to_pylist(), metrics=specs, control="control"
    )
    assert replay.sequential_snapshot() == appended
    assert _rows(replay.run()) == _rows(auto_later.run())

    # A changed roster, assignment or prefix refuses to continue.
    with pytest.raises(CapabilityError) as roster_changed:
        _summary(later, specs, automatic.model_copy(update={"q": 0.2}), design).capture_sequential(
            finalized=True, previous=snapshot
        )
    assert roster_changed.value.code == "sequential.continuation.rewrite"
    grown = Randomized(
        control_group="control", allocation={"control": 0.5, "a": 0.2, "b": 0.2, "c": 0.1}
    )
    with pytest.raises(CapabilityError) as assignment_changed:
        _summary(later, specs, automatic, grown).capture_sequential(
            finalized=True, previous=snapshot
        )
    assert assignment_changed.value.code == "sequential.continuation.rewrite"
    rewritten = later.copy()
    rewritten.loc[0, "outcome"] = 1 - rewritten.loc[0, "outcome"]
    with pytest.raises(CapabilityError) as prefix_changed:
        _summary(rewritten, specs, automatic, design).capture_sequential(
            finalized=True, previous=snapshot
        )
    assert prefix_changed.value.code == "sequential.continuation.rewrite"
    with pytest.raises(CapabilityError) as unregistered_arm:
        _summary(_arm_rows(("control", "a", "c")), specs, automatic, design)
    assert unregistered_arm.value.code == "sequential.source.invalid"


@pytest.mark.slow
def test_automatic_segments_match_the_manual_breakout_and_retain_absent_levels(tmp_path):
    import pyarrow.parquet as pq

    from increment.semantics.design import Randomized
    from increment.semantics.models import AnalysisPlan, InferenceSpec

    design = Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5})
    specs = [MetricSpec(name="outcome", type="conversion")]
    levels = ("x", "y", "absent")
    automatic = AnalysisPlan(
        primary="outcome",
        inference=InferenceSpec(kind="always_valid", segments={"segment": levels}),
    )
    roster = tuple(
        SequentialCell(
            metric="outcome",
            group_id="treatment",
            segment=(("segment", level),),
            family=True,
            alpha=Fraction(0.1) / 3,
        )
        for level in levels
    )
    manual = AnalysisPlan(
        primary="outcome",
        inference=InferenceSpec(
            kind="always_valid", registration=_explicit_bernoulli(specs, roster, design=design)
        ),
    )
    frame = _arm_rows(("control", "treatment"), levels=("x", "y"))
    auto_source = _summary(frame, specs, automatic, design)
    manual_source = _summary(frame, specs, manual, design)
    snapshot = auto_source.sequential_snapshot()
    assert snapshot == manual_source.sequential_snapshot()
    assert {(state.segment, state.n > 0) for state in snapshot.states} == {
        ((("segment", "x"),), True),
        ((("segment", "y"),), True),
        ((("segment", "absent"),), False),
    }
    rows = auto_source.run_breakout()
    assert _rows(rows) == _rows(manual_source.run_breakout())
    by_level = {row.dimension_value: row for row in rows}
    assert set(by_level) == set(levels)
    absent = by_level["absent"]
    assert absent.discovery is False
    assert absent.sequential_result is not None
    assert absent.sequential_result.checkpoint.status == "missing"
    assert all(row.family_axes == ("metric", "arm", "segment") for row in rows)

    # Whole-window and as-of views keep their existing segmented refusals.
    for source in (auto_source, manual_source):
        with pytest.raises(CapabilityError) as refused:
            source.run()
        assert refused.value.code == "sequential.source.invalid"

    # A replayed checkpoint retains the segmented state; a moments cube still
    # declares no breakout dimension.
    path = tmp_path / "segments.parquet"
    auto_source.export(path)
    replay = Analysis.from_moments(
        pq.read_table(path).to_pylist(), metrics=specs, control="control"
    )
    assert replay.sequential_snapshot() == snapshot
    with pytest.raises(CapabilityError) as no_catalog:
        replay.run_breakout()
    assert no_catalog.value.code == "readout.source.dimension"

    # An undeclared level in the data joins no cell; a redeclared family refuses.
    extra = _arm_rows(("control", "treatment"), levels=("x", "y", "z"), offset=60)
    assert auto_source.capture_sequential(finalized=True, previous=snapshot) == snapshot
    import pandas as pd

    grown = _summary(pd.concat([frame, extra], ignore_index=True), specs, automatic, design)
    appended = grown.capture_sequential(finalized=True, previous=snapshot)
    assert appended.parent_id == snapshot.prefix_id
    assert {state.segment[0][1] for state in appended.states} == set(levels)
    redeclared = automatic.model_copy(
        update={"inference": InferenceSpec(kind="always_valid", segments={"segment": ("x", "y")})}
    )
    with pytest.raises(CapabilityError) as family_changed:
        _summary(frame, specs, redeclared, design).capture_sequential(
            finalized=True, previous=snapshot
        )
    assert family_changed.value.code == "sequential.continuation.rewrite"


@pytest.mark.slow
def test_direct_frame_totals_construction_reads_registered_segments_like_from_frame():
    import narwhals as nw

    from increment.frame import FrameTotalsSource, from_unit_summary
    from increment.semantics.design import Randomized
    from increment.semantics.models import AnalysisPlan, InferenceSpec

    design = Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5})
    specs = [MetricSpec(name="outcome", type="conversion")]
    levels = ("x", "y")
    plan = AnalysisPlan(
        primary="outcome",
        inference=InferenceSpec(kind="always_valid", segments={"segment": levels}),
    )
    frame = _arm_rows(("control", "treatment"), levels=levels)
    built = from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        control="control",
        metrics=specs,
        design=design,
        experiment_id="experiment",
        plan=plan,
        exposure_date="exposure",
    )
    direct = FrameTotalsSource(
        frame=nw.from_native(frame, eager_only=True),
        unit="unit",
        group="arm",
        moments=[],
        metrics=specs,
        control="control",
        experiment_id="experiment",
        design=design,
        plan=built.plan,
        exposure_date="exposure",
        synthesised_metrics=built.context.metrics,
    )
    snapshot = built.sequential_snapshot()
    assert direct.sequential_snapshot() == snapshot
    assert {state.segment[0][1] for state in snapshot.states} == set(levels)


@pytest.mark.slow
def test_direct_frame_totals_refuses_a_registered_segment_column_absent_from_the_frame():
    import narwhals as nw

    from increment.frame import FrameTotalsSource, from_unit_summary
    from increment.semantics.design import Randomized
    from increment.semantics.models import AnalysisPlan, InferenceSpec

    design = Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5})
    specs = [MetricSpec(name="outcome", type="conversion")]
    plan = AnalysisPlan(
        primary="outcome",
        inference=InferenceSpec(kind="always_valid", segments={"segment": ("x", "y")}),
    )
    frame = _arm_rows(("control", "treatment"), levels=("x", "y"))
    built = from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        control="control",
        metrics=specs,
        design=design,
        experiment_id="experiment",
        plan=plan,
        exposure_date="exposure",
    )
    with pytest.raises(InvalidRequestError) as direct:
        FrameTotalsSource(
            frame=nw.from_native(frame.drop(columns=["segment"]), eager_only=True),
            unit="unit",
            group="arm",
            moments=[],
            metrics=specs,
            control="control",
            experiment_id="experiment",
            design=design,
            plan=built.plan,
            exposure_date="exposure",
            synthesised_metrics=built.context.metrics,
        )
    # The public factory validates the same column before projection.
    with pytest.raises(InvalidRequestError) as factory:
        _summary(frame.drop(columns=["segment"]), specs, plan, design)
    assert factory.value.code == "source.frame.metric_missing"
    assert direct.value.code == factory.value.code
    assert direct.value.context.keys() == factory.value.context.keys()
    assert direct.value.context["missing"] == factory.value.context["missing"]


@pytest.mark.slow
def test_automatic_segments_on_a_unit_panel_match_the_manual_registration():
    import pyarrow as pa

    from increment.semantics.design import Randomized
    from increment.semantics.models import AnalysisPlan, InferenceSpec

    design = Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5})
    specs = [MetricSpec(name="outcome", type="conversion", window_days=2)]
    levels = ("x", "y")
    automatic = AnalysisPlan(
        primary="outcome",
        inference=InferenceSpec(kind="always_valid", segments={"segment": levels}),
    )
    roster = tuple(
        SequentialCell(
            metric="outcome",
            group_id="treatment",
            segment=(("segment", level),),
            family=True,
            alpha=Fraction(0.05),
        )
        for level in levels
    )
    manual = AnalysisPlan(
        primary="outcome",
        inference=InferenceSpec(
            kind="always_valid",
            registration=_explicit_bernoulli(specs, roster, design=design, date="day"),
        ),
    )
    rows = []
    for i in range(40):
        for arm in ("control", "treatment"):
            for day in range(3):
                rows.append(
                    {
                        "unit": f"{i:04d}-{arm}",
                        "arm": arm,
                        "day": day,
                        "exposed": 0,
                        "segment": levels[i % 2],
                        "outcome": int(
                            day == 1 and (i * 7 + (5 if arm == "treatment" else 0)) % 10 < 5
                        ),
                    }
                )
    table = pa.Table.from_pylist(rows)

    def panel(plan):
        source = Analysis.from_unit_panel(
            table,
            unit="unit",
            group="arm",
            date="day",
            exposure_date="exposed",
            design=design,
            metrics=specs,
            experiment_id="experiment",
            plan=plan,
            observation_end=10,
        )
        source.capture_sequential(finalized=True, as_of=2)
        return source

    auto_source, manual_source = panel(automatic), panel(manual)
    assert auto_source.sequential_snapshot() == manual_source.sequential_snapshot()
    assert _rows(auto_source.run_breakout()) == _rows(manual_source.run_breakout())
    assert {row.dimension_value for row in auto_source.run_breakout()} == set(levels)


def test_automatic_segments_under_an_explicit_bh_view_read_out_at_the_view_level():
    """A declared BH breakout view selects at its own q, and the readout requires
    the registration to carry that level: the automatic roster is registered
    at the view's q and reads out exactly like the manual form written there,
    where a registration left at the plan's q would refuse at the family."""
    from increment import MultiplicitySpec
    from increment.semantics.design import Randomized
    from increment.semantics.models import AnalysisPlan, InferenceSpec

    design = Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5})
    specs = [MetricSpec(name="outcome", type="conversion")]
    levels = ("x", "y")
    automatic = AnalysisPlan(
        primary="outcome",
        view_multiplicity=MultiplicitySpec(correction="bh", q=0.3),
        inference=InferenceSpec(kind="always_valid", segments={"segment": levels}),
    )
    roster = tuple(
        SequentialCell(
            metric="outcome",
            group_id="treatment",
            segment=(("segment", level),),
            family=True,
            alpha=Fraction(0.05),
        )
        for level in levels
    )
    manual = automatic.model_copy(
        update={
            "inference": InferenceSpec(
                kind="always_valid",
                registration=_explicit_bernoulli(specs, roster, design=design, q=Fraction(0.3)),
            )
        }
    )
    frame = _arm_rows(("control", "treatment"), levels=levels)
    auto_source = _summary(frame, specs, automatic, design)
    manual_source = _summary(frame, specs, manual, design)
    assert auto_source.sequential_snapshot() == manual_source.sequential_snapshot()
    assert auto_source.sequential_snapshot().registration.q == Fraction(0.3)
    rows = auto_source.run_breakout()
    assert _rows(rows) == _rows(manual_source.run_breakout())
    assert {row.family_q for row in rows if row.method_role == "decision"} == {0.3}
    plan_level = automatic.model_copy(
        update={
            "inference": InferenceSpec(
                kind="always_valid",
                registration=_explicit_bernoulli(specs, roster, design=design),
            )
        }
    )
    with pytest.raises(CapabilityError) as refused:
        _summary(frame, specs, plan_level, design).run_breakout()
    assert refused.value.code == "sequential.source.invalid"


def test_automatic_segments_under_encouragement_keep_the_breakout_estimand_refusal():
    """The manual segmented form under an encouragement design constructs and then
    refuses at run_breakout for want of a declared estimand; the automatic form
    must refuse there too, with the same reason, not earlier with a new one."""
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec
    from increment.semantics.models import AnalysisPlan, InferenceSpec

    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="inert encouragement"
        ),
        allocation={"control": 0.5, "treatment": 0.5},
    )
    specs = [MetricSpec(name="outcome", type="conversion")]
    automatic = AnalysisPlan(
        primary="outcome",
        inference=InferenceSpec(kind="always_valid", segments={"segment": ("x", "y")}),
        compliance=SequentialCompliancePolicy(alpha=Fraction(1, 20)),
    )
    frame = _arm_rows(("control", "treatment"), levels=("x", "y"))
    frame["clicked"] = [int(arm != "control" and i % 3 != 0) for i, arm in enumerate(frame["arm"])]
    source = _summary(frame, specs, automatic, design)
    roster = source.sequential_snapshot().registration.roster
    assert {(c.metric, c.estimand, c.segment[0][1]) for c in roster} == {
        (metric, estimand, level)
        for metric, estimand in (("outcome", "itt"), ("uptake", "compliance"))
        for level in ("x", "y")
    }
    assert all(not c.family for c in roster)
    assert {c.alpha for c in roster if c.estimand == "compliance"} == {Fraction(1, 40)}
    with pytest.raises(CapabilityError) as refused:
        source.run_breakout()
    assert refused.value.code == "sequential.route.unsupported"


@pytest.mark.slow
def test_automatic_exact_multi_arm_on_definitions_matches_the_manual_plan():
    from increment.plan import bind_automatic_sequential_plan
    from increment.semantics.design import Randomized
    from increment.semantics.models import AnalysisPlan, Definitions, InferenceSpec
    from increment.sequential_source import native_observation_mapping
    from tests.analysis_factory import make_analysis
    from tests.parity_harness import dataset as ds

    plan = AnalysisPlan(primary="purchase_rate", inference=InferenceSpec(kind="always_valid"))
    definition = ds.multiplicity_definitions_dict(plan=plan)
    definition["metrics"] = [m for m in definition["metrics"] if m["name"] == "purchase_rate"]
    defs = Definitions.model_validate(definition)
    experiment = defs.experiment("exp")
    assert experiment is not None
    design = Randomized(control_group="control", allocation=ds.multiplicity_allocation())
    metrics = [m for m in defs.metrics if m.name in experiment.metric_names]
    mapping = native_observation_mapping(defs, experiment, on_mixed_assignment="error")
    bound = bind_automatic_sequential_plan(
        experiment.plan, metrics, design=design, source_id="exp", source_mapping=mapping
    )
    assert bound is not None and bound.inference is not None
    registration = bound.inference.registration
    assert registration is not None
    assert {(c.group_id, c.alpha) for c in registration.roster} == {
        ("treatment_a", Fraction(0.05) / 2),
        ("treatment_b", Fraction(0.05) / 2),
    }
    manual = plan.model_copy(
        update={"inference": InferenceSpec(kind="always_valid", registration=registration)}
    )
    as_of = ds._EXPERIMENT_END - timedelta(days=1)
    connection = ds.duckdb_connection(ds.multiplicity_event_rows())
    try:
        automatic_source = make_analysis(connection, defs, experiment="exp", plan=bound)
        manual_source = make_analysis(connection, defs, experiment="exp", plan=manual)
        snapshot = automatic_source.capture_sequential(finalized=True, as_of=as_of)
        assert snapshot == manual_source.capture_sequential(finalized=True, as_of=as_of)
        assert len(snapshot.records) == 240
        rows = automatic_source.run()
        assert _rows(rows) == _rows(manual_source.run())
        assert {r.group_id for r in rows if isinstance(r, LiftEstimate)} == {
            "treatment_a",
            "treatment_b",
        }
        segmented = bind_automatic_sequential_plan(
            plan.model_copy(
                update={
                    "inference": InferenceSpec(
                        kind="always_valid", segments={"store": ("s0", "s1")}
                    )
                }
            ),
            metrics,
            design=design,
            source_id="exp",
            source_mapping=mapping,
        )
        relational = make_analysis(connection, defs, experiment="exp", plan=segmented)
        with pytest.raises(CapabilityError) as refused:
            relational.capture_sequential(finalized=True, as_of=as_of)
        assert refused.value.code == "sequential.route.unsupported"
    finally:
        connection.disconnect()


# -- Typed Boolean/null segment labels: the canonical-string frame is the oracle.

_BOOL_LABELS = {"true": True, "false": False, "__null__": None}
_BOOL_KINDS = ("pandas", "polars", "arrow")


def typed_segment_frame(frame, kind, *, column="segment"):
    """*frame* with *column* rebuilt as the nullable Boolean column its canonical strings name."""
    import pandas as pd

    values = [_BOOL_LABELS[label] for label in frame[column]]
    base = frame.drop(columns=column)
    if kind == "pandas":
        return base.assign(**{column: pd.array(values, dtype="boolean")})
    if kind == "polars":
        import polars as pl

        return pl.from_pandas(base).with_columns(pl.Series(column, values, dtype=pl.Boolean))
    import pyarrow as pa

    return pa.Table.from_pandas(base, preserve_index=False).append_column(
        column, pa.array(values, pa.bool_())
    )


def _occupancy(snapshot):
    return {(state.segment[0][1], state.n > 0) for state in snapshot.states}


@pytest.mark.parametrize("kind", _BOOL_KINDS)
@pytest.mark.parametrize("registered", ["automatic", "explicit"])
def test_boolean_and_null_summary_segments_match_the_canonical_string_oracle(kind, registered):
    from increment.semantics.design import Randomized
    from increment.semantics.models import AnalysisPlan, InferenceSpec

    design = Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5})
    specs = [MetricSpec(name="outcome", type="conversion")]
    levels = ("true", "false", "__null__", "absent")
    if registered == "automatic":
        plan = AnalysisPlan(
            primary="outcome",
            inference=InferenceSpec(kind="always_valid", segments={"segment": levels}),
        )
    else:
        roster = tuple(
            SequentialCell(
                metric="outcome",
                group_id="treatment",
                segment=(("segment", level),),
                family=True,
                alpha=Fraction(0.1) / len(levels),
            )
            for level in levels
        )
        plan = AnalysisPlan(
            primary="outcome",
            inference=InferenceSpec(
                kind="always_valid",
                registration=_explicit_bernoulli(specs, roster, design=design),
            ),
        )
    canonical = _arm_rows(("control", "treatment"), levels=levels[:3])
    oracle = _summary(canonical, specs, plan, design)
    typed = _summary(typed_segment_frame(canonical, kind), specs, plan, design)

    snapshot = typed.sequential_snapshot()
    assert snapshot == oracle.sequential_snapshot()
    assert _occupancy(snapshot) == {(level, level != "absent") for level in levels}
    rows = typed.run_breakout()
    assert _rows(rows) == _rows(oracle.run_breakout())
    by_level = {row.dimension_value: row for row in rows}
    assert set(by_level) == set(levels)
    for level, row in by_level.items():
        status = row.sequential_result.checkpoint.status
        assert (status == "missing") is (level == "absent")


@pytest.mark.parametrize("kind", _BOOL_KINDS)
def test_boolean_and_null_panel_segments_match_the_canonical_string_oracle(kind):
    import pandas as pd

    from increment.semantics.design import Randomized
    from increment.semantics.models import AnalysisPlan, InferenceSpec

    design = Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5})
    specs = [MetricSpec(name="outcome", type="conversion", window_days=2)]
    levels = ("true", "false", "__null__", "absent")
    plan = AnalysisPlan(
        primary="outcome",
        inference=InferenceSpec(kind="always_valid", segments={"segment": levels}),
    )
    canonical = pd.DataFrame(
        [
            {
                "unit": f"{i:04d}-{arm}",
                "arm": arm,
                "day": day,
                "exposed": 0,
                "segment": levels[i % 3],
                "outcome": int(day == 1 and (i * 7 + (5 if arm == "treatment" else 0)) % 10 < 5),
            }
            for i in range(60)
            for arm in ("control", "treatment")
            for day in range(3)
        ]
    )

    def capture(frame):
        source = Analysis.from_unit_panel(
            frame,
            unit="unit",
            group="arm",
            date="day",
            exposure_date="exposed",
            design=design,
            metrics=specs,
            experiment_id="experiment",
            plan=plan,
            observation_end=10,
        )
        source.capture_sequential(finalized=True, as_of=2)
        return source

    oracle = capture(canonical)
    typed = capture(typed_segment_frame(canonical, kind))
    snapshot = typed.sequential_snapshot()
    assert snapshot == oracle.sequential_snapshot()
    assert _occupancy(snapshot) == {(level, level != "absent") for level in levels}
    assert _rows(typed.run_breakout()) == _rows(oracle.run_breakout())


def test_segment_named_like_a_role_column_keeps_every_role_intact():
    """A registered segment that is also the group column labels units by its own values
    and leaves the group column intact."""
    from increment.semantics.design import Randomized
    from increment.semantics.models import AnalysisPlan, InferenceSpec

    design = Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5})
    specs = [MetricSpec(name="outcome", type="conversion")]
    frame = _arm_rows(("control", "treatment"))
    oracle_frame = frame.assign(seg=frame["arm"])
    levels = ("control", "treatment")
    oracle = _summary(
        oracle_frame,
        specs,
        AnalysisPlan(
            primary="outcome",
            inference=InferenceSpec(kind="always_valid", segments={"seg": levels}),
        ),
        design,
    )
    overlapped = _summary(
        frame,
        specs,
        AnalysisPlan(
            primary="outcome",
            inference=InferenceSpec(kind="always_valid", segments={"arm": levels}),
        ),
        design,
    )
    key = lambda s: (s.group_id, s.segment[0][1], s.n, s.mean)  # noqa: E731
    assert sorted(map(key, overlapped.sequential_snapshot().states)) == sorted(
        map(key, oracle.sequential_snapshot().states)
    )


def test_pre_normalization_bool_null_prefix_replays_but_cannot_be_relabelled_forward():
    """The frozen prefix recorded raw ``True``/``None`` spellings; it replays unchanged, an
    identical string prefix still continues, and canonical labels never continue it."""
    import json
    from pathlib import Path

    import pandas as pd

    from increment.semantics.design import Randomized
    from increment.semantics.models import AnalysisPlan, InferenceSpec

    rows = json.loads(
        (
            Path(__file__).parent / "fixtures" / "sequential_bool_null_prefix_prerelease.json"
        ).read_text()
    )
    frozen = rows[0]["sequential_snapshot"]
    specs = [MetricSpec(name="outcome", type="conversion")]
    design = Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5})
    old = ("True", "False", "None")
    plan = AnalysisPlan(
        primary="outcome",
        inference=InferenceSpec(
            kind="always_valid", segments={"segment": (*old, "true", "false", "__null__")}
        ),
    )

    replay = Analysis.from_moments(rows, metrics=specs, control="control")
    assert replay.sequential_snapshot().model_dump_json() == frozen
    previous = replay.sequential_snapshot()

    spelled = _arm_rows(("control", "treatment"), levels=old)
    captured = _summary(spelled, specs, plan, design).sequential_snapshot()
    assert captured.records == previous.records
    assert captured.states == previous.states
    assert captured.assignment_counts == {"control": 60, "treatment": 60}
    later = pd.concat([spelled, _arm_rows(("control", "treatment"), offset=60, levels=old)])
    appended = _summary(later, specs, plan, design).capture_sequential(
        finalized=True, previous=previous
    )
    assert appended.parent_id == previous.prefix_id

    canonical = _arm_rows(("control", "treatment"), levels=("true", "false", "__null__"))
    for kind in _BOOL_KINDS:
        relabelled = _summary(typed_segment_frame(canonical, kind), specs, plan, design)
        with pytest.raises(CapabilityError) as refused:
            relabelled.capture_sequential(finalized=True, previous=previous)
        assert refused.value.code == "sequential.continuation.rewrite"


@pytest.mark.slow
def test_registered_native_and_artifact_capture_share_source_identity():
    from increment.errors import CodedError
    from increment.plan import bind_automatic_sequential_plan
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.design import Randomized
    from increment.semantics.models import AnalysisPlan, Definitions, InferenceSpec
    from increment.sequential_source import native_observation_mapping
    from tests.analysis_factory import make_analysis
    from tests.parity_harness import dataset as ds

    plan = AnalysisPlan(primary="purchase_rate", inference=InferenceSpec(kind="always_valid"))
    definition = ds.multiplicity_definitions_dict(plan=plan)
    definition["metrics"] = [m for m in definition["metrics"] if m["name"] == "purchase_rate"]
    defs = Definitions.model_validate(definition)
    experiment = defs.experiment("exp")
    assert experiment is not None
    design = Randomized(control_group="control", allocation=ds.multiplicity_allocation())
    metrics = [m for m in defs.metrics if m.name in experiment.metric_names]
    mapping = native_observation_mapping(defs, experiment, on_mixed_assignment="error")
    assert set(mapping) == {"source_mapping_format", "recipe_sha256"}
    bound = bind_automatic_sequential_plan(
        experiment.plan, metrics, design=design, source_id="exp", source_mapping=mapping
    )
    as_of = ds._EXPERIMENT_END - timedelta(days=1)
    connection = ds.duckdb_connection(ds.multiplicity_event_rows())
    try:
        native = make_analysis(connection, defs, experiment="exp", plan=bound)
        store = WarehouseArtifactStore(connection, schema_name="artifacts")  # ty: ignore[invalid-argument-type]
        context = artifact_context(defs, experiment, "error")
        ref = native.publish_unit_day_artifact(store)
        adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
        native_snapshot = native.capture_sequential(finalized=True, as_of=as_of)
        artifact_snapshot = adopted.capture_sequential(finalized=True, as_of=as_of)
        # prefix_id/records order depends on accumulation order; identity and state do not.
        assert native_snapshot.registration == artifact_snapshot.registration
        assert native_snapshot.states == artifact_snapshot.states
        assert len(native_snapshot.records) == len(artifact_snapshot.records)

        raw = {"definitions": defs.model_dump(include={"dialect"}), "on_mixed_assignment": "error"}
        legacy = bind_automatic_sequential_plan(
            experiment.plan, metrics, design=design, source_id="exp", source_mapping=raw
        )
        with pytest.raises(CodedError) as refused:
            make_analysis(connection, defs, experiment="exp", plan=legacy)
        assert refused.value.code == "sequential.source.invalid"
    finally:
        connection.disconnect()


def test_a_registration_from_before_window_days_were_bound_refuses_new_native_capture():
    """The frozen registration hashed a source recipe without the derived window days, which
    cannot say which window the data was computed under, so it never continues."""
    import json
    from pathlib import Path

    from increment.errors import CodedError
    from increment.semantics.models import AnalysisPlan, Definitions
    from increment.sequential_source import native_observation_mapping
    from tests.analysis_factory import make_analysis
    from tests.parity_harness import dataset as ds

    frozen = json.loads(
        (
            Path(__file__).parent / "fixtures" / "unit_day_artifact_legacy_window_identities.json"
        ).read_text()
    )["native_registration"]
    defs = Definitions.model_validate(frozen["definitions"])
    experiment = defs.experiment("exp")
    assert experiment is not None
    legacy_plan = AnalysisPlan.model_validate(frozen["plan"])
    assert native_observation_mapping(defs, experiment) != frozen["source_mapping"]
    connection = ds.duckdb_connection(ds.multiplicity_event_rows())
    try:
        with pytest.raises(CodedError) as refused:
            make_analysis(connection, defs, experiment="exp", plan=legacy_plan)
        assert refused.value.code == "sequential.source.invalid"
    finally:
        connection.disconnect()
