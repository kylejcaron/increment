from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
import warnings
from dataclasses import replace
from typing import Any, cast

import pandas as pd
import pytest

from increment import Analysis, MetricSpec, Randomized, RunResult, Winsorization, run
from increment.builders import from_definitions, from_frame, from_moments
from increment.errors import CodedError, IncrementWarning
from increment.kernel.contracts import Registration, Scope
from increment.kernel.experiment import Experiment, SourceReference
from increment.kernel.resolver import runnable
from increment.workflows import export_moments, materialize
from tests.analysis_factory import replace_bound_source, view_readout
from tests.test_evidence_identity_parity import _capture
from tests.warning_codes import warning_codes


@pytest.mark.parametrize("winsorization", [None, Winsorization(upper_percentile=0.9)])
def test_moments_builder_preserves_estimates_and_raw_outcome_refusals(tmp_path, winsorization):
    import pyarrow.parquet as pq

    metric = MetricSpec(
        name="value",
        type="mean",
        winsorization=winsorization,
    )
    design = Randomized(control_group="control")
    summary_experiment, summary_resolver = from_frame(
        _frame().assign(value=[1.0, 3.0, 2.0, 4.0] * 10),
        shape="summary",
        unit="unit",
        group="arm",
        metrics=[metric],
        design=design,
    )
    path = tmp_path / "moments.parquet"
    export_moments(summary_experiment, summary_resolver, path)
    rows = pq.read_table(path).to_pylist()

    experiment, resolver = from_moments(
        rows,
        metrics=[metric],
        design=design,
    )
    from tests.test_metric_evidence import _CountingSource

    counted = _CountingSource(resolver.binding.source)
    replace_bound_source(resolver, counted)

    restored = Experiment.from_json(experiment.to_json())
    assert restored.hash() == experiment.hash()
    refusals = runnable(restored, "value")["randomized.unadjusted"].static_refusals
    if winsorization is not None:
        assert tuple(refusal.code for refusal in refusals) == ("evidence.raw_outcomes_unavailable",)
        with pytest.raises(CodedError) as raised:
            run(restored, resolver)
        assert raised.value.code == "evidence.raw_outcomes_unavailable"
    else:
        assert refusals == ()
        (cell,) = run(restored, resolver).cells
        assert cell.summary is not None
        assert cell.summary.point == pytest.approx(4 / 3)
    assert counted.unit_frame_reads == 0


def _frame():
    return pd.DataFrame(
        {
            "unit": [f"u{i}" for i in range(40)],
            "arm": ["control", "treatment"] * 20,
            "value": [float(i % 7) for i in range(40)],
        }
    )


@pytest.mark.integration
def test_package_import_does_not_require_fcntl():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['fcntl'] = None; from increment import Analysis; Analysis",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("shape", ["summary", "panel"])
def test_from_frame_builder_estimates_survive_specification_round_trip(shape):
    frame = _frame().assign(value=[1.0, 3.0, 2.0, 4.0] * 10, day="2025-01-01")
    if shape == "summary":
        experiment, resolver = from_frame(
            frame,
            shape="summary",
            unit="unit",
            group="arm",
            control="control",
            metrics={"value": "mean"},
            intervention_grain="unit",
        )
    else:
        experiment, resolver = from_frame(
            frame,
            shape="panel",
            unit="unit",
            group="arm",
            date="day",
            control="control",
            metrics={"value": "mean"},
            observation_end="2025-01-02",
        )
    restored = Experiment.from_json(experiment.to_json())
    assert restored.hash() == experiment.hash()
    (cell,) = run(restored, resolver).cells
    assert cell.identity.key.metric == "value"
    assert cell.summary is not None
    assert cell.summary.point == pytest.approx(4 / 3)


def test_frame_run_matches_analysis_run():

    frame = _frame()
    design = Randomized(control_group="control")
    expected = view_readout(
        *from_frame(
            frame,
            shape="summary",
            unit="unit",
            group="arm",
            design=design,
            metrics={"value": "mean"},
        )
    )
    experiment, resolver = from_frame(
        frame,
        shape="summary",
        unit="unit",
        group="arm",
        design=design,
        metrics={"value": "mean"},
    )
    actual = run(experiment, resolver)
    assert isinstance(actual, RunResult)
    expected_cells = expected.as_cells()
    assert len(actual.cells) == len(expected_cells)
    for actual_cell, expected_cell in zip(actual.cells, expected_cells, strict=True):
        assert actual_cell.identity.key == expected_cell.identity.key
        assert actual_cell.summary is not None
        assert expected_cell.summary is not None
        assert actual_cell.summary.point == pytest.approx(expected_cell.summary.point, rel=1e-9)
        assert actual_cell.summary.lower == pytest.approx(expected_cell.summary.lower, rel=1e-9)
        assert actual_cell.summary.upper == pytest.approx(expected_cell.summary.upper, rel=1e-9)


def test_analysis_summary_facade_delegates_to_public_builder():

    analysis = from_frame(
        _frame(),
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics={"value": "mean"},
    )
    experiment, resolver = analysis

    assert isinstance(experiment, Experiment)
    assert set(experiment.source_refs[0].canonical_payload()) == {
        "name",
        "uri",
        "expected_digest",
        "schema",
        "capabilities",
        "declaration_version",
    }
    assert isinstance(run(experiment, resolver), RunResult)


@pytest.mark.integration
def test_definition_exploratory_metric_runs_with_cell_identity(seeded_defs, seeded_con):
    experiment, resolver = from_definitions(
        seeded_defs, seeded_con, experiment_name="new_onboarding_v2"
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.warns(IncrementWarning):
                baseline = run(experiment, resolver)
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.warns(IncrementWarning):
                result = run(
                    experiment,
                    resolver,
                    exploratory_metrics=["aov"],
                )
    finally:
        cast(Any, resolver).close()

    assert result.decisions.program == baseline.decisions.program
    assert isinstance(result, RunResult)
    exploratory = {
        cell.identity.hash() for cell in result.cells if cell.identity.key.metric == "aov"
    }
    assert exploratory
    assert {
        verdict.role for verdict in result.decisions.cells if verdict.identity_hash in exploratory
    } == {"exploratory"}


@pytest.mark.integration
def test_builder_close_releases_owned_tables_preserves_caller_connection(seeded_defs, seeded_con):
    seeded_con.raw_sql("CREATE TEMP TABLE binding_close_sentinel(value INTEGER)")
    seeded_con.raw_sql("INSERT INTO binding_close_sentinel VALUES (37)")

    first_experiment, first = from_definitions(
        seeded_defs, seeded_con, experiment_name="new_onboarding_v2", store="always"
    )
    second_experiment, second = from_definitions(
        seeded_defs, seeded_con, experiment_name="new_onboarding_v2", store="always"
    )
    first_bound = cast(Any, first)
    second_bound = cast(Any, second)

    def materialized_names(experiment: Experiment, resolver: Any) -> tuple[str, ...]:
        materialize(experiment, resolver)
        session = resolver.binding.session
        names = tuple(item.name for item in session._materialized)
        assert names
        assert set(names) <= set(seeded_con.list_tables())
        return names

    first_names = materialized_names(first_experiment, first_bound)
    second_names = materialized_names(second_experiment, second_bound)
    assert set(first_names).isdisjoint(second_names)

    first_bound.close()
    first_bound.close()
    assert not set(first_names) & set(seeded_con.list_tables())
    assert set(second_names) <= set(seeded_con.list_tables())
    assert seeded_con.raw_sql("SELECT value FROM binding_close_sentinel").fetchone()[0] == 37

    with second_bound:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IncrementWarning)
            assert run(second_experiment, second)
    assert not set(second_names) & set(seeded_con.list_tables())
    assert seeded_con.raw_sql("SELECT value FROM binding_close_sentinel").fetchone()[0] == 37


@pytest.mark.integration
def test_closing_derived_bindings_preserves_parent_session(seeded_defs, seeded_con):
    experiment, resolver = from_definitions(
        seeded_defs, seeded_con, experiment_name="new_onboarding_v2", store="always"
    )
    bound_resolver = cast(Any, resolver)
    materialize(experiment, resolver)
    parent_names = tuple(item.name for item in bound_resolver.binding.session._materialized)
    assert parent_names

    analysis = cast(Any, Analysis.__new__(Analysis))
    analysis._binding = bound_resolver.binding
    added = analysis._exploratory_metrics(["aov"], caller="run")
    exploratory = analysis._exploratory_analysis(added, caller="run")._binding

    def resolver_with_binding(binding: Any) -> Any:
        derived_resolver = copy.copy(bound_resolver)
        derived_resolver.binding = binding
        return derived_resolver

    derived_resolver = resolver_with_binding(exploratory)

    try:
        breakout_resolver = resolver_with_binding(bound_resolver.binding.with_breakouts(()))
        breakout_resolver.close()
        assert set(parent_names) <= set(seeded_con.list_tables())

        derived_resolver.close()
        assert set(parent_names) <= set(seeded_con.list_tables())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IncrementWarning)
            assert run(experiment, resolver)
    finally:
        bound_resolver.close()


@pytest.mark.integration
def test_close_live_run_close_releases_new_warehouse_tables(seeded_defs, seeded_con):
    experiment, resolver = from_definitions(
        seeded_defs, seeded_con, experiment_name="new_onboarding_v2", store="always"
    )
    bound_resolver = cast(Any, resolver)
    materialize(experiment, resolver)
    session = bound_resolver.binding.session
    bound_resolver.close()
    assert not session._materialized

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IncrementWarning)
        assert run(experiment, resolver, coherence="live")

    live_names = tuple(item.name for item in session._materialized)
    assert live_names
    assert set(live_names) <= set(seeded_con.list_tables())
    bound_resolver.close()
    assert not set(live_names) & set(seeded_con.list_tables())


@pytest.mark.integration
def test_builder_exploratory_binding_does_not_close_parent_source(seeded_defs, seeded_con):
    experiment, resolver = from_definitions(
        seeded_defs, seeded_con, experiment_name="new_onboarding_v2", store="always"
    )
    bound_resolver = cast(Any, resolver)
    try:
        materialize(experiment, resolver)
        session = bound_resolver.binding.session
        parent_names = tuple(item.name for item in session._materialized)
        assert parent_names
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.warns(IncrementWarning):
                run(experiment, resolver, exploratory_metrics=["aov"])
        assert set(parent_names) <= set(seeded_con.list_tables())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IncrementWarning)
            assert run(experiment, resolver)
    finally:
        bound_resolver.close()


def test_offline_definitions_builder_refuses_missing_catalog(tmp_path):
    definitions = tmp_path / "definitions.yaml"
    definitions.write_text("experiments: []\n", encoding="utf-8")

    with pytest.raises(CodedError) as raised:
        from_definitions(definitions)

    assert raised.value.code == "kernel.definitions.catalog_missing"


def test_offline_definitions_builder_refuses_stale_catalog(tmp_path):
    definitions = tmp_path / "definitions.yaml"
    definitions.write_text("experiments: []\n", encoding="utf-8")
    catalog = definitions.with_suffix(".yaml.kernel.json")
    catalog.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "definitions_digest": "stale",
                "experiments": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CodedError) as raised:
        from_definitions(definitions)

    assert raised.value.code == "kernel.definitions.catalog_stale"


def test_offline_definitions_builder_loads_fresh_catalog_without_connection(tmp_path):
    definitions = tmp_path / "definitions.yaml"
    definitions.write_text("experiments: []\n", encoding="utf-8")
    encoded_path = definitions.name.encode("utf-8")
    contents = definitions.read_bytes()
    digest = hashlib.sha256(
        len(encoded_path).to_bytes(8, "big")
        + encoded_path
        + len(contents).to_bytes(8, "big")
        + contents
    ).hexdigest()
    experiment = Experiment(
        experiment_id="exp",
        design={},
        metrics={},
        registration=Registration(registration_hash="registration-hash"),
        source_refs=(SourceReference(name="events", uri="warehouse://events"),),
        scopes=(Scope(view="run", grain="total"),),
    )
    catalog = definitions.with_suffix(".yaml.kernel.json")
    catalog.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "definitions_digest": digest,
                "experiments": {"exp": json.loads(experiment.to_json())},
            }
        ),
        encoding="utf-8",
    )

    restored, resolver = from_definitions(definitions)

    assert restored.hash() == experiment.hash()
    changed = replace(
        restored,
        registration=restored.registration.model_copy(update={"registration_hash": "changed"}),
    )
    assert changed.source_refs == restored.source_refs
    with pytest.raises(CodedError) as invalid:
        run(changed, resolver)
    assert invalid.value.code == "kernel.run.execution_unavailable"
    with pytest.raises(CodedError) as raised:
        run(restored, resolver)
    assert raised.value.code == "kernel.run.resolver_missing"


def test_module_builders_are_root_exports():
    import increment

    assert all(
        callable(getattr(increment, name))
        for name in (
            "from_frame",
            "from_definitions",
            "from_unit_day_artifact",
            "from_switchback_frame",
            "from_moments",
        )
    )


@pytest.mark.integration
def test_concurrent_definitions_builders_retain_both_catalog_entries(
    tmp_path, seeded_defs, seeded_con, con, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    definitions = tmp_path / "definitions"
    shutil.copytree(seeded_defs, definitions)
    catalog = definitions / ".increment-kernel-catalog.json"
    catalog.unlink(missing_ok=True)
    ready = Barrier(2, timeout=30)
    decode = json.loads

    def decode_together(value, *args, **kwargs):
        payload = decode(value, *args, **kwargs)
        if isinstance(payload, dict) and "experiment_id" in payload and "source_refs" in payload:
            ready.wait()
        return payload

    with monkeypatch.context() as patch:
        patch.setattr(json, "loads", decode_together)
        with ThreadPoolExecutor(max_workers=2) as pool:
            writers = [
                pool.submit(from_definitions, definitions, connection, experiment_name=name)
                for name, connection in (
                    ("new_onboarding_v2", seeded_con),
                    ("pricing_tier_test", con),
                )
            ]
            bound = [writer.result()[0] for writer in writers]

    for experiment in bound:
        restored, _ = from_definitions(definitions, experiment_name=experiment.experiment_id)
        assert restored.hash() == experiment.hash()


@pytest.mark.integration
@pytest.mark.slow
def test_definitions_catalog_round_trips_schema_capabilities_and_runnability(
    tmp_path, seeded_defs, seeded_con
):
    definitions = tmp_path / "definitions"
    shutil.copytree(seeded_defs, definitions)

    catalog_path = definitions / ".increment-kernel-catalog.json"
    if not catalog_path.exists():
        from_definitions(definitions, seeded_con, experiment_name="new_onboarding_v2")
    cached, _ = from_definitions(definitions, experiment_name="new_onboarding_v2")
    online, online_resolver = from_definitions(
        definitions, seeded_con, experiment_name="new_onboarding_v2"
    )
    assert online.source_refs[0].schema == {}
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always", IncrementWarning)
        assert run(
            cached,
            online_resolver,
            metrics=["purchase_rate"],
        )
    assert warning_codes(captured_warnings) == ["analysis.cuped.covariate_unused"]
    catalog_path = definitions / ".increment-kernel-catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    offline, resolver = from_definitions(definitions, experiment_name="new_onboarding_v2")

    assert offline.source_refs[0].schema
    assert catalog["experiments"]["new_onboarding_v2"]["source_refs"][0]["schema"]
    recorded_digest = online.source_refs[0].expected_digest
    assert recorded_digest is not None
    assert (
        catalog["experiments"]["new_onboarding_v2"]["source_refs"][0]["expected_digest"]
        == recorded_digest
    )
    assert offline.source_refs[0].expected_digest == recorded_digest
    provider = online_resolver.resolve(online.source_refs)
    assert online_resolver.verify(offline.source_refs, provider) == []
    changed = (replace(offline.source_refs[0], expected_digest="sha256:changed"),)
    assert [error.code for error in online_resolver.verify(changed, provider)] == [
        "kernel.resolver.digest_mismatch"
    ]
    assert offline.source_refs[0].capabilities == online.source_refs[0].capabilities
    assert runnable(offline, "purchase_rate") == runnable(online, "purchase_rate")
    assert Experiment.from_json(online.to_json()).hash() == online.hash()
    assert Experiment.from_json(offline.to_json()).hash() == offline.hash()

    definitions.joinpath("experiments.yaml").write_text("experiments: []\n", encoding="utf-8")

    assert tuple(error.code for error in resolver.verify(offline.source_refs, object())) == (
        "kernel.definitions.catalog_stale",
    )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.filterwarnings(
    "ignore:metric .* has window_days=None:increment.errors.IncrementWarning"
)
@pytest.mark.filterwarnings(
    "ignore:metric .*quantile sequential inference is refused:increment.errors.IncrementWarning"
)
@pytest.mark.parametrize(
    ("scenario_id", "ingress", "expected"),
    [
        pytest.param(scenario_id, ingress, expected, id=f"{scenario_id}-{ingress}")
        for scenario_id, scenario in _capture()["scenarios"].items()
        for ingress, expected in scenario.items()
    ],
)
def test_frozen_capture_builder_specification_round_trip(
    scenario_id, ingress, expected, record_property
):
    from tests.parity_harness.cases import PARITY_CASES, _close_parity_binding

    cases = {case.id: case for case in PARITY_CASES}
    base_id = scenario_id.removeprefix("asof_lift-").removeprefix("asof-")
    case = cases.get(base_id)
    if case is None:
        assert expected["kind"] in {"refusal", "absent"}
        record_property(
            "not_expressible",
            expected.get("reason", "isolated refusal probe, not an analysis construction"),
        )
        return
    if ingress not in case.build:
        assert expected["kind"] == "absent"
        record_property("not_expressible", "no constructor for this declared ingress")
        return
    try:
        bound = case.build[ingress]()
    except CodedError as error:
        assert expected["kind"] == "refusal"
        assert error.code == expected["code"]
        record_property("not_expressible", f"construction refuses: {error.code}")
        return
    try:
        experiment, _ = bound
        assert Experiment.from_json(experiment.to_json()).hash() == experiment.hash()
        record_property("specification_hash", experiment.hash())
    finally:
        _close_parity_binding(bound)


@pytest.mark.integration
def test_saved_artifact_experiment_replays_with_fresh_resolver():
    from increment.builders import from_unit_day_artifact
    from tests.test_unit_day_artifact_facade import _published

    con, (native_experiment, native_resolver), context, store, ref = _published()
    try:
        experiment, _initial_resolver = from_unit_day_artifact(store, ref, expected_context=context)
        assert experiment.analysis_context["source_name"] == "_ArtifactFacadeSource"
        saved_experiment = Experiment.from_json(experiment.to_json())

        _fresh_experiment, fresh_resolver = from_unit_day_artifact(
            store, ref, expected_context=context
        )

        assert saved_experiment.hash() == experiment.hash()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.warns(IncrementWarning):
                expected = view_readout(native_experiment, native_resolver).as_cells()
        actual = run(saved_experiment, fresh_resolver).cells
        assert len(actual) == len(expected)
        for cell, reference in zip(actual, expected, strict=True):
            assert cell.identity.key == reference.identity.key
            assert cell.summary is not None
            assert reference.summary is not None
            assert cell.summary.point == pytest.approx(reference.summary.point)
            assert cell.summary.lower == pytest.approx(reference.summary.lower)
            assert cell.summary.upper == pytest.approx(reference.summary.upper)
        _initial_resolver.close()
        fresh_resolver.close()
    finally:
        con.disconnect()


@pytest.mark.integration
def test_artifact_resolver_reopens_owned_snapshot_after_close():
    from increment.builders import from_unit_day_artifact
    from tests.test_unit_day_artifact_facade import _published

    con, _native, context, store, ref = _published()
    con.raw_sql("CREATE TEMP TABLE artifact_binding_sentinel(value INTEGER)")
    con.raw_sql("INSERT INTO artifact_binding_sentinel VALUES (53)")
    resolver = None
    try:
        experiment, resolver = from_unit_day_artifact(store, ref, expected_context=context)
        first_source = resolver.binding.state.source
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IncrementWarning)
            assert run(experiment, resolver).cells
        resolver.close()
        assert getattr(first_source, "closed", False)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IncrementWarning)
            assert run(experiment, resolver).cells
        second_source = resolver.binding.state.source
        assert second_source is not first_source
        resolver.close()
        assert getattr(second_source, "closed", False)
        assert con.raw_sql("SELECT value FROM artifact_binding_sentinel").fetchone()[0] == 53
    finally:
        if resolver is not None:
            resolver.close()
        con.disconnect()


@pytest.mark.integration
def test_definitions_positional_arguments_remain_executable(seeded_defs, seeded_con):
    builder = cast(Any, from_definitions)
    for build in (
        lambda: builder("new_onboarding_v2", seeded_defs, seeded_con),
        lambda: builder("new_onboarding_v2", seeded_defs, con=seeded_con),
        lambda: builder("new_onboarding_v2", definitions_path=seeded_defs, con=seeded_con),
    ):
        experiment, resolver = build()
        with resolver, warnings.catch_warnings():
            warnings.simplefilter("ignore", IncrementWarning)
            result = run(experiment, resolver, metrics=["purchase_rate"])
        assert result.cells
        assert all(
            cell.summary is not None and cell.summary.point is not None for cell in result.cells
        )
