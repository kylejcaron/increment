from __future__ import annotations

import copy
import hashlib
import json
import pickle
import shutil
import subprocess
import sys
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any, NoReturn, cast

import pandas as pd
import pytest
import yaml

from increment import MetricSpec, Randomized, RunResult, Winsorization, run
from increment.builders import from_definitions, from_frame, from_moments
from increment.errors import CodedError, IncrementWarning
from increment.kernel.contracts import Registration, View
from increment.kernel.experiment import Experiment, SourceReference
from increment.kernel.resolver import runnable
from increment.workflows import export_moments, materialize, unit_counts
from tests.analysis_factory import bound_source, replace_bound_source, view_readout
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

    counted = _CountingSource(resolver._binding.source)
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
            "import sys; sys.modules['fcntl'] = None; from increment import from_definitions, run",
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


def test_frame_builder_records_source_reference_fields():

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


# Exploratory cell identity compares complete native runs before and after selection.
@pytest.mark.slow
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
def test_saved_definitions_experiment_keeps_its_declaration_and_runs(seeded_defs, seeded_con):
    reference = json.loads(
        (Path(__file__).parent / "fixtures" / "definitions_experiment_reference.json").read_text()
    )
    saved = Experiment.from_json(json.dumps(reference["experiment"]))
    experiment, resolver = from_definitions(
        seeded_defs, seeded_con, experiment_name="new_onboarding_v2"
    )
    try:
        assert json.loads(experiment.to_json()) == reference["experiment"]
        assert experiment.hash() == reference["hash"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IncrementWarning)
            assert isinstance(run(saved, resolver), RunResult)
    finally:
        cast(Any, resolver).close()


@pytest.mark.integration
def test_offline_and_connected_definitions_share_identity_across_runs(seeded_defs, seeded_con):
    built, reader = from_definitions(seeded_defs, seeded_con, experiment_name="new_onboarding_v2")
    before, _ = from_definitions(seeded_defs, experiment_name="new_onboarding_v2")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IncrementWarning)
            run(built, reader)
    finally:
        cast(Any, reader).close()
    after, _ = from_definitions(seeded_defs, experiment_name="new_onboarding_v2")

    assert built.hash() == before.hash() == after.hash()
    assert built.analysis_spec_hash == before.analysis_spec_hash == after.analysis_spec_hash
    assert after.provides()


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
        session = resolver._binding.session
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
    from increment import ExperimentReader, _exploratory

    experiment, resolver = from_definitions(
        seeded_defs, seeded_con, experiment_name="new_onboarding_v2", store="always"
    )
    bound_resolver = cast(Any, resolver)
    materialize(experiment, resolver)
    parent_names = tuple(item.name for item in bound_resolver._binding.session._materialized)
    assert parent_names

    added = _exploratory.select_exploratory(bound_resolver._binding, ["aov"], caller="run")
    exploratory = _exploratory.exploratory_sibling(bound_resolver._binding, added, caller="run")

    def resolver_with_binding(binding: Any) -> Any:
        return ExperimentReader(binding, experiment)

    derived_resolver = resolver_with_binding(exploratory)

    try:
        breakout_resolver = resolver_with_binding(bound_resolver._binding.with_breakouts(()))
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
def test_closed_warehouse_reader_refuses_a_live_run_before_materializing(seeded_defs, seeded_con):
    experiment, resolver = from_definitions(
        seeded_defs, seeded_con, experiment_name="new_onboarding_v2", store="always"
    )
    materialize(experiment, resolver)
    session = resolver._binding.session
    resolver.close()
    assert not session._materialized

    with pytest.raises(CodedError) as closed:
        run(experiment, resolver, coherence="live")

    assert closed.value.code == "kernel.run.reader_closed"
    assert not session._materialized


# Materialization and repeated exploratory runs protect the parent session lifecycle.
@pytest.mark.slow
@pytest.mark.integration
def test_builder_exploratory_binding_does_not_close_parent_source(seeded_defs, seeded_con):
    experiment, resolver = from_definitions(
        seeded_defs, seeded_con, experiment_name="new_onboarding_v2", store="always"
    )
    bound_resolver = cast(Any, resolver)
    try:
        materialize(experiment, resolver)
        session = bound_resolver._binding.session
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
                "schema_version": 2,
                "definitions_digest": "stale",
                "experiments": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CodedError) as raised:
        from_definitions(definitions)

    assert raised.value.code == "kernel.definitions.catalog_stale"


def _write_offline_catalog(tmp_path: Path) -> tuple[Path, Experiment]:
    """Definitions with a fresh offline catalog of one experiment, written without a connection."""
    from increment.semantics import load

    definitions = tmp_path / "definitions.yaml"
    definitions.write_text("experiments: []\n", encoding="utf-8")
    declared = json.dumps(
        load(definitions).model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(declared.encode("utf-8")).hexdigest()
    experiment = Experiment(
        experiment_id="exp",
        design={},
        metrics={},
        registration=Registration(registration_hash="registration-hash"),
        source_refs=(SourceReference(name="events", uri="warehouse://events"),),
        views=(View(view="run", grain="total"),),
    )
    catalog = definitions.with_suffix(".yaml.kernel.json")
    catalog.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "definitions_digest": digest,
                "experiments": {"exp": json.loads(experiment.to_json())},
            }
        ),
        encoding="utf-8",
    )
    return definitions, experiment


def test_offline_definitions_builder_loads_fresh_catalog_without_connection(tmp_path):
    definitions, experiment = _write_offline_catalog(tmp_path)

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


def test_reader_public_surface_is_pinned(tmp_path):
    from increment import ExperimentReader

    _, frame_reader = from_frame(
        _frame(),
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics={"value": "mean"},
    )
    definitions, _ = _write_offline_catalog(tmp_path)
    _, offline_reader = from_definitions(definitions)

    for reader in (frame_reader, offline_reader):
        assert type(reader) is ExperimentReader
        assert {name for name in dir(reader) if not name.startswith("_")} == {
            "experiment",
            "closed",
            "close",
        }
    assert repr(frame_reader) == (
        "ExperimentReader(experiment_id='frame', reads='dataframe snapshot', closed=False)"
    )
    offline_reader.close()
    assert repr(offline_reader) == (
        "ExperimentReader(experiment_id='exp', reads='offline: cannot read', closed=True)"
    )


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_frame_reader_keeps_the_rows_it_was_built_from(backend):
    import numpy as np
    import polars as pl

    rng = np.random.default_rng(11)
    arm = np.where(np.arange(200) % 2, "treatment", "control")
    columns = {
        "unit": [f"u{index}" for index in range(200)],
        "arm": arm,
        "revenue": rng.gamma(2.0, 5.0, 200),
        "latency": rng.gamma(3.0, 20.0, 200),
    }
    frame = pl.DataFrame(columns) if backend == "polars" else pd.DataFrame(columns)

    def lifts(experiment, reader):
        from increment.results import LiftEstimates

        readout = run(experiment, reader).readout()
        assert isinstance(readout, LiftEstimates)
        return {row.metric: row.require_lift().value for row in readout}

    def build():
        return from_frame(
            frame,
            shape="summary",
            unit="unit",
            group="arm",
            control="control",
            metrics=[
                MetricSpec(name="revenue", type="mean"),
                MetricSpec(name="latency", type="quantile", quantile=0.5),
            ],
        )

    experiment, reader = build()
    built = lifts(experiment, reader)
    treated = arm == "treatment"
    if isinstance(frame, pl.DataFrame):
        for name in ("revenue", "latency"):
            scaled = frame[name].to_numpy() * np.where(treated, 5.0, 1.0)
            frame.replace_column(frame.get_column_index(name), pl.Series(name, scaled))
    else:
        frame.loc[treated, ["revenue", "latency"]] *= 5.0

    assert lifts(experiment, reader) == built
    rebuilt = lifts(*build())
    assert all(rebuilt[name] != built[name] for name in ("revenue", "latency"))


def test_closed_reader_refuses_every_read(tmp_path):
    import polars as pl

    experiment, reader = from_frame(
        pl.from_pandas(_frame()),
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics={"value": "mean"},
    )
    with reader:
        assert run(experiment, reader).cells
    definitions, _ = _write_offline_catalog(tmp_path)
    offline_experiment, offline_reader = from_definitions(definitions)
    offline_reader.close()

    for closed_experiment, closed_reader in (
        (experiment, reader),
        (offline_experiment, offline_reader),
    ):
        assert closed_reader.closed is True
        for read in (run, unit_counts):
            with pytest.raises(CodedError) as closed:
                read(closed_experiment, closed_reader)
            assert closed.value.code == "kernel.run.reader_closed"
            assert dict(closed.value.context) == {"experiment_id": closed_experiment.experiment_id}


@pytest.mark.parametrize(
    "reads", ["frame", pytest.param("warehouse", marks=pytest.mark.integration)]
)
def test_reader_refuses_pickle_and_deepcopy(request, reads):
    if reads == "frame":
        experiment, reader = from_frame(
            _frame(),
            shape="summary",
            unit="unit",
            group="arm",
            control="control",
            metrics={"value": "mean"},
        )
    else:
        experiment, reader = from_definitions(
            request.getfixturevalue("seeded_defs"),
            request.getfixturevalue("seeded_con"),
            experiment_name="new_onboarding_v2",
        )
    with reader:
        for portable in (pickle.dumps, copy.deepcopy):
            with pytest.raises(CodedError) as refused:
                portable(reader)
            assert refused.value.code == "kernel.run.reader_unportable"
            assert dict(refused.value.context) == {"experiment_id": experiment.experiment_id}
    restored = pickle.loads(pickle.dumps(refused.value))
    assert (restored.code, dict(restored.context)) == (
        refused.value.code,
        dict(refused.value.context),
    )


def test_reads_refuse_anything_but_a_reader():
    experiment, reader = from_frame(
        _frame(),
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics={"value": "mean"},
    )

    class SiblingReader:
        """Carries a real reader's whole surface without being an ExperimentReader."""

        def __getattr__(self, name: str) -> object:
            return getattr(reader, name)

    refused_arguments = (
        (experiment, None),
        (reader, experiment),
        (experiment, object()),
        (experiment, SiblingReader()),
    )
    for read in (run, unit_counts):
        for arguments in refused_arguments:
            with pytest.raises(CodedError) as refused:
                read(*cast(Any, arguments))
            assert refused.value.code == "kernel.run.reader_required"
            assert refused.value.context["received"] == type(arguments[1]).__name__


@pytest.mark.parametrize(
    "field", ["experiment_id", "design", "registration", "analysis_context", "metrics"]
)
def test_mismatch_refusal_names_the_first_differing_field(field):
    _, reader = from_frame(
        _frame(),
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics={"value": "mean"},
    )
    declared = reader.experiment
    changed = {
        "experiment_id": "other",
        "design": {**declared.design, "control_group": "treatment"},
        "registration": declared.registration.model_copy(update={"registration_hash": "changed"}),
        "analysis_context": {**declared.analysis_context, "source_name": "other"},
        "metrics": {**declared.metrics, "other": declared.metrics["value"]},
    }[field]
    experiment = replace(declared, **{field: changed})

    with pytest.raises(CodedError) as mismatch:
        run(experiment, reader)

    assert mismatch.value.code == "kernel.run.execution_unavailable"
    assert dict(mismatch.value.context) == {
        "experiment_id": experiment.experiment_id,
        "reader_experiment_id": reader.experiment.experiment_id,
        "field": field,
    }


def test_bound_resolver_refusals_use_provider_codes():
    from types import SimpleNamespace

    from increment.evidence import BoundResolver, EvidenceCapabilities, EvidenceProvider

    capabilities = EvidenceCapabilities(
        kind="unit_summary",
        grains=frozenset({"total"}),
        breakouts=(),
        shape="unit_summary",
        operations=frozenset(),
        unit_rows=True,
        raw_outcomes=True,
        method_fit="none",
        compliance_axis="none",
        count_grains=frozenset({"unit"}),
        populations=frozenset({"assigned"}),
        sequential_capture="none",
        scalar=None,
        available_covariates=frozenset(),
    )
    source = SourceReference(
        name="events",
        uri="frame://events",
        expected_digest="source-recorded",
        capabilities=capabilities,
    )
    changed = cast(
        EvidenceProvider,
        SimpleNamespace(evidence=capabilities, context=SimpleNamespace(source_digest="current")),
    )
    undeclared = cast(
        EvidenceProvider,
        SimpleNamespace(evidence=None, context=SimpleNamespace(source_digest="source-recorded")),
    )

    assert [error.code for error in BoundResolver(changed).verify((source,), changed)] == [
        "kernel.provider.digest_mismatch"
    ]
    assert [error.code for error in BoundResolver(undeclared).verify((source,))] == [
        "kernel.provider.capabilities_missing"
    ]


def test_digest_mismatch_carries_one_context_from_every_raise_site():
    from types import SimpleNamespace

    from increment.evidence import BoundResolver, EvidenceProvider

    experiment, reader = from_frame(
        _frame(),
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics={"value": "mean"},
    )
    reference = experiment.source_refs[0]
    refusals = []
    for changed in (
        replace(reference, uri="other://source"),
        replace(reference, expected_digest="sha256:other"),
    ):
        with pytest.raises(CodedError) as mismatch:
            run(replace(experiment, source_refs=(changed,)), reader)
        refusals.append(mismatch.value)
    provider = cast(
        EvidenceProvider,
        SimpleNamespace(
            evidence=reference.capabilities, context=SimpleNamespace(source_digest="sha256:now")
        ),
    )
    refusals.extend(BoundResolver(provider).verify((reference,), provider))

    assert [(refusal.code, dict(refusal.context)) for refusal in refusals] == [
        ("kernel.provider.digest_mismatch", {"source_refs": (reference.name,)})
    ] * 3


@pytest.mark.integration
def test_offline_reader_supports_with_and_refuses_reads(seeded_defs, seeded_con):
    _, connected = from_definitions(seeded_defs, seeded_con, experiment_name="new_onboarding_v2")
    connected.close()
    experiment, reader = from_definitions(seeded_defs, experiment_name="new_onboarding_v2")

    with reader, pytest.raises(CodedError) as offline:
        run(experiment, reader)

    assert offline.value.code == "kernel.run.reader_offline"
    display = repr(reader)
    assert "0x" not in display and repr(experiment.experiment_id) in display


def _change_metric_window(definitions: Path) -> None:
    metrics_path = definitions / "metrics.yaml"
    declared = yaml.safe_load(metrics_path.read_text(encoding="utf-8"))
    purchase_rate = next(m for m in declared["metrics"] if m["name"] == "purchase_rate")
    purchase_rate["window_days"] = 7
    metrics_path.write_text(yaml.safe_dump(declared), encoding="utf-8")


def _write_impossible_date(definitions: Path) -> None:
    experiments_path = definitions / "experiments.yaml"
    text = experiments_path.read_text(encoding="utf-8")
    experiments_path.write_text(
        text.replace("start: 2025-01-15", "start: 2025-13-15"), encoding="utf-8"
    )


def _write_undecodable_bytes(definitions: Path) -> None:
    experiments_path = definitions / "experiments.yaml"
    experiments_path.write_bytes(experiments_path.read_bytes() + b"\xff\n")


class _ReadRefusingSource:
    """Answer a bound source's declarations; fail the test on any evidence read."""

    def __init__(self, source: Any) -> None:
        self.evidence = source.evidence
        self.context = source.context
        self.declaration_identity = source.declaration_identity

    def _read(self, request: object) -> NoReturn:
        pytest.fail(f"edited definitions were read: {type(request).__name__}")

    moments = breakout_method_moments = unit_rows = method_fit = _read
    compliance = compliance_series = counts = capture_sequential = scope = _read


@pytest.mark.integration
@pytest.mark.parametrize(
    "edit",
    [_change_metric_window, _write_impossible_date, _write_undecodable_bytes],
    ids=["changed_metric", "impossible_date", "undecodable_bytes"],
)
def test_edited_definitions_refuse_before_any_read(seeded_defs, seeded_con, edit):
    import duckdb
    from packaging.version import Version

    from tests.parity_harness.query_log import disable_query_log, enable_query_log, query_log

    experiment, reader = from_definitions(
        seeded_defs, seeded_con, experiment_name="new_onboarding_v2"
    )
    source = bound_source(reader)
    edit(Path(seeded_defs))
    # The stand-in fails on evidence reads; the query log sees any statement on the connection.
    replace_bound_source(reader, _ReadRefusingSource(source))
    logs_queries = enable_query_log(seeded_con)
    # DuckDB's QueryLog first ships in 1.3.0.
    assert logs_queries == (Version(duckdb.__version__) >= Version("1.3.0"))
    try:
        for read in (run, unit_counts):
            logged = len(query_log(seeded_con)) if logs_queries else 0
            with warnings.catch_warnings(), pytest.raises(CodedError) as stale:
                warnings.simplefilter("ignore", IncrementWarning)
                read(experiment, reader)
            if logs_queries:
                statements = [
                    statement
                    for statement in query_log(seeded_con)[logged:]
                    if "duckdb_logs" not in statement
                ]
                assert statements == [], read
            assert stale.value.code == "kernel.definitions.catalog_stale", read
            assert dict(stale.value.context) == {"path": str(Path(seeded_defs))}
    finally:
        if logs_queries:
            disable_query_log(seeded_con)
        replace_bound_source(reader, source)
        cast(Any, reader).close()
    with pytest.raises(CodedError) as offline:
        from_definitions(seeded_defs)
    assert offline.value.code == "kernel.definitions.catalog_stale"
    assert dict(offline.value.context) == {"path": str(Path(seeded_defs))}


@pytest.mark.integration
def test_comment_only_definitions_edit_still_runs(seeded_defs, seeded_con):
    experiment, reader = from_definitions(
        seeded_defs, seeded_con, experiment_name="new_onboarding_v2"
    )
    experiments_path = Path(seeded_defs) / "experiments.yaml"
    experiments_path.write_text(
        experiments_path.read_text(encoding="utf-8") + "# note\n", encoding="utf-8"
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IncrementWarning)
            result = run(experiment, reader)
    finally:
        cast(Any, reader).close()

    assert result.cells
    # The build records no column types; the run's catalog refresh records them.
    restored, _ = from_definitions(seeded_defs)
    assert restored.source_refs[0].schema


@pytest.mark.integration
def test_version_one_catalog_refuses_with_rebuild_route(seeded_defs, seeded_con):
    def build() -> None:
        _, reader = from_definitions(seeded_defs, seeded_con, experiment_name="new_onboarding_v2")
        cast(Any, reader).close()

    build()
    catalog_path = Path(seeded_defs) / ".increment-kernel-catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog_path.write_text(json.dumps({**catalog, "schema_version": 1}), encoding="utf-8")

    with pytest.raises(CodedError) as stale:
        from_definitions(seeded_defs)
    assert stale.value.code == "kernel.definitions.catalog_stale"
    assert dict(stale.value.context) == {"path": str(Path(seeded_defs))}

    build()
    restored, _ = from_definitions(seeded_defs)
    assert restored.experiment_id == "new_onboarding_v2"


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
    offline, _ = from_definitions(definitions, experiment_name="new_onboarding_v2")

    assert offline.source_refs[0].schema
    assert catalog["experiments"]["new_onboarding_v2"]["source_refs"][0]["schema"]
    recorded_digest = online.source_refs[0].expected_digest
    assert recorded_digest is not None
    assert (
        catalog["experiments"]["new_onboarding_v2"]["source_refs"][0]["expected_digest"]
        == recorded_digest
    )
    assert offline.source_refs[0].expected_digest == recorded_digest
    provider = online_resolver._resolve(online.source_refs)
    assert online_resolver._verify(offline.source_refs, provider) == []
    changed = (replace(offline.source_refs[0], expected_digest="sha256:changed"),)
    assert [error.code for error in online_resolver._verify(changed, provider)] == [
        "kernel.provider.digest_mismatch"
    ]
    assert offline.source_refs[0].capabilities == online.source_refs[0].capabilities
    assert runnable(offline, "purchase_rate") == runnable(online, "purchase_rate")
    assert Experiment.from_json(online.to_json()).hash() == online.hash()
    assert Experiment.from_json(offline.to_json()).hash() == offline.hash()


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


# Closing a real artifact reader releases its snapshot, keeps the caller's connection and
# refuses later reads; a fresh reader over the same reference reads again.
@pytest.mark.slow
@pytest.mark.integration
def test_closed_artifact_reader_refuses_and_a_fresh_reader_reads():
    from increment.builders import from_unit_day_artifact
    from tests.test_unit_day_artifact_facade import _published

    con, _native, context, store, ref = _published()
    con.raw_sql("CREATE TEMP TABLE artifact_binding_sentinel(value INTEGER)")
    con.raw_sql("INSERT INTO artifact_binding_sentinel VALUES (53)")
    try:
        experiment, resolver = from_unit_day_artifact(store, ref, expected_context=context)
        source = resolver._binding.state.source
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IncrementWarning)
            with resolver:
                assert run(experiment, resolver).cells
        assert getattr(source, "closed", False)

        with pytest.raises(CodedError) as closed:
            run(experiment, resolver)
        assert closed.value.code == "kernel.run.reader_closed"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IncrementWarning)
            with from_unit_day_artifact(store, ref, expected_context=context)[1] as fresh:
                assert run(experiment, fresh).cells
        assert con.raw_sql("SELECT value FROM artifact_binding_sentinel").fetchone()[0] == 53
    finally:
        con.disconnect()


@pytest.mark.integration
def test_reader_repr_names_how_it_reads_without_reopening_a_closed_artifact():
    from increment.builders import from_unit_day_artifact
    from tests.test_unit_day_artifact_facade import _published

    con, _native, context, store, ref = _published()
    resolver = None
    try:
        experiment, resolver = from_unit_day_artifact(store, ref, expected_context=context)
        assert "closed=False" in repr(resolver)
        binding = resolver._binding
        resolver.close()

        display = repr(resolver)

        assert "reads='artifact'" in display and repr(experiment.experiment_id) in display
        assert "closed=True" in display and "0x" not in display
        assert resolver._bound is binding
        assert getattr(binding.state.source, "closed", False)
    finally:
        if resolver is not None:
            resolver.close()
        con.disconnect()
