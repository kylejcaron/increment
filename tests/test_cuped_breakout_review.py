from __future__ import annotations

from typing import Any, Literal, cast

import pyarrow as pa
import pytest

from increment import Analysis, AnalysisPlan, Method, MetricSpec, MultiplicitySpec
from increment.estimation.results import LiftEstimate
from increment.frame import CupedCovariate


def _linear_metric(
    *, metric_type: Literal["mean", "conversion", "ratio", "retention", "quantile"] = "mean"
) -> MetricSpec:
    return MetricSpec(
        name="revenue",
        type=metric_type,
        decision_method=Method.cuped_multicovariate(
            name="linear", covariates=CupedCovariate(name="pre", kind="categorical")
        ),
    )


def _breakout_analysis(*, unsupported: bool = False, null_segment: bool = False) -> Analysis:
    rows: list[dict[str, Any]] = []
    segments: tuple[str | None, ...] = (
        ("north", "south", "unsupported")
        if unsupported
        else (
            "north",
            "south",
        )
    )
    if null_segment:
        segments = ("north", None)
    for segment in segments:
        for group in ("control", "treatment"):
            for index in range(18):
                pre = (
                    "rare"
                    if segment == "unsupported" and group == "treatment" and index == 0
                    else ("common" if index < 9 else "other")
                )
                rows.append(
                    {
                        "unit": f"{segment}-{group}-{index}",
                        "group": group,
                        "day": 1,
                        "segment": segment,
                        "pre": pre,
                        "revenue": float(index + (group == "treatment") * (3 + index % 3)),
                    }
                )
    return Analysis.from_unit_panel(
        pa.Table.from_pylist(rows),
        unit="unit",
        group="group",
        date="day",
        control="control",
        metrics=[_linear_metric()],
        breakouts=["segment"],
    )


def test_public_breakout_localizes_failed_cuped_projection_and_keeps_siblings() -> None:
    from increment.errors import IncrementWarning

    analysis = _breakout_analysis(unsupported=True)
    with pytest.warns(IncrementWarning, match="estimation.cuped_linear.level_support"):
        rows = analysis.run_breakout()

    by_segment = {row.dimension_value: row for row in rows}
    assert set(by_segment) == {"north", "south", "unsupported"}
    assert by_segment["north"].lift is not None
    assert by_segment["south"].lift is not None
    assert by_segment["unsupported"].lift is None
    assert by_segment["unsupported"].excluded == "estimation_failed"
    unavailable = by_segment["unsupported"]
    assert unavailable.failure_code == "estimation.cuped_linear.level_support"
    assert unavailable.failure_context is not None
    assert unavailable.failure_context["column"] == "pre"
    assert unavailable.sampling_available is False
    assert unavailable.sampling_reason_code == unavailable.failure_code
    assert unavailable.sampling_reason_context == unavailable.failure_context
    assert rows.metadata is not None
    record = next(r for r in rows.metadata.cells if r.cell.dimension_value == "unsupported")
    assert record.failure is not None and record.failure.code == unavailable.failure_code
    assert record.sampling is not None and record.sampling.available is False
    assert record.sampling.reason_code == unavailable.failure_code
    assert record.sampling.reason_context == unavailable.failure_context


def test_public_breakout_sensitivity_failure_does_not_enter_decision_family() -> None:
    analysis = _breakout_analysis(unsupported=True)
    rows = analysis.run_breakout(
        decision_method=Method(name="plain"),
        sensitivity_methods=[
            Method.cuped_multicovariate(
                name="linear", covariates=CupedCovariate(name="pre", kind="categorical")
            )
        ],
    )

    unsupported = [row for row in rows if row.dimension_value == "unsupported"]
    assert {row.method for row in unsupported} == {"plain", "linear"}
    assert next(row for row in unsupported if row.method == "plain").lift is not None
    assert (
        next(row for row in unsupported if row.method == "linear").excluded == "estimation_failed"
    )
    failed = next(row for row in unsupported if row.method == "linear")
    assert failed.failure_code == "estimation.cuped_linear.level_support"
    assert failed.failure_context is not None
    assert failed.failure_context["column"] == "pre"
    assert failed.sampling_available is False
    assert all(
        row.family_axes == ("metric", "arm", "segment") for row in rows if row.method == "plain"
    )


def test_public_breakout_null_segment_uses_its_unit_rows() -> None:
    analysis = _breakout_analysis(null_segment=True)
    rows = analysis.run_breakout()

    null_rows = [row for row in rows if row.dimension_value == "__null__"]
    assert len(null_rows) == 1
    assert null_rows[0].lift is not None
    assert null_rows[0].n_control == null_rows[0].n_treat == 18


def test_public_breakout_keeps_all_conversion_methods_in_mixed_cuped_request() -> None:
    rows: list[dict[str, Any]] = []
    for segment in ("north", "south"):
        for group in ("control", "treatment"):
            for index in range(24):
                rows.append(
                    {
                        "unit": f"{segment}-{group}-{index}",
                        "group": group,
                        "day": 1,
                        "segment": segment,
                        "pre": float(index),
                        "region": "a" if index % 2 else "b",
                        "converted": int(index < (8 if group == "control" else 14)),
                    }
                )
    methods = [
        Method(name="plain"),
        Method.cuped(covariates=CupedCovariate(name="pre", kind="continuous"), name="scalar"),
        Method.cuped_multicovariate(
            name="linear",
            covariates=(
                CupedCovariate(name="pre", kind="continuous"),
                CupedCovariate(name="region", kind="categorical"),
            ),
        ),
    ]
    metric = MetricSpec(
        name="conversion",
        type="conversion",
        value_column="converted",
        decision_method=methods[0],
        sensitivity_methods=methods[1:],
    )
    analysis = Analysis.from_unit_panel(
        pa.Table.from_pylist(rows),
        unit="unit",
        group="group",
        date="day",
        control="control",
        metrics=[metric],
        breakouts=["segment"],
    )

    result = analysis.run_breakout()
    assert {(row.dimension_value, row.method) for row in result} == {
        (segment, method.name) for segment in ("north", "south") for method in methods
    }
    assert all(row.lift is not None for row in result)


@pytest.mark.integration
def test_definitions_and_artifact_breakouts_supply_cuped_unit_evidence(tmp_path) -> None:
    import yaml

    from increment.query.artifact_contract import unit_day_artifact_extension_catalog
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.models import Definitions
    from tests.covariate_cases import _CATEGORICAL_DESIGN, categorical_defs_and_con

    defs_path, con, _units = categorical_defs_and_con(tmp_path)
    text = defs_path.read_text().replace(
        "    plan: {primary: revenue}",
        "    plan: {primary: {metric: revenue, "
        "decision_method: {name: linear, variance_reduction: cuped_multicovariate, "
        "covariates: [{name: region, kind: categorical, property: region, source: events}]}}}\n"
        "    breakouts: [{property: region, source: events}]",
    )
    defs_path.write_text(text.replace(_CATEGORICAL_DESIGN, ""))
    native = Analysis.from_definitions("cat_test", defs_path, con)
    definitions = Definitions.model_validate(yaml.safe_load(defs_path.read_text()))
    experiment = definitions.experiment("cat_test")
    assert experiment is not None
    store = WarehouseArtifactStore(con, schema_name="breakout_review_artifacts")
    context = artifact_context(definitions, experiment, "error")
    extensions = [entry.request for entry in unit_day_artifact_extension_catalog(context)]
    reference = native.publish_unit_day_artifact(store, extensions=extensions)
    restored = Analysis.from_unit_day_artifact(store, reference, expected_context=context)

    native_rows = native.run_breakout()
    artifact_rows = restored.run_breakout()

    def identity(rows):
        return [
            (
                row.dimension_value,
                row.group_id,
                row.method,
                row.excluded,
                row.family_axes,
                row.family_procedure,
            )
            for row in rows
        ]

    assert identity(native_rows) == identity(artifact_rows)
    assert [row.require_lift().value for row in native_rows] == pytest.approx(
        [row.require_lift().value for row in artifact_rows]
    )
    assert native_rows.metadata is not None and artifact_rows.metadata is not None
    assert native_rows.metadata.scope is not None and artifact_rows.metadata.scope is not None
    assert native_rows.metadata.scope.cells == artifact_rows.metadata.scope.cells


def test_explicit_scalar_cuped_breakout_matches_frame_oracle(tmp_path) -> None:
    from tests.covariate_cases import categorical_defs_and_con

    defs_path, con, units = categorical_defs_and_con(tmp_path)
    text = defs_path.read_text().replace(
        "    plan: {primary: revenue}",
        "    plan: {primary: {metric: revenue, decision_method: {name: scalar, "
        "variance_reduction: cuped, covariates: [{name: pre_tenure, kind: continuous, "
        "property: tenure, source: events, missing: zero}]}}}\n"
        "    breakouts: [{property: region, source: events}]",
    )
    defs_path.write_text(
        text.replace(
            "    design:\n"
            "      mechanism: observational\n"
            "      covariates:\n"
            "        - {property: tenure, source: events}\n"
            "        - {property: region, source: events}\n",
            "",
        )
    )
    native = Analysis.from_definitions("cat_test", defs_path, con)
    frame = Analysis.from_unit_panel(
        pa.Table.from_pylist(
            [
                {
                    "unit": unit,
                    "group": values["variant"],
                    "day": 1,
                    "segment": values["region"],
                    "pre_tenure": values["tenure"],
                    "revenue": values["revenue"],
                }
                for unit, values in units.items()
            ]
        ),
        unit="unit",
        group="group",
        date="day",
        control="control",
        metrics=[
            MetricSpec(
                name="revenue",
                decision_method=Method.cuped(
                    name="scalar",
                    covariates=CupedCovariate(name="pre_tenure", kind="continuous", missing="zero"),
                ),
            )
        ],
        breakouts=["segment"],
    )
    native_rows = native.run_breakout()
    frame_rows = frame.run_breakout()
    assert [(row.dimension_value, row.method) for row in native_rows] == [
        (row.dimension_value, row.method) for row in frame_rows
    ]
    assert [row.require_lift().value for row in native_rows] == pytest.approx(
        [row.require_lift().value for row in frame_rows], rel=1e-10
    )


def test_public_breakout_retains_relative_ratio_when_absolute_projection_overflows() -> None:
    rows: list[dict[str, Any]] = []
    for group in ("control", "treatment"):
        for index in range(18):
            rows.append(
                {
                    "unit": f"{group}-{index}",
                    "group": group,
                    "day": 1,
                    "segment": "all",
                    "pre": ("a", "b", "c")[index % 3],
                    "revenue": 1e150 * (1.0 + index / 1000.0 + (group == "treatment") / 100.0),
                    "orders": 1e-200,
                }
            )
    metric = MetricSpec(
        name="revenue_per_order",
        type="ratio",
        numerator="revenue",
        denominator="orders",
        decision_method=Method.cuped_multicovariate(
            name="linear", covariates=CupedCovariate(name="pre", kind="categorical")
        ),
    )
    analysis = Analysis.from_unit_panel(
        pa.Table.from_pylist(rows),
        unit="unit",
        group="group",
        date="day",
        control="control",
        metrics=[metric],
        breakouts=["segment"],
    )

    (row,) = analysis.run_breakout()

    assert row.lift is not None
    assert row.abs_diff is None and row.abs_se is None
    assert row.absolute_error_code == "estimation.cuped_linear.nonfinite_projection"
    assert row.absolute_error_context is not None
    assert set(row.absolute_error_context) == {"value", "se", "dof"}
    assert row.discovery is True
    (headline,) = analysis.run()
    assert isinstance(headline, LiftEstimate)
    assert headline.absolute_error_code == row.absolute_error_code
    assert headline.absolute_error_context == row.absolute_error_context
    import copy
    import pickle

    from increment.breakout.estimates import BreakoutEstimate

    for estimate in (headline, row):
        model = BreakoutEstimate if estimate is row else LiftEstimate
        for restored in (
            model.model_validate_json(estimate.model_dump_json()),
            copy.deepcopy(estimate),
            pickle.loads(pickle.dumps(estimate)),
        ):
            assert isinstance(restored, (LiftEstimate, BreakoutEstimate))
            assert restored.absolute_error_code == estimate.absolute_error_code
            assert restored.absolute_error_context == estimate.absolute_error_context
            if isinstance(estimate, LiftEstimate):
                assert isinstance(restored, LiftEstimate)
                assert restored.p_value() == pytest.approx(estimate.p_value())
            assert restored.require_lift().value == pytest.approx(estimate.require_lift().value)
            with pytest.raises(TypeError):
                cast(dict[str, object], restored.absolute_error_context)["value"] = 0


def test_public_breakout_localizes_nonpositive_adjusted_arm_with_positive_raw_mean() -> None:
    rows: list[dict[str, Any]] = []
    offsets = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    for segment in ("north", "south"):
        for group in ("control", "treatment"):
            x_center, y_center = (
                (0.0, 10.0)
                if group == "control"
                else ((0.0, 12.0) if segment == "north" else (100.0, 5.0))
            )
            for index in range(21):
                offset = offsets[index % len(offsets)]
                rows.append(
                    {
                        "unit": f"{segment}-{group}-{index}",
                        "group": group,
                        "day": 1,
                        "segment": segment,
                        "pre": x_center + offset,
                        "revenue": y_center + offset,
                    }
                )
    metric = MetricSpec(
        name="revenue",
        decision_method=Method.cuped_multicovariate(
            name="linear", covariates=CupedCovariate(name="pre", kind="continuous")
        ),
    )
    analysis = Analysis.from_unit_panel(
        pa.Table.from_pylist(rows),
        unit="unit",
        group="group",
        date="day",
        control="control",
        metrics=[metric],
        breakouts=["segment"],
        plan=AnalysisPlan(view_multiplicity=MultiplicitySpec(correction="none")),
    )

    with pytest.warns():
        result = analysis.run_breakout()

    by_cell = {(row.dimension_value, row.group_id): row for row in result}
    assert by_cell[("north", "treatment")].lift is not None
    assert by_cell[("south", "treatment")].lift is None
    assert by_cell[("south", "treatment")].excluded == "estimation_failed"
    refused = by_cell[("south", "treatment")]
    assert refused.failure_code == "estimation.cuped_linear.nonpositive_component"
    assert refused.failure_context == {
        "family": "outcome",
        "arm": "treatment",
        "value": pytest.approx(-45.0),
    }
    assert refused.sampling_available is False


@pytest.mark.integration
@pytest.mark.slow
def test_native_breakout_qr_never_uses_unit_frame(tmp_path, monkeypatch):
    from increment.query.native_source import DefinitionsMomentSource
    from tests.covariate_cases import _CATEGORICAL_DESIGN, categorical_defs_and_con

    path, con, _units = categorical_defs_and_con(tmp_path)
    path.write_text(
        path.read_text()
        .replace(_CATEGORICAL_DESIGN, "")
        .replace(
            "    plan: {primary: revenue}",
            "    plan: {primary: {metric: revenue, "
            "decision_method: {name: linear, variance_reduction: cuped_multicovariate, covariates: "
            "[{name: region, kind: categorical, property: region, source: events}]}}}\n"
            "    breakouts: [{property: region, source: events}]",
        )
    )
    analysis = Analysis.from_definitions("cat_test", path, con, store="none")
    methods: dict[str, Any] = {
        "sensitivity_methods": [
            Method.cuped_multicovariate(
                name="linear-sensitivity",
                covariates=CupedCovariate(name="region", kind="categorical"),
            )
        ],
    }
    baseline = analysis.run_breakout(**methods)

    def forbidden_unit_frame(*args, **kwargs):
        pytest.fail("warehouse mixed-CUPED breakout fetched a per-unit frame")

    monkeypatch.setattr(DefinitionsMomentSource, "unit_frame", forbidden_unit_frame)
    try:
        after = analysis.run_breakout(**methods)
        assert [(r.dimension_value, r.method) for r in after] == [
            (r.dimension_value, r.method) for r in baseline
        ]
        assert [r.require_lift().value for r in after] == pytest.approx(
            [r.require_lift().value for r in baseline], rel=1e-10, abs=1e-10
        )
    finally:
        analysis.close()
        con.disconnect()


@pytest.mark.slow
def test_triggered_native_breakout_linear_matches_triggered_unit_oracle(tmp_path):
    import datetime as dt

    import ibis
    import narwhals as nw

    from increment import SourceSnapshotEvidence
    from increment.estimation.engine import estimate_cuped_linear
    from tests.analysis_factory import _native_source
    from tests.test_analysis_trigger import _defs_yaml, _events, _with_preassignment_region

    path = tmp_path / "breakout-trigger.yaml"
    text = (
        _defs_yaml()
        .replace(
            "      - name: saw_surface\n        column: null",
            "      - name: saw_surface\n        column: null\n    properties:\n"
            "      - {name: region, column: region, dtype: string, as_of: pre_exposure}",
        )
        .replace(
            "    plan:\n      secondaries: [revenue]",
            "    plan: {primary: {metric: revenue, "
            "decision_method: {name: linear, variance_reduction: cuped_multicovariate, covariates: "
            "[{name: region, kind: categorical, property: region, source: events}]}}, secondaries: [median]}\n"
            "    breakouts: [{property: region, source: events}]",
        )
    )
    text = text.replace(
        "experiments:",
        "  - name: median\n    type: quantile\n"
        "    quantile: 0.5\n    entity: user_id\n    fact: revenue\n"
        "    aggregation: sum\n    window_days: 7\nexperiments:",
    )
    path.write_text(text)
    con = ibis.duckdb.connect()
    con.create_table(
        "events", obj=_with_preassignment_region(_events(n_per_arm=100, trigger_rate=0.5))
    )
    analysis = Analysis.from_definitions(
        "exp",
        path,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
            {"events": dt.datetime(2024, 1, 16, tzinfo=dt.UTC)},
        ),
    )
    try:
        (headline,) = analysis.run(metrics=["revenue"], population="triggered")
        companion = next(
            row
            for row in analysis.run(metrics=["revenue", "median"], population="triggered")
            if row.metric == "revenue"
        )
        assert isinstance(headline, LiftEstimate) and isinstance(companion, LiftEstimate)
        assert companion.require_lift().value == pytest.approx(
            headline.require_lift().value, rel=1e-10
        )
        assert companion.analysis_population == "triggered"
        source = _native_source(analysis)
        metric = next(m for m in source.context.metrics if m.name == "revenue")
        declarations = cast(
            tuple[CupedCovariate, ...],
            next(c.covariates for c in source.context.configs if c.metric.name == "revenue"),
        )
        frame = nw.from_native(
            source.unit_frame(metric, covariates=("region",), population="triggered"),
            eager_only=True,
        )
        actual = analysis.run_breakout(metrics=["revenue"], population="triggered")
        for row in actual:
            cell = frame.filter(nw.col("region") == row.dimension_value)
            expected = estimate_cuped_linear(
                metric,
                cell.to_native(),
                control_group="C",
                covariates=declarations,
                method=Method.cuped_multicovariate(name="linear", covariates=declarations),
                method_role="decision",
                alpha=0.05,
                alternative="two-sided",
            ).results[0]
            assert row.analysis_population == "triggered"
            assert row.require_lift().value == pytest.approx(
                expected.require_lift().value, rel=1e-10
            )
            assert row.n_control is not None and row.n_treat is not None
            assert row.n_control + row.n_treat == cell.shape[0]
    finally:
        analysis.close()
        con.disconnect()


def _triggered_linear_artifact(tmp_path):
    import datetime as dt

    import ibis

    from increment import SourceSnapshotEvidence
    from increment.query.artifact_contract import unit_day_artifact_extension_catalog
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.loader import load
    from tests.test_analysis_trigger import _defs_yaml, _events, _with_preassignment_region

    path = tmp_path / "triggered-cuped-artifact.yaml"
    text = (
        _defs_yaml()
        .replace(
            "      - name: saw_surface\n        column: null",
            "      - name: saw_surface\n        column: null\n    properties:\n"
            "      - {name: region, column: region, dtype: string, as_of: pre_exposure}",
        )
        .replace(
            "    plan:\n      secondaries: [revenue]",
            "    plan: {primary: {metric: revenue, "
            "decision_method: {name: linear, variance_reduction: cuped_multicovariate, covariates: "
            "[{name: region, kind: categorical, property: region, source: events}]}}}\n"
            "    breakouts: [{property: region, source: events}]",
        )
    )
    path.write_text(text)
    connection = ibis.duckdb.connect()
    connection.create_table(
        "events", obj=_with_preassignment_region(_events(n_per_arm=100, trigger_rate=0.5))
    )
    source_evidence = SourceSnapshotEvidence(
        dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
        {"events": dt.datetime(2024, 1, 16, tzinfo=dt.UTC)},
    )
    native = Analysis.from_definitions(
        "exp", path, connection, source_snapshot_evidence=source_evidence
    )
    definitions = load(path)
    experiment = definitions.experiment("exp")
    assert experiment is not None
    context = artifact_context(definitions, experiment, "error")
    store = WarehouseArtifactStore(connection, schema_name="triggered_cuped")
    required_kinds = {
        "trigger_population",
        "trigger_measure_stats",
        "assignment_counts",
        "unit_covariate_level",
        "breakout_dimension",
    }
    extensions = [
        entry.request
        for entry in unit_day_artifact_extension_catalog(context)
        if entry.request.kind in required_kinds
    ]
    reference = native.publish_unit_day_artifact(store, extensions=extensions)
    adopted = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    return native, adopted, connection


@pytest.mark.slow
def test_triggered_cuped_definition_and_artifact_runs_match(tmp_path):
    from tests.parity_harness.comparison import nested_close

    native, artifact, connection = _triggered_linear_artifact(tmp_path)

    def stats(rows):
        return {
            (
                row.metric,
                row.group_id,
                row.method,
                row.method_role,
                getattr(row, "dimension_value", None),
            ): (
                row.lift.value if row.lift is not None else None,
                row.lift.log_se if row.lift is not None else None,
                row.dof,
                (row.lift.lb, row.lift.ub) if row.lift is not None else None,
            )
            for row in rows
        }

    try:
        assert nested_close(
            stats(native.run(metrics=["revenue"], population="triggered")),
            stats(artifact.run(metrics=["revenue"], population="triggered")),
        )
        assert nested_close(
            stats(native.run_breakout(metrics=["revenue"], population="triggered")),
            stats(artifact.run_breakout(metrics=["revenue"], population="triggered")),
        )
    finally:
        artifact.close()
        native.close()
        connection.disconnect()


def test_breakout_equivalent_linear_methods_share_one_fit_per_cell(monkeypatch):
    import increment.estimation.cuped.multicovariate as cuped_multicovariate

    original = cuped_multicovariate.fit_cuped_multicovariate
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(cuped_multicovariate, "fit_cuped_multicovariate", counted)
    rows = _breakout_analysis().run_breakout(
        sensitivity_methods=[
            Method.cuped_multicovariate(
                name="other-linear",
                covariates=CupedCovariate(name="pre", kind="categorical"),
            )
        ]
    )
    assert all(r.lift is not None for r in rows)
    assert {(r.dimension_value, r.method) for r in rows} == {
        (segment, method) for segment in ("north", "south") for method in ("linear", "other-linear")
    }
    assert calls == 2


@pytest.mark.integration
def test_breakout_unit_evidence_keeps_qualified_source_and_distinct_covariate(tmp_path):
    import datetime as dt
    import warnings

    import yaml

    from tests.covariate_cases import categorical_defs_and_con

    path, con, units = categorical_defs_and_con(tmp_path)
    definitions = yaml.safe_load(path.read_text())
    definitions["fact_sources"].append(
        {
            "name": "profiles",
            "sql": "SELECT * FROM profiles",
            "timestamp_column": "ts",
            "entities": ["user_id"],
            "facts": [{"name": "profile", "column": None}],
            "properties": [
                {"name": "region", "column": "region", "dtype": "string", "as_of": "pre_exposure"}
            ],
        }
    )
    experiment = definitions["experiments"][0]
    experiment.pop("design")
    experiment["breakouts"] = [
        {"property": "region", "source": source} for source in ("events", "profiles")
    ]
    experiment["plan"] = {
        "primary": {
            "metric": "revenue",
            "decision_method": {
                "name": "linear",
                "variance_reduction": "cuped_multicovariate",
                "covariates": [
                    {
                        "name": "region",
                        "kind": "continuous",
                        "property": "tenure",
                        "source": "events",
                    }
                ],
            },
        }
    }
    profiles = {unit: ("a" if int(unit[1:]) % 2 else "b") for unit in units}
    con.create_table(
        "profiles",
        obj=[
            {"user_id": unit, "ts": dt.datetime(2025, 7, 20), "region": value, "event": "profile"}
            for unit, value in profiles.items()
        ],
    )
    path.write_text(yaml.safe_dump(definitions))
    analysis = Analysis.from_definitions("cat_test", path, con)
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            actual = analysis.run_breakout()
        for source in ("events", "profiles"):
            oracle = Analysis.from_unit_panel(
                pa.Table.from_pylist(
                    [
                        {
                            "unit": unit,
                            "group": values["variant"],
                            "day": 1,
                            "revenue": values["revenue"],
                            "region": values["tenure"],
                            "segment": values["region"] if source == "events" else profiles[unit],
                        }
                        for unit, values in units.items()
                    ]
                ),
                unit="unit",
                group="group",
                date="day",
                control="control",
                metrics=[
                    MetricSpec(
                        name="revenue",
                        decision_method=Method.cuped_multicovariate(
                            name="linear",
                            covariates=CupedCovariate(name="region", kind="continuous"),
                        ),
                    )
                ],
                breakouts=["segment"],
            )
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                expected = {row.dimension_value: row for row in oracle.run_breakout()}
            qualified = [row for row in actual if row.source == source]
            assert {row.dimension_value for row in qualified} == set(expected)
            for row in qualified:
                reference = expected[row.dimension_value]
                assert row.n_treat == reference.n_treat and row.n_control == reference.n_control
                assert row.failure_code == reference.failure_code
                if reference.lift is not None:
                    assert row.require_lift().value == pytest.approx(
                        reference.require_lift().value, rel=1e-10
                    )
    finally:
        analysis.close()
        con.disconnect()


def test_native_breakout_override_without_covariates_refuses_before_queries(tmp_path, monkeypatch):
    from increment.errors import CodedError
    from tests.covariate_cases import _CATEGORICAL_DESIGN, categorical_defs_and_con

    path, con, _units = categorical_defs_and_con(tmp_path)
    path.write_text(
        path.read_text()
        .replace(_CATEGORICAL_DESIGN, "")
        .replace(
            "    plan: {primary: revenue}",
            "    plan: {primary: revenue}\n    breakouts: [{property: region, source: events}]",
        )
    )
    analysis = Analysis.from_definitions("cat_test", path, con)
    materializations = 0
    original = con.to_pyarrow

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal materializations
        materializations += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(con, "to_pyarrow", counted)
    try:
        with pytest.raises(CodedError) as raised:
            Method.cuped_multicovariate(covariates=())
        assert raised.value.code == "estimation.engine.method_covariates"
        assert materializations == 0
    finally:
        analysis.close()
        con.disconnect()


def test_breakout_covariate_named_like_internal_dimension_keeps_its_values() -> None:
    rows: list[dict[str, Any]] = []
    for segment in ("north", "south"):
        for group in ("control", "treatment"):
            for index in range(18):
                rows.append(
                    {
                        "unit": f"{segment}-{group}-{index}",
                        "group": group,
                        "day": 1,
                        "segment": segment,
                        "pre": ("a", "b", "c")[index % 3],
                        "revenue": float(index % 3 * 4 + index + (group == "treatment") * 2),
                    }
                )
    table = pa.Table.from_pylist(rows)

    def run(covariate: str) -> dict[str, Any]:
        frame = table.rename_columns(
            [covariate if name == "pre" else name for name in table.column_names]
        )
        metric = MetricSpec(
            name="revenue",
            decision_method=Method.cuped_multicovariate(
                name="linear", covariates=CupedCovariate(name=covariate, kind="categorical")
            ),
        )
        analysis = Analysis.from_unit_panel(
            frame,
            unit="unit",
            group="group",
            date="day",
            control="control",
            metrics=[metric],
            breakouts=["segment"],
        )
        return {row.dimension_value: row for row in analysis.run_breakout()}

    reference = run("pre")
    colliding = run("__breakout_dimension")
    assert set(colliding) == set(reference) == {"north", "south"}
    for segment, expected in reference.items():
        actual = colliding[segment]
        assert actual.require_lift().value == pytest.approx(expected.require_lift().value)
        assert actual.require_lift().lb == pytest.approx(expected.require_lift().lb)
        assert actual.discovery == expected.discovery


def test_readout_rows_forward_projection_refusal_sidecars() -> None:
    from increment.tables import estimates_to_readout

    rows: list[dict[str, Any]] = []
    for group in ("control", "treatment"):
        for index in range(18):
            rows.append(
                {
                    "unit": f"{group}-{index}",
                    "group": group,
                    "day": 1,
                    "segment": "all",
                    "pre": ("a", "b", "c")[index % 3],
                    "revenue": 1e150 * (1.0 + index / 1000.0 + (group == "treatment") / 100.0),
                    "orders": 1e-200,
                }
            )
    metric = MetricSpec(
        name="revenue_per_order",
        type="ratio",
        numerator="revenue",
        denominator="orders",
        decision_method=Method.cuped_multicovariate(
            name="linear", covariates=CupedCovariate(name="pre", kind="categorical")
        ),
    )
    analysis = Analysis.from_unit_panel(
        pa.Table.from_pylist(rows),
        unit="unit",
        group="group",
        date="day",
        control="control",
        metrics=[metric],
        breakouts=["segment"],
    )
    (headline,) = analysis.run()
    (segment,) = analysis.run_breakout()
    assert isinstance(headline, LiftEstimate)
    for estimate in (headline, segment):
        (converted,) = estimates_to_readout([estimate])
        assert converted["lift"] == pytest.approx(estimate.require_lift().value)
        assert converted["failure_code"] is None
        assert converted["abs_lb"] is None and converted["abs_ub"] is None
        assert converted["absolute_error_code"] == "estimation.cuped_linear.nonfinite_projection"
        assert converted["absolute_error_context"] == estimate.absolute_error_context
        assert converted["relative_error_code"] is None
        assert converted["relative_error_context"] is None


def _ratio_rows_with_dead_arm(
    *,
    dead_control: bool = False,
    groups: tuple[str, ...] = ("control", "treatment", "dead"),
    zero_groups: tuple[str, ...] = ("dead",),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in ("north", "south"):
        for group in groups:
            zero = group in zero_groups or (dead_control and group == "control")
            for index in range(18):
                rows.append(
                    {
                        "unit": f"{segment}-{group}-{index}",
                        "group": group,
                        "day": 1,
                        "segment": segment,
                        "pre": ("a", "b", "c")[index % 3],
                        "revenue": float(10 + index % 5 + (group == "treatment") * 2),
                        "orders": 0.0 if zero else float(2 + index % 3),
                    }
                )
    return rows


def _ratio_analysis(
    rows: list[dict[str, Any]],
    method: Method,
    *,
    include_linear_sensitivity: bool = False,
) -> Analysis:
    metric = MetricSpec(
        name="rpo",
        type="ratio",
        numerator="revenue",
        denominator="orders",
        decision_method=method,
        sensitivity_methods=(
            (
                Method.cuped_multicovariate(
                    name="linear",
                    covariates=CupedCovariate(name="pre", kind="categorical"),
                ),
            )
            if include_linear_sensitivity
            else ()
        ),
    )
    return Analysis.from_unit_panel(
        pa.Table.from_pylist(rows),
        unit="unit",
        group="group",
        date="day",
        control="control",
        metrics=[metric],
        breakouts=["segment"],
    )


def test_zero_denominator_arm_is_withheld_like_the_unadjusted_path() -> None:
    linear = _ratio_analysis(
        _ratio_rows_with_dead_arm(),
        Method.cuped_multicovariate(
            name="linear", covariates=CupedCovariate(name="pre", kind="categorical")
        ),
    )
    unadjusted = _ratio_analysis(_ratio_rows_with_dead_arm(), Method(name="unadjusted"))

    def by_arm(analysis: Analysis) -> dict[str, LiftEstimate]:
        with pytest.warns():
            rows = list(analysis.run())
        estimates = [row for row in rows if isinstance(row, LiftEstimate)]
        assert len(estimates) == len(rows)
        return {row.group_id: row for row in estimates}

    linear_run = by_arm(linear)
    unadjusted_run = by_arm(unadjusted)
    assert set(linear_run) == set(unadjusted_run) == {"dead", "treatment"}
    assert linear_run["treatment"].lift is not None
    assert linear_run["treatment"].failure_code is None
    dead = linear_run["dead"]
    assert dead.lift is None
    assert dead.failure_code == unadjusted_run["dead"].failure_code
    assert dead.failure_context == {"metric": "rpo", "group_id": "dead", "d_bar": 0.0}
    assert dead.sampling_available is False

    with pytest.warns():
        linear_cells = {(row.dimension_value, row.group_id): row for row in linear.run_breakout()}
    with pytest.warns():
        unadjusted_cells = {
            (row.dimension_value, row.group_id): row for row in unadjusted.run_breakout()
        }
    assert set(linear_cells) == set(unadjusted_cells)
    for key, reference in unadjusted_cells.items():
        cell = linear_cells[key]
        assert cell.excluded == reference.excluded
        assert (cell.lift is None) == (reference.lift is None)
    assert linear_cells[("north", "dead")].excluded == "nonpositive_mean"
    adjusted = linear_cells[("north", "treatment")].require_lift()
    raw = unadjusted_cells[("north", "treatment")].require_lift()
    assert adjusted.log_se is not None and raw.log_se is not None
    assert adjusted.log_se < raw.log_se


def test_zero_denominator_control_fails_every_linear_contrast_without_abort() -> None:
    from increment.errors import CodedError

    linear = _ratio_analysis(
        _ratio_rows_with_dead_arm(dead_control=True),
        Method.cuped_multicovariate(
            name="linear", covariates=CupedCovariate(name="pre", kind="categorical")
        ),
    )
    unadjusted = _ratio_analysis(
        _ratio_rows_with_dead_arm(dead_control=True), Method(name="unadjusted")
    )
    for analysis in (linear, unadjusted):
        with pytest.raises(CodedError) as raised, pytest.warns():
            analysis.run()
        assert raised.value.code == "readout.estimate_lift_every"
        if analysis is linear:
            failures = cast(list[dict[str, Any]], raised.value.context["failures"])
            assert {failure["group_id"] for failure in failures} == {"dead", "treatment"}
            assert {failure["context"]["group_id"] for failure in failures} == {"control"}
            assert {failure["code"] for failure in failures} == {
                "estimation.variance.ratio_moments_nonpositive_denominator_mean"
            }


@pytest.mark.parametrize(
    "zero_groups",
    [("treatment",), ("treatment", "dead")],
    ids=["single_treatment", "every_non_control_arm"],
)
def test_every_withheld_contrast_arm_refuses_exactly_like_unadjusted(
    zero_groups: tuple[str, ...],
) -> None:
    from increment.errors import CodedError

    groups = ("control", *zero_groups)
    linear = _ratio_analysis(
        _ratio_rows_with_dead_arm(groups=groups, zero_groups=zero_groups),
        Method.cuped_multicovariate(
            name="linear", covariates=CupedCovariate(name="pre", kind="categorical")
        ),
    )
    unadjusted = _ratio_analysis(
        _ratio_rows_with_dead_arm(groups=groups, zero_groups=zero_groups),
        Method(name="unadjusted"),
    )
    outcomes: dict[str, list[dict[str, Any]]] = {}
    for label, analysis in (("linear", linear), ("unadjusted", unadjusted)):
        with pytest.raises(CodedError) as raised, pytest.warns():
            analysis.run()
        assert raised.value.code == "readout.estimate_lift_every"
        outcomes[label] = cast(list[dict[str, Any]], raised.value.context["failures"])
    assert {failure["group_id"] for failure in outcomes["linear"]} == set(zero_groups)
    for label in ("linear", "unadjusted"):
        assert {failure["code"] for failure in outcomes[label]} == {
            "estimation.variance.ratio_moments_nonpositive_denominator_mean"
        }
        assert {failure["context"]["group_id"] for failure in outcomes[label]} == set(zero_groups)
    with pytest.warns():
        linear_cells = {
            (row.dimension_value, row.group_id): row.excluded for row in linear.run_breakout()
        }
    with pytest.warns():
        unadjusted_cells = {
            (row.dimension_value, row.group_id): row.excluded for row in unadjusted.run_breakout()
        }
    assert linear_cells == unadjusted_cells
    assert set(linear_cells.values()) == {"nonpositive_mean"}


def _dead_arm_rows_with_unsupported_sibling() -> list[dict[str, Any]]:
    rows = _ratio_rows_with_dead_arm()
    for row in rows:
        if row["group"] == "treatment" and row["unit"].endswith("-0"):
            row["pre"] = "rare"
    return rows


def test_withheld_arm_keeps_its_refusal_when_the_sibling_fit_fails() -> None:
    from increment.errors import CodedError

    denominator_code = "estimation.variance.ratio_moments_nonpositive_denominator_mean"
    expected = {
        "dead": (denominator_code, {"metric": "rpo", "group_id": "dead", "d_bar": 0.0}),
        "treatment": ("estimation.cuped_linear.level_support", None),
    }
    linear = Method.cuped_multicovariate(
        name="linear", covariates=CupedCovariate(name="pre", kind="categorical")
    )

    # Every decision cell refused: the readout refuses as a whole, carrying each arm's own reason.
    decision = _ratio_analysis(_dead_arm_rows_with_unsupported_sibling(), linear)
    with pytest.raises(CodedError) as raised, pytest.warns():
        decision.run()
    assert raised.value.code == "readout.estimate_lift_every"
    failures = {
        failure["group_id"]: failure
        for failure in cast(list[dict[str, Any]], raised.value.context["failures"])
    }
    assert set(failures) == set(expected)
    for arm, (code, context) in expected.items():
        assert failures[arm]["code"] == code
        if context is not None:
            assert dict(failures[arm]["context"]) == context
    assert failures["treatment"]["context"]["level"] == "rare"

    # A valid decision alongside: the linear sensitivity rows each keep their own refusal.
    sensitivity = _ratio_analysis(
        _dead_arm_rows_with_unsupported_sibling(),
        Method(name="unadjusted"),
        include_linear_sensitivity=True,
    )
    with pytest.warns():
        results = sensitivity.run(sensitivity_methods=[linear])
    by_cell = {(row.group_id, row.method): row for row in results if isinstance(row, LiftEstimate)}
    assert by_cell[("treatment", "unadjusted")].lift is not None
    for arm, (code, context) in expected.items():
        row = by_cell[(arm, "linear")]
        assert row.lift is None and row.failure_code == code
        if context is not None:
            assert row.failure_context == context
    unsupported = by_cell[("treatment", "linear")].failure_context
    assert unsupported is not None and unsupported["level"] == "rare"


def test_withheld_arm_reports_requested_posterior_unavailable() -> None:
    from increment.estimation.inference import Normal

    analysis = _ratio_analysis(
        _ratio_rows_with_dead_arm(),
        Method.cuped_multicovariate(
            name="linear", covariates=CupedCovariate(name="pre", kind="categorical")
        ),
    )

    with pytest.warns():
        results = analysis.run(prior=Normal(mu=0.0, sigma=0.1))

    by_arm = {row.group_id: row for row in results if isinstance(row, LiftEstimate)}
    for arm in ("dead", "treatment"):
        row = by_arm[arm]
        assert row.posterior_available is False
        assert row.posterior_reason_code == (
            "estimation.results.lift.t_posterior_sufficient_statistics"
        )
        assert row.posterior_reason_context is not None
        assert row.posterior_reason_context["group_id"] == arm
    assert by_arm["treatment"].lift is not None
    assert by_arm["dead"].lift is None
    assert results.metadata is not None
    posterior_by_arm = {
        record.cell.group_id: record.posterior
        for record in results.metadata.cells
        if record.posterior is not None
    }
    assert posterior_by_arm["dead"].available is False
    assert posterior_by_arm["dead"].reason_code == by_arm["dead"].posterior_reason_code
    assert posterior_by_arm["dead"].reason_context == by_arm["dead"].posterior_reason_context
