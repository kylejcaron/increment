"""Public DuckDB journey: definitions -> unit-day artifact -> portable moments -> replay."""

from __future__ import annotations

import json
import math
import re
from typing import Any, cast

import ibis
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from increment import (
    Method,
    Scope,
    from_definitions,
    from_frame,
    from_moments,
    from_unit_day_artifact,
)
from increment.artifacts import artifact_context, publish_unit_day_artifact
from increment.errors import CodedError
from increment.estimation.results import LiftEstimate
from increment.frame import MetricSpec
from increment.semantics.models import CupedCovariateBinding
from increment.workflows import export_moments
from tests.analysis_factory import view_readout

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_CATEGORICAL = Method.cuped_multicovariate(
    name="segments",
    covariates=(
        CupedCovariateBinding(
            name="country",
            kind="categorical",
            property="country",
            source="events",
            missing="impute-indicator",
        ),
        CupedCovariateBinding(
            name="device", kind="categorical", property="device", source="events"
        ),
    ),
)
_COMPANION = Method.cuped(name="cuped")
_EXPLICIT_COMPANION = Method.cuped(
    name="history_b",
    covariates=CupedCovariateBinding(
        name="history_b", kind="continuous", property="history_b", source="events"
    ),
)
# Columns of transferred joint-cell outcome aggregates: ref_<f>, first_<f>, cross_<f>_<g>.
_OUTCOME_CELL = re.compile(r"(ref|first|cross)_\d")


def _units(*, missing_country: bool = True) -> list[dict[str, Any]]:
    """Unequal crossed categories; spend is a continuous per-unit ratio denominator."""
    units = []
    for a, arm in enumerate(("control", "treatment")):
        for pattern, count in enumerate((41, 29, 37, 33)):
            for i in range(count):
                noise = math.sin(0.7 * i + a)
                units.append(
                    {
                        "user_id": f"{arm}-{pattern}-{i}",
                        "variant": arm,
                        "revenue": 8.0 + 0.5 * a + 0.2 * pattern + noise,
                        "spend": 3.0 + 0.1 * pattern + 0.1 * math.cos(1.3 * i + a),
                        "orders": 1,
                        "signups": 0,
                        "pre_revenue": 7.5 + 0.2 * pattern + 0.8 * noise + 0.05 * (i % 5),
                        "history_b": 2.0 + 0.1 * pattern + 0.6 * math.cos(0.3 * i + a),
                        "region": "north" if i % 2 else "south",
                        "country": (
                            None
                            if missing_country and i % 11 == 0
                            else ("A" if pattern < 2 else "B")
                        ),
                        "device": "mobile" if pattern % 2 else "desktop",
                        "new_user": "no",
                    }
                )
    return units


def _events(units) -> list[dict[str, Any]]:
    from tests.warehouse_cuped_cases import mixed_cuped_event_rows

    events = mixed_cuped_event_rows(units)
    by_user = {unit["user_id"]: unit for unit in units}
    for event in events:
        event["history_b"] = (
            by_user[event["user_id"]]["history_b"] if event["event"] == "history" else None
        )
    outcome = next(
        row["ts"]
        for row in events
        if row["event"] == "revenue" and row["value"] == units[0]["revenue"]
    )
    blank = dict.fromkeys(events[0])
    events.extend(
        {
            **blank,
            "user_id": unit["user_id"],
            "ts": outcome,
            "event": "spend",
            "value": unit["spend"],
        }
        for unit in units
    )
    return events


def _definitions(tmp_path, *, fixed_caps: bool, covariate_name: str = "country"):
    from tests.warehouse_cuped_cases import mixed_cuped_definitions_yaml

    data = yaml.safe_load(
        mixed_cuped_definitions_yaml(
            "events",
            breakout=True,
            scalar_cuped=True,
            n_pre_periods=3,
            fixed_winsorization=fixed_caps,
        )
    )
    data["fact_sources"][0]["facts"].append({"name": "spend", "column": "value"})
    data["fact_sources"][0]["properties"].append(
        {
            "name": "history_b",
            "column": "history_b",
            "dtype": "float",
            "as_of": "pre_exposure",
        }
    )
    data["metrics"].append(
        {
            "name": "revenue_per_spend",
            "type": "ratio",
            "entity": "user_id",
            "numerator": {"fact": "revenue", "aggregation": "sum", "window_days": 2},
            "denominator": {"fact": "spend", "aggregation": "sum", "window_days": 2},
            "preferred_direction": "increase",
        }
    )
    decision = _CATEGORICAL.model_dump(mode="json", exclude_none=True)
    decision["covariates"][0]["name"] = covariate_name
    companion = _COMPANION.model_dump(mode="json", exclude_none=True)
    explicit_companion = _EXPLICIT_COMPANION.model_dump(mode="json", exclude_none=True)
    plan = data["experiments"][0]["plan"]
    plan["primary"]["decision_method"] = decision
    plan["primary"]["sensitivity_methods"] = [companion, explicit_companion]
    guardrail = dict(decision)
    plan["guardrails"] = [{"metric": "revenue_per_spend", "decision_method": guardrail}]
    plan.pop("secondaries")
    path = tmp_path / "portable.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def _specs():
    """Metric shapes only: the cube's embedded plan supplies the declared methods and roles."""
    return [
        MetricSpec(name="revenue"),
        MetricSpec(
            name="revenue_per_spend", type="ratio", numerator="revenue", denominator="spend"
        ),
    ]


def _publish(con, native: tuple[Any, Any]) -> tuple[tuple[Any, Any], list[Any]]:
    from increment.query.artifact_contract import unit_day_artifact_extension_catalog
    from increment.query.session import WarehouseArtifactStore

    experiment, resolver = native
    store = WarehouseArtifactStore(con, schema_name="portable")
    context = artifact_context(experiment, resolver)
    requests = [entry.request for entry in unit_day_artifact_extension_catalog(context)]
    ref = publish_unit_day_artifact(experiment, resolver, store, extensions=requests)
    return from_unit_day_artifact(store, ref, expected_context=context), requests


def _guard_unit_transfers(monkeypatch, con, *, outcome_cells: bool) -> list[tuple[str, ...]]:
    """Refuse any connection transfer of a unit-keyed relation; record transferred columns."""
    transferred: list[tuple[str, ...]] = []

    def checked(real):
        def transfer(expression, *args, **kwargs):
            names = tuple(expression.schema().names) if hasattr(expression, "schema") else ()
            assert "unit_id" not in names, f"unit rows crossed the client boundary: {names}"
            if not outcome_cells:
                assert not any(_OUTCOME_CELL.match(name) for name in names)
            transferred.append(names)
            return real(expression, *args, **kwargs)

        return transfer

    monkeypatch.setattr(con, "to_pyarrow", checked(con.to_pyarrow))
    monkeypatch.setattr(con, "to_pyarrow_batches", checked(con.to_pyarrow_batches))
    for owner in _unit_frame_owners():
        monkeypatch.setattr(
            owner,
            "unit_frame",
            lambda *args, **kwargs: pytest.fail("portable export must not read unit frames"),
        )
    return transferred


def _design(path):
    from increment.semantics.loader import load

    experiment = load(path).experiment("mixed_cuped")
    assert experiment is not None
    return experiment.resolved_design()


def _unit_frame_owners():
    from increment.query.native_source import DefinitionsMomentSource
    from increment.query.source import _ArtifactFacadeSource

    return (DefinitionsMomentSource, _ArtifactFacadeSource)


def _cells(rows):
    return {
        (row.metric, row.method, row.group_id): row for row in rows if isinstance(row, LiftEstimate)
    }


@pytest.mark.parametrize("ingress", ["definitions", "artifact"])
def test_categorical_covariate_named_n_exports_identically(tmp_path, ingress):
    analyses = []
    connection = ibis.duckdb.connect()
    connection.create_table("events", obj=pa.Table.from_pylist(_events(_units())))
    try:
        for name in ("country", "n"):
            path = _definitions(tmp_path, fixed_caps=False, covariate_name=name)
            native_experiment, native_resolver = from_definitions(
                path, connection, experiment_name="mixed_cuped"
            )
            adopted_resolver = None
            try:
                source_experiment, source_resolver = native_experiment, native_resolver
                if ingress == "artifact":
                    (source_experiment, adopted_resolver), _ = _publish(
                        connection, (native_experiment, native_resolver)
                    )
                    source_resolver = adopted_resolver
                expected = _cells(view_readout(native_experiment, native_resolver))
                destination = tmp_path / f"{ingress}-{name}.parquet"
                export_moments(source_experiment, source_resolver, destination)
                rows = pq.read_table(destination).to_pylist()
                design = _design(path)
                experiment, resolver = from_moments(
                    rows, metrics=_specs(), design=design, experiment_id="mixed_cuped"
                )
                try:
                    analyses.append((_cells(view_readout(experiment, resolver)), expected))
                finally:
                    resolver.close()
            finally:
                if adopted_resolver is not None:
                    adopted_resolver.close()
                cast(Any, native_resolver).close()
    finally:
        connection.disconnect()
    reference = analyses[0][1]
    for _, expected in analyses[1:]:
        assert expected.keys() == reference.keys()
        for key, row in reference.items():
            comparison = expected[key]
            for field in ("value", "log_se", "lb", "ub"):
                assert getattr(comparison.require_lift(), field) == pytest.approx(
                    getattr(row.require_lift(), field), rel=1e-10, abs=1e-12
                )
    for actual, expected in analyses:
        assert actual.keys() == expected.keys()
        for key, row in expected.items():
            replayed = actual[key]
            for field in ("value", "log_se", "lb", "ub"):
                assert getattr(replayed.require_lift(), field) == pytest.approx(
                    getattr(row.require_lift(), field), rel=1e-10, abs=1e-12
                )


@pytest.mark.parametrize("ingress", ["artifact", "definitions"])
def test_definitions_artifact_moments_journey_replays_companions_without_unit_transfer(
    tmp_path, monkeypatch, ingress
):
    path = _definitions(tmp_path, fixed_caps=False)
    con = ibis.duckdb.connect()
    con.create_table("events", obj=pa.Table.from_pylist(_events(_units())))
    native_experiment, native_resolver = from_definitions(path, con, experiment_name="mixed_cuped")
    native = (native_experiment, native_resolver)
    adopted_resolver = None
    try:
        expected = _cells(view_readout(native_experiment, native_resolver))
        assert set(expected) == {
            ("revenue", "segments", "treatment"),
            ("revenue", "cuped", "treatment"),
            ("revenue", "history_b", "treatment"),
            ("revenue_per_spend", "segments", "treatment"),
        }
        transferred = _guard_unit_transfers(monkeypatch, con, outcome_cells=True)
        source_experiment, source_resolver = native
        if ingress == "artifact":
            (source_experiment, adopted_resolver), requests = _publish(con, native)
            assert {"cuped_preperiod", "breakout_dimension"} <= {item.kind for item in requests}
            source_resolver = adopted_resolver
        destination = tmp_path / "portable.parquet"
        export_moments(source_experiment, source_resolver, destination)
        # Joint cells and both method-scoped scalar families cross only as aggregates.
        assert any(_OUTCOME_CELL.match(name) for names in transferred for name in names)
        assert all("unit_id" not in names for names in transferred)
        rows = pq.read_table(destination).to_pylist()
        assert {row["moments_format"] for row in rows} == {13}
        envelopes = sorted(
            (payload["kind"], payload["metric"])
            for payload in (
                json.loads(row["cuped_evidence"])
                for row in rows
                if row.get("record_kind") == "cuped_evidence"
            )
        )
        assert envelopes == [
            ("categorical_cells", "revenue"),
            ("categorical_cells", "revenue_per_spend"),
            ("scalar_moments", "revenue"),
            ("scalar_moments", "revenue"),
        ]
        design = _design(path)
        experiment, resolver = from_moments(
            rows, metrics=_specs(), design=design, experiment_id="mixed_cuped"
        )
        try:
            actual = _cells(view_readout(experiment, resolver))
            with pytest.raises(CodedError) as breakout:
                view_readout(experiment, resolver, Scope(view="breakout", grain="total"))
        finally:
            resolver.close()
        assert breakout.value.code == "evidence.breakout_unavailable"
    finally:
        if adopted_resolver is not None:
            adopted_resolver.close()
        cast(Any, native_resolver).close()
        con.disconnect()
    assert actual.keys() == expected.keys()
    for key, row in expected.items():
        replayed = actual[key]
        assert (replayed.role, replayed.method_role) == (row.role, row.method_role)
        for field in ("value", "log_se", "lb", "ub"):
            assert getattr(replayed.require_lift(), field) == pytest.approx(
                getattr(row.require_lift(), field), rel=1e-10, abs=1e-12
            )
        for field in ("abs_diff", "abs_se", "abs_lb", "abs_ub"):
            assert getattr(replayed, field) == pytest.approx(
                getattr(row, field), rel=1e-10, abs=1e-12
            )


@pytest.mark.parametrize("ingress", ["artifact", "definitions"])
def test_fixed_caps_export_fits_with_warehouse_qr_and_replays_without_unit_transfer(
    tmp_path, monkeypatch, ingress
):
    path = _definitions(tmp_path, fixed_caps=True)
    con = ibis.duckdb.connect()
    con.create_table("events", obj=pa.Table.from_pylist(_events(_units())))
    native_experiment, native_resolver = from_definitions(path, con, experiment_name="mixed_cuped")
    native = (native_experiment, native_resolver)
    adopted_resolver = None
    try:
        expected = _cells(view_readout(native_experiment, native_resolver))
        capped = expected[("revenue", "segments", "treatment")]
        assert np.isfinite(capped.require_lift().value)
        _guard_unit_transfers(monkeypatch, con, outcome_cells=True)
        source_experiment, source_resolver = native
        if ingress == "artifact":
            (source_experiment, adopted_resolver), _ = _publish(con, native)
            source_resolver = adopted_resolver
        destination = tmp_path / "capped.parquet"
        export_moments(source_experiment, source_resolver, destination)
        rows = pq.read_table(destination).to_pylist()
        assert {row["moments_format"] for row in rows} == {13}
        envelopes = [
            json.loads(row["cuped_evidence"])
            for row in rows
            if row.get("record_kind") == "cuped_evidence"
        ]
        assert any(payload["kind"] == "fitted_multicovariate" for payload in envelopes)
        from increment.semantics.models import Winsorization

        metrics = _specs()
        metrics[0] = MetricSpec(
            name="revenue",
            winsorization=Winsorization(lower_value=7.0, upper_value=11.0),
        )
        experiment, resolver = from_moments(
            rows, metrics=metrics, design=_design(path), experiment_id="mixed_cuped"
        )
        try:
            actual = _cells(view_readout(experiment, resolver))
        finally:
            resolver.close()
    finally:
        if adopted_resolver is not None:
            adopted_resolver.close()
        cast(Any, native_resolver).close()
        con.disconnect()
    assert actual.keys() == expected.keys()
    for key, row in expected.items():
        replayed = actual[key]
        for field in ("value", "log_se", "lb", "ub"):
            assert getattr(replayed.require_lift(), field) == pytest.approx(
                getattr(row.require_lift(), field), rel=1e-10, abs=1e-12
            )
        for field in ("abs_diff", "abs_se", "abs_lb", "abs_ub"):
            assert getattr(replayed, field) == pytest.approx(
                getattr(row, field), rel=1e-10, abs=1e-12
            )


def test_fitted_envelope_replays_across_frame_native_and_artifact_sources(tmp_path, monkeypatch):
    import datetime as dt

    from increment.semantics.design import Randomized
    from increment.semantics.loader import load
    from increment.semantics.models import AnalysisPlan
    from tests.warehouse_cuped_cases import mixed_cuped_definitions_yaml

    units = _units(missing_country=False)
    path = tmp_path / "mixed.yaml"
    path.write_text(mixed_cuped_definitions_yaml("events"))
    experiment = load(path).experiment("mixed_cuped")
    assert experiment is not None
    binding = experiment.bindings["revenue"]
    assert binding.decision_method is not None
    method = Method.model_validate(binding.decision_method.model_dump())
    design = experiment.resolved_design()
    frame_metric = MetricSpec(name="revenue", decision_method=method)
    frame_design = Randomized(
        control_group="control", allocation={"control": 0.5, "treatment": 0.5}
    )
    frame_plan = AnalysisPlan(primary="revenue")
    summary = from_frame(
        pa.Table.from_pylist(units),
        shape="summary",
        unit="user_id",
        group="variant",
        metrics=[frame_metric],
        design=frame_design,
        plan=frame_plan,
    )
    panel_rows = [{**row, "day": dt.date(2025, 1, 10)} for row in units]
    panel = from_frame(
        pa.Table.from_pylist(panel_rows),
        shape="panel",
        unit="user_id",
        group="variant",
        date="day",
        metrics=[frame_metric],
        design=frame_design,
        plan=frame_plan,
    )
    con = ibis.duckdb.connect()
    con.create_table("events", obj=pa.Table.from_pylist(_events(units)))
    native_experiment, native_resolver = from_definitions(path, con, experiment_name="mixed_cuped")
    native = (native_experiment, native_resolver)
    adopted = None
    try:
        native_rows = _cells(view_readout(native_experiment, native_resolver, metrics=["revenue"]))
        _guard_unit_transfers(monkeypatch, con, outcome_cells=True)
        adopted, _ = _publish(con, native)
        sources = (summary, panel, native, adopted)
        for index, source in enumerate(sources):
            source_experiment, source_resolver = source
            expected = (
                native_rows
                if source is native or source is adopted
                else _cells(view_readout(source_experiment, source_resolver))
            )
            assert expected.keys() == native_rows.keys()
            for key, row in expected.items():
                reference = native_rows[key]
                for field in ("value", "log_se", "lb", "ub"):
                    assert getattr(row.require_lift(), field) == pytest.approx(
                        getattr(reference.require_lift(), field), rel=1e-10, abs=1e-12
                    )
            destination = tmp_path / f"fitted-{index}.parquet"
            export_moments(source_experiment, source_resolver, destination)
            rows = pq.read_table(destination).to_pylist()
            assert {row["moments_format"] for row in rows} == {13}
            envelopes = [
                json.loads(row["cuped_evidence"])
                for row in rows
                if row.get("record_kind") == "cuped_evidence"
            ]
            fitted = [
                payload
                for payload in envelopes
                if payload["kind"] == "fitted_multicovariate" and payload["metric"] == "revenue"
            ]
            assert len(fitted) == 1
            restored_experiment, restored_resolver = from_moments(
                rows,
                metrics=[frame_metric],
                design=design,
                plan=AnalysisPlan(primary="revenue"),
                experiment_id="mixed_cuped",
            )
            try:
                actual = _cells(
                    view_readout(restored_experiment, restored_resolver, metrics=["revenue"])
                )
            finally:
                restored_resolver.close()
            assert actual.keys() == expected.keys()
            for key, row in expected.items():
                replayed = actual[key]
                for field in ("value", "log_se", "lb", "ub"):
                    assert getattr(replayed.require_lift(), field) == pytest.approx(
                        getattr(row.require_lift(), field), rel=1e-10, abs=1e-12
                    )
    finally:
        if adopted is not None:
            adopted[1].close()
        summary[1].close()
        panel[1].close()
        cast(Any, native_resolver).close()
        con.disconnect()
