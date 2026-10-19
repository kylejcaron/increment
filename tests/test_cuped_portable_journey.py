"""Public DuckDB journey: definitions -> unit-day artifact -> portable moments -> replay."""

from __future__ import annotations

import json
import math
import re
from typing import Any

import ibis
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from increment import Analysis, Method
from increment.errors import CapabilityError, CodedError
from increment.estimation.results import LiftEstimate
from increment.frame import MetricSpec
from increment.semantics.models import CupedCovariateBinding

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_CATEGORICAL = Method.cuped_multicovariate(
    name="segments",
    covariates=(
        CupedCovariateBinding(
            name="country", kind="categorical", property="country", source="events"
        ),
        CupedCovariateBinding(
            name="device", kind="categorical", property="device", source="events"
        ),
    ),
)
_COMPANION = Method.cuped(name="cuped")
# Columns of transferred joint-cell outcome aggregates: ref_<f>, first_<f>, cross_<f>_<g>.
_OUTCOME_CELL = re.compile(r"(ref|first|cross)_\d")


def _units() -> list[dict[str, Any]]:
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
                        "region": "north" if i % 2 else "south",
                        "country": "A" if pattern < 2 else "B",
                        "device": "mobile" if pattern % 2 else "desktop",
                        "new_user": "no",
                    }
                )
    return units


def _events(units) -> list[dict[str, Any]]:
    from tests.warehouse_cuped_cases import mixed_cuped_event_rows

    events = mixed_cuped_event_rows(units)
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


def _definitions(tmp_path, *, fixed_caps: bool):
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
    plan = data["experiments"][0]["plan"]
    plan["primary"]["decision_method"] = decision
    plan["primary"]["sensitivity_methods"] = [_COMPANION.model_dump(mode="json", exclude_none=True)]
    plan["guardrails"] = [{"metric": "revenue_per_spend", "decision_method": decision}]
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


def _publish(con, native):
    from increment.query.artifact_contract import unit_day_artifact_extension_catalog
    from increment.query.session import WarehouseArtifactStore

    store = WarehouseArtifactStore(con, schema_name="portable")
    context = native.artifact_context
    requests = [entry.request for entry in unit_day_artifact_extension_catalog(context)]
    ref = native.publish_unit_day_artifact(store, extensions=requests)
    return Analysis.from_unit_day_artifact(store, ref, expected_context=context), requests


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


@pytest.mark.parametrize("ingress", ["artifact", "definitions"])
def test_definitions_artifact_moments_journey_replays_companions_without_unit_transfer(
    tmp_path, monkeypatch, ingress
):
    path = _definitions(tmp_path, fixed_caps=False)
    con = ibis.duckdb.connect()
    con.create_table("events", obj=pa.Table.from_pylist(_events(_units())))
    native = Analysis.from_definitions("mixed_cuped", path, con)
    adopted = None
    try:
        expected = _cells(native.run())
        assert set(expected) == {
            ("revenue", "segments", "treatment"),
            ("revenue", "cuped", "treatment"),
            ("revenue_per_spend", "segments", "treatment"),
        }
        transferred = _guard_unit_transfers(monkeypatch, con, outcome_cells=True)
        if ingress == "artifact":
            adopted, requests = _publish(con, native)
            assert {"cuped_preperiod", "breakout_dimension"} <= {item.kind for item in requests}
            source = adopted
        else:
            source = native
        destination = tmp_path / "portable.parquet"
        source.export(destination)
        # The joint cells crossed the same guarded connection that refuses unit rows.
        assert any(_OUTCOME_CELL.match(name) for names in transferred for name in names)
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
        ]
        design = _design(path)
        restored = Analysis.from_moments(
            rows, metrics=_specs(), design=design, experiment_id="mixed_cuped"
        )
        try:
            actual = _cells(restored.run())
            with pytest.raises(CodedError) as breakout:
                restored.run_breakout()
        finally:
            restored.close()
        assert breakout.value.code == "facade.analysis.operation"
    finally:
        if adopted is not None:
            adopted.close()
        native.close()
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
def test_fixed_caps_route_the_portable_export_to_general_qr_without_transfer(
    tmp_path, monkeypatch, ingress
):
    path = _definitions(tmp_path, fixed_caps=True)
    con = ibis.duckdb.connect()
    con.create_table("events", obj=pa.Table.from_pylist(_events(_units())))
    native = Analysis.from_definitions("mixed_cuped", path, con)
    adopted = None
    try:
        capped = _cells(native.run())[("revenue", "segments", "treatment")]
        assert np.isfinite(capped.require_lift().value)
        _guard_unit_transfers(monkeypatch, con, outcome_cells=False)
        source = native
        if ingress == "artifact":
            adopted, _ = _publish(con, native)
            source = adopted
        destination = tmp_path / "capped.parquet"
        with pytest.raises(CapabilityError) as raised:
            source.export(destination)
    finally:
        if adopted is not None:
            adopted.close()
        native.close()
        con.disconnect()
    assert raised.value.code == "query.cuped.general_qr_required"
    assert raised.value.context["reason"] == "outcome_domain"
    route = raised.value.context["route"]
    assert isinstance(route, str) and "from_unit_summary" in route
    assert not destination.exists()
