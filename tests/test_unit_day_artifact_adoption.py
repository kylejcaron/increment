from __future__ import annotations

import json
import math
from contextlib import contextmanager
from datetime import UTC, date, datetime
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import ibis
import ibis.expr.types as ir
import pytest

from increment.errors import CapabilityError
from increment.frame import MetricSpec
from increment.query.artifact_contract import (
    ArtifactContractError,
    ArtifactStore,
    _aggregate_mean_error_bound,
    _aggregate_sum_within_extrema,
    _aggregate_sum_within_extrema_sql,
    compile_unit_day_artifact_context,
)
from increment.query.artifact_digest import digest_relation, manifest_sha256
from increment.query.artifact_reader import ArtifactMomentSource
from increment.query.builders import (
    asof_group_summary,
    cohort_group_summary,
    daily_group_summary,
    first_exposures,
    group_summary,
    metric_events,
    panel_spine,
    post_exposure_stats,
    unit_day_panel,
    unit_totals,
    validate_unit_day_aggregate_rows,
    window_bound_stats,
    winsorize_unit_totals,
)
from increment.query.schemas import (
    UNIT_DAY_ARTIFACT_PRIMARY_KEYS,
    UNIT_DAY_ARTIFACT_RELATION_SCHEMAS,
)
from increment.semantics.artifact import (
    ArtifactRelationRef,
    BaseRelations,
    Freshness,
    MeasureManifest,
    RatioMetricMeasure,
    RelationLocator,
    SimpleMetricMeasure,
    UnitDayArtifactManifest,
    UnitDayArtifactRef,
)
from increment.semantics.models import (
    AnalysisPlan,
    Definitions,
    Experiment,
    Exposure,
    Fact,
    FactSource,
    MeanMetric,
    Measure,
    Metric,
    QuantileMetric,
    RatioMetric,
    RetentionMetric,
    Winsorization,
)
from tests.semantics.test_unit_day_artifact import _manifest


@pytest.fixture
def reader_snapshot():
    from increment._window import NO_DATA_SIGNAL

    manifest = _manifest()
    manifest = manifest.model_copy(
        update={
            "measures": tuple(
                measure.model_copy(update={"freshness": Freshness(loaded_through=NO_DATA_SIGNAL)})
                for measure in manifest.measures
            )
        }
    )
    exposures = [
        {
            "experiment_id": "exp",
            "unit_id": "u1",
            "group_id": "control",
            "first_exposure_ts": datetime(2025, 1, 1, tzinfo=UTC),
            "first_exposure_date": date(2025, 1, 1),
        },
        {
            "experiment_id": "exp",
            "unit_id": "u2",
            "group_id": "treatment",
            "first_exposure_ts": datetime(2025, 1, 2, tzinfo=UTC),
            "first_exposure_date": date(2025, 1, 2),
        },
    ]
    stats = [
        {
            "experiment_id": "exp",
            "unit_id": "u1",
            "ds": date(2025, 1, 2),
            "measure_key": "orders",
            "n_events": 1,
            "sum_value": 2.0,
            "min_value": 2.0,
            "max_value": 2.0,
        },
        {
            "experiment_id": "exp",
            "unit_id": "u2",
            "ds": date(2025, 1, 3),
            "measure_key": "orders",
            "n_events": 1,
            "sum_value": 4.0,
            "min_value": 4.0,
            "max_value": 4.0,
        },
    ]
    refs = {}
    for role, rows in (("exposures", exposures), ("measure_stats", stats)):
        digest = digest_relation(
            role,
            UNIT_DAY_ARTIFACT_RELATION_SCHEMAS[role],
            rows,
            primary_key=UNIT_DAY_ARTIFACT_PRIMARY_KEYS[role],
        )
        refs[role] = getattr(manifest.base, role).model_copy(
            update={
                "schema_sha256": digest.schema_sha256,
                "content_sha256": digest.content_sha256,
                "row_count": len(rows),
            }
        )
    manifest = manifest.model_copy(update={"base": manifest.base.model_copy(update=refs)})
    manifest = manifest.model_copy(update={"manifest_sha256": manifest_sha256(manifest)})
    ref = UnitDayArtifactRef(
        artifact_id=manifest.artifact_id,
        generation_id=manifest.generation_id,
        manifest=RelationLocator(name="manifest"),
        manifest_sha256=manifest.manifest_sha256,
    )

    class Snapshot:
        artifact_id = manifest.artifact_id
        generation_id = manifest.generation_id

        def __init__(self):
            self.calls = {"exposures": 0, "measure_stats": 0}
            self.executions = []

        def read_manifest(self, locator, *, expected_sha256):
            return manifest

        def verify_relation(self, relation, *, expected_role):
            self.calls[expected_role] += 1
            rows = exposures if expected_role == "exposures" else stats
            return ibis.memtable(rows)

        def execute(self, expression):
            assert isinstance(expression, ir.Table)
            self.executions.append(tuple(expression.columns))
            return con.to_pyarrow(expression)

    snapshot = Snapshot()

    class Store:
        def validate_locator(self, locator, **kwargs):
            return None

        @contextmanager
        def open_snapshot(self, fixed_ref):
            yield snapshot

    con = ibis.duckdb.connect()
    try:
        yield manifest, ref, snapshot, cast(ArtifactStore, Store())
    finally:
        con.disconnect()


@pytest.mark.slow
def test_artifact_reducer_uses_one_snapshot_handle_and_caches_relations(
    reader_snapshot, monkeypatch
) -> None:
    manifest, ref, _snapshot, store = reader_snapshot

    def direct_execution_forbidden(*args, **kwargs):
        pytest.fail("artifact queries must use the pinned snapshot executor")

    monkeypatch.setattr(ir.Expr, "execute", direct_execution_forbidden)
    source = ArtifactMomentSource.open(
        store,
        ref,
        expected_context=manifest.context,
        metrics=(MetricSpec(name="conversion", type="mean", value_column="orders"),),
    )
    metric = source.context.metrics[0]
    total = source.moments(metric)
    source.moments(metric, grain="daily")
    source.moments(metric, grain="asof")
    _artifact_rows_equal(source.moments(metric), total)
    assert {row["group_id"]: row["n"] for row in total} == {"control": 1, "treatment": 1}
    assert source.unit_counts() == {"control": 1, "treatment": 1}
    source.close()
    bad = ArtifactMomentSource.open(
        store,
        ref,
        expected_context=manifest.context,
        metrics=(MetricSpec(name="conversion", type="mean", value_column="orders", window_days=2),),
    )
    with pytest.raises(ArtifactContractError) as raised:
        _ = bad.context
    assert raised.value.code == "artifact.metric.binding_mismatch"
    bad.close()


@pytest.mark.slow
@pytest.mark.parametrize("role", ["exposures", "measure_stats"])
@pytest.mark.parametrize("row_count", [0, 1, 3])
def test_artifact_reader_rejects_inconsistent_base_row_count(
    reader_snapshot, monkeypatch, role, row_count
) -> None:
    manifest, ref, snapshot, store = reader_snapshot
    relation = getattr(manifest.base, role).model_copy(update={"row_count": row_count})
    manifest = manifest.model_copy(
        update={"base": manifest.base.model_copy(update={role: relation})}
    )
    manifest = manifest.model_copy(update={"manifest_sha256": manifest_sha256(manifest)})
    ref = ref.model_copy(update={"manifest_sha256": manifest.manifest_sha256})
    monkeypatch.setattr(snapshot, "read_manifest", lambda *args, **kwargs: manifest)
    source = ArtifactMomentSource.open(
        store,
        ref,
        expected_context=manifest.context,
        metrics=(MetricSpec(name="conversion", type="mean", value_column="orders"),),
    )
    try:
        metric = source.context.metrics[0]
        for _ in range(2):
            with pytest.raises(ArtifactContractError) as raised:
                source.moments(metric)
            assert raised.value.code == "artifact.relation.digest_mismatch"
    finally:
        source.close()


@pytest.mark.slow
@pytest.mark.parametrize("explicit_metrics", [False, True])
@pytest.mark.parametrize(
    "damage", ["missing_type", "duplicate_name", "extra_declaration", "omitted_declaration"]
)
def test_direct_source_refuses_invalid_or_off_roster_metric_declarations(
    reader_snapshot, monkeypatch, damage, explicit_metrics
) -> None:
    import json

    from increment.query.artifact_digest import canonical_json
    from increment.semantics.artifact import ArtifactContext, _artifact_context_digest

    manifest, ref, snapshot, store = reader_snapshot
    payload = json.loads(manifest.context.canonical_json)
    metrics = payload["definitions"]["metrics"]
    if damage == "missing_type":
        for item in metrics:
            item.pop("type")
    elif damage == "duplicate_name":
        metrics.append(dict(metrics[0]))
    elif damage == "extra_declaration":
        # A valid declaration the experiment's plan never lists and the manifest never binds.
        metrics.append({**metrics[0], "name": "unlisted"})
    else:
        # The plan lists a metric that neither the declarations nor the manifest carry.
        payload["experiment"]["plan"]["secondaries"] = ["undeclared"]
    raw = canonical_json(payload)
    context = ArtifactContext(canonical_json=raw, sha256=_artifact_context_digest(raw))
    manifest = manifest.model_copy(update={"context": context})
    manifest = manifest.model_copy(update={"manifest_sha256": manifest_sha256(manifest)})
    ref = ref.model_copy(update={"manifest_sha256": manifest.manifest_sha256})
    monkeypatch.setattr(snapshot, "read_manifest", lambda *args, **kwargs: manifest)
    source = ArtifactMomentSource.open(
        store,
        ref,
        expected_context=context,
        metrics=(
            (MetricSpec(name="conversion", type="mean", value_column="orders"),)
            if explicit_metrics
            else None
        ),
    )
    try:
        with pytest.raises(ArtifactContractError) as raised:
            _ = source.context
        assert raised.value.code == "artifact.context.mismatch"
    finally:
        source.close()


_ARTIFACT_FIRST_DS = date(2025, 1, 1)
_ARTIFACT_LAST_DS = date(2025, 1, 6)


def _artifact_metric_case(
    shape: str,
) -> tuple[
    MetricSpec,
    Metric,
    tuple[tuple[str, str | None, Literal["numerator", "denominator"]], ...],
]:
    if shape == "windowed_mean":
        return (
            MetricSpec(name=shape, type="mean", value_column="purchase", window_days=2),
            MeanMetric(
                name=shape,
                entity="unit_id",
                fact="purchase",
                aggregation="sum",
                window_days=2,
            ),
            (("purchase", "value", "numerator"),),
        )
    if shape == "winsorized_mean":
        winsorization = Winsorization(lower_value=3.0, upper_value=10.0)
        return (
            MetricSpec(
                name=shape,
                type="mean",
                value_column="purchase",
                winsorization=winsorization,
            ),
            MeanMetric(
                name=shape,
                entity="unit_id",
                fact="purchase",
                aggregation="sum",
                winsorization=winsorization,
            ),
            (("purchase", "value", "numerator"),),
        )
    if shape == "ratio":
        return (
            MetricSpec(
                name=shape,
                type="ratio",
                numerator="numerator",
                denominator="denominator",
                window_days=2,
            ),
            RatioMetric(
                name=shape,
                entity="unit_id",
                numerator=Measure(fact="numerator", aggregation="sum", window_days=2),
                denominator=Measure(fact="denominator", aggregation="count", window_days=2),
            ),
            (
                ("numerator", "value", "numerator"),
                ("denominator", None, "denominator"),
            ),
        )
    if shape in ("avg_event", "quantile"):
        metric = (
            QuantileMetric(
                name=shape,
                entity="unit_id",
                fact="purchase",
                aggregation="avg_event",
                quantile=0.5,
            )
            if shape == "quantile"
            else MeanMetric(
                name=shape,
                entity="unit_id",
                fact="purchase",
                aggregation="avg_event",
            )
        )
        return (
            MetricSpec(
                name=shape,
                type="quantile" if shape == "quantile" else "mean",
                value_column="purchase",
                quantile=0.5 if shape == "quantile" else None,
            ),
            metric,
            (("purchase", "value", "numerator"),),
        )
    if shape in ("retention", "retention_unbounded"):
        threshold = 3 if shape == "retention_unbounded" else (1, 3)
        return (
            MetricSpec(
                name=shape, type="retention", value_column="return", threshold_days=threshold
            ),
            RetentionMetric(
                name=shape,
                entity="unit_id",
                fact="return",
                threshold_days=threshold,
            ),
            (("return", None, "numerator"),),
        )
    raise AssertionError(f"unknown artifact shape {shape!r}")


def _artifact_raw_exposure_rows() -> list[dict[str, object]]:
    return [
        {
            "experiment_id": "exp",
            "unit_id": unit_id,
            "group_id": group,
            "ts": datetime(2025, 1, day, 10, tzinfo=UTC),
        }
        for unit_id, group, day in (
            ("u1", "control", 1),
            ("u2", "control", 2),
            ("u3", "treatment", 1),
            ("u4", "treatment", 2),
        )
    ]


def _artifact_raw_event_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add(unit_id: str, day: int, event: str, value: float | None = None) -> None:
        rows.append(
            {
                "unit_id": unit_id,
                "ts": datetime(2025, 1, day, 12, tzinfo=UTC),
                "event": event,
                "value": value,
            }
        )

    # Exposure-time events are deliberately present and must be excluded by
    # post_exposure_stats' strict timestamp boundary.
    add("u1", 1, "purchase", 100.0)
    add("u2", 2, "purchase", 100.0)
    add("u3", 1, "purchase", 100.0)
    add("u4", 2, "purchase", 100.0)
    add("u1", 2, "purchase", 2.0)
    add("u1", 2, "purchase", 4.0)
    add("u1", 4, "purchase", 8.0)
    add("u2", 3, "purchase", 4.0)
    add("u2", 3, "purchase", 6.0)
    add("u3", 2, "purchase", 5.0)
    add("u3", 3, "purchase", 7.0)
    add("u4", 3, "purchase", 20.0)
    add("u4", 4, "purchase", 1.0)

    add("u1", 2, "numerator", 10.0)
    add("u1", 3, "numerator", 8.0)
    add("u2", 3, "numerator", 5.0)
    add("u3", 2, "numerator", 6.0)
    add("u4", 3, "numerator", 9.0)
    for unit_id, day, count in (
        ("u1", 2, 2),
        ("u1", 3, 1),
        ("u2", 3, 1),
        ("u3", 2, 1),
        ("u3", 3, 1),
        ("u4", 3, 2),
    ):
        for _ in range(count):
            add(unit_id, day, "denominator")

    add("u1", 3, "return")
    add("u2", 4, "return")
    add("u3", 4, "return")
    add("u4", 3, "return")
    return rows


def _artifact_rows_equal(
    actual: list[dict[str, object]], expected: list[dict[str, object]]
) -> None:
    key_columns = ("ds", "experiment_id", "metric", "group_id")

    def sort_key(row: dict[str, object]) -> tuple[str, ...]:
        return tuple(str(row.get(key)) for key in key_columns)

    actual_rows = sorted(actual, key=sort_key)
    expected_rows = sorted(expected, key=sort_key)
    assert [row.keys() for row in actual_rows] == [row.keys() for row in expected_rows]
    assert len(actual_rows) == len(expected_rows)
    for actual_row, expected_row in zip(actual_rows, expected_rows, strict=True):
        assert actual_row.keys() == expected_row.keys()
        for key, actual_value in actual_row.items():
            expected_value = expected_row[key]
            if isinstance(actual_value, (int, float)) and isinstance(expected_value, (int, float)):
                # Warehouse aggregation order is not fixed, so centered sums near zero
                # differ by rounding noise far below any real moment change.
                assert actual_value == pytest.approx(expected_value, abs=1e-8), key
            else:
                assert actual_value == expected_value, key


def _adopted_artifact_fixture(
    shape: str, *, event_rows: list[dict[str, object]] | None = None
) -> dict[str, Any]:
    spec, metric, measure_specs = _artifact_metric_case(shape)
    exposure_rows = _artifact_raw_exposure_rows()
    event_rows = _artifact_raw_event_rows() if event_rows is None else event_rows
    con = ibis.duckdb.connect()
    import pyarrow as pa

    con.create_table("raw_exposures", pa.Table.from_pylist(exposure_rows))
    raw_events = con.create_table("raw_events", pa.Table.from_pylist(event_rows))
    raw_exposures = con.table("raw_exposures")
    experiment = Experiment(
        name="exp",
        exposure="assigned",
        unit="unit_id",
        start=datetime.combine(_ARTIFACT_FIRST_DS, datetime.min.time(), tzinfo=UTC),
        control_group="control",
        plan=AnalysisPlan(primary=shape),
    )
    exposure_base = first_exposures(raw_exposures, experiment)
    exposures_expr = exposure_base.mutate(
        first_exposure_date=exposure_base.first_exposure_ts.cast("date")
    ).select(
        "experiment_id",
        "unit_id",
        "group_id",
        "first_exposure_ts",
        "first_exposure_date",
    )
    exposures = con.to_pyarrow(exposures_expr).to_pylist()

    stats_by_key: dict[str, list[dict[str, object]]] = {}
    for measure_key, value_column, part in measure_specs:
        events = metric_events(
            raw_events,
            metric,
            value_column=value_column,
            part=part,
        )
        stats_expr = post_exposure_stats(
            events,
            exposures_expr,
            source_key=measure_key,
            experiment=experiment,
        ).mutate(experiment_id=ibis.literal("exp"))
        stats_expr = stats_expr.select(
            "experiment_id",
            "unit_id",
            "ds",
            "source_key",
            "n_events",
            "sum_value",
            "min_value",
            "max_value",
        ).rename(measure_key="source_key")
        stats_by_key[measure_key] = con.to_pyarrow(stats_expr).to_pylist()
    stats = [row for rows in stats_by_key.values() for row in rows]
    stats.sort(
        key=lambda row: (row["experiment_id"], row["unit_id"], row["ds"], row["measure_key"])
    )

    def relation_ref(role: str, rows: list[dict[str, object]]) -> ArtifactRelationRef:
        role_key = cast(
            Literal[
                "exposures",
                "measure_stats",
                "breakout_dimension",
                "factor_dimension",
                "cluster_identity",
                "cuped_preperiod",
                "assignment_counts",
                "trigger_population",
                "encouragement_uptake",
                "site_volume",
            ],
            role,
        )
        digests = digest_relation(
            role_key,
            UNIT_DAY_ARTIFACT_RELATION_SCHEMAS[role_key],
            rows,
            primary_key=UNIT_DAY_ARTIFACT_PRIMARY_KEYS[role_key],
        )
        return ArtifactRelationRef(
            artifact_id=artifact_id,
            generation_id=generation_id,
            role=role_key,
            relation=RelationLocator(catalog="warehouse", schema="analytics", name=role),
            schema_sha256=digests.schema_sha256,
            content_sha256=digests.content_sha256,
            row_count=digests.row_count,
            primary_key=UNIT_DAY_ARTIFACT_PRIMARY_KEYS[role_key],
        )

    fact_source = FactSource(
        name="events",
        sql="SELECT * FROM raw_events",
        timestamp_column="ts",
        entities=["unit_id"],
        facts=[
            Fact(name="exposure", column=None),
            Fact(name="purchase", column="value"),
            Fact(name="numerator", column="value"),
            Fact(name="denominator", column=None),
            Fact(name="return", column=None),
        ],
    )
    definitions = Definitions(
        dialect="duckdb",
        fact_sources=[fact_source],
        exposures=[Exposure(name="assigned", fact="exposure")],
        metrics=[metric],
        experiments=[experiment],
    )
    context = compile_unit_day_artifact_context("exp", definitions)
    artifact_id = uuid4()
    generation_id = uuid4()
    base = BaseRelations(
        exposures=relation_ref("exposures", exposures),
        measure_stats=relation_ref("measure_stats", stats),
    )
    measures = tuple(
        MeasureManifest(
            measure_key=measure_key,
            source_provenance_sha256=digest_relation(
                "measure_stats",
                UNIT_DAY_ARTIFACT_RELATION_SCHEMAS["measure_stats"],
                rows,
                primary_key=UNIT_DAY_ARTIFACT_PRIMARY_KEYS["measure_stats"],
            ).content_sha256,
            freshness=Freshness(loaded_through=_ARTIFACT_LAST_DS, declared_complete=True),
        )
        for measure_key, rows in sorted(stats_by_key.items())
    )
    bindings = (
        RatioMetricMeasure(
            metric_name=shape,
            numerator_measure_key="numerator",
            denominator_measure_key="denominator",
        )
        if shape == "ratio"
        else SimpleMetricMeasure(metric_name=shape, measure_key=measure_specs[0][0]),
    )
    values: dict[str, Any] = {
        "artifact_id": artifact_id,
        "generation_id": generation_id,
        "experiment_id": "exp",
        "created_at": datetime(2025, 1, 1, tzinfo=UTC),
        "day_boundary": "UTC",
        "first_ds": _ARTIFACT_FIRST_DS,
        "last_ds": _ARTIFACT_LAST_DS,
        "base": base,
        "measures": measures,
        "metric_measures": bindings,
        "context": context,
        "manifest_sha256": "0" * 64,
    }
    values["manifest_sha256"] = manifest_sha256(UnitDayArtifactManifest.model_construct(**values))
    manifest = UnitDayArtifactManifest(**values)
    ref = UnitDayArtifactRef(
        artifact_id=artifact_id,
        generation_id=generation_id,
        manifest=RelationLocator(name="manifest"),
        manifest_sha256=manifest.manifest_sha256,
    )

    class Snapshot:
        def __init__(self):
            self.artifact_id = artifact_id
            self.generation_id = generation_id

        def read_manifest(self, locator, *, expected_sha256):
            return manifest

        def verify_relation(self, relation, *, expected_role):
            rows = exposures if expected_role == "exposures" else stats
            return ibis.memtable(rows)

        def execute(self, expression):
            return con.to_pyarrow(expression)

    snapshot = Snapshot()

    class Store:
        def validate_locator(self, locator, **kwargs):
            return None

        @contextmanager
        def open_snapshot(self, fixed_ref):
            yield snapshot

    source = ArtifactMomentSource.open(
        cast(ArtifactStore, Store()),
        ref,
        expected_context=context,
        metrics=(spec,),
    )
    reader_experiment = Experiment(
        name="exp",
        exposure="adopted",
        unit="unit_id",
        start=datetime.combine(_ARTIFACT_FIRST_DS, datetime.min.time(), tzinfo=UTC),
        control_group="control",
        plan=AnalysisPlan(),
    )
    expected: dict[str, list[dict[str, object]]] = {}
    exposure_table = ibis.memtable(exposures)
    spine = panel_spine(
        exposure_table,
        reader_experiment,
        end_date=ibis.literal(_ARTIFACT_LAST_DS),
    )

    def stats_table(measure_key: str) -> Any:
        return ibis.memtable(stats_by_key[measure_key])

    total_stats = stats_table(measure_specs[0][0])
    den_stats = stats_table("denominator") if shape == "ratio" else None
    totals = unit_totals(
        spine,
        total_stats,
        metric,
        reader_experiment,
        den_stats=den_stats,
        data_as_of=_ARTIFACT_LAST_DS,
        warn_on_censoring=False,
    )
    totals = winsorize_unit_totals(totals, metric)
    expected["total"] = con.to_pyarrow(group_summary(totals)).to_pylist()

    def panel_for(
        measure_key: str,
        value_column: str | None,
        part: Literal["numerator", "denominator"],
    ) -> Any:
        events = metric_events(raw_events, metric, value_column=value_column, part=part)
        return unit_day_panel(
            exposure_table,
            events,
            reader_experiment,
            metric.name,
            end_date=ibis.literal(_ARTIFACT_LAST_DS),
        )

    panel = panel_for(measure_specs[0][0], measure_specs[0][1], measure_specs[0][2])
    den_panel = panel_for("denominator", None, "denominator") if shape == "ratio" else None
    if isinstance(metric, RetentionMetric) and metric.band[1] is None:
        # Unbounded retention has no daily evidence; the source refuses that grain.
        pass
    elif shape == "retention":
        expected["daily"] = con.to_pyarrow(
            cohort_group_summary(
                spine,
                total_stats,
                metric,
                reader_experiment,
                data_as_of=_ARTIFACT_LAST_DS,
                warn_on_censoring=False,
            )
        ).to_pylist()
    else:
        expected["daily"] = con.to_pyarrow(
            daily_group_summary(
                window_bound_stats(panel, metric),
                metric=metric,
                den_panel=(
                    window_bound_stats(den_panel, metric) if den_panel is not None else None
                ),
            )
        ).to_pylist()
    if not isinstance(metric, QuantileMetric):
        expected["asof"] = con.to_pyarrow(
            asof_group_summary(panel, metric, den_panel=den_panel)
        ).to_pylist()
    return {
        "source": source,
        "ref": ref,
        "context": context,
        "store": Store(),
        "spec": spec,
        "expected": expected,
    }


# Metric shapes the adopted artifact reduction is verified against (and so supports).
ARTIFACT_SUPPORTED_SHAPES = ("windowed_mean", "winsorized_mean", "ratio", "avg_event", "retention")


@pytest.mark.parametrize("shape", ARTIFACT_SUPPORTED_SHAPES)
def test_adopted_source_matches_builder_reduction_matrix(shape: str) -> None:
    fixture = _adopted_artifact_fixture(shape)
    source = fixture["source"]
    metric = source.context.metrics[0]
    for grain in ("total", "daily", "asof"):
        _artifact_rows_equal(
            source.moments(metric, grain=grain),
            fixture["expected"][grain],
        )
    source.close()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("window_days", 2),
        ("winsorization", Winsorization(lower_value=1.0)),
    ),
)
def test_adopted_source_rejects_caller_semantics_absent_from_trusted_context(
    field: str, value: object
) -> None:
    fixture = _adopted_artifact_fixture("avg_event")
    spec = fixture["spec"]
    bad_spec = spec.model_copy(update={field: value})
    source = ArtifactMomentSource.open(
        fixture["store"],
        fixture["ref"],
        expected_context=fixture["context"],
        metrics=(bad_spec,),
    )
    with pytest.raises(ArtifactContractError) as error:
        _ = source.context
    assert error.value.code == "artifact.metric.binding_mismatch"
    source.close()


def test_adopted_export_is_parquet_reloadable_by_analysis_from_moments(tmp_path) -> None:
    fixture = _adopted_artifact_fixture("windowed_mean")
    path = tmp_path / "moments.parquet"
    fixture["source"].export_moments(path)
    import pyarrow.parquet as pq

    from increment import Analysis

    rows = pq.read_table(path).to_pylist()
    reloaded = Analysis.from_moments(rows, metrics=[fixture["spec"]], control="control")
    assert reloaded.run()
    fixture["source"].close()


def test_adopted_avg_event_covariate_requires_cuped_extension() -> None:
    fixture = _adopted_artifact_fixture("avg_event")
    metric = fixture["source"].context.metrics[0]
    with pytest.raises(ArtifactContractError) as raised:
        fixture["source"].moments(metric, include_covariate=True)
    assert raised.value.code == "artifact.extension.missing"
    fixture["source"].close()


def test_completed_windows_gates_bounded_mean_and_retention_but_is_noop_for_unbounded() -> None:
    mean_fixture = _adopted_artifact_fixture("windowed_mean")
    mean_metric = mean_fixture["source"].context.metrics[0]
    baseline = mean_fixture["source"].moments(mean_metric, grain="asof")
    completed = mean_fixture["source"].moments(
        mean_metric, grain="asof", completed_windows_only=True
    )
    # A bounded no-uptake window completes after first_exposure_date +
    # window_days, as in builders.py's _completed_asof_rows. Days before a
    # group's first completion drop out; while only the earlier-exposed unit has
    # completed, only its count remains.
    assert {(str(row["ds"]), row["group_id"]) for row in baseline} == {
        (f"2025-01-0{day}", group) for day in range(1, 7) for group in ("control", "treatment")
    }
    assert {(str(row["ds"]), row["group_id"], row["n"]) for row in completed} == {
        ("2025-01-03", "control", 1),
        ("2025-01-03", "treatment", 1),
        ("2025-01-04", "control", 2),
        ("2025-01-04", "treatment", 2),
        ("2025-01-05", "control", 2),
        ("2025-01-05", "treatment", 2),
        ("2025-01-06", "control", 2),
        ("2025-01-06", "treatment", 2),
    }
    mean_fixture["source"].close()

    unbounded_mean_fixture = _adopted_artifact_fixture("avg_event")
    unbounded_mean_metric = unbounded_mean_fixture["source"].context.metrics[0]
    unbounded_baseline = unbounded_mean_fixture["source"].moments(
        unbounded_mean_metric, grain="asof"
    )
    unbounded_completed = unbounded_mean_fixture["source"].moments(
        unbounded_mean_metric, grain="asof", completed_windows_only=True
    )
    _artifact_rows_equal(unbounded_completed, unbounded_baseline)
    unbounded_mean_fixture["source"].close()
    retention_fixture = _adopted_artifact_fixture("retention")
    retention_metric = retention_fixture["source"].context.metrics[0]
    bounded = retention_fixture["source"].moments(
        retention_metric, grain="asof", completed_windows_only=True
    )
    expected = [
        ("2025-01-04", "control", 1, 1.0),
        ("2025-01-04", "treatment", 1, 0.0),
        ("2025-01-05", "control", 2, 1.0),
        ("2025-01-05", "treatment", 2, 0.5),
        ("2025-01-06", "control", 2, 1.0),
        ("2025-01-06", "treatment", 2, 0.5),
    ]
    assert (
        sorted((str(row["ds"]), row["group_id"], row["n"], row["ref_y"]) for row in bounded)
        == expected
    )
    mutated = retention_metric.model_copy(update={"threshold_days": 3})
    with pytest.raises(ArtifactContractError) as raised:
        retention_fixture["source"].moments(
            mutated,
            grain="asof",
            completed_windows_only=True,
        )
    assert raised.value.code == "artifact.metric.binding_mismatch"
    retention_fixture["source"].close()

    unbounded_fixture = _adopted_artifact_fixture("retention_unbounded")
    unbounded = unbounded_fixture["source"].context.metrics[0]
    _artifact_rows_equal(
        unbounded_fixture["source"].moments(unbounded, grain="total"),
        unbounded_fixture["source"].moments(unbounded, grain="total", completed_windows_only=True),
    )
    with pytest.raises(ArtifactContractError) as raised:
        unbounded_fixture["source"].moments(unbounded, grain="daily")
    assert raised.value.code == "artifact.evidence.unavailable"
    with pytest.raises(ArtifactContractError) as raised:
        unbounded_fixture["source"].moments(unbounded, grain="daily", completed_windows_only=True)
    assert raised.value.code == "artifact.evidence.unavailable"
    with pytest.raises(ArtifactContractError) as raised:
        unbounded_fixture["source"].moments(
            unbounded,
            grain="asof",
            completed_windows_only=True,
        )
    assert raised.value.code == "artifact.evidence.unavailable"
    unbounded_fixture["source"].close()


def test_unsupported_grain_and_raw_sql_stay_capability_refusals() -> None:
    fixture = _adopted_artifact_fixture("windowed_mean")
    source = fixture["source"]
    metric = source.context.metrics[0]
    source.capabilities = frozenset({"total"})
    with pytest.raises(CapabilityError) as raised:
        source.moments(metric, grain="daily")
    assert raised.value.code == "artifact.operation.unsupported"
    with pytest.raises(CapabilityError) as raised:
        source.sql()
    assert raised.value.code == "artifact.operation.unsupported"
    source.close()


@pytest.mark.slow
@pytest.mark.parametrize("shape", ["avg_event", "quantile"])
@pytest.mark.parametrize("adapter", ["reader", "facade"])
def test_simple_aggregation_mutation_refuses_before_data_read(shape, adapter, monkeypatch):
    from increment.query.source import ArtifactMomentSource as FacadeSource

    fixture = _adopted_artifact_fixture(shape)
    source_class = ArtifactMomentSource if adapter == "reader" else FacadeSource
    source = source_class.open(
        fixture["store"],
        fixture["ref"],
        expected_context=fixture["context"],
    )
    try:
        metric = source.context.metrics[0]
        assert isinstance(metric, MeanMetric | QuantileMetric)
        assert metric.aggregation == "avg_event"

        def unexpected_read(*args, **kwargs):
            pytest.fail("binding mismatch must refuse before reading artifact data")

        with monkeypatch.context() as patch:
            with fixture["store"].open_snapshot(fixture["ref"]) as snapshot:
                patch.setattr(snapshot, "verify_relation", unexpected_read)
                patch.setattr(snapshot, "execute", unexpected_read)
                with pytest.raises(ArtifactContractError) as raised:
                    source.moments(metric.model_copy(update={"aggregation": "sum"}), grain="daily")
                assert raised.value.code == "artifact.metric.binding_mismatch"
        _artifact_rows_equal(source.moments(metric, grain="daily"), fixture["expected"]["daily"])
    finally:
        source.close()
        fixture["source"].close()


def test_ratio_nested_window_mutation_refuses_before_reduction() -> None:
    fixture = _adopted_artifact_fixture("ratio")
    metric = fixture["source"].context.metrics[0]
    mutations = (
        (
            "numerator",
            metric.numerator.model_copy(update={"window_days": 99}),
        ),
        (
            "denominator",
            metric.denominator.model_copy(update={"aggregation": "sum"}),
        ),
    )
    for field, part in mutations:
        mutated = metric.model_copy(update={field: part})
        with pytest.raises(ArtifactContractError) as raised:
            fixture["source"].moments(mutated)
        assert raised.value.code == "artifact.metric.binding_mismatch"
    fixture["source"].close()


def test_panel_materialization_keeps_trusted_ratio_context() -> None:
    fixture = _adopted_artifact_fixture("ratio")
    source = fixture["source"]
    ratio = source.context.metrics[0]
    # unit_frame() on this unit-panel shape must raise a coded capability
    # refusal, not crash, and must keep the manifest-derived context rather than
    # panel-synthesized metric semantics.
    with pytest.raises(CapabilityError) as raised:
        source.unit_frame(ratio)
    assert raised.value.code == "source.frame.unit_frame_panel"
    assert source.context.metrics[0] is ratio
    rows = source.moments(ratio)
    assert rows
    source.close()


def test_unit_counts_answers_every_enrolled_unit_via_a_bounded_aggregate() -> None:
    """`unit_counts()` is a per-group `group_by` count straight off
    `exposures` -- every enrolled unit counts, regardless of whether that
    unit's outcome window is itself observable yet. `cluster_counts()`
    refuses on this no-extension reader: no cluster concept exists at
    this layer (only the cluster_identity-extension-backed facade answers
    a cluster-grain count)."""
    fixture = _adopted_artifact_fixture("windowed_mean")
    source = fixture["source"]
    assert source.unit_counts() == {"control": 2, "treatment": 2}
    with pytest.raises(CapabilityError) as raised:
        source.cluster_counts()
    assert raised.value.code == "artifact.operation.unsupported"
    source.close()


@pytest.mark.slow
def test_published_manifest_records_the_experiment_day_boundary(tmp_path) -> None:
    """Every date in the artifact is bucketed under Experiment.day_boundary and
    the reader re-derives first_exposure_date from manifest.day_boundary, so
    the manifest must carry the experiment's boundary, not the definitions default."""
    import shutil

    import ibis

    from examples._seed import EXPERIMENT_ID, seed_event_log
    from increment import Analysis
    from increment.query.session import WarehouseArtifactStore

    definitions = tmp_path / "definitions"
    shutil.copytree("examples/definitions", definitions)
    experiments = definitions / "experiments.yaml"
    experiments.write_text(
        experiments.read_text().replace(
            "    unit: user_id", '    unit: user_id\n    day_boundary: "UTC-05:00"', 1
        )
    )

    con = ibis.duckdb.connect()
    seed_event_log(con, n_units=300, seed=13, with_pre_period=True)
    analysis = Analysis.from_definitions(EXPERIMENT_ID, definitions, con, store="none")
    assert analysis.experiment.day_boundary == "UTC-05:00"

    store = WarehouseArtifactStore(con, schema_name="day_boundary_artifact")
    ref = analysis.publish_unit_day_artifact(store)
    with store.open_snapshot(ref) as snapshot:
        manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)

    assert manifest.day_boundary == analysis.experiment.day_boundary


@pytest.mark.slow
def test_published_event_horizon_matches_native_and_older_artifacts_still_read(
    tmp_path,
) -> None:
    """The persisted per-measure event_horizon is native's own filtered
    metric_events() watermark, not the unfiltered freshness watermark: a
    late NULL-valued fact row advances freshness.loaded_through but must
    not advance the adopted spine, on a RUNNING experiment (no declared end
    or observation_end) where the spine edge actually falls back to the
    persisted horizon instead of a declared bound. `asof` grain is the
    grain whose observed day axis genuinely tracks that fallback: `daily`
    grain consumes the same shared spine edge but is additionally clipped
    per unit by `window_bound_stats`, so its own max day never moves in
    this fixture either way and would not have caught a regression here.
    An older artifact predating this field still parses and falls back to
    freshness-based sizing."""
    from examples._seed import seed_event_log
    from increment import Analysis
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.query.source import open_artifact
    from increment.semantics.artifact import Freshness, MeasureManifest
    from increment.semantics.loader import load
    from tests.test_unit_day_artifact_facade import _definitions

    definitions = _definitions(tmp_path, open_ended=True)
    con = ibis.duckdb.connect()
    seed_event_log(con, n_units=200, seed=7, with_pre_period=True)
    analysis = Analysis.from_definitions("new_onboarding_v2", definitions, con, store="none")
    assert analysis.experiment.observation_horizon is None
    metric = next(m for m in analysis.metrics if m.name == "avg_session_duration")

    store = WarehouseArtifactStore(con, schema_name="event_horizon_artifact")
    ref = analysis.publish_unit_day_artifact(store)
    with store.open_snapshot(ref) as snapshot:
        manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
    binding = next(b for b in manifest.metric_measures if b.metric_name == metric.name)
    measure = next(m for m in manifest.measures if m.measure_key == binding.measure_key)

    # Compile the trusted context from the definitions the caller already
    # owns, so opening below cross-checks publish-time against recomputed.
    loaded = load(definitions)
    experiment = loaded.experiment("new_onboarding_v2")
    assert experiment is not None
    context = artifact_context(loaded, experiment, "error")
    adopted = open_artifact(store, ref, expected_context=context)
    baseline_max_ds = max(row["ds"] for row in adopted.moments(metric, grain="asof"))

    # avg_session_duration reads session_end.duration_s, filtered to
    # platform=web; a NULL-valued late row is dropped by metric_events()
    # but still advances the fact's unfiltered freshness watermark.
    con.raw_sql(
        "INSERT INTO analytics.event_log "
        "(event_at, user_id, session_id, event, duration_s, device_type) "
        "VALUES ('2025-02-20 00:00:00', 'u00031', 'u00031-late', 'session_end', NULL, 'web')"
    )
    late_analysis = Analysis.from_definitions("new_onboarding_v2", definitions, con, store="none")
    late_store = WarehouseArtifactStore(con, schema_name="event_horizon_artifact_late")
    late_ref = late_analysis.publish_unit_day_artifact(late_store)
    with late_store.open_snapshot(late_ref) as snapshot:
        late_manifest = snapshot.read_manifest(
            late_ref.manifest, expected_sha256=late_ref.manifest_sha256
        )
    late_measure = next(m for m in late_manifest.measures if m.measure_key == binding.measure_key)
    assert late_measure.freshness.loaded_through == date(2025, 2, 20)
    assert late_measure.freshness.loaded_through > measure.freshness.loaded_through
    assert late_measure.event_horizon == measure.event_horizon

    # A freshness-only republish must not move the compiled context, so the
    # independently derived context still binds the late artifact.
    assert late_manifest.context.sha256 == context.sha256
    late_adopted = open_artifact(late_store, late_ref, expected_context=context)
    late_max_ds = max(row["ds"] for row in late_adopted.moments(metric, grain="asof"))
    native_asof = late_analysis.run_asof(metrics=[metric.name])
    assert late_max_ds == max(row.ds for row in native_asof)
    # The freshness-only change never moved the adopted spine: it stayed
    # pinned to the persisted (filtered) event horizon both times.
    assert late_max_ds == baseline_max_ds
    assert late_max_ds < late_measure.freshness.loaded_through

    # An older artifact published before this field existed omits it; the
    # manifest still parses (nullable default) and its per-measure edge
    # falls back to the freshness watermark.
    stale_measures = tuple(
        MeasureManifest.model_validate(
            {
                "measure_key": m.measure_key,
                "source_provenance_sha256": m.source_provenance_sha256,
                "freshness": Freshness(
                    loaded_through=m.freshness.loaded_through,
                    declared_complete=m.freshness.declared_complete,
                ),
            }
        )
        for m in late_manifest.measures
    )
    assert all(m.event_horizon is None for m in stale_measures)
    stale_measure = next(m for m in stale_measures if m.measure_key == binding.measure_key)
    assert stale_measure.freshness.loaded_through == date(2025, 2, 20)
    analysis.close()
    late_analysis.close()


@pytest.mark.slow
def test_legacy_manifest_json_without_event_horizon_verifies_through_the_store() -> None:
    """A manifest published before `event_horizon` existed never emitted the
    key in its stored `manifest_json` at all. Overwrite a freshly published
    artifact's persisted row to strip the key from every measure (as if it
    predated the field) and recompute the digest the same way a legacy
    writer would have; opening it back through the real store must still
    verify -- `MeasureManifest`'s `_ExplicitOnlyFields` mixin drops the key
    from serialization exactly when it was never set on parse, keeping a
    legacy artifact's stored digest byte-identical."""

    from examples._seed import EXPERIMENT_ID, seed_event_log
    from increment import Analysis
    from increment.query.artifact_digest import manifest_sha256
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.artifact import MeasureManifest

    con = ibis.duckdb.connect()
    seed_event_log(con, n_units=200, seed=7, with_pre_period=True)
    analysis = Analysis.from_definitions(EXPERIMENT_ID, "examples/definitions", con, store="none")
    store = WarehouseArtifactStore(con, schema_name="legacy_event_horizon_artifact")
    ref = analysis.publish_unit_day_artifact(store)
    with store.open_snapshot(ref) as snapshot:
        manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
    assert all("event_horizon" in m.model_fields_set for m in manifest.measures)

    legacy_measures = tuple(
        MeasureManifest.model_validate(
            {k: v for k, v in m.model_dump(mode="python").items() if k != "event_horizon"}
        )
        for m in manifest.measures
    )
    assert all("event_horizon" not in m.model_fields_set for m in legacy_measures)
    legacy_manifest = manifest.model_copy(update={"measures": legacy_measures})
    legacy_digest = manifest_sha256(legacy_manifest)
    payload = legacy_manifest.model_dump(mode="json")
    payload["manifest_sha256"] = legacy_digest
    assert all("event_horizon" not in row for row in payload["measures"])

    table_ref = '"legacy_event_horizon_artifact"."ud_manifest_index"'
    escaped = json.dumps(payload, separators=(",", ":")).replace("'", "''")
    con.raw_sql(
        f"UPDATE {table_ref} SET manifest_json = '{escaped}', "
        f"manifest_sha256 = '{legacy_digest}' WHERE manifest_sha256 = '{ref.manifest_sha256}'"
    )
    # A fresh store on the same schema re-reads the persisted row, exactly as
    # a later process opening this artifact would.
    reopened_store = WarehouseArtifactStore(con, schema_name="legacy_event_horizon_artifact")
    legacy_ref = ref.model_copy(update={"manifest_sha256": legacy_digest})
    with reopened_store.open_snapshot(legacy_ref) as snapshot:
        reopened = snapshot.read_manifest(
            legacy_ref.manifest, expected_sha256=legacy_ref.manifest_sha256
        )
    assert all("event_horizon" not in m.model_fields_set for m in reopened.measures)
    assert all(m.event_horizon is None for m in reopened.measures)
    assert reopened.manifest_sha256 == manifest_sha256(reopened) == legacy_digest
    analysis.close()


@pytest.mark.slow
def test_explicit_null_event_horizon_verifies_and_contributes_no_spine_edge() -> None:
    """An explicitly persisted `"event_horizon": null` is NOT legacy
    omission -- the publisher set the field and confirmed the measure is
    eventless, same as the `NO_DATA_SIGNAL` sentinel. `_ExplicitOnlyFields`
    keeps an explicitly-set null in the digest body (unlike an omitted
    key), so such a manifest still verifies; the reader must exclude that
    measure from the spine union outright rather than falling back to its
    freshness watermark."""

    from increment.query.artifact_digest import manifest_sha256
    from increment.query.artifact_reader import _measure_edge, _outcome_edge, _union_outcome_edge
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.artifact import MeasureManifest
    from tests.semantics.test_unit_day_artifact import _manifest as _bare_manifest

    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="explicit_null_event_horizon_artifact")
    base = _bare_manifest()
    explicit_null_measures = tuple(
        MeasureManifest.model_validate({**m.model_dump(mode="python"), "event_horizon": None})
        for m in base.measures
    )
    assert all("event_horizon" in m.model_fields_set for m in explicit_null_measures)
    assert all(m.event_horizon is None for m in explicit_null_measures)
    manifest = base.model_copy(update={"measures": explicit_null_measures})
    digest = manifest_sha256(manifest)
    payload = manifest.model_dump(mode="json")
    payload["manifest_sha256"] = digest
    assert all(row.get("event_horizon", "missing") is None for row in payload["measures"])

    manifest_name = f"ud_{manifest.artifact_id.hex[:12]}_{manifest.generation_id.hex[:12]}_manifest"
    table_ref = '"explicit_null_event_horizon_artifact"."ud_manifest_index"'
    con.raw_sql(
        f"INSERT INTO {table_ref} "
        "(artifact_id, generation_id, manifest_name, manifest_sha256, manifest_json) VALUES "
        f"('{manifest.artifact_id}', '{manifest.generation_id}', '{manifest_name}', '{digest}', "
        f"'{json.dumps(payload, separators=(',', ':')).replace(chr(39), chr(39) * 2)}')"
    )
    from increment.semantics.artifact import RelationLocator, UnitDayArtifactRef

    trusted_ref = UnitDayArtifactRef(
        artifact_id=manifest.artifact_id,
        generation_id=manifest.generation_id,
        manifest=RelationLocator(schema="explicit_null_event_horizon_artifact", name=manifest_name),
        manifest_sha256=digest,
    )
    with store.open_snapshot(trusted_ref) as snapshot:
        reopened = snapshot.read_manifest(
            trusted_ref.manifest, expected_sha256=trusted_ref.manifest_sha256
        )
    assert reopened.manifest_sha256 == manifest_sha256(reopened) == digest
    reopened_measure = reopened.measures[0]
    assert "event_horizon" in reopened_measure.model_fields_set
    assert reopened_measure.event_horizon is None

    assert _measure_edge(reopened_measure, prefer_persisted_horizon=True) is None
    key = reopened_measure.measure_key
    empty_stats_schema = ibis.schema(
        {
            "experiment_id": "string",
            "unit_id": "string",
            "ds": "date",
            "measure_key": "string",
            "n_events": "int64",
            "sum_value": "float64",
            "min_value": "float64",
            "max_value": "float64",
        }
    )
    # ibis<=10.x drops every declared column of a memtable built from an empty list.
    empty_stats = ibis.memtable(
        empty_stats_schema.to_pyarrow().empty_table(), schema=empty_stats_schema
    )
    assert (
        _outcome_edge(
            reopened.measures,
            empty_stats,
            frozenset({key}),
            con.to_pyarrow,
            prefer_persisted_horizon=True,
        )
        is None
    )
    assert _union_outcome_edge(reopened.measures, frozenset({key})) is None


def test_winsorization_metadata_missing_is_coded():
    from increment.semantics.models import MeanMetric
    from increment.sources import MomentsSource
    from tests.test_sources import _format3_row

    metric = MeanMetric(name="revenue", entity="user", fact="revenue")
    row = _format3_row(
        winsor_lower_percentile=0.01,
        winsor_lower_bound=None,
        winsor_n_lower=1,
        winsor_upper_percentile=None,
        winsor_upper_bound=None,
        winsor_n_upper=0,
        winsor_n=1,
    )
    with pytest.raises(CapabilityError) as raised:
        MomentsSource([row], metrics=[metric], study_id="exp")
    assert raised.value.code == "moments.winsorization.metadata_missing"


def test_validate_stats_primary_key_is_coded(reader_snapshot, monkeypatch) -> None:
    manifest, ref, snapshot, store = reader_snapshot
    verified = snapshot.verify_relation

    def duplicate_unit_day(relation, *, expected_role):
        table = verified(relation, expected_role=expected_role)
        if expected_role != "measure_stats":
            return table
        rows = snapshot.execute(table).to_pylist()
        return ibis.memtable([*rows, rows[0]])

    monkeypatch.setattr(snapshot, "verify_relation", duplicate_unit_day)
    source = ArtifactMomentSource.open(
        store,
        ref,
        expected_context=manifest.context,
        metrics=(MetricSpec(name="conversion", type="mean", value_column="orders"),),
    )
    try:
        metric = source.context.metrics[0]
        with pytest.raises(CapabilityError) as raised:
            source.moments(metric)
        assert raised.value.code == "artifact.relation.invalid"
        assert raised.value.context["check"] == "primary_key"
    finally:
        source.close()


@pytest.mark.slow
def test_streaming_snapshot_isolates_external_mutation():
    from pathlib import Path

    from increment.analysis import Analysis
    from tests.test_unit_day_artifact_facade import _native

    con, native, context, store = _native(
        definitions=Path(__file__).resolve().parents[1] / "examples" / "definitions"
    )
    adopted = None
    try:
        ref = native.publish_unit_day_artifact(store)
        adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
        with store.open_snapshot(ref) as snapshot:
            manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
        metric = adopted.metrics[0]
        before = adopted.run(metrics=[metric.name])
        assert before
        published = manifest.base.measure_stats.relation
        con.raw_sql(
            f'UPDATE "{published.schema_name}"."{published.name}" SET sum_value = sum_value + 1000'
        )
        # The adopted analysis keeps reading the snapshot it opened with, so the
        # externally mutated relation never reaches its numbers.
        assert adopted.run(metrics=[metric.name]) == before
        with store.open_snapshot(ref) as fresh:
            with pytest.raises(ArtifactContractError) as error:
                fresh.verify_relation(manifest.base.measure_stats, expected_role="measure_stats")
        assert error.value.code == "artifact.snapshot.mixed"
    finally:
        if adopted is not None:
            adopted.close()
        native.close()
        con.disconnect()


def test_open_refuses_a_manifest_edited_in_place() -> None:
    """Mirrors drop_generation's own tamper check (session.py): a
    manifest_json row edited without updating its manifest_sha256 column
    must be refused on open, the same way drop_generation already refuses
    it. Publishes the manifest row directly via one raw SQL INSERT into the
    metadata table (matching test_explicit_null_event_horizon_verifies_and_contributes_no_spine_edge,
    same file) rather than through begin_publication, so the store's
    in-memory generation cache is never populated for this artifact_id --
    open_snapshot's _generation_for_ref call is forced onto its cache-miss
    path, which re-reads and re-parses manifest_json from the table every
    time, exactly the code path being tested. The tampered body is built by
    forging a self-consistent manifest (UnitDayArtifactManifest's own model
    validator already rejects a manifest whose embedded manifest_sha256
    field disagrees with its own other fields, so a naive single-field edit
    never reaches this check at all -- it must be a manifest that parses and
    validates cleanly, whose independently-recomputed digest differs from
    the original, untouched database column)."""

    from increment.query.session import WarehouseArtifactStore
    from tests.semantics.test_unit_day_artifact import _manifest as _bare_manifest

    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="tampered_manifest_artifact")
    manifest = _bare_manifest()
    digest = manifest_sha256(manifest)
    payload = manifest.model_dump(mode="json")
    payload["manifest_sha256"] = digest

    manifest_name = f"ud_{manifest.artifact_id.hex[:12]}_{manifest.generation_id.hex[:12]}_manifest"
    table_ref = f'"{store._schema}"."{store._meta_name}"'
    con.raw_sql(
        f"INSERT INTO {table_ref} "
        "(artifact_id, generation_id, manifest_name, manifest_sha256, manifest_json) VALUES "
        f"('{manifest.artifact_id}', '{manifest.generation_id}', '{manifest_name}', '{digest}', "
        f"'{json.dumps(payload, separators=(',', ':')).replace(chr(39), chr(39) * 2)}')"
    )

    # Forge a self-consistent tampered manifest via model instances throughout
    # (never a raw JSON-mode dict -- mixing dict-mode and model-mode
    # representations produces a different canonical encoding for
    # UUID/datetime fields and silently breaks the digest comparison).
    retagged = manifest.model_copy(update={"experiment_id": "tampered-in-place"})
    forged_digest = manifest_sha256(retagged)
    assert forged_digest != digest
    tampered_manifest = retagged.model_copy(update={"manifest_sha256": forged_digest})
    tampered_json = json.dumps(
        tampered_manifest.model_dump(mode="json"), separators=(",", ":")
    ).replace(chr(39), chr(39) * 2)
    con.raw_sql(
        f"UPDATE {table_ref} SET manifest_json = '{tampered_json}' "
        f"WHERE artifact_id = '{manifest.artifact_id}' AND generation_id = '{manifest.generation_id}'"
    )

    trusted_ref = UnitDayArtifactRef(
        artifact_id=manifest.artifact_id,
        generation_id=manifest.generation_id,
        manifest=RelationLocator(schema=store._schema, name=manifest_name),
        manifest_sha256=digest,  # the DB column is left untouched -- only manifest_json changed
    )
    # A fresh store instance on the same connection/schema, matching drop_generation's
    # own re-read contract: nothing here relies on the publishing store's own cache.
    fresh_store = WarehouseArtifactStore(con, schema_name="tampered_manifest_artifact")
    with pytest.raises(ArtifactContractError) as raised:
        with fresh_store.open_snapshot(trusted_ref):
            pass
    assert raised.value.code == "artifact.manifest.invalid"
    con.disconnect()


def test_scaled_binary64_enclosure_admits_repeated_addition_and_rejects_outside() -> None:
    for n, value in ((10, 0.1), (100, 0.1), (10, 0.5)):
        total = 0.0
        for _ in range(n):
            total += value
        validate_unit_day_aggregate_rows(
            [{"n_events": n, "sum_value": total, "min_value": value, "max_value": value}]
        )
        assert _aggregate_sum_within_extrema(total, n, value, value)
    with pytest.raises(CapabilityError) as raised:
        validate_unit_day_aggregate_rows(
            [{"n_events": 10, "sum_value": 1.0001, "min_value": 0.1, "max_value": 0.1}]
        )
    assert raised.value.code == "artifact.aggregate.impossible"


def test_scaled_binary64_enclosure_keeps_structural_refusals_exact() -> None:
    for n_events, total, minimum, maximum in (
        (0, 0.0, 0.0, 0.0),
        (True, 1.0, 1.0, 1.0),
        (1.5, 1.5, 1.0, 1.0),
        (2**53 + 1, float(2**53), 1.0, 1.0),
        (2, math.inf, 1.0, 1.0),
        (2, 2.0, math.nan, 1.0),
        (2, 2.0, 1.0, math.inf),
        (2, 1 << 1100, 1.0, 1.0),
        (2, 2.0, 2.0, 1.0),
        (1, math.nextafter(1.0, math.inf), 1.0, 1.0),
        (2, math.ulp(0.0), 0.0, 0.0),
    ):
        with pytest.raises(CapabilityError) as raised:
            validate_unit_day_aggregate_rows(
                [
                    {
                        "n_events": n_events,
                        "sum_value": total,
                        "min_value": minimum,
                        "max_value": maximum,
                    }
                ]
            )
        assert raised.value.code == "artifact.aggregate.impossible"


def test_scaled_binary64_enclosure_bounds_exact_rounding_oracle() -> None:
    with localcontext() as context:
        context.prec = 1800
        u = Decimal(2) ** -53
        eta = Decimal.from_float(math.ulp(0.0))
        largest = float.fromhex("0x1.fffffffffffffp+1023")
        for n in (2, 3, 10, 100, 1024, 10**6, 2**26, 2**52, 2**53 - 1, 2**53):
            for magnitude in (
                math.ulp(0.0),
                math.nextafter(2.0**-1022, 0.0),
                2.0**-1022,
                1e-300,
                0.1,
                math.nextafter(0.1, math.inf),
                0.5,
                1e300,
                largest / n,
                largest,
            ):
                denominator = Decimal(1) - Decimal(n - 1) * u
                beta = Decimal(n) * u / denominator
                exact_required = (
                    (beta + u) * Decimal.from_float(magnitude) + (1 / denominator + 5 + 2 * u) * eta
                ) / (1 - u)
                bound = _aggregate_mean_error_bound(n, magnitude)
                assert Decimal.from_float(bound) >= exact_required
                if exact_required < Decimal.from_float(largest) / 2:
                    assert math.isfinite(bound)


def test_scaled_binary64_enclosure_python_and_sql_decisions_match() -> None:
    largest = float.fromhex("0x1.fffffffffffffp+1023")
    eta = math.ulp(0.0)
    cases = [
        (2, 0.3, 0.1, 0.2, True),
        (10, 0.9999999999999999, 0.1, 0.1, True),
        (100, 9.99999999999998, 0.1, 0.1, True),
        (2**53, float(2**53), 1.0, 1.0, True),
        (2**53 + 1, float(2**53), 1.0, 1.0, False),
        (2**53, 0.0, 1.0, 1.0, False),
        (2**53, 0.5, 1.0, 1.0, False),
        (2**53, 0.0, -1.0, -1.0, False),
        (2**53, -0.5, -1.0, -1.0, False),
        (0, 0.0, 0.0, 0.0, False),
        (1, math.nextafter(1.0, math.inf), 1.0, 1.0, False),
        (2, eta, 0.0, 0.0, False),
        (10, 10 * eta, eta, eta, True),
        (10, 1000 * eta, eta, eta, False),
        (10, 1e-299, 1e-300, 1e-300, True),
        (10, 2e-299, 1e-300, 1e-300, False),
        (2, 0.0, -largest, largest, True),
        (2**53, 0.0, -largest, largest, True),
        (2, largest, 0.75 * largest, 0.75 * largest, False),
        (2, 4.0, 1.0, 1.0, False),
        (2, 2.0, 2.0, 1.0, False),
    ]
    records = [
        {"case": i, "n_events": n, "sum_value": total, "min_value": low, "max_value": high}
        for i, (n, total, low, high, _expected) in enumerate(cases)
    ]
    expected = [accepted for *_statistics, accepted in cases]
    observed = []
    for row in records:
        try:
            validate_unit_day_aggregate_rows([row])
        except CapabilityError as error:
            assert error.code == "artifact.aggregate.impossible"
            observed.append(False)
        else:
            observed.append(True)
    assert observed == expected
    connection = ibis.duckdb.connect()
    try:
        table = connection.create_table("aggregate_cases", ibis.memtable(records))
        accepted = _aggregate_sum_within_extrema_sql(
            table.sum_value, table.n_events, table.min_value, table.max_value
        )
        actual = table.select("case", accepted=accepted).order_by("case").execute()
        assert actual["accepted"].tolist() == expected
    finally:
        connection.disconnect()


def test_scaled_binary64_enclosure_accepts_reordered_and_partitioned_sums() -> None:
    largest = float.fromhex("0x1.fffffffffffffp+1023")
    for values in (
        [0.1] * 100,
        [1e16, 1.0, -1e16, -0.5],
        [1e16, math.nextafter(1e16, math.inf)] * 10,
        [1e-300, math.nextafter(1e-300, math.inf)] * 10,
        [math.nextafter(largest / 3, 0.0)] * 3,
    ):
        totals = []
        for ordered in (values, list(reversed(values))):
            total = 0.0
            for value in ordered:
                total += value
            totals.append(total)
        totals.append(
            math.fsum(
                (math.fsum(values[: len(values) // 2]), math.fsum(values[len(values) // 2 :]))
            )
        )
        for total in totals:
            validate_unit_day_aggregate_rows(
                [
                    {
                        "n_events": len(values),
                        "sum_value": total,
                        "min_value": min(values),
                        "max_value": max(values),
                    }
                ]
            )
            assert _aggregate_sum_within_extrema(total, len(values), min(values), max(values))


@pytest.mark.slow
def test_artifact_adoption_preserves_rounded_decimal_event_averages() -> None:
    events: list[dict[str, object]] = [
        {
            "unit_id": unit,
            "ts": datetime(2025, 1, 3, 12, tzinfo=UTC),
            "event": "purchase",
            "value": 0.1,
        }
        for unit, count in (("u1", 10), ("u2", 100), ("u3", 10), ("u4", 100))
        for _ in range(count)
    ]
    fixture = _adopted_artifact_fixture("avg_event", event_rows=events)
    source = fixture["source"]
    try:
        rows = source.moments(source.context.metrics[0])
        _artifact_rows_equal(rows, fixture["expected"]["total"])
        assert {row["group_id"] for row in rows} == {"control", "treatment"}
        assert all(row["ref_y"] + row["cy1"] / row["n"] == pytest.approx(0.1) for row in rows)
    finally:
        source.close()


# Frozen artifacts published by the previous writer. Two retention-only experiments that
# differ only in the spelling of `end` produced byte-identical contexts, so neither the
# context nor its pin can say which window produced the relations. Never regenerate.
_LEGACY_COLLISION = Path(__file__).parent / "fixtures" / "unit_day_artifact_legacy_collision.json"


def _decode_cell(value: Any) -> Any:
    if isinstance(value, dict):
        if "$dt" in value:
            return datetime.fromisoformat(value["$dt"])
        if "$date" in value:
            return date.fromisoformat(value["$date"])
        if "$dec" in value:
            return Decimal(value["$dec"])
        if "$b" in value:
            return bytes.fromhex(value["$b"])
    return value


def _legacy_collision_definitions(tmp_path: Path, end: str) -> Path:
    import yaml

    payload: dict[str, Any] = {}
    root = Path(__file__).parents[1] / "examples" / "definitions"
    for name in ("fact_sources.yaml", "exposures.yaml", "metrics.yaml", "experiments.yaml"):
        payload.update(yaml.safe_load((root / name).read_text()))
    experiment = next(e for e in payload["experiments"] if e["name"] == "new_onboarding_v2")
    experiment.update(
        start="2025-01-05T00:00:00Z",
        end=end,
        day_boundary="UTC-05:00",
        plan={"primary": "d7_retention"},
    )
    path = tmp_path / "legacy_collision.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def _restored_legacy_artifact(label: str) -> tuple[Any, Any, Any, Any]:
    """A warehouse holding the frozen artifact: (connection, store, ref, original pin)."""
    from examples._seed import seed_event_log
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.artifact import ArtifactContext

    frozen = json.loads(_LEGACY_COLLISION.read_text())["artifacts"][label]
    connection = ibis.duckdb.connect()
    seed_event_log(connection, n_units=300)
    connection.raw_sql("CREATE SCHEMA IF NOT EXISTS artifacts")
    for name, table in frozen["tables"].items():
        schema = ibis.schema([(column, ibis.dtype(kind)) for column, kind in table["schema"]])
        rows = [{key: _decode_cell(value) for key, value in row.items()} for row in table["rows"]]
        # ibis<=10.x drops every declared column of a memtable built from an empty list.
        data = (
            ibis.memtable(rows, schema=schema)
            if rows
            else ibis.memtable(schema.to_pyarrow().empty_table(), schema=schema)
        )
        connection.create_table(name, data, database="artifacts", overwrite=False)
    store = WarehouseArtifactStore(connection, schema_name="artifacts")
    ref = UnitDayArtifactRef.model_validate(frozen["ref"])
    return connection, store, ref, ArtifactContext.model_validate(frozen["context"])


@pytest.mark.parametrize("label", ["z", "offset"])
def test_frozen_collision_artifacts_are_refused_at_every_entry_point(
    tmp_path: Path, label: str
) -> None:
    from increment.analysis import Analysis
    from increment.query.artifact_publish import artifact_context
    from increment.query.source import open_artifact
    from increment.semantics.loader import load

    connection, store, ref, original_pin = _restored_legacy_artifact(label)
    end = "2025-01-25T00:00:00Z" if label == "z" else "2025-01-24T19:00:00-05:00"
    definitions_path = _legacy_collision_definitions(tmp_path, end)
    definitions = load(definitions_path)
    experiment = definitions.experiment("new_onboarding_v2")
    assert experiment is not None
    compiled_pin = artifact_context(definitions, experiment, "error")
    entry_points = {
        "adopt": lambda pin: Analysis.from_unit_day_artifact(store, ref, expected_context=pin),
        "source": lambda pin: open_artifact(store, ref, expected_context=pin),
        "reader": lambda pin: ArtifactMomentSource.open(store, ref, expected_context=pin),
    }
    try:
        for name, opener in entry_points.items():
            for pin_name, pin in (("original", original_pin), ("compiled", compiled_pin)):
                with pytest.raises(ArtifactContractError) as refused:
                    opener(pin)
                assert refused.value.code == "artifact.context.mismatch", (name, pin_name)
                assert refused.value.context["missing"] == "window_days", (name, pin_name)
        analysis = Analysis.from_definitions("new_onboarding_v2", definitions_path, connection)
        try:
            with pytest.raises(ArtifactContractError) as refreshed:
                analysis.publish_unit_day_artifact(store, refresh_of=ref)
            assert refreshed.value.code == "artifact.context.mismatch"
            assert refreshed.value.context["missing"] == "window_days"
        finally:
            analysis.close()
    finally:
        connection.disconnect()
