"""Unit-frame schemas, pinned execution, and refusal ordering for adopted artifacts."""

import gc
import tracemalloc
import warnings
from datetime import datetime

import ibis
import ibis.expr.types as ir
import narwhals as nw
import pytest

from increment.analysis import Analysis
from increment.errors import CapabilityError, IncrementWarning
from increment.query.artifact_publish import artifact_context
from increment.query.artifact_reader import ArtifactMomentSource
from increment.query.session import WarehouseArtifactStore
from increment.query.source import open_artifact
from increment.semantics.artifact import ClusterIdentityRequest
from increment.semantics.loader import load
from increment.semantics.models import Breakout
from tests.analysis_factory import _native_source, native_connection

pytestmark = pytest.mark.slow


def test_publication_rejects_conflicting_canonical_cluster_identity(published_units):
    from increment.errors import InvalidRequestError

    native, _ = published_units
    con = native_connection(native)
    con.raw_sql("""INSERT INTO events
        SELECT user_id, event_at, event, experiment_id, group_id, 'conflict', revenue
        FROM events WHERE user_id = 'u1' AND event = 'exposure'""")
    store = WarehouseArtifactStore(con, schema_name="conflicting_artifacts")
    with pytest.raises(InvalidRequestError) as caught:
        native.publish_unit_day_artifact(store)
    assert caught.value.code == "query.integrity.cluster_conflict"


@pytest.fixture
def published_units(tmp_path):
    definitions = tmp_path / "definitions.yaml"
    definitions.write_text("""
dialect: duckdb
fact_sources:
  - name: events
    sql: SELECT * FROM events
    timestamp_column: event_at
    entities: [user_id]
    facts:
      - {name: exposure, column: null}
      - {name: purchase, column: revenue}
exposures:
  - {name: enrolled, fact: exposure}
metrics:
  - name: revenue
    type: mean
    entity: user_id
    fact: purchase
    aggregation: sum
    window_days: 2
  - name: average_order
    type: ratio
    entity: user_id
    numerator: {fact: purchase, aggregation: sum, window_days: 2}
    denominator: {fact: purchase, aggregation: count, window_days: 2}
experiments:
  - name: test
    exposure: enrolled
    unit: user_id
    cluster: store_id
    start: 2025-08-01
    end: 2025-08-03
    control_group: control
    plan: {secondaries: [revenue, average_order]}
""")
    events = []
    for unit, group, day, values in (
        ("u1", "control", 1, [10.0]),
        ("u2", "treatment", 1, [5.0, 15.0]),
        ("u3", "control", 3, [30.0]),
        ("u4", "treatment", 3, [40.0]),
    ):
        events.append(
            {
                "user_id": unit,
                "event_at": datetime(2025, 8, day, 9),
                "event": "exposure",
                "experiment_id": "test",
                "group_id": group,
                "store_id": f"s{unit}",
                "revenue": None,
            }
        )
        for value in values:
            events.append(
                {
                    "user_id": unit,
                    "event_at": datetime(2025, 8, max(day, 2), 10),
                    "event": "purchase",
                    "experiment_id": None,
                    "group_id": None,
                    "store_id": None,
                    "revenue": value,
                }
            )
    connection = ibis.duckdb.connect()
    connection.create_table("events", obj=events)
    native = Analysis.from_definitions("test", definitions, connection)
    loaded = load(definitions)
    context = artifact_context(loaded, loaded.experiments[0], "error")
    store = WarehouseArtifactStore(connection, schema_name="artifacts")
    reference = native.publish_unit_day_artifact(
        store, extensions=[ClusterIdentityRequest(cluster_name="store_id")]
    )
    adopted = open_artifact(store, reference, expected_context=context)
    try:
        yield _native_source(native), adopted
    finally:
        adopted.close()
        native.close()
        connection.disconnect()


@pytest.mark.parametrize("metric_name", ["revenue", "average_order"])
@pytest.mark.parametrize("source_kind", ["native", "artifact"])
def test_published_unit_frames_keep_outcomes_and_censor_through_snapshot(
    published_units, monkeypatch, metric_name, source_kind
):
    native, adopted = published_units
    source = native if source_kind == "native" else adopted
    metric = next(m for m in source.context.metrics if m.name == metric_name)

    def direct_execution_forbidden(*args, **kwargs):
        pytest.fail("artifact execution must use the pinned snapshot")

    if source_kind == "artifact":
        monkeypatch.setattr(ir.Expr, "execute", direct_execution_forbidden)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", UserWarning)
        actual = nw.from_native(source.unit_frame(metric), eager_only=True).sort("unit_id")
    assert any(issubclass(item.category, UserWarning) for item in recorded)
    assert actual["unit_id"].to_list() == ["u1", "u2"]
    assert actual["group_id"].to_list() == ["control", "treatment"]
    assert actual["cluster_id"].to_list() == ["su1", "su2"]
    assert actual["y"].to_list() == [10.0, 20.0]
    if metric_name == "average_order":
        assert actual["y_den"].to_list() == [1.0, 2.0]
    else:
        assert "y_den" not in actual.columns


@pytest.mark.parametrize("grain", ["daily", "asof"])
@pytest.mark.parametrize("operation", ["plain", "dimension", "cuped", "dimensioned", "breakout"])
def test_clustered_non_total_refuses_before_extension_resolution(published_units, grain, operation):
    _, adopted = published_units
    metric = adopted.context.metrics[0]

    # The grain refusal precedes extension resolution: the dimensioned and
    # covariate cases name a missing extension, yet report the grain code.
    with pytest.raises(CapabilityError) as caught:
        if operation == "breakout":
            adopted.breakout_moments(metric, Breakout(property="missing"), grain=grain)
        else:
            adopted.moments(
                metric,
                grain=grain,
                by=["missing"] if operation in {"dimension", "dimensioned"} else [],
                include_covariate=operation in {"cuped", "dimensioned"},
            )
    assert caught.value.code == "artifact.operation.unsupported"


def _published_plain_reader(root, *, n_units: int, n_days: int, aggregation: str = "sum"):
    """A plain (no-extension) artifact reader over an unwindowed mean metric.

    Every unit is exposed on day one and purchases once per day for
    *n_days* days, so the per-unit outcome under ``avg_event`` (the mean
    over events) differs from the per-day sum whenever *n_days* > 1.
    """
    root.mkdir(parents=True, exist_ok=True)
    definitions = root / "definitions.yaml"
    definitions.write_text(f"""
dialect: duckdb
fact_sources:
  - name: events
    sql: SELECT * FROM events
    timestamp_column: event_at
    entities: [user_id]
    facts:
      - {{name: exposure, column: null}}
      - {{name: purchase, column: revenue}}
exposures:
  - {{name: enrolled, fact: exposure}}
metrics:
  - name: revenue
    type: mean
    entity: user_id
    fact: purchase
    aggregation: {aggregation}
experiments:
  - name: test
    exposure: enrolled
    unit: user_id
    start: 2025-08-01
    end: 2025-08-28
    control_group: control
    plan: {{primary: revenue}}
""")
    events = []
    for index in range(n_units):
        unit = f"u{index:05d}"
        group = "control" if index % 2 == 0 else "treatment"
        events.append(
            {
                "user_id": unit,
                "event_at": datetime(2025, 8, 1, 9),
                "event": "exposure",
                "experiment_id": "test",
                "group_id": group,
                "revenue": None,
            }
        )
        for day in range(1, n_days + 1):
            events.append(
                {
                    "user_id": unit,
                    "event_at": datetime(2025, 8, day, 10),
                    "event": "purchase",
                    "experiment_id": None,
                    "group_id": None,
                    "revenue": float(day + index % 3),
                }
            )
    connection = ibis.duckdb.connect()
    connection.create_table("events", obj=events)
    # An unwindowed mean metric is loaded with a coded warning; the reader under
    # test is the point, not the loader's advice.
    with pytest.warns(IncrementWarning):
        native = Analysis.from_definitions("test", definitions, connection)
    with pytest.warns(IncrementWarning):
        loaded = load(definitions)
    context = artifact_context(loaded, loaded.experiments[0], "error")
    store = WarehouseArtifactStore(connection, schema_name="artifacts")
    reference = native.publish_unit_day_artifact(store)
    reader = ArtifactMomentSource.open(store, reference, expected_context=context)
    return reader, native, connection


def test_plain_reader_unit_frame_agrees_with_its_own_total_moments(tmp_path):
    """unit_frame and moments(grain="total") see the same per-unit values:
    under avg_event each unit's value is its mean over events, and the
    frame's per-group mean of those values is the moments' mean."""
    reader, native, connection = _published_plain_reader(
        tmp_path, n_units=8, n_days=3, aggregation="avg_event"
    )
    try:
        metric = reader.context.metrics[0]
        frame = nw.from_native(reader.unit_frame(metric), eager_only=True)
        for row in reader.moments(metric):
            values = frame.filter(nw.col("group_id") == row["group_id"])["y"].to_list()
            assert len(values) == row["n"]
            mean = row["ref_y"] + row["cy1"] / row["n"]
            assert sum(values) / len(values) == pytest.approx(mean, rel=1e-12)
    finally:
        reader.close()
        native.close()
        connection.disconnect()


def _unit_frame_peak_bytes(reader, metric) -> int:
    reader.moments(metric)  # warm caches (metric specs, verified relations)
    gc.collect()
    tracemalloc.start()
    try:
        reader.unit_frame(metric)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def test_plain_reader_unit_frame_memory_does_not_grow_with_the_day_axis(tmp_path):
    """Serving a per-unit frame must not materialize the unit x day panel in
    Python: four times the days must not add more than a bounded amount."""
    short = _published_plain_reader(tmp_path / "short", n_units=2000, n_days=4)
    long = _published_plain_reader(tmp_path / "long", n_units=2000, n_days=16)
    try:
        peak_short = _unit_frame_peak_bytes(short[0], short[0].context.metrics[0])
        peak_long = _unit_frame_peak_bytes(long[0], long[0].context.metrics[0])
        assert peak_long - peak_short < 2 * 1024 * 1024, (peak_short, peak_long)
    finally:
        for reader, native, connection in (short, long):
            reader.close()
            native.close()
            connection.disconnect()
