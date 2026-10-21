"""``_rename_to_builder_cols``: property renames must never clobber a
canonical builder column or a fact's own value column."""

import datetime as dt

import ibis
import pytest

from increment import Analysis
from increment.errors import DefinitionError, InvalidRequestError
from increment.query.fact_resolution import (
    _find_fact_source,
    _rename_to_builder_cols,
    _resolve_breakout_fact_source,
)
from increment.query.session import WarehouseSession
from increment.semantics.design import Randomized
from increment.semantics.models import Breakout, Definitions, Experiment, Fact, FactSource, Property
from tests._unit_boundary import backend_fixture

pytestmark = [pytest.mark.creates_tables, pytest.mark.integration]


def test_find_fact_source_reports_the_missing_name_and_every_declared_fact():
    defs = Definitions.model_validate(
        {
            "fact_sources": [
                {
                    "name": "orders",
                    "sql": "unused",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [{"name": "purchase", "column": "amount"}],
                }
            ]
        }
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        _find_fact_source(defs, "ghost_fact")
    assert exc_info.value.code == "query.fact_resolution.fact_declared_any"
    assert exc_info.value.context["fact_name"] == "ghost_fact"
    assert exc_info.value.context["available"] == ("purchase",)


@pytest.mark.parametrize(
    "control, design",
    [(None, None), ("control", Randomized(control_group="control"))],
)
def test_require_design_rejects_zero_or_two_of_control_and_design(control, design):
    with pytest.raises(InvalidRequestError) as exc_info:
        Analysis.from_moments((), metrics=(), control=control, design=design)
    assert exc_info.value.code == "query.fact_resolution.pass_exactly_one"
    assert exc_info.value.context["constructor"] == "Analysis.from_moments"


def _breakout_fact_source():
    return FactSource.model_construct(
        name="events",
        sql="unused",
        timestamp_column="ts",
        entities=("user_id",),
        facts=(Fact(name="page_view", column=None),),
        properties=(Property(name="country", column="country", as_of="static"),),
        dims=(),
    )


def test_resolve_breakout_fact_source_rejects_an_unknown_explicit_source():
    fs = _breakout_fact_source()
    defs = Definitions.model_construct(fact_sources=(fs,))
    experiment = Experiment.model_construct(unit="user_id")
    breakout = Breakout(property="country", source="ghost_source")
    with pytest.raises(InvalidRequestError) as exc_info:
        _resolve_breakout_fact_source(defs, experiment, breakout)
    assert exc_info.value.code == "query.fact_resolution.breakout_property_references"
    assert exc_info.value.context["property"] == "country"
    assert exc_info.value.context["source"] == "ghost_source"


def test_resolve_breakout_fact_source_rejects_an_unresolvable_property():
    fs = _breakout_fact_source()
    defs = Definitions.model_construct(fact_sources=(fs,))
    experiment = Experiment.model_construct(unit="user_id")
    breakout = Breakout(property="ghost_property")
    with pytest.raises(InvalidRequestError) as exc_info:
        _resolve_breakout_fact_source(defs, experiment, breakout)
    assert exc_info.value.code == "query.fact_resolution.breakout_property_could"
    assert exc_info.value.context["property"] == "ghost_property"
    assert exc_info.value.context["unit"] == "user_id"


def _source(properties):
    return FactSource.model_validate(
        {
            "name": "orders",
            "sql": "SELECT * FROM orders",
            "timestamp_column": "ts",
            "entities": ["user_id"],
            "facts": [{"name": "purchase", "column": "amount"}],
            "properties": properties,
        }
    )


def _unvalidated_source(properties):
    """A FactSource that skipped validation.

    The declaration boundary refuses these collisions at construction, so the
    resolution-layer guard is only reachable for a source built without
    validation. Exercising it keeps that defence honest rather than dead.
    """
    return FactSource.model_construct(
        name="orders",
        sql="unused",
        timestamp_column="ts",
        entities=("user_id",),
        facts=(Fact(name="purchase", column="amount"),),
        properties=tuple(Property.model_construct(**p) for p in properties),
    )


def _table(con):
    return con.create_table(
        "orders",
        ibis.memtable(
            {
                "user_id": ["u1"],
                "ts": [dt.datetime(2025, 1, 1)],
                "amount": [100.0],
                "event": ["purchase"],
                "country": [7],
            }
        ),
    )


def _fact_table(con, fs, unit="user_id"):
    """Resolve a fact source the way the builders do."""
    defs = Definitions.model_construct(fact_sources=(fs,))
    return WarehouseSession(con, defs).fact_table(fs, unit)


@backend_fixture()
def con():
    return ibis.duckdb.connect()


def test_property_renaming_onto_a_fact_value_column_is_rejected(con):
    # Property(name="amount", column="country") would rename "country" onto
    # the existing "amount" value column, silently overwriting it.
    # Primary boundary: the declaration refuses it at construction.
    with pytest.raises(DefinitionError) as raised:
        _source([{"name": "amount", "column": "country", "as_of": "event_time"}])
    assert raised.value.code == "definition.fact.source_renames_propert"

    # Defence in depth: still refused if a source skipped validation.
    fs = _unvalidated_source([{"name": "amount", "column": "country", "as_of": "event_time"}])
    with pytest.raises(InvalidRequestError) as exc_info:
        _rename_to_builder_cols(_table(con), fs, "user_id")
    assert exc_info.value.code == "query.fact_resolution.fact_source_property"
    assert exc_info.value.context["name"] == "orders"
    assert exc_info.value.context["prop_name"] == "amount"


def test_property_renaming_away_a_fact_value_column_is_rejected(con):
    # Property(name="country", column="amount") would rename the fact's own
    # "amount" value column away, deleting it under a different name. A
    # name-only check on the destination alone would miss this.
    # Primary boundary: the declaration refuses it at construction.
    with pytest.raises(DefinitionError) as raised:
        _source([{"name": "country", "column": "amount", "as_of": "event_time"}])
    assert raised.value.code == "definition.fact.source_renames_its"

    # Defence in depth: still refused if a source skipped validation.
    fs = _unvalidated_source([{"name": "country", "column": "amount", "as_of": "event_time"}])
    with pytest.raises(InvalidRequestError) as exc_info:
        _rename_to_builder_cols(_table(con), fs, "user_id")
    assert exc_info.value.code == "query.fact_resolution.fact_source_property_deletes_value_column"
    assert exc_info.value.context["name"] == "orders"
    assert exc_info.value.context["prop_column"] == "amount"


def test_identity_property_declaration_is_allowed(con):
    # Property(name="amount", column="amount") performs no rename at all
    # and must not be flagged, even though "amount" is a fact value column.
    fs = _source([{"name": "amount", "column": "amount", "as_of": "event_time"}])
    _table(con)
    tbl = _fact_table(con, fs)
    assert tbl.execute()["amount"].tolist() == [100.0]


def test_property_named_after_a_canonical_builder_column_is_rejected(con):
    # "ts"/"event"/"experiment_id" are canonical builder columns even with no fact
    # on that physical column; the builder's rename lands on the name, so identity
    # mapping is as unsafe as a rename. Primary boundary: construction refuses it.
    for column in ("country", "ts"):
        with pytest.raises(DefinitionError) as raised:
            _source([{"name": "ts", "column": column, "as_of": "event_time"}])
        assert raised.value.code == "definition.fact.source_declares_propert"

    # Defence in depth: still refused if a source skipped validation, for the
    # identity declaration too.
    tbl = _table(con)
    for column in ("country", "ts"):
        fs = _unvalidated_source([{"name": "ts", "column": column, "as_of": "event_time"}])
        with pytest.raises(InvalidRequestError) as exc_info:
            _rename_to_builder_cols(tbl, fs, "user_id")
        assert exc_info.value.code == "query.fact_resolution.fact_source_property"
        assert exc_info.value.context["prop_name"] == "ts"


def test_unrelated_property_rename_still_applies(con):
    fs = _source([{"name": "region", "column": "country", "as_of": "event_time"}])
    _table(con)
    tbl = _fact_table(con, fs)
    assert tbl.execute()["region"].tolist() == [7]


def test_property_reading_the_fallback_entity_column_is_rejected(con):
    """When the analysed `unit` is not one of this source's entities -- a
    denominator-only fact source, say -- the rename falls back to the first
    declared entity. A property reading THAT column is renamed away just the
    same, so the guard and the rename resolve the entity source once and share
    it rather than each deciding independently."""
    con.create_table(
        "orders",
        ibis.memtable(
            {
                "session_id": ["s1"],
                "user_id": ["u1"],
                "ts": [dt.datetime(2025, 1, 1)],
                "event": ["purchase"],
                "amount": [100.0],
            }
        ),
    )
    fs = FactSource(
        name="orders",
        sql="SELECT * FROM orders",
        timestamp_column="ts",
        entities=["session_id"],
        facts=[Fact(name="purchase", column="amount")],
        properties=[Property(name="sess", column="session_id", as_of="event_time")],
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        _fact_table(con, fs, "user_id")
    assert (
        exc_info.value.code == "query.fact_resolution.fact_source_property_reads_canonical_column"
    )
    assert exc_info.value.context["prop_column"] == "session_id"


def test_fact_source_missing_event_refuses_before_data_query_and_survives_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import copy
    import pickle

    from increment.query.fact_resolution import _dim_joined_fact_table

    con = ibis.duckdb.connect()
    fs = FactSource.model_validate(
        {
            "name": "orders",
            "sql": "SELECT user_id, event_time AS ts FROM orders",
            "timestamp_column": "event_time",
            "entities": ["user_id"],
            "facts": [{"name": "purchase", "column": "amount"}],
        }
    )
    defs = Definitions.model_construct(fact_sources=(fs,), dim_sources=())
    relation = ibis.memtable({"user_id": ["u1"], "event_time": [dt.datetime(2025, 1, 1)]})

    def unexpected_query(*args: object, **kwargs: object) -> None:
        raise AssertionError("schema refusal must precede data execution")

    monkeypatch.setattr(con, "execute", unexpected_query)
    with pytest.raises(DefinitionError) as raised:
        _dim_joined_fact_table(con, defs, fs, "user_id", {}, resolve_sql=lambda _sql: relation)
    error = raised.value
    assert error.code == "definition.fact_source.missing_columns"
    assert error.context["fact_source"] == "orders"
    assert error.context["missing"] == ("event",)
    for restored in (copy.deepcopy(error), pickle.loads(pickle.dumps(error))):
        assert restored.code == error.code
        assert dict(restored.context) == dict(error.context)


@pytest.mark.parametrize("missing_source", ["exposure", "metric"])
@pytest.mark.parametrize("operation", ["run", "publish"])
@pytest.mark.integration
def test_missing_fact_source_event_refuses_before_public_data_query(
    tmp_path, monkeypatch: pytest.MonkeyPatch, missing_source: str, operation: str
) -> None:
    import yaml

    from tests.test_unit_day_artifact_facade import _definitions, _native

    definitions = _definitions(tmp_path)
    payload = yaml.safe_load(definitions.read_text())
    experiment = next(
        item for item in payload["experiments"] if item["name"] == "new_onboarding_v2"
    )
    if missing_source == "metric":
        experiment.pop("factors", None)
        experiment.pop("breakouts", None)
    experiment["plan"] = {"secondaries": ["purchase_rate"]}
    if missing_source == "metric":
        source = next(item for item in payload["fact_sources"] if item["name"] == "event_log")
        source["facts"] = [fact for fact in source["facts"] if fact["name"] != "purchase"]
        payload["fact_sources"].append(
            {
                "name": "purchase_rows",
                "sql": (
                    "SELECT user_id, session_id, event_at, revenue, country_code, "
                    "device_type, plan FROM analytics.event_log WHERE event = 'purchase'"
                ),
                "timestamp_column": "event_at",
                "entities": ["user_id", "session_id"],
                "facts": [{"name": "purchase", "column": "revenue"}],
                "properties": source["properties"],
            }
        )
    definitions.write_text(yaml.safe_dump(payload, sort_keys=False))
    connection, analysis, _context, store = _native(definitions=definitions)
    if missing_source == "exposure":
        connection.raw_sql(
            """
            CREATE OR REPLACE TABLE analytics.event_log AS
            SELECT * EXCLUDE (event) FROM analytics.event_log
            """
        )

    data_queries: list[object] = []

    def reject_data_query(*args: object, **kwargs: object) -> None:
        data_queries.append((args, kwargs))
        raise AssertionError("schema refusal must precede source snapshot execution")

    monkeypatch.setattr(connection, "create_table", reject_data_query)
    try:
        with pytest.raises(DefinitionError) as raised:
            if operation == "run":
                analysis.run()
            else:
                analysis.publish_unit_day_artifact(store)
        assert raised.value.code == "definition.fact_source.missing_columns"
        assert raised.value.context["missing"] == ("event",)
        assert not data_queries
    finally:
        analysis.close()
