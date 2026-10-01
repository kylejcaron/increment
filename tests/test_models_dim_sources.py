"""Validation behaviour of DimSource / DimValidity / Definitions.dim_sources."""

import pytest

from increment.errors import DefinitionError
from increment.semantics.models import Definitions, DimSource, DimValidity


def _static_dim(**overrides):
    base = {
        "name": "users",
        "sql": "SELECT user_id, country FROM dim_user",
        "entity": "user_id",
        "properties": [
            {"name": "country", "column": "country", "dtype": "string", "as_of": "static"}
        ],
    }
    base.update(overrides)
    return base


def _versioned_dim(**overrides):
    base = {
        "name": "user_plan",
        "sql": "SELECT user_id, plan, valid_from, valid_to FROM snap_user_plan",
        "entity": "user_id",
        "validity": {"valid_from": "valid_from", "valid_to": "valid_to"},
        "properties": [
            {"name": "plan", "column": "plan", "dtype": "string", "as_of": "pre_exposure"}
        ],
    }
    base.update(overrides)
    return base


def _fact_source(**overrides):
    base = {
        "name": "orders",
        "sql": "SELECT 'order' AS event, user_id, ordered_at, amount FROM fact_orders",
        "timestamp_column": "ordered_at",
        "entities": ["user_id"],
        "dims": ["users"],
        "facts": [{"name": "order", "column": "amount"}],
    }
    base.update(overrides)
    return base


class TestDimValidity:
    def test_ranges_shape_accepted(self):
        v = DimValidity(valid_from="valid_from", valid_to="valid_to")
        assert v.changed_at is None

    def test_changelog_shape_accepted(self):
        v = DimValidity(changed_at="updated_at")
        assert v.valid_from is None

    def test_mixing_ranges_and_changelog_rejected(self):
        with pytest.raises(DefinitionError) as exc_info:
            DimValidity(valid_from="valid_from", valid_to="valid_to", changed_at="updated_at")
        assert exc_info.value.code == "definition.dim.validity_mutually_exclusive_encoding"

    def test_half_a_range_rejected(self):
        with pytest.raises(DefinitionError) as exc_info:
            DimValidity(valid_from="valid_from")
        assert exc_info.value.code == "definition.dim.validity_ranges_need"

    def test_empty_validity_rejected(self):
        with pytest.raises(DefinitionError) as exc_info:
            DimValidity()
        assert exc_info.value.code == "definition.dim.validity_requires_one_encoding"


class TestDimSourceAsOf:
    def test_static_dim_accepts_static_properties(self):
        DimSource.model_validate(_static_dim())

    def test_static_dim_rejects_time_varying_property(self):
        bad = _static_dim(
            properties=[{"name": "country", "column": "country", "as_of": "pre_exposure"}]
        )
        with pytest.raises(DefinitionError) as exc_info:
            DimSource.model_validate(bad)
        assert exc_info.value.code == "definition.dim.source_no_validity"

    def test_versioned_dim_rejects_static_property(self):
        bad = _versioned_dim(properties=[{"name": "plan", "column": "plan", "as_of": "static"}])
        with pytest.raises(DefinitionError) as exc_info:
            DimSource.model_validate(bad)
        assert exc_info.value.code == "definition.dim.source_versioned_validity"

    def test_versioned_dim_accepts_event_time_property(self):
        ok = _versioned_dim(properties=[{"name": "plan", "column": "plan", "as_of": "event_time"}])
        DimSource.model_validate(ok)

    def test_dim_requires_at_least_one_property(self):
        with pytest.raises(DefinitionError) as exc_info:
            DimSource.model_validate(_static_dim(properties=[]))
        assert exc_info.value.code == "definition.dim.source_declares_no"

    def test_duplicate_property_name_within_one_dim_rejected(self):
        # Regression: DimSource had no intrinsic uniqueness check on its own
        # properties list (unlike fact sources) - two properties named 'country' on the SAME dim source used to load silently, the second entry winning any name-keyed lookup downstream.
        bad = _static_dim(
            properties=[
                {"name": "country", "column": "country_code", "as_of": "static"},
                {"name": "country", "column": "country_name", "as_of": "static"},
            ]
        )
        with pytest.raises(DefinitionError) as exc_info:
            DimSource.model_validate(bad)
        assert exc_info.value.code == "definition.dim.duplicate_property_name"


class TestDefinitionsCrossChecks:
    def test_unknown_dim_reference_rejected(self):
        with pytest.raises(DefinitionError) as exc_info:
            Definitions.model_validate(
                {"fact_sources": [_fact_source(dims=["users"])], "dim_sources": []}
            )
        assert exc_info.value.code == "definition.invalid"
        assert "definition.index_sources.fact_source_references" in {
            c
            for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
        }

    def test_unknown_dim_reference_with_metric_on_source_still_raises_definition_error(self):
        # Regression: an unknown dim reference used to escape as a raw KeyError,
        # discarding every other accumulated cross-check error.
        with pytest.raises(DefinitionError) as exc_info:
            Definitions.model_validate(
                {
                    "fact_sources": [_fact_source(dims=["ghost"])],
                    "dim_sources": [],
                    "metrics": [
                        {"name": "conv", "type": "conversion", "entity": "user_id", "fact": "order"}
                    ],
                }
            )
        assert exc_info.value.code == "definition.invalid"
        assert "definition.index_sources.fact_source_references" in {
            c
            for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
        }

    def test_dim_entity_must_be_in_fact_source_entities(self):
        dim = _static_dim(entity="account_id")
        with pytest.raises(DefinitionError) as exc_info:
            Definitions.model_validate({"fact_sources": [_fact_source()], "dim_sources": [dim]})
        assert exc_info.value.code == "definition.invalid"
        assert "definition.index_sources.fact_source_joins" in {
            c
            for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
        }

    def test_dim_property_colliding_with_own_property_rejected(self):
        fs = _fact_source(properties=[{"name": "country", "column": "country", "as_of": "static"}])
        with pytest.raises(DefinitionError) as exc_info:
            Definitions.model_validate({"fact_sources": [fs], "dim_sources": [_static_dim()]})
        assert exc_info.value.code == "definition.invalid"
        assert "definition.index_sources.duplicate_property_name" in {
            c
            for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
        }

    def test_two_dims_colliding_property_rejected(self):
        other = _static_dim(name="users2")
        with pytest.raises(DefinitionError) as exc_info:
            Definitions.model_validate(
                {
                    "fact_sources": [_fact_source(dims=["users", "users2"])],
                    "dim_sources": [_static_dim(), other],
                }
            )
        assert exc_info.value.code == "definition.invalid"
        assert "definition.index_sources.duplicate_property_name" in {
            c
            for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
        }

    def test_duplicate_dim_source_names_rejected(self):
        with pytest.raises(DefinitionError) as exc_info:
            Definitions.model_validate(
                {"fact_sources": [], "dim_sources": [_static_dim(), _static_dim()]}
            )
        assert exc_info.value.code == "definition.invalid"
        assert "definition.index_sources.duplicate_dim_source" in {
            c
            for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
        }

    def test_properties_of_merges_dim_properties(self):
        defs = Definitions.model_validate(
            {"fact_sources": [_fact_source()], "dim_sources": [_static_dim()]}
        )
        fs = defs.fact_sources[0]
        assert [p.name for p in defs.properties_of(fs)] == ["country"]
        # fs.properties itself is NOT mutated - round-trip safety.
        assert fs.properties == ()

    def test_fact_source_for_resolves_dim_property(self):
        defs = Definitions.model_validate(
            {"fact_sources": [_fact_source()], "dim_sources": [_static_dim()]}
        )
        assert defs.fact_source_for("country") is defs.fact_sources[0]

    def test_fact_precedes_property_across_source_order(self):
        profile = _fact_source(
            name="profile",
            dims=[],
            facts=[{"name": "profile_event", "column": None}],
            properties=[{"name": "purchase", "column": "purchase", "as_of": "static"}],
        )
        events = _fact_source(
            name="events",
            dims=[],
            facts=[{"name": "purchase", "column": None}],
        )
        for sources in ((profile, events), (events, profile)):
            defs = Definitions.model_validate({"fact_sources": list(sources)})
            owner = defs.fact_source_for("purchase")
            assert owner is not None
            assert owner.name == "events"

    def test_fact_source_for_resolves_own_property(self):
        source = _fact_source(
            name="orders",
            dims=[],
            properties=[{"name": "region", "column": "region", "as_of": "static"}],
        )
        defs = Definitions.model_validate({"fact_sources": [source]})
        owner = defs.fact_source_for("region")
        assert owner is not None
        assert owner.name == "orders"

    def test_dim_property_named_like_reserved_column_rejected(self):
        # `ts`/`unit_id`/`event`/`experiment_id` are builder-reserved;
        # `valid_from`/`valid_to` are generated by the range-join projection. A dim property with any of these names would silently shadow or collide downstream.
        dim = _static_dim(properties=[{"name": "ts", "column": "ts_col", "as_of": "static"}])
        with pytest.raises(DefinitionError) as exc_info:
            Definitions.model_validate({"fact_sources": [_fact_source()], "dim_sources": [dim]})
        assert exc_info.value.code == "definition.invalid"
        assert "definition.index_sources.dim_source_property" in {
            c
            for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
        }

    def test_dim_backed_breakout_dtype_checked_against_filters(self):
        # A metric filter on a dim-contributed property goes through the same
        # dtype-compatibility check as an inline property.
        defs = {
            "fact_sources": [_fact_source()],
            "dim_sources": [_static_dim()],
            "metrics": [
                {
                    "name": "orders_de",
                    "type": "conversion",
                    "entity": "user_id",
                    "fact": "order",
                    "filters": [{"property": "country", "op": "equals", "values": ["DE"]}],
                }
            ],
        }
        Definitions.model_validate(defs)  # must not raise "unknown property"
