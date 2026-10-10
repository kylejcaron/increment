"""Coded-refusal regression tests for increment.query.artifact_contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from increment.errors import InvalidRequestError
from increment.query import artifact_contract as ac
from increment.semantics import Definitions

_EXPERIMENT = "site_volume_test"


def test_artifact_contract_error_without_code_carries_refusal_code() -> None:
    with pytest.raises(InvalidRequestError) as exc_info:
        ac.ArtifactContractError("not a known refusal code")
    assert exc_info.value.code == "artifact.artifact_contract.refusal_code"


def _definitions() -> Any:
    """One unit-grain fact source, one mean metric, one experiment using it."""
    return Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT 1",
                    "timestamp_column": "ts",
                    "entities": ["unit_id"],
                    "facts": [{"name": "purchase", "column": "value"}],
                }
            ],
            "exposures": [{"name": "assigned", "fact": "purchase"}],
            "metrics": [
                {
                    "name": "revenue",
                    "type": "mean",
                    "entity": "unit_id",
                    "fact": "purchase",
                    "aggregation": "sum",
                    "window_days": 7,
                }
            ],
            "experiments": [
                {
                    "name": _EXPERIMENT,
                    "exposure": "assigned",
                    "unit": "unit_id",
                    "control_group": "control",
                    "start": "2025-08-01T00:00:00+00:00",
                    "plan": {"secondaries": ["revenue"]},
                }
            ],
        }
    )


def _coverage(first_ds: Any, last_ds: str) -> dict[str, Any]:
    return {
        "first_ds": first_ds,
        "last_ds": last_ds,
        "freshness": {"loaded_through": "2024-01-05"},
    }


def test_compile_context_rejects_unknown_on_mixed_assignment() -> None:
    with pytest.raises(InvalidRequestError) as exc_info:
        ac.compile_unit_day_artifact_context(
            _EXPERIMENT, _definitions(), on_mixed_assignment=cast("Any", "bogus")
        )
    assert exc_info.value.code == "artifact.on_mixed_assignment"


def test_artifact_context_round_trips_declared_allocation_scheme() -> None:
    definitions = _definitions()
    experiment = definitions.experiment(_EXPERIMENT)
    assert experiment is not None
    declared = experiment.model_copy(update={"allocation_scheme": "independent"})
    definitions = definitions.model_copy(update={"experiments": (declared,)})

    context = ac.compile_unit_day_artifact_context(_EXPERIMENT, definitions)
    stored = json.loads(context.canonical_json)["experiment"]
    assert stored["allocation_scheme"] == "independent"
    legacy = ac.compile_unit_day_artifact_context(_EXPERIMENT, _definitions())
    assert "allocation_scheme" not in json.loads(legacy.canonical_json)["experiment"]

    assert definitions.experiment(_EXPERIMENT).resolved_design().allocation_scheme == "independent"


def test_compile_context_rejects_non_date_site_volume_coverage() -> None:
    with pytest.raises(InvalidRequestError) as exc_info:
        ac.compile_unit_day_artifact_context(
            _EXPERIMENT, _definitions(), site_volume_coverage=_coverage(20240101, "2024-01-02")
        )
    assert exc_info.value.code == "artifact.coverage_dates_date"


def test_compile_context_rejects_first_ds_after_last_ds() -> None:
    definitions = _definitions()
    context = ac.compile_unit_day_artifact_context(
        _EXPERIMENT, definitions, site_volume_coverage=_coverage("2024-01-01", "2024-01-01")
    )
    (site_volume,) = [
        entry
        for entry in ac.unit_day_artifact_extension_catalog(context)
        if entry.request.kind == "site_volume"
    ]
    assert json.loads(site_volume.canonical_definition_json)["first_ds"] == "2024-01-01"
    with pytest.raises(InvalidRequestError) as exc_info:
        ac.compile_unit_day_artifact_context(
            _EXPERIMENT, definitions, site_volume_coverage=_coverage("2024-02-01", "2024-01-01")
        )
    assert exc_info.value.code == "artifact.first_ds_last"


def test_refusal_registry_is_read_only() -> None:
    from increment.query.artifact_contract import REFUSALS

    code = next(iter(REFUSALS))
    with pytest.raises(TypeError):
        REFUSALS[code] = None  # type: ignore[index]  # ty: ignore[invalid-assignment]
    with pytest.raises(TypeError):
        del REFUSALS[code]  # ty: ignore[not-subscriptable]


def test_extension_catalog_offers_exactly_the_declared_observational_covariates(tmp_path):
    from increment.query.artifact_publish import artifact_context
    from increment.semantics import load
    from tests.covariate_cases import covariate_defs_and_con

    for observational, expected in ((True, [("tenure", "events")]), (False, [])):
        defs_path, con, *_ = covariate_defs_and_con(tmp_path, observational=observational)
        con.disconnect()
        defs = load(defs_path)
        experiment = defs.experiment("cov_test")
        assert experiment is not None
        context = artifact_context(defs, experiment, "error")
        offered = [
            (entry.request.property_name, entry.request.source_name)
            for entry in ac.unit_day_artifact_extension_catalog(context)
            if entry.request.kind == "unit_covariate"
        ]
        assert offered == expected


def test_extension_catalog_types_a_string_covariate_as_a_level_extension(tmp_path):
    """The declared dtype picks the wire relation: a numeric property is
    offered as `unit_covariate`, a string property as `unit_covariate_level`,
    each requestable only under its own typed request."""
    from increment.query.artifact_publish import artifact_context
    from increment.semantics import load
    from increment.semantics.artifact import UnitCovariateLevelRequest, UnitCovariateRequest
    from tests.covariate_cases import categorical_defs_and_con

    defs_path, con, _units = categorical_defs_and_con(tmp_path)
    con.disconnect()
    defs = load(defs_path)
    experiment = defs.experiment("cat_test")
    assert experiment is not None
    context = artifact_context(defs, experiment, "error")
    offered = {
        entry.request.property_name: entry.request
        for entry in ac.unit_day_artifact_extension_catalog(context)
        if entry.request.kind in {"unit_covariate", "unit_covariate_level"}
    }
    assert offered == {
        "tenure": UnitCovariateRequest(property_name="tenure", source_name="events"),
        "region": UnitCovariateLevelRequest(property_name="region", source_name="events"),
    }


# ── format-2 provenance: hash-only, non-executable contexts ─────────────────

_SECRETS = {
    "fact_used": "SECRET_FACT_USED_5b1e",
    "fact_unused": "SECRET_FACT_UNUSED_77c2",
    "dim": "SECRET_DIM_a90d",
    "exposure": "SECRET_EXPOSURE_31fe",
    "trigger": "SECRET_TRIGGER_c4d8",
}
_CUPED = {"name": "cuped", "variance_reduction": "cuped"}


def _secret_payload() -> dict[str, Any]:
    """Definitions whose every source SQL carries a unique sentinel, used or not."""
    start = "2025-08-01T00:00:00+00:00"
    common = {"unit": "unit_id", "control_group": "control", "start": start}
    return {
        "dialect": "duckdb",
        "fact_sources": [
            {
                "name": "events",
                "sql": f"SELECT * FROM events /* {_SECRETS['fact_used']} */",
                "timestamp_column": "ts",
                "entities": ["unit_id"],
                "facts": [
                    {"name": "purchase", "column": "value"},
                    {"name": "clicked", "column": None},
                ],
                "properties": [
                    {"name": "country", "column": "country", "dtype": "string", "as_of": "static"}
                ],
                "dims": ["users"],
            },
            {
                "name": "unused_events",
                "sql": f"SELECT 2 /* {_SECRETS['fact_unused']} */",
                "timestamp_column": "ts",
                "entities": ["unit_id"],
                "facts": [{"name": "other", "column": "o"}],
            },
        ],
        "dim_sources": [
            {
                "name": "users",
                "sql": f"SELECT * FROM users /* {_SECRETS['dim']} */",
                "entity": "unit_id",
                "properties": [
                    {"name": "segment", "column": "segment", "dtype": "string", "as_of": "static"}
                ],
            }
        ],
        "exposures": [
            {
                "name": "direct",
                "sql": f"SELECT unit_id, ts, group_id FROM enrolled /* {_SECRETS['exposure']} */",
            },
            {
                "name": "trig",
                "sql": f"SELECT unit_id, ts, group_id FROM triggered /* {_SECRETS['trigger']} */",
            },
            {"name": "by_fact", "fact": "purchase"},
        ],
        "metrics": [
            {
                "name": "revenue",
                "type": "mean",
                "entity": "unit_id",
                "fact": "purchase",
                "aggregation": "sum",
                "window_days": 7,
            },
            {"name": "unrelated", "type": "mean", "entity": "unit_id", "fact": "other"},
        ],
        "experiments": [
            {
                **common,
                "name": "breakout_test",
                "exposure": "direct",
                "trigger": "trig",
                "breakouts": [{"property": "segment"}],
                "plan": {"secondaries": ["revenue"]},
            },
            {
                **common,
                "name": "cuped_test",
                "exposure": "by_fact",
                "n_pre_periods": 7,
                "plan": {"secondaries": [{"metric": "revenue", "sensitivity_methods": [_CUPED]}]},
            },
            {
                **common,
                "name": "encouragement_test",
                "exposure": "direct",
                "design": {"mechanism": "encouragement", "uptake": {"fact": "clicked"}},
                "plan": {"secondaries": ["revenue"]},
            },
        ],
    }


def _compile(payload: dict[str, Any], experiment: str) -> Any:
    return ac.compile_unit_day_artifact_context(experiment, Definitions.model_validate(payload))


@pytest.mark.parametrize("experiment", ["breakout_test", "cuped_test", "encouragement_test"])
def test_compiled_context_and_every_catalog_entry_carry_no_source_sql(experiment: str) -> None:
    context = _compile(_secret_payload(), experiment)
    entries = ac.unit_day_artifact_extension_catalog(context)
    assert entries
    serialized = context.canonical_json + "".join(entry.model_dump_json() for entry in entries)
    for sentinel in _SECRETS.values():
        assert sentinel not in serialized
    assert "SELECT" not in serialized


def test_source_mutations_change_provenance_without_emitting_content() -> None:
    base = _secret_payload()
    context = _compile(base, "breakout_test")
    base_entries = {
        entry.request.kind: entry for entry in ac.unit_day_artifact_extension_catalog(context)
    }
    for secret in ("fact_used", "fact_unused", "dim", "exposure"):
        changed = _secret_payload()
        for section in ("fact_sources", "dim_sources", "exposures"):
            for item in changed[section]:
                if "sql" in item:
                    item["sql"] = item["sql"].replace(_SECRETS[secret], "REPLACED")
        mutated = _compile(changed, "breakout_test")
        assert mutated.sha256 != context.sha256, secret
        assert ac.artifact_source_mapping(mutated) != ac.artifact_source_mapping(context), secret
        assert "REPLACED" not in mutated.canonical_json
    used = _secret_payload()
    used["fact_sources"][0]["sql"] += " WHERE 1 = 1"
    mutated_entries = {
        entry.request.kind: entry
        for entry in ac.unit_day_artifact_extension_catalog(_compile(used, "breakout_test"))
    }
    assert (
        mutated_entries["breakout_dimension"].source_provenance_sha256
        != base_entries["breakout_dimension"].source_provenance_sha256
    )
    assert (
        mutated_entries["breakout_dimension"].definition_sha256
        == base_entries["breakout_dimension"].definition_sha256
    )


@pytest.mark.parametrize(
    "mutation",
    [{"aggregation": "count"}, {"type": "conversion"}],
    ids=["aggregation", "type"],
)
def test_metric_declaration_mutation_changes_the_context_digest(mutation: dict[str, str]) -> None:
    context = _compile(_secret_payload(), "breakout_test")
    changed = _secret_payload()
    metric = changed["metrics"][0]
    if mutation.get("type") == "conversion":
        metric.pop("aggregation")
        metric.pop("window_days")
    metric.update(mutation)
    assert _compile(changed, "breakout_test").sha256 != context.sha256


def _reseal(context: Any, mutate: Any) -> Any:
    """A model-valid context whose payload was rewritten and re-hashed by an attacker."""
    from increment.query.artifact_digest import canonical_json
    from increment.semantics.artifact import ArtifactContext, _artifact_context_digest

    payload = json.loads(context.canonical_json)
    mutate(payload)
    raw = canonical_json(payload)
    return ArtifactContext(canonical_json=raw, sha256=_artifact_context_digest(raw))


def _resealed_entry(context: Any, edit: Any) -> Any:
    """A context whose first catalog entry was rewritten with consistent self-hashes."""

    def mutate(payload: dict[str, Any]) -> None:
        entry = payload["extension_catalog"][0]
        request = ac.unit_day_artifact_extension_catalog(context)[0].request
        recipe = edit(json.loads(entry["canonical_source_recipe_json"]))
        entry["canonical_source_recipe_json"] = ac.canonical_json(recipe)
        entry["source_provenance_sha256"] = ac._extension_hash(
            "extension-source", request, entry["canonical_source_recipe_json"]
        )

    return _reseal(context, mutate)


def test_a_source_recipe_carrying_sql_is_refused_even_when_self_consistent() -> None:
    context = _compile(_secret_payload(), "breakout_test")
    for edit in (
        lambda recipe: {**recipe, "sql": "SELECT secret"},
        lambda recipe: {"source": {"sql": "SELECT secret"}},
        lambda recipe: {**recipe, "source_recipe_format": 1},
    ):
        with pytest.raises(ac.ArtifactContractError) as refused:
            ac.unit_day_artifact_extension_catalog(_resealed_entry(context, edit))
        assert refused.value.code == "artifact.extension.invalid"


def test_source_mapping_is_a_validated_format_two_digest() -> None:
    context = _compile(_secret_payload(), "breakout_test")
    mapping = ac.artifact_source_mapping(context)
    assert set(mapping) == {"source_mapping_format", "recipe_sha256"}
    assert mapping["source_mapping_format"] == 2
    for bad in (
        {"source_mapping_format": 1, "recipe_sha256": mapping["recipe_sha256"]},
        {"source_mapping_format": 2, "recipe_sha256": "XYZ"},
        {"source_mapping_format": 2},
        {"definitions": {"sql": "SELECT secret"}},
    ):
        tampered = _reseal(context, lambda payload, bad=bad: payload.update(source_mapping=bad))
        with pytest.raises(ac.ArtifactContractError) as refused:
            ac.artifact_source_mapping(tampered)
        assert refused.value.code == "artifact.context.mismatch"


def test_context_format_one_is_refused_by_name_with_versions() -> None:
    context = _compile(_secret_payload(), "breakout_test")
    old = context.model_copy(update={"context_format": 1})
    with pytest.raises(ac.ArtifactContractError) as refused:
        ac.validate_artifact_context(old)
    assert refused.value.code == "artifact.format.unsupported"
    assert refused.value.context["received"] == 1
    assert refused.value.context["supported"] == 2


def test_inner_and_outer_context_format_must_agree_at_model_and_admission() -> None:
    from increment.errors import DefinitionError
    from increment.query.artifact_digest import canonical_json
    from increment.semantics.artifact import ArtifactContext, _artifact_context_digest

    context = _compile(_secret_payload(), "breakout_test")
    payload = json.loads(context.canonical_json)
    payload["context_format"] = 1
    raw = canonical_json(payload)
    with pytest.raises(DefinitionError) as at_model:
        ArtifactContext(canonical_json=raw, sha256=_artifact_context_digest(raw))
    assert at_model.value.code == "definition.artifact.context_format_disagrees"
    bypassed = context.model_copy(
        update={"canonical_json": raw, "sha256": _artifact_context_digest(raw)}
    )
    with pytest.raises(ac.ArtifactContractError) as at_admission:
        ac.validate_artifact_context(bypassed)
    assert at_admission.value.code == "artifact.context.mismatch"


@pytest.mark.parametrize("raw", ["[]", "null", "1", '"x"', "[2]"])
def test_a_non_object_context_is_a_coded_refusal_at_the_model(raw: str) -> None:
    import copy
    import pickle

    from increment.errors import DefinitionError
    from increment.semantics.artifact import ArtifactContext, _artifact_context_digest

    with pytest.raises(DefinitionError) as refused:
        ArtifactContext(canonical_json=raw, sha256=_artifact_context_digest(raw))
    for error in (
        refused.value,
        copy.deepcopy(refused.value),
        pickle.loads(pickle.dumps(refused.value)),
    ):
        assert error.code == "definition.encode_json_object"


def test_duplicate_json_keys_are_refused_at_model_and_admission() -> None:
    from increment.errors import DefinitionError
    from increment.semantics.artifact import ArtifactContext, _artifact_context_digest

    context = _compile(_secret_payload(), "breakout_test")
    raw = '{"context_format":2,"context_format":2,"experiment_name":"x"}'
    with pytest.raises(DefinitionError):
        ArtifactContext(canonical_json=raw, sha256=_artifact_context_digest(raw))
    bypassed = context.model_copy(
        update={"canonical_json": raw, "sha256": _artifact_context_digest(raw)}
    )
    with pytest.raises(ac.ArtifactContractError) as at_admission:
        ac.validate_artifact_context(bypassed)
    assert at_admission.value.code == "artifact.context.mismatch"


def test_equal_instants_in_different_offsets_compile_to_one_digest() -> None:
    utc = _secret_payload()
    offset = _secret_payload()
    for payload, spelling in ((utc, "2026-01-01T00:00:00Z"), (offset, "2025-12-31T19:00:00-05:00")):
        for experiment in payload["experiments"]:
            experiment["start"] = spelling
    for experiment in ("breakout_test", "encouragement_test"):
        assert _compile(utc, experiment).sha256 == _compile(offset, experiment).sha256


def test_naive_utc_declarations_compile_to_the_bytes_of_their_aware_spelling() -> None:
    from increment.query.artifact_publish import artifact_context

    naive = _secret_payload()
    aware = _secret_payload()
    for payload, spelling in ((naive, "2026-01-01T00:00:00"), (aware, "2026-01-01T00:00:00+00:00")):
        for experiment in payload["experiments"]:
            experiment["start"] = spelling
    naive_defs = Definitions.model_validate(naive)
    direct = ac.compile_unit_day_artifact_context("breakout_test", naive_defs)
    assert direct.sha256 == _compile(aware, "breakout_test").sha256
    experiment = naive_defs.experiment("breakout_test")
    assert experiment is not None
    assert artifact_context(naive_defs, experiment, "error") == direct


def test_context_metric_roster_must_match_the_experiment_exactly() -> None:
    from increment.query.source import _artifact_source_context

    context = _compile(_secret_payload(), "breakout_test")
    experiment, source_context = _artifact_source_context(context)
    assert [metric.name for metric in source_context.metrics] == experiment.metric_names
    for edit in (
        lambda payload: payload["definitions"].update(metrics=[]),
        lambda payload: payload["definitions"]["metrics"].append(
            payload["definitions"]["metrics"][0]
        ),
        lambda payload: payload["definitions"]["metrics"].append(
            {**payload["definitions"]["metrics"][0], "name": "extra"}
        ),
    ):
        with pytest.raises(ac.ArtifactContractError) as refused:
            _artifact_source_context(_reseal(context, edit))
        assert refused.value.code == "artifact.context.mismatch"


def test_effective_encouragement_design_survives_a_query_free_round_trip() -> None:
    from increment.query.source import _artifact_source_context
    from increment.semantics.design import Encouragement, UptakeSpec

    payload = _secret_payload()
    payload["experiments"][2]["design"] = None
    definitions = Definitions.model_validate(payload)
    effective = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked", window_days=3),
        one_sided=True,
        min_first_stage_z=0.001,
    )
    context = ac.compile_unit_day_artifact_context(
        "encouragement_test", definitions, encouragement_uptake=effective
    )
    from increment.semantics.models import EncouragementDeclaration

    experiment, source_context = _artifact_source_context(context)
    assert source_context.design == effective
    assert isinstance(experiment.design, EncouragementDeclaration)
    assert experiment.design.uptake == effective.uptake
    original = definitions.experiment("encouragement_test")
    assert original is not None
    assert original.design is None

    declared = Definitions.model_validate(_secret_payload())
    declared_experiment = declared.experiment("encouragement_test")
    assert declared_experiment is not None
    conflicting = effective.model_copy(update={"one_sided": False})
    with pytest.raises(ac.ArtifactContractError) as refused:
        ac.compile_unit_day_artifact_context(
            "encouragement_test", declared, encouragement_uptake=conflicting
        )
    assert refused.value.code == "artifact_contract.encouragement_uptake_conflict"
    _, declared_context = _artifact_source_context(
        ac.compile_unit_day_artifact_context("encouragement_test", declared)
    )
    assert declared_context.design == declared_experiment.resolved_design()


def test_undeclared_day_boundary_is_inherited_from_definitions_and_explicit_wins() -> None:
    from increment.query.source import _artifact_source_context

    inherited = _secret_payload()
    inherited["day_boundary"] = "UTC-05:00"
    context = _compile(inherited, "breakout_test")
    experiment, _ = _artifact_source_context(context)
    assert experiment.day_boundary == "UTC-05:00"
    assert "day_boundary" not in experiment.model_dump()

    explicit = _secret_payload()
    explicit["day_boundary"] = "UTC-05:00"
    explicit["experiments"][0]["day_boundary"] = "UTC+01:00"
    experiment, _ = _artifact_source_context(_compile(explicit, "breakout_test"))
    assert experiment.day_boundary == "UTC+01:00"
    assert experiment.model_dump()["day_boundary"] == "UTC+01:00"


# ── window days bound into the context identity ─────────────────────────────

_WINDOW_IDENTITIES = json.loads(
    (
        Path(__file__).parents[1] / "fixtures" / "unit_day_artifact_legacy_window_identities.json"
    ).read_text()
)["identities"]


def _window_definitions(start: str, end: str, day_boundary: str = "UTC-05:00") -> Any:
    payload = _secret_payload()
    payload["day_boundary"] = day_boundary
    for experiment in payload["experiments"]:
        experiment.update(start=start, end=end)
    return Definitions.model_validate(payload)


def _window_context(name: str, definitions: Any) -> Any:
    from increment.query.artifact_publish import artifact_context

    experiment = definitions.experiment(name)
    assert experiment is not None
    return artifact_context(definitions, experiment, "error")


def test_context_binds_the_declared_window_days_for_every_experiment() -> None:
    definitions = _window_definitions("2025-01-09T19:00:00-05:00", "2025-01-19T19:00:00-05:00")
    for name in ("breakout_test", "cuped_test", "encouragement_test"):
        stored = json.loads(_window_context(name, definitions).canonical_json)["window_days"]
        assert stored == {
            "start": "2025-01-09",
            "end": "2025-01-19",
            "observation_horizon": "2025-01-19",
        }
    open_ended = json.loads(_compile(_secret_payload(), "breakout_test").canonical_json)
    assert open_ended["window_days"] == {
        "start": "2025-08-01",
        "end": None,
        "observation_horizon": None,
    }


def test_admission_requires_stored_window_days_of_valid_shape() -> None:
    context = _compile(_secret_payload(), "breakout_test")
    ac.validate_artifact_context(context)
    missing = _reseal(context, lambda payload: payload.pop("window_days"))
    with pytest.raises(ac.ArtifactContractError) as refused:
        ac.validate_artifact_context(missing)
    assert refused.value.code == "artifact.context.mismatch"
    assert refused.value.context["missing"] == "window_days"
    assert refused.value.context["route"] == "republish from trusted definitions"
    for damaged in (
        {"start": "2025-08-01"},
        {"start": "not-a-date", "end": None, "observation_horizon": None},
        {"start": "2025-08-09", "end": "2025-08-01", "observation_horizon": "2025-08-01"},
        {"start": "2025-08-01", "end": None, "observation_horizon": None, "extra": None},
        [],
    ):
        tampered = _reseal(context, lambda payload, bad=damaged: payload.update(window_days=bad))
        with pytest.raises(ac.ArtifactContractError) as invalid:
            ac.validate_artifact_context(tampered)
        assert invalid.value.code == "artifact.context.mismatch"


def test_legacy_identity_collision_classes_split_by_window_after_the_change() -> None:
    """Legacy identities discarded the declaration's spelling; the derived window is bound now."""
    z = _WINDOW_IDENTITIES["aware_z"]
    offset = _WINDOW_IDENTITIES["aware_offset"]
    naive = _WINDOW_IDENTITIES["naive"]
    assert z["context_sha256"] == offset["context_sha256"] == naive["context_sha256"]
    assert (
        z["native_recipe_sha256"] == offset["native_recipe_sha256"] == naive["native_recipe_sha256"]
    )

    from increment.sequential_source import native_observation_mapping

    contexts, recipes = {}, {}
    for label, cell in _WINDOW_IDENTITIES.items():
        definitions = _window_definitions(cell["start"], cell["end"])
        contexts[label] = _window_context("breakout_test", definitions)
        experiment = definitions.experiment("breakout_test")
        recipes[label] = native_observation_mapping(definitions, experiment)
    assert contexts["aware_z"].sha256 == contexts["aware_offset"].sha256
    assert recipes["aware_z"] == recipes["aware_offset"]
    assert contexts["naive"].sha256 != contexts["aware_z"].sha256
    assert recipes["naive"] != recipes["aware_z"]
    for label, cell in _WINDOW_IDENTITIES.items():
        assert contexts[label].sha256 != cell["context_sha256"]
        assert recipes[label]["recipe_sha256"] != cell["native_recipe_sha256"]


def test_public_compile_reads_naive_declarations_as_wall_clock_at_the_day_boundary() -> None:
    from increment.query.artifact_publish import artifact_context
    from increment.query.source import _artifact_source_context

    definitions = _window_definitions("2025-01-15", "2025-01-20")
    direct = ac.compile_unit_day_artifact_context("breakout_test", definitions)
    stored = json.loads(direct.canonical_json)["window_days"]
    assert stored == {
        "start": "2025-01-15",
        "end": "2025-01-20",
        "observation_horizon": "2025-01-20",
    }
    experiment, _ = _artifact_source_context(direct)
    assert {
        "start": experiment.start_day.isoformat(),
        "end": None if experiment.end_day is None else experiment.end_day.isoformat(),
        "observation_horizon": (
            None
            if experiment.observation_horizon_day is None
            else experiment.observation_horizon_day.isoformat()
        ),
    } == stored
    declared = definitions.experiment("breakout_test")
    assert declared is not None
    published = json.loads(artifact_context(definitions, declared, "error").canonical_json)
    assert published["experiment"] == json.loads(direct.canonical_json)["experiment"]
    assert published["window_days"] == stored


def test_same_day_naive_declaration_stays_ordered_at_a_shifted_boundary() -> None:
    from increment.query.source import _artifact_source_context

    definitions = _window_definitions("2025-01-15T09:00:00", "2025-01-15T00:00:00")
    context = ac.compile_unit_day_artifact_context("breakout_test", definitions)
    stored = json.loads(context.canonical_json)["window_days"]
    assert stored == {
        "start": "2025-01-15",
        "end": "2025-01-15",
        "observation_horizon": "2025-01-15",
    }
    experiment, _ = _artifact_source_context(context)
    assert experiment.start_day.isoformat() == stored["start"]
    assert experiment.end_day is not None
    assert experiment.end_day.isoformat() == stored["end"]
    assert experiment.observation_horizon_day is not None
    assert experiment.observation_horizon_day.isoformat() == stored["observation_horizon"]


@pytest.mark.parametrize(
    "damaged",
    [
        {"start": "2025-08-01", "end": "2025-08-09", "observation_horizon": None},
        {"start": "2025-08-01", "end": None, "observation_horizon": "2025-08-09"},
        {"start": "2025-08-01", "end": "2025-08-09", "observation_horizon": "2025-08-05"},
        {"start": "2025-08-09", "end": None, "observation_horizon": None, "extra": 1},
    ],
)
def test_admission_refuses_inconsistent_window_day_relationships(damaged: dict[str, Any]) -> None:
    context = _compile(_secret_payload(), "breakout_test")
    tampered = _reseal(context, lambda payload: payload.update(window_days=damaged))
    with pytest.raises(ac.ArtifactContractError) as invalid:
        ac.validate_artifact_context(tampered)
    assert invalid.value.code == "artifact.context.mismatch"
    assert invalid.value.context["invalid"] == "window_days"


def test_admission_accepts_an_observation_horizon_past_the_end() -> None:
    context = _compile(_secret_payload(), "breakout_test")
    window = {"start": "2025-08-01", "end": "2025-08-09", "observation_horizon": "2025-08-20"}
    ac.validate_artifact_context(
        _reseal(context, lambda payload: payload.update(window_days=window))
    )


@pytest.mark.parametrize("code_first", [True, False], ids=["code-first", "message-first"])
def test_error_keeps_an_explicit_context_mapping_in_either_argument_order(code_first: bool) -> None:
    code, message = "artifact.snapshot.mixed", "stored identity differs"
    args = (code, message) if code_first else (message, code)
    error = ac.ArtifactContractError(*args, context={"artifact_id": "a1"})
    assert error.code == code
    assert dict(error.context) == {"artifact_id": "a1"}


def test_error_merges_keyword_context_with_an_explicit_mapping() -> None:
    error = ac.ArtifactContractError(
        "artifact.snapshot.mixed", "m", context={"artifact_id": "a1", "role": "explicit"}, role="kw"
    )
    # Both contributions survive; the explicit mapping wins a clash.
    assert dict(error.context) == {"artifact_id": "a1", "role": "explicit"}
    kw_only = ac.ArtifactContractError("artifact.snapshot.mixed", "m", generation_id="g1")
    assert dict(kw_only.context) == {"generation_id": "g1"}


def test_error_context_survives_pickle_and_deepcopy() -> None:
    import copy
    import pickle

    error = ac.ArtifactContractError("artifact.refresh.invalid_ref", "m", context={"name": "n1"})
    for clone in (pickle.loads(pickle.dumps(error)), copy.deepcopy(error)):
        assert clone.code == "artifact.refresh.invalid_ref"
        assert dict(clone.context) == {"name": "n1"}


def test_trigger_measure_stats_types_are_exported_from_package_root():
    import increment
    from increment import TriggerMeasureStatsExtension, TriggerMeasureStatsRequest
    from increment.semantics.artifact import TriggerMeasureStatsExtension as ArtifactExtension
    from increment.semantics.artifact import TriggerMeasureStatsRequest as ArtifactRequest

    assert TriggerMeasureStatsRequest is ArtifactRequest
    assert TriggerMeasureStatsExtension is ArtifactExtension
    assert {"TriggerMeasureStatsRequest", "TriggerMeasureStatsExtension"} <= set(increment.__all__)


def test_trigger_measure_request_is_closed_to_one_metric():
    from increment.semantics.models import TriggerMeasureStatsRequest

    request = TriggerMeasureStatsRequest.model_validate(
        {
            "kind": "trigger_measure_stats",
            "trigger_name": "checkout",
            "metric_names": ["revenue"],
        }
    )
    assert request.metric_names == ("revenue",)
    with pytest.raises(InvalidRequestError):
        TriggerMeasureStatsRequest.model_validate(
            {
                "kind": "trigger_measure_stats",
                "trigger_name": "checkout",
                "metric_names": ["revenue", "orders"],
            }
        )


@pytest.mark.parametrize("legacy_version", [1, 2])
def test_legacy_trigger_anchor_versions_are_rejected_and_require_republish(legacy_version):
    from uuid import UUID

    from increment.semantics.artifact import TriggerPopulationExtension

    relation = {
        "artifact_id": UUID("00000000-0000-0000-0000-000000000001"),
        "generation_id": UUID("00000000-0000-0000-0000-000000000002"),
        "role": "trigger_population",
        "relation": {"name": "trigger_population"},
        "schema_sha256": "0" * 64,
        "content_sha256": "1" * 64,
        "row_count": 0,
        "primary_key": ["experiment_id", "unit_id"],
    }
    payload = {
        "extension_version": legacy_version,
        "relation": relation,
        "definition_sha256": "2" * 64,
        "source_provenance_sha256": "3" * 64,
        "trigger_name": "checkout",
        "observation_cutoff_ts": "2025-01-20T12:00:00Z",
        "complete_through_ts": None,
    }
    with pytest.raises(ValidationError, match="extension_version"):
        TriggerPopulationExtension.model_validate(payload)
