"""Tests for the YAML loader — file I/O, directory merging, duplicate
detection with file origins, SQL validation, and example integration."""

import copy
import pickle
import warnings
from pathlib import Path

import pytest
import sqlglot
import yaml

from increment.errors import DefinitionError, IncrementWarning, InvalidRequestError
from increment.semantics import load
from increment.semantics.loader import admit_read_only_sql
from increment.semantics.models import Definitions, RatioMetric
from tests.warning_codes import warning_codes


def _dump_yaml(data: object, path: Path) -> None:
    path.write_text(yaml.dump(data))


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def tmp_defs_dir(tmp_path: Path) -> Path:
    """Return a temporary directory with a minimal valid definitions file."""
    data = {
        "dialect": "duckdb",
        "fact_sources": [
            {
                "name": "events",
                "sql": "SELECT * FROM events",
                "timestamp_column": "ts",
                "entities": ["user_id"],
                "facts": [{"name": "page_view", "column": None}],
            }
        ],
        "exposures": [{"name": "enrolled", "fact": "page_view"}],
        "metrics": [
            {
                "type": "conversion",
                "name": "visit_rate",
                "entity": "user_id",
                "fact": "page_view",
                "window_days": 7,
            }
        ],
        "experiments": [
            {
                "name": "exp_1",
                "plan": {},
                "exposure": "enrolled",
                "unit": "user_id",
                "start": "2024-06-01",
                "control_group": "C",
            }
        ],
    }
    f = tmp_path / "definitions.yaml"
    with open(f, "w") as fh:
        yaml.dump(data, fh)
    return tmp_path


# ── Single file round-trip ─────────────────────────────────────────────


def test_load_minimal_yaml(tmp_defs_dir: Path):
    yaml_path = tmp_defs_dir / "definitions.yaml"
    defs = load(yaml_path)
    assert isinstance(defs, Definitions)
    assert defs.dialect == "duckdb"
    assert len(defs.fact_sources) == 1
    assert len(defs.exposures) == 1
    assert len(defs.metrics) == 1
    assert len(defs.experiments) == 1


def test_load_missing_path_raises():
    with pytest.raises(DefinitionError) as exc_info:
        load("/no/such/path.yaml")
    assert exc_info.value.code == "definition.path.missing"


def test_load_invalid_yaml_raises(tmp_path: Path):
    f = tmp_path / "bad.yaml"
    f.write_text("{unbalanced: [}")
    with pytest.raises(DefinitionError) as exc_info:
        load(f)
    assert exc_info.value.code == "definition.yaml"


@pytest.mark.parametrize("section", ["metrics", "fact_sources"])
def test_merge_section_must_be_a_list_for_file_and_directory(tmp_path: Path, section: str):
    bad_yaml = f"{section}:\n  name: broken\n"

    single = tmp_path / "bad.yaml"
    single.write_text(bad_yaml)
    with pytest.raises(DefinitionError) as exc_info:
        load(single)
    assert exc_info.value.code == "definition.section"

    definitions = tmp_path / "defs"
    definitions.mkdir()
    (definitions / "bad.yaml").write_text(bad_yaml)
    with pytest.raises(DefinitionError) as exc_info:
        load(definitions)
    assert exc_info.value.code == "definition.section"


def test_duplicate_scalar_key_in_single_file_rejected(tmp_path: Path):
    """PyYAML's default constructor keeps the LAST of a duplicate key with
    no warning; a duplicate-detecting loader must refuse instead."""
    f = tmp_path / "defs.yaml"
    f.write_text("dialect: duckdb\ndialect: postgres\n")
    with pytest.raises(DefinitionError) as exc_info:
        load(f)
    assert exc_info.value.code == "definition.yaml"
    assert (exc_info.value.context["line"], exc_info.value.context["column"]) == (2, 1)


def test_duplicate_scalar_key_across_files_names_both_files(tmp_path: Path):
    (tmp_path / "a.yaml").write_text("dialect: duckdb\n")
    (tmp_path / "b.yaml").write_text("dialect: duckdb\n")
    with pytest.raises(DefinitionError) as exc_info:
        load(tmp_path)
    assert exc_info.value.code == "definition.duplicate_scalar"
    assert exc_info.value.context["field"] == "dialect"
    assert exc_info.value.context["files"] == (
        str(tmp_path / "a.yaml"),
        str(tmp_path / "b.yaml"),
    )


def test_duplicate_key_within_a_list_item_rejected(tmp_path: Path):
    f = tmp_path / "defs.yaml"
    f.write_text(
        "fact_sources:\n"
        "- name: events\n"
        "  sql: SELECT 1\n"
        "  sql: SELECT 2\n"
        "  timestamp_column: ts\n"
        "  entities: [user_id]\n"
        "  facts: []\n"
    )
    with pytest.raises(DefinitionError) as exc_info:
        load(f)
    assert exc_info.value.code == "definition.yaml"


@pytest.mark.parametrize("section", ["fact_sources", "metrics"])
def test_non_mapping_list_element_rejected(tmp_path: Path, section: str):
    """A list element that is not a mapping (e.g. a bare string) must
    refuse with file/section/index context, not crash with a raw
    AttributeError from `.get()` deep inside the loader."""
    f = tmp_path / "defs.yaml"
    f.write_text(f"{section}:\n- oops\n")
    with pytest.raises(DefinitionError) as exc_info:
        load(f)
    assert exc_info.value.path == str(f)
    assert exc_info.value.code == "definition.section"


@pytest.mark.parametrize(
    ("yaml_body", "field", "invalid_input"),
    [
        (
            "fact_sources:\n- name: source\n  sql: SELECT 1\n  facts: [oops]\n",
            "fact_sources.0.facts.0",
            "oops",
        ),
        (
            "fact_sources:\n- name: source\n  sql: SELECT 1\n  facts: null\n",
            "fact_sources.0.facts",
            None,
        ),
        ("metrics:\n- name: [x]\n", "metrics.0", {"name": ["x"]}),
        (
            "fact_sources: [{name: s, sql: 'SELECT 1', properties: null}]",
            "fact_sources.0.properties",
            None,
        ),
        (
            "fact_sources: [{name: s, sql: 'SELECT 1', properties: [{name: [x]}]}]",
            "fact_sources.0.properties.0.name",
            ["x"],
        ),
    ],
)
def test_malformed_nested_yaml_raises_coded_field_errors(
    tmp_path: Path, yaml_body: str, field: str, invalid_input: object
):
    """Malformed nested values reach Pydantic instead of duplicate prechecks."""
    from ast import literal_eval
    from collections.abc import Mapping
    from typing import cast

    definition_path = tmp_path / "defs.yaml"
    definition_path.write_text(yaml_body)

    with pytest.raises(DefinitionError) as raised:
        load(definition_path)

    error = raised.value
    assert error.code == "definition.validation"
    assert error.context["path"] == str(definition_path)
    validation_errors = cast(tuple[Mapping[str, object], ...], error.context["errors"])
    matching = next(item for item in validation_errors if item["field"] == field)
    observed_input = matching["input"]
    if isinstance(observed_input, str) and not isinstance(invalid_input, str):
        observed_input = literal_eval(observed_input)
    assert observed_input == invalid_input
    with pytest.raises(TypeError):
        matching["field"] = "changed"  # ty: ignore[invalid-assignment]
    assert copy.deepcopy(error).context == error.context
    assert pickle.loads(pickle.dumps(error)).context == error.context


def test_sql_field_wrong_type_rejected(tmp_path: Path):
    f = tmp_path / "defs.yaml"
    f.write_text(
        "fact_sources:\n"
        "- name: events\n"
        "  sql: 42\n"
        "  timestamp_column: ts\n"
        "  entities: [user_id]\n"
        "  facts: []\n"
    )
    with pytest.raises(DefinitionError) as exc_info:
        load(f)
    assert exc_info.value.path == str(f)
    assert exc_info.value.code == "definition.section"


def test_field_constraint_violation_surfaces_as_a_coded_definition_error(tmp_path: Path):
    f = tmp_path / "defs.yaml"
    f.write_text(
        "fact_sources:\n"
        "- name: events\n"
        "  sql: SELECT 1\n"
        "  timestamp_column: ts\n"
        "  entities: []\n"
        "  facts: []\n"
    )
    with pytest.raises(DefinitionError) as raised:
        load(f)
    assert raised.value.code == "definition.validation"
    (error,) = raised.value.context["errors"]  # ty: ignore[not-iterable]
    assert (error["field"], error["type"]) == ("fact_sources.0.entities", "too_short")
    restored = pickle.loads(pickle.dumps(raised.value))
    assert restored.context == raised.value.context


def test_recursive_yaml_anchor_rejected_not_infinite_loop(tmp_path: Path):
    """A self-referencing anchor list must refuse cleanly (caught by the
    non-mapping-element shape guard) instead of crashing or hanging."""
    f = tmp_path / "defs.yaml"
    f.write_text("fact_sources: &x\n- *x\n")
    with pytest.raises(DefinitionError) as exc_info:
        load(f)
    assert exc_info.value.code == "definition.section"


# ── Directory merging ──────────────────────────────────────────────────


def test_load_directory_merges(tmp_path: Path):
    """Files in a directory are merged by top-level list keys."""
    src_file = tmp_path / "fact_sources.yaml"
    _dump_yaml(
        {
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT 1",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [{"name": "ev", "column": None}],
                }
            ]
        },
        src_file,
    )

    exp_file = tmp_path / "experiments.yaml"
    _dump_yaml(
        {
            "exposures": [{"name": "enrolled", "fact": "ev"}],
            "metrics": [
                {
                    "type": "conversion",
                    "name": "rate",
                    "entity": "user_id",
                    "fact": "ev",
                    "window_days": 7,
                }
            ],
            "experiments": [
                {
                    "name": "e1",
                    "plan": {},
                    "exposure": "enrolled",
                    "unit": "user_id",
                    "start": "2024-01-01",
                    "control_group": "C",
                }
            ],
        },
        exp_file,
    )

    defs = load(tmp_path)
    assert len(defs.fact_sources) == 1
    assert len(defs.exposures) == 1
    assert len(defs.metrics) == 1
    assert len(defs.experiments) == 1


def test_load_directory_ignores_non_yaml(tmp_path: Path):
    """Non-.yaml files are ignored during directory load."""
    (tmp_path / "definitions.yaml").write_text(
        yaml.dump(
            {
                "fact_sources": [
                    {
                        "name": "s",
                        "sql": "SELECT 1",
                        "timestamp_column": "ts",
                        "entities": ["u"],
                        "facts": [{"name": "f", "column": None}],
                    }
                ],
                "exposures": [{"name": "ex", "fact": "f"}],
                "metrics": [
                    {
                        "type": "conversion",
                        "name": "m",
                        "entity": "u",
                        "fact": "f",
                        "window_days": 7,
                    }
                ],
                "experiments": [
                    {
                        "name": "e",
                        "plan": {},
                        "exposure": "ex",
                        "unit": "u",
                        "start": "2024-01-01",
                        "control_group": "C",
                    }
                ],
            }
        )
    )
    (tmp_path / "notes.txt").write_text("not yaml")
    defs = load(tmp_path)
    assert len(defs.fact_sources) == 1


def test_load_directory_reads_yml_extension(tmp_path: Path):
    """`.yml` is standard YAML and must be read: a definitions split across `sources.yml` +
    `experiments.yaml` must not silently drop the `.yml` file with no error."""
    (tmp_path / "sources.yml").write_text(
        yaml.dump(
            {
                "fact_sources": [
                    {
                        "name": "s",
                        "sql": "SELECT 1",
                        "timestamp_column": "ts",
                        "entities": ["u"],
                        "facts": [{"name": "f", "column": None}],
                    }
                ]
            }
        )
    )
    (tmp_path / "experiments.yaml").write_text(
        yaml.dump(
            {
                "exposures": [{"name": "ex", "fact": "f"}],
                "metrics": [
                    {
                        "type": "conversion",
                        "name": "m",
                        "entity": "u",
                        "fact": "f",
                        "window_days": 7,
                    }
                ],
                "experiments": [
                    {
                        "name": "e",
                        "plan": {},
                        "exposure": "ex",
                        "unit": "u",
                        "start": "2024-01-01",
                        "control_group": "C",
                    }
                ],
            }
        )
    )
    defs = load(tmp_path)
    assert len(defs.fact_sources) == 1  # came from the .yml file
    assert len(defs.experiments) == 1


# ── Duplicate detection with file names ────────────────────────────────


def test_duplicate_fact_source_name_across_files(tmp_path: Path):
    _dump_yaml(
        {
            "fact_sources": [
                {
                    "name": "dup",
                    "sql": "SELECT 1",
                    "timestamp_column": "ts",
                    "entities": ["u"],
                    "facts": [{"name": "f1", "column": None}],
                }
            ],
            "exposures": [],
            "metrics": [],
            "experiments": [],
        },
        (tmp_path / "a.yaml"),
    )
    _dump_yaml(
        {
            "fact_sources": [
                {
                    "name": "dup",
                    "sql": "SELECT 2",
                    "timestamp_column": "ts",
                    "entities": ["u"],
                    "facts": [{"name": "f2", "column": None}],
                }
            ],
            "exposures": [],
            "metrics": [],
            "experiments": [],
        },
        (tmp_path / "b.yaml"),
    )

    with pytest.raises(DefinitionError) as raised:
        load(tmp_path)

    error = raised.value
    assert error.code == "definition.duplicates"
    assert error.path == str(tmp_path)
    assert error.sources["fact_source:dup"] == (
        str(tmp_path / "a.yaml"),
        str(tmp_path / "b.yaml"),
    )
    assert error.context["sources"] == error.sources


def test_duplicate_fact_name_across_sources(tmp_path: Path):
    """Same fact name in two fact sources (across files) is an error."""
    _dump_yaml(
        {
            "fact_sources": [
                {
                    "name": "src_a",
                    "sql": "SELECT 1",
                    "timestamp_column": "ts",
                    "entities": ["u"],
                    "facts": [{"name": "conflict", "column": None}],
                }
            ],
            "exposures": [],
            "metrics": [],
            "experiments": [],
        },
        (tmp_path / "a.yaml"),
    )
    _dump_yaml(
        {
            "fact_sources": [
                {
                    "name": "src_b",
                    "sql": "SELECT 2",
                    "timestamp_column": "ts",
                    "entities": ["u"],
                    "facts": [{"name": "conflict", "column": None}],
                }
            ],
            "exposures": [],
            "metrics": [],
            "experiments": [],
        },
        (tmp_path / "b.yaml"),
    )

    with pytest.raises(DefinitionError) as exc_info:
        load(tmp_path)
    assert exc_info.value.code == "definition.duplicates"


# ── Reference validation through the loader ────────────────────────────


def test_metric_entity_mismatch_via_loader(tmp_path: Path):
    _dump_yaml(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "src",
                    "sql": "SELECT 1",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [{"name": "ev", "column": None}],
                }
            ],
            "exposures": [{"name": "e", "fact": "ev"}],
            "metrics": [{"type": "mean", "name": "m", "entity": "company_id", "fact": "ev"}],
            "experiments": [
                {
                    "name": "exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2024-01-01",
                    "control_group": "C",
                    "plan": {"secondaries": ["m"]},
                }
            ],
        },
        (tmp_path / "defs.yaml"),
    )

    with pytest.raises(DefinitionError) as exc_info:
        load(tmp_path / "defs.yaml")
    assert exc_info.value.code == "definition.invalid"


def test_ratio_shorthand_with_n_pre_periods_loads_via_loader(tmp_path: Path):
    """A ratio metric without a CUPED-requesting binding coexists with n_pre_periods>0; only a
    binding that actually requests CUPED is refused (see test below)."""
    _dump_yaml(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "src",
                    "sql": "SELECT 1",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [{"name": "rev", "column": "a"}, {"name": "ord", "column": "b"}],
                }
            ],
            "exposures": [{"name": "e", "sql": "SELECT 1"}],
            "metrics": [
                {
                    "type": "ratio",
                    "name": "r",
                    "entity": "user_id",
                    "numerator": {"fact": "rev", "aggregation": "sum"},
                    "denominator": {"fact": "ord", "aggregation": "count"},
                }
            ],
            "experiments": [
                {
                    "name": "exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2024-01-01",
                    "control_group": "C",
                    "plan": {"secondaries": ["r"]},
                    "n_pre_periods": 7,
                }
            ],
        },
        (tmp_path / "defs.yaml"),
    )

    defs = load(tmp_path / "defs.yaml")
    assert defs.experiments[0].metric_names == ["r"]


def test_ratio_cuped_binding_loads_via_loader(tmp_path: Path):
    """The YAML binding form (metric:/decision_method:/prior:) round-trips
    through the loader with its CUPED declaration on a ratio metric."""
    _dump_yaml(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "src",
                    "sql": "SELECT 1",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [{"name": "rev", "column": "a"}, {"name": "ord", "column": "b"}],
                }
            ],
            "exposures": [{"name": "e", "sql": "SELECT 1"}],
            "metrics": [
                {
                    "type": "ratio",
                    "name": "r",
                    "entity": "user_id",
                    "numerator": {"fact": "rev", "aggregation": "sum"},
                    "denominator": {"fact": "ord", "aggregation": "count"},
                }
            ],
            "experiments": [
                {
                    "name": "exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2024-01-01",
                    "control_group": "C",
                    "plan": {
                        "secondaries": [
                            {
                                "metric": "r",
                                "decision_method": {"name": "cuped", "variance_reduction": "cuped"},
                            }
                        ]
                    },
                    "n_pre_periods": 7,
                }
            ],
        },
        (tmp_path / "defs.yaml"),
    )

    defs = load(tmp_path / "defs.yaml")
    experiment = defs.experiment("exp")
    assert experiment is not None
    assert experiment.bindings["r"].wants_cuped


def test_experiment_metric_binding_loads_via_loader(tmp_path: Path):
    """The full binding form (decision method, sensitivity method AND prior)
    round-trips through the loader and the resolved Experiment carries
    wants_cuped."""
    _dump_yaml(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "src",
                    "sql": "SELECT 1",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [{"name": "rev", "column": "a"}, {"name": "ord", "column": "b"}],
                }
            ],
            "exposures": [{"name": "e", "sql": "SELECT 1"}],
            "metrics": [
                {
                    "type": "mean",
                    "name": "orders",
                    "entity": "user_id",
                    "fact": "ord",
                    "aggregation": "sum",
                    "window_days": 7,
                }
            ],
            "experiments": [
                {
                    "name": "exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2024-01-01",
                    "control_group": "C",
                    "n_pre_periods": 14,
                    "plan": {
                        "secondaries": [
                            {
                                "metric": "orders",
                                "decision_method": {"name": "unadjusted"},
                                "sensitivity_methods": [
                                    {"name": "cuped", "variance_reduction": "cuped"}
                                ],
                                "prior": {"mu": 0.0, "sigma": 0.03},
                            }
                        ]
                    },
                }
            ],
        },
        (tmp_path / "defs.yaml"),
    )

    defs = load(tmp_path / "defs.yaml")
    binding = defs.experiments[0].bindings["orders"]
    assert binding.wants_cuped
    assert binding.prior is not None
    assert binding.prior.sigma == 0.03


def test_experiment_with_quantile_metric_loads(tmp_path: Path):
    """A quantile metric on a warehouse experiment is estimable from the
    retained per-unit rows, so it loads like any other metric."""
    _dump_yaml(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "src",
                    "sql": "SELECT 1",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [{"name": "lat", "column": "a"}],
                }
            ],
            "exposures": [{"name": "e", "sql": "SELECT 1"}],
            "metrics": [
                {
                    "type": "quantile",
                    "name": "p90_lat",
                    "entity": "user_id",
                    "fact": "lat",
                    "quantile": 0.9,
                }
            ],
            "experiments": [
                {
                    "name": "exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2024-01-01",
                    "control_group": "C",
                    "plan": {"secondaries": ["p90_lat"]},
                }
            ],
        },
        (tmp_path / "defs.yaml"),
    )

    defs = load(tmp_path / "defs.yaml")
    metric = defs.metric("p90_lat")
    assert metric is not None and metric.type == "quantile"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "WITH rows AS (SELECT 1 AS value) SELECT value FROM rows",
        "SELECT 1 UNION ALL SELECT 2",
    ],
)
def test_admit_read_only_sql_accepts_queries(sql: str):
    admit_read_only_sql(sql, dialect="duckdb", label="test query")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM source WITH (NOLOCK)",
        "SELECT * FROM source WITH (INDEX(ix_source))",
        "SELECT * FROM source WITH (READPAST)",
    ],
)
def test_admit_read_only_sql_accepts_benign_table_hints(sql: str):
    admit_read_only_sql(sql, dialect="tsql", label="test query")


@pytest.mark.parametrize(
    ("sql", "dialect", "code"),
    [
        ("SELECT 1; DROP TABLE victim", "duckdb", "definition.sql.statement_count"),
        ("DELETE FROM victim", "duckdb", "definition.sql.not_read_only"),
        ("INSERT INTO victim VALUES (1)", "duckdb", "definition.sql.not_read_only"),
        ("UPDATE victim SET value = 1", "duckdb", "definition.sql.not_read_only"),
        (
            "WITH x AS (SELECT 1 AS value) INSERT INTO victim SELECT value FROM x",
            "duckdb",
            "definition.sql.not_read_only",
        ),
        (
            "WITH x AS (SELECT 1 AS value) UPDATE victim SET value = x.value FROM x",
            "duckdb",
            "definition.sql.not_read_only",
        ),
        (
            "WITH x AS (SELECT 1 AS value) DELETE FROM victim USING x",
            "duckdb",
            "definition.sql.not_read_only",
        ),
        ("CREATE TABLE victim (value INTEGER)", "duckdb", "definition.sql.not_read_only"),
        ("ALTER TABLE victim ADD COLUMN value INTEGER", "duckdb", "definition.sql.not_read_only"),
        ("DROP TABLE victim", "duckdb", "definition.sql.not_read_only"),
        ("SELECT * INTO victim FROM source", "postgres", "definition.sql.not_read_only"),
        ("SELECT * FROM source FOR UPDATE", "postgres", "definition.sql.not_read_only"),
        ("SELECT * FROM source FOR SHARE", "postgres", "definition.sql.not_read_only"),
        ("SELECT * FROM source LOCK IN SHARE MODE", "mysql", "definition.sql.not_read_only"),
        ("SELECT * FROM source WITH (UPDLOCK)", "tsql", "definition.sql.not_read_only"),
        ("SELECT * FROM source WITH (SERIALIZABLE)", "tsql", "definition.sql.not_read_only"),
        ("SELECT * FROM source WITH (REPEATABLEREAD)", "tsql", "definition.sql.not_read_only"),
    ],
)
def test_admit_read_only_sql_rejects_side_effects(sql: str, dialect: str, code: str):
    with pytest.raises(InvalidRequestError) as exc_info:
        admit_read_only_sql(sql, dialect=dialect, label="test query")
    assert exc_info.value.code == code


@pytest.mark.parametrize(
    ("sql", "dialect"),
    [
        ("SELECT pg_advisory_lock(1)", "postgres"),
        ("SELECT pg_sleep(1)", "postgres"),
        ("SELECT lo_export(1, 'out.csv')", "postgres"),
        ("SELECT lo_put(1, 0, 'bytes')", "postgres"),
        ("SELECT lo_from_bytea(0, 'bytes')", "postgres"),
        ("SELECT get_lock('mylock', 5)", "mysql"),
        ("SELECT count(*) FROM t WHERE pg_terminate_backend(pid) IS NOT NULL", "postgres"),
        ("SELECT write_csv(t, 'out.csv') FROM t", "duckdb"),
        ("SELECT pg_promote()", "postgres"),
        ("SELECT pg_rotate_logfile()", "postgres"),
        ("SELECT dblink_disconnect('name')", "postgres"),
        ("SELECT pg_notify('channel', 'payload')", "postgres"),
        (
            "SELECT * FROM dblink('dbname=other', 'DELETE FROM victim RETURNING id') AS r(id int)",
            "postgres",
        ),
        ("SELECT pg_sleep_for('1 second')", "postgres"),
        ("SELECT pg_sleep_until(now())", "postgres"),
        ("SELECT SYSTEM$CANCEL_QUERY('query-id')", "snowflake"),
        ("SELECT SYSTEM$ABORT_SESSION(123)", "snowflake"),
        ("SELECT SYSTEM$WAIT(1)", "snowflake"),
        ("SELECT MASTER_POS_WAIT('binlog.000001', 4)", "mysql"),
        ("SELECT WAIT_FOR_EXECUTED_GTID_SET('uuid:1')", "mysql"),
    ],
)
def test_admit_read_only_sql_rejects_side_effecting_function_calls(sql: str, dialect: str):
    """Reject side-effecting function calls even within a plain SELECT."""
    with pytest.raises(InvalidRequestError) as exc_info:
        admit_read_only_sql(sql, dialect=dialect, label="test query")
    assert exc_info.value.code == "definition.sql.not_read_only"


@pytest.mark.parametrize(
    ("sql", "dialect", "violation", "construct"),
    [
        ("DELETE FROM victim", "duckdb", "non-query statement", "Delete"),
        ("SELECT * FROM source FOR UPDATE", "postgres", "write or lock clause", "Lock"),
        ("SELECT * FROM source WITH (UPDLOCK)", "tsql", "locking table hint", "UPDLOCK"),
        ("SELECT pg_sleep(1)", "postgres", "side-effecting function", "pg_sleep"),
    ],
)
def test_admit_read_only_sql_names_the_rejected_construct(
    sql: str, dialect: str, violation: str, construct: str
):
    with pytest.raises(InvalidRequestError) as exc_info:
        admit_read_only_sql(sql, dialect=dialect, label="test query")
    context = exc_info.value.context
    assert (context["violation"], context["construct"], context["dialect"]) == (
        violation,
        construct,
        dialect,
    )


def test_admit_read_only_sql_counts_the_parsed_statements():
    with pytest.raises(InvalidRequestError) as exc_info:
        admit_read_only_sql("SELECT 1; SELECT 2", dialect="duckdb", label="test query")
    assert (exc_info.value.context["statements"], exc_info.value.context["empty"]) == (2, 0)


@pytest.mark.parametrize(
    ("function", "argument"),
    [("lowrite", "decode('DEADBEEF', 'hex')"), ("lo_truncate", "0"), ("lo_truncate64", "0")],
)
@pytest.mark.parametrize(
    "query",
    [
        "SELECT {function}(1, {argument})",
        "SELECT {function}(lo_open(12345, 131072), {argument})",
        "SELECT pg_catalog.{function}(lo_open(12345, 131072), {argument})",
        "WITH result AS (SELECT coalesce(pg_catalog.{function}(1, {argument}), 0) AS n) "
        "SELECT n FROM result",
    ],
)
def test_admit_read_only_sql_rejects_large_object_descriptor_writes(
    function: str, argument: str, query: str
):
    with pytest.raises(InvalidRequestError) as exc_info:
        admit_read_only_sql(
            query.format(function=function.upper(), argument=argument),
            dialect="postgres",
            label="test query",
        )
    assert exc_info.value.code == "definition.sql.not_read_only"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT pg_catalog.lo_get(12345, 0, 4)",
        "SELECT 'lowrite(1, x)' AS lo_truncate, 0 AS lo_truncate64",
    ],
)
def test_admit_read_only_sql_allows_large_object_reads_and_write_names_as_data(sql: str):
    admit_read_only_sql(sql, dialect="postgres", label="test query")


def test_admit_read_only_sql_allows_ordinary_function_calls():
    admit_read_only_sql(
        "SELECT count(*), sum(x), coalesce(y, 0) FROM t", dialect="duckdb", label="test query"
    )


def test_admit_read_only_sql_rejects_unknown_dialect():
    with pytest.raises(InvalidRequestError) as exc_info:
        admit_read_only_sql("SELECT 1", dialect="not_a_real_dialect", label="test query")
    assert exc_info.value.code == "definition.sql.dialect"


@pytest.mark.parametrize(
    ("sql", "dialect"),
    [
        ("SELECT 1; DROP TABLE victim", "duckdb"),
        ("DELETE FROM victim", "duckdb"),
        ("INSERT INTO victim VALUES (1)", "duckdb"),
        ("UPDATE victim SET value = 1", "duckdb"),
        ("CREATE TABLE victim (value INTEGER)", "duckdb"),
        ("ALTER TABLE victim ADD COLUMN value INTEGER", "duckdb"),
        ("DROP TABLE victim", "duckdb"),
        ("SELECT * INTO victim FROM source", "postgres"),
        ("SELECT * FROM source FOR UPDATE", "postgres"),
        ("SELECT * FROM source WITH (UPDLOCK)", "tsql"),
        ("SELECT * FROM source WITH (SERIALIZABLE)", "tsql"),
        ("SELECT * FROM source WITH (REPEATABLEREAD)", "tsql"),
    ],
)
def test_loader_rejects_side_effects_with_definition_path(tmp_path: Path, sql: str, dialect: str):
    definitions = tmp_path / "definitions.yaml"
    _dump_yaml(
        {
            "dialect": dialect,
            "fact_sources": [{"name": "unsafe", "sql": sql}],
        },
        definitions,
    )

    with pytest.raises(DefinitionError) as exc_info:
        load(definitions)
    assert exc_info.value.path == str(definitions)
    assert exc_info.value.code == (
        "definition.sql.statement_count" if ";" in sql else "definition.sql.not_read_only"
    )


def test_from_definitions_reports_specific_sql_failure_codes(tmp_path: Path):
    """Different SQL admission hazards retain their distinct public codes."""
    cases = {
        "definition.sql.parse": "select select select",
        "definition.sql.statement_count": "select 1; select 2;",
        "definition.sql.not_read_only": "insert into t values (1)",
    }
    for expected_code, sql in cases.items():
        yaml_path = tmp_path / f"{expected_code.replace('.', '_')}.yaml"
        yaml_path.write_text(
            "dialect: duckdb\n"
            "fact_sources:\n"
            "  - name: f\n"
            f'    sql: "{sql}"\n'
            "    entity: user_id\n"
            "    ts: ts\n"
        )
        with pytest.raises(DefinitionError) as raised:
            load(yaml_path)
        assert raised.value.code == expected_code, f"sql={sql!r}"


# ── SQL validation ─────────────────────────────────────────────────────


def test_invalid_sql_raises(tmp_path: Path):
    _dump_yaml(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "bad_src",
                    "sql": "SEL ECT * FORM events",  # intentional typos
                    "timestamp_column": "ts",
                    "entities": ["u"],
                    "facts": [{"name": "f", "column": None}],
                }
            ],
            "exposures": [{"name": "e", "fact": "f"}],
            "metrics": [{"type": "conversion", "name": "m", "entity": "u", "fact": "f"}],
            "experiments": [
                {
                    "name": "x",
                    "plan": {},
                    "exposure": "e",
                    "unit": "u",
                    "start": "2024-01-01",
                    "control_group": "C",
                }
            ],
        },
        (tmp_path / "defs.yaml"),
    )

    with pytest.raises(DefinitionError) as exc_info:
        load(tmp_path / "defs.yaml")
    assert exc_info.value.code == "definition.sql.parse"


def test_invalid_exposure_sql_raises(tmp_path: Path):
    """Exposure SQL is user-supplied like fact-source SQL: with a declared dialect, a parse
    failure must be a hard error at load, not a backend error far from the definition."""
    _dump_yaml(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM events",
                    "timestamp_column": "ts",
                    "entities": ["u"],
                    "facts": [{"name": "f", "column": None}],
                }
            ],
            "exposures": [{"name": "exp1", "sql": "TOTALLY ((( NOT SQL @@@"}],
            "metrics": [{"type": "conversion", "name": "m", "entity": "u", "fact": "f"}],
            "experiments": [
                {
                    "name": "x",
                    "plan": {},
                    "exposure": "exp1",
                    "unit": "u",
                    "start": "2024-01-01",
                    "control_group": "C",
                }
            ],
        },
        (tmp_path / "defs.yaml"),
    )

    with pytest.raises(DefinitionError) as exc_info:
        load(tmp_path / "defs.yaml")
    assert exc_info.value.code == "definition.sql.parse"


def test_invalid_exposure_sql_without_dialect_raises(tmp_path: Path):
    """An invalid exposure query is rejected even when no dialect is declared."""
    _dump_yaml(
        {
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM events",
                    "timestamp_column": "ts",
                    "entities": ["u"],
                    "facts": [{"name": "f", "column": None}],
                }
            ],
            "exposures": [{"name": "exp1", "sql": "SEL ECT * FORM events"}],
            "metrics": [
                {"type": "conversion", "name": "m", "entity": "u", "fact": "f", "window_days": 7}
            ],
            "experiments": [
                {
                    "name": "x",
                    "plan": {},
                    "exposure": "exp1",
                    "unit": "u",
                    "start": "2024-01-01",
                    "control_group": "C",
                }
            ],
        },
        (tmp_path / "defs.yaml"),
    )

    with pytest.warns(IncrementWarning) as rec:
        with pytest.raises(DefinitionError) as exc_info:
            load(tmp_path / "defs.yaml")
    assert exc_info.value.path == str(tmp_path / "defs.yaml")
    assert exc_info.value.code == "definition.sql.parse"
    assert "definition.sql.dialect_guess_parse_failed" in warning_codes(rec)
    dialect_warning = next(
        w.message
        for w in rec
        if getattr(w.message, "code", None) == "definition.sql.dialect_guess_parse_failed"
    )
    assert isinstance(dialect_warning, IncrementWarning)
    assert isinstance(dialect_warning.context["parse_exc"], str)
    restored = pickle.loads(pickle.dumps(dialect_warning))
    assert isinstance(restored, IncrementWarning)
    assert restored.context == dialect_warning.context


def test_invalid_sql_without_dialect_raises(tmp_path: Path):
    """An invalid fact-source query is rejected even when no dialect is declared."""
    _dump_yaml(
        {
            "fact_sources": [
                {
                    "name": "bad_src",
                    "sql": "SEL ECT * FORM events",  # intentional typos
                    "timestamp_column": "ts",
                    "entities": ["u"],
                    "facts": [{"name": "f", "column": None}],
                }
            ],
            "exposures": [{"name": "e", "fact": "f"}],
            "metrics": [
                {"type": "conversion", "name": "m", "entity": "u", "fact": "f", "window_days": 7}
            ],
            "experiments": [
                {
                    "name": "x",
                    "plan": {},
                    "exposure": "e",
                    "unit": "u",
                    "start": "2024-01-01",
                    "control_group": "C",
                }
            ],
        },
        (tmp_path / "defs.yaml"),
    )

    with pytest.warns(IncrementWarning):
        with pytest.raises(DefinitionError) as exc_info:
            load(tmp_path / "defs.yaml")
    assert exc_info.value.path == str(tmp_path / "defs.yaml")
    assert exc_info.value.code == "definition.sql.parse"


def test_empty_sql_is_fine(tmp_path: Path):
    """An empty or whitespace-only sql string is skipped by the SQL checker."""
    _dump_yaml(
        {
            "fact_sources": [
                {
                    "name": "empty_src",
                    "sql": "",
                    "timestamp_column": "ts",
                    "entities": ["u"],
                    "facts": [{"name": "f", "column": None}],
                }
            ],
            "exposures": [{"name": "e", "fact": "f"}],
            "metrics": [
                {"type": "conversion", "name": "m", "entity": "u", "fact": "f", "window_days": 7}
            ],
            "experiments": [
                {
                    "name": "x",
                    "plan": {},
                    "exposure": "e",
                    "unit": "u",
                    "start": "2024-01-01",
                    "control_group": "C",
                }
            ],
        },
        (tmp_path / "defs.yaml"),
    )

    defs = load(tmp_path / "defs.yaml")
    # Must load despite empty sql (the model requires it, but empty string is fine)
    assert defs.fact_sources[0].sql == ""


def test_snowflake_semi_structured_sql_loads_with_declared_dialect(tmp_path: Path):
    """Snowflake VARIANT-path syntax parses cleanly and warning-free once
    `Definitions.dialect: snowflake` is declared."""
    _dump_yaml(
        {
            "dialect": "snowflake",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT payload:user.id::string AS uid, ts FROM raw_events",
                    "timestamp_column": "ts",
                    "entities": ["u"],
                    "facts": [{"name": "f", "column": None}],
                }
            ],
            "exposures": [{"name": "e", "fact": "f"}],
            "metrics": [
                {
                    "type": "conversion",
                    "name": "m",
                    "entity": "u",
                    "fact": "f",
                    "window_days": 7,
                }
            ],
            "experiments": [
                {
                    "name": "x",
                    "plan": {},
                    "exposure": "e",
                    "unit": "u",
                    "start": "2024-01-01",
                    "control_group": "C",
                }
            ],
        },
        (tmp_path / "defs.yaml"),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        defs = load(tmp_path / "defs.yaml")
    assert defs.dialect == "snowflake"


def test_snowflake_semi_structured_sql_without_dialect_raises(tmp_path: Path):
    """Snowflake-specific SQL needs its dialect declared for read-only admission."""
    _dump_yaml(
        {
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT payload:user.id::string AS uid, ts FROM raw_events",
                    "timestamp_column": "ts",
                    "entities": ["u"],
                    "facts": [{"name": "f", "column": None}],
                }
            ],
            "exposures": [{"name": "e", "fact": "f"}],
            "metrics": [
                {"type": "conversion", "name": "m", "entity": "u", "fact": "f", "window_days": 7}
            ],
            "experiments": [
                {
                    "name": "x",
                    "plan": {},
                    "exposure": "e",
                    "unit": "u",
                    "start": "2024-01-01",
                    "control_group": "C",
                }
            ],
        },
        (tmp_path / "defs.yaml"),
    )

    with pytest.warns(IncrementWarning):
        with pytest.raises(DefinitionError) as exc_info:
            load(tmp_path / "defs.yaml")
    assert exc_info.value.path == str(tmp_path / "defs.yaml")
    assert exc_info.value.code == "definition.sql.parse"


def test_duckdb_fact_sql_transpiles_to_other_dialects(tmp_path: Path):
    """duckdb-authored fact SQL loads cleanly and transpiles to other dialects without raising:
    `con.sql(sql, dialect=defs.dialect)` transpiles via sqlglot whenever dialects differ (see
    `ibis.backends.sql.SQLBackend._transpile_sql`); this is a compile-only stand-in for running
    against a live non-duckdb warehouse."""
    sql = (
        "SELECT user_id, ts + INTERVAL 7 DAY AS future_ts FROM events "
        "QUALIFY ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY ts) = 1"
    )
    _dump_yaml(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": sql,
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [{"name": "f", "column": None}],
                }
            ],
            "exposures": [{"name": "e", "fact": "f"}],
            "metrics": [
                {
                    "type": "conversion",
                    "name": "m",
                    "entity": "user_id",
                    "fact": "f",
                    "window_days": 7,
                }
            ],
            "experiments": [
                {
                    "name": "x",
                    "plan": {},
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2024-01-01",
                    "control_group": "C",
                }
            ],
        },
        (tmp_path / "defs.yaml"),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        defs = load(tmp_path / "defs.yaml")
    assert defs.dialect == "duckdb"

    fact_sql = defs.fact_sources[0].sql
    for target_dialect in ("snowflake", "bigquery"):
        transpiled = sqlglot.transpile(fact_sql, read=defs.dialect, write=target_dialect)[0]
        assert transpiled


# ── extra="forbid" enforced via loader ─────────────────────────────────


def test_extra_forbid_nested_metric_key_via_loader(tmp_path: Path):
    """A typo inside a metric definition is caught by extra='forbid'."""
    _dump_yaml(
        {
            "fact_sources": [
                {
                    "name": "src",
                    "sql": "SELECT 1",
                    "timestamp_column": "ts",
                    "entities": ["u"],
                    "facts": [{"name": "f", "column": None}],
                }
            ],
            "exposures": [{"name": "e", "fact": "f"}],
            "metrics": [
                {
                    "type": "mean",
                    "name": "m",
                    "entity": "u",
                    "fact": "f",
                    "threshhold_days": 7,  # intentional typo
                }
            ],
            "experiments": [],
        },
        (tmp_path / "defs.yaml"),
    )

    with pytest.raises(DefinitionError) as exc_info:
        load(tmp_path / "defs.yaml")
    assert exc_info.value.code == "definition.validation"


# ── Example integration test ───────────────────────────────────────────


def test_example_loads():
    """The examples/definitions/ directory must load without error."""
    example_dir = Path(__file__).parent.parent.parent / "examples" / "definitions"
    assert example_dir.is_dir(), f"missing checked-in example directory: {example_dir}"

    defs = load(example_dir)
    assert isinstance(defs, Definitions)
    # Must have at least the expected categories populated
    assert len(defs.fact_sources) >= 1
    assert len(defs.exposures) >= 1
    assert len(defs.metrics) >= 1
    assert len(defs.experiments) >= 1


def test_realistic_demo_example_loads():
    """The examples/realistic_demo/definitions/ directory must load without error: a fast,
    warehouse-free (YAML + sqlglot only) guard against the YAML files rotting out of sync with
    generate.py (e.g. a renamed fact/property/table or a SQL typo)."""
    example_dir = (
        Path(__file__).parent.parent.parent / "examples" / "realistic_demo" / "definitions"
    )
    assert example_dir.is_dir(), f"missing checked-in example directory: {example_dir}"
    defs = load(example_dir)
    assert isinstance(defs, Definitions)
    assert {s.name for s in defs.fact_sources} == {"orders", "web_events", "experiment_exposure"}
    assert len(defs.exposures) >= 1
    assert {m.name for m in defs.metrics} == {
        "conversion_rate",
        "revenue_per_user",
        "average_order_value",
        "d7_retention",
        "checkout_latency_ms",
    }
    latency = next(m for m in defs.metrics if m.name == "checkout_latency_ms")
    assert isinstance(latency, RatioMetric)
    assert latency.numerator.fact == "page_load"
    assert latency.numerator.aggregation == "sum"
    assert latency.numerator.window_days == 14
    assert latency.denominator.fact == "page_load"
    assert latency.denominator.aggregation == "count"
    assert latency.denominator.window_days == 14
    assert latency.numerator.filters == latency.denominator.filters
    assert len(defs.experiments) == 1


# ── Regression: window_days=None warning ────────────────────────────


def test_window_days_none_warns_for_mean_metric(tmp_path: Path):
    """MeanMetric with window_days=None produces a warning."""
    _dump_yaml(
        {
            "fact_sources": [
                {
                    "name": "s",
                    "sql": "SELECT 1",
                    "timestamp_column": "ts",
                    "entities": ["u"],
                    "facts": [{"name": "ev", "column": "a"}],
                }
            ],
            "exposures": [{"name": "e", "fact": "ev"}],
            "metrics": [{"type": "mean", "name": "m", "entity": "u", "fact": "ev"}],
            "experiments": [
                {
                    "name": "x",
                    "plan": {},
                    "exposure": "e",
                    "unit": "u",
                    "start": "2024-01-01",
                    "control_group": "C",
                }
            ],
        },
        (tmp_path / "defs.yaml"),
    )
    with pytest.warns(IncrementWarning):
        load(tmp_path / "defs.yaml")


def test_window_days_none_warns_for_conversion_metric(tmp_path: Path):
    """ConversionMetric with window_days=None produces a warning."""
    _dump_yaml(
        {
            "fact_sources": [
                {
                    "name": "s",
                    "sql": "SELECT 1",
                    "timestamp_column": "ts",
                    "entities": ["u"],
                    "facts": [{"name": "ev", "column": None}],
                }
            ],
            "exposures": [{"name": "e", "fact": "ev"}],
            "metrics": [{"type": "conversion", "name": "m", "entity": "u", "fact": "ev"}],
            "experiments": [
                {
                    "name": "x",
                    "plan": {},
                    "exposure": "e",
                    "unit": "u",
                    "start": "2024-01-01",
                    "control_group": "C",
                }
            ],
        },
        (tmp_path / "defs.yaml"),
    )
    with pytest.warns(IncrementWarning):
        load(tmp_path / "defs.yaml")


def test_window_days_set_no_warning(tmp_path: Path):
    """Metric with explicit window_days produces no warning."""
    _dump_yaml(
        {
            "fact_sources": [
                {
                    "name": "s",
                    "sql": "SELECT 1",
                    "timestamp_column": "ts",
                    "entities": ["u"],
                    "facts": [{"name": "ev", "column": "a"}],
                }
            ],
            "exposures": [{"name": "e", "fact": "ev"}],
            "metrics": [
                {"type": "mean", "name": "m", "entity": "u", "fact": "ev", "window_days": 30}
            ],
            "experiments": [
                {
                    "name": "x",
                    "plan": {},
                    "exposure": "e",
                    "unit": "u",
                    "start": "2024-01-01",
                    "control_group": "C",
                }
            ],
        },
        (tmp_path / "defs.yaml"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load(tmp_path / "defs.yaml")


# ── Regression: property-name duplicates with file origins ─────────


def test_duplicate_property_name_in_source_rejected(tmp_path: Path):
    """Duplicate property names within a fact source include file names."""
    _dump_yaml(
        {
            "fact_sources": [
                {
                    "name": "src",
                    "sql": "SELECT 1",
                    "timestamp_column": "ts",
                    "entities": ["u"],
                    "facts": [{"name": "ev", "column": None}],
                    "properties": [
                        {"name": "dup", "column": "a", "dtype": "string"},
                        {"name": "dup", "column": "b", "dtype": "int"},
                    ],
                }
            ],
            "exposures": [{"name": "e", "fact": "ev"}],
            "metrics": [{"type": "conversion", "name": "m", "entity": "u", "fact": "ev"}],
            "experiments": [
                {
                    "name": "x",
                    "plan": {},
                    "exposure": "e",
                    "unit": "u",
                    "start": "2024-01-01",
                    "control_group": "C",
                }
            ],
        },
        (tmp_path / "defs.yaml"),
    )
    with pytest.raises(DefinitionError) as exc_info:
        load(tmp_path / "defs.yaml")
    assert exc_info.value.code == "definition.duplicates"


# ── day_boundary: definitions-level default flows onto experiments ─────


def test_definitions_day_boundary_flows_to_experiments_unless_overridden(tmp_path: Path):
    """A definitions-level day_boundary is inherited by experiments that do
    not declare their own; an explicit per-experiment value is kept."""
    _dump_yaml(
        {
            "dialect": "duckdb",
            "day_boundary": "UTC-08:00",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM events",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [{"name": "page_view", "column": None}],
                }
            ],
            "exposures": [{"name": "enrolled", "fact": "page_view"}],
            "experiments": [
                {
                    "name": "inherits",
                    "plan": {},
                    "exposure": "enrolled",
                    "unit": "user_id",
                    "start": "2024-06-01",
                    "control_group": "C",
                },
                {
                    "name": "explicit_utc",
                    "plan": {},
                    "exposure": "enrolled",
                    "unit": "user_id",
                    "start": "2024-06-01",
                    "control_group": "C",
                    "day_boundary": "UTC",
                },
            ],
        },
        (tmp_path / "defs.yaml"),
    )

    defs = load(tmp_path / "defs.yaml")
    inherits = defs.experiment("inherits")
    explicit = defs.experiment("explicit_utc")
    assert inherits is not None and explicit is not None
    assert inherits.day_boundary == "UTC-08:00"
    assert explicit.day_boundary == "UTC"


def test_definitions_day_boundary_inheritance_survives_dump_reload_round_trip(tmp_path: Path):
    """Round-trip fidelity (yjc9): `model_dump()` writes every field's
    resolved value regardless of whether it was declared. An experiment
    that only INHERITED its day_boundary from the definitions-level
    default must still serialise as unset, so a later edit to just the
    top-level default -- the common "export, tweak the org default,
    reload" workflow -- re-propagates instead of the experiment staying
    silently pinned to the stale value it happened to resolve to at the
    time of the first dump."""
    _dump_yaml(
        {
            "dialect": "duckdb",
            "day_boundary": "UTC-08:00",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM events",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [{"name": "page_view", "column": None}],
                }
            ],
            "exposures": [{"name": "enrolled", "fact": "page_view"}],
            "experiments": [
                {
                    "name": "inherits",
                    "plan": {},
                    "exposure": "enrolled",
                    "unit": "user_id",
                    "start": "2024-06-01",
                    "control_group": "C",
                },
            ],
        },
        (tmp_path / "defs.yaml"),
    )
    defs = load(tmp_path / "defs.yaml")
    inherits = defs.experiment("inherits")
    assert inherits is not None
    assert inherits.day_boundary == "UTC-08:00"

    # First round trip: dump and reload with the SAME top-level default.
    # The resolved value must survive unchanged either way, so this alone
    # would not distinguish "declared" from "inherited".
    dumped = defs.model_dump(mode="json")
    _dump_yaml(dumped, tmp_path / "roundtrip1.yaml")
    reloaded = load(tmp_path / "roundtrip1.yaml")
    reloaded_exp = reloaded.experiment("inherits")
    assert reloaded_exp is not None
    assert reloaded_exp.day_boundary == "UTC-08:00"

    # Change only the dumped top-level default and reload: an experiment that
    # never declared day_boundary must inherit the new value, not stay pinned
    # because the round trip marked the old value as explicitly declared.
    dumped["day_boundary"] = "UTC"
    _dump_yaml(dumped, tmp_path / "roundtrip2.yaml")
    repropagated = load(tmp_path / "roundtrip2.yaml")
    repropagated_exp = repropagated.experiment("inherits")
    assert repropagated_exp is not None
    assert repropagated_exp.day_boundary == "UTC"


def test_empty_definitions_directory_warns(tmp_path: Path):
    """A directory with zero YAML files loads as an empty Definitions but must WARN: a typo'd
    path is the likely cause, and its downstream symptom ('unknown metric') would land far
    from here if silent. Composing from nothing stays valid, so it is a warning, not a refusal."""
    with pytest.warns(IncrementWarning):
        defs = load(tmp_path)
    assert defs.fact_sources == ()
    assert defs.metrics == ()


# ── Experiment.plan migration: legacy metrics:/guardrails: keys ────────


def _plan_migration_dict(tmp_path: Path, experiment_extra: dict) -> Path:
    _dump_yaml(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM events",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [{"name": "page_view", "column": None}],
                }
            ],
            "exposures": [{"name": "enrolled", "fact": "page_view"}],
            "metrics": [
                {
                    "type": "conversion",
                    "name": "visit_rate",
                    "entity": "user_id",
                    "fact": "page_view",
                    "window_days": 7,
                }
            ],
            "experiments": [
                {
                    "name": "exp_1",
                    "exposure": "enrolled",
                    "unit": "user_id",
                    "start": "2024-06-01",
                    "control_group": "C",
                    **experiment_extra,
                }
            ],
        },
        (tmp_path / "defs.yaml"),
    )
    return tmp_path / "defs.yaml"


def test_experiment_plan_round_trips_via_loader(tmp_path: Path):
    """A declared `experiments[0].plan` loads and round-trips through the
    loader unchanged."""
    path = _plan_migration_dict(
        tmp_path, {"plan": {"secondaries": ["visit_rate"], "guardrails": []}}
    )
    defs = load(path)
    assert defs.experiments[0].plan.secondaries == ("visit_rate",)
    assert defs.experiments[0].metric_names == ["visit_rate"]


def test_legacy_experiment_metrics_key_raises_migration_error(tmp_path: Path):
    """A legacy experiment-level `metrics:` key is caught in the raw-dict
    walk, before pydantic's `extra='forbid'`, and named explicitly."""
    path = _plan_migration_dict(tmp_path, {"metrics": ["visit_rate"]})
    with pytest.raises(DefinitionError) as exc_info:
        load(path)
    assert exc_info.value.code == "definition.migration.metrics"


def test_legacy_experiment_guardrails_key_raises_migration_error(tmp_path: Path):
    """A legacy experiment-level `guardrails:` key is caught the same way,
    naming `plan.guardrails`."""
    path = _plan_migration_dict(tmp_path, {"guardrails": ["visit_rate"]})
    with pytest.raises(DefinitionError) as exc_info:
        load(path)
    assert exc_info.value.code == "definition.migration.guardrails"


def test_missing_plan_key_raises_pydantic_missing_field_error(tmp_path: Path):
    """No `plan:` at all -- not a migration case, just a required field --
    fails pydantic validation naming `plan`."""
    path = _plan_migration_dict(tmp_path, {})
    with pytest.raises(DefinitionError) as exc_info:
        load(path)
    assert exc_info.value.code == "definition.validation"


def test_metric_names_orders_primary_then_secondary_then_guardrail(tmp_path: Path):
    """`Experiment.metric_names` spans every plan role in declaration
    order: primaries, then secondaries, then guardrails."""
    _dump_yaml(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM events",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [{"name": "page_view", "column": None}],
                }
            ],
            "exposures": [{"name": "enrolled", "fact": "page_view"}],
            "metrics": [
                {
                    "type": "conversion",
                    "name": name,
                    "entity": "user_id",
                    "fact": "page_view",
                    "window_days": 7,
                }
                for name in ("prim", "sec", "guard")
            ],
            "experiments": [
                {
                    "name": "exp_1",
                    "exposure": "enrolled",
                    "unit": "user_id",
                    "start": "2024-06-01",
                    "control_group": "C",
                    "plan": {
                        "primary": "prim",
                        "secondaries": ["sec"],
                        "guardrails": ["guard"],
                    },
                }
            ],
        },
        (tmp_path / "defs.yaml"),
    )
    defs = load(tmp_path / "defs.yaml")
    assert defs.experiments[0].metric_names == ["prim", "sec", "guard"]
    assert defs.experiments[0].guardrail_names == ["guard"]


class TestDuplicateKeyScanPreservesLoaderSemantics:
    """The duplicate-key scan runs before SafeConstructor, so it must not
    change what SafeLoader otherwise accepts or how it reports errors."""

    def test_anchor_merge_keys_still_flatten(self, tmp_path: Path) -> None:
        # The merge key is flattened by the constructor, not a real key: scanning
        # it as one both fails to construct and looks like a duplicate.
        (tmp_path / "defs.yaml").write_text(
            "fact_sources:\n"
            "  - &base\n"
            "    name: events\n"
            '    sql: "SELECT * FROM events"\n'
            "    timestamp_column: ts\n"
            "    entities: [u]\n"
            "    facts: [{name: f1, column: null}]\n"
            "  - <<: *base\n"
            "    name: events_copy\n"
            "    facts: [{name: f2, column: null}]\n"
        )
        defs = load(tmp_path / "defs.yaml")
        assert {source.name for source in defs.fact_sources} == {"events", "events_copy"}
        # `sql` is declared only by the anchor the second source merges.
        assert defs.fact_sources[1].sql == "SELECT * FROM events"

    def test_a_real_duplicate_is_still_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "defs.yaml"
        path.write_text("dialect: duckdb\ndialect: duckdb\n")
        with pytest.raises(DefinitionError) as raised:
            load(path)
        assert raised.value.code == "definition.yaml"
        assert raised.value.path == str(path)

    def test_an_unhashable_key_raises_the_yaml_error_not_a_type_error(self, tmp_path: Path) -> None:
        # _read_yaml() wraps ConstructorError as a path-aware DefinitionError; a
        # bare TypeError would escape that translation.
        path = tmp_path / "defs.yaml"
        path.write_text("dialect: duckdb\n? [a, b]\n: v\n")
        with pytest.raises(DefinitionError) as raised:
            load(path)
        assert raised.value.code == "definition.yaml"
        assert raised.value.path == str(path)


class TestRepeatedMergeKeysAreRefused:
    """A mapping repeating ``<<`` silently keeps only the last one, the same
    ambiguity the duplicate-key check exists to refuse."""

    def test_two_merge_keys_in_one_mapping_are_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "defs.yaml"
        path.write_text(
            "fact_sources:\n"
            "  - &base\n"
            "    name: events\n"
            '    sql: "SELECT * FROM events"\n'
            "    timestamp_column: ts\n"
            "    entities: [u]\n"
            "    facts: [{name: f, column: null}]\n"
            "  - &extra\n"
            "    name: events_copy\n"
            "    facts: [{name: f, column: null}]\n"
            "  - <<: *base\n"
            "    <<: *extra\n"
        )
        with pytest.raises(DefinitionError) as raised:
            load(path)
        assert raised.value.code == "definition.yaml"
        assert raised.value.path == str(path)

    def test_one_merge_key_taking_a_sequence_still_works(self, tmp_path: Path) -> None:
        # The declared way to merge several anchors, with explicit precedence:
        # the first anchor of the sequence wins a conflicting key.
        (tmp_path / "defs.yaml").write_text(
            "fact_sources:\n"
            "  - &first\n"
            "    name: events\n"
            '    sql: "SELECT * FROM events"\n'
            "    timestamp_column: ts\n"
            "    entities: [u]\n"
            "    facts: [{name: f1, column: null}]\n"
            "  - &second\n"
            "    name: events_alt\n"
            '    sql: "SELECT * FROM other"\n'
            "    timestamp_column: ts\n"
            "    entities: [u]\n"
            "    facts: [{name: f2, column: null}]\n"
            "  - <<: [*first, *second]\n"
            "    name: events_merged\n"
            "    facts: [{name: f3, column: null}]\n"
        )
        defs = load(tmp_path / "defs.yaml")
        assert {source.name for source in defs.fact_sources} == {
            "events",
            "events_alt",
            "events_merged",
        }
        assert defs.fact_sources[2].sql == "SELECT * FROM events"


def test_load_refuses_ambiguous_breakout_source_with_code(tmp_path):
    from increment.errors import DefinitionError
    from increment.semantics.loader import load

    (tmp_path / "defs.yaml").write_text(
        """
fact_sources:
  - name: a
    sql: "SELECT * FROM a"
    timestamp_column: ts
    entities: [user_id]
    facts: [{name: a_fact, column: v}]
    properties: [{name: country, column: country, dtype: string, as_of: static}]
  - name: b
    sql: "SELECT * FROM b"
    timestamp_column: ts
    entities: [user_id]
    facts: [{name: b_fact, column: v}]
    properties: [{name: country, column: country, dtype: string, as_of: static}]
exposures:
  - name: assignment
    sql: "SELECT 1"
experiments:
  - name: e1
    unit: user_id
    control_group: control
    exposure: assignment
    start: "2026-01-01"
    breakouts: [{property: country}]
    plan: {primary: []}
"""
    )
    with pytest.raises(DefinitionError) as exc_info:
        load(tmp_path)
    assert exc_info.value.code == "definition.breakout.ambiguous_source"
    assert exc_info.value.path == str(tmp_path)


def test_admit_read_only_sql_unknown_dialect_is_coded():
    from increment.errors import InvalidRequestError

    with pytest.raises(InvalidRequestError) as raised:
        admit_read_only_sql("select 1", dialect="not-a-real-dialect", label="fact source 'x'")
    assert raised.value.code == "definition.sql.dialect"


def test_admit_read_only_sql_not_read_only_is_coded():
    from increment.errors import InvalidRequestError

    with pytest.raises(InvalidRequestError) as raised:
        admit_read_only_sql("delete from t", dialect=None, label="fact source 'x'")
    assert raised.value.code == "definition.sql.not_read_only"


# ── Observational adjustment covariates ───────────────────────────────


def _observational_defs(tmp_path: Path, *, sources: str, covariates: str) -> Path:
    text = f"""
dialect: duckdb
fact_sources:
{sources}
exposures:
  - name: assignment
    fact: exposure
metrics:
  - {{name: outcome, type: mean, entity: user_id, fact: outcome, aggregation: sum, window_days: 7}}
experiments:
  - name: exp
    exposure: assignment
    unit: user_id
    control_group: control
    start: 2025-01-01
    plan: {{secondaries: [outcome]}}
    design:
      mechanism: observational
      covariates:
{covariates}
"""
    path = tmp_path / "defs.yaml"
    path.write_text(text)
    return path


_ONE_SOURCE = """
  - name: a
    sql: "SELECT * FROM a"
    timestamp_column: ts
    entities: [user_id]
    facts: [{name: exposure, column: null}, {name: outcome, column: v}]
    properties:
      - {name: tenure, column: tenure, dtype: float, as_of: pre_exposure}
      - {name: last_page, column: last_page, dtype: float, as_of: event_time}
      - {name: country, column: country, dtype: string, as_of: static}
      - {name: signup_date, column: signup_date, dtype: date, as_of: static}
"""


def test_observational_covariate_binds_to_pre_exposure_numeric_property(tmp_path):
    from increment.semantics.design import Observational
    from increment.semantics.models import AdjustmentCovariate, ObservationalDeclaration

    path = _observational_defs(
        tmp_path, sources=_ONE_SOURCE, covariates="        - {property: tenure, source: a}"
    )
    exp = load(path).experiment("exp")
    assert exp is not None and isinstance(exp.design, ObservationalDeclaration)
    assert exp.design.covariates == (AdjustmentCovariate(property="tenure", source="a"),)
    design = exp.resolved_design()
    assert isinstance(design, Observational)
    assert design.adjustment.covariates == ("tenure",)


def test_observational_covariate_without_source_resolves_when_unambiguous(tmp_path):
    from increment.semantics.design import Observational

    path = _observational_defs(
        tmp_path, sources=_ONE_SOURCE, covariates="        - {property: tenure}"
    )
    exp = load(path).experiment("exp")
    assert exp is not None
    design = exp.resolved_design()
    assert isinstance(design, Observational)
    assert design.adjustment.covariates == ("tenure",)


def test_observational_covariate_binds_to_a_string_property_as_categorical(tmp_path):
    """A string property is a categorical adjustment covariate: it resolves
    into the design's adjustment set beside a numeric one with no encoding
    declared anywhere in the definitions."""
    from increment.semantics.design import Observational

    path = _observational_defs(
        tmp_path,
        sources=_ONE_SOURCE,
        covariates="        - {property: tenure, source: a}\n        - {property: country, source: a}",
    )
    exp = load(path).experiment("exp")
    assert exp is not None
    design = exp.resolved_design()
    assert isinstance(design, Observational)
    assert design.adjustment.covariates == ("tenure", "country")


def test_observational_covariate_ambiguous_without_source_refuses(tmp_path):
    sources = """
  - name: a
    sql: "SELECT * FROM a"
    timestamp_column: ts
    entities: [user_id]
    facts: [{name: exposure, column: null}]
    properties:
      - {name: tenure, column: tenure, dtype: float, as_of: pre_exposure}
  - name: b
    sql: "SELECT * FROM b"
    timestamp_column: ts
    entities: [user_id]
    facts: [{name: outcome, column: v}]
    properties:
      - {name: tenure, column: tenure, dtype: float, as_of: pre_exposure}
"""
    path = _observational_defs(tmp_path, sources=sources, covariates="        - {property: tenure}")
    with pytest.raises(DefinitionError) as raised:
        load(path)
    assert raised.value.code == "definition.observational.ambiguous_covariate_source"
    assert raised.value.context["candidates"] == ("a", "b")


@pytest.mark.parametrize(
    ("covariate", "code"),
    [
        ("{property: last_page, source: a}", "definition.check_observational.covariate_as_of"),
        ("{property: signup_date, source: a}", "definition.check_observational.covariate_dtype"),
        (
            "{property: tenure, source: nope}",
            "definition.check_observational.covariate_source_found",
        ),
        ("{property: missing}", "definition.check_observational.covariate_could"),
    ],
)
def test_observational_covariate_refuses_unusable_binding(tmp_path, covariate, code):
    path = _observational_defs(tmp_path, sources=_ONE_SOURCE, covariates=f"        - {covariate}")
    with pytest.raises(DefinitionError) as raised:
        load(path)
    errors = raised.value.context["errors"]
    assert code in {c for c, _ in errors}  # ty: ignore[not-iterable]
