"""Loader behaviour for dim_sources: merge, SQL lint, duplicates."""

import textwrap

import pytest

from increment.errors import DefinitionError
from increment.semantics.loader import load

# No `dialect:` here - a scalar key may appear in only ONE file of a
# definitions directory (the loader refuses duplicate scalars), and the duplicate-dim test below writes this document twice.
DIM_YAML = textwrap.dedent(
    """
    dim_sources:
      - name: users
        sql: SELECT user_id, country FROM dim_user
        entity: user_id
        properties:
          - name: country
            column: country
            as_of: static
    """
)

FACT_YAML = textwrap.dedent(
    """
    dialect: duckdb
    fact_sources:
      - name: orders
        sql: SELECT 'order' AS event, user_id, ordered_at, amount FROM fact_orders
        timestamp_column: ordered_at
        entities: [user_id]
        dims: [users]
        facts:
          - name: order
            column: amount
    """
)


def test_dim_sources_merge_across_files(tmp_path):
    (tmp_path / "dims.yaml").write_text(DIM_YAML)
    (tmp_path / "facts.yaml").write_text(FACT_YAML)
    defs = load(tmp_path)
    assert [d.name for d in defs.dim_sources] == ["users"]
    assert defs.fact_sources[0].dims == ("users",)


def test_duplicate_dim_source_across_files_names_both_files(tmp_path):
    (tmp_path / "a.yaml").write_text(DIM_YAML)
    (tmp_path / "b.yaml").write_text(DIM_YAML)
    (tmp_path / "facts.yaml").write_text(FACT_YAML)
    with pytest.raises(DefinitionError) as exc_info:
        load(tmp_path)
    assert exc_info.value.code == "definition.duplicates"
    files = exc_info.value.sources["dim_source:users"]
    assert {"a.yaml", "b.yaml"} <= {f.split("/")[-1] for f in files}


def test_dim_source_sql_is_checked(tmp_path):
    # With a declared dialect, unparseable SQL is a hard load error (same
    # contract fact sources get); the label must name the dim source.
    bad = DIM_YAML.replace("SELECT user_id, country FROM dim_user", "SELEC oops FRM")
    (tmp_path / "dims.yaml").write_text(bad)
    (tmp_path / "facts.yaml").write_text(FACT_YAML)
    with pytest.raises(DefinitionError) as exc_info:
        load(tmp_path)
    assert exc_info.value.code == "definition.sql.parse"
