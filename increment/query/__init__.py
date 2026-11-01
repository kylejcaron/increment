from __future__ import annotations

import warnings

from increment.query._workflows import build_panel_for_metric as build_panel_for_metric
from increment.query._workflows import panel_sql as panel_sql
from increment.query._workflows import summary_sql as summary_sql

# Ibis' DuckDB backend calls DuckDB's deprecated fetch_arrow_table API.
# Remove when Ibis switches APIs (lock: Ibis 12.0.0 / DuckDB 1.5.5;
# floors: Ibis 10.1.0 / DuckDB 1.3.2 on Python 3.12-13, Ibis 11.0.0 /
# DuckDB 1.4.3 on Python 3.14).
warnings.filterwarnings(
    "ignore",
    message=r"fetch_arrow_table\(\) is deprecated",
    category=DeprecationWarning,
    module=r"ibis\.backends\.duckdb",
)
