"""Optional presentation helpers for an experiment dashboard.

Bind your own experiment, prepare one snapshot, and render sections:

    con = ibis.duckdb.connect(...)
    analysis = Analysis.from_definitions("my_experiment", "definitions/", con)
    config = DashboardConfig(expected_allocation={"control": 0.5, "treatment": 0.5})
    snapshot = prepare_dashboard(analysis, config=config)
    render_health(snapshot)

Install with the ``dashboard`` extra. Core Increment never imports this
package: estimation, power, and the dataframe entry points stay free of
marimo, CoefTable, and pandas.
"""

from increment.dashboard._data import (
    DashboardConfig,
    DashboardSnapshot,
    ExploreView,
    group_data_csv,
    load_explore,
    prepare_dashboard,
    readout_csv,
)
from increment.dashboard._html import (
    dashboard_styles,
    render_details,
    render_explore,
    render_header,
    render_health,
    render_metric_details,
    render_results,
)

__all__ = [
    "DashboardConfig",
    "DashboardSnapshot",
    "ExploreView",
    "dashboard_styles",
    "group_data_csv",
    "load_explore",
    "prepare_dashboard",
    "readout_csv",
    "render_details",
    "render_explore",
    "render_header",
    "render_health",
    "render_metric_details",
    "render_results",
]
