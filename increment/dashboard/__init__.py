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

from increment.dashboard._app import render_dashboard
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
from increment.dashboard._theme import (
    MIDNIGHT,
    DashboardCharts,
    DashboardLayout,
    DashboardPalette,
    DashboardPrint,
    DashboardTheme,
    DashboardTypography,
)

__all__ = [
    "MIDNIGHT",
    "DashboardCharts",
    "DashboardLayout",
    "DashboardPalette",
    "DashboardPrint",
    "DashboardTheme",
    "DashboardTypography",
    "DashboardConfig",
    "DashboardSnapshot",
    "ExploreView",
    "dashboard_styles",
    "group_data_csv",
    "load_explore",
    "prepare_dashboard",
    "readout_csv",
    "render_dashboard",
    "render_details",
    "render_explore",
    "render_header",
    "render_health",
    "render_metric_details",
    "render_results",
]
