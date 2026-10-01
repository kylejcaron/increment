# /// script
# [tool.marimo.display]
# theme = "light"
# ///
import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full", css_file="ab_testing_dashboard.css")


@app.cell
def _():
    from pathlib import Path

    import _dashboard_demo as demo
    import ibis
    import marimo as mo

    from increment import Analysis
    from increment.dashboard import (
        DashboardConfig,
        dashboard_styles,
        group_data_csv,
        load_explore,
        prepare_dashboard,
        readout_csv,
        render_details,
        render_explore,
        render_header,
        render_health,
        render_metric_details,
        render_results,
    )

    return (
        Analysis,
        DashboardConfig,
        Path,
        dashboard_styles,
        demo,
        group_data_csv,
        ibis,
        load_explore,
        mo,
        prepare_dashboard,
        readout_csv,
        render_details,
        render_explore,
        render_header,
        render_health,
        render_metric_details,
        render_results,
    )


@app.cell
def _(Path, demo):
    # The synthetic warehouse is generated, never committed. Only this example
    # needs the repository root: its definitions hold relative parquet paths.
    demo_manifest = demo.ensure_demo_warehouse(Path.cwd())
    return (demo_manifest,)


@app.cell
def _(ibis):
    con = ibis.duckdb.connect()
    return (con,)


@app.cell
def _(Analysis, DashboardConfig, con, demo, demo_manifest):
    analysis = Analysis.from_definitions(
        "checkout_redesign", "examples/realistic_demo/definitions", con
    )
    config = DashboardConfig(
        expected_allocation={"control": 0.5, "treatment": 0.5},
        title="Checkout redesign",
        source_label="Synthetic snapshot",
        metric_units={
            "revenue_per_user": "USD",
            "average_order_value": "USD",
            "checkout_latency_ms": "ms",
        },
        provenance=demo.demo_provenance(demo_manifest),
    )
    return analysis, config


@app.cell
def _(analysis, config, prepare_dashboard):
    snapshot = prepare_dashboard(analysis, config=config)
    return (snapshot,)


@app.cell(hide_code=True)
def _(dashboard_styles):
    # Injects packaged section styles; no data changes. The app's css_file
    # separately styles the notebook page. Keep this cell for HTML exports.
    dashboard_styles()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
        <header class="inc-dashboard-root inc-dashboard-navigation">
          <a class="inc-dashboard-brand" href="#overview">increment</a>
          <nav aria-label="Dashboard sections">
            <a href="#overview">Overview</a>
            <a href="#health">Health</a>
            <a href="#results">Results</a>
            <a href="#explore">Explore</a>
            <a href="#metric-details">Metric details</a>
            <a href="#details">Details</a>
          </nav>
        </header>
    """)
    return


@app.cell(hide_code=True)
def _(render_header, snapshot):
    render_header(snapshot)
    return


@app.cell(hide_code=True)
def _(render_health, snapshot):
    render_health(snapshot)
    return


@app.cell(hide_code=True)
def _(render_results, snapshot):
    render_results(snapshot)
    return


@app.cell
def _(mo):
    explore_view = mo.ui.dropdown(
        options={
            "Cumulative lift": "cumulative_lift",
            "Daily metric values": "daily_values",
            "Cumulative metric values": "cumulative_values",
            "Segments": "segments",
        },
        value="Cumulative lift",
        label="View",
    )
    return (explore_view,)


@app.cell
def _(mo, snapshot):
    segment_choice = mo.ui.dropdown(
        options={
            f"{dimension} · {source}": (source, dimension)
            for source, dimension in snapshot.breakouts
        },
        value=next((f"{dimension} · {source}" for source, dimension in snapshot.breakouts), None),
        label="Breakout",
    )
    return (segment_choice,)


@app.cell
def _(explore_view, mo):
    # Recreated per view: the maturity gate is cumulative-only, and a daily
    # request must never silently carry it.
    completed_windows = mo.ui.switch(
        label="Completed windows only",
        disabled=explore_view.value in ("daily_values", "segments"),
    )
    return (completed_windows,)


@app.cell(hide_code=True)
def _(completed_windows, explore_view, mo, segment_choice, snapshot):
    _controls = mo.vstack(
        [
            explore_view,
            mo.hstack(
                [segment_choice]
                if explore_view.value == "segments" and snapshot.breakouts
                else [completed_windows]
                if explore_view.value in ("cumulative_lift", "cumulative_values")
                else [],
                justify="start",
                wrap=True,
            ),
        ]
    )
    mo.Html(f'<div class="inc-dashboard-root inc-dashboard-controls">{_controls.text}</div>')
    return


@app.cell
def _(
    analysis,
    completed_windows,
    explore_view,
    load_explore,
    segment_choice,
    snapshot,
):
    explore_data = load_explore(
        analysis,
        snapshot=snapshot,
        metric=None,
        view=explore_view.value,
        completed_windows_only=completed_windows.value,
        breakout=segment_choice.value,
    )
    return (explore_data,)


@app.cell(hide_code=True)
def _(completed_windows, explore_data, explore_view, render_explore, snapshot):
    render_explore(
        snapshot,
        explore_data,
        metric=None,
        view=explore_view.value,
        completed_windows_only=completed_windows.value,
    )
    return


@app.cell
def _(mo, snapshot):
    metric_choice = mo.ui.dropdown(
        options=[metric.name for metric in snapshot.metrics],
        value=snapshot.primary_metric,
        label="Metric details",
    )
    mo.Html(f'<div class="inc-dashboard-root inc-dashboard-controls">{metric_choice.text}</div>')
    return (metric_choice,)


@app.cell(hide_code=True)
def _(metric_choice, render_metric_details, snapshot):
    render_metric_details(snapshot, metric=metric_choice.value)
    return


@app.cell
def _(group_data_csv, metric_choice, mo, snapshot):
    mo.download(
        data=group_data_csv(snapshot, metric=metric_choice.value),
        filename=f"{snapshot.experiment_name}-{metric_choice.value}-groups.csv",
        mimetype="text/csv",
        label="Download data by group",
    )
    return


@app.cell(hide_code=True)
def _(render_details, snapshot):
    render_details(snapshot)
    return


@app.cell
def _(mo, readout_csv, snapshot):
    mo.download(
        data=readout_csv(snapshot),
        filename=f"{snapshot.experiment_name}-readout.csv",
        mimetype="text/csv",
        label="Download headline readout",
    )
    return


if __name__ == "__main__":
    app.run()
