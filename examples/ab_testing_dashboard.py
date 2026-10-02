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

    from increment import Analysis
    from increment.dashboard import DashboardConfig, prepare_dashboard, render_dashboard

    return Analysis, DashboardConfig, Path, demo, ibis, prepare_dashboard, render_dashboard


@app.cell
def _(Path, demo):
    # Only the example needs the repository root for relative fixture paths.
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
def _(analysis, render_dashboard, snapshot):
    render_dashboard(analysis, snapshot=snapshot)
    return


if __name__ == "__main__":
    app.run()
