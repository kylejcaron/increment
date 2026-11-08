# /// script
# [tool.marimo.display]
# theme = "light"
# ///
import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full", css_file="dashboard.css")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from pathlib import Path

    import _dashboard_demo as demo
    import ibis

    import increment
    from increment.dashboard import DashboardConfig, prepare_dashboard, render_dashboard
    from increment.workflows import experiment_definition

    return (
        DashboardConfig,
        Path,
        demo,
        experiment_definition,
        ibis,
        increment,
        prepare_dashboard,
        render_dashboard,
    )


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
def _(demo, mo):
    experiment_name = mo.ui.dropdown(
        options=demo.EXPERIMENTS, value=demo.EXPERIMENTS[0], label="Experiment"
    )
    experiment_name
    return (experiment_name,)


@app.cell
def _(con, demo, demo_manifest, experiment_definition, experiment_name, increment):
    experiment, reader = increment.from_definitions(
        demo.DEFINITIONS, con, experiment_name=experiment_name.value
    )
    if experiment_definition(experiment, reader).trigger is not None:
        # A triggered population is read up to the snapshot's event-time cutoff.
        experiment, reader = increment.from_definitions(
            demo.DEFINITIONS,
            con,
            experiment_name=experiment_name.value,
            source_snapshot_evidence=demo.snapshot_evidence(demo_manifest),
        )
    return experiment, reader


@app.cell
def _(DashboardConfig, demo, demo_manifest, experiment, experiment_definition, reader):
    definition = experiment_definition(experiment, reader)
    config = DashboardConfig(
        expected_allocation=definition.allocation,
        title=definition.name.replace("_", " ").capitalize(),
        source_label="Synthetic snapshot",
        metric_units={
            "revenue_per_user": "USD",
            "average_order_value": "USD",
            "checkout_latency_ms": "ms",
        },
        provenance=demo.demo_provenance(demo_manifest),
    )
    return (config,)


@app.cell
def _(config, experiment, prepare_dashboard, reader):
    snapshot = prepare_dashboard(experiment, reader, config=config)
    return (snapshot,)


@app.cell(hide_code=True)
def _(experiment, render_dashboard, snapshot):
    render_dashboard(experiment, snapshot=snapshot)
    return


if __name__ == "__main__":
    app.run()
