import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell
def _():
    import datetime as dt
    import tempfile
    from pathlib import Path

    import ibis
    import numpy as np
    import pyarrow as pa

    from increment import Analysis, SourceSnapshotEvidence
    from increment.dashboard import DashboardConfig, prepare_dashboard, render_dashboard

    return (
        Analysis,
        DashboardConfig,
        Path,
        SourceSnapshotEvidence,
        dt,
        ibis,
        np,
        pa,
        prepare_dashboard,
        render_dashboard,
        tempfile,
    )


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        """
        This reproducible fixture separates assignment from trigger eligibility. The dashboard
        defaults to the triggered cohort and switches the headline, Readout, inspector, Health,
        allocation history, group evidence, and daily/as-of Explore series together. Triggered-only
        inference is appropriate only when triggering is unaffected by treatment; a refusal keeps
        the triggered series unavailable and offers the assignment-level view as a separate route.
        """
    )
    return


@app.cell
def _(Path, tempfile):
    directory = Path(tempfile.mkdtemp(prefix="increment-triggered-dashboard-"))
    definitions = directory / "definitions.yaml"
    definitions.write_text(
        """dialect: duckdb
fact_sources:
  - name: events
    sql: SELECT * FROM events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: revenue
        column: value
      - name: enrolled
        column: null
      - name: saw_surface
        column: null
exposures:
  - name: assignment
    fact: enrolled
  - name: saw_surface
    fact: saw_surface
metrics:
  - name: revenue
    type: mean
    entity: user_id
    fact: revenue
    aggregation: sum
    window_days: 7
experiments:
  - name: checkout
    exposure: assignment
    trigger: saw_surface
    unit: user_id
    start: 2024-01-01T00:00:00
    control_group: C
    allocation:
      C: 0.5
      T: 0.5
    allocation_scheme: independent
    plan:
      primary: revenue
      alternative: two-sided
"""
    )
    return definitions, directory


@app.cell
def _(ibis, np, pa):
    rng = np.random.default_rng(37)
    records = {
        "user_id": [],
        "group_id": [],
        "ts": [],
        "event": [],
        "value": [],
        "experiment_id": [],
    }
    assignment = np.datetime64("2024-01-02T00:00:00")
    outcome = np.datetime64("2024-01-08T00:00:00")
    for arm in ("C", "T"):
        for index in range(40):
            user = f"{arm}{index}"
            records["user_id"].append(user)
            records["group_id"].append(arm)
            records["ts"].append(assignment)
            records["event"].append("enrolled")
            records["value"].append(0.0)
            records["experiment_id"].append("checkout")
            triggered = index % 2 == 0
            if triggered:
                records["user_id"].append(user)
                records["group_id"].append(arm)
                records["ts"].append(assignment)
                records["event"].append("saw_surface")
                records["value"].append(0.0)
                records["experiment_id"].append("checkout")
            baseline = float(rng.lognormal(0.0, 0.35))
            value = baseline * (1.7 if arm == "T" and triggered else 1.0)
            records["user_id"].append(user)
            records["group_id"].append(arm)
            records["ts"].append(outcome)
            records["event"].append("revenue")
            records["value"].append(value)
            records["experiment_id"].append("checkout")
    con = ibis.duckdb.connect()
    con.create_table("events", obj=pa.table(records))
    return con


@app.cell
def _(Analysis, SourceSnapshotEvidence, con, definitions, dt):
    analysis = Analysis.from_definitions(
        "checkout",
        definitions,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            observation_cutoff_ts=dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
            complete_through_by_feed={"events": dt.datetime(2024, 1, 16, tzinfo=dt.UTC)},
        ),
    )
    return (analysis,)


@app.cell
def _(DashboardConfig, analysis, prepare_dashboard):
    config = DashboardConfig(expected_allocation={"C": 0.5, "T": 0.5}, title="Triggered checkout")
    snapshot = prepare_dashboard(analysis, config=config)
    return (snapshot,)


@app.cell(hide_code=True)
def _(analysis, render_dashboard, snapshot):
    render_dashboard(analysis, snapshot=snapshot)
    return


if __name__ == "__main__":
    app.run()
