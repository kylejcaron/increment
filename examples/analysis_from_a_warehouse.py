import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def imports():
    from pathlib import Path

    import ibis
    import marimo as mo
    import pyarrow.parquet as pq

    from increment import MetricSpec, Scope, from_definitions, from_moments, run
    from increment.query import summary_sql
    from increment.tables import estimates_to_readout, readout_table
    from increment.workflows import experiment_definition, export_moments, srm

    DEFINITIONS = "examples/realistic_demo/definitions/"
    EXPERIMENT = "checkout_redesign"
    MOMENTS_PARQUET = Path.cwd() / "increment_moments.parquet"
    WAREHOUSE = Path("examples/realistic_demo/warehouse")
    return (
        DEFINITIONS,
        EXPERIMENT,
        MOMENTS_PARQUET,
        MetricSpec,
        Path,
        Scope,
        WAREHOUSE,
        estimates_to_readout,
        experiment_definition,
        export_moments,
        from_definitions,
        from_moments,
        ibis,
        mo,
        pq,
        readout_table,
        run,
        srm,
        summary_sql,
    )


@app.cell(hide_code=True)
def _():
    # Register Markdown list preprocessors below superfences so fenced YAML is
    # protected before marimo's indentation passes inspect list-like lines.
    # Re-registering by name replaces the existing entries and is idempotent.
    import marimo._output.md as _md_internals

    _md, _md_lock = _md_internals._get_markdown()
    _md.preprocessors.register(_md.preprocessors["flexible_indent"], "flexible_indent", 24)
    _md.preprocessors.register(
        _md.preprocessors["breakless_lists_preproc"], "breakless_lists_preproc", 23
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Analyzing an Experiment From a Warehouse

    In this example, we look at a realistic checkout redesign A/B test and walk through measuring it.

    The experiment has one primary metric, two secondary metrics, and two
    guardrails. The data lives in a normalized warehouse, so this starts in
    the same place a production analysis would: with definitions for the
    experiment, metrics, exposures, facts, and dimensions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Looking at the semantic layer

    We use YAML as the source of truth for where data comes from and how the
    experiment should be analyzed. The full set of definitions is below:
    """)
    return


@app.cell
def _(DEFINITIONS, Path, mo):
    def _preview(fname):
        _yaml_text = (Path(DEFINITIONS) / fname).read_text()
        return mo.md(f"```yaml\n{_yaml_text}\n```")

    mo.accordion(
        {
            "Experiments": _preview("experiments.yaml"),
            "Exposures": _preview("exposures.yaml"),
            "Fact Sources": _preview("fact_sources.yaml"),
            "Dim Sources": _preview("dim_sources.yaml"),
            "Metrics": _preview("metrics.yaml"),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Connecting to a warehouse

    DuckDB stands in for a production warehouse here. The semantic layer emits
    SQL through ibis, so the analysis code stays the same when the connection
    points at another supported backend.
    """)
    return


@app.cell
def _(ibis):
    # Instantiate a local warehouse connection.
    con = ibis.duckdb.connect()
    return (con,)


@app.cell(hide_code=True)
def _(WAREHOUSE):
    # The warehouse is generated, never committed. Regenerate it when it is
    # missing or was produced by older generator semantics.
    import importlib.util as _importlib_util
    import json as _json

    _spec = _importlib_util.spec_from_file_location(
        "_demo_generate", "examples/realistic_demo/generate.py"
    )
    _generate_mod = _importlib_util.module_from_spec(_spec)
    _spec.loader.exec_module(_generate_mod)

    _manifest_path = WAREHOUSE / "manifest.json"
    warehouse_manifest = (
        _json.loads(_manifest_path.read_text()) if _manifest_path.is_file() else None
    )
    if (
        warehouse_manifest is None
        or warehouse_manifest.get("generator_version") != _generate_mod.GENERATOR_VERSION
    ):
        WAREHOUSE.mkdir(parents=True, exist_ok=True)
        _row_counts = _generate_mod.generate(
            WAREHOUSE, partitions=2, users_per_partition=2000, seed=42
        )
        warehouse_manifest = _generate_mod.warehouse_manifest(
            partitions=2,
            users_per_partition=2000,
            seed=42,
            row_counts=_row_counts,
        )
        _manifest_path.write_text(_json.dumps(warehouse_manifest, indent=2) + "\n")

    # This sentinel makes the file dependency explicit in marimo's DAG.
    warehouse_ready = WAREHOUSE
    return warehouse_manifest, warehouse_ready


@app.cell(hide_code=True)
def _(mo, warehouse_manifest):
    warehouse_summary = (
        f"The fixture contains {warehouse_manifest['partitions']} parquet "
        f"partitions with {warehouse_manifest['users_per_partition']:,} "
        "assigned users each.\n\n"
        "| Warehouse relation | Rows |\n"
        "|---|---:|\n"
        + "\n".join(
            f"| `{table}` | {count:,} |"
            for table, count in warehouse_manifest["row_counts"].items()
        )
    )
    mo.md(warehouse_summary)
    return


@app.cell
def _(
    DEFINITIONS,
    EXPERIMENT,
    con,
    experiment_definition,
    from_definitions,
    warehouse_ready,
):
    assert (warehouse_ready / "manifest.json").exists(), (
        "warehouse missing; run the generation cell above first"
    )
    experiment, resolver = from_definitions(
        DEFINITIONS, con, experiment_name=EXPERIMENT, store="always"
    )
    definition = experiment_definition(experiment, resolver)
    return definition, experiment, resolver


@app.cell(hide_code=True)
def _(definition, mo):
    roles = definition.plan.role_names()
    primary = next(name for name, role in roles.items() if role == "primary")
    secondaries = [name for name, role in roles.items() if role == "secondary"]
    guardrails = [name for name, role in roles.items() if role == "guardrail"]
    breakouts = [breakout.property for breakout in definition.breakouts]
    _inference = (
        definition.plan.inference.kind
        if definition.plan.inference is not None
        else "method-selected fixed"
    )

    mo.md(
        f"""
        ## Experiment: {definition.name.replace("_", " ").title()}

        {definition.description}

        | | |
        |---|---|
        | Window | {definition.start:%b %d, %Y} to {definition.end:%b %d, %Y} |
        | Unit | `{definition.unit}` |
        | Control | `{definition.control_group}` |
        | Primary | `{primary}` |
        | Secondaries | {", ".join(f"`{name}`" for name in secondaries)} |
        | Guardrails | {", ".join(f"`{name}`" for name in guardrails)} |
        | Breakouts | {", ".join(f"`{name}`" for name in breakouts)} |
        | Inference | `{_inference}` |
        """
    )
    return


@app.cell(hide_code=True)
def _(experiment, mo, resolver, summary_sql):
    conversion_sql = summary_sql(experiment, resolver)["conversion_rate"]
    mo.vstack(
        [
            mo.md(r"""
            ## Inspecting the SQL

            The public SQL preview shows the warehouse reduction before it
            runs. Each metric produces the same small per-arm moments shape
            consumed by the estimator.
            """),
            mo.accordion(
                {"SQL for the conversion_rate summary": mo.md(f"```sql\n{conversion_sql}\n```")}
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(experiment, mo, resolver, srm):
    # The realistic generator uses independent 50/50 assignment; declare the anytime-valid check.
    srm_result = srm(
        experiment, resolver, expected={"control": 0.5, "treatment": 0.5}, inference="always_valid"
    )
    srm_status = "SRM detected" if srm_result.is_srm else "No SRM detected"
    mo.md(
        f"""
        ## Checking the allocation

        The generator assigns users 50/50. Reaching checkout is independent of
        treatment in this fixture, so the enrolled population remains valid
        for a sample-ratio check.

        **{srm_status}.**

        | Arm | Enrolled units |
        |---|---:|
        | control | {srm_result.observed["control"]:,} |
        | treatment | {srm_result.observed["treatment"]:,} |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Running the analysis

    One call runs every metric named in the experiment plan. The plan also
    supplies metric roles, fixed-horizon inference, and the multiplicity policy
    used by downstream views.

    With `store="always"`, each readout builds fresh materialized tables.
    Metrics share those tables within the call; later headline, breakout,
    and monitoring calls rebuild them to pick up warehouse corrections.
    """)
    return


@app.cell
def _(experiment, resolver, run):
    # This readout materializes once for all declared metrics.
    estimates = run(experiment, resolver).readout()
    return (estimates,)


@app.cell
def _(estimates, estimates_to_readout, readout_table):
    headline_table = readout_table(
        estimates_to_readout(estimates),
        title="Checkout Redesign: Headline Readout",
        subtitle="All 5 declared metrics, assigned population, fixed-horizon intervals",
    )
    headline_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exploring the data with breakouts

    We can break the result out by segment or follow it over time. Both views
    are useful, but they also create more chances to read noise as signal. The
    experiment plan supplies the correction for the declared breakout views.
    """)
    return


@app.cell
def _(Scope, estimates_to_readout, experiment, readout_table, resolver, run):
    breakout_scope = Scope(view="breakout", grain="total")
    breakout_estimates = run(experiment.with_views(breakout_scope), resolver).readout(
        breakout_scope
    )
    country_breakout_estimates = [
        estimate
        for estimate in breakout_estimates
        if estimate.metric == "conversion_rate" and estimate.dimension == "country"
    ]

    breakout_table = readout_table(
        estimates_to_readout(country_breakout_estimates),
        title="Checkout Redesign: Breakouts",
        subtitle="conversion_rate, by country",
        nest_by="segment",
    )
    breakout_table
    return


@app.cell
def _(Scope, estimates, estimates_to_readout, experiment, readout_table, resolver, run):
    asof_lift_scope = Scope(view="asof", grain="asof")
    daily_estimates = run(
        experiment.with_views(asof_lift_scope),
        resolver,
        metrics=["conversion_rate"],
    ).readout(asof_lift_scope)

    trend_table = readout_table(
        estimates_to_readout([e for e in estimates if e.metric == "conversion_rate"]),
        title="Checkout Redesign: conversion_rate over time",
        subtitle="Daily lift trajectory",
        trend=list(daily_estimates),
    )
    trend_table
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exporting moments for reuse

    `workflows.export_moments()` writes additive moments, not event rows. Each row
    carries a `moments_format=7` version stamp and the complete compiled
    decision plan - methods, roles, inference, and view policies travel
    with the evidence. A second experiment can load that compact cube and
    reproduce the headline estimates without a warehouse connection;
    `from_moments` validates the embedded plan and refuses legacy or
    partial exports, so a rehydrated readout cannot silently decide under
    a different procedure.
    """)
    return


@app.cell
def _(MOMENTS_PARQUET, experiment, export_moments, pq, resolver):
    export_moments(experiment, resolver, MOMENTS_PARQUET)
    exported_moments = pq.read_table(MOMENTS_PARQUET).to_pylist()
    return (exported_moments,)


@app.cell
def _(MetricSpec, definition, exported_moments, from_moments, run):
    moment_metrics = [
        MetricSpec(
            name="conversion_rate",
            type="conversion",
            window_days=14,
            preferred_direction="increase",
        ),
        MetricSpec(
            name="revenue_per_user",
            window_days=14,
            preferred_direction="increase",
        ),
        MetricSpec(
            name="average_order_value",
            type="ratio",
            numerator="revenue",
            denominator="orders",
            window_days=14,
            preferred_direction="increase",
        ),
        MetricSpec(
            name="d7_retention",
            type="retention",
            threshold_days=(7, 14),
            preferred_direction="increase",
        ),
        MetricSpec(
            name="checkout_latency_ms",
            type="ratio",
            numerator="load_time",
            denominator="events",
            window_days=14,
            preferred_direction="decrease",
        ),
    ]
    _experiment, _resolver = from_moments(
        exported_moments,
        metrics=moment_metrics,
        control=definition.control_group,
        experiment_id=definition.name,
    )
    rehydrated_estimates = run(_experiment, _resolver).readout()
    return (rehydrated_estimates,)


@app.cell
def _(estimates, estimates_to_readout, readout_table, rehydrated_estimates):
    warehouse_lifts = {
        (estimate.metric, estimate.group_id): estimate.lift.value for estimate in estimates
    }
    rehydrated_lifts = {
        (estimate.metric, estimate.group_id): estimate.lift.value
        for estimate in rehydrated_estimates
    }
    assert rehydrated_lifts.keys() == warehouse_lifts.keys()
    max_lift_difference = max(
        abs(rehydrated_lifts[key] - value) for key, value in warehouse_lifts.items()
    )
    assert max_lift_difference < 1e-12

    rehydrated_table = readout_table(
        estimates_to_readout(rehydrated_estimates),
        title="Checkout Redesign: Rehydrated Readout",
        subtitle="The same five metrics, loaded from additive moments",
    )
    rehydrated_table
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Where to go next

    The [data model guide](../guides/data-model/) opens up the warehouse and
    query pipeline. The [dataframe tutorial](analysis_from_a_dataframe.html)
    shows the same builders and `run` when the input is already one row per unit
    or a unit-day panel.
    """)
    return


if __name__ == "__main__":
    app.run()
