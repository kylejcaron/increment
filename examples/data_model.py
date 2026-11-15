import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import datetime as dt
    from pathlib import Path

    import ibis
    import marimo as mo
    import pandas as pd

    import increment
    from increment.query import build_panel_for_metric, summary_sql
    from increment.tables import estimates_to_readout, readout_table
    from increment.workflows import (
        breakout_summaries,
        experiment_definition,
        export_moments,
        metrics,
    )

    DEFINITIONS = "examples/realistic_demo/definitions/"
    EXPERIMENT = "checkout_redesign"
    WAREHOUSE = Path("examples/realistic_demo/warehouse")
    return (
        DEFINITIONS,
        EXPERIMENT,
        Path,
        WAREHOUSE,
        breakout_summaries,
        build_panel_for_metric,
        dt,
        estimates_to_readout,
        experiment_definition,
        export_moments,
        ibis,
        increment,
        metrics,
        mo,
        pd,
        readout_table,
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
    # What does the data model look like? :

    In this example, we look at a realistic checkout redesign A/B test and observe the internal data transforms that exist in getting data out of the warehouse and into an analysis.
    """)
    return


@app.cell
def _(ibis, mo):
    con = ibis.duckdb.connect()
    mo.show_code(position="above")
    return (con,)


@app.cell(hide_code=True)
def _(WAREHOUSE):
    # The warehouse is generated, never committed (see .gitignore).
    # Regenerate when missing or produced by older generator semantics.
    import importlib.util
    import json

    _spec = importlib.util.spec_from_file_location(
        "_demo_generate", "examples/realistic_demo/generate.py"
    )
    assert _spec is not None and _spec.loader is not None
    _generate_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_generate_mod)

    _manifest_path = WAREHOUSE / "manifest.json"
    _manifest = json.loads(_manifest_path.read_text()) if _manifest_path.is_file() else None
    if _manifest is None or _manifest.get("generator_version") != _generate_mod.GENERATOR_VERSION:
        WAREHOUSE.mkdir(parents=True, exist_ok=True)
        _row_counts = _generate_mod.generate(
            WAREHOUSE, partitions=2, users_per_partition=2000, seed=42
        )
        _manifest_path.write_text(
            json.dumps(
                _generate_mod.warehouse_manifest(
                    partitions=2,
                    users_per_partition=2000,
                    seed=42,
                    row_counts=_row_counts,
                ),
                indent=2,
            )
            + "\n"
        )

    # Explicit sentinel so cells reading the warehouse depend on this cell in
    # marimo's DAG, not on file position - without it, running the builder
    # cell alone on a warehouse-less checkout could race ahead of generation.
    warehouse_ready = WAREHOUSE
    return (warehouse_ready,)


@app.cell
def _(
    DEFINITIONS,
    EXPERIMENT,
    con,
    experiment_definition,
    increment,
    mo,
    warehouse_ready,
):
    assert (warehouse_ready / "manifest.json").exists(), (
        "warehouse missing - run the generation cell above first"
    )
    experiment, reader = increment.from_definitions(
        DEFINITIONS, con, experiment_name=EXPERIMENT, store="always"
    )
    definition = experiment_definition(experiment, reader)
    mo.show_code(position="above")
    return definition, experiment, reader


@app.cell(hide_code=True)
def _(definition, mo):
    roles = definition.plan.role_names()
    primary_names = [name for name, role in roles.items() if role == "primary"]
    secondary_count = sum(1 for role in roles.values() if role == "secondary")
    guardrail_names = set(definition.guardrail_names)
    mo.vstack(
        [
            mo.md(f"## Experiment: {definition.name.replace('_', ' ').title()}"),
            mo.hstack(
                [
                    mo.stat(
                        value=f"{(definition.end - definition.start).days}d",
                        label="Duration",
                        caption=f"{definition.start:%b %d} → {definition.end:%b %d, %Y}",
                        bordered=True,
                    ),
                    mo.stat(
                        value=definition.unit,
                        label="Randomization unit",
                        caption=f"exposure: {definition.exposure}",
                        bordered=True,
                    ),
                    mo.stat(
                        value=primary_names[0] if primary_names else "(none declared)",
                        label="Primary Metric",
                        caption=f"""
                {secondary_count} secondary metrics,
                {len(guardrail_names)} guardrails
                """,
                        bordered=True,
                    ),
                ],
                widths="equal",
                gap=1,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    <br>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Warehouse map

    Seven tables: one experiment-metadata dimension, two user dimensions
    (one static, one slowly-changing), three fact tables, and a raw event
    stream. The two user dimensions are what the definitions declare as
    `dim_sources:` further down; the engine joins them into the fact
    sources that ask for them.

    Columns and types below are read LIVE off the Parquet files by
    DuckDB's `DESCRIBE` - not hand-typed - so this diagram cannot drift
    from the real schema.
    """)
    return


@app.cell(hide_code=True)
def wh_map(con, mo, warehouse_ready):
    # Column names and types come straight from DuckDB: no table->file map to keep
    # in step with renames, and no pyarrow->mermaid type table. `part` is the Hive
    # partition key - an artifact of the layout, not a column of the fact.
    _PREFIX_RANK = ("dim_", "snap_", "fact_", "")  # dims, history, facts, event stream

    def _describe(table):
        rows = con.raw_sql(
            f"DESCRIBE SELECT * FROM read_parquet('{warehouse_ready / table}/**/*.parquet')"
        ).fetchall()
        return {column: dtype for column, dtype, *_ in rows if column != "part"}

    table_schemas = {
        t: _describe(t)
        for t in sorted(
            (p.name for p in warehouse_ready.iterdir() if p.is_dir()),
            key=lambda t: (min(i for i, p in enumerate(_PREFIX_RANK) if t.startswith(p)), t),
        )
    }

    # Types and columns come from DuckDB; PK/FK markers and crow's-foot lines
    # remain hand-authored because the warehouse does not encode those relations.
    _PRIMARY_KEYS = {
        "dim_experiment": {"experiment_id"},
        "dim_user": {"user_id"},
        "fact_orders": {"order_id"},
        "events": {"event_id"},
    }
    _FOREIGN_KEYS = {
        "snap_user_plan": {"user_id"},
        "fact_assignment": {"user_id", "experiment_id"},
        "fact_exposure": {"user_id", "experiment_id"},
        "fact_orders": {"user_id"},
        "events": {"user_id"},
    }
    _RELATIONSHIPS = [
        "    dim_user ||--o{ snap_user_plan : user_id",
        "    dim_user ||--o{ fact_assignment : user_id",
        "    dim_user ||--o{ fact_exposure : user_id",
        "    dim_user ||--o{ fact_orders : user_id",
        "    dim_user ||--o{ events : user_id",
        "    dim_experiment ||--o{ fact_assignment : experiment_id",
        "    dim_experiment ||--o{ fact_exposure : experiment_id",
    ]

    def _entity_block(name, schema):
        lines = [f"    {name} {{"]
        for column, dtype in schema.items():
            if column in _PRIMARY_KEYS.get(name, ()):
                marker = " PK"
            elif column in _FOREIGN_KEYS.get(name, ()):
                marker = " FK"
            else:
                marker = ""
            lines.append(f"        {dtype.lower()} {column}{marker}")
        lines.append("    }")
        return "\n".join(lines)

    warehouse_erd = "\n".join(
        ["erDiagram", *_RELATIONSHIPS]
        + [_entity_block(name, schema) for name, schema in table_schemas.items()]
    )

    mo.mermaid(warehouse_erd)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    <br>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Headline readout

    The table covers all five declared metrics for the assigned population,
    using fixed-horizon inference.

    Each `run()` reads from a consistent captured source snapshot. The stage
    previews below are separate live queries, not the run's physical execution plan.

    `store="always"` controls direct source workflows. A snapshot-backed `run()`
    does not materialize the spine/stats pair illustrated below.
    """)
    return


@app.cell
def _(experiment, increment, mo, reader):
    estimates = increment.run(experiment, reader).readout()
    mo.show_code(position="above")
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


@app.cell
def _(mo):
    mo.md(r"""
    ---

    # Logical pipeline: warehouse data to conversion estimates

    The diagram follows the conversion metric's data shapes, not physical TEMP
    tables. The spine comes from enrollment; sparse event statistics are built
    separately and then joined to it. Later sections inspect these live relations.
    """)
    return


@app.cell(hide_code=True)
def _(experiment, metrics, reader):
    metric = next(m for m in metrics(experiment, reader) if m.name == "conversion_rate")
    return (metric,)


@app.cell(hide_code=True)
def _(dt, mo, pd):
    # Presentation helpers for the pipeline flow diagram below.
    def _fmt_cell(v, width=13):
        """Format one dataframe value for the mini preview tables."""
        if v is None:
            return "&mdash;"
        if isinstance(v, float):
            s = f"{v:,.2f}"
        elif isinstance(v, (dt.datetime, dt.date)):
            s = str(v)[:10]
        else:
            s = str(v)
        if len(s) > width:
            s = s[: width - 1] + "\u2026"
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _shape(tbl, n_preview=1):
        """(row_count, preview_frame) for either an ibis table or a DataFrame.

        Lets every card below take whichever it has on hand and still read both
        numbers live - nothing in the diagram hardcodes a count that could rot
        if the warehouse is regenerated at a different scale.
        """
        if isinstance(tbl, pd.DataFrame):
            return len(tbl), tbl.head(n_preview)
        return tbl.count().execute(), tbl.head(n_preview).execute()

    def mini_table(df, cols):
        """Tiny HTML preview: header row plus the sample rows in `df`."""
        head = "".join(
            f'<th style="text-align:left;padding:0 10px 2px 0;font-weight:600;'
            f'white-space:nowrap;opacity:.6;">{c}</th>'
            for c in cols
        )
        body = ""
        for _, row in df.iterrows():
            body += (
                "<tr>"
                + "".join(
                    f'<td style="padding:0 10px 0 0;white-space:nowrap;'
                    f'font-variant-numeric:tabular-nums;opacity:.85;">{_fmt_cell(row[c])}</td>'
                    for c in cols
                )
                + "</tr>"
            )
        return (
            '<table style="border-collapse:collapse;font:9.5px/1.5 ui-monospace,'
            f'SFMono-Regular,Menlo,monospace;"><thead><tr>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table>"
        )

    def _pill(text, color):
        return (
            f'<span style="flex:0 0 auto;font:600 8px/1 system-ui;'
            f"letter-spacing:.06em;text-transform:uppercase;color:{color};"
            f"border:1px solid {color};border-radius:6px;padding:2px 4px;"
            f'opacity:.85;">{text}</span>'
        )

    def defs_card(rows, root):
        """The declarative layer: which YAML file declares what.

        Not a data stage - dashed border, no row count, because these are the
        program the pipeline below executes, not a table it produces.
        """
        body = "".join(
            f"""<div style="min-width:0;">
                  <span style="font:9.5px/1.45 ui-monospace,Menlo,monospace;
                               opacity:.8;">{name}</span><br>
                  <span style="font:9px/1.4 system-ui;opacity:.55;">{note}</span>
                </div>"""
            for name, note in rows
        )
        body = f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:2px 14px;">{body}</div>'
        return f"""
        <div style="border:1px dashed rgba(128,128,128,.4);border-radius:9px;
                    padding:6px 10px;background:rgba(128,128,128,.03);">
          <div style="display:flex;align-items:center;gap:7px;margin-bottom:4px;">
            {_pill("declared", "rgb(120,120,130)")}
            <span style="font:600 12px/1.25 system-ui;">definitions</span>
            <span style="font:9px/1.3 ui-monospace,monospace;opacity:.45;">{root}</span>
          </div>
          {body}
        </div>"""

    def stage_row(n, title, tbl, cols, note="", accent=False):
        """One pipeline stage as a strip: number, name, live count, live preview."""
        ink = "rgb(99,102,241)" if accent else "currentColor"
        bg = "rgba(99,102,241,.07)" if accent else "transparent"
        ring = "rgba(99,102,241,.35)" if accent else "rgba(128,128,128,.16)"
        n_rows, preview = _shape(tbl)
        return f"""
        <div style="display:grid;grid-template-columns:150px 96px 1fr;
                    align-items:center;gap:10px;padding:4px 10px;
                    background:{bg};border:1px solid {ring};border-radius:9px;">
          <div style="display:flex;align-items:center;gap:6px;min-width:0;
                      flex-wrap:wrap;">
            <span style="flex:0 0 auto;width:17px;height:17px;border-radius:50%;
                         border:1.5px solid {ink};color:{ink};opacity:.85;
                         font:600 10px/14px system-ui;text-align:center;">{n}</span>
            <span style="font:600 12px/1.25 system-ui;color:{ink};">{title}</span>
          </div>
          <div style="text-align:right;">
            <div style="font:700 15px/1 system-ui;
                        font-variant-numeric:tabular-nums;">{n_rows:,}</div>
            <div style="font:8.5px/1.3 system-ui;opacity:.45;">rows</div>
          </div>
          <div style="min-width:0;overflow-x:auto;">
            {mini_table(preview, cols)}
            <div style="font:9px/1.3 ui-monospace,monospace;opacity:.45;
                        margin-top:2px;">{note}</div>
          </div>
        </div>"""

    def moment_card(title, tbl, builder, note=""):
        """One terminal moment cube in the fan-out band."""
        n_rows, _ = _shape(tbl)
        return f"""
        <div style="flex:1 1 0;min-width:0;border:1px solid rgba(99,102,241,.35);
                    border-radius:9px;padding:6px 9px;
                    background:rgba(99,102,241,.07);">
          <div style="font:600 11.5px/1.25 system-ui;color:rgb(99,102,241);">{title}</div>
          <div style="display:flex;align-items:baseline;gap:4px;margin:2px 0 3px;">
            <span style="font:700 15px/1 system-ui;
                         font-variant-numeric:tabular-nums;">{n_rows:,}</span>
            <span style="font:8.5px/1 system-ui;opacity:.45;">rows</span>
          </div>
          <div style="font:9px/1.35 ui-monospace,Menlo,monospace;opacity:.7;
                      word-break:break-word;">{builder}</div>
          <div style="font:9px/1.35 system-ui;opacity:.45;margin-top:2px;">{note}</div>
        </div>"""

    def fanout(cards):
        """The moment cubes, side by side, off one materialized pair."""
        return '<div style="display:flex;gap:7px;align-items:stretch;">' + "".join(cards) + "</div>"

    def flow_step(label):
        """The transform between two stages: a down arrow plus what runs."""
        return f"""
        <div style="display:flex;align-items:center;gap:8px;padding:1px 0 1px 62px;">
          <span style="font-size:13px;opacity:.3;line-height:1;">&darr;</span>
          <span style="font:9.5px/1.3 ui-monospace,SFMono-Regular,Menlo,monospace;
                       opacity:.6;">{label}</span>
        </div>"""

    def pipeline_flow(parts):
        """Stack stage strips and transform steps into one vertical flow."""
        return mo.Html(
            '<div style="display:flex;flex-direction:column;padding:2px 0 4px;">'
            + "".join(parts)
            + "</div>"
        )

    return defs_card, fanout, flow_step, moment_card, pipeline_flow, stage_row


@app.cell(hide_code=True)
def _(
    breakout_summaries,
    data_as_of,
    definition,
    experiment,
    ibis,
    metric,
    panel,
    reader,
):
    # Match unit_totals' conversion semantics from the public dense panel:
    # bound each event to the metric window, discard immature units, then
    # reduce each unit to a binary any-event outcome.
    bounded_panel = panel
    if metric.window_days is not None:
        window_end = panel.first_exposure_date + ibis.literal(metric.window_days).as_interval("D")
        observable_end = (
            ibis.literal(definition.observation_horizon.date())
            if definition.observation_horizon is not None
            else panel.ds.max()
        )
        if data_as_of is not None:
            observable_end = ibis.least(observable_end, data_as_of.cast("date"))
        inclusive_last_day = window_end - ibis.literal(1).as_interval("D")
        bounded_panel = panel.filter(
            (panel.ds < window_end) & (inclusive_last_day <= observable_end)
        )

    unit_totals_tbl = bounded_panel.group_by(["unit_id", "group_id"]).aggregate(
        y=(bounded_panel.n_events > 0).max().cast("float64")
    )

    moment_tables = breakout_summaries(experiment, reader, metrics=[metric])
    _country_key = next(key for key in moment_tables if key.startswith(f"{metric.name}:country:"))
    breakout_moments_tbl = moment_tables[_country_key]["group_summary"].to_pandas()
    daily_moments_tbl = moment_tables[_country_key]["daily_group_summary"].to_pandas()
    return breakout_moments_tbl, daily_moments_tbl, unit_totals_tbl


@app.cell(hide_code=True)
def _(
    DEFINITIONS,
    breakout_moments_tbl,
    daily_moments_tbl,
    definition,
    defs_card,
    exposures,
    fact_tbl,
    fanout,
    flow_step,
    metric_events,
    moment_card,
    pipeline_flow,
    reader,
    spine,
    stage_row,
    stats,
    summary,
    unit_totals_tbl,
):
    pipeline_flow(
        [
            defs_card(
                [
                    (
                        "experiments.yaml",
                        f"selected: {definition.name} &middot; unit={definition.unit} &middot; "
                        f"{len(definition.breakouts)} breakouts",
                    ),
                    (
                        "metrics.yaml",
                        # Internals peek: definitions metadata has no public
                        # count property; keep this display-only tour read-only.
                        f"{len(reader._binding.definitions.metrics)} metrics &mdash; type, fact, window",
                    ),
                    (
                        "fact_sources.yaml",
                        f"{len(reader._binding.definitions.fact_sources)} fact sources &mdash; "
                        "SQL, entities, facts",
                    ),
                    (
                        "dim_sources.yaml",
                        f"{len(reader._binding.definitions.dim_sources)} dim sources &mdash; "
                        "unit properties, joined on demand",
                    ),
                    (
                        "exposures.yaml",
                        f"{len(reader._binding.definitions.exposures)} exposure &mdash; {definition.exposure}",
                    ),
                ],
                DEFINITIONS,
            ),
            flow_step("the declarations above drive every stage below"),
            stage_row(
                1,
                "exposures",
                exposures,
                ["unit_id", "group_id"],
                "one row per enrolled unit; first qualifying exposure",
            ),
            flow_step("declared metric &rarr; its fact source SQL"),
            stage_row(
                2,
                "fact table",
                fact_tbl,
                ["unit_id", "ts", "total_amount"],
                "fact_orders &#8904; dim_user &#8904; snap_user_plan",
            ),
            flow_step("metric_events &mdash; filter to fact, resolve value"),
            stage_row(
                3,
                "metric events",
                metric_events,
                ["unit_id", "metric", "value"],
                "conversion is occurrence-only, so value = 1.0",
            ),
            flow_step("from enrollment &mdash; dense unit &times; day grid"),
            stage_row(
                "4a",
                "spine",
                spine,
                ["unit_id", "day_offset", "ds"],
                "through the observation horizon; metric windows apply later",
            ),
            flow_step("from post-exposure events &mdash; sparse per-day aggregates"),
            stage_row(
                "4b",
                "stats",
                stats,
                ["unit_id", "ds", "n_events"],
                "one row per unit and observed event day",
            ),
            flow_step(
                "join sparse stats to the spine, zero-fill, then apply window and maturity rules"
            ),
            stage_row(
                5,
                "unit totals",
                unit_totals_tbl,
                ["unit_id", "group_id", "y"],
                "y = did this unit convert?",
            ),
            flow_step("&hellip; reduced to moments three different ways:"),
            fanout(
                [
                    moment_card(
                        "summary",
                        summary,
                        "group_summary(unit_totals)",
                        "selected conversion metric &times; arm",
                    ),
                    moment_card(
                        "segment breakout",
                        breakout_moments_tbl,
                        'group_summary(..., by=["country"])',
                        "per country &times; arm",
                    ),
                    moment_card(
                        "daily breakout",
                        daily_moments_tbl,
                        "daily_group_summary(panel, metric=metric)",
                        "per day &times; country &times; arm",
                    ),
                ]
            ),
            flow_step("centered moments &rarr; conversion lift in Python"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    <br>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Stage 1: enrolled units and their first exposure

    `first_exposures` restricts enrollment to the experiment's dates and keeps
    each unit's first qualifying exposure and arm. Missing unit IDs or timestamps,
    missing arm assignments, and conflicting assignments are excluded.
    """)
    return


@app.cell
def _(spine):
    # One row per enrolled unit: its arm and first exposure, read off the panel spine
    # that build_panel_for_metric returns below.
    exposures = spine.select("unit_id", "experiment_id", "group_id", "first_exposure_ts").distinct()
    exposures.execute().head()
    return (exposures,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    <br>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Stage 2: assemble the fact table based on the declared metrics

    We wanted to analyze `conversion_rate` for the experiment - but how do we calculate it? We start with a yaml definition for the metric
    """)
    return


@app.cell(hide_code=True)
def _(DEFINITIONS, Path, mo):
    _metrics_yaml = (Path(DEFINITIONS) / "metrics.yaml").read_text()
    _name_idx = _metrics_yaml.index("name: conversion_rate")
    _start = _metrics_yaml.rindex("\n  - type:", 0, _name_idx) + 1
    _end = _metrics_yaml.index("\n\n", _start)
    mo.md(f"""
    ```yaml
    {_metrics_yaml[_start:_end]}
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice the line `fact: order`. This declares the conversion_rate metric actually comes from a defined `fact`: an order.

    The next step in the pipeline is to look at the necessary facts to assemble to calculate our metrics, and start chaining those together. For that, we go to our fact sources
    """)
    return


@app.cell(hide_code=True)
def _(DEFINITIONS, Path, mo):

    # helper code to visualize the yaml file
    _yaml_text = (Path(DEFINITIONS) / "fact_sources.yaml").read_text()
    _start = _yaml_text.rindex("\n", 0, _yaml_text.index("name: orders")) + 1
    _end = _yaml_text.index("\n  # events", _start)
    mo.md(f"""
    ```yaml
    {_yaml_text[_start:_end].rstrip()}
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `dims: [users, user_plan]` line is why that SQL is only a scan of
    `fact_orders`: no JOIN to `dim_user`, no plan-history range predicate
    to hand-write. Dimension tables are declared once in
    `dim_sources.yaml` and referenced by name; the engine generates the
    join into every fact source that asks for them.
    """)
    return


@app.cell(hide_code=True)
def _(DEFINITIONS, Path, mo):
    # the dim sources the `dims:` list above refers to, minus the file header
    _dims_yaml = (Path(DEFINITIONS) / "dim_sources.yaml").read_text()
    mo.md(f"""
    ```yaml
    {_dims_yaml[_dims_yaml.index("dim_sources:") :].rstrip()}
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `users` has no `validity`, so it is a plain dim: one row per user,
    equi-joined on `user_id`, and every property it carries is
    `as_of: static`. `user_plan` declares `validity`, so it is a version
    history: the join carries a range predicate on the fact source's own
    timestamp, giving each order the plan that was current when the order
    was placed. Both joins are LEFT, so a fact row with no matching dim
    row keeps NULL properties instead of disappearing.

    Dims carry attributes *of the unit*. Attributes of the event itself
    stay inline on the fact source: `web_events` declares `page` in its
    own `properties:` block, because a page path belongs to the page view,
    not to the user.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The public `increment.query.build_panel_for_metric()` hook returns live
    warehouse expressions for the fact table, metric events, panel, spine and
    stats. These logical relations are independent of source-owned TEMP tables
    that other workflow calls may refresh.

    Use the tabs below to inspect each returned value. Relation previews show at
    most 20 rows, with column types; they are not the complete warehouse relations.
    """)
    return


@app.cell
def _(build_panel_for_metric, experiment, metric, mo, reader):
    panel, spine, stats, metric_events, den_events, fact_tbl, value_col, data_as_of = (
        build_panel_for_metric(experiment, reader, metric)
    )
    mo.show_code(position="above")
    return (
        data_as_of,
        den_events,
        fact_tbl,
        metric_events,
        panel,
        spine,
        stats,
        value_col,
    )


@app.cell(hide_code=True)
def _(mo):
    def preview_pipeline_relation(relation, description):
        """Inspect at most twenty rows without counting the full relation."""
        if relation is None:
            content = mo.md("**Not applicable for this metric.**")
        else:
            content = mo.ui.table(
                relation.limit(20).to_pyarrow(),
                selection=None,
                pagination=False,
                show_column_summaries=False,
                show_data_types=True,
                label="Preview: up to 20 rows",
            )
        return mo.vstack([mo.md(description), content])

    def preview_pipeline_value(value, description, missing_message):
        """Show a resolved scalar rather than its warehouse expression."""
        content = mo.md(missing_message) if value is None else mo.as_html(value)
        return mo.vstack([mo.md(description), content])

    return preview_pipeline_relation, preview_pipeline_value


@app.cell(hide_code=True)
def _(
    data_as_of,
    den_events,
    fact_tbl,
    metric_events,
    mo,
    panel,
    preview_pipeline_relation,
    preview_pipeline_value,
    spine,
    stats,
    value_col,
):
    _previews = {
        "panel": lambda: preview_pipeline_relation(
            panel,
            "Dense unit-day relation: enrolled units joined to daily event statistics.",
        ),
        "spine": lambda: preview_pipeline_relation(
            spine,
            "Enrollment and calendar days, before joining any event values.",
        ),
        "stats": lambda: preview_pipeline_relation(
            stats,
            "Sparse post-exposure event counts and values, grouped by unit and day.",
        ),
        "metric_events": lambda: preview_pipeline_relation(
            metric_events,
            "Events selected for this metric, with their resolved outcome values.",
        ),
        "den_events": lambda: preview_pipeline_relation(
            den_events,
            "Denominator events are used by ratio metrics; this conversion metric has none.",
        ),
        "fact_tbl": lambda: preview_pipeline_relation(
            fact_tbl,
            "The resolved fact source, including its declared dimension joins.",
        ),
        "value_col": lambda: preview_pipeline_value(
            value_col,
            "Resolved value column on the fact source. Conversion events contribute 1.0 instead.",
            "**Not applicable.** No fact value column is required for this metric.",
        ),
        "data_as_of": lambda: preview_pipeline_value(
            data_as_of.execute(),
            "Latest observed event date for this source. The observation horizon can still cap the analysis.",
            "**Unavailable.** No observed event date was returned.",
        ),
    }
    # A live HTTP request can service lazy callbacks; fresh exports have no request.
    _interactive = mo.app_meta().request is not None
    pipeline_tabs = mo.ui.tabs(
        {
            name: mo.lazy(preview, show_loading_indicator=True) if _interactive else preview()
            for name, preview in _previews.items()
        },
        value="panel",
        label="Inspect pipeline outputs",
    )
    pipeline_tabs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    <br>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Stage 3: select this metric's events and resolve their values

    `metric_events` selects the declared `order` fact and applies the metric's
    filters. For this occurrence-only conversion metric, each selected event
    contributes `value = 1.0`.

    Exposure timing and the metric's observation window are enforced later.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    <br>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Stage 4: spine, sparse statistics and the dense panel

    The **spine** has one row per enrolled unit and calendar day, through the
    observation horizon. It does not stop at the selected metric's window.

    **Stats** aggregates strictly post-exposure events per unit and day: count,
    sum, minimum and maximum. Joining these sparse statistics onto the spine
    retains eventless days and fills them with zeros.

    Later reductions apply the metric's observation window and maturity rules.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    <br>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Stage 5: unit outcomes and centered moments

    1. `unit_totals` combines the spine and event statistics, applies the metric
       window and maturity rules, and reduces each eligible unit to one outcome.
       For conversion, `y` records whether the unit had any qualifying event.
    2. `group_summary` aggregates those outcomes per arm into centered moments:
       `n`, `ref_y`, `cy1`, `cy2`, and related fields.

    The basic conversion readout fetches these aggregates for inference instead
    of the whole unit-day panel. This notebook also fetches rows to display stage
    previews; other statistical procedures can request different evidence.
    """)
    return


@app.cell
def _(con, experiment, metric, mo, reader, summary_sql):
    summary = con.sql(summary_sql(experiment, reader)[metric.name])
    mo.show_code(position="above")
    return (summary,)


@app.cell(hide_code=True)
def _(summary):
    summary.execute()[["group_id", "n", "ref_y", "cy1", "cy2"]].round(3)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Quick reference: the moments**

    | column | meaning |
    |--|--|
    | `n` | eligible units in this arm |
    | `ref_y` | centering reference; here, the arm's mean conversion outcome |
    | `cy1` | sum of `(y - ref_y)`; near zero here, but retained for accurate moment merging |
    | `cy2` | sum of `(y - ref_y)^2`; used with `cy1` to recover variance |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    <br>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Stage 6: export moments and replay one metric

    **Export:** `workflows.export_moments()` writes portable moment rows for all
    five declared metrics, together with their stored decision plan, source
    identity and assignment metadata. The `summary` relation above previews only
    `conversion_rate`, not the entire exported payload.

    **Replay:** the code below supplies only the conversion metric to
    `from_moments()`. `run()` uses the stored plan for that selected metric.

    **Check:** the comparison verifies its lift and bounds against the original
    readout. It does not demonstrate replay of all five metrics or the guardrails.

    Registered raw sequential monitoring uses a separate finalized checkpoint;
    see the sequential-inference guide.
    """)
    return


@app.cell(hide_code=True)
def _():
    import tempfile

    # Keep the export available when downstream cells are rerun.
    moments_dir = tempfile.TemporaryDirectory()
    return (moments_dir,)


@app.cell
def _(Path, experiment, export_moments, mo, moments_dir, reader):
    moments_path = Path(moments_dir.name) / "moments.parquet"
    export_moments(experiment, reader, moments_path)
    mo.show_code(position="above")
    return (moments_path,)


@app.cell(hide_code=True)
def _(increment, metric, moments_path):
    import pyarrow.parquet as pq

    exported_moments = pq.read_table(moments_path).to_pylist()
    moments_metric = increment.MetricSpec(
        name=metric.name,
        type="conversion",
        window_days=metric.window_days,
        preferred_direction=metric.preferred_direction,
    )
    return exported_moments, moments_metric


@app.cell
def _(
    definition,
    estimates,
    exported_moments,
    increment,
    metric,
    mo,
    moments_metric,
):
    moments_experiment, moments_reader = increment.from_moments(
        exported_moments,
        metrics=[moments_metric],
        control=definition.control_group,
        experiment_id=definition.name,
    )
    moments_estimate = next(
        e
        for e in increment.run(
            moments_experiment, moments_reader, metrics=[moments_metric]
        ).readout()
        if e.metric == metric.name
    )
    original_estimates = next(e for e in estimates if e.metric == metric.name)
    mo.show_code(position="above")
    return moments_estimate, original_estimates


@app.cell
def _(mo, moments_estimate, original_estimates):
    mo.md(f"""
    | | from exported moments | `run(experiment, reader)` |
    |--|--:|--:|
    | lift | {moments_estimate.lift.value:.6f} | {original_estimates.lift.value:.6f} |
    | lift CI | [{moments_estimate.lift.lb:.4f}, {moments_estimate.lift.ub:.4f} ] | [{original_estimates.lift.lb:.4f}, {original_estimates.lift.ub:.4f} ] |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    <br>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Breakouts: estimates within each segment

    A breakout estimates the effect separately within each segment. It is useful
    for investigating differences, but exploratory comparisons can reflect noise.

    `workflows.breakout_summaries()` returns arm-level and day-level moment
    relations for each metric and declared breakout. Below, `country` makes the
    additional grouping concrete.
    """)
    return


@app.cell
def _(breakout_summaries, experiment, metric, mo, reader):
    breakout_tables = breakout_summaries(experiment, reader, metrics=[metric])
    mo.show_code(position="above")
    return (breakout_tables,)


@app.cell(hide_code=True)
def _(breakout_tables, pd):
    rows = []
    for key, tables in breakout_tables.items():
        _, property_name, source_name = key.split(":", 2)
        grouped = (
            tables["group_summary"]
            .to_pandas()
            .groupby(property_name, dropna=False)["n"]
            .sum()
            .reset_index()
        )
        rows.extend(
            {
                "source": source_name,
                "property": property_name,
                "group": group,
                "count": count,
            }
            for group, count in grouped.itertuples(index=False, name=None)
        )
    pd.DataFrame(rows, columns=["source", "property", "group", "count"])
    return


@app.cell
def _(mo):
    mo.md(r"""
    Both breakout properties come from dim sources, not from a fact
    source's SQL. `country` resolves `as_of: static` (a user's signup
    country never changes); `plan` above resolves `as_of: pre_exposure`
    instead - each unit's value is whatever `snap_user_plan` said was
    current *strictly before* that unit's first exposure, not whatever's
    current when an event is logged. This warehouse deliberately lands every
    plan upgrade after the experiment window, so that resolution always
    lands on `free` here - a data-model detail worth knowing, not
    something we'll build a breakout table out of below (a segment with
    one value can't contrast against anything).
    """)
    return


@app.cell
def _(breakout_tables, metric):
    _country_key = next(key for key in breakout_tables if key.startswith(f"{metric.name}:country:"))
    breakout_moments = breakout_tables[_country_key]["group_summary"].to_pandas()
    breakout_moments[["country", "group_id", "n", "ref_y", "cy1", "cy2"]].round(3)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Each country's conversion lift is estimated from that country's units. The
    table below is a per-segment readout, not the pooled headline estimate
    rearranged for display.
    """)
    return


@app.cell
def _(experiment, increment, mo, reader):
    breakout_scope = increment.View.breakout(())
    breakout_estimates = increment.run(
        experiment.with_views(breakout_scope),
        reader,
        views=(breakout_scope,),
        metrics=["conversion_rate"],
    ).readout(breakout_scope)
    mo.show_code(position="above")
    return (breakout_estimates,)


@app.cell(hide_code=True)
def _(breakout_estimates, estimates_to_readout, readout_table):
    country_breakout_estimates = [
        e for e in breakout_estimates if e.metric == "conversion_rate" and e.dimension == "country"
    ]
    breakout_table = readout_table(
        estimates_to_readout(country_breakout_estimates),
        title="Checkout Redesign: Breakouts",
        subtitle="conversion_rate, by country",
        nest_by="segment",
    )
    breakout_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    <br>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Cumulative lift over time

    `View.asof()` estimates lift from observations available by each calendar-date
    cutoff. The warehouse computes cumulative day-level summaries; Python then
    estimates lift for each retained date. It does not rerun the whole-window
    SQL pipeline from scratch for every day.

    Per-unit windows still apply, and monitoring can include incomplete follow-up.
    This differs from `View.daily()`, which reports separate daily or cohort outcomes.
    """)
    return


@app.cell
def _(experiment, increment, metric, mo, reader):
    asof_lift_scope = increment.View.asof()
    asof_estimates = increment.run(
        experiment.with_views(asof_lift_scope),
        reader,
        views=(asof_lift_scope,),
        metrics=[metric],
    ).readout(asof_lift_scope)
    mo.show_code(position="above")
    return (asof_estimates,)


@app.cell(hide_code=True)
def _(asof_estimates, estimates_to_readout, pd):
    pd.DataFrame(estimates_to_readout(list(asof_estimates)))[
        ["ds", "group_id", "lift", "lower", "higher", "stat_sig"]
    ].sort_values(["group_id", "ds"]).head(8)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The sparkline pairs the whole-window conversion estimate with its cumulative
    as-of series. The series describes how the evidence evolves by calendar date;
    it is not a set of independent daily effects.
    """)
    return


@app.cell
def _(asof_estimates, estimates_to_readout, original_estimates, readout_table):
    trend_table = readout_table(
        estimates_to_readout([original_estimates]),
        title="Checkout Redesign: conversion_rate over time",
        subtitle="Cumulative lift by as-of date",
        trend=list(asof_estimates),
    )
    trend_table
    return


if __name__ == "__main__":
    app.run()
