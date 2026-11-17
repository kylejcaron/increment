# Use the experiment dashboard

<span id="ab-testing-dashboard"></span>

Prepare one experiment, open its four-tab dashboard, and inspect or export the
same captured results. Use this with a two-arm randomized experiment loaded
through `from_definitions`; the dashboard presents your analysis plan, not a
ship/no-ship recommendation.

## 1. Prepare the data

<span id="install"></span>

For this local example, install:

```bash
pip install --pre "increment[dashboard,demo]"
```

<span id="supported-experiments"></span>

The dashboard needs a declared primary metric and explicit expected allocation.
Unequal allocation and renamed arms are supported; never assume an even split.
Fixed-horizon and registered Bernoulli/asymptotic-mean plans are supported.
For sequential plans, capture finalized observations first; an unregistered
`always_valid` declaration is not enough. See
[supported experiments and configuration](../reference/dashboard-output.md#supported-experiments-and-configuration).

This complete synthetic example creates a local warehouse with 20 users per
arm. Purchases occur after enrollment; a later background event establishes
that the one-day measurement window has finished.

```python
import datetime as dt
from pathlib import Path
from tempfile import TemporaryDirectory

import ibis
import pandas as pd
import increment
from increment.dashboard import DashboardConfig, prepare_dashboard, render_dashboard

rows = []
for i in range(40):
    for event in ["page_view", "purchase"]:
        rows.append(
            {
                "user_id": f"u{i}",
                "group_id": "control" if i < 20 else "treatment",
                "experiment_id": "demo",
                "ts": dt.datetime(2025, 1, 1, 12, 1 if event == "purchase" else 0),
                "event": event,
                "revenue": float(1 + i % 5 + (i >= 20)) if event == "purchase" else None,
            }
        )
rows.append(
    {
        "user_id": "background",
        "group_id": None,
        "experiment_id": None,
        "ts": dt.datetime(2025, 1, 3, 12),
        "event": "purchase",
        "revenue": 1.0,
    }
)
con = ibis.duckdb.connect()
con.create_table("demo_events", pd.DataFrame(rows))

with TemporaryDirectory() as definitions:
    Path(definitions, "demo.yaml").write_text("""\
dialect: duckdb
fact_sources:
  - name: demo_events
    sql: SELECT * FROM demo_events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - {name: page_view, column: null}
      - {name: purchase, column: revenue}
exposures:
  - {name: enrolled, fact: page_view}
metrics:
  - {name: revenue, type: mean, entity: user_id, fact: purchase, aggregation: sum, window_days: 1}
experiments:
  - name: demo
    exposure: enrolled
    unit: user_id
    start: 2025-01-01
    end: 2025-01-01
    observation_end: 2025-01-02
    control_group: control
    allocation: {control: 1, treatment: 1}
    plan: {primary: revenue}
""")
    experiment, reader = increment.from_definitions(definitions, con, experiment_name="demo")
    config = DashboardConfig(
        expected_allocation={"control": 1, "treatment": 1},
        source_label="Synthetic example",
        metric_units={"revenue": "USD"},
    )
    with reader:
        snapshot = prepare_dashboard(experiment, reader, config=config)
con.disconnect()
```

Preparation captures headlines, group data, diagnostics, and supported Explore
views from the same pinned data. The snapshot has no open connection; the
example closes its reader and database after preparation.

<span id="bind-your-own-experiment"></span>

For your warehouse, follow [warehouse setup](data-model.md), install its Ibis
driver plus `increment[dashboard,warehouse]`, and pass your bound experiment
and reader to the same `prepare_dashboard` call. Use your declared allocation
and explicit unit labels; currency is not inferred from metric names.
You own the reader and connection: keep them for a deliberate refresh, or
close them when finished.

## 2. Open the dashboard

Run this in a separate marimo notebook cell:

```python
render_dashboard(experiment, snapshot=snapshot)
```

The synthetic Readout contains one revenue comparison. Inspect **Data by group**
to see the observed means, 3 USD for control and 4 USD for treatment: a relative
increase of about 33%. The interval expresses uncertainty; the example is not
evidence about a real checkout.

<span id="run-the-example"></span>

Prefer a ready-made notebook? Open the
[dashboard example](../examples/dashboard.md), or run its
[local launch command](../reference/dashboard-output.md#running-the-bundled-notebook).

<span id="live-notebook-vs-static-export"></span>

Browser controls select prepared data, including in static HTML; they do not
query the warehouse or change the plan. Re-run preparation to refresh data.
Static exports cannot change the **Added metrics** selection; that control
requires a running notebook.

<span id="theme-presets"></span>

Use the masthead toggle for light/dark mode. It preserves data and selections;
printing always uses light mode. To customize fonts, colors, or sizes, see
[theme presets](../reference/dashboard-output.md#theme-presets).

## 3. Read the result

<span id="sections"></span>

| Tab | Use it to |
|---|---|
| **Readout** | Read the declared primary, secondary, and guardrail results |
| **Explore** | Compare segments, added metrics, and changes over time |
| **Health** | Inspect allocation, enrollment, and diagnostic warnings |
| **Report** | Export the full captured readout or print a summary |

For individual notebook sections, see the
[renderer functions](../reference/dashboard-output.md#renderer-functions).

Readout and Report show short warnings for failed checks, missing results, or
unverified assignment. Routine passes and complete-result counts stay in
**Health**, alongside the detailed checks.

<span id="population-selection"></span>
<span id="assigned-and-triggered-populations"></span>

With a declared trigger, the dashboard opens on **Triggered**. **Population**
switches all tabs, inspectors, and downloads together, never pooling populations.
Without a trigger, only Assigned is available. Triggered inference requires
treatment not to affect trigger membership or timing. Triggered balance does
not validate assignment; the assignment check still uses the assigned
population. See [population and check details](../reference/dashboard-output.md#populations-and-assignment-checks).

<span id="read-the-intervals"></span>
<span id="what-the-dashboard-will-not-do"></span>

Read the estimate, its uncertainty, and visible warnings together. **Unavailable**
or `N/A` means no answer, with a reason—not zero or nonsignificant.
**Inconclusive** means the available evidence did not establish the tested claim.
Color also respects the original multiple-testing selection, not just the sign.
Hiding a failed guardrail cannot make the required comparisons complete.

**Guardrail passed** rules out the declared unacceptable loss, not necessarily
a decline. An inconclusive safety test does not establish safety or prove harm;
separate adverse interval evidence remains visible. Open interval ends can be
intentional one-sided bounds, whereas undefined ends carry reasons. Expand
**How to read these intervals** or consult
[interval geometry](../reference/dashboard-output.md#interval-and-confidence-set-geometry).

## 4. Inspect a metric or time series

<span id="observed-data-by-group"></span>

Select a metric's name to inspect its definition, analysis policy, exact bounds,
and observed **Data by group**. The header's enrollment count is not a metric
denominator: inspect eligibility, maturity, and exclusions. Group values precede
adjustment; added exploratory metrics have no group-data panel or download.
See [group-value definitions](../reference/dashboard-output.md#observed-data-by-group).

When a metric has several methods, dates, or segments, choose the result you want;
the inspector keeps that result's evidence rather than substituting a headline.

<span id="explore"></span>
<span id="overview"></span>

Explore starts on **Overview**. **Compare by** adds declared segments beneath
whole-experiment rows. Rows labeled “as in Readout” keep their original analysis.
Other eligible cells share one exploratory multiple-testing family, even when
hidden. Its Benjamini–Hochberg correction assumes independence or positive
dependence, which is not checked here. Re-preparing with more data is a new
look, not continuation of that family. See
[Overview and correction](../reference/dashboard-output.md#overview-and-correction).

<span id="added-exploratory-metrics"></span>

In a running notebook, **+ Add metrics** selects saved metrics already captured
with the headline. This changes display only; added metrics never enter Readout
or Report. In static HTML, their captured selection cannot change. See
[offered metrics and exclusions](../reference/dashboard-output.md#added-exploratory-metrics).

<span id="time-series"></span>

Choose **Time series** to inspect changes over time. **Relative lift** is
cumulative; **Absolute values** shows both arms, daily or cumulative.
**Completed windows only**
excludes partial follow-up in cumulative views; it does not apply to daily
values or delay open-ended metrics. Assignment starts the Assigned clock;
first trigger starts the Triggered clock. Retention's daily dates identify
exposure cohorts, not return days.

Expand **Unavailable** for missing-series reasons and **Latest readings and
qualifications** for exact values. Choosing Assigned can show an available
trajectory but does not replace missing Triggered results. Fixed-horizon plots
are descriptive monitoring, not permission for repeated decisions. Registered
Bernoulli and asymptotic-mean plans keep their respective guarantees; asymptotic
does not mean finite-sample exact. See
[time-series rules](../reference/dashboard-output.md#time-series-and-dates).

<span id="allocation-history"></span>

In Health, expand allocation history. Its pointwise 95% Wilson bands describe
observed shares, not sequential mismatch thresholds or repeated-look guarantees;
read the separate allocation check. See
[allocation-history details](../reference/dashboard-output.md#allocation-history).

<span id="no-breakout-experiments"></span>

No segment selector is expected when no breakouts were declared; this is a
[supported case](../reference/dashboard-output.md#no-breakout-experiments).

## 5. Export the captured result

Open **Report** for the full-readout CSV or **Print / Save PDF**. Report keeps
all declared metrics regardless of Explore selections. Explore offers each
declared metric's group-data CSV. Downloads use the snapshot, not fresh data;
unavailable numeric values remain empty.

Small reports fit one A4 page; larger reports paginate rather than shrinking
textual evidence below the configured font floor. A CSV is a readout, not an
inference checkpoint. For exact fields, spreadsheet safety, pagination, and
programmatic downloads, see [CSV exports](../reference/dashboard-output.md#csv-exports)
and [static HTML and printing](../reference/dashboard-output.md#static-html-and-printing).
