# A/B Testing Dashboard

`increment.dashboard` is an optional presentation layer over one bound
experiment. The bundled `examples/ab_testing_dashboard.py` notebook uses
`render_dashboard` for a shared experiment summary and four tabs: **Readout**,
**Explore**, **Health**, and a printable **Report**. Native CoefTable plots
retain the engine's uncertainty; CSV downloads retain the captured readout
and observed group data. Section-level helpers remain available for custom
notebook compositions.

Core Increment never imports this package: estimation, power, and the
dataframe entry points stay free of marimo, CoefTable, and pandas.

## Install

```bash
pip install "increment[dashboard]"
```

The extra adds `marimo`, `anywidget`, `coeftable>=0.13.1`, and `pandas`. Running the
bundled notebook against its synthetic fixture also needs the local `demo`
extra (DuckDB):

```bash
pip install "increment[dashboard,demo]"
```

## Bind your own experiment

Changing experiments is a source-binding operation, not a renderer change:
construct your own `Analysis`, supply a `DashboardConfig`, and reuse the same
rendering calls. The package has no "checkout" conditional and never assumes
a 50/50 allocation.

<!-- skip: next "requires increment[dashboard] and a configured warehouse" -->

```python
import ibis
from increment import Analysis
from increment.dashboard import (
    DashboardConfig,
    prepare_dashboard,
    render_dashboard,
)

con = ibis.snowflake.connect(...)
analysis = Analysis.from_definitions("checkout_redesign", "definitions/", con)
config = DashboardConfig(
    expected_allocation={"control": 0.5, "treatment": 0.5},
    source_label="Production warehouse",
    metric_units={"revenue_per_user": "USD", "checkout_latency_ms": "ms"},
)

snapshot = prepare_dashboard(analysis, config=config)

render_dashboard(analysis, snapshot=snapshot)
```

Put preparation and rendering in separate notebook cells.
Install your warehouse's Ibis backend separately.

The default preset pairs **Midnight · Daylight** with **Midnight**. The
complete dashboard embeds its theme, shared section stylesheet, native-table
adapter, and browser controls; no external browser packages or Studio server
are required. The example's host stylesheet stays beside the notebook.

`prepare_dashboard` is the computation boundary. It validates the experiment and
config, then invokes one source-owned snapshot operation. Allocation, enrollment
history, headline estimates, observed group data, and every supported Explore
choice share its pinned inputs, including declared breakout property sources.
The caller's `Analysis` is not rebound. Captured sequential decisions retain
each displayed metric's checkpoint, including earlier frozen secondaries.
These results form an immutable `DashboardSnapshot`. Preparation computes the
temporal and breakout views as well as the headline.

`render_dashboard` embeds the captured evidence without querying the warehouse.
Its browser controls select prepared evidence; they never alter the analysis
plan or recompute confirmatory results. That includes **Added metrics** in
Explore: every offered metric is read with the headline, so changing which
ones are shown re-renders the same snapshot. Section-level
renderers also accept already-prepared data. Re-executing preparation
intentionally creates a new snapshot that reflects subsequent source changes.
You retain ownership of `analysis` and its connection; the package does not
close or replace them. When deliberately rebinding to a different experiment,
the caller must close the old connection.

Explore requests must match the snapshot's experiment and metric
configuration, including arms, windows, breakouts, and policy. Matching only
the name is insufficient.

## Supported experiments

The dashboard supports ordinary, definitions-backed, two-arm randomized
experiments (`Analysis.from_definitions`) with a declared primary metric and
fixed-horizon, registered Bernoulli, or registered asymptotic-mean inference.
Capture finalized sequential observations before preparation; an unregistered
`always_valid` declaration is not supported. Ordinary cases include:

- Renamed arms (the configured control arm must match the bound experiment;
  the treatment label is whatever the other declared arm is called).
- Unequal allocation — `expected_allocation` is a required mapping because
  the definitions-backed SRM check has no safe implicit default; an assumed
  even split would silently turn an unequal design into a false mismatch.
- A different declared primary metric, and any number of secondaries and
  guardrails.
- Experiments with no declared breakouts.

`DashboardConfig.expected_allocation` must declare exactly two distinct,
non-empty arm labels with finite positive weights; an unusable configuration
is refused with `InvalidRequestError` (`dashboard.invalid_config`) before
any data loads. Metric roles, decision methods, inference, population,
direction, tested alternative, intervals, and the multiplicity policy are
all inherited from your `Analysis` — there is no confidence slider or method
selector that silently changes the analysis contract. An experiment whose
`run()` returns contrast results rather than lift estimates (for example a
switchback design), or that declares an encouragement or observational
`design`, is refused with `CapabilityError` (`dashboard.unsupported_experiment`)
before any read, rather than shown as a randomized result.

`metric_units` contains presentation labels, not statistical settings. Unknown
metric names and empty labels are refused before source reads. Conversion and
retention values use percentages; other metrics use explicit units or the
generic label `value`. Currency is never inferred from a metric's name.

## Theme presets

`DashboardConfig.theme` accepts an immutable `DashboardTheme`. Start from
`MIDNIGHT` and replace only the categories you want to change:

```python
from dataclasses import replace
from increment.dashboard import MIDNIGHT, DashboardConfig

editorial = replace(
    MIDNIGHT,
    name="Editorial",
    light=replace(MIDNIGHT.light, accent="#9b4d18", canvas="#fff7ed"),
    dark=replace(MIDNIGHT.dark, accent="#ffb66b", canvas="#152923"),
    typography=replace(MIDNIGHT.typography, heading_font="Arial, sans-serif"),
    layout=replace(MIDNIGHT.layout, page_padding=28, radius=10),
    charts=replace(MIDNIGHT.charts, forest_width=333),
    printing=replace(MIDNIGHT.printing, min_font_size_pt=9),
)
config = DashboardConfig(
    expected_allocation={"control": 1, "treatment": 1},
    theme=editorial,
)
```

| Category | Controls |
|---|---|
| `light`, `dark` (`DashboardPalette`) | Semantic colors for surfaces, text, arms, and evidence direction. |
| `typography` (`DashboardTypography`) | Body, heading, and monospace font stacks; body, table, estimate, and interval text sizes. |
| `layout` (`DashboardLayout`) | Dashboard and standalone-section widths, page padding, corner radius, and native-table cell padding. |
| `charts` (`DashboardCharts`) | Headline and segment forests, lift trajectories, absolute trajectories, and allocation-chart dimensions. |
| `printing` (`DashboardPrint`) | Report prose and numeric-evidence font floor, in points. |

Dimensions and screen text sizes use CSS pixels. Colors must be hexadecimal;
font stacks cannot contain CSS rules. Invalid categories or values use
`InvalidRequestError` (`dashboard.invalid_theme`) before source reads.
Replacing a preset never mutates `MIDNIGHT` or changes captured evidence.
The same categories drive native plots and the browser shell; exported
dashboards carry their configuration and need no separate theme files.

For standalone sections, emit `dashboard_styles(theme=config.theme)` once
alongside the section outputs. Its tokens are scoped to `.inc-dashboard-root`
and do not restyle the notebook canvas.

## Sections

| Call | Renders |
|---|---|
| `render_dashboard(analysis, snapshot=...)` | The complete interactive four-tab dashboard from captured evidence, as a marimo anywidget; native plots, scoped controls, metric inspectors, health, provenance, and a readable Report with the full captured metric family. Its Explore **Added metrics** control re-renders the same snapshot in a running notebook; it never reads the warehouse. |
| `dashboard_styles(theme=MIDNIGHT)` | The configured theme and scoped stylesheet as one style block. Reads packaged resources, so it needs no repository-relative path. |
| `render_header(snapshot)` | Experiment identity, window, population, inference, arms, and three summary cards: enrolled units, observed arm split, and a compact primary lift with its bracketed interval and direction-aware significance status. |
| `render_health(snapshot)` | Visual allocation bars with target markers, the SRM verdict, and visible assignment warnings and result caveats. Allocation over time and evidence expands to a CoefTable with one row per variant, cumulative enrolled share, and the check statistics. |
| `render_results(snapshot)` | The whole declared family through CoefTable, grouped by role, with forest plots and an explicit adverse-guardrail callout. Redundant confidence, significance, and confidence-set display columns are omitted; levels and interpretation remain in notes and CSV. |
| `render_metric_details(snapshot, metric=...)` | Lift, interval, and direction chips; policy and evidence-geometry details; observed data by group, including eligibility, exclusions, counts, units, and provenance. |
| `load_explore(analysis, snapshot=..., metric=..., view=..., completed_windows_only=..., breakout=...)` | One advanced view captured during preparation. `metric=None` loads every declared metric together; a declared or added metric name selects its own captured series. Original coded refusals remain specific to each state. Validates experiment binding and options, without warehouse queries or materialization. |
| `render_explore(snapshot, data, metric=..., view=..., completed_windows_only=...)` | One Explore section with a combined cumulative-lift table or separate absolute daily/cumulative tables per metric, actual date basis/range, and visible unavailable-point reasons. Applicable headline evidence geometry is labeled separately from the series. Pass the same metric selection used to load data; `metric=None` renders every declared metric together. |
| `render_details(snapshot)` | Collapsible experiment metadata, per-metric policies, provenance, and static-snapshot limitations. |
| `readout_csv(snapshot)` | The current headline readout as UTF-8 CSV bytes, for `mo.download`. |
| `group_data_csv(snapshot, metric=None)` | The observed group-data rows as UTF-8 CSV bytes. Select one declared metric or export all; no warehouse query occurs. |

Headline and metric-detail intervals retain declared one-sided bounds:
`(−∞, upper]` or `[lower, +∞)`. Relative effects use percentages; absolute
effects use outcome units. A declared open endpoint is not missing data.
An ordinary, non-FCR point-backed Fieller result retains its central display
interval: for a 5% one-sided test, that interval has 90% coverage. The interval
disclosure and Report retain the 95% directional set explicitly. FCR-selected
directional rows retain their re-estimated open display interval instead.

Binomial, Fieller, and winsorized confidence sets retain their geometry even
without a finite point estimate. **How to read these intervals** retains numeric
set endpoints. **Methods & assumptions** and Report notes disclose each row's
tested-null verdict separately from family selection, qualified by method,
role, and arm when needed. Disconnected sets remain unions of intervals; their
forest bars are omitted rather than filling the excluded gap. Confidence levels
use up to two decimal places: an adjusted 98.333…% level displays as 98.33%.
An undefined endpoint is shown with its reason and any finite opposite bound,
never as infinity. Its numeric interval and forest bar are omitted rather
than drawing a false open interval; an available point is still shown.
Unavailable inference is not labeled nonsignificant. A usable absolute-null
decision remains visible when only the relative confidence set is unavailable.
Readout and Report both call out wholly adverse decision intervals, including
one-sided guardrails whose declared null was not rejected.

**How to read these intervals** explains captured test direction, exact binary counts,
and confidence-set shape without changing the inference. In the shipped
checkout demo, D7 retention is an increasing one-sided guardrail: its test
constrains the lower relative-lift bound and intentionally leaves the upper
direction unconstrained. This is different from a zero-control-count result;
inspect **Data by group** for the observed rates, counts, and maturity.
Observed retention rates remain between 0% and 100%; relative lift is a
different quantity. A binary decreasing test also retains the physical −100%
relative-lift floor rather than inventing an open lower endpoint.

Multiple applicable explanations remain visible. Disconnected sets retain
their excluded gap, whole-line sets retain both infinite ends, and a wide but
finite interval is not reclassified by a size threshold. When relative
evidence is unavailable, its recorded reason and any available absolute
bounds appear separately in metric units (percentage points for rates).
Explore's interval notes distinguish captured headline evidence from the selected
trajectory, rather than claiming a new inference for each daily value.

CSV downloads neutralize formula-leading string cells with an apostrophe.
Numeric cells remain numeric and unavailable numbers remain empty.
Confidence-set cells contain unrounded JSON metadata, omitting retained
samples and solver references. This CSV is a readout, not a portable inference
checkpoint.

`ExploreView` is the literal type `"cumulative_lift" | "daily_values" |
"cumulative_values" | "segments"`. An unknown metric/view, a daily-values or
segments request carrying the cumulative-only `completed_windows_only` gate, or an
undeclared breakout selection is refused with `InvalidRequestError`
(`dashboard.invalid_view`); the refusal context names the rejected metric, view,
or breakout.

### Overview

Explore opens on **Overview**: one CoefTable of every metric's relative lift.
With **Compare by** set to a declared breakout, each metric's segments nest
beneath its whole-experiment row. Selecting a metric opens its **Time series**.

- The experiment's own metrics on the whole experiment are the Readout's rows,
  labelled "as in Readout", with their declared roles and corrections; they
  are never re-corrected here.
- Every other cell is exploratory: the whole-experiment rows of every saved
  metric the definitions offer, and every segment cell of every metric and
  declared breakout. The eligible cells form one Benjamini-Hochberg family at
  the plan's `q`, fixed for the snapshot, so switching **Compare by** never
  changes a cell. Eligible cells count whether or not their metric is shown,
  so choosing what to display, before or after seeing results, never changes
  the correction. Changing the selection reads nothing; only re-running
  preparation reads the warehouse, and a read on more data is a new look, not
  a continuation of this family. Only a
  discovery is coloured as a finding, with its FCR-adjusted interval; other
  exploratory intervals are unadjusted and drawn neutral even when they
  exclude zero. BH controls the false discovery rate under independence or
  positive dependence; these cells share units and the control arm, and
  [that condition is not checked](../limitations.md#fixed-horizon-fdr-control-assumes-a-dependence-condition-that-is-not-checked).
- A cell the family cannot take (for example a quantile metric or an
  informative prior) is left out with its reason in **Overview notes**, shown
  or not; it is never marked as a discovery and never blocks the other cells.
  Under a registered sequential plan the added metrics and uncorrected
  segments are refused with their coded reasons.

### Added exploratory metrics

`DashboardConfig(exploratory_metrics=(...))` names the metrics from
`analysis.available_metrics` to show: saved per-unit metrics on the
experiment's unit that the experiment does not declare (report-only
`total`/`active` metrics and other entities' metrics are not offered). Every
offered metric is read in the same pinned snapshot and its eligible cells are
counted in the exploratory family; the shown ones also appear in the
Overview's Exploratory group and in Time series. None enters the Readout or the
Report. Offered metrics are read together where their corrections never span
metrics: one read per time-series state and scope (five states for the whole
experiment and for each declared breakout) and one overview read for the whole
experiment and per breakout, however many metrics are offered. Segment views
are read one metric at a time, because under a Benjamini-Hochberg breakout
policy a joint read would correct every metric's segments as one family. A
read the source refuses is repeated one metric at a time, so each metric keeps
its own refusal.
`metric_units` may name any
offered metric. A name the definitions do not offer is refused with
`dashboard.invalid_config` before any read. In Explore, **Added metrics** lists
the selection; **+ Add metrics** searches the saved definitions. Applying a
change re-renders the same snapshot and reopens Explore; nothing is read
again, so the results and the correction cannot change. A name the snapshot
did not offer is refused and the current page stays.

### Time series

Time series selects a metric and a declared breakout, then switches between
**Relative lift** and **Absolute values**. Relative lift is cumulative;
absolute values show the control and treatment arms on daily or cumulative
scales. **Completed windows only** applies to cumulative views, not daily
values. The breakout changes both estimates and uncertainty, not just labels.
Each declared `(source, property)` breakout is captured separately, so two
sources of the same property never mix, and a refusal from one never hides
the other. Unsupported engine requests display their refusal instead of
substituting whole-experiment results.

Temporal plots use a separate value scale per metric. Date-basis disclosures
distinguish observation dates from retention cohort dates. Missing estimates
remain gaps; **Plot notes** count unavailable points by reason. One-sided
intervals show their finite bound without inventing an endpoint.

Absolute daily and cumulative plots show numeric y-axis ticks. Both arms
share a scale within each metric; unrelated metrics keep separate scales
and formatters. Rates and explicit `%` units display percentages, `USD`
displays currency, and `count` retains fractional values. Other declared
units appear as suffixes. Formatting changes neither values nor intervals.

### Allocation history

Health uses first-assignment enrollment counts, independent of outcome
availability and retention maturity. The plot shows each variant's cumulative
share on observed enrollment dates, using the experiment's day boundary.
Each row has a dashed reference line at its configured target allocation
and a shaded pointwise 95% Wilson interval around its observed share.
These intervals remain nonzero at observed shares of zero or one.
They are descriptive, not corrected for repeated looks, and are not
sequential SRM thresholds; the separate allocation check supplies that verdict.

`snapshot.allocation_history` stores immutable rows from
`Analysis.allocation_history()`. A capability refusal is retained in
`snapshot.allocation_history_refusal` and displayed without disabling the
headline results. A refused allocation check does not query history.
Cluster-randomized and non-native sources do not supply this unit timeline.

## Truthful inference, not a verdict

- There is no automatic ship/no-ship verdict. Allocation checks, metric
  evidence, and guardrail evidence are rendered as separate things.
- Unavailable values render as `N/A` with the reason supplied by the result,
  never as a zero-filled number or an implied passing check.
- The primary headline is amber when the estimate is not statistically
  significant. Significant estimates are green or red only after comparing
  the observed side with the metric's declared preferred direction.
  Unavailable or directionless estimates stay neutral; a significant
  directionless result still receives a significance badge.
- `stat_sig=False` on a guardrail means its declared test did
  not reject; it does **not** mean "no harm" or "safe". An interval
  entirely on the adverse side stays visibly unfavorable even when
  `stat_sig=False`; `render_results` calls this out explicitly rather than
  relying on someone reading the table correctly.
- A fixed-horizon cumulative monitoring view is labeled descriptive
  monitoring, never "safe for repeated decisions". Registered Bernoulli
  inference retains its sequential guarantee; asymptotic-mean monitoring is
  labeled asymptotic and does not claim a finite-sample guarantee.
- Guardrail tests are labeled by their own tested tail (`alternative`), read
  from each original estimate — never hardcoded to "all guardrails test
  improvement".

## Observed data by group

Each metric's disclosure and group-data download use the same captured rows.
They show assigned and eligible units, the observed arm value, counts and
totals where meaningful, the observation cutoff and window, and exclusion
reasons. The accounting is
`assigned = eligible + not mature + no observed day + other excluded`.
The header's enrolled count is not a metric denominator.
Available group data remains inspectable when the snapshot has no decision
row for that metric. This does not bypass analysis refusals for invalid inputs.

Values precede CUPED, prior shrinkage, and winsorization while retaining the
same cohort, filters and windows. Ratios are ratios of group component totals,
not averages of individual ratios. Quantiles use the linear sample-quantile
definition. A sum of per-unit averages is not an event-level revenue total.
Converted/retained unit counts are distinct from qualifying event counts.

Sequential disclosures use the checkpoint displayed by that metric, never
current warehouse outcomes alongside an earlier stopped result. The source
kind and prefix identify the evidence. Historical exclusions, raw event
counts, and pre-transform values that were not retained are unavailable with
reasons; retained transformed inputs are labeled separately.

Unavailable numeric CSV cells stay empty, with reasons in JSON. Finite means
can remain available when a display total is outside the numeric range.
Rows include experiment, preparation timestamp, binding fingerprint, and
checkpoint prefix where applicable; unit identifiers and event rows are
never exported. String cells are spreadsheet-safe, and numeric values are
unrounded. Preparing a new snapshot cannot change an earlier download.

## No-breakout experiments

`snapshot.breakouts` is a tuple of `(declared_source, dimension)` pairs read
from the bound experiment. When it is empty, the notebook omits the segment
selector. Selecting the segments view shows "No declared breakouts" and
issues no breakout query. This is an
ordinary supported case, not a rendering failure: a real experiment with one
declared breakout dimension and a single resolved segment is equally valid.

## Live notebook vs. static export

Both live notebooks and static HTML exports contain the prepared **Readout**,
**Explore**, **Health**, and **Report** tabs. Metric, declared-breakout,
relative/absolute, daily/cumulative, and maturity controls select embedded
evidence in the browser; no Python kernel is needed after export. A static
export shows its captured added metrics, but changing them needs a running
notebook; the control says so instead of applying the change.

By default, the dashboard uses **Midnight · Daylight** in light mode and
**Midnight** in dark mode; custom presets supply their own paired palettes.
The masthead toggle changes tables, charts, and surrounding
surfaces without changing the captured analysis or current selections.
It follows the operating-system preference until you choose a mode, then
remembers that choice in browser storage when available. The same toggle
works in static exports. Printing always uses the light palette and leaves
the selected screen mode unchanged.

Explore keeps charts at their native size so axis labels and uncertainty
bands remain legible. Narrow tables scroll horizontally instead of shrinking
the plots.

Readout and Report always retain the complete captured family, independent
of Explore selections. Both the hero and Report primary card retain the
captured uncertainty interval; the Report uses smaller, muted interval text.
Tables omit the repeated **Confidence** column without changing statistical
metadata or interval levels.

Report retains every metric result and interval, compact allocation health,
essential inference notes, and source/capture context. **Print / Save PDF**
uses one A4 page when the report fits without taking prose or numeric
evidence below `theme.printing.min_font_size_pt` (8pt by default). Larger
families flow across pages with repeated table headers instead of shrinking
the evidence. Over-wide tables split into column bands that repeat metric
and arm identity; report whitespace is bounded for the page. Plot axes
retain their native chart styling.

Printing expands folded groups and restores them afterwards. Detailed
policies, observed arm data, allocation history, and provenance remain in
Readout, metric inspection, Explore, and Health rather than filling the PDF.

The full-readout CSV download belongs to Report, not Readout. Explore retains
its metric-specific group-data CSV; all downloads use the captured snapshot.

Preparation can be more expensive than a single-view export because it
computes the declared temporal choices. This is a captured analysis, not a
live data-refresh surface. Rebind and prepare in Python to update the data
or change the analysis policy.

## Run the example

See the [A/B testing dashboard notebook](../examples/ab_testing_dashboard.md)
for the bundled `checkout_redesign` fixture end to end, and the
[API reference](../api.md) for the complete `increment.dashboard` surface.

```bash
uv run --extra dashboard --extra demo marimo edit examples/ab_testing_dashboard.py
```

The equivalent `marimo run` serves the same notebook read-only, and
`uv run --extra dashboard --extra demo python examples/ab_testing_dashboard.py`
runs it as a script for a quick smoke check. Notebook-local `# /// script`
metadata keeps marimo's own chrome light; the embedded dashboard has its
independent light/dark toggle. That metadata is also why the bare
`uv run examples/ab_testing_dashboard.py` form is not supported here:
without an explicit `python`/`marimo` command, `uv run` treats a directly
executed `.py` file carrying that metadata as an isolated script environment
and never installs the `dashboard`/`demo` extras at all.
