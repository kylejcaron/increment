# A/B Testing Dashboard

`increment.dashboard` renders one bound experiment's captured evidence as an
interactive Readout, Explore, Health, and Report. Use it to present a
supported two-arm randomized `Analysis.from_definitions` without changing its
analysis plan; the [example notebook](../examples/ab_testing_dashboard.md)
shows the full flow.

All four tabs share the same evidence. They do not replace the plan or turn
diagnostics into a validity verdict; for population and selection status, see
[Reading results](reading-results.md) and [Multiplicity](multiplicity.md).

## Install

```bash
pip install "increment[dashboard]"
```

The extra adds `marimo`, `anywidget`, `coeftable>=0.13.1`, and `pandas`.
Running the bundled notebook against its synthetic fixture also needs the
local `demo` extra (DuckDB):

```bash
pip install "increment[dashboard,demo]"
```

## Bind your own experiment

Construct your own `Analysis`, supply a `DashboardConfig`, and reuse the same
rendering calls. Changing experiments is a source-binding operation, not a
renderer change; the package never assumes a 50/50 allocation.

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

Keep preparation and rendering in separate notebook cells, and install the
warehouse's Ibis backend separately. The dashboard embeds its theme,
stylesheet, table adapter, and controls, so it needs no extra browser package
or Studio server; keep the example's host stylesheet beside the notebook.

`prepare_dashboard` validates the experiment and config, then captures
allocation, enrollment, headline estimates, group data, and supported Explore
views from one pinned source snapshot into an immutable `DashboardSnapshot`.
It does not rebind your `Analysis`. `render_dashboard` reads captured evidence only:
browser controls select prepared data and never query the warehouse, change
the plan, or recompute confirmatory results. To update data or policy, rebind
and prepare again; you own `analysis` and its connection, so close the old
connection when rebinding to a different experiment.

`prepare_dashboard` is the computation boundary. It validates the experiment and
config, then invokes one source-owned snapshot operation. Allocation, enrollment
history, headline estimates, observed group data, and every supported Explore
choice share its pinned inputs, including declared breakout property sources.
The caller's `Analysis` is not rebound. Captured sequential decisions retain
each displayed metric's checkpoint, including earlier frozen secondaries.
These results form an immutable `DashboardSnapshot`. Preparation computes the
temporal and breakout views as well as the headline.

## Population selection

When an experiment declares a trigger, preparation captures assigned and triggered
whole-window estimates, group evidence, allocation diagnostics, and supported
Explore views together. The dashboard defaults to **Triggered** and its
**Population** control switches every readout, breakout, group-data panel,
allocation history, daily/as-of series, inspector, report, and CSV to the same
population; populations are never pooled. Without a declared trigger, the
dashboard remains assigned-only.

Switching populations retains the selected Readout metric and Explore scope,
metric, and series controls when those selections exist in both views.

The selected population's interpretation is shown with its source identity,
integrity, required-scope completeness and retained family. Switching assigned
and triggered populations changes the selected evidence, not the family that
was retained when each population was analyzed. Filtering a failed guardrail
out of the visible rows does not make the primary's required scope complete.

Summary, Readout, and Report distinguish the analysis-design assignment check
from the dashboard's configured allocation check. The first uses retained
analysis integrity evidence; if the design did not declare an allocation, it
is shown as not run, never as a pass. The second compares the selected
population with `DashboardConfig.expected_allocation` and reports its own
evidence. When that dashboard check detects a mismatch, its source and
population are named before effect summaries.

The integrity code `integrity.allocation_missing_or_nonconstant` refers to
missing or nonconstant declared target allocation weights; it does not mean
that the observed group counts are unavailable. The observed allocation
diagnostic is reported separately.

For the Triggered view, the assigned-population integrity result remains
visible as the assignment check; it is not replaced by triggered-population
integrity metadata. Relative configured allocation weights are displayed as
normalized shares (for example, `{control: 1, treatment: 1}` means 50% / 50%).
The complete dashboard also retains the date window, inference description,
and configured source label in the hero metadata beside the qualifications.

The complete dashboard places the selected qualifications in the hero before
the primary effect and at the start of Report before its effect and results
table. An added metric's inspector uses that metric's captured collection
evidence and retained qualifications, not the confirmatory Readout's rows.
The Health tab keeps the allocation summary and available, warning, and
unavailable checks visible. Checks that do not apply because no trigger is
declared collapse into one summary line; diagnostic codes and context are in
the **Technical details** disclosure.

Assigned allocation checks describe assignment integrity. Triggered allocation
checks describe balance in the triggered cohort and are not assignment-integrity
checks. Triggered-cohort inference is appropriate only if triggering is
unaffected by treatment; filtering to the triggered population does not establish
that condition. A triggered view that the source cannot provide stays unavailable
with its refusal code and reason. For a triggered daily/as-of refusal, choose
**Assigned** to view the assignment-level trajectory; this does not replace or
reinterpret the unavailable triggered series. Under a registered sequential plan
each population's group evidence is decoded from its own retained chain of the
captured checkpoint, so a triggered row carries the triggered chain's prefix and
never borrows the assigned one; a process committed without a triggered chain
shows its triggered cells as unavailable (`triggered_chain_uncommitted`).

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

The standalone `render_header` and `render_metric_details` sections use the same
selected-population default and accept `population="assigned"` or
`population="triggered"` for an explicit view. Metric group evidence is scoped to
that population; assignment counts remain separately labeled.

Readout tables keep the core metric and arm identity, lift, and lift plot, with
failure, family-discovery, or prior-advisory columns when those values exist.
Numerical qualification appears in the metric inspector's interval-reading
disclosure rather than the headline interval.

Explore requests must match the snapshot's experiment and metric
configuration, including arms, windows, breakouts, and policy. Matching only
the name is insufficient.

## Supported experiments

Ordinary, definitions-backed, two-arm randomized experiments
(`Analysis.from_definitions`) with a declared primary metric and
fixed-horizon, registered Bernoulli, or registered asymptotic-mean inference.
Capture finalized sequential observations before preparation; an unregistered
`always_valid` declaration is not supported. Renamed arms, unequal
allocation, any declared primary metric, any number of secondaries and
guardrails, and experiments with no declared breakouts are all ordinary
cases.

Metric roles, decision methods, inference, population, direction, tested
alternative, intervals, and the multiplicity policy are inherited from your
`Analysis`; there is no confidence slider or method selector that silently
changes the analysis contract.

| Config | Rule |
|---|---|
| `DashboardConfig.expected_allocation` (`expected_allocation`) | Required: exactly two distinct, non-empty arm labels with finite positive weights. The definitions-backed SRM check has no safe implicit default; an assumed even split would silently turn an unequal design into a false mismatch. The configured control arm must match the bound experiment; the treatment label is whatever the other declared arm is called. |
| `metric_units` | Presentation labels, not statistical settings. Conversion and retention values use percentages; other metrics use the explicit unit or the generic label `value`. Currency is never inferred from a metric's name. Unknown metric names and empty labels are refused before source reads. |

**When it refuses.** An unusable configuration raises `InvalidRequestError`
(`dashboard.invalid_config`) before any data loads. An experiment whose
`run()` returns contrast results rather than lift estimates (for example a
switchback design), or that declares an encouragement or observational
`design`, raises `CapabilityError` (`dashboard.unsupported_experiment`) before
any read, rather than being shown as a randomized result.

## Assigned and triggered populations

With a declared trigger, preparation captures both populations' whole-window
estimates, group evidence, allocation diagnostics, and supported Explore
views. The dashboard opens on **Triggered**; the **Population** control
switches every readout, breakout, group-data panel, allocation history,
daily/as-of series, inspector, report, and CSV together, and never pools
populations. Without a trigger, the dashboard is assigned-only.

!!! warning "Triggered balance is not a randomization check"
    Assigned allocation checks evaluate assignment integrity. Triggered
    checks describe triggered-cohort balance and cannot validate
    randomization: triggered inference is appropriate only when treatment
    does not affect trigger membership or timing, and balanced rates cannot
    prove that. Use
    [`increment.workflows.health_check()`](../api.md#diagnostics-and-robustness)
    as a sanity check, not a causal-validity badge. An incomplete feed or
    immature window is not a null estimate.

??? info "Population behavior in detail"
    Selections persist across populations when available in both. Health
    shows available, warning, and unavailable checks; inapplicable checks
    collapse into one line when no trigger is declared, with their codes and
    context in **Technical details**. A source-unavailable triggered view
    retains its refusal code and reason. For an unavailable triggered
    daily/as-of series, **Assigned** can show the assignment-level trajectory
    but does not replace the missing triggered series. Registered sequential
    results use each population's own retained checkpoint chain; without a
    committed triggered chain, its cells remain unavailable as
    `triggered_chain_uncommitted`.

    Standalone `render_header` and `render_metric_details` use the selected
    population by default and accept `population="assigned"` or
    `population="triggered"`. Group evidence is population-scoped;
    assignment counts remain separate. Readout keeps metric and arm
    identity, lift, and plot, plus available failure, family-discovery, and
    prior-advisory columns. Numerical qualifications appear in the
    inspector's interval disclosure.

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

Dimensions and screen text sizes use CSS pixels; colors must be hexadecimal;
font stacks cannot contain CSS rules. Invalid categories or values raise
`InvalidRequestError` (`dashboard.invalid_theme`) before source reads.
Replacing a preset never mutates `MIDNIGHT` or changes captured evidence.
The default pairs **Midnight · Daylight** (light) with **Midnight** (dark);
custom presets supply their own paired palettes, and exported dashboards
carry their configuration and need no separate theme files. For standalone
sections, emit `dashboard_styles(theme=config.theme)` once alongside the
section outputs; its tokens are scoped to `.inc-dashboard-root` and do not
restyle the notebook canvas.

## Sections

| Call | Renders |
|---|---|
| `render_dashboard(analysis, snapshot=...)` | The complete four-tab dashboard as a marimo anywidget: native plots, scoped controls, metric inspectors, health, provenance, and a Report with the full captured metric family. Its Explore **Added metrics** control re-renders the same snapshot in a running notebook; it never reads the warehouse. |
| `dashboard_styles(theme=MIDNIGHT)` | The configured theme and scoped stylesheet as one style block, read from packaged resources. |
| `render_header(snapshot)` | Experiment identity, window, population, inference, arms, and three summary cards: enrolled units, observed arm split, and the primary lift with its bracketed interval and direction-aware significance status. |
| `render_health(snapshot)` | Allocation bars with target markers, the SRM verdict, and visible assignment warnings and result caveats. Allocation over time expands to a CoefTable with one row per variant, cumulative enrolled share, and the check statistics. |
| `render_results(snapshot)` | The whole declared family through CoefTable, grouped by role, with forest plots and an explicit adverse-guardrail callout. Redundant confidence, significance, and confidence-set columns are omitted; levels and interpretation remain in notes and CSV. |
| `render_metric_details(snapshot, metric=...)` | Lift, interval, and direction chips; policy and evidence-geometry details; for the experiment's own metrics, observed data by group with eligibility, exclusions, counts, units, and provenance. An added metric shows its Overview result instead. |
| `load_explore(analysis, snapshot=..., metric=..., view=..., completed_windows_only=..., breakout=...)` | One advanced view captured during preparation. `metric=None` loads every declared metric together; a declared or added metric name selects its own captured series. Validates the experiment binding and options without warehouse queries or materialization; coded refusals stay specific to each state. |
| `render_explore(snapshot, data, metric=..., view=..., completed_windows_only=...)` | One Explore section: a combined cumulative-lift table or separate absolute daily/cumulative tables per metric, the actual date basis and range, and visible unavailable-point reasons. Applicable headline evidence geometry is labeled separately from the series. Pass the same metric selection used to load the data. |
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
samples and solver references. Scoped readouts also carry the source/population,
integrity status and context, required-cell availability, filtered/hidden
identities and reasons, retained family, and retained advisory codes and
explanations. Nested evidence is canonical JSON, not Python representations;
numeric nulls remain empty numeric cells. Filtering a row does not discard the
collection's required scope: visible rows report the view as partial and keep
the original family. A CSV is a row transport, so filtering every row leaves
no row to carry metadata; keep the scoped collection or its saved result when
an empty view must retain its full scope. This CSV is a readout, not a portable
inference checkpoint.

`ExploreView` is the literal type `"cumulative_lift" | "daily_values" |
"cumulative_values" | "segments"`. An unknown metric or view, a daily-values
or segments request carrying the cumulative-only `completed_windows_only`
gate, or an undeclared breakout selection is refused with
`InvalidRequestError` (`dashboard.invalid_view`); the refusal context names
the rejected metric, view, or breakout.

## Read the intervals

Headline and metric-detail intervals keep their declared shape. A one-sided
bound shows as `(−∞, upper]` or `[lower, +∞)`; a declared open endpoint is
not missing data. Relative effects use percentages and absolute effects use
outcome units. An undefined endpoint is shown with its reason and any finite
opposite bound, never as infinity, and its numeric interval and forest bar
are omitted rather than drawing a false open interval; an available point is
still shown. Unavailable inference is never labeled nonsignificant. **How to
read these intervals** explains the captured test direction, exact binary
counts, and confidence-set shape without changing the inference.

!!! warning "A guardrail that did not reject is not safe"
    `stat_sig=False` on a guardrail means its declared test did not reject;
    it does not mean "no harm". An interval entirely on the adverse side
    stays visibly unfavorable even when `stat_sig=False`: Readout and Report
    call out wholly adverse decision intervals, including one-sided
    guardrails whose declared null was not rejected; `render_results` makes
    this explicit rather than relying on someone reading the table correctly.

??? info "Statistical details: interval display rules"
    An ordinary, non-FCR point-backed Fieller result retains its central
    display interval: for a 5% one-sided test, that interval has 90%
    coverage. The interval disclosure and Report retain the 95% directional
    set explicitly. FCR-selected directional rows retain their re-estimated
    open display interval instead.

    Binomial, Fieller, and winsorized confidence sets retain their geometry
    even without a finite point estimate; **How to read these intervals**
    retains numeric set endpoints. **Methods & assumptions** and Report notes
    disclose each row's tested-null verdict separately from family
    selection, qualified by method, role, and arm when needed. Disconnected
    sets remain unions of intervals and retain their excluded gap; their
    forest bars are omitted rather than filling the gap. Whole-line sets
    retain both infinite ends, and a wide but finite interval is not
    reclassified by a size threshold. Confidence levels use up to two decimal
    places: an adjusted 98.333…% level displays as 98.33%. A usable
    absolute-null decision remains visible when only the relative confidence
    set is unavailable; the recorded reason and any available absolute bounds
    appear separately in metric units (percentage points for rates).
    Multiple applicable explanations remain visible.

    In the shipped checkout demo, D7 retention is an increasing one-sided
    guardrail: its test constrains the lower relative-lift bound and
    intentionally leaves the upper direction unconstrained. This is different
    from a zero-control-count result; inspect **Data by group** for the
    observed rates, counts, and maturity. Observed retention rates remain
    between 0% and 100%; relative lift is a different quantity. A binary
    decreasing test also retains the physical −100% relative-lift floor
    rather than inventing an open lower endpoint.

    Explore's interval notes distinguish captured headline evidence from the
    selected trajectory rather than claiming a new inference for each daily
    value. CSV downloads neutralize formula-leading string cells with an
    apostrophe; numeric cells remain numeric and unavailable numbers remain
    empty. Confidence-set cells contain unrounded JSON metadata, omitting
    retained samples and solver references. The CSV is a readout, not a
    portable inference checkpoint.

## Explore

### Overview

Explore opens on **Overview**: one CoefTable of every metric's relative lift.
With **Compare by** set to a declared breakout, each metric's segments nest
beneath its whole-experiment row. Selecting a metric opens its **Time
series**.

The experiment's own whole-experiment rows are the Readout's rows, labelled
"as in Readout", with their declared roles and corrections; they are never
re-corrected here. Every other cell is exploratory: every saved metric the
definitions offer and every segment of every metric and declared breakout.
Only a discovery is coloured as a finding, with its FCR-adjusted interval;
other exploratory intervals are unadjusted and drawn neutral even when they
exclude zero. A cell the family cannot take (for example a quantile metric or
an informative prior) is left out with its reason in **Overview notes**, shown
or not; it is never marked as a discovery and never blocks the other cells.

!!! warning "Exploratory discoveries carry an unchecked assumption"
    The exploratory cells form one Benjamini-Hochberg family at the plan's
    `q`. BH controls the false discovery rate under independence or
    positive dependence; these cells share units and the control arm, and
    [that condition is not checked](../limitations.md#fixed-horizon-fdr-control-assumes-a-dependence-condition-that-is-not-checked).
    Changing the selection reads nothing; only re-running preparation reads
    the warehouse, and a read on more data is a new look, not a
    continuation of this family.

??? info "Statistical details: the exploratory family"
    The family is fixed for the snapshot, so switching **Compare by** never
    changes a cell. Eligible cells count whether or not their metric is
    shown, so choosing what to display, before or after seeing results, never
    changes the correction. Under a registered sequential plan the added
    metrics and uncorrected segments are refused with their coded reasons.

### Added exploratory metrics

`DashboardConfig(exploratory_metrics=(...))` names metrics from
`analysis.available_metrics` to show: saved per-unit metrics on the
experiment's unit that the experiment does not declare. Report-only
`total`/`active` metrics and other entities' metrics are not offered, and a
name the definitions do not offer is refused with `dashboard.invalid_config`
before any read. Every offered metric is read in the same pinned snapshot and
its eligible cells count in the exploratory family; the shown ones also
appear in Overview's Exploratory group and in Time series. None enters the
Readout or the Report. `metric_units` may name any offered metric.

In Explore, **Added metrics** lists the selection and **+ Add metrics**
searches the saved definitions. Applying a change re-renders the same
snapshot and reopens Explore; nothing is read again, so the results and the
correction cannot change. A name the snapshot did not offer is refused and
the current page stays. Selecting an added metric's row opens its Overview
result with its family correction; data by group and its download are
captured for the experiment's own metrics only.

??? info "How offered metrics are read"
    Offered metrics are read together where their corrections never span
    metrics: one read per time-series state and scope (five states for the
    whole experiment and for each declared breakout) and one overview read
    for the whole experiment and per breakout, however many metrics are
    offered. Segment views are read one metric at a time, because under a
    Benjamini-Hochberg breakout policy a joint read would correct every
    metric's segments as one family. A read the source refuses is repeated
    one metric at a time, so each metric keeps its own refusal.

### Time series

Time series selects a metric and a declared breakout, then switches between
**Relative lift** (cumulative) and **Absolute values** (control and treatment
arms on daily or cumulative scales). **Completed windows only** applies to
cumulative views, not daily values. The breakout changes both estimates and
uncertainty, not just labels; each declared `(source, property)` breakout is
captured separately, so two sources of the same property never mix and a
refusal from one never hides the other. Unsupported engine requests display
their refusal instead of substituting whole-experiment results.

??? info "Plot conventions"
    Temporal plots use a separate value scale per metric; both arms share a
    scale within each metric, and unrelated metrics keep separate scales and
    formatters. Date-basis disclosures distinguish observation dates from
    retention cohort dates. Missing estimates remain gaps; **Plot notes**
    count unavailable points by reason. One-sided intervals show their finite
    bound without inventing an endpoint. Absolute daily and cumulative plots
    show numeric y-axis ticks: rates and explicit `%` units display
    percentages, `USD` displays currency, `count` retains fractional values,
    and other declared units appear as suffixes. Formatting changes neither
    values nor intervals.

### Allocation history

Health uses first-assignment enrollment counts, independent of outcome
availability and retention maturity. The plot shows each variant's cumulative
share on observed enrollment dates, using the experiment's day boundary, with
a dashed reference line at its configured target allocation and a shaded
pointwise 95% Wilson interval around its observed share.

!!! warning "The shaded bands are descriptive"
    They are not corrected for repeated looks and are not sequential SRM
    thresholds; the separate allocation check supplies that verdict. The
    bands remain nonzero at observed shares of zero or one.

??? info "`snapshot.allocation_history`"
    It stores immutable rows from the allocation history operation
    (`increment.workflows.allocation_history()`). Each row
    carries its `analysis_population` (`assigned` or `triggered`) so
    enrollment and first-trigger cohorts remain distinct; rows are ordered by
    date and arm within population. A capability refusal is retained in
    `snapshot.allocation_history_refusal` and displayed without disabling the
    headline results. A refused allocation check does not query history.
    Cluster-randomized and non-native sources do not supply this unit
    timeline.

## What the dashboard will not do

- Issue a ship/no-ship verdict. Allocation checks, metric evidence, and
  guardrail evidence are rendered as separate things.
- Fill in unavailable values. They render as `N/A` with the reason supplied
  by the result, never as a zero-filled number or an implied passing check.
- Colour by sign alone. The primary headline is amber when the estimate is
  not statistically significant; significant estimates are green or red only
  after comparing the observed side with the metric's declared preferred
  direction. Unavailable or directionless estimates stay neutral; a
  significant directionless result still receives a significance badge.
- Call fixed-horizon monitoring "safe for repeated decisions". A cumulative
  monitoring view is labeled descriptive monitoring. Registered Bernoulli
  inference retains its sequential guarantee; asymptotic-mean monitoring is
  labeled asymptotic and does not claim a finite-sample guarantee.
- Assume every guardrail tests improvement. Each is labeled by its own tested
  tail (`alternative`), read from the original estimate.

## Observed data by group

Each declared metric's disclosure and group-data download use the same
captured rows; added exploratory metrics have neither. They show assigned
and eligible units, the observed arm value, counts and totals where
meaningful, the observation cutoff and window, and exclusion reasons, with
`assigned = eligible + not mature + no observed day + other excluded`. The
header's enrolled count is not a metric denominator. Group data remains
inspectable when the snapshot has no decision row for that metric; this does
not bypass analysis refusals for invalid inputs.

??? info "What the group values are"
    Values precede CUPED, prior shrinkage, and winsorization while retaining
    the same cohort, filters and windows. Ratios are ratios of group
    component totals, not averages of individual ratios. Quantiles use the
    linear sample-quantile definition. A sum of per-unit averages is not an
    event-level revenue total. Converted/retained unit counts are distinct
    from qualifying event counts. Sequential disclosures use the checkpoint
    displayed by that metric, never current warehouse outcomes alongside an
    earlier stopped result; the source kind and prefix identify the
    evidence. Historical exclusions, raw event counts, and pre-transform
    values that were not retained are unavailable with reasons; retained
    transformed inputs are labeled separately.

    In the CSV, unavailable numeric cells stay empty with reasons in JSON;
    finite means can remain available when a display total is outside the
    numeric range. Rows include experiment, preparation timestamp, binding
    fingerprint, and checkpoint prefix where applicable; unit identifiers
    and event rows are never exported. String cells are spreadsheet-safe and
    numeric values are unrounded. Preparing a new snapshot cannot change an
    earlier download.

## No-breakout experiments

`snapshot.breakouts` is a tuple of `(declared_source, dimension)` pairs read
from the bound experiment. When it is empty, the notebook omits the segment
selector, the segments view shows "No declared breakouts", and no breakout
query is issued. This is an ordinary supported case, not a rendering failure;
one declared breakout dimension with a single resolved segment is equally
valid.

## Live notebook vs. static export

Live notebooks and static HTML exports both contain the prepared **Readout**,
**Explore**, **Health**, and **Report** tabs. Metric, declared-breakout,
relative/absolute, daily/cumulative, and maturity controls select embedded
evidence in the browser; no Python kernel is needed after export. A static
export shows its captured added metrics, but changing them needs a running
notebook; the control says so instead of applying the change. This is a
captured analysis, not a live data-refresh surface.

The masthead toggle switches light and dark mode for tables, charts, and
surrounding surfaces without changing the captured analysis or current
selections. It follows the operating-system preference until you choose a
mode, then remembers that choice in browser storage when available; the same
toggle works in static exports. Printing always uses the light palette and
leaves the selected screen mode unchanged.

Readout and Report always retain the complete captured family, independent
of Explore selections. The full-readout CSV download belongs to Report, not
Readout; Explore keeps its metric-specific group-data CSV for the
experiment's own metrics. All downloads use the captured snapshot.

??? info "Layout and print details"
    Explore keeps charts at their native size so axis labels and uncertainty
    bands remain legible; narrow tables scroll horizontally instead of
    shrinking the plots. Both the hero and Report primary card retain the
    captured uncertainty interval; the Report uses smaller, muted interval
    text. Tables omit the repeated **Confidence** column without changing
    statistical metadata or interval levels.

    Report retains every metric result and interval, compact allocation
    health, essential inference notes, and source/capture context.
    **Print / Save PDF** uses one A4 page when the report fits without taking
    prose or numeric evidence below `theme.printing.min_font_size_pt` (8pt
    by default). Larger families flow across pages with repeated table
    headers instead of shrinking the evidence. Over-wide tables split into
    column bands that repeat metric and arm identity; report whitespace is
    bounded for the page. Plot axes retain their native chart styling.
    Printing expands folded groups and restores them afterwards. Detailed
    policies, observed arm data, allocation history, and provenance remain
    in Readout, metric inspection, Explore, and Health rather than filling
    the PDF.

## Run the example

See the [A/B testing dashboard notebook](../examples/ab_testing_dashboard.md)
for the bundled `checkout_redesign` fixture end to end, and the
[API reference](../api.md) for the complete `increment.dashboard` surface.

```bash
uv run --extra dashboard --extra demo marimo edit examples/ab_testing_dashboard.py
```

`marimo run` serves the same notebook read-only, and
`uv run --extra dashboard --extra demo python examples/ab_testing_dashboard.py`
runs it as a script for a quick smoke check.

??? info "Why the bare `uv run examples/ab_testing_dashboard.py` form is not supported"
    Notebook-local `# /// script` metadata keeps marimo's own chrome light;
    the embedded dashboard has its independent light/dark toggle. Without an
    explicit `python`/`marimo` command, `uv run` treats a directly executed
    `.py` file carrying that metadata as an isolated script environment and
    never installs the `dashboard`/`demo` extras at all.
