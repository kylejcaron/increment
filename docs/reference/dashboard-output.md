# Dashboard output reference

Use this page to look up what the dashboard captures, displays, and exports.
For a complete setup and the reading order, see [Use the experiment dashboard](../guides/dashboard.md).

## Supported experiments and configuration

The dashboard accepts definitions-backed, two-arm randomized experiments bound
with `from_definitions`, with a declared primary metric and fixed-horizon,
registered Bernoulli, or registered asymptotic-mean inference. Capture finalized
sequential observations before preparation; an unregistered `always_valid`
declaration is not supported. Renamed arms, unequal allocation, any declared
primary metric, any number of secondaries and guardrails, and no breakouts are
supported.

Metric roles, decision methods, inference, population, direction, tested
alternative, intervals, and multiple-testing policy come from the experiment.
There is no confidence slider or method selector that changes them.

- **`DashboardConfig.expected_allocation`** is required: exactly two distinct,
  non-empty arm labels with finite positive weights. The control label must
  match the experiment; the other label identifies treatment. Weights are
  normalized, so `{control: 1, treatment: 1}` means 50% / 50%. There is no
  implicit even split: that would misdiagnose an unequal design.
- **`metric_units`** supplies presentation labels, not statistical settings.
  Conversion and retention use percentages. Other metrics use the declared
  unit or `value`; currency is never inferred from a name. Unknown metric
  names and empty labels are refused before reads. Any offered exploratory
  metric may have a unit label.
- **`exploratory_metrics`** selects offered metrics to display; see
  [Added exploratory metrics](#added-exploratory-metrics).
- **`theme`** accepts an immutable `DashboardTheme`; see [Theme presets](#theme-presets).

Invalid configuration raises `InvalidRequestError` (`dashboard.invalid_config`)
before data loads. Contrast-result experiments such as switchbacks, and
experiments declaring encouragement or observational designs, raise
`CapabilityError` (`dashboard.unsupported_experiment`) before reads, rather
than being presented as randomized results.

### Installation

For Snowflake, install `increment[dashboard,warehouse]` and
`ibis-framework[snowflake]`; use your warehouse's Ibis driver for other backends.
The local example uses `increment[dashboard,demo]`, which includes warehouse
support and DuckDB. The dashboard extra supplies `marimo`, `anywidget`,
`coeftable>=0.13.1`, and `pandas`.

Without that extra, `import increment.dashboard` raises
`install.dashboard_extra_missing` before importing dashboard modules. The
exception is also an `ImportError`; its `install` context gives
`pip install "increment[dashboard]"`. Missing warehouse support raises
`install.warehouse_extra_missing` separately when binding.

## Preparation and capture identity

`prepare_dashboard(experiment, reader, config=config)` validates the experiment
and configuration, then invokes one source-owned snapshot operation. Allocation,
enrollment history, headline estimates, observed group data, and supported
Explore choices share pinned inputs, including declared breakout property
sources. Preparation computes temporal and breakout views as well as headlines.
The immutable `DashboardSnapshot` keeps no reader, source, or connection.

Rendering and `load_explore` read captured data only. Browser choices never
query the warehouse, change the plan, or recompute confirmatory results.
Re-running preparation deliberately makes a new snapshot. The caller's reader
is neither rebound nor closed; the caller owns it and the connection and must
close them when binding another experiment.

Explore requests must match the experiment and metric configuration captured
in the snapshot, including arms, windows, breakouts, and policy. A matching
experiment name alone is insufficient. Sequential decisions keep each displayed
metric's checkpoint, including an earlier checkpoint for a stopped secondary.

Pass `interactions=attachment` to add an optional [interaction scan](../api.md#interaction-discovery).
Preparation independently checks that the attachment and the focal dashboard
use the same definition, fixed window, cutoff, and sealed DuckDB data. A
matching experiment name, timestamp, or row count is not enough.

### Statistical cautions

A successful preparation combines repeated ratio cautions into one
`dashboard.statistical_cautions` summary, not one warning per date, segment,
or arm. **Results needing caution** groups distinct affected readings by metric
and explains their consequences. Search or filter its result picker to inspect
the original date, segment, arm, and check values. An early-date caution does not
qualify an unaffected headline. These checks are heuristics, not measured
coverage for the experiment; estimates and intervals are unchanged. Unrelated
warnings remain visible, and Python warning filters still apply, including
warnings promoted to errors.

## Populations and assignment checks

A declared trigger causes preparation to capture assigned and triggered
whole-window estimates, group data, allocation diagnostics, and supported
Explore views together. The initial population is **Triggered**; without a
trigger it is assigned-only. **Population** switches the readout, breakouts,
group data, allocation history, daily/as-of series, inspector, report, and CSV
together. Populations are never pooled. Readout metric and Explore scope,
metric, and series selections persist when available in both populations.

The selected population keeps its source identity, integrity status, required
comparison completeness, and original multiple-testing family. Switching
populations does not change either population's analyzed family. Hiding a
failed guardrail does not complete the primary metric's required comparisons.

Health distinguishes two checks:

- **Analysis-design assignment check:** uses the analysis's recorded integrity
  evidence. No declared allocation means **not run**, never a pass.
  `integrity.allocation_missing_or_nonconstant` means declared target weights
  are missing or nonconstant, not that observed group counts are missing.
- **Dashboard allocation check:** compares the selected population with
  `expected_allocation`. A mismatch names its source and population before
  effect summaries. Observed allocation is reported separately from the
  analysis-design check.

The Triggered view still warns if the assigned population's check failed or
could not be verified; passing triggered balance never replaces assignment
evidence. Routine passes, complete-result counts, and detailed check values
stay in Health, not Readout or Report.
Assigned allocation describes assignment integrity; triggered allocation
only describes cohort balance. Triggered inference requires treatment not to
affect trigger membership or timing. Balance cannot establish this condition;
[`health_check()`](../api.md#diagnostics-and-robustness) is a diagnostic, not
proof of causal validity. An incomplete feed or immature window is not a null
estimate.

A source-unavailable triggered view keeps its code and reason. Selecting
**Assigned** can show an assignment-level trajectory, but does not replace or
reinterpret a refused triggered daily/as-of series. Sequential group data uses
each population's own chain from the displayed checkpoint; it never borrows
the assigned prefix for a triggered row. An uncommitted triggered chain leaves
triggered cells unavailable with `triggered_chain_uncommitted`.

## Renderer functions

The complete dashboard embeds its theme, stylesheet, table adapter, and
controls; it needs no extra browser package or Studio server. Keep the bundled
notebook's host stylesheet beside that notebook. Standalone sections also
accept prepared data; emit `dashboard_styles(theme=config.theme)` once with them.

### `render_dashboard(experiment, snapshot=...)`

Returns the four-tab marimo anywidget with plots, scoped controls, metric
inspectors, Health, provenance, and Report. Readout and Report always contain
the complete captured metric family, independent of Explore selections.
**Added metrics** can change the displayed selection in a running notebook,
without rereading data.

### `dashboard_styles(theme=MIDNIGHT)`

Returns the configured theme and scoped stylesheet as one style block from
packaged resources. Tokens are scoped to `.inc-dashboard-root`; they do not
restyle the notebook canvas.

### `render_header(snapshot)`

Shows identity, window, population, inference, arms, and three summary cards:
enrolled units, observed arm split, and primary lift with bracketed interval
and direction-aware status. The complete dashboard's hero includes the source
label. Qualifications belong in Readout and Health, not repeated in the hero.

### `render_health(snapshot)`

Shows allocation bars and target markers, the sample-ratio-mismatch verdict,
and visible assignment warnings and result caveats. Available, warning, and
unavailable checks stay visible. Checks inapplicable without a trigger share
one summary line; **Technical details** retains diagnostic codes and contexts.
Expandable allocation history has one row per variant with cumulative share
and check statistics.

### `render_results(snapshot)`

Shows the declared family grouped by role in CoefTable, with forest plots and
an adverse-guardrail callout. Redundant confidence, significance, and
confidence-set columns are omitted; levels and interpretation remain in notes
and CSV.

### `render_metric_details(snapshot, metric=...)`

Shows lift, interval, direction, policy, and confidence-set details. Declared
metrics also show observed group data with eligibility, exclusions, counts,
units, and provenance. Added metrics instead show their Overview result and
its own captured collection evidence, not confirmatory Readout rows.

`render_header` and `render_metric_details` use the selected-population default
and accept `population="assigned"` or `population="triggered"`. Group data is
population-specific; assignment counts remain separately labeled.

### `load_explore(experiment, snapshot=..., metric=..., view=..., completed_windows_only=..., breakout=...)`

Loads a captured view without resolving a source, querying, or materializing.
`metric=None` loads all declared metrics; a declared or added name selects its
own series. Experiment and options are validated; refusals remain specific to
each captured state.

`ExploreView` is `"cumulative_lift" | "daily_values" | "cumulative_values" |
"segments"`. An unknown metric/view, an undeclared breakout, or
`completed_windows_only` on daily values or segments raises
`InvalidRequestError` (`dashboard.invalid_view`). Context names the rejected
metric, view, or breakout.

### `render_explore(snapshot, data, metric=..., view=..., completed_windows_only=...)`

Shows CoefTable cumulative-lift or daily/cumulative absolute trajectories,
their date basis and range, and unavailable-point reasons. Exact readings and
qualifications expand together. Applicable headline geometry is labeled
separately from the series. Pass the metric selection used to load the data.

### `render_details(snapshot)`

Shows collapsible experiment metadata, per-metric policies, provenance, and
static-snapshot limitations.

### `readout_csv(snapshot)` and `group_data_csv(snapshot, metric=None)`

Return UTF-8 CSV bytes suitable for `mo.download`, without a warehouse query.
The first exports headline rows; the second exports observed group data for
one declared metric or all of them. See [CSV exports](#csv-exports).

## Readout tables and disclosures

Dashboard tables retain metric and arm identity, lift, uncertainty, and a
readable outcome. Failure and prior-advisory columns appear when applicable.
Colors support the outcome; they do not replace it. Original results and CSV
retain full test and result details.

Plot color uses the recorded family selection when applicable, otherwise the
comparison's own test. Green/red require an established favorable/adverse
result, not merely a positive/negative estimate. Blue means a detected
difference without established favorability. Inconclusive and unavailable
results are gray; a comparison not selected as a discovery stays gray even
if its own test rejects. The dashboard makes no ship/no-ship decision.

### Evidence disclosures

- Select a metric's name to inspect its result. When it represents several
  methods, sources, dates, or segments, use the result picker; the inspector
  never substitutes a headline estimate.
  Picker labels distinguish the original analysis from Overview correction
  and identify the views that use each reading.
- **Test and interval details:** the selected result's estimate, exact interval,
  confidence level and retained budget, individual test verdict, and
  multiple-testing selection. **Technical record for this result** retains
  original records and links to matching source, check, and correction groups.
  These records are not repeated beneath every table.
- **How to read these intervals:** metric-specific exceptions, numeric set
  endpoints, and shared explanations. Construction and numerical
  qualifications have a separate reference.
- **Methods & assumptions:** shared tests, boundaries, and confidence levels
  grouped by role. Individual verdicts belong to the selected result.
- **Technical details:** one diagnostic per card, with nested context as
  labeled fields/lists; inapplicable trigger checks share an explanation.
- **Results needing caution:** distinct affected readings, grouped by metric
  with consequences and a searchable result picker.
- **Experiment provenance & definitions:** a brief experiment summary, with
  reproduction settings and source records available separately.

Missing evidence, incomplete decisions, adverse results, unavailability,
exclusion, reliability, and safety warnings stay visible outside disclosures.
Routine numerical assumptions remain in technical evidence and the inspector's
**Technical record for this result**, not warnings.

Result-specific caveats use numbered markers beside the affected lift and
short notes below the table. Identical full caveats share a marker; distinct
technical details get separate numbers. Notes do not repeat row coordinates.
They survive column filtering in Readout, Explore, and printed Report. CSV
and saved results retain original evidence rather than display markers.

The inspector's **Technical record for this result** retains assumptions,
result identity, and captured control/treatment successes and trials when
available. The standalone `render_metric_details()` remains self-contained.
The metric summary and selected result both keep the declared measurement
window or retention band, even when the metric has no written description.

### Guardrails

A row labels its declared limit, for example “Guardrail: no more than a 2%
relative decrease.” This is a threshold description, not a pass verdict.
Absolute margins are labeled in absolute units; required improvement is not
called allowed deterioration. **Guardrail** and its dashed threshold marker
share an orange color.

**Guardrail passed** rules out the declared unacceptable loss; it does not
establish improvement, and a small decline can pass. Failing the one-sided
safety test is **Inconclusive**, not proof of harm. Wholly adverse decision
intervals are called out separately in Readout and Report, even when the
guardrail's declared null was not rejected.

Supported fixed-horizon one-sided guardrails show the bound used by their
safety test and its one-sided confidence level across methods. A lower-bound
guardrail leaves the upper end open. The display level can differ from the
stored central interval; captured data and CSV keep that original interval.
This changes neither the verdict nor the table's column set.

## Interval and confidence-set geometry

Relative effects use percentages; absolute effects use outcome units, in both
tables and qualifications. A declared one-sided bound, `(−∞, upper]` or
`[lower, ∞)`, is not missing data. An undefined endpoint instead keeps its
reason and any finite opposite bound, never a fabricated infinity. Its numeric
interval and forest bar are omitted, while an available point remains visible.
Unavailable inference is never labeled nonsignificant.

Binomial, Fieller, and winsorized confidence sets keep their geometry even
without a finite point estimate. Disconnected sets remain unions with an
excluded gap; forest bars are omitted rather than filled across that gap.
Whole-line sets retain both infinite ends. A wide finite interval is not
reclassified by a size threshold. Multiple applicable explanations remain
visible. A usable absolute-null decision survives unavailable relative-set
inference; its reason and available absolute bounds appear separately in
metric units, including percentage points for rates.

An ordinary, non-FCR point-backed Fieller result retains its central display
interval: a 5% one-sided test has a 90% central interval. The interval
disclosure and Report also keep the 95% directional set. A directional row
selected for false-coverage-rate (FCR) adjustment instead keeps its
re-estimated open display interval. Tested-null verdicts and family selection
are disclosed separately in **Test and interval details** and Report,
identified by method, role, and arm where needed.

Confidence levels normally use up to two decimal places: 98.333…% becomes
98.33%. Fractional levels near 0% or 100% keep enough precision not to round
to either endpoint. If binary floating-point storage has already rounded a
level to 100% despite a positive noncoverage budget, the display keeps 100%
and shows that interval's own α budget. A relative confidence set shows its
own budget separately.

In the checkout demo, D7 retention is an increasing one-sided guardrail: its
lower relative-lift bound is constrained and its upper direction intentionally
unconstrained. This is not a zero-control-count result. **Data by group** shows
the rates, counts, and maturity. Observed retention rates stay between 0% and
100%; relative lift is a different quantity. A decreasing binary test retains
the physical −100% relative-lift floor, not an invented open lower endpoint.
Explore notes distinguish headline evidence from the selected trajectory;
they do not assert new inference for each daily value.

## Explore views

### Overview and correction

Overview contains every displayed metric's relative lift. **Compare by** nests
segments beneath each metric's whole-experiment row. The experiment's own
whole-experiment rows are labeled “as in Readout” and keep their roles and
corrections; they are never corrected again here.

Other cells are exploratory: offered saved metrics and every segment of every
metric and declared breakout. Eligible cells form one Benjamini–Hochberg (BH)
family at the plan's `q`, whether shown or not. The family is fixed for the
snapshot; changing **Compare by** or metric visibility before or after seeing
results cannot change the correction or a cell's result. Only discoveries
receive finding colors and FCR-adjusted intervals. Other exploratory intervals
are unadjusted and neutral even if they exclude zero.

BH controls false discovery rate under independence or positive dependence.
Cells share units and the control arm;
[the required dependence condition is not checked](../limitations.md#fixed-horizon-fdr-control-assumes-a-dependence-condition-that-is-not-checked).
Re-preparation on more data is a new look, not continuation of that family.
Ineligible cells, such as quantile metrics or informative priors, keep their
stored intervals. Their exclusion reasons appear beside the relevant visible
metric and breakout, never another source's same-named segment. They cannot
become discoveries or block other cells. Correction settings remain in
**How Explore comparisons work**. Under registered sequential plans, added
metrics and uncorrected segments are refused with their codes.

### Added exploratory metrics

`DashboardConfig(exploratory_metrics=(...))` chooses from
`workflows.available_metrics(experiment, reader)`: saved per-unit metrics on
the experiment's unit that it does not declare. Report-only `total`/`active`
metrics and other entities' metrics are not offered. A name absent from the
definitions raises `dashboard.invalid_config` before reads.

Every offered metric is captured with the headline, and every eligible cell
counts in the exploratory family. Selected metrics appear in Overview's
Exploratory group and Time series, never Readout or Report. **Added metrics**
lists them; **+ Add metrics** searches the saved definitions. Applying a change
re-renders the snapshot and reopens Explore; results and correction cannot
change. An unoffered name is refused and leaves the current page intact.
Selecting an added metric's name shows its corrected Overview result. Only
declared metrics have captured group data and group-data downloads.

Offered metrics are read together where corrections do not span metrics:
one read per time-series state and scope (five states for the whole experiment
and each breakout), plus one Overview read for the whole experiment and each
breakout. Segment reads are per metric; combining them under a BH breakout
policy would incorrectly join their families. A refused joint read is retried
per metric so each keeps its own refusal.

### Interactions

Explore shows one row per concurrent experiment for the selected metric.
Each row estimates how much the focal experiment's effect changes when the other
experiment is on rather than off. Pair details show both effects, their
difference, and the four assignment groups. Overview and pair plots share a
scale within the selected metric.

- **% lift vs. control** divides each effect by its focal-off mean with the
  other experiment at the same setting. The difference between those lifts
  uses percentage points of lift (`pp of lift`). This is the default when
  the metric has a bounded relative comparison.
- The option named for the metric shows effects and their difference in its
  own units. For a rate, `pp of save rate` means percentage points of save
  rate, not of lift.

| Inventory coverage | What the screen can say |
|---|---|
| Not checked | No scan was supplied or the inventory was not captured. |
| Partial | Known pairs can show descriptive estimates, but no scan-wide intervals or conclusions. |
| Complete | The source confirms which experiments were in scope, including if there were none. |

Complete coverage requires both the inventory frozen before entry and the
record at the cutoff to cover entry through the end of the outcome window
and agree on the assignment revisions in scope.

Complete inventory coverage does not make all pair data complete. A
comparison that needs a missing, pending, or empty group has no estimate;
the affected comparison and its exact reason remain visible. An available
estimate can lack an interval, or an interval can be too weakly supported
to justify a conclusion. **Higher with other on** or **Lower with other on**
means the supported interval for the difference excludes zero. **None detected**
means it contains or touches zero. Neither outcome is a rollout decision.

#### Interaction family and intervals

Absolute intervals use the two-sided Welch approximation. Relative intervals
account for uncertainty in the baseline using a two-sided Fieller interval.
A zero, negative, or imprecisely measured baseline can prevent a relative
result even when the absolute result is available. The interval for the
difference in relative lift subtracts the two adjusted lift intervals,
using the lowest and highest possible differences of their endpoints.

Bonferroni adjusts each scale separately; coverage is not guaranteed jointly
across both scales. This adjustment is separate from Explore Overview's BH
correction and the registered pair workflow. The family counts three
comparisons per frozen candidate and metric: the two effects and their difference.
For three candidates and two metrics, the divisor is 18, even if a row is
unavailable or a metric is hidden. Changing the metric, selecting a pair
by name, or opening it from Readout does not change the family, level, or
bounds. See **Scope and interval adjustment** for the family size, level,
and fixed look. The adjustment does not require independent pairs; the
within-pair independent assignment and outcome requirements still apply.

These exploratory comparisons describe pairs, not a combination of three
or more treatments. The **Possible interaction** badge is based on the
adjusted absolute interval. It opens the detected pair in **% lift vs. control**
if that scale also detects the change. The primary result is unchanged.

### Time series and dates

**Whole experiment** shows all primary, secondary, and guardrail metrics;
selected added metrics follow as exploratory rows. **Relative lift** is
cumulative; **Absolute values** shows both arms, cumulative or daily. Each
metric retains its units and scale. Unavailable metrics keep a row and an
expandable reason. A breakout restores **Focus metric** and shows that metric's
segments; it changes estimates and uncertainty, not just labels. Each
`(source, property)` is captured separately. A refused source is never replaced
by another source or whole-experiment results.

Cumulative dates are the calendar dates as of which estimates are computed.
Daily dates follow the basis beneath each chart: retention uses exposure
cohorts, not return dates. The clock starts at assignment for Assigned and at
first trigger for Triggered.

**Completed windows only** applies to cumulative views, never daily values.
It excludes users until their full metric window elapses; otherwise partial
follow-up may contribute. A January 15 assignment completes a 14-day window
on January 29; a January 20 first trigger completes it on February 3. This
can shift the first plotted date and later contributing users. Open-ended
metrics need no maturity wait: events contribute from the selected population's
start through the captured observation cutoff.

Whole-experiment rows share a date range, not a value scale; breakout rows
share the selected metric's scales. Both arms share a row and scale for
absolute values. Different date bases are labeled per metric; dates appear
on charts rather than in another column. **Latest readings and qualifications**
contains exact values, intervals, dates, and qualifications, plus tested nulls
for lift. **Focus metric** groups by role and scrolls at a fixed height. From
nine metrics, a name/role filter narrows the list without changing a selected
metric hidden by that filter.

Missing estimates remain plot gaps; **Plot notes** counts them by reason.
Lift shading needs two finite bounds. Open sides are not drawn as separate
lines or filled bands; their latest bounds remain in the readings disclosure.
Absolute plots have numeric y-axis ticks: rates and `%` use percentages,
`USD` uses currency, `count` retains fractions, and other units appear as
suffixes. Formatting changes neither values nor intervals. Charts keep their
native size; narrow tables scroll rather than shrink labels and bands.

Fixed-horizon cumulative monitoring is descriptive, not safe for repeated
decisions. Registered Bernoulli inference retains its sequential guarantee;
asymptotic-mean monitoring is labeled asymptotic, not finite-sample guaranteed.

### Allocation history

Health uses first-assignment enrollment counts independently of outcomes and
retention maturity. Each variant's cumulative share is plotted on observed
enrollment dates in the experiment's day boundary, with a dashed configured
target and a pointwise 95% Wilson band. Bands remain nonzero at shares zero or
one. They are descriptive, not adjusted for repeated looks or sequential
sample-ratio-mismatch thresholds; the separate allocation check supplies its
verdict.

`snapshot.allocation_history` holds immutable rows from
`increment.workflows.allocation_history()`. Each has `analysis_population`
(`assigned` or `triggered`), preserving enrollment versus first-trigger cohorts;
rows are ordered by date and arm within population. A refusal is retained in
`snapshot.allocation_history_refusal` and displayed without disabling headlines.
A refused allocation check does not query history. Cluster-randomized and
non-native sources do not supply this unit timeline.

### No-breakout experiments

`snapshot.breakouts` contains `(declared_source, dimension)` pairs. An empty
tuple omits the segment selector, shows “No declared breakouts” in segments,
and issues no breakout query. This is supported, as is a dimension with one
resolved segment.

## Observed data by group

Declared metrics' disclosures and downloads use the same captured rows;
added metrics have neither. Rows include assigned/eligible units, observed
arm value, meaningful counts/totals, cutoff/window, and exclusion reasons:
`assigned = eligible + not mature + no observed day + other excluded`.
The header's enrolled count is not a metric denominator. Group data can remain
inspectable without a decision row, but does not bypass invalid-input refusals.

Values precede CUPED, prior shrinkage, and winsorization while keeping cohort,
filters, and windows. Ratios divide group component totals, not average
individual ratios. Quantiles use the linear sample-quantile definition. A sum
of per-unit averages is not event-level revenue. Converted/retained unit
counts differ from qualifying event counts.

Sequential disclosures use each metric's displayed checkpoint, never current
outcomes next to an earlier stopped result; source kind and prefix identify
it. Unretained historical exclusions, raw event counts, and pre-transform
values stay unavailable with reasons. Retained transformed inputs are labeled
separately.

## CSV exports

Report owns the full-readout download; Explore owns metric-specific group-data
downloads for declared metrics. All use the captured snapshot. Formula-leading
string cells get an apostrophe; numeric cells stay numeric and unrounded,
and unavailable numbers stay empty, not zero.

### Readout CSV

Confidence-set cells contain unrounded JSON metadata without retained samples
or solver references. Scoped rows also retain source/population, integrity
status/context, required-comparison availability, filtered/hidden identities
and reasons, original testing family, and advisory codes/explanations. Nested
evidence is canonical JSON, not Python representations.

Filtering keeps the original required comparisons and family; visible rows
mark a partial view. Filtering every row leaves no CSV row to carry metadata.
Keep the scoped collection or its saved result if an empty view must retain
its full scope. A readout CSV is not a portable inference checkpoint.

`coverage_alpha` is the nullable noncoverage budget of that row's captured
displayed interval or set; it is blank if no budget was retained. For older
snapshots whose readout rows predate this field, `readout_csv` projects it from
the captured estimate. It never reconstructs a budget from a rounded level
or borrows another interval's budget.

### Group-data CSV

Unavailable numeric cells are empty with reasons in JSON. A finite mean can
remain available when a display total is out of numeric range. Rows identify
the experiment, preparation timestamp, binding fingerprint, and applicable
checkpoint prefix; unit identifiers and event rows are never exported.
Preparing another snapshot cannot change an earlier download.

## Static HTML and printing

Live notebooks and static exports contain all four tabs. Population, metric,
declared-breakout, relative/absolute, daily/cumulative, and maturity controls
select embedded data without a Python kernel. Static exports show captured
added metrics, but changing that selection requires a running notebook; the
control explains this rather than applying the change. Neither form refreshes
data without re-preparation.

Static HTML keeps the same captured interaction rows, intervals, counts,
and reasons as the notebook. Metric selection and named pair navigation
work offline. When shown in Report and print, the interaction evidence is
identified as exploratory and does not enter the headline CSV. There is no
separate saved scan: HTML is a read-only export, not an input to preparation.
Keep the [original inputs](../guides/dashboard.md#check-for-interactions-with-other-experiments)
to reproduce the same look.

The masthead's light/dark toggle changes tables, charts, and surrounding
surfaces without changing evidence or selections. It follows the operating
system until chosen, then uses browser storage when available. It also works
in static HTML. Printing uses the light palette without changing screen mode.

Both hero and Report primary card retain uncertainty intervals; Report uses
smaller, muted interval text. Tables omit a standalone **Interval** column
but retain forest plots; exact bounds and levels remain in inspection, notes,
and CSV.

Report includes every captured metric result and interval, visible material
limitations, displayed confidence levels, and shared interpretation. Individual
tests and group selections have a compact comparison table. Stored budgets
remain in a separate reference; routine check passes and allocation values
stay in Health.
Material check warnings precede the primary effect and results table,
including in printed reports.

**Print / Save PDF** uses one A4 page only if prose and numeric evidence stay
at or above `theme.printing.min_font_size_pt` (default 8pt). Larger families
paginate with repeated headers. Over-wide tables split into column bands
repeating metric and arm identity; page whitespace is bounded and axes retain
native styling. Printing expands folded groups and individual test decisions,
then restores them afterwards. Material qualifications and complete
confidence-set geometry remain on paper; routine specifications and raw
records do not. Observed arm data, allocation history, and reproduction
records remain in the dashboard rather than filling the PDF.

## Theme presets

Start from immutable `MIDNIGHT` and replace the categories you need:

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

- **`light`, `dark` (`DashboardPalette`):** surface, text, arm, and
  evidence-direction colors.
- **`typography` (`DashboardTypography`):** body/heading/monospace fonts;
  body/table/estimate/interval sizes.
- **`layout` (`DashboardLayout`):** dashboard/section widths, padding, radius,
  table-cell padding.
- **`charts` (`DashboardCharts`):** forest, trajectory, and allocation-chart dimensions.
- **`printing` (`DashboardPrint`):** prose and numeric-evidence font floor in points.

Dimensions and screen text sizes use CSS pixels. Colors must be hexadecimal;
font stacks cannot contain CSS rules. Invalid categories/values raise
`InvalidRequestError` (`dashboard.invalid_theme`) before reads. Replacement
never mutates `MIDNIGHT` or captured data.

The default pairs **Midnight · Daylight** (light) and **Midnight** (dark).
Custom themes provide their own paired palettes; exports carry the
configuration without separate theme files.

## Running the bundled notebook

From a repository checkout, run:

```bash
uv run --extra dashboard --extra demo marimo edit examples/dashboard.py
```

Use `marimo run` instead of `marimo edit` for read-only serving, or
`uv run --extra dashboard --extra demo python examples/dashboard.py` for script
execution. The notebook includes `checkout_redesign` and triggered
`express_checkout`; [open the exported example](../examples/dashboard.md).

Do not use bare `uv run examples/dashboard.py`. Its `# /// script` metadata
makes that form use an isolated script environment without the dashboard/demo
extras. Use explicit `python` or `marimo` as above. The metadata keeps marimo's
chrome light; the embedded dashboard has an independent light/dark toggle.
