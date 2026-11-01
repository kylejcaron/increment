# Sitewide impact

`increment.estimators.sitewide(experiment, resolver, metric)` estimates the
whole-site change if the per-unit effect had applied to every enrolled unit
during the experiment window. Use it for a rollout estimate, not as another
per-unit effect.

It needs the site's raw event volume, so it runs on an experiment built by
`from_definitions` or on an adopted `from_unit_day_artifact` published with
the metric's `site_volume` extension. Pre-melted summaries and moments cannot
supply it; see the
[artifact extension matrix](data-model.md#operation-and-extension-matrix).

!!! warning "Read only the window observed"
    Sitewide assumes the observed per-unit effect applies to all enrolled
    units, and it reports only the experiment's enrollment window
    (`experiment.start .. experiment.end`), never an annualized or
    longer-horizon projection. An effect measured in one window does not
    establish its size or statistical signature in a later one; for
    retention, even a constant per-unit effect can become less detectable as
    the window grows ([why](metric-types.md#unbounded-retention-doesnt-keep-getting-more-precise)).
    Check [when it refuses](#when-it-refuses) before choosing a source or
    metric.

## Run it

This example publishes one immutable artifact carrying trigger membership and
site-volume evidence, then reads the assigned and triggered estimates from it:

```python
from pathlib import Path
import datetime as dt

import ibis
import polars as pl
import warnings

from increment import from_definitions, from_unit_day_artifact
from increment.artifacts import (
    SourceSnapshotEvidence,
    artifact_context,
    publish_unit_day_artifact,
    unit_day_artifact_extension_catalog,
)
from increment.query.session import WarehouseArtifactStore
from increment.estimators import IncrementWarning, sitewide

Path("triggered-definitions").mkdir(exist_ok=True)
Path("triggered-definitions/fact_sources.yaml").write_text("""\
dialect: duckdb
fact_sources:
  - name: events
    sql: SELECT * FROM events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: enrolled
        column: null
      - name: saw_surface
        column: null
      - name: revenue
        column: value
      - name: sessions
        column: value
""")
Path("triggered-definitions/exposures.yaml").write_text("""\
exposures:
  - name: assignment
    fact: enrolled
  - name: saw_surface
    fact: saw_surface
""")
Path("triggered-definitions/metrics.yaml").write_text("""\
metrics:
  - name: revenue
    type: mean
    entity: user_id
    fact: revenue
    aggregation: sum
  - name: revenue_per_session
    type: ratio
    entity: user_id
    numerator: {fact: revenue, aggregation: sum}
    denominator: {fact: sessions, aggregation: sum}
""")
Path("triggered-definitions/experiments.yaml").write_text("""\
experiments:
  - name: triggered_launch
    exposure: assignment
    trigger: saw_surface
    unit: user_id
    start: 2025-03-01
    end: 2025-03-10
    control_group: control
    plan:
      secondaries: [revenue, revenue_per_session]
""")

triggered_rows = []
trigger_start = dt.datetime(2025, 3, 1, 9)
for i in range(40):
    user = f"u{i:03d}"
    arm = "treatment" if i % 2 else "control"
    event_base = {
        "user_id": user,
        "group_id": arm,
        "experiment_id": "triggered_launch",
    }
    triggered_rows.append({**event_base, "ts": trigger_start, "event": "enrolled", "value": None})
    if i % 3 != 0:
        triggered_rows.append(
            {**event_base, "ts": trigger_start, "event": "saw_surface", "value": None}
        )
    triggered_rows.extend(
        [
            {
                **event_base,
                "ts": trigger_start + dt.timedelta(days=1),
                "event": "revenue",
                "value": 12.0 + (4.0 if arm == "treatment" and i % 3 != 0 else 0.0),
            },
            {
                **event_base,
                "ts": trigger_start + dt.timedelta(days=1),
                "event": "sessions",
                "value": 1.0,
            },
        ]
    )
# Unexposed site activity contributes to site volume but not enrolled outcomes.
triggered_rows.append(
    {
        "user_id": "background",
        "group_id": None,
        "experiment_id": None,
        "ts": trigger_start + dt.timedelta(days=2),
        "event": "revenue",
        "value": 80.0,
    }
)
triggered_events = pl.DataFrame(triggered_rows)
triggered_con = ibis.duckdb.connect()
triggered_con.create_table("events", triggered_events.to_arrow())
with warnings.catch_warnings():
    warnings.simplefilter("ignore", IncrementWarning)
    triggered_experiment, triggered_resolver = from_definitions(
        "triggered-definitions/",
        triggered_con,
        experiment_name="triggered_launch",
        source_snapshot_evidence=SourceSnapshotEvidence(
            observation_cutoff_ts=dt.datetime(2025, 3, 12, tzinfo=dt.UTC),
            complete_through_by_feed={"events": dt.datetime(2025, 3, 12, tzinfo=dt.UTC)},
        ),
    )
triggered_store = WarehouseArtifactStore(triggered_con, schema_name="sitewide_artifacts")
triggered_context = artifact_context(triggered_experiment, triggered_resolver)
triggered_catalog = unit_day_artifact_extension_catalog(triggered_context)
triggered_requests = [
    entry.request
    for entry in triggered_catalog
    if entry.request.kind in {"site_volume", "trigger_population", "assignment_counts"}
]
triggered_reference = publish_unit_day_artifact(
    triggered_experiment, triggered_resolver, triggered_store, extensions=triggered_requests
)
incomplete_reference = publish_unit_day_artifact(
    triggered_experiment,
    triggered_resolver,
    triggered_store,
    extensions=[
        entry.request for entry in triggered_catalog if entry.request.kind == "site_volume"
    ],
)
incomplete_experiment, incomplete_resolver = from_unit_day_artifact(
    triggered_store, incomplete_reference, expected_context=triggered_context
)
experiment, resolver = from_unit_day_artifact(
    triggered_store, triggered_reference, expected_context=triggered_context
)
```

```python
for m in ("revenue", "revenue_per_session"):
    assigned = sitewide(experiment, resolver, metric=m)
    triggered = sitewide(experiment, resolver, metric=m, population="triggered")
    print(f"{m}: assigned={assigned}; triggered={triggered}")
    assert assigned.snapshot_id == triggered.snapshot_id
    assert assigned.window == triggered.window
    assert triggered.membership_cutoff is not None
```

## Read the result

`sitewide()` returns a `SitewideImpact` for a mean metric and a
`SitewideRatioImpact` for a ratio metric. Both are `exploratory_unadjusted`:
no family adjustment is applied, and their fixed-horizon intervals are not
anytime-valid. Both baselines net out every enrolled non-control arm's lift
(`other_arm_ids`), not only the arm being scored.

| Field (both models) | Meaning |
|---|---|
| `absolute_impact`, `absolute_impact_lb`, `absolute_impact_ub` | The ship-to-all gain with its interval. For a mean metric it is in the metric's own units; for a ratio metric it is `shipped_ratio - baseline_ratio`, a difference of site ratios. |
| `relative_impact`, `relative_impact_lb`, `relative_impact_ub` | That gain as a fraction of the baseline (`baseline_volume` or `baseline_ratio`). |
| `analysis_population`, `membership_cutoff`, `window`, `snapshot_id` | The population read, its trigger-membership cutoff, the experiment window, and the evidence identity. |
| `n_clusters`, `absolute_dof`, `relative_dof` | `None` at unit grain; otherwise the cluster count and the degrees of freedom each interval was cut at. |

| `SitewideImpact` (mean metric) | `SitewideRatioImpact` (ratio metric) | Meaning |
|---|---|---|
| `delta` | `delta_num`, `delta_den` | The per-unit absolute lift being scaled up; a ratio metric has one per component. |
| `site_total_volume` | `site_total_numerator`, `site_total_denominator` | The observed site-window total(s). |
| `baseline_volume` | `baseline_numerator`, `baseline_denominator`, `baseline_ratio` | The counterfactual total(s) with nobody exposed to treatment. |
| — | `shipped_numerator`, `shipped_denominator`, `shipped_ratio` | The ship-to-all total(s) with every enrolled unit treated. |

??? info "Statistical details: the ship-to-all counterfactual"
    Given a per-unit absolute lift

    $$
    \delta = \bar y_T - \bar y_C
    $$

    and its Welch-style standard error

    $$
    \mathrm{se}(\delta) = \sqrt{\frac{\mathrm{Var}(y_T)}{n_T} + \frac{\mathrm{Var}(y_C)}{n_C}},
    $$

    start from the observed **site-window volume** $V$: everything that
    happened on the metric's raw event stream during the experiment's
    enrollment window, not just what the exposed population did. $V$ already
    contains the treatment arm's lift, so subtracting its contribution
    (its $n_T$ units, each $\delta$ above what they would have done under
    control) recovers the **counterfactual baseline** -- what the whole site
    would have produced with nobody exposed to treatment:

    $$
    V_0 = V - \delta\,n_T
    $$

    The **ship-to-all impact** is the absolute gain from rolling the lift out
    to every enrolled unit, control and treatment alike
    ($N_{\text{exp}} = n_C + n_T$):

    $$
    I = \delta\,N_{\text{exp}}
    $$

    and the **relative impact** expresses that gain as a fraction of the
    counterfactual baseline:

    $$
    \frac{I}{V_0}
    $$

    $V$ and $N_{\text{exp}}$ are data, not estimated quantities -- fixed
    constants for the intervals below, which propagate only $\delta$'s
    sampling variance. Absolute impact is linear in $\delta$, so its interval
    is just the lift's Wald interval scaled by $N_{\text{exp}}$:

    $$
    \mathrm{se}(I) = N_{\text{exp}}\,\mathrm{se}(\delta)
    $$

    Relative impact is not linear -- $V_0$ is itself a function of $\delta$ --
    but the quotient rule's cross-terms cancel to a clean closed form:

    $$
    \mathrm{se}\!\left(\frac{I}{V_0}\right)
    = \left|\frac{N_{\text{exp}}\,V}{V_0^2}\right|\,\mathrm{se}(\delta),
    $$

    the same first-order delta-method (Taylor) approximation this library
    uses everywhere a ratio depends on an estimated quantity.

    !!! note
        A non-positive $V_0$ makes "relative impact" undefined -- a fraction
        of a baseline that isn't positive has no sensible sign or magnitude.
        `sitewide()` raises `ValueError` rather than return one when
        the baseline is non-positive.

    **Ratio metrics.** A ratio metric (revenue per session, CTR) needs two
    lifts instead of one: a numerator lift $\delta_{\text{num}}$ and a
    denominator lift $\delta_{\text{den}}$, plus their covariance (they come
    from the same per-arm sums, not independent draws). The counterfactual
    and ship-to-all totals are built the same way as $V_0$ above, once per
    side:

    $$
    N_0 = V_{\text{num}} - n_T\,\delta_{\text{num}}, \qquad
    D_0 = V_{\text{den}} - n_T\,\delta_{\text{den}}
    $$

    $$
    N_1 = N_0 + N_{\text{exp}}\,\delta_{\text{num}}, \qquad
    D_1 = D_0 + N_{\text{exp}}\,\delta_{\text{den}}
    $$

    and the impact is the **difference of the two ratios**, not a ratio of
    differences:

    $$
    \text{impact} = \frac{N_1}{D_1} - \frac{N_0}{D_0}
    $$

    Its delta-method standard error follows from the two partial derivatives
    (with respect to $\delta_{\text{num}}$ and $\delta_{\text{den}}$) and
    includes the numerator/denominator covariance term, since the two are
    not independent:

    $$
    \mathrm{Var}(\text{impact}) = g_{\text{num}}^2\,\mathrm{Var}(\delta_{\text{num}})
    + g_{\text{den}}^2\,\mathrm{Var}(\delta_{\text{den}})
    + 2\,g_{\text{num}}\,g_{\text{den}}\,\mathrm{Cov}(\delta_{\text{num}}, \delta_{\text{den}})
    $$

    `sitewide()` reports a ratio metric's `relative_impact` as
    `absolute_impact / baseline_ratio`. Its uncertainty uses covariance-aware
    delta-method gradients for the numerator and denominator lifts; the
    reported intervals use the resulting relative-impact standard error.

    **Multi-arm experiments.** With a second treatment arm enrolled, $V$
    carries *both* arms' lifts: $V = V_0 + \delta_A n_A + \delta_B n_B$.
    Netting out only the arm being scored would leave the other arm's lift
    counted as baseline, so every enrolled non-control arm is subtracted:

    $$
    V_0 = V - \sum_i \delta_i\,n_i
    $$

    $N_{\text{exp}}$ counts all enrolled units. The arms' lifts are not
    independent -- every $\delta_i$ subtracts the same control mean, so
    $\mathrm{Cov}(\delta_i, \delta_j) = \mathrm{Var}(\bar y_C) = \mathrm{Var}(y_C)/n_C$,
    which is *positive*. Absolute impact depends only on the selected arm's
    lift; relative impact also carries the shared-baseline covariance into
    its interval.

## Assigned or triggered

With a declared `trigger:`, `population="assigned"` (the default) estimates
the intent-to-treat impact of assignment. `population="triggered"` scales
the effect among finalized trigger members by the triggered enrolled count.

| Rule | Both populations |
|---|---|
| Window | The experiment calendar window (`start .. end`), not `run()`'s trigger-anchored windows. Later outcomes are excluded even when `observation_end` extends past the experiment; a late trigger can still be a member, but its outcomes come from the experiment window. |
| Site volume | Always the full site, never only triggered units. |
| Identification | Randomized only; observational and encouragement designs refuse before evidence loads. |
| Completeness | Outcomes must be complete through the experiment end, or the call refuses with `facade.analysis.sitewide_outcomes_incomplete`. Rerun after the feeds reach the end day. |

!!! warning "The triggered estimate is conditional on its assumptions"
    It assumes trigger membership is independent of treatment assignment,
    units that never trigger are unaffected, and enrolled units exhaust
    exposure. The result carries an allocation-balance diagnostic, but a
    passed diagnostic cannot establish those assumptions.

??? info "Statistical details: completeness evidence and calibration"
    Triggered results require a feed watermark in `SourceSnapshotEvidence`
    (or the artifact) proving completeness through the end day. Assigned
    results refuse if a snapshot cutoff or declared watermark is earlier; a
    native analysis without snapshot evidence has no cutoff to check.
    `snapshot_id` equality is content-bound for artifacts only; native
    identity reflects declared `SourceSnapshotEvidence`, not table contents.

    In two-arm simulations whose assumptions held, both readings covered the
    realized site impact at 95.1%–95.7% for nominal 95% intervals. When units
    that never trigger were affected, the triggered reading missed by the
    omitted effect while the assigned reading stayed calibrated. See
    [limitations](../limitations.md) for the designs, the precision
    comparison and what the campaign does not cover.

## Multi-arm experiments

`sitewide()` scores one arm at a time: pass `arm="treatment_b"` to select it
when several treatment arms are enrolled. The question is what would happen if that arm alone
shipped to everyone, compared with no treatment; arms are never combined into
an unshippable scenario. Every enrolled arm's lift is removed from site volume
to form the baseline, so co-enrolled arms add sampling variance to the
relative impact's interval. The formula is in the fold above.

## Cluster-randomized experiments

`Experiment.cluster` (see [The data model](data-model.md#randomization-grain-vs-analysis-grain))
names the stores, markets, or other units whose assignment actually changed,
not the individually measured units. Clustered readings come from unwindowed
`mean` and `ratio` metrics with sum- or count-aggregated components; any other
declared type raises `CapabilityError`. Conversion, retention and quantile
metrics refuse for the reasons in [when it refuses](#when-it-refuses) whether
or not the experiment is clustered.

Declaring a cluster never moves the point estimate. It changes the standard
error, because units within a cluster are not independent draws, and moves
inference from a Normal to a $t$ reference whose degrees of freedom depend on
what is reported. `absolute_dof` and `relative_dof` echo the degrees of
freedom each interval was cut at.

**How many clusters is enough.** Below 40 total clusters, a `RuntimeWarning`
names the over-rejection risk, and borderline significance should be treated
as fragile. Each enrolled arm needs at least two clusters so its
between-cluster variance is defined. More clusters buy real precision; adding
units inside each cluster adds little.

!!! warning "Enrolled units must exhaust the treated population"
    Cluster randomization usually treats the whole cluster, every register at
    a store or every session in a market, not just the units that logged an
    enrollment event. A returning customer who skipped the enrolling page but
    still bought at a treated store is real treated volume with no enrollment
    row, so it lands in the counterfactual baseline as though untouched. The
    bias has a fixed direction: the baseline is biased upward and the reported
    ship-to-all impact is understated, never overstated. The library does not
    implement a cluster-restricted site total or spillover correction.

??? info "Statistical details: cluster standard errors and degrees of freedom"
    Each arm becomes a ratio of cluster totals: outcome summed per cluster
    divided by units summed per cluster, the same cluster-robust reduction
    used by `run()` and other cluster-aware estimators. The point estimate
    matches a unit-grain mean; only the standard error changes. Cluster count
    is a qualification, not an admission floor: structurally valid contrasts
    use a qualified Normal/$t$ reference.

    - **Absolute impact for a plain (non-ratio) metric** combines the target
      and control arm mean variances. Its $t$ reference uses a two-component
      Welch–Satterthwaite correction, pairing each variance component with
      its own arm's cluster degrees of freedom ($K_i - 1$). It does not
      pool the cluster count or automatically use the pairwise sum.
    - **Relative impact, and a ratio metric's absolute impact,** mix
      contributions from every enrolled arm. A co-enrolled arm's lift enters
      the shared baseline used by either quantity, so more arms add sampling
      variance to the standard error, not necessarily a more conservative
      reference distribution. The combined estimate uses a Welch–Satterthwaite
      correction across every contributing arm's cluster count (each arm's own
      $K_i - 1$, including control), weighted by that arm's variance
      contribution. It is not a simple average of the arms' individual degrees
      of freedom. The result falls between the smallest contributing arm's
      $K_i - 1$ and the sum of all contributing arms' $K_i - 1$, and can fall
      below the target-control pairwise degrees of freedom. A co-enrolled arm
      with fewer clusters can lower an every-arm reference further.

    The two kinds of reference need not move in the same direction when a
    cluster is declared. The [cluster-randomized rollout](#a-cluster-randomized-rollout)
    below shows both at 42.0245 with 24 cluster degrees of freedom per arm.

## When it refuses

| Case | What happens | Do instead |
|---|---|---|
| `RetentionMetric` or `QuantileMetric` | `CapabilityError` before any moments are built: site volume is undefined, because a retention outcome is anchored to each unit's own exposure time and a quantile is not additive across events. | Use an additive mean, count, or ratio metric. |
| Windowed or non-additive metric (`avg_event`, `min`, `count_distinct`, a per-unit window), including ratio components | Refuses before evidence loads: site totals cover the calendar window `[start, end]` and cannot align. | Use an unwindowed sum or count aggregation. |
| Conversion metric | `facade.analysis.sitewide_metric_unaligned` before evidence loads, native or artifact: a converter indicator is a maximum over a unit's events, not an additive total. | Declare a `count` metric (`type: mean`, `aggregation: count`) on the conversion fact. |
| `from_frame` summaries and panels, `from_moments` | Pre-melted outcomes or moments carry no whole-site raw-event total. | Use `from_definitions`, or adopt a unit-day artifact published with the metric's `site_volume` extension ([extension matrix](data-model.md#operation-and-extension-matrix)). |
| Artifact without trigger membership | `artifact.extension.missing` for `population="triggered"`. | Republish with the `trigger_population` extension, as below. |
| Outcomes incomplete through the end day | `facade.analysis.sitewide_outcomes_incomplete`, either population. | Rerun after the feeds reach the end day. |
| Observational or encouragement design | Refuses before evidence loads: the unadjusted sitewide contrast does not implement those estimands. | Use `run(experiment, resolver)`. |
| Several non-control arms and no `arm=` | `ValueError` naming the candidates. | Pass `arm=`. |

The artifact published in [Run it](#run-it) carries trigger membership.
The `incomplete_reference` artifact, published with only `site_volume`, refuses the
triggered reading until the source is republished with the required evidence:


```python
from increment.errors import InvalidRequestError

try:
    sitewide(incomplete_experiment, incomplete_resolver, "revenue", population="triggered")
except InvalidRequestError as err:
    print(err.code)

recovered_reference = publish_unit_day_artifact(
    triggered_experiment,
    triggered_resolver,
    triggered_store,
    extensions=triggered_requests,
)
recovered_experiment, recovered_resolver = from_unit_day_artifact(
    triggered_store, recovered_reference, expected_context=triggered_context
)
recovered = sitewide(recovered_experiment, recovered_resolver, "revenue", population="triggered")
print(recovered.analysis_population, recovered.membership_cutoff)
incomplete_resolver.close()
recovered_resolver.close()
resolver.close()
triggered_resolver.close()
```

## Worked example

This example uses the native `from_definitions` path: a definitions directory
plus a DuckDB connection, as in [The data model](data-model.md#running-it).
The event log carries off-experiment activity on purpose: a `page_view` marks
enrollment, `purchase` is the revenue metric, `session_end` backs a ratio
metric's denominator, and background purchases from users who were never
exposed still count toward the site total.

<!-- invisible-code-block: python
from pathlib import Path

Path("definitions").mkdir(exist_ok=True)
Path("definitions/fact_sources.yaml").write_text("""\
dialect: duckdb

fact_sources:
  - name: event_log
    sql: SELECT * FROM analytics.event_log
    timestamp_column: event_at
    entities:
      - user_id
    facts:
      - name: page_view
        column: null
        description: A user viewed a page (occurrence-only event)
      - name: purchase
        column: revenue
        description: A purchase event with its revenue amount
      - name: session_end
        column: session_count
        description: One per session end event, for additive session totals
""")
Path("definitions/exposures.yaml").write_text("""\
exposures:
  - name: first_page_view
    fact: page_view
    description: >
      The first page view inside the experiment window marks the unit
      as enrolled.
""")
Path("definitions/metrics.yaml").write_text("""\
metrics:
  - type: mean
    name: revenue_per_user
    description: "Total purchase revenue per user over the full calendar window"
    entity: user_id
    preferred_direction: increase
    fact: purchase
    aggregation: sum
  - type: ratio
    name: revenue_per_session
    description: "Revenue per session (ratio of additive sums)"
    entity: user_id
    preferred_direction: increase
    numerator:
      fact: purchase
      aggregation: sum
    denominator:
      fact: session_end
      aggregation: sum
""")
Path("definitions/experiments.yaml").write_text("""\
experiments:
  - name: site_launch
    description: "New landing page vs the current one"
    exposure: first_page_view
    unit: user_id
    start: 2025-03-01
    end: 2025-03-15
    observation_end: 2025-03-29
    plan:
      secondaries:
        - revenue_per_user
        - revenue_per_session
    control_group: control

  - name: store_rollout
    description: "New checkout flow, rolled out store by store"
    exposure: first_page_view
    unit: user_id
    cluster: store_id
    start: 2025-04-01
    end: 2025-04-15
    observation_end: 2025-04-29
    plan:
      secondaries:
        - revenue_per_user
        - revenue_per_session
    control_group: control
""")
-->

```python
import datetime as dt
import random
import warnings

import ibis
import polars as pl

from increment import from_definitions
from increment.estimators import IncrementWarning, sitewide

random.seed(4)
start = dt.datetime(2025, 3, 1, 9, 0)
rows = []
for i in range(200):
    user = f"u{i:04d}"
    group = "treatment" if i % 2 else "control"
    enrolled = start + dt.timedelta(days=i % 10)
    rows.append(
        {
            "event_at": enrolled,
            "user_id": user,
            "event": "page_view",
            "revenue": None,
            "session_count": None,
            "experiment_id": "site_launch",
            "group_id": group,
            "store_id": None,
        }
    )
    rows.append(
        {
            "event_at": enrolled + dt.timedelta(hours=1),
            "user_id": user,
            "event": "session_end",
            "revenue": None,
            "experiment_id": None,
            "group_id": None,
            "store_id": None,
        }
    )
    if random.random() < (0.35 if group == "treatment" else 0.20):
        rows.append(
            {
                "event_at": enrolled + dt.timedelta(hours=2),
                "user_id": user,
                "event": "purchase",
                "revenue": round(random.uniform(10.0, 50.0), 2),
                "experiment_id": None,
                "group_id": None,
                "store_id": None,
            }
        )

# Background activity from users never exposed to the experiment at all
# -- still real site volume during the window, so sitewide() must count
# it in the site total even though these users have no enrollment row.
for d in range(30):
    rows.append(
        {
            "event_at": dt.datetime(2025, 3, 1, 12, 0) + dt.timedelta(days=d),
            "user_id": f"bg{d:03d}",
            "event": "purchase",
            "revenue": round(random.uniform(10.0, 50.0), 2),
            "experiment_id": None,
            "group_id": None,
            "store_id": None,
        }
    )
    rows.append(
        {
            "event_at": dt.datetime(2025, 3, 1, 12, 0) + dt.timedelta(days=d),
            "user_id": f"bg{d:03d}",
            "event": "session_end",
            "revenue": None,
            "experiment_id": None,
            "group_id": None,
            "store_id": None,
        }
    )
events = pl.DataFrame(rows).with_columns(
    pl.when(pl.col("event") == "session_end")
    .then(1)
    .otherwise(None)
    .cast(pl.Int64)
    .alias("session_count"),
    pl.col("store_id").cast(pl.Utf8),
)

con = ibis.duckdb.connect()  # in-memory; swap for Snowflake/BigQuery/Postgres
con.create_database("analytics")
con.create_table("event_log", events.to_arrow(), database="analytics")

with warnings.catch_warnings():
    warnings.simplefilter("ignore", IncrementWarning)
    experiment, resolver = from_definitions("definitions/", con, experiment_name="site_launch")

sum_result = sitewide(experiment, resolver, "revenue_per_user")
print(f"observed site volume:   {sum_result.site_total_volume:,.2f}")
print(f"counterfactual (V0):    {sum_result.baseline_volume:,.2f}")
print(f"per-unit lift (delta):  {sum_result.delta:+.2f}")
print(
    f"ship-to-all impact:     {sum_result.absolute_impact:+,.2f} "
    f"[{sum_result.absolute_impact_lb:+,.2f}, {sum_result.absolute_impact_ub:+,.2f}]"
)
print(
    f"relative to baseline:   {sum_result.relative_impact:+.2%} "
    f"[{sum_result.relative_impact_lb:+.2%}, {sum_result.relative_impact_ub:+.2%}]"
)

ratio_result = sitewide(experiment, resolver, "revenue_per_session")
print(f"baseline ratio:         {ratio_result.baseline_ratio:.4f}")
print(f"shipped ratio:          {ratio_result.shipped_ratio:.4f}")
print(
    f"ship-to-all impact:     {ratio_result.absolute_impact:+.4f} "
    f"[{ratio_result.absolute_impact_lb:+.4f}, {ratio_result.absolute_impact_ub:+.4f}]"
)
```

<!-- output -->
```text
observed site volume:   2,095.10
counterfactual (V0):    1,502.88
per-unit lift (delta):  +5.92
ship-to-all impact:     +1,184.44 [+385.90, +1,982.98]
relative to baseline:   +78.81% [+4.74%, +152.88%]
baseline ratio:         6.9901
shipped ratio:          12.4992
ship-to-all impact:     +5.5090 [+1.7949, +9.2232]
```

`revenue_per_user` returns a `SitewideImpact`; `revenue_per_session` returns a
`SitewideRatioImpact`. Both intervals exclude zero here: the lift is large
enough, relative to this toy dataset's small enrolled population, to move the
whole-site total by a detectable amount.

### A cluster-randomized rollout

The second experiment, `store_rollout`, declares `cluster: store_id`. The
store ID comes from exposure (`page_view`) rows, as described in
[The data model](data-model.md#randomization-grain-vs-analysis-grain).
Background purchases keep `data_as_of` through the experiment calendar window.

```python
random.seed(7)
start2 = dt.datetime(2025, 4, 1, 9, 0)
cluster_rows = []
n_stores_per_arm = 25
users_per_store = 8
store_i = 0
for arm_name in ("control", "treatment"):
    for s in range(n_stores_per_arm):
        store_id = f"store_{arm_name}_{s:03d}"
        store_shock = random.uniform(-3.0, 3.0)  # per-store correlated shock
        for u in range(users_per_store):
            user = f"su_{store_id}_{u:02d}"
            enrolled = start2 + dt.timedelta(days=(store_i + u) % 10)
            cluster_rows.append(
                {
                    "event_at": enrolled,
                    "user_id": user,
                    "event": "page_view",
                    "revenue": None,
                    "experiment_id": "store_rollout",
                    "group_id": arm_name,
                    "store_id": store_id,
                }
            )
            cluster_rows.append(
                {
                    "event_at": enrolled + dt.timedelta(hours=1),
                    "user_id": user,
                    "event": "session_end",
                    "revenue": None,
                    "experiment_id": None,
                    "group_id": None,
                    "store_id": None,
                }
            )
            buy_rate = 0.20 + (0.15 if arm_name == "treatment" else 0.0)
            if random.random() < buy_rate:
                amount = round(random.uniform(10.0, 50.0) + store_shock, 2)
                cluster_rows.append(
                    {
                        "event_at": enrolled + dt.timedelta(hours=2),
                        "user_id": user,
                        "event": "purchase",
                        "revenue": amount,
                        "experiment_id": None,
                        "group_id": None,
                        "store_id": None,
                    }
                )
        store_i += 1

# Background purchases cover the experiment's full calendar window.
for d in range(30):
    cluster_rows.append(
        {
            "event_at": start2 + dt.timedelta(days=d),
            "user_id": f"sbg{d:03d}",
            "event": "purchase",
            "revenue": round(random.uniform(10.0, 50.0), 2),
            "experiment_id": None,
            "group_id": None,
            "store_id": None,
        }
    )
    cluster_rows.append(
        {
            "event_at": start2 + dt.timedelta(days=d),
            "user_id": f"sbg{d:03d}",
            "event": "session_end",
            "revenue": None,
            "experiment_id": None,
            "group_id": None,
            "store_id": None,
        }
    )

cluster_events = pl.DataFrame(cluster_rows).with_columns(
    pl.when(pl.col("event") == "session_end")
    .then(1)
    .otherwise(None)
    .cast(pl.Int64)
    .alias("session_count"),
    pl.col("store_id").cast(pl.Utf8),
)
all_events = pl.concat([events, cluster_events.select(events.columns)])
con.create_table("event_log", all_events.to_arrow(), database="analytics", overwrite=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", IncrementWarning)
    cluster_experiment, cluster_resolver = from_definitions(
        "definitions/", con, experiment_name="store_rollout"
    )

with warnings.catch_warnings(record=True) as cluster_warnings:
    # This is the documented cluster-population advisory, not an error in
    # this example. Keep the capture local so unrelated warnings still fail.
    warnings.filterwarnings(
        "always",
        message=r"^sitewide under a declared cluster:",
        category=UserWarning,
    )
    csum = sitewide(cluster_experiment, cluster_resolver, "revenue_per_user")
    cratio = sitewide(cluster_experiment, cluster_resolver, "revenue_per_session")

assert len(cluster_warnings) == 2
assert all(
    str(item.message).startswith("sitewide under a declared cluster:") for item in cluster_warnings
)
print(f"n_clusters:              {csum.n_clusters}")
print(f"absolute_dof:            {csum.absolute_dof:g}")
print(f"relative_dof:            {csum.relative_dof:g}")
print(f"observed site volume:    {csum.site_total_volume:,.2f}")
print(f"counterfactual (V0):     {csum.baseline_volume:,.2f}")
print(f"per-unit lift (delta):   {csum.delta:+.2f}")
print(
    f"ship-to-all impact:      {csum.absolute_impact:+,.2f} "
    f"[{csum.absolute_impact_lb:+,.2f}, {csum.absolute_impact_ub:+,.2f}]"
)
print(
    f"relative to baseline:    {csum.relative_impact:+.2%} "
    f"[{csum.relative_impact_lb:+.2%}, {csum.relative_impact_ub:+.2%}]"
)

print(f"ratio n_clusters:        {cratio.n_clusters}")
print(f"ratio absolute_dof:      {cratio.absolute_dof:g}")
print(f"ratio relative_dof:      {cratio.relative_dof:g}")
print(f"baseline ratio:          {cratio.baseline_ratio:.4f}")
print(f"shipped ratio:           {cratio.shipped_ratio:.4f}")
print(
    f"ship-to-all impact:      {cratio.absolute_impact:+.4f} "
    f"[{cratio.absolute_impact_lb:+.4f}, {cratio.absolute_impact_ub:+.4f}]"
)
```

<!-- output -->
```text
n_clusters:              50
absolute_dof:            42.0245
relative_dof:            42.0245
observed site volume:    4,125.25
counterfactual (V0):     2,912.12
per-unit lift (delta):   +6.07
ship-to-all impact:      +2,426.26 [+1,127.91, +3,724.61]
relative to baseline:    +83.32% [+20.16%, +146.47%]
ratio n_clusters:        50
ratio absolute_dof:      42.0245
ratio relative_dof:      42.0245
baseline ratio:          7.0172
shipped ratio:           12.8636
ship-to-all impact:      +5.8464 [+2.7179, +8.9750]
```

50 total clusters (25 stores per arm) clear the 40-cluster floor, so no
`RuntimeWarning` fires. `n_clusters` is contrast-level metadata;
`absolute_dof` and `relative_dof` echo the degrees of freedom each interval
was actually cut at, not a single pooled figure.

??? info "Statistical details: why both references are 42.0245 here"
    `revenue_per_user`'s plain-metric absolute impact uses the two-component
    Welch–Satterthwaite reference, pairing the control and target mean
    variances with their own 24 cluster degrees of freedom. 48 would be the
    special case of equal variance contributions, not an automatic
    `target.n + control.n - 2` result; the unequal contributions here give
    42.0245. Relative impact and `revenue_per_session`'s absolute and
    relative impacts use the every-arm Welch–Satterthwaite reference instead
    (see "Degrees of freedom" in `increment.estimation.sitewide`'s module
    docstring): each variance term is paired with its own arm's cluster
    degrees of freedom. With only control and one target, those are 24 each,
    and the same unequal variance weights yield 42.0245. Co-enrolled arms add
    their own variance terms and cluster degrees of freedom to that
    reduction.

See the [API reference](../api.md) for `SitewideImpact` and
`SitewideRatioImpact`'s full field lists, and
[The data model](data-model.md) for the definitions-directory / DuckDB
setup this guide reuses.
