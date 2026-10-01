# Sitewide impact

An experiment measures only the units it enrolled. `Analysis.sitewide()`
translates that exposed-population lift into one whole-site number: if the
lift had shipped to every enrolled unit, how much would the metric's total
have moved during the experiment's actual window? This is the business
impact that an enrolled per-unit lift alone cannot answer.

## The ship-to-all counterfactual

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
    `Analysis.sitewide()` raises `ValueError` rather than return one when
    the baseline is non-positive.

## Ratio metrics

A ratio metric (revenue per session, CTR) needs two lifts instead of
one: a numerator lift $\delta_{\text{num}}$ and a denominator lift
$\delta_{\text{den}}$, plus their covariance (they come from the same
per-arm sums, not independent draws). The counterfactual and ship-to-all
totals are built the same way as $V_0$ above, once per side:

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

`Analysis.sitewide()` reports ratio metrics' `relative_impact` as absolute
impact divided by the baseline ratio (`absolute_impact / baseline_ratio`).
Its uncertainty uses covariance-aware delta-method gradients for the
numerator and denominator lifts, including their covariance; the reported
intervals use the resulting relative-impact standard error.

## Window only -- no projection beyond it

`sitewide()` reports the impact over exactly the experiment's declared
enrollment window (`experiment.start .. experiment.end`) and refuses to
extrapolate it into a longer or different horizon: no "annualized"
number, no per-day rate multiplied out to a quarter or a year.

That refusal is not conservatism for its own sake. As
[Metric types](metric-types.md#unbounded-retention-doesnt-keep-getting-more-precise)
shows for retention, a real, *constant* per-unit effect does not read the
same at every observation horizon: watched longer, later and
weakly-informative observations dilute the running estimate, to the
point that a genuinely constant effect can look statistically
undetectable purely from observing further out. If an effect's own
statistical signature is not stable across horizons, assuming its
*size* stays flat past the window you actually measured is exactly the
same unearned assumption. `sitewide()` reports what happened in the
window that was run, not a projection of what might happen in a window
that wasn't.

## Multi-arm experiments

With a second treatment arm enrolled, $V$ carries *both* arms' lifts:
$V = V_0 + \delta_A n_A + \delta_B n_B$. Netting out only the arm being
scored would leave the other arm's lift counted as baseline, so every
enrolled non-control arm is subtracted:

$$
V_0 = V - \sum_i \delta_i\,n_i
$$

and $N_{\text{exp}}$ counts every enrolled unit, not just two arms' worth.
The reported number therefore answers: if this arm had shipped to every
enrolled unit, and no other treatment arm had ever been exposed, how much
higher would the site-window total be than under no treatment at all? Both
worlds it compares are worlds a ship decision can actually choose -- a unit
sits in exactly one arm, so "this arm ships *and* the other keeps running"
is not a realizable alternative for one shared population.

`sitewide()` scores one arm per call: pass `arm="treatment_b"` to say
which. It is optional only when a single non-control arm is enrolled, and
the co-enrolled arms are netted out of the baseline either way.

The arms' lifts are not independent -- every $\delta_i$ subtracts the same
control mean, so $\mathrm{Cov}(\delta_i, \delta_j) = \mathrm{Var}(\bar
y_C) = \mathrm{Var}(y_C)/n_C$, which is *positive*. Absolute impact is
unaffected (it stays linear in the scored arm's $\delta$ alone), but the
relative impact's interval carries that covariance through the shared
baseline.

## Cluster-randomized experiments

`Experiment.cluster` (see [The data model](data-model.md#randomization-grain-vs-analysis-grain))
is recognized here as elsewhere in the library: it identifies the stores,
markets, or other units whose assignment actually changed, not the
individually measured units. `Analysis.sitewide()` allows `mean`,
`conversion`, `retention`, and `ratio` metrics; any other declared type
raises `CapabilityError` naming the metric type and cluster. In practice,
`mean`, `conversion`, and `ratio` produce clustered readings.
`RetentionMetric` still refuses for the site-volume reason in
["What refuses"](#what-refuses), whether or not the experiment is clustered.

The per-unit lift $\delta$ and its variance are no longer a plain arm mean
and Welch SE. Each arm becomes a ratio of cluster totals: outcome summed per
cluster divided by units summed per cluster. This is the same cluster-robust
reduction used by `run()` and other cluster-aware estimators. The point
estimate matches a unit-grain mean; only the standard error changes because
units within a cluster are not independent draws.

Inference also moves from a Normal reference to a $t$ reference. The
applicable degrees of freedom depend on what is reported:

- **Absolute impact for a plain (non-ratio) metric** depends only on the
  control and target arm -- no other enrolled arm ever enters its
  variance -- so its interval uses the plain pairwise $t$ reference
  between those two arms' cluster counts, the same as a two-arm
  experiment would.
- **Relative impact, and a ratio metric's absolute impact,** both mix
  contributions from every enrolled arm. A co-enrolled arm's lift enters the
  shared baseline used by either quantity, so more arms add sampling variance
  to the standard error, not necessarily a more conservative reference
  distribution. The combined estimate uses a Satterthwaite correction across
  every contributing arm's cluster count (each arm's own $K_i - 1$,
  including control), weighted by that arm's variance contribution. It is not
  the plain pairwise reference or a simple average of the arms' individual
  degrees of freedom. The result falls between the smallest contributing
  arm's $K_i - 1$ and the sum of all contributing arms' $K_i - 1$. It can
  fall below the target-control pairwise degrees of freedom. In the guide's
  balanced single-arm example, control and target each have 24 degrees of
  freedom, so the pairwise reference is 48 and the Satterthwaite combination
  is about 42; a co-enrolled arm with fewer clusters can lower it further.

!!! note
    Declaring a cluster never moves the point estimate. It changes the
    standard error and, for a multi-arm quantity, the reference distribution's
    degrees of freedom; those changes need not move in the same direction.

### How many clusters is enough

Cluster count is a qualification, not an arbitrary admission floor. Structurally
valid contrasts use a qualified Normal/$t$ reference; below 40 total clusters,
`RuntimeWarning` names the over-rejection risk, and borderline significance
should be treated as fragile. Each enrolled arm still needs at least two
clusters so its between-cluster variance is defined. More clusters buy real
precision; adding units inside each cluster adds little.

!!! warning "Enrolled units must exhaust the treated population"
    Cluster randomization usually treats the whole cluster -- every
    register at a store, every session in a market -- not just the units
    that happened to trigger an enrollment event. `sitewide()` assumes
    enrolled units exhaust the treated population.

    That assumption breaks concretely whenever a treated cluster has
    real activity from someone who was never exposure-logged there: a
    returning customer who did not view the enrolling page this window,
    but still bought something at the treated store. That purchase is
    real treated-population volume, but `sitewide()` has no enrollment
    row to attribute it to, so it lands in the counterfactual baseline as
    though it were untouched by treatment.

    The bias has a fixed direction: the baseline is biased upward, and the
    reported ship-to-all impact is understated, never overstated. The library
    does not implement a cluster-restricted site total or spillover correction;
    this is a known limitation, not behavior that `sitewide()` works around.

## What refuses

- **`RetentionMetric` and `QuantileMetric`.** Site volume is undefined
  for both: a retention outcome is anchored to each unit's own exposure
  time (the observation band), not a raw event stream that sums
  independently of exposure, and a quantile is a distributional
  statistic that is not additive across events. Both raise
  `CapabilityError` before any moments are even built.
- **A clustered experiment's metric type.** `Analysis.sitewide()`
  recognizes a declared cluster for `mean`, `conversion`, `retention`,
  and `ratio` metrics; any other type raises `CapabilityError` naming
  the metric's type and the cluster. In practice that gate never turns
  away anything reachable today: `TotalMetric` and `ActiveMetric` cannot
  be declared on any native experiment's `metrics`/`guardrails` at all,
  clustered or not -- a load-time restriction that predates this feature
  (they have no per-unit variance the estimation engine can serve;
  `increment.Report` is the intended path for them instead).
  `RetentionMetric` still refuses for the site-volume reason above
  regardless of `cluster`. `mean`, `conversion`, and `ratio` are
  therefore the metric types that actually produce a clustered sitewide
  reading.
- **A `MomentSource`-backed `Analysis`.** `from_unit_summary`,
`from_unit_panel`, and `from_moments` all start from a pre-melted unit
summary or a precomputed moments cube -- none retains the raw event stream
`sitewide()` needs to sum a whole-site total over. Only a native instance
(`Analysis(experiment_name, definitions_path, con)` /
`Analysis.from_definitions(...)`) supports it; the others raise
`CapabilityError` naming `from_definitions` as the fix.
- **An ambiguous arm.** With several non-control arms enrolled and no
  `arm=` given, `sitewide()` raises `ValueError` naming the candidates
  rather than silently picking one for you.

## Worked example

`sitewide()` needs the native (`from_definitions`) path: a definitions
directory plus a DuckDB connection, same as
[The data model](data-model.md#running-it). The event log below carries
two kinds of "off-experiment" activity on purpose -- a `page_view`
marks enrollment, `purchase` is the revenue metric, `session_end` backs
a ratio metric's denominator, and a block of background purchases from
users who were never exposed at all still count toward the site total:

<!-- invisible-code-block: python
from pathlib import Path

Path("definitions").mkdir(exist_ok=True)
Path("definitions/fact_sources.yaml").write_text("""\
dialect: duckdb

fact_sources:
  - name: event_log
    sql: |
      SELECT * FROM analytics.event_log
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
        column: null
        description: A session ended (occurrence-only event)
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
    description: "Total purchase revenue per user over 14 days"
    entity: user_id
    preferred_direction: increase
    fact: purchase
    aggregation: sum
    window_days: 14

  - type: ratio
    name: revenue_per_session
    description: "Revenue per session (ratio of sums)"
    entity: user_id
    preferred_direction: increase
    numerator:
      fact: purchase
      aggregation: sum
      window_days: 14
    denominator:
      fact: session_end
      aggregation: count
      window_days: 14
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

from increment import Analysis

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
events = pl.DataFrame(rows).with_columns(pl.col("store_id").cast(pl.Utf8))

con = ibis.duckdb.connect()  # in-memory; swap for Snowflake/BigQuery/Postgres
con.create_database("analytics")
con.create_table("event_log", events.to_arrow(), database="analytics")

analysis = Analysis.from_definitions("site_launch", "definitions/", con)

sum_result = analysis.sitewide("revenue_per_user")
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

ratio_result = analysis.sitewide("revenue_per_session")
print(f"baseline ratio:         {ratio_result.baseline_ratio:.4f}")
print(f"shipped ratio:          {ratio_result.shipped_ratio:.4f}")
print(
    f"ship-to-all impact:     {ratio_result.absolute_impact:+.4f} "
    f"[{ratio_result.absolute_impact_lb:+.4f}, {ratio_result.absolute_impact_ub:+.4f}]"
)
```

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

`revenue_per_user` returns a `SitewideImpact`; `revenue_per_session`, a
ratio metric, returns a `SitewideRatioImpact` with the numerator and
denominator fields described in "Ratio metrics" above. Both intervals exclude
zero here: the lift is large enough, relative to this toy dataset's small
enrolled population, to move the whole-site total by a statistically
detectable amount.

### A cluster-randomized rollout

The second experiment, `store_rollout`, declares `cluster: store_id`.
The store ID comes from exposure (`page_view`) rows, as described in
[The data model](data-model.md#randomization-grain-vs-analysis-grain).
Background purchases keep `data_as_of` past every enrolled store's
14-day window close.

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

# Background purchases past every store's 14-day window close, so the
# fact table's data_as_of does not censor the whole experiment.
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

all_events = pl.concat([events, pl.DataFrame(cluster_rows)])
con.create_table("event_log", all_events.to_arrow(), database="analytics", overwrite=True)

cluster_analysis = Analysis.from_definitions("store_rollout", "definitions/", con)

with warnings.catch_warnings(record=True) as cluster_warnings:
    # This is the documented cluster-population advisory, not an error in
    # this example. Keep the capture local so unrelated warnings still fail.
    warnings.filterwarnings(
        "always",
        message=r"^sitewide under a declared cluster:",
        category=UserWarning,
    )
    csum = cluster_analysis.sitewide("revenue_per_user")
    cratio = cluster_analysis.sitewide("revenue_per_session")

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

```text
n_clusters:              50
absolute_dof:            48
relative_dof:            42.0245
observed site volume:    4,125.25
counterfactual (V0):     2,912.12
per-unit lift (delta):   +6.07
ship-to-all impact:      +2,426.26 [+1,132.68, +3,719.84]
relative to baseline:    +83.32% [+20.16%, +146.47%]
ratio n_clusters:        50
ratio absolute_dof:      42.0245
ratio relative_dof:      42.0245
baseline ratio:          7.0172
shipped ratio:           12.8636
ship-to-all impact:      +5.8464 [+2.7179, +8.9750]
```

50 total clusters (25 stores per arm) clear the 40-cluster floor
cleanly, so no `RuntimeWarning` fires here. `n_clusters` is echoed as
contrast-level metadata; `absolute_dof`/`relative_dof` echo the dof
each interval above was actually cut at, not a single pooled figure.
`revenue_per_user`'s absolute impact uses the plain pairwise reference
(`target.n + control.n - 2`) described above -- with a single
non-control arm enrolled, `absolute_dof` lands at 48 exactly, since
there is no other arm to separate it from. Relative impact and
`revenue_per_session`'s absolute impact use the every-arm-aware
Satterthwaite reference instead (see "Degrees of freedom" in
`increment.estimation.sitewide`'s module docstring for the exact
reduction): each variance term is paired with its own arm's cluster
count minus 1 -- 24 for control and 24 for target here -- so
`relative_dof`/`ratio absolute_dof` land between 24 and their sum of 48,
and below the pairwise 48 whenever the two terms carry unequal variance
weight, as they visibly do at about 42. That holds even with a single
non-control arm enrolled; a co-enrolled arm only adds its own
$K_i - 1$ to the mix.

See the [API reference](../api.md) for `SitewideImpact` and
`SitewideRatioImpact`'s full field lists, and
[The data model](data-model.md) for the definitions-directory / DuckDB
setup this guide reuses.
