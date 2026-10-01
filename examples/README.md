# Examples

Runnable [marimo](https://marimo.io) notebooks exercising the public API end
to end. Most of these run against an in-memory DuckDB warehouse -- no
external services needed -- but `analysis_from_a_dataframe.py` and
`observational.py` need no warehouse at all. Each notebook's own intro cell
explains the scenario in detail; this is just a map of what's here.

- **`analysis_from_a_dataframe.py`** -- a guided dataframe tutorial covering
  a baseline readout, CUPED, an informative prior, a ratio metric, and
  dataframe-panel windows and retention. Needs the `tables` extra:

  ```bash
  uv run --extra tables marimo edit examples/analysis_from_a_dataframe.py
  ```

- **`power.py`** -- design-time power analysis: given control-arm
  assumptions, solve for required sample size, achieved power at a fixed
  budget, and minimum detectable effect.

  ```bash
  uv run marimo edit examples/power.py
  ```

- **`ttest_benchmark.py`** -- compares Increment’s simplest fixed-horizon
  mean analysis with an explicit log-scale z-test and Welch’s t-test. Needs
  the `tables` extra:

  ```bash
  uv run --extra tables marimo edit examples/ttest_benchmark.py
  ```

- **`cuped.py`** -- runs one experiment's estimates twice, unadjusted and
  with CUPED variance reduction, and compares interval widths side by side.
  Needs the `demo` and `tables` extras:

  ```bash
  uv run --extra demo --extra tables marimo edit examples/cuped.py
  ```

- **`analysis_from_a_warehouse.py`** — the full warehouse workflow: inspect
  semantic definitions, run a realistic checkout experiment, check allocation,
  render headline and segmented readouts, follow lift over time, and export
  moments for offline reuse. Needs the `demo` and `tables` extras:

  ```bash
  uv run --extra demo --extra tables marimo edit examples/analysis_from_a_warehouse.py
  ```

- **`data_model.py`** — a deeper tour of the normalized warehouse and the
  query pipeline that reduces event rows to additive moments. Needs the `demo`
  and `tables` extras:

  ```bash
  uv run --extra demo --extra tables marimo edit examples/data_model.py
  ```

- **`breakout.py`** -- per-segment lift breakdowns and per-day metric/lift
  trends: the same estimator `analysis_from_a_warehouse.py` uses, sliced by a
  declared breakout dimension (`country`) and by calendar day instead of
  collapsing the experiment into one top-line number. Needs the `demo` and
  `tables` extras for every table, including the per-day views with
  `coeftable`'s inline sparkline column:

  ```bash
  uv run --extra demo --extra tables marimo edit examples/breakout.py
  ```

- **`ab_testing_dashboard.py`** — reusable presentation helpers over one
  bound experiment: allocation health, the full CoefTable readout, optional
  metric time trends and segments, and a downloadable CSV. The renderers
  are the product; this notebook is one readable composition over them
  against the bundled `checkout_redesign` fixture. Needs the `demo` and
  `dashboard` extras:

  ```bash
  uv run --extra dashboard --extra demo marimo edit examples/ab_testing_dashboard.py
  ```

  `python examples/ab_testing_dashboard.py` also runs a script smoke check
  with the same extras installed; bare `uv run examples/ab_testing_dashboard.py`
  does not, because the notebook's `# /// script` metadata makes uv treat it
  as an isolated script environment.

- **`observational.py`** -- no-warehouse, straight from a dataframe like
  `analysis_from_a_dataframe.py`: a naive (confounded) comparison side by
  side with an `Observational` + IPTW-adjusted one, against a known synthetic
  truth, so the bias and the correction are visible in the same run. Needs
  the `tables` extra:

  ```bash
  uv run --extra tables marimo edit examples/observational.py
  ```

- **`hte.py`** -- heterogeneous treatment effects: `estimate_cate`,
  `validate_cate` and `targeting_rule` over a simulated cohort whose per-unit
  effect is known, so every estimate is printed beside its truth. The centre
  of the notebook is a second cohort with *no* heterogeneity at all, where
  the in-sample top group reads well above the truth and the held-out
  re-estimation does not -- which is why the targeting rule refuses to hand
  back a threshold unless the honest split earned one. Needs the `tables`
  extra:

  ```bash
  uv run --extra tables marimo edit examples/hte.py
  ```

## Encouragement design

`encouragement.py` demonstrates randomized encouragement with imperfect
uptake. It reports the intention-to-treat effect, the compliance first
stage, and the complier average causal effect under the declared exclusion
restriction. Run it when assignment is randomized but treatment receipt is
not.

```bash
uv run --extra tables marimo edit examples/encouragement.py
```

`_seed.py` is shared infrastructure -- it fills the synthetic
`analytics.event_log` table the other notebooks analyse. It isn't a notebook
and isn't meant to be run directly.
