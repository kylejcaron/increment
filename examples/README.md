# Examples

Run these [marimo](https://marimo.io) notebooks locally; no external services
are needed. Start with **`analysis_from_a_dataframe.py`** for a first result,
or **`analysis_from_a_warehouse.py`** for YAML and SQL.

Tables show estimates first, with unchecked or incomplete evidence above them.
Expand **Table evidence and checks** for assignment checks, multiple-testing
plans, source references and result qualifications.

Run mode shows the analysis calls alongside their results. Use `marimo edit`
to inspect the full source, including imports, sample-data generation and plots.
The commands below run from this repository.

- **`analysis_from_a_dataframe.py`**: a guided dataframe tutorial covering
  a baseline readout, CUPED, an informative prior, a ratio metric, and
  dataframe-panel windows and retention. Needs the `tables` extra:

  ```bash
  uv run --extra tables marimo edit examples/analysis_from_a_dataframe.py
  ```

- **`power.py`**: design-time power analysis: given control-arm
  assumptions, solve for required sample size, achieved power at a fixed
  budget, and minimum detectable effect.

  ```bash
  uv run marimo edit examples/power.py
  ```

- **`ttest_benchmark.py`**: compares Increment’s simplest fixed-horizon
  mean analysis with an explicit log-scale z-test and Welch’s t-test. Needs
  the `tables` extra:

  ```bash
  uv run --extra tables marimo edit examples/ttest_benchmark.py
  ```

- **`cuped.py`**: runs one experiment's estimates twice, unadjusted and
  with CUPED, and reports the measured variance reduction. Guardrail bounds
  use the same one-sided confidence convention for both methods.
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
  query pipeline that reduces event rows to additive moments. Eight tabs inspect
  the pipeline outputs: live sessions load bounded previews on demand, while
  fresh HTML exports include precomputed previews. Needs the `demo` and `tables`
  extras:

  ```bash
  uv run --extra demo --extra tables marimo edit examples/data_model.py
  ```

- **`breakout.py`**: compare declared country segments and daily trends using
  the warehouse example's estimator. Needs `demo` and `tables`, including for
  daily tables with inline sparklines:

  ```bash
  uv run --extra demo --extra tables marimo edit examples/breakout.py
  ```

- **`dashboard.py`**: explore `checkout_redesign` or `express_checkout` through
  the same dashboard calls. The latter has separate assigned and triggered
  populations. Uses synthetic warehouse data; needs `demo` and `dashboard`:

  ```bash
  uv run --extra dashboard --extra demo marimo edit examples/dashboard.py
  ```

  `python examples/dashboard.py` also runs a script smoke check
  with the same extras installed; bare `uv run examples/dashboard.py`
  does not, because the notebook's `# /// script` metadata makes uv treat it
  as an isolated script environment.

- **`observational.py`**: compare a naive, confounded dataframe estimate with
  an `Observational` + IPTW-adjusted estimate against known synthetic truth.
  No warehouse needed; uses the `tables` extra:

  ```bash
  uv run --extra tables marimo edit examples/observational.py
  ```

- **`hte.py`**: estimate and validate conditional effects, then evaluate a
  targeting rule on simulated cohorts with known effects. A second cohort has
  no real heterogeneity: its apparently strong in-sample top group disappears
  under held-out re-estimation, and targeting is refused. Uses `estimate_cate`,
  `validate_cate` and `targeting_rule`; needs the `tables` extra:

  ```bash
  uv run --extra tables marimo edit examples/hte.py
  ```

## Encouragement design

`encouragement.py` demonstrates randomized encouragement with imperfect
uptake. It reports the intention-to-treat effect, the compliance first
stage, and the complier average causal effect under the declared exclusion
restriction, then monitors that effect over time for a staggered cohort as
each user's windows close. Run it when assignment is randomized but treatment
receipt is not.

```bash
uv run --extra tables marimo edit examples/encouragement.py
```

`_seed.py` is shared infrastructure; it fills the synthetic
`analytics.event_log` table the other notebooks analyse. It isn't a notebook
and isn't meant to be run directly.
