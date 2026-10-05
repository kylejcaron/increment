# Parity matrix coverage

Counts and trackers only. Every per-ingress outcome, reason and authority lives in
`tests/parity_harness/matrix.py` (`RULES`, `_SPECS`); `tests/test_parity_matrix.py` runs every cell.

Every count below was computed from `matrix.classify` over `matrix.iter_cells()` when this file was last changed; no test enforces the numbers. `tests/test_parity_matrix.py` enforces the dispositions themselves, cell by cell.

## Axes

| Axis | Values |
|---|---|
| metric | 14: mean, conversion, ratio, retention, quantile, total, active and `windowed_<base>` for each |
| view | run, breakout, daily, asof (`daily` = `run_daily` + `run_daily_lift`; `asof` = `run_asof` + `run_asof_lift`) |
| option | none, cuped, winsor_fixed, winsor_percentile, cluster, sequential, observational, ni_margin |
| day_boundary | utc, fixed_offset (`UTC-05:00`) |
| missing | error, zero, drop, impute |

Cells: 3,584. Ingress-cells (six constructors each): 21,504. Every cell is executed; none is collapsed.

## Cells per disposition

A cell's status is its most significant per-ingress verdict (`Disposition.status`).

| Status | Cells |
|---|---|
| supported | 522 |
| source_limited | 512 |
| construction_limited | 1,316 |
| not_expressible | 960 |
| unfinished | 274 |
| unsound | 0 |
| total | 3,584 |

## Ingress verdicts per status

Verdicts are per leg: a day-axis view has a value leg and a lift leg that refuse independently, so 5,376 legs carry 32,256 verdicts.

| Ingress | supported | source_limited | construction_limited | not_expressible | unfinished | unsound |
|---|---|---|---|---|---|---|
| from_definitions | 676 | 0 | 504 | 3,976 | 220 | 0 |
| from_unit_day_artifact | 672 | 0 | 500 | 3,976 | 228 | 0 |
| from_unit_summary | 102 | 1,536 | 1,544 | 1,824 | 370 | 0 |
| from_unit_panel | 480 | 920 | 1,940 | 1,632 | 404 | 0 |
| from_switchback_panel | 4 | 2,216 | 1,428 | 1,440 | 288 | 0 |
| from_moments | 144 | 1,582 | 1,628 | 1,728 | 294 | 0 |

Cells where, in at least one leg, an ingress runs and another does not (each non-runner carries a status, reason and authority): 522.

## Unfinished cells and trackers

Each tracker is a kata issue linked to t8tc; each cell below carries the code now raised.

| Tracker | Cells | Code now raised, per ingress | What is unfinished |
|---|---|---|---|
| `0f6d` | 64 | definitions: breakout.quantile; unit_day_artifact: query.builders.asof_group_summary_metric_type_not_implemented, readout.metric.quantile_grain, readout.observational.quantile; unit_panel: frame.asof.quantile_unsupported, readout.metric.quantile_grain | the same quantile x day-axis hazard raises the breakout code here, not the catalog's readout.metric.quantile_grain |
| `1cr4` | 48 | definitions: sequential.route.unsupported; unit_day_artifact: sequential.route.unsupported; unit_panel: sequential.route.unsupported; unit_summary: sequential.source.invalid | every observation window must be bounded by the common registered reveal window |
| `66mg` | 36 | definitions: arm.metric.quantile_cluster, arm.metric.quantile_cuped; moments: frame.metric.cuped_does_apply; switchback_panel: frame.metric.cuped_does_apply; unit_day_artifact: arm.metric.quantile_cluster, artifact.extension.invalid; unit_panel: frame.metric.cuped_does_apply; unit_summary: frame.metric.cuped_does_apply, source.frame.cluster_capability | a quantile has no mean to adjust |
| `6z2f` | 160 | moments: frame.metric.window_days_supported; switchback_panel: frame.metric.window_days_supported; unit_day_artifact: frame.metric.window_days_supported; unit_panel: frame.metric.window_days_supported; unit_summary: frame.metric.window_days_supported | the artifact route reads a windowed quantile through the frame declaration and refuses it, while the definitions route that published it runs |
| `r3bg` | 10 | definitions: readout.metric.quantile_alternative; moments: readout.metric.quantile_alternative; unit_day_artifact: readout.metric.quantile_alternative; unit_panel: readout.metric.quantile_alternative; unit_summary: readout.metric.quantile_alternative | a one-sided alternative (a non-inferiority margin) is not supported for quantile metrics yet |

A hazard that several routes refuse with different codes is unfinished on every route that raises it (`matrix._reconcile_hazards`); `check_disposition` rejects a diverging hazard with no tracker. Tracker `66mg` covers both quantile x CUPED (`arm.metric.quantile_cuped`, `frame.metric.cuped_does_apply`, `artifact.extension.invalid`) and quantile x cluster (`arm.metric.quantile_cluster`, `source.frame.cluster_capability`); `0f6d` and `1cr4` cover the quantile day-axis and unbounded-sequential hazards the same way.

An observational design has no quantile estimator. The readout seam (`increment/estimation/_readout_refusals.py::refuse_observational_quantile`) refuses `run()` with `readout.observational.quantile` on the definitions, artifact, unit-summary, unit-panel and portable (`from_moments`) ingresses. A windowed quantile reaches the seam on the warehouse routes only; the frame routes refuse its window declaration first (`6z2f`). Descriptive day-axis value legs do not call `validate_readout_adjustment`; the artifact's unwindowed `run_asof()` value leg raises `readout.observational.quantile` from its source guard and remains under `0f6d` with the other quantile x day-axis ordering discrepancies.

### Portable quantile cells

A quantile has no moments representation, so no producer exports a quantile cube. Every `from_moments` quantile verdict is therefore measured on a real scalar-moments source, not on a producer's refusal to export a quantile: `matrix_cases.py::_Ingress._quantile_moments` declares the cell's quantile for replay first, exports a real randomized scalar mean over the quantile's outcome column (carrying a cluster column or a sequential registration where the cell declares one; a unit-summary producer holds its construction-time checkpoint, so the export carries that checkpoint and is never recaptured at an as-of horizon), replays it with the cell's design/plan, and reads the cell's actual `run`/`run_breakout`/`run_daily`/`run_daily_lift`/`run_asof`/`run_asof_lift`. A verdict is attributed to the stage that raised it, measured per cell: the replay's quantile `MetricSpec` declaration (built first), the mean producer's construction, the producer's export, or `from_moments`. Unwindowed quantile, `error`/`zero`/`drop`, measured (`impute` is refused by the replay's quantile declaration, `frame.metric.missing_impute`, before any producer is built, on every option):

| Option | `run()` | `run_breakout()` | day-axis (4 methods) |
|---|---|---|---|
| none | `source.moments.unit_grain` (a cube holds no per-unit rows) | `facade.analysis.operation` | `facade.analysis.no_definitions` |
| ni_margin | `readout.metric.quantile_alternative` (`r3bg`) | `facade.analysis.operation` | `facade.analysis.no_definitions` |
| observational | `readout.observational.quantile` (estimator refusal) | `facade.analysis.operation` | `facade.analysis.no_definitions` |
| cluster | `source.moments.cluster_grain` (the producer cannot export a clustered cube) | same | same |
| sequential | `sequential.route.unsupported` from `from_moments` under `error`/`zero` (the exported construction-time checkpoint is replayed under a quantile declaration: scalar mean inference requires a mean, conversion or retention metric); `sequential.route.unsupported` under `drop` is raised earlier, by the mean producer's own registration (outcome-dependent row deletion), before any cube exists | same | same |
| cuped, winsorization, any window | declaration: `frame.metric.cuped_does_apply`, `frame.metric.winsorization_applies_type`, `frame.metric.window_days_supported` (the replay's own `MetricSpec`) | same | same |

Only `run()` can reach an estimator: breakout and day-axis requests are refused at the source gate before any estimator reads the cube, so they never carry the estimator refusals (`readout.observational.quantile`, `readout.metric.quantile_alternative`). The separate randomized quantile export still raises `source.frame.quantile_no_moments` (scenarios `audit-quantile-family` and `quantile_tied_rounded_outcomes`, `cases.py`); the matrix no longer records it for any cell. The scenario `observational_quantile_refused_at_the_readout_seam` exercises the observational `run()` leg on its own.

## Tiers and cost

| Tier | Cells | Wall time (`-n 2`, measured on an arm64 laptop) |
|---|---|---|
| fast | 2,552 | about 60 s: no ingress runs or reaches the request stage; each declaration either refuses with a code or is structurally absent (a schema or keyword the ingress cannot express, with the declared field named by the error) |
| slow | 1,032 | about 11 minutes: some ingress runs, or refuses only once a request is read |

The tiers count `test_cell` items, so they total the 3,584 cells; the unparameterized disposition test (`test_every_cell_is_dispositioned_and_its_split_explained`) is not a cell and is in the fast tier on its own. A cell is slow when any of its six ingresses builds data and runs or reads a request (`Disposition.all_verdicts`), not only a warehouse route.

Every cell builds and runs each of its ingresses itself; no result is shared between cells, so a cell never depends on test order or on a sibling having run. Cells that differ only in `missing` error versus zero therefore repeat the warehouse read; that repetition is the cost of independence.

## Collapse proofs

None. No cell is collapsed into a representative.

## Existing scenario cases

The 69 `PARITY_CASES` keep their own scenario assertions (sequential families, encouragement/LATE, priors, multiplicity roles, observational covariate shapes, the observational quantile refusal). The matrix does not map them to cells; it executes every cell directly. A row is compared on every public field (`model_dump()` of the emitted row), with 1e-9 relative tolerance and a 1e-9 absolute floor for floats, and exactly otherwise; see `runner.py` for the few path-naming fields it drops.

## Fixture choices that bound what a cell proves

- `missing=error` cells feed the dataframe routes null-free columns; the refusal of a null under `error` is pinned by `tests/test_frame_missing.py::test_null_metric_value_refuses_naming_both_fixes`. `zero` and `drop` cells feed NULLs for units with no purchase.
- `winsor_fixed` and `winsor_percentile` cells use all-positive outcomes (a percentile winsorization pilot refuses non-positive outcomes, `estimation.winsor.pilot_nonpositive_outcome`, pinned in `tests/estimation/test_winsor_bootstrap.py`).
- A panel conversion column carries one 0/1 per unit for `run`/`asof`/`breakout` (a total reads it once) and a per-day indicator for `daily`; one frame cannot serve both, so the frame is built per view.
- Edge units are exposed at 03:00Z (22:00 the previous day under `UTC-05:00`); retention and window cells carry events whose band or window day differs by boundary, so a route bucketing in UTC fails them.
- `observational` is declared IPTW with one numeric pre-exposure covariate; `sequential` uses `asymptotic_mean` (mean, ratio) or `always_valid` (conversion, retention) with a 50/50 allocation and no explicit registration; `ni_margin` is a relative margin on a guardrail; `cluster` is 40 clusters per arm of one or two units.
- The switchback sub-case uses one fixed schedule (8 units, 2 cycles, 2 periods, washout 1) and a shared frame carrying every metric column.
- A `from_moments` cube is exported by a producer and replayed. A retention or windowed mean, conversion or ratio metric reads dates, so the dataframe panel produces it; the panel refuses a pre-period covariate there, so a CUPED retention (unwindowed), windowed mean, windowed conversion or windowed ratio cube under `error`/`zero` is exported from `from_definitions` instead and the cell records what `from_moments` does with that cube. An unwindowed CUPED mean, conversion or ratio cube is exported from `from_unit_summary` as before. A quantile cube is never exported: see "Portable quantile cells" above. Windowed retention stops at declaration (`definition.retention.metric_window_days`) on every ingress before any cube exists. Under `drop` the warehouse route cannot declare the policy, so no producer exists and the panel's refusal stands.

## Unresolved remainder

- Option x option interactions (`PAIRS`, one ingress each) are outside the single `option` axis; not claimed.
- Sequential family variations (registered rosters, composed compliance, multi-arm), encouragement/LATE, priors and observational covariate missingness stay with the scenario cases and their own tests.
- AIPW/DML adjustment, absolute margins and one-sided alternatives other than the margin are not enumerated.
- Float tolerance is 1e-9 relative with a 1e-9 absolute floor; the matrix runs on DuckDB in memory. A divergence found only on PostgreSQL, Snowflake or BigQuery is outside this ticket (CONTRIBUTING, Backend verification).
- Refusal before data load is not asserted per cell; the gate order is pinned by `tests/test_refusal_uniqueness.py` and `tests/test_source_capabilities.py`.
