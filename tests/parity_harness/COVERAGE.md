# Parity matrix coverage

Counts and trackers only. Every per-ingress outcome, reason and authority lives in
`tests/parity_harness/matrix.py` (`RULES`, `_SPECS`); `tests/test_parity_matrix.py` runs every cell.

Last verified at commit 0da41f2.

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
| supported | 450 |
| source_limited | 512 |
| construction_limited | 1,386 |
| not_expressible | 960 |
| unfinished | 276 |
| unsound | 0 |
| total | 3,584 |

## Ingress verdicts per status

| Ingress | supported | source_limited | construction_limited | not_expressible | unfinished | unsound |
|---|---|---|---|---|---|---|
| from_definitions | 420 | 0 | 436 | 2,648 | 80 | 0 |
| from_unit_day_artifact | 420 | 0 | 432 | 2,648 | 84 | 0 |
| from_unit_summary | 102 | 984 | 1,062 | 1,216 | 220 | 0 |
| from_unit_panel | 276 | 628 | 1,416 | 1,088 | 176 | 0 |
| from_switchback_panel | 4 | 1,476 | 984 | 960 | 160 | 0 |
| from_moments | 112 | 950 | 1,210 | 1,152 | 160 | 0 |

Cells where at least one ingress runs and another does not (each non-runner carries a status, reason and authority): 454.

## Unfinished cells and trackers

Each tracker is a kata issue linked to t8tc; each cell below carries the code now raised.

| Tracker | Cells | Code now raised, per ingress | What is unfinished |
|---|---|---|---|
| `0f6d` | 64 | definitions: breakout.quantile; unit_day_artifact: query.builders.asof_group_summary_metric_type_not_implemented; unit_panel: frame.asof.quantile_unsupported | the same quantile x day-axis hazard raises the breakout code here, not the catalog's readout.metric.quantile_grain |
| `1cr4` | 48 | unit_summary: sequential.source.invalid | an unbounded-window sequential request raises a source code here while the definitions and panel routes raise sequential.route.unsupported |
| `66mg` | 32 | unit_day_artifact: artifact.extension.invalid | publishing the CUPED extension for a quantile raises an artifact code instead of the arm-compatibility code the other routes raise |
| `6z2f` | 160 | moments: frame.metric.window_days_supported; switchback_panel: frame.metric.window_days_supported; unit_day_artifact: frame.metric.window_days_supported; unit_panel: frame.metric.window_days_supported; unit_summary: frame.metric.window_days_supported | a windowed quantile is declared by the warehouse model and read from per-unit totals, yet refused on the frame path with no stated statistical reason |
| `r3bg` | 10 | definitions: readout.metric.quantile_alternative; unit_day_artifact: readout.metric.quantile_alternative; unit_panel: readout.metric.quantile_alternative; unit_summary: readout.metric.quantile_alternative | a one-sided alternative (a non-inferiority margin) is not supported for quantile metrics yet |
| `t8ae` | 10 | definitions: source.native.operation; unit_summary: source.frame.quantile_no_moments | an adjusted quantile reads per-unit rows on the panel and artifact routes, but the native source refuses the moments operation instead |

## Tiers and cost

| Tier | Cells | Wall time (12 cores, `-n auto`) |
|---|---|---|
| fast | 2,725 | about 20 s: no warehouse route reads data |
| slow | 860 | about 75 s: a warehouse route publishes or reads a request |

A warehouse route is memoised per process on its exact inputs (the digest of its Definitions payload, view, option and artifact extensions). Cells differing only in `missing` error versus zero declare the same payload, share an `xdist_group` and read once. This reuses identical work; it collapses no cell.

## Collapse proofs

None. No cell is collapsed into a representative.

## Existing scenario cases

The 68 `PARITY_CASES` are unchanged and keep their own scenario assertions (sequential families, encouragement/LATE, priors, multiplicity roles, observational covariate shapes). The matrix does not map them to cells; it executes every cell directly.

## Fixture choices that bound what a cell proves

- `missing=error` cells feed the dataframe routes null-free columns; the refusal of a null under `error` is pinned by `tests/test_frame_missing.py::test_null_metric_value_refuses_naming_both_fixes`. `zero` and `drop` cells feed NULLs for units with no purchase.
- `winsor_fixed` and `winsor_percentile` cells use all-positive outcomes (a percentile winsorization pilot refuses non-positive outcomes, `estimation.winsor.pilot_nonpositive_outcome`, pinned in `tests/estimation/test_winsor_bootstrap.py`).
- A panel conversion column carries one 0/1 per unit for `run`/`asof`/`breakout` (a total reads it once) and a per-day indicator for `daily`; one frame cannot serve both, so the frame is built per view.
- Edge units are exposed at 03:00Z (22:00 the previous day under `UTC-05:00`); retention and window cells carry events whose band or window day differs by boundary, so a route bucketing in UTC fails them.
- `observational` is declared IPTW with one numeric pre-exposure covariate; `sequential` uses `asymptotic_mean` (mean, ratio) or `always_valid` (conversion, retention) with a 50/50 allocation and no explicit registration; `ni_margin` is a relative margin on a guardrail; `cluster` is 40 clusters per arm of one or two units.
- The switchback sub-case uses one fixed schedule (8 units, 2 cycles, 2 periods, washout 1) and a shared frame carrying every metric column.

## Unresolved remainder

- Option x option interactions (`PAIRS`, one ingress each) are outside the single `option` axis; not claimed.
- Sequential family variations (registered rosters, composed compliance, multi-arm), encouragement/LATE, priors and observational covariate missingness stay with the 68 scenario cases and their own tests.
- AIPW/DML adjustment, absolute margins and one-sided alternatives other than the margin are not enumerated.
- Tolerance is the runner's 1e-9 relative; the matrix runs on DuckDB in memory. A divergence found only on PostgreSQL, Snowflake or BigQuery is outside this ticket (CONTRIBUTING, Backend verification).
- Refusal before data load is not asserted per cell; the gate order is pinned by `tests/test_refusal_uniqueness.py` and `tests/test_source_capabilities.py`.
