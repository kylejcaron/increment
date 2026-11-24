# Capabilities by entry point

This page records, for each of the six experiment builders, which capabilities run and which refuse.
The [Method compatibility reference](../guides/compatibility.md) is organised by metric type and method; neither page restates the other's cells.
Where they disagree, the probe and harness results govern: `scripts/probe_capability_table.py`, `tests/test_composition_matrix.py` and `tests/parity_harness/matrix.py`.

## What runs where

A capability is a claim only when its path is named. Six builders feed
`run(experiment, resolver)`, in three families: **dataframe**
(`from_frame(..., shape="summary")`, one row per unit;
`from_frame(..., shape="panel")`, one row per unit per day;
`from_switchback_frame`); **warehouse** (`from_definitions`, compiling from raw
event facts; `from_unit_day_artifact`, reading unit × day relations previously
published into the warehouse); and **portable** (`from_moments`, a file of
pre-reduced moments exported from any of the others).

The tables project recorded consumer and parity contracts. Cells marked
`not measured` or `no disposition` are not capability claims; regenerating the
page does not execute its evidence. Matched-arm parity uses unit summary as its
oracle. Switchback estimates a different quantity (a fixed-horizon contrast
over a switchback schedule), so it is exercised independently rather than
compared row for row with that oracle.

Regenerate the evidence-backed tables with `uv run python scripts/render_capability_table.py` (or check them with `uv run python scripts/render_capability_table.py --check`). A bounded behavioral probe is `uv run --extra demo --extra tables --extra dashboard python scripts/probe_capability_table.py --case encouragement_margin_guardrail`; it exercises only that named parity case, not the full corpus.

<!-- capability-table:ingresses:start -->
Select a status for its conditions, codes and reason. A conditional cell is not a claim about every request.

| Capability | Unit summary | Unit panel | Switchback panel | Definitions | Unit-day artifact | Portable moments |
|---|---|---|---|---|---|---|
| Shared definition: completed-order ratio | [not comparable](#capability-ingresses-shared-warehouse-metric-from_unit_summary) | [not comparable](#capability-ingresses-shared-warehouse-metric-from_unit_panel) | [not comparable](#capability-ingresses-shared-warehouse-metric-from_switchback_panel) | [supported](#capability-ingresses-shared-warehouse-metric-from_definitions) | [supported](#capability-ingresses-shared-warehouse-metric-from_unit_day_artifact) | [not comparable](#capability-ingresses-shared-warehouse-metric-from_moments) |
| Declared + exploratory metrics in one physical capture | [refused (conditional)](#capability-ingresses-shared-capture-exploratory-from_unit_summary) | [refused (conditional)](#capability-ingresses-shared-capture-exploratory-from_unit_panel) | [refused (conditional)](#capability-ingresses-shared-capture-exploratory-from_switchback_panel) | [supported (conditional)](#capability-ingresses-shared-capture-exploratory-from_definitions) | [refused (conditional)](#capability-ingresses-shared-capture-exploratory-from_unit_day_artifact) | [refused (conditional)](#capability-ingresses-shared-capture-exploratory-from_moments) |
| Whole-window + daily scopes in one physical capture | [refused (conditional)](#capability-ingresses-shared-capture-multiview-from_unit_summary) | [supported (conditional)](#capability-ingresses-shared-capture-multiview-from_unit_panel) | [refused (conditional)](#capability-ingresses-shared-capture-multiview-from_switchback_panel) | [supported (conditional)](#capability-ingresses-shared-capture-multiview-from_definitions) | [supported (conditional)](#capability-ingresses-shared-capture-multiview-from_unit_day_artifact) | [refused (conditional)](#capability-ingresses-shared-capture-multiview-from_moments) |
| Mean CUPED, fixed horizon | [supported](#capability-ingresses-mean-cuped-fixed-horizon-from_unit_summary) | [supported](#capability-ingresses-mean-cuped-fixed-horizon-from_unit_panel) | [refused](#capability-ingresses-mean-cuped-fixed-horizon-from_switchback_panel) | [supported](#capability-ingresses-mean-cuped-fixed-horizon-from_definitions) | [supported](#capability-ingresses-mean-cuped-fixed-horizon-from_unit_day_artifact) | [supported](#capability-ingresses-mean-cuped-fixed-horizon-from_moments) |
| Ratio CUPED, fixed horizon | [supported](#capability-ingresses-ratio-cuped-fixed-horizon-from_unit_summary) | [supported](#capability-ingresses-ratio-cuped-fixed-horizon-from_unit_panel) | [refused](#capability-ingresses-ratio-cuped-fixed-horizon-from_switchback_panel) | [supported](#capability-ingresses-ratio-cuped-fixed-horizon-from_definitions) | [supported](#capability-ingresses-ratio-cuped-fixed-horizon-from_unit_day_artifact) | [supported](#capability-ingresses-ratio-cuped-fixed-horizon-from_moments) |
| Mixed-covariate CUPED, fixed-horizon mean/ratio | [supported (conditional)](#capability-ingresses-mixed-covariate-cuped-fixed-horizon-from_unit_summary) | [supported (conditional)](#capability-ingresses-mixed-covariate-cuped-fixed-horizon-from_unit_panel) | [source limited (conditional)](#capability-ingresses-mixed-covariate-cuped-fixed-horizon-from_switchback_panel) | [supported (conditional)](#capability-ingresses-mixed-covariate-cuped-fixed-horizon-from_definitions) | [supported (conditional)](#capability-ingresses-mixed-covariate-cuped-fixed-horizon-from_unit_day_artifact) | [supported (conditional)](#capability-ingresses-mixed-covariate-cuped-fixed-horizon-from_moments) |
| Ratio denominator-precision advisory, fixed horizon | [supported (conditional)](#capability-ingresses-ratio-denominator-advisory-fixed-horizon-from_unit_summary) | [supported (conditional)](#capability-ingresses-ratio-denominator-advisory-fixed-horizon-from_unit_panel) | [refused](#capability-ingresses-ratio-denominator-advisory-fixed-horizon-from_switchback_panel) | [supported (conditional)](#capability-ingresses-ratio-denominator-advisory-fixed-horizon-from_definitions) | [supported (conditional)](#capability-ingresses-ratio-denominator-advisory-fixed-horizon-from_unit_day_artifact) | [supported (conditional)](#capability-ingresses-ratio-denominator-advisory-fixed-horizon-from_moments) |
| Informative Normal/Student-t/mixture log-lift priors, fixed-horizon mean/ratio/conversion/mean CUPED | [supported (conditional)](#capability-ingresses-informative-log-lift-priors-from_unit_summary) | [supported (conditional)](#capability-ingresses-informative-log-lift-priors-from_unit_panel) | [not comparable (conditional)](#capability-ingresses-informative-log-lift-priors-from_switchback_panel) | [supported (conditional)](#capability-ingresses-informative-log-lift-priors-from_definitions) | [supported (conditional)](#capability-ingresses-informative-log-lift-priors-from_unit_day_artifact) | [supported (conditional)](#capability-ingresses-informative-log-lift-priors-from_moments) |
| Relative Normal/Student-t/mixture priors, fixed-horizon mean/ratio/conversion/mean CUPED | [supported (conditional)](#capability-ingresses-informative-relative-lift-priors-from_unit_summary) | [supported (conditional)](#capability-ingresses-informative-relative-lift-priors-from_unit_panel) | [not comparable (conditional)](#capability-ingresses-informative-relative-lift-priors-from_switchback_panel) | [supported (conditional)](#capability-ingresses-informative-relative-lift-priors-from_definitions) | [supported (conditional)](#capability-ingresses-informative-relative-lift-priors-from_unit_day_artifact) | [supported (conditional)](#capability-ingresses-informative-relative-lift-priors-from_moments) |
| Relative Normal prior on breakout rows | [supported](#capability-ingresses-relative-prior-breakout-from_unit_summary) | [supported](#capability-ingresses-relative-prior-breakout-from_unit_panel) | [not comparable](#capability-ingresses-relative-prior-breakout-from_switchback_panel) | [supported](#capability-ingresses-relative-prior-breakout-from_definitions) | [supported](#capability-ingresses-relative-prior-breakout-from_unit_day_artifact) | [refused](#capability-ingresses-relative-prior-breakout-from_moments) |
| Relative Normal prior on mature retention breakouts | [refused](#capability-ingresses-relative-prior-retention-from_unit_summary) | [supported](#capability-ingresses-relative-prior-retention-from_unit_panel) | [not comparable](#capability-ingresses-relative-prior-retention-from_switchback_panel) | [supported](#capability-ingresses-relative-prior-retention-from_definitions) | [supported](#capability-ingresses-relative-prior-retention-from_unit_day_artifact) | [refused](#capability-ingresses-relative-prior-retention-from_moments) |
| Relative Normal prior on quantiles | [supported](#capability-ingresses-relative-prior-quantile-from_unit_summary) | [supported](#capability-ingresses-relative-prior-quantile-from_unit_panel) | [not comparable](#capability-ingresses-relative-prior-quantile-from_switchback_panel) | [supported](#capability-ingresses-relative-prior-quantile-from_definitions) | [supported](#capability-ingresses-relative-prior-quantile-from_unit_day_artifact) | [refused](#capability-ingresses-relative-prior-quantile-from_moments) |
| Relative Normal prior on triggered daily lift | [not comparable](#capability-ingresses-relative-prior-daily-from_unit_summary) | [not comparable](#capability-ingresses-relative-prior-daily-from_unit_panel) | [not comparable](#capability-ingresses-relative-prior-daily-from_switchback_panel) | [supported](#capability-ingresses-relative-prior-daily-from_definitions) | [supported](#capability-ingresses-relative-prior-daily-from_unit_day_artifact) | [not comparable](#capability-ingresses-relative-prior-daily-from_moments) |
| Relative Normal prior on triggered as-of lift | [not comparable](#capability-ingresses-relative-prior-asof-from_unit_summary) | [not comparable](#capability-ingresses-relative-prior-asof-from_unit_panel) | [not comparable](#capability-ingresses-relative-prior-asof-from_switchback_panel) | [supported](#capability-ingresses-relative-prior-asof-from_definitions) | [supported](#capability-ingresses-relative-prior-asof-from_unit_day_artifact) | [not comparable](#capability-ingresses-relative-prior-asof-from_moments) |
| Fixed-threshold winsorization under sequential inference | [supported (conditional)](#capability-ingresses-fixed-threshold-winsor-sequential-from_unit_summary) | [supported](#capability-ingresses-fixed-threshold-winsor-sequential-from_unit_panel) | [refused](#capability-ingresses-fixed-threshold-winsor-sequential-from_switchback_panel) | [supported](#capability-ingresses-fixed-threshold-winsor-sequential-from_definitions) | [supported](#capability-ingresses-fixed-threshold-winsor-sequential-from_unit_day_artifact) | [supported](#capability-ingresses-fixed-threshold-winsor-sequential-from_moments) |
| Pooled percentile winsorization, fixed horizon, default size route | [supported (conditional)](#capability-ingresses-pooled-percentile-winsor-routes-from_unit_summary) | [refused (conditional)](#capability-ingresses-pooled-percentile-winsor-routes-from_unit_panel) | [refused](#capability-ingresses-pooled-percentile-winsor-routes-from_switchback_panel) | [supported (conditional)](#capability-ingresses-pooled-percentile-winsor-routes-from_definitions) | [supported (conditional)](#capability-ingresses-pooled-percentile-winsor-routes-from_unit_day_artifact) | [refused (conditional)](#capability-ingresses-pooled-percentile-winsor-routes-from_moments) |
| Ratio metric under sequential inference | [supported](#capability-ingresses-ratio-sequential-from_unit_summary) | [supported](#capability-ingresses-ratio-sequential-from_unit_panel) | [refused](#capability-ingresses-ratio-sequential-from_switchback_panel) | [supported](#capability-ingresses-ratio-sequential-from_definitions) | [supported](#capability-ingresses-ratio-sequential-from_unit_day_artifact) | [supported](#capability-ingresses-ratio-sequential-from_moments) |
| CUPED under sequential inference | [supported (conditional)](#capability-ingresses-mean-cuped-sequential-from_unit_summary) | [refused](#capability-ingresses-mean-cuped-sequential-from_unit_panel) | [refused](#capability-ingresses-mean-cuped-sequential-from_switchback_panel) | [supported](#capability-ingresses-mean-cuped-sequential-from_definitions) | [supported](#capability-ingresses-mean-cuped-sequential-from_unit_day_artifact) | [supported](#capability-ingresses-mean-cuped-sequential-from_moments) |
| Ratio CUPED under sequential inference | [supported (conditional)](#capability-ingresses-ratio-cuped-sequential-from_unit_summary) | [refused (conditional)](#capability-ingresses-ratio-cuped-sequential-from_unit_panel) | [refused](#capability-ingresses-ratio-cuped-sequential-from_switchback_panel) | [supported (conditional)](#capability-ingresses-ratio-cuped-sequential-from_definitions) | [supported (conditional)](#capability-ingresses-ratio-cuped-sequential-from_unit_day_artifact) | [supported (conditional)](#capability-ingresses-ratio-cuped-sequential-from_moments) |
| Bounded conversion under asymptotic sequential inference | [supported (conditional)](#capability-ingresses-bounded-conversion-asymptotic-sequential-from_unit_summary) | [supported (conditional)](#capability-ingresses-bounded-conversion-asymptotic-sequential-from_unit_panel) | [refused](#capability-ingresses-bounded-conversion-asymptotic-sequential-from_switchback_panel) | [supported (conditional)](#capability-ingresses-bounded-conversion-asymptotic-sequential-from_definitions) | [supported (conditional)](#capability-ingresses-bounded-conversion-asymptotic-sequential-from_unit_day_artifact) | [supported (conditional)](#capability-ingresses-bounded-conversion-asymptotic-sequential-from_moments) |
| Bounded retention (a start and end threshold) under asymptotic sequential inference | [refused](#capability-ingresses-bounded-retention-asymptotic-sequential-from_unit_summary) | [supported](#capability-ingresses-bounded-retention-asymptotic-sequential-from_unit_panel) | [refused](#capability-ingresses-bounded-retention-asymptotic-sequential-from_switchback_panel) | [supported](#capability-ingresses-bounded-retention-asymptotic-sequential-from_definitions) | [supported](#capability-ingresses-bounded-retention-asymptotic-sequential-from_unit_day_artifact) | [supported](#capability-ingresses-bounded-retention-asymptotic-sequential-from_moments) |
| Bounded conversion under automatic exact Bernoulli monitoring without a registration | [supported (conditional)](#capability-ingresses-bounded-conversion-exact-bernoulli-from_unit_summary) | [supported (conditional)](#capability-ingresses-bounded-conversion-exact-bernoulli-from_unit_panel) | [refused](#capability-ingresses-bounded-conversion-exact-bernoulli-from_switchback_panel) | [supported (conditional)](#capability-ingresses-bounded-conversion-exact-bernoulli-from_definitions) | [supported (conditional)](#capability-ingresses-bounded-conversion-exact-bernoulli-from_unit_day_artifact) | [supported (conditional)](#capability-ingresses-bounded-conversion-exact-bernoulli-from_moments) |
| Bounded retention under automatic exact Bernoulli monitoring without a registration | [refused](#capability-ingresses-bounded-retention-exact-bernoulli-from_unit_summary) | [supported](#capability-ingresses-bounded-retention-exact-bernoulli-from_unit_panel) | [refused](#capability-ingresses-bounded-retention-exact-bernoulli-from_switchback_panel) | [supported](#capability-ingresses-bounded-retention-exact-bernoulli-from_definitions) | [supported](#capability-ingresses-bounded-retention-exact-bernoulli-from_unit_day_artifact) | [supported](#capability-ingresses-bounded-retention-exact-bernoulli-from_moments) |
| Automatic exact multi-arm bounded-conversion monitoring | [supported](#capability-ingresses-multi-arm-bounded-conversion-exact-from_unit_summary) | [supported](#capability-ingresses-multi-arm-bounded-conversion-exact-from_unit_panel) | [refused (conditional)](#capability-ingresses-multi-arm-bounded-conversion-exact-from_switchback_panel) | [supported](#capability-ingresses-multi-arm-bounded-conversion-exact-from_definitions) | [supported](#capability-ingresses-multi-arm-bounded-conversion-exact-from_unit_day_artifact) | [supported](#capability-ingresses-multi-arm-bounded-conversion-exact-from_moments) |
| Automatic exact multi-arm bounded-retention monitoring | [refused](#capability-ingresses-multi-arm-bounded-retention-exact-from_unit_summary) | [not measured](#capability-ingresses-multi-arm-bounded-retention-exact-from_unit_panel) | [refused (conditional)](#capability-ingresses-multi-arm-bounded-retention-exact-from_switchback_panel) | [not measured](#capability-ingresses-multi-arm-bounded-retention-exact-from_definitions) | [not measured](#capability-ingresses-multi-arm-bounded-retention-exact-from_unit_day_artifact) | [not measured](#capability-ingresses-multi-arm-bounded-retention-exact-from_moments) |
| Unbounded conversion under sequential inference | [refused](#capability-ingresses-unbounded-conversion-sequential-from_unit_summary) | [refused](#capability-ingresses-unbounded-conversion-sequential-from_unit_panel) | [refused](#capability-ingresses-unbounded-conversion-sequential-from_switchback_panel) | [refused](#capability-ingresses-unbounded-conversion-sequential-from_definitions) | [refused](#capability-ingresses-unbounded-conversion-sequential-from_unit_day_artifact) | [refused](#capability-ingresses-unbounded-conversion-sequential-from_moments) |
| Unbounded retention (an integer threshold) under sequential inference | [refused (conditional)](#capability-ingresses-unbounded-retention-sequential-from_unit_summary) | [no disposition](#capability-ingresses-unbounded-retention-sequential-from_unit_panel) | [refused (conditional)](#capability-ingresses-unbounded-retention-sequential-from_switchback_panel) | [refused (conditional)](#capability-ingresses-unbounded-retention-sequential-from_definitions) | [no disposition](#capability-ingresses-unbounded-retention-sequential-from_unit_day_artifact) | [refused (conditional)](#capability-ingresses-unbounded-retention-sequential-from_moments) |
| Whole-window breakouts from one-row-per-unit summaries | [supported](#capability-ingresses-unit-summary-whole-window-breakouts-from_unit_summary) | [supported](#capability-ingresses-unit-summary-whole-window-breakouts-from_unit_panel) | [not comparable](#capability-ingresses-unit-summary-whole-window-breakouts-from_switchback_panel) | [supported](#capability-ingresses-unit-summary-whole-window-breakouts-from_definitions) | [supported](#capability-ingresses-unit-summary-whole-window-breakouts-from_unit_day_artifact) | [refused](#capability-ingresses-unit-summary-whole-window-breakouts-from_moments) |
| Randomized registered segmented sequential family (automatic predeclared levels or explicit roster) | [supported (conditional)](#capability-ingresses-registered-segmented-sequential-family-from_unit_summary) | [supported (conditional)](#capability-ingresses-registered-segmented-sequential-family-from_unit_panel) | [refused](#capability-ingresses-registered-segmented-sequential-family-from_switchback_panel) | [refused (conditional)](#capability-ingresses-registered-segmented-sequential-family-from_definitions) | [refused (conditional)](#capability-ingresses-registered-segmented-sequential-family-from_unit_day_artifact) | [refused (conditional)](#capability-ingresses-registered-segmented-sequential-family-from_moments) |
| Boolean and null segment labels in a registered segmented sequential family (canonical true/false/null spellings matching the string-labelled oracle) | [supported (conditional)](#capability-ingresses-boolean-null-segment-labels-from_unit_summary) | [supported (conditional)](#capability-ingresses-boolean-null-segment-labels-from_unit_panel) | [refused](#capability-ingresses-boolean-null-segment-labels-from_switchback_panel) | [refused (conditional)](#capability-ingresses-boolean-null-segment-labels-from_definitions) | [refused (conditional)](#capability-ingresses-boolean-null-segment-labels-from_unit_day_artifact) | [refused (conditional)](#capability-ingresses-boolean-null-segment-labels-from_moments) |
| Registered sequential monitoring of the triggered population (Experiment.trigger with InferenceSpec; assigned and triggered chains) | [refused (conditional)](#capability-ingresses-triggered-sequential-registration-from_unit_summary) | [refused (conditional)](#capability-ingresses-triggered-sequential-registration-from_unit_panel) | [refused (conditional)](#capability-ingresses-triggered-sequential-registration-from_switchback_panel) | [supported / refused (conditional)](#capability-ingresses-triggered-sequential-registration-from_definitions) | [supported / refused (conditional)](#capability-ingresses-triggered-sequential-registration-from_unit_day_artifact) | [supported / refused (conditional)](#capability-ingresses-triggered-sequential-registration-from_moments) |
| Triggered encouragement compliance under a registered plan | [refused (conditional)](#capability-ingresses-triggered-sequential-uptake-from_unit_summary) | [refused (conditional)](#capability-ingresses-triggered-sequential-uptake-from_unit_panel) | [refused (conditional)](#capability-ingresses-triggered-sequential-uptake-from_switchback_panel) | [unavailable (conditional)](#capability-ingresses-triggered-sequential-uptake-from_definitions) | [unavailable (conditional)](#capability-ingresses-triggered-sequential-uptake-from_unit_day_artifact) | [unavailable (conditional)](#capability-ingresses-triggered-sequential-uptake-from_moments) |
| Pooled-mean imputation helper with declared roles; a frame metric declaring an impute missing policy | [refused / supported (conditional)](#capability-ingresses-impute-pooled-mean-roles-from_unit_summary) | [refused / supported (conditional)](#capability-ingresses-impute-pooled-mean-roles-from_unit_panel) | [refused / supported (conditional)](#capability-ingresses-impute-pooled-mean-roles-from_switchback_panel) | [not comparable / not expressible (conditional)](#capability-ingresses-impute-pooled-mean-roles-from_definitions) | [not comparable / not expressible (conditional)](#capability-ingresses-impute-pooled-mean-roles-from_unit_day_artifact) | [not comparable / refused (conditional)](#capability-ingresses-impute-pooled-mean-roles-from_moments) |
| Encouragement ITT, compliance and LATE, fixed horizon, declared as the study design | [supported](#capability-ingresses-encouragement-design-readouts-from_unit_summary) | [supported](#capability-ingresses-encouragement-design-readouts-from_unit_panel) | [refused (conditional)](#capability-ingresses-encouragement-design-readouts-from_switchback_panel) | [supported](#capability-ingresses-encouragement-design-readouts-from_definitions) | [supported](#capability-ingresses-encouragement-design-readouts-from_unit_day_artifact) | [supported](#capability-ingresses-encouragement-design-readouts-from_moments) |
| Explicit ITT/compliance without a LATE exclusion declaration (fixed horizon or sequential monitoring) | [supported (conditional)](#capability-ingresses-encouragement-no-exclusion-from_unit_summary) | [supported (conditional)](#capability-ingresses-encouragement-no-exclusion-from_unit_panel) | [refused (conditional)](#capability-ingresses-encouragement-no-exclusion-from_switchback_panel) | [supported (conditional)](#capability-ingresses-encouragement-no-exclusion-from_definitions) | [supported (conditional)](#capability-ingresses-encouragement-no-exclusion-from_unit_day_artifact) | [supported (conditional)](#capability-ingresses-encouragement-no-exclusion-from_moments) |
| Fixed-horizon design-level compliance with an empty outcome catalog | [supported](#capability-ingresses-encouragement-empty-outcome-catalog-from_unit_summary) | [supported](#capability-ingresses-encouragement-empty-outcome-catalog-from_unit_panel) | [refused (conditional)](#capability-ingresses-encouragement-empty-outcome-catalog-from_switchback_panel) | [supported](#capability-ingresses-encouragement-empty-outcome-catalog-from_definitions) | [supported](#capability-ingresses-encouragement-empty-outcome-catalog-from_unit_day_artifact) | [supported](#capability-ingresses-encouragement-empty-outcome-catalog-from_moments) |
| Encouragement guardrail with a declared non-inferiority margin | [supported / refused (conditional)](#capability-ingresses-encouragement-margin-guardrail-from_unit_summary) | [supported / refused (conditional)](#capability-ingresses-encouragement-margin-guardrail-from_unit_panel) | [refused (conditional)](#capability-ingresses-encouragement-margin-guardrail-from_switchback_panel) | [supported / refused (conditional)](#capability-ingresses-encouragement-margin-guardrail-from_definitions) | [supported / refused (conditional)](#capability-ingresses-encouragement-margin-guardrail-from_unit_day_artifact) | [supported / refused (conditional)](#capability-ingresses-encouragement-margin-guardrail-from_moments) |
| Encouragement BH-selected secondary's LATE row re-estimated at the corrected level | [supported](#capability-ingresses-encouragement-late-fcr-reestimated-from_unit_summary) | [supported](#capability-ingresses-encouragement-late-fcr-reestimated-from_unit_panel) | [refused (conditional)](#capability-ingresses-encouragement-late-fcr-reestimated-from_switchback_panel) | [supported](#capability-ingresses-encouragement-late-fcr-reestimated-from_definitions) | [supported](#capability-ingresses-encouragement-late-fcr-reestimated-from_unit_day_artifact) | [supported](#capability-ingresses-encouragement-late-fcr-reestimated-from_moments) |
| Encouragement ITT on a conversion metric takes the count-routed conversion route | [supported (conditional)](#capability-ingresses-encouragement-itt-conversion-route-from_unit_summary) | [supported (conditional)](#capability-ingresses-encouragement-itt-conversion-route-from_unit_panel) | [refused (conditional)](#capability-ingresses-encouragement-itt-conversion-route-from_switchback_panel) | [supported (conditional)](#capability-ingresses-encouragement-itt-conversion-route-from_definitions) | [supported (conditional)](#capability-ingresses-encouragement-itt-conversion-route-from_unit_day_artifact) | [supported (conditional)](#capability-ingresses-encouragement-itt-conversion-route-from_moments) |
| Conversion/retention counts one below, at and one above the dense threshold, on the success and failure side, route identically and report the same interval and p-value | [supported](#capability-ingresses-conversion-route-threshold-from_unit_summary) | [supported](#capability-ingresses-conversion-route-threshold-from_unit_panel) | [not comparable](#capability-ingresses-conversion-route-threshold-from_switchback_panel) | [supported](#capability-ingresses-conversion-route-threshold-from_definitions) | [supported](#capability-ingresses-conversion-route-threshold-from_unit_day_artifact) | [supported](#capability-ingresses-conversion-route-threshold-from_moments) |
| Conversion breakout with dense, sparse and threshold-straddling segments in one metric: each segment row is routed by its own counts | [supported](#capability-ingresses-conversion-breakout-route-from_unit_summary) | [supported](#capability-ingresses-conversion-breakout-route-from_unit_panel) | [not comparable](#capability-ingresses-conversion-breakout-route-from_switchback_panel) | [supported](#capability-ingresses-conversion-breakout-route-from_definitions) | [supported](#capability-ingresses-conversion-breakout-route-from_unit_day_artifact) | [refused](#capability-ingresses-conversion-breakout-route-from_moments) |
| Encouragement ITT rows of a BH secondary family are routed at their own level and carry composite p-value evidence | [supported](#capability-ingresses-conversion-route-encouragement-family-from_unit_summary) | [supported](#capability-ingresses-conversion-route-encouragement-family-from_unit_panel) | [refused (conditional)](#capability-ingresses-conversion-route-encouragement-family-from_switchback_panel) | [supported](#capability-ingresses-conversion-route-encouragement-family-from_definitions) | [supported](#capability-ingresses-conversion-route-encouragement-family-from_unit_day_artifact) | [supported](#capability-ingresses-conversion-route-encouragement-family-from_moments) |
| Observational unadjusted rows of a BH secondary family are routed at their own level and carry composite p-value evidence | [supported](#capability-ingresses-conversion-route-observational-family-from_unit_summary) | [supported](#capability-ingresses-conversion-route-observational-family-from_unit_panel) | [refused](#capability-ingresses-conversion-route-observational-family-from_switchback_panel) | [supported](#capability-ingresses-conversion-route-observational-family-from_definitions) | [supported](#capability-ingresses-conversion-route-observational-family-from_unit_day_artifact) | [supported](#capability-ingresses-conversion-route-observational-family-from_moments) |
| Encouragement design with a retention metric (ITT and LATE readouts) | [refused (conditional)](#capability-ingresses-encouragement-retention-readouts-from_unit_summary) | [refused (conditional)](#capability-ingresses-encouragement-retention-readouts-from_unit_panel) | [refused (conditional)](#capability-ingresses-encouragement-retention-readouts-from_switchback_panel) | [refused (conditional)](#capability-ingresses-encouragement-retention-readouts-from_definitions) | [refused (conditional)](#capability-ingresses-encouragement-retention-readouts-from_unit_day_artifact) | [refused (conditional)](#capability-ingresses-encouragement-retention-readouts-from_moments) |
| Encouragement design with a retention metric, compliance-only request (which ignores the retention outcome) | [refused (conditional)](#capability-ingresses-encouragement-retention-compliance-only-from_unit_summary) | [refused (conditional)](#capability-ingresses-encouragement-retention-compliance-only-from_unit_panel) | [refused (conditional)](#capability-ingresses-encouragement-retention-compliance-only-from_switchback_panel) | [supported / refused (conditional)](#capability-ingresses-encouragement-retention-compliance-only-from_definitions) | [not measured](#capability-ingresses-encouragement-retention-compliance-only-from_unit_day_artifact) | [not measured](#capability-ingresses-encouragement-retention-compliance-only-from_moments) |
| Encouragement conversion ITT breakouts retain exact confidence sets at zero control counts | [supported](#capability-ingresses-encouragement-conversion-breakout-zero-control-from_unit_summary) | [supported](#capability-ingresses-encouragement-conversion-breakout-zero-control-from_unit_panel) | [not comparable](#capability-ingresses-encouragement-conversion-breakout-zero-control-from_switchback_panel) | [supported](#capability-ingresses-encouragement-conversion-breakout-zero-control-from_definitions) | [supported](#capability-ingresses-encouragement-conversion-breakout-zero-control-from_unit_day_artifact) | [refused](#capability-ingresses-encouragement-conversion-breakout-zero-control-from_moments) |
| Bounded-retention breakouts with mature exposure cohorts | [refused](#capability-ingresses-retention-breakout-cohorts-from_unit_summary) | [supported](#capability-ingresses-retention-breakout-cohorts-from_unit_panel) | [not comparable](#capability-ingresses-retention-breakout-cohorts-from_switchback_panel) | [supported](#capability-ingresses-retention-breakout-cohorts-from_definitions) | [supported](#capability-ingresses-retention-breakout-cohorts-from_unit_day_artifact) | [refused](#capability-ingresses-retention-breakout-cohorts-from_moments) |
| Encouragement continuous ITT + Bernoulli uptake compliance, jointly monitored (composed sequential family, e-BH selected) | [supported (conditional)](#capability-ingresses-encouragement-joint-sequential-from_unit_summary) | [supported (conditional)](#capability-ingresses-encouragement-joint-sequential-from_unit_panel) | [refused (conditional)](#capability-ingresses-encouragement-joint-sequential-from_switchback_panel) | [supported (conditional)](#capability-ingresses-encouragement-joint-sequential-from_definitions) | [supported (conditional)](#capability-ingresses-encouragement-joint-sequential-from_unit_day_artifact) | [supported (conditional)](#capability-ingresses-encouragement-joint-sequential-from_moments) |
| Uptake-only sequential compliance with an empty outcome catalog | [supported](#capability-ingresses-uptake-only-sequential-empty-catalog-from_unit_summary) | [supported](#capability-ingresses-uptake-only-sequential-empty-catalog-from_unit_panel) | [refused (conditional)](#capability-ingresses-uptake-only-sequential-empty-catalog-from_switchback_panel) | [supported](#capability-ingresses-uptake-only-sequential-empty-catalog-from_definitions) | [supported](#capability-ingresses-uptake-only-sequential-empty-catalog-from_unit_day_artifact) | [supported](#capability-ingresses-uptake-only-sequential-empty-catalog-from_moments) |
| Uptake-only sequential compliance that retains an unbounded retention metric in the outcome catalog | [refused](#capability-ingresses-uptake-only-sequential-unbounded-retention-catalog-from_unit_summary) | [refused](#capability-ingresses-uptake-only-sequential-unbounded-retention-catalog-from_unit_panel) | [refused (conditional)](#capability-ingresses-uptake-only-sequential-unbounded-retention-catalog-from_switchback_panel) | [supported / refused (conditional)](#capability-ingresses-uptake-only-sequential-unbounded-retention-catalog-from_definitions) | [supported / refused (conditional)](#capability-ingresses-uptake-only-sequential-unbounded-retention-catalog-from_unit_day_artifact) | [supported / refused (conditional)](#capability-ingresses-uptake-only-sequential-unbounded-retention-catalog-from_moments) |
| Observational weight diagnostics (IPTW/AIPW; unit-level mean and conversion parity) | [supported (conditional)](#capability-ingresses-observational-weight-diagnostics-from_unit_summary) | [supported (conditional)](#capability-ingresses-observational-weight-diagnostics-from_unit_panel) | [refused (conditional)](#capability-ingresses-observational-weight-diagnostics-from_switchback_panel) | [supported (conditional)](#capability-ingresses-observational-weight-diagnostics-from_definitions) | [supported (conditional)](#capability-ingresses-observational-weight-diagnostics-from_unit_day_artifact) | [refused (conditional)](#capability-ingresses-observational-weight-diagnostics-from_moments) |
| Assignment-integrity scope (status, observed counts, randomization grain and retained metadata; triggered populations are not assignment-law evidence) | [supported (conditional)](#capability-ingresses-assignment-integrity-scope-from_unit_summary) | [supported / not comparable (conditional)](#capability-ingresses-assignment-integrity-scope-from_unit_panel) | [not comparable](#capability-ingresses-assignment-integrity-scope-from_switchback_panel) | [supported](#capability-ingresses-assignment-integrity-scope-from_definitions) | [supported](#capability-ingresses-assignment-integrity-scope-from_unit_day_artifact) | [supported / not comparable (conditional)](#capability-ingresses-assignment-integrity-scope-from_moments) |
| Observational confounding sensitivity (conversion RR) | [supported (conditional)](#capability-ingresses-observational-confounding-sensitivity-from_unit_summary) | [supported (conditional)](#capability-ingresses-observational-confounding-sensitivity-from_unit_panel) | [refused (conditional)](#capability-ingresses-observational-confounding-sensitivity-from_switchback_panel) | [supported (conditional)](#capability-ingresses-observational-confounding-sensitivity-from_definitions) | [supported (conditional)](#capability-ingresses-observational-confounding-sensitivity-from_unit_day_artifact) | [refused (conditional)](#capability-ingresses-observational-confounding-sensitivity-from_moments) |
| Observational IPTW covariate adjustment | [supported](#capability-ingresses-observational-iptw-mean-from_unit_summary) | [supported](#capability-ingresses-observational-iptw-mean-from_unit_panel) | [refused](#capability-ingresses-observational-iptw-mean-from_switchback_panel) | [supported](#capability-ingresses-observational-iptw-mean-from_definitions) | [supported](#capability-ingresses-observational-iptw-mean-from_unit_day_artifact) | [refused](#capability-ingresses-observational-iptw-mean-from_moments) |
| Observational AIPW/DML covariate adjustment | [supported](#capability-ingresses-observational-aipw-dml-from_unit_summary) | [supported](#capability-ingresses-observational-aipw-dml-from_unit_panel) | [refused](#capability-ingresses-observational-aipw-dml-from_switchback_panel) | [supported](#capability-ingresses-observational-aipw-dml-from_definitions) | [supported](#capability-ingresses-observational-aipw-dml-from_unit_day_artifact) | [refused](#capability-ingresses-observational-aipw-dml-from_moments) |
| Categorical observational adjustment (string covariates) | [supported](#capability-ingresses-categorical-observational-adjustment-from_unit_summary) | [supported](#capability-ingresses-categorical-observational-adjustment-from_unit_panel) | [refused](#capability-ingresses-categorical-observational-adjustment-from_switchback_panel) | [supported](#capability-ingresses-categorical-observational-adjustment-from_definitions) | [supported](#capability-ingresses-categorical-observational-adjustment-from_unit_day_artifact) | [refused](#capability-ingresses-categorical-observational-adjustment-from_moments) |
| Categorical nulls with a nondefault missing-value policy | [supported](#capability-ingresses-categorical-missing-policy-from_unit_summary) | [supported](#capability-ingresses-categorical-missing-policy-from_unit_panel) | [refused](#capability-ingresses-categorical-missing-policy-from_switchback_panel) | [refused](#capability-ingresses-categorical-missing-policy-from_definitions) | [refused](#capability-ingresses-categorical-missing-policy-from_unit_day_artifact) | [refused](#capability-ingresses-categorical-missing-policy-from_moments) |
| Observational primary + secondary + guardrail together (role multiplicity) | [supported](#capability-ingresses-observational-role-multiplicity-from_unit_summary) | [supported](#capability-ingresses-observational-role-multiplicity-from_unit_panel) | [refused](#capability-ingresses-observational-role-multiplicity-from_switchback_panel) | [supported](#capability-ingresses-observational-role-multiplicity-from_definitions) | [supported](#capability-ingresses-observational-role-multiplicity-from_unit_day_artifact) | [refused](#capability-ingresses-observational-role-multiplicity-from_moments) |
| Quantile inference on tied/rounded outcomes | [supported](#capability-ingresses-quantile-tied-rounded-outcomes-from_unit_summary) | [supported / refused (conditional)](#capability-ingresses-quantile-tied-rounded-outcomes-from_unit_panel) | [refused](#capability-ingresses-quantile-tied-rounded-outcomes-from_switchback_panel) | [supported](#capability-ingresses-quantile-tied-rounded-outcomes-from_definitions) | [supported](#capability-ingresses-quantile-tied-rounded-outcomes-from_unit_day_artifact) | [refused (conditional)](#capability-ingresses-quantile-tied-rounded-outcomes-from_moments) |
| Registered conversion checkpoint retaining an unmodeled, unwindowed quantile catalog | [supported](#capability-ingresses-quantile-checkpoint-retention-from_unit_summary) | [supported](#capability-ingresses-quantile-checkpoint-retention-from_unit_panel) | [not comparable](#capability-ingresses-quantile-checkpoint-retention-from_switchback_panel) | [supported](#capability-ingresses-quantile-checkpoint-retention-from_definitions) | [supported](#capability-ingresses-quantile-checkpoint-retention-from_unit_day_artifact) | [supported / refused (conditional)](#capability-ingresses-quantile-checkpoint-retention-from_moments) |
| Quantile metric under an Observational design (two-sided, zero-null request) | [refused (conditional)](#capability-ingresses-observational-quantile-refusal-from_unit_summary) | [refused (conditional)](#capability-ingresses-observational-quantile-refusal-from_unit_panel) | [refused](#capability-ingresses-observational-quantile-refusal-from_switchback_panel) | [refused (conditional)](#capability-ingresses-observational-quantile-refusal-from_definitions) | [refused / not expressible (conditional)](#capability-ingresses-observational-quantile-refusal-from_unit_day_artifact) | [refused (conditional)](#capability-ingresses-observational-quantile-refusal-from_moments) |
| Clustered ratio metric with a non-positive control-arm numerator | [supported](#capability-ingresses-clustered-negative-mean-from_unit_summary) | [refused](#capability-ingresses-clustered-negative-mean-from_unit_panel) | [not expressible](#capability-ingresses-clustered-negative-mean-from_switchback_panel) | [supported](#capability-ingresses-clustered-negative-mean-from_definitions) | [supported](#capability-ingresses-clustered-negative-mean-from_unit_day_artifact) | [refused](#capability-ingresses-clustered-negative-mean-from_moments) |
| Clustered zero relative variance retains its point and any available additive interval | [supported](#capability-ingresses-clustered-zero-relative-variance-from_unit_summary) | [refused](#capability-ingresses-clustered-zero-relative-variance-from_unit_panel) | [not expressible](#capability-ingresses-clustered-zero-relative-variance-from_switchback_panel) | [supported](#capability-ingresses-clustered-zero-relative-variance-from_definitions) | [supported](#capability-ingresses-clustered-zero-relative-variance-from_unit_day_artifact) | [refused](#capability-ingresses-clustered-zero-relative-variance-from_moments) |
| Planning baseline, mean metric | [supported (conditional)](#capability-ingresses-planning-baseline-mean-from_unit_summary) | [supported (conditional)](#capability-ingresses-planning-baseline-mean-from_unit_panel) | [supported (conditional)](#capability-ingresses-planning-baseline-mean-from_switchback_panel) | [supported (conditional)](#capability-ingresses-planning-baseline-mean-from_definitions) | [supported (conditional)](#capability-ingresses-planning-baseline-mean-from_unit_day_artifact) | [supported (conditional)](#capability-ingresses-planning-baseline-mean-from_moments) |
| Planning baseline, quantile metric | [supported (conditional)](#capability-ingresses-planning-baseline-quantile-from_unit_summary) | [supported (conditional)](#capability-ingresses-planning-baseline-quantile-from_unit_panel) | [refused (conditional)](#capability-ingresses-planning-baseline-quantile-from_switchback_panel) | [supported (conditional)](#capability-ingresses-planning-baseline-quantile-from_definitions) | [supported (conditional)](#capability-ingresses-planning-baseline-quantile-from_unit_day_artifact) | [refused (conditional)](#capability-ingresses-planning-baseline-quantile-from_moments) |
| Planning baseline with a declared cluster (per-unit mean/var; ICC on the metric's per-unit score) | [supported (conditional)](#capability-ingresses-planning-baseline-cluster-from_unit_summary) | [refused (conditional)](#capability-ingresses-planning-baseline-cluster-from_unit_panel) | [not expressible (conditional)](#capability-ingresses-planning-baseline-cluster-from_switchback_panel) | [supported (conditional)](#capability-ingresses-planning-baseline-cluster-from_definitions) | [supported (conditional)](#capability-ingresses-planning-baseline-cluster-from_unit_day_artifact) | [refused (conditional)](#capability-ingresses-planning-baseline-cluster-from_moments) |
| Planning baseline under a declared trigger (triggered-population moments and trigger rate) | [source limited (conditional)](#capability-ingresses-planning-baseline-trigger-from_unit_summary) | [source limited (conditional)](#capability-ingresses-planning-baseline-trigger-from_unit_panel) | [source limited (conditional)](#capability-ingresses-planning-baseline-trigger-from_switchback_panel) | [supported (conditional)](#capability-ingresses-planning-baseline-trigger-from_definitions) | [supported / refused (conditional)](#capability-ingresses-planning-baseline-trigger-from_unit_day_artifact) | [source limited (conditional)](#capability-ingresses-planning-baseline-trigger-from_moments) |
| Triggered daily/as-of outcome value and lift history (membership is limited to units whose first eligible trigger is observed by each day, and outcome windows anchor at that unit's trigger) | [refused (conditional)](#capability-ingresses-triggered-day-axis-histories-from_unit_summary) | [refused (conditional)](#capability-ingresses-triggered-day-axis-histories-from_unit_panel) | [refused (conditional)](#capability-ingresses-triggered-day-axis-histories-from_switchback_panel) | [supported (conditional)](#capability-ingresses-triggered-day-axis-histories-from_definitions) | [supported (conditional)](#capability-ingresses-triggered-day-axis-histories-from_unit_day_artifact) | [refused (conditional)](#capability-ingresses-triggered-day-axis-histories-from_moments) |
| Triggered encouragement compliance (daily and as-of lift, compliance only; uptake stays assignment-anchored while membership enters at first eligible trigger) | [refused](#capability-ingresses-triggered-encouragement-compliance-from_unit_summary) | [refused](#capability-ingresses-triggered-encouragement-compliance-from_unit_panel) | [refused](#capability-ingresses-triggered-encouragement-compliance-from_switchback_panel) | [supported](#capability-ingresses-triggered-encouragement-compliance-from_definitions) | [supported](#capability-ingresses-triggered-encouragement-compliance-from_unit_day_artifact) | [refused](#capability-ingresses-triggered-encouragement-compliance-from_moments) |
| Triggered clustered daily/as-of outcome readouts | [refused (conditional)](#capability-ingresses-triggered-clustered-day-axis-from_unit_summary) | [refused (conditional)](#capability-ingresses-triggered-clustered-day-axis-from_unit_panel) | [refused (conditional)](#capability-ingresses-triggered-clustered-day-axis-from_switchback_panel) | [refused / supported (conditional)](#capability-ingresses-triggered-clustered-day-axis-from_definitions) | [refused / supported (conditional)](#capability-ingresses-triggered-clustered-day-axis-from_unit_day_artifact) | [refused (conditional)](#capability-ingresses-triggered-clustered-day-axis-from_moments) |
| Two-stage spillover: policy difference under personal nonoffer | [supported (conditional)](#capability-ingresses-untreated-spillover-from_unit_summary) | [not expressible (conditional)](#capability-ingresses-untreated-spillover-from_unit_panel) | [not expressible (conditional)](#capability-ingresses-untreated-spillover-from_switchback_panel) | [supported (conditional)](#capability-ingresses-untreated-spillover-from_definitions) | [not expressible (conditional)](#capability-ingresses-untreated-spillover-from_unit_day_artifact) | [not expressible (conditional)](#capability-ingresses-untreated-spillover-from_moments) |
| Logged-policy contrast | [not expressible (conditional)](#capability-ingresses-logged-policy-contrast-from_unit_summary) | [not expressible (conditional)](#capability-ingresses-logged-policy-contrast-from_unit_panel) | [not expressible (conditional)](#capability-ingresses-logged-policy-contrast-from_switchback_panel) | [not expressible (conditional)](#capability-ingresses-logged-policy-contrast-from_definitions) | [not expressible (conditional)](#capability-ingresses-logged-policy-contrast-from_unit_day_artifact) | [not expressible (conditional)](#capability-ingresses-logged-policy-contrast-from_moments) |
| TabularPolicy persistence (JSON dump and validate) | [not comparable (conditional)](#capability-ingresses-tabular-policy-persistence-from_unit_summary) | [not comparable (conditional)](#capability-ingresses-tabular-policy-persistence-from_unit_panel) | [not comparable (conditional)](#capability-ingresses-tabular-policy-persistence-from_switchback_panel) | [not comparable (conditional)](#capability-ingresses-tabular-policy-persistence-from_definitions) | [not comparable (conditional)](#capability-ingresses-tabular-policy-persistence-from_unit_day_artifact) | [not comparable (conditional)](#capability-ingresses-tabular-policy-persistence-from_moments) |
| Mixed-covariate CUPED × scalar cuped | [supported](#capability-ingresses-cuped-linear-interaction-scalar_cuped-from_unit_summary) | [supported](#capability-ingresses-cuped-linear-interaction-scalar_cuped-from_unit_panel) | [source limited](#capability-ingresses-cuped-linear-interaction-scalar_cuped-from_switchback_panel) | [supported](#capability-ingresses-cuped-linear-interaction-scalar_cuped-from_definitions) | [supported](#capability-ingresses-cuped-linear-interaction-scalar_cuped-from_unit_day_artifact) | [supported](#capability-ingresses-cuped-linear-interaction-scalar_cuped-from_moments) |
| Mixed-covariate CUPED × ratio | [supported](#capability-ingresses-cuped-linear-interaction-ratio-from_unit_summary) | [supported](#capability-ingresses-cuped-linear-interaction-ratio-from_unit_panel) | [source limited](#capability-ingresses-cuped-linear-interaction-ratio-from_switchback_panel) | [supported](#capability-ingresses-cuped-linear-interaction-ratio-from_definitions) | [supported](#capability-ingresses-cuped-linear-interaction-ratio-from_unit_day_artifact) | [supported](#capability-ingresses-cuped-linear-interaction-ratio-from_moments) |
| Mixed-covariate CUPED × clustering | [unfinished](#capability-ingresses-cuped-linear-interaction-clustering-from_unit_summary) | [source limited](#capability-ingresses-cuped-linear-interaction-clustering-from_unit_panel) | [source limited](#capability-ingresses-cuped-linear-interaction-clustering-from_switchback_panel) | [unfinished](#capability-ingresses-cuped-linear-interaction-clustering-from_definitions) | [unfinished](#capability-ingresses-cuped-linear-interaction-clustering-from_unit_day_artifact) | [source limited](#capability-ingresses-cuped-linear-interaction-clustering-from_moments) |
| Mixed-covariate CUPED × sequential | [unfinished](#capability-ingresses-cuped-linear-interaction-sequential-from_unit_summary) | [unfinished](#capability-ingresses-cuped-linear-interaction-sequential-from_unit_panel) | [source limited](#capability-ingresses-cuped-linear-interaction-sequential-from_switchback_panel) | [unfinished](#capability-ingresses-cuped-linear-interaction-sequential-from_definitions) | [unfinished](#capability-ingresses-cuped-linear-interaction-sequential-from_unit_day_artifact) | [source limited](#capability-ingresses-cuped-linear-interaction-sequential-from_moments) |
| Mixed-covariate CUPED × winsorization fixed value | [supported](#capability-ingresses-cuped-linear-interaction-winsorization_fixed_value-from_unit_summary) | [supported](#capability-ingresses-cuped-linear-interaction-winsorization_fixed_value-from_unit_panel) | [source limited](#capability-ingresses-cuped-linear-interaction-winsorization_fixed_value-from_switchback_panel) | [supported](#capability-ingresses-cuped-linear-interaction-winsorization_fixed_value-from_definitions) | [supported](#capability-ingresses-cuped-linear-interaction-winsorization_fixed_value-from_unit_day_artifact) | [supported](#capability-ingresses-cuped-linear-interaction-winsorization_fixed_value-from_moments) |
| Mixed-covariate CUPED × winsorization percentile | [unfinished](#capability-ingresses-cuped-linear-interaction-winsorization_percentile-from_unit_summary) | [unfinished](#capability-ingresses-cuped-linear-interaction-winsorization_percentile-from_unit_panel) | [source limited](#capability-ingresses-cuped-linear-interaction-winsorization_percentile-from_switchback_panel) | [unfinished](#capability-ingresses-cuped-linear-interaction-winsorization_percentile-from_definitions) | [unfinished](#capability-ingresses-cuped-linear-interaction-winsorization_percentile-from_unit_day_artifact) | [source limited](#capability-ingresses-cuped-linear-interaction-winsorization_percentile-from_moments) |
| Mixed-covariate CUPED × breakouts | [source limited](#capability-ingresses-cuped-linear-interaction-breakouts-from_unit_summary) | [supported](#capability-ingresses-cuped-linear-interaction-breakouts-from_unit_panel) | [source limited](#capability-ingresses-cuped-linear-interaction-breakouts-from_switchback_panel) | [supported](#capability-ingresses-cuped-linear-interaction-breakouts-from_definitions) | [supported](#capability-ingresses-cuped-linear-interaction-breakouts-from_unit_day_artifact) | [source limited](#capability-ingresses-cuped-linear-interaction-breakouts-from_moments) |
| Mixed-covariate CUPED × multiplicity roles | [supported](#capability-ingresses-cuped-linear-interaction-multiplicity_roles-from_unit_summary) | [supported](#capability-ingresses-cuped-linear-interaction-multiplicity_roles-from_unit_panel) | [source limited](#capability-ingresses-cuped-linear-interaction-multiplicity_roles-from_switchback_panel) | [supported](#capability-ingresses-cuped-linear-interaction-multiplicity_roles-from_definitions) | [supported](#capability-ingresses-cuped-linear-interaction-multiplicity_roles-from_unit_day_artifact) | [supported](#capability-ingresses-cuped-linear-interaction-multiplicity_roles-from_moments) |
| Mixed-covariate CUPED × triggered population | [source limited](#capability-ingresses-cuped-linear-interaction-triggered_population-from_unit_summary) | [source limited](#capability-ingresses-cuped-linear-interaction-triggered_population-from_unit_panel) | [source limited](#capability-ingresses-cuped-linear-interaction-triggered_population-from_switchback_panel) | [supported](#capability-ingresses-cuped-linear-interaction-triggered_population-from_definitions) | [supported](#capability-ingresses-cuped-linear-interaction-triggered_population-from_unit_day_artifact) | [source limited](#capability-ingresses-cuped-linear-interaction-triggered_population-from_moments) |

<span id="capability-ingresses-shared-warehouse-metric"></span>

### Shared definition: completed-order ratio

<span id="capability-ingresses-shared-warehouse-metric-from_unit_summary"></span>

#### Shared definition: completed-order ratio — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** not comparable

**Classification:** SOURCE

**Reason:** this comparison starts with native and shared warehouse YAML.

<span id="capability-ingresses-shared-warehouse-metric-from_unit_panel"></span>

#### Shared definition: completed-order ratio — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** not comparable

**Classification:** SOURCE

**Reason:** this comparison starts with native and shared warehouse YAML.

<span id="capability-ingresses-shared-warehouse-metric-from_switchback_panel"></span>

#### Shared definition: completed-order ratio — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** not comparable

**Classification:** SOURCE

**Reason:** this comparison starts with native and shared warehouse YAML.

<span id="capability-ingresses-shared-warehouse-metric-from_definitions"></span>

#### Shared definition: completed-order ratio — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

**Note:** Opt in to selected shared metrics from the native definitions file.

<span id="capability-ingresses-shared-warehouse-metric-from_unit_day_artifact"></span>

#### Shared definition: completed-order ratio — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

**Note:** Matches the prepared unit-day data; does not read shared YAML.

<span id="capability-ingresses-shared-warehouse-metric-from_moments"></span>

#### Shared definition: completed-order ratio — Portable moments

**Recorded path:** `from_moments`

**Status:** not comparable

**Classification:** SOURCE

**Reason:** this comparison starts with native and shared warehouse YAML.


<span id="capability-ingresses-shared-capture-exploratory"></span>

### Declared + exploratory metrics in one physical capture

<span id="capability-ingresses-shared-capture-exploratory-from_unit_summary"></span>

#### Declared + exploratory metrics in one physical capture — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** saved definitions are unavailable; reject the selected whole-window view before reads

**Status:** refused

**Codes:** `facade.analysis.exploratory_metrics_source_limited`

**Reason:** Only the definitions ingress can add undeclared saved metrics.

**Operation:** `run(exploratory_metrics=...)`

**Stage:** read

<span id="capability-ingresses-shared-capture-exploratory-from_unit_panel"></span>

#### Declared + exploratory metrics in one physical capture — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** saved definitions are unavailable; reject the selected whole-window view before reads

**Status:** refused

**Codes:** `facade.analysis.exploratory_metrics_source_limited`

**Reason:** Only the definitions ingress can add undeclared saved metrics.

**Operation:** `run(exploratory_metrics=...)`

**Stage:** read

<span id="capability-ingresses-shared-capture-exploratory-from_switchback_panel"></span>

#### Declared + exploratory metrics in one physical capture — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** saved definitions are unavailable; reject the selected whole-window view before reads

**Status:** refused

**Codes:** `facade.analysis.exploratory_metrics_source_limited`

**Reason:** Only the definitions ingress can add undeclared saved metrics.

**Operation:** `run(exploratory_metrics=...)`

**Stage:** read

<span id="capability-ingresses-shared-capture-exploratory-from_definitions"></span>

#### Declared + exploratory metrics in one physical capture — Definitions

**Recorded path:** `from_definitions`

**Condition:** declared and exploratory families share physical capture, retaining original family horizons

**Status:** supported

**Operation:** `run(exploratory_metrics=...)`

**Stage:** read

<span id="capability-ingresses-shared-capture-exploratory-from_unit_day_artifact"></span>

#### Declared + exploratory metrics in one physical capture — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** saved definitions are unavailable; reject the selected whole-window view before reads

**Status:** refused

**Codes:** `facade.analysis.exploratory_metrics_source_limited`

**Reason:** Only the definitions ingress can add undeclared saved metrics.

**Operation:** `run(exploratory_metrics=...)`

**Stage:** read

<span id="capability-ingresses-shared-capture-exploratory-from_moments"></span>

#### Declared + exploratory metrics in one physical capture — Portable moments

**Recorded path:** `from_moments`

**Condition:** saved definitions are unavailable; reject the selected whole-window view before reads

**Status:** refused

**Codes:** `facade.analysis.exploratory_metrics_source_limited`

**Reason:** Only the definitions ingress can add undeclared saved metrics.

**Operation:** `run(exploratory_metrics=...)`

**Stage:** read


<span id="capability-ingresses-shared-capture-multiview"></span>

### Whole-window + daily scopes in one physical capture

<span id="capability-ingresses-shared-capture-multiview-from_unit_summary"></span>

#### Whole-window + daily scopes in one physical capture — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** a later daily scope cannot be served; reject before the whole-window sibling reads

**Status:** refused

**Codes:** `evidence.grain_unavailable`

**Reason:** Shared execution does not create a temporal axis or a new contrast estimator.

**Operation:** `run(views=...)`

**Stage:** read

<span id="capability-ingresses-shared-capture-multiview-from_unit_panel"></span>

#### Whole-window + daily scopes in one physical capture — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** whole-window and daily scopes share one capture and retain standalone cells

**Status:** supported

**Operation:** `run(views=...)`

**Stage:** read

<span id="capability-ingresses-shared-capture-multiview-from_switchback_panel"></span>

#### Whole-window + daily scopes in one physical capture — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** a later daily scope cannot be served; reject before the whole-window sibling reads

**Status:** refused

**Codes:** `facade.analysis.contrast_unavailable`

**Reason:** Shared execution does not create a temporal axis or a new contrast estimator.

**Operation:** `run(views=...)`

**Stage:** read

<span id="capability-ingresses-shared-capture-multiview-from_definitions"></span>

#### Whole-window + daily scopes in one physical capture — Definitions

**Recorded path:** `from_definitions`

**Condition:** whole-window and daily scopes share one capture and retain standalone cells

**Status:** supported

**Operation:** `run(views=...)`

**Stage:** read

**Note:** As-of, breakout and assigned/triggered scope unions also retain original standalone identities; capture provenance is separate.

<span id="capability-ingresses-shared-capture-multiview-from_unit_day_artifact"></span>

#### Whole-window + daily scopes in one physical capture — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** whole-window and daily scopes share one capture and retain standalone cells

**Status:** supported

**Operation:** `run(views=...)`

**Stage:** read

<span id="capability-ingresses-shared-capture-multiview-from_moments"></span>

#### Whole-window + daily scopes in one physical capture — Portable moments

**Recorded path:** `from_moments`

**Condition:** a later daily scope cannot be served; reject before the whole-window sibling reads

**Status:** refused

**Codes:** `evidence.grain_unavailable`

**Reason:** Shared execution does not create a temporal axis or a new contrast estimator.

**Operation:** `run(views=...)`

**Stage:** read


<span id="capability-ingresses-mean-cuped-fixed-horizon"></span>

### Mean CUPED, fixed horizon

<span id="capability-ingresses-mean-cuped-fixed-horizon-from_unit_summary"></span>

#### Mean CUPED, fixed horizon — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

<span id="capability-ingresses-mean-cuped-fixed-horizon-from_unit_panel"></span>

#### Mean CUPED, fixed horizon — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

<span id="capability-ingresses-mean-cuped-fixed-horizon-from_switchback_panel"></span>

#### Mean CUPED, fixed horizon — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.metric`

**Reason:** switchback procedures apply no decision-method variance reduction

**Stage:** declaration

<span id="capability-ingresses-mean-cuped-fixed-horizon-from_definitions"></span>

#### Mean CUPED, fixed horizon — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

<span id="capability-ingresses-mean-cuped-fixed-horizon-from_unit_day_artifact"></span>

#### Mean CUPED, fixed horizon — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

<span id="capability-ingresses-mean-cuped-fixed-horizon-from_moments"></span>

#### Mean CUPED, fixed horizon — Portable moments

**Recorded path:** `from_moments`

**Status:** supported


<span id="capability-ingresses-ratio-cuped-fixed-horizon"></span>

### Ratio CUPED, fixed horizon

<span id="capability-ingresses-ratio-cuped-fixed-horizon-from_unit_summary"></span>

#### Ratio CUPED, fixed horizon — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

<span id="capability-ingresses-ratio-cuped-fixed-horizon-from_unit_panel"></span>

#### Ratio CUPED, fixed horizon — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

<span id="capability-ingresses-ratio-cuped-fixed-horizon-from_switchback_panel"></span>

#### Ratio CUPED, fixed horizon — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.metric`

**Reason:** switchback metrics must be a mean or a conversion

**Stage:** declaration

<span id="capability-ingresses-ratio-cuped-fixed-horizon-from_definitions"></span>

#### Ratio CUPED, fixed horizon — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

<span id="capability-ingresses-ratio-cuped-fixed-horizon-from_unit_day_artifact"></span>

#### Ratio CUPED, fixed horizon — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

<span id="capability-ingresses-ratio-cuped-fixed-horizon-from_moments"></span>

#### Ratio CUPED, fixed horizon — Portable moments

**Recorded path:** `from_moments`

**Status:** supported


<span id="capability-ingresses-mixed-covariate-cuped-fixed-horizon"></span>

### Mixed-covariate CUPED, fixed-horizon mean/ratio

<span id="capability-ingresses-mixed-covariate-cuped-fixed-horizon-from_unit_summary"></span>

#### Mixed-covariate CUPED, fixed-horizon mean/ratio — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** same per-unit mean, ratio, categorical and missing-history data

**Status:** supported

**Operation:** `run() mixed-covariate mean/ratio/conversion`

**Stage:** read

<span id="capability-ingresses-mixed-covariate-cuped-fixed-horizon-from_unit_panel"></span>

#### Mixed-covariate CUPED, fixed-horizon mean/ratio — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** same per-unit mean, ratio, categorical and missing-history data

**Status:** supported

**Operation:** `run() mixed-covariate mean/ratio/conversion`

**Stage:** read

<span id="capability-ingresses-mixed-covariate-cuped-fixed-horizon-from_switchback_panel"></span>

#### Mixed-covariate CUPED, fixed-horizon mean/ratio — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** switchback contrasts use a different design and construction

**Status:** source limited

**Classification:** SOURCE

**Reason:** use an independent-unit randomized ingress

**Operation:** `run() mixed-covariate mean`

**Stage:** construction

<span id="capability-ingresses-mixed-covariate-cuped-fixed-horizon-from_definitions"></span>

#### Mixed-covariate CUPED, fixed-horizon mean/ratio — Definitions

**Recorded path:** `from_definitions`

**Condition:** same per-unit mean, ratio, categorical and missing-history data; warehouse QR keeps unit values in place and leaves only the bounded triangular solve and fit finalization client-side (verified on DuckDB, PostgreSQL, Snowflake and BigQuery)

**Status:** supported

**Operation:** `run() mixed-covariate mean/ratio/conversion`

**Stage:** read

<span id="capability-ingresses-mixed-covariate-cuped-fixed-horizon-from_unit_day_artifact"></span>

#### Mixed-covariate CUPED, fixed-horizon mean/ratio — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** format-4 artifact extensions are SQL-verified; native QR keeps unit values in the warehouse and leaves only the bounded triangular solve and fit finalization client-side (verified on DuckDB, PostgreSQL, Snowflake and BigQuery)

**Status:** supported

**Operation:** `run() mixed-covariate mean/ratio/conversion`

**Stage:** read

<span id="capability-ingresses-mixed-covariate-cuped-fixed-horizon-from_moments"></span>

#### Mixed-covariate CUPED, fixed-horizon mean/ratio — Portable moments

**Recorded path:** `from_moments`

**Condition:** format-13 joint-cell evidence exported from a unit, definitions or artifact source

**Status:** supported

**Operation:** `run() categorical-only mixed-covariate mean/ratio with scalar companions`

**Stage:** read

**Condition:** format-13 fitted_multicovariate evidence replays the retained QR fit at a changed alpha

**Status:** supported

**Operation:** `run() continuous/mixed-covariate mean/ratio`

**Stage:** read


<span id="capability-ingresses-ratio-denominator-advisory-fixed-horizon"></span>

### Ratio denominator-precision advisory, fixed horizon

<span id="capability-ingresses-ratio-denominator-advisory-fixed-horizon-from_unit_summary"></span>

#### Ratio denominator-precision advisory, fixed horizon — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** fixed horizon; same matched units; denominator relative SE exceeds advisory threshold

**Status:** supported

**Operation:** `ratio lift with an imprecise denominator`

**Stage:** read

<span id="capability-ingresses-ratio-denominator-advisory-fixed-horizon-from_unit_panel"></span>

#### Ratio denominator-precision advisory, fixed horizon — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** fixed horizon; same matched units; denominator relative SE exceeds advisory threshold

**Status:** supported

**Operation:** `ratio lift with an imprecise denominator`

**Stage:** read

<span id="capability-ingresses-ratio-denominator-advisory-fixed-horizon-from_switchback_panel"></span>

#### Ratio denominator-precision advisory, fixed horizon — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.metric`

**Reason:** switchback metrics must be a mean or a conversion

**Stage:** declaration

<span id="capability-ingresses-ratio-denominator-advisory-fixed-horizon-from_definitions"></span>

#### Ratio denominator-precision advisory, fixed horizon — Definitions

**Recorded path:** `from_definitions`

**Condition:** fixed horizon; same matched units; denominator relative SE exceeds advisory threshold

**Status:** supported

**Operation:** `ratio lift with an imprecise denominator`

**Stage:** read

<span id="capability-ingresses-ratio-denominator-advisory-fixed-horizon-from_unit_day_artifact"></span>

#### Ratio denominator-precision advisory, fixed horizon — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** fixed horizon; same matched units; denominator relative SE exceeds advisory threshold

**Status:** supported

**Operation:** `ratio lift with an imprecise denominator`

**Stage:** read

<span id="capability-ingresses-ratio-denominator-advisory-fixed-horizon-from_moments"></span>

#### Ratio denominator-precision advisory, fixed horizon — Portable moments

**Recorded path:** `from_moments`

**Condition:** fixed horizon; same matched units; denominator relative SE exceeds advisory threshold

**Status:** supported

**Operation:** `ratio lift with an imprecise denominator`

**Stage:** read


<span id="capability-ingresses-informative-log-lift-priors"></span>

### Informative Normal/Student-t/mixture log-lift priors, fixed-horizon mean/ratio/conversion/mean CUPED

<span id="capability-ingresses-informative-log-lift-priors-from_unit_summary"></span>

#### Informative Normal/Student-t/mixture log-lift priors, fixed-horizon mean/ratio/conversion/mean CUPED — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** Normal prior

**Status:** supported

**Condition:** Student-t prior

**Status:** supported

**Condition:** mixture prior

**Status:** supported

<span id="capability-ingresses-informative-log-lift-priors-from_unit_panel"></span>

#### Informative Normal/Student-t/mixture log-lift priors, fixed-horizon mean/ratio/conversion/mean CUPED — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** Normal prior

**Status:** supported

**Condition:** Student-t prior

**Status:** supported

**Condition:** mixture prior

**Status:** supported

<span id="capability-ingresses-informative-log-lift-priors-from_switchback_panel"></span>

#### Informative Normal/Student-t/mixture log-lift priors, fixed-horizon mean/ratio/conversion/mean CUPED — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** Normal prior

**Status:** not comparable

**Classification:** SOURCE

**Reason:** from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.

**Condition:** Student-t prior

**Status:** not comparable

**Classification:** SOURCE

**Reason:** from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.

**Condition:** mixture prior

**Status:** not comparable

**Classification:** SOURCE

**Reason:** from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.

<span id="capability-ingresses-informative-log-lift-priors-from_definitions"></span>

#### Informative Normal/Student-t/mixture log-lift priors, fixed-horizon mean/ratio/conversion/mean CUPED — Definitions

**Recorded path:** `from_definitions`

**Condition:** Normal prior

**Status:** supported

**Condition:** Student-t prior

**Status:** supported

**Condition:** mixture prior

**Status:** supported

<span id="capability-ingresses-informative-log-lift-priors-from_unit_day_artifact"></span>

#### Informative Normal/Student-t/mixture log-lift priors, fixed-horizon mean/ratio/conversion/mean CUPED — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** Normal prior

**Status:** supported

**Condition:** Student-t prior

**Status:** supported

**Condition:** mixture prior

**Status:** supported

<span id="capability-ingresses-informative-log-lift-priors-from_moments"></span>

#### Informative Normal/Student-t/mixture log-lift priors, fixed-horizon mean/ratio/conversion/mean CUPED — Portable moments

**Recorded path:** `from_moments`

**Condition:** Normal prior

**Status:** supported

**Condition:** Student-t prior

**Status:** supported

**Condition:** mixture prior

**Status:** supported


<span id="capability-ingresses-informative-relative-lift-priors"></span>

### Relative Normal/Student-t/mixture priors, fixed-horizon mean/ratio/conversion/mean CUPED

<span id="capability-ingresses-informative-relative-lift-priors-from_unit_summary"></span>

#### Relative Normal/Student-t/mixture priors, fixed-horizon mean/ratio/conversion/mean CUPED — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** relative Normal prior

**Status:** supported

**Condition:** relative Student-t prior

**Status:** supported

**Condition:** relative mixture prior

**Status:** supported

<span id="capability-ingresses-informative-relative-lift-priors-from_unit_panel"></span>

#### Relative Normal/Student-t/mixture priors, fixed-horizon mean/ratio/conversion/mean CUPED — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** relative Normal prior

**Status:** supported

**Condition:** relative Student-t prior

**Status:** supported

**Condition:** relative mixture prior

**Status:** supported

<span id="capability-ingresses-informative-relative-lift-priors-from_switchback_panel"></span>

#### Relative Normal/Student-t/mixture priors, fixed-horizon mean/ratio/conversion/mean CUPED — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** relative Normal prior

**Status:** not comparable

**Classification:** SOURCE

**Reason:** from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.

**Condition:** relative Student-t prior

**Status:** not comparable

**Classification:** SOURCE

**Reason:** from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.

**Condition:** relative mixture prior

**Status:** not comparable

**Classification:** SOURCE

**Reason:** from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.

<span id="capability-ingresses-informative-relative-lift-priors-from_definitions"></span>

#### Relative Normal/Student-t/mixture priors, fixed-horizon mean/ratio/conversion/mean CUPED — Definitions

**Recorded path:** `from_definitions`

**Condition:** relative Normal prior

**Status:** supported

**Condition:** relative Student-t prior

**Status:** supported

**Condition:** relative mixture prior

**Status:** supported

<span id="capability-ingresses-informative-relative-lift-priors-from_unit_day_artifact"></span>

#### Relative Normal/Student-t/mixture priors, fixed-horizon mean/ratio/conversion/mean CUPED — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** relative Normal prior

**Status:** supported

**Condition:** relative Student-t prior

**Status:** supported

**Condition:** relative mixture prior

**Status:** supported

<span id="capability-ingresses-informative-relative-lift-priors-from_moments"></span>

#### Relative Normal/Student-t/mixture priors, fixed-horizon mean/ratio/conversion/mean CUPED — Portable moments

**Recorded path:** `from_moments`

**Condition:** relative Normal prior

**Status:** supported

**Condition:** relative Student-t prior

**Status:** supported

**Condition:** relative mixture prior

**Status:** supported


<span id="capability-ingresses-relative-prior-breakout"></span>

### Relative Normal prior on breakout rows

<span id="capability-ingresses-relative-prior-breakout-from_unit_summary"></span>

#### Relative Normal prior on breakout rows — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

<span id="capability-ingresses-relative-prior-breakout-from_unit_panel"></span>

#### Relative Normal prior on breakout rows — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

<span id="capability-ingresses-relative-prior-breakout-from_switchback_panel"></span>

#### Relative Normal prior on breakout rows — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** not comparable

**Classification:** SOURCE

**Reason:** from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.

<span id="capability-ingresses-relative-prior-breakout-from_definitions"></span>

#### Relative Normal prior on breakout rows — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

<span id="capability-ingresses-relative-prior-breakout-from_unit_day_artifact"></span>

#### Relative Normal prior on breakout rows — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

<span id="capability-ingresses-relative-prior-breakout-from_moments"></span>

#### Relative Normal prior on breakout rows — Portable moments

**Recorded path:** `from_moments`

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.breakout_unavailable`

**Reason:** a moments cube contains aggregate rows, not unit-level breakout dimensions.


<span id="capability-ingresses-relative-prior-retention"></span>

### Relative Normal prior on mature retention breakouts

<span id="capability-ingresses-relative-prior-retention-from_unit_summary"></span>

#### Relative Normal prior on mature retention breakouts — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.constructor`

**Reason:** unit summaries cannot supply retention exposure cohorts.

<span id="capability-ingresses-relative-prior-retention-from_unit_panel"></span>

#### Relative Normal prior on mature retention breakouts — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

<span id="capability-ingresses-relative-prior-retention-from_switchback_panel"></span>

#### Relative Normal prior on mature retention breakouts — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** not comparable

**Classification:** SOURCE

**Reason:** from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.

<span id="capability-ingresses-relative-prior-retention-from_definitions"></span>

#### Relative Normal prior on mature retention breakouts — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

<span id="capability-ingresses-relative-prior-retention-from_unit_day_artifact"></span>

#### Relative Normal prior on mature retention breakouts — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

<span id="capability-ingresses-relative-prior-retention-from_moments"></span>

#### Relative Normal prior on mature retention breakouts — Portable moments

**Recorded path:** `from_moments`

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.breakout_unavailable`

**Reason:** portable moments retain no breakout operation or cohort identity.


<span id="capability-ingresses-relative-prior-quantile"></span>

### Relative Normal prior on quantiles

<span id="capability-ingresses-relative-prior-quantile-from_unit_summary"></span>

#### Relative Normal prior on quantiles — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

<span id="capability-ingresses-relative-prior-quantile-from_unit_panel"></span>

#### Relative Normal prior on quantiles — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

<span id="capability-ingresses-relative-prior-quantile-from_switchback_panel"></span>

#### Relative Normal prior on quantiles — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** not comparable

**Classification:** SOURCE

**Reason:** switchback metrics must be type='mean' or type='conversion'; quantile is refused before any frame is read (confirmed live: source.frame.switchback.metric).

<span id="capability-ingresses-relative-prior-quantile-from_definitions"></span>

#### Relative Normal prior on quantiles — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

<span id="capability-ingresses-relative-prior-quantile-from_unit_day_artifact"></span>

#### Relative Normal prior on quantiles — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

<span id="capability-ingresses-relative-prior-quantile-from_moments"></span>

#### Relative Normal prior on quantiles — Portable moments

**Recorded path:** `from_moments`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.quantile_no_moments`

**Reason:** a quantile has no moments representation -- export() itself refuses before a from_moments cube could ever exist.


<span id="capability-ingresses-relative-prior-daily"></span>

### Relative Normal prior on triggered daily lift

<span id="capability-ingresses-relative-prior-daily-from_unit_summary"></span>

#### Relative Normal prior on triggered daily lift — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** not comparable

**Classification:** SOURCE

**Reason:** unit summaries carry no trigger declaration or per-unit trigger anchors.

<span id="capability-ingresses-relative-prior-daily-from_unit_panel"></span>

#### Relative Normal prior on triggered daily lift — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** not comparable

**Classification:** SOURCE

**Reason:** unit panels carry no trigger declaration or per-unit trigger anchors.

<span id="capability-ingresses-relative-prior-daily-from_switchback_panel"></span>

#### Relative Normal prior on triggered daily lift — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** not comparable

**Classification:** SOURCE

**Reason:** switchback contrasts carry no triggered arm day-axis evidence.

<span id="capability-ingresses-relative-prior-daily-from_definitions"></span>

#### Relative Normal prior on triggered daily lift — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

<span id="capability-ingresses-relative-prior-daily-from_unit_day_artifact"></span>

#### Relative Normal prior on triggered daily lift — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

<span id="capability-ingresses-relative-prior-daily-from_moments"></span>

#### Relative Normal prior on triggered daily lift — Portable moments

**Recorded path:** `from_moments`

**Status:** not comparable

**Classification:** SOURCE

**Reason:** portable moments carry no per-unit trigger anchors.


<span id="capability-ingresses-relative-prior-asof"></span>

### Relative Normal prior on triggered as-of lift

<span id="capability-ingresses-relative-prior-asof-from_unit_summary"></span>

#### Relative Normal prior on triggered as-of lift — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** not comparable

**Classification:** SOURCE

**Reason:** unit summaries carry no trigger declaration or per-unit trigger anchors.

<span id="capability-ingresses-relative-prior-asof-from_unit_panel"></span>

#### Relative Normal prior on triggered as-of lift — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** not comparable

**Classification:** SOURCE

**Reason:** unit panels carry no trigger declaration or per-unit trigger anchors.

<span id="capability-ingresses-relative-prior-asof-from_switchback_panel"></span>

#### Relative Normal prior on triggered as-of lift — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** not comparable

**Classification:** SOURCE

**Reason:** switchback contrasts carry no triggered arm day-axis evidence.

<span id="capability-ingresses-relative-prior-asof-from_definitions"></span>

#### Relative Normal prior on triggered as-of lift — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

<span id="capability-ingresses-relative-prior-asof-from_unit_day_artifact"></span>

#### Relative Normal prior on triggered as-of lift — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

<span id="capability-ingresses-relative-prior-asof-from_moments"></span>

#### Relative Normal prior on triggered as-of lift — Portable moments

**Recorded path:** `from_moments`

**Status:** not comparable

**Classification:** SOURCE

**Reason:** portable moments carry no per-unit trigger anchors.


<span id="capability-ingresses-fixed-threshold-winsor-sequential"></span>

### Fixed-threshold winsorization under sequential inference

<span id="capability-ingresses-fixed-threshold-winsor-sequential-from_unit_summary"></span>

#### Fixed-threshold winsorization under sequential inference — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

**Condition:** upper_value=20; the retained scalar-mean state holds the clipped per-unit means

**Status:** supported

**Operation:** `mean with fixed-threshold winsorization under asymptotic_mean`

**Stage:** read

<span id="capability-ingresses-fixed-threshold-winsor-sequential-from_unit_panel"></span>

#### Fixed-threshold winsorization under sequential inference — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

<span id="capability-ingresses-fixed-threshold-winsor-sequential-from_switchback_panel"></span>

#### Fixed-threshold winsorization under sequential inference — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.metric`

**Reason:** switchback aggregation applies no winsorization

**Stage:** declaration

<span id="capability-ingresses-fixed-threshold-winsor-sequential-from_definitions"></span>

#### Fixed-threshold winsorization under sequential inference — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

<span id="capability-ingresses-fixed-threshold-winsor-sequential-from_unit_day_artifact"></span>

#### Fixed-threshold winsorization under sequential inference — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

<span id="capability-ingresses-fixed-threshold-winsor-sequential-from_moments"></span>

#### Fixed-threshold winsorization under sequential inference — Portable moments

**Recorded path:** `from_moments`

**Status:** supported

**Note:** replays the exported checkpoint

<span id="capability-ingresses-fixed-threshold-winsor-sequential-family"></span>

#### Fixed-threshold winsorization under sequential inference: family-owned operations

**Condition:** fixed lower/upper value thresholds are predictable

**Status:** supported

**Operation:** `validate sequential transform`

**Stage:** read


<span id="capability-ingresses-pooled-percentile-winsor-routes"></span>

### Pooled percentile winsorization, fixed horizon, default size route

<span id="capability-ingresses-pooled-percentile-winsor-routes-from_unit_summary"></span>

#### Pooled percentile winsorization, fixed horizon, default size route — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** positive pool below the size route

**Status:** supported

**Condition:** zero-inclusive pool below the size route

**Status:** supported

**Condition:** zero-inclusive pool on the analytic size route

**Status:** supported

**Note:** zero outcomes are ordinary pool members; the executed route is recorded on the confidence set

<span id="capability-ingresses-pooled-percentile-winsor-routes-from_unit_panel"></span>

#### Pooled percentile winsorization, fixed horizon, default size route — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** positive pool below the size route

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.raw_outcomes_unavailable`

**Reason:** panel collapse does not retain exact pre-winsor unit outcomes

**Condition:** zero-inclusive pool below the size route

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.raw_outcomes_unavailable`

**Reason:** panel collapse does not retain exact pre-winsor unit outcomes

**Condition:** zero-inclusive pool on the analytic size route

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.raw_outcomes_unavailable`

**Reason:** panel collapse does not retain exact pre-winsor unit outcomes

<span id="capability-ingresses-pooled-percentile-winsor-routes-from_switchback_panel"></span>

#### Pooled percentile winsorization, fixed horizon, default size route — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.metric`

**Reason:** switchback aggregation applies no winsorization

**Stage:** declaration

<span id="capability-ingresses-pooled-percentile-winsor-routes-from_definitions"></span>

#### Pooled percentile winsorization, fixed horizon, default size route — Definitions

**Recorded path:** `from_definitions`

**Condition:** positive pool below the size route

**Status:** supported

**Condition:** zero-inclusive pool below the size route

**Status:** supported

**Condition:** zero-inclusive pool on the analytic size route

**Status:** supported

<span id="capability-ingresses-pooled-percentile-winsor-routes-from_unit_day_artifact"></span>

#### Pooled percentile winsorization, fixed horizon, default size route — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** positive pool below the size route

**Status:** supported

**Condition:** zero-inclusive pool below the size route

**Status:** supported

**Condition:** zero-inclusive pool on the analytic size route

**Status:** supported

<span id="capability-ingresses-pooled-percentile-winsor-routes-from_moments"></span>

#### Pooled percentile winsorization, fixed horizon, default size route — Portable moments

**Recorded path:** `from_moments`

**Condition:** positive pool below the size route

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.raw_outcomes_unavailable`

**Reason:** portable moments do not retain exact pre-winsor unit outcomes

**Condition:** zero-inclusive pool below the size route

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.raw_outcomes_unavailable`

**Reason:** portable moments do not retain exact pre-winsor unit outcomes

**Condition:** zero-inclusive pool on the analytic size route

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.raw_outcomes_unavailable`

**Reason:** portable moments do not retain exact pre-winsor unit outcomes


<span id="capability-ingresses-ratio-sequential"></span>

### Ratio metric under sequential inference

<span id="capability-ingresses-ratio-sequential-from_unit_summary"></span>

#### Ratio metric under sequential inference — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

<span id="capability-ingresses-ratio-sequential-from_unit_panel"></span>

#### Ratio metric under sequential inference — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

<span id="capability-ingresses-ratio-sequential-from_switchback_panel"></span>

#### Ratio metric under sequential inference — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.metric`

**Reason:** switchback metrics must be a mean or a conversion

**Stage:** declaration

<span id="capability-ingresses-ratio-sequential-from_definitions"></span>

#### Ratio metric under sequential inference — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

<span id="capability-ingresses-ratio-sequential-from_unit_day_artifact"></span>

#### Ratio metric under sequential inference — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

<span id="capability-ingresses-ratio-sequential-from_moments"></span>

#### Ratio metric under sequential inference — Portable moments

**Recorded path:** `from_moments`

**Status:** supported

**Note:** replays the exported checkpoint


<span id="capability-ingresses-mean-cuped-sequential"></span>

### CUPED under sequential inference

<span id="capability-ingresses-mean-cuped-sequential-from_unit_summary"></span>

#### CUPED under sequential inference — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

**Condition:** predeclared coefficient and center; retained adjusted scalar narrows interval

**Status:** supported

**Operation:** `mean CUPED under asymptotic_mean`

**Stage:** read

**Condition:** in-experiment coefficient; the adjusted_mean law retains joint (Y, X) moments

**Status:** supported

**Operation:** `mean CUPED under asymptotic_mean`

**Stage:** read

**Note:** asymptotic route

<span id="capability-ingresses-mean-cuped-sequential-from_unit_panel"></span>

#### CUPED under sequential inference — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** refused

**Classification:** CONSTRUCTION

**Codes:** `frame.validation.from_unit_panel`

**Reason:** panel sequential capture needs a bounded window, and a windowed metric has no per-unit collapse to attach the CUPED covariate to.

<span id="capability-ingresses-mean-cuped-sequential-from_switchback_panel"></span>

#### CUPED under sequential inference — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.plan`

**Reason:** the switchback plan gate refuses the CUPED decision override and the sequential inference before any frame is read; its contrasts are fixed-horizon.

<span id="capability-ingresses-mean-cuped-sequential-from_definitions"></span>

#### CUPED under sequential inference — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

**Note:** asymptotic route

<span id="capability-ingresses-mean-cuped-sequential-from_unit_day_artifact"></span>

#### CUPED under sequential inference — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

**Note:** asymptotic route

<span id="capability-ingresses-mean-cuped-sequential-from_moments"></span>

#### CUPED under sequential inference — Portable moments

**Recorded path:** `from_moments`

**Status:** supported

**Note:** replays the exported checkpoint

<span id="capability-ingresses-mean-cuped-sequential-family"></span>

#### CUPED under sequential inference: family-owned operations

**Condition:** an in-experiment CUPED coefficient on the exact Bernoulli route

**Status:** refused

**Codes:** `sequential.transform.unpredictable`

**Reason:** the exact Bernoulli e-process admits no CUPED coefficient

**Operation:** `validate sequential methods`

**Stage:** construction


<span id="capability-ingresses-ratio-cuped-sequential"></span>

### Ratio CUPED under sequential inference

<span id="capability-ingresses-ratio-cuped-sequential-from_unit_summary"></span>

#### Ratio CUPED under sequential inference — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** overlapping arm denominators

**Status:** supported

**Condition:** separated arm denominators

**Status:** supported

**Note:** asymptotic route

<span id="capability-ingresses-ratio-cuped-sequential-from_unit_panel"></span>

#### Ratio CUPED under sequential inference — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** overlapping arm denominators

**Status:** refused

**Classification:** SOURCE

**Codes:** `frame.validation.from_unit_panel`

**Reason:** registered panel capture cannot supply the per-unit pre-period covariate for adjusted_ratio_mean; use a unit-summary frame.

**Condition:** separated arm denominators

**Status:** refused

**Classification:** SOURCE

**Codes:** `frame.validation.from_unit_panel`

**Reason:** registered panel capture cannot supply the per-unit pre-period covariate for adjusted_ratio_mean; use a unit-summary frame.

<span id="capability-ingresses-ratio-cuped-sequential-from_switchback_panel"></span>

#### Ratio CUPED under sequential inference — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.metric`

**Reason:** switchback metrics must be a mean or a conversion

**Stage:** declaration

<span id="capability-ingresses-ratio-cuped-sequential-from_definitions"></span>

#### Ratio CUPED under sequential inference — Definitions

**Recorded path:** `from_definitions`

**Condition:** overlapping arm denominators

**Status:** supported

**Condition:** separated arm denominators

**Status:** supported

**Note:** asymptotic route

<span id="capability-ingresses-ratio-cuped-sequential-from_unit_day_artifact"></span>

#### Ratio CUPED under sequential inference — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** overlapping arm denominators

**Status:** supported

**Condition:** separated arm denominators

**Status:** supported

**Note:** asymptotic route

<span id="capability-ingresses-ratio-cuped-sequential-from_moments"></span>

#### Ratio CUPED under sequential inference — Portable moments

**Recorded path:** `from_moments`

**Condition:** overlapping arm denominators

**Status:** supported

**Condition:** separated arm denominators

**Status:** supported

**Note:** replays the exported checkpoint


<span id="capability-ingresses-bounded-conversion-asymptotic-sequential"></span>

### Bounded conversion under asymptotic sequential inference

<span id="capability-ingresses-bounded-conversion-asymptotic-sequential-from_unit_summary"></span>

#### Bounded conversion under asymptotic sequential inference — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** asymptotic_mean; retain Bernoulli registration, model state and interval equal to unit-summary oracle

**Status:** supported

**Operation:** `bounded conversion under sequential inference`

**Stage:** read

**Note:** pre-windowed per-unit outcomes with an exposure date

<span id="capability-ingresses-bounded-conversion-asymptotic-sequential-from_unit_panel"></span>

#### Bounded conversion under asymptotic sequential inference — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** asymptotic_mean; retain Bernoulli registration, model state and interval equal to unit-summary oracle

**Status:** supported

**Operation:** `bounded conversion under sequential inference`

**Stage:** read

<span id="capability-ingresses-bounded-conversion-asymptotic-sequential-from_switchback_panel"></span>

#### Bounded conversion under asymptotic sequential inference — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.plan`

**Reason:** switchback contrasts support fixed inference only

**Stage:** declaration

<span id="capability-ingresses-bounded-conversion-asymptotic-sequential-from_definitions"></span>

#### Bounded conversion under asymptotic sequential inference — Definitions

**Recorded path:** `from_definitions`

**Condition:** asymptotic_mean; retain Bernoulli registration, model state and interval equal to unit-summary oracle

**Status:** supported

**Operation:** `bounded conversion under sequential inference`

**Stage:** read

<span id="capability-ingresses-bounded-conversion-asymptotic-sequential-from_unit_day_artifact"></span>

#### Bounded conversion under asymptotic sequential inference — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** asymptotic_mean; retain Bernoulli registration, model state and interval equal to unit-summary oracle

**Status:** supported

**Operation:** `bounded conversion under sequential inference`

**Stage:** read

<span id="capability-ingresses-bounded-conversion-asymptotic-sequential-from_moments"></span>

#### Bounded conversion under asymptotic sequential inference — Portable moments

**Recorded path:** `from_moments`

**Condition:** asymptotic_mean; retain Bernoulli registration, model state and interval equal to unit-summary oracle

**Status:** supported

**Operation:** `bounded conversion under sequential inference`

**Stage:** read

**Note:** replays the exported checkpoint


<span id="capability-ingresses-bounded-retention-asymptotic-sequential"></span>

### Bounded retention (a start and end threshold) under asymptotic sequential inference

<span id="capability-ingresses-bounded-retention-asymptotic-sequential-from_unit_summary"></span>

#### Bounded retention (a start and end threshold) under asymptotic sequential inference — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.constructor`

**Reason:** a retention band needs an exposure date to count days from; a summary has none

**Stage:** declaration

<span id="capability-ingresses-bounded-retention-asymptotic-sequential-from_unit_panel"></span>

#### Bounded retention (a start and end threshold) under asymptotic sequential inference — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

**Reason:** runs and agrees with every other running matched-arm ingress once the request declares an exposure date and an observation band; a registered asymptotic or exact Bernoulli inference, a declared allocation and windows covered by the common reveal window

**Stage:** request

<span id="capability-ingresses-bounded-retention-asymptotic-sequential-from_switchback_panel"></span>

#### Bounded retention (a start and end threshold) under asymptotic sequential inference — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.metric`

**Reason:** switchback metrics must be a mean or a conversion

**Stage:** declaration

<span id="capability-ingresses-bounded-retention-asymptotic-sequential-from_definitions"></span>

#### Bounded retention (a start and end threshold) under asymptotic sequential inference — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

**Reason:** runs and agrees with every other running matched-arm ingress once the request declares an exposure date and an observation band; a registered asymptotic or exact Bernoulli inference, a declared allocation and windows covered by the common reveal window

**Stage:** request

<span id="capability-ingresses-bounded-retention-asymptotic-sequential-from_unit_day_artifact"></span>

#### Bounded retention (a start and end threshold) under asymptotic sequential inference — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

**Reason:** runs and agrees with every other running matched-arm ingress once the request declares an exposure date and an observation band; a registered asymptotic or exact Bernoulli inference, a declared allocation and windows covered by the common reveal window

**Stage:** request

<span id="capability-ingresses-bounded-retention-asymptotic-sequential-from_moments"></span>

#### Bounded retention (a start and end threshold) under asymptotic sequential inference — Portable moments

**Recorded path:** `from_moments`

**Status:** supported

**Reason:** runs and agrees with every other running matched-arm ingress once the request declares an exposure date and an observation band; a registered asymptotic or exact Bernoulli inference, a declared allocation and windows covered by the common reveal window

**Stage:** request


<span id="capability-ingresses-bounded-conversion-exact-bernoulli"></span>

### Bounded conversion under automatic exact Bernoulli monitoring without a registration

<span id="capability-ingresses-bounded-conversion-exact-bernoulli-from_unit_summary"></span>

#### Bounded conversion under automatic exact Bernoulli monitoring without a registration — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** always_valid; retain Bernoulli registration, model state and interval equal to unit-summary oracle

**Status:** supported

**Operation:** `bounded conversion under sequential inference`

**Stage:** read

**Note:** pre-windowed per-unit outcomes with an exposure date

<span id="capability-ingresses-bounded-conversion-exact-bernoulli-from_unit_panel"></span>

#### Bounded conversion under automatic exact Bernoulli monitoring without a registration — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** always_valid; retain Bernoulli registration, model state and interval equal to unit-summary oracle

**Status:** supported

**Operation:** `bounded conversion under sequential inference`

**Stage:** read

<span id="capability-ingresses-bounded-conversion-exact-bernoulli-from_switchback_panel"></span>

#### Bounded conversion under automatic exact Bernoulli monitoring without a registration — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.plan`

**Reason:** switchback contrasts support fixed inference only

**Stage:** declaration

<span id="capability-ingresses-bounded-conversion-exact-bernoulli-from_definitions"></span>

#### Bounded conversion under automatic exact Bernoulli monitoring without a registration — Definitions

**Recorded path:** `from_definitions`

**Condition:** always_valid; retain Bernoulli registration, model state and interval equal to unit-summary oracle

**Status:** supported

**Operation:** `bounded conversion under sequential inference`

**Stage:** read

<span id="capability-ingresses-bounded-conversion-exact-bernoulli-from_unit_day_artifact"></span>

#### Bounded conversion under automatic exact Bernoulli monitoring without a registration — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** always_valid; retain Bernoulli registration, model state and interval equal to unit-summary oracle

**Status:** supported

**Operation:** `bounded conversion under sequential inference`

**Stage:** read

<span id="capability-ingresses-bounded-conversion-exact-bernoulli-from_moments"></span>

#### Bounded conversion under automatic exact Bernoulli monitoring without a registration — Portable moments

**Recorded path:** `from_moments`

**Condition:** always_valid; retain Bernoulli registration, model state and interval equal to unit-summary oracle

**Status:** supported

**Operation:** `bounded conversion under sequential inference`

**Stage:** read

**Note:** replays the exported checkpoint


<span id="capability-ingresses-bounded-retention-exact-bernoulli"></span>

### Bounded retention under automatic exact Bernoulli monitoring without a registration

<span id="capability-ingresses-bounded-retention-exact-bernoulli-from_unit_summary"></span>

#### Bounded retention under automatic exact Bernoulli monitoring without a registration — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.constructor`

**Reason:** a retention band needs an exposure date to count days from; a summary has none

**Stage:** declaration

<span id="capability-ingresses-bounded-retention-exact-bernoulli-from_unit_panel"></span>

#### Bounded retention under automatic exact Bernoulli monitoring without a registration — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

**Reason:** runs and agrees with every other running matched-arm ingress once the request declares an exposure date and an observation band; a registered asymptotic or exact Bernoulli inference, a declared allocation and windows covered by the common reveal window

**Stage:** request

<span id="capability-ingresses-bounded-retention-exact-bernoulli-from_switchback_panel"></span>

#### Bounded retention under automatic exact Bernoulli monitoring without a registration — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.metric`

**Reason:** switchback metrics must be a mean or a conversion

**Stage:** declaration

<span id="capability-ingresses-bounded-retention-exact-bernoulli-from_definitions"></span>

#### Bounded retention under automatic exact Bernoulli monitoring without a registration — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

**Reason:** runs and agrees with every other running matched-arm ingress once the request declares an exposure date and an observation band; a registered asymptotic or exact Bernoulli inference, a declared allocation and windows covered by the common reveal window

**Stage:** request

<span id="capability-ingresses-bounded-retention-exact-bernoulli-from_unit_day_artifact"></span>

#### Bounded retention under automatic exact Bernoulli monitoring without a registration — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

**Reason:** runs and agrees with every other running matched-arm ingress once the request declares an exposure date and an observation band; a registered asymptotic or exact Bernoulli inference, a declared allocation and windows covered by the common reveal window

**Stage:** request

<span id="capability-ingresses-bounded-retention-exact-bernoulli-from_moments"></span>

#### Bounded retention under automatic exact Bernoulli monitoring without a registration — Portable moments

**Recorded path:** `from_moments`

**Status:** supported

**Reason:** runs and agrees with every other running matched-arm ingress once the request declares an exposure date and an observation band; a registered asymptotic or exact Bernoulli inference, a declared allocation and windows covered by the common reveal window

**Stage:** request


<span id="capability-ingresses-multi-arm-bounded-conversion-exact"></span>

### Automatic exact multi-arm bounded-conversion monitoring

<span id="capability-ingresses-multi-arm-bounded-conversion-exact-from_unit_summary"></span>

#### Automatic exact multi-arm bounded-conversion monitoring — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

**Note:** pre-windowed per-unit outcomes with an exposure date

<span id="capability-ingresses-multi-arm-bounded-conversion-exact-from_unit_panel"></span>

#### Automatic exact multi-arm bounded-conversion monitoring — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

<span id="capability-ingresses-multi-arm-bounded-conversion-exact-from_switchback_panel"></span>

#### Automatic exact multi-arm bounded-conversion monitoring — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** Randomized allocation has control plus multiple treatment arms; conversion metric

**Status:** refused

**Codes:** `source.frame.switchback.identification`

**Reason:** Switchback identification requires exactly one control and one treatment arm, before metric or sequential-plan validation.

**Operation:** `from_switchback_frame()`

**Stage:** construction

**Condition:** request reduced to control plus one treatment; sequential plan

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.plan`

**Reason:** switchback contrasts support fixed inference only

**Stage:** declaration

<span id="capability-ingresses-multi-arm-bounded-conversion-exact-from_definitions"></span>

#### Automatic exact multi-arm bounded-conversion monitoring — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

<span id="capability-ingresses-multi-arm-bounded-conversion-exact-from_unit_day_artifact"></span>

#### Automatic exact multi-arm bounded-conversion monitoring — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

<span id="capability-ingresses-multi-arm-bounded-conversion-exact-from_moments"></span>

#### Automatic exact multi-arm bounded-conversion monitoring — Portable moments

**Recorded path:** `from_moments`

**Status:** supported

**Note:** replays the exported checkpoint


<span id="capability-ingresses-multi-arm-bounded-retention-exact"></span>

### Automatic exact multi-arm bounded-retention monitoring

<span id="capability-ingresses-multi-arm-bounded-retention-exact-from_unit_summary"></span>

#### Automatic exact multi-arm bounded-retention monitoring — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.constructor`

**Reason:** a retention band needs an exposure date to count days from; a summary has none

**Stage:** declaration

<span id="capability-ingresses-multi-arm-bounded-retention-exact-from_unit_panel"></span>

#### Automatic exact multi-arm bounded-retention monitoring — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** not measured

**Reason:** no parity case exercises multi-arm retention monitoring

<span id="capability-ingresses-multi-arm-bounded-retention-exact-from_switchback_panel"></span>

#### Automatic exact multi-arm bounded-retention monitoring — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** Randomized allocation has control plus multiple treatment arms; retention metric

**Status:** refused

**Codes:** `source.frame.switchback.identification`

**Reason:** Switchback identification requires exactly one control and one treatment arm, before metric or sequential-plan validation.

**Operation:** `from_switchback_frame()`

**Stage:** construction

**Condition:** request reduced to control plus one treatment; retention metric

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.metric`

**Reason:** switchback metrics must be a mean or a conversion

**Stage:** declaration

<span id="capability-ingresses-multi-arm-bounded-retention-exact-from_definitions"></span>

#### Automatic exact multi-arm bounded-retention monitoring — Definitions

**Recorded path:** `from_definitions`

**Status:** not measured

**Reason:** no parity case exercises multi-arm retention monitoring

<span id="capability-ingresses-multi-arm-bounded-retention-exact-from_unit_day_artifact"></span>

#### Automatic exact multi-arm bounded-retention monitoring — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** not measured

**Reason:** no parity case exercises multi-arm retention monitoring

<span id="capability-ingresses-multi-arm-bounded-retention-exact-from_moments"></span>

#### Automatic exact multi-arm bounded-retention monitoring — Portable moments

**Recorded path:** `from_moments`

**Status:** not measured

**Reason:** no parity case exercises multi-arm retention monitoring


<span id="capability-ingresses-unbounded-conversion-sequential"></span>

### Unbounded conversion under sequential inference

<span id="capability-ingresses-unbounded-conversion-sequential-from_unit_summary"></span>

#### Unbounded conversion under sequential inference — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** refused

**Classification:** UNFINISHED

**Codes:** `sequential.source.invalid`

**Reason:** an unbounded-window sequential request raises a source code here while the definitions and panel routes raise sequential.route.unsupported

**Stage:** request

**Tracked issue:** `1cr4`

<span id="capability-ingresses-unbounded-conversion-sequential-from_unit_panel"></span>

#### Unbounded conversion under sequential inference — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** refused

**Classification:** UNFINISHED

**Codes:** `sequential.route.unsupported`

**Reason:** panel metrics need bounded windows covered by the common joint-reveal window; the hazard raises 2 codes across routes (sequential.route.unsupported, sequential.source.invalid), one hazard needs one code

**Stage:** declaration

**Tracked issue:** `1cr4`

<span id="capability-ingresses-unbounded-conversion-sequential-from_switchback_panel"></span>

#### Unbounded conversion under sequential inference — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.plan`

**Reason:** switchback contrasts support fixed inference only

**Stage:** declaration

<span id="capability-ingresses-unbounded-conversion-sequential-from_definitions"></span>

#### Unbounded conversion under sequential inference — Definitions

**Recorded path:** `from_definitions`

**Status:** refused

**Classification:** UNFINISHED

**Codes:** `sequential.route.unsupported`

**Reason:** every observation window must be bounded by the common registered reveal window; the hazard raises 2 codes across routes (sequential.route.unsupported, sequential.source.invalid), one hazard needs one code

**Stage:** request

**Tracked issue:** `1cr4`

<span id="capability-ingresses-unbounded-conversion-sequential-from_unit_day_artifact"></span>

#### Unbounded conversion under sequential inference — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** refused

**Classification:** UNFINISHED

**Codes:** `sequential.route.unsupported`

**Reason:** every observation window must be bounded by the common registered reveal window; the hazard raises 2 codes across routes (sequential.route.unsupported, sequential.source.invalid), one hazard needs one code

**Stage:** request

**Tracked issue:** `1cr4`

<span id="capability-ingresses-unbounded-conversion-sequential-from_moments"></span>

#### Unbounded conversion under sequential inference — Portable moments

**Recorded path:** `from_moments`

**Status:** refused

**Classification:** SOURCE

**Codes:** `sequential.source.invalid`

**Reason:** a cube replays an exported checkpoint; it holds no per-unit source to start a sequential process from

**Stage:** declaration


<span id="capability-ingresses-unbounded-retention-sequential"></span>

### Unbounded retention (an integer threshold) under sequential inference

<span id="capability-ingresses-unbounded-retention-sequential-from_unit_summary"></span>

#### Unbounded retention (an integer threshold) under sequential inference — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** summary has no exposure dates to resolve the unbounded retention band

**Status:** refused

**Codes:** `source.frame.constructor`

**Reason:** retention needs an exposure date; a one-row-per-unit summary has none

**Operation:** `sequential retention with threshold_days=int`

**Stage:** construction

<span id="capability-ingresses-unbounded-retention-sequential-from_unit_panel"></span>

#### Unbounded retention (an integer threshold) under sequential inference — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** no disposition

<span id="capability-ingresses-unbounded-retention-sequential-from_switchback_panel"></span>

#### Unbounded retention (an integer threshold) under sequential inference — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** switchback source refuses retention metric before sequential monitoring

**Status:** refused

**Codes:** `source.frame.switchback.metric`

**Reason:** switchback source only constructs its supported mean/conversion metrics

**Operation:** `sequential retention with threshold_days=int`

**Stage:** construction

<span id="capability-ingresses-unbounded-retention-sequential-from_definitions"></span>

#### Unbounded retention (an integer threshold) under sequential inference — Definitions

**Recorded path:** `from_definitions`

**Condition:** the unbounded observation band has no finalized window

**Status:** refused

**Codes:** `sequential.route.unsupported`

**Reason:** sequential capture requires every observation window to have a bounded end

**Operation:** `capture_sequential(retention, threshold_days=int)`

**Stage:** read

<span id="capability-ingresses-unbounded-retention-sequential-from_unit_day_artifact"></span>

#### Unbounded retention (an integer threshold) under sequential inference — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** no disposition

<span id="capability-ingresses-unbounded-retention-sequential-from_moments"></span>

#### Unbounded retention (an integer threshold) under sequential inference — Portable moments

**Recorded path:** `from_moments`

**Condition:** cube exported from the fixed-horizon definitions readout; the registered route captured no checkpoint for the unbounded band

**Status:** refused

**Codes:** `sequential.capture.required`

**Reason:** portable moments only replay an upstream checkpoint, and none can be captured for an unbounded retention band

**Operation:** `capture_sequential(retention, threshold_days=int)`

**Stage:** read


<span id="capability-ingresses-unit-summary-whole-window-breakouts"></span>

### Whole-window breakouts from one-row-per-unit summaries

<span id="capability-ingresses-unit-summary-whole-window-breakouts-from_unit_summary"></span>

#### Whole-window breakouts from one-row-per-unit summaries — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

**Note:** breakouts do not create daily or as-of capabilities

<span id="capability-ingresses-unit-summary-whole-window-breakouts-from_unit_panel"></span>

#### Whole-window breakouts from one-row-per-unit summaries — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

<span id="capability-ingresses-unit-summary-whole-window-breakouts-from_switchback_panel"></span>

#### Whole-window breakouts from one-row-per-unit summaries — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** not comparable

**Classification:** SOURCE

**Reason:** from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.

<span id="capability-ingresses-unit-summary-whole-window-breakouts-from_definitions"></span>

#### Whole-window breakouts from one-row-per-unit summaries — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

<span id="capability-ingresses-unit-summary-whole-window-breakouts-from_unit_day_artifact"></span>

#### Whole-window breakouts from one-row-per-unit summaries — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

<span id="capability-ingresses-unit-summary-whole-window-breakouts-from_moments"></span>

#### Whole-window breakouts from one-row-per-unit summaries — Portable moments

**Recorded path:** `from_moments`

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.breakout_unavailable`

**Reason:** a moments cube contains aggregate rows, not unit-level breakout dimensions.

**Note:** portable moments contain no per-unit segment labels


<span id="capability-ingresses-registered-segmented-sequential-family"></span>

### Randomized registered segmented sequential family (automatic predeclared levels or explicit roster)

<span id="capability-ingresses-registered-segmented-sequential-family-from_unit_summary"></span>

#### Randomized registered segmented sequential family (automatic predeclared levels or explicit roster) — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** automatic levels, continuous metric

**Status:** supported

**Condition:** automatic levels, discrete metric

**Status:** supported

**Condition:** explicit roster, continuous metric

**Status:** supported

**Condition:** explicit roster, discrete metric

**Status:** supported

<span id="capability-ingresses-registered-segmented-sequential-family-from_unit_panel"></span>

#### Randomized registered segmented sequential family (automatic predeclared levels or explicit roster) — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** automatic levels, continuous metric

**Status:** supported

**Condition:** automatic levels, discrete metric

**Status:** supported

**Condition:** explicit roster, continuous metric

**Status:** supported

**Condition:** explicit roster, discrete metric

**Status:** supported

<span id="capability-ingresses-registered-segmented-sequential-family-from_switchback_panel"></span>

#### Randomized registered segmented sequential family (automatic predeclared levels or explicit roster) — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.plan`

**Reason:** switchback contrasts support fixed inference only

**Stage:** declaration

<span id="capability-ingresses-registered-segmented-sequential-family-from_definitions"></span>

#### Randomized registered segmented sequential family (automatic predeclared levels or explicit roster) — Definitions

**Recorded path:** `from_definitions`

**Condition:** automatic levels, continuous metric

**Status:** refused

**Classification:** SOURCE

**Codes:** `sequential.route.unsupported`

**Reason:** given the same segmented registration as from_unit_summary, a relational source (validate_relational_capture, shared by native_source.py and artifact_reader.py) refuses that segmented sequential roster outright -- 'relational segment capture needs a registered immutable property relation; use finalized joint records or frame segments'.

**Condition:** automatic levels, discrete metric

**Status:** refused

**Classification:** SOURCE

**Codes:** `sequential.route.unsupported`

**Reason:** given the same segmented registration as from_unit_summary, a relational source (validate_relational_capture, shared by native_source.py and artifact_reader.py) refuses that segmented sequential roster outright -- 'relational segment capture needs a registered immutable property relation; use finalized joint records or frame segments'.

**Condition:** explicit roster, continuous metric

**Status:** refused

**Classification:** SOURCE

**Codes:** `sequential.route.unsupported`

**Reason:** given the same segmented registration as from_unit_summary, a relational source (validate_relational_capture, shared by native_source.py and artifact_reader.py) refuses that segmented sequential roster outright -- 'relational segment capture needs a registered immutable property relation; use finalized joint records or frame segments'.

**Condition:** explicit roster, discrete metric

**Status:** refused

**Classification:** SOURCE

**Codes:** `sequential.route.unsupported`

**Reason:** given the same segmented registration as from_unit_summary, a relational source (validate_relational_capture, shared by native_source.py and artifact_reader.py) refuses that segmented sequential roster outright -- 'relational segment capture needs a registered immutable property relation; use finalized joint records or frame segments'.

<span id="capability-ingresses-registered-segmented-sequential-family-from_unit_day_artifact"></span>

#### Randomized registered segmented sequential family (automatic predeclared levels or explicit roster) — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** automatic levels, continuous metric

**Status:** refused

**Classification:** SOURCE

**Codes:** `sequential.route.unsupported`

**Reason:** publish/adopt re-derives the experiment's declared plan's own automatic registration -- segmented from the predeclared family, so the relational capture check above refuses it for the same reason.

**Condition:** automatic levels, discrete metric

**Status:** refused

**Classification:** SOURCE

**Codes:** `sequential.route.unsupported`

**Reason:** publish/adopt re-derives the experiment's declared plan's own automatic registration -- segmented from the predeclared family, so the relational capture check above refuses it for the same reason.

**Condition:** explicit roster, continuous metric

**Status:** refused

**Classification:** SOURCE

**Codes:** `sequential.route.unsupported`

**Reason:** publish/adopt always re-derives an automatic (unsegmented) registration from the experiment's declared plan -- a caller-built segmented registration never reaches it -- and the relational capture check above would refuse it regardless.

**Condition:** explicit roster, discrete metric

**Status:** refused

**Classification:** SOURCE

**Codes:** `sequential.route.unsupported`

**Reason:** publish/adopt always re-derives an automatic (unsegmented) registration from the experiment's declared plan -- a caller-built segmented registration never reaches it -- and the relational capture check above would refuse it regardless.

<span id="capability-ingresses-registered-segmented-sequential-family-from_moments"></span>

#### Randomized registered segmented sequential family (automatic predeclared levels or explicit roster) — Portable moments

**Recorded path:** `from_moments`

**Condition:** automatic levels, continuous metric

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.breakout_unavailable`

**Reason:** MomentsSource declares no breakout catalog at all -- a replayed checkpoint carries the segmented registration intact (export survives it), but run_breakout still refuses because a moments cube has no breakout dimension to serve.

**Condition:** automatic levels, discrete metric

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.breakout_unavailable`

**Reason:** MomentsSource declares no breakout catalog at all -- a replayed checkpoint carries the segmented registration intact (export survives it), but run_breakout still refuses because a moments cube has no breakout dimension to serve.

**Condition:** explicit roster, continuous metric

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.breakout_unavailable`

**Reason:** MomentsSource declares no breakout catalog at all -- a replayed checkpoint carries the segmented registration intact (export survives it), but run_breakout still refuses because a moments cube has no breakout dimension to serve.

**Condition:** explicit roster, discrete metric

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.breakout_unavailable`

**Reason:** MomentsSource declares no breakout catalog at all -- a replayed checkpoint carries the segmented registration intact (export survives it), but run_breakout still refuses because a moments cube has no breakout dimension to serve.

**Note:** replays the retained segment state; a cube carries no breakout catalog


<span id="capability-ingresses-boolean-null-segment-labels"></span>

### Boolean and null segment labels in a registered segmented sequential family (canonical true/false/null spellings matching the string-labelled oracle)

<span id="capability-ingresses-boolean-null-segment-labels-from_unit_summary"></span>

#### Boolean and null segment labels in a registered segmented sequential family (canonical true/false/null spellings matching the string-labelled oracle) — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** automatic roster, exact_bernoulli law

**Status:** supported

**Condition:** automatic roster, asymptotic_mean law

**Status:** supported

**Condition:** registered roster, exact_bernoulli law

**Status:** supported

**Condition:** registered roster, asymptotic_mean law

**Status:** supported

**Note:** measured on pandas, Polars and Arrow Boolean columns

<span id="capability-ingresses-boolean-null-segment-labels-from_unit_panel"></span>

#### Boolean and null segment labels in a registered segmented sequential family (canonical true/false/null spellings matching the string-labelled oracle) — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** automatic roster, exact_bernoulli law

**Status:** supported

**Condition:** automatic roster, asymptotic_mean law

**Status:** supported

**Condition:** registered roster, exact_bernoulli law

**Status:** supported

**Condition:** registered roster, asymptotic_mean law

**Status:** supported

**Note:** measured on pandas, Polars and Arrow Boolean columns

<span id="capability-ingresses-boolean-null-segment-labels-from_switchback_panel"></span>

#### Boolean and null segment labels in a registered segmented sequential family (canonical true/false/null spellings matching the string-labelled oracle) — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.plan`

**Reason:** switchback contrasts support fixed inference only

**Stage:** declaration

<span id="capability-ingresses-boolean-null-segment-labels-from_definitions"></span>

#### Boolean and null segment labels in a registered segmented sequential family (canonical true/false/null spellings matching the string-labelled oracle) — Definitions

**Recorded path:** `from_definitions`

**Condition:** automatic roster, exact_bernoulli law

**Status:** refused

**Classification:** SOURCE

**Codes:** `sequential.route.unsupported`

**Reason:** given the same segmented registration as from_unit_summary, a relational source (validate_relational_capture, shared by native_source.py and artifact_reader.py) refuses that segmented sequential roster outright -- 'relational segment capture needs a registered immutable property relation; use finalized joint records or frame segments'.

**Condition:** automatic roster, asymptotic_mean law

**Status:** refused

**Classification:** SOURCE

**Codes:** `sequential.route.unsupported`

**Reason:** given the same segmented registration as from_unit_summary, a relational source (validate_relational_capture, shared by native_source.py and artifact_reader.py) refuses that segmented sequential roster outright -- 'relational segment capture needs a registered immutable property relation; use finalized joint records or frame segments'.

**Condition:** registered roster, exact_bernoulli law

**Status:** refused

**Classification:** SOURCE

**Codes:** `sequential.route.unsupported`

**Reason:** given the same segmented registration as from_unit_summary, a relational source (validate_relational_capture, shared by native_source.py and artifact_reader.py) refuses that segmented sequential roster outright -- 'relational segment capture needs a registered immutable property relation; use finalized joint records or frame segments'.

**Condition:** registered roster, asymptotic_mean law

**Status:** refused

**Classification:** SOURCE

**Codes:** `sequential.route.unsupported`

**Reason:** given the same segmented registration as from_unit_summary, a relational source (validate_relational_capture, shared by native_source.py and artifact_reader.py) refuses that segmented sequential roster outright -- 'relational segment capture needs a registered immutable property relation; use finalized joint records or frame segments'.

<span id="capability-ingresses-boolean-null-segment-labels-from_unit_day_artifact"></span>

#### Boolean and null segment labels in a registered segmented sequential family (canonical true/false/null spellings matching the string-labelled oracle) — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** automatic roster, exact_bernoulli law

**Status:** refused

**Classification:** SOURCE

**Codes:** `sequential.route.unsupported`

**Reason:** publish/adopt re-derives the experiment's declared plan's own automatic registration -- segmented from the predeclared family, so the relational capture check above refuses it for the same reason.

**Condition:** automatic roster, asymptotic_mean law

**Status:** refused

**Classification:** SOURCE

**Codes:** `sequential.route.unsupported`

**Reason:** publish/adopt re-derives the experiment's declared plan's own automatic registration -- segmented from the predeclared family, so the relational capture check above refuses it for the same reason.

**Condition:** registered roster, exact_bernoulli law

**Status:** refused

**Classification:** SOURCE

**Codes:** `sequential.route.unsupported`

**Reason:** publish/adopt always re-derives an automatic (unsegmented) registration from the experiment's declared plan -- a caller-built segmented registration never reaches it -- and the relational capture check above would refuse it regardless.

**Condition:** registered roster, asymptotic_mean law

**Status:** refused

**Classification:** SOURCE

**Codes:** `sequential.route.unsupported`

**Reason:** publish/adopt always re-derives an automatic (unsegmented) registration from the experiment's declared plan -- a caller-built segmented registration never reaches it -- and the relational capture check above would refuse it regardless.

<span id="capability-ingresses-boolean-null-segment-labels-from_moments"></span>

#### Boolean and null segment labels in a registered segmented sequential family (canonical true/false/null spellings matching the string-labelled oracle) — Portable moments

**Recorded path:** `from_moments`

**Condition:** automatic roster, exact_bernoulli law

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.breakout_unavailable`

**Reason:** MomentsSource declares no breakout catalog at all -- a replayed checkpoint carries the segmented registration intact (export survives it), but run_breakout still refuses because a moments cube has no breakout dimension to serve.

**Condition:** automatic roster, asymptotic_mean law

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.breakout_unavailable`

**Reason:** MomentsSource declares no breakout catalog at all -- a replayed checkpoint carries the segmented registration intact (export survives it), but run_breakout still refuses because a moments cube has no breakout dimension to serve.

**Condition:** registered roster, exact_bernoulli law

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.breakout_unavailable`

**Reason:** MomentsSource declares no breakout catalog at all -- a replayed checkpoint carries the segmented registration intact (export survives it), but run_breakout still refuses because a moments cube has no breakout dimension to serve.

**Condition:** registered roster, asymptotic_mean law

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.breakout_unavailable`

**Reason:** MomentsSource declares no breakout catalog at all -- a replayed checkpoint carries the segmented registration intact (export survives it), but run_breakout still refuses because a moments cube has no breakout dimension to serve.

**Note:** replays the retained segment state; a cube carries no breakout catalog


<span id="capability-ingresses-triggered-sequential-registration"></span>

### Registered sequential monitoring of the triggered population (Experiment.trigger with InferenceSpec; assigned and triggered chains)

<span id="capability-ingresses-triggered-sequential-registration-from_unit_summary"></span>

#### Registered sequential monitoring of the triggered population (Experiment.trigger with InferenceSpec; assigned and triggered chains) — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** frame source or moments without a retained triggered checkpoint

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** This source carries no trigger declaration or trigger membership.

**Operation:** `run(population='triggered')`

**Stage:** run

<span id="capability-ingresses-triggered-sequential-registration-from_unit_panel"></span>

#### Registered sequential monitoring of the triggered population (Experiment.trigger with InferenceSpec; assigned and triggered chains) — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** frame source or moments without a retained triggered checkpoint

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** This source carries no trigger declaration or trigger membership.

**Operation:** `run(population='triggered')`

**Stage:** run

<span id="capability-ingresses-triggered-sequential-registration-from_switchback_panel"></span>

#### Registered sequential monitoring of the triggered population (Experiment.trigger with InferenceSpec; assigned and triggered chains) — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** frame source or moments without a retained triggered checkpoint

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** This source carries no trigger declaration or trigger membership.

**Operation:** `run(population='triggered')`

**Stage:** run

<span id="capability-ingresses-triggered-sequential-registration-from_definitions"></span>

#### Registered sequential monitoring of the triggered population (Experiment.trigger with InferenceSpec; assigned and triggered chains) — Definitions

**Recorded path:** `from_definitions`

**Condition:** registered plan and certified feeds

**Status:** supported

**Condition:** uncertified contributing feed

**Status:** refused

**Codes:** `sequential.source.invalid`

**Reason:** Every contributing feed must be certified complete through the capture day.

**Operation:** `capture_sequential(finalized=True, as_of=...)`

**Stage:** capture

**Condition:** SQL-declared trigger

**Status:** refused

**Codes:** `sequential.source.invalid`

**Reason:** A SQL-declared trigger has no feed that can certify its completeness.

**Operation:** `capture_sequential(finalized=True, as_of=...)`

**Stage:** capture

**Condition:** missing pinned source evidence

**Status:** refused

**Codes:** `source.native.trigger_evidence_required`

**Reason:** Trigger membership requires pinned source snapshot evidence.

**Operation:** `capture_sequential(finalized=True, as_of=...)`

**Stage:** capture

<span id="capability-ingresses-triggered-sequential-registration-from_unit_day_artifact"></span>

#### Registered sequential monitoring of the triggered population (Experiment.trigger with InferenceSpec; assigned and triggered chains) — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** trigger extensions and certified feeds

**Status:** supported

**Condition:** uncertified trigger feed

**Status:** refused

**Codes:** `sequential.source.invalid`

**Reason:** Stored trigger evidence must certify completeness through the capture day.

**Operation:** `capture_sequential(finalized=True, as_of=...)`

**Stage:** capture

**Note:** capture uses trigger_population, trigger_measure_stats and triggered assignment_counts; encouragement also needs encouragement_uptake

<span id="capability-ingresses-triggered-sequential-registration-from_moments"></span>

#### Registered sequential monitoring of the triggered population (Experiment.trigger with InferenceSpec; assigned and triggered chains) — Portable moments

**Recorded path:** `from_moments`

**Condition:** retained triggered checkpoint

**Status:** supported

**Condition:** frame source or moments without a retained triggered checkpoint

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** This source carries no trigger declaration or trigger membership.

**Operation:** `run(population='triggered')`

**Stage:** run

**Note:** replays retained assigned and triggered state; cannot start either process


<span id="capability-ingresses-triggered-sequential-uptake"></span>

### Triggered encouragement compliance under a registered plan

<span id="capability-ingresses-triggered-sequential-uptake-from_unit_summary"></span>

#### Triggered encouragement compliance under a registered plan — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** frame source or moments without a retained triggered checkpoint

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** This source carries no trigger declaration or trigger membership.

**Operation:** `run(population='triggered')`

**Stage:** run

<span id="capability-ingresses-triggered-sequential-uptake-from_unit_panel"></span>

#### Triggered encouragement compliance under a registered plan — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** frame source or moments without a retained triggered checkpoint

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** This source carries no trigger declaration or trigger membership.

**Operation:** `run(population='triggered')`

**Stage:** run

<span id="capability-ingresses-triggered-sequential-uptake-from_switchback_panel"></span>

#### Triggered encouragement compliance under a registered plan — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** frame source or moments without a retained triggered checkpoint

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** This source carries no trigger declaration or trigger membership.

**Operation:** `run(population='triggered')`

**Stage:** run

<span id="capability-ingresses-triggered-sequential-uptake-from_definitions"></span>

#### Triggered encouragement compliance under a registered plan — Definitions

**Recorded path:** `from_definitions`

**Condition:** registered encouragement plan with a triggered checkpoint

**Status:** unavailable

**Codes:** `readout.cell.unsupported_request`

**Reason:** triggered_uptake_unsupported

**Operation:** `run(estimands=('itt', 'compliance'))`

**Stage:** run

**Note:** assigned compliance remains monitored

<span id="capability-ingresses-triggered-sequential-uptake-from_unit_day_artifact"></span>

#### Triggered encouragement compliance under a registered plan — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** registered encouragement plan with a triggered checkpoint

**Status:** unavailable

**Codes:** `readout.cell.unsupported_request`

**Reason:** triggered_uptake_unsupported

**Operation:** `run(estimands=('itt', 'compliance'))`

**Stage:** run

**Note:** assigned compliance remains monitored

<span id="capability-ingresses-triggered-sequential-uptake-from_moments"></span>

#### Triggered encouragement compliance under a registered plan — Portable moments

**Recorded path:** `from_moments`

**Condition:** registered encouragement plan with a triggered checkpoint

**Status:** unavailable

**Codes:** `readout.cell.unsupported_request`

**Reason:** triggered_uptake_unsupported

**Operation:** `run(estimands=('itt', 'compliance'))`

**Stage:** run

**Note:** assigned compliance remains monitored


<span id="capability-ingresses-impute-pooled-mean-roles"></span>

### Pooled-mean imputation helper with declared roles; a frame metric declaring an impute missing policy

<span id="capability-ingresses-impute-pooled-mean-roles-from_unit_summary"></span>

#### Pooled-mean imputation helper with declared roles; a frame metric declaring an impute missing policy — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** frame prepared for from_unit_summary with declared roles; the outcome column is rejected before any requested column is filled

**Status:** refused

**Codes:** `impute.pooled_mean_outcome`

**Reason:** outcome columns must not be imputed as covariates

**Operation:** `increment.estimators.impute.pooled_mean`

**Stage:** construction

**Condition:** frame prepared for from_unit_summary without declared roles; filled columns emit one coded warning with per-column counts

**Status:** supported

**Codes:** `impute.pooled_mean_role_undeclared`

**Operation:** `increment.estimators.impute.pooled_mean`

**Stage:** construction

**Condition:** frame source refuses imputation policy

**Status:** refused

**Codes:** `frame.metric.missing_impute`

**Reason:** frame metric declarations do not support missing='impute'

**Operation:** `frame metric with missing='impute'`

**Stage:** construction

<span id="capability-ingresses-impute-pooled-mean-roles-from_unit_panel"></span>

#### Pooled-mean imputation helper with declared roles; a frame metric declaring an impute missing policy — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** frame prepared for from_unit_panel with declared roles; the outcome column is rejected before any requested column is filled

**Status:** refused

**Codes:** `impute.pooled_mean_outcome`

**Reason:** outcome columns must not be imputed as covariates

**Operation:** `increment.estimators.impute.pooled_mean`

**Stage:** construction

**Condition:** frame prepared for from_unit_panel without declared roles; filled columns emit one coded warning with per-column counts

**Status:** supported

**Codes:** `impute.pooled_mean_role_undeclared`

**Operation:** `increment.estimators.impute.pooled_mean`

**Stage:** construction

**Condition:** frame source refuses imputation policy

**Status:** refused

**Codes:** `frame.metric.missing_impute`

**Reason:** frame metric declarations do not support missing='impute'

**Operation:** `frame metric with missing='impute'`

**Stage:** construction

<span id="capability-ingresses-impute-pooled-mean-roles-from_switchback_panel"></span>

#### Pooled-mean imputation helper with declared roles; a frame metric declaring an impute missing policy — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** frame prepared for from_switchback_panel with declared roles; the outcome column is rejected before any requested column is filled

**Status:** refused

**Codes:** `impute.pooled_mean_outcome`

**Reason:** outcome columns must not be imputed as covariates

**Operation:** `increment.estimators.impute.pooled_mean`

**Stage:** construction

**Condition:** frame prepared for from_switchback_panel without declared roles; filled columns emit one coded warning with per-column counts

**Status:** supported

**Codes:** `impute.pooled_mean_role_undeclared`

**Operation:** `increment.estimators.impute.pooled_mean`

**Stage:** construction

**Condition:** frame source refuses imputation policy

**Status:** refused

**Codes:** `frame.metric.missing_impute`

**Reason:** frame metric declarations do not support missing='impute'

**Operation:** `frame metric with missing='impute'`

**Stage:** construction

<span id="capability-ingresses-impute-pooled-mean-roles-from_definitions"></span>

#### Pooled-mean imputation helper with declared roles; a frame metric declaring an impute missing policy — Definitions

**Recorded path:** `from_definitions`

**Condition:** a warehouse experiment declares metrics over event facts, not a prepared frame

**Status:** not comparable

**Reason:** a warehouse Metric cannot express missing='impute'; the helper prepares dataframes only

**Operation:** `increment.estimators.impute.pooled_mean`

**Stage:** declaration

**Condition:** impute missing policy

**Status:** not expressible

**Reason:** a warehouse Metric cannot express a missing-value policy: absent events are zero by construction of the event log

**Stage:** declaration

<span id="capability-ingresses-impute-pooled-mean-roles-from_unit_day_artifact"></span>

#### Pooled-mean imputation helper with declared roles; a frame metric declaring an impute missing policy — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** a warehouse experiment declares metrics over event facts, not a prepared frame

**Status:** not comparable

**Reason:** a warehouse Metric cannot express missing='impute'; the helper prepares dataframes only

**Operation:** `increment.estimators.impute.pooled_mean`

**Stage:** declaration

**Condition:** impute missing policy

**Status:** not expressible

**Reason:** a warehouse Metric cannot express a missing-value policy: absent events are zero by construction of the event log

**Stage:** declaration

<span id="capability-ingresses-impute-pooled-mean-roles-from_moments"></span>

#### Pooled-mean imputation helper with declared roles; a frame metric declaring an impute missing policy — Portable moments

**Recorded path:** `from_moments`

**Condition:** a portable cube holds arm moments only

**Status:** not comparable

**Reason:** a moments cube carries no per-unit rows for the helper to repair

**Operation:** `increment.estimators.impute.pooled_mean`

**Stage:** read

**Condition:** impute missing policy

**Status:** refused

**Classification:** CONSTRUCTION

**Codes:** `frame.metric.missing_impute`

**Reason:** filling an outcome with its pooled mean shrinks variance and biases the estimate

**Stage:** declaration


<span id="capability-ingresses-encouragement-design-readouts"></span>

### Encouragement ITT, compliance and LATE, fixed horizon, declared as the study design

<span id="capability-ingresses-encouragement-design-readouts-from_unit_summary"></span>

#### Encouragement ITT, compliance and LATE, fixed horizon, declared as the study design — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

<span id="capability-ingresses-encouragement-design-readouts-from_unit_panel"></span>

#### Encouragement ITT, compliance and LATE, fixed horizon, declared as the study design — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

<span id="capability-ingresses-encouragement-design-readouts-from_switchback_panel"></span>

#### Encouragement ITT, compliance and LATE, fixed horizon, declared as the study design — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** Encouragement design supplies identification to the switchback builder

**Status:** refused

**Codes:** `source.frame.switchback.identification`

**Reason:** Switchback identification requires a supported randomized design.

**Operation:** `from_switchback_frame(identification=Encouragement)`

**Stage:** construction

<span id="capability-ingresses-encouragement-design-readouts-from_definitions"></span>

#### Encouragement ITT, compliance and LATE, fixed horizon, declared as the study design — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

<span id="capability-ingresses-encouragement-design-readouts-from_unit_day_artifact"></span>

#### Encouragement ITT, compliance and LATE, fixed horizon, declared as the study design — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

<span id="capability-ingresses-encouragement-design-readouts-from_moments"></span>

#### Encouragement ITT, compliance and LATE, fixed horizon, declared as the study design — Portable moments

**Recorded path:** `from_moments`

**Status:** supported


<span id="capability-ingresses-encouragement-no-exclusion"></span>

### Explicit ITT/compliance without a LATE exclusion declaration (fixed horizon or sequential monitoring)

<span id="capability-ingresses-encouragement-no-exclusion-from_unit_summary"></span>

#### Explicit ITT/compliance without a LATE exclusion declaration (fixed horizon or sequential monitoring) — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** fixed horizon

**Status:** supported

**Condition:** sequential monitoring

**Status:** supported

<span id="capability-ingresses-encouragement-no-exclusion-from_unit_panel"></span>

#### Explicit ITT/compliance without a LATE exclusion declaration (fixed horizon or sequential monitoring) — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** fixed horizon

**Status:** supported

**Condition:** sequential monitoring

**Status:** supported

<span id="capability-ingresses-encouragement-no-exclusion-from_switchback_panel"></span>

#### Explicit ITT/compliance without a LATE exclusion declaration (fixed horizon or sequential monitoring) — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** Encouragement design supplies identification to the switchback builder

**Status:** refused

**Codes:** `source.frame.switchback.identification`

**Reason:** Switchback identification requires a supported randomized design.

**Operation:** `from_switchback_frame(identification=Encouragement)`

**Stage:** construction

<span id="capability-ingresses-encouragement-no-exclusion-from_definitions"></span>

#### Explicit ITT/compliance without a LATE exclusion declaration (fixed horizon or sequential monitoring) — Definitions

**Recorded path:** `from_definitions`

**Condition:** fixed horizon

**Status:** supported

**Condition:** sequential monitoring

**Status:** supported

<span id="capability-ingresses-encouragement-no-exclusion-from_unit_day_artifact"></span>

#### Explicit ITT/compliance without a LATE exclusion declaration (fixed horizon or sequential monitoring) — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** fixed horizon

**Status:** supported

**Condition:** sequential monitoring

**Status:** supported

<span id="capability-ingresses-encouragement-no-exclusion-from_moments"></span>

#### Explicit ITT/compliance without a LATE exclusion declaration (fixed horizon or sequential monitoring) — Portable moments

**Recorded path:** `from_moments`

**Condition:** fixed horizon

**Status:** supported

**Condition:** sequential monitoring

**Status:** supported

**Note:** sequential monitoring replays the exported checkpoint


<span id="capability-ingresses-encouragement-empty-outcome-catalog"></span>

### Fixed-horizon design-level compliance with an empty outcome catalog

<span id="capability-ingresses-encouragement-empty-outcome-catalog-from_unit_summary"></span>

#### Fixed-horizon design-level compliance with an empty outcome catalog — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

<span id="capability-ingresses-encouragement-empty-outcome-catalog-from_unit_panel"></span>

#### Fixed-horizon design-level compliance with an empty outcome catalog — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

<span id="capability-ingresses-encouragement-empty-outcome-catalog-from_switchback_panel"></span>

#### Fixed-horizon design-level compliance with an empty outcome catalog — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** Encouragement design supplies identification to the switchback builder

**Status:** refused

**Codes:** `source.frame.switchback.identification`

**Reason:** Switchback identification requires a supported randomized design.

**Operation:** `from_switchback_frame(identification=Encouragement)`

**Stage:** construction

<span id="capability-ingresses-encouragement-empty-outcome-catalog-from_definitions"></span>

#### Fixed-horizon design-level compliance with an empty outcome catalog — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

<span id="capability-ingresses-encouragement-empty-outcome-catalog-from_unit_day_artifact"></span>

#### Fixed-horizon design-level compliance with an empty outcome catalog — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

<span id="capability-ingresses-encouragement-empty-outcome-catalog-from_moments"></span>

#### Fixed-horizon design-level compliance with an empty outcome catalog — Portable moments

**Recorded path:** `from_moments`

**Status:** supported

**Note:** a design summary carrying the source identity and the empty catalog


<span id="capability-ingresses-encouragement-margin-guardrail"></span>

### Encouragement guardrail with a declared non-inferiority margin

<span id="capability-ingresses-encouragement-margin-guardrail-from_unit_summary"></span>

#### Encouragement guardrail with a declared non-inferiority margin — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** request includes itt

**Status:** supported

**Condition:** request excludes itt

**Status:** refused

**Classification:** CONSTRUCTION

**Codes:** `readout.encouragement.margin`

**Reason:** the request excludes the ITT guardrail result

**Note:** the ITT row carries the margin verdict

<span id="capability-ingresses-encouragement-margin-guardrail-from_unit_panel"></span>

#### Encouragement guardrail with a declared non-inferiority margin — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** request includes itt

**Status:** supported

**Condition:** request excludes itt

**Status:** refused

**Classification:** CONSTRUCTION

**Codes:** `readout.encouragement.margin`

**Reason:** the request excludes the ITT guardrail result

**Note:** the ITT row carries the margin verdict

<span id="capability-ingresses-encouragement-margin-guardrail-from_switchback_panel"></span>

#### Encouragement guardrail with a declared non-inferiority margin — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** Encouragement design supplies identification to the switchback builder

**Status:** refused

**Codes:** `source.frame.switchback.identification`

**Reason:** Switchback identification requires a supported randomized design.

**Operation:** `from_switchback_frame(identification=Encouragement)`

**Stage:** construction

<span id="capability-ingresses-encouragement-margin-guardrail-from_definitions"></span>

#### Encouragement guardrail with a declared non-inferiority margin — Definitions

**Recorded path:** `from_definitions`

**Condition:** request includes itt

**Status:** supported

**Condition:** request excludes itt

**Status:** refused

**Classification:** CONSTRUCTION

**Codes:** `readout.encouragement.margin`

**Reason:** the request excludes the ITT guardrail result

**Note:** the ITT row carries the margin verdict

<span id="capability-ingresses-encouragement-margin-guardrail-from_unit_day_artifact"></span>

#### Encouragement guardrail with a declared non-inferiority margin — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** request includes itt

**Status:** supported

**Condition:** request excludes itt

**Status:** refused

**Classification:** CONSTRUCTION

**Codes:** `readout.encouragement.margin`

**Reason:** the request excludes the ITT guardrail result

**Note:** the ITT row carries the margin verdict

<span id="capability-ingresses-encouragement-margin-guardrail-from_moments"></span>

#### Encouragement guardrail with a declared non-inferiority margin — Portable moments

**Recorded path:** `from_moments`

**Condition:** request includes itt

**Status:** supported

**Condition:** request excludes itt

**Status:** refused

**Classification:** CONSTRUCTION

**Codes:** `readout.encouragement.margin`

**Reason:** the request excludes the ITT guardrail result

**Note:** the ITT row carries the margin verdict


<span id="capability-ingresses-encouragement-late-fcr-reestimated"></span>

### Encouragement BH-selected secondary's LATE row re-estimated at the corrected level

<span id="capability-ingresses-encouragement-late-fcr-reestimated-from_unit_summary"></span>

#### Encouragement BH-selected secondary's LATE row re-estimated at the corrected level — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

<span id="capability-ingresses-encouragement-late-fcr-reestimated-from_unit_panel"></span>

#### Encouragement BH-selected secondary's LATE row re-estimated at the corrected level — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

<span id="capability-ingresses-encouragement-late-fcr-reestimated-from_switchback_panel"></span>

#### Encouragement BH-selected secondary's LATE row re-estimated at the corrected level — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** Encouragement design supplies identification to the switchback builder

**Status:** refused

**Codes:** `source.frame.switchback.identification`

**Reason:** Switchback identification requires a supported randomized design.

**Operation:** `from_switchback_frame(identification=Encouragement)`

**Stage:** construction

<span id="capability-ingresses-encouragement-late-fcr-reestimated-from_definitions"></span>

#### Encouragement BH-selected secondary's LATE row re-estimated at the corrected level — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

<span id="capability-ingresses-encouragement-late-fcr-reestimated-from_unit_day_artifact"></span>

#### Encouragement BH-selected secondary's LATE row re-estimated at the corrected level — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

<span id="capability-ingresses-encouragement-late-fcr-reestimated-from_moments"></span>

#### Encouragement BH-selected secondary's LATE row re-estimated at the corrected level — Portable moments

**Recorded path:** `from_moments`

**Status:** supported


<span id="capability-ingresses-encouragement-itt-conversion-route"></span>

### Encouragement ITT on a conversion metric takes the count-routed conversion route

<span id="capability-ingresses-encouragement-itt-conversion-route-from_unit_summary"></span>

#### Encouragement ITT on a conversion metric takes the count-routed conversion route — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** dense counts, delta-method route

**Status:** supported

**Condition:** sparse counts, finite-sample route

**Status:** supported

<span id="capability-ingresses-encouragement-itt-conversion-route-from_unit_panel"></span>

#### Encouragement ITT on a conversion metric takes the count-routed conversion route — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** dense counts, delta-method route

**Status:** supported

**Condition:** sparse counts, finite-sample route

**Status:** supported

<span id="capability-ingresses-encouragement-itt-conversion-route-from_switchback_panel"></span>

#### Encouragement ITT on a conversion metric takes the count-routed conversion route — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** Encouragement design supplies identification to the switchback builder

**Status:** refused

**Codes:** `source.frame.switchback.identification`

**Reason:** Switchback identification requires a supported randomized design.

**Operation:** `from_switchback_frame(identification=Encouragement)`

**Stage:** construction

<span id="capability-ingresses-encouragement-itt-conversion-route-from_definitions"></span>

#### Encouragement ITT on a conversion metric takes the count-routed conversion route — Definitions

**Recorded path:** `from_definitions`

**Condition:** dense counts, delta-method route

**Status:** supported

**Condition:** sparse counts, finite-sample route

**Status:** supported

<span id="capability-ingresses-encouragement-itt-conversion-route-from_unit_day_artifact"></span>

#### Encouragement ITT on a conversion metric takes the count-routed conversion route — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** dense counts, delta-method route

**Status:** supported

**Condition:** sparse counts, finite-sample route

**Status:** supported

<span id="capability-ingresses-encouragement-itt-conversion-route-from_moments"></span>

#### Encouragement ITT on a conversion metric takes the count-routed conversion route — Portable moments

**Recorded path:** `from_moments`

**Condition:** dense counts, delta-method route

**Status:** supported

**Condition:** sparse counts, finite-sample route

**Status:** supported


<span id="capability-ingresses-conversion-route-threshold"></span>

### Conversion/retention counts one below, at and one above the dense threshold, on the success and failure side, route identically and report the same interval and p-value

<span id="capability-ingresses-conversion-route-threshold-from_unit_summary"></span>

#### Conversion/retention counts one below, at and one above the dense threshold, on the success and failure side, route identically and report the same interval and p-value — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

<span id="capability-ingresses-conversion-route-threshold-from_unit_panel"></span>

#### Conversion/retention counts one below, at and one above the dense threshold, on the success and failure side, route identically and report the same interval and p-value — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

<span id="capability-ingresses-conversion-route-threshold-from_switchback_panel"></span>

#### Conversion/retention counts one below, at and one above the dense threshold, on the success and failure side, route identically and report the same interval and p-value — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** not comparable

**Classification:** SOURCE

**Reason:** from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.

<span id="capability-ingresses-conversion-route-threshold-from_definitions"></span>

#### Conversion/retention counts one below, at and one above the dense threshold, on the success and failure side, route identically and report the same interval and p-value — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

<span id="capability-ingresses-conversion-route-threshold-from_unit_day_artifact"></span>

#### Conversion/retention counts one below, at and one above the dense threshold, on the success and failure side, route identically and report the same interval and p-value — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

<span id="capability-ingresses-conversion-route-threshold-from_moments"></span>

#### Conversion/retention counts one below, at and one above the dense threshold, on the success and failure side, route identically and report the same interval and p-value — Portable moments

**Recorded path:** `from_moments`

**Status:** supported


<span id="capability-ingresses-conversion-breakout-route"></span>

### Conversion breakout with dense, sparse and threshold-straddling segments in one metric: each segment row is routed by its own counts

<span id="capability-ingresses-conversion-breakout-route-from_unit_summary"></span>

#### Conversion breakout with dense, sparse and threshold-straddling segments in one metric: each segment row is routed by its own counts — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

<span id="capability-ingresses-conversion-breakout-route-from_unit_panel"></span>

#### Conversion breakout with dense, sparse and threshold-straddling segments in one metric: each segment row is routed by its own counts — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

<span id="capability-ingresses-conversion-breakout-route-from_switchback_panel"></span>

#### Conversion breakout with dense, sparse and threshold-straddling segments in one metric: each segment row is routed by its own counts — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** not comparable

**Classification:** SOURCE

**Reason:** from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.

<span id="capability-ingresses-conversion-breakout-route-from_definitions"></span>

#### Conversion breakout with dense, sparse and threshold-straddling segments in one metric: each segment row is routed by its own counts — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

<span id="capability-ingresses-conversion-breakout-route-from_unit_day_artifact"></span>

#### Conversion breakout with dense, sparse and threshold-straddling segments in one metric: each segment row is routed by its own counts — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

<span id="capability-ingresses-conversion-breakout-route-from_moments"></span>

#### Conversion breakout with dense, sparse and threshold-straddling segments in one metric: each segment row is routed by its own counts — Portable moments

**Recorded path:** `from_moments`

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.breakout_unavailable`

**Reason:** the source carries no breakout catalog: run_breakout needs a native or panel source

**Stage:** request


<span id="capability-ingresses-conversion-route-encouragement-family"></span>

### Encouragement ITT rows of a BH secondary family are routed at their own level and carry composite p-value evidence

<span id="capability-ingresses-conversion-route-encouragement-family-from_unit_summary"></span>

#### Encouragement ITT rows of a BH secondary family are routed at their own level and carry composite p-value evidence — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

<span id="capability-ingresses-conversion-route-encouragement-family-from_unit_panel"></span>

#### Encouragement ITT rows of a BH secondary family are routed at their own level and carry composite p-value evidence — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

<span id="capability-ingresses-conversion-route-encouragement-family-from_switchback_panel"></span>

#### Encouragement ITT rows of a BH secondary family are routed at their own level and carry composite p-value evidence — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** Encouragement design supplies identification to the switchback builder

**Status:** refused

**Codes:** `source.frame.switchback.identification`

**Reason:** Switchback identification requires a supported randomized design.

**Operation:** `from_switchback_frame(identification=Encouragement)`

**Stage:** construction

<span id="capability-ingresses-conversion-route-encouragement-family-from_definitions"></span>

#### Encouragement ITT rows of a BH secondary family are routed at their own level and carry composite p-value evidence — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

<span id="capability-ingresses-conversion-route-encouragement-family-from_unit_day_artifact"></span>

#### Encouragement ITT rows of a BH secondary family are routed at their own level and carry composite p-value evidence — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

<span id="capability-ingresses-conversion-route-encouragement-family-from_moments"></span>

#### Encouragement ITT rows of a BH secondary family are routed at their own level and carry composite p-value evidence — Portable moments

**Recorded path:** `from_moments`

**Status:** supported


<span id="capability-ingresses-conversion-route-observational-family"></span>

### Observational unadjusted rows of a BH secondary family are routed at their own level and carry composite p-value evidence

<span id="capability-ingresses-conversion-route-observational-family-from_unit_summary"></span>

#### Observational unadjusted rows of a BH secondary family are routed at their own level and carry composite p-value evidence — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

<span id="capability-ingresses-conversion-route-observational-family-from_unit_panel"></span>

#### Observational unadjusted rows of a BH secondary family are routed at their own level and carry composite p-value evidence — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

<span id="capability-ingresses-conversion-route-observational-family-from_switchback_panel"></span>

#### Observational unadjusted rows of a BH secondary family are routed at their own level and carry composite p-value evidence — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.identification`

**Reason:** a switchback frame identifies a Randomized design only: restricted, observational and encouragement designs are unsupported

**Stage:** declaration

<span id="capability-ingresses-conversion-route-observational-family-from_definitions"></span>

#### Observational unadjusted rows of a BH secondary family are routed at their own level and carry composite p-value evidence — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

<span id="capability-ingresses-conversion-route-observational-family-from_unit_day_artifact"></span>

#### Observational unadjusted rows of a BH secondary family are routed at their own level and carry composite p-value evidence — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

<span id="capability-ingresses-conversion-route-observational-family-from_moments"></span>

#### Observational unadjusted rows of a BH secondary family are routed at their own level and carry composite p-value evidence — Portable moments

**Recorded path:** `from_moments`

**Status:** supported


<span id="capability-ingresses-encouragement-retention-readouts"></span>

### Encouragement design with a retention metric (ITT and LATE readouts)

<span id="capability-ingresses-encouragement-retention-readouts-from_unit_summary"></span>

#### Encouragement design with a retention metric (ITT and LATE readouts) — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** retention threshold_days needs per-unit dates the summary does not carry

**Status:** refused

**Codes:** `source.frame.constructor`

**Reason:** One-row-per-unit summaries carry no dates for retention thresholds.

**Operation:** `from_frame(shape="summary") with retention outcome`

**Stage:** construction

<span id="capability-ingresses-encouragement-retention-readouts-from_unit_panel"></span>

#### Encouragement design with a retention metric (ITT and LATE readouts) — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** Encouragement design declares a retention outcome

**Status:** refused

**Codes:** `readout.encouragement.retention`

**Reason:** The panel refuses the retention outcome before any readout.

**Operation:** `from_frame(shape="panel") with retention outcome`

**Stage:** construction

<span id="capability-ingresses-encouragement-retention-readouts-from_switchback_panel"></span>

#### Encouragement design with a retention metric (ITT and LATE readouts) — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** Encouragement design supplies identification to the switchback builder

**Status:** refused

**Codes:** `source.frame.switchback.identification`

**Reason:** Switchback identification requires a supported randomized design.

**Operation:** `from_switchback_frame(identification=Encouragement)`

**Stage:** construction

<span id="capability-ingresses-encouragement-retention-readouts-from_definitions"></span>

#### Encouragement design with a retention metric (ITT and LATE readouts) — Definitions

**Recorded path:** `from_definitions`

**Condition:** Encouragement design retains a retention outcome in the requested estimands

**Status:** refused

**Codes:** `readout.encouragement.retention`

**Reason:** Retention is not estimable under an Encouragement design.

**Operation:** `run() ITT/LATE with retention outcome`

**Stage:** read

<span id="capability-ingresses-encouragement-retention-readouts-from_unit_day_artifact"></span>

#### Encouragement design with a retention metric (ITT and LATE readouts) — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** Encouragement design retains a retention outcome in the requested estimands

**Status:** refused

**Codes:** `readout.encouragement.retention`

**Reason:** Retention is not estimable under an Encouragement design.

**Operation:** `run() ITT/LATE with retention outcome`

**Stage:** read

<span id="capability-ingresses-encouragement-retention-readouts-from_moments"></span>

#### Encouragement design with a retention metric (ITT and LATE readouts) — Portable moments

**Recorded path:** `from_moments`

**Condition:** Encouragement design retains a retention outcome in the requested estimands

**Status:** refused

**Codes:** `readout.encouragement.retention`

**Reason:** Retention is not estimable under an Encouragement design.

**Operation:** `run() ITT/LATE with retention outcome`

**Stage:** read


<span id="capability-ingresses-encouragement-retention-compliance-only"></span>

### Encouragement design with a retention metric, compliance-only request (which ignores the retention outcome)

<span id="capability-ingresses-encouragement-retention-compliance-only-from_unit_summary"></span>

#### Encouragement design with a retention metric, compliance-only request (which ignores the retention outcome) — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** retention threshold_days needs per-unit dates the summary does not carry

**Status:** refused

**Codes:** `source.frame.constructor`

**Reason:** One-row-per-unit summaries carry no dates for retention thresholds.

**Operation:** `from_frame(shape="summary") with retention outcome`

**Stage:** construction

**Note:** the retention declaration itself is the limit, before any request

<span id="capability-ingresses-encouragement-retention-compliance-only-from_unit_panel"></span>

#### Encouragement design with a retention metric, compliance-only request (which ignores the retention outcome) — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** Encouragement design declares a retention outcome

**Status:** refused

**Codes:** `readout.encouragement.retention`

**Reason:** The panel refuses the retention outcome before any readout.

**Operation:** `from_frame(shape="panel") with retention outcome`

**Stage:** construction

**Note:** the retention declaration itself is the limit, before any request

<span id="capability-ingresses-encouragement-retention-compliance-only-from_switchback_panel"></span>

#### Encouragement design with a retention metric, compliance-only request (which ignores the retention outcome) — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** Encouragement design supplies identification to the switchback builder

**Status:** refused

**Codes:** `source.frame.switchback.identification`

**Reason:** Switchback identification requires a supported randomized design.

**Operation:** `from_switchback_frame(identification=Encouragement)`

**Stage:** construction

<span id="capability-ingresses-encouragement-retention-compliance-only-from_definitions"></span>

#### Encouragement design with a retention metric, compliance-only request (which ignores the retention outcome) — Definitions

**Recorded path:** `from_definitions`

**Condition:** compliance-only request ignores the retention outcome configuration

**Status:** supported

**Operation:** `run(estimands=('compliance',))`

**Stage:** read

**Condition:** same experiment requests the retention ITT outcome

**Status:** refused

**Codes:** `readout.encouragement.retention`

**Reason:** Retention remains refused for the ITT outcome.

**Operation:** `run(estimands=('itt',))`

**Stage:** read

<span id="capability-ingresses-encouragement-retention-compliance-only-from_unit_day_artifact"></span>

#### Encouragement design with a retention metric, compliance-only request (which ignores the retention outcome) — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** not measured

**Reason:** no consumer exercises a compliance-only request on an adopted artifact

<span id="capability-ingresses-encouragement-retention-compliance-only-from_moments"></span>

#### Encouragement design with a retention metric, compliance-only request (which ignores the retention outcome) — Portable moments

**Recorded path:** `from_moments`

**Status:** not measured

**Reason:** no consumer exercises a compliance-only request on a replayed cube


<span id="capability-ingresses-encouragement-conversion-breakout-zero-control"></span>

### Encouragement conversion ITT breakouts retain exact confidence sets at zero control counts

<span id="capability-ingresses-encouragement-conversion-breakout-zero-control-from_unit_summary"></span>

#### Encouragement conversion ITT breakouts retain exact confidence sets at zero control counts — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

<span id="capability-ingresses-encouragement-conversion-breakout-zero-control-from_unit_panel"></span>

#### Encouragement conversion ITT breakouts retain exact confidence sets at zero control counts — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

<span id="capability-ingresses-encouragement-conversion-breakout-zero-control-from_switchback_panel"></span>

#### Encouragement conversion ITT breakouts retain exact confidence sets at zero control counts — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** not comparable

**Classification:** SOURCE

**Reason:** from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.

<span id="capability-ingresses-encouragement-conversion-breakout-zero-control-from_definitions"></span>

#### Encouragement conversion ITT breakouts retain exact confidence sets at zero control counts — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

<span id="capability-ingresses-encouragement-conversion-breakout-zero-control-from_unit_day_artifact"></span>

#### Encouragement conversion ITT breakouts retain exact confidence sets at zero control counts — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

<span id="capability-ingresses-encouragement-conversion-breakout-zero-control-from_moments"></span>

#### Encouragement conversion ITT breakouts retain exact confidence sets at zero control counts — Portable moments

**Recorded path:** `from_moments`

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.breakout_unavailable`

**Reason:** portable moments retain no per-unit breakout dimension


<span id="capability-ingresses-retention-breakout-cohorts"></span>

### Bounded-retention breakouts with mature exposure cohorts

<span id="capability-ingresses-retention-breakout-cohorts-from_unit_summary"></span>

#### Bounded-retention breakouts with mature exposure cohorts — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.constructor`

**Reason:** unit summaries cannot supply retention exposure cohorts.

<span id="capability-ingresses-retention-breakout-cohorts-from_unit_panel"></span>

#### Bounded-retention breakouts with mature exposure cohorts — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

**Note:** breakout rows

<span id="capability-ingresses-retention-breakout-cohorts-from_switchback_panel"></span>

#### Bounded-retention breakouts with mature exposure cohorts — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** not comparable

**Classification:** SOURCE

**Reason:** from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.

<span id="capability-ingresses-retention-breakout-cohorts-from_definitions"></span>

#### Bounded-retention breakouts with mature exposure cohorts — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

**Note:** breakout rows and cohort-aware breakout summaries

<span id="capability-ingresses-retention-breakout-cohorts-from_unit_day_artifact"></span>

#### Bounded-retention breakouts with mature exposure cohorts — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

**Note:** breakout rows and cohort-aware breakout summaries

<span id="capability-ingresses-retention-breakout-cohorts-from_moments"></span>

#### Bounded-retention breakouts with mature exposure cohorts — Portable moments

**Recorded path:** `from_moments`

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.breakout_unavailable`

**Reason:** portable moments retain no breakout operation or cohort identity.


<span id="capability-ingresses-encouragement-joint-sequential"></span>

### Encouragement continuous ITT + Bernoulli uptake compliance, jointly monitored (composed sequential family, e-BH selected)

<span id="capability-ingresses-encouragement-joint-sequential-from_unit_summary"></span>

#### Encouragement continuous ITT + Bernoulli uptake compliance, jointly monitored (composed sequential family, e-BH selected) — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** ITT and uptake only

**Status:** supported

**Condition:** with in-family secondaries

**Status:** supported

<span id="capability-ingresses-encouragement-joint-sequential-from_unit_panel"></span>

#### Encouragement continuous ITT + Bernoulli uptake compliance, jointly monitored (composed sequential family, e-BH selected) — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** ITT and uptake only

**Status:** supported

**Condition:** with in-family secondaries

**Status:** supported

<span id="capability-ingresses-encouragement-joint-sequential-from_switchback_panel"></span>

#### Encouragement continuous ITT + Bernoulli uptake compliance, jointly monitored (composed sequential family, e-BH selected) — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** declared design

**Status:** refused

**Codes:** `source.frame.switchback.identification`

**Reason:** Switchback identification requires a supported randomized design.

**Operation:** `from_switchback_frame(identification=Encouragement)`

**Stage:** construction

**Condition:** sequential plan

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.plan`

**Reason:** switchback contrasts support fixed inference only

**Stage:** declaration

<span id="capability-ingresses-encouragement-joint-sequential-from_definitions"></span>

#### Encouragement continuous ITT + Bernoulli uptake compliance, jointly monitored (composed sequential family, e-BH selected) — Definitions

**Recorded path:** `from_definitions`

**Condition:** ITT and uptake only

**Status:** supported

**Condition:** with in-family secondaries

**Status:** supported

<span id="capability-ingresses-encouragement-joint-sequential-from_unit_day_artifact"></span>

#### Encouragement continuous ITT + Bernoulli uptake compliance, jointly monitored (composed sequential family, e-BH selected) — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** ITT and uptake only

**Status:** supported

**Condition:** with in-family secondaries

**Status:** supported

<span id="capability-ingresses-encouragement-joint-sequential-from_moments"></span>

#### Encouragement continuous ITT + Bernoulli uptake compliance, jointly monitored (composed sequential family, e-BH selected) — Portable moments

**Recorded path:** `from_moments`

**Condition:** ITT and uptake only

**Status:** supported

**Condition:** with in-family secondaries

**Status:** supported

**Note:** replays the exported checkpoint


<span id="capability-ingresses-uptake-only-sequential-empty-catalog"></span>

### Uptake-only sequential compliance with an empty outcome catalog

<span id="capability-ingresses-uptake-only-sequential-empty-catalog-from_unit_summary"></span>

#### Uptake-only sequential compliance with an empty outcome catalog — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

<span id="capability-ingresses-uptake-only-sequential-empty-catalog-from_unit_panel"></span>

#### Uptake-only sequential compliance with an empty outcome catalog — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

<span id="capability-ingresses-uptake-only-sequential-empty-catalog-from_switchback_panel"></span>

#### Uptake-only sequential compliance with an empty outcome catalog — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** declared design

**Status:** refused

**Codes:** `source.frame.switchback.identification`

**Reason:** Switchback identification requires a supported randomized design.

**Operation:** `from_switchback_frame(identification=Encouragement)`

**Stage:** construction

**Condition:** sequential plan

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.plan`

**Reason:** switchback contrasts support fixed inference only

**Stage:** declaration

<span id="capability-ingresses-uptake-only-sequential-empty-catalog-from_definitions"></span>

#### Uptake-only sequential compliance with an empty outcome catalog — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

<span id="capability-ingresses-uptake-only-sequential-empty-catalog-from_unit_day_artifact"></span>

#### Uptake-only sequential compliance with an empty outcome catalog — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

<span id="capability-ingresses-uptake-only-sequential-empty-catalog-from_moments"></span>

#### Uptake-only sequential compliance with an empty outcome catalog — Portable moments

**Recorded path:** `from_moments`

**Status:** supported

**Note:** replays the exported checkpoint with the empty catalog


<span id="capability-ingresses-uptake-only-sequential-unbounded-retention-catalog"></span>

### Uptake-only sequential compliance that retains an unbounded retention metric in the outcome catalog

<span id="capability-ingresses-uptake-only-sequential-unbounded-retention-catalog-from_unit_summary"></span>

#### Uptake-only sequential compliance that retains an unbounded retention metric in the outcome catalog — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.constructor`

**Reason:** a one-row-per-unit summary has no per-unit dates to resolve a retention band against, so the catalog cannot be declared.

<span id="capability-ingresses-uptake-only-sequential-unbounded-retention-catalog-from_unit_panel"></span>

#### Uptake-only sequential compliance that retains an unbounded retention metric in the outcome catalog — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** refused

**Classification:** COMBINATION

**Codes:** `readout.encouragement.retention`

**Reason:** a unit panel refuses a retention metric under an Encouragement design at construction, before any request is read.

<span id="capability-ingresses-uptake-only-sequential-unbounded-retention-catalog-from_switchback_panel"></span>

#### Uptake-only sequential compliance that retains an unbounded retention metric in the outcome catalog — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** declared design

**Status:** refused

**Codes:** `source.frame.switchback.identification`

**Reason:** Switchback identification requires a supported randomized design.

**Operation:** `from_switchback_frame(identification=Encouragement)`

**Stage:** construction

**Condition:** sequential plan

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.plan`

**Reason:** switchback contrasts support fixed inference only

**Stage:** declaration

<span id="capability-ingresses-uptake-only-sequential-unbounded-retention-catalog-from_definitions"></span>

#### Uptake-only sequential compliance that retains an unbounded retention metric in the outcome catalog — Definitions

**Recorded path:** `from_definitions`

**Condition:** run(estimands=('compliance',)) with the retention catalog retained

**Status:** supported

**Condition:** uptake-only registration retains an unbounded retention outcome; compliance-only as-of request

**Status:** supported

**Operation:** `run_asof_lift(estimands=('compliance',), completed_windows_only=True)`

**Stage:** read

**Condition:** same retained retention catalog; ITT as-of request with either completion policy

**Status:** refused

**Codes:** `breakout.retention.encouragement`

**Reason:** Retention outcomes remain refused under encouragement even when compliance-only monitoring succeeds.

**Operation:** `run_asof_lift(estimands=('itt',), completed_windows_only=False/True)`

**Stage:** read

**Condition:** same uptake-only registration and retained retention catalog; ITT outcome request

**Status:** refused

**Codes:** `sequential.source.invalid`

**Reason:** The uptake-only checkpoint supplies no retained sequential state for the retention ITT outcome.

**Operation:** `run(estimands=('itt',))`

**Stage:** read

<span id="capability-ingresses-uptake-only-sequential-unbounded-retention-catalog-from_unit_day_artifact"></span>

#### Uptake-only sequential compliance that retains an unbounded retention metric in the outcome catalog — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** run(estimands=('compliance',)) with the retention catalog retained

**Status:** supported

**Condition:** uptake-only registration retains an unbounded retention outcome; compliance-only as-of request

**Status:** supported

**Operation:** `run_asof_lift(estimands=('compliance',), completed_windows_only=True)`

**Stage:** read

**Condition:** same retained retention catalog; ITT as-of request with either completion policy

**Status:** refused

**Codes:** `breakout.retention.encouragement`

**Reason:** Retention outcomes remain refused under encouragement even when compliance-only monitoring succeeds.

**Operation:** `run_asof_lift(estimands=('itt',), completed_windows_only=False/True)`

**Stage:** read

**Condition:** same uptake-only registration and retained retention catalog; ITT outcome request

**Status:** refused

**Codes:** `sequential.source.invalid`

**Reason:** The uptake-only checkpoint supplies no retained sequential state for the retention ITT outcome.

**Operation:** `run(estimands=('itt',))`

**Stage:** read

**Note:** the retention metric declared on the experiment and its catalog are retained

<span id="capability-ingresses-uptake-only-sequential-unbounded-retention-catalog-from_moments"></span>

#### Uptake-only sequential compliance that retains an unbounded retention metric in the outcome catalog — Portable moments

**Recorded path:** `from_moments`

**Condition:** run(estimands=('compliance',)) with the retention catalog retained

**Status:** supported

**Condition:** uptake-only registration retains an unbounded retention outcome; compliance-only as-of request

**Status:** supported

**Operation:** `run_asof_lift(estimands=('compliance',), completed_windows_only=True)`

**Stage:** read

**Condition:** same retained retention catalog; ITT as-of request with either completion policy

**Status:** refused

**Codes:** `breakout.retention.encouragement`

**Reason:** Retention outcomes remain refused under encouragement even when compliance-only monitoring succeeds.

**Operation:** `run_asof_lift(estimands=('itt',), completed_windows_only=False/True)`

**Stage:** read

**Condition:** same uptake-only registration and retained retention catalog; ITT outcome request

**Status:** refused

**Codes:** `sequential.source.invalid`

**Reason:** The uptake-only checkpoint supplies no retained sequential state for the retention ITT outcome.

**Operation:** `run(estimands=('itt',))`

**Stage:** read

**Note:** replays the exported checkpoint with the retention catalog retained


<span id="capability-ingresses-observational-weight-diagnostics"></span>

### Observational weight diagnostics (IPTW/AIPW; unit-level mean and conversion parity)

<span id="capability-ingresses-observational-weight-diagnostics-from_unit_summary"></span>

#### Observational weight diagnostics (IPTW/AIPW; unit-level mean and conversion parity) — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** IPTW, mean

**Status:** supported

**Condition:** IPTW, conversion

**Status:** supported

**Condition:** AIPW, mean

**Status:** supported

**Condition:** AIPW, conversion

**Status:** supported

**Condition:** IPTW mean, declared cluster grain

**Status:** supported

**Operation:** `run()`

**Stage:** read

<span id="capability-ingresses-observational-weight-diagnostics-from_unit_panel"></span>

#### Observational weight diagnostics (IPTW/AIPW; unit-level mean and conversion parity) — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** IPTW, mean

**Status:** supported

**Condition:** IPTW, conversion

**Status:** supported

**Condition:** AIPW, mean

**Status:** supported

**Condition:** AIPW, conversion

**Status:** supported

<span id="capability-ingresses-observational-weight-diagnostics-from_switchback_panel"></span>

#### Observational weight diagnostics (IPTW/AIPW; unit-level mean and conversion parity) — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** mean

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.identification`

**Reason:** a switchback frame identifies a Randomized design only: restricted, observational and encouragement designs are unsupported

**Stage:** declaration

**Condition:** conversion

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.identification`

**Reason:** a switchback frame identifies a Randomized design only: restricted, observational and encouragement designs are unsupported

**Stage:** declaration

<span id="capability-ingresses-observational-weight-diagnostics-from_definitions"></span>

#### Observational weight diagnostics (IPTW/AIPW; unit-level mean and conversion parity) — Definitions

**Recorded path:** `from_definitions`

**Condition:** IPTW, mean

**Status:** supported

**Condition:** IPTW, conversion

**Status:** supported

**Condition:** AIPW, mean

**Status:** supported

**Condition:** AIPW, conversion

**Status:** supported

**Condition:** IPTW mean, declared cluster grain

**Status:** supported

**Operation:** `run()`

**Stage:** read

<span id="capability-ingresses-observational-weight-diagnostics-from_unit_day_artifact"></span>

#### Observational weight diagnostics (IPTW/AIPW; unit-level mean and conversion parity) — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** IPTW, mean

**Status:** supported

**Condition:** IPTW, conversion

**Status:** supported

**Condition:** AIPW, mean

**Status:** supported

**Condition:** AIPW, conversion

**Status:** supported

<span id="capability-ingresses-observational-weight-diagnostics-from_moments"></span>

#### Observational weight diagnostics (IPTW/AIPW; unit-level mean and conversion parity) — Portable moments

**Recorded path:** `from_moments`

**Condition:** IPTW, mean

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.covariate_unavailable`

**Reason:** a moments cube holds no per-unit rows to attach a covariate to.

**Condition:** IPTW, conversion

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.covariate_unavailable`

**Reason:** a moments cube holds no per-unit rows to attach a covariate to.

**Condition:** AIPW, mean

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.covariate_unavailable`

**Reason:** a moments cube holds no per-unit rows to attach a covariate to.

**Condition:** AIPW, conversion

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.covariate_unavailable`

**Reason:** a moments cube holds no per-unit rows to attach a covariate to.


<span id="capability-ingresses-assignment-integrity-scope"></span>

### Assignment-integrity scope (status, observed counts, randomization grain and retained metadata; triggered populations are not assignment-law evidence)

<span id="capability-ingresses-assignment-integrity-scope-from_unit_summary"></span>

#### Assignment-integrity scope (status, observed counts, randomization grain and retained metadata; triggered populations are not assignment-law evidence) — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** cluster-grain scope

**Status:** supported

**Condition:** summary with cluster identities counts assigned clusters rather than member units

**Status:** supported

**Operation:** `srm()`

**Stage:** read

**Note:** unassigned totals are separate

<span id="capability-ingresses-assignment-integrity-scope-from_unit_panel"></span>

#### Assignment-integrity scope (status, observed counts, randomization grain and retained metadata; triggered populations are not assignment-law evidence) — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** multiple daily rows per unit must still count each assigned unit once

**Status:** supported

**Operation:** `srm()`

**Stage:** read

**Condition:** cluster grain

**Status:** not comparable

**Classification:** SOURCE

**Reason:** the panel constructor has no cluster key, so it cannot preserve the declared randomization grain.

<span id="capability-ingresses-assignment-integrity-scope-from_switchback_panel"></span>

#### Assignment-integrity scope (status, observed counts, randomization grain and retained metadata; triggered populations are not assignment-law evidence) — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** not comparable

**Classification:** SOURCE

**Reason:** a switchback contrast does not expose parallel-arm assignment counts.

<span id="capability-ingresses-assignment-integrity-scope-from_definitions"></span>

#### Assignment-integrity scope (status, observed counts, randomization grain and retained metadata; triggered populations are not assignment-law evidence) — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

**Note:** unit or cluster-grain counts with mixed/unassigned audit totals

<span id="capability-ingresses-assignment-integrity-scope-from_unit_day_artifact"></span>

#### Assignment-integrity scope (status, observed counts, randomization grain and retained metadata; triggered populations are not assignment-law evidence) — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

**Note:** cluster grain requires cluster identity; mixed/unassigned totals require assignment-count evidence

<span id="capability-ingresses-assignment-integrity-scope-from_moments"></span>

#### Assignment-integrity scope (status, observed counts, randomization grain and retained metadata; triggered populations are not assignment-law evidence) — Portable moments

**Recorded path:** `from_moments`

**Condition:** replayed unit-grain moments preserve assignment counts, not day-row counts

**Status:** supported

**Operation:** `srm()`

**Stage:** read

**Condition:** cluster grain

**Status:** not comparable

**Classification:** SOURCE

**Reason:** this clustered case has no portable cluster-identity payload.


<span id="capability-ingresses-observational-confounding-sensitivity"></span>

### Observational confounding sensitivity (conversion RR)

<span id="capability-ingresses-observational-confounding-sensitivity-from_unit_summary"></span>

#### Observational confounding sensitivity (conversion RR) — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** IPTW/AIPW fixed-horizon conversion RR, binary outcome from the row's declared metric type

**Status:** supported

**Operation:** `confounding_sensitivity`

**Stage:** read

<span id="capability-ingresses-observational-confounding-sensitivity-from_unit_panel"></span>

#### Observational confounding sensitivity (conversion RR) — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** IPTW/AIPW fixed-horizon conversion RR, binary outcome from the row's declared metric type

**Status:** supported

**Operation:** `confounding_sensitivity`

**Stage:** read

<span id="capability-ingresses-observational-confounding-sensitivity-from_switchback_panel"></span>

#### Observational confounding sensitivity (conversion RR) — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** switchback identification accepts a randomized design only

**Status:** refused

**Codes:** `source.frame.switchback.identification`

**Reason:** observational identification is not supported by switchback

**Operation:** `observational design construction`

**Stage:** construction

<span id="capability-ingresses-observational-confounding-sensitivity-from_definitions"></span>

#### Observational confounding sensitivity (conversion RR) — Definitions

**Recorded path:** `from_definitions`

**Condition:** IPTW/AIPW fixed-horizon conversion RR, binary outcome from the row's declared metric type

**Status:** supported

**Operation:** `confounding_sensitivity`

**Stage:** read

<span id="capability-ingresses-observational-confounding-sensitivity-from_unit_day_artifact"></span>

#### Observational confounding sensitivity (conversion RR) — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** IPTW/AIPW fixed-horizon conversion RR, binary outcome from the row's declared metric type

**Status:** supported

**Operation:** `confounding_sensitivity`

**Stage:** read

<span id="capability-ingresses-observational-confounding-sensitivity-from_moments"></span>

#### Observational confounding sensitivity (conversion RR) — Portable moments

**Recorded path:** `from_moments`

**Condition:** portable moments cannot supply adjustment covariates

**Status:** refused

**Codes:** `evidence.covariate_unavailable`

**Reason:** a moments cube retains no per-unit adjustment covariates

**Operation:** `observational adjusted conversion readout`

**Stage:** read


<span id="capability-ingresses-observational-iptw-mean"></span>

### Observational IPTW covariate adjustment

<span id="capability-ingresses-observational-iptw-mean-from_unit_summary"></span>

#### Observational IPTW covariate adjustment — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

<span id="capability-ingresses-observational-iptw-mean-from_unit_panel"></span>

#### Observational IPTW covariate adjustment — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

<span id="capability-ingresses-observational-iptw-mean-from_switchback_panel"></span>

#### Observational IPTW covariate adjustment — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.identification`

**Reason:** a switchback frame identifies a Randomized design only: restricted, observational and encouragement designs are unsupported

**Stage:** declaration

<span id="capability-ingresses-observational-iptw-mean-from_definitions"></span>

#### Observational IPTW covariate adjustment — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

<span id="capability-ingresses-observational-iptw-mean-from_unit_day_artifact"></span>

#### Observational IPTW covariate adjustment — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

<span id="capability-ingresses-observational-iptw-mean-from_moments"></span>

#### Observational IPTW covariate adjustment — Portable moments

**Recorded path:** `from_moments`

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.covariate_unavailable`

**Reason:** a moments cube holds no per-unit rows to attach a covariate to.


<span id="capability-ingresses-observational-aipw-dml"></span>

### Observational AIPW/DML covariate adjustment

<span id="capability-ingresses-observational-aipw-dml-from_unit_summary"></span>

#### Observational AIPW/DML covariate adjustment — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

<span id="capability-ingresses-observational-aipw-dml-from_unit_panel"></span>

#### Observational AIPW/DML covariate adjustment — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

<span id="capability-ingresses-observational-aipw-dml-from_switchback_panel"></span>

#### Observational AIPW/DML covariate adjustment — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.identification`

**Reason:** a switchback frame identifies a Randomized design only: restricted, observational and encouragement designs are unsupported

**Stage:** declaration

<span id="capability-ingresses-observational-aipw-dml-from_definitions"></span>

#### Observational AIPW/DML covariate adjustment — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

<span id="capability-ingresses-observational-aipw-dml-from_unit_day_artifact"></span>

#### Observational AIPW/DML covariate adjustment — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

<span id="capability-ingresses-observational-aipw-dml-from_moments"></span>

#### Observational AIPW/DML covariate adjustment — Portable moments

**Recorded path:** `from_moments`

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.covariate_unavailable`

**Reason:** a moments cube holds no per-unit rows to attach a covariate to.


<span id="capability-ingresses-categorical-observational-adjustment"></span>

### Categorical observational adjustment (string covariates)

<span id="capability-ingresses-categorical-observational-adjustment-from_unit_summary"></span>

#### Categorical observational adjustment (string covariates) — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

<span id="capability-ingresses-categorical-observational-adjustment-from_unit_panel"></span>

#### Categorical observational adjustment (string covariates) — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

<span id="capability-ingresses-categorical-observational-adjustment-from_switchback_panel"></span>

#### Categorical observational adjustment (string covariates) — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.identification`

**Reason:** a switchback frame identifies a Randomized design only: restricted, observational and encouragement designs are unsupported

**Stage:** declaration

<span id="capability-ingresses-categorical-observational-adjustment-from_definitions"></span>

#### Categorical observational adjustment (string covariates) — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

<span id="capability-ingresses-categorical-observational-adjustment-from_unit_day_artifact"></span>

#### Categorical observational adjustment (string covariates) — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

<span id="capability-ingresses-categorical-observational-adjustment-from_moments"></span>

#### Categorical observational adjustment (string covariates) — Portable moments

**Recorded path:** `from_moments`

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.covariate_unavailable`

**Reason:** a moments cube holds no per-unit rows to attach a covariate to.


<span id="capability-ingresses-categorical-missing-policy"></span>

### Categorical nulls with a nondefault missing-value policy

<span id="capability-ingresses-categorical-missing-policy-from_unit_summary"></span>

#### Categorical nulls with a nondefault missing-value policy — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

<span id="capability-ingresses-categorical-missing-policy-from_unit_panel"></span>

#### Categorical nulls with a nondefault missing-value policy — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

<span id="capability-ingresses-categorical-missing-policy-from_switchback_panel"></span>

#### Categorical nulls with a nondefault missing-value policy — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.identification`

**Reason:** a switchback frame identifies a Randomized design only: restricted, observational and encouragement designs are unsupported

**Stage:** declaration

<span id="capability-ingresses-categorical-missing-policy-from_definitions"></span>

#### Categorical nulls with a nondefault missing-value policy — Definitions

**Recorded path:** `from_definitions`

**Status:** refused

**Classification:** SOURCE

**Codes:** `adjust.identification.missing_covariates`

**Reason:** the definitions grammar declares no missing-covariate policy, so this path resolves the default missing='refuse' and must name the NULL level.

**Note:** the definitions grammar declares no missing-covariate policy, so the default applies

<span id="capability-ingresses-categorical-missing-policy-from_unit_day_artifact"></span>

#### Categorical nulls with a nondefault missing-value policy — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** refused

**Classification:** SOURCE

**Codes:** `adjust.identification.missing_covariates`

**Reason:** the definitions grammar declares no missing-covariate policy, so this path resolves the default missing='refuse' and must name the NULL level.

**Note:** the default missing-covariate policy is inherited from definitions

<span id="capability-ingresses-categorical-missing-policy-from_moments"></span>

#### Categorical nulls with a nondefault missing-value policy — Portable moments

**Recorded path:** `from_moments`

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.covariate_unavailable`

**Reason:** a moments cube holds no per-unit rows to attach a covariate to.


<span id="capability-ingresses-observational-role-multiplicity"></span>

### Observational primary + secondary + guardrail together (role multiplicity)

<span id="capability-ingresses-observational-role-multiplicity-from_unit_summary"></span>

#### Observational primary + secondary + guardrail together (role multiplicity) — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

<span id="capability-ingresses-observational-role-multiplicity-from_unit_panel"></span>

#### Observational primary + secondary + guardrail together (role multiplicity) — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

<span id="capability-ingresses-observational-role-multiplicity-from_switchback_panel"></span>

#### Observational primary + secondary + guardrail together (role multiplicity) — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.identification`

**Reason:** a switchback frame identifies a Randomized design only: restricted, observational and encouragement designs are unsupported

**Stage:** declaration

<span id="capability-ingresses-observational-role-multiplicity-from_definitions"></span>

#### Observational primary + secondary + guardrail together (role multiplicity) — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

<span id="capability-ingresses-observational-role-multiplicity-from_unit_day_artifact"></span>

#### Observational primary + secondary + guardrail together (role multiplicity) — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

<span id="capability-ingresses-observational-role-multiplicity-from_moments"></span>

#### Observational primary + secondary + guardrail together (role multiplicity) — Portable moments

**Recorded path:** `from_moments`

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.covariate_unavailable`

**Reason:** a moments cube holds no per-unit rows to attach a covariate to.


<span id="capability-ingresses-quantile-tied-rounded-outcomes"></span>

### Quantile inference on tied/rounded outcomes

<span id="capability-ingresses-quantile-tied-rounded-outcomes-from_unit_summary"></span>

#### Quantile inference on tied/rounded outcomes — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

<span id="capability-ingresses-quantile-tied-rounded-outcomes-from_unit_panel"></span>

#### Quantile inference on tied/rounded outcomes — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** unwindowed metric

**Status:** supported

**Condition:** windowed metric

**Status:** refused

**Classification:** UNFINISHED

**Codes:** `frame.metric.window_days_supported`

**Reason:** a windowed quantile is declared by the warehouse model and read from per-unit totals, yet refused on the frame path with no stated statistical reason

**Stage:** declaration

**Tracked issue:** `6z2f`

**Note:** the panel's per-unit collapse is the same per-unit total mean/ratio already use

<span id="capability-ingresses-quantile-tied-rounded-outcomes-from_switchback_panel"></span>

#### Quantile inference on tied/rounded outcomes — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.metric`

**Reason:** switchback metrics must be a mean or a conversion

**Stage:** declaration

<span id="capability-ingresses-quantile-tied-rounded-outcomes-from_definitions"></span>

#### Quantile inference on tied/rounded outcomes — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

<span id="capability-ingresses-quantile-tied-rounded-outcomes-from_unit_day_artifact"></span>

#### Quantile inference on tied/rounded outcomes — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

<span id="capability-ingresses-quantile-tied-rounded-outcomes-from_moments"></span>

#### Quantile inference on tied/rounded outcomes — Portable moments

**Recorded path:** `from_moments`

**Condition:** genuine quantile export

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.quantile_no_moments`

**Reason:** a quantile has no moments representation -- export() itself refuses before a from_moments cube could ever exist.

**Condition:** quantile declared over a scalar cube

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.unit_rows_unavailable`

**Reason:** unit-level outcomes are unavailable from an aggregated moments source

**Stage:** request

**Condition:** breakout over a scalar cube

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.breakout_unavailable`

**Reason:** the source carries no breakout catalog: run_breakout needs a native or panel source

**Stage:** request

**Condition:** day-axis read over a scalar cube

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.grain_unavailable`

**Reason:** a constructor cannot provide the requested grain (for example, a summary has no day axis or a moments cube did not retain the requested grain)

**Stage:** request


<span id="capability-ingresses-quantile-checkpoint-retention"></span>

### Registered conversion checkpoint retaining an unmodeled, unwindowed quantile catalog

<span id="capability-ingresses-quantile-checkpoint-retention-from_unit_summary"></span>

#### Registered conversion checkpoint retaining an unmodeled, unwindowed quantile catalog — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

**Note:** export and replay

<span id="capability-ingresses-quantile-checkpoint-retention-from_unit_panel"></span>

#### Registered conversion checkpoint retaining an unmodeled, unwindowed quantile catalog — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

**Note:** export and replay

<span id="capability-ingresses-quantile-checkpoint-retention-from_switchback_panel"></span>

#### Registered conversion checkpoint retaining an unmodeled, unwindowed quantile catalog — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** not comparable

**Classification:** SOURCE

**Reason:** switchback accepts only mean/conversion metrics and refuses a quantile catalog (source.frame.switchback.metric); it also has no registered sequential construction. This dataset has no switchback schedule.

<span id="capability-ingresses-quantile-checkpoint-retention-from_definitions"></span>

#### Registered conversion checkpoint retaining an unmodeled, unwindowed quantile catalog — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

**Note:** export and replay

<span id="capability-ingresses-quantile-checkpoint-retention-from_unit_day_artifact"></span>

#### Registered conversion checkpoint retaining an unmodeled, unwindowed quantile catalog — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

**Note:** export and replay

<span id="capability-ingresses-quantile-checkpoint-retention-from_moments"></span>

#### Registered conversion checkpoint retaining an unmodeled, unwindowed quantile catalog — Portable moments

**Recorded path:** `from_moments`

**Condition:** checkpoint replay

**Status:** supported

**Condition:** a moments-replayed source receives an export request for a sequential checkpoint

**Status:** refused

**Codes:** `evidence.operation_unavailable`

**Reason:** A moments replay cannot be re-exported as a checkpoint.

**Operation:** `workflows.export_moments(sequential checkpoint)`

**Stage:** read


<span id="capability-ingresses-observational-quantile-refusal"></span>

### Quantile metric under an Observational design (two-sided, zero-null request)

<span id="capability-ingresses-observational-quantile-refusal-from_unit_summary"></span>

#### Quantile metric under an Observational design (two-sided, zero-null request) — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** two-sided zero null

**Status:** refused

**Classification:** CONSTRUCTION

**Codes:** `readout.observational.quantile`

**Reason:** no quantile estimator under an observational design (readout.observational.quantile)

**Condition:** observational quantile with absolute margin

**Status:** refused

**Codes:** `readout.metric.quantile_alternative`

**Reason:** Observational quantile requests retain construction/read refusal precedence.

**Operation:** `from_unit_summary()/run()`

**Stage:** read

**Condition:** observational quantile with one-sided alternative

**Status:** refused

**Codes:** `readout.metric.quantile_alternative`

**Reason:** Observational quantile requests retain construction/read refusal precedence.

**Operation:** `from_unit_summary()/run()`

**Stage:** read

**Condition:** observational quantile with relative margin

**Status:** refused

**Codes:** `plan.observational.relative_margin`

**Reason:** Observational quantile requests retain construction/read refusal precedence.

**Operation:** `from_unit_summary()/run()`

**Stage:** construction

<span id="capability-ingresses-observational-quantile-refusal-from_unit_panel"></span>

#### Quantile metric under an Observational design (two-sided, zero-null request) — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** two-sided zero null

**Status:** refused

**Classification:** CONSTRUCTION

**Codes:** `readout.observational.quantile`

**Reason:** no quantile estimator under an observational design (readout.observational.quantile)

**Condition:** observational relative-margin plan

**Status:** refused

**Codes:** `plan.observational.relative_margin`

**Reason:** Observational quantile requests retain construction/read refusal precedence.

**Operation:** `from_unit_panel()/run()`

**Stage:** construction

**Condition:** observational quantile with absolute margin

**Status:** refused

**Codes:** `readout.metric.quantile_alternative`

**Reason:** Observational quantile requests retain construction/read refusal precedence.

**Operation:** `from_unit_panel()/run()`

**Stage:** read

**Condition:** observational quantile with one-sided alternative

**Status:** refused

**Codes:** `readout.metric.quantile_alternative`

**Reason:** Observational quantile requests retain construction/read refusal precedence.

**Operation:** `from_unit_panel()/run()`

**Stage:** read

<span id="capability-ingresses-observational-quantile-refusal-from_switchback_panel"></span>

#### Quantile metric under an Observational design (two-sided, zero-null request) — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.identification`

**Reason:** a switchback frame identifies a Randomized design only: restricted, observational and encouragement designs are unsupported

**Stage:** declaration

<span id="capability-ingresses-observational-quantile-refusal-from_definitions"></span>

#### Quantile metric under an Observational design (two-sided, zero-null request) — Definitions

**Recorded path:** `from_definitions`

**Condition:** two-sided zero null

**Status:** refused

**Classification:** CONSTRUCTION

**Codes:** `readout.observational.quantile`

**Reason:** an observational design has no quantile estimator (readout.observational.quantile)

**Condition:** observational quantile with absolute margin

**Status:** refused

**Codes:** `readout.metric.quantile_alternative`

**Reason:** Quantile one-sided or shifted-null requests refuse before observational dispatch.

**Operation:** `run()`

**Stage:** read

**Condition:** observational quantile with one sided

**Status:** refused

**Codes:** `readout.metric.quantile_alternative`

**Reason:** Quantile one-sided or shifted-null requests refuse before observational dispatch.

**Operation:** `run()`

**Stage:** read

**Condition:** observational mean guardrail with relative margin

**Status:** refused

**Codes:** `plan.observational.relative_margin`

**Reason:** Observational relative-margin plans refuse before loading evidence.

**Operation:** `from_definitions()`

**Stage:** construction

<span id="capability-ingresses-observational-quantile-refusal-from_unit_day_artifact"></span>

#### Quantile metric under an Observational design (two-sided, zero-null request) — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** two-sided zero null

**Status:** refused

**Classification:** CONSTRUCTION

**Codes:** `readout.observational.quantile`

**Reason:** no quantile estimator under an observational design (readout.observational.quantile)

**Condition:** observational quantile with absolute margin

**Status:** refused

**Codes:** `readout.metric.quantile_alternative`

**Reason:** Quantile one-sided or shifted-null requests refuse before observational dispatch.

**Operation:** `run()`

**Stage:** read

**Condition:** observational quantile with one sided

**Status:** refused

**Codes:** `readout.metric.quantile_alternative`

**Reason:** Quantile one-sided or shifted-null requests refuse before observational dispatch.

**Operation:** `run()`

**Stage:** read

**Condition:** observational relative-margin plan

**Status:** not expressible

**Reason:** The producer refuses plan.observational.relative_margin before publication; artifact adoption cannot override its immutable plan.

**Operation:** `produce artifact context`

**Stage:** construction

<span id="capability-ingresses-observational-quantile-refusal-from_moments"></span>

#### Quantile metric under an Observational design (two-sided, zero-null request) — Portable moments

**Recorded path:** `from_moments`

**Condition:** two-sided zero null over a scalar cube

**Status:** refused

**Classification:** CONSTRUCTION

**Codes:** `readout.observational.quantile`

**Reason:** the quantile declaration over scalar moments must refuse its absent observational estimator before using those moments (readout.observational.quantile).

**Condition:** Observational quantile request is two-sided and has no shifted margin

**Status:** refused

**Codes:** `readout.observational.quantile`

**Reason:** No observational quantile estimator is available.

**Operation:** `run() quantile`

**Stage:** read

**Condition:** Observational quantile request declares a one-sided alternative

**Status:** refused

**Codes:** `readout.metric.quantile_alternative`

**Reason:** Quantiles do not support one-sided alternative inference.

**Operation:** `run() quantile with one-sided alternative`

**Stage:** read

**Condition:** Observational quantile request declares a marginless guardrail

**Status:** refused

**Codes:** `plan.guardrail.margin_required`

**Reason:** A guardrail without a margin is refused when its plan compiles, before quantile readout dispatch.

**Operation:** `from_moments(..., quantile guardrail without a margin)`

**Stage:** construction

**Condition:** Observational quantile request declares absolute margin

**Status:** refused

**Codes:** `readout.metric.quantile_alternative`

**Reason:** Quantile shifted-null inference is unsupported.

**Operation:** `run() quantile with absolute margin`

**Stage:** read

**Condition:** Quantile metric has a declared relative margin under Observational design

**Status:** refused

**Codes:** `plan.observational.relative_margin`

**Reason:** Declared relative-margin validation precedes quantile readout dispatch.

**Operation:** `from_moments(..., quantile + relative margin)`

**Stage:** construction

**Condition:** Observational design declares relative margin

**Status:** refused

**Codes:** `plan.observational.relative_margin`

**Reason:** Observational readouts do not implement a shifted relative null.

**Operation:** `from_moments(..., plan=relative margin)`

**Stage:** construction

**Condition:** breakout over a scalar cube

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.breakout_unavailable`

**Reason:** the source carries no breakout catalog: run_breakout needs a native or panel source

**Stage:** request

**Condition:** day-axis read over a scalar cube

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.grain_unavailable`

**Reason:** a constructor cannot provide the requested grain (for example, a summary has no day axis or a moments cube did not retain the requested grain)

**Stage:** request

**Note:** genuine observational quantile export raises the same estimator limit


<span id="capability-ingresses-clustered-negative-mean"></span>

### Clustered ratio metric with a non-positive control-arm numerator

<span id="capability-ingresses-clustered-negative-mean-from_unit_summary"></span>

#### Clustered ratio metric with a non-positive control-arm numerator — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

<span id="capability-ingresses-clustered-negative-mean-from_unit_panel"></span>

#### Clustered ratio metric with a non-positive control-arm numerator — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.cluster_unavailable`

**Reason:** the source cannot supply cluster identities or cluster counts

**Stage:** declaration

**Note:** build clustered arm totals with the unit summary instead

<span id="capability-ingresses-clustered-negative-mean-from_switchback_panel"></span>

#### Clustered ratio metric with a non-positive control-arm numerator — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** not expressible

**Reason:** from_switchback_panel takes no cluster= parameter: a switchback contrast has no arm-cluster shape

**Stage:** declaration

<span id="capability-ingresses-clustered-negative-mean-from_definitions"></span>

#### Clustered ratio metric with a non-positive control-arm numerator — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

<span id="capability-ingresses-clustered-negative-mean-from_unit_day_artifact"></span>

#### Clustered ratio metric with a non-positive control-arm numerator — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

<span id="capability-ingresses-clustered-negative-mean-from_moments"></span>

#### Clustered ratio metric with a non-positive control-arm numerator — Portable moments

**Recorded path:** `from_moments`

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.cluster_unavailable`

**Reason:** export_source_moments refuses a clustered, non-Encouragement source ('cluster transport requires a complete encouragement compliance payload') -- the moments wire format carries no cluster marker, so from_moments would rehydrate the cluster rows as unit-grain ratio moments and silently misread them.


<span id="capability-ingresses-clustered-zero-relative-variance"></span>

### Clustered zero relative variance retains its point and any available additive interval

<span id="capability-ingresses-clustered-zero-relative-variance-from_unit_summary"></span>

#### Clustered zero relative variance retains its point and any available additive interval — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

<span id="capability-ingresses-clustered-zero-relative-variance-from_unit_panel"></span>

#### Clustered zero relative variance retains its point and any available additive interval — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.cluster_unavailable`

**Reason:** the source cannot supply cluster identities or cluster counts

**Stage:** declaration

**Note:** build clustered arm totals with the unit summary instead

<span id="capability-ingresses-clustered-zero-relative-variance-from_switchback_panel"></span>

#### Clustered zero relative variance retains its point and any available additive interval — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** not expressible

**Reason:** from_switchback_panel takes no cluster= parameter: a switchback contrast has no arm-cluster shape

**Stage:** declaration

<span id="capability-ingresses-clustered-zero-relative-variance-from_definitions"></span>

#### Clustered zero relative variance retains its point and any available additive interval — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

<span id="capability-ingresses-clustered-zero-relative-variance-from_unit_day_artifact"></span>

#### Clustered zero relative variance retains its point and any available additive interval — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

<span id="capability-ingresses-clustered-zero-relative-variance-from_moments"></span>

#### Clustered zero relative variance retains its point and any available additive interval — Portable moments

**Recorded path:** `from_moments`

**Status:** refused

**Classification:** SOURCE

**Codes:** `evidence.cluster_unavailable`

**Reason:** export_source_moments refuses a clustered, non-Encouragement source ('cluster transport requires a complete encouragement compliance payload') -- the moments wire format carries no cluster marker, so from_moments would rehydrate the cluster rows as unit-grain ratio moments and silently misread them.


<span id="capability-ingresses-planning-baseline-mean"></span>

### Planning baseline, mean metric

<span id="capability-ingresses-planning-baseline-mean-from_unit_summary"></span>

#### Planning baseline, mean metric — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** unclustered mean baseline is the oracle

**Status:** supported

**Operation:** `planning_baseline(mean)`

**Stage:** read

<span id="capability-ingresses-planning-baseline-mean-from_unit_panel"></span>

#### Planning baseline, mean metric — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** unclustered mean baseline matches the unit-summary oracle field for field

**Status:** supported

**Operation:** `planning_baseline(mean)`

**Stage:** read

<span id="capability-ingresses-planning-baseline-mean-from_switchback_panel"></span>

#### Planning baseline, mean metric — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** the pilot-fitted SwitchbackBaseline feeds the switchback planner

**Status:** supported

**Operation:** `planning_baseline(mean)`

**Stage:** read

**Note:** the pilot-fitted switchback baseline feeds the switchback solvers

<span id="capability-ingresses-planning-baseline-mean-from_definitions"></span>

#### Planning baseline, mean metric — Definitions

**Recorded path:** `from_definitions`

**Condition:** unclustered mean baseline matches the unit-summary oracle field for field

**Status:** supported

**Operation:** `planning_baseline(mean)`

**Stage:** read

<span id="capability-ingresses-planning-baseline-mean-from_unit_day_artifact"></span>

#### Planning baseline, mean metric — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** unclustered mean baseline matches the unit-summary oracle field for field

**Status:** supported

**Operation:** `planning_baseline(mean)`

**Stage:** read

<span id="capability-ingresses-planning-baseline-mean-from_moments"></span>

#### Planning baseline, mean metric — Portable moments

**Recorded path:** `from_moments`

**Condition:** unclustered mean baseline matches the unit-summary oracle field for field

**Status:** supported

**Operation:** `planning_baseline(mean)`

**Stage:** read


<span id="capability-ingresses-planning-baseline-quantile"></span>

### Planning baseline, quantile metric

<span id="capability-ingresses-planning-baseline-quantile-from_unit_summary"></span>

#### Planning baseline, quantile metric — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** quantile baseline from raw unit values

**Status:** supported

**Operation:** `planning_baseline(quantile)`

**Stage:** read

<span id="capability-ingresses-planning-baseline-quantile-from_unit_panel"></span>

#### Planning baseline, quantile metric — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** unwindowed quantile; raw per-unit panel values remain available

**Status:** supported

**Operation:** `planning_baseline(quantile)`

**Stage:** read

**Note:** unwindowed metrics only, as for quantile inference on the panel

<span id="capability-ingresses-planning-baseline-quantile-from_switchback_panel"></span>

#### Planning baseline, quantile metric — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** a quantile metric is refused at the switchback metric-type gate before any frame is read

**Status:** refused

**Codes:** `source.frame.switchback.metric`

**Reason:** switchback metrics are mean or conversion only

**Operation:** `planning_baseline(quantile)`

**Stage:** construction

<span id="capability-ingresses-planning-baseline-quantile-from_definitions"></span>

#### Planning baseline, quantile metric — Definitions

**Recorded path:** `from_definitions`

**Condition:** declared trigger; the quantile reads triggered control values

**Status:** supported

**Operation:** `planning_baseline(quantile)`

**Stage:** read

<span id="capability-ingresses-planning-baseline-quantile-from_unit_day_artifact"></span>

#### Planning baseline, quantile metric — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** declared trigger with metric-scoped trigger evidence; the artifact quantile matches definitions

**Status:** supported

**Operation:** `planning_baseline(quantile)`

**Stage:** read

<span id="capability-ingresses-planning-baseline-quantile-from_moments"></span>

#### Planning baseline, quantile metric — Portable moments

**Recorded path:** `from_moments`

**Condition:** a quantile declared over an exported scalar cube

**Status:** refused

**Codes:** `analysis.planning_baseline.quantile_source_unavailable`

**Reason:** moments do not retain per-unit values required by quantile planning

**Operation:** `planning_baseline(quantile)`

**Stage:** read

**Condition:** genuine quantile export

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.quantile_no_moments`

**Reason:** a quantile has no moments representation -- export() itself refuses before a from_moments cube could ever exist.


<span id="capability-ingresses-planning-baseline-cluster"></span>

### Planning baseline with a declared cluster (per-unit mean/var; ICC on the metric's per-unit score)

<span id="capability-ingresses-planning-baseline-cluster-from_unit_summary"></span>

#### Planning baseline with a declared cluster (per-unit mean/var; ICC on the metric's per-unit score) — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** declared cluster column; per-unit mean/var with ICC on the metric's per-unit score

**Status:** supported

**Operation:** `planning_baseline(clustered mean)`

**Stage:** read

<span id="capability-ingresses-planning-baseline-cluster-from_unit_panel"></span>

#### Planning baseline with a declared cluster (per-unit mean/var; ICC on the metric's per-unit score) — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** cluster= on a per-day panel whose per-unit collapse cannot carry randomization grain

**Status:** refused

**Codes:** `evidence.cluster_unavailable`

**Reason:** panel grain cannot express clustered randomization

**Operation:** `planning_baseline(clustered mean)`

**Stage:** construction

<span id="capability-ingresses-planning-baseline-cluster-from_switchback_panel"></span>

#### Planning baseline with a declared cluster (per-unit mean/var; ICC on the metric's per-unit score) — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** cluster= on the switchback constructor

**Status:** not expressible

**Reason:** from_switchback_frame has no cluster= parameter: a switchback contrast has no arm-cluster shape

**Operation:** `planning_baseline(clustered mean)`

**Stage:** construction

<span id="capability-ingresses-planning-baseline-cluster-from_definitions"></span>

#### Planning baseline with a declared cluster (per-unit mean/var; ICC on the metric's per-unit score) — Definitions

**Recorded path:** `from_definitions`

**Condition:** declared experiment cluster; the plan agrees with the clustered unit-summary oracle

**Status:** supported

**Operation:** `planning_baseline(clustered mean)`

**Stage:** read

<span id="capability-ingresses-planning-baseline-cluster-from_unit_day_artifact"></span>

#### Planning baseline with a declared cluster (per-unit mean/var; ICC on the metric's per-unit score) — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** published cluster identity; the plan agrees with the clustered unit-summary oracle

**Status:** supported

**Operation:** `planning_baseline(clustered mean)`

**Stage:** read

<span id="capability-ingresses-planning-baseline-cluster-from_moments"></span>

#### Planning baseline with a declared cluster (per-unit mean/var; ICC on the metric's per-unit score) — Portable moments

**Recorded path:** `from_moments`

**Condition:** exporting a clustered unit summary to a portable cube

**Status:** refused

**Codes:** `evidence.cluster_unavailable`

**Reason:** portable moments do not retain randomized cluster identity

**Operation:** `planning_baseline(clustered mean)`

**Stage:** export


<span id="capability-ingresses-planning-baseline-trigger"></span>

### Planning baseline under a declared trigger (triggered-population moments and trigger rate)

<span id="capability-ingresses-planning-baseline-trigger-from_unit_summary"></span>

#### Planning baseline under a declared trigger (triggered-population moments and trigger rate) — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** no trigger can be declared; the assigned population is planned

**Status:** source limited

**Classification:** SOURCE

**Reason:** the constructor accepts no Experiment declaration, so no trigger resolves and the assigned population is planned with trigger_rate=1.0

**Operation:** `planning_baseline(triggered mean)`

**Stage:** read

**Note:** the assigned population is analyzed instead

<span id="capability-ingresses-planning-baseline-trigger-from_unit_panel"></span>

#### Planning baseline under a declared trigger (triggered-population moments and trigger rate) — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** no trigger can be declared; the assigned population is planned

**Status:** source limited

**Classification:** SOURCE

**Reason:** the constructor accepts no Experiment declaration, so no trigger resolves and the assigned population is planned with trigger_rate=1.0

**Operation:** `planning_baseline(triggered mean)`

**Stage:** read

<span id="capability-ingresses-planning-baseline-trigger-from_switchback_panel"></span>

#### Planning baseline under a declared trigger (triggered-population moments and trigger rate) — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** no trigger can be declared; the pilot contrast carries no triggered population

**Status:** source limited

**Classification:** SOURCE

**Reason:** a switchback contrast has no assignment population to trigger into; its SwitchbackBaseline carries no trigger rate

**Operation:** `planning_baseline(triggered mean)`

**Stage:** read

<span id="capability-ingresses-planning-baseline-trigger-from_definitions"></span>

#### Planning baseline under a declared trigger (triggered-population moments and trigger rate) — Definitions

**Recorded path:** `from_definitions`

**Condition:** declared trigger reads actual triggered population

**Status:** supported

**Operation:** `planning_baseline(triggered mean)`

**Stage:** read

<span id="capability-ingresses-planning-baseline-trigger-from_unit_day_artifact"></span>

#### Planning baseline under a declared trigger (triggered-population moments and trigger rate) — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** artifact trigger evidence agrees with definitions

**Status:** supported

**Operation:** `planning_baseline(triggered mean)`

**Stage:** read

**Condition:** declared trigger but artifact omits required evidence

**Status:** refused

**Codes:** `analysis.planning_baseline.trigger_evidence_unavailable`

**Reason:** artifact lacks declared trigger evidence

**Operation:** `planning_baseline(triggered mean)`

**Stage:** read

**Note:** trigger population, triggered assignment counts and metric-scoped trigger measure statistics, with explicit source snapshot evidence

<span id="capability-ingresses-planning-baseline-trigger-from_moments"></span>

#### Planning baseline under a declared trigger (triggered-population moments and trigger rate) — Portable moments

**Recorded path:** `from_moments`

**Condition:** no trigger can be declared; the assigned population is planned

**Status:** source limited

**Classification:** SOURCE

**Reason:** the constructor accepts no Experiment declaration, so no trigger resolves and the assigned population is planned with trigger_rate=1.0

**Operation:** `planning_baseline(triggered mean)`

**Stage:** read


<span id="capability-ingresses-triggered-day-axis-histories"></span>

### Triggered daily/as-of outcome value and lift history (membership is limited to units whose first eligible trigger is observed by each day, and outcome windows anchor at that unit's trigger)

<span id="capability-ingresses-triggered-day-axis-histories-from_unit_summary"></span>

#### Triggered daily/as-of outcome value and lift history (membership is limited to units whose first eligible trigger is observed by each day, and outcome windows anchor at that unit's trigger) — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** daily values

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Triggered day-axis reads require a supported definitions or unit-day artifact source.

**Operation:** `run_daily(population='triggered')`

**Stage:** read

**Condition:** daily lift

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Triggered day-axis reads require a supported definitions or unit-day artifact source.

**Operation:** `run_daily_lift(population='triggered')`

**Stage:** read

**Condition:** as-of values

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Triggered day-axis reads require a supported definitions or unit-day artifact source.

**Operation:** `run_asof(population='triggered')`

**Stage:** read

**Condition:** as-of lift

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Triggered day-axis reads require a supported definitions or unit-day artifact source.

**Operation:** `run_asof_lift(population='triggered')`

**Stage:** read

<span id="capability-ingresses-triggered-day-axis-histories-from_unit_panel"></span>

#### Triggered daily/as-of outcome value and lift history (membership is limited to units whose first eligible trigger is observed by each day, and outcome windows anchor at that unit's trigger) — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** daily values

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Triggered day-axis reads require a supported definitions or unit-day artifact source.

**Operation:** `run_daily(population='triggered')`

**Stage:** read

**Condition:** daily lift

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Triggered day-axis reads require a supported definitions or unit-day artifact source.

**Operation:** `run_daily_lift(population='triggered')`

**Stage:** read

**Condition:** as-of values

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Triggered day-axis reads require a supported definitions or unit-day artifact source.

**Operation:** `run_asof(population='triggered')`

**Stage:** read

**Condition:** as-of lift

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Triggered day-axis reads require a supported definitions or unit-day artifact source.

**Operation:** `run_asof_lift(population='triggered')`

**Stage:** read

<span id="capability-ingresses-triggered-day-axis-histories-from_switchback_panel"></span>

#### Triggered daily/as-of outcome value and lift history (membership is limited to units whose first eligible trigger is observed by each day, and outcome windows anchor at that unit's trigger) — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** daily values

**Status:** refused

**Codes:** `facade.analysis.contrast_unavailable`

**Reason:** Switchback sources do not provide the parallel-arm contrast used by these day-axis APIs.

**Operation:** `run_daily(population='triggered')`

**Stage:** read

**Condition:** daily lift

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Triggered day-axis reads require a supported definitions or unit-day artifact source.

**Operation:** `run_daily_lift(population='triggered')`

**Stage:** read

**Condition:** as-of values

**Status:** refused

**Codes:** `facade.analysis.contrast_unavailable`

**Reason:** Switchback sources do not provide the parallel-arm contrast used by these day-axis APIs.

**Operation:** `run_asof(population='triggered')`

**Stage:** read

**Condition:** as-of lift

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Triggered day-axis reads require a supported definitions or unit-day artifact source.

**Operation:** `run_asof_lift(population='triggered')`

**Stage:** read

<span id="capability-ingresses-triggered-day-axis-histories-from_definitions"></span>

#### Triggered daily/as-of outcome value and lift history (membership is limited to units whose first eligible trigger is observed by each day, and outcome windows anchor at that unit's trigger) — Definitions

**Recorded path:** `from_definitions`

**Condition:** daily values

**Status:** supported

**Condition:** daily lift

**Status:** supported

**Condition:** as-of values

**Status:** supported

**Condition:** as-of lift

**Status:** supported

**Note:** explicit source snapshot evidence; outcome events are strictly after each unit's first eligible trigger and are cut off by source evidence

<span id="capability-ingresses-triggered-day-axis-histories-from_unit_day_artifact"></span>

#### Triggered daily/as-of outcome value and lift history (membership is limited to units whose first eligible trigger is observed by each day, and outcome windows anchor at that unit's trigger) — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** daily values

**Status:** supported

**Condition:** daily lift

**Status:** supported

**Condition:** as-of values

**Status:** supported

**Condition:** as-of lift

**Status:** supported

**Note:** trigger population and metric-compatible trigger measure statistics

<span id="capability-ingresses-triggered-day-axis-histories-from_moments"></span>

#### Triggered daily/as-of outcome value and lift history (membership is limited to units whose first eligible trigger is observed by each day, and outcome windows anchor at that unit's trigger) — Portable moments

**Recorded path:** `from_moments`

**Condition:** daily values

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Triggered day-axis reads require a supported definitions or unit-day artifact source.

**Operation:** `run_daily(population='triggered')`

**Stage:** read

**Condition:** daily lift

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Triggered day-axis reads require a supported definitions or unit-day artifact source.

**Operation:** `run_daily_lift(population='triggered')`

**Stage:** read

**Condition:** as-of values

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Triggered day-axis reads require a supported definitions or unit-day artifact source.

**Operation:** `run_asof(population='triggered')`

**Stage:** read

**Condition:** as-of lift

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Triggered day-axis reads require a supported definitions or unit-day artifact source.

**Operation:** `run_asof_lift(population='triggered')`

**Stage:** read


<span id="capability-ingresses-triggered-encouragement-compliance"></span>

### Triggered encouragement compliance (daily and as-of lift, compliance only; uptake stays assignment-anchored while membership enters at first eligible trigger)

<span id="capability-ingresses-triggered-encouragement-compliance-from_unit_summary"></span>

#### Triggered encouragement compliance (daily and as-of lift, compliance only; uptake stays assignment-anchored while membership enters at first eligible trigger) — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** refused

**Classification:** SOURCE

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** unit summaries have no trigger declaration or trigger anchors.

<span id="capability-ingresses-triggered-encouragement-compliance-from_unit_panel"></span>

#### Triggered encouragement compliance (daily and as-of lift, compliance only; uptake stays assignment-anchored while membership enters at first eligible trigger) — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** refused

**Classification:** SOURCE

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** unit panels have no trigger declaration or trigger anchors.

<span id="capability-ingresses-triggered-encouragement-compliance-from_switchback_panel"></span>

#### Triggered encouragement compliance (daily and as-of lift, compliance only; uptake stays assignment-anchored while membership enters at first eligible trigger) — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** refused

**Classification:** SOURCE

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** switchback contrasts have no triggered assignment population.

<span id="capability-ingresses-triggered-encouragement-compliance-from_definitions"></span>

#### Triggered encouragement compliance (daily and as-of lift, compliance only; uptake stays assignment-anchored while membership enters at first eligible trigger) — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

**Note:** explicit source snapshot evidence; uptake windows use assignment time and triggered membership uses the first-trigger day

<span id="capability-ingresses-triggered-encouragement-compliance-from_unit_day_artifact"></span>

#### Triggered encouragement compliance (daily and as-of lift, compliance only; uptake stays assignment-anchored while membership enters at first eligible trigger) — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

**Note:** trigger population and encouragement uptake evidence

<span id="capability-ingresses-triggered-encouragement-compliance-from_moments"></span>

#### Triggered encouragement compliance (daily and as-of lift, compliance only; uptake stays assignment-anchored while membership enters at first eligible trigger) — Portable moments

**Recorded path:** `from_moments`

**Status:** refused

**Classification:** SOURCE

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** portable moments have no per-unit trigger anchors.


<span id="capability-ingresses-triggered-clustered-day-axis"></span>

### Triggered clustered daily/as-of outcome readouts

<span id="capability-ingresses-triggered-clustered-day-axis-from_unit_summary"></span>

#### Triggered clustered daily/as-of outcome readouts — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** daily values

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Triggered day-axis reads require a supported definitions or unit-day artifact source.

**Operation:** `run_daily(population='triggered')`

**Stage:** read

**Condition:** daily lift

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Triggered day-axis reads require a supported definitions or unit-day artifact source.

**Operation:** `run_daily_lift(population='triggered')`

**Stage:** read

**Condition:** as-of values

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Triggered day-axis reads require a supported definitions or unit-day artifact source.

**Operation:** `run_asof(population='triggered')`

**Stage:** read

**Condition:** as-of lift

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Triggered day-axis reads require a supported definitions or unit-day artifact source.

**Operation:** `run_asof_lift(population='triggered')`

**Stage:** read

<span id="capability-ingresses-triggered-clustered-day-axis-from_unit_panel"></span>

#### Triggered clustered daily/as-of outcome readouts — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** daily values

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Triggered day-axis reads require a supported definitions or unit-day artifact source.

**Operation:** `run_daily(population='triggered')`

**Stage:** read

**Condition:** daily lift

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Triggered day-axis reads require a supported definitions or unit-day artifact source.

**Operation:** `run_daily_lift(population='triggered')`

**Stage:** read

**Condition:** as-of values

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Triggered day-axis reads require a supported definitions or unit-day artifact source.

**Operation:** `run_asof(population='triggered')`

**Stage:** read

**Condition:** as-of lift

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Triggered day-axis reads require a supported definitions or unit-day artifact source.

**Operation:** `run_asof_lift(population='triggered')`

**Stage:** read

<span id="capability-ingresses-triggered-clustered-day-axis-from_switchback_panel"></span>

#### Triggered clustered daily/as-of outcome readouts — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** daily values

**Status:** refused

**Codes:** `facade.analysis.contrast_unavailable`

**Reason:** Switchback sources do not provide the parallel-arm contrast used by these day-axis APIs.

**Operation:** `run_daily(population='triggered')`

**Stage:** read

**Condition:** daily lift

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Triggered day-axis reads require a supported definitions or unit-day artifact source.

**Operation:** `run_daily_lift(population='triggered')`

**Stage:** read

**Condition:** as-of values

**Status:** refused

**Codes:** `facade.analysis.contrast_unavailable`

**Reason:** Switchback sources do not provide the parallel-arm contrast used by these day-axis APIs.

**Operation:** `run_asof(population='triggered')`

**Stage:** read

**Condition:** as-of lift

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Triggered day-axis reads require a supported definitions or unit-day artifact source.

**Operation:** `run_asof_lift(population='triggered')`

**Stage:** read

<span id="capability-ingresses-triggered-clustered-day-axis-from_definitions"></span>

#### Triggered clustered daily/as-of outcome readouts — Definitions

**Recorded path:** `from_definitions`

**Condition:** clustered experiment requests triggered outcome history

**Status:** refused

**Codes:** `facade.analysis.clustered_day_axis`

**Reason:** Clustered triggered outcome day-axis is refused.

**Operation:** `run_daily()/run_asof() outcome history`

**Stage:** read

**Condition:** clustered triggered experiment requests uptake-only history

**Status:** supported

**Operation:** `run_daily_lift()/run_asof_lift() compliance-only history`

**Stage:** read

**Condition:** clustered experiment requests triggered outcome population

**Status:** supported

**Operation:** `run()`

**Stage:** read

<span id="capability-ingresses-triggered-clustered-day-axis-from_unit_day_artifact"></span>

#### Triggered clustered daily/as-of outcome readouts — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** clustered experiment requests triggered outcome history from an adopted artifact

**Status:** refused

**Codes:** `facade.analysis.clustered_day_axis`

**Reason:** Clustered triggered outcome day-axis is refused for artifact-backed history.

**Operation:** `run_daily()/run_asof() outcome history`

**Stage:** read

**Condition:** clustered triggered experiment requests uptake-only history from an adopted artifact

**Status:** supported

**Operation:** `run_daily_lift()/run_asof_lift() compliance-only history`

**Stage:** read

<span id="capability-ingresses-triggered-clustered-day-axis-from_moments"></span>

#### Triggered clustered daily/as-of outcome readouts — Portable moments

**Recorded path:** `from_moments`

**Condition:** daily values

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Triggered day-axis reads require a supported definitions or unit-day artifact source.

**Operation:** `run_daily(population='triggered')`

**Stage:** read

**Condition:** daily lift

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Triggered day-axis reads require a supported definitions or unit-day artifact source.

**Operation:** `run_daily_lift(population='triggered')`

**Stage:** read

**Condition:** as-of values

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Triggered day-axis reads require a supported definitions or unit-day artifact source.

**Operation:** `run_asof(population='triggered')`

**Stage:** read

**Condition:** as-of lift

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Triggered day-axis reads require a supported definitions or unit-day artifact source.

**Operation:** `run_asof_lift(population='triggered')`

**Stage:** read


<span id="capability-ingresses-untreated-spillover"></span>

### Two-stage spillover: policy difference under personal nonoffer

<span id="capability-ingresses-untreated-spillover-from_unit_summary"></span>

#### Two-stage spillover: policy difference under personal nonoffer — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** Every eligible person has an assignment; everyone without an offer has a measured conversion

**Status:** supported

**Operation:** `from_frame(..., shape='summary', design=...) and run()`

**Stage:** read

<span id="capability-ingresses-untreated-spillover-from_unit_panel"></span>

#### Two-stage spillover: policy difference under personal nonoffer — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** The input does not carry store policy, individual assignment and planned offer counts

**Status:** not expressible

**Reason:** Use a one-row-per-person summary or a named warehouse experiment.

**Operation:** `run(experiment, reader)`

**Stage:** construction

<span id="capability-ingresses-untreated-spillover-from_switchback_panel"></span>

#### Two-stage spillover: policy difference under personal nonoffer — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** The input does not carry store policy, individual assignment and planned offer counts

**Status:** not expressible

**Reason:** Use a one-row-per-person summary or a named warehouse experiment.

**Operation:** `run(experiment, reader)`

**Stage:** construction

<span id="capability-ingresses-untreated-spillover-from_definitions"></span>

#### Two-stage spillover: policy difference under personal nonoffer — Definitions

**Recorded path:** `from_definitions`

**Condition:** Both assignment logs match the frozen roster and the purchase feed covers all windows

**Status:** supported

**Operation:** `from_definitions(..., experiment_name=...) and run()`

**Stage:** read

<span id="capability-ingresses-untreated-spillover-from_unit_day_artifact"></span>

#### Two-stage spillover: policy difference under personal nonoffer — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** The input does not carry store policy, individual assignment and planned offer counts

**Status:** not expressible

**Reason:** Use a one-row-per-person summary or a named warehouse experiment.

**Operation:** `run(experiment, reader)`

**Stage:** construction

<span id="capability-ingresses-untreated-spillover-from_moments"></span>

#### Two-stage spillover: policy difference under personal nonoffer — Portable moments

**Recorded path:** `from_moments`

**Condition:** The input does not carry store policy, individual assignment and planned offer counts

**Status:** not expressible

**Reason:** Use a one-row-per-person summary or a named warehouse experiment.

**Operation:** `run(experiment, reader)`

**Stage:** construction


<span id="capability-ingresses-logged-policy-contrast"></span>

### Logged-policy contrast

<span id="capability-ingresses-logged-policy-contrast-from_unit_summary"></span>

#### Logged-policy contrast — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** Experiment builders cannot carry a LoggedTrace, which is the required contrast input

**Status:** not expressible

**Reason:** Policy contrast is a separate family-owned estimand, not an experiment readout.

**Operation:** `estimate_policy_contrast()`

**Stage:** read

<span id="capability-ingresses-logged-policy-contrast-from_unit_panel"></span>

#### Logged-policy contrast — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** Experiment builders cannot carry a LoggedTrace, which is the required contrast input

**Status:** not expressible

**Reason:** Policy contrast is a separate family-owned estimand, not an experiment readout.

**Operation:** `estimate_policy_contrast()`

**Stage:** read

<span id="capability-ingresses-logged-policy-contrast-from_switchback_panel"></span>

#### Logged-policy contrast — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** Experiment builders cannot carry a LoggedTrace, which is the required contrast input

**Status:** not expressible

**Reason:** Policy contrast is a separate family-owned estimand, not an experiment readout.

**Operation:** `estimate_policy_contrast()`

**Stage:** read

<span id="capability-ingresses-logged-policy-contrast-from_definitions"></span>

#### Logged-policy contrast — Definitions

**Recorded path:** `from_definitions`

**Condition:** Experiment builders cannot carry a LoggedTrace, which is the required contrast input

**Status:** not expressible

**Reason:** Policy contrast is a separate family-owned estimand, not an experiment readout.

**Operation:** `estimate_policy_contrast()`

**Stage:** read

<span id="capability-ingresses-logged-policy-contrast-from_unit_day_artifact"></span>

#### Logged-policy contrast — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** Experiment builders cannot carry a LoggedTrace, which is the required contrast input

**Status:** not expressible

**Reason:** Policy contrast is a separate family-owned estimand, not an experiment readout.

**Operation:** `estimate_policy_contrast()`

**Stage:** read

<span id="capability-ingresses-logged-policy-contrast-from_moments"></span>

#### Logged-policy contrast — Portable moments

**Recorded path:** `from_moments`

**Condition:** Experiment builders cannot carry a LoggedTrace, which is the required contrast input

**Status:** not expressible

**Reason:** Policy contrast is a separate family-owned estimand, not an experiment readout.

**Operation:** `estimate_policy_contrast()`

**Stage:** read

<span id="capability-ingresses-logged-policy-contrast-family"></span>

#### Logged-policy contrast: family-owned operations

**Condition:** trace has valid logging-policy evidence and support

**Status:** supported

**Operation:** `estimate_policy_contrast()`

**Stage:** read


<span id="capability-ingresses-tabular-policy-persistence"></span>

### TabularPolicy persistence (JSON dump and validate)

<span id="capability-ingresses-tabular-policy-persistence-from_unit_summary"></span>

#### TabularPolicy persistence (JSON dump and validate) — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** experiment runs save effect-analysis results rather than policy tables

**Status:** not comparable

**Reason:** TabularPolicy persistence is family-owned and not an experiment run persistence operation.

**Operation:** `TabularPolicy JSON/Python persistence`

**Stage:** read

<span id="capability-ingresses-tabular-policy-persistence-from_unit_panel"></span>

#### TabularPolicy persistence (JSON dump and validate) — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** experiment runs save effect-analysis results rather than policy tables

**Status:** not comparable

**Reason:** TabularPolicy persistence is family-owned and not an experiment run persistence operation.

**Operation:** `TabularPolicy JSON/Python persistence`

**Stage:** read

<span id="capability-ingresses-tabular-policy-persistence-from_switchback_panel"></span>

#### TabularPolicy persistence (JSON dump and validate) — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** experiment runs save effect-analysis results rather than policy tables

**Status:** not comparable

**Reason:** TabularPolicy persistence is family-owned and not an experiment run persistence operation.

**Operation:** `TabularPolicy JSON/Python persistence`

**Stage:** read

<span id="capability-ingresses-tabular-policy-persistence-from_definitions"></span>

#### TabularPolicy persistence (JSON dump and validate) — Definitions

**Recorded path:** `from_definitions`

**Condition:** experiment runs save effect-analysis results rather than policy tables

**Status:** not comparable

**Reason:** TabularPolicy persistence is family-owned and not an experiment run persistence operation.

**Operation:** `TabularPolicy JSON/Python persistence`

**Stage:** read

<span id="capability-ingresses-tabular-policy-persistence-from_unit_day_artifact"></span>

#### TabularPolicy persistence (JSON dump and validate) — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** experiment runs save effect-analysis results rather than policy tables

**Status:** not comparable

**Reason:** TabularPolicy persistence is family-owned and not an experiment run persistence operation.

**Operation:** `TabularPolicy JSON/Python persistence`

**Stage:** read

<span id="capability-ingresses-tabular-policy-persistence-from_moments"></span>

#### TabularPolicy persistence (JSON dump and validate) — Portable moments

**Recorded path:** `from_moments`

**Condition:** experiment runs save effect-analysis results rather than policy tables

**Status:** not comparable

**Reason:** TabularPolicy persistence is family-owned and not an experiment run persistence operation.

**Operation:** `TabularPolicy JSON/Python persistence`

**Stage:** read

<span id="capability-ingresses-tabular-policy-persistence-family"></span>

#### TabularPolicy persistence (JSON dump and validate): family-owned operations

**Condition:** string-keyed policy table uses JSON mode

**Status:** supported

**Operation:** `TabularPolicy.model_dump_json()/model_validate_json()`

**Stage:** read

**Condition:** integer-keyed policy table uses JSON mode

**Status:** refused

**Codes:** `logged_policy.policy.json_context_key`

**Reason:** Typed context keys cannot be represented by JSON object keys.

**Operation:** `TabularPolicy.model_dump_json()`

**Stage:** read

**Condition:** typed-key policy table uses Python mode or trusted pickle

**Status:** supported

**Operation:** `TabularPolicy.model_dump(mode='python')/trusted pickle`

**Stage:** read


<span id="capability-ingresses-cuped-linear-interaction-scalar_cuped"></span>

### Mixed-covariate CUPED × scalar cuped

<span id="capability-ingresses-cuped-linear-interaction-scalar_cuped-from_unit_summary"></span>

#### Mixed-covariate CUPED × scalar cuped — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

**Reason:** Frames declare scalar and mixed covariates on their respective Method.cuped(covariates=...) declarations; definitions derive the scalar covariate from n_pre_periods > 0 and bind mixed covariates separately. Publish the cuped_preperiod artifact extension.

**Stage:** read

<span id="capability-ingresses-cuped-linear-interaction-scalar_cuped-from_unit_panel"></span>

#### Mixed-covariate CUPED × scalar cuped — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

**Reason:** Frames declare scalar and mixed covariates on their respective Method.cuped(covariates=...) declarations; definitions derive the scalar covariate from n_pre_periods > 0 and bind mixed covariates separately. Publish the cuped_preperiod artifact extension.

**Stage:** read

<span id="capability-ingresses-cuped-linear-interaction-scalar_cuped-from_switchback_panel"></span>

#### Mixed-covariate CUPED × scalar cuped — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** source limited

**Classification:** SOURCE

**Codes:** `source.frame.switchback.metric`

**Reason:** Switchback is a different design from independent-unit CUPED.

**Stage:** construction

<span id="capability-ingresses-cuped-linear-interaction-scalar_cuped-from_definitions"></span>

#### Mixed-covariate CUPED × scalar cuped — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

**Reason:** Frames declare scalar and mixed covariates on their respective Method.cuped(covariates=...) declarations; definitions derive the scalar covariate from n_pre_periods > 0 and bind mixed covariates separately. Publish the cuped_preperiod artifact extension.

**Stage:** read

<span id="capability-ingresses-cuped-linear-interaction-scalar_cuped-from_unit_day_artifact"></span>

#### Mixed-covariate CUPED × scalar cuped — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

**Reason:** Frames declare scalar and mixed covariates on their respective Method.cuped(covariates=...) declarations; definitions derive the scalar covariate from n_pre_periods > 0 and bind mixed covariates separately. Publish the cuped_preperiod artifact extension.

**Stage:** read

<span id="capability-ingresses-cuped-linear-interaction-scalar_cuped-from_moments"></span>

#### Mixed-covariate CUPED × scalar cuped — Portable moments

**Recorded path:** `from_moments`

**Status:** supported

**Reason:** Format-13 fitted_multicovariate evidence supports continuous/mixed QR replay with a changed alpha; categorical-only requests retain joint-cell replay.

**Stage:** read


<span id="capability-ingresses-cuped-linear-interaction-ratio"></span>

### Mixed-covariate CUPED × ratio

<span id="capability-ingresses-cuped-linear-interaction-ratio-from_unit_summary"></span>

#### Mixed-covariate CUPED × ratio — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

**Stage:** read

<span id="capability-ingresses-cuped-linear-interaction-ratio-from_unit_panel"></span>

#### Mixed-covariate CUPED × ratio — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

**Stage:** read

<span id="capability-ingresses-cuped-linear-interaction-ratio-from_switchback_panel"></span>

#### Mixed-covariate CUPED × ratio — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** source limited

**Classification:** SOURCE

**Codes:** `source.frame.switchback.metric`

**Reason:** Switchback is a different design from independent-unit CUPED.

**Stage:** construction

<span id="capability-ingresses-cuped-linear-interaction-ratio-from_definitions"></span>

#### Mixed-covariate CUPED × ratio — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

**Stage:** read

<span id="capability-ingresses-cuped-linear-interaction-ratio-from_unit_day_artifact"></span>

#### Mixed-covariate CUPED × ratio — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

**Stage:** read

<span id="capability-ingresses-cuped-linear-interaction-ratio-from_moments"></span>

#### Mixed-covariate CUPED × ratio — Portable moments

**Recorded path:** `from_moments`

**Status:** supported

**Reason:** Format-13 fitted_multicovariate evidence supports continuous/mixed QR replay with a changed alpha; categorical-only requests retain joint-cell replay.

**Stage:** read


<span id="capability-ingresses-cuped-linear-interaction-clustering"></span>

### Mixed-covariate CUPED × clustering

<span id="capability-ingresses-cuped-linear-interaction-clustering-from_unit_summary"></span>

#### Mixed-covariate CUPED × clustering — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** unfinished

**Classification:** UNFINISHED

**Codes:** `estimation.engine.cuped_linear_cluster`

**Reason:** Implement cluster-robust HC2/Bell-McCaffrey inference for the WLS fit.

**Stage:** construction

<span id="capability-ingresses-cuped-linear-interaction-clustering-from_unit_panel"></span>

#### Mixed-covariate CUPED × clustering — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** source limited

**Classification:** SOURCE

**Codes:** `evidence.cluster_unavailable`

**Reason:** The panel is collapsed to one row per unit before clustering could apply.

**Stage:** construction

<span id="capability-ingresses-cuped-linear-interaction-clustering-from_switchback_panel"></span>

#### Mixed-covariate CUPED × clustering — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** source limited

**Classification:** SOURCE

**Codes:** `source.frame.switchback.metric`

**Reason:** Switchback is a different design from independent-unit CUPED.

**Stage:** construction

<span id="capability-ingresses-cuped-linear-interaction-clustering-from_definitions"></span>

#### Mixed-covariate CUPED × clustering — Definitions

**Recorded path:** `from_definitions`

**Status:** unfinished

**Classification:** UNFINISHED

**Codes:** `estimation.engine.cuped_linear_cluster`

**Reason:** Implement cluster-robust HC2/Bell-McCaffrey inference for the WLS fit.

**Stage:** construction

<span id="capability-ingresses-cuped-linear-interaction-clustering-from_unit_day_artifact"></span>

#### Mixed-covariate CUPED × clustering — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** unfinished

**Classification:** UNFINISHED

**Codes:** `estimation.engine.cuped_linear_cluster`

**Reason:** Implement cluster-robust HC2/Bell-McCaffrey inference for the WLS fit.

**Stage:** construction

<span id="capability-ingresses-cuped-linear-interaction-clustering-from_moments"></span>

#### Mixed-covariate CUPED × clustering — Portable moments

**Recorded path:** `from_moments`

**Status:** source limited

**Classification:** SOURCE

**Codes:** `facade.analysis_config.cuped_linear_moments`

**Reason:** Portable moments carry joint cells only for categorical-only fixed-horizon requests.

**Stage:** construction


<span id="capability-ingresses-cuped-linear-interaction-sequential"></span>

### Mixed-covariate CUPED × sequential

<span id="capability-ingresses-cuped-linear-interaction-sequential-from_unit_summary"></span>

#### Mixed-covariate CUPED × sequential — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** unfinished

**Classification:** UNFINISHED

**Codes:** `estimation.engine.cuped_linear_sequential`

**Reason:** Implement predictable multivariate sequential construction; tracked by kata 6rhs.

**Stage:** construction

<span id="capability-ingresses-cuped-linear-interaction-sequential-from_unit_panel"></span>

#### Mixed-covariate CUPED × sequential — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** unfinished

**Classification:** UNFINISHED

**Codes:** `estimation.engine.cuped_linear_sequential`

**Reason:** Implement predictable multivariate sequential construction; tracked by kata 6rhs.

**Stage:** construction

<span id="capability-ingresses-cuped-linear-interaction-sequential-from_switchback_panel"></span>

#### Mixed-covariate CUPED × sequential — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** source limited

**Classification:** SOURCE

**Codes:** `source.frame.switchback.metric`

**Reason:** Switchback is a different design from independent-unit CUPED.

**Stage:** construction

<span id="capability-ingresses-cuped-linear-interaction-sequential-from_definitions"></span>

#### Mixed-covariate CUPED × sequential — Definitions

**Recorded path:** `from_definitions`

**Status:** unfinished

**Classification:** UNFINISHED

**Codes:** `estimation.engine.cuped_linear_sequential`

**Reason:** Implement predictable multivariate sequential construction; tracked by kata 6rhs.

**Stage:** construction

<span id="capability-ingresses-cuped-linear-interaction-sequential-from_unit_day_artifact"></span>

#### Mixed-covariate CUPED × sequential — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** unfinished

**Classification:** UNFINISHED

**Codes:** `estimation.engine.cuped_linear_sequential`

**Reason:** Implement predictable multivariate sequential construction; tracked by kata 6rhs.

**Stage:** construction

<span id="capability-ingresses-cuped-linear-interaction-sequential-from_moments"></span>

#### Mixed-covariate CUPED × sequential — Portable moments

**Recorded path:** `from_moments`

**Status:** source limited

**Classification:** SOURCE

**Codes:** `facade.analysis_config.cuped_linear_moments`

**Reason:** Portable moments carry joint cells only for categorical-only fixed-horizon requests.

**Stage:** construction


<span id="capability-ingresses-cuped-linear-interaction-winsorization_fixed_value"></span>

### Mixed-covariate CUPED × winsorization fixed value

<span id="capability-ingresses-cuped-linear-interaction-winsorization_fixed_value-from_unit_summary"></span>

#### Mixed-covariate CUPED × winsorization fixed value — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

**Reason:** Fixed lower and upper value bounds are exercised on each supported unit ingress and compared with shared row and retained-scope normalization.

**Stage:** read

<span id="capability-ingresses-cuped-linear-interaction-winsorization_fixed_value-from_unit_panel"></span>

#### Mixed-covariate CUPED × winsorization fixed value — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

**Reason:** Fixed lower and upper value bounds are exercised on each supported unit ingress and compared with shared row and retained-scope normalization.

**Stage:** read

<span id="capability-ingresses-cuped-linear-interaction-winsorization_fixed_value-from_switchback_panel"></span>

#### Mixed-covariate CUPED × winsorization fixed value — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** source limited

**Classification:** SOURCE

**Codes:** `source.frame.switchback.metric`

**Reason:** Switchback is a different design from independent-unit CUPED.

**Stage:** construction

<span id="capability-ingresses-cuped-linear-interaction-winsorization_fixed_value-from_definitions"></span>

#### Mixed-covariate CUPED × winsorization fixed value — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

**Reason:** Fixed lower and upper value bounds are exercised on each supported unit ingress and compared with shared row and retained-scope normalization.

**Stage:** read

<span id="capability-ingresses-cuped-linear-interaction-winsorization_fixed_value-from_unit_day_artifact"></span>

#### Mixed-covariate CUPED × winsorization fixed value — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

**Reason:** Fixed lower and upper value bounds are exercised on each supported unit ingress and compared with shared row and retained-scope normalization.

**Stage:** read

<span id="capability-ingresses-cuped-linear-interaction-winsorization_fixed_value-from_moments"></span>

#### Mixed-covariate CUPED × winsorization fixed value — Portable moments

**Recorded path:** `from_moments`

**Status:** supported

**Stage:** construction


<span id="capability-ingresses-cuped-linear-interaction-winsorization_percentile"></span>

### Mixed-covariate CUPED × winsorization percentile

<span id="capability-ingresses-cuped-linear-interaction-winsorization_percentile-from_unit_summary"></span>

#### Mixed-covariate CUPED × winsorization percentile — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** unfinished

**Classification:** UNFINISHED

**Codes:** `estimation.engine.cuped_linear_winsorization`

**Reason:** Implement winsorized CUPED fitting and interval construction for percentile bounds.

**Stage:** construction

<span id="capability-ingresses-cuped-linear-interaction-winsorization_percentile-from_unit_panel"></span>

#### Mixed-covariate CUPED × winsorization percentile — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** unfinished

**Classification:** UNFINISHED

**Codes:** `estimation.engine.cuped_linear_winsorization`

**Reason:** Implement winsorized CUPED fitting and interval construction for percentile bounds.

**Stage:** construction

<span id="capability-ingresses-cuped-linear-interaction-winsorization_percentile-from_switchback_panel"></span>

#### Mixed-covariate CUPED × winsorization percentile — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** source limited

**Classification:** SOURCE

**Codes:** `source.frame.switchback.metric`

**Reason:** Switchback is a different design from independent-unit CUPED.

**Stage:** construction

<span id="capability-ingresses-cuped-linear-interaction-winsorization_percentile-from_definitions"></span>

#### Mixed-covariate CUPED × winsorization percentile — Definitions

**Recorded path:** `from_definitions`

**Status:** unfinished

**Classification:** UNFINISHED

**Codes:** `estimation.engine.cuped_linear_winsorization`

**Reason:** Implement winsorized CUPED fitting and interval construction for percentile bounds.

**Stage:** construction

<span id="capability-ingresses-cuped-linear-interaction-winsorization_percentile-from_unit_day_artifact"></span>

#### Mixed-covariate CUPED × winsorization percentile — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** unfinished

**Classification:** UNFINISHED

**Codes:** `estimation.engine.cuped_linear_winsorization`

**Reason:** Implement winsorized CUPED fitting and interval construction for percentile bounds.

**Stage:** construction

<span id="capability-ingresses-cuped-linear-interaction-winsorization_percentile-from_moments"></span>

#### Mixed-covariate CUPED × winsorization percentile — Portable moments

**Recorded path:** `from_moments`

**Status:** source limited

**Classification:** SOURCE

**Codes:** `facade.analysis_config.cuped_linear_moments`

**Reason:** Portable moments carry joint cells only for categorical-only fixed-horizon requests.

**Stage:** construction


<span id="capability-ingresses-cuped-linear-interaction-breakouts"></span>

### Mixed-covariate CUPED × breakouts

<span id="capability-ingresses-cuped-linear-interaction-breakouts-from_unit_summary"></span>

#### Mixed-covariate CUPED × breakouts — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** source limited

**Classification:** SOURCE

**Codes:** `evidence.breakout_unavailable`

**Reason:** A unit-summary frame has no breakout dimension.

**Stage:** read

<span id="capability-ingresses-cuped-linear-interaction-breakouts-from_unit_panel"></span>

#### Mixed-covariate CUPED × breakouts — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

**Stage:** construction

<span id="capability-ingresses-cuped-linear-interaction-breakouts-from_switchback_panel"></span>

#### Mixed-covariate CUPED × breakouts — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** source limited

**Classification:** SOURCE

**Codes:** `source.frame.switchback.metric`

**Reason:** Switchback is a different design from independent-unit CUPED.

**Stage:** construction

<span id="capability-ingresses-cuped-linear-interaction-breakouts-from_definitions"></span>

#### Mixed-covariate CUPED × breakouts — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

**Stage:** read

<span id="capability-ingresses-cuped-linear-interaction-breakouts-from_unit_day_artifact"></span>

#### Mixed-covariate CUPED × breakouts — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

**Stage:** read

<span id="capability-ingresses-cuped-linear-interaction-breakouts-from_moments"></span>

#### Mixed-covariate CUPED × breakouts — Portable moments

**Recorded path:** `from_moments`

**Status:** source limited

**Classification:** SOURCE

**Codes:** `evidence.breakout_unavailable`

**Reason:** Portable moments retain no per-unit breakout dimension.

**Stage:** read


<span id="capability-ingresses-cuped-linear-interaction-multiplicity_roles"></span>

### Mixed-covariate CUPED × multiplicity roles

<span id="capability-ingresses-cuped-linear-interaction-multiplicity_roles-from_unit_summary"></span>

#### Mixed-covariate CUPED × multiplicity roles — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** supported

**Stage:** read

<span id="capability-ingresses-cuped-linear-interaction-multiplicity_roles-from_unit_panel"></span>

#### Mixed-covariate CUPED × multiplicity roles — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** supported

**Stage:** read

<span id="capability-ingresses-cuped-linear-interaction-multiplicity_roles-from_switchback_panel"></span>

#### Mixed-covariate CUPED × multiplicity roles — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** source limited

**Classification:** SOURCE

**Codes:** `source.frame.switchback.metric`

**Reason:** Switchback is a different design from independent-unit CUPED.

**Stage:** construction

<span id="capability-ingresses-cuped-linear-interaction-multiplicity_roles-from_definitions"></span>

#### Mixed-covariate CUPED × multiplicity roles — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

**Stage:** read

<span id="capability-ingresses-cuped-linear-interaction-multiplicity_roles-from_unit_day_artifact"></span>

#### Mixed-covariate CUPED × multiplicity roles — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

**Stage:** read

<span id="capability-ingresses-cuped-linear-interaction-multiplicity_roles-from_moments"></span>

#### Mixed-covariate CUPED × multiplicity roles — Portable moments

**Recorded path:** `from_moments`

**Status:** supported

**Reason:** Format-13 fitted_multicovariate evidence supports continuous/mixed QR replay with a changed alpha; categorical-only requests retain joint-cell replay.

**Stage:** read


<span id="capability-ingresses-cuped-linear-interaction-triggered_population"></span>

### Mixed-covariate CUPED × triggered population

<span id="capability-ingresses-cuped-linear-interaction-triggered_population-from_unit_summary"></span>

#### Mixed-covariate CUPED × triggered population — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** source limited

**Classification:** SOURCE

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Dataframe ingresses do not retain the source trigger chain.

**Stage:** construction

<span id="capability-ingresses-cuped-linear-interaction-triggered_population-from_unit_panel"></span>

#### Mixed-covariate CUPED × triggered population — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** source limited

**Classification:** SOURCE

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** Dataframe ingresses do not retain the source trigger chain.

**Stage:** construction

<span id="capability-ingresses-cuped-linear-interaction-triggered_population-from_switchback_panel"></span>

#### Mixed-covariate CUPED × triggered population — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** source limited

**Classification:** SOURCE

**Codes:** `source.frame.switchback.metric`

**Reason:** Switchback is a different design from independent-unit CUPED.

**Stage:** construction

<span id="capability-ingresses-cuped-linear-interaction-triggered_population-from_definitions"></span>

#### Mixed-covariate CUPED × triggered population — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

**Stage:** read

<span id="capability-ingresses-cuped-linear-interaction-triggered_population-from_unit_day_artifact"></span>

#### Mixed-covariate CUPED × triggered population — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Status:** supported

**Stage:** read

<span id="capability-ingresses-cuped-linear-interaction-triggered_population-from_moments"></span>

#### Mixed-covariate CUPED × triggered population — Portable moments

**Recorded path:** `from_moments`

**Status:** source limited

**Classification:** SOURCE

**Codes:** `facade.analysis_config.cuped_linear_moments`

**Reason:** Portable moments carry joint cells only for categorical-only fixed-horizon requests.

**Stage:** construction
<!-- capability-table:ingresses:end -->

Triggered encouragement compliance histories are dense calendar series from
the first observed trigger-entry date through the observed uptake/observation
horizon. Uptake remains assignment-anchored; `completed_windows_only` also
requires the assignment-anchored window to fit within both the requested
as-of day and the inclusive certified uptake-feed edge. Dimensioned daily
compliance requests are explicitly refused with
`facade.analysis.daily_compliance_dimension_unsupported`.

## Triggered auxiliary readouts

Triggered segmented readouts are separate populations, not a filter silently
applied to the assigned view. A breakout view and `workflows.breakout_summaries()` emit
assigned and triggered results with population-qualified family/table identity.
Definitions are supported; a unit-day artifact is supported for breakouts only
when it contains the declared breakout-dimension extension and current trigger
evidence. Segment properties declared `as_of: pre_exposure` are resolved at
assignment, before either population's outcome anchor. The consumer parity
regression in `tests/test_analysis_trigger.py` compares population-qualified
breakout/factor summaries and public breakout rows across definitions and
unit-day artifact ingresses.

<!-- capability-table:triggered-auxiliary:start -->
Select a status for its conditions, codes and reason. A conditional cell is not a claim about every request.

| Read | Definitions | Unit-day artifact | Other entry points |
|---|---|---|---|
| breakout view / `breakout_summaries()` with a declared trigger | [supported (conditional)](#capability-triggered-auxiliary-triggered-read-breakout-from_definitions) | [supported / refused (conditional)](#capability-triggered-auxiliary-triggered-read-breakout-from_unit_day_artifact) | [refused (conditional)](#capability-triggered-auxiliary-triggered-read-breakout-other_entry_points) |
| `factor_summaries()` with a declared trigger | [supported (conditional)](#capability-triggered-auxiliary-triggered-read-factor-summaries-from_definitions) | [supported / refused (conditional)](#capability-triggered-auxiliary-triggered-read-factor-summaries-from_unit_day_artifact) | [refused (conditional)](#capability-triggered-auxiliary-triggered-read-factor-summaries-other_entry_points) |
| `prepare_dashboard()` group data per population | [supported (conditional)](#capability-triggered-auxiliary-triggered-read-dashboard-group-from_definitions) | [refused (conditional)](#capability-triggered-auxiliary-triggered-read-dashboard-group-from_unit_day_artifact) | [refused (conditional)](#capability-triggered-auxiliary-triggered-read-dashboard-group-other_entry_points) |
| `allocation_history(population=...)` | [supported (conditional)](#capability-triggered-auxiliary-triggered-read-allocation-history-from_definitions) | [supported / refused (conditional)](#capability-triggered-auxiliary-triggered-read-allocation-history-from_unit_day_artifact) | [refused (conditional)](#capability-triggered-auxiliary-triggered-read-allocation-history-other_entry_points) |
| `sitewide(population='assigned')` with a declared trigger | [supported / refused (conditional)](#capability-triggered-auxiliary-triggered-read-sitewide-assigned-from_definitions) | [supported / refused (conditional)](#capability-triggered-auxiliary-triggered-read-sitewide-assigned-from_unit_day_artifact) | [refused (conditional)](#capability-triggered-auxiliary-triggered-read-sitewide-assigned-other_entry_points) |
| `sitewide(population='triggered')` with a declared trigger | [supported / refused (conditional)](#capability-triggered-auxiliary-triggered-read-sitewide-from_definitions) | [supported / refused (conditional)](#capability-triggered-auxiliary-triggered-read-sitewide-from_unit_day_artifact) | [refused (conditional)](#capability-triggered-auxiliary-triggered-read-sitewide-other_entry_points) |
| `estimate_cate()`, `validate_cate()`, `targeting_rule()`, `select_targeting_rule()` with a declared trigger | [supported / refused (conditional)](#capability-triggered-auxiliary-triggered-read-cate-from_definitions) | [supported / refused (conditional)](#capability-triggered-auxiliary-triggered-read-cate-from_unit_day_artifact) | [not comparable (conditional)](#capability-triggered-auxiliary-triggered-read-cate-other_entry_points) |
| `health_check()` source diagnostics | [supported (conditional)](#capability-triggered-auxiliary-callable-experiment-health-from_definitions) | [supported (conditional)](#capability-triggered-auxiliary-callable-experiment-health-from_unit_day_artifact) | [supported (conditional)](#capability-triggered-auxiliary-callable-experiment-health-other_entry_points) |

<span id="capability-triggered-auxiliary-triggered-read-breakout"></span>

### breakout view / `breakout_summaries()` with a declared trigger

<span id="capability-triggered-auxiliary-triggered-read-breakout-from_definitions"></span>

#### breakout view / `breakout_summaries()` with a declared trigger — Definitions

**Recorded path:** `from_definitions`

**Condition:** definitions source includes SourceSnapshotEvidence and per-unit trigger membership

**Status:** supported

**Operation:** `run_breakout()/breakout_summaries()`

**Stage:** read

<span id="capability-triggered-auxiliary-triggered-read-breakout-from_unit_day_artifact"></span>

#### breakout view / `breakout_summaries()` with a declared trigger — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** unit-day artifact has all read-specific extensions and explicit trigger evidence

**Status:** supported

**Operation:** `run_breakout()/breakout_summaries()`

**Stage:** read

**Condition:** required extension kind is not published

**Status:** refused

**Codes:** `artifact.extension.missing`

**Reason:** Artifact lacks the operation-specific extension.

**Operation:** `run_breakout()/breakout_summaries()`

**Stage:** read

**Condition:** extension producer cannot obtain operation-specific source evidence

**Status:** refused

**Codes:** `artifact.evidence.unavailable`

**Reason:** The source cannot provide required evidence for the extension.

**Operation:** `run_breakout()/breakout_summaries()`

**Stage:** publication

<span id="capability-triggered-auxiliary-triggered-read-breakout-other_entry_points"></span>

#### breakout view / `breakout_summaries()` with a declared trigger — Other entry points

**Recorded path:** `other_entry_points`

**Condition:** from_unit_summary

**Status:** refused

**Codes:** `evidence.breakout_unavailable`

**Reason:** This source cannot supply the requested auxiliary operation.

**Operation:** `run_breakout()/breakout_summaries()`

**Stage:** read

**Condition:** from_unit_panel

**Status:** refused

**Codes:** `evidence.breakout_unavailable`

**Reason:** This source cannot supply the requested auxiliary operation.

**Operation:** `run_breakout()/breakout_summaries()`

**Stage:** read

**Condition:** from_switchback_panel

**Status:** refused

**Codes:** `facade.analysis.contrast_unavailable`

**Reason:** This source cannot supply the requested auxiliary operation.

**Operation:** `run_breakout()/breakout_summaries()`

**Stage:** read

**Condition:** from_moments

**Status:** refused

**Codes:** `evidence.breakout_unavailable`

**Reason:** This source cannot supply the requested auxiliary operation.

**Operation:** `run_breakout()/breakout_summaries()`

**Stage:** read


<span id="capability-triggered-auxiliary-triggered-read-factor-summaries"></span>

### `factor_summaries()` with a declared trigger

<span id="capability-triggered-auxiliary-triggered-read-factor-summaries-from_definitions"></span>

#### `factor_summaries()` with a declared trigger — Definitions

**Recorded path:** `from_definitions`

**Condition:** definitions source includes SourceSnapshotEvidence and per-unit trigger membership

**Status:** supported

**Operation:** `factor_summaries()`

**Stage:** read

<span id="capability-triggered-auxiliary-triggered-read-factor-summaries-from_unit_day_artifact"></span>

#### `factor_summaries()` with a declared trigger — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** unit-day artifact has all read-specific extensions and explicit trigger evidence

**Status:** supported

**Operation:** `factor_summaries()`

**Stage:** read

**Condition:** required extension kind is not published

**Status:** refused

**Codes:** `evidence.operation_unavailable`

**Reason:** Artifact lacks the operation-specific extension.

**Operation:** `factor_summaries()`

**Stage:** read

**Condition:** extension producer cannot obtain operation-specific source evidence

**Status:** refused

**Codes:** `artifact.evidence.unavailable`

**Reason:** The source cannot provide required evidence for the extension.

**Operation:** `factor_summaries()`

**Stage:** publication

<span id="capability-triggered-auxiliary-triggered-read-factor-summaries-other_entry_points"></span>

#### `factor_summaries()` with a declared trigger — Other entry points

**Recorded path:** `other_entry_points`

**Condition:** from_unit_summary

**Status:** refused

**Codes:** `evidence.operation_unavailable`

**Reason:** This source cannot supply the requested auxiliary operation.

**Operation:** `factor_summaries()`

**Stage:** read

**Condition:** from_unit_panel

**Status:** refused

**Codes:** `evidence.operation_unavailable`

**Reason:** This source cannot supply the requested auxiliary operation.

**Operation:** `factor_summaries()`

**Stage:** read

**Condition:** from_switchback_panel

**Status:** refused

**Codes:** `facade.analysis.contrast_unavailable`

**Reason:** This source cannot supply the requested auxiliary operation.

**Operation:** `factor_summaries()`

**Stage:** read

**Condition:** from_moments

**Status:** refused

**Codes:** `evidence.operation_unavailable`

**Reason:** This source cannot supply the requested auxiliary operation.

**Operation:** `factor_summaries()`

**Stage:** read


<span id="capability-triggered-auxiliary-triggered-read-dashboard-group"></span>

### `prepare_dashboard()` group data per population

<span id="capability-triggered-auxiliary-triggered-read-dashboard-group-from_definitions"></span>

#### `prepare_dashboard()` group data per population — Definitions

**Recorded path:** `from_definitions`

**Condition:** assigned and triggered populations are requested

**Status:** supported

**Operation:** `prepare_dashboard() group data`

**Stage:** read

<span id="capability-triggered-auxiliary-triggered-read-dashboard-group-from_unit_day_artifact"></span>

#### `prepare_dashboard()` group data per population — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** dashboard preparation requested from an adopted artifact

**Status:** refused

**Codes:** `dashboard.unsupported_experiment`

**Reason:** An adopted artifact records no primary metric, so the dashboard refuses it before any read.

**Operation:** `prepare_dashboard() group data`

**Stage:** read

<span id="capability-triggered-auxiliary-triggered-read-dashboard-group-other_entry_points"></span>

#### `prepare_dashboard()` group data per population — Other entry points

**Recorded path:** `other_entry_points`

**Condition:** from_unit_summary

**Status:** refused

**Codes:** `dashboard.unsupported_experiment`

**Reason:** This source cannot supply the requested auxiliary operation.

**Operation:** `prepare_dashboard() group data`

**Stage:** read

**Condition:** from_unit_panel

**Status:** refused

**Codes:** `dashboard.unsupported_experiment`

**Reason:** This source cannot supply the requested auxiliary operation.

**Operation:** `prepare_dashboard() group data`

**Stage:** read

**Condition:** from_switchback_panel

**Status:** refused

**Codes:** `facade.analysis.contrast_unavailable`

**Reason:** This source cannot supply the requested auxiliary operation.

**Operation:** `prepare_dashboard() group data`

**Stage:** read

**Condition:** from_moments

**Status:** refused

**Codes:** `dashboard.unsupported_experiment`

**Reason:** This source cannot supply the requested auxiliary operation.

**Operation:** `prepare_dashboard() group data`

**Stage:** read


<span id="capability-triggered-auxiliary-triggered-read-allocation-history"></span>

### `allocation_history(population=...)`

<span id="capability-triggered-auxiliary-triggered-read-allocation-history-from_definitions"></span>

#### `allocation_history(population=...)` — Definitions

**Recorded path:** `from_definitions`

**Condition:** assigned and triggered allocation histories are requested

**Status:** supported

**Operation:** `allocation_history(population=...)`

**Stage:** read

<span id="capability-triggered-auxiliary-triggered-read-allocation-history-from_unit_day_artifact"></span>

#### `allocation_history(population=...)` — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** artifact retains assignment_rows and assigned history is requested

**Status:** supported

**Operation:** `allocation_history(population=...)`

**Stage:** read

**Condition:** artifact has no assignment_rows identity relation

**Status:** refused

**Codes:** `evidence.assignment.identity_unavailable`

**Reason:** The artifact has reduced counts but no retained row-level assignment identity.

**Operation:** `allocation_history(population=...)`

**Stage:** read

<span id="capability-triggered-auxiliary-triggered-read-allocation-history-other_entry_points"></span>

#### `allocation_history(population=...)` — Other entry points

**Recorded path:** `other_entry_points`

**Condition:** unit summary

**Status:** refused

**Codes:** `evidence.assignment.identity_unavailable`

**Reason:** This source retains only reduced assignment data, not row-level assignment identity.

**Operation:** `allocation_history(population=...)`

**Stage:** read

**Condition:** unit panel

**Status:** refused

**Codes:** `evidence.assignment.identity_unavailable`

**Reason:** This source retains only reduced assignment data, not row-level assignment identity.

**Operation:** `allocation_history(population=...)`

**Stage:** read

**Condition:** switchback panel

**Status:** refused

**Codes:** `evidence.assignment.identity_unavailable`

**Reason:** This source retains only reduced assignment data, not row-level assignment identity.

**Operation:** `allocation_history(population=...)`

**Stage:** read

**Condition:** portable moments

**Status:** refused

**Codes:** `evidence.operation_unavailable`

**Reason:** A moments experiment does not retain raw allocation events.

**Operation:** `allocation_history(population=...)`

**Stage:** read


<span id="capability-triggered-auxiliary-triggered-read-sitewide-assigned"></span>

### `sitewide(population='assigned')` with a declared trigger

<span id="capability-triggered-auxiliary-triggered-read-sitewide-assigned-from_definitions"></span>

#### `sitewide(population='assigned')` with a declared trigger — Definitions

**Recorded path:** `from_definitions`

**Condition:** sitewide requested with the default assigned population on an experiment that declares a trigger

**Status:** supported

**Operation:** `sitewide(population=...)`

**Stage:** read

**Condition:** sitewide requested with the default assigned population on an observational design

**Status:** refused

**Codes:** `facade.analysis.sitewide_requires_randomized`

**Reason:** Sitewide impact requires randomized identification.

**Operation:** `sitewide(population=...)`

**Stage:** read

**Condition:** sitewide requested with a windowed, non-additive, or conversion metric

**Status:** refused

**Codes:** `facade.analysis.sitewide_metric_unaligned`

**Reason:** The per-unit outcome cannot align with the additive calendar-window site total.

**Operation:** `sitewide(population=...)`

**Stage:** read

**Condition:** sitewide requested with a conversion metric for either population

**Status:** refused

**Codes:** `facade.analysis.sitewide_metric_unaligned`

**Reason:** Converter indicators are not additive; declare a count metric on the conversion fact.

**Operation:** `sitewide(population=...)`

**Stage:** read

<span id="capability-triggered-auxiliary-triggered-read-sitewide-assigned-from_unit_day_artifact"></span>

#### `sitewide(population='assigned')` with a declared trigger — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** sitewide requested with the default assigned population and a site-volume extension

**Status:** supported

**Operation:** `sitewide(population=...)`

**Stage:** read

**Condition:** sitewide requested with a windowed, non-additive, or conversion metric

**Status:** refused

**Codes:** `facade.analysis.sitewide_metric_unaligned`

**Reason:** Per-unit outcomes cannot align with the additive calendar-window site total.

**Operation:** `sitewide(population=...)`

**Stage:** read

**Condition:** sitewide requested with a conversion metric for either population

**Status:** refused

**Codes:** `facade.analysis.sitewide_metric_unaligned`

**Reason:** Converter indicators are not additive; declare a count metric on the conversion fact.

**Operation:** `sitewide(population=...)`

**Stage:** read

<span id="capability-triggered-auxiliary-triggered-read-sitewide-assigned-other_entry_points"></span>

#### `sitewide(population='assigned')` with a declared trigger — Other entry points

**Recorded path:** `other_entry_points`

**Condition:** from_unit_summary

**Status:** refused

**Codes:** `evidence.operation_unavailable`

**Reason:** This source cannot supply the requested auxiliary operation.

**Operation:** `sitewide(population=...)`

**Stage:** read

**Condition:** from_unit_panel

**Status:** refused

**Codes:** `evidence.operation_unavailable`

**Reason:** This source cannot supply the requested auxiliary operation.

**Operation:** `sitewide(population=...)`

**Stage:** read

**Condition:** from_switchback_panel

**Status:** refused

**Codes:** `facade.analysis.contrast_unavailable`

**Reason:** This source cannot supply the requested auxiliary operation.

**Operation:** `sitewide(population=...)`

**Stage:** read

**Condition:** from_moments

**Status:** refused

**Codes:** `evidence.operation_unavailable`

**Reason:** This source cannot supply the requested auxiliary operation.

**Operation:** `sitewide(population=...)`

**Stage:** read


<span id="capability-triggered-auxiliary-triggered-read-sitewide"></span>

### `sitewide(population='triggered')` with a declared trigger

<span id="capability-triggered-auxiliary-triggered-read-sitewide-from_definitions"></span>

#### `sitewide(population='triggered')` with a declared trigger — Definitions

**Recorded path:** `from_definitions`

**Condition:** sitewide requested with population='triggered' on an experiment that declares a trigger

**Status:** supported

**Operation:** `sitewide(population=...)`

**Stage:** read

**Condition:** sitewide requested with a windowed, non-additive, or conversion metric

**Status:** refused

**Codes:** `facade.analysis.sitewide_metric_unaligned`

**Reason:** The per-unit outcome cannot align with the additive calendar-window site total.

**Operation:** `sitewide(population=...)`

**Stage:** read

**Condition:** sitewide requested with a conversion metric for either population

**Status:** refused

**Codes:** `facade.analysis.sitewide_metric_unaligned`

**Reason:** Converter indicators are not additive; declare a count metric on the conversion fact.

**Operation:** `sitewide(population=...)`

**Stage:** read

<span id="capability-triggered-auxiliary-triggered-read-sitewide-from_unit_day_artifact"></span>

#### `sitewide(population='triggered')` with a declared trigger — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** sitewide requested with population='triggered' and published trigger/site-volume evidence

**Status:** supported

**Operation:** `sitewide(population=...)`

**Stage:** read

**Condition:** sitewide requested with a windowed, non-additive, or conversion metric

**Status:** refused

**Codes:** `facade.analysis.sitewide_metric_unaligned`

**Reason:** Per-unit outcomes cannot align with the additive calendar-window site total.

**Operation:** `sitewide(population=...)`

**Stage:** read

**Condition:** sitewide requested with a conversion metric for either population

**Status:** refused

**Codes:** `facade.analysis.sitewide_metric_unaligned`

**Reason:** Converter indicators are not additive; declare a count metric on the conversion fact.

**Operation:** `sitewide(population=...)`

**Stage:** read

<span id="capability-triggered-auxiliary-triggered-read-sitewide-other_entry_points"></span>

#### `sitewide(population='triggered')` with a declared trigger — Other entry points

**Recorded path:** `other_entry_points`

**Condition:** unit summary

**Status:** refused

**Codes:** `evidence.operation_unavailable`

**Reason:** This source cannot provide triggered sitewide evidence.

**Operation:** `sitewide(population=...)`

**Stage:** read

**Condition:** unit panel

**Status:** refused

**Codes:** `evidence.operation_unavailable`

**Reason:** This source cannot provide triggered sitewide evidence.

**Operation:** `sitewide(population=...)`

**Stage:** read

**Condition:** switchback panel

**Status:** refused

**Codes:** `facade.analysis.contrast_unavailable`

**Reason:** This source cannot provide triggered sitewide evidence.

**Operation:** `sitewide(population=...)`

**Stage:** read

**Condition:** portable moments

**Status:** refused

**Codes:** `evidence.operation_unavailable`

**Reason:** This source cannot provide triggered sitewide evidence.

**Operation:** `sitewide(population=...)`

**Stage:** read


<span id="capability-triggered-auxiliary-triggered-read-cate"></span>

### `estimate_cate()`, `validate_cate()`, `targeting_rule()`, `select_targeting_rule()` with a declared trigger

<span id="capability-triggered-auxiliary-triggered-read-cate-from_definitions"></span>

#### `estimate_cate()`, `validate_cate()`, `targeting_rule()`, `select_targeting_rule()` with a declared trigger — Definitions

**Recorded path:** `from_definitions`

**Condition:** randomized definitions source includes per-unit trigger and covariate evidence

**Status:** supported

**Operation:** `estimate_cate()/validate_cate()/targeting_rule()/select_targeting_rule()`

**Stage:** read

**Condition:** validate_cate() on an observational definitions source with complete covariate and trigger evidence

**Status:** supported

**Operation:** `validate_cate()`

**Stage:** read

**Condition:** targeting_rule() on an observational definitions source with complete covariate and trigger evidence

**Status:** supported

**Operation:** `targeting_rule()`

**Stage:** read

**Condition:** select_targeting_rule() on an observational definitions source with complete covariate and trigger evidence

**Status:** supported

**Operation:** `select_targeting_rule()`

**Stage:** read

**Condition:** estimate_cate() on an observational definitions source with complete covariate and trigger evidence

**Status:** refused

**Codes:** `cate.identification.randomized_only`

**Reason:** Published covariates do not establish randomized assignment or an observational CATE estimator.

**Operation:** `estimate_cate()`

**Stage:** read

<span id="capability-triggered-auxiliary-triggered-read-cate-from_unit_day_artifact"></span>

#### `estimate_cate()`, `validate_cate()`, `targeting_rule()`, `select_targeting_rule()` with a declared trigger — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** randomized estimate_cate(interact=[]) with explicit trigger extensions

**Status:** supported

**Operation:** `estimate_cate(interact=[])`

**Stage:** read

**Condition:** validate_cate() on an observational artifact source with complete covariate and trigger evidence

**Status:** supported

**Operation:** `validate_cate()`

**Stage:** read

**Condition:** targeting_rule() on an observational artifact source with complete covariate and trigger evidence

**Status:** supported

**Operation:** `targeting_rule()`

**Stage:** read

**Condition:** select_targeting_rule() on an observational artifact source with complete covariate and trigger evidence

**Status:** supported

**Operation:** `select_targeting_rule()`

**Stage:** read

**Condition:** estimate_cate() on an observational artifact with complete covariate and trigger evidence

**Status:** refused

**Codes:** `cate.identification.randomized_only`

**Reason:** Published covariates do not establish randomized assignment or an observational CATE estimator.

**Operation:** `estimate_cate()`

**Stage:** read

**Condition:** required extension kind is not published

**Status:** refused

**Codes:** `artifact.extension.missing`

**Reason:** Artifact lacks the operation-specific extension.

**Operation:** `estimate_cate()/validate_cate()/targeting_rule()/select_targeting_rule()`

**Stage:** read

**Condition:** extension producer cannot obtain operation-specific source evidence

**Status:** refused

**Codes:** `artifact.evidence.unavailable`

**Reason:** The source cannot provide required evidence for the extension.

**Operation:** `estimate_cate()/validate_cate()/targeting_rule()/select_targeting_rule()`

**Stage:** publication

<span id="capability-triggered-auxiliary-triggered-read-cate-other_entry_points"></span>

#### `estimate_cate()`, `validate_cate()`, `targeting_rule()`, `select_targeting_rule()` with a declared trigger — Other entry points

**Recorded path:** `other_entry_points`

**Condition:** from_unit_summary

**Status:** not comparable

**Reason:** The frame/moments/switchback entry points carry no declared trigger membership for this read.

**Operation:** `estimate_cate()/validate_cate()/targeting_rule()/select_targeting_rule()`

**Stage:** construction

**Condition:** from_unit_panel

**Status:** not comparable

**Reason:** The frame/moments/switchback entry points carry no declared trigger membership for this read.

**Operation:** `estimate_cate()/validate_cate()/targeting_rule()/select_targeting_rule()`

**Stage:** construction

**Condition:** from_switchback_panel

**Status:** not comparable

**Reason:** The frame/moments/switchback entry points carry no declared trigger membership for this read.

**Operation:** `estimate_cate()/validate_cate()/targeting_rule()/select_targeting_rule()`

**Stage:** construction

**Condition:** from_moments

**Status:** not comparable

**Reason:** The frame/moments/switchback entry points carry no declared trigger membership for this read.

**Operation:** `estimate_cate()/validate_cate()/targeting_rule()/select_targeting_rule()`

**Stage:** construction


<span id="capability-triggered-auxiliary-callable-experiment-health"></span>

### `health_check()` source diagnostics

<span id="capability-triggered-auxiliary-callable-experiment-health-from_definitions"></span>

#### `health_check()` source diagnostics — Definitions

**Recorded path:** `from_definitions`

**Condition:** declared trigger with retained native source evidence

**Status:** supported

**Operation:** `health_check()`

**Stage:** read

**Note:** matched native definitions health parity test uses explicit source cutoff and feed watermark

<span id="capability-triggered-auxiliary-callable-experiment-health-from_unit_day_artifact"></span>

#### `health_check()` source diagnostics — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** published trigger population, assignment counts and metric-stat extensions

**Status:** supported

**Operation:** `health_check()`

**Stage:** read

**Condition:** artifact omits trigger membership and rate extensions

**Status:** supported

**Operation:** `health_check()`

**Stage:** read

**Note:** matched artifact health parity requires trigger_population, assignment_counts and trigger_measure_stats

<span id="capability-triggered-auxiliary-callable-experiment-health-other_entry_points"></span>

#### `health_check()` source diagnostics — Other entry points

**Recorded path:** `other_entry_points`

**Condition:** summary-backed analysis with declared randomized allocation

**Status:** supported

**Operation:** `health_check()`

**Stage:** read

**Condition:** unit panel has no trigger declaration or raw trigger membership

**Status:** supported

**Operation:** `health_check()`

**Stage:** read

**Condition:** switchback assignment law cannot be tested as a unit-arm allocation

**Status:** supported

**Operation:** `health_check()`

**Stage:** read

**Condition:** moments replay retains no declared trigger membership

**Status:** supported

**Operation:** `health_check()`

**Stage:** read

**Note:** unit summary, unit panel, switchback and moments do not retain trigger population evidence
<!-- capability-table:triggered-auxiliary:end -->

Triggered sequential breakout inference remains explicitly unsupported
(`sequential.route.unsupported`): relational segment capture has no immutable
property contract for either population. It does not fall back to assigned
segments.

For `from_definitions`, pinning preserves dimensions joined by freshness-bearing
fact sources. Qualifying non-enrolled events can establish that later data has
arrived without entering the enrolled outcome population. In a `from_frame` panel,
natural day labels such as `d9` and `d10` use numeric order when inferring
maturity, matching the same explicitly declared observation bound.
Windowed ratios use the earlier numerator/denominator observation bound for
totals and completed as-of cohorts. An entirely null component under
`missing="zero"` retains the panel-extent fallback; an explicitly supplied
`observation_end` remains authoritative.

For clustered triggered planning, `from_definitions` and `from_unit_day_artifact`
retain both populations: assigned mean cluster size determines recruitment;
contributing-cluster participation, analyzed size CV, and analyzed-score ICC
determine the planning approximation. The artifact needs `cluster_identity`
as well as the trigger and assignment-count extensions. These pilot estimates
do not certify future participation or finite-sample coverage. Other
constructors cannot supply the declared trigger membership identified in the table.
Both clustered trigger routes refuse a pilot whose metric rows omit
trigger-eligible control units (SOURCE,
`analysis.planning_baseline.trigger_metric_population_unavailable`).
Missing rows can reflect unfinished windows or outcomes undefined even after
full follow-up, such as `avg_event` for a unit with no events. Complete
unfinished windows; for structurally undefined outcomes, choose a metric
defined for every eligible unit. The adapter does not silently substitute a
complete-case population for the declared trigger population.

For trigger-aware planning through `from_definitions` and an adopted unit-day
artifact, the consumer regression covers mean and ratio CUPED moments,
an unadjusted median baseline, and the observed trigger rate. Each uses the
first eligible trigger as the outcome-window anchor; mean and ratio CUPED
retain covariates measured before each unit's own assignment, even with
staggered assignment dates. Enrollment counts remain assignment-based, while
the analyzed-trigger count remains distinct in baseline-derived sample-size
results. These claims apply to those two warehouse ingresses, not to the
summary, panel, switchback, or portable-moment routes marked refused above.

Scalar CUPED readouts require moments for each effective method's covariate
and missing-value policy. A source adapter without that capability refuses
with `readout.scalar_method_evidence`, naming the route, before reading rows.
Portable single-family moments retain their established scalar covariate;
they do not acquire a different named family from unit data.

Mean/Ratio CUPED on the unit panel resolves the declared covariate the same
way observational adjustment already does: through the per-unit value
`unit_frame()`/`moments(grain="total")` join in, constant across a unit's
own rows, refused by name (`frame.frame_panel.unit_covariate_varies`) when
it genuinely varies. A windowed or retention metric still refuses a CUPED
covariate (CONSTRUCTION, `frame.validation.from_unit_panel`): those metrics
have no per-unit collapse for that value to attach to. For a windowed
metric, compute each unit's windowed value upstream and declare it as an
unwindowed metric on a `from_frame` summary; no frame source serves a per-unit
retention value, so remove the covariate to run retention without CUPED.
The switchback panel supports neither sequential inference
nor CUPED; its contrasts are fixed-horizon. "Asymptotic route" means CUPED
is admitted under `InferenceSpec(kind="asymptotic_mean")`: fitted coefficients
use retained joint moments, whose confidence sequence accounts for their
estimation; pre-period coefficients use the scalar-mean adjustment. The exact
Bernoulli e-process route (`AlwaysValid`) admits neither form of CUPED.
"Checkpoint replay" means `from_moments` cannot start a sequential process,
because it holds no per-unit source to register against, but reproduces an
exported checkpoint exactly. On the warehouse the covariate for a ratio metric
is the numerator's pre-period total; a separate denominator covariate is not
expressible, and the CUPED guide says why. An encouragement design's continuous
ITT and Bernoulli-uptake compliance, monitored jointly, compose into one
sequential family automatically (no explicit registration to declare); the
family reports one shared `family_guarantee`, equal to the weakest regime any
member carries -- with an asymptotic scalar-mean ITT cell in the family, that
value is `asymptotic_sequential` on every selected row, including the exact
Bernoulli compliance cell's own row, not "exact for one cell, asymptotic for
the other." The `from_switchback_frame` encouragement and observational cells are a structural
limit of that constructor, not a gap. It has no `design=` parameter, so a `design:` declaration
cannot be written and Python raises `TypeError` (no `.code`). It accepts only a
switchback-shaped frame and `identification: Randomized`; an `Encouragement`/`Observational`
object passed as `identification=` (as the parity matrix's observational cells do) is refused
with `source.frame.switchback.identification` (reason `unsupported_identification`), as is a
quantile metric (`source.frame.switchback.metric`, reason `unsupported_metric_type`).

Parity is asserted, not assumed: identical per-unit data through each path must
reproduce the same adjusted moments and interval, and the same retained
sequential state byte for byte. That check runs against DuckDB in the ordinary
suite and against a live PostgreSQL in the warehouse gate. For the two binary
rows it reads one per-unit dataset through the unit panel (under the same
two-day window the definitions declare), the definitions reader, the unit-day
artifact reader and a replayed portable checkpoint, and requires each to retain
the unit-summary oracle's state, registration and interval exactly.

The informative-prior row uses the existing approximate Normal likelihood
on log lift, including for conversion metrics; it is not a binomial
posterior or a finite-sample sampling guarantee. The three enumerated
prior cases compare posterior bounds, probabilities, metadata and row
identity across all five applicable constructors on DuckDB and PostgreSQL.
See [Priors and Bayesian decisions](../guides/priors-and-decisions.md) for
binary boundary behavior and the distinction from prior-free sampling.

Binary sequential monitoring (either automatic route) composes as follows, measured on
a `from_frame` summary and pinned per axis. Multiplicity: every in-family secondary on the
metric/arm axis -- Bernoulli, scalar-mean, or a mix of both -- is registered into the
family at a nominal per-cell allocation and selected by e-BH at the plan's `q`
(supported); a selected member's interval is reinverted at `min(q * R / m, nominal_alpha)`,
and the family reports one shared `family_guarantee` (`finite_sample` only when every
member is exact, `asymptotic_sequential` when any member is asymptotic). This is
unrelated to the registered breakout/segment axis, whose own correction is declared per
registration: a continuous (scalar-mean) breakout registers `correction="bonferroni"` and
keeps fixed per-cell Bonferroni (FWER, no reselection); a discrete (Bernoulli) breakout
registers `correction="bh"` and is selected by the same e-BH-at-`q` family as an ordinary
metric/arm secondary, reinverted the same way. This is also unrelated to guardrails, which test at the
full plan `alpha` outside every family. CUPED: a pre-period coefficient and centre declared through
`InferenceSpec.adjustments` is supported on the asymptotic route; an in-experiment
coefficient is refused on the exact route (`sequential.transform.unpredictable`) because
it re-weights past increments with information unavailable when they were revealed, a
construction this package does not have; the asymptotic route retains the joint moments
instead. Clustered assignment: refused before any observation is read
(`sequential.route.unsupported`); no cluster-robust sequential boundary is built.
Winsorization: it does not apply to a 0/1 outcome, so a binary metric refuses a
percentile or fixed clip at declaration (`frame.metric.winsorization_applies_type`); on a
mean metric a fixed clip is supported and a percentile clip refuses
(`sequential.transform.unpredictable`) because its threshold depends on accumulated data.
Quantile metrics: refused on the exact route (`definition.inference.always_valid_metric_type`)
because the Bernoulli law does not describe them; use fixed-horizon inference (valid for one planned analysis, not repeated looks).
Breakouts: `InferenceSpec.segments` fixes one string-valued dimension and its
allowed levels before outcomes on `from_frame` summaries and panels;
an explicit registration is optional. Empty levels stay in the roster.
Asymptotic segment membership must be determined before assignment and uses
the existing fixed-roster Bonferroni construction, not BH. Segment labels are
canonical strings on both frame routes: `true`/`false` for Booleans and
`__null__` for missing values, so summary and panel sources label the same
units identically. A checkpoint recorded under an earlier raw spelling (`True`,
`None`) replays unchanged but cannot be continued by a canonically labelled
source (COMBINATION -- earlier recorded prefix with a different roster
spelling; `sequential.continuation.rewrite`). This is a stored-state boundary, not an
inherent limitation; recovery is in [Persisted formats](../api.md#persisted-formats).
`from_definitions` and `from_unit_day_artifact` still refuse segmented capture
(`sequential.route.unsupported`). A moments cube can replay retained segment
state but has no breakout catalog, so a breakout view refuses
(`evidence.breakout_unavailable`). Switchback: refused before any frame is read
(`source.frame.switchback.plan`, reason `unsupported_inference`) for every sequential kind,
automatic or explicit; the switchback contrast has no sequential construction.
Continuous intent-to-treat under an encouragement design registers under the public asymptotic
mean law exactly as a randomized design does (`sequential.route.unsupported` still names the
Bernoulli route on any other mechanism).

### Stored-state compatibility by entry point

These rows describe what happens when stored state meets the current reader, or the artifact store's recovery path. Recovery steps are in [Persisted formats](../api.md#persisted-formats).

<!-- capability-table:stored-state:start -->
Select a status for its conditions, codes and reason. A conditional cell is not a claim about every request.

| Capability | Unit summary | Unit panel | Switchback panel | Definitions | Unit-day artifact | Portable moments |
|---|---|---|---|---|---|---|
| Registered process committed without a triggered chain continued under a declared trigger | [refused (conditional)](#capability-stored-state-triggered-sequential-continuation-from_unit_summary) | [refused (conditional)](#capability-stored-state-triggered-sequential-continuation-from_unit_panel) | [refused (conditional)](#capability-stored-state-triggered-sequential-continuation-from_switchback_panel) | [unavailable / refused (conditional)](#capability-stored-state-triggered-sequential-continuation-from_definitions) | [unavailable / refused (conditional)](#capability-stored-state-triggered-sequential-continuation-from_unit_day_artifact) | [unavailable / refused (conditional)](#capability-stored-state-triggered-sequential-continuation-from_moments) |
| Continuing a checkpoint recorded under earlier raw Boolean/null spellings with canonically labelled units | [refused / supported (conditional)](#capability-stored-state-stored-boolean-null-continuation-from_unit_summary) | [refused (conditional)](#capability-stored-state-stored-boolean-null-continuation-from_unit_panel) | [refused](#capability-stored-state-stored-boolean-null-continuation-from_switchback_panel) | [refused (conditional)](#capability-stored-state-stored-boolean-null-continuation-from_definitions) | [refused (conditional)](#capability-stored-state-stored-boolean-null-continuation-from_unit_day_artifact) | [replay only (conditional)](#capability-stored-state-stored-boolean-null-continuation-from_moments) |
| Artifact context format 2 (hash-only source recipes; no source SQL) | [not comparable](#capability-stored-state-artifact-context-formats-from_unit_summary) | [not comparable](#capability-stored-state-artifact-context-formats-from_unit_panel) | [not comparable](#capability-stored-state-artifact-context-formats-from_switchback_panel) | [supported](#capability-stored-state-artifact-context-formats-from_definitions) | [supported / refused (conditional)](#capability-stored-state-artifact-context-formats-from_unit_day_artifact) | [not comparable](#capability-stored-state-artifact-context-formats-from_moments) |
| Experiment window days (start, end and observation-horizon days) follow the day boundary: a naive value is the wall clock at the boundary, an aware value is converted | [not comparable (conditional)](#capability-stored-state-experiment-window-day-boundary-from_unit_summary) | [not comparable (conditional)](#capability-stored-state-experiment-window-day-boundary-from_unit_panel) | [not comparable (conditional)](#capability-stored-state-experiment-window-day-boundary-from_switchback_panel) | [supported / refused (conditional)](#capability-stored-state-experiment-window-day-boundary-from_definitions) | [supported / refused (conditional)](#capability-stored-state-experiment-window-day-boundary-from_unit_day_artifact) | [not comparable (conditional)](#capability-stored-state-experiment-window-day-boundary-from_moments) |
| Publication abort after the manifest insert began invalidates the generation (tombstone first, then relation erase; if the tombstone cannot be written the relations are kept and named, and a fresh process hides the generation before they are dropped) | [not comparable (conditional)](#capability-stored-state-artifact-publication-abort-from_unit_summary) | [not comparable (conditional)](#capability-stored-state-artifact-publication-abort-from_unit_panel) | [not comparable (conditional)](#capability-stored-state-artifact-publication-abort-from_switchback_panel) | [supported (conditional)](#capability-stored-state-artifact-publication-abort-from_definitions) | [refused (conditional)](#capability-stored-state-artifact-publication-abort-from_unit_day_artifact) | [not comparable (conditional)](#capability-stored-state-artifact-publication-abort-from_moments) |

<span id="capability-stored-state-triggered-sequential-continuation"></span>

### Registered process committed without a triggered chain continued under a declared trigger

<span id="capability-stored-state-triggered-sequential-continuation-from_unit_summary"></span>

#### Registered process committed without a triggered chain continued under a declared trigger — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** frame source or moments without a retained triggered checkpoint

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** This source carries no trigger declaration or trigger membership.

**Operation:** `run(population='triggered')`

**Stage:** run

<span id="capability-stored-state-triggered-sequential-continuation-from_unit_panel"></span>

#### Registered process committed without a triggered chain continued under a declared trigger — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** frame source or moments without a retained triggered checkpoint

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** This source carries no trigger declaration or trigger membership.

**Operation:** `run(population='triggered')`

**Stage:** run

<span id="capability-stored-state-triggered-sequential-continuation-from_switchback_panel"></span>

#### Registered process committed without a triggered chain continued under a declared trigger — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** frame source or moments without a retained triggered checkpoint

**Status:** refused

**Codes:** `facade.analysis.trigger_unsupported`

**Reason:** This source carries no trigger declaration or trigger membership.

**Operation:** `run(population='triggered')`

**Stage:** run

<span id="capability-stored-state-triggered-sequential-continuation-from_definitions"></span>

#### Registered process committed without a triggered chain continued under a declared trigger — Definitions

**Recorded path:** `from_definitions`

**Condition:** assigned-only commitment: read under a declared trigger

**Status:** unavailable

**Codes:** `readout.cell.unsupported_request`

**Reason:** triggered_chain_uncommitted

**Operation:** `run()`

**Stage:** run

**Condition:** assigned-only commitment: retroactive backfill

**Status:** refused

**Codes:** `sequential.continuation.legacy`

**Reason:** A committed assigned-only process cannot acquire a triggered chain retroactively.

**Operation:** `SequentialSnapshot.verify_parent()`

**Stage:** verify_parent

**Note:** continues the assigned chain only

<span id="capability-stored-state-triggered-sequential-continuation-from_unit_day_artifact"></span>

#### Registered process committed without a triggered chain continued under a declared trigger — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** assigned-only commitment: read under a declared trigger

**Status:** unavailable

**Codes:** `readout.cell.unsupported_request`

**Reason:** triggered_chain_uncommitted

**Operation:** `run()`

**Stage:** run

**Condition:** assigned-only commitment: retroactive backfill

**Status:** refused

**Codes:** `sequential.continuation.legacy`

**Reason:** A committed assigned-only process cannot acquire a triggered chain retroactively.

**Operation:** `SequentialSnapshot.verify_parent()`

**Stage:** verify_parent

**Note:** continues the assigned chain only

<span id="capability-stored-state-triggered-sequential-continuation-from_moments"></span>

#### Registered process committed without a triggered chain continued under a declared trigger — Portable moments

**Recorded path:** `from_moments`

**Condition:** assigned-only commitment: read under a declared trigger

**Status:** unavailable

**Codes:** `readout.cell.unsupported_request`

**Reason:** triggered_chain_uncommitted

**Operation:** `run()`

**Stage:** run

**Condition:** assigned-only commitment: retroactive backfill

**Status:** refused

**Codes:** `sequential.continuation.legacy`

**Reason:** A committed assigned-only process cannot acquire a triggered chain retroactively.

**Operation:** `SequentialSnapshot.verify_parent()`

**Stage:** verify_parent

**Note:** replays the retained assigned-only commitment


<span id="capability-stored-state-stored-boolean-null-continuation"></span>

### Continuing a checkpoint recorded under earlier raw Boolean/null spellings with canonically labelled units

<span id="capability-stored-state-stored-boolean-null-continuation-from_unit_summary"></span>

#### Continuing a checkpoint recorded under earlier raw Boolean/null spellings with canonically labelled units — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** incoming typed boolean/null levels are canonically relabelled

**Status:** refused

**Codes:** `sequential.continuation.rewrite`

**Reason:** Canonical relabeling changes the frozen continuation identity.

**Operation:** `capture_sequential(previous=...)`

**Stage:** read

**Condition:** incoming rows preserve historical raw spellings

**Status:** supported

**Operation:** `capture_sequential(previous=...)`

**Stage:** read

**Note:** the checkpoint itself replays unchanged

<span id="capability-stored-state-stored-boolean-null-continuation-from_unit_panel"></span>

#### Continuing a checkpoint recorded under earlier raw Boolean/null spellings with canonically labelled units — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** canonical labels attempt to continue a legacy unbounded-window prefix at an explicit panel horizon

**Status:** refused

**Codes:** `sequential.route.unsupported`

**Reason:** Panel capture requires bounded metric windows covered by a common joint-reveal window; continue this legacy finalized prefix through unit summary.

**Operation:** `capture_sequential(previous=...)`

**Stage:** read

<span id="capability-stored-state-stored-boolean-null-continuation-from_switchback_panel"></span>

#### Continuing a checkpoint recorded under earlier raw Boolean/null spellings with canonically labelled units — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** refused

**Classification:** SOURCE

**Codes:** `source.frame.switchback.plan`

**Reason:** switchback contrasts support fixed inference only

**Stage:** declaration

<span id="capability-stored-state-stored-boolean-null-continuation-from_definitions"></span>

#### Continuing a checkpoint recorded under earlier raw Boolean/null spellings with canonically labelled units — Definitions

**Recorded path:** `from_definitions`

**Condition:** exact Bernoulli law

**Status:** refused

**Classification:** SOURCE

**Codes:** `sequential.route.unsupported`

**Reason:** given the same segmented registration as from_unit_summary, a relational source (validate_relational_capture, shared by native_source.py and artifact_reader.py) refuses that segmented sequential roster outright -- 'relational segment capture needs a registered immutable property relation; use finalized joint records or frame segments'.

**Condition:** asymptotic mean law

**Status:** refused

**Classification:** SOURCE

**Codes:** `sequential.route.unsupported`

**Reason:** given the same segmented registration as from_unit_summary, a relational source (validate_relational_capture, shared by native_source.py and artifact_reader.py) refuses that segmented sequential roster outright -- 'relational segment capture needs a registered immutable property relation; use finalized joint records or frame segments'.

<span id="capability-stored-state-stored-boolean-null-continuation-from_unit_day_artifact"></span>

#### Continuing a checkpoint recorded under earlier raw Boolean/null spellings with canonically labelled units — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** exact Bernoulli law

**Status:** refused

**Classification:** SOURCE

**Codes:** `sequential.route.unsupported`

**Reason:** publish/adopt always re-derives an automatic (unsegmented) registration from the experiment's declared plan -- a caller-built segmented registration never reaches it -- and the relational capture check above would refuse it regardless.

**Condition:** asymptotic mean law

**Status:** refused

**Classification:** SOURCE

**Codes:** `sequential.route.unsupported`

**Reason:** publish/adopt always re-derives an automatic (unsegmented) registration from the experiment's declared plan -- a caller-built segmented registration never reaches it -- and the relational capture check above would refuse it regardless.

<span id="capability-stored-state-stored-boolean-null-continuation-from_moments"></span>

#### Continuing a checkpoint recorded under earlier raw Boolean/null spellings with canonically labelled units — Portable moments

**Recorded path:** `from_moments`

**Condition:** frozen checkpoint contains historical raw True/False/None spellings

**Status:** replay only

**Reason:** Historical checkpoint replay is not proof of a live-source continuation.

**Operation:** `sequential_snapshot()`

**Stage:** read


<span id="capability-stored-state-artifact-context-formats"></span>

### Artifact context format 2 (hash-only source recipes; no source SQL)

<span id="capability-stored-state-artifact-context-formats-from_unit_summary"></span>

#### Artifact context format 2 (hash-only source recipes; no source SQL) — Unit summary

**Recorded path:** `from_unit_summary`

**Status:** not comparable

**Classification:** SOURCE

**Reason:** no warehouse artifact context is carried by a frame

<span id="capability-stored-state-artifact-context-formats-from_unit_panel"></span>

#### Artifact context format 2 (hash-only source recipes; no source SQL) — Unit panel

**Recorded path:** `from_unit_panel`

**Status:** not comparable

**Classification:** SOURCE

**Reason:** no warehouse artifact context is carried by a frame

<span id="capability-stored-state-artifact-context-formats-from_switchback_panel"></span>

#### Artifact context format 2 (hash-only source recipes; no source SQL) — Switchback panel

**Recorded path:** `from_switchback_panel`

**Status:** not comparable

**Classification:** SOURCE

**Reason:** no warehouse artifact context is carried by a switchback

<span id="capability-stored-state-artifact-context-formats-from_definitions"></span>

#### Artifact context format 2 (hash-only source recipes; no source SQL) — Definitions

**Recorded path:** `from_definitions`

**Status:** supported

**Note:** publishes format 2

<span id="capability-stored-state-artifact-context-formats-from_unit_day_artifact"></span>

#### Artifact context format 2 (hash-only source recipes; no source SQL) — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** format 2

**Status:** supported

**Condition:** format 1

**Status:** refused

**Classification:** COMBINATION

**Codes:** `artifact.format.unsupported`

**Reason:** the current reader refuses a format-1 context

**Note:** a format-1 context must be republished from trusted definitions

<span id="capability-stored-state-artifact-context-formats-from_moments"></span>

#### Artifact context format 2 (hash-only source recipes; no source SQL) — Portable moments

**Recorded path:** `from_moments`

**Status:** not comparable

**Classification:** SOURCE

**Reason:** portable moments carry no warehouse artifact context


<span id="capability-stored-state-experiment-window-day-boundary"></span>

### Experiment window days (start, end and observation-horizon days) follow the day boundary: a naive value is the wall clock at the boundary, an aware value is converted

<span id="capability-stored-state-experiment-window-day-boundary-from_unit_summary"></span>

#### Experiment window days (start, end and observation-horizon days) follow the day boundary: a naive value is the wall clock at the boundary, an aware value is converted — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** zulu and offset spellings of one instant

**Status:** not comparable

**Classification:** SOURCE

**Reason:** from_unit_summary has no experiment-window lever -- the day boundary is applied by the caller pre-bucketing ds/exposure_date (see tests/test_frame_window_parity.py), so the window spelling is not an input.

**Condition:** two offset spellings of one instant

**Status:** not comparable

**Classification:** SOURCE

**Reason:** from_unit_summary has no experiment-window lever -- the day boundary is applied by the caller pre-bucketing ds/exposure_date (see tests/test_frame_window_parity.py), so the window spelling is not an input.

<span id="capability-stored-state-experiment-window-day-boundary-from_unit_panel"></span>

#### Experiment window days (start, end and observation-horizon days) follow the day boundary: a naive value is the wall clock at the boundary, an aware value is converted — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** zulu and offset spellings of one instant

**Status:** not comparable

**Classification:** SOURCE

**Reason:** from_unit_panel has no experiment-window lever -- the day boundary is applied by the caller pre-bucketing ds/exposure_date (see tests/test_frame_window_parity.py), so the window spelling is not an input.

**Condition:** two offset spellings of one instant

**Status:** not comparable

**Classification:** SOURCE

**Reason:** from_unit_panel has no experiment-window lever -- the day boundary is applied by the caller pre-bucketing ds/exposure_date (see tests/test_frame_window_parity.py), so the window spelling is not an input.

<span id="capability-stored-state-experiment-window-day-boundary-from_switchback_panel"></span>

#### Experiment window days (start, end and observation-horizon days) follow the day boundary: a naive value is the wall clock at the boundary, an aware value is converted — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** zulu and offset spellings of one instant

**Status:** not comparable

**Classification:** SOURCE

**Reason:** from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.

**Condition:** two offset spellings of one instant

**Status:** not comparable

**Classification:** SOURCE

**Reason:** from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.

<span id="capability-stored-state-experiment-window-day-boundary-from_definitions"></span>

#### Experiment window days (start, end and observation-horizon days) follow the day boundary: a naive value is the wall clock at the boundary, an aware value is converted — Definitions

**Recorded path:** `from_definitions`

**Condition:** zulu and offset spellings of one instant

**Status:** supported

**Condition:** two offset spellings of one instant

**Status:** supported

**Condition:** day-boundary-compatible start/end/horizon window identity

**Status:** supported

**Operation:** `capture_sequential()`

**Stage:** read

**Condition:** legacy native registration predates bound window_days

**Status:** refused

**Codes:** `sequential.source.invalid`

**Reason:** Legacy registration does not identify the derived observation window.

**Operation:** `capture_sequential()`

**Stage:** construction

**Note:** a registration made before the window was bound must be re-registered

<span id="capability-stored-state-experiment-window-day-boundary-from_unit_day_artifact"></span>

#### Experiment window days (start, end and observation-horizon days) follow the day boundary: a naive value is the wall clock at the boundary, an aware value is converted — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** zulu and offset spellings of one instant

**Status:** supported

**Condition:** two offset spellings of one instant

**Status:** supported

**Condition:** day-boundary-compatible start/end/horizon window identity

**Status:** supported

**Operation:** `capture_sequential()`

**Stage:** read

**Condition:** artifact context omits window_days

**Status:** refused

**Codes:** `artifact.context.mismatch`

**Reason:** Artifact context identity omits the experiment window.

**Operation:** `open/adopt artifact`

**Stage:** admission

**Note:** the context binds the window days; a context without them must be republished

<span id="capability-stored-state-experiment-window-day-boundary-from_moments"></span>

#### Experiment window days (start, end and observation-horizon days) follow the day boundary: a naive value is the wall clock at the boundary, an aware value is converted — Portable moments

**Recorded path:** `from_moments`

**Condition:** zulu and offset spellings of one instant

**Status:** not comparable

**Classification:** SOURCE

**Reason:** from_moments replays an exported moment cube; it carries no declared experiment window to spell.

**Condition:** two offset spellings of one instant

**Status:** not comparable

**Classification:** SOURCE

**Reason:** from_moments replays an exported moment cube; it carries no declared experiment window to spell.


<span id="capability-stored-state-artifact-publication-abort"></span>

### Publication abort after the manifest insert began invalidates the generation (tombstone first, then relation erase; if the tombstone cannot be written the relations are kept and named, and a fresh process hides the generation before they are dropped)

<span id="capability-stored-state-artifact-publication-abort-from_unit_summary"></span>

#### Publication abort after the manifest insert began invalidates the generation (tombstone first, then relation erase; if the tombstone cannot be written the relations are kept and named, and a fresh process hides the generation before they are dropped) — Unit summary

**Recorded path:** `from_unit_summary`

**Condition:** this ingress has no WarehouseArtifactStore publication-generation lifecycle

**Status:** not comparable

**Reason:** Artifact-store generation lifecycle is not an operation on this experiment builder.

**Operation:** `WarehouseArtifactStore publication/abort`

**Stage:** read

<span id="capability-stored-state-artifact-publication-abort-from_unit_panel"></span>

#### Publication abort after the manifest insert began invalidates the generation (tombstone first, then relation erase; if the tombstone cannot be written the relations are kept and named, and a fresh process hides the generation before they are dropped) — Unit panel

**Recorded path:** `from_unit_panel`

**Condition:** this ingress has no WarehouseArtifactStore publication-generation lifecycle

**Status:** not comparable

**Reason:** Artifact-store generation lifecycle is not an operation on this experiment builder.

**Operation:** `WarehouseArtifactStore publication/abort`

**Stage:** read

<span id="capability-stored-state-artifact-publication-abort-from_switchback_panel"></span>

#### Publication abort after the manifest insert began invalidates the generation (tombstone first, then relation erase; if the tombstone cannot be written the relations are kept and named, and a fresh process hides the generation before they are dropped) — Switchback panel

**Recorded path:** `from_switchback_panel`

**Condition:** this ingress has no WarehouseArtifactStore publication-generation lifecycle

**Status:** not comparable

**Reason:** Artifact-store generation lifecycle is not an operation on this experiment builder.

**Operation:** `WarehouseArtifactStore publication/abort`

**Stage:** read

<span id="capability-stored-state-artifact-publication-abort-from_definitions"></span>

#### Publication abort after the manifest insert began invalidates the generation (tombstone first, then relation erase; if the tombstone cannot be written the relations are kept and named, and a fresh process hides the generation before they are dropped) — Definitions

**Recorded path:** `from_definitions`

**Condition:** a definitions experiment publishes its warehouse artifact

**Status:** supported

**Operation:** `publish_unit_day_artifact()`

**Stage:** read

**Note:** warehouse artifact store publication

<span id="capability-stored-state-artifact-publication-abort-from_unit_day_artifact"></span>

#### Publication abort after the manifest insert began invalidates the generation (tombstone first, then relation erase; if the tombstone cannot be written the relations are kept and named, and a fresh process hides the generation before they are dropped) — Unit-day artifact

**Recorded path:** `from_unit_day_artifact`

**Condition:** an already-adopted reader attempts to read a generation tombstoned by the store

**Status:** refused

**Codes:** `artifact.generation.dropped`

**Reason:** The artifact generation was tombstoned and erased.

**Operation:** `run an adopted artifact experiment after generation drop`

**Stage:** read

<span id="capability-stored-state-artifact-publication-abort-from_moments"></span>

#### Publication abort after the manifest insert began invalidates the generation (tombstone first, then relation erase; if the tombstone cannot be written the relations are kept and named, and a fresh process hides the generation before they are dropped) — Portable moments

**Recorded path:** `from_moments`

**Condition:** this ingress has no WarehouseArtifactStore publication-generation lifecycle

**Status:** not comparable

**Reason:** Artifact-store generation lifecycle is not an operation on this experiment builder.

**Operation:** `WarehouseArtifactStore publication/abort`

**Stage:** read

<span id="capability-stored-state-artifact-publication-abort-family"></span>

#### Publication abort after the manifest insert began invalidates the generation (tombstone first, then relation erase; if the tombstone cannot be written the relations are kept and named, and a fresh process hides the generation before they are dropped): family-owned operations

**Condition:** manifest insert has begun and publication body aborts

**Status:** refused

**Codes:** `artifact.generation.dropped`

**Reason:** Aborted generation is tombstoned before its relations are erased.

**Operation:** `abort publication after manifest insert`

**Stage:** read

**Condition:** cleanup tombstone write fails after body error

**Status:** supported

**Operation:** `abort publication with tombstone failure`

**Stage:** read

**Condition:** tombstone cannot be persisted

**Status:** supported

**Operation:** `publication abort with tombstone failure`

**Stage:** read

**Condition:** a new process reopens the store and abandons/erases a generation left visible by the prior process

**Status:** supported

**Operation:** `abandon_generation()/drop_generation()`

**Stage:** read
<!-- capability-table:stored-state:end -->
