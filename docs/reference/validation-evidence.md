# Validation evidence

Use this reference to trace a checked claim to its assumptions, source and tolerance.
Start with [the validation summary](../validation.md) for what is and is not established,
or [statistical limitations](../limitations.md#at-a-glance) for what to do with a result.
A resolved test reference is not a passing test, and a bounded calibration is not a
mathematical guarantee.

## Find an evidence record

| Method or question | Record |
|---|---|
| Welch mean difference | [EV-WELCH-T](#ev-welch-t) |
| Clopper–Pearson single-arm interval | [EV-CP-BINOM-TEST](#ev-cp-binom-test) |
| Benjamini–Hochberg selection | [EV-BH-P-ADJUST](#ev-bh-p-adjust) |
| Meta-analysis: DL, HKSJ and HT | [EV-META-DL-HKSJ-HT](#ev-meta-dl-hksj-ht) |
| Quantile order-statistic bracket | [EV-QUANTILE-QBINOM](#ev-quantile-qbinom) |
| DML, AIPW and IPTW aggregation | [EV-ADJ-DML-AIPW-IPTW](#ev-adj-dml-aipw-iptw) |
| Eight contrasts for two independent experiments | [EV-OVERLAP-WELCH](#ev-overlap-welch) |
| Marginal confounding bound | [EV-CONFOUNDING-MARGINAL](#ev-confounding-marginal) |
| Segment posterior quadrature | [EV-META-POSTERIOR-QUAD](#ev-meta-posterior-quad) |
| Welch arithmetic | [EV-WELCH-ARITHMETIC](#ev-welch-arithmetic) |
| Binomial risk-ratio enumeration | [EV-BINOMIAL-RR-ENUM](#ev-binomial-rr-enum) |
| Binomial additive score test | [EV-BINOMIAL-SCORE-SIZE](#ev-binomial-score-size) |
| Cluster covariance witnesses | [EV-CLUSTER-WITNESSES](#ev-cluster-witnesses) |
| Joint relative confidence sets | [EV-JOINT-RELATIVE](#ev-joint-relative) |
| Outward-rounded arithmetic | [EV-CERTIFIED-ARITH](#ev-certified-arith) |
| HKSJ at small K | [EV-HKSJ-SMALL-K](#ev-hksj-small-k) |
| Rare-event support and coverage | [EV-RARE-EVENT-GRID](#ev-rare-event-grid) |
| Scalar CUPED coefficient | [EV-CUPED-THETA](#ev-cuped-theta) |
| Unclustered CATE and targeting | [EV-CATE-UNCLUSTERED-CAL](#ev-cate-unclustered-cal) |
| Clustered CATE and targeting | [EV-CATE-CLUSTER-SCREEN](#ev-cate-cluster-screen) |

For methods with no matched independent comparison, use the
[method evidence index](#methods-without-an-independent-reference). Other records cover
[user journeys](#ordinary-user-journey-contracts-and-current-evidence),
[campaign measurements](#campaign-measurements-and-scope),
[reproduction](#reproduce), and [discrepancies](#discrepancy-log).

## Numerical qualification reconciliation
The exact-binomial confidence-set construction and enumerated power use SciPy special functions
and float64 summation. Their finite-sample statistical construction is distinct from the
arithmetic enclosure, which is conditional on the deployed SciPy/Boost error model. The
`scipy_special_function_error_model_conditional_v1` result metadata does not claim a uniform
proof across builds. The repository lock currently resolves SciPy 1.18.0, but the cited decimal
oracle evidence does not identify that locked version as the executed build. The prior measured
error envelope was arm64 with fused multiply-subtract; x86 has not been executed. This is a
verification gap, not a reproduced bound failure.

PR38–45 implemented the previously open caching/support-window, endpoint-tolerance, nuisance
stopping/error-model/ceiling, and automatic dense-route changes; those completed changes are not
open validation items here. `1tce` remains an investigation of the special-function allowance
above about $2^{26}$ trials per arm; undercoverage has not been demonstrated. `xeth` remains the
separate deployed-numerics verification obligation for sequential Bernoulli likelihood,
inversion and decisions. Neither is discharged by the local qualification label.

Targeted independent review remains pending in tracker `0p4v`: the risk-ratio nuisance
construction and error bounds; quantile-bracket-to-interval conversion; and the sequential
likelihood, inversion and decision chain. This repository's own probes and derivations are not
independent cases or outside review.

The older binomial evidence rows retain references to trackers that predate these
implementation credits. Do not interpret those references as a claim that PR38–45
changes were absent: the remaining documented numerical investigation is `1tce`,
the sequential deployed-numerics obligation is `xeth`, and targeted independent review
is pending in `0p4v`. `4xwx` separately owns remeasuring the remaining planning cost;
no current large-design timing is inferred from the earlier PR45 probes.

## Reference inventory

### Reference classes

- **`third_party_tool`:** frozen output of an external package at a matched
  estimand and assumptions. An independent third-party reference.
- **`independent_reimplementation`:** an external primitive (for example R
  `qbinom`) recomputes the same formula. Independent only for that primitive.
- **`independent_derivation_in_repo`:** hand arithmetic, quadrature or exact
  enumeration written in this repository. Not independent: same authorship.
- **`calibration_evidence`:** a bounded Monte Carlo grid with preselected cells.
  Behavior evidence, not an independent reference.
- **`none_found`:** a matched reference was looked for and not found.
  Independent-reference classification is not applicable.

Ordinary tests read frozen JSON only: no R, no network, no external package at test time. The
comparisons below are at estimator level unless the direct-coverage fields name an entry point. A
transitive ingress is claimed only where a parity-matrix cell or parity case exercises the same
estimator path; cross-ingress equivalence is owned by the parity harness
(`helper:tests/parity_harness/matrix.py`), not by these oracles.

A tolerance written as relative below is enforced as `pytest.approx(expected, rel=..., abs=0.0)`.
`pytest.approx` otherwise keeps a default absolute floor of `1e-12` next to a relative tolerance,
which would dominate a near-zero value, so the tests pass `abs=0.0` and no absolute allowance exists.
A tolerance that is absolute is stated as absolute.

## Frozen comparisons against a third-party tool or independent primitive

These records compare the named estimator and assumptions, not every use of the method.

### EV-OVERLAP-WELCH

**Method:** `increment.overlap.analyze_overlap` estimates prespecified fixed-window marginal, conditional, joint-launch and interaction effects in original outcome units.

**Guarantee and assumptions:** approximate Welch intervals with Bonferroni adjustment, on independent units assigned under a registered product law. Requires pretreatment eligibility, no cross-unit interference, stable outcome definitions and complete fixed-window measurement. The adjustment does not make the component approximation exact.

**Independent reference:** R 4.4.2 saturated regression and its signed covariance transformed from cell means, with independent `qt` and `pt` calls. `helper:tests/oracles/overlap_contrasts.R` generates `helper:tests/fixtures/overlap/regression_reference.json` (SHA-256 `73f0e5180fb3e0799f7bdd1d9f839df6510cf03ceb0053af750d04ef4d6faa05`). Compare all eight contrasts, covariance, degrees of freedom, adjusted bands and boundary probabilities in `test:tests/estimation/test_overlap.py::test_saturated_regression_reference`. Small-support and extreme-tail cases are checked separately in that module.

**Input checks:** `test:tests/test_overlap_workflow.py::test_public_sdk_external_frame_warehouse_and_wire` compares archived assignments, an independent external log, frame and DuckDB capture, and checked replay. The aggregate-only DuckDB witness `test:tests/query/test_overlap.py::test_query_egress_is_aggregate_only` returned 9 rows × 83 scalar columns for both 3 and 1,003 supplied units, and 1 × 83 for a duplicate-key refusal (DuckDB 1.5.5, Ibis 12.0.0, macOS arm64). This is a transfer-size measurement, not a runtime promise.

**Frozen simulation:** 960,000 synthetic attempts across 12 registered cases, with 20,000 attempts per case for each of four model-by-algorithm combinations. All 172 checks passed: 88 exact-binomial checks, 36 matched planning-rate checks and 48 support checks. The worst upper bound on false-claim approval was 1.860%, against a 6% limit; the worst upper bound on false joint clearance was 0.433%, also against 6%. Among the dense cases, the lowest lower bounds were 99.954% for complete issuance and 96.487% for simultaneous issued-and-cover, against 99% and 94% limits. The largest difference between matched planning rates was 0.605 percentage points. These are bounds and observations from the specified simulations, not guarantees for other data.

**Reproduce:** `campaign:calibration.overlap --protocol calibration/protocols/overlap.md --out overlap-qualification.json`. The frozen protocol SHA-256 is `5863fb24ba484fcaf0ab9ce6338f0799f38b0c75371367da17135583375021ad`, with seed `2026111983`. The recorded run used source `bcb0365a`, Python 3.15.2, NumPy 2.5.4 and SciPy 1.18.1; its JSON SHA-256 is `f3528bf3c46c660366690b5f1d2f89689c8fa86a0a1ea9820d8cb1832ff8092f`. A later reporting correction counts overlapping reasons for unavailable reference rows separately. That changes reason totals only, not attempts, estimates, intervals, decisions or gates.

**Limits:** Does not certify a real producer's assignment law or operating data. No clusters, dependent assignments, repeated looks, covariate adjustment, relative effects or warehouse engines other than the documented DuckDB path were verified for this workflow.


### EV-WELCH-T

**Method and public symbol:** Fixed-horizon, prior-free unweighted mean difference, `run(experiment, reader)` without `prior`, absolute-axis fields

**Guarantee:** `asymptotic`: additive difference of independent unweighted arm means, with `ddof=1` arm variances and their own Welch-Satterthwaite t reference; a working approximation, not a finite-sample guarantee ([limitations](../limitations.md#uncertainty-is-estimated-on-every-path))

**Metric types:** mean

**Estimand and assumptions:** Treatment minus control difference of arm means, Welch variance, Welch-Satterthwaite degrees of freedom, two-sided 95% t interval; independent units

**Reference class:** `third_party_tool`

**Reference tool and version:** R 4.4.2 `stats::t.test(var.equal = FALSE)`

**Direct entry-point coverage:** `from_frame(shape="summary")`

**Transitive entry-point coverage:** parity cell `mean-run-none-utc-error` (`supported` on `from_definitions`, `from_unit_day_artifact`, `from_frame(shape="summary")`, `from_frame(shape="panel")`, `from_moments`); `from_switchback_frame` is a separate estimand and is not covered by this oracle

**Tolerance and rationale:** relative `1e-12` on difference, SE, df and both endpoints; the largest observed relative difference was `6.7e-15`, so this is floating-point headroom

**Fixture and generator:** `helper:tests/oracles/fixtures/welch_t.json` `abf22cc43957`; `helper:tests/oracles/generate/gen_welch_oracle.R`

**Test selectors:** `tests/oracles/test_welch_oracle.py::test_welch_difference_interval_matches_r_t_test`

**Regimes not covered:** Two cases (14 vs 22 units with unequal variance; 5 vs 6 units, near-equal variance). Arms of fewer than 2 units are not compared. Informative-prior runs (not exercised by this frozen comparison; a prior now leaves sampling estimates, intervals and references unchanged, with a separately stored posterior; see [Bayesian results](bayesian-results.md)), weighted, clustered, covariate-adjusted, ratio, winsorized and sequential contrasts are not compared. The p-value is not compared

**Regeneration command:** `Rscript tests/oracles/generate/gen_welch_oracle.R`

**Regeneration tools:** R 4.4.2, jsonlite 1.8.9

**Committed digest and provenance:** `helper:tests/oracles/fixtures/welch_t.json` `abf22cc43957`

**One-time regeneration (2026-10-03, macOS arm64):** generated for this page; recorded generating commit `37b9b34`


### EV-CP-BINOM-TEST

**Method and public symbol:** `increment.estimation.binomial_rr.clopper_pearson`

**Guarantee:** `exact` for the single-arm primitive; the risk-ratio route that uses it is the exact-binomial row of [Methods without an independent reference](#methods-without-an-independent-reference), with its numerical qualifications ([limitations](../limitations.md#conversion-and-retention-arms-dense-counts-use-the-delta-method-sparse-counts-are-estimated-exactly-not-refused)); open numerical items (internal trackers): `1tce`, `fd17`, `9af7`, `rsf0`

**Metric types:** conversion, retention (primitive of the exact-binomial route)

**Estimand and assumptions:** Exact two-sided single-arm Clopper-Pearson `1 - beta` interval; production rounds outward by a documented relative slack

**Reference class:** `third_party_tool`

**Reference tool and version:** R 4.4.2 `stats::binom.test()$conf.int`

**Direct entry-point coverage:** none: the primitive is called directly

**Transitive entry-point coverage:** none claimed; the primitive feeds the exact-binomial risk-ratio set, which this oracle does not compare

**Tolerance and rationale:** enclosure of R's interval within `1e-15`, then an outward gap of at most `1.01 * 1e-6` of the lower bound below and of one minus the upper bound above; the band is the documented slack, and every measured gap equals it to within `5e-6` of itself

**Fixture and generator:** `helper:tests/oracles/fixtures/clopper_pearson.json` `7f53220dd69a`; `helper:tests/oracles/generate/gen_clopper_pearson_oracle.R`

**Test selectors:** `tests/oracles/test_clopper_pearson_oracle.py::test_interval_encloses_and_matches_r_binom_test`

**Regimes not covered:** 12 cases: `(x, n)` in `(0,20) (1,20) (7,40) (39,40) (40,40) (500,1000)` at `beta` 0.05 and 0.001. Two-arm risk-ratio sets are not compared. Arms above about 2^26 are an open numerical investigation (internal tracker `1tce`: the SciPy primitive-error assumption behind the certificate margin is falsified at that size; public undercoverage is not demonstrated and nothing here claims it repaired)

**Regeneration command:** `Rscript tests/oracles/generate/gen_clopper_pearson_oracle.R`

**Regeneration tools:** R 4.4.2, jsonlite 1.8.9

**Committed digest and provenance:** `helper:tests/oracles/fixtures/clopper_pearson.json` `7f53220dd69a`

**One-time regeneration (2026-10-03, macOS arm64):** generated for this page; recorded generating commit `37b9b34`


### EV-BH-P-ADJUST

**Method and public symbol:** `increment.estimation.family.bh_select`

**Guarantee:** `exact` for the selection step given finite-sample-valid (super-uniform) input p-values: controls FDR at the stated level under independence or positive dependence (PRDS), a condition that is not checked. The class describes the primitive only; a fixed-horizon family takes the weakest inference route among its members, so a family containing an asymptotic Welch or delta-method p-value has no finite-sample FDR statement ([limitations](../limitations.md#fixed-horizon-fdr-control-assumes-a-dependence-condition-that-is-not-checked))

**Metric types:** any metric (selection over p-values)

**Estimand and assumptions:** Benjamini-Hochberg step-up rejection set at FDR level `q` over a fixed p-value family, ties selected together; index `i` is rejected iff R's BH-adjusted p-value is `<= q`

**Reference class:** `third_party_tool`

**Reference tool and version:** R 4.4.2 `stats::p.adjust(method = "BH")`

**Direct entry-point coverage:** none: the selection function is called directly

**Transitive entry-point coverage:** none claimed

**Tolerance and rationale:** exact equality of index sets (no numeric tolerance)

**Fixture and generator:** `helper:tests/oracles/fixtures/bh_p_adjust.json` `aa3f442ec7eb`; `helper:tests/oracles/generate/gen_bh_oracle.R`

**Test selectors:** `tests/oracles/test_bh_oracle.py::test_selected_set_matches_r_p_adjust_bh`

**Regimes not covered:** 7 families at `q` 0.05, 0.1 and 0.25, chosen away from the `k * q / m` boundary (see [Discrepancy log](#discrepancy-log)). Dependence conditions, family construction, e-BH and sequential selection are not compared

**Regeneration command:** `Rscript tests/oracles/generate/gen_bh_oracle.R`

**Regeneration tools:** R 4.4.2, jsonlite 1.8.9

**Committed digest and provenance:** `helper:tests/oracles/fixtures/bh_p_adjust.json` `aa3f442ec7eb`

**One-time regeneration (2026-10-03, macOS arm64):** generated for this page; recorded generating commit `37b9b34`


### EV-META-DL-HKSJ-HT

**Method and public symbol:** `increment.estimation.meta.cochran_q`, `hksj_pooled_mean`

**Guarantee:** `asymptotic`: DerSimonian-Laird heterogeneity estimate with an unmodified Hartung-Knapp-Sidik-Jonkman adjustment on a `t_(K-1)` reference; no finite-sample guarantee is stated, and the plug-in interval is anticonservative at small K ([limitations](../limitations.md#meta-analysis-can-be-anticonservative-at-small-k))

**Metric types:** segment estimates with variances (log relative-lift rows)

**Estimand and assumptions:** DerSimonian-Laird tau^2, Hartung-Knapp-Sidik-Jonkman pooled mean on a `t_(K-1)` reference, Higgins-Thompson test-based I^2 interval

**Reference class:** `third_party_tool`

**Reference tool and version:** R 4.4.2, metafor 4.8.0 (`rma` with `method = "DL"`, `test = "knha"`; `confint(type = "HT")`)

**Direct entry-point coverage:** none: estimator on supplied estimates and variances

**Transitive entry-point coverage:** none claimed

**Tolerance and rationale:** tau^2 relative `1e-8`, HKSJ and I^2 interval absolute `1e-6`; closed forms, so the headroom covers summation order

**Fixture and generator:** `helper:tests/oracles/fixtures/meta_dl_hksj.json` `8eaa14439a24`; `helper:tests/oracles/generate/gen_meta_oracle.R`

**Test selectors:** `tests/oracles/test_meta_oracle.py::test_dl_tau2_matches_metafor`, `tests/oracles/test_meta_oracle.py::test_hksj_pooled_mean_matches_metafor_knha`, `tests/oracles/test_meta_oracle.py::test_i2_ci_matches_metafor_higgins_thompson`

**Regimes not covered:** Two cases (K = 5, K = 8). Smaller K is covered only by calibration (see below). metafor's default Q-profile interval is a different method and is not compared

**Regeneration command:** `Rscript tests/oracles/generate/gen_meta_oracle.R`

**Regeneration tools:** R 4.4.2, metafor 4.8.0, jsonlite 1.8.9

**Committed digest and provenance:** `helper:tests/oracles/fixtures/meta_dl_hksj.json` `8eaa14439a24`, introduced in `94f4b6b` (2026-10-01)

**One-time regeneration (2026-10-03, macOS arm64):** byte-identical


### EV-QUANTILE-QBINOM

**Method and public symbol:** `increment.estimation.quantile.log_quantile_se`

**Guarantee:** `asymptotic` for the reported two-arm interval (the standard error comes from an order-statistic bracket and is divided by a normal critical value); the per-arm bracket's coverage is distribution-free, but the two-arm log-ratio interval carries its own qualifications, so no `exact` class is claimed from the primitive ([limitations](../limitations.md#a-quantiles-reported-interval-depends-on-the-alpha-you-asked-for), [path table](../limitations.md#uncertainty-is-estimated-on-every-path))

**Metric types:** quantile

**Estimand and assumptions:** Log-scale half-width of the discrete order-statistic bracket whose ranks come from the exact binomial quantile

**Reference class:** `independent_reimplementation`

**Reference tool and version:** R 4.4.2 `qbinom` (not a port of SciPy's); sample generated with numpy `default_rng(20260915)`

**Direct entry-point coverage:** none: estimator on a sample

**Transitive entry-point coverage:** parity cell `quantile-run-none-utc-error` (`supported`: four ingresses; `from_moments` refuses `evidence.unit_rows_unavailable`)

**Tolerance and rationale:** relative `1e-12` on the standard error; the rank selection is bit-identical, so this is floating-point headroom

**Fixture and generator:** `helper:tests/oracles/fixtures/quantile_woodruff.json` `4c7560999fa8`; `helper:tests/oracles/generate/gen_quantile_samples.py`, `helper:tests/oracles/generate/gen_quantile_oracle.R`

**Test selectors:** `tests/oracles/test_quantile_oracle.py::test_log_scale_half_width_matches_independent_qbinom_bracket`

**Regimes not covered:** The independence is of the binomial-quantile primitive, not of the interval method; `survey::oldsvyquantile` was rejected because it is a different estimator. Two samples (n = 40 at q = 0.9; n = 120 at q = 0.5)

**Regeneration command:** `uv run python tests/oracles/generate/gen_quantile_samples.py`, then `Rscript tests/oracles/generate/gen_quantile_oracle.R`

**Regeneration tools:** R 4.4.2, numpy 2.5.1 (the fixture records only R)

**Committed digest and provenance:** `helper:tests/oracles/fixtures/quantile_woodruff.json` `4c7560999fa8`, `helper:tests/oracles/data/quantile_values.csv` `c65951504fa2`, introduced in `94f4b6b`

**One-time regeneration (2026-10-03, macOS arm64):** CSV and fixture byte-identical


### EV-ADJ-DML-AIPW-IPTW

**Method and public symbol:** `increment.estimation.adjust.estimate_ate` with `Method(name="dml" / "aipw" / "iptw")`

**Guarantee:** `asymptotic`: influence-function sandwich with an asymptotic Normal reference under identification; DML reports a partially-linear slope, not an average treatment effect ([limitations](../limitations.md#cluster-references-are-qualified-working-approximations), [DML](../limitations.md#dml-reports-a-partially-linear-slope-not-an-average-treatment-effect))

**Metric types:** mean (covariate-adjusted observational ATE)

**Estimand and assumptions:** Aggregation formulas given identical frozen out-of-fold nuisances: partially-linear DML slope, AIPW ATE (`normalize_ipw=True`), Hajek IPTW with an HC0 sandwich

**Reference class:** `third_party_tool`

**Reference tool and version:** doubleml 0.11.4 (PLR, IRM), statsmodels 0.15.0 (WLS HC0), numpy 2.5.1, scikit-learn 1.9.1

**Direct entry-point coverage:** `from_frame(shape="summary")`

**Transitive entry-point coverage:** parity case `observational_aipw_dml_covariate_adjusted` covers AIPW/DML on `from_definitions`, `from_unit_day_artifact`, `from_frame(shape="summary")` and `from_frame(shape="panel")` (`from_moments` refuses `evidence.covariate_unavailable`); no transitive IPTW standard-error comparison claimed: `observational_iptw_covariate_adjusted_ate` uses the different native fitted-logistic covariance; matrix cells `mean-run-observational-utc-error` and `conversion-run-observational-utc-error` are `supported`, `ratio-run-observational-utc-error` is `construction_limited`

**Tolerance and rationale:** `1e-9` relative on estimates and on DML and IPTW standard errors; AIPW standard error `1e-4` relative (measured about `7e-5`)

**Fixture and generator:** `helper:tests/oracles/fixtures/dml_aipw_iptw.json` `c426e24f472d`, `helper:tests/oracles/data/dml_synthetic.csv` `99473fdc78f2`; `helper:tests/oracles/generate/gen_dml_oracle.py`

**Test selectors:** `tests/oracles/test_dml_oracle.py::test_dml_matches_doubleml_external_predictions`, `tests/oracles/test_dml_oracle.py::test_aipw_matches_doubleml_irm_ate_score`, `tests/oracles/test_dml_oracle.py::test_iptw_matches_statsmodels_wls_hajek`

**Regimes not covered:** Nuisance fitting and cross-fitting are not validated: the nuisances come from this repository's own fold assignment. One dataset (n = 500, 5 folds). The DML estimand is the partially-linear slope, not an average treatment effect ([limitations](../limitations.md#dml-reports-a-partially-linear-slope-not-an-average-treatment-effect)). Clustered and trimmed variants are not compared. Default untrimmed native-logistic IPTW covariance, including propensity-fitting estimating equations, is not compared with the frozen WLS-HC0 reference

**Regeneration command:** `uv run --python 3.12 --with numpy==2.5.1 --with scikit-learn==1.9.1 --with statsmodels==0.15.0 --with doubleml==0.11.4 python tests/oracles/generate/gen_dml_oracle.py`

**Regeneration tools:** CPython 3.12, numpy 2.5.1, scikit-learn 1.9.1, statsmodels 0.15.0, doubleml 0.11.4

**Committed digest and provenance:** `helper:tests/oracles/fixtures/dml_aipw_iptw.json` `c426e24f472d`, `helper:tests/oracles/data/dml_synthetic.csv` `99473fdc78f2`, introduced in `94f4b6b`

**One-time regeneration (2026-10-03, macOS arm64):** CSV byte-identical; fixture differs only in the trailing newline, every value identical


## In-repo derivations and calibration evidence

These are not independent third-party references. They record what exists beyond the frozen comparisons.


### EV-CONFOUNDING-MARGINAL

**Method and public symbol:** `increment.estimators.confounding_sensitivity`, `increment.results.ConfoundingSensitivity`

**Guarantee:** Deterministic conservative marginal bound; interval inherits the row's asymptotic working-approximation guarantee, not a new finite-sample coverage claim ([guide](../guides/observational.md#how-strong-would-an-omitted-confounder-need-to-be))

**Estimand, assumptions and measured scope:** Binary outcome from the row's declared metric type (conversion or retention); consistency, positivity and nuisance-model correctness in the reported population; exchangeability given measured C and omitted U; strength maxima over strata and every cohort arm pair in both directions; no homogeneous conditional effect assumption

**Reference class:** `independent_derivation_in_repo`

**Test selectors and tolerances:** Enumeration oracle and no-confounding limit: `tests/estimation/test_confounding.py::test_bound_holds_on_random_populations`, `tests/estimation/test_confounding.py::test_no_confounding_population_recovers_identified_rr` (300 populations each, seed 20261013; bound absolute slack 1e-12, no-confounding relative tolerance 1e-12); counterexample negative control: `tests/estimation/test_confounding.py::test_counterexample_is_bounded`; independent bounding-factor root check: `tests/estimation/test_confounding.py::test_e_value_solves_bounding_factor_equation` (relative tolerance 1e-10); end-to-end RR-scale check: `tests/test_confounding_sensitivity_public.py::test_iptw_and_aipw_rows_report_marginal_risk_ratio` (40,000 units, seed 20261013, within four interval-derived standard errors); wrong-scale negative controls: `tests/test_confounding_sensitivity_public.py::test_marginal_risk_ratio_check_rejects_other_scales` (coherent log-RR and absolute risk-difference point/intervals, for IPTW and AIPW); metric-type eligibility from the stamped row: `tests/test_confounding_sensitivity_public.py::test_mean_metric_row_is_refused_from_its_metric_type`; complete sensitivity payload parity: `tests/test_confounding_sensitivity_public.py::test_cross_ingress_sensitivity_parity`

**Regimes not covered:** Same authorship; no third-party tool reference or coverage study. Not selection/measurement bias, ratio or continuous outcomes, DML slopes, or a sharper estimator-scoped bound.


### EV-META-POSTERIOR-QUAD

**Method and public symbol:** `marginalized_segment_intervals`

**Guarantee:** none stated: Wald intervals on posterior means and variances carry no finite-sample frequentist coverage guarantee ([limitations](../limitations.md#meta-analysis-can-be-anticonservative-at-small-k))

**Estimand, assumptions and measured scope:** Tau-marginalized segment posterior (REML marginal likelihood times a half-normal prior on tau; Morris 1983) against continuous quadrature written from the model's formulas

**Reference class:** `independent_derivation_in_repo`

**Test selectors and tolerances:** `tests/oracles/test_meta_oracle.py::test_marginalized_intervals_match_continuous_quadrature`

**Regimes not covered:** The recorded escaped-posterior counterexample and budget behavior are pinned, not generalized; Wald intervals on posterior means carry no frequentist coverage guarantee


### EV-WELCH-ARITHMETIC

**Method and public symbol:** `IndependentMeanReference`

**Guarantee:** `asymptotic`: as `EV-WELCH-T` (additive difference of independent unweighted arm means with `ddof=1` arm variances and their own Welch-Satterthwaite t reference; a working approximation, not a finite-sample guarantee)

**Estimand, assumptions and measured scope:** Welch component arithmetic and the Welch-Satterthwaite reference against hand arithmetic and normal/chi-square integration

**Reference class:** `independent_derivation_in_repo`

**Test selectors and tolerances:** `tests/estimation/test_independent_mean_reference.py`

**Regimes not covered:** Same authorship as the code under test; `EV-WELCH-T` is the third-party check


### EV-BINOMIAL-RR-ENUM

**Method and public symbol:** Exact binomial risk-ratio sets

**Guarantee:** `exact`: finite-sample relative-risk confidence set and relative-scale decisions for `reference_kind="binomial"` rows; the additive `abs_lb`/`abs_ub` interval is the Miettinen-Nurminen score interval for the difference of proportions (`abs_reference_kind="score"`), which is asymptotic and not exact (`EV-BINOMIAL-SCORE-SIZE`, [limitations](../limitations.md#uncertainty-is-estimated-on-every-path)); open numerical items (internal trackers): `1tce`, `fd17`, `9af7`, `rsf0`

**Estimand, assumptions and measured scope:** Decimal and `math.comb` joint-binomial enumeration (a valid lower bound on the supremum p-value) and a Bonferroni rectangle of Clopper-Pearson quantiles from `scipy.stats.beta` directly

**Reference class:** `independent_derivation_in_repo`

**Test selectors and tolerances:** `tests/estimation/test_binomial_rr.py`, `tests/estimation/test_rare_event_calibration.py`

**Regimes not covered:** Enumeration is bounded by `MAX_ARM_SIZE`; the grid maximum is a bound, never an equality oracle


### EV-BINOMIAL-SCORE-SIZE

**Method and public symbol:** Additive interval and absolute-margin test of exact-binomial rows

**Guarantee:** `asymptotic`: score test inversion on a Normal reference; not a finite-sample guarantee

**Estimand, assumptions and measured scope:** Exact size, summed over the count lattice, of the one-sided 0.05 margin test at the boundary `p_t = p_c - 0.01` with 1,000 units per arm and 2, 5, 10, 50 or 100 expected treatment events: at most 0.0505 (measured 0.0432 to 0.0502; the Wald rule it replaced reached 0.0785 at 2 events). Bounds invert the score test, with the restricted rates solved independently from the likelihood equation

**Reference class:** `calibration_evidence`

**Test selectors and tolerances:** `tests/estimation/test_binomial_abs_sidecar.py::test_score_margin_test_holds_its_size_at_the_margin_boundary`; `tests/estimation/test_binomial_abs_sidecar.py::test_score_bounds_invert_score_test`

**Regimes not covered:** Size is measured at one arm size, margin and level; the composite null away from its boundary is not summed


### EV-CLUSTER-WITNESSES

**Method and public symbol:** Cluster-randomized AIPW and DML variance

**Guarantee:** `asymptotic`: qualified working approximation; neither a Bessel multiplier nor a generic t choice establishes finite-sample validity for arbitrary small, unbalanced, or skewed clusters ([limitations](../limitations.md#cluster-references-are-qualified-working-approximations))

**Estimand, assumptions and measured scope:** Hand-derived cases showing the between-arm superpopulation variation is retained

**Reference class:** `independent_derivation_in_repo`

**Test selectors and tolerances:** `tests/estimation/test_i13_independent_witnesses.py`

**Regimes not covered:** The witnesses identify particular covariance defects; they do not replace small-cluster scientific gates


### EV-JOINT-RELATIVE

**Method and public symbol:** Joint relative (Fieller) inversion

**Guarantee:** `asymptotic`: relative Fieller set on a fixed t reference for clustered unadjusted arms; preserves bounded, disconnected, one-sided, full-real, empty or unavailable geometry ([limitations](../limitations.md#cluster-references-are-qualified-working-approximations))

**Estimand, assumptions and measured scope:** Numerical and directional contracts of the inversion

**Reference class:** `independent_derivation_in_repo`

**Test selectors and tolerances:** `tests/estimation/test_joint_relative_reference.py`

**Regimes not covered:** No third-party package compared


### EV-CERTIFIED-ARITH

**Method and public symbol:** Certified outward-rounded arithmetic layer

**Guarantee:** none stated: an arithmetic layer beneath the `exact` class of the exact-binomial row, not an interval of its own

**Estimand, assumptions and measured scope:** Mathematical behavior of the freestanding arithmetic

**Reference class:** `independent_derivation_in_repo`

**Test selectors and tolerances:** `tests/estimation/test_certified.py`

**Regimes not covered:** Arithmetic layer only


### EV-HKSJ-SMALL-K

**Method and public symbol:** HKSJ pooled interval at K = 2 to 12

**Guarantee:** `asymptotic`: as `EV-META-DL-HKSJ-HT`

**Estimand, assumptions and measured scope:** Coverage of the random-effects mean over 18 cells of 10000 replications

**Reference class:** `calibration_evidence`

**Test selectors and tolerances:** `tests/calibration/test_hksj_small_k.py`

**Regimes not covered:** One design: unequal variances with log-scale standard errors 0.10 to 0.20; see [limitations](../limitations.md#meta-analysis-can-be-anticonservative-at-small-k)


### EV-RARE-EVENT-GRID

**Method and public symbol:** Exact-binomial relative-lift set

**Guarantee:** `exact`: as `EV-BINOMIAL-RR-ENUM`; fixed-horizon only, it does not itself prove validity under optional stopping ([limitations](../limitations.md#conversion-and-retention-arms-dense-counts-use-the-delta-method-sparse-counts-are-estimated-exactly-not-refused)); open numerical items (internal trackers): `1tce`, `fd17`, `9af7`, `rsf0`

**Estimand, assumptions and measured scope:** Support and geometry checks over all 336 cells of the truth and support manifest; pointwise Monte Carlo coverage over six prespecified cells only

**Reference class:** `calibration_evidence`

**Test selectors and tolerances:** `tests/estimation/test_rare_event_calibration.py`, `tests/estimation/test_binomial_grid_calibration.py`

**Regimes not covered:** Empirical coverage is measured at six preselected cells, not across the 336-cell grid; the Monte Carlo cells are marked `slow`


### EV-CUPED-THETA

**Method and public symbol:** CUPED interval

**Guarantee:** `asymptotic`: the standard error conditions on a coefficient estimated from the same data it adjusts, and the uncertainty in that estimate is not propagated ([limitations](../limitations.md#cuped-treats-its-adjustment-coefficient-as-known))

**Estimand, assumptions and measured scope:** 16 cells of 3000 replications against a known-coefficient comparator

**Reference class:** `calibration_evidence`

**Test selectors and tolerances:** `tests/calibration/test_cuped_theta_uncertainty.py`

**Regimes not covered:** Equal allocation and one homogeneous slope; the comparator is a delta-method construction, not exact inference


### EV-CATE-UNCLUSTERED-CAL

**Method and public symbol:** `estimate_cate` and `validate_cate` (the same estimators behind `targeting_rule` and `select_targeting_rule`), unclustered, randomized or doubly-robust scores

**Guarantee:** none stated: no finite-sample guarantee; for `Observational` scores identification requires conditional ignorability and overlap ([limitations](../limitations.md#targeting-validation-depends-on-identification-and-overlap))

**Estimand, assumptions and measured scope:** Interval coverage of the ATE and one interaction slope on a Gaussian design (n = 4000, 1000 replications, accepted band 0.93 to 0.97); ATE coverage at a 10/90 allocation with unequal arm variances (n = 4000, 2000 replications, at least 0.93) and for a Bernoulli outcome (1000 replications, band 0.92 to 0.98, also at 10/90); one-sided AUTOC null rejection rate (n = 2000, 2000 replications, band 0.03 to 0.07); doubly-robust AUTOC null rejection rate under confounding (500 replications, within three Monte Carlo standard errors of 0.05, with plain IPW rejecting above 0.15)

**Reference class:** `calibration_evidence`

**Test selectors and tolerances:** `tests/test_cate_calibration.py::TestIntervalsCoverAtNominal`, `tests/test_cate_calibration.py::TestSandwichSurvivesSkewedAllocation`, `tests/test_cate_calibration.py::TestBernoulliOutcomeCoverage`, `tests/test_cate_calibration.py::TestBernoulliOutcomeCoverageAtSkewedAllocation`, `tests/test_cate_calibration.py::TestAutocNullSize`, `tests/test_cate_calibration.py::TestDrPsiCalibrationUnderConfounding`

**Regimes not covered:** Estimator-level draws (`fit_cate`, `validate_cate_arrays`), not a source ingress; a few fixed designs whose bands are regression screens, not coverage guarantees; identification is assumed


### EV-CATE-CLUSTER-SCREEN

**Method and public symbol:** Clustered `validate_cate` statistics (AUTOC, Qini, GATES, CLAN) and clustered `estimate_cate`

**Guarantee:** `asymptotic`: clustered `estimate_cate` uses a weighted CR1/HC0 score sandwich with `K-1` reference degrees of freedom; the held-out validation statistics carry no distribution-free finite-cluster coverage ([limitations](../limitations.md#clustered-cate-uncertainty-is-cluster-asymptotic))

**Estimand, assumptions and measured scope:** Null screen of the held-out validation statistics on frozen oracle scores: 72 designs (10, 40 or 200 clusters; equal sizes of 20 or repeating sizes 5/20/100; intracluster correlation 0, 0.2 or 0.5; Gaussian or skewed errors with one high-leverage score per cluster; two cluster weightings), 256 replications and 199 bootstrap draws each, held to family-aware binomial miss bounds. The full grid runs with `python -m calibration.mc_safety c07 --out c07.jsonl` (`campaign-module:calibration.mc_safety c07 --out c07.jsonl`); the pytest selector runs the reduced four-design sentinel set. A separate 288-cell diagnostic of the `estimate_cate` covariance (128 draws per cell) gates variance moments and records, but does not gate, coverage and Wald rejection. A public-path replay of `validate_cate`, `targeting_rule` and `select_targeting_rule` on eight fixed cluster witnesses records behavior

**Reference class:** `calibration_evidence`

**Test selectors and tolerances:** `tests/test_cate_calibration.py::test_c07_cluster_null_rejection_and_coverage`, `tests/test_cate_calibration.py::test_cluster_fit_covariance_moments_and_coverage_diagnostics`, `tests/test_cate_calibration.py::test_e6v0_original_varying_noise_cluster_public_replay`

**Regimes not covered:** A regression screen, not a guarantee: it calls the internal validation function with frozen oracle scores rather than a fitted pipeline, and the replay is not an error-control certificate. Few unequal clusters with skewed errors still undercover


## Methods without an independent reference

Status describes the available evidence, not the strength of the guarantee.
`calibration_only` means bounded measured evidence; `derivation_only` means an
in-repository derivation without a matched third-party tool; `campaign_pending`
means the full stated campaign has not run; `none_found` records no matched reference.

| Method | Evidence status |
|---|---|
| [Scalar CUPED](#scalar-cuped) | `calibration_only` |
| [Mixed-covariate CUPED](#mixed-covariate-cuped) | `campaign_pending` |
| [Cluster-robust contrasts](#cluster-robust-contrasts) | `calibration_only` |
| [Ratio metrics](#ratio-metrics) | `calibration_only` |
| [Joint relative inversion](#joint-relative-inversion) | `derivation_only` |
| [Exact-binomial relative-lift sets](#exact-binomial-relative-lift-sets) | `calibration_only` |
| [Asymptotic sequential means](#asymptotic-sequential-means) | `campaign_pending` |
| [Bernoulli e-processes](#bernoulli-e-processes) | `campaign_pending` |
| [Sequential selection and intervals](#sequential-selection-and-intervals) | `derivation_only` |
| [Fixed-horizon selected intervals](#fixed-horizon-selected-intervals) | `calibration_only` |
| [Winsorization](#winsorization) | `calibration_only` |
| [Encouragement](#encouragement) | `calibration_only` |
| [Switchback unit-cycle t](#switchback-unit-cycle-t) | `calibration_only` |
| [Switchback variance envelope](#switchback-variance-envelope) | `campaign_pending` |
| [Switchback shared-block t](#switchback-shared-block-t) | `calibration_only` |
| [Parallel-arm planning](#parallel-arm-planning) | `calibration_only` |
| [Binomial decision planning](#binomial-decision-planning) | `derivation_only` |
| [Sequential planning](#sequential-planning) | `calibration_only` |
| [Switchback planning](#switchback-planning) | `calibration_only` |
| [Unit-cycle power lower bound](#unit-cycle-power-lower-bound) | `derivation_only` |
| [Segment heterogeneity planning](#segment-heterogeneity-planning) | `calibration_only` |
| [Logged-policy evaluation](#logged-policy-evaluation) | `calibration_only` |
| [Positivity and weight diagnostics](#positivity-and-weight-diagnostics) | `none_found` |
| [Total, active and retention metrics](#total-active-and-retention-metrics) | `none_found` |
| [CATE and targeting](#cate-and-targeting) | `calibration_only` |

Finite-sample binomial confidence sets distinguish their statistical construction from the
arithmetic qualification: new rows record `scipy_special_function_error_model_conditional_v1`,
conditional on the deployed SciPy/Boost error model rather than a uniform cross-build proof.
Rows saved before that marker existed use `legacy_unrecorded_v1` and make no numerical
qualification claim. Enumerated binomial planning points publish as points when their
computed enclosure resolves to absolute error at most `1e-6` conditional on that model;
`power_basis="lower_bound"` states the enclosure's lower end as `power`, with both ends in
`power_lower`/`power_upper`, under the same conditional qualification.

### Scalar CUPED

**Method:** CUPED point estimate and standard error

**Guarantee:** `asymptotic`: the standard error conditions on a coefficient estimated from the same data it adjusts; that uncertainty is not propagated, so the interval is slightly narrower than one holding the coefficient fixed ([limitations](../limitations.md#cuped-treats-its-adjustment-coefficient-as-known))

**Evidence status:** `calibration_only`

**Basis and evidence limits:** The ANCOVA `lm` comparison was examined and does not match: increment reports a per-arm Welch delta-method standard error on the log scale with Welch degrees of freedom, whereas ANCOVA uses a pooled residual variance with `N - 3` degrees of freedom. On a heteroskedastic scratch dataset (not frozen) the adjusted difference agreed with the ANCOVA treatment coefficient to about `1e-14` relative at equal allocation and differed by about 0.8% at 25 vs 40 units, and the standard errors differed (0.533 vs 0.553 at equal allocation, 0.533 vs 0.611 at 25 vs 40). No comparison was frozen. [Limitations](../limitations.md#cuped-treats-its-adjustment-coefficient-as-known) Calibration: `EV-CUPED-THETA` (16 cells, 3000 replications each, equal allocation, one homogeneous slope).

**Entry points and parity:** [matrix](capabilities-by-entry-point.md#what-runs-where): `mean-run-cuped-utc-error`, `ratio-run-cuped-utc-error`, `conversion-run-cuped-utc-error`, `retention-run-cuped-utc-error`: `supported`; `quantile-run-cuped-utc-error`: `unfinished` (`66mg`: a quantile has no mean to adjust)


### Mixed-covariate CUPED

**Method:** Mixed-covariate linear CUPED (`Method.cuped_multicovariate`)

**Guarantee:** `asymptotic`: fixed-horizon HC2 uncertainty with a Bell–McCaffrey Student-t approximation; this does not establish finite-sample coverage ([limitations](../limitations.md#mixed-covariate-linear-cuped-is-fixed-horizon-approximate))

**Evidence status:** `campaign_pending`

**Basis and evidence limits:** Preregistered full and reduced coverage profiles exist in `campaign-module:calibration.cuped_linear` and neither has been run. Coverage is not established; runtime support is not certification.

**Entry points and parity:** Four-ingress per-unit parity is enumerated separately in `helper:tests/parity_harness/cases.py` and the capability reference.


### Cluster-robust contrasts

**Method:** Cluster-robust contrasts (disjoint-arm Fieller set, additive Welch, cluster IPTW/AIPW/DML)

**Guarantee:** `asymptotic`: qualified working approximations; neither a Bessel multiplier nor a generic t choice establishes finite-sample validity for arbitrary small, unbalanced, or skewed clusters ([limitations](../limitations.md#cluster-references-are-qualified-working-approximations))

**Evidence status:** `calibration_only`

**Basis and evidence limits:** Hand-derived witnesses plus bounded Monte Carlo cells: `tests/estimation/test_cluster_coverage.py` and `tests/estimation/test_cluster_ratio_coverage.py` use 400 replications, 40 clusters per arm and 20 units per cluster; the ratio file also checks a negative control mean. `tests/estimation/test_adjust_cluster_coverage.py` covers IPTW, AIPW and DML with 400 replications, 80 clusters total and 20 units per cluster; stochastic assignment can give unequal arm counts. These cells do not cover arbitrary small or unbalanced clusters. `sandwich::vcovCL` was unavailable, so no external comparison was attempted. [Limitations](../limitations.md#cluster-references-are-qualified-working-approximations) The complete existing cluster calibration matrices are open (internal trackers `q468`, `h80b`).

**Entry points and parity:** [matrix](capabilities-by-entry-point.md#what-runs-where): `mean-run-cluster-utc-error`, `conversion-run-cluster-utc-error`, `ratio-run-cluster-utc-error`: `supported`; `quantile-run-cluster-utc-error`: `unfinished` (`66mg`: a quantile has no mean to adjust); `total-run-cluster-utc-error`: `source_limited`


### Ratio metrics

**Method:** Ratio metrics (delta method)

**Guarantee:** `asymptotic`: unit-grain ratio lift uses a delta-method log-scale uncertainty against a Welch-Satterthwaite t reference; a first-order approximation is furthest from the truth at small n, for heavy-tailed outcomes and for ratio metrics, and the `ratio_denominator_precision` note is an advisory, not a corrected interval ([limitations](../limitations.md#uncertainty-is-estimated-on-every-path))

**Evidence status:** `calibration_only`

**Basis and evidence limits:** `tests/estimation/test_ratio.py::TestRatioStatisticalValidation::test_se_within_5pc_of_bootstrap` compares the delta SE with 2000 bootstrap replications on 2000 units, within 5%; `tests/estimation/test_ratio.py::test_unclustered_ratio_interval_coverage_is_nominal` measures coverage at a 20% lift over 1500 draws with 500 units per arm, within three Monte Carlo standard errors of 95%. These bounded simulations do not cover arbitrary rates or denominator distributions. No third-party comparison was attempted because `mratios` was unavailable. [Limitations](../limitations.md#signed-effects-require-a-suitable-scale-and-reference)

**Entry points and parity:** [matrix](capabilities-by-entry-point.md#what-runs-where): `ratio-run-none-utc-error`: `supported` (`from_switchback_frame` refuses it as `source_limited`, `source.frame.switchback.metric`)


### Joint relative inversion

**Method:** Joint relative inversion

**Guarantee:** `asymptotic`: relative Fieller set on a fixed t reference for clustered unadjusted arms; preserves bounded, disconnected, one-sided, full-real, empty or unavailable geometry ([limitations](../limitations.md#cluster-references-are-qualified-working-approximations))

**Evidence status:** `derivation_only`

**Basis and evidence limits:** `EV-JOINT-RELATIVE`; no third-party comparison. [Limitations](../limitations.md#signed-effects-require-a-suitable-scale-and-reference)

**Entry points and parity:** See `EV-JOINT-RELATIVE` above.


### Exact-binomial relative-lift sets

**Method:** Exact-binomial relative-lift set (conversion, retention)

**Guarantee:** `exact`: finite-sample relative-risk confidence set and relative-scale decisions for eligible unadjusted unit-grain fixed-horizon conversion and retention (`reference_kind="binomial"`); the additive interval is the asymptotic Miettinen-Nurminen score interval (`EV-BINOMIAL-SCORE-SIZE`) and is not exact, and no validity under optional stopping is claimed. CUPED, ratio-denominator and clustered conversion and retention keep their own approximate routes and do not inherit this class. Under `conversion_inference="auto"`, a prior does not change count-based sampling routing: sparse counts retain exact binomial sampling inference even when the separate working-model posterior is unavailable under its existing guards. An explicit `conversion_inference="finite_sample"` request with a prior refuses; see [binary posterior availability](bayesian-results.md#binary-metrics-and-path-coverage). An Encouragement ITT on a conversion metric keeps this route (ITT and LATE readouts of a retention metric under Encouragement are refused with `readout.encouragement.retention` on the constructors that reach the readout, `from_frame(shape="summary")` refuses earlier with `source.frame.constructor`, and `from_switchback_frame` cannot express an Encouragement design). Open numerical items (internal trackers): `1tce` (certificate margin above about 2^26 per arm), `fd17` (endpoint-inversion error not yet bounded and disclosed), `9af7` (latency near one million units per arm), `rsf0` (usability above the exact-route size ceiling); an open item is not a demonstrated coverage failure and is not described as repaired ([limitations](../limitations.md#conversion-and-retention-arms-dense-counts-use-the-delta-method-sparse-counts-are-estimated-exactly-not-refused))

**Evidence status:** `calibration_only`

**Basis and evidence limits:** Enumeration and a Bonferroni Clopper-Pearson rectangle accompany `EV-RARE-EVENT-GRID`: six prespecified Monte Carlo coverage cells, not all 336 manifest cells. Only the single-arm Clopper-Pearson primitive has a third-party comparison. [Limitations](../limitations.md#conversion-and-retention-arms-dense-counts-use-the-delta-method-sparse-counts-are-estimated-exactly-not-refused) Further evidence: `EV-RARE-EVENT-GRID` (336-cell manifest) and `EV-CP-BINOM-TEST` (single-arm primitive).

**Entry points and parity:** [matrix](capabilities-by-entry-point.md#what-runs-where): `conversion-run-none-utc-error`, `retention-run-none-utc-error`: `supported`


### Asymptotic sequential means

**Method:** Sequential asymptotic mean confidence sequence

**Guarantee:** `asymptotic`: the public mean, adjusted-mean and ratio routes provide asymptotic confidence sequences and family approximations; estimated variance does not preserve the exact e-value martingale argument ([limitations](../limitations.md#stopping-date-guarantees-require-the-registered-reveal-contract))

**Evidence status:** `campaign_pending`

**Basis and evidence limits:** The complete empirical campaign has not been executed. [Limitations](../limitations.md#the-sequential-certification-campaign-has-not-been-executed) Bounded evidence exists for three laws at one design: `tests/estimation/test_asymptotic_joint.py::test_time_uniform_miscoverage_across_repeated_looks` (slow, parameter recovery) runs 300 replications per law (adjusted-mean, ratio-mean and adjusted-ratio-mean; Normal covariates, treatment effect 0.5), adding 50 units per arm (100 total) at each of six looks, and requires an exact one-sided Clopper-Pearson upper bound on the probability that any look's set misses the truth, at a Monte Carlo family error budget of 0.01 over the three laws, of at most 0.055 (the nominal 0.05 plus the repository's scientific tolerance); a fast twin (`tests/estimation/test_asymptotic_joint.py::test_time_uniform_coverage_smoke`) runs 12 replications of three looks per law. That is one design per law over six looks, not a statement about other laws, horizons, effect sizes or distributions. Other bounded release cells are not inventoried here: `unknown`.

**Entry points and parity:** [matrix](capabilities-by-entry-point.md#what-runs-where): bounded observation windows: `windowed_mean-run-sequential-utc-error` and `windowed_ratio-run-sequential-utc-error` are `supported` on `from_definitions`, `from_unit_day_artifact`, `from_frame(shape="panel")` and `from_moments`, and `source_limited` on `from_frame(shape="summary")` and `from_switchback_frame`; unbounded observation windows: `mean-run-sequential-utc-error` and `ratio-run-sequential-utc-error` are `unfinished` on `from_definitions`, `from_unit_day_artifact`, `from_frame(shape="summary")` and `from_frame(shape="panel")` (`1cr4`: every observation window must be bounded by the common registered reveal window) and `source_limited` on `from_moments` and `from_switchback_frame`. These cells declare no CUPED method and the matrix does not probe option interactions, so the adjusted-mean and adjusted-ratio sequential laws have no matrix cell: `unknown`. The scenario case `sequential_asymptotic_mean_revenue` (not mapped to a matrix cell) is parity-compared on `from_definitions`, `from_unit_day_artifact`, `from_frame(shape="summary")`, `from_frame(shape="panel")` and `from_moments`; it waives `from_switchback_frame`. A bounded retention metric on this asymptotic route is enumerated as 64 variant cells (the retention and windowed-retention `sequential` cells re-run with `inference = "asymptotic_mean"`, ids ending `-asymptotic_mean`, counts in `helper:tests/parity_harness/COVERAGE.md`; `tests/test_parity_matrix.py::test_asymptotic_retention_cell`): 12 are `supported`, in the `run` cells `from_definitions`, `from_unit_day_artifact`, `from_frame(shape="panel")` and `from_moments` agree while `from_frame(shape="summary")` is `source_limited` (it refuses `source.frame.constructor`), and every ingress that captures sequential state retains the scalar-mean law


### Bernoulli e-processes

**Method:** Bernoulli e-process routes

**Guarantee:** `exact`: finite-sample anytime-valid derivation under the registered model, assignment and reveal assumptions. The deployed numerical implementation is verified separately (internal tracker `xeth`, open). The campaign not having been executed does not change this class ([limitations](../limitations.md#the-sequential-certification-campaign-has-not-been-executed))

**Evidence status:** `campaign_pending`

**Basis and evidence limits:** A finite-sample derivation exists under the registered assumptions; verification of the deployed implementation is a separate obligation. [Limitations](../limitations.md#the-sequential-certification-campaign-has-not-been-executed) Finite Bernoulli witnesses independent of the deployed code (`helper:tests/estimation/_sequential_proof_oracle.py`, used by `tests/estimation/test_sequential_proof_acceptance.py`) are `independent_derivation_in_repo`; they do not certify the full campaign, arbitrary horizons or other observation models.

**Entry points and parity:** [matrix](capabilities-by-entry-point.md#what-runs-where): bounded observation windows: `windowed_conversion-run-sequential-utc-error` is `supported` on `from_definitions`, `from_unit_day_artifact`, `from_frame(shape="panel")` and `from_moments`, and `source_limited` on `from_frame(shape="summary")` and `from_switchback_frame`; `retention-run-sequential-utc-error`, which declares the bounded `(3, 5)` retention band, is `supported` on `from_definitions`, `from_unit_day_artifact`, `from_frame(shape="panel")` and `from_moments`, and `source_limited` on `from_frame(shape="summary")` (a summary carries no dates) and `from_switchback_frame`; unbounded observation windows: `conversion-run-sequential-utc-error` is `unfinished` on `from_definitions`, `from_unit_day_artifact`, `from_frame(shape="summary")` and `from_frame(shape="panel")` (`1cr4`: every observation window must be bounded by the common registered reveal window) and `source_limited` on `from_moments` and `from_switchback_frame`; an unbounded retention metric (`RetentionMetric(threshold_days=int)`) has no matrix cell and no recorded parity disposition (`unknown`): warehouse capture (`from_definitions`) is tested to refuse it with `sequential.route.unsupported`, and `from_frame(shape="summary")` refuses every retention metric with `source.frame.constructor`; multi-arm monitoring of a bounded retention metric has no parity case (`unknown`; the multi-arm case covers conversion only)


### Sequential selection and intervals

**Method:** e-BH and sequential selected intervals

**Guarantee:** `exact` only when every family member has a finite-sample Bernoulli construction; `asymptotic` for the whole family when any member is asymptotic, including selected Bernoulli rows. Family validity requires the declared common joint-unit filtration, and selected intervals are reinverted at `min(q * R / m, nominal alpha)` ([limitations](../limitations.md#sequential-selected-intervals-use-the-same-stopped-likelihood))

**Evidence status:** `derivation_only`

**Basis and evidence limits:** No third-party implementation of the same procedure was found. In-repository evidence is an exhaustive enumeration, not a simulation: `tests/estimation/test_sequential_public_proof_acceptance.py::test_public_mixed_family_predictable_stop_bounds_stopped_fdr_and_fcr` (slow) replays every positive-probability path of independent Bernoulli reveals through the public selection and selected-interval route for one two-cell family (a null cell at ratio 1 and a non-null cell at ratio 2 against a control success rate of 1/2, the non-null treatment always succeeding), horizon 4, `q = 1/3`, nominal interval alpha 1/2 and a stop decided one reveal earlier, and asserts in exact rationals that the stopped false discovery rate and the selected-set noncoverage rate are positive and at most `q`, and that each selected set equals the exact e-BH set. It covers one family, one horizon and one stopping rule; it does not show the bound for other families, asymptotic members or dependence structures. [Limitations](../limitations.md#sequential-selected-intervals-use-the-same-stopped-likelihood)

**Entry points and parity:** [matrix](capabilities-by-entry-point.md#what-runs-where): no matrix row (the matrix has no multiplicity axis)


### Fixed-horizon selected intervals

**Method:** Fixed-horizon FCR-adjusted intervals

**Guarantee:** `exact` for the Benjamini-Hochberg selection step only when every family member's p-value is finite-sample valid, and under independence or positive dependence (PRDS), a condition that is not checked; `asymptotic` for a family containing an asymptotic Welch or delta-method p-value (the weakest route in the family); the guarantee of the adjusted intervals themselves: none stated (no limitations entry states it) ([limitations](../limitations.md#fixed-horizon-fdr-control-assumes-a-dependence-condition-that-is-not-checked))

**Evidence status:** `calibration_only`

**Basis and evidence limits:** [Limitations](../limitations.md#fixed-horizon-fdr-control-assumes-a-dependence-condition-that-is-not-checked) The BH selection step alone has a third-party comparison: `EV-BH-P-ADJUST`. Bounded Monte Carlo evidence for the adjusted one-sided intervals (slow): `tests/estimation/test_fcr_one_sided.py::test_one_sided_fcr_lower_bound_realized_fcr_is_controlled` draws 100,000 replications of four independent unit-variance Normal statistics (one signal of 2, 4 or 10 and three nulls), selects with `bh_select` at `q = 0.1`, and requires the realized fraction of selected intervals missing their truth to be at most `q` plus four Monte Carlo standard errors; `tests/estimation/test_fcr_one_sided.py::test_fixed_fcr_exact_tail_when_all_signals_are_selected` checks that, when every hypothesis is selected, the miss fraction equals its nominal marginal rate within four Monte Carlo standard errors. Both use independent known-scale statistics and one family size, so they say nothing about dependence, estimated variances or other family sizes.

**Entry points and parity:** [matrix](capabilities-by-entry-point.md#what-runs-where): no matrix row (the matrix has no multiplicity axis)


### Winsorization

**Method:** Winsorization inference

**Guarantee:** `experimental`: typed `status="experimental"` on `WinsorInferenceSpec` with a persisted `qualification` (`pointwise_asymptotic_model_conditioned_v1` for the bootstrap, `uniform_support_conditioned_v1` for the rank method); the descriptive confidence set is reported and decision evidence is withheld (`evidence.experimental_reference`) ([metric types](../guides/metric-types.md#winsorization))

**Evidence status:** `calibration_only`

**Basis and evidence limits:** Typed status `experimental`; 12 alternative rows of the preserved stress grid remain unresolved. [Metric types](../guides/metric-types.md#winsorization)

**Entry points and parity:** [matrix](capabilities-by-entry-point.md#what-runs-where): `mean-run-winsor_percentile-utc-error`: `supported` (`from_frame(shape="panel")` and `from_moments` refuse it as `source_limited`, `evidence.raw_outcomes_unavailable`); `conversion-run-winsor_percentile-utc-error`, `retention-run-winsor_percentile-utc-error`, `ratio-run-winsor_percentile-utc-error`, `quantile-run-winsor_percentile-utc-error`: `construction_limited`


### Encouragement

**Method:** Encouragement ITT, compliance and LATE

**Guarantee:** `asymptotic` for the clustered LATE additive interval (Welch-Satterthwaite t reference over the two arms' cluster-ratio variance components); an ITT on a conversion metric keeps the exact-binomial route above; ITT and LATE readouts of a retention metric under Encouragement are refused (`readout.encouragement.retention`), while a compliance-only request (`estimands=("compliance",)`) ignores outcome configuration and runs where the constructor accepts the declaration; every other component-based reference: none stated ([limitations](../limitations.md#cluster-references-are-qualified-working-approximations))

**Evidence status:** `calibration_only`

**Basis and evidence limits:** Component-based references and bounded LATE coverage cells in `tests/estimation/test_encouragement_recovery.py`: additive LATE with one-sided uptake, additive LATE with two-sided uptake, and complier-relative LATE with one-sided uptake; no external comparison. This evidence does not certify every encouragement component or design. [Encouragement guide](../guides/encouragement.md) The clustered LATE additive interval is additionally calibrated (`tests/calibration/test_cluster_late_welch_df.py`, 2000 replications per cell over 5/5, 5/25 and 10/50 treatment/control clusters): with equal cluster sizes it covers 94.3-96.0% at nominal 95%, and log-normally dispersed sizes cover 88.5-92.7%. That calibration is not an external comparison.

**Entry points and parity:** [matrix](capabilities-by-entry-point.md#what-runs-where): no matrix cell: the scenario parity cases that cover encouragement are not mapped to cells (`helper:tests/parity_harness/COVERAGE.md`)


### Switchback unit-cycle t

**Method:** Switchback unit-cycle t reference

**Guarantee:** `asymptotic`: the default `UnitCycleTApproximation` (sample variance, `dof = n_units - 1`) is not a claim of finite-sample coverage and relies on approximately Normal unit means, a condition outside the declared assumptions ([limitations](../limitations.md#the-switchback-t-reference-is-itself-an-approximation))

**Evidence status:** `calibration_only`

**Basis and evidence limits:** An analytic lower bound on the miscoverage of the unit-t pivot in one small-N cell (4 units, 1 cycle, CT probability 0.9) is 0.0565 at effect 0 and 0.0564 at effect 1 against a nominal 0.05, pinned by `tests/simulate/test_unit_cycle_calibration.py::test_corrected_counterexample_and_corrected_envelope`; it is a derivation that contradicts nominal coverage there. Two slow parameter-recovery cells run the default independent unit-cycle path end to end (`tests/simulate/test_switchback.py::test_supported_switchback_calibration_is_covered` and `tests/simulate/test_switchback.py::test_switchback_null_effect_type_i_calibration`: 120 units, 4 cycles, 24 replications each, treatment effects 2.0 and 0.0); each asserts only that unconditional coverage lies between 0.80 and 1.0 and that bias is below 0.15, a weak floor on a small replication count that does not establish nominal coverage.

**Entry points and parity:** [matrix](capabilities-by-entry-point.md#what-runs-where): `mean-run-none-utc-error` and `conversion-run-none-utc-error`: `from_switchback_frame` is `supported`; `ratio-run-none-utc-error`, `retention-run-none-utc-error` and `quantile-run-none-utc-error`: `from_switchback_frame` is `source_limited` (`source.frame.switchback.metric`)


### Switchback variance envelope

**Method:** Switchback unit-cycle variance envelope

**Guarantee:** `exact`: a valid prospective `UnitCycleVarianceEnvelope` gives a finite-sample bound under the declared prospective residual-variance assumption, which must be justified externally; a fitted pilot variance is not a valid envelope. No open deployed-numerics tracker is recorded for this route in this inventory ([limitations](../limitations.md#the-switchback-t-reference-is-itself-an-approximation))

**Evidence status:** `campaign_pending`

**Basis and evidence limits:** The preselected `RELEASE_CELLS` check (`tests/simulate/test_unit_cycle_calibration.py`, 1,536 repetitions per checkpoint) evaluates an envelope procedure, and its thresholds are acceptance bands the campaign is written to enforce, not returned results ([switchback guide](../guides/switchback.md#representative-release-checks)): no coverage, availability or power figure is recorded. The 4,320-cell manifest `helper:tests/unit_cycle_prospective.json` declares independent Bernoulli unit-cycle orders only, and complete unit-cycle research matrices remain uncertified.

**Entry points and parity:** [matrix](capabilities-by-entry-point.md#what-runs-where): no matrix cell declares a `UnitCycleVarianceEnvelope`; the cells in the unit-cycle t reference row run the default reference


### Switchback shared-block t

**Method:** Switchback shared-block t reference

**Guarantee:** `asymptotic`: block-level Student-t reference with `n_blocks - 1` degrees of freedom; the independent replicate is the block, not the roster unit, so more roster units neither add replication nor establish finite-sample coverage ([limitations](../limitations.md#switchback-inference-requires-independent-units-or-blocks))

**Evidence status:** `calibration_only`

**Basis and evidence limits:** One slow parameter-recovery cell exercises the shared-schedule block-t path (`tests/simulate/test_switchback.py::test_shared_schedule_block_t_refuses_degenerate_cycle_order_and_recovers_coverage`): 3 units, 6 blocks, `probability_ct=0.8`, 400 replications, null treatment effect. About a quarter of replications realize a single cycle order and are refused (`source.frame.switchback.schedule`); the asserted bound is coverage of at least 0.85 conditional on an interval being produced, a floor and not nominal coverage. No other shared-block cell is inventoried, and the independent unit-cycle calibration cells declare independent Bernoulli unit-cycle orders only.

**Entry points and parity:** [matrix](capabilities-by-entry-point.md#what-runs-where): no matrix row (the matrix declares independent Bernoulli orders only)


### Parallel-arm planning

**Method:** Sample-size, power and MDE solvers: log-ratio planning model

**Guarantee:** `planning approximation`: the log-ratio Normal/noncentral-t planning model with a planned variance (`power_basis="asymptotic"`) behind the parallel-arm solvers (`required_sample_size`, `achieved_power`, `minimum_detectable_effect`, `power_curve`) for every fixed-horizon plan the runtime does not decide with the exact binomial test; it plans from declared baselines and alternatives and is not an inferential guarantee, and a quantile pilot is one draw whose projected variance is not a certified bound. Sequential, switchback, unit-cycle and segment planners have their own rows ([limitations](../limitations.md#arm-power-planning-depends-on-declared-alternative-arm-variance-shapes), [power guide](../guides/power-analysis.md), [quantile planning](../limitations.md#a-quantile-metrics-planned-variance-is-a-projection-from-the-pilot-not-a-certified-bound))

**Evidence status:** `calibration_only`

**Basis and evidence limits:** The planning scale is the log relative lift with a planned variance, whereas `power.t.test` is an additive pooled-variance, equal-allocation calculation; the estimands do not match and no estimator was added to force a match, so no third-party comparison exists. Bounded Monte Carlo evidence is one slow cell (`tests/power/test_calibration.py::TestCalibration::test_power_calibration`: a Negative Binomial count at a 10% lift planned for 0.80 power, 200 replications through the full pipeline, empirical power within 0.07 of the design-aware analytic power, and planning power at most 0.05 above it); it is not a calibration grid. [Limitations](../limitations.md#arm-power-planning-depends-on-declared-alternative-arm-variance-shapes)

**Entry points and parity:** [matrix](capabilities-by-entry-point.md#what-runs-where): no matrix row (planning is not a readout)


### Binomial decision planning

**Method:** Sample-size, power and MDE solvers: binomial decision planning

**Guarantee:** `planning approximation`: an unadjusted, unclustered, fixed-horizon conversion or retention plan may enumerate the runtime's rejection decisions at the planned counts. `power_basis="exact"` describes exact enumeration arithmetic for the decision map, not a universal finite-sample statistical guarantee: with `conversion_inference="auto"`, the map may be hybrid delta-method and Berger–Boos, as well as finite-sample-only. A point is returned when its computed enclosure resolves to `1e-6`; an enclosure the evaluation budgets leave wider is stated by its lower end (`power_basis="lower_bound"`, both ends in `power_lower`/`power_upper`), and a size or effect reaches a target only when the lower end does. Neither this planning model nor the separate asymptotic planning model is an inferential guarantee ([power guide](../guides/power-analysis.md), [limitations](../limitations.md#arm-power-planning-depends-on-declared-alternative-arm-variance-shapes))

**Evidence status:** `derivation_only`

**Basis and evidence limits:** Exact enumeration is compared with runtime decisions (`tests/power/test_binomial_planning.py::TestExactRouteMatchesRuntime`). The default hybrid route is exercised at borderline sizes in `tests/power/test_hybrid_planning.py::TestBorderlinePlansAreTheUnionOfBothRoutes::test_the_classification_is_borderline_and_the_figure_exact`; these are in-repository decision comparisons, not a third-party tool.

**Entry points and parity:** [matrix](capabilities-by-entry-point.md#what-runs-where): no matrix row (planning is not a readout)


### Sequential planning

**Method:** Sample-size, power and MDE solvers: sequential planning (`GaussianScoreMixture` boundary)

**Guarantee:** `planning approximation`: the probability that the planned asymptotic-mean boundary is crossed by a drifted Normal path with a planned variance; it sizes a sequential design and is not an inferential guarantee ([planning guide](../guides/power-analysis.md#fixed-and-sequential-planning), [limitations](../limitations.md#arm-power-planning-depends-on-declared-alternative-arm-variance-shapes))

**Evidence status:** `calibration_only`

**Basis and evidence limits:** The crossing probability is compared with integrals that share no code with `increment.power.sequential` (`helper:tests/power/_references.py`): closed-form one-look tails, a two-look conditional-normal integral by `scipy.integrate.quad`, and a Simpson-rule recursion for three or more looks (`tests/power/test_sequential_evaluator.py::TestOneLookExactTails`, `tests/power/test_sequential_evaluator.py::TestTwoLookQuadReference`, `tests/power/test_sequential_evaluator.py::TestMultiLookSimpsonReference`). The MDE inversion is checked against the Simpson power at its pinned candidate effects for 5 and 14 looks (`tests/power/test_sequential_inverse.py::TestOrdinaryInversions`, slow). The plan-time boundary is held equal to the runtime's asymptotic-mean boundary (`tests/estimation/test_asymptotic_mean_planning_equivalence.py`). One slow cell runs 4000 replications of Normal arms through that runtime boundary (`tests/power/test_asymptotic_mean_sizing.py::test_planned_n_achieves_nominal_power_under_the_runtime_boundary`: 14 looks, 5% lift, equal arm variances as planned) and asserts empirical power of at least 0.80 minus three Monte Carlo standard errors at the planned size; it is one design. The status is `calibration_only` because bounded Monte Carlo evidence sits beside the in-repository derivations; no third-party tool is involved.

**Entry points and parity:** [matrix](capabilities-by-entry-point.md#what-runs-where): no matrix row (planning is not a readout)


### Switchback planning

**Method:** Sample-size, power and MDE solvers: switchback planning (`switchback_achieved_power`, `switchback_minimum_detectable_effect`, `switchback_required_blocks_or_units`)

**Guarantee:** `planning approximation`: moment-t noncentral-t power (`power_kind="moment_t_approximation"`) conditioned on fitted pilot estimates; it is not a finite-branch rejection law, does not account for the estimation uncertainty of the pilot, and certifies no finite-sample calibration ([limitations](../limitations.md#switchback-planning-requires-its-own-model))

**Evidence status:** `calibration_only`

**Basis and evidence limits:** The noncentral-t power is compared with direct numerical integration over its chi factor and with analytic quadratic roots at hand-computed two-branch moments (`tests/power/test_switchback.py::test_independent_branch_oracle_and_two_analytic_roots`), and the two-period case with a closed form (`tests/power/test_switchback.py::test_two_period_power_matches_the_one_dof_oracle_and_replays`). Shared-schedule power scaled by the admission probability is checked against a 200,000-replication simulation of the two-stage process within four Monte Carlo standard errors (`tests/power/test_switchback.py::test_shared_schedule_power_available_matches_bounded_simulation`, slow). Four pytest sentinels check the planning variance against the actual source and estimator under a frozen bounded law; the full 32-cell grid runs with `python -m calibration.mc_safety switchback --out switchback.jsonl`. Neither gates coverage or power. The status is `calibration_only` because bounded Monte Carlo evidence sits beside the in-repository derivations; no third-party tool is involved.

**Entry points and parity:** [matrix](capabilities-by-entry-point.md#what-runs-where): no matrix row (planning is not a readout)


### Unit-cycle power lower bound

**Method:** Sample-size, power and MDE solvers: switchback unit-cycle power lower bound (`unit_cycle_power_lower_bound`)

**Guarantee:** `planning approximation`: a supplementary Cantelli lower bound on the rejection probability under the declared prospective residual-variance envelope, which must be justified externally; it is a lower bound for planning, not an inferential guarantee, and a zero bound is not an unattainability claim ([limitations](../limitations.md#the-switchback-t-reference-is-itself-an-approximation))

**Evidence status:** `derivation_only`

**Basis and evidence limits:** The bound is recomputed in exact rationals (`tests/power/test_unit_cycle_power.py::test_lower_bound_matches_rational_cantelli_and_keeps_effect_scale`), and the unit-cycle law moments are checked against an independent full-mask enumeration (`tests/power/test_unit_cycle_power.py::test_moments_match_independent_full_mask_enumeration`). In-repository comparisons with no third-party tool.

**Entry points and parity:** [matrix](capabilities-by-entry-point.md#what-runs-where): no matrix row (planning is not a readout)


### Segment heterogeneity planning

**Method:** Sample-size, power and MDE solvers: segment heterogeneity (`segment_pairwise_*`, `joint_q_power_fixed`, `joint_q_power_random`)

**Guarantee:** `planning approximation`: Cochran's Q power at the declared segment variances; the random-effects form is exact only when every variance is equal and a mean-matched approximation otherwise (its docstring reports up to +12.8% relative error at a 50-fold variance spread), so it is not a calibrated design tool at very unequal variances

**Evidence status:** `calibration_only`

**Basis and evidence limits:** `joint_q_power_fixed` is compared with a 2000-replication simulation at one unequal-variance cell (absolute difference below 0.05, `tests/test_hte_calibration.py::TestPowerSolverCalibration`); `joint_q_power_random` with an 80,000-replication simulation at one cell with an 8-fold variance spread (relative difference below 5%, `tests/power/test_core.py::TestJointQPower::test_random_effects_fixed_effects_agree_on_calibration`); the pairwise sizing formula is recomputed in closed form and its achieved power is held within one percentage point of the target (`tests/test_hte_calibration.py::test_pairwise_power_non_partition_shares`). [Heterogeneity guide](../guides/heterogeneity-and-rollout.md)

**Entry points and parity:** [matrix](capabilities-by-entry-point.md#what-runs-where): no matrix row (planning is not a readout)


### Logged-policy evaluation

**Method:** Off-policy evaluation of logged decisions

**Guarantee:** `asymptotic`: the interval is asymptotic in the number of units, under a fixed logger only; adaptive loggers are refused ([limitations](../limitations.md#off-policy-evaluation-of-logged-decisions-is-calibrated-only-under-a-fixed-logger))

**Evidence status:** `calibration_only`

**Basis and evidence limits:** Fixed-logger coverage cells; adaptive loggers are refused. [Limitations](../limitations.md#off-policy-evaluation-of-logged-decisions-is-calibrated-only-under-a-fixed-logger) Fixed-logger calibration: 26 cells and 9,188 emitted intervals by band of the smallest per-index effective sample size (`tests/calibration/test_logged_policy_ess_floor.py`), and nine fixed-logger cells with coverage 0.931 to 0.960 (`tests/calibration/test_logged_policy_adaptive.py`).

**Entry points and parity:** [matrix](capabilities-by-entry-point.md#what-runs-where): no matrix row: the family runs on no experiment builder path and has its own ingress (`LoggedTrace.from_records`, `from_frame`)


### Positivity and weight diagnostics

**Method:** Positivity gate and weight diagnostics

**Guarantee:** none stated: a fixed `[0.01, 0.99]` propensity threshold; passing it is not evidence of good overlap ([limitations](../limitations.md#the-positivity-gate-is-a-fixed-threshold-not-an-overlap-assessment))

**Evidence status:** `none_found`

**Basis and evidence limits:** A fixed threshold, not an overlap assessment; effective sample size and maximum weight share are diagnostics, not a sensitivity analysis. [Limitations](../limitations.md#the-positivity-gate-is-a-fixed-threshold-not-an-overlap-assessment)

**Entry points and parity:** [matrix](capabilities-by-entry-point.md#what-runs-where): no matrix row for the gate itself


### Total, active and retention metrics

**Method:** Total, active and retention metric types

**Guarantee:** retention: no type-wide guarantee class; each route states its own. Standalone exact-binomial inference (`EV-CP-BINOM-TEST`) and eligible unadjusted, unit-grain, fixed-horizon, prior-free relative-risk inference are `exact` (exact-binomial row); sequential routes are `exact` or `asymptotic` by family (Bernoulli e-process and e-BH rows); the adjusted intervals of fixed-horizon FCR families state no guarantee (FCR row); other routes keep their own qualifications. Total and active: none stated (no limitations entry states a class)

**Evidence status:** `none_found`

**Basis and evidence limits:** No frozen third-party comparison exists for these metric types. [Compatibility](../guides/compatibility.md)

**Entry points and parity:** [matrix](capabilities-by-entry-point.md#what-runs-where): `retention-run-none-utc-error`: `supported`; `total-run-none-utc-error`, `active-run-none-utc-error`: `not_expressible` (report-layer metric types: a frame names no such type)


### CATE and targeting

**Method:** CATE estimation and targeting (`estimate_cate`, `validate_cate`, `targeting_rule`, `select_targeting_rule`)

**Guarantee:** none stated: no finite-sample guarantee; identification requires conditional ignorability and overlap for `Observational` scores, group-effect and characteristic tests are Bonferroni-corrected inside their own families, and no guarantee spans the whole workflow ([limitations](../limitations.md#targeting-validation-depends-on-identification-and-overlap), [no joint error control](../limitations.md#the-targeting-workflow-carries-no-joint-error-control)); clustered `estimate_cate` is cluster-asymptotic ([limitations](../limitations.md#clustered-cate-uncertainty-is-cluster-asymptotic))

**Evidence status:** `calibration_only`

**Basis and evidence limits:** `EV-CATE-UNCLUSTERED-CAL` and `EV-CATE-CLUSTER-SCREEN` are bounded Monte Carlo cells; no third-party comparison is recorded.

**Entry points and parity:** [matrix](capabilities-by-entry-point.md#what-runs-where): no matrix row (the matrix has no CATE axis)


## Ordinary-user journey contracts and current evidence

These journeys describe user-facing contracts. A named assertion supports only the behavior it
exercises; its presence does not establish untested parts of a journey.

For matched-arm inputs, the parity harness evaluates `from_definitions`,
`from_unit_day_artifact`, `from_frame(shape="summary")`, `from_frame(shape="panel")`, and `from_moments` by matrix cell;
only `supported` cells assert matching output. A constructor being present is not itself evidence
of support. `source_limited`, `construction_limited`, `not_expressible`, `unfinished`, and
`unsound` retain their meanings in the matrix above. `from_switchback_frame` has its own
contrast/support journey and is never compared as an equivalent arm estimator. Trigger-declared
parity is not implied by the existing matched-arm matrix.

### 1. Two-arm mean dataframe

**Contract:** **1. Two-arm mean dataframe.** Show effect and honest interval/reference; disclose a declared plan without treating an unadjusted `role=None` result as exploratory.

**Existing evidence:** `tests/test_frame_summary.py::test_mean_lift_matches_hand_computed_ratio_of_means`

**Limit:** The assertion covers a basic mean result, not plan-status disclosure.


### 2. Matched constructors

**Contract:** **2. Matched constructors.** Equivalent supported per-unit input produces equivalent estimates and retained sequential state; warehouse query work stays outside estimation. Switchback has its own contrast.

**Existing evidence:** `tests/test_parity_harness.py::test_parity_case[mean_ratio_conversion_cuped_fixed_horizon]`; `tests/test_parity_matrix.py`; `helper:tests/parity_harness/matrix.py`

**Limit:** Only `supported` matched-arm matrix cells assert parity; switchback is not compared with those estimands.


### 3. Planned primary and guardrail

**Contract:** **3. Planned primary and guardrail.** Every required cell is an estimate or structured failure; partial output cannot appear complete.

**Existing evidence:** No named assertion in this inventory establishes the complete-result contract.

**Limit:** Existing behavior is not credited without an assertion covering failure and missing cells.


### 4. Informative prior

**Contract:** **4. Informative prior.** Posterior output is distinct from sampling significance; a prior may change the posterior but not the sampling calculation.

**Existing evidence:** `tests/estimation/test_conversion_route.py::TestExplicitFiniteSample::test_an_informative_prior_is_refused_by_code_instead_of_served_by_the_delta_method`; `tests/estimation/test_conversion_route.py::TestExplicitFiniteSample::test_the_prior_is_served_by_auto_on_the_delta_method_path_as_before`

**Limit:** These conversion assertions do not establish the mean/readout contract.


### 5. CUPED

**Contract:** **5. CUPED.** An explicit `Method(name="cuped", variance_reduction="cuped")` uses a pre-assignment covariate; pre-trigger timing alone is insufficient.

**Existing evidence:** `tests/test_frame_summary.py::test_metric_spec_roles_drive_run_dispatch`; `tests/test_frame_summary.py::test_cuped_corrects_pre_period_imbalance`

**Limit:** These checks do not establish assignment-anchored source eligibility.


### 6. Integrity

**Contract:** **6. Integrity.** Cumulative assigned counts are checked under a declared independent constant-allocation scheme; unsupported inputs are explicit.

**Existing evidence:** Existing diagnostic component assertions do not establish automatic `run(experiment, reader)` behavior.

**Limit:** The automatic behavior is not credited by these component checks.


### 7. Dense conversion

**Contract:** **7. Dense conversion.** Automatic inference uses its efficient dense route and matched reference under a prespecified efficiency budget.

**Existing evidence:** `tests/estimation/test_conversion_route.py::TestRowsCarryTheirRoute::test_dense_counts_take_the_delta_method_route_and_say_so`; `tests/estimation/test_conversion_route_calibration.py::TestSmoke::test_a_dense_cell_takes_the_delta_method_route_and_covers_each_tail`

**Limit:** Route assertions are not planning benchmarks or measured speed evidence.


### 8. Sparse, small, or skewed inputs

**Contract:** **8. Sparse, small, or skewed inputs.** Partial/null rows retain exact reasons; approximation limits and ratio counterexamples remain visible.

**Existing evidence:** `tests/estimation/test_conversion_route.py::TestRowsCarryTheirRoute::test_dense_counts_take_the_delta_method_route_and_say_so`; `tests/estimation/test_conversion_route.py::TestRowsCarryTheirRoute::test_finite_sample_mode_labels_a_dense_row_binomial`

**Limit:** These route checks do not establish the full partial-result contract or the ratio counterexample disposition.


### 9. Repeated conversion

**Contract:** **9. Repeated conversion.** Preserve continuation identity and disclose reveal and assignment assumptions.

**Existing evidence:** `tests/test_sequential_public_sources.py::test_frame_zero_event_prefix_append_and_rewrite_do_not_reset_process`

**Limit:** This checks conversion continuation; `tests/test_parity_harness.py::test_parity_case[sequential_asymptotic_mean_revenue]` covers a mean case, not conversion.


### 10. Save, reopen, and consume

**Contract:** **10. Save, reopen, and consume.** Replay live fixed/sequential state, expose concise certificate references, and exercise actual refusal routes.

**Existing evidence:** `tests/test_sequential_public_sources.py::test_frame_current_wire_and_moments_replay_same_actual_decision`; `tests/test_sequential_public_sources.py::test_current_envelope_rejects_relabeling_duplicate_rows_and_legacy`; `tests/test_realistic_warehouse.py::test_export_from_moments_round_trips_run_exactly`; `tests/test_realistic_warehouse.py::test_realistic_warehouse_artifact_replay_matches_native`; `tests/test_realistic_warehouse.py::test_export_round_trips_registered_native_actual_state`; `tests/test_readout_public_scope.py::test_missing_arm_for_one_metric_is_a_typed_cell_failure_while_another_observes_it`; `tests/test_unit_day_artifact_parity.py::test_cluster_extension_spine_parity_matches_native_for_late_enrolled_treatment`; `tests/test_unit_day_artifact_parity.py::test_boolean_dimension_labels_match_native_and_string_case_is_preserved`; `tests/test_experiment_health.py::test_health_missing_certificates_are_unknown`

**Limit:** The realistic demo's in-memory warehouse, artifact, and moments replays establish matching results; the moments path also establishes source-bound identities and family procedure/size parity. It checks available unequal-allocation health evidence, while feed completeness is not applicable for this source. A separate artifact witness exercises a late-enrolled treatment; the nullable boolean dimension-label test preserves null labels, not nullable optional-covariate behavior. Missing feed certificates remain unavailable. Failed-cell replay preserves reason and incomplete scope. These checks do not establish a dataframe replay of the same source or certificate presentation.


### 10a. Messy-data sequential parity

**Contract:** **10a. Messy-data sequential parity.** Exercise the messy event rows through single-horizon constructor parity across supported ingresses, plus two native-to-moments checkpoint replays at two looks.

**Existing evidence:** `tests/test_parity_harness.py::test_parity_case[audit-sequential-messy-mean-conversion]`

**Limit:** The witness derives 60/40 assignment counts and late/missing feed slices from the parity dataset. Its declared CUPED `pre_revenue` covariate has two missing values; unit-summary and panel construction exercise the configured imputation. The checkpoint probe checks semantic parity and exact switchback refusal (`source.frame.switchback.identification`). Panel sequential capture with `window_days=None` is refused as requiring a bounded window (`sequential.route.unsupported`); this is distinct from constructor refusal `frame.validation.from_unit_panel` for a CUPED metric with a finite window. A non-finalized capture is refused (`sequential.source.invalid`), so it does not claim replay of an immature checkpoint with unavailable reasons.


### 11. Trigger-declared analysis

**Contract:** **11. Trigger-declared analysis.** Anchor population/windows at trigger without changing assignment grain or pre-assignment adjustment; parity applies only to supported inputs.

**Existing evidence:** The parity runner includes `analysis_population` in row identity.

**Limit:** Existing matched-arm assertions do not establish trigger-declared producer behavior or a complete dashboard journey.


The public dataframe example can be exported with `uv run --extra demo --extra tables --extra dashboard python -m marimo export html examples/analysis_from_a_dataframe.py -o /tmp/increment-analysis-from-a-dataframe.html`; this smoke alone does not establish the journey contracts above. The narrow mean assertion in row 1 is not evidence for other journeys. Build documentation with `nox -s docs`.

## Mixed-covariate linear CUPED campaign

`Method.cuped_multicovariate` uses fixed-horizon HC2 uncertainty and a Bell–McCaffrey
Student-t approximation. It is intentionally different from scalar `cuped`;
neither method inherits the other's evidence. The preregistered campaign in
`campaign-module:calibration.cuped_linear` is coded, but **neither the full nor the reduced
profile has been run**. Coverage is not established. Runtime support, the
example's observed interval, and passing the M1 support checks are not
calibration evidence.

The registered profiles are prospective gates, not results:

| Profile | Draws per statistic | Gate | Detectable undercoverage alternative |
|---|---:|---|---|
| `reduced` | 28,000 | coverage lower bound at least 0.94 | 0.92 coverage |
| `full` (Revision 7) | 107,100 | coverage lower bound at least 0.945 | 0.935 coverage |

These are the campaign's predeclared thresholds and detection targets only.
The profiles share a 67-statistic allocation; the campaign documentation
describes its accounting and the eight descriptive edge cells. Neither a
gate nor an alternative is an observed coverage value.

The campaign checks that production refuses fits outside the frozen M1
support region with the registered code and observables, then exercises the
public estimator on its preregistered grid. Probe prerequisites and project
cost before deciding whether to run it:

```bash
uv run python -m calibration.cuped_linear --campaign reduced --prerequisites
uv run python -m calibration.cuped_linear --campaign reduced --project
uv run python -m calibration.cuped_linear --campaign full --project
```

The actual certifying commands, from the checkout, are:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  uv run --extra tables python -m calibration.cuped_linear \
  --campaign reduced --output .test-evidence/intro-usability/cuped-linear-reduced --workers 8
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  uv run --extra tables python -m calibration.cuped_linear \
  --campaign full --output .test-evidence/intro-usability/cuped-linear-full --workers 8
```

Each output directory must be new. These commands are supplied for later
execution; this documentation change did not run them. External-prediction
training (tracker #72) and longitudinal dependence (tracker #80) are not
implemented. They are implementation gaps, not claims about the statistical
limit of the existing same-experiment fixed-horizon estimator.

## Campaign measurements and scope

These are recorded measurements, not new runs. Each record keeps its tested design,
exclusions and numerical qualifications. The [limitations](../limitations.md)
remain the place to check whether a method fits your decision.


### Small-arm mean intervals

On a mean metric the relative interval (a log-ratio delta method) can be the weaker of a
row's two intervals at small n on a skewed outcome. With gamma(0.5) outcomes (skewness
2.8) and a true ratio of 1.1, it covered 93.5% at 30 units per arm and 94.3% at 100, while
the same rows' additive Welch interval (`abs_lb`/`abs_ub`) covered 95.6% and 95.0%
(`campaign-module:calibration.small_arm_intervals mean`: 20,000 draws per cell, seed 77,
Monte Carlo standard error at most 0.18 points; the 0.65% of 30-unit draws that the
log-scale admission guard refuses are not counted).

### Ratio-denominator skew calibration

The cutoff comes from `campaign-module:calibration.ratio_skew` (seed 20261010, 6,000 replications
per cell, lognormal denominators with an independent `Normal(2, 1)` numerator, coverage
among the rows the log-scale admission guard admits at nominal 95%):

| denominator | n = 50 | n = 100 | n = 200 | n = 400 |
|---|---|---|---|---|
| lognormal(0.5) | 95.1% | 94.8% | 95.5% | 94.5% |
| lognormal(1.0) | 93.3% | 94.6% | 94.3% | 94.8% |
| lognormal(1.5) | 90.2% (16% refused) | 92.7% | 93.1% | 93.7% |
| lognormal(2.0) | 85.1% (62% refused) | 87.6% (33% refused) | 90.1% (15% refused) | 91.7% (5% refused) |

Pooled over the grid, rows whose observed statistic lies below 0.25 cover at 94.3–94.8%,
rows in [0.25, 0.30) at 93.6%, and every bin from 0.30 to 0.75 at 92.0–93.0%; the cutoff
is the first bin edge from which two consecutive bins cover below 93.5%. At 0.30 the
warning reaches 83–100% of admitted lognormal(1.5) and lognormal(2.0) rows, 24–85% of
lognormal(1.0) rows, and 0–20% of lognormal(0.5) rows (20% at n = 50, 0% at n = 400);
flagged rows cover at 92.2% pooled and unflagged rows at 94.3%. The statistic is read off
the sample, and the samples that miss a heavy tail look benign while under-covering most,
so the warning marks where observed skew begins to predict under-coverage rather than a
line that separates safe rows from unsafe ones. The precision note sees only the
denominator's own spread: a numerator that moves with the denominator covers close to
nominal even when flagged, and a numerator independent of it under-covers most. The
heavy-tail correction itself remains unresolved.

### Triggered sitewide calibration

The full calibration gates use two-arm, unclustered designs with 20% trigger
reach and measure coverage against realized potential-outcome site totals;
each gated statistic uses 107,100 replications. Run the campaign with an
explicit new output directory:

```bash
python -m calibration.triggered_sitewide --campaign full \
  --output .cache/triggered-sitewide-calibration
```

For sum metrics (20% and 100% of the site enrolled) and a benign ratio metric,
the assigned and triggered readings each covered 95.12%–95.66% for absolute
and relative impact at nominal 95%; every Clopper–Pearson lower bound, at a
family-allocated error of 7.5e-5, cleared 94.5%. The campaign checks the
Normal-reference intervals the sitewide functions report; it does not cover
count metrics, multi-arm netting, clustered designs, or uncertainty about
future traffic or trigger reach.

Calibration does not establish the real-world assumptions needed to interpret
the triggered result as whole-site impact. The explicit assumption-failure
simulations demonstrate this limitation: when never-triggered units have a
0.1 effect, the triggered reading undercounts by the omitted complement total
(about −800 of a true 1,400) and covers 3%, while the assigned reading still
covers 95%. When membership depends on treatment despite equal trigger rates,
the triggered reading covers almost never, while the triggered sample-ratio
check rejects in only 4.7% of runs.

Precision relative to the assigned reading was measured separately in the T3
descriptive designs: 10% trigger reach and 4,000 replications for each
variance ratio. The observed ratios were 0.91, 0.65 and 0.10 for a binary
post-trigger-only outcome, a lognormal post-trigger-only outcome and a broad
metric respectively. Each was within 5% of:

$$
\frac{p\sigma_T^2}{p\sigma_T^2 + (1-p)\sigma_{NT}^2 + p(1-p)(\mu_T-\mu_{NT})^2}
$$

### Clustered encouragement LATE calibration

The clustered encouragement LATE cuts its additive interval at a Welch–Satterthwaite t
reference over the two arms' own cluster-ratio variance components. Measured by
`tests/calibration/test_cluster_late_welch_df.py` (2000 replications per cell over 5/5,
5/25 and 10/50 treatment/control clusters, treatment between-cluster variance 1, 4 and 16
times the control's, mean cluster size 20; binomial Monte-Carlo standard error 0.5 points
at 95% and 0.9 points at 82%): with equal cluster sizes the interval covers the true LATE
94.3–96.0% at nominal 95% in every cell, including 94.55% at 5/25 clusters with 16 times
the variance in the small arm, where a pooled `t_(K_treatment + K_control - 2)` reference
over the same components would cover 89.15%. Log-normally dispersed cluster sizes (sigma 0.75)
cost about 3–6 points that the reference does not recover: 88.5–92.7% across those nine
cells, 89.7% at 5/25 with the 16x ratio, against 82.0% pooled. The first-stage gate withheld
at most 4 of 2000 replications in any cell.

### Cluster diagnostics release checks

Representative release checks are diagnostics, not certification of the original
stress matrix. Run the preserved cases and paired-width criteria explicitly:

```bash
uv run python -m calibration.cluster_diagnostics --output /tmp/cluster-diagnostics --repetitions 4
uv run python -m calibration.cluster_diagnostics --selection full --output /tmp/cluster-stress
```

The module entrypoints are `campaign-module:calibration.cluster_diagnostics --output /tmp/cluster-diagnostics --repetitions 4` and `campaign-module:calibration.cluster_diagnostics --selection full --output /tmp/cluster-stress`.

Output directories must be new. Budgets are bounded at one hour; started,
completed, refused, failed, unfinished, and unstarted work remain distinguishable.
The full route preserves original seeds, repetition rules, and paired-width
thresholds. Interrupted or diagnostic prefixes never pass those historical gates.

### Scalar CUPED coefficient measurement

The CUPED standard error conditions on a theta estimated from the same data it adjusts. The
uncertainty in that estimate is not propagated, so the reported interval is slightly
narrower than one holding the coefficient fixed. Measured by
`tests/calibration/test_cuped_theta_uncertainty.py` (16 cells: covariate correlation 0.1 to
0.9 by 20 to 2000 units per arm, equal allocation, one homogeneous slope, 3000 replications
each, intervals cut at the row's Welch–Satterthwaite t reference; binomial Monte-Carlo
standard error 0.4 points): the fitted interval is 1.3–1.4% narrower on the log scale than
a known-coefficient comparator at 20 units per arm, 0.5% at 50, 0.1% at 200 and 0.01% at
2000, the same at every covariate correlation, and within Monte-Carlo error of
`sqrt(1 - 1/(N - 2))` with `N` the total sample. Coverage of the fitted interval is
93.4–95.3% at nominal 95% across the 16 cells, against 93.8–95.6% for the comparator, and
the reported standard error runs 0.94–1.01 times the empirical spread of the log estimate
(0.95–0.98 at 20 per arm). The comparator is the same log-scale delta method with the true
coefficient substituted, not exact finite-sample inference, so the width ratio measures the
residual-variance shrinkage from fitting theta and does not isolate the added sampling
variance of the estimate. Unequal allocation and arm-specific slopes are outside that design.

### Finite-sample width and candidate ordering

Its ordering is the `r = 1` difference for every candidate, so at moderate counts the set
is wider than the log-scale Wald (Katz) interval by 1.07× (20,000/arm, 10%), 1.14×
(5,000/arm, 10%, r = 1), 1.24× (r = 1.2) and 1.36× (2,000/arm, 5%) in log width (the
audit's earlier measurement of the same shapes: 1.07×, 1.16×, 1.24× and 1.45×), and its
null p-value is 1.7–4× the pivotal (score-ordered) exact p-value at those shapes
(500/5000 vs 600/5000: 2.38e-3 against 1.39e-3; 50/500 vs 75/500: 4.35e-2 against
1.69e-2; 20/200 vs 40/200: 2.19e-2 against 5.18e-3). Smaller arms widen the gap. On the
exact count lattice of the default row at two-sided 95%, where every count pair takes the
finite-sample route (`campaign-module:calibration.small_arm_intervals conversion`, under
`1e-8` of probability omitted), the set covers 97.47% at 1,000 units per arm, a 10%
control rate and r = 1.08 (Katz 95.04%), with 1.355× Katz's expected log width. At 200
units per arm, 10% and r = 1 it covers 99.47% (Katz 95.64%), and its lower endpoint is a
ratio of zero (a -100% lift) with probability 0.43 although treatment has events: a
candidate near zero is tested at the lowest control rate the nuisance range admits, where
a control count as small as the observed difference between the arms is not surprising.
Over the other outcomes its expected log width is 2.04× Katz's. A candidate-ordered
construction that removes the gap at exact coverage was measured against the shipped
ordering on the same Berger-Boos frame: exact-lattice coverage at or above nominal on all
372 grid cells (worst one-sided miss `2.0e-6` below `alpha / 2`), log width 1.000–1.004×
Katz at the moderate-count shapes above, eligibility unchanged. It is held back by the
cost of certifying its nuisance supremum (1.8–15× the shipped runtime at the shipped
endpoint tolerance, 0.7–4.2× at an eight-times coarser one) and by a
threshold-monotonicity lemma that is measured (0 violations in 6,000 candidate steps per
count pair) but unproven. Adopting it would change the stored-set construction token, so
persisted sets would be refused by `estimation.results.binomial.obsolete_construction` and
re-run.

### Dense-count threshold measurements

`dense_min_count(tail) = max(412, ceil(145 z^4))` with `z = Phi^-1(1 - tail)`. It is an envelope
of a measured requirement, not a fit: the requirement is the least count at which the
delta-method interval's one-sided noncoverage in the boundary cells below stays within
`tests.mc.scientific_delta(tail)` of the tail (a tenth of the tail below 0.05, 0.005 at and above
it) at that count and at every larger step of a geometric ladder (ratio 1.15) through 1.25 times
it. The law is at least 1.25 times the requirement at every production tail; the 0.1 tail sets
its floor and the 0.0005 tail its slope. The tail error of a log risk ratio Wald interval grows
like `z^3` relative to the tail, so extreme tails need thousands of events. The requirement
grows more slowly than `z^4` below them, so the middle tails carry a margin well above 1.25
(about 3.7 times at 0.025 and 0.05), and counts the interval already covers take the
finite-sample route there, which is valid at every count. A tail below 0.0005 is extrapolated by
the same formula, not measured. Per tail, the requirement is a step of a ladder of that ratio
with every step of that candidate's own ladder through 1.25 times it measured; the ladders are
anchored at 10 for the tails from 0.025 up and at 1,637 below (the 0.005 tail's on the
continuation of an interrupted scan, at 2,489.5). A scan records its ladder with every step.
The same numeric rung counts regardless of which scan measured it, but a differently numbered
rung cannot replace a missing candidate rung. The `python -m calibration.conversion_route required` command (`campaign-module:calibration.conversion_route required`)
reads saved scans, inferring compatible anchors for legacy rows and refusing a legacy scan
that no single ladder fits:

| One-sided tail | 0.0005 | 0.001 | 0.005 | 0.01 | 0.025 | 0.05 | 0.1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Measured requirement | 13,320 | 10,072 | 2,863 | 2,165 | 576 | 286 | 329 |
| `dense_min_count` | 17,000 | 13,224 | 6,384 | 4,247 | 2,140 | 1,062 | 412 |
| Law over requirement | 1.28 | 1.31 | 2.23 | 1.96 | 3.72 | 3.71 | 1.25 |

`python -m calibration.conversion_route select` (`campaign-module:calibration.conversion_route select`) integrates the interval's noncoverage over
the exact binomial lattice in rare, failure-limited and central boundary cells (control sizes
1e3 to 1e7 at every 1, 2 and 5 of a decade, allocations 1:1, 1:4 and 4:1, risk ratios 0.5 to 2)
and reports, per tail, the worst excess over the tail in tolerance units (at most 1 passes); the
failure-limited cells bind at every tail. The excess just below each requirement and at it:

| One-sided tail | 0.0005 | 0.001 | 0.005 | 0.01 | 0.025 | 0.05 | 0.1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Last failing step | 11,583 | 8,758 | 2,490 | 1,883 | 501 | 249 | 286 |
| Worst excess there | 1.15 | 1.09 | 1.29 | 1.01 | 1.16 | 1.21 | 1.03 |
| Requirement | 13,320 | 10,072 | 2,863 | 2,165 | 576 | 286 | 329 |
| Worst excess there | 0.87 | 0.98 | 0.97 | 0.99 | 0.98 | 0.93 | 0.97 |

and, at counts every tail was measured at (worst excess in tolerance units):

| Count `m` | 0.0005 | 0.001 | 0.005 | 0.01 | 0.025 | 0.05 | 0.1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 435 | 6.84 | 5.47 | 3.07 | 2.25 | 1.39 | 0.88 | 0.95 |
| 1,637 | 2.86 | 2.35 | 1.37 | 1.03 | 0.64 | 0.41 | 0.45 |
| 2,165 | 2.73 | 2.25 | 1.31 | 0.99 | 0.62 | 0.39 | 0.42 |

`verify` re-checks the shipped count at each tail and at 1.25 times it in the same cells; every
cell passes (worst excess in tolerance units, at the count and at 1.25 times it):

| One-sided tail | 0.0005 | 0.001 | 0.005 | 0.01 | 0.025 | 0.05 | 0.1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| At `dense_min_count` | 0.84 | 0.72 | 0.62 | 0.69 | 0.62 | 0.56 | 0.97 |
| At 1.25 times it | 0.83 | 0.71 | 0.65 | 0.60 | 0.47 | 0.42 | 0.76 |

The 0.1 tail passes by 3% at its floor. `hybrid` sums the production pipeline's noncoverage
exactly over the count lattice at design counts two below to two above the shipped count (the
three cells with the most routed noncoverage at each offset, two-sided at `alpha = 2 * tail`,
plus the worst cell read directionally at `alpha = tail`: `greater`, and `less` at the tails a
production alpha reaches directionally, run at the 0.1 and 0.05 tails). At every one of the
seven tails every cell stays within tolerance; the worst excess over each tail's cells is
0.50 / +0.34 / +0.40 / +0.48 / +0.53 / +0.63 / +0.67 at the 0.1 / 0.05 / 0.025 / 0.01 / 0.005 /
0.001 / 0.0005 tails. The production pipeline on 57,601, 30,400 and 62,400 seeded draws at the
0.1, 0.05 and 0.025 tails agrees with the exact value within four Monte Carlo standard errors;
that check was not run below the 0.025 tail. The calibration-only vectorised interval uses
the runtime's `student_t_isf` for both direct quantiles and the interpolation nodes, rather
than a separate `scipy.stats.t.isf` inversion. On the measured `conformance` cells with
Python 3.14 and SciPy 1.16.3, it agreed with `estimate_lift` within `2.0e-14` (relative),
and the interpolated Student quantile agreed with the runtime quantile within `2.6e-15`.
The conformance bounds remain `1e-9` and `1e-12`, respectively. These empirical comparisons
are not numerical certificates; planning and exact enumeration use the runtime calculation
directly. A bounded grid is evidence at those cells, not coverage for every data-generating
process.

A BH-selected row is re-estimated at its own corrected level and routed there, so its label can
differ from the label of the nominal row whose p-value selected it.

### Sequential campaign scope

The public Bernoulli likelihood route has a finite-sample anytime-valid
derivation under its registered model, assignment and reveal assumptions.
That argument and verification of the deployed numerical implementation are
separate obligations. Continuous-mean, adjusted-mean and ratio sequential
routes carry asymptotic qualifications; they do not inherit the Bernoulli
finite-sample guarantee.

The complete empirical campaign over the registered runtime manifest has
not been executed. Its recorded cost is months on twelve cores even after
the measured speedups; budgeted profiles deliberately cover only subsets.
The presence of this machinery is a specification of the check, not evidence
that its cells passed. Complete clustered and unit-cycle research matrices
likewise remain uncertified.

Release review for public sequential and difficult-cluster methods instead
uses a bounded claim audit, independent numerical oracles and preselected
risk-focused calibration, with unchanged per-case thresholds. Those checks
can identify concrete failures and establish evidence for their stated cases;
they cannot establish universal calibration, resolve unrun regimes, or turn
small-cluster/asymptotic approximations into finite-sample guarantees.
Incomplete or insufficiently precise checks remain unverified. This narrower
release evidence requirement does not claim completion of the full campaigns
or change the separate unit-cycle research obligations.

Evidence status for these campaigns is recorded under
[Unvalidated regimes](../validation.md#unvalidated-regimes).

### Adaptive-logger measurements

A trace logged under more than one policy version is refused (CONSTRUCTION,
`logged_policy.inference.adaptive_logging_unsupported`). Measured before that gate
existed, with the same estimator on batch-refit adaptive loggers at horizon 5 under
context dependence 0.8 (`tests/calibration/test_logged_policy_adaptive.py`'s design, 400
replications at 200 units): an epsilon-greedy logger covered 86.4%, with the estimate's
bias within its Monte-Carlo error but the sandwich standard error 27% smaller than the
estimate's realized spread; a Thompson-sampling logger covered 89.6% over the 67% of
replications its own propensities kept above the floor; and a contextual Thompson logger
at horizon 2, 1000 units, covered 87.2% over the 24% of replications that survived the
floor, with a bias of seven Monte-Carlo standard errors in that surviving slice. The
variance understatement comes from units coupled through the batch refits and from
heavy-tailed cumulative weights; the surviving-slice bias is selection on the outcomes
that drove the propensities down. The construction that repairs both (stabilized or
adaptively weighted estimators with a martingale variance; Hadad et al. 2021, Zhang,
Janson and Murphy 2021) is not implemented, so the refusal names one policy version per
trace as the route forward.

The record of where these figures come from is under
[Unvalidated regimes](../validation.md#unvalidated-regimes).

### Fixed-logger calibration

Under a fixed logger the interval is nominal from the effective-sample-size floor up.
Measured on 26 fixed-logger cells over target divergence, horizon and unit count (9188
emitted intervals; `tests/calibration/test_logged_policy_ess_floor.py`), coverage by band
of the smallest per-index ESS was 0.862 / 0.892 / 0.933 / 0.933 / 0.943 / 0.947 / 0.949 for
`[2,5)`, `[5,10)`, `[10,20)`, `[20,40)`, `[40,80)`, `[80,160)` and `[160,inf)`, with
700 to 2190 intervals per band; `[20,40)` is the smallest band from which every band is
within three Monte-Carlo standard errors of nominal, so `ESS_FLOOR = 20` and, since the
ESS never exceeds the unit count, the independent-unit floor is 20 as well. The two
bands just above the floor sit 1 to 2 points under nominal -- within tolerance, not at
it -- because the sandwich standard error is about 7% too small there; from 80 effective
units on it is within 1 point. On the nine fixed-logger cells of the adaptive study
(T in {1, 2, 5}, 50 to 1000 units, 160 to 800 replications each) coverage ranged from
0.931 to 0.960, every cell inside its own three-MCSE band; at horizon 5 with 100 units
the floor screened 126 of 600 replications and the emitted intervals covered 0.941.

The cumulative-product weighting is what makes the estimator target the dynamic policy
value. Measured against a current-step-only comparator at horizon 5, 1000 units, under
context dependence 0.8 (truth 0.0275): the current-step estimator is biased by +0.0026,
more than three Monte-Carlo standard errors from zero, while the shipped estimator's
bias of +0.0006 stays within its own.

### Binomial planning comparison campaign

`python -m calibration.conversion_route bound` (`campaign-module:calibration.conversion_route bound`) compares each route's plan with the pipeline's
exact rejection probability, summed over the count lattice, and judges each by the claim of its
route. The 534 designs run so far reach every production tail at central and rare-event rates,
with falls (negative lifts) at the 0.01, 0.025, 0.05 and 0.1 tails; the `tails` and `window`
grids define every production request at its own alpha, two-sided against a rise and
directional in both directions, and the directional requests have been run at the 0.05 and 0.1
tails (central and rare-event) and two 0.01 designs. Those figures were measured against the
interim planner (the smaller of the replayed and the closed-form power for a plan whose counts
straddle the threshold, and a Normal-tail replay above 16,000 cells): a sparse plan was
between 1.8e-5 above and 3.3e-4 below the pipeline's exact probability, and a borderline plan
between 1.6e-5 above and 0.072 below it. Neither is a claim of the hybrid planner, whose
enclosure is checked against the same enumeration with no tolerance beyond its own undecided
mass and numerical error (the dense closed form keeps its 0.005 ceiling; its two failures, at
1,236 per arm and a 50% baseline with a 6% lift, are now enumerated).

A `bound --out` checkpoint records each design's enumeration beside the finite-sample
construction and routing floor it was summed under, and its plan beside the planner model. A
resumed run retains compatible runtime enumerations and replans retired-model records with the
current planner, `hybrid_finite_plus_delta_v3`. Its v1 and v2 predecessors must be
recomputed, not relabelled. Unknown models and earlier undated layouts refuse with
the file line and a route to a new `--out`. Saved earlier files remain unchanged as evidence.

### Binomial replay cost measurements

Under default `"auto"`, dense plans using the closed-form route have no replay and no
such bound; small enumerated dense plans remain subject to the replay bound. Measured with
`scripts/measure_binomial_ceiling.py planning` (`--conversion-inference finite_sample` or
`auto`) on a 12-core machine, cold CPU seconds of the call and the process's peak resident set,
under a shared load that varied by several times between cells (null cells are the null
rectangle). Explicit `finite_sample` plans at 1e5, 1e6, 5e6 and 5e7 units per arm, dense (a 5%
baseline) and sparse (about a hundred events per arm):

#### Baseline, units per arm: 5%, 100,000

**Call:** `achieved_power`, lift 5%

**CPU / peak:** 9.8 s / 1.34 GiB

**Outcome:** 0.97M null cells, companion effect available

#### Baseline, units per arm: 5%, 1,000,000

**Call:** `achieved_power`, lift 1.5%

**CPU / peak:** 198 s / 1.72 GiB

**Outcome:** 9.67M null cells; companion effect `numerical_resolution`

#### Baseline, units per arm: 5%, 5,000,000

**Call:** `achieved_power` or `minimum_detectable_effect`

**CPU / peak:** refused within 1 ms / 0.12 GiB

**Outcome:** 48.3M null cells

#### Baseline, units per arm: 5%, 50,000,000

**Call:** `achieved_power` or `minimum_detectable_effect`

**CPU / peak:** refused within 1 ms / 0.12 GiB

**Outcome:** 483M null cells

#### Baseline, units per arm: 0.1% at 1e5, 0.01% at 1e6, 0.002% at 5e6, 0.0002% at 5e7 (100 events)

**Call:** `achieved_power`, lift 50%, and `minimum_detectable_effect`

**CPU / peak:** 0.19 to 0.20 s / 0.21 to 0.26 GiB

**Outcome:** Historical timing for 20,164 null cells per size on the internal Normal-tail sizing proposal, not a returned power or MDE. Current public calls admit only a resolved exact value or refuse.


The same calls under `"auto"`:

#### Baseline, units per arm: 5%, 1e5 to 5e7

**Call:** `achieved_power`, `minimum_detectable_effect` and `required_sample_size`

**CPU / peak:** 1 ms / 0.12 GiB

**Outcome:** `power_basis="asymptotic"`; at 5e6 the power for lift 1% is 0.951 and the MDE 0.77%, at 5e7 the power for lift 0.3% is 0.930 and the MDE 0.24%

#### Baseline, units per arm: 50%, 1e5 and 1.9e5

**Call:** `minimum_detectable_effect`, `achieved_power`

**CPU / peak:** 1 ms / 0.12 GiB

**Outcome:** asymptotic, answered where the replay refuses

#### Baseline, units per arm: 100 events per arm at 1e5, 1e6, 5e6, 5e7

**Call:** `achieved_power`, lift 50%

**CPU / peak:** 0.19 to 0.21 s / 0.23 to 0.25 GiB

**Outcome:** Historical timing of the internal Normal-tail sizing proposal; not a returned power. Current public calls admit only resolved exact/asymptotic values or refuse; no current timing was re-measured here.


Earlier measurements of the replay at other sizes (an Apple M3 Pro at `45829f1`, `2da355f`),
which an explicit `finite_sample` plan still takes and which were not repeated at this commit.
The cells marked † were re-measured at `45829f1`, after a companion effect search became a solve
of its own (it no longer inherits the supplied effect's cells, so an unresolved companion costs
what a refused `minimum_detectable_effect` does); the others are from `2da355f`:

| Baseline, units per arm | Call | CPU / peak | Outcome |
|---|---|---:|---|
| 5%, 250,000 | `achieved_power`, lift 3% | 16.5 s / 1.6 GiB | 2.4M null cells, companion effect available |
| 5%, 500,000 † | `achieved_power`, lift 2% | 35.3 s / 1.8 GiB | 4.8M null cells, companion effect available |
| 5%, 500,000 † | `minimum_detectable_effect` | 35.7 s / 1.9 GiB | answered |
| 5%, 1,000,000 † | `minimum_detectable_effect` | refused after 61.0 s / 1.8 GiB | needs 19.7M cells |
| 5%, lift 1.75% | `required_sample_size` | 277 s / 1.7 GiB | 993,064 per arm; companion effect `numerical_resolution` (not re-measured) |
| 50%, 100,000 | `minimum_detectable_effect` | refused after 57.5 s / 1.3 GiB | needs 10.6M cells (5.1M null) |
| 50%, 190,000 † | `achieved_power`, lift 3% | 201 s / 1.7 GiB | 9.7M null cells; companion effect `numerical_resolution` |
| 50%, 190,000 † | `minimum_detectable_effect` | refused after 210 s / 1.1 GiB | needs 10.8M cells |
| 1e-4 at 1e6, 1e-6 at 1e8, 1e-7 at 1e9 (100 events) | `achieved_power`, lift 50% | 0.85 s / 0.2 GiB | 20,164 null cells, answered |

Not measured: `required_sample_size` at a 50% baseline or beyond a million units per arm under
`finite_sample`. Planning a design above the bound has no replay: under `auto` a dense one is
planned in closed form, an explicit `finite_sample` one is refused. Before the bound existed a
call took 574 s and 3.4 GiB at 4,000,000 per arm (39 million cells).

What the effect search's reserved endpoints and the shared decisions save, measured as
paired runs (the two revisions alternating, three cold processes each, the best of three;
a shared 12-core Apple arm64 host under a load of 15 to 140, so the absolute times are
contended and the ratios are the evidence) at commit `e39f3ce` against its parent
`e51117e`, with the finite-sample replay rows each call ran (the replay kernel is unchanged,
so the row count is the work). Every decision, route, size, power (to `1e-11`), effect (to
`1e-9` relative) and refusal is the same at every cell; every call is faster, by 1.0 to 2.8
times, and peak resident growth stays within 1.4 times:

| Design | Call | Replay rows before -> after | Wall before -> after | Outcome |
|---|---|---:|---:|---|
| 10%, 701/arm, `finite_sample` | `achieved_power`, lift 50% / `minimum_detectable_effect` / `required_sample_size` | 96,728 -> 41,132 / 96,954 -> 41,132 / 190,340 -> 121,142 | 5.3 -> 2.2 s / 4.5 -> 1.8 s / 7.3 -> 4.6 s | exact; sized 831 |
| 10%, 2,600/arm, `finite_sample` | the same three calls | 449,080 -> 152,864 / 449,516 -> 152,864 / 745,142 -> 424,798 | 13.5 -> 6.7 s / 10.6 -> 5.4 s / 21.5 -> 14.4 s | exact; sized 2,926 |
| 10%, 5,000/arm, `finite_sample` | the same three calls, and a 2x2 `power_curve` | 663,072 -> 249,790 / 663,096 -> 249,790 / 1,080,367 -> 659,139 / 1,430,212 -> 527,340 | 25.1 -> 13.5 s / 23.8 -> 13.9 s / 48.3 -> 35.1 s / 57.4 -> 28.6 s | exact; sized 5,473 |
| 10%, 7,000/arm, `finite_sample` | the same three calls | 732,830 -> 246,166 / 732,540 -> 246,162 / 1,241,231 -> 750,463 | 50.3 -> 18.2 s / 26.3 -> 16.1 s / 40.3 -> 31.6 s | exact; sized 7,570, companion `numerical_resolution` |
| 2%, 20,000/arm, `finite_sample` | the same three calls | 738,438 -> 221,306 / 738,612 -> 221,306 / 1,183,354 -> 662,616 | 20.0 -> 11.1 s / 17.1 -> 10.6 s / 31.2 -> 21.0 s | exact; sized 22,384 |
| 10%, 6,000 treatment at allocation 0.8, `finite_sample` | the same three calls, and a 2x2 curve | 551,282 -> 180,580 / 552,610 -> 180,580 / 1,772,330 -> 1,354,440 / 1,268,806 -> 438,350 | 38.2 -> 19.4 s / 32.8 -> 13.5 s / 81.7 -> 77.9 s / 45.4 -> 23.2 s | exact; sized 7,245 |
| 10%, 2,000 treatment at allocation 0.2, `finite_sample` | the same three calls | 565,918 -> 206,776 / 565,836 -> 206,776 / 1,308,035 -> 947,653 | 13.0 -> 7.2 s / 13.6 -> 6.6 s / 66.3 -> 39.8 s | exact; sized 2,041 |
| 10%, 7,000/arm, `greater`, `null_lift=0.2`, `finite_sample` | the same three calls | 645,735 -> 208,348 / 649,673 -> 208,348 / 1,528,176 -> 1,528,176 | 13.3 -> 7.5 s / 13.9 -> 7.0 s / 26.4 -> 26.1 s | exact; sizing refused `power.binomial_probability_unresolved` at its ceiling |
| 30%, 3,500/arm, `less`, `null_lift=-0.1`, `finite_sample` | the same three calls | 575,856 -> 197,604 / 572,760 -> 197,604 / 1,544,062 -> 1,544,062 | 16.2 -> 8.7 s / 17.6 -> 8.7 s / 37.7 -> 30.7 s | exact; sizing refused likewise |
| 10%, 20,000 assigned at trigger 0.35, `finite_sample` | the same three calls | 732,830 -> 246,166 / 732,540 -> 246,162 / 1,241,231 -> 750,463 | 33.1 -> 19.7 s / 32.8 -> 21.2 s / 61.8 -> 53.0 s | exact at 7,000 analyzed; sized 21,628, companion `numerical_resolution` |
| 0.1% at 100,000/arm and 0.01% at 1,000,000/arm, `auto` (sparse) | the same three calls | 384,840 -> 69,102 / 385,692 -> 69,102 / 607,933 -> 251,555 (and 444,308 -> 67,684 / 445,160 -> 67,684 / 676,550 -> 250,146) | 4.2 -> 1.6 s / 3.5 -> 1.9 s / 8.7 -> 5.9 s (and 2.9 -> 1.5 s / 3.2 -> 1.6 s / 7.9 -> 5.6 s) | exact; sized 122,373 and 1,223,792 |
| 10%, 5,000/arm, `auto` (sparse) | the same three calls | as the explicit `finite_sample` plan | 22.1 -> 12.4 s / 23.0 -> 12.7 s / 42.5 -> 31.1 s | exact; sized 5,473 |
| 10%, 23,000/arm, `auto` (borderline) | the same three calls | 149,894 -> 149,894 / 718,770 -> 273,622 / 278,591 -> 278,591 | 10.5 -> 10.5 s / 34.6 -> 27.9 s / 17.4 -> 16.8 s | refused `power.binomial_probability_unresolved` (effect: `numerical_resolution`): the budgets leave 4e-3 of the mass undecided |

A supplied effect's own evaluation and a sizing's heuristic probes replay as before; the
companion search after them now costs a fraction of the evaluation it follows rather than a
multiple. The outcome column is historical: its refusals of
`power.binomial_probability_unresolved` are what stood when the table was measured. A rerun
today states the enclosure's lower end as `power` (`power_basis="lower_bound"`) at the same
sizes and times, and a sizing whose lower end never reaches the target climbs to its ceiling
and refuses there: as an unresolved probability (`power.binomial_probability_unresolved`)
when the ceiling's enclosure holds the target, otherwise with
`power.binomial_replay_bound_exceeded` or
`power.binomial_size_search_unreachable`. An explicit `finite_sample` plan at the ticket's
dense shapes (10,000 per arm and above at a 10% rate, 20,000 at 50%) has windows of 190,000
to 2.3 million cells, more than the per-evaluation budgets decide, so the enclosure stays
wider than `1e-6` and the lower end is stated after one evaluation (10 to 50 s). Under the
default `auto` the same budgets reach the designs the count rule does not route to the delta
method with near certainty: at a 10% rate 10,000 per arm is `sparse` and 20,000 to 23,000
`borderline` (the routing floor is 2,140 successes and failures per arm at the default
two-sided alpha, where those arms expect 1,000 to 2,300), so they are enumerated and stated
likewise; 35,000 per arm at 10% and 20,000 at 50% are `dense` and planned in closed form
without a replay.

### Clustered targeting null screen

Measured behaviour on a null screen with frozen oracle scores (constant effect,
independent CLAN covariate), 256 replications and 199 draws per cell: every one
of 72 designs stays within its family-aware binomial miss bound for AUTOC, Qini,
each GATES interval, the joint GATES family and CLAN, and within the rejection
bound for the AUTOC and Qini tests. The designs cross 10, 40 or 200 clusters,
equal sizes of 20 or repeating sizes 5/20/100, intracluster correlation 0, 0.2
or 0.5, Gaussian or skewed errors with one high-leverage score per cluster, and
both weightings. The tightest design has 10 unequal clusters, skewed errors,
member weighting and correlation 0.5: each GATES interval missed 18 of 256
times against a nominal 2.5% (6.4 expected, bound 19). Few unequal clusters
therefore still undercover; the screen is a regression check, not a guarantee.

### HKSJ small-K calibration

Pooling uses a DerSimonian–Laird heterogeneity estimate followed by an unmodified
Hartung–Knapp–Sidik–Jonkman adjustment, with no variance floor. Measured by
`tests/calibration/test_hksj_small_k.py` (10000 replications per cell, unequal segment
variances with log-scale standard errors 0.10–0.20, between-segment variance 0, 0.5 and 2
times the average sampling variance; binomial Monte-Carlo standard error 0.2 points at 95%
and 0.4 points at 80%): the HKSJ interval covers the random-effects mean 93.7–95.2% at
nominal 95% for every K from 2 to 12 and every heterogeneity level. The plug-in
random-effects interval covers 80.0% at K = 2, 85.2% at K = 3, 87.2% at K = 4, 88.6% at
K = 5, 90.9% at K = 8 and 92.2% at K = 12 when the between-segment variance is twice the
average sampling variance, and over-covers (96.0–96.4%) when there is none. HKSJ is the
narrower of the two in 12–35% of replications without heterogeneity and 2–13% with it. The
degeneracy fallback to the plug-in interval fired in 1 of 10000 replications in two of the
K = 2 cells and never otherwise on continuous data. The anticonservatism at small K is real
but bounded at about 1.3 points in this design; treat a pooled interval over two or three
segments as indicative for that reason, not because it is narrower than plug-in.

### Adaptive-logger measurement provenance

Provenance of the adaptive-logger figures: coverage figures 86.4/89.6/87.2% for batch-refit
adaptive loggers were measured before the refusal gate; the generating estimator is not retained
and the figures are not reproducible from this repository; the shipped test
(`tests/calibration/test_logged_policy_adaptive.py`) asserts only the refusal. The figures first
appear in the 2026-09-30 import commit `e132d0f` of the `initial-pre-release` history, and
`git log -S'covered 86.4%' -- docs/limitations.md` finds only the squashed commit `94f4b6b`
(2026-10-01) on `main`. No earlier commit contains `estimate_policy_contrast`. The simulator
(`simulate_logged_run`) and a test-local trajectory estimator survive, so a fresh measurement is
possible, but it would be a new measurement, not a reproduction.

## Reproduce

Ordinary tests need only the repository:

```bash
make test TESTS=tests/oracles/test_welch_oracle.py
make test TESTS=tests/oracles/test_clopper_pearson_oracle.py
make test TESTS=tests/oracles/test_bh_oracle.py
make test TESTS=tests/oracles/test_meta_oracle.py
make test TESTS=tests/oracles/test_quantile_oracle.py
make test TESTS=tests/oracles/test_dml_oracle.py
```

Regenerating a fixture needs the tools listed in its evidence record and is a deliberate act: a numeric difference found
by regeneration is a discrepancy to record, never a reason to widen a tolerance. Generators run
from the repository root. Digests are SHA-256 of the committed files.

The three older fixtures predate provenance blocks: they record tool versions but not the
increment commit, generation date, platform, `jsonlite` or Python version, so the commit and date
in those records come from `git log`. The three newer fixtures carry `generator`, tool versions,
`generated_on`, `increment_commit` (the parent of the commit that added them) and a one-sentence
`estimand`. The regeneration run used a scratch worktree and nothing it wrote was committed.

## Case ledger

No independently sourced case has been recorded.

An entry belongs here only when someone who is neither a maintainer nor an automated agent supplies
a real experiment's edge case (aggregate moments or sanitized rows) or an independent review run of
a matched comparison, with permission to describe it, enough information to rerun it on a stated
increment version, and their result. Maintainer-, agent- or synthetic-only comparisons, including
every row above, do not count as independently sourced.

## Discrepancy log

The following discrepancies are explained, not dismissed.


### BH boundary rounding

**Date:** 2026-10-03

**Comparison:** `EV-BH-P-ADJUST`

**Observation:** For the family `[0.01, 0.5, 0.01, 0.01, 0.2]` at `q = 0.25`, R selects four indices and `bh_select` selects three. The fifth p-value, 0.2, equals `k * q / m` for `k = 4, m = 5` in decimal. R computes the adjusted value `5/4 * 0.2` as exactly 0.25 and includes it. The stored double 0.2 is slightly above 0.2, and `_conservative_ratio` rounds the threshold downward by design, so the p-value exceeds the threshold on the stored doubles and `bh_select` excludes it.

**Status:** explained (measured on this input; the rounding direction is the documented behavior of `_conservative_ratio`). The fixture's vectors avoid exact boundaries.


### Regeneration of the three earlier fixtures

**Date:** 2026-10-03

**Comparison:** Regeneration of the three earlier fixtures

**Observation:** No numeric difference. The DML fixture differs only in the trailing newline.

**Status:** explained (byte comparison)


No other discrepancy has been found between increment and these references.
