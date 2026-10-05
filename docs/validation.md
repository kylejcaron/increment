# External validation coverage

Frozen third-party references exist for a small set of estimators. This page states which, at what
tolerance, through which entry points, and which regimes have none. It is an inventory of evidence,
not a certification: a row means the named estimand agreed with the named reference at the stated
tolerance on the stated inputs, never that a whole method is validated. Campaigns that have not
been executed are listed as unexecuted, and the guarantees themselves are stated on
[Statistical limitations](limitations.md).

Selector audit: nothing automated checks that the test selectors on this page still resolve. To
audit one, run `uv run pytest --collect-only -q <selector>` (add `-m slow` for the slow-marked
calibration modules); a selector that does not collect is a defect in this page. Each generator
command in [Reproduce](#reproduce) was run once, on the date and platform recorded there; the
page does not claim a later re-run.

## Reference inventory

### Reference classes

| Class | Meaning | Independent third-party reference? |
|---|---|---|
| `third_party_tool` | Frozen output of an external package at a matched estimand and assumptions | yes |
| `independent_reimplementation` | An external primitive (for example R `qbinom`) recomputes the same formula | partially: the primitive only |
| `independent_derivation_in_repo` | Hand arithmetic, quadrature or exact enumeration written in this repository | no: same authorship |
| `calibration_evidence` | Bounded Monte Carlo grid with preselected cells | no: behavior evidence, not a reference |
| `none_found` | A matched reference was looked for and not found | not applicable |

Ordinary tests read frozen JSON only: no R, no network, no external package at test time. The
comparisons below are at estimator level unless the ingress columns name an entry point. A
transitive ingress is claimed only where a parity-matrix cell or parity case exercises the same
estimator path; cross-ingress equivalence is owned by the parity harness
(`tests/parity_harness/matrix.py`), not by these oracles.

A tolerance written as relative below is enforced as `pytest.approx(expected, rel=..., abs=0.0)`.
`pytest.approx` otherwise keeps a default absolute floor of `1e-12` next to a relative tolerance,
which would dominate a near-zero value, so the tests pass `abs=0.0` and no absolute allowance exists.
A tolerance that is absolute is stated as absolute.

### Frozen comparisons against a third-party tool or independent primitive

Column key: **Evidence id**, **Capability and public symbol**, **Metric types**, **Estimand and
assumptions matched**, **Reference class**, **Reference tool and version**, **Ingress exercised
directly**, **Ingress covered transitively** (parity-matrix cell id or parity case id),
**Stated tolerance and how it was chosen**, **Fixture / generator** (SHA-256 prefix of the
committed fixture), **Test selector**, **Regimes not covered**.

| Evidence id | Capability and public symbol | Metric types | Estimand and assumptions matched | Reference class | Reference tool and version | Ingress exercised directly | Ingress covered transitively | Stated tolerance and how it was chosen | Fixture / generator | Test selector | Regimes not covered |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `EV-WELCH-T` | Fixed-horizon, prior-free unweighted mean difference, `Analysis.run` without `prior`, absolute-axis fields | mean | Treatment minus control difference of arm means, Welch variance, Welch-Satterthwaite degrees of freedom, two-sided 95% t interval; independent units | `third_party_tool` | R 4.4.2 `stats::t.test(var.equal = FALSE)` | `from_unit_summary` | parity cell `mean-run-none-utc-error` (`from_definitions`, `from_unit_day_artifact`, `from_unit_summary`, `from_unit_panel`, `from_moments`); `from_switchback_panel` is a separate estimand and is not covered by this oracle | relative `1e-12` on difference, SE, df and both endpoints; the largest observed relative difference was `6.7e-15`, so this is floating-point headroom | `welch_t.json` `abf22cc43957`; `gen_welch_oracle.R` | `tests/oracles/test_welch_oracle.py::test_welch_difference_interval_matches_r_t_test` | Two cases (14 vs 22 units with unequal variance; 5 vs 6 units, near-equal variance). Arms of fewer than 2 units are not compared. Weighted, clustered, covariate-adjusted, ratio, winsorized and sequential contrasts are not compared. Runs with an informative `prior` are not covered: the absolute interval then uses a Normal reference (`abs_reference_kind == "normal"`) instead of the Welch-Satterthwaite t reference. The p-value is not compared |
| `EV-CP-BINOM-TEST` | `increment.estimation.binomial_rr.clopper_pearson` | conversion, retention (primitive of the exact-binomial route) | Exact two-sided single-arm Clopper-Pearson `1 - beta` interval; production rounds outward by a documented relative slack | `third_party_tool` | R 4.4.2 `stats::binom.test()$conf.int` | none: the primitive is called directly | none claimed; the primitive feeds the exact-binomial risk-ratio set, which this oracle does not compare | enclosure of R's interval within `1e-15`, then an outward gap of at most `1.01 * 1e-6` of the lower bound below and of one minus the upper bound above; the band is the documented slack, and every measured gap equals it to within `5e-6` of itself | `clopper_pearson.json` `7f53220dd69a`; `gen_clopper_pearson_oracle.R` | `tests/oracles/test_clopper_pearson_oracle.py::test_interval_encloses_and_matches_r_binom_test` | 12 cases: `(x, n)` in `(0,20) (1,20) (7,40) (39,40) (40,40) (500,1000)` at `beta` 0.05 and 0.001. Two-arm risk-ratio sets are not compared. Arms above about 2^26 are an open numerical investigation (internal tracker `1tce`: the SciPy primitive-error assumption behind the certificate margin is falsified at that size; public undercoverage is not demonstrated and nothing here claims it repaired) |
| `EV-BH-P-ADJUST` | `increment.estimation.family.bh_select` | any metric (selection over p-values) | Benjamini-Hochberg step-up rejection set at FDR level `q` over a fixed p-value family, ties selected together; index `i` is rejected iff R's BH-adjusted p-value is `<= q` | `third_party_tool` | R 4.4.2 `stats::p.adjust(method = "BH")` | none: the selection function is called directly | none claimed | exact equality of index sets (no numeric tolerance) | `bh_p_adjust.json` `aa3f442ec7eb`; `gen_bh_oracle.R` | `tests/oracles/test_bh_oracle.py::test_selected_set_matches_r_p_adjust_bh` | 7 families at `q` 0.05, 0.1 and 0.25, chosen away from the `k * q / m` boundary (see [Discrepancy log](#discrepancy-log)). Dependence conditions, family construction, e-BH and sequential selection are not compared |
| `EV-META-DL-HKSJ-HT` | `increment.estimation.meta.cochran_q`, `hksj_pooled_mean` | segment estimates with variances (log relative-lift rows) | DerSimonian-Laird tau^2, Hartung-Knapp-Sidik-Jonkman pooled mean on a `t_(K-1)` reference, Higgins-Thompson test-based I^2 interval | `third_party_tool` | R 4.4.2, metafor 4.8.0 (`rma` with `method = "DL"`, `test = "knha"`; `confint(type = "HT")`) | none: estimator on supplied estimates and variances | none claimed | tau^2 relative `1e-8`, HKSJ and I^2 interval absolute `1e-6`; closed forms, so the headroom covers summation order | `meta_dl_hksj.json` `8eaa14439a24`; `gen_meta_oracle.R` | `tests/oracles/test_meta_oracle.py::test_dl_tau2_matches_metafor`, `::test_hksj_pooled_mean_matches_metafor_knha`, `::test_i2_ci_matches_metafor_higgins_thompson` | Two cases (K = 5, K = 8). Smaller K is covered only by calibration (see below). metafor's default Q-profile interval is a different method and is not compared |
| `EV-QUANTILE-QBINOM` | `increment.estimation.quantile.log_quantile_se` | quantile | Log-scale half-width of the discrete order-statistic bracket whose ranks come from the exact binomial quantile | `independent_reimplementation` | R 4.4.2 `qbinom` (not a port of SciPy's); sample generated with numpy `default_rng(20260915)` | none: estimator on a sample | parity cell `quantile-run-none-utc-error` (four ingresses; `from_moments` refuses `source.frame.quantile_no_moments`) | relative `1e-12` on the standard error; the rank selection is bit-identical, so this is floating-point headroom | `quantile_woodruff.json` `4c7560999fa8`; `gen_quantile_samples.py`, `gen_quantile_oracle.R` | `tests/oracles/test_quantile_oracle.py::test_log_scale_half_width_matches_independent_qbinom_bracket` | The independence is of the binomial-quantile primitive, not of the interval method; `survey::oldsvyquantile` was rejected because it is a different estimator. Two samples (n = 40 at q = 0.9; n = 120 at q = 0.5) |
| `EV-ADJ-DML-AIPW-IPTW` | `increment.estimation.adjust.estimate_ate` with `Method(name="dml" / "aipw" / "iptw")` | mean (covariate-adjusted observational ATE) | Aggregation formulas given identical frozen out-of-fold nuisances: partially-linear DML slope, AIPW ATE (`normalize_ipw=True`), Hajek IPTW with an HC0 sandwich | `third_party_tool` | doubleml 0.11.4 (PLR, IRM), statsmodels 0.15.0 (WLS HC0), numpy 2.5.1, scikit-learn 1.9.1 | `from_unit_summary` | parity case `observational_aipw_dml_covariate_adjusted` covers AIPW/DML on `from_definitions`, `from_unit_day_artifact`, `from_unit_summary` and `from_unit_panel` (`from_moments` refuses `source.moments.covariate_unavailable`); no transitive IPTW standard-error comparison claimed: `observational_iptw_covariate_adjusted_ate` uses the different native fitted-logistic covariance | `1e-9` relative on estimates and on DML and IPTW standard errors; AIPW standard error `1e-4` relative (measured about `7e-5`) | `dml_aipw_iptw.json` `c426e24f472d`, `data/dml_synthetic.csv`; `gen_dml_oracle.py` | `tests/oracles/test_dml_oracle.py::test_dml_matches_doubleml_external_predictions`, `::test_aipw_matches_doubleml_irm_ate_score`, `::test_iptw_matches_statsmodels_wls_hajek` | Nuisance fitting and cross-fitting are not validated: the nuisances come from this repository's own fold assignment. One dataset (n = 500, 5 folds). The DML estimand is the partially-linear slope, not an average treatment effect ([limitations](limitations.md#dml-reports-a-partially-linear-slope-not-an-average-treatment-effect)). Clustered and trimmed variants are not compared. Default untrimmed native-logistic IPTW covariance, including propensity-fitting estimating equations, is not compared with the frozen WLS-HC0 reference |

### In-repo derivations and calibration evidence

These are not independent third-party references. They are listed so a reader can see what
exists beyond the frozen comparisons above.

| Evidence id | Capability and public symbol | Estimand and assumptions matched | Reference class | Test selector | Regimes not covered |
|---|---|---|---|---|---|
| `EV-META-POSTERIOR-QUAD` | `marginalized_segment_intervals` | Tau-marginalized segment posterior (REML marginal likelihood times a half-normal prior on tau; Morris 1983) against continuous quadrature written from the model's formulas | `independent_derivation_in_repo` | `tests/oracles/test_meta_oracle.py::test_marginalized_intervals_match_continuous_quadrature` | The recorded escaped-posterior counterexample and budget behavior are pinned, not generalized; Wald intervals on posterior means carry no frequentist coverage guarantee |
| `EV-WELCH-ARITHMETIC` | `IndependentMeanReference` | Welch component arithmetic and the Welch-Satterthwaite reference against hand arithmetic and normal/chi-square integration | `independent_derivation_in_repo` | `tests/estimation/test_independent_mean_reference.py` | Same authorship as the code under test; `EV-WELCH-T` is the third-party check |
| `EV-BINOMIAL-RR-ENUM` | Exact binomial risk-ratio sets | Decimal and `math.comb` joint-binomial enumeration (a valid lower bound on the supremum p-value) and a Bonferroni rectangle of Clopper-Pearson quantiles from `scipy.stats.beta` directly | `independent_derivation_in_repo` | `tests/estimation/test_binomial_rr.py`, `tests/estimation/test_rare_event_calibration.py` | Enumeration is bounded by `MAX_ARM_SIZE`; the grid maximum is a bound, never an equality oracle |
| `EV-CLUSTER-WITNESSES` | Cluster-randomized AIPW and DML variance | Hand-derived cases showing the between-arm superpopulation variation is retained | `independent_derivation_in_repo` | `tests/estimation/test_i13_independent_witnesses.py` | The witnesses identify particular covariance defects; they do not replace small-cluster scientific gates |
| `EV-JOINT-RELATIVE` | Joint relative (Fieller) inversion | Numerical and directional contracts of the inversion | `independent_derivation_in_repo` | `tests/estimation/test_joint_relative_reference.py` | No third-party package compared |
| `EV-CERTIFIED-ARITH` | Certified outward-rounded arithmetic layer | Mathematical behavior of the freestanding arithmetic | `independent_derivation_in_repo` | `tests/estimation/test_certified.py` | Arithmetic layer only |
| `EV-HKSJ-SMALL-K` | HKSJ pooled interval at K = 2 to 12 | Coverage of the random-effects mean over 18 cells of 10000 replications | `calibration_evidence` | `tests/calibration/test_hksj_small_k.py` | One design: unequal variances with log-scale standard errors 0.10 to 0.20; see [limitations](limitations.md#meta-analysis-can-be-anticonservative-at-small-k) |
| `EV-RARE-EVENT-GRID` | Exact-binomial relative-lift set | Support and geometry checks over all 336 cells of the truth and support manifest; pointwise Monte Carlo coverage over six prespecified cells only | `calibration_evidence` | `tests/estimation/test_rare_event_calibration.py`, `tests/estimation/test_binomial_grid_calibration.py` | Empirical coverage is measured at six preselected cells, not across the 336-cell grid; the Monte Carlo cells are marked `slow` |
| `EV-CUPED-THETA` | CUPED interval | 16 cells of 3000 replications against a known-coefficient comparator | `calibration_evidence` | `tests/calibration/test_cuped_theta_uncertainty.py` | Equal allocation and one homogeneous slope; the comparator is a delta-method construction, not exact inference |
| `EV-CATE-UNCLUSTERED-CAL` | `estimate_cate` and `validate_cate` (the same estimators behind `targeting_rule` and `select_targeting_rule`), unclustered, randomized or doubly-robust scores | Interval coverage of the ATE and one interaction slope on a Gaussian design (n = 4000, 1000 replications, accepted band 0.93 to 0.97); ATE coverage at a 10/90 allocation with unequal arm variances (n = 4000, 2000 replications, at least 0.93) and for a Bernoulli outcome (1000 replications, band 0.92 to 0.98, also at 10/90); one-sided AUTOC null rejection rate (n = 2000, 2000 replications, band 0.03 to 0.07); doubly-robust AUTOC null rejection rate under confounding (500 replications, within three Monte Carlo standard errors of 0.05, with plain IPW rejecting above 0.15) | `calibration_evidence` | `tests/test_cate_calibration.py::TestIntervalsCoverAtNominal`, `::TestSandwichSurvivesSkewedAllocation`, `::TestBernoulliOutcomeCoverage`, `::TestBernoulliOutcomeCoverageAtSkewedAllocation`, `::TestAutocNullSize`, `::TestDrPsiCalibrationUnderConfounding` | Estimator-level draws (`fit_cate`, `validate_cate_arrays`), not a source ingress; a few fixed designs whose bands are regression screens, not coverage guarantees; identification (conditional ignorability, overlap) is assumed, see [limitations](limitations.md#targeting-validation-depends-on-identification-and-overlap) |
| `EV-CATE-CLUSTER-SCREEN` | Clustered `validate_cate` statistics (AUTOC, Qini, GATES, CLAN) and clustered `estimate_cate` | Null screen of the held-out validation statistics on frozen oracle scores: 72 designs (10, 40 or 200 clusters; equal sizes of 20 or repeating sizes 5/20/100; intracluster correlation 0, 0.2 or 0.5; Gaussian or skewed errors with one high-leverage score per cluster; two cluster weightings), 256 replications and 199 bootstrap draws each, held to family-aware binomial miss bounds. A separate 288-cell diagnostic of the `estimate_cate` covariance (128 draws per cell) gates variance moments and records, but does not gate, coverage and Wald rejection. A public-path replay of `validate_cate`, `targeting_rule` and `select_targeting_rule` on eight fixed cluster witnesses records behavior | `calibration_evidence` | `tests/test_cate_calibration.py::test_c07_cluster_null_rejection_and_coverage`, `::test_cluster_fit_covariance_moments_and_coverage_diagnostics`, `::test_e6v0_original_varying_noise_cluster_public_replay` | A regression screen, not a guarantee: it calls the internal validation function with frozen oracle scores rather than a fitted pipeline, and the replay is not an error-control certificate. Few unequal clusters with skewed errors still undercover ([limitations](limitations.md#clustered-cate-uncertainty-is-cluster-asymptotic)) |

## Methods without an independent reference

Status values: `none_found` (a matched reference was not found or does not match the estimand),
`derivation_only` (in-repo derivation or enumeration only), `calibration_only` (bounded Monte
Carlo evidence only), `campaign_pending` (a registered campaign exists and has not been executed).

| Method | Status | Basis and where the guarantee is stated |
|---|---|---|
| CUPED point estimate and standard error | `calibration_only` | The ANCOVA `lm` comparison was examined and does not match: increment reports a per-arm Welch delta-method standard error on the log scale with Welch degrees of freedom, whereas ANCOVA uses a pooled residual variance with `N - 3` degrees of freedom. On a heteroskedastic scratch dataset (not frozen) the adjusted difference agreed with the ANCOVA treatment coefficient to about `1e-14` relative at equal allocation and differed by about 0.8% at 25 vs 40 units, and the standard errors differed (0.533 vs 0.553 at equal allocation, 0.533 vs 0.611 at 25 vs 40). No comparison was frozen. [Limitations](limitations.md#cuped-treats-its-adjustment-coefficient-as-known) |
| Cluster-robust contrasts (disjoint-arm Fieller set, additive Welch, cluster IPTW/AIPW/DML) | `derivation_only` | Hand-derived witnesses and a bounded diagnostic set; `sandwich::vcovCL` was not available in the generation environment, so no comparison was attempted. [Limitations](limitations.md#cluster-references-are-qualified-working-approximations) |
| Ratio metrics (delta method) and joint relative inversion | `derivation_only` | `EV-JOINT-RELATIVE`; `mratios` was not available in the generation environment, so no comparison was attempted. [Limitations](limitations.md#signed-effects-require-a-suitable-scale-and-reference) |
| Exact-binomial relative-lift set (conversion, retention) | `derivation_only` | Enumeration and a Bonferroni Clopper-Pearson rectangle; only the Clopper-Pearson primitive has a third-party comparison. [Limitations](limitations.md#rare-events-on-an-unadjusted-conversionretention-arm-are-estimated-exactly-not-refused) |
| Sequential asymptotic mean confidence sequence | `campaign_pending` | The complete empirical campaign has not been executed. [Limitations](limitations.md#the-sequential-certification-campaign-has-not-been-executed) |
| Bernoulli e-process routes | `campaign_pending` | A finite-sample derivation exists under the registered assumptions; verification of the deployed implementation is a separate obligation. [Limitations](limitations.md#the-sequential-certification-campaign-has-not-been-executed) |
| e-BH and sequential selected intervals | `none_found` | No third-party implementation of the same procedure was found. [Limitations](limitations.md#sequential-selected-intervals-use-the-same-stopped-likelihood) |
| Fixed-horizon FCR-adjusted intervals | `none_found` | [Limitations](limitations.md#fixed-horizon-fdr-control-assumes-a-dependence-condition-that-is-not-checked) |
| Winsorization inference | `calibration_only` | Typed status `experimental`; 12 alternative rows of the preserved stress grid remain unresolved. [Metric types](guides/metric-types.md#winsorization) |
| Encouragement ITT, compliance and LATE | `derivation_only` | Component-based references; no external comparison. [Encouragement guide](guides/encouragement.md) |
| Switchback unit-cycle t reference | `calibration_only` | An analytic lower bound on the miscoverage of the unit-t pivot in one small-N cell (0.0565 against a nominal 0.05; `tests/simulate/test_unit_cycle_calibration.py::test_corrected_counterexample_and_corrected_envelope`) is a derivation that contradicts nominal coverage there. Two slow parameter-recovery cells run the default independent unit-cycle path end to end (`tests/simulate/test_switchback.py::test_supported_switchback_calibration_is_covered` and `::test_switchback_null_effect_type_i_calibration`: 120 units, 4 cycles, 24 replications each, treatment effects 2.0 and 0.0); each asserts only that unconditional coverage lies between 0.80 and 1.0 and that bias is below 0.15, a weak floor on a small replication count that does not establish nominal coverage. [Limitations](limitations.md#the-switchback-t-reference-is-itself-an-approximation) |
| Switchback shared-block t reference | `calibration_only` | One slow parameter-recovery cell exercises the shared-schedule block-t path (`tests/simulate/test_switchback.py::test_shared_schedule_block_t_refuses_degenerate_cycle_order_and_recovers_coverage`): 3 units, 6 blocks, `probability_ct=0.8`, 400 replications, null treatment effect. About a quarter of replications realize a single cycle order and are refused (`source.frame.switchback.schedule`); the asserted bound is coverage of at least 0.85 conditional on an interval being produced, a floor and not nominal coverage. No other shared-block cell is inventoried, and the independent unit-cycle calibration cells declare independent Bernoulli unit-cycle orders only. |
| Fixed-horizon arm sample-size, power and MDE solvers (log-ratio planning model) | `calibration_only` | The planning scale is the log relative lift with a planned variance, whereas `power.t.test` is an additive pooled-variance, equal-allocation calculation; the estimands do not match and no estimator was added to force a match, so no third-party comparison exists. Bounded Monte Carlo evidence is one slow cell (`tests/power/test_calibration.py::TestCalibration::test_power_calibration`: a Negative Binomial count at a 10% lift planned for 0.80 power, 200 replications through the full pipeline, empirical power within 0.07 of the design-aware analytic power, and planning power at most 0.05 above it); it is not a calibration grid. [Limitations](limitations.md#arm-power-planning-depends-on-declared-alternative-arm-variance-shapes) |
| Exact-binomial planning for unadjusted, unclustered, fixed-horizon conversion and retention (`power_basis="exact"` and `"approximate"`) | `derivation_only` | The exact route is compared with an enumeration of the runtime's own decisions (`tests/power/test_binomial_planning.py::TestExactRouteMatchesRuntime`: three cases to `1e-11` absolute, and every decision of three arm-size cells equal to the runtime's). The approximate route is held to the errors measured against the runtime on eight retained witness rows (`TestApproximateRoute`) and a rare large-arm decision witness. Both are in-repository comparisons with no third-party tool. [Limitations](limitations.md#conversion-planning-matches-the-exact-binomial-decision-only-within-a-budget) |
| Sequential sample-size, power and MDE planning (`GaussianScoreMixture` boundary) | `derivation_only` | The probability of crossing the planned boundary under a drifted Normal path is compared with integrals that share no code with `increment.power.sequential` (`tests/power/_references.py`): closed-form one-look tails, a two-look conditional-normal integral by `scipy.integrate.quad`, and a Simpson-rule recursion for three or more looks (`tests/power/test_sequential_evaluator.py::TestOneLookExactTails`, `::TestTwoLookQuadReference`, `::TestMultiLookSimpsonReference`). The MDE inversion is checked against the Simpson power at its pinned candidate effects for 5 and 14 looks (`tests/power/test_sequential_inverse.py::TestOrdinaryInversions`, slow). The plan-time boundary is held equal to the runtime's asymptotic-mean boundary (`tests/estimation/test_asymptotic_mean_planning_equivalence.py`). One slow cell runs 4000 replications of Normal arms through that runtime boundary (`tests/power/test_asymptotic_mean_sizing.py::test_planned_n_achieves_nominal_power_under_the_runtime_boundary`: 14 looks, 5% lift, equal arm variances as planned) and asserts empirical power of at least 0.80 minus three Monte Carlo standard errors at the planned size; it is one design. [Planning guide](guides/power-analysis.md#fixed-and-sequential-planning) |
| Switchback planning (`switchback_achieved_power`, `switchback_minimum_detectable_effect`, `switchback_required_blocks_or_units`) | `derivation_only` | The moment-t noncentral-t power (`power_kind="moment_t_approximation"`, an approximation and not a finite-branch rejection law) is compared with direct numerical integration of the noncentral-t over its chi factor and with analytic quadratic roots at hand-computed two-branch moments (`tests/power/test_switchback.py::test_independent_branch_oracle_and_two_analytic_roots`), and the two-period case with a closed form (`::test_two_period_power_matches_the_one_dof_oracle_and_replays`). Shared-schedule power scaled by the admission probability is checked against a 200,000-replication simulation of the two-stage process within four Monte Carlo standard errors (`::test_shared_schedule_power_available_matches_bounded_simulation`, slow). A slow 32-cell experiment checks the planning variance against the actual source and estimator under a frozen bounded law; it gates variances only, not coverage or power (`tests/power/test_switchback_planning_calibration.py::test_source_estimator_and_planning_variance_calibration`). Planning conditions on pilot estimates and does not account for their uncertainty. [Limitations](limitations.md#switchback-planning-requires-its-own-model) |
| Switchback unit-cycle power lower bound (`unit_cycle_power_lower_bound`) | `derivation_only` | A Cantelli lower bound on the rejection probability under a prospective residual-variance envelope, recomputed in exact rationals (`tests/power/test_unit_cycle_power.py::test_lower_bound_matches_rational_cantelli_and_keeps_effect_scale`), with the unit-cycle law moments checked against an independent full-mask enumeration (`::test_moments_match_independent_full_mask_enumeration`). Its validity rests on the envelope assumption, which must be justified externally, and a zero bound is not an unattainability claim. [Limitations](limitations.md#the-switchback-t-reference-is-itself-an-approximation) |
| Segment-heterogeneity power (`segment_pairwise_*`, `joint_q_power_fixed`, `joint_q_power_random`) | `calibration_only` | `joint_q_power_fixed` is compared with a 2000-replication simulation at one unequal-variance cell (absolute difference below 0.05, `tests/test_hte_calibration.py::TestPowerSolverCalibration`); `joint_q_power_random` with an 80,000-replication simulation at one cell with an 8-fold variance spread (relative difference below 5%, `tests/power/test_core.py`, `test_random_effects_fixed_effects_agree_on_calibration`), is exact only for equal variances, and its docstring reports up to +12.8% relative error at a 50-fold spread; the pairwise sizing formula is recomputed in closed form and its achieved power is held within one percentage point of the target (`tests/test_hte_calibration.py::test_pairwise_power_non_partition_shares`). [Heterogeneity guide](guides/heterogeneity-and-rollout.md) |
| Off-policy evaluation of logged decisions | `calibration_only` | Fixed-logger coverage cells; adaptive loggers are refused. [Limitations](limitations.md#off-policy-evaluation-of-logged-decisions-is-calibrated-only-under-a-fixed-logger) |
| Positivity gate and weight diagnostics | `none_found` | A fixed threshold with no weight diagnostics. [Limitations](limitations.md#the-positivity-gate-is-a-fixed-threshold-with-no-weight-diagnostics) |
| Total, active and retention metric types | `none_found` | No frozen third-party comparison exists for these metric types. [Compatibility](guides/compatibility.md) |
| CATE estimation and targeting (`estimate_cate`, `validate_cate`, `targeting_rule`, `select_targeting_rule`) | `calibration_only` | `EV-CATE-UNCLUSTERED-CAL` and `EV-CATE-CLUSTER-SCREEN` are bounded Monte Carlo cells; no third-party comparison is recorded. Group-effect and characteristic tests are Bonferroni-corrected inside their own families and no guarantee spans the whole workflow. [Identification and overlap](limitations.md#targeting-validation-depends-on-identification-and-overlap), [no joint error control](limitations.md#the-targeting-workflow-carries-no-joint-error-control), [clustered uncertainty](limitations.md#clustered-cate-uncertainty-is-cluster-asymptotic) |

## Unvalidated regimes

These restate published gaps; this page adds no new number. Each links to the section that states
the guarantee.

- Sequential inference: the complete certification campaign over the registered runtime manifest
  has not been executed, and clustered and unit-cycle research matrices remain uncertified
  ([limitations](limitations.md#the-sequential-certification-campaign-has-not-been-executed)).
  Bounded release evidence cannot establish universal calibration.
- Winsorization inference is experimental, and 12 alternative rows of its preserved stress grid
  remain unresolved ([metric types](guides/metric-types.md#winsorization)).
- Meta-analysis at small K: the plug-in random-effects interval is anticonservative, and HKSJ is
  bounded by one calibration design
  ([limitations](limitations.md#meta-analysis-can-be-anticonservative-at-small-k)).
- Switchback carryover is assumed away, not tested, and the unit-cycle t reference is an
  approximation ([carryover](limitations.md#switchback-carryover-is-assumed-away-not-tested),
  [t reference](limitations.md#the-switchback-t-reference-is-itself-an-approximation)).
- Off-policy evaluation of logged decisions under adaptive logging is refused. The coverage
  figures for batch-refit adaptive loggers (86.4% epsilon-greedy, 89.6% Thompson, 87.2% contextual
  Thompson) were measured before the refusal gate existed. Their provenance is recorded below
  ([limitations](limitations.md#off-policy-evaluation-of-logged-decisions-is-calibrated-only-under-a-fixed-logger)).
- Exact-binomial arms above about 2^26 per arm: the certificate margin's SciPy primitive-error
  assumption is falsified at that size and is tracked as an open numerical investigation
  (internal tracker `1tce`); public undercoverage is not demonstrated.
- Causal-feature calibration against independent oracles (internal tracker `ryyg`) is open; this
  page makes no claim about it.

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

Regenerating a fixture needs the tools below and is a deliberate act: a numeric difference found
by regeneration is a discrepancy to record, never a reason to widen a tolerance. Generators run
from the repository root. Digests are SHA-256 of the committed files.

| Oracle | Generator command | Tool versions | Committed digest | One-time regeneration (2026-10-03, macOS arm64) |
|---|---|---|---|---|
| `EV-META-DL-HKSJ-HT` | `Rscript tests/oracles/generate/gen_meta_oracle.R` | R 4.4.2, metafor 4.8.0, jsonlite 1.8.9 | `meta_dl_hksj.json` `8eaa14439a24`, introduced in `94f4b6b` (2026-10-01) | byte-identical |
| `EV-QUANTILE-QBINOM` | `uv run python tests/oracles/generate/gen_quantile_samples.py`, then `Rscript tests/oracles/generate/gen_quantile_oracle.R` | R 4.4.2, numpy 2.5.1 (the fixture records only R) | `quantile_woodruff.json` `4c7560999fa8`, `data/quantile_values.csv` `c65951504fa2`, introduced in `94f4b6b` | CSV and fixture byte-identical |
| `EV-ADJ-DML-AIPW-IPTW` | `uv run --python 3.12 --with numpy==2.5.1 --with scikit-learn==1.9.1 --with statsmodels==0.15.0 --with doubleml==0.11.4 python tests/oracles/generate/gen_dml_oracle.py` | CPython 3.12, numpy 2.5.1, scikit-learn 1.9.1, statsmodels 0.15.0, doubleml 0.11.4 | `dml_aipw_iptw.json` `c426e24f472d`, `data/dml_synthetic.csv` `99473fdc78f2`, introduced in `94f4b6b` | CSV byte-identical; fixture differs only in the trailing newline, every value identical |
| `EV-WELCH-T` | `Rscript tests/oracles/generate/gen_welch_oracle.R` | R 4.4.2, jsonlite 1.8.9 | `welch_t.json` `abf22cc43957` | generated for this page; recorded generating commit `37b9b34` |
| `EV-CP-BINOM-TEST` | `Rscript tests/oracles/generate/gen_clopper_pearson_oracle.R` | R 4.4.2, jsonlite 1.8.9 | `clopper_pearson.json` `7f53220dd69a` | generated for this page; recorded generating commit `37b9b34` |
| `EV-BH-P-ADJUST` | `Rscript tests/oracles/generate/gen_bh_oracle.R` | R 4.4.2, jsonlite 1.8.9 | `bh_p_adjust.json` `aa3f442ec7eb` | generated for this page; recorded generating commit `37b9b34` |

The three older fixtures predate provenance blocks: they record tool versions but not the
increment commit, generation date, platform, `jsonlite` or Python version, so the commit and date
above come from `git log`. The three newer fixtures carry `generator`, tool versions,
`generated_on`, `increment_commit` (the parent of the commit that added them) and a one-sentence
`estimand`. The regeneration run used a scratch worktree and nothing it wrote was committed.

## Report a discrepancy

1. Open a [GitHub issue](https://github.com/kylejcaron/increment/issues/new) naming the increment version or commit, the entry point, and the reference tool and version.
2. Give the two numbers (increment's and the reference's) and the tolerance at which you consider them to disagree.
3. Attach per-arm aggregate moments or the reference tool's output; never identifiers, credentials, or raw private data.

A reported discrepancy is listed in the [Discrepancy log](#discrepancy-log) until it is fixed or
explained with evidence. It is not closed as dismissed.

## Case ledger

No independently sourced case has been recorded yet.

An entry belongs here only when someone who is neither a maintainer nor an automated agent supplies
a real experiment's edge case (aggregate moments or sanitized rows) or an independent review run of
a matched comparison, with permission to describe it, enough information to rerun it on a stated
increment version, and their result. Maintainer-, agent- or synthetic-only comparisons, including
every row above, do not count as independently sourced.

## Discrepancy log

| Date | Comparison | Observation | Status |
|---|---|---|---|
| 2026-10-03 | `EV-BH-P-ADJUST` | For the family `[0.01, 0.5, 0.01, 0.01, 0.2]` at `q = 0.25`, R selects four indices and `bh_select` selects three. The fifth p-value, 0.2, equals `k * q / m` for `k = 4, m = 5` in decimal. R computes the adjusted value `5/4 * 0.2` as exactly 0.25 and includes it. The stored double 0.2 is slightly above 0.2, and `_conservative_ratio` rounds the threshold downward by design, so the p-value exceeds the threshold on the stored doubles and `bh_select` excludes it. | explained (measured on this input; the rounding direction is the documented behavior of `_conservative_ratio`). The fixture's vectors avoid exact boundaries. |
| 2026-10-03 | Regeneration of the three earlier fixtures | No numeric difference. The DML fixture differs only in the trailing newline. | explained (byte comparison) |

No other discrepancy has been found between increment and the references above.
