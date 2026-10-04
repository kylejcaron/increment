# External validation coverage

Frozen third-party references exist for a small set of estimators. This page states which, at what
tolerance, through which entry points, and which regimes have none. It is an inventory of evidence,
not a certification: a row means the named estimand agreed with the named reference at the stated
tolerance on the stated inputs, never that a whole method is validated. Campaigns that have not
been executed are listed as unexecuted, and the guarantees themselves are stated on
[Statistical limitations](limitations.md).

Verified at commit `02b8eb2` (2026-10-03): every test selector below resolves under
`uv run pytest --collect-only -q <selector>` (add `-m slow` for the slow-marked calibration
modules), and every generator command in [Reproduce](#reproduce) was run once.

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
| Switchback unit-cycle t reference | `calibration_only` | A small-N counterexample to nominal coverage is recorded. [Limitations](limitations.md#the-switchback-t-reference-is-itself-an-approximation) |
| Switchback shared-block t reference | `none_found` | No calibration cells for the shared-block reference are inventoried: `unknown`. The unit-cycle calibration cells declare independent Bernoulli unit-cycle orders only. |
| Sample-size, power and MDE solvers | `none_found` | The planning scale is the log relative lift with a planned variance, whereas `power.t.test` is an additive pooled-variance, equal-allocation calculation; the estimands do not match and no estimator was added to force a match. [Limitations](limitations.md#arm-power-planning-depends-on-declared-alternative-arm-variance-shapes) |
| Off-policy evaluation of logged decisions | `calibration_only` | Fixed-logger coverage cells; adaptive loggers are refused. [Limitations](limitations.md#off-policy-evaluation-of-logged-decisions-is-calibrated-only-under-a-fixed-logger) |
| Positivity gate and weight diagnostics | `none_found` | A fixed threshold with no weight diagnostics. [Limitations](limitations.md#the-positivity-gate-is-a-fixed-threshold-with-no-weight-diagnostics) |
| Total, active and retention metric types | `none_found` | No frozen third-party comparison exists for these metric types. [Compatibility](guides/compatibility.md) |

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
