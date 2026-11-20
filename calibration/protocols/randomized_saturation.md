# Randomized saturation: can untreated people benefit from more treated peers?

## Question, design and estimand

This fixed-horizon evaluation of [#95](https://github.com/kylejcaron/increment/issues/95) concerns assignment, not actual uptake. Freeze a finite roster of `N` eligible people partitioned into clusters of sizes `n_g` and strata `h`. In each stratum independently, choose exactly `m_hH` of `G_h` clusters uniformly for high saturation; the other `m_hL` receive low. Given its level `r`, each cluster independently treats a uniform subset of exactly `k_gr` members, with `0 < k_gL < k_gH < n_g`. Use the declared integers, not intended percentages or observed arm frequencies. Own untreated exposure at `r` has probability `(m_hr/G_h) (n_g-k_gr)/n_g`.

Let `mu_gir` be person `i`'s expected fixed-horizon response **when assigned untreated**, averaging over exactly `k_gr` treated peers sampled from the other `n_g-1` members. Peer identities may matter. The target is `tau = sum_gi(mu_giH - mu_giL)/N`, not a contrast among two changing pools of untreated people. Assume no influence from other clusters and well-defined assignment and measurement. Interpreting a policy effect as a peer effect further rules out a separate effect of announcing the policy.

## Candidate and error model

For each cluster let `T_g = n_g * mean(observed untreated outcomes)` and `t_gr = sum_i mu_gir`. Inverse untreated probability gives `E_2[T_g | r] = t_gr`; inverse first-stage probability then gives `E[tau_hat] = tau`, where

```
w_h     = G_h / N
tau_hat = sum_h w_h (mean_H(T_g) - mean_L(T_g))
V_hat   = sum_h w_h² (s²_H(T_g)/m_hH + s²_L(T_g)/m_hL).
```

The sample variances use clusters and divisor `m_hr-1`, not person rows. Write `S²_h` for finite-population variance across the `G_h` clusters (divisor `G_h-1`), and `v_gr = Var_2(T_g | r)`. Conditional variance adds `mean_g(v_gH)/m_hH + mean_g(v_gL)/m_hL` per stratum to both the expected variance estimate and the sampling variance (before multiplying by `w_h²`). Between-policy allocation adds `S²_h(t_gH)/m_hH + S²_h(t_gL)/m_hL` to both, **minus** `S²_h(t_gH - t_gL)/G_h` only for the sampling variance. Thus

```
E[V_hat] - Var(tau_hat) = sum_h G_h S²_h(t_gH - t_gL) / N² >= 0.
```

This is conservatism *in expectation*, not coverage of a realized interval. The candidate tests two-sided `H0: tau=0` at alpha `0.05`: reject exactly when `abs(tau_hat) > z * sqrt(V_hat)`, with `z = norm.isf(0.025)`. If any stratum has fewer than two clusters in either arm, or variance is zero or nonfinite, the interval is unavailable and there is **no rejection**. Keep the point; an unavailable interval does not cover. Many informative clusters, no dominant contribution and adequate untreated exposure would be necessary for a Normal approximation; a cluster count or positive marginal exposure alone is not a calibration argument.

## Frozen evaluation

Earlier exploratory calculations had already identified the tiny, sentinel and hypergeometric examples. This is an independent implementation and arithmetic comparison, **not** a retrospective preregistration or blind validation. The following laws, outcome schedules, grid and criteria were fixed for this run. There is no sampling, parameter fitting, sequential look or Monte Carlo extrapolation: seed **none**, random draws **zero**, MC error **zero**. Use integer multiplicities and rational moments and probability masses. Evaluate the Normal upper-tail quantile via its lower-tail equivalent `-NormalDist().inv_cdf(0.025)`; compare squared errors to its square. Rates use **joint mass of all attempted assignments**, including every refusal. Record states evaluated, unavailable by reason, rejection of zero, available intervals missing the truth, and misses plus unavailability. Compressed states have unequal masses. Inputs are synthetic and complete; unexpected exceptions fail the evaluation rather than dropping a state.

**Predeclared comparison.** Require exact equality of rational means and variances to the arithmetic oracles below. For a correctly specified null fixture, flag the interval if either the exact null rejection probability or the probability of noncoverage-or-unavailability exceeds `0.055` (nominal `0.05` plus `min(0.005, 0.1*alpha)`, the repository's absolute scientific tolerance). No binomial confidence bounds or multiplicity adjustment for MC noise apply to exact sums. A power crossing is interpretable only for the same null-calibrated fixture; a failure blocks release, not publication of the adverse result. No claim about behavior between grid points.

1. **Unequal sizes.** One stratum with sizes `(3,3,3,3,6)`, two high and three low; treat `2 n_g/3` or `n_g/3`. Enumerate every stage-one subset and every stage-two subset (10 × 3⁴ × 15 assignments). Person and cluster indices start at zero. For untreated people use `g+i` plus the following shifts: (a) none, (b) one under high, (c) cluster-specific `(2,-1,0,1,-1)` under high, a heterogeneous person-weighted zero-average null. An own-treatment effect of +7 can be assigned to treated responses; they never enter this estimand. Independently derive moments using `t_gr = n_g * (g + (n_g-1)/2 + shift_gr)` and `v_gr = n_g² (1-u_gr/n_g) S²(i)/u_gr`, where `u_gr=n_g-k_gr`. Also report the naive difference of the **pooled** untreated person means. Independent sharp-null and unit-shift anchors: pooled expectations `-2/15` and `13/15`; sampling variances `18245/1296` and `20189/1296`; expected variance estimates `18245/1296` and `20225/1296`.
2. **Leakage control.** Same unequal-size roster; add only the fraction of *other clusters* assigned high. Holding their policies fixed, the own-cluster effect is zero. Fixed stage-one counts imply shifts `1/4` for a high cluster and `2/4` for a low cluster. The full allocation contrast must recover `-1/4`, **not** the controlled own-cluster effect. Count decisions against zero descriptively; this is an assumption violation, not a null calibration cell.
3. **Stratification.** Four size-three clusters (two high) and five size-six clusters (three high), same within-cluster quota fractions as above. Within each stratum the untreated baseline is the zero-based cluster index; the high shift is `+5` in the first stratum, `-2` in the second. All members share their cluster's outcome so the second stage integrates out. The person-weighted effect is zero; enumerate the independent fixed-count draws, not a pooled stage-one draw. The sampling variance and mean candidate variance must both equal `(4²*15 + 5²*75)/42²`: the baseline cluster-total variances are 15 and 90, respectively.
4. **Rare exposure / influential outcomes.** Forty clusters per level, size 100, exactly one person per cluster has untreated response one and all other responses are zero, irrespective of level. Use high/low treated quotas `(75,25)` and `(99,1)`. First-stage permutations are symmetric. The two counts `b_r` of distinguished people left untreated are independent `Binomial(40, u_r/100)`; use their product mass, not a binomial model for all rows. Feed `b_r` observed cluster totals `100/u_r` and `40-b_r` zeros into the same candidate. This is a sharp no-spillover null with fixed potential responses.
5. **Matched planning.** One stratum, `2m` size-three clusters, half with constant baseline one, half zero, identical people within cluster; treat two or one. High adds `d` to each untreated response. With `x` of the `m` ones allocated high, `P(x)=choose(m,x)²/choose(2m,m)`. Stage two affects who is observed but not this cluster's mean. Reconstruct observed totals and compare the candidate for **each** state to `tau_hat=d+2x/m-1`, `V_hat=2(x/m)(1-x/m)/(m-1)` (variance unavailable for `m=1`). Null cells at `m=1,3,40`; power grid at `m=3,40` over `d=0.00,0.01,...,0.50`. Record rejection of zero and interval availability across all attempts. Report the first grid point reaching 80% only when the matching null passes both error criteria; otherwise distinguish a failed null from a grid not attaining power. This is a specified outcome schedule, not a power model learned from pilot data.

## Ingress and limits

A usable study would need a versioned **pre-assignment** roster (person, cluster, stratum, eligibility, unique IDs), both declared uniform assignment laws and integer quotas, and auditable realized stage-one **and** stage-two logs linked to that roster. Reconcile levels, counts, within-cluster selection and unchanged membership before using the measurements. Require a prespecified horizon, outcome definition and ascertainment for every expected untreated person. Unknown assignment is not low/untreated; missing response is not zero. Selective missingness, post-assignment membership and treatment receipt cannot substitute for those records. A randomization log cannot establish absence of cross-cluster interference. Aggregate arm summaries, switchback schedules and whole-cluster assignments do not identify this design.

The study excludes overlapping rosters, adaptive saturation, exposure defined by observed behavior, unknown spillover networks, repeated looks and population extrapolation. Sizing would additionally require a credible outcome/dependence model, including influential people and variation across clusters, matched to the proposed **cluster counts, sizes, strata and both quotas**, not just total enrollment or an ICC. If these assumptions or exposure support are implausible, redesign the randomization or measure the relevant network; these results are not a substitute.

## Reproduction

From the repository root, run `python calibration/randomized_saturation.py` (Python standard library only). The command checks exact identities and prints the weighted decision record and grid summary. No public estimator, reader or planner is installed. The design follows allocation-policy estimands in [Hudgens and Halloran (2008)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2600548/); see also [Baird et al., randomized saturation designs](https://pdel.ucsd.edu/_files/WorkingPaper_Spillovers_20160902.pdf) for the distinct model-based planning problem. The [method acceptance criteria](../../CONTRIBUTING.md#adding-a-method-or-design) apply before any production reconsideration.

## Executed evaluation and decision (2026-11-19)

**Defer production inference and sizing; reject the proposed Normal decision as a general 5% procedure.** The design identifies the roster-weighted policy contrast under partial interference. It is the interval, not identification under that assumption, that fails on complete, correctly randomized rosters. Even 40 clusters per arm with both levels interior did not suffice. The research command is the only implementation.

Command executed from the worktree: `.venv/bin/python calibration/randomized_saturation.py` (exit 0). The exact moment, mass and statewise hypergeometric comparisons completed. The tables transcribe its output; the linked [research command](../randomized_saturation.py) reproduces the integer numerators and full grid. No replications were sampled or discarded.

**Unequal groups.** Each row enumerated 12,150 equally likely assignments. “Misses” counts available intervals excluding the target, including the controlled zero reference in the leakage row. No interval in these rows was unavailable.

| Untreated outcome schedule | Target | E(estimate) | E(pooled untreated difference) | Var(estimate) | E(variance estimate) | Reject zero / 12,150 | Misses / 12,150 |
|---|---:|---:|---:|---:|---:|---:|---:|
| No spillover | 0 | 0 | −2/15 | 18245/1296 | 18245/1296 | 162 | 162 |
| High adds one | 1 | 1 | 13/15 | 20189/1296 | 20225/1296 | 160 | 151 |
| Heterogeneous zero-average effect | 0 | 0 | −1/30 | 6985/648 | 3605/324 | 615 | 615 |
| Other clusters' policies only | 0 controlled | −1/4 | −23/60 | 154927/10368 | 154945/10368 | 145 | 145 |

The naive contrast missed even the sharp null: setting the same treated fraction in every cluster assigned a given policy does not make the realized untreated pools comparable across policies. The other-cluster control illustrates a different error: comparison within this allocation law changes the other clusters' high fraction, even with perfect logs. Its `−1/4` is not spillover from one's own cluster.

**Null decisions.** All probabilities below have the full assignment distribution as denominator, not just the assignments yielding intervals. “Miss” is an available interval excluding the true zero, so here it equals rejection. Decimal probabilities are rounded; the two tiny positive refusal rates are given exactly. The stratified row has two strata of four and five clusters.

| Null schedule | Evaluated states | Probability denominator¹ | P(reject) = P(miss) | P(unavailable), reason | P(miss or unavailable) | Meets both 0.055 limits? |
|---|---:|---:|---:|---|---:|---|
| Unequal sizes, sharp | 12,150 | 12,150 | 0.01333333 | 0 | 0.01333333 | Yes |
| Unequal sizes, heterogeneous | 12,150 | 12,150 | 0.05061728 | 0 | 0.05061728 | Yes |
| Stratified, heterogeneous | 60 | 60 | **0.20000000** | 0 | **0.20000000** | No |
| Sentinel, 75 vs 25 treated | 1,681 | 4⁸⁰ | **0.06157826** | `((1+3^40)/4^40)^2`, zero variance | **0.06157826** | No |
| Sentinel, 99 vs 1 treated | 1,681 | 100⁸⁰ | **0.22149770** | **0.44752321**, zero variance | **0.66902091** | No |
| Homogeneous clusters, 1 per arm | 2 | 2 | 0 | 1, insufficient clusters | 1 | No |
| Homogeneous clusters, 3 per arm | 4 | 20 | 0 | 2/20, zero variance | **0.10000000** | No |
| Homogeneous clusters, 40 per arm | 41 | `C(80,40)` | 0.04350955 | `2/C(80,40)`, zero variance | 0.04350955 | Yes |

¹ Integer mass after integrating out assignments giving the same observations. For instance the sentinel count pairs have binomial-product weights, **not** probability `1/1681`. Every stated null had mean estimate zero. The stratified row rejected in 12 of 60 draws despite `Var(estimate) = E(variance estimate) = 235/196`. The two sentinel pairs also had equality of variances (`1/120000` and `4901/19800000` respectively). There were no nonfinite-variance refusals. Zero variance is possible when all informative people fall on the same side within each level; declaring those intervals unavailable does not fix excessive rejection on the remaining mass.

**Power conditional on the specified outcome schedule, unconditional over assignments.** At three clusters per arm no effect on the grid attained 80%; power was zero on every grid point and the matching null also failed availability. With 40 per arm, the common assignment denominator was `C(80,40) = 107507208733336176461620`. Exactly two assignments by this count were unavailable at every grid point; every available interval that missed the true shift was counted.

| Shift in outcome units | Rejecting assignment mass | P(reject) | P(miss or unavailable at true shift) |
|---:|---:|---:|---:|
| 0.00 | 4677590724673981944218 | 0.04350955 | 0.04350955 |
| 0.32 | 80489588921742610073198 | 0.74869016 | 0.04350955 |
| **0.33** | 93344672687579149313198 | **0.86826431** | 0.04350955 |

The **first 80% grid crossing was 0.33** for the 80-cluster binary-baseline schedule. This is not a continuous MDE: discrete two-sided power even fell from 0.08025487 at `d=0.03` to 0.06520399 at `d=0.04`. It is not a recommended sample size, a binary risk difference (shifted outcomes are not constrained to 0/1), or a power claim for the sentinel or stratified failures.

**What would change the decision?** A candidate that controls both null rejection and the combined probability of noncoverage or unavailability in a specified use region, with checkable exposure and influence conditions, plus an ingress contract that preserves the experimental roster and both assignment stages. Its planning calculation must repeat that same decision and its refusals under credible prospective outcomes. A larger participant count, agreement of expected variances, more exact power-grid points, or tighter Monte Carlo intervals cannot repair the measured failures of this candidate. The [current assignment evidence and replay contract](../../docs/reference/artifacts-and-replay.md#assignment-evidence-and-replay) does not by itself provide the two-stage evidence required here.
