# Relative-prior statistical validation protocol

## Status and scope

Frozen before generating outcomes for this investigation. Seed, sample counts, scenarios and pass criteria below are not changed after observing results. This is preimplementation evidence for the owner-approved direct-relative Normal working likelihood, not production-ingress certification. No physical-support truncation, new eligibility policy or alternative likelihood is introduced.

The statistical distinction is essential: accurate numerical integration, calibration under an assumed joint model, and approximation quality for raw experiment data are different claims.

## 1. Model-conditional calibration

For each family, generate independent latent effects r from exactly the analysis prior and observations x=r+s*z with z~Normal(0,1), fixed known s=0.10. Evaluate U=posterior_CDF(r | x,s). A correct continuous posterior gives uniform U under this joint model, including when the prior has no finite moments.

Four families, 2048 replications each:

1. Normal(mean=0, SD=0.10).
2. Normal mixture with weights (0.8,0.2), means (0,0.20), SDs (0.05,0.15).
3. Zero-centered Student-t(nu=1, scale=0.10).
4. Zero-centered Student-t(nu=4, scale=0.10).

Normal/mixture updates are independently coded from conjugacy; Student-t uses the existing research candidate without editing it. Mixture component selection is latent simulation input, not conditioned on by the posterior. No quantile solver is required for PIT; central 95% coverage is the event 0.025<=U<=0.975.

**Randomness:** NumPy Generator with SeedSequence(2026111503). Spawn independent streams in the listed family order, followed by the raw scenario order below and their proxy-observation streams. Record versions and seed indices. Never reroll a failure.

**Predeclared pass criteria:** zero numerical failures/nonfinite probabilities, probabilities in [0,1], and both checks below. Family-wise false-rejection allocation is at most 0.01 across the four families and two checks per family:

- Uniformity: empirical two-sided CDF distance D must not exceed sqrt(log(2/alpha_cell)/(2*N)), alpha_cell=0.005/4. This is the finite-sample Dvoretzky–Kiefer–Wolfowitz bound, not an asymptotic fitted-distribution test.
- Central 95% inclusion: the two-sided exact Binomial(N,0.95) test has p>=0.005/4. Report count and exact confidence interval as well as p.

A failure is investigated, not erased by increasing N or changing the seed. Numerical failures are not silently removed from the denominator. Empirical success does not certify all parameter values or real-data calibration.

## 2. Raw-outcome approximation study

Use 4096 independent experiments per scenario, batched in at most 128 experiments. Equal arm sizes n=200 and n=2000; fixed true relative effects r=0 and r=0.20. Cross seven families with those sizes/effects: 28 prespecified scenarios. These are exploratory scientific scenarios, not 28 permanent test items.

The first five families are independent arms with control estimand 1 except binary outcomes. Treatment estimand is control*(1+r).

| Family | Data-generating process | Estimand and uncertainty |
|---|---|---|
| Gamma mean | Gamma shape 1, arm-specific scale equal to the arm mean | Mean ratio; unbiased sample arm variance / n |
| Skewed lognormal mean | Log variance log(10), log location log(arm mean)-log(10)/2; CV=3 | Mean ratio; sample arm variance / n |
| Dense binary | Bernoulli control p=0.20, treatment p=0.20*(1+r) | Conversion ratio; unbiased sample variance / n |
| Sparse binary | Bernoulli control p=0.01, treatment p=0.01*(1+r) | Same; retain all zero/saturated and unreliable outcomes in refusal counts |
| Correlated numerator/denominator ratio | D~Gamma(shape=2,scale=1), V~Exponential(mean=arm estimand), independent within unit; numerator N=D*V | Arm ratio E[N]/E[D]; centered delta variance including Cov(N,D) |
| Fixed-reference scalar adjustment | X~Normal(0,1), epsilon~Normal(0,0.6^2), Y=arm mean+0.8*X+epsilon; known coefficient 0.8 and fixed reference E[X]=0 | Adjusted mean of Y-0.8*X; actual residual sample variance / n |
| Shared-reference scalar adjustment | Same data and fixed coefficient, but adjusted arm means use the pooled X mean as reference | Preserve induced cross-arm covariance through independent-arm influence contributions; do not treat adjusted means as independent |

### Independent variance checks

For independent arm estimators T,C with variance estimates Vt,Vc, direct relative variance is Vt/C^2 + T^2*Vc/C^4. Independently compute per-arm log SEs and s_relative=(T/C)*hypot(se_log_t,se_log_c). Compare the two on every eligible row using a scale-aware floating-point tolerance; report maximum discrepancy. Never call the existing relative-likelihood helper as its own oracle.

For a ratio arm, the direct covariance expression is [Var(N)+Rhat^2*Var(D)-2*Rhat*Cov(N,D)]/(n*Dbar^2). Compare against Var(N-Rhat*D)/(n*Dbar^2). In the population, E[D]=2, Var(D)=2, E[D^2]=6, E[V]=m and Var(V)=m^2, giving per-unit ratio influence variance 1.5*m^2 and log influence variance 1.5. These are independent derivation checks, not fitted expectations.

For shared-reference adjustment, with equal n and beta=0.8:

- C=Ybar_c-beta*(Xbar_c-Xbar_pool), T=Ybar_t-beta*(Xbar_t-Xbar_pool).
- Put epsilon_a=Y_a-beta*X_a and g=beta/2*(1/T-1/C).
- Independent-arm log influence scores are g*(X_c-Xbar_c)-(epsilon_c-epsilonbar_c)/C and g*(X_t-Xbar_t)+(epsilon_t-epsilonbar_t)/T.
- The direct relative-gradient contributions are beta/2*(1/C-T/C^2) on each X arm, -T/C^2 on epsilon_c and 1/C on epsilon_t. Compute their variance with each arm's empirical X/epsilon covariance, then compare against (T/C)^2 times the log-score variance sum.

The coefficient is fixed by construction. This does not certify fitted-coefficient CUPED, multicovariate CUPED, or production evidence acquisition. The shared reference specifically probes the cross-arm covariance hazard; its reduction to independent raw-arm scores is not a claim that adjusted arm means are independent.

### Admission and reporting population

Call the existing `infer_lift` with the derived joint log ratio and per-arm independent log-score SEs, prior=None, method_role='decision', no sequential/cluster reference. Read-only use of this estimator does not implement the proposed prior wrapper. Count its actual LiftGuardError reasons. Before taking logarithms, classify nonpositive/nonfinite arm estimands explicitly; these are standalone data-admission exclusions, not invented public refusal codes.

Every scenario reports attempted count=4096, estimand exclusions, estimator refusal reasons, unexpected failures and admitted count. Do not pool admitted-only results across scenarios or hide sparse exclusions. Record the baseline commit and describe this as numerical-seam admission, not full builder/preflight/ingress parity.

### Posterior and diagnostic outputs

For each admitted raw estimate x and its estimated s, use the planned relative Normal prior Normal(0,0.10). Compute the conjugate posterior analytically and report:

- observed lift bias, standardized error mean and central quantiles;
- coverage of the direct Normal likelihood interval x +/- z_0.975*s;
- descriptive posterior central-95% coverage of fixed truth;
- mean probability r>0 and frequency of posterior probability r>0 reaching 0.95;
- mean and maximum posterior mass below -1;
- numerical and admission failures with their denominators.

For comparison, independently draw x_proxy=r_true+s*z using each admitted run's observed s, the same fixed prior and independent z. Report the paired difference in interval coverage and probability decisions between raw and proxy observations with Monte Carlo standard errors. This holds the empirical SE distribution fixed but breaks its dependence on the observed estimate, and does not reproduce raw admission selection. Therefore it measures a combined working-likelihood/estimated-SE/selection contrast—not an isolated test of one assumption and not a new recommended model.

**No nominal-coverage pass gate for fixed-truth informative posteriors.** Shrinkage alone can change their repeated-sampling coverage, even under an exact likelihood. Raw results are diagnostic; do not declare the solver incorrect solely from informative-posterior coverage, or claim uniform validity from favorable cases. Flag a raw-versus-proxy difference exceeding max(0.03, 4 Monte Carlo SE) for investigation, not as an automatic public refusal or a revised acceptance threshold.

## 3. Reproduction and evidence

The script records protocol SHA256, source versions, family/scenario definitions, random streams, all counts and numeric outputs in JSON. Use stable log tails for probabilities; avoid recovering small tails as 1-CDF. Use absolute paths and single-core numerical libraries; no full repository suites. stdout/stderr are captured in a durable log while the finite visible terminal closes automatically.

Review the script's formulas and the report independently. Any added diagnostic has a separate label and reason; the original protocol/results remain intact. Findings can clarify the existing implementation plan but cannot silently replace owner-approved A, add support, truncate priors or create an automatic historical-prior fitter.
