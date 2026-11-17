# Secondary diagnostic: raw-data prior-predictive calibration

## Why this was added

The original frozen study completed before this protocol was written. Its four working-model calibration cases passed; raw-data comparisons flagged decision-rate differences and one skewed-outcome interval difference. The Normal proxy holds the empirical SE distribution fixed but breaks the estimate/SE dependence and raw admission selection. Those contrasts cannot alone establish posterior calibration or isolate the approximation responsible.

This separately labeled diagnostic evaluates a coherent joint prior/data simulation, retaining the original protocol, results and pass criteria unchanged. It is not a reroll or a replacement acceptance gate.

## Frozen before this diagnostic's outcomes

- SeedSequence(2026111504), one stream per scenario in the original seven-family order, then n=200 and n=2000.
- 2048 attempted experiments per scenario, 14 scenarios. Draw each true relative effect independently from Normal(0,0.10), then generate the original family's raw data conditional on that effect. Reuse the original data-generating formulas and numerical-seam admission, without changing its code.
- For positive-outcome families, a generated effect at or below -1 is physically invalid; binary families additionally require treatment probability below 1. Record every such generation exclusion. The Normal prior's lower invalid mass is Phi(-10), about 7.62e-24; upper binary invalid masses are even smaller for the chosen baselines. The simulation's physical validity check does not truncate or change the analysis prior. Any conditional-calibration claim is qualified by that negligible generating-prior mismatch and the observed exclusion count.
- Analyze each admitted run with the same relative Normal(0,0.10) prior and its independently estimated relative SE. Compute posterior PIT at the generated truth and the forecast probability of a positive effect.
- Report attempted, generating exclusions, estimand exclusions, estimator refusals, admitted and unexpected failures separately. No silent removal of numerical failures.
- With N admitted, compare the empirical PIT CDF distance to sqrt(log(2/(0.01/14))/(2*N)). This is a simultaneous diagnostic reference under a correct continuous posterior, not a promise that the raw likelihood is exact. If N=0, calibration is unavailable, not a pass.
- Report central-95% posterior inclusion and its exact binomial confidence interval, PIT mean/quantiles and ten fixed probability-reliability bins [0,.1),...,[.9,1]. Each bin retains count, mean forecast and actual positive-effect frequency; do not grade sparsely populated bins as if they were precise.
- Overall computational success requires no unexpected failures and finite valid outputs. Statistical discrepancy flags remain scientific findings, not script failures to be tuned away.

## Interpretation limits

These scenarios use fixed known nuisance coefficients and selected raw distributions, not every supported metric or actual historical experiments. Only the Normal relative prior is used in this secondary raw-data study; Student-t and mixture calibration remain established only under the earlier assumed working model. Conditioning on data-based admission does not excuse a wrong posterior, but any diagnostic is for the admitted population and reports its denominator. No new guard, method, prior family or likelihood is introduced from these observations without a separate scope decision.
