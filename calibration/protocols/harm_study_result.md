# Fraction harmed: defer implementation

The exact checks and all 88 prespecified synthetic calibration counters passed. The independent field evaluation could not start: its input manifest was absent. There is no authorized, representative field readout with an eligible decision owner, so **defer**. This is neither evidence that the method changes a real action nor evidence that it cannot. No method, owner tolerance, or economic gain is recommended or inferred.

## What was tested

For a fixed-window binary outcome where 1 is beneficial, the target is the population fraction that would have outcome 1 under control and 0 under treatment. Randomization identifies the two marginal outcome rates, not the joint outcomes on each person. Sharp bounds follow from the possible couplings of the marginals within each baseline group, mixed using the population's group proportions. A favorable mean effect does not imply a small harmed fraction: the observational twins have the same observed law but harmed fractions 0 and .50. Neither these bounds nor a treatment-effect model identify which people would lose the outcome.

The [frozen protocol](harm_study_v1.md) compares full-data unstratified (`U0`), fixed one-feature (`F`), and fixed two-feature (`Q`) bounds with honest split-sample adjustment (`H`) and a deliberately misspecified control (`M`). The operational pipelines `P0` to `P3` intersect, in order, the outer bands from `U0`, `F`, `Q`, and `H`; `M` does not enter a pipeline. On the simultaneous coverage event the intersections cover the full two-feature identified interval. Clear only when the entire outer band is below the tolerance, rule out only when entirely above, and otherwise abstain, including at equality or if the bands disagree.

The numerical oracle passed with zero mismatches: 165 potential-outcome tables, 81 marginal pairs, 1,225 ordered pairs of stratified tables, five simplex cases, and 147 exact sampling cells. The largest absolute difference in the audited binomial interval endpoints was `3.33e-15`.

## Synthetic calibration

The completed run crossed seven frozen laws, two sizes (`160`, `4096`), and two assignment probabilities (`.5`, `.1`). All 560,000 attempts were evaluated: 20,000 per cell × 28 cells. Both mass modes and all five methods returned 5,600,000 valid sampling bands. There were also 291,684 null plug-in estimates due to empty arm-leaves; these are not missing or invalid sampling bands. Each error count is a union across the comparisons specified for that trial and cell, not a pooled average or a count of distinct errors across decision rows.

| Error event | Largest count among the 28 cells | Simultaneous upper limit in that cell |
| --- | ---: | ---: |
| Joint endpoint noncoverage or disjoint pipeline | 68 / 20,000 | .00528401542348 |
| Any false clearance | 68 / 20,000 | .00528401542348 |
| Any false rule-out | 3 / 20,000 | .000829573082806 |

All 84 error upper limits were at or below the frozen .05 benchmark. All four positive-control lower limits exceeded the frozen .80 power benchmark:

| Positive control (estimated mass, `n=4096`, `e=.5`) | Successes / 20,000 | Simultaneous lower limit |
| --- | ---: | ---: |
| D2: `P0` rules out at .50 | 20,000 | .999511336726 |
| D3: `P0` clears at .50 | 20,000 | .999511336726 |
| D4: `P1` clears at .32 while `P0` abstains | 20,000 | .999511336726 |
| D5: `P1∩H` clears at .32 while `P1` abstains | 19,887 | .992013825285 |

The confidence level for the Monte Carlo limits is at least .99 over the 88 counters. The D5 increment is against the fixed-S pipeline `P1`, not against full-data `Q`: at that coordinate and threshold, `P2` (which includes `Q`) cleared in 20,000/20,000 trials and `P1∩H` in 19,887/20,000. The learned partition has not demonstrated an advantage over full `Q` in a field decision. None of these thresholds or synthetic losses establishes an owner's tolerance or costs.

## Independent field evidence

The prespecified historical benchmark compares assignment to the men's email against no email, with purchase within two weeks as the beneficial outcome. It could not start: `missing_case_manifest`. Analysis/reporting permission and design were not verified. The aggregate packet has null bounds, null raw input and card digests, no threshold rows, and no owner evaluation or post-readout acknowledgment. These are unavailable results, not zeros. There is no estimate of field harm, field calibration, sample adequacy, or field utility. Even a permitted historical benchmark would not substitute for a trial matched to an authorized current decision.

To reconsider, obtain independently verified permission to analyze and report aggregates, document independent units, assignment, complete follow-up and the target population, and stage a representative trial with a prospectively signed card for the actual decision. Before results are seen, name the action alternatives, tolerance, risk ceiling, cost bearers, loss scale, costs, and minimum gain. After the readout, obtain independent confirmation bound to the exact decision and card digests for the simplest feasible changed action with strictly sufficient worst-case gain. More data can reduce sampling uncertainty; it cannot on its own shrink the population identification interval at a fixed feature resolution.

## Limits and receipts

The projected exact-binomial bands jointly cover their respective population endpoints with probability at least .95 in real arithmetic under the frozen assumptions: independent sampling from the same population, constant independent two-arm assignment, a complete binary outcome at a fixed window, consistency, and no interference. The guarantee is not point identification, a posterior, exact .95 coverage, or transport beyond the sampled population. The study does not extend to CUPED residuals, ratios, winsorization, covariate-dependent assignment, clustering, sequential looks, declared subgroups, or general multiplicity. Calibration supports the frozen construction and scenarios, not a public interface or deployment decision.

| Receipt | SHA-256 |
| --- | --- |
| Frozen protocol | `516a7ac0697769f2b477e22f27b257fd75405eeb8a9515f1ed2f0f5ed7e585ac` |
| Exact oracle bound by the campaign | `e42ad036ca7cf92cf37b60fd89248e53fe96693128c7639f20409a4732131c5d` |
| Full campaign manifest | `6d3cf8fe16cfb0a087b61d8f83570eec623c6b7e13e521e160d2b6a29a205f4f` |
| Field decision v2 | `6a92e69b7960c9026d7c718ca898ac3b56036735faf4a34663947958abf4cdd0` |

The campaign source revision was `132d665c765806c89afd400d3857e7cd2f13e84a`; receipts bind its exact source files, lockfile and aggregate artifacts. The locked runtime was Python 3.15.2, NumPy 2.5.4, SciPy 1.18.1 and Pydantic 2.14.2. The join reads only the aggregate evidence and writes a fresh recommendation; exit 2 means a valid defer, not input corruption. No trial data were accessed for this report.
