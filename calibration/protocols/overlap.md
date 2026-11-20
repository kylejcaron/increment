# Overlap qualification, version 1

This protocol tests a fixed-horizon family of approximate Welch bands and their decision rules, not assignment integrity or field validity. Execute once on the integrated workflow:

```sh
uv run --no-sync python -m calibration.overlap --protocol calibration/protocols/overlap.md --out overlap-qualification.json
```

The run is fixed at 20,000 attempts per case, model and algorithm. A failed case or unexpected fault counts against qualification; do not retry a trial, change a threshold, change a seed, or substitute a draw. The output must be a new file and contain the exact protocol SHA-256, software versions, seeds, attempted counts, failures and the overall decision.

## Design and outcomes

Use two independent Bernoulli assignments with probabilities `a` and `b`, in cell order `00,10,01,11`. Cross each row of this table with the global null, then the mixed null, before going to the next row.

| Regime | Target N | a | b | Outcome | Gate class |
|---|---:|---:|---:|---|---|
| Dense normal | 512 | .5 | .5 |  Normal | Dense |
| Unequal normal | 4096 | .25 | .75 | Normal | Dense |
| Dense skew | 4096 | .5 | .5 | Centered lognormal | Dense |
| Dense binary | 4096 | .5 | .5 | Binary, base .20 | Dense |
| Sparse joint | 48 | .1 | .2 | Normal | Sparse |
| Rare binary | 512 | .25 | .75 | Binary, base .005 | Sparse |

Use 24 claims: all eight registered contrasts of `value` (primary, higher benefit except two-sided interaction), `cost` (lower-is-better guardrail, absolute margin .25) and `activity` (secondary, two-sided difference). In claim order use A marginal, B marginal, A at B-off, A at B-on, B at A-off, B at A-on, both versus neither, and full interaction. Family alpha is .05. All trials have complete mature measurement and no delivery conflict. A fitted row involving a cell with fewer than 30 units, or a binary cell with fewer than 10 successes or failures, can issue an approximate band but cannot decide. A guardrail clears only in the favorable direction; its adverse-side failure is not a clearance. Compound joint clearance requires the joint primary and guardrail to clear.

Per unit draw independent standard normals Z and U and set `V = -.5 Z + sqrt(.75) U`. With cell scales `(1,1.5,1,2)`, continuous outcomes are `value = m[cell] + scale[cell] * noise(Z)`, `cost = k[cell] + scale[cell] * noise(V)`, and `activity = m[cell] + noise(Z)`. Noise is the identity for normal or `exp(z-.5)-1` for centered lognormal. Global null uses zero vectors m and k. Mixed null uses `m=(0,.3,.2,.8)` and `k=(0,0,0,.25)`. For the two binary regimes, value and activity are both `1{Phi(Z) < r[cell]}` with base risk as in the table and mixed-null increments `(0,.02,.01,.04)`; cost stays normal with k as above. The true means and role/direction determine false-claim masks before drawing trials. In both states, at least one of the necessary joint claims is null or on its boundary.

The runtime parametric algorithm draws the multinomial assignment counts, generates the prescribed outcome law independently within each cell, and sends complete typed cell statistics to the same decision kernel used by readout. The independent reference generates individual A and B draws, then individual outcomes, and uses a saturated group regression with separate residual variances. The two methods do not share random outputs. For the matched empirical comparison, take a separate fixed sample of 256 paired vectors per cell from the stated law. Both algorithms target this exact empirical distribution, not the parametric law: production uses conditional multinomial frequencies within assignment cells and the independent reference samples row indices with replacement. Every comparison preserves pairing across metrics.

## Random streams

Use PCG64 with `SeedSequence(2026111983).spawn(12)` in table order and state order (global, mixed). At each case spawn five children, in order: runtime parametric, reference parametric, empirical distribution, runtime empirical, reference empirical. Use the children directly except for the public runtime empirical draw seed: use the low 53 bits of the child's first generated `uint64` word as the portable integer seed. Generate the empirical law exactly once per case, in canonical cell order, and reuse its vectors only for the matched comparison. Reference batches contain up to 64 trials; generate A and B together as `(trial, entrant, A-or-B)` uniforms and draw independent outcome normals (parametric) or empirical indices (matched).

## Reporting and gates

For each algorithm and model retain per-claim truth, bias, empirical SD, mean fitted SE, point/fit/decision-ready counts, favorable decisions and reason counts. Bias and empirical SD use the point denominator; mean fitted SE and fitted-only 5/50/95% SE quantiles use the fit denominator. Give every such denominator explicitly. Report assignment opportunity (all four counts at least two), FWER (one or more false claim approvals), false compound clearance, compound clearance and per-metric interaction decision rates. Coverage of the whole family has three mutually exclusive counts: all bands issued and cover, all bands issued and at least one miss, and not all issued. Small-support bands count as issued; abstention never counts as coverage. Unconditional rates always divide by all attempts.

For each parametric algorithm and case require one-sided exact Clopper–Pearson upper bounds at most .06 for FWER and false compound clearance. For each dense case additionally require one-sided exact lower bounds of at least .99 on complete issuance and .94 on simultaneous issued-and-cover. For the sparse cases, the exact multinomial probability that all four cells have at least two units must lie within the two-sided exact interval for observed opportunity, and no decision may approve an unsupported claim. Sparse regimes do not promise dense availability. There are 88 statistical exact-binomial gates: for each of the two parametric algorithms, 24 false-decision gates, 16 dense issuance/coverage gates and 4 sparse occupancy gates. Use `.01/88` per gate (split in two for the sparse two-sided interval). Use upper-tail inversion for upper bounds; the zero and all-success limits are exact.

For each paired empirical case compare primary joint, primary interaction and compound joint decision rates to the independent matched reference. For each require absolute difference at most `.01 + 4*sqrt(se1**2 + se2**2)` with `se = sqrt(rate*(1-rate)/20000)`. Report rates and tolerances even if a gate fails. Check that unsupported claims receive no favorable decisions in either model or algorithm. These are prospective qualification tolerances, not a proof of universal Welch coverage, a guarantee of enrollment, or an estimate from observed overlap results.
