# Published intervals for untreated spillover

These are independent benchmarks on synthetic binary outcomes, not methods selected from a live trial. All signs below are higher-quota policy minus reference (the reverse of `interferenceCI`'s `IE`), in probability units. Groups do not interfere with one another. Within each stratum, fixed counts of groups receive each policy uniformly; within each group, its quota of offers is drawn uniformly without replacement. Both compared policies leave at least one person unoffered in every group.

From the repository root, with the project Python environment active:

```sh
python tests/oracles/generate/gen_spillover_inversion.py
python tests/oracles/generate/gen_spillover_null.py
Rscript --vanilla tests/oracles/generate/gen_spillover_published.R
```

Only the second command needs NumPy (reference: 2.5.4). The optional R command uses `shasum` to verify two commit- and SHA-256-pinned upstream files in a temporary directory, then executes them. No package install or network call in ordinary tests. Reference: R 4.4.2, RNG `Mersenne-Twister`, `Inversion`, `Rejection`.

## Two different targets and two different inversions

For group size $n_i$, policy $s$, offer quota $m_{is}$, and first-stage probability $q_{is}$, let $T_{is}$ be the number of people who would have an event if unoffered at that quota. Rigdon–Hudgens assume **count-only interference**: identities of offered neighbors do not change a person's outcome at a given quota. This is distinct from blocking the first-stage randomization. With observed unoffered events $e_i$, the observed quota allows $e_i\le T_{is}\le e_i+m_{is}$; an unobserved quota allows any count from 0 to $n_i$. Equal quotas share a count, even when policies differ. All such margins can be realized by individual schedules, and no draw uses two margins from the same group simultaneously.

For a pair $b-a$, the target is $\tau=\sum_i w_i(T_{ib}-T_{ia})/n_i$: $w_i=1/k$ for the papers' average of $k$ group means, $w_i=n_i/\sum_j n_j$ for the person average. The estimator replaces $T_{is}/n_i$ with $1\{S_i=s\}e_i/[q_{is}(n_i-m_{is})]$. For each sharp schedule, $p=P(|\hat\tau^*-\tau|\ge|\hat\tau_{obs}-\tau|)$ averages over full policy assignments and, conditionally, independent hypergeometric unoffered counts. Do not flatten the two stages into equiprobable rows. [Rigdon & Hudgens §4.1](https://pmc.ncbi.nlm.nih.gov/articles/PMC4504023/) retains a value if **any** compatible schedule has $p\ge\alpha$. The tiny script searches from each extreme inwards with exact rational masses. The result is the hull, not necessarily a set of accepted interior values. Person weights, blocking and a third policy are our extensions.

[Tchetgen Tchetgen & VanderWeele, Theorem 3](https://pmc.ncbi.nlm.nih.gov/articles/PMC4216807/), does not require count-only interference. For its two-policy, unblocked **group** target, the squared radius is $[\max_s q_s^{-2}+\sum_i L_i^2/k]\log(2/\alpha)/(2k)$, where $L_i=\max_s\{(1-1/\binom{n_i}{m_{is}})/q_s\}$. The [R `TV` code](https://github.com/cran/interferenceCI/blob/9352a8ec903a01f0abf4eb83f7e689e5ff1922dd/R/interferenceCI.R#L272-L281) divides by $q_s^2$ inside $L_i$ instead. Both intervals below are clipped to $[-1,1]$; neither off-the-shelf formula covers the unequal-size person target by relabeling.

## Reference output

The tiny two-policy tables have quotas `(0,1)`, assignments `0,0,1,1` and unoffered events `0,1,1,2`. Sizes are `(3,3,3,3)` or `(2,3,2,4)`. The blocked table has two strata each of sizes `(2,3,3)`, quota vectors `(0,1,1), (0,1,2), (0,1,2)` in each, assignments `0,1,2 | 2,0,1`, and events `0,1,1 | 1,2,0`. Each stratum randomizes **all three** policies. Its two prespecified *contrast tests* use Bonferroni $.05/2$; the other rows use $.05$. This allocation belongs to this inversion benchmark, not to the production policy-mean interval.

| Exact tiny inversion | Point | Hull |
|---|---:|---:|
| Equal, group = person, 1−0 | 7/12 | [−1/4, 5/6] |
| Unequal, group, 1−0 | 2/3 | [−13/48, 41/48] |
| Unequal, person, 1−0 | 2/3 | [−3/11, 9/11] |
| Blocked, person, 1−0 | −3/32 | [−11/16, 5/8] |
| Blocked, person, 2−0 | 9/16 | [−1/2, 7/8] |

| Published group comparison | Theorem 3 | R `TV` | R sampled `exactCI`, seeds 519 / 520 |
|---|---:|---:|---:|
| Equal | [−1, 1] | [−1, 1] | [−.333333, .833333] / [−.208333, .833333] |
| Unequal | [−.926854, 1] | [−1, 1] | [−.083333, .854167] / [−.119792, .854167] |
| Package's 10×10 example | [−1, .897548] | [−1, 1] | [−.543333, −.057500] / [−.545000, −.046000] |
| 240×59 | [−.220506, .271217] | [−.361388, .412099] | Not run: first-stage matrix is infeasible |

The [package search](https://github.com/cran/interferenceCI/blob/9352a8ec903a01f0abf4eb83f7e689e5ff1922dd/R/interferenceCI.R#L598-L653) samples compatible schedules and randomizations (`B2=C2=100`), then interpolates. The upstream `min`/`max` warnings on tiny tables occur when no sampled rejection lies beyond a retained extreme. Seed 520 misses the exact equal-size value −1/4. These draws are not coverage estimates.

## 240×59: a known truth, not a computed exact interval

Four groups have 59 event-positive people each; the other 236 have 14 each. No person's outcome changes with assignment. Half the groups make one offer and half none. In the displayed draw all four exceptional groups make an offer, and 28 ordinary offers hit event-positive people. The no-offer groups each have 14/59 events; the one-offer rates are 58/58 (4 groups), 13/58 (28), 14/58 (88). The estimate is +.025355543; the old Wald interval is [.000720029, .049991056].

$X$, the number of exceptional groups making an offer, is hypergeometric with population 240, four successes and 120 draws. Given $X=x$, ordinary offers hit $Y\sim\mathrm{Binomial}(120-x,14/59)$ event-positive people. These counts determine the point and the Wald variance estimate $(s_0^2+s_1^2)/120$, with within-arm sample variances of group rates. **Across assignments**, the nominal 95% Wald interval falsely excludes zero with probability .107438779537 (exact enumeration). **On the displayed assignment**, the no-effect schedule has exact two-sided $p=.071107526717$: zero belongs to the full inversion set.

The R search constructs a matrix of $\binom{240}{120}$ first stages; 240 double columns would require about $10^{74.24}$ bytes. There are $2^{2(240)(59)-120(59+58)}$ compatible labeled untreated schedules and $\binom{240}{120}59^{120}$ assignments. Instead the Python script runs the §4.2 search on **per-group counts**: binomial draws impute missing outcomes; a uniformly offered person has an event with probability $T_{i1}/59$. Integer scores decide ties. PCG64 gives [.009435028, .041666667] (519) and [.007768362, .043008475] (520). Both exclude the compatible truth. These are not upstream R draws or certified endpoints. General inversion uses per-group outcome counts, not the stratum/policy moments returned by the warehouse.
