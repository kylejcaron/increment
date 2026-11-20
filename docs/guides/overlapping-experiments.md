# Analyze two overlapping experiments

Use `increment.overlap` when two experiments independently assign the same kind of unit (for example, an account) to off or on, and you want to decide whether to ship both or add one after the other ships. An interaction measures how much one effect changes when the other treatment is on; it does not answer either launch question by itself.

## First establish who could have received either treatment

Start with the original assignment configuration files, not dictionaries re-exported from them. Use `sdk_overlap_plan` with the archived configuration, the two target keys and focal experiments, the off/on arm labels, and your registered population, dates, metrics, and launch criteria. The archive establishes the allocator rules, revisions, and conditional arm probabilities. You supply the business eligibility rules, proof they were assessed before either assignment, and the serving and measurement dates. A hash checks that the captured inputs match a pinned receipt; it does not prove that the inputs or the described launch policy are true.

Both experiments must cover one shared enrollment period. Base the decision on the intersection: units that met both eligibility rules and were randomized by both experiments. If each eligible account was certain to enter both, this is the jointly eligible audience. With additional enrollment gates, it is the jointly randomized audience, not everyone who could have qualified. A recorded gate decline is not randomized control. A missing decision is not control either. Two experiments competing for one mutually exclusive layer cannot provide this design.

Prepare four inputs for a common fixed measurement window:

| Input | What to retain |
|---|---|
| Roster | Stable text unit IDs, original entry time, both eligibility answers and when each was fixed. Keep entrants who never saw either treatment. |
| Decisions | The actual first decision for each eligible target, including randomized off assignments and explicit gate declines. Convert persisted SDK decisions with `sdk_overlap_decision`. Do not regenerate absent records. |
| Deliveries | Actual treatment deliveries with event IDs and timestamps, including delivery that did not follow the assigned arm. Focal deliveries may start at the common start, but must follow both eligibility determinations. Events at or after the window ends do not change its delivery summaries. |
| Outcomes | One finalized outcome per registered metric and assigned unit for the same window. Record a measured zero as zero, not as a missing value. |

Use `OverlapFeed` and `OverlapCapture` with source versions and coverage through the required dates. Supply materialized dataframes from the same captured generation to `capture_overlap_frame`; the dataframe itself cannot certify its source version. For warehouse inputs, use [verified local DuckDB tables](../reference/overlap-warehouse.md) with `capture_overlap_warehouse`. It checks the four versions against the actual relations and returns aggregate evidence, not unit rows.

An external randomizer can supply an `OverlapLaw` and `OverlapFactor` with a documented independent assignment mechanism and an equivalent decision log. A pair of exposure tables or single-experiment summaries cannot supply that evidence.

## Read the decision that matches the launch

Run `analyze_overlap` on the captured evidence. Compare effects in the outcome's original units per originally assigned account. For a binary metric, an estimate of `0.01` means one percentage point. The labels `00`, `10`, `01`, and `11` mean neither treatment, A only, B only, and both.

| Question | Row | Comparison |
|---|---|---|
| Ship both rather than neither? | `joint_launch` | `11 − 00` |
| Add B when A is already on? | `b_given_a_on` | `11 − 10` |
| Does B's effect change when A is on? | `interaction` | `(11 − 10) − (01 − 00)` |

The readout also includes A's corresponding comparisons and the marginal effects under the other experiment's **planned** allocation, not the allocation observed in this run. A positive joint effect does not establish that adding B helps. A small or uncertain interaction does not establish that either launch clears its criteria.

Run the synthetic checkout-and-promotion example from the repository root:

```sh
uv run python examples/overlap.py tests/fixtures/overlap/sdk_config.json
```

It uses the archived allocator to assign a fixed cohort. The constructed outcomes improve by about two orders per account if both treatments launch, but fall by about two if B is added after A. A second, three-account rehearsal cannot support a joint decision. These are illustrations, not estimates from customer data.

Register the metric–comparison pairs that can drive decisions before looking at outcomes. The family is fixed: if a cell is absent, the affected rows report why and still count in the adjustment. The reported intervals use unequal-variance Welch approximations with a Bonferroni adjustment across the registered family. They are not exact finite-sample guarantees. The approximation does not become valid because multiplicity was accounted for. Use a benefit interval entirely above zero for a higher-is-better claim. For a lower-is-better guardrail with a maximum tolerated increase, use the upper bound: it must be strictly below the tolerance. A bound exactly on a threshold is inconclusive.

`criteria_met` for a joint launch requires every registered primary benefit and every registered guardrail to clear. A secondary result cannot make up for a failed or missing criterion. If any required cell has fewer than 30 assigned units (or a binary cell has fewer than 10 successes or 10 failures), the interval may be shown, but the claim is not usable for a decision. A missing, pending, or empty required cell does not produce a point estimate; a singleton or sample-constant cell does not produce an interval. No outcomes are imputed to empty cells. If deliveries disagree with the randomized assignments, the estimates still describe the actual implementation as assigned; a clean launch is not recommended on that basis.

## Rehearse a design before starting

Use `simulate_overlap` with one representative dataframe of joint metric vectors per cell and the planned number of entrants expected to be randomized into both experiments. Estimate that count from both eligibility and enrollment together. The simulation varies cell sizes using the registered allocation and resamples metrics together so a success and its associated cost stay paired. Give each cell a prespecified outcome distribution, even if that cell is expected to be rare. The estimates assume complete mature measurements and delivery as designed. They do not forecast attrition or prove that those distributions describe your traffic.

Report decisions across **all** simulated trials. A draw with an empty cell or insufficient support is not a success, and remains in the denominator. The Monte Carlo interval describes uncertainty from running a finite number of simulations, not uncertainty about the effect in the actual experiment. Compare the chance of clearing all launch criteria separately from the chance of detecting an interaction.

## Check whether your design fits

- **Supported:** Two concurrent binary experiments, independent assignment with constant (possibly unequal) weights, independent units, absolute unadjusted means or binary means, a fixed family of claims and one final look. Use frames, verified warehouse aggregates, or checked evidence replay for the same plan. Archive replay uses a separately pinned SHA-256. Portable exact counts and allocation weights are limited to `2**53 − 1`.
- **Inputs cannot establish the result:** Separate experiment aggregates, exposure-only logs, and feeds without pretreatment eligibility, unit identity, assignment history or coherent source versions.
- **Not covered by this method:** Relative lift and ratio-of-means, adjustment with covariates, clusters or blocks, winsorization, arbitrary subgroup breakouts, more than two factors, multiple treatment arms or dependent assignment. The standard `run` builders do not express this design.
- **Do not stop early on these intervals:** Repeatedly applying a fixed-horizon test does not preserve its error limit. Finish the registered outcome window and analyze once.
