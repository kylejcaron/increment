# Analyze two overlapping experiments

If two experiments ran on the same accounts, “Should we launch both?” and “Should we add B when A is already on?” can have different answers. Use `increment.overlap` when each experiment independently assigns accounts to off or on and you will analyze the results once, after a fixed measurement window.

## See why the launch decisions differ

From the repository root, run the synthetic checkout-and-promotion example:

```sh
uv run python examples/overlap.py tests/fixtures/overlap/sdk_config.json
```

Launching both clears the preset criteria, with an estimated gain of **2 orders per account** versus launching neither. Adding B after A is already on estimates a **loss of 2 orders per account**, but that result is inconclusive. A second example with three accounts reports `insufficient_evidence`. Neither example uses customer data.

The result labels `00`, `10`, `01`, and `11` mean neither treatment, A only, B only, and both.

| Question | Result row | Comparison |
|---|---|---|
| Launch both rather than neither? | `joint_launch` | `11 − 00` |
| Add B when A is already on? | `b_given_a_on` | `11 − 10` |
| Does B's effect change when A is on? | `interaction` | `(11 − 10) − (01 − 00)` |

The interaction describes how B's effect changes with A. It does not say whether either launch meets its criteria.

## Check which units belong in the analysis

The experiments must share one enrollment period. Analyze only units that met both eligibility rules **before either assignment** and were randomized in both experiments. That may be fewer units than were eligible for both: a gate decline or missing decision is not a control assignment. Experiments in the same mutually exclusive assignment layer cannot enroll a unit in both and do not qualify.

This method compares absolute means or binary rates on independent units. Assignment probabilities can differ between experiments or arms, but must stay fixed. Do not use it for more than two experiments, multi-arm or dependent assignments, clusters or blocks, relative lift or ratios, covariate adjustment, winsorization, or subgroup comparisons. Do not use these intervals to decide when to stop early.

## Prepare the plan and inputs

Read the original assignment configuration files as bytes. Pass those bytes, both target keys and experiment IDs, and their off/on labels to `sdk_overlap_plan`. Do not use dictionaries re-exported from the files. The archive supplies assignment rules, versions, and probabilities. You supply the eligibility rules, proof of when you applied them, and the registered dates, metrics, and launch criteria. If you use another randomizer, supply an `OverlapLaw`, `OverlapFactor` definitions, and records that establish independent assignment.

| Input | What to keep |
|---|---|
| Roster | Stable text IDs, entry times, both eligibility answers and when each was fixed. Include units that received neither treatment. |
| Assignments | The actual first decision for each eligible target, including off assignments and gate declines. Use `sdk_overlap_decision` for saved SDK decisions; do not reconstruct missing rows. |
| Deliveries | Event IDs and times, even if delivery disagreed with assignment. Delivery can start at the shared start time, but must follow both eligibility checks. Only events inside the measurement window count toward delivery summaries. |
| Outcomes | One finalized value per assigned unit and registered metric for the same window. Record a measured zero as zero, not as missing. |

Record the source versions and dates covered in `OverlapFeed` and `OverlapCapture`. For dataframes, pass materialized inputs from the same snapshot to `capture_overlap_frame`; a dataframe cannot verify where it came from. For DuckDB, follow the [warehouse capture steps](../reference/overlap-warehouse.md). You can also [reload saved evidence for the same plan](../reference/saved-formats.md#overlap-evidence-and-results). Exposure logs or separate experiment summaries alone are not enough.

## Read the result

Pass the captured evidence to `analyze_overlap`, not the single-experiment `run` builders. Effects use the metric's original units per assigned unit: for a binary metric, `0.01` is one percentage point. The result also shows A's corresponding comparisons and each experiment's average effect under the other's **planned** assignment probabilities, not the observed split.

Register the metric and comparison pairs that can drive decisions before looking at outcomes. The Welch intervals are approximate, even after adjusting for all registered claims (Bonferroni). An unavailable result still counts in that adjustment.

- `criteria_met`: Every registered primary benefit and guardrail **for launching both** passed.
- `not_met`: The required results were available, but at least one did not pass. This does not necessarily mean harm.
- `insufficient_evidence`: A required result was unavailable or had too little data, or delivery disagreed with assignment. If delivery disagreed, estimates still describe what happened under the original assignments, not what a rollout as planned would do.
- `not_requested`: No primary claim was registered for launching both.

A benefit passes when the whole interval favors treatment. For a guardrail where lower is better, the upper bound must be below the allowed increase. Where higher is better, the lower bound must be above the negative of the allowed decrease. A bound exactly on the threshold is inconclusive.

Secondary results cannot make up for a failed or unavailable criterion. A required group needs at least 30 assigned units; for a binary outcome it also needs at least 10 successes and 10 failures. An interval may be shown below those limits, but cannot support a decision. Missing, pending, or empty groups yield no estimate for comparisons that need them. A group with one outcome or identical outcomes yields no interval for those comparisons.

## Plan for detecting an interaction before launch

Use `simulate_overlap` before starting to ask: if B's effect changes by an amount that matters when A is on, how often will this design detect it? Supply the expected number of accounts **randomized in both experiments**, not the sum of two enrollment forecasts or the number merely eligible for both.

From the repository root, compare two forecasts for the same checkout-and-promotion pair:

```sh
uv run python examples/overlap_power.py
```

The example registers one two-sided interaction claim for orders per account. The four dataframes describe outcomes for neither treatment, A only, B only, and both. Each is a hypothetical distribution with four equally likely account outcomes and the same spread. Their means are 10, 11, 13, and 12, respectively: B adds 3 orders without A but only 1 with A, an interaction of −2 orders per account. Each simulation draws new group sizes using the planned assignment probabilities and samples account outcomes with replacement within each group. With multiple metrics, keep each account's metrics together in one row. Use samples that reflect the registered measurement window and metric definitions; the simulation assumes complete outcomes and delivery as planned.

A and B independently treat 20% and 10% of jointly enrolled accounts, so only 2% are expected to get both. At **1,200** accounts, expect 24 in that group, below the 30-account minimum for a decision. Only 264 of 2,000 runs were ready to test the interaction, and 136 detected it. The estimated detection chance is **6.8%** (136 out of **all 2,000** runs, not just the 264 ready ones). At **4,000** accounts, expect 80 in the both-on group. All 2,000 runs were ready, and **91.3%** detected the same interaction.

If you need to know whether B works differently with A before choosing a rollout sequence, plan for more joint enrollment before starting or don't rely on this design to settle the choice. A run that doesn't detect an interaction doesn't show that B works the same with and without A.

The 95% Monte Carlo intervals describe uncertainty from the finite number of simulated runs, not whether the assumed outcomes or effect reflect real traffic. Power here means the chance of detecting the assumed nonzero interaction. If you instead assume exactly zero interaction, detections estimate the false-detection rate. This example does not request joint launch criteria; if you register them, check the chance of meeting them separately from the chance of detecting an interaction.

This is planning for one specified pair, not for discovery across a larger set of pairs. Correcting for multiple comparisons limits false detections; it cannot make up for small groups. The pair simulation does not estimate power for a discovery scan.
