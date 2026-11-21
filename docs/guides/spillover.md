# Does making more offers help people who get no offer?

Suppose you can offer training to either a small or a large number of people in each office. You want to know whether the **people not offered training** buy more often when more colleagues receive an offer. This is a spillover: compare the untreated purchase rate under the high-offer policy with the untreated rate under the low-offer policy. A difference of `0.05` means **five percentage points**, not a 5% relative increase. The target gives each person on the eligible roster equal weight, not each office.

## Before assignment: freeze the design

1. Record the eligible people in each office, the binary outcome and its observation window. Each person must belong to exactly one office. If you randomize separately by region, for example, label each office with that region as its `stratum`.
2. Independently in each stratum, draw a fixed number of offices for the high policy; assign the rest low. Then, given those policies, independently draw each office's fixed number of people for offers. At each draw, every eligible subset of that size must be equally likely. Do not reuse the same draw across regions or offices: correct quotas in the logs cannot establish independence. Both policies must leave untreated people, and the low quota must be positive and strictly smaller than the high quota.
3. Choose the smallest **signed absolute change** worth detecting. If available, record one comparable pre-assignment event rate. You do not need individual histories, office-specific rates or a holdout. Save the roster, both allocation rules, outcome, window, target, optional rate and package version to repeat the plan. Never choose these after seeing the trial outcomes.

Here are six offices in one stratum. This example omits the historical rate, so there is no simulation and no assessment of power. It still identifies an effect if the two assignments are randomized as planned.

```python
from increment.spillover import SpilloverDesign, SpilloverGroup, plan_spillover

# (office, number eligible, offers if low, offers if high)
quotas = [
    ("A", 4, 1, 3),
    ("B", 5, 2, 3),
    ("C", 6, 2, 4),
    ("D", 8, 3, 5),
    ("E", 7, 2, 4),
    ("F", 9, 3, 6),
]
design = SpilloverDesign(
    groups=tuple(
        SpilloverGroup(office, tuple(f"{office}{j}" for j in range(size)), low, high)
        for office, size, low, high in quotas
    ),
    high_groups={"all": 3},
    outcome="made a purchase",
    horizon="within 30 days of assignment",
)
plan = plan_spillover(design, meaningful_effect=0.05)
assert plan.evidence.status == "not_calibrated"
print(plan.message)
```

## Decide whether to launch

With a comparable rate of, say, 25%, pass `baseline_rate=0.25` to the same planning call **before assignment**. This is the only historical input; it does not adjust the later estimate. The plan models that rate as events spread across offices, concentrated in the largest offices, or concentrated in the smallest. For each profile it checks no change, offsetting gains and losses with zero average change, and the requested change spread across offices or concentrated in a few. Read `plan.message` for the advice and `plan.evidence.status` to act on it:

The planner rounds the change to the nearest whole number of people. For example, on a roster of 20,000, `meaningful_effect=0.09998` models exactly 2,000 additional events, or **+10 percentage points**. The launch message names that modeled change. Its detection advice is about the six scenarios with that change, not the six with no average change.

| Status | What to do |
|---|---|
| `supported` | The interval checks passed. The lower detection bound reaches 80% in all six scenarios with the modeled change. If those scenarios fit your setting, the design meets that detection target. |
| `underpowered` | The interval checks passed, but even the upper detection bound is below 80% for at least one alternative. Consider adding independently randomized offices before launch. If you proceed, an interval may still be available; not detecting the change will not rule it out. |
| `inconclusive` | The interval checks passed, but the detection bounds do not settle the 80% decision. Do not describe the design as adequately powered. |
| `not_calibrated` | Read `plan.evidence.reason`. No usable power or interval was established. A missing baseline or an infeasible modeled scenario does **not** invalidate a properly randomized comparison. |

For example, at a 1% baseline on 10,000 people, a **+5-point** target is 500 additional events, not five additional events. There are only 100 baseline events to lose, so a zero-average stress scenario can offset at most 100 gains with 100 losses. It does not shrink the positive target to make the check pass. A comparable rate cannot tell you which offices have events or prove how colleagues affect one another.

In the [worked comparison](../reference/validation-evidence.md#ev-spillover-rct2), 1,000 offices of 20 people each could usually detect an increase from 25% to 35%: even the hardest modeled scenario detected it about 94% of the time. The **same offices** detected a rise from 25% to 26% only about 6% of the time in the hardest scenario. With 500 offices, the interval itself missed too often, so the plan would return only the estimate. These are examples, not office-count rules. Enter your roster and the change that matters to your decision; use a negative `meaningful_effect` for a decrease.

## After the window closes: reconcile both logs

Use the **same plan**, the recorded policy of every office, and the recorded offer status of every person (including those with no offer). Use assignment, not attendance. These are example *realized logs*, not instructions to assign the first people on a roster. Each digit in the second log is one person's recorded 0/1 assignment in member-ID order.

```python
policy_log = {"A": 1, "B": 0, "C": 1, "D": 0, "E": 0, "F": 1}
offer_log = {
    "A": "1110",
    "B": "11000",
    "C": "111100",
    "D": "11100000",
    "E": "1100000",
    "F": "111111000",
}
treated = {
    f"{office}{j}": int(bit) for office, bits in offer_log.items() for j, bit in enumerate(bits)
}
# Outcomes as recorded, including explicit zeros (no purchase).
observed = (
    "A3:1 B2:1 B3:0 B4:0 C4:0 C5:0 D3:1 D4:1 D5:0 D6:0 D7:0 E2:1 E3:1 E4:1 E5:1 E6:0 F6:1 F7:0 F8:0"
)
outcomes = {person: int(event) for person, event in (row.split(":") for row in observed.split())}
result = plan.analyze(group_high=policy_log, treated=treated, outcomes=outcomes)
assert round(100 * result.estimate, 1) == -17.8
assert result.lower is None and result.upper is None
print(result.unavailable_reason)
```

The estimate is about **17.8 percentage points lower** under the high policy. It is not simply the difference between two pooled untreated rates: offices differ in size and in the fraction offered training. Increment estimates each office's total from its untreated mean and its full roster size, then compares policies within strata. With unequal offices, the estimate for a particular assignment can fall outside `[-1, 1]`, even though the true risk difference cannot. Neither the estimate nor the approximate endpoints are clipped; they are not individual purchase probabilities.

Supply every untreated outcome; a missing purchase record is not a nonpurchase. Treated outcomes are optional and do not enter this contrast. Missing assignments, foreign IDs or counts inconsistent with the plan are errors, not point estimates.

`result.planning_status` preserves the prelaunch decision. If the plan passed the interval checks **and** the observations support a variance estimate, `result.lower` and `result.upper` give an approximate two-sided 95% interval in proportion units. One office per policy in a stratum, for example, can identify a point but not that variance. Inspect `result.unavailable_reason` when bounds are `None`. Missing bounds are not evidence of no spillover. An interval crossing zero isn't proof of no change either: compare its endpoints with the changes that would matter to you.

## When the interpretation holds

Offices must not influence each other, and outcome measurement must be comparable. The point comparison averages over which peers received offers. **For the interval, the number of treated peers may matter, but their identities must not.** Calling the difference peer-mediated also requires that announcing an office's policy has no separate effect. Neither the logs nor historical rates verify these assumptions. A passing plan describes the modeled outcomes, not every future trial.

The design accounts for office-level randomization; adding a cluster label to a person-randomized experiment is not equivalent. This standalone route does not take ratio metrics, CUPED adjustment, winsorization, breakouts, repeated looks or multiplicity options. It does not estimate the effect on people offered training. See the [API](../api.md#two-stage-spillover) and [validation record](../reference/validation-evidence.md#ev-spillover-rct2) for the published comparison and the planning bounds.
