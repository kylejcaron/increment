# Does making more offers help people who get no offer?

If more people in an office get a training offer, do their colleagues buy more often **even when they get no offer themselves**? To measure that spillover, compare purchases by people who got no offer in offices that made more offers versus offices that made fewer. This does not measure the effect on people offered training.

## Plan before you assign anyone

Use this design only if one office’s offers cannot affect another office’s purchases. Measure purchases the same way everywhere. To report a reliable range for the effect, you also need to assume that **how many** colleagues get offers can matter, but **which** colleagues get them does not.

1. List the eligible people in each office. Each person belongs to one office. Choose a yes/no outcome and when to measure it, such as “made a purchase within 30 days.”
2. Set the number of offers each office would make under each policy. Both counts must be positive, the high count must exceed the low count, and both must leave some people without offers. Decide how many offices will use each policy. If you draw offices separately by region, use the region as the `stratum`.
3. Choose the smallest change that would affect your decision. For a rise from 25% to 30%, use `meaningful_effect=0.05`: **five percentage points**, not a 5% relative increase. Use a negative value for a decrease.

Here’s a plan for six offices. If you have a comparable historical purchase rate, pass it to `plan_spillover` as `baseline_rate` (for example, `0.25`). You don’t need individual purchase histories. This example leaves it out.

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

In a [simulation with 1,000 offices](../reference/validation-evidence.md#ev-spillover-rct2), a rise from **25% to 35%** was detected about **94%** of the time, even in the hardest scenario. For a rise from **25% to 26%**, the same offices detected it only **6%** of the time in the hardest scenario. Check your own roster rather than using 1,000 as a cutoff.

Read `plan.message` before you launch. The planner checks two things in simulations: does the reported range usually include the true effect, and does the experiment detect a change as large as the one you care about? `plan.evidence.status` tells you what to do:

- **`supported`** — If the scenarios fit your setting, the plan meets the goal: at least an 80% chance of detecting the modeled change in each scenario where it occurs.
- **`underpowered`** — The range checks passed, but at least one scenario fell short of 80% detection. Consider adding more offices. If you go ahead, not detecting a change won’t rule it out.
- **`inconclusive`** — The range checks passed, but the simulations couldn’t settle the 80% question. Don’t rely on this plan to detect the change.
- **`not_calibrated`** — Check `plan.evidence.reason`. You can still estimate the effect after a valid experiment, but this plan hasn’t established a reliable range. That’s the outcome for the six offices above: no historical rate was supplied.

## If you launch, make two random draws

Save the plan inputs and package version so you can reproduce the check. Then:

1. Draw the planned number of high-offer offices; the rest use the low policy. If you grouped offices by region, make a separate, independent draw in each region. Record the policy for every office.
2. In each office, draw the people who get offers, using that office’s assigned count. Make this draw independently of the other offices. Record whether **each** eligible person got an offer, even if they didn’t attend training.

At both steps, every possible selection of the required size must be equally likely. Matching the planned counts is not enough if you reused the same draw across regions or offices.

## After the window closes: reconcile both logs

Use the same plan and both assignment logs. Supply a purchase outcome for **everyone who got no offer**: `1` for a purchase, `0` for a confirmed nonpurchase. A missing record is not a zero. Outcomes for people who got offers are optional.

Here are the recorded assignments for the six offices. In `policy_log`, `1` means the office made more offers. In `offer_log`, each digit records one person’s offer in roster order. These are example records, not a rule for choosing people.

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
purchased = "A3 B2 D3 D4 E2 E3 E4 E5 F6".split()
did_not_purchase = "B3 B4 C4 C5 D5 D6 D7 E6 F7 F8".split()
outcomes = {person: 1 for person in purchased} | {person: 0 for person in did_not_purchase}
result = plan.analyze(group_high=policy_log, treated=treated, outcomes=outcomes)
assert round(100 * result.estimate, 1) == -17.8
assert result.lower is None and result.upper is None
print(result.unavailable_reason)
```

We estimate that people who got no offer were **17.8 percentage points less likely to buy** in offices that made more offers. We didn’t supply a historical purchase rate, so the plan gives no range around that estimate (`missing_baseline_rate`). The estimate alone doesn’t show that making more offers hurt purchases.

When the plan and the observed outcomes both support it, `result.lower` and `result.upper` give an approximate 95% range for the effect. If it includes zero, look at both ends: the data may still be consistent with a benefit or harm large enough to matter. If there’s no range, check `result.unavailable_reason`.

The comparison measures the effect of the office policy on people who got no offer. If announcing the policy itself changed how they shopped, you can’t attribute the whole difference to their colleagues. See the [API reference](../api.md#two-stage-spillover) for how offices of different sizes count toward the estimate.
