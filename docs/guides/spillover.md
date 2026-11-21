# Does making more offers help people who get no offer?

A spillover experiment asks whether offering something to more people changes outcomes for people who **weren’t offered it**. You can compare two or more offer levels for the same treatment. This is not a comparison of different products and not a switchback over time.

For example, a retailer might assign stores to make few, some or many offers and compare purchases by customers who received none. Each customer belongs to one store. Offers in one store must not influence customers in another. To use the interval, assume that **how many** peers receive offers can matter, but **which** peers receive them does not.

## Before launch: choose comparisons and freeze the roster

Give every policy a name. Set the number of offers each store would make under **each** policy and the number of stores that will use each policy. If you draw stores separately by region, use a separate set of store counts for each region (a *stratum*). Zero offers is fine. A policy that offers to everyone cannot be used to measure what happens without an offer.

Choose the direction and size of each comparison before seeing outcomes. `policy: many, reference: few` means **many minus few**. `meaningful_effect: 0.05` means a rise of five percentage points, not a 5% relative increase. A negative value means a decrease. For a precision check, add a comparable pre-experiment event rate as `baseline_rate` on that comparison. Omit it and you still get an estimate, but no interval.

Save these definitions alongside the **frozen**, versioned roster and offer counts. In this six-store example, two stores receive each policy. The size and names are examples, not limits.

```python
from pathlib import Path

study = Path("offers.yaml")
study.write_text("""\
spillovers:
  - name: store_offers
    design:
      outcome: purchased
      horizon: within seven days
      groups_per_policy: {all: {few: 2, some: 2, many: 2}}
      comparisons:
        - {policy: some, reference: few, meaningful_effect: 0.05}
        - {policy: many, reference: few, meaningful_effect: 0.05}
    relations:
      roster: eligible_customers_v1
      offer_counts: planned_counts_v1
      group_policy: store_assignment
      offered: customer_assignment
      outcomes: seven_day_purchase
""")
```

Point `relations` to tables or views for **this experiment**. A short table name means the connection's current schema at registration. Give a fully qualified name if the table lives elsewhere. The reader stays bound to those locations even if you switch schemas. Use a view to rename columns or limit the rows. Each customer has one roster row. Give the full plan, not just the offer count for the policy a store eventually received.

| Relation | Columns | When needed |
|---|---|---|
| `roster` | `person_id`, `group_id`, `stratum` | Before launch |
| `offer_counts` | `group_id`, `policy`, `offers` | Before launch: one row per store and possible policy |
| `group_policy` | `group_id`, `policy` | After launch: the policy assigned to each store |
| `offered` | `person_id`, `offered` | After launch: the actual offer assignment for **every** eligible customer |
| `outcomes` | `person_id`, `outcome` | After the window closes: a yes/no outcome for **every** customer without an offer |

<details markdown="1">
<summary>Create the example’s planning tables in DuckDB</summary>

```python
import ibis
import pyarrow as pa

con = ibis.duckdb.connect()
stores = list("ABCDEF")
people = [f"{s}{i}" for s in stores for i in range(4)]
roster = pa.table(
    {"person_id": people, "group_id": [p[0] for p in people], "stratum": ["all"] * 24}
)
counts = pa.table(
    {
        "group_id": [s for s in stores for _ in range(3)],
        "policy": ["few", "some", "many"] * 6,
        "offers": [0, 1, 2] * 6,
    }
)
con.create_table("eligible_customers_v1", roster)
con.create_table("planned_counts_v1", counts)
```

</details>

Use your Ibis connection as `con`. Planning reads eligibility and offer counts, not assignments or purchases. Save `registration_id` **before either random draw**.

```python
from increment import from_definitions, run

experiment, reader = from_definitions(study, con, family="spillover")
with reader:
    for comparison in experiment.planning:
        print(comparison.policy, comparison.reference, comparison.status)
    registration = experiment.registration_id
```

`supported` means the simulated designs met the detection target for the chosen change. `underpowered` means the interval checks passed but at least one scenario missed that target. `inconclusive` means the simulations could not settle the power question. `not_calibrated` means plan for a point estimate only; read the comparison's `reason`. None of these labels proves how a real population will respond.

## Draw the assignments, then measure

First, randomly allocate the exact number of stores to each policy within each stratum. Then randomly choose the planned number of customers in each store to receive an offer. Each possible selection of the required size must be equally likely. Make the draws independently across strata and, at the second stage, across stores. Logging the right counts doesn’t prove the draws were random. Record **who got an offer**, not who took it up.

After seven days, an explicit zero means a confirmed nonpurchase. A missing outcome for a customer who got no offer is an error, not a zero. Outcomes for customers who got an offer are optional.

<details markdown="1">
<summary>Load the example’s recorded assignments and purchases</summary>

```python
labels = ["few", "some", "many"] * 2
policy = pa.table({"group_id": stores, "policy": labels})
selected = {"B0", "C0", "C1", "E0", "F0", "F1"}
offers = pa.table({"person_id": people, "offered": [p in selected for p in people]})
unoffered = [p for p in people if p not in selected]
bought = {"D0", "D1", "B1", "E1", "E2", "C2", "F2", "F3"}
purchases = pa.table({"person_id": unoffered, "outcome": [p in bought for p in unoffered]})
con.create_table("store_assignment", policy)
con.create_table("customer_assignment", offers)
con.create_table("seven_day_purchase", purchases)
```

</details>

## Read the comparisons you registered

```python
experiment, reader = from_definitions(
    study, con, family="spillover", expected_registration=registration
)
with reader:
    rows = run(experiment, reader).comparisons
for row in rows:
    print(
        f"{row.policy} minus {row.reference}: {100 * row.estimate:+.1f} points; {row.unavailable_reason}"
    )
```

<!-- output -->
```text
some minus few: +25.0 points; missing_baseline_rate
many minus few: +50.0 points; missing_baseline_rate
```

The estimated purchase rate among customers who got **no offer** was 25 percentage points higher under “some” than under “few.” The example supplied no historical rate, so there’s no interval: the point alone is not enough to conclude that more offers helped.

With a qualified plan and enough variation in the observed data, `lower` and `upper` give an approximate range in proportion units. The ranges account for **all** registered comparisons, even if one can't report an interval. People and store outcome rows stay in the warehouse; Python receives policy summaries. Each eligible person counts equally, which is not the same as pooling all customers who got no offer.

Already have dataframes? Use [`increment.spillover.from_frame`](../api.md#two-stage-spillover) with the same five roles, then `run(experiment, reader)`. No warehouse connection is needed. See the [validation evidence](../reference/validation-evidence.md#ev-spillover-rct2) for the published comparison and the limits of the planning check.
