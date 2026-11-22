# Does making more offers help people who get no offer?

A spillover experiment asks whether offering something to more people changes outcomes for people who **weren’t offered it**. You can compare two or more offer levels for the same treatment. This is not a comparison of different products and not a switchback over time.

For example, a retailer might assign stores to make few, some or many offers and compare purchases by customers who received none. Each customer belongs to one store. Offers in one store must not influence customers in another. To use the interval, assume that **how many** peers receive offers can matter, but **which** peers receive them does not.

## Before launch: choose comparisons and freeze the roster

Give every policy a name. Set the number of offers each store would make under **each** policy and the number of stores that will use each policy. If you draw stores separately by region, use a separate set of store counts for each region (a *stratum*). Zero offers is fine. A policy that offers to everyone cannot be used to measure what happens without an offer.

Choose the direction and size of each comparison before seeing outcomes. `policy: many, reference: few` means **many minus few**. `meaningful_effect: 0.05` means a rise of five percentage points, not a 5% relative increase. Set `alpha` once for the experiment. At the default `0.05`, the mathematical rule limits the chance that **any** registered range misses its true effect to 5%, under the assignment rules below. Smaller alpha generally means wider ranges. Software approximates the endpoints; don't interpret a difference at the last decimal place. The [evidence and assumptions](../reference/validation-evidence.md#ev-spillover-rct2) give the details.

Want to know whether the experiment could detect that change? Give the comparison a comparable pre-experiment event rate as `baseline_rate`. Without it, Increment skips the power check, **not** the final interval.

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
      alpha: 0.01
      groups_per_policy: {all: {few: 2, some: 2, many: 2}}
      comparisons:
        - {policy: some, reference: few, meaningful_effect: 0.05}
        - {policy: many, reference: few, meaningful_effect: 0.05}
    relations:
      roster: eligible_customers_v1
      offer_counts: planned_counts_v1
      group_policy: store_assignment_v1
      offered: customer_assignment_v1
      outcomes: seven_day_purchase_v1
""")
```

Use durable tables or views for **this experiment**. Short names use the connection's current schema at registration; use fully qualified names to reopen from another schema. The registration checks the roster and planned counts, not the contents of future logs or outcomes. To reproduce a result, bind snapshots that cannot be overwritten and keep them. A version-looking name on a mutable view is not a snapshot.

With DuckDB, attach the databases before registering, even if the outcome tables do not exist yet. Use absolute file paths to reopen on another connection; an in-memory database works only on its original connection. With Snowflake, avoid a temporary table or view with the same name as a registered input. The check applies to the named object, not the tables inside a view.

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

database = Path("offers.duckdb").resolve()
con = ibis.duckdb.connect(database)
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

Read `experiment.planning` before launch. `supported` means all modeled scenarios met the 80% detection target; `underpowered` means at least one fell short. `inconclusive` means the simulations couldn't settle it; `not_assessed` gives a reason, such as missing history. These labels describe power, not whether the assignment logs and outcomes are ready. `run` checks those after the trial.

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
con.create_table("store_assignment_v1", policy)
con.create_table("customer_assignment_v1", offers)
con.create_table("seven_day_purchase_v1", purchases)
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
    point = f"{row.policy} minus {row.reference}: {100 * row.estimate:+.1f} points"
    if row.lower is None or row.upper is None:
        print(f"{point}; no interval: {row.unavailable_reason}")
        continue
    print(f"{point}; range {100 * row.lower:+.1f} to {100 * row.upper:+.1f}; {row.direction}")
```

<!-- output -->
```text
some minus few: +25.0 points; range -97.7 to +99.1; inconclusive
many minus few: +50.0 points; range -97.3 to +100.0; inconclusive
```

The best estimate for “some” versus “few” is a **25-point increase** in purchases by customers who got no offer. But only two stores ran each policy. The range includes both a loss and a gain, so the result is **inconclusive**. It is not “no effect,” and it is not a missing test.

If the whole range is above zero the row reports `increase`; if it is below zero, `decrease`. Whether that is good news depends on the outcome. A range that touches zero is inconclusive. All registered comparisons are protected without needing a previous event rate. Each eligible person counts equally; simply pooling everyone who got no offer answers a different question.

Only people in stores that would make **different numbers of offers** can contribute to this spillover. If the two policies would make exactly the same number in every store, the effect is zero by design: the point and both ends of the range are zero. This follows from the assumption that the number of offers matters, not which coworkers get them or what the policy is called. It is not evidence that two genuinely different offers have the same effect.

Warehouse analysis returns totals by region and policy, not rows labeled with a customer or store ID. That is not anonymization: if just one customer went without an offer in a cell, its event count reveals that customer's outcome. Apply your usual access and small-cell reporting rules.

Already have dataframes? Use [`increment.spillover.from_frame`](../api.md#two-stage-spillover) with the same five roles, then `run(experiment, reader)`. No warehouse connection is needed. See the [validation evidence](../reference/validation-evidence.md#ev-spillover-rct2) for the published comparison and the limits of the planning check.

## Keep a result you can reproduce

Spillover results aren't written to Increment's results store. The registration ID identifies the design, not a particular set of purchases. Keep the definitions, the versions of all five tables **as analyzed**, and the comparison rows you reported. This example keeps one DuckDB file and records its checksum so you can verify it later. In a shared warehouse, bind immutable snapshots and retain their version identifiers instead.

```python
import json
from dataclasses import asdict
from hashlib import file_digest

import yaml

declared = yaml.safe_load(study.read_text())
schema = con.current_database
con.disconnect()
with database.open("rb") as snapshot:
    snapshot_sha256 = file_digest(snapshot, "sha256").hexdigest()
archive = {
    "registration_id": registration,
    "definitions": declared,
    "database": str(database),
    "schema": schema,
    "snapshot_sha256": snapshot_sha256,
    "comparisons": [asdict(row) for row in rows],
}
with Path("store-offers-result.json").open("x") as output:
    json.dump(archive, output, allow_nan=False, indent=2)
```

Retain the database file read-only alongside the JSON; a checksum can't recover deleted data. No customer rows are exported from the warehouse to build the report. If purchases are corrected later, keep the original snapshots and report the corrected analysis separately.
