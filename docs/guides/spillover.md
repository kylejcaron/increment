# Does making more offers help customers who get no offer?

A retailer wants to know whether making more coupon offers in a store changes a customer's chance of buying **when that customer is not offered a coupon**. This is a spillover: other people's offers may affect your outcome. The retailer will compare three offer policies using its usual seven-day purchase metric, then save and reopen the results.

## 1. Name the experiment and its metric

In this example, `few`, `some` and `many` mean **zero, one and two offers per store**, respectively. They are not the customer's own assignment. `offered` and `no_offer` are the two possible assignments for a customer. These names, the table names, `store_offers` and `purchases_7d` are choices, not reserved words.

Put the following declarations in your definitions directory. The DuckDB example writes them into one file; the purchase metric can also be shared with other experiments.

```python
from pathlib import Path

definitions = Path("definitions")
definitions.mkdir(exist_ok=True)
(definitions / "offers.yaml").write_text(
    """\
dialect: duckdb
fact_sources:
  - name: purchases
    sql: SELECT customer_id, ordered_at, 'purchase' AS event, payment_state FROM orders
    timestamp_column: ordered_at
    entities: [customer_id]
    facts: [{name: purchase, column: null}]
    properties: [{name: payment_state, column: payment_state}]
metrics:
  - name: purchases_7d
    type: conversion
    entity: customer_id
    fact: purchase
    filters: [{property: payment_state, op: equals, values: [paid]}]
    window_days: 7
exposures:
  - name: coupon_assignment
    sql: |
      SELECT customer_id AS unit_id, assigned_at AS ts, assignment AS group_id, experiment_id
      FROM customer_assignments
experiments:
  - name: store_offers
    unit: customer_id
    exposure: coupon_assignment
    start: 2025-09-01
    end: 2025-09-02
    observation_end: 2025-09-08
    day_boundary: UTC
    control_group: no_offer
    plan: {primary: purchases_7d, alpha: 0.01}
    design:
      mechanism: spillover
      offered_group: offered
      groups_per_policy: {all: {few: 2, some: 2, many: 2}}
      comparisons:
        - {policy: some, reference: few, meaningful_effect: 0.05}
        - {policy: many, reference: few, meaningful_effect: 0.05}
      roster: eligible_customers_v1
      offer_counts: planned_counts_v1
      group_policy: store_assignments
""",
    encoding="utf-8",
)
```

The three extra tables describe the two draws. Point the three names at your own tables or views; use qualified names when needed.

| Table | One row for… | What it records |
|---|---|---|
| `roster` | Each eligible customer, before launch | `person_id`, their `group_id` (store), and `stratum` (the set of stores drawn together, here `all`). |
| `offer_counts` | Each store and **possible** policy, before launch | `group_id`, `policy`, and `offers`: how many customers would get offers if that policy were drawn. Every policy you compare must leave at least one customer in each store without an offer. |
| `group_policy` | Each store, after the first draw | `group_id` and the one `policy` actually assigned. The customer's assignment and its time come separately from `coupon_assignment`. |

With DuckDB, attach the databases and create any views before registering. Assignment and purchase **tables** can be populated or created later at their registered names; a new view is a different definition. Use absolute database paths to reopen from another connection.

## 2. Register before either random draw

Freeze the eligible people and planned counts. Two stores per policy gives six stores here; each has four eligible customers. No assignments or purchases need to exist yet.

```python
import ibis
import pyarrow as pa

from increment import Experiment, from_definitions, run
from increment.power import spillover_power

con = ibis.duckdb.connect(Path("offers.duckdb").resolve())
stores = list("ABCDEF")
people = [f"{s}{i}" for s in stores for i in range(4)]
roster = pa.table(
    {"person_id": people, "group_id": [p[0] for p in people], "stratum": ["all"] * len(people)}
)
counts = pa.table(
    {
        "group_id": [s for s in stores for _ in range(3)],
        "policy": ["few", "some", "many"] * len(stores),
        "offers": [0, 1, 2] * len(stores),
    }
)
con.create_table("eligible_customers_v1", roster)
con.create_table("planned_counts_v1", counts)

trial, planning_reader = from_definitions(definitions, con, experiment_name="store_offers")
saved_plan = Path("registered-offers.json")
with planning_reader:
    saved_plan.write_text(trial.to_json(), encoding="utf-8")
    for check in spillover_power(trial):
        print(check.policy, "minus", check.reference, check.message)
```

`meaningful_effect: 0.05` means a rise of **five percentage points**. For a power check, also give the comparison a comparable historical purchase rate as `baseline_rate`. `supported` means all modeled scenarios met the 80% detection target; `underpowered` means at least one fell short; `inconclusive` means the simulations couldn't settle it. Without history, the check says `not_assessed`, not that the final interval is unavailable. Save the experiment file before launch and keep it unchanged.

## 3. Draw the assignments, then measure

Draw exactly the planned number of stores per policy in each stratum. In each store, draw exactly its planned number of people to offer the coupon. Every subset of the required size must be equally likely. Draw independently across strata and, within the second stage, across stores. Record assignments for **everyone**, not redemptions, opened messages or visits.

<details markdown="1">
<summary>Load the example's assignments, orders and first batch receipt</summary>

```python
from datetime import UTC, datetime


def september(day, hour=0):
    return datetime(2025, 9, day, hour, tzinfo=UTC)


policies = pa.table({"group_id": stores, "policy": ["few", "some", "many"] * 2})
con.create_table("store_assignments", policies)
selected = {"B0", "C0", "C1", "E0", "F0", "F1"}
assignments = ["offered" if p in selected else "no_offer" for p in people]
con.create_table(
    "customer_assignments",
    pa.table(
        {
            "customer_id": people,
            "assigned_at": [september(2, 10)] * len(people),
            "assignment": assignments,
            "experiment_id": ["store_offers"] * len(people),
        }
    ),
)

early_buyers = ["D0", "D1", "B1", "E1", "E2", "C2", "F2"]
# B1 buys twice; A0 has an unpaid order. Neither adds another conversion.
orders = early_buyers + ["B1", "A0"]
con.create_table(
    "orders",
    pa.table(
        {
            "customer_id": orders,
            "ordered_at": [september(3)] * len(orders),
            "payment_state": ["paid"] * (len(orders) - 1) + ["unpaid"],
        }
    ),
)


def load_receipt(batch, cutoff, through):
    return pa.table({"batch": [batch], "cutoff": [cutoff], "complete_through": [through]})


con.create_table("purchase_load_receipts", load_receipt("first", september(9), september(8)))
```

</details>

Match the purchase feed's **load receipt**, not today's date or the most recent purchase, to the declared window. A qualifying paid order strictly after assignment and before midnight at the start of September 9 counts once. Everyone's window must be complete, even if they have already bought. Only then does no qualifying order mean zero.

The warehouse copies only purchases that could fall within an eligible customer’s measurement window, not the store’s full purchase history. If a shared metric requires unique order IDs, it still checks for duplicates across the full purchase feed.

```python
from increment.artifacts import SourceSnapshotEvidence
from increment.errors import CodedError
from increment.store import ParquetResultsStore

registered = Experiment.from_json(saved_plan.read_text(encoding="utf-8"))
store = ParquetResultsStore("results")
run_id = "offers-final"


def measure(batch):
    loads = con.table("purchase_load_receipts")
    (receipt,) = loads.filter(loads.batch == batch).to_pyarrow().to_pylist()
    evidence = SourceSnapshotEvidence(
        observation_cutoff_ts=receipt["cutoff"],
        complete_through_by_feed={"purchases": receipt["complete_through"]},
    )
    _, reader = from_definitions(
        definitions,
        con,
        experiment_name=registered.experiment_id,
        source_snapshot_evidence=evidence,
    )
    with reader:
        return run(registered, reader, results_store=store, run_id=run_id)


try:
    measure("first")
except CodedError as error:
    assert error.code == "spillover.window.incomplete"
else:
    raise AssertionError("The September 8 purchases have not all arrived")

key = (registered.experiment_id, registered.analysis_spec_hash, run_id)
try:
    store.open_run(*key)
except CodedError as error:
    assert error.code == "store.run.not_found"
else:
    raise AssertionError("An incomplete experiment should have no saved run")
```

The first job guarantees purchases only through the **start** of September 8. Wait for the next completed job. Here its last batch contains F3's purchase on September 8 and certifies the full day. Use a fresh reader with that receipt and the **original** experiment, not a shorter window or a newly registered plan.

```python
con.insert(
    "orders",
    pa.table(
        {
            "customer_id": ["F3"],
            "ordered_at": [september(8, 17)],
            "payment_state": ["paid"],
        }
    ),
)
con.insert("purchase_load_receipts", load_receipt("final", september(9), september(9)))
result = measure("final")
print(result.readout())
con.disconnect()
```

## 4. Read the saved comparisons without reconnecting

```python
from increment import RunResult

handle = ParquetResultsStore("results").open_run(*key)
reopened = RunResult.from_handle(handle, registration=registered.registration)
assert tuple(reopened.readout()) == tuple(result.readout())
table = reopened.to_frame()

for row in reopened.readout():
    point = f"{row.policy} minus {row.reference}: {100 * row.estimate:+.1f} points"
    if row.lower is None or row.upper is None:
        print(f"{point}; no interval: {row.unavailable_reason}")
    else:
        print(f"{point}; range {100 * row.lower:+.1f} to {100 * row.upper:+.1f}; {row.direction}")
```

Expected output for these recorded purchases (percentage points):

<!-- output -->
```text
some minus few: +25.0 points; range -97.7 to +99.1; inconclusive
many minus few: +50.0 points; range -97.3 to +100.0; inconclusive
```

For a formatted comparison table, pass `reopened` to `increment.tables.readout_table`. It shows both policy names and expresses the difference and its range in percentage points.

The estimates compare policies for the **whole eligible population if each customer personally received no offer**. Only customers drawn to receive no offer supply these outcomes. With equal-sized stores here, their purchase rates average 25%, 50% and 75% across stores under `few`, `some` and `many`. Hence `some` minus `few` is +25 points. With unequal store sizes, simply pooling nonoffered customers gives the wrong weights.

Only two stores ran each policy. Both ranges allow a loss as well as a gain: **inconclusive is not no effect**. A range wholly above or below zero reports an increase or decrease. If the interval calculation fails, the point stays visible with a missing range and its reason. The family of registered ranges uses one `alpha` (here `.01`), not one test per customer.

This interpretation assumes that customers in one store cannot affect another store, and that **how many** other customers get the offer matters, not **which** customers get it. If two policies assign identical counts in every store, the effect is zero by design under those assumptions. See the [evidence and limits](../reference/validation-evidence.md#ev-spillover-rct2).

The results store keeps the comparison rows, not the customer records. Keep immutable input versions if you need to repeat the calculation. Even a warehouse total can reveal a customer's outcome when just one person contributes; apply your usual access rules.

## Already have one row per customer?

Use `from_frame(shape="summary")` with the same policies, comparisons and plan. Here `group` names the **individual assignment** column and `cluster` names the **store** column. Supply a measured yes/no value for every customer without an offer; missing is not zero. `window_days=7` describes the supplied flag. The summary reader does not calculate a window from timestamps.

```python
from math import isclose

from increment import AnalysisPlan, MetricSpec, Spillover, SpilloverComparison, from_frame

buyers = {*early_buyers, "F3"}
summary = roster.append_column("assignment", pa.array(assignments)).append_column(
    "purchases_7d", pa.array([None if p in selected else p in buyers for p in people])
)
local, reader = from_frame(
    summary,
    shape="summary",
    experiment_id="store_offers",
    unit="person_id",
    group="assignment",
    cluster="group_id",
    stratum="stratum",
    design=Spillover(
        control_group="no_offer",
        offered_group="offered",
        groups_per_policy={"all": {"few": 2, "some": 2, "many": 2}},
        comparisons=(
            SpilloverComparison("some", "few", 0.05),
            SpilloverComparison("many", "few", 0.05),
        ),
    ),
    offer_counts=counts,
    group_policy=policies,
    metrics=[MetricSpec(name="purchases_7d", type="conversion", window_days=7)],
    plan=AnalysisPlan(primary="purchases_7d", alpha=0.01),
)
with reader:
    local_rows = run(local, reader).readout()
for left, right in zip(local_rows, reopened.readout(), strict=True):
    assert (left.policy, left.reference) == (right.policy, right.reference)
    assert isclose(left.estimate, right.estimate, abs_tol=1e-12)
    assert isclose(left.lower, right.lower, abs_tol=1e-10)
    assert isclose(left.upper, right.upper, abs_tol=1e-10)
```
