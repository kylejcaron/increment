# Capture overlap evidence from DuckDB

Use `increment.query.overlap.capture_overlap_warehouse` when the roster, first decisions, deliveries, and finalized outcomes for two independent experiments are already in DuckDB. The result contains counts and per-cell outcome summaries. It does not contain unit keys or event records.

## Bind the source tables

Use four native tables in the same local, file-backed DuckDB catalog, on one Ibis connection. Pass catalog- and schema-qualified relations, for example `connection.table("roster", database=("experiments", "main"))`. Use persisted columns with exact, case-sensitive string comparisons and timezone-aware timestamps. Views, prepared SQL, Ibis expressions over tables, in-memory databases, generated columns, defaults on the input columns, custom collations, remote sources, and other database engines are not qualified for this operation.

Pass your `OverlapPlan`, a producer-supplied `OverlapCapture`, and the four tables to `increment.query.overlap.bind_overlap_warehouse`. That function keeps the plan and coverage declarations, but writes the source IDs, versions, and generation ID from the actual tables:

- Each source ID identifies the resolved local database path, catalog, schema, and table. The receipt contains a hash, not the path.
- Each version hashes the declared columns of that study's rows. A change to a row, or to the number of copies of a row, changes the version. Reordering rows does not. Outcome values are encoded as doubles and timestamps as UTC instants.
- The generation identifies the four source ID and version pairs together.

Use the returned receipt as `capture` and pass the same four qualified tables to `capture_overlap_warehouse`. The warehouse checks that the receipt still matches while counting invalid records and computing the assignment cells in one SQL statement. If any feed has changed, capture refuses with `overlap.capture.invalid` (`role="snapshot"`, `reason="unbound_generation"`). Rebind only if the new contents still match your declared coverage and cutoff. A matching receipt cannot establish that the producer included every eligible unit, that events were logged when they happened, or that the archived randomization law was followed.

## Work done by the warehouse

Binding and capture both scan the study inputs. DuckDB hashes each projected row, sorts hashes within 4,096 possible buckets, and hashes those bucket summaries in order. Only the four final feed digests leave the database. Capture also checks key uniqueness and joins the roster with each first decision. It counts delivery events per unit before joining assignments and uses two centered passes for complete additive outcomes; binary outcomes use exact counts. Data transfer is bounded by one summary row, ten delivery groups, and four cell rows per declared metric, regardless of how many units were enrolled. On a data validation failure, no delivery groups or cells leave DuckDB.
