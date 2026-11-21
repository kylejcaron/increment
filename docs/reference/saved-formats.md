# Saved formats

Each saved file records its format version. The entries below give the current
writer version, the oldest readable version, the complete supported set and
the migration route. Dashboards are not saved.

A reader never upgrades a file silently and never guesses at a version it does
not know. A file written by a newer release refuses with a stable code; upgrade
Increment to read it. Before 1.0, a release may stop reading an old version;
its release notes list each removal with its migration route.

## Saved runs

Applies to `ParquetResultsStore` and `WarehouseResultsStore`.

**Writes**: `increment.run_manifest.v2`

**Oldest readable**: `increment.run_manifest.v2`

**Support window**: Version 2 only. Any other version refuses with `store.run.manifest_unsupported`; its context lists the versions `found` in the file and the versions this release reads (`supported`).

**Migration**: None needed: version 2 is the only version written so far.

## Result cells

Applies to `ResultCell` and `AnalysisEnvelope`, including cells in a saved run.

**Writes**: schema 3

**Oldest readable**: schema 1

**Support window**: Schemas 1–3. A newer schema, or a result kind this release does not know, refuses the whole file with `results.cell.schema_unsupported`. A switchback envelope contrast (`switchback_unit_variance_envelope`) saved before it recorded `residual_side` refuses with `estimation.contrast.legacy_envelope`.

**Migration**: Schemas 1 and 2 are read in place. Recompute an envelope contrast that refuses from its source data.

## Moments cubes

`export_moments` writes these; `from_moments` reads them.

**Writes**: Fixed-horizon format 12, or 13 when CUPED evidence is needed; sequential format 10

**Oldest readable**: Fixed-horizon format 10; sequential format 9

**Support window**: Fixed-horizon formats 10–13 and sequential formats 9–10. Formats below 9 refuse with `moments.format.unsupported_legacy`; formats above 13 with `moments.format.unsupported_future`.

**Migration**: Re-export an older cube from its original data with `workflows.export_moments()`.

## Sequential checkpoints

**Writes**: version 2

**Oldest readable**: version 2

**Support window**: Version 2 only. Any other version refuses with `sequential.continuation.legacy`.

**Migration**: None. Keep the release that wrote the checkpoint, or start a new registered process by capturing without `previous=`.

## CUPED evidence in moments cubes

**Writes**: version 1

**Oldest readable**: version 1

**Support window**: Version 1 only. Any other version refuses with `moments.cuped_evidence.unsupported`.

**Migration**: None. Re-export the cube from its original data.

## Overlap evidence and results

`OverlapEvidence.to_json()` and `OverlapResult.to_json()` write format 1.
Their `from_json(..., expected_sha256=...)` readers accept only format 1 and
require a separately saved SHA-256 digest. Invalid versions or mismatched
digests raise `overlap.format.invalid`. Keep the evidence to rerun the same
plan without rereading the source. Counts and assignment weights cannot
exceed `2**53 - 1`.

For captured inputs and saved results, a matching digest shows that the
contents match the original record. It cannot prove that the producer
included all the data, followed its assignment rules, or described the
launch policy accurately.

## Unit-day artifact context

**Writes**: format 2

**Oldest readable**: format 2

**Support window**: Format 2 only. Format 1 refuses with `artifact.format.unsupported`.

**Migration**: Republish from trusted definitions.

## Artifact relation digests

**Writes**: format 4 for warehouse publication

**Oldest readable**: format 1, subject to the warehouse verification route

**Support window**: Formats 1–4 are described by stored manifests. Format 4 has SQL verification on DuckDB, PostgreSQL, Snowflake and BigQuery; format 2 has DuckDB-only verification. Formats 1 and 3 refuse adoption where no verifier is available, with `query.session.warehouse_artifact.verification_unavailable`.

**Migration**: Republish from trusted definitions with warehouse-side integrity enabled. Never relabel a digest.

## Artifact extensions

**Writes**: Version 1 for general records; 2 for encouragement uptake; 3 for trigger populations; 5 for assignment identity rows

**Oldest readable**: version 1 for general records and assignment counts; version 2 for usable encouragement uptake; version 3 for trigger populations

**Support window**: Kind-specific versions, not interchangeable: general version 1; uptake 1–2 (version 1 cannot serve timestamp-bearing compliance); trigger 3; assignment counts 1, 4 and 5.

**Migration**: Republish from trusted definitions when newer evidence is required.

## Decision logs

**Writes**: schema 2 when a decision records its decision plan, otherwise schema 1

**Oldest readable**: schema 1

**Support window**: Schemas 1–2. A newer schema refuses when the log is opened, with `store.decision.log_corrupt`.

**Migration**: None needed.

## Decision plans

These versions apply to plans bound to saved runs and decision logs.

**Writes**: version 2 when the plan records a favorable side (a metric declares `preferred_direction="increase"` or `"decrease"`, or the plan is one-sided), otherwise version 1

**Oldest readable**: version 1

**Support window**: Versions 1–2. A version-1 plan records no favorable side, so a rejecting primary decided from it reads `decision.primary.direction_unspecified`. Any other version, and a version-2 plan read by an earlier release, makes the record unreadable: `store.run.corrupt` for a saved run, `store.decision.log_corrupt` for a decision log. A moments cube stores its plan without this version: a cube exported by an earlier release reads with no favorable side, and an earlier release refuses a plan that records one with `wire.payload.invalid`.

**Migration**: None needed. Run the experiment again to record its favorable sides.

## Offline definitions catalog

**Writes**: schema 2

**Oldest readable**: schema 2

**Support window**: Schema 2 for unchanged definitions; comment and whitespace edits do not count as changes. Schema 1, any other schema, or changed definitions refuse offline with `kernel.definitions.catalog_stale`.

**Migration**: Rebind the definitions with a connection to rebuild it. The catalog is a cache, not a portable format.


## When a saved run is unreadable

A saved run that this release cannot read also stops `open_run` without a
`run_id`, and `history`, from reading that store directory or warehouse schema:
the run's experiment cannot be read, so it cannot be skipped safely. Both
refuse with the run's own code, as opening it by its `run_id` does. Opening a
different run by its `run_id` still works.

Artifact relation digests and extension records carry their own versions; see
[Persisted formats](../api.md#persisted-formats) and
[Unit-day artifacts](artifacts-and-replay.md#canonical-unit-day-artifacts).
