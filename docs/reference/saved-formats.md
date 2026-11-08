# Saved formats

Increment writes files that a later release may read back: saved runs, moments
cubes, sequential checkpoints, artifacts and decision logs. Each file records
its format version. The table lists, for this release, the version each writer
produces, the oldest version each reader accepts, the full set of versions it
reads, and the route forward for a file outside that set.

A reader never upgrades a file silently and never guesses at a version it does
not know. A file written by a newer release refuses with a stable code; upgrade
Increment to read it. Before 1.0, a release may stop reading an old version;
its release notes list each removal with its migration route.

| Format | Writes | Oldest readable | Support window | Migration |
|---|---|---|---|---|
| Saved runs in `ParquetResultsStore` | `increment.run_manifest.v2` | `increment.run_manifest.v2` | Version 2 only. Any other version refuses with `store.run.manifest_unsupported`; its context lists the versions `found` in the file and the versions this release reads (`supported`). | None needed: version 2 is the only version written so far. |
| Result cells (`ResultCell`, `AnalysisEnvelope`), including the cells of a saved run | schema 3 | schema 1 | Schemas 1–3. A newer schema, or a result kind this release does not know, refuses the whole file with `results.cell.schema_unsupported`. A switchback envelope contrast (`switchback_unit_variance_envelope`) saved before it recorded `residual_side` refuses with `estimation.contrast.legacy_envelope`. | Schemas 1 and 2 are read in place. Recompute an envelope contrast that refuses from its source data. |
| Moments cubes (`export_moments`, `from_moments`) | Fixed-horizon format 12, or 13 when CUPED evidence is needed; sequential format 10 | Fixed-horizon format 10; sequential format 9 | Fixed-horizon formats 10–13 and sequential formats 9–10. Formats below 9 refuse with `moments.format.unsupported_legacy`; formats above 13 with `moments.format.unsupported_future`. | Re-export an older cube from its original data with `workflows.export_moments()`. |
| Sequential checkpoints | version 2 | version 2 | Version 2 only. Any other version refuses with `sequential.continuation.legacy`. | None. Keep the release that wrote the checkpoint, or start a new registered process by capturing without `previous=`. |
| CUPED evidence in moments cubes | version 1 | version 1 | Version 1 only. Any other version refuses with `moments.cuped_evidence.unsupported`. | None. Re-export the cube from its original data. |
| Unit-day artifact context | format 2 | format 2 | Format 2 only. Format 1 refuses with `artifact.format.unsupported`. | Republish from trusted definitions. |
| Artifact relation digests | format 4 for warehouse publication | format 1, subject to the warehouse verification route | Formats 1–4 are described by stored manifests. Format 4 has SQL verification on DuckDB, PostgreSQL, Snowflake and BigQuery; format 2 has DuckDB-only verification. Formats 1 and 3 refuse adoption where no verifier is available, with `query.session.warehouse_artifact.verification_unavailable`. | Republish from trusted definitions with warehouse-side integrity enabled. Never relabel a digest. |
| Artifact extensions | Version 1 for general records; 2 for encouragement uptake; 3 for trigger populations; 5 for assignment identity rows | version 1 for general records and assignment counts; version 2 for usable encouragement uptake; version 3 for trigger populations | Kind-specific versions, not interchangeable: general version 1; uptake 1–2 (version 1 cannot serve timestamp-bearing compliance); trigger 3; assignment counts 1, 4 and 5. | Republish from trusted definitions when newer evidence is required. |
| Decision logs | schema 2 when a decision records its decision plan, otherwise schema 1 | schema 1 | Schemas 1–2. A newer schema refuses when the log is opened, with `store.decision.log_corrupt`. | None needed. |
| Decision plans bound to saved runs and decision logs | version 2 when the plan records a favorable side (a metric declares `preferred_direction="increase"` or `"decrease"`, or the plan is one-sided), otherwise version 1 | version 1 | Versions 1–2. A version-1 plan records no favorable side, so a rejecting primary decided from it reads `decision.primary.direction_unspecified`. Any other version, and a version-2 plan read by an earlier release, makes the record unreadable: `store.run.corrupt` for a saved run, `store.decision.log_corrupt` for a decision log. A moments cube stores its plan without this version: a cube exported by an earlier release reads with no favorable side, and an earlier release refuses a plan that records one with `wire.payload.invalid`. | None needed. Run the experiment again to record its favorable sides. |
| Offline definitions catalog | schema 2 | schema 2 | Schema 2 for unchanged definitions; comment and whitespace edits do not count as changes. Schema 1, any other schema, or changed definitions refuse offline with `kernel.definitions.catalog_stale`. | Rebind the definitions with a connection to rebuild it. The catalog is a cache, not a portable format. |

Dashboards are not saved.

A saved run that this release cannot read also stops `open_run` without a
`run_id`, and `history`, from reading that store directory: the run's
experiment cannot be read, so it cannot be skipped safely. Opening a different
run by its `run_id` still works.

Artifact relation digests and extension records carry their own versions; see
[Persisted formats](../api.md#persisted-formats) and
[Canonical unit-day artifacts](../guides/data-model.md#canonical-unit-day-artifacts).
