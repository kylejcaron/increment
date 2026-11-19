# Saved data, results and replay

Saving an experiment declaration does not save its rows or results. Choose the boundary you need:

| Keep | Use |
|---|---|
| Analysis settings | `Experiment.to_json()` and `Experiment.from_json()` |
| Warehouse unit-day data | A published unit-day artifact |
| Reduced data for another analysis | `export_moments` and `from_moments` |
| A completed run and its decisions | A results store and `RunResult.from_handle` |

See [saved-format versions](saved-formats.md) before moving data between releases. For a complete declare/run/save workflow, start with [experiment specifications](../guides/experiment-specifications.md).

## Canonical unit-day artifacts

### Session-local materialization

`increment.workflows.materialize(experiment, reader)` rebuilds the unit-day core in TEMP relations on the current connection. Each call rescans, except with `store="none"` or no metrics. Readouts rebuild from their first call with `"always"` and second with `"auto"`; nested reductions share only that operation's copy. Refreshing or closing the reader removes the TEMP relations. This is neither portable storage nor an atomic snapshot across source tables.

### Publish an immutable generation

`increment.artifacts.publish_unit_day_artifact(experiment, reader, store, extensions=(), refresh_of=None)` returns a `UnitDayArtifactRef`. Get its trusted context with `increment.artifacts.artifact_context(experiment, reader)`; reopen it with `from_unit_day_artifact(store, ref, expected_context=...)` without rebuilding facts or exposures.

Publication writes the canonical core, format-2 context and format-4 relation digests. A refresh retains `artifact_id`, creates a new `generation_id` and requires the previous generation's compiled context. Timestamps, `data_as_of`, publication time and experiment end do **not** prove completeness; only supplied certifications do.

!!! warning "Do not adopt a failed publication"
    Failure after publication invalidates the generation with `artifact.generation.dropped`. If cleanup fails, retain the exception's artifact/generation IDs and relation names and follow the recovery instructions below.

### What publication captures

One source execution captures facts, exposures, dimensions, triggers and extensions in warehouse TEMP storage. Assignment/cluster validation, measures, freshness, event horizons and coverage use those same inputs. No raw relations are collected into Python; later reductions cannot fall back to live data. Persisted rows are the rows used for their digest.

`SourceSnapshotEvidence` supplies an aware event-time cutoff and optional per-feed `complete_through` certifications. Missing certifications stay unknown. Trigger membership uses the trigger feed's certification; each metric's triggered outcomes use its contributing fact feeds, at that same cutoff.

Percentile-winsorization snapshots, artifact publication, `prepare_dashboard` and sequential capture scope each captured fact/dimension source to enrolled units. For sources bounded by the analysis window, the lower limit is `start - (n_pre_periods + 1)` days. There is no pushed upper limit: freshness and breakout/factor/covariate lookups need later rows.

The manifest is written last. It records identity, experiment/day boundary, coverage, freshness, required `exposures` and `measure_stats` relations, metric bindings, trusted `ArtifactContext` and extensions. Every relation reference is bound to that generation and role.

### Stored relation digests

The warehouse store persists each relation before hashing it. Format 4 is SQL-verifiable on DuckDB, PostgreSQL, Snowflake and BigQuery; historical format 2 is verify-only on DuckDB. Unsupported formats refuse before unit transfer. Only bounded hash/count summaries return to Python: `RelationDigests.rows_bytes` is `None` here, while `digest_relation()` includes canonical row bytes.

Format 4 encodes each finite binary64 as its sign, exact decimal-integer significand, `p` and exact binary exponent. It preserves stored zero's sign; subnormals use exponent -1074. Ingestion may normalize `-0.0` to `+0.0` (including BigQuery Parquet loads); the digest describes stored content. NaN/infinity retain canonical spellings. These tokens are unique per finite IEEE-754 value and avoid backend decimal expansion.

Dates use ISO `YYYY-MM-DD`, not session settings. PostgreSQL timestamp conversion uses `EXTRACT(EPOCH)` before integer microseconds, without floating-point seconds. Text is explicitly UTF-8 bytes, so backslashes remain literal rather than bytea escapes.

Exposure metadata/integrity checks use SQL reductions, joins and anti-joins. Explicit row APIs such as `unit_frame()` still collect requested rows; backend query buffers may grow with relation size. Adoption verifies and pins a warehouse-backed copy: later external table writes cannot change it. Closing the read handle removes its private tables.

## Recovering from a failed publication

**Occupied name — leave the existing relation alone.** `query.relation.name_occupied` identifies `backend`, `catalog`, `schema`, `name` and `route`. Increment does not overwrite or erase it. Retry with another publication/scratch name; delete the existing relation only after independently establishing ownership.

**Before submitting the manifest-index insert.** The generation was never visible. Cleanup attempts every relation, including a failed write whose immediate cleanup also failed. An aborting exception lists retained relations. A publication ending normally instead raises `query.session.warehouse_artifact.publication_cleanup_incomplete` with `relations`, `artifact_id`, `generation_id` and `route`. Remove the listed relations by name.

**After submitting the insert.** It may have committed, or may commit remotely after an exception. Cleanup must tombstone the generation **before** erasing relations; late manifest rows then remain hidden and reads refuse with `artifact.generation.dropped`. Every erasure is attempted even if another fails.

- If tombstoning fails, **all relations are preserved**. From a fresh process, call `store.abandon_generation(artifact_id, generation_id)` first. This durable, repeatable operation writes only the tombstone and hides any present or later manifest. Then remove the listed relations. If a manifest exists, `drop_generation` also works.
- If only erasure fails, the generation is already hidden. Remove the listed retained relations.
- An aborting exception retains the original failure and notes artifact/generation IDs plus catalog/schema-qualified relation names. A normally ending publication with an ambiguous insert and no manifest raises `query.session.warehouse_artifact.publication_state_unknown`, chained to the underlying failure. Its context includes `relations` (a tuple of qualified names), `artifact_id`, `generation_id`, `tombstoned` and `route`.

**After the manifest is published.** The publication context invalidates a failed generation through `drop_generation`, tombstone first. Fresh stores list nothing; `open_snapshot` and `verify_relation` refuse with `artifact.generation.dropped`. If invalidation fails, the original exception is re-raised with the IDs. Retry `store.drop_generation(artifact_id, generation_id)` from a fresh process.

## Dropping a published generation

`WarehouseArtifactStore.drop_generation()` commits a durable tombstone before touching relations. Every store in that namespace refreshes visibility on each access; the generation is then hidden and refused everywhere. The manifest-index row remains a cleanup receipt.

Success means every named relation was erased. Erasure failure leaves the generation invalid; retrying revalidates the receipt and attempts every relation, including on an already-dropped generation. Already-erased relations are left alone. A never-published pair has no effect; use `abandon_generation` to hide it. Exported, cached or copied rows are not revoked.

Before enabling deletion in an existing namespace:

1. Upgrade **every reader and writer**; older versions ignore tombstones and are unsafe during incomplete cleanup.
2. Create the tombstone table with namespace-migration privileges before constructing any store. Construction fails closed if the table is missing and cannot be created.
3. Explicitly drop generations retired by an older in-memory-only release; it left no durable deletion record.

`reader.close()` releases only an adopted snapshot handle, never durable artifact data. Coordinate other backend cleanup policies separately.

Artifact implementations load lazily through the typed `increment.artifact_ports` entry point; this does not change public references, format identities or replay behavior.

## Operation and extension matrix

Every adopted artifact exposes `moments_source`, `day_source`, and
`export_moments` (which exports a moments cube; it does not re-publish the
artifact). On an adopted artifact, `workflows.materialize()` and `query.panel_sql()` are
refused with `evidence.operation_unavailable`. Optional operations are enabled only by
their published evidence:

- `breakout_dimension` enables `breakout_source`, `breakout_sources` and `breakout_summaries`.
- `factor_dimension` enables `factor_summaries`.
- `site_volume` enables `sitewide_evidence`.
- `trigger_population` plus `assignment_counts` enables `triggered_counts`.
- Add the metric's `trigger_measure_stats` to those trigger/count extensions for metric-specific `triggered_source` outcomes.

An `assignment_counts` extension with an `assignment_rows` relation also retains
row-level assignment identity. It enables typed assignment evidence and daily
`allocation_history()`; count-only artifacts report reduced assignment evidence
and refuse identity-dependent history with
`evidence.assignment.identity_unavailable`. History is an aggregate by exposure
day and arm, not per-unit data for assignment replay verification.

`cluster_identity`, `cuped_preperiod`, and `encouragement_uptake` provide
relation evidence consumed by the corresponding total/day reductions; they do
not add a generic operation. The extension catalog is closed and immutable: a
request must match exactly one catalog entry, including its canonical
definition and source-provenance hashes. Supported requests are
`breakout_dimension`, `factor_dimension`, `cluster_identity`,
`cuped_preperiod`, `assignment_counts`, `trigger_population`,
`trigger_measure_stats`, `encouragement_uptake`, and `site_volume`.


### Triggered outcomes and site volume

Triggered outcome summaries are raw-event-filtered from the unit's first
eligible trigger, then aggregated by that trigger-relative local day. The
exclusive window edge is the first local day after the declared number of
days. Trigger-population evidence before version 3 and outcome artifacts
without the selected metric's summary cannot serve a triggered outcome;
use definitions-backed input or rebuild and republish with version-3
trigger membership and the metric's `trigger_measure_stats` request.

The publisher filters trigger-relative raw events at the explicit pinned
event-time cutoff before making daily aggregates. The reader rejects
aggregate dates later than the cutoff's local day, while permitting a
partial cutoff-day aggregate. Daily rows contain no raw timestamps, so the
reader cannot independently prove which within-day events were admitted;
exact cutoff filtering on that day is trusted publisher behavior bound to
the artifact's source provenance and cutoff metadata.

Logical metrics may share one physical site-volume recipe. Publication
writes that recipe once; each metric retains its binding to the shared
evidence when the artifact is adopted. Site volume covers the enrollment
window `[start, end]` in days at the declared `day_boundary`, even when
`observation_end` extends past `end`; an experiment without an `end`
publishes no site-volume coverage.

## Artifact refusal codes

- `artifact.format.unsupported`: the context format is not version 2 or the
  artifact format is not version 1; republish from trusted definitions.
- `artifact.manifest.invalid`: the manifest body, identity, ordering, or
  self-digest fails validation.
- `artifact.identifier.unsafe`: a locator, namespace identifier, or digest
  identifier is malformed or outside the permitted identifier grammar.
- `artifact.context.mismatch`: canonical context JSON, its hash, or a supplied
  context payload does not agree.
- `artifact.refresh.invalid_ref`: a pinned ref is malformed, unpublished,
  generation-bound incorrectly, or has a locator/digest mismatch.
- `artifact.refresh.context_mismatch`: a refresh or open uses a context different
  from the caller-pinned/previous generation context.
- `artifact.relation.schema_mismatch` and `artifact.relation.digest_mismatch`:
  a stored relation's schema or content digest differs from its manifest ref.
- `artifact.snapshot.mixed`: a relation, manifest, or generation changes or is
  bound to a different snapshot while a read is in progress.
- `artifact.digest.type`, `artifact.digest.nullable`, and
  `artifact.digest.primary_key`: a digest cell has the wrong physical type,
  violates nullability, or has a missing/duplicate/non-unique primary key;
  multi-row relations must provide a primary key.
- `artifact.digest.row`: a digest row is not a mapping/model or is missing a
  field declared by the relation schema.
- `artifact.digest.rows`: the relation rows container is not a supported
  sequence.
- `artifact.digest.nonfinite`, `artifact.digest.schema`,
  `artifact.digest.timestamp`, `artifact.digest.range`, `artifact.digest.role`,
  `artifact.digest.json`, and `artifact.digest.recanonicalize`: digest input
  violates the corresponding finite-number, schema, UTC timestamp,
  integer-range, ASCII-role, JSON, or canonical-representation rule.
- `artifact.aggregate.impossible`: publication encounters an aggregate with
  missing sufficient statistics, impossible min/max or event counts, or
  non-finite/contradictory single-event values.
- `artifact.extension.missing`, `artifact.extension.invalid`, and
  `artifact.metric.binding_mismatch`: required evidence is absent,
  malformed/tampered, or inconsistent with the manifest metric binding.
- `artifact.operation.unsupported`: a bare artifact reader cannot provide
  `cluster_counts()` without cluster identity; use a definitions-backed
  source with the `cluster_identity` extension. Unsupported grains use
  `evidence.grain_unavailable`; unsupported operations use
  `evidence.operation_unavailable`.
- `artifact.evidence.unavailable`: the requested evidence or shape is not
  available from the selected source or published artifact, even though the
  broader operation is supported — a source lacking a required operation or
  unit-grain evidence shape during extension publication raises this too.
- `artifact.store.namespace`: the store's own catalog/schema namespace does
  not exist or is invalid, or the deletion tombstone table specifically is
  missing and cannot be created, or has an unexpected schema. The
  manifest-index table is validated at store construction too: it must
  contain the columns `artifact_id`, `generation_id`, `manifest_name`,
  `manifest_sha256`, and `manifest_json`, all of string type, with no
  missing, extra, or wrong-type columns (physical column order is free),
  and an absent index that cannot be created is refused as well.
- `artifact.manifest.unreferenced_relation`: a publication wrote a relation
  that its manifest does not reference (in base relations or extensions);
  publication is sealed against this before the manifest is persisted.
- `artifact.generation.dropped`: the requested generation has a durable
  deletion tombstone; every read (`open_snapshot`, `verify_relation`) is
  refused with this code. Dropping it again is not refused: `drop_generation`
  revalidates the retained receipt and retries erasing every named relation
  idempotently.

## Trust, namespace, and read consistency

The caller-pinned `UnitDayArtifactRef` (artifact UUID, generation UUID,
namespace-qualified manifest locator, and manifest digest) is the trust root.
The store, not Increment, owns authorization for the catalog/schema namespace;
locators outside that namespace or outside the pinned generation are refused.

For BigQuery, a store resolves its effective project and dataset once during
construction, before creating or reading its control-plane tables. An omitted
namespace binds to the connection's current data project and dataset; a
dataset-only namespace binds to the billing project and that dataset. A
catalog-only argument retains BigQuery's existing creation behavior and binds
to the connection's current data project and dataset. Supplying both catalog
and dataset binds exactly that pair. The store consistently uses this bound
namespace for metadata, relation creation, reads, tombstones, and cleanup;
later `set_database()` calls on a borrowed connection do not retarget an
existing store.

Older BigQuery artifact receipts that omit a data relation's physical project
or dataset cannot prove which relation was created. They are refused with
`artifact.identifier.unsafe` before looking up or copying that relation's data,
and before tombstoning or erasure. Control metadata may be read first. A valid
manifest checksum does not establish the missing location. Do not replace it
with the current connection defaults; recover or retire the receipt only with
independently verified original location evidence.
Opening validates the pinned manifest and then holds one immutable
`ArtifactSnapshot` for verification and every reduction, so later writes or
rebinding cannot mix generations. If both the artifact and the supplied ref
are replaced together, Increment cannot detect that replacement: the caller's
ref distribution and store authorization are the authenticity boundary.

!!! warning "Digests are integrity evidence, not secrecy"
    Do not put credentials in SQL literals. Hashing does not conceal a secret
    that someone can guess, and copies of a format-1 manifest that were
    already stored or shared still contain the SQL until an operator handles
    them.


### Context and digest construction

`ArtifactContext` is compiled from an already-loaded `Definitions` model,
not from a path or hidden warehouse I/O. Its `context_format` is `2`, and
the same `context_format` appears inside `canonical_json` so the digest
binds it; the two must agree. Its `canonical_json` is compact RFC 8785
canonical JSON: finite values, no whitespace, and object keys ordered by
UTF-16BE code units. Its `sha256` is
`SHA256(b"increment.unit-day-artifact\x00v1\x00context\x00" +
canonical_json_bytes)`, where `\x00` denotes the actual NUL separator byte.

A format-2 context carries the typed experiment (including the effective
encouragement design), the complete selected metric roster, the dialect
and day boundary, and the extension catalog. It never carries fact,
dimension or exposure SQL: the native source recipe and every extension
source recipe are represented by domain-tagged SHA-256 digests, and a
receipt is never executed. The context also carries `window_days`, the
boundary-local day of each declared window edge (`start`, `end`,
`observation_horizon`), and readers use those stored days rather than
deriving an edge from a timestamp. A format-2 context without
`window_days` is refused by `artifact.context.mismatch` (route: republish
from trusted definitions), before any manifest, relation or pin is used.

Context format 1 is retired. Opening or publishing with a format-1 context
is refused by `artifact.format.unsupported`, naming the received and
supported versions. Republish from trusted definitions to obtain a
format-2 artifact; the refusal never upgrades an old context. Existing
artifacts are not deleted and credentials are not rotated automatically.
Relation digest formats are independent of the context format. Publication
on DuckDB, PostgreSQL, Snowflake and BigQuery writes digest format 4, which
the warehouse reader verifies on each of them; format 2 remains verifiable
on DuckDB for existing artifacts. Formats 1 and 3, and format 2 on other
backends, are refused with
`query.session.warehouse_artifact.verification_unavailable` and structured
relation/format context. Format 1 remains available only to the explicit
historical local codec. Republish from trusted definitions to produce a
warehouse-verifiable artifact; this does not upgrade an existing manifest
or transfer its unverified rows.

Relation schema/content and manifest digests use the same domain root,
role-specific markers, schema bytes, row count, and rows sorted by primary
key. The manifest digest excludes its own `manifest_sha256` field. All
digest strings are lowercase 64-character hexadecimal SHA-256.

## Moments wire migration

`from_moments` reads centered, complete `moments_format=12`
fixed-horizon rows, including a complete embedded `decision_plan`, a source
identity record and, on every denominator row, the ratio denominator's
centered third moment `cden3`. Format 13 adds method-scoped CUPED evidence
envelopes when an export needs them: scalar-family and categorical-cell
evidence, or a fitted continuous/mixed multicovariate QR state. Fitted
continuous/mixed envelopes replay from `from_moments`, including
reintervalization at a changed alpha, but do not refit the model (see
[Portable CUPED evidence](../guides/cuped.md#portable-cuped-evidence)). The reader also
accepts earlier fixed-horizon formats 10 and 11 and sequential checkpoint
formats 9 and 10. Formats above 13 are refused with
`moments.format.unsupported_future`; fixed-horizon formats 1–8
with `moments.format.unsupported_legacy`. `from_moments` is separate from the
unit-day artifact boundary.

!!! warning "An old cube cannot be upgraded in place"
    Re-export from the original observations, or pin the Increment revision
    that wrote the old cube. A cube alone cannot recover exact binary counts
    from rounded moments, and a format-10 or format-11 cube cannot recover the
    third moment: its ratio rows load, and every lift row estimated from them
    reports the denominator-skew coverage check as unavailable rather than
    passing silently. Rounded fixed-horizon moments and legacy format-8
    sequential envelopes cannot resume a sequential process.


### Row contracts, hand-authored summaries and envelopes

Fixed-horizon format 10 and sequential format 9 lack a source identity
field, and a format 11 cube may be loaded without one. Their replay
identity includes a legacy-format namespace and study ID; the original
provenance cannot be reconstructed. A `source_identity` embedded in a
moments cube is self-asserted, not authenticated provenance.

For hand-authored per-arm summaries, `moments_from_summaries(...)` builds
format-12 rows for fixed-horizon mean and conversion metrics. Mean rows
require `n >= 2`, `mean`, and unbiased `var` (ddof=1), then use
`ref_y=mean`, `cy1=0`, and `cy2=(n-1)·var`; conversion rows require exact
integer `successes` and retain that count. Rows have a strict schema;
extra fields are refused with `moments.summaries.invalid`. Optional
`assignment_counts` records source-level enrollment counts independently
of metric counts. When omitted, `workflows.unit_counts()`, `workflows.srm()`, and the health
allocation check are unavailable with `source.moments.assignment_counts`.
The helper does not infer ratio denominators, covariates, cluster
structure, or sequential registration; use `from_frame(..., shape="summary")` for those
richer summaries.

Each ordinary row carries integer `n >= 1` and nullable integer
`successes` between zero and `n`. Eligible unit-grain conversion and
retention outcomes retain their original 0/1 success counts;
cluster-aggregate rows and other outcomes carry null. Partition merges add
counts as integers, never via floating totals. Missing counts are not
reconstructed from a mean. Import and export validate count ranges on
every row, including unselected imported metrics, with
`moments.count_out_of_range` for invalid ranges. Exports from definitions,
unit-day artifacts, and dataframe sources refuse invalid counts before
creating the Parquet file. Parquet exports use nullable `int64` for
`successes`; preserve that type when loading through a dataframe,
including pandas' nullable `Int64` dtype. Warehouse daily conversion
summaries also retain integer successes: each unit contributes at most one
success per day, regardless of its event count.

The `metrics` declaration selects which metrics to replay. Winsorization
declarations must match selected metrics; an unselected clipped metric
does not require its configuration to replay an ordinary sibling. Format
stamps and wire-schema validation still cover every row of the imported
cube.

A metric declared `missing="skip"` is estimated over the units that
observed it. Every row of a format-12 or format-13 export records those
metrics in an optional `metric_missingness` field, a JSON object such as
`{"score":"skip"}`, so a replay that names the metric and its type keeps
the observed-only population without declaring `missing="skip"` again,
and a replay declaring another policy does not relabel those rows. When an
export records skipped metrics, a replay that declares `missing="skip"` for
a metric the field leaves out is refused with `moments.plan.conflict`: the
export reduced that metric over every unit. The field is absent when no
metric skips, and such an export replays as before, without that check. A
cube exported before the field existed reopens unchanged; declare
`missing="skip"` on replay to qualify its population. A field that is not
such an object, names another policy, or differs between rows is refused
with `moments.format.invalid`, and one that repeats a metric with
`moments.duplicate_key`.

An empty-metric fixed-horizon Encouragement export carries one current
format-12 `design_summary` envelope instead of outcome rows. It includes
the experiment identity, complete assignment counts and compliance state,
source identity, and empty-metric fixed-horizon decision plan. Reload with
`metrics=[]` and the same design. Missing state, mixed rows, outcome
fields, a different identity, or a nonempty metric catalog are refused;
overriding the plan cannot bypass envelope validation. This envelope is
not a sequential checkpoint.

Registered sequential exports use one current format-10
`sequential_checkpoint` envelope carrying the exact snapshot, retained
roster, registration (schema version 2, including source mapping identity
and compliance policy), parent-linked record proof, and source identity.
Their embedded compiled plan uses wire version 3. The preceding sequential
format 9 remains readable. See
[Sequential inference](../guides/sequential-inference.md) for explicit finalization,
idempotent replay and append-only continuation.

## Versioned assignment evidence

`AssignmentRequest` acquires reduced assignment evidence separately from the
existing aggregate `CountsRequest(kind="assignment")` and `ComplianceRequest`.
It reports arm counts by allocation version and period, the assignment
independence grain, and any declared allocation probabilities. `AssignmentEvidence`
is reduced metadata and does not return ledger rows or event timestamps. When a
source retains allocated/exposed/triggered timestamps, the public
`FrameLedger.reduce(...).rows` API preserves them independently; unknown times
remain null, and first exposure is never substituted for allocation.

Warehouse-backed definitions can bind an identity-bearing relation directly on
an experiment. The query must return one row per assignment key, and `roles`
maps the semantic fields to the relation's actual column names:

```yaml
experiments:
  - name: checkout
    # Existing experiment fields omitted.
    assignment_ledger:
      sql: SELECT * FROM analytics.assignment_ledger
      roles:
        unit: subject_id
        experiment: experiment_id
        arm: assigned_arm
        version: allocation_version
        period: allocation_period
        allocated_at: allocated_at
        exposed_at: first_exposed_at
        triggered_at: first_triggered_at
```

The `unit`, `experiment`, `arm`, and `version` roles are required. `cluster`,
`period`, `layer`, and the three event-time roles are optional; omitted event
times stay unknown. Requests reduce this relation inside the warehouse source
boundary and return only immutable `AssignmentEvidence`. `run` and the
`increment.workflows` functions issue these requests through the reader; a
custom provider answers the same request, and
`increment.evidence.request_evidence(provider, request)` applies the shared
capability checks before calling it:

```python
from increment.evidence import AssignmentRequest

request = AssignmentRequest(experiment="checkout", population="assigned")
```

The selection is deliberately explicit: a missing version or period is
accepted only when the source has one unambiguous value. Switchback callers may
request `all_periods=True`; observation-day columns are not allocation periods.
The one-shot group-column compatibility evidence is version 1 with a null
period and null event times, and is not a replay certificate. Sources without
assignment capability refuse the relation request with
[`evidence.assignment.unavailable`](../guides/refusals.md#find-the-next-step-by-code-family)
before data access; aggregate counts and compliance requests keep their existing
behavior.

The assignment relation does not change estimator behavior. CUPED, ratio
metrics, winsorization, sequential inference, breakouts, and multiplicity keep
their existing method-specific contracts; this request does not infer
eligibility or create a new analysis method. Cluster assignments report the
cluster as the independence grain and require consistent unit-to-cluster
membership and arm allocation. Replay is `not_certified` without a replay
specification; sampled verification remains a sample check and never certifies
the full ledger. See [Assignment and exposure roadmap issue #93](https://github.com/increment/Increment/issues/93)
and downstream work in [#83](https://github.com/increment/Increment/issues/83),
[#91](https://github.com/increment/Increment/issues/91),
[#92](https://github.com/increment/Increment/issues/92),
[#78](https://github.com/increment/Increment/issues/78), and
[#95](https://github.com/increment/Increment/issues/95).

## Assignment evidence and replay

The assignment evidence request is distinct from `CountsRequest(kind="assignment")`,
which remains an aggregate count operation. Assignment rows use an explicit
identity key `(unit or cluster, experiment, version, period)`; a period is not
the panel's outcome date. Allocation, first exposure and trigger timestamps are
separate nullable fields. Unknown times remain null.

When a source retains no assignment relation or exact reduced assignment
evidence, the typed request refuses with `evidence.assignment.unavailable`.
Do not infer assignment identities from aggregate counts. Mixed allocation
versions require an explicit version selection. A source retaining only
one-shot counts refuses unsupported version or period selections with
`evidence.assignment.selection_unavailable`; switchback consumers may
explicitly select all periods. Design declarations are authoritative for
randomization law, target probabilities, independence grain and carryover;
assignment-integrity checks derive those inputs from the declared design,
not replay metadata. `AssignmentSpec` contains only deterministic replay inputs
(the salt, bucket allocation, hash and encoding, and eligible roster). It
cannot override the design's integrity declaration. Legacy schema documents
that embed integrity declarations are rejected with
`evidence.assignment.invalid`.
The public `increment.evidence.replay` helper returns a validated identity
ledger. Warehouse adapters apply experiment/version selection before reading
assignment rows; exposure joins return only period/arm tuples and a
next-period neighbour-arm lookup. Full and sampled replay checks are distinct;
without a replay declaration, the ledger is `not_certified`, not disproven.
Assignment evidence is needed for the versioned allocation and exposure work
tracked in #93 and is a prerequisite for follow-on assignment and network
readouts (#83, #91, #92, #78, #95).


## Experiment identity

An `Experiment` records a design, metrics, source references and views. Portable declarations preserve their canonical identity through `to_json()` and `from_json()`. A source reference may include a digest, column-name/type schema and versioned `EvidenceCapabilities`; a missing capability declaration means unknown, not supported.

Design mechanisms are a versioned set: an older reader refuses an unknown
mechanism with `design.mechanism_unsupported` rather than guessing its
semantics. New mechanisms are added in minor releases; existing mechanism
contracts are not silently reinterpreted.

Artifact readers retain `artifact.context.mismatch` for other invalid experiment
declarations, including boolean values in integer fields.

Raw learner factories use exact object identity while bound in memory, so
different closures, partials, or callable instances cannot be substituted for
one another. Their registration identity is process-local, not a portable
implementation/configuration identifier. Stable experiment hashes and portable
exports refuse raw factories even when `implementation_refs` records their
provenance. An `ImplementationRef` does not replace or reconstruct a raw
factory; experiments containing one remain in-memory-only. They still
deep-copy and pickle when the factory itself pickles, such as a module-level
function or class; unpickle only data you trust.


Every builder returns an experiment declaration and a reader for reading its
source. Saving and loading the declaration preserves
its hash; it does not serialize the data or a warehouse connection. Column types
recorded for a source are saved with the declaration (`SourceReference.schema`), and
`experiment.provides()` lists the recorded columns, but the types are not part of its
`hash()` or `analysis_spec_hash`. `hash(experiment)` agrees with
`==`, so equal experiments are one dictionary key; `experiment.hash()` can still
differ between declarations that compare equal, such as `Fraction(1, 2)` and
`0.5`.

### Declared views and saved request options

`run(experiment, reader)` returns one `RunResult` for all declared supported
views, sharing one physical evidence capture. `views=(view, ...)` selects
declared views by their hashes; execution follows declaration order, not
selection order. Empty, duplicate or undeclared selections refuse before reads.
`result.readout(view)` returns the typed collection each executed view
produced; see [reading results](../guides/reading-results.md).

`values=True` requests per-arm descriptive values for daily or as-of views.
Without it, those views request lift. Values and lift at the same time
coordinates are separate views; the default lift view keeps its existing
canonical identity. Overall, breakout and switchback views do not support values.
If an in-memory learner configuration cannot be serialized, `RunResult`
remains usable in memory, while storing or serializing it refuses with
`wire.procedure.runtime_method`.


Builder specifications retain the compiled plan, metric configurations,
cluster and trigger declarations, and the bound operations' accepted arguments.
Readout operations are selected through the declared views: daily values and
daily lift are separate views even when they share time coordinates, and have
different compatibility requirements. Preflight is conservative and sound rather than complete: it reports
known static refusals using the execution refusal code, but does not estimate
data-dependent conditions or guarantee that an unrefused request will execute.


The analysis specification identity includes the effective readout operation for
each view and saved request options, including priors and estimands. Omitting
an operation and explicitly selecting its default have the same specification
identity. These declarations also contribute to the execution identity; they
do not change evidence identities.

Preflight restores saved request options and checks known metric selection and
configuration overrides, including an explicit `prior=None` that clears a
declared prior. It cannot run every execution validator offline, so some
configuration checks may first appear during execution. Daily and as-of values
use descriptive validation, without restrictions that apply only to lift or
causal inference. A registered as-of lift projects the current finalized
checkpoint, so it requires sequential history rather than separately retained
as-of moments.

Estimator selection retains declared method settings such as conversion
inference and CUPED covariates where available; multicovariate CUPED requires
its covariate configuration. Switchback views require a switchback experiment.
Switchback requests reject arm-only arguments and role or prior overrides with
the same checks as execution.
`capture` is an execution step, not a saved request option. To capture or adopt
a sequential checkpoint, call
`increment.workflows.capture_sequential(experiment, reader, ...)` with the
pair returned by a module builder, then run the experiment. Pass an `as_of`
horizon when required by the source. `run(experiment, reader,
capture=SequentialCaptureRequest(...))` instead takes the capture inside the
run's own shared evidence capture, before any view reads.
Specifications containing `capture` refuse with
`kernel.request.capture_not_supported`. Population is declared on a view
constructor, such as `View.daily(population=...)`; a request option named
`population` refuses with
`kernel.request.population_scope_only`, as does a `run(population=...)` that
differs from a selected view's population.


??? info "Complete sequential capture example"

    For a panel-backed sequential definition, capture through the builder's pair
    before running. Supply an `as_of` horizon when required by the source:
    
    ```python
    import polars as pl
    
    import increment
    from increment.workflows import capture_sequential
    
    panel = pl.DataFrame(
        {
            "user_id": [f"u{index}" for index in range(8)],
            "variant": ["control"] * 4 + ["treatment"] * 4,
            "day": [1] * 8,
            "enrolled_on": [1] * 8,
            "revenue": [float(index + 1) for index in range(8)],
        }
    )
    experiment, reader = increment.from_frame(
        panel,
        shape="panel",
        unit="user_id",
        group="variant",
        date="day",
        exposure_date="enrolled_on",
        design=increment.Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics=[increment.MetricSpec(name="revenue", type="mean", window_days=1)],
        observation_end=2,
        plan=increment.AnalysisPlan(
            primary="revenue",
            inference=increment.InferenceSpec(kind="asymptotic_mean"),
        ),
    )
    capture_sequential(experiment, reader, finalized=True, as_of=1)
    result = increment.run(experiment, reader)
    ```

### Offline preflight

`Experiment.compile()` returns one run-wide `EvidenceScope` inventory and one `AnalysisRequest` for each selected view. Both contain the declared metrics, with any breakout or switchback view's metric selection applied. Each request retains its view, registration hash, and source-reference name.

`runnable(experiment, metric)` is an offline preflight. It does not open a source. If a source has no recorded capabilities, the returned estimator entries name those capabilities as `unknown`; known data-dependent conditions such as usable denominators and sample floors remain deferred until execution. Its `static_refusals` contain only known incompatibilities.

Pass a metric name declared for the experiment, or an eligible metric you can
add from its definitions with `run(..., exploratory_metrics=[name])`. A name
that is neither raises `facade.analysis_config.unknown_metric_declared` before
reading data. If a saved experiment lacks enough information to decide
whether it can add that name, the check reports it under `unknown` instead.

An unknown source declaration does not erase refusals already established by
another source. Alternative evidence routes include their count and named
covariate requirements. For switchback experiments, `switchback.contrast`
identifies the supported contrast procedure; arm estimators refuse with
`readout.contrast.override`.

### Reader matching and lifetime

`run(experiment, reader)` and every function that reads data accept only the
`ExperimentReader` a module builder returns with the experiment; anything else
refuses with `kernel.run.reader_required`, whose context names the type that
was passed. A custom evidence provider is verified, not run, with
`increment.evidence.BoundResolver`; see the
[provider guide](../development/adding-an-estimator.md#custom-evidence-providers).
`run` verifies the reader's source declarations before reading.

A closed reader stays closed: after `close()` or the end of a `with` block,
every read refuses with `kernel.run.reader_closed`; build a new reader with the
same builder to read again. A reader that `from_definitions(path)` restores from
the offline catalog cannot read and refuses with `kernel.run.reader_offline`;
build it with a connection to run. `repr(reader)` names the experiment, how the
reader reads and whether it is closed, without reading data.

The specification format describes declarations; it does not itself open a warehouse or infer a source snapshot. Reproducible execution still requires the reader and execution context to establish the actual snapshot identity.

An experiment's identity and its source digests cover declarations, not rows,
so readers built with identical declarations are interchangeable: `run`
accepts any of them with the experiment and reads whatever rows that reader
holds. A dataframe reader keeps the rows its frame held when the builder ran;
a warehouse reader reads live on every call.

A built-in reader executes selected overall, daily, as-of, breakout
or switchback views using each view's declared metric selection and
population. It validates every selected operation before opening the shared
capture. Existing source limitations still apply: combining views does not
create temporal, breakout, raw-outcome or clustering evidence.
Whole-window overall views, including the switchback contrast view, are evaluated
against the registered decision plan once per result. Daily, as-of, breakout,
compliance-only and exploratory-metric cells retain their own view and role and
receive `not_evaluated` verdicts. Cells, effects and any live confidence-set rows
stay aligned in one flat result with one run ID. Changed designs, decision
plans, metric definitions or unsupported view coordinates refuse with
`kernel.run.execution_unavailable`, before reading evidence. When the experiment
is not the one the reader was built for, its `field` context names the first
declaration that differs (`experiment_id`, `design`, `registration`,
`analysis_context` or `metrics`) and `reader_experiment_id` names the reader's
experiment. Source names, URIs, recorded digests, and capabilities must identify
the bound source; otherwise the read refuses with `kernel.provider.digest_mismatch`
or `kernel.provider.capability_mismatch`.
`run` accepts readout options as keywords:
`decision_method`, `sensitivity_methods`, `prior`, `metrics`, `estimands`,
`value_scale`, `exploratory_metrics`, `population` and `coherence`. Only the
options you pass apply, so `prior=None` removes the registered prior while an
omitted `prior` keeps it. A `population` must match every selected view;
declare a different population with the matching view constructor, such as
`View.daily(population=...)`.
`result.readout(view)` returns the `LiftEstimates`, `ContrastResults` or
other typed collection one view produced.

Use the reader as a context manager when it owns temporary warehouse tables or
an opened artifact snapshot. Exiting the block releases those resources; it
does not close the warehouse connection you supplied:

??? info "Complete warehouse reader-lifetime example"

    ```python
    from pathlib import Path
    from tempfile import TemporaryDirectory
    
    import ibis
    import pyarrow as pa
    
    import increment
    
    rows = [
        {
            "user_id": f"u{index}",
            "group_id": "control" if index < 4 else "treatment",
            "experiment_id": "demo",
            "ts": f"2025-01-01 12:{event:02}:00",
            "event": kind,
            "revenue": float(index + 1) if kind == "purchase" else None,
        }
        for index in range(8)
        for event, kind in ((0, "page_view"), (1, "purchase"))
    ]
    con = ibis.duckdb.connect()
    con.create_table("demo_events", pa.Table.from_pylist(rows))
    
    with TemporaryDirectory() as definitions:
        Path(definitions, "demo.yaml").write_text(
            """\
    dialect: duckdb
    fact_sources:
      - name: demo_events
        sql: SELECT * FROM demo_events
        timestamp_column: ts
        entities: [user_id]
        facts:
          - {name: page_view, column: null}
          - {name: purchase, column: revenue}
    exposures:
      - {name: enrolled, fact: page_view}
    metrics:
      - {name: revenue, type: mean, entity: user_id, fact: purchase, aggregation: sum, window_days: 1}
    experiments:
      - name: demo
        exposure: enrolled
        unit: user_id
        start: 2025-01-01
        end: 2025-01-01
        observation_end: 2025-01-02
        control_group: control
        plan: {primary: revenue}
    """,
            encoding="utf-8",
        )
        experiment, reader = increment.from_definitions(
            definitions, con, experiment_name="demo", store="always"
        )
        with reader:
            result = increment.run(experiment, reader)
    assert result
    assert con.raw_sql("SELECT 1").fetchone()[0] == 1
    con.disconnect()
    ```

### Builder options and offline catalogs

`from_frame` requires `shape="summary"` or `shape="panel"`. Each shape has
its own typed options. A summary accepts `intervention_grain="unit"` or
`"cluster"`; a panel requires `date` and accepts `observation_end`. Both accept
`unit`, `group`, `metrics`, `control` or `design`, `plan`, `uptake`,
`exposure_date`, `breakouts`, `cluster`, and `on_unassigned`.
`control` and `design` are alternative ways to declare the comparison.

The definitions call is path-first:
`from_definitions(definitions_path, con=None, *, experiment_name=None,
backend=None, store="auto", on_mixed_assignment="error",
source_snapshot_evidence=None, catalog_path=None)`. Omit `experiment_name` only
when the definitions or offline catalog contain exactly one experiment.

By default, binding with a connection writes `.increment-kernel-catalog.json`
inside a definitions directory, or `<file>.kernel.json` next to a single
definitions file. For read-only definitions, set `catalog_path` to a writable
file location and pass the same path when you reopen without a connection.
Relative paths are relative to your working directory. Use a new file or a
catalog previously written by Increment, not a YAML input or a symbolic link.
The generated `.lock` file stays beside the catalog to serialize updates.
`store="none"` does not disable this catalog.

Reopening refuses when the catalog is missing or the definitions have changed.
You can inspect the offline experiment with `runnable`, but need a connected
reader to execute it.

The other source builders expose only the options their data support:

<!-- skip: next "requires a published artifact, exported moments, and a switchback frame" -->
```python
import increment

artifact_experiment, artifact_reader = increment.from_unit_day_artifact(
    artifact_store,
    artifact_ref,
    expected_context=artifact_context,
    verification="lazy_digest",
)
moments_experiment, moments_reader = increment.from_moments(
    moment_rows,
    metrics={"revenue": "mean"},
    control="control",
    experiment_id="checkout_redesign",
)
switchback_experiment, switchback_reader = increment.from_switchback_frame(
    switchback_rows,
    unit="user_id",
    cycle="cycle",
    period="period",
    step="step",
    group="variant",
    metrics={"revenue": "mean"},
    identification=randomized_design,
    assignment=switchback_assignment,
    contrast_references=contrast_references,
)
```

Connected definitions builders record dtypes lazily from the selected input
relations during the first execution, then refresh their offline catalog.
Recording them does not change the experiment's identity: the connected and
offline experiments share one `hash()` and `analysis_spec_hash` before and after
that first execution.
Construction performs no additional schema query. The definitions builder takes
the path first and the experiment name as the `experiment_name=` keyword.
Catalog refreshes retain the definitions digest from the original bind. The
digest covers the parsed definitions, so comment and whitespace edits do not
change it. After any other change to those files, `run(experiment, reader)`
and every function that reads data through the reader refuse with
`kernel.definitions.catalog_stale` before reading anything; rebind the
definitions with a connection to continue.
Catalog updates lock the complete read-merge-replace operation, so concurrent
bindings or schema refreshes retain each experiment's entry.

The catalog records each loader's `declaration_identity()` digest alongside its
schema and capabilities. A connected reader recomputes that declaration digest
before every read and refuses a changed or unavailable recorded digest with
`kernel.provider.digest_mismatch`. It does not hash live rows: snapshot and
content digests belong to the runtime evidence, not the saved specification.

### Shared captures and exploratory metrics

Definitions-backed `run(experiment, reader, exploratory_metrics=...)` also shares one
physical capture across declared and exploratory families. Each family retains
the horizon of its original standalone declaration, including unselected
declared metrics needed to determine a running calendar-day horizon. Such
capture-only dependencies do not become authorized readout metrics. Request-
and evidence-bound `source_snapshot_id` values remain distinct when appropriate;
the shared physical capture is recorded separately in `metadata.captures`.
No source snapshot hash or wire-format version changes.

## Frozen-capture construction coverage

The specification round-trip tests enumerate the frozen capture's own
scenario/constructor inventory, including its as-of variants. In-memory definitions
are written through the public definitions constructor; explicitly registered
segment families are retained in the serialized inference declaration rather
than replaced with an automatic, unsegmented registration.

An absent constructor is not a failed round-trip: it is not declared
for that captured scenario. A constructor refusal also produces no `Experiment`.
Both outcomes are recorded with their reason or refusal code. Among the
analysis-construction scenarios, `edge-nonfinite-ratio-denominator` has no
successful captured constructor: its unit-summary input refuses with
`source.frame.non_finite`, and the other constructor routes are absent.

The following frozen scenarios are isolated refusal probes, not analysis
constructions. They supply invalid arm statistics, compatibility requests or
contrast objects directly to the owning statistical function; there is no
builder-produced specification to serialize:

- `allocation.alpha.underflow`
- `arm.adjustment.cluster_cuped`, `arm.adjustment.cluster_prior`,
  `arm.adjustment.sequential_prior`
- `arm.baseline.cluster_size_required`, `arm.baseline.compliance_undeclared`,
  `arm.baseline.iid_cluster_knobs`
- `arm.inference.cluster`
- `arm.metric.quantile_cluster`, `arm.metric.quantile_cuped`,
  `arm.metric.quantile_sequential`
- `compatibility.support_unsupported`
- `contrast.decision`, `contrast.inference`, `contrast.metric`
- `estimation.engine.arm.duplicate_rows`, `estimation.engine.arm.invalid_count`
- `estimation.engine.carries_format_moments`,
  `estimation.engine.carries_unsupported_weighted`
- `estimation.engine.cluster.denominator`, `estimation.engine.control_group_found`
- `estimation.engine.cuped_linear_cluster`,
  `estimation.engine.cuped_linear_encouragement`,
  `estimation.engine.cuped_linear_observational`,
  `estimation.engine.cuped_linear_sequential`,
  `estimation.engine.cuped_linear_unit_evidence`,
  `estimation.engine.cuped_linear_winsorization`
- `estimation.engine.group_summary_missing`, `estimation.engine.group_summary_row`
- `estimation.engine.method_name_observational`,
  `estimation.engine.method_names_unique`, `estimation.engine.metric_found_group`
- `estimation.engine.ratio.cuped`, `estimation.engine.ratio.negative_variance`
- `estimation.engine.winsorization_metadata_differs`,
  `estimation.engine.winsorization_metadata_present`
- `estimation.engine.x_role_declared`

The remaining non-construction probes exercise planning-only admission
coordinates, rather than experiment declarations:

- `compatibility.denominator`: only the private conservative-ratio helper accepts an arbitrary denominator; public plans derive it from validated inputs.
- `compatibility.multiplier`: only that private helper accepts an arbitrary multiplier; public plans derive it from validated inputs.
- `arm.adjustment.sequential_cuped`: the public power planner rejects always-valid inference before this admission point and admits CUPED under asymptotic inference.
- `arm.baseline.absorption_undeclared`: the public sample-size planner derives factor absorption from nonzero baseline ICC before admission.
- `arm.baseline.triggering_undeclared`: it derives the triggered population from the baseline trigger rate before admission.
- `arm.baseline.cuped_undeclared`: it derives the CUPED method from the baseline correlation before admission.
- `arm.metric.unsupported`: validated planning-procedure capabilities exclude the probe's unsupported metric shape before admission.
