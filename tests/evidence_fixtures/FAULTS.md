# Acquired-evidence fixtures

Fixtures compare the typed values delivered to estimators, after acquisition has
selected the dataframe, warehouse, artifact or moments implementation. They do
not replace the independent reducers. `test_frame_window_parity.py` is unchanged.

## Schema 1

`capture_acquired_evidence` runs public `run` and requires a nonempty, explicit
`frozenset[CaptureKey]`. It checks exact roster equality: missing and unexpected
keys fail, and inserting a duplicate fails immediately. `assert_same_evidence`
rejects empty records. `to_payload()` emits JSON-compatible values under
`fixture_version: 1`, sorting records by their complete key. Nulls are retained;
nonfinite numbers have an explicit tagged representation. Neither source digests
nor source-kind labels stand in for actual numerical evidence.

The key is `(estimator_id, method_identity, metric, scope_hash,
request_coordinates, arm)`. Method identity hashes the **full** declaration,
including name, covariates and their missingness policies. Request coordinates
contain population, grain, day and segment; segment null and the string
`"__null__"` remain distinct. Scenario and source names belong outside the
record, so identity scenarios can store these versioned records without changing
their key schema. Auxiliary acquisitions use their producer/request kind as the
estimator identifier and empty method/metric/arm fields only where not applicable.

For unit-randomized experiments, assignment and unit counts that establish the
same arm roster share `counts.arm_roster`; their response population is retained
and their statistical grain is `unit`. Cluster-grain counts are not folded into
that normalization. Source-qualified breakout dimensions retain their qualifier
in segment coordinates. Acquisitions shared before several views execute are
recorded under each consuming scope.

Day-axis segment coordinates use each consumed `DayAxisSlice.source_name`, not
the request's or row's source context, so equal property/day/value rows from two
fact sources remain distinct. Method-fit records retain the full
`MethodFitRequest` and numerical `envelope`, `cells`, or `failure`; source-varying
`origin`, `fit_identity`, and `evidence_sha256` are excluded from parity. The
full method declaration remains in the key and the full request remains in the
payload, so changing fitted numerical values still fails comparison.

## Frozen input and readout ownership

The compressed operations captures define the complete scenario inventory.
Every declared source success, refusal and absence remains covered by acquired
input parity. The smaller public-journey catalogue does not reduce that
inventory or the retained population and coherence checks.

Pure readout consumers receive closed data, effective calculation and identity
plans, declared context, rosters and requests. They do not receive a Source,
provider, capabilities or acquisition callback. Published rows and complete
metadata retain every archived key. Current snapshot and family identifiers are
reconstructed from their complete authentic input preimages before historical
identifiers are rebound in a comparison copy. Canonical sorting of opaque family
identifiers after rebinding does not reorder rows, days, checkpoints or source
components.

The archived nonempty daily domains have four owners: 31 ordinary or encouragement
breakouts, 16 sequential breakouts, 24 daily/as-of coordinates and one contrast.
Three categorical-CUPED coordinates have no archived segment estimates. Their
additional segment scopes exercise current source-input identity, fit/frame
association and metadata declarations only; they are not extra numerical gold.
Their archived main-run estimates remain covered by the fixed readout consumer.

Ordinary breakout inputs retain 27 directly materialized slice paths and seven
genuine derived summary-source paths. The latter do not relabel the root's
unsupported typed response. Every materialized scope has its actual pinned
source context, request and normalized caller route. Missing required input
receipts fail rather than reducing the eligible domain.

Warehouse fit-record components are independently rebuilt from the genuine
rehydrated fit, its complete field inventory and real relative/ratio projections.
Projection bounds are recomputed through the Student-t inverse, so an archived
component commits to the capture platform's rounding of them, not to decoded
data. Archived components are rehashed with their witnessed historical bounds
(`breakout_component_witnesses.json`, keyed by fixture, receipt, entry, kind,
metric, arm and alpha). A witness has authority only because that rehash equals
the protected archived digest; the locally recomputed bounds must agree with the
witnessed ones at the original parity contract, and the witness inventory must
match the archived components exactly. Live components are rehashed on their own
runtime. Neither checks an independent implementation of projection mathematics.
Local frame components retain their original columns and values and are hashed
after the actual declared dimension-alias transformation.

Bootstrap-t root series are owned by replicate index: count, order, `None`
positions and failure indices are exact, every root keeps the original parity
contract, and every root must agree on the consumer scale `point - root * se`
(`expm1` for the log-relative series) within the consumer interval bound. Pivot
error is absolute, so a relative rule on a nearly centred root rejects platform
rounding no consumer can observe.

Warehouse raw-relation fingerprints are a distinct protocol from fitted-wire
hashes and fit-record components. Their receipts retain ordered backend schema,
the exact two backend aggregate sum strings and row count. An additional bounded
acquisition inside the same pinned source lifetime supplies these preimages;
it is not an interception of the original aggregate execution. Each frozen and
current fingerprint is independently rehashed before comparison. No column
sorting, float conversion of decimal strings or admitted-arm filtering is
applied. The backend's hash implementation itself is not independently proven.
Retained categorical and fitted payload hashes still use their original strings.

Original source frame/table/state/manifest receipts without stored raw bytes are
historical provenance, not independently rehashed frozen byte oracles. Current
original-byte hashes are checked before semantic keying. Consumed numeric and
categorical values retain their independent references and unchanged bounds.
Current source payloads use one canonical byte stream: sorted-key, compact JSON
with ASCII escaping, strict nonfinite handling and UTF-8 encoding. Population
rows are fetched or converted in batches of at most 1,024; complete populations
are never materialized as Python row graphs or JSON buffers just for a receipt.
Observation, row and column order and numeric/null spellings are preserved.
Claim and audit SHA-256 states are updated separately from the owned chunks;
neither calculation copies the other's digest or hash state. The detached audit
tuple is compared with emitted receipt claims before projection. Literal byte
and empty/partial-batch assertions own the shared encoder and array framing;
a truncated-claim preimage fault must fail the retained input comparison.
The narrow post-capture alpha and absolute-contrast serialization projections
operate only on comparison copies; actual result objects and archived records
remain unmodified.


Population membership identity is captured from the source’s actual declared population and metric roster. For artifact sources, the test-only observer uses a private source-owned membership projection over the same eligible unit reduction and pinned snapshot, returning only unit and arm IDs. It skips analyst censoring-warning bookkeeping only for this identity observation; public `unit_frame` reads retain their warning behavior. Membership deduplication, canonical encoding and hashing remain in the capture helper, and this observer adds no public evidence capability or request schema.

## Producer inventory

The wrapper inventory is `capture.PRODUCERS`. Aliases already imported by readouts
are wrapped too; aliases imported during `run` are restored on exit. Nested
acquisitions are represented by the enclosing terminal envelope rather than
recording the same observations twice. Lazy day-axis slices are recorded as they
are consumed, without materializing them outside the owning source lifetime.
`prepare_multicovariate_fit` is a pure consumer of the acquired method-fit answers,
not an acquisition boundary: its FCR re-estimation passes share one captured
`MethodFitAnswer` inside `MetricEvidence`.

| Readout / evidence | Producers covered by the wrapper | Reduced implementations underneath |
| --- | --- | --- |
| Run, scalar CUPED, quantile, percentile winsorization | `_metric_rows.load_metric_evidence` covers `_load_metric_evidence` and `_load_percentile_winsor_evidence`, all seven run `MetricEvidence` construction branches | `_evidence_loaders.load_moments`, `load_unit_rows`; dataframe / warehouse / artifact / moments |
| Breakout | `_metric_rows._load_breakout_metric_rows` (the eighth `_metric_rows` constructor); `_evidence.load_breakout_evidence`, `load_breakout_method_moments` | `BreakoutSlices`, `BreakoutSlice`, `BreakoutUnitRows`; `frame.FramePanelSource.breakout_moments_for_method`, `sources.BreakoutMomentsSource.breakout_moments_for_method` produce `ScalarBreakoutRows` |
| Daily and as-of values/lifts | `_evidence.load_day_axis_rows` covers `_load_day_axis_rows` and its custom-provider branch; `_asof._asof_metric_rows` | `DayAxisRows` / `DayAxisSlice` from `_NativeDayAxis`, `_ArtifactDayAxis`, `_MomentsDayAxis`; lower `load_builtin_day_moments` delegates back to this boundary |
| Multicovariate CUPED | `_evidence.load_method_fit`; the `prepare_multicovariate_fit` consumer selects a captured answer | `MethodFitAnswer`, `PreparedMulticovariateFit`, `warehouse_cuped_fit_evidence` / `_WarehouseCupedFitEvidence`; retained categorical cells or fitted/local inputs |
| Sequential run, daily/as-of and breakout | `_evidence.load_sequential_capture`, `current_source_snapshot` | `loader_snapshot`, `retained_sequential_capture`, `retained_sequential_snapshot` delegate to those boundaries; source-owned `SequentialCapture` snapshots |
| Switchback | `_evidence_loaders.load_contrast_stats`; `_evidence.load_assignment` | `acquire_contrast_stats`; assignment results from `switchback`, `query.native_source`, `query.artifact_reader`, or the custom provider |
| Arm roster / integrity | `_evidence.load_counts`, `load_assignment` | `CountEvidence`, `AssignmentEvidence`; nested assignment counts are retained inside the assignment envelope |
| Encouragement | `_evidence.load_compliance`, `load_compliance_series` | `ComplianceSummary` with `ComplianceArm`, including source-owned series |
| Standalone aggregate/unit acquisitions | `_evidence_loaders.load_moments`, `load_unit_rows` | `MomentRows`, `UnitRows` aliases; suppressed only when the enclosing envelope already contains them |

`increment._evidence_types` also declares request objects, source/view lifetime
objects (`EvidenceScope`, `EvidenceScopeResult`), source construction state
(`SourceContext`), stream/provenance identity (`EvidenceStream`,
`ArtifactProvenance`) and breakout selectors (`BreakoutIdentity`, `BreakoutCell`).
They are not additional observations delivered to an estimator: requests supply
capture coordinates; view/source objects are never serialized; numerical
acquisitions under them are captured at the boundaries above. Constructors used
before `run` are covered through their returned acquisition envelope, not by
patching dataclass constructors. No estimator implementation registry is used.

## Loader fault catalogue

The shared detector is
`tests/test_evidence_level_parity.py::test_mean_and_conversion_evidence_matches_across_sources`.
It executes the same line-one mean/conversion/ratio scenario through summary and
panel `from_frame`, DuckDB `from_definitions`, `from_unit_day_artifact`, and
`from_moments`. Its expected roster has six metric/arm inputs and two assignment
counts. The scenario's denominators, exact successes and optional null moments
are compared, not just the final estimate.


### Original loader fault proofs

The measurements below record the original fault campaign at
`25f819138e4b82367748b42bb5e66b7ea505fa4a`. Each replacement changed exactly one
production line, restored before the next execution. Every fault failed at an
evidence comparison or capture-roster assertion, not an import failure, timeout,
estimator refusal or final readout difference. These elapsed times are historical
observations, not performance thresholds.

Command for each row (substitute its ID for `<id>`):

```sh
uv run --no-sync python -m scripts.test_slots -- make test \
  TESTS=tests/test_evidence_level_parity.py::test_mean_and_conversion_evidence_matches_across_sources \
  PYTEST_ARGS="--tb=short -p no:tach --junitxml=/tmp/te-t10-<id>.xml"
```

| ID | One-line location | Observed detector | Result |
| --- | --- | --- | --- |
| `frame-dropped-arm-proof` | `increment/_frame_moments.py:684` | capture roster: missing revenue/treatment | `1 failed in 8.08s`; make exit 2 |
| `frame-day-window-proof` | `increment/_frame_moments.py:1515` | moments.ref_y: 4.666666666666667 versus 104.66666666666667 | `1 failed in 7.85s`; make exit 2 |
| `frame-wrong-denominator-proof` | `increment/_frame_moments.py:684` | moments.n: 60 versus 61 | `1 failed in 7.71s`; make exit 2 |
| `frame-lost-null-proof` | `increment/_frame_moments.py:684` | moments.successes: null versus 0 | `1 failed in 7.82s`; make exit 2 |
| `frame-swapped-population-proof` | `increment/_frame_moments.py:636` | moments.cy2: 733.3333333333336 versus 80.0 | `1 failed in 11.48s`; make exit 2 |
| `warehouse-dropped-arm-proof` | `increment/query/builders.py:2190` | capture roster: missing revenue/treatment | `1 failed in 4.57s`; make exit 2 |
| `warehouse-day-window-proof` | `increment/query/builders.py:1517` | moments.ref_y: 4.666666666666667 versus 104.66666666666667 | `1 failed in 8.23s`; make exit 2 |
| `warehouse-wrong-denominator-proof` | `increment/query/builders.py:2190` | moments.n: 60 versus 61 | `1 failed in 7.38s`; make exit 2 |
| `warehouse-lost-null-proof` | `increment/query/builders.py:2190` | moments.successes: null versus 0 | `1 failed in 7.10s`; make exit 2 |
| `warehouse-swapped-population-proof` | `increment/query/builders.py:2190` | moments.cy2: 733.3333333333336 versus 80.0 | `1 failed in 7.45s`; make exit 2 |
| `artifact-dropped-arm-proof` | `increment/query/artifact_reader.py:1443` | capture roster: missing revenue/treatment | `1 failed in 6.06s`; make exit 2 |
| `artifact-day-window-proof` | `increment/query/artifact_reader.py:1640` | moments.ref_y: 4.666666666666667 versus 104.66666666666667 | `1 failed in 7.46s`; make exit 2 |
| `artifact-wrong-denominator-proof` | `increment/query/artifact_reader.py:1443` | moments.n: 60 versus 61 | `1 failed in 7.64s`; make exit 2 |
| `artifact-lost-null-proof` | `increment/query/artifact_reader.py:1443` | moments.successes: null versus 0 | `1 failed in 7.96s`; make exit 2 |
| `artifact-swapped-population-proof` | `increment/query/artifact_reader.py:1819` | moments.cy2: 733.3333333333336 versus 80.0 | `1 failed in 6.63s`; make exit 2 |
| `moments-dropped-arm-proof` | `increment/sources.py:1834` | capture roster: missing revenue/treatment | `1 failed in 6.88s`; make exit 2 |
| `moments-day-window-proof` | `increment/sources.py:1834` | capture roster: unexpected day=2025-01-11 instead of the total-window coordinate | `1 failed in 6.33s`; make exit 2 |
| `moments-wrong-denominator-proof` | `increment/sources.py:1834` | moments.n: 60 versus 61 | `1 failed in 6.83s`; make exit 2 |
| `moments-lost-null-proof` | `increment/sources.py:1834` | moments.successes: null versus 0 | `1 failed in 6.30s`; make exit 2 |
| `moments-swapped-population-proof` | `increment/sources.py:1834` | moments.n: 60 versus 40 (converter count substituted for assigned count) | `1 failed in 6.29s`; make exit 2 |

The three computing loaders extend the half-open window by one day, admitting
nonzero purchases exactly at the original excluded boundary. Moments ingress
cannot recompute windows from unit events: its temporal fault instead assigns
the next-day window coordinate to carried total rows, caught as an unexpected
key. This is a replay-identity proof, not a claim of an independent moments
window reducer. The population faults select only purchasers for revenue on the
computing loaders; moments ingress substitutes the converter-population count
for the assigned count. The requested population identity stays assigned, so
comparing the actual evidence, rather than only that label, is essential.

Exact replacements (original line followed by its faulty replacement):

#### frame-dropped-arm-proof

`increment/_frame_moments.py:684`

```diff
-    return rows
+    return [dict(row) for row in rows if row["group_id"] == "control"]
```

#### frame-day-window-proof

`increment/_frame_moments.py:1515`

```diff
-            kept.filter((nw.col(day_index) >= 0) & (nw.col(day_index) < right_edge))
+            kept.filter((nw.col(day_index) >= 0) & (nw.col(day_index) < right_edge + 1))
```

#### frame-wrong-denominator-proof

`increment/_frame_moments.py:684`

```diff
-    return rows
+    return [dict(row, n=row["n"] + (row["metric"] == "revenue")) for row in rows]
```

#### frame-lost-null-proof

`increment/_frame_moments.py:684`

```diff
-    return rows
+    return [dict(row, successes=0 if row.get("successes") is None else row["successes"]) for row in rows]
```

#### frame-swapped-population-proof

`increment/_frame_moments.py:636`

```diff
-        long = frame.select(
+        long = frame.filter((nw.col(spec.y_column) > 0) | nw.lit(spec.name != "revenue")).select(
```

#### warehouse-dropped-arm-proof

`increment/query/builders.py:2190`

```diff
-    return emit_centered_moments(totals, UNIT_GRAIN, keys=group_cols, columns=columns)
+    return emit_centered_moments(totals, UNIT_GRAIN, keys=group_cols, columns=columns).filter(lambda t: t.group_id == "control")
```

#### warehouse-day-window-proof

`increment/query/builders.py:1517`

```diff
-            window_end = fe_date + _days_literal(window_days).as_interval("D")
+            window_end = fe_date + _days_literal(window_days + 1).as_interval("D")
```

#### warehouse-wrong-denominator-proof

`increment/query/builders.py:2190`

```diff
-    return emit_centered_moments(totals, UNIT_GRAIN, keys=group_cols, columns=columns)
+    return emit_centered_moments(totals, UNIT_GRAIN, keys=group_cols, columns=columns).mutate(n=lambda t: t.n + (t.metric == "revenue").cast("int64"))
```

#### warehouse-lost-null-proof

`increment/query/builders.py:2190`

```diff
-    return emit_centered_moments(totals, UNIT_GRAIN, keys=group_cols, columns=columns)
+    return emit_centered_moments(totals, UNIT_GRAIN, keys=group_cols, columns=columns).mutate(successes=lambda t: t.successes.fill_null(0))
```

#### warehouse-swapped-population-proof

`increment/query/builders.py:2190`

```diff
-    return emit_centered_moments(totals, UNIT_GRAIN, keys=group_cols, columns=columns)
+    return emit_centered_moments(totals.filter((totals.metric != "revenue") | (totals.y > 0)), UNIT_GRAIN, keys=group_cols, columns=columns)
```

#### artifact-dropped-arm-proof

`increment/query/artifact_reader.py:1443`

```diff
-        return _rows(self._snapshot.execute(query))
+        return [dict(row) for row in _rows(self._snapshot.execute(query)) if row["group_id"] == "control"]
```

#### artifact-day-window-proof

`increment/query/artifact_reader.py:1640`

```diff
-        metric = self._normalize_metric(metric)
+        metric = self._normalize_metric(metric).model_copy(update={"window_days": 2})
```

#### artifact-wrong-denominator-proof

`increment/query/artifact_reader.py:1443`

```diff
-        return _rows(self._snapshot.execute(query))
+        return [dict(row, n=row["n"] + (row["metric"] == "revenue")) for row in _rows(self._snapshot.execute(query))]
```

#### artifact-lost-null-proof

`increment/query/artifact_reader.py:1443`

```diff
-        return _rows(self._snapshot.execute(query))
+        return [dict(row, successes=0 if row.get("successes") is None else row["successes"]) for row in _rows(self._snapshot.execute(query))]
```

#### artifact-swapped-population-proof

`increment/query/artifact_reader.py:1819`

```diff
-            totals = winsorize_unit_totals(totals, metric)
+            totals = winsorize_unit_totals(totals, metric).filter((totals.metric != "revenue") | (totals.y > 0))
```

#### moments-dropped-arm-proof

`increment/sources.py:1834`

```diff
-        return [dict(r) for r in self._rows if r["metric"] == metric.name]
+        return [dict(r) for r in self._rows if r["metric"] == metric.name and r["group_id"] == "control"]
```

#### moments-day-window-proof

`increment/sources.py:1834`

```diff
-        return [dict(r) for r in self._rows if r["metric"] == metric.name]
+        return [dict(r, ds="2025-01-11") for r in self._rows if r["metric"] == metric.name]
```

#### moments-wrong-denominator-proof

`increment/sources.py:1834`

```diff
-        return [dict(r) for r in self._rows if r["metric"] == metric.name]
+        return [dict(r, n=r["n"] + (metric.name == "revenue")) for r in self._rows if r["metric"] == metric.name]
```

#### moments-lost-null-proof

`increment/sources.py:1834`

```diff
-        return [dict(r) for r in self._rows if r["metric"] == metric.name]
+        return [dict(r, successes=0 if r.get("successes") is None else r["successes"]) for r in self._rows if r["metric"] == metric.name]
```

#### moments-swapped-population-proof

`increment/sources.py:1834`

```diff
-        return [dict(r) for r in self._rows if r["metric"] == metric.name]
+        return [dict(r, n=next(other["successes"] for other in self._rows if other["metric"] == "purchase_rate" and other["group_id"] == r["group_id"]) if metric.name == "revenue" else r["n"]) for r in self._rows if r["metric"] == metric.name]
```

### Fixture guard proofs

Four additional one-line fixture faults were executed and restored:

| Fault | Test | Observed failure |
| --- | --- | --- |
| Omit covariates from the full Method hash | `test_cuped_companions_with_different_covariates_capture_separately` | same-name methods with covariates `a` and `b` acquired the same hash; assertion failed |
| Remove duplicate-key rejection | `test_duplicate_capture_key_raises` | `DID NOT RAISE AssertionError` |
| Allow comparison of two empty records | `test_capture_roster_mismatch_fails` | `DID NOT RAISE AssertionError` |
| Skip per-key numerical comparison | `test_cuped_companions_with_different_covariates_capture_separately` | changing either companion independently no longer raised; `DID NOT RAISE AssertionError` |

SHA-256 before and after all production faults was identical for each loader:

```text
170228bf426583d9a594a59fe65080ebcb81be79ef3ffd42000b77995cb8443c  increment/_frame_moments.py
a830721cafc3268450ede86a27efe6c6f0df91a01311ca50cc3de6aa08bf26d5  increment/query/builders.py
ce54d1465e7a667ad5ef5b731b5cc30d2b4caf4a2f73483d72b61e4dc068e780  increment/query/artifact_reader.py
9ec34a66d3dea4c8b3d6210ea1b61695bfbecac1a83fc19571d2f9810c73917c  increment/sources.py
```

The producer inventory includes daily, as-of, sequential, breakout, switchback
and multicovariate-fit boundaries. The shared loader detector acquires typed
evidence directly, without running a successful public readout. Frozen-input
consumers own statistical and assembly fields; the public journey catalogue owns
source-coupled completion, identity, reopening and forwarding contracts.

### Duplicate producer output

A separate production fault duplicated the acquired `MetricEvidence.rows` at
`increment/readouts/_metric_rows.py:201`:

```diff
-    return _load_metric_evidence(src, request, _snapshot_active=scope_active)
+    return __import__("dataclasses").replace(evidence := _load_metric_evidence(src, request, _snapshot_active=scope_active), rows=evidence.rows + evidence.rows)
```

The original public-run detector failed inside capture, before estimation:
`AssertionError: Duplicate capture key` for revenue/control (`1 failed in 4.08s`).
The retained typed-acquisition detector preserves duplicate-key rejection.
Each fault proof requires a separate pristine execution after restoration.

## Public-journey fault proofs

At `d8bcdc4a2c1a9c676171ae18a6fc86cffcfe1660`, four temporary production
faults exercised all 16 retained public witnesses. Each failed its intended
public contract, then passed in a separate pristine execution after byte-exact
restoration. No fixture, expected result, assertion, tolerance or marker changed.
All executions used `INCREMENT_REQUIRE_ALL_REGISTERED=1 make test TESTS=...`
with fail-fast disabled.

| Fault | Production boundary | Observed failure | Pristine result |
| --- | --- | --- | --- |
| A: captured evidence mislabeled live | `load_evidence_scope` capture status | All 13 standard catalogue entries failed their captured-coherence assertion: 13 failed, 0 errors, 15.68s | Same 13 entries passed, 15.40s |
| B: captured switchback evidence mislabeled live | `_SwitchbackScopeProvider.scope` capture status | The switchback catalogue entry failed its captured-coherence assertion: 1 failed, 0 errors, 2.84s | Same entry passed, 2.58s |
| C: skip reopening a closed artifact | `ResolverBinding.resolve` artifact-open guard | The second public read after `resolver.close()` failed with a missing snapshot-table `CatalogException`: 1 failed, 0 errors, 3.51s | Same adoption/reopen witness passed, 3.44s |
| D: forward the wrong population | `request_evidence` primitive dispatch, replacing the dataclass population with `triggered` | Native results diverged from the forwarding provider's coded refusal at the existing parity assertion: 1 failed, 0 errors, 2.97s | Same forwarding witness passed, 6.68s |

The exact catalogue coordinates exercised by A and B were:

| Scenario | Source route | Fault |
| --- | --- | --- |
| `mean_ratio_conversion_cuped_fixed_horizon` | `from_definitions` | A |
| `conversion_route_straddles_the_dense_threshold` | `from_unit_summary` | A |
| `mean_ratio_conversion_cuped_fixed_horizon` | `from_moments` | A |
| `mixed-cuped-multicovariate-fixed-horizon` | `from_unit_day_artifact` | A |
| `quantile_tied_rounded_outcomes` | `from_unit_panel` | A |
| `sequential-mixed-quantile-catalog-checkpoint` | `from_unit_summary` | A |
| `triggered_daily` | `from_definitions` | A |
| `triggered_asof` | `from_unit_day_artifact` | A |
| `triggered-sitewide-population-results` | `from_definitions` | A |
| `triggered_encouragement_compliance_asof` | `from_unit_day_artifact` | A |
| `breakout_store_dimension` | `from_definitions` | A |
| `clustered_assignment_integrity_scope` | `from_unit_day_artifact` | A |
| `multiplicity_primary_secondaries_guardrail` | `from_definitions` | A |
| `audit-switchback-neighboring-integers` | `from_switchback_panel` | B |

C used `test_public_artifact_adoption_constructs_and_reopens`; its failure was
the real second-read lifecycle error, not a startup failure or an asserted
error expectation. D used `test_forwarding_provider_boundary_cell[run]` and
reached the result/refusal parity assertion; the request remained a real typed
dataclass and no fabricated provider was substituted.
