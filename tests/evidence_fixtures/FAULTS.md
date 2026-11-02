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

## Producer inventory

The wrapper inventory is `capture.PRODUCERS`. Aliases already imported by readouts
are wrapped too; aliases imported during `run` are restored on exit. Nested
acquisitions are represented by the enclosing terminal envelope rather than
recording the same observations twice. Lazy day-axis slices are recorded as they
are consumed, without materializing them outside the owning source lifetime.

| Readout / evidence | Producers covered by the wrapper | Reduced implementations underneath |
| --- | --- | --- |
| Run, scalar CUPED, quantile, percentile winsorization | `_metric_rows.load_metric_evidence` covers `_load_metric_evidence` and `_load_percentile_winsor_evidence`, all seven run `MetricEvidence` construction branches | `_evidence_loaders.load_moments`, `load_unit_rows`; dataframe / warehouse / artifact / moments |
| Breakout | `_metric_rows._load_breakout_metric_rows` (the eighth `_metric_rows` constructor); `_evidence.load_breakout_evidence`, `load_breakout_method_moments` | `BreakoutSlices`, `BreakoutSlice`, `BreakoutUnitRows`; `frame.FramePanelSource.breakout_moments_for_method`, `sources.BreakoutMomentsSource.breakout_moments_for_method` produce `ScalarBreakoutRows` |
| Daily and as-of values/lifts | `_evidence.load_day_axis_rows` covers `_load_day_axis_rows` and its custom-provider branch; `_asof._asof_metric_rows` | `DayAxisRows` / `DayAxisSlice` from `_NativeDayAxis`, `_ArtifactDayAxis`, `_MomentsDayAxis`; lower `load_builtin_day_moments` delegates back to this boundary |
| Multicovariate CUPED | `_evidence.load_method_fit`; `estimation.cuped.prepare.prepare_multicovariate_fit` | `MethodFitAnswer`, `PreparedMulticovariateFit`, `warehouse_cuped_fit_evidence` / `_WarehouseCupedFitEvidence`; retained categorical cells or fitted/local inputs |
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


### Executed fault proofs

All twenty replacements below were executed separately in the detached disposable
worktree `/tmp/increment-evidence-faults`, based on
`25f819138e4b82367748b42bb5e66b7ea505fa4a`. Each replacement changes exactly one
production line; that line was restored before the next execution. Every final
fault failed at an **evidence comparison or capture-roster assertion**, not merely
an import failure, timeout, estimator refusal or readout difference. The pristine
boundary-sensitive scenario passed before the replacements (`1 passed in 8.73s`)
and after restoration (`1 passed in 6.02s`).

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
and multicovariate-fit boundaries. The four tests added here execute fixed-horizon
mean/conversion/ratio and two scalar-CUPED companions; the inventory is not a
claim that every other readout scenario was executed here. Existing parity and
end-to-end scenarios remain in place. No tests or assertions were deleted.

### Duplicate producer output

A separate production fault duplicated the acquired `MetricEvidence.rows` at
`increment/readouts/_metric_rows.py:201`:

```diff
-    return _load_metric_evidence(src, request, _snapshot_active=scope_active)
+    return __import__("dataclasses").replace(evidence := _load_metric_evidence(src, request, _snapshot_active=scope_active), rows=evidence.rows + evidence.rows)
```

The same public-run evidence test failed **inside capture**, before estimation:
`AssertionError: Duplicate capture key` for revenue/control (`1 failed in 4.08s`).
The original producer was restored. The restored scenario was executed again;
its result is recorded with the verification evidence.
