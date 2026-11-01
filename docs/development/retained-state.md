# Retained state for result envelopes

The 3a adapter is `LiftEstimate.as_cells()`; `LiftEstimates.as_cells()` stamps one aware timestamp for the run and passes its existing `ReadoutMetadata` through unchanged. Supported cell states retain the serialized `LiftEstimate`, including nested inference sets, relative and additive references, status fields, and sequential checkpoints. Confidence sets that retain raw outcomes are not serialized: those cells preserve identity and an unavailable reason without row state. Supported retained rows are restored with `LiftEstimate.model_validate`; no live fit object is needed.

| Pilot row | Classifier route | Retained construction state | Supported sampling operations |
|---|---|---|---|
| Fixed-horizon mean and ratio | `wald` or `conversion` when persisted relative statistics exist; `binomial` for exact conversion sets; `joint` for relative confidence sets | Complete serialized `LiftEstimate`; relative log statistics and absolute sidecar remain separate | Portable `relevel(cell, alpha=...)` supports retained `t` intervals only when the requested alpha is no greater than the stored interval's alpha; it can preserve or widen an interval, but cannot narrow it. Directional intervals with an open endpoint are not relevelable. `select_exploratory_family()` only selects eligible uncorrected rows |
| CUPED | Same `wald`/`conversion` construction as its fixed-horizon lift | Complete serialized row, including fitted inference answer already present on the result. No live fit handle is retained. Capturing `MethodFitAnswer` or `PreparedMulticovariateFit` awaits the operation that needs those fit internals. | Same row-level operations as the matching fixed-horizon construction |
| Asymptotic sequential lift | Sampling classifier describes the retained row, while the sequential component refers to its `AsymptoticSequentialResult` within that row | Complete serialized row, including checkpoint, bounds, decision alpha, ceiling, and validity regime | None in 3a; alpha is registration-bound and step 8 must declare any procedure operations |
| Unsupported sampling construction | `classify()`'s `None` tag and exact reason | Complete serialized row when it contains no raw outcomes; raw-outcome confidence sets retain identity and reason but no row state | None; a requested operation refuses with `results.operation.unsupported` and the classifier reason |

Posterior state is an independent component backed by posterior sufficient statistics on the same retained row; no 3a posterior operation is declared. `compare_margin` and `combine_family` are not supported in 3a.

Normal-reference Wald rows are not re-leveled in 3a: the portable `relevel()` operation uses the existing `LiftEstimate.reintervalize()` constructor, which rebuilds persisted `t` rows (or a winsor confidence set), but the portable cell operation is deliberately conservative and refuses requests above the retained interval alpha, which would narrow the interval. Direct row-level `LiftEstimate.reintervalize()` retains its existing reissue behavior. A new normal-reference relevel constructor would change estimator behavior. Winsor confidence sets, quantile rows, and independent-mean references are not in the portable relevel pilot slice.

## Saving results

Use the collection's `model_dump_json()` when you need to save and reopen the
collection as `result.readout(scope)` returns it, including its rows and
collection-level metadata.

Use `ResultCell` or `AnalysisEnvelope` when you need portable per-cell
identities, inference status, supported interval operations, and summaries.
`ResultCell.to_payload()` / `from_payload()` provide the JSON-compatible form;
`ResultCell.to_arrow()` / `from_arrow()` provide Arrow tables.
`AnalysisEnvelope.to_payload()` / `from_payload()` and `to_arrow()` /
`from_arrow()` also preserve the readout metadata that describes the source
scope. Supported retained rows serialize their `LiftEstimate` state; cells
whose confidence sets retain raw outcomes preserve identity and the unavailable
reason without serializing the raw data. Neither format retains live analysis
or fit objects.

Choose the saved representation by lifecycle. `ReadoutResults` JSON preserves
the typed result collection and collection metadata. `ResultCell` and
`AnalysisEnvelope` provide portable per-cell projections, supported operations,
rendering, and Arrow interchange. `ResultsStore` persists a `Run` with its cells,
execution identity, decision-plan binding, and manifest metadata in Arrow-backed
storage. The collection, interchange envelope, and persisted execution record
are complementary representations, not competing choices for one format.

The version-2 envelope adds a tagged report-only row payload for result classes
that are not `LiftEstimate` rows. These payloads are not deserialized as
`LiftEstimate` and have no portable operations unless a construction-specific
adapter retains the required state. Version-1 JSON and Arrow result cells
continue to decode as `lift_estimate`.

| Public result row or collection | Retained state and identity | Portable operation ruling |
|---|---|---|
| `BreakoutEstimate` / `BreakoutEstimates` | Full typed row, including segment identity, population, source, raw and prior-shrunken result fields, relative/additive references, confidence-set geometry, counts, exclusion/unavailable status, multiplicity and posterior fields. | Report-only; no lift cast or relevel inference. Segment identity remains distinct from total rows. |
| `DailyLiftEstimate` / `DailyLiftEstimates` | Full typed row, including `ds`, `ds_basis`, segment identity, source, set geometry, unavailable reason and sequential state. Daily and as-of rows have distinct row-kind and scope identities. | Construction-specific margin comparison and fixed-family evidence adaptation use retained state; no lift cast or relevel inference. |
| `DailyMetricValue` / `DailyMetricValues` | Full typed row, including absolute arm mean or its exact unavailable reason, `ds`, `ds_basis`, segment and source. Daily and as-of values have distinct row-kind and scope identities. Both identity and summary explicitly use absolute scale. | Report-only; arm-level absolute values are not treatment contrasts. |
| `CateResult` | Versioned report-only payload contains only fields in its public serialized form. The producing facade captures metric, control, treatment, scope and canonical source identity; an explicit source snapshot ID is retained when available. Omitted fit state is not reconstructed. | No fit-dependent operations. `cate`, `contrast` and `score` remain on the live result. |
| Honest CATE / targeting outputs (`GroupEffect`, `CateValidation`, `TargetingRule`, `TargetingSelection`) | Typed report payload retains held-out group effects, gate/qualification, selected fraction and policy readout with numeric-null reasons. The selection estimand follows the outer targeting rule population; the inner selection population stays in the report payload. Withheld policy values remain null. | Report-only; no replay or policy mutation. |
| `ConfoundingSensitivity` | Typed payload preserves the observed ratio, derived bounds and exact interval-null reason. The cell summary projects the ratio and both bounds to relative lift; table display formats those summaries as percentages. | Report-only; sensitivity is not a second sampling interval. |
| `ContrastResult` / `ContrastResults` | Typed payload preserves contrast identity, estimate, descriptive or t-reference variance, null SE/dof reasons, realized order counts, status, refusal context, advisories and collection metadata. Its identity and summary use absolute scale. | Descriptive `switchback_unit_variance_envelope` does not support sampling operations; t-based methods remain construction-specific and are not granted lift operations. |

Report-only cells can be reopened and passed through the same table and renderer entry points as their live rows; the renderer rehydrates typed report details rather than interpreting report payloads as `LiftEstimate`. Collection concatenation refuses incompatible daily/as-of view or completion-policy scopes with a coded error.

For opaque CATE, targeting, and sensitivity report kinds, core does not import
the implementation model to rebuild the summary. The serialized `EffectEstimate`
is the authoritative portable summary projection used for summary rendering;
the retained report payload remains available for full typed report display, but
its internal values are not re-derived or cross-validated against that
projection. Standard, daily/as-of, and switchback rows have typed core adapters,
so their restored
summaries are projected from the retained typed row.

Posterior mixture ordering and zero-weight components are serialized in model
order; tuple-valued state is represented as canonical JSON arrays. Unordered
advisories and operation sets are sorted. Runtime-only handles are excluded.
Duplicate row identities are not collapsed: each input row remains a cell.

See the [API reference's result format guidance](../api.md#portable-result-cells)
for the user-facing format choice.
