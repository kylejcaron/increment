# Retained state for result-envelope pilot

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
collection as it is produced by `Analysis.run()`, including its rows and
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

Save `ReadoutResults` JSON to preserve and reopen the full result collection.
Use `ResultCell` or `AnalysisEnvelope` for portable per-cell summaries,
available operations, and Arrow interchange. The long-term choice of saved
result format is tracked in kata issue `cade`.

The envelope adapter covers fixed-horizon mean, ratio and CUPED lift rows and
asymptotic sequential lift rows. Only operations listed for a cell can be
applied after reopening it. The format does not represent every analysis row
type, reconstruct source data or fit objects, or add margin comparisons and
family combination operations.

See the [API reference's result format guidance](../api.md#portable-result-cells)
for the user-facing format choice.
