<!-- BEGIN KATA (managed by `kata init --with-agents`) -->
## kata issue tracker

This project uses [kata](https://github.com/kenn-io/kata) as its shared issue
ledger. Run `kata quickstart` at the start of each session for the full agent
contract. The short version:

- Search before creating: `kata search "<keywords>" --agent`.
- Prefer updating existing issues over duplicates (`kata comment`, `kata label add`, `kata edit`).
- Default to `--agent` for ordinary reads and mutations; use `--json` only when a script needs structured data.
- Close only verified work: `kata close <ref> --done --message "<scope + verification>" --commit <sha>`.
- If work is incomplete, label `needs-review` and comment what remains rather than closing.
- Never `kata delete` or `kata purge` without explicit user authorization.

## kata work.* conventions (agent orchestration)

When working a kata-tracked issue, keep its `work.*` metadata truthful:

- On claim/start: `kata meta set <ref> work.attention ok`; if the work has a
  dedicated branch, stamp it once with `kata meta set <ref> work.branch <branch>`.
- Signal live state: `kata meta set <ref> work.attention stuck|needs-human|ok`
  plus a one-line `work.attention_msg` saying why. Raise `stuck` when you cannot
  proceed, `needs-human` when you want review; clear back to `ok` when unblocked.
- Never stop with the signal stale: close the issue, or leave the attention
  pair reflecting the hand-off.
- Coordinators read `work.*` on issues they delegated; only the working agent
  writes them. `work.*` on closed issues is meaningless.
<!-- END KATA -->

# Contributor guidelines

## Architecture

- `increment/semantics/models.py` contains Pydantic definitions only and MUST
  NOT import Ibis or SQLGlot. `increment/semantics/loader.py` MAY import SQLGlot
  lazily to validate user SQL; other semantics modules MUST NOT.
- `increment/query/` owns warehouse query construction and Ibis integration.
- `increment/estimation/` consumes additive moments or per-unit frames and MUST
  NOT construct warehouse queries.
- `increment/frame.py` owns the Narwhals dataframe entry points and MUST NOT
  import Ibis or `increment.query`.
- `increment/power/` owns sample-size, power, and MDE solvers; it MUST NOT
  access warehouses.
- `calibration/` contains certification campaigns and intentionally sits
  outside pytest `testpaths`. A bare `pytest` MUST NOT collect it.

## Tests and validation

- Unit tests MUST NOT use the network or a real warehouse. Live warehouse
  checks belong under `integration/warehouse_execution/`.
- Tests MUST be independent under `pytest-xdist`. A test MUST NOT rely on
  execution order or state created by another test. Modules that mutate a
  shared filesystem path MUST use `pytest.mark.xdist_group("<name>")`.
- Tests slower than the ordinary suite use `@pytest.mark.slow`.
  Simulation-based coverage, bias, or parameter-recovery checks additionally
  use `@pytest.mark.parameter_recovery` and retain a small fast smoke case.
- Tests assert consumer-visible behavior: stable codes, numerical contracts,
  tolerances, invariants, and cross-path parity. They MUST NOT pin permitted
  message wording, private structure, internal wiring, or exact bits where a
  tolerance defines the contract.
- Start with focused validation:
  `make test TESTS=tests/file.py::test_name`.
- Run `make check` before pushing. Reserve `make test-all` for final
  integration or release candidates that change estimation or a wire format.
- Use `make fmt` for formatting. Build documentation with `nox -s docs`, not
  bare MkDocs; the Nox session exports the notebook-backed pages first.

## Analysis ingress and parity

The public analysis ingress surface is:

- `Analysis.from_definitions`
- `Analysis.from_unit_day_artifact`
- `Analysis.from_unit_summary`
- `Analysis.from_unit_panel`
- `Analysis.from_switchback_panel`
- `Analysis.from_moments`

A capability claim MUST name its supported ingress paths. Identical per-unit
data through supported paths MUST produce equivalent emitted rows, estimates,
intervals, and retained sequential state. Add or update the enumerated parity
harness for every capability; path presence alone is not evidence of parity.

Where a path cannot support a capability, refuse explicitly and distinguish a
source that cannot supply required input from an estimator or construction
that is unfinished. Keep the path-by-capability table in
`docs/reference/capabilities-by-entry-point.md` aligned with measured behavior.

## Method compatibility and limitations

- A new method MUST address its interaction with CUPED, ratio metrics,
  clustering, sequential inference, winsorization, breakouts, and
  multiplicity roles. Classify each applicable combination as supported,
  source-limited, unfinished, or mathematically unsound.
- A refusal MUST cover only the variant its reason applies to and MUST name a
  route forward when one exists.
- Under sequential inference, a transform, threshold, or coefficient is
  admissible only when it is predictable when each observation arrives or a
  construction accounting for its estimation is established.
- Document a limitation only when it is inherent to the source, the available
  construction, or statistical validity. Missing implementation belongs in
  the issue tracker, not in the limitations page.

## Public API and refusals

- Public inputs describe experiments, metrics, and decisions in terms users
  can act on. Do not expose internal proof obligations or mathematical tuning
  controls merely because an implementation needs them.
- Derive defensible defaults when possible. Keep assumptions and the strength
  of guarantees explicit; ease of use MUST NOT depend on hiding either.
- Unsupported public requests use the designated `CodedError` subclass and
  the module's `RefusalSpec`, with a stable package-wide code and immutable,
  structured context, before loading data or calculating results.
- The same hazard MUST use the same refusal code across ingress paths.
  Refusals MUST survive pickle and deepcopy round trips.

## Numerical and data contracts

- When planning and runtime, or dataframe and query paths, derive the same
  quantity independently, add an equivalence test at matched inputs.
- For small probabilities, allocate family and tail error conservatively. Use
  survival or log-tail functions; do not recover a tail through
  `1 - alpha`.
- Aggregate calculations cover large offsets, neighboring floating-point
  values, input-order changes, merged partitions, and values near overflow.
  Prefer centered, mergeable moments and overflow-safe cross terms.
- When a result can be unavailable, preserve a numeric null and its exact
  reason through schemas, conversions, serialization, and saved fixtures.
- Portable or hashed formats define collection order, tuple encoding, alias
  behavior, duplicate handling, and treatment of runtime-only values.
- Read-only SQL is parsed using the dialect that will execute it and is
  rejected when read-only safety cannot be established.

## Complete changes

- Public API changes update production callers, exports and registries,
  docstrings, guides and examples, fixture builders, saved fixtures, tests,
  and type-checked callers.
- Removing a capability updates every surface that produces, plans, validates,
  serializes, or documents it in the same change: runtime dispatch, planners,
  wire formats, refusal registries, guides, limitations, fixtures, and tests.
  A surviving planner or fixture is evidence that the removal is incomplete.
- Comments and docstrings remain concise and evergreen. Preserve necessary
  derivations and assumptions; use `make audit-verbosity` to check them.
