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

Increment turns experiment data into decisions. Six public builder routes load
data (`from_frame(shape="summary")`, `from_frame(shape="panel")`,
`from_switchback_frame`, `from_definitions`, `from_unit_day_artifact`,
`from_moments`), each returning an experiment and its resolver; `run(experiment,
resolver)` executes them; one evidence layer answers typed requests; estimators
compute from data; readouts assemble results. A new method should touch an
estimator, the evidence request it needs, the parity matrix and a guide — not
every loader.

## Layers

| Layer | Owns | Must not |
|---|---|---|
| `increment/semantics/` | Pydantic declarations of experiments, metrics, designs, plans | import Ibis or SQLGlot (`loader.py` may import SQLGlot lazily to validate SQL) |
| `increment/query/` | Warehouse relations, SQL builders, Ibis sessions, artifacts | hold statistical logic |
| `increment/frame.py`, `increment/_frame_*.py` | Narwhals dataframe loaders and reducers | import Ibis or `increment.query` |
| Evidence layer: `increment/_evidence_types.py` (leaf request/result types), `increment/_source_types.py` (declarations and provider protocol), `increment/_evidence_refusals.py` (declaration validation), and acquisition in `increment/readouts/_evidence.py` plus `increment/_evidence_loaders.py` (when present) | Typed evidence requests, loader capability declarations, refusals, digests | compute estimates |
| `increment/estimation/` | Estimators and inference on data values | call a source, construct a query, or branch on loader kind |
| `increment/readouts/` (except `readouts/_evidence.py` and the lower `increment/_evidence_loaders.py` acquisition layer), `increment/breakout/`, `increment/_day_axis.py` | Assemble results from evidence | call source methods or probe a source with `isinstance`/`getattr`/`hasattr` |
| `increment/power/` | Sample size, power and MDE solvers | access a warehouse |
| `calibration/` | Certification campaigns | be collected by a bare `pytest` |

- Statistics code receives data (rows, moments, fitted evidence), never a source object. Evidence acquisition lives only in `increment/readouts/_evidence.py` and `increment/_evidence_loaders.py` when present. The public builders and `run` (`builders.py`, `_run.py`), the `workflows`, `query`, `artifacts` and `dashboard` functions, `cate.py::estimate_cate`, and the named estimation facades validate and delegate; they do not introduce a second acquisition path. Custom sources MUST declare `EvidenceCapabilities` or implement the public `EvidenceProvider` before reads. The boundary is enforced by `tests/test_evidence_boundary.py`.
- Each built-in loader declares what it can answer once through `evidence`; legacy plain attributes are private or derived from that declaration. Refusals are validated against declaration data before any read. `tests/test_evidence_capabilities.py` pins the declaration contract. One hazard, one code, on every constructor. Raise the registered `RefusalSpec`; `tests/test_refusal_uniqueness.py::test_each_evidence_hazard_has_one_code_on_every_constructor` enforces the public-constructor contract; never add a per-loader variant of an existing hazard.
- `tach check` enforces module dependencies; add one only where the table allows.

## Making a change

- Find the owner first: estimators in `estimation/`, loaders in `frame.py`,
  `query/` and `sources.py`, assembly in `readouts/`. Editing all three usually
  means a value is being threaded through layers; pass a typed object instead
  of adding parameters.
- A new method or design states the evidence it requests and classifies its
  interaction with CUPED, ratio metrics, clustering, sequential inference,
  winsorization, breakouts and multiplicity as supported, source-limited,
  construction-limited, not-expressible, unfinished or unsound (mathematically
  invalid, with its derivation) — the parity matrix's vocabulary
  (`tests/parity_harness/matrix.py`). The matrix and the recorded
  consumer contracts are the capability record;
  `docs/reference/capabilities-by-entry-point.md` and
  `docs/guides/compatibility.md` are rendered from them
  (`scripts/render_capability_table.py`, `scripts/render_compatibility.py`),
  never edited by hand.
- Under sequential inference a transform, threshold or coefficient is admissible
  only when it is fixed before each observation arrives or an established
  construction accounts for its estimation.
- Cutovers are clean: migrate every caller and delete the old path. A
  compatibility alias needs a removal date in the release notes.
- Removing a capability removes it everywhere it is produced, planned,
  validated, serialized or documented, in one change.

## Public API, outputs and refusals

- Public inputs describe experiments, metrics and decisions; they do not expose
  proof obligations or tuning knobs. Derive defaults; keep assumptions and the
  strength of guarantees explicit.
- An unsupported request refuses before data is read with a `CodedError`
  carrying a stable package-wide code and immutable structured context, and
  names a route forward when one exists. Refusals survive pickle and deepcopy.
- Every requested cell is a result or an unavailable row that keeps its
  identity and exact reason. `assigned` and `triggered` populations and
  multiplicity status stay distinct through filtering. Priors never change
  sampling inference.
- User-facing strings — refusal templates, reprs, warnings, README, guides —
  are written for an analyst who has not read the guides; internal vocabulary
  stays internal; a term is allowed only where a guide defines it. Root exports
  are listed with a one-line user task in `tests/public_exports.txt` (and public
  parameter spellings in `tests/public_parameters.txt` where that inventory
  exists); `tests/test_consumer_standard.py` enforces the inventories and the
  vocabulary check. A term a guide defines for users is listed with that
  guide in `tests/consumer_jargon.txt`; existing occurrences are counted per
  file in `tests/consumer_jargon_baseline.txt`, and those counts may fall but
  never grow. A root-export change is an API cutover recorded in
  release notes.

## Numbers

- Planning and runtime, or dataframe and query, derive the same quantity
  independently; an equivalence test at matched inputs pins them.
- Small probabilities use survival or log-tail functions; never recover a tail
  as `1 - alpha`.
- Aggregates use centered, mergeable moments with overflow-safe cross terms and
  are tested at large offsets, neighboring floats, reordered input and merged
  partitions. Outside a realistic range, refuse with a code.
- An unavailable number travels as a numeric null with its exact reason through
  schemas, serialization and fixtures.
- Portable or hashed formats define collection order, tuple encoding, aliases,
  duplicates and runtime-only values.
- Read-only SQL is parsed in the executing dialect and rejected when safety
  cannot be established.
- A change that can move a published number cites the pinned baseline
  (`tests/fixtures/parity_baseline.json`) and the contract it verifies.

## Tests

- A test exists to fail when the code is wrong. Assert contracts: codes,
  tolerances, invariants, parity. Never wording, private wiring or exact bits
  where a tolerance defines the contract.
- Test each behavior once, on plain data or one constructor. Cross-constructor
  equality is the parity matrix's job (`tests/test_parity_matrix.py`); add a
  scenario to `tests/parity_harness/cases.py` only for a shape the matrix axes
  cannot express.
- Markers: unmarked tests are unit tests (no DuckDB, processes or sockets;
  `tests/_unit_boundary.py` enforces this); `integration` marks component,
  query, UI and subprocess boundaries; `slow` marks expensive work;
  `parameter_recovery` and `numerical_certification` mark scientific checks and
  are always also `slow`; `examples` marks notebooks. A `parametrize` past
  twelve items becomes boundary cases plus a Hypothesis property or carries
  `@pytest.mark.grid("<reason>")`.
- Tests are independent under xdist; modules that touch a shared path use
  `pytest.mark.xdist_group("<name>")`. Unit tests never use the network or a
  live warehouse; live checks live under `integration/warehouse_execution/`.
- A new test is shown failing first (a one-line break in the code under test or
  a detached worktree at the base, never `git stash`). Expected values come
  from the specification, not from running the implementation.
- Commands: `make test TESTS=<nodeid>` while editing; `make test-affected
  BASE=<task-base-sha>` before committing; `make check` optionally; CI is the
  gate. Collected-item growth needs a `tests: +N items — reason` line in the
  PR body; removing or weakening tests needs the `test-reduction-approved`
  label. Mutation evidence comes
  from `scripts/mutation_campaign.py`, nightly or on demand, never per push.
- A failure under contention is rerun alone before it is called a flake
  (`uv run --no-sync python -m scripts.test_slots -- make test TESTS=<nodeid>`);
  never widen a tolerance, timeout or threshold to pass. When Hypothesis finds
  a failure, add its shrunk case as an `@example` before changing code; never
  lower `max_examples`.

## Documentation

- Generated pages are regenerated, never edited: the capability page and the
  compatibility reference. `nox -s docs` builds the site;
  `scripts/check_doc_references.py --check` verifies references.
- Guides explain what a result means and what was assumed. Limitations record
  only what is inherent to a source, a construction or statistical validity;
  missing implementation goes in the issue tracker.
- Comments and docstrings stay short and evergreen: keep derivations and
  assumptions, drop narration (`make audit-verbosity`). `make fmt` formats;
  `make lint` runs ruff, tach, pydoclint and vulture.

## Working with others

- Work in a worktree on a branch. Record its absolute path, base commit and
  verification commands in the issue ledger; use absolute paths in every
  command. Do not overwrite another writer's dirty work; one writer per shared
  schema or wire format.
- Every PR gets a human diff read. Statistical, numerical and wire-format
  changes get domain review; security-sensitive boundaries get a security
  review. Review tooling and handoff mechanics are in `CONTRIBUTING.md`.
- Commit messages, code and documentation describe the change in product
  terms.
