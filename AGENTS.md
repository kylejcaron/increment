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
  use `@pytest.mark.parameter_recovery`. Full simulation calibration grids live
  in `calibration/` and run before releases or when the covered inference
  changes; the slow tier retains representative sentinels plus a small fast
  smoke.
- Tests assert consumer-visible behavior: stable codes, numerical contracts,
  tolerances, invariants, and cross-path parity. They MUST NOT pin permitted
  message wording, private structure, internal wiring, or exact bits where a
  tolerance defines the contract.
- Serialization replay tests MUST run claimed consumers on the restored
  object, not the original.
- Start with the exact regression:
  `make test TESTS=tests/file.py::test_name`.
- Then run `make test-affected BASE=<task-base-sha>` (default fast, serial);
  use `TIER=slow` or numeric xdist (`-n 1..8`, with `--dist loadgroup`)
  when relevant. The affected runner accepts only typed selectors/controls:
  `-k`, `-m`, `-x`/`--maxfail`, `-q`, `-v`, `--durations`, numeric `-n`,
  `--dist loadgroup`, `--evidence-root`, and `--runtime-diagnostics`.
  Arbitrary pytest arguments, alternate configs, addopts/plugins environment,
  response files, automatic/logical workers, and custom xdist transports are
  refused. `PYTEST_XDIST_AUTO_NUM_WORKERS` does not override the numeric policy.
  Resolve and record the task base at dispatch, reuse it across commits/resumes,
  and use `BASE=HEAD` only for a working-tree-only comparison.
- Normal test tiers run testmon in selection-disabled mode, recording per-test
  executed code into the ignored `.testmondata` map. Supervised pytest children
  carry an explicit owner-child role; nested normal runners bypass Testmon
  reads/writes and certification. Without a valid owner token, a nested normal
  runner skips Testmon entirely and does not acquire its database lock; only
  reusing a live same-worktree lock requires the verified owner token.
- Only an unfiltered, fully executed fast/slow/full tier writes an ignored
  completeness marker; file/node selectors, sharding, help, inherited
  `PYTEST_ADDOPTS`, and collection-only runs cannot certify it. Full runs
  refresh the `.testmondata` per-test execution map and a companion marker bound
  to the Testmon database, exact Python/dependency environment, and tracked
  source snapshots. They record each tier's collected node IDs, with
  `loadgroup` suffixes when that scheduler is active, import-time executed lines
  gathered in a fresh interpreter (module imports and full test collection under
  coverage), plus AST fingerprints for module/class-level state. A fast or slow
  run certifies
  only the tier it executes; an `all` run certifies both. Testmon's xdist
  synchronization may discard tests from the other tier, so a tier's marker
  is retained only while every recorded node ID remains present in
  `.testmondata`. Affected runs preserve the marker and never certify a tier.
  The marker also checks resolved pytest
  config/selection controls and requires terminal reports for every collected
  test node ID. Affected startup pins Testmon to the worktree's `.testmondata`
  and validates the marker before Testmon opens and mutates its database.
  Testmon is primary only when the map covers the requested tier and changed
  function bodies. On that primary path, import-graph consumers of changed test
  modules are protected from Testmon deselection. Changes to import-executed lines
  or any function containing them fall back to the import graph. Module/class-level
  changes use the same fallback. Testmon's xdist synchronization is supported;
  affected collection restores longest-first duration ordering after Testmon's
  collection hook. Missing, new, unreadable/corrupt, stale, or incomplete maps
  fall back to the conservative local AST import graph. That graph follows
  file-level imports transitively through source modules and tests, including
  consumers of changed test helper modules; it resolves public lazy exports
  from the package export map and retains the full test set if any graph file
  is unparseable. Changed test files, dynamic-import consumers, subprocess test
  files, and filesystem guard tests remain selected in either mode. Config,
  dependency/plugin, conftest/shared-fixture, data, and public lazy-export
  hazards still refuse with a broader route; changed files that themselves
  dynamically load arbitrary paths remain refusals. A zero-selection run is
  rejected, and successful acceptance requires owned finished evidence whose
  terminal node IDs exactly match the selected node IDs. Affected evidence
  records the selector route as `testmon` or `import_graph`. Affected-fast
  does not waive slow, parity, fixture, public-smoke, docs, scientific, or
  full-suite gates.
- Dispatch/resume notes record base SHA, commands, numeric/serial worker
  allowance, remaining acceptance, and evidence location. Selection evidence
  is tied to source, worktree/environment, and dirty inputs; changed inputs
  require rerunning it.
- Run `make install` once per clone. It installs the prek pre-commit (Ruff,
  ty, prose audit) and pre-push hooks; prek chains an existing roborev
  pre-push hook as `pre-push.legacy`. Do not bypass them with `--no-verify`.
- Run `make check` before pushing. Reserve `make test-all` for final integration
  or release candidates that change estimation or a wire format.
- Use `make fmt` for formatting. Build documentation with `nox -s docs`, not
  bare MkDocs; the Nox session exports the notebook-backed pages first.

## Parallel work and handoffs

- Record the user-selected integration branch, absolute worktree path, and
  source SHA before dispatch. Every worker uses its assigned absolute path;
  a session's working directory is not evidence of the intended branch.
- Ignored planning notes, contracts, and temporary measurement scripts remain
  local. Do not force-add them to obtain commit-based reviews. Ship product
  documentation and behavioral verification, not execution scaffolding.
- Review a design contract at its agreed gate, resolve remaining decisions,
  then verify the implementation. Do not restart a design-only review loop
  for every clarification or treat a contract as an implemented capability.
- Give shared result schemas and serialization boundaries one integration
  writer. Consumers use a concrete, exercised implementation handoff, not
  independently invented carriers based on the same design document.
- Coordinate aggregate test workers across the host. Use focused checks
  during implementation; one coordinator owns broad integration checks.
  Docs-only clarifications do not justify repeated broad suites. Ordinary
  focused and xdist runs may overlap. Reserve an exclusive window only for
  CPU/RSS timing measurements; correctness oracles run under load, with any
  elapsed time labeled contended.
- Integrate a lane as soon as its focused checks and review pass, then run
  the combined focused selection on the integration branch. Do not hold
  verified work on side branches; late merges hide cross-lane failures.
  The integration gate before each commit includes the `slow`-marked tests
  of every changed test file, not only the default selection.
- Where roborev is configured for the checkout, run
  `roborev fix --list --branch <branch>` before closing a tracked issue and
  after each integration commit. Resolve or triage open findings before
  starting the next task. Close a review only with cited code or test
  evidence for every finding; "a later review found nothing" is not evidence.
  Contributors without roborev are not required to install it.
- Update the issue ledger when work state changes (commit, merge,
  verification, blocker), not in end-of-session batches.
- When several writers share one worktree, assign disjoint file ownership,
  route other hunks through the owner, and keep every module importable
  after each edit.
- A handoff names exact commits, dirty and untracked files, verification
  evidence, unresolved decisions, and the next action. Distinguish
  implemented, verified, integrated, and issue-closed work. Preserve partial
  edits and stop old workers before a new coordinator takes ownership.

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
## Review outcomes, not structure

- Review analysis changes by asking whether an ordinary analyst can reach and
  correctly interpret the result for the affected user journey. Use the
  eleven journey contracts and evidence inventory in `docs/validation.md`;
  structural proxies such as constructor presence, row counts, or a passing
  primitive test do not establish an outcome contract.
- For scoped results, check decision completeness: every required cell is
  present as a result or an identity-preserving unavailable row with its
  reason. A partial collection MUST NOT look like a complete answer.
- Keep sampling evidence and posterior summaries distinct. A prior MUST NOT
  change sampling inference; posterior-dependent decisions require an actually
  available posterior and its recorded reason when unavailable.
- Check automatic assignment integrity in `Analysis.run()` at the assigned
  population and declared assignment grain. `NotApplicable` is not a pass;
  it means the integrity question could not be evaluated under the source or
  assignment contract. Triggered counts are not assignment-law evidence.
- Check multiplicity status and family scope on the result, not only the
  displayed estimate. In particular, `exploratory_unadjusted` explicitly
  means no family adjustment, and filtering must not shrink the retained
  family.
- Treat `assigned` and `triggered` as distinct populations and family scopes.
  A triggered estimate is anchored at each unit's first eligible trigger; it
  does not validate random assignment or establish that conditioning on
  triggering is causally valid.
- Reuse `tests/readout_journeys.py` assertions where applicable and verify the
  public result a user consumes. Do not replace semantic checks with wording,
  private-wiring, or type/name-count assertions.

## Import organization and user-facing language

- Keep authoring and analysis configuration distinct from receive-only result
  consumption in documentation and examples. Show the intended import path for
  each role; do not move root exports, add compatibility aliases, or create
  parallel namespaces without an approved API cutover.
- Use one term for each public concept across README, guides, API reference, and
  release notes. Keep statistical status labels and population names exact
  (`exploratory_unadjusted`, `assigned`, `triggered`); explain them in ordinary
  analyst language rather than inventing synonyms.
- Update the existing release-note mechanism when one exists. Do not introduce
  a new changelog or claim an API/release change for documentation-only
  clarification.


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
- Assignment counts state their grain and population. Unassigned and
  mixed-assignment accounting buckets are not experiment arms; keep them out
  of arm rosters while preserving their audit counts.
- A scoped collection represents every requested cell, either as a result or
  as an unavailable row that keeps the cell's identity fields.
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
