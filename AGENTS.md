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
- Affected selection reuses Tach plus filesystem guards; config,
  dependency/plugin, conftest/shared-fixture, data, and public lazy-export
  hazards refuse with a broader route. On source changes it also retains
  detected dynamic-import consumer files and reports any not run by the
  current tier/selector; changed files that themselves dynamically load
  arbitrary paths remain refusals. A zero-selection run is rejected, and
  successful acceptance requires owned finished evidence whose terminal
  node IDs exactly match the selected node IDs. Affected-fast does not waive
  slow, parity, fixture, public-smoke, docs, scientific, or full-suite gates.
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
