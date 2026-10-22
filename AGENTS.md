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

- Unit tests MUST NOT use the network or real warehouses. Live checks belong
  under `integration/warehouse_execution/`.
- Unmarked test categories default to cheap unit/small regression checks.
  Mark component/query/UI/subprocess boundaries `integration`; use `slow`
  for expensive work and `stress` plus `slow` for large deterministic grids.
  Integration tests remain in PR fast CI; the unit tier excludes them for
  local iteration. Legacy unmarked tests require source inspection, not an
  assumed unit label.
- Recovery, coverage and bias studies use `parameter_recovery` plus `slow`;
  deterministic inferential-certification families use
  `numerical_certification` plus `slow`. Notebook exports use `examples` plus
  `slow`. Collection enforces the cost mark, including inherited marks.
  Integration alone does not exclude a test.
- Review recorded setup/body/teardown costs above five seconds; prioritize
  above ten. Thresholds trigger investigation, not flaky timing failures.
  Keep assertions and scientific settings when splitting smoke/campaigns;
  exclusion requires an equivalent required execution gate. See
  [test categories and runtime expectations](CONTRIBUTING.md#test-categories-and-runtime-expectations).
- Tests MUST be independent under xdist. Modules that mutate shared paths use
  `pytest.mark.xdist_group("<name>")`; dependent Markdown snippets retain source
  order within their file.
- Slow tests use `@pytest.mark.slow`; simulation coverage, bias and recovery
  checks also use `@pytest.mark.parameter_recovery`, while deterministic
  inferential certification uses `@pytest.mark.numerical_certification`.
  Keep exact designs, seeds, repetitions, thresholds and tolerances in both
  classes; the required certification gate covers relevant changes, and the
  nightly runs both scientific categories. Full calibration grids live in
  `calibration/` and run before releases or covered-inference changes; retain
  representative slow sentinels and a fast smoke.
- Assert consumer-visible behavior: codes, numerical tolerances, boundaries,
  invariants and parity. Do not pin permitted wording, private wiring or exact
  bits where a tolerance defines the contract. Replay tests exercise the
  claimed consumers on the restored object.
- Load the `writing-tests` skill before adding or changing tests and
  `test-audit` before pruning. Each new test names the bug it catches and is
  shown failing first: a one-line break in the code under test or a detached
  worktree at the base, never `git stash`. Derive expected values from the
  spec, not from running the implementation.
- One test file per source module. Assert query and estimation behavior
  through one ingress path plus the parity harness, never once per ingress.
  Assert dashboard behavior on the prepared payload; render only when the
  behavior is the rendering.
- A `parametrize` past twelve items becomes boundary cases plus a Hypothesis
  property or carries `@pytest.mark.grid("<reason>")`. Do not add cases to
  `tests/test_parity_matrix.py` or
  `tests/estimation/test_sequential_screen_acceptance.py`.
- Mutation evidence comes from `scripts/mutation_campaign.py` on demand or
  nightly, never per push; see [test economy](CONTRIBUTING.md#test-economy).
- Start with `make test TESTS=tests/file.py::test_name`, then run
  `make test-affected BASE=<task-base-sha>` and include relevant slow cases.
  Resolve the base at dispatch and reuse it across commits/resumes;
  `BASE=HEAD` is only a working-tree comparison.
- Affected selection uses the changed-path rules in
  [CONTRIBUTING.md](CONTRIBUTING.md#tests-and-quality-checks). It is local
  feedback, may select no tests, and does not waive slow, parity, fixtures,
  public smokes, documentation, scientific or full-suite gates. CI is the
  authoritative PR gate; a required nightly covers the full test union.
  Numerical certification is selected from certified modules' transitive
  `tach.toml` dependencies, certification-test changes and CI/build config.
- Test targets use plain pytest marker expressions: fast excludes `slow`,
  `parameter_recovery` and `numerical_certification`; slow selects `slow`
  excluding those scientific markers and `examples`; scientific is the union
  of `parameter_recovery` and `numerical_certification` and runs nightly; all
  uses `slow or not slow`. Tags require successful `ci-ok` and a successful
  nightly run on the exact SHA unless the explicit `force` input is used. Each
  test has the configured 300-second timeout.
- The default local worker count is `max(1, CPU count // TEST_SLOTS, CPU count // held slots)`, measured after acquiring a test slot; CI passes
  `WORKERS=auto`. Tests MUST remain xdist-independent. The ignored
  `.test_durations` cache controls stable longest-first ordering.

### Pre-push checks

- The pre-push hook runs lint, type-checking, `test-weakening` (warning only),
  and `prepush-tests`, which runs changed unit tests through testmon after its
  first unit-tier seed and records successful tree/merge-base receipts to skip
  identical pushes. `make check` is an optional broader local check; CI is the
  authoritative PR gate.
- Pushes MAY use `git push --no-verify`; CI remains the required gate. CI treats
  test-weakening as approved only with the PR `test-reduction-approved` label;
  local checks continue to accept the `Test-Reduction-Approved:` commit trailer.
- Use `make fmt`; build docs with `nox -s docs`, not bare MkDocs.
- The unit boundary is enforced by `tests/_unit_boundary.py`: unit-selected tests cannot open DuckDB, start processes or open Internet sockets, and cannot request a `backend_fixture`; mark a real integration boundary or test a pure function on in-memory data.
- `scripts/test_slots.py` coordinates heavy Make/Nox runs; focused `make test TESTS=...` uses xdist under a slot for files/directories, while one node ID stays unwrapped and single-process. DuckDB defaults and Hypothesis profiles are configured in root `conftest.py`.
- Before/after: unexplained expected value `tests/estimation/test_sampling_posterior.py:120` → independent 100-digit reference comment and parametrized cases in `tests/estimation/test_tails.py:23-26`.
- Before/after: estimation assertion behind DuckDB `tests/test_cuped_identity.py::test_definitions_and_artifact_keep_decision_family_when_companion_changes` (`tests/covariate_cases.py:121`) → moments-route scalar-family evidence in `tests/test_cuped_route_evidence.py::test_single_cube_transports_carry_each_scalar_family`.
- Run `make test TESTS=<nodeid>` while editing; before commit run `make test-affected BASE=$BASE`.
- Before calling a contention failure a flake, rerun the single test alone with `uv run --no-sync python -m scripts.test_slots -- make test TESTS=<nodeid>`; never widen a tolerance, timeout or threshold to pass.
- When Hypothesis finds a failure, add its shrunk case as an `@example` before changing code; never lower `max_examples`.
- The weakening check reports fewer test functions or assertions (requiring the owner trailer), while the item ratchet reports more collected items or a literal grid over 12 without `grid` (requiring a stated reason); collapsing a grid to boundary cases plus a property lowers items and keeps functions, passing both.

## Coordination and review

- Record the integration branch, absolute worktree, immutable base, worker
  allowance, exact commands and remaining acceptance. Assign disjoint files;
  shared schemas and serialization have one writer. Consume exercised
  handoffs, not design-only carriers. Keep modules importable and preserve
  other writers' work. Delegated workers use absolute paths in their assigned
  worktree for every read, edit and command; relative tool paths resolve to the
  coordinator's worktree.
- One coordinator owns broad checks; focused suites may overlap within the
  host allowance. Timing measurements require an uncontended window.
- Integrate reviewed slices promptly. Include relevant slow tests of changed
  test files before integration; changed source or environment inputs require
  rerunning affected checks.
- Before any merge, inspect main and the integration worktree's status; do not
  overwrite unrelated dirty work. Each task's reviewer is independent of its
  writer and receives the brief, report and diff, not the controller conversation.
- Every PR requires a human diff read before merging. Where configured, pin
  Roborev's repo and branch; check its backlog before issue closure and after
  integration commits. Audit by job ID since the effort began (`roborev show
  --job N`); branch-filtered lists miss side-branch jobs. Run a branch-range
  review per task branch and for the PR range before merging. Resolve each
  finding with concrete code/test evidence, not a later clean review. See
  [review and handoffs](CONTRIBUTING.md#review-and-handoffs).
- Ignored plans and temporary probes stay local; never force-add scaffolding.
  Record decisions, verification and blockers in the issue ledger as they change.
- Handoffs identify commits, dirty/untracked files, exact verification commands,
  remaining decisions and next action. Distinguish implemented, verified,
  integrated and closed state. Stop old writers before transferring ownership.
  Preserve original scientific and human acceptance; a design review is not
  implementation verification.

### Recommended local verification and review

Keep Roborev per commit. This cadence does not relax required CI or scientific
acceptance; it does not authorize shared-hook changes.

| Boundary | Checks and tests | Review |
| --- | --- | --- |
| Commit | Existing format/lint/type hooks; relevant regressions while editing, not a broad suite merely because a commit is created. | Roborev per commit; resolve findings with concrete code/test results. |
| Task / AI task | Focused regressions, affected fast and relevant affected slow against the immutable task base, plus the changed public-behavior smoke. | One independent high-depth correctness review of the completed task; consume commit findings. |
| PR | Required CI full-suite and applicable docs/build, parity and scientific gates; pre-push runs lint, type-checking and the unit tests the change executed. `make check` is optional. | Human full-diff read and coherent branch-range correctness review; consume task and commit reports. |
| Stack / final integration | Same baseline on the frozen combined source, plus cross-PR journeys and applicable scientific campaigns or `make test-all`. | High-depth integration review of interactions and unreviewed resolutions; do not re-review every constituent PR from scratch. |

- A standalone PR is the final integration gate. Rerun affected checks when
  their source or environment inputs change.
- Escalate statistical, numerical and wire-format changes to domain review;
  add targeted security review for security-sensitive boundaries. A review does
  not replace required execution checks.


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

## Consumer outcomes

- Verify the public outcome using the eleven journeys in `docs/validation.md`
  and `tests/readout_journeys.py`; constructor presence or row counts alone
  are not acceptance.
- Every requested cell is a result or identity-preserving unavailable row with
  its exact reason. A partial collection MUST NOT look complete.
- Priors MUST NOT change sampling inference. Posterior-dependent decisions
  require an available posterior or its exact unavailable reason.
- Automatic assignment integrity is evaluated in `Analysis.run()` on the
  assigned population and declared grain. `NotApplicable` is not a pass.
  Unassigned/mixed audit buckets are not arms; triggered counts are not
  assignment-law evidence.
- Preserve multiplicity status and retained family through filtering.
  `exploratory_unadjusted` means no family adjustment.
- Keep `assigned` and `triggered` populations/families distinct. Triggered
  outcomes anchor at first eligible trigger; conditioning does not establish
  causal validity or validate random assignment.


## Import organization and user-facing language

- Every root export from `increment` MUST serve a clear user task and have a
  one-line justification in `tests/public_exports.txt`. Root exports may be
  moved, consolidated, or removed through an explicit API cutover recorded in
  release notes; compatibility aliases are allowed only when those notes state
  a removal date.
- A reader who has not read the guides is the audience for every user-facing
  string: refusal and error templates, representations, display output,
  warnings, README, and guides MUST make sense to an analyst on first use.
  Internal design vocabulary (for example, ingress, seam, retained family,
  roster completeness, and proof obligation) MUST stay out of those surfaces
  unless a guide defines the term for users.
- Keep one term for each public concept across README, guides, API reference,
  and release notes. Preserve exact statistical labels and population names
  (`exploratory_unadjusted`, `assigned`, `triggered`), explaining them in
  ordinary analyst language rather than inventing synonyms.
- The root-export and user-facing vocabulary checks live in
  `tests/test_consumer_standard.py`. Add a root-export justification only for
  a name that serves a user task; remove obsolete allowlist entries when exports
  leave the root. For jargon, define a term in a guide only when that definition
  helps users, then record the defining guide in `tests/consumer_jargon.txt`.
  Baseline existing occurrences in `tests/consumer_jargon_baseline.txt`; counts
  may fall but MUST NOT grow. New wording MUST use analyst language.
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

- Changes that can move a published number cite the pinned baseline and verify
  the applicable numerical contract; ownership-only extraction preserves its
  existing result and identity values.
- When planning and runtime, or dataframe and query paths, derive the same
  quantity independently, add an equivalence test at matched inputs.
- For small probabilities, allocate family and tail error conservatively. Use
  survival or log-tail functions; do not recover a tail through
  `1 - alpha`.
- Aggregate calculations cover large offsets, neighboring floating-point
  values, input-order changes and merged partitions within a realistic data
  range. Beyond it, refuse with a code rather than crash or return a wrong
  value; do not add tests pinning exact results near float64 limits or
  subnormals. Prefer centered, mergeable moments and overflow-safe cross terms.
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
