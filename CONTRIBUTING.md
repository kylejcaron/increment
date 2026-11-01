# Contributing to increment

Thank you for your interest in contributing.

## Submitting changes

If you do not have write access, fork the repository, create a branch in your
fork, and open a pull request into `main`.

A pull request can merge only when CI (`ci-ok`) and the CLA check pass and a
code owner has approved it. Commits are squash-merged.

Release notes are generated from merged PR titles. Choose one change category:

| Label | Use |
| --- | --- |
| `feature` | Adds or extends supported behavior. |
| `bug` | Corrects behavior that violates the documented contract, including statistical errors. |
| `documentation` | Documentation-only changes. |
| `internal` | Maintenance without an intended public contract change. |

Add `breaking` when a promised public contract changes and consumers need a
migration, such as a removed API, changed result schema, or changed stable
refusal code. It supplements the change category and takes precedence in release
notes. Include the affected paths and migration route in the PR description.
Rejecting previously invalid inputs earlier does not alone warrant `breaking`.

Use `skip-changelog` to omit a PR from release notes. The automation labels
`dependencies`, `python:uv`, and `weekly-failure` are not contributor categories;
dependency updates are excluded from release notes.

Increment is alpha: breaking changes between 0.x prereleases are allowed but
must carry the `breaking` label and name the affected paths and a migration
route. The full policy, including persisted formats and stable refusal codes,
is in the [pre-1.0 compatibility policy](docs/api.md#pre-10-compatibility).

## License and the CLA

increment is licensed under the [Apache License 2.0](LICENSE).

Before your first pull request can be merged, you must sign the
[Contributor License Agreement](.github/CLA.md). Comment the following on your PR and
the CLA assistant records it:

```
I have read the CLA Document and I hereby sign the CLA
```

You sign once and it covers every future contribution.

**You keep the copyright in your work.** The CLA is a license, not an
assignment -- you may reuse, sell, or relicense your own contributions freely.
What it grants increment is the right to sublicense contributed code, which is
what makes commercial licensing and any future relicensing of the project
possible. Without it, a single unreachable contributor can permanently block
both.

Contributing on behalf of a company whose legal team requires its own
agreement? Email kyle.j.caron@gmail.com before opening a pull request.

## Development

Increment supports Python 3.12 through 3.14 and uses
[uv](https://docs.astral.sh/uv/) for dependency management. From a checkout,
create or update the development environment with:

```bash
uv sync --group dev --extra demo --extra tables --extra dashboard
```

`make install` synchronizes dependencies and installs hooks using the pinned
stable `prek` user tool. If it is absent, the installer prints the exact
`uv tool install prek==<version>` command from `uv.lock`; install it explicitly,
then rerun `make install`. Hook execution must not depend on a disposable
checkout's `.venv`. Inspect existing custom/legacy ownership before installation;
never force-replace shared hooks while active worktrees are unprepared.

The pre-push checks run lint, type-checking, a warning-only `test-weakening`
report and `prepush-tests`. The latter runs the changed unit tests, uses
pytest-testmon after seeding its database from the unit tier, and records a
successful receipt keyed by the tree and merge base so identical pushes skip
testing. Pushes MAY use `git push --no-verify`; CI is the required PR gate.
Numerical certification is selected from the certified modules' transitive
`tach.toml` import graph, certification-test changes and CI/build configuration;
unrelated module changes such as `increment/dashboard/_html.py` do not select it.
The parity-harness certification case runs nightly; its facade is not a
certification root because the other harness cases exercise those ingress paths
on every PR.
The PostgreSQL gate is selected for query dependency, warehouse-execution,
PostgreSQL-test, and build/workflow changes; docs-only and unrelated test-only
PRs skip it. PostgreSQL runs nightly on main; numerical certification runs
nightly inside the scientific tier.
Tagging requires successful `ci-ok` and a successful nightly run on the exact
tagged SHA. The existing `force` input bypasses both checks; hosted
branch-protection activation is an owner gate.

The Python 3.12 fast shards upload optional executed-code coverage artifacts.
After the required jobs, `ci-ok` combines matching shard artifacts and
publishes a combined report when every shard is complete. This advisory step
does not affect the required test result; missing, failed, or mismatched shard
data means no combined report is published. Coverage counts `increment/` source
statements executed during the fast tier (including pytest-cov's measured test
workers); it does not measure the slow or scientific tiers, establish
whole-suite coverage, or rate test quality.

Commit `uv.lock` whenever a dependency change updates the resolution.

Notebook examples use Marimo; the development environment does not include
Jupyter Notebook, JupyterLab, or ipywidgets.

### Tests and quality checks

CI measures import reach with `uv run --no-sync python scripts/import_reach.py --summary`; run the same command locally to inspect module counts and import times for each supported entry point. Pass explicit module names to measure a subset, or use `--json PATH` to save the results as JSON.
Policy: see AGENTS.md › Tests.

#### Test categories and runtime expectations

Unmarked categories default to cheap unit or small deterministic regression
checks. Most unit bodies should be sub-second; the existing local timeout is
a hang guard, not a target. Legacy unmarked tests are not automatically units:
inspect their assertions, fixtures and helpers before classifying them.
Scheduling/metadata marks such as `xdist_group`, `skip` and `compatibility`
do not define a test's purpose. Do not add redundant `unit` or `fast` marks.

| Purpose | Marks | Execution |
| --- | --- | --- |
| Unit/small regression | No category mark | `make test-unit`; ordinary local `make test` |
| Local integration boundary | `integration` | `make test-integration-fast` and `make test-fast`; PR CI |
| Expensive functional regression | `slow`, plus its purpose where useful | Relevant affected slow and required CI |
| Large-scale/exhaustive deterministic validation | `stress` and `slow` | Currently functional-slow; relevant changes and final integration/release |
| Monte Carlo coverage, bias or parameter recovery | `parameter_recovery` and `slow` | Nightly required gate and relevant inference validation |
| Deterministic inferential certification | `numerical_certification` and `slow` | Required for relevant changes and in the nightly gate |
| Runnable notebook exports | `examples` and `slow` | Nightly required gate and notebook acceptance/documentation |

`slow` describes cost, not purpose. A solver regression at ordinary inputs
does not become stress merely because it is slow: investigate production
performance or redundant test work. Category checks refuse recovery,
numerical certification, examples or stress tests lacking an effective `slow` mark before execution,
including CI; module, class and parameter inheritance count.

Integration tests remain in the fast tier and PR CI. `make test-unit` excludes
them for a smaller local loop; `make test-fast` runs unit plus integration-fast.

`make test-unit` and `make test-fast` report unit-selected call-phase process CPU
above 100 ms; the list contains only unbaselined tests. Baseline candidates have
exceeded 50 ms in any of three full-run samples, leaving headroom below the
reporting threshold. This is report-only because per-test cost varies with test
order and shared caches; use `--unit-cpu-budget-enforce` to opt into failures.
Stale baseline entries are notes for removal. Focused `make test` runs do not
enable measurement by default.
Stress alone remains in the functional-slow tier; an inferential capacity
campaign also carries numerical certification. Moving tests outside routine CI
requires an executable, required gate for relevant changes and a scheduled/release
route, with failure evidence retained; a label or proposed workflow is insufficient.
Monte Carlo and deterministic numerical certification are excluded from fast
and functional-slow. The certification job requires relevant changes selected
from the `tach.toml` module graph, while the nightly runs both scientific
purposes, examples, full fast and slow tiers, and a randomized-order fast tier.
Weekly retains dependency-floor matrices and live warehouse backends.
`TestEnumerationEnclosesTheExactMass::test_the_interval_holds_the_exact_rational_mass[central]`
and `[rare]`, plus `test_parity_case[exact_binomial_beyond_the_former_arm_ceiling]`.
Weekly execution does not replace required scientific validation before merging
changes to the covered estimator, numerical construction or guarantee.

For runtime audits, review every recorded test whose summed setup, call and
teardown exceeds five seconds; prioritize those exceeding ten. Also inspect
aggregate module/fixture costs, collection/startup and interrupted tests.
Consistently above five seconds under representative conditions means `slow`
until optimized; attribute shared setup before classifying dependent tests.
The thresholds are not single-run, hardware-sensitive CI failures. Contended,
partial or older-source runs identify candidates but cannot establish current
wall-time improvements or complete acceptance.

Test execution uses ordinary pytest options and markers. `make test-fast` is the
default CI selection; the Make targets define supported tiers:

| Target | Marker expression |
| --- | --- |
| `make test-unit` | `not slow and not integration and not parameter_recovery and not numerical_certification and not examples and not stress and not benchmark` |
| `make test-integration-fast` | `not slow and integration and not parameter_recovery and not numerical_certification and not examples and not stress and not benchmark` |
| `make test-fast` | Unit plus integration-fast (benchmarks excluded) |
| `make test-slow` | `slow and not parameter_recovery and not numerical_certification and not examples and not compile_registry and not benchmark` |
| `make test-tier TIER=compile-registry` | `compile_registry and not benchmark` (the warehouse compile registry; its own CI job) |
| `make test-parameter-recovery SPLITS=N GROUP=K` | `parameter_recovery and not benchmark` |
| `make test-numerical-certification` | `numerical_certification and not benchmark` |
| `make test-scientific SPLITS=N GROUP=K` | `(parameter_recovery or numerical_certification) and not benchmark` |
| `make test-examples` | `examples and not benchmark` |
| `make test-all` | `(slow or not slow) and not benchmark` |

The parameter-recovery, numerical-certification, scientific and examples tests
also carry `slow`; the marker expressions keep their purpose-specific gates
separate. `pytest-timeout` supplies the 300-second per-test hang guard. There
are no tier-level wall-clock budgets.

Ordinary test lanes pass `--benchmark-skip` because pytest-benchmark's xdist
collection warning is fatal under the project's warning policy. Run benchmarks
explicitly with `make benchmark`. To compare code changes against your machine,
capture a local baseline before editing:

```bash
uv run pytest tests/benchmarks/test_analysis_benchmark.py -m benchmark -n 0 \
  --benchmark-only --benchmark-save=before-change
make benchmark BENCHMARK_BASELINE=/path/to/baseline.json
```

The plugin writes the baseline JSON under `.benchmarks/`; pass that generated
path to `BENCHMARK_BASELINE`.

The committed `tests/fixtures/benchmarks/step4-pre-kernel.json` is a
reference-machine diagnostic, not a portable performance gate. `make benchmark`
records results by default; comparison happens only when `BENCHMARK_BASELINE`
is explicitly supplied.


Start with the exact regression:

```bash
make test TESTS=tests/file.py::test_name
```

Then run the affected fast tier against the task's immutable base:

```bash
make test-affected BASE=<task-base-sha>
make test-affected BASE=<task-base-sha> TIER=slow
```

`BASE` is required. Resolve it at dispatch and reuse that commit across later
commits and resumes; `BASE=HEAD` is only appropriate for a working-tree-only
comparison. `scripts.affected` compares the merge base and working-tree
changes. Configuration changes that affect selection or test infrastructure
run the full requested tier; changed test modules and their importers are
selected directly; test `conftest.py` changes select tests beneath that
directory; production Python changes use Tach with the immutable base, and
tests importing lazy root exports are also selected when the export's internal
import closure reaches the changed module; changed README/docs pages run their
snippets. Other changed paths are reported as not covered locally because CI
is responsible for them. Deleted test paths are not selected. A zero-selection
result is a successful no-op, not evidence that uncovered paths were tested.

Affected checks are local feedback, not a replacement for required slow,
parity, fixtures, public-smoke, documentation, scientific or full-suite gates.
CI is the authoritative full-suite gate. The pre-push hook runs lint,
type-checking, the warning-only weakening report and changed unit tests through
`prepush-tests`; it skips a successful receipt for an identical tree and merge
base. Pushes MAY use `git push --no-verify`, with CI as the required gate.
CI accepts test-reduction approval only through the `test-reduction-approved`
PR label; local checks continue to accept the `Test-Reduction-Approved:`
trailer on a reducing commit. `make check` is an optional broader local check.
Send pytest options through `PYTEST_ARGS` when using Make targets.

For an optional local inner loop, run `make test-changed` to use
pytest-testmon and run the whole files containing selected tests. The first
run builds the coverage database; this target is iteration feedback only and
never replaces any required test gate.

For runtime audits, review every recorded test whose summed setup, call and
teardown exceeds five seconds; prioritize those exceeding ten. Also inspect
aggregate module/fixture costs, collection/startup and interrupted tests.
These thresholds trigger investigation, not flaky timing failures. Retag only
after representative runs and inspect shared fixture/setup costs.

Record each family's contract, current source anchor, expensive workload,
disposition (keep, optimize, split, reclassify or remove genuinely redundant
coverage), required execution gate and exact parameter identities. Group
shared causes without overlooking expensive boundary variants. Keep small
deterministic witnesses in the routine tier; preserve full campaign
assertions, seeds, repetitions and tolerances separately. A small smoke does
not establish empirical coverage. Update marker selection, campaign scripts,
references and CI consumers together so reclassification cannot lose tests.
Use the existing `calibration/` directory for full certification campaigns
when appropriate; it is outside ordinary pytest collection.

Compare library-computed diagnostic readings within their declared numerical
reproduction budget, not bitwise. Frozen manifest bytes, scientific settings,
decisions and benchmark inputs remain exact; reproduction error is not
statistical tolerance.

Marker-only retags require collection and exact selected/deselected identity
checks, not rerunning test bodies or statistical campaigns. Collect changed
files with the marker filter cleared, then verify every affected tier and
the full inventory so no case disappears. Run focused execution when test
bodies, fixtures, assertions or selection/guard logic change. Batch those
checks at meaningful milestones rather than after each decorator edit.

#### Test economy

The suite's cost is item count. In October 2026 the recorded CI durations showed
about 20,000 collected items from about 9,000 test functions (fast tier
18,189 items, 619 CPU-seconds, median 2 ms; slow tier 1,825 items, 1,760
CPU-seconds, median 0.3 s; certification 82 items, 852 CPU-seconds).
Collecting the tree takes close to a minute per xdist worker in every shard,
so item count and shard wall time are the budgets to watch, not CPU. These
numbers date quickly; the duration files under `.github/` are current.

Tier intent:

- `fast` is the PR gate: unit plus in-process `integration` (DuckDB, Ibis,
  dataframe entry points). Keep each shard around ten minutes of wall time;
  add shards before trimming tests.
- `slow` holds deterministic functional work: large-N parity, power
  enclosure, artifact replay, dashboard capture. Same shard target.
- `numerical_certification` runs on PRs only when certified modules change;
  `parameter_recovery` is nightly. Neither moves into `fast` or `slow` to
  see it on a PR.

To inspect the fast-tier test items selected by representative one-file changes,
run `uv run --no-sync python scripts/selection_probe.py`. The probe reports the
selected items, fast-tier total, direct-file and Tach components, and end-to-end
`make test-affected` time for each module. Use `--json PATH` to save the rows,
`--summary` to append the table to a GitHub Actions step summary, or
`--no-timing` to collect selection counts without running the affected tier.

- `integration/warehouse_execution/` is weekly or on touch and never
  collected from `testpaths`.

Where new tests go:

- One test file per source module, `tests/<pkg>/test_<module>.py`. Do not
  add a file for a facet of an existing feature; the
  `test_unit_day_artifact_*` family is the pattern to stop, not extend.
- Dashboard behavior is asserted on the prepared payload
  (`increment.dashboard` data structures). Rendering tests are a small set of
  `slow` plus `integration` smokes; a new dashboard test renders only when the
  behavior is the rendering. Rendering is about a quarter of the slow tier.
- Power-planning and parity work over large N belongs in `slow`; a fast-tier
  sentinel is one representative case, not a grid.

The sequential screen and fitted-evidence shape grids have been collapsed to
retained boundary cases plus a property. The parity matrix remains a strength-3
covering array with disposition sentinels; the full grid runs nightly. Do not
restore the collapsed grids. Any `parametrize` that expands past twelve items
becomes boundary cases plus a property or carries `@pytest.mark.grid("<reason>")`.
Check with `--collect-only -q` on the changed file.

Red check: every new test is observed failing before it is accepted. For a
bugfix, run it against the pre-change code; for new code, introduce a
deliberate one-line break in the code under test, watch the test catch it,
and revert. Use a detached worktree
(`git worktree add --detach /tmp/red-check <base>`) when the old code is
needed; never `git stash`, which disturbs other worktrees' uncommitted work.
Quote the failure in the PR.

Mutation evidence: `scripts/mutation_campaign.py` runs bounded mutmut 3.8
campaigns over named functions in a fresh interpreter per mutant (the
numerical stack is not fork-safe, so the mutmut CLI is not used directly).
Campaigns are scoped by `[tool.mutmut]` in `pyproject.toml` and the campaign
table in the script. A change under `increment/estimation/`, `increment/power/`
or `increment/query/` that adds tests states which campaign covers the touched
function, or that none does yet; run the campaign on demand under the usual
local resource limits, never as a per-push step. Survivors are reported, not
fixed, unless the task is about them. When a campaign exists, a new test
should kill a mutant no prior test killed; otherwise the PR names, in one
line per test, the bug it catches that no existing test catches.

Long descriptive names and one behavior per test are the house style; do not
merge tests to shorten names. The duplication to remove is two tests no
single-line mutant can separate, which only a per-test kill matrix shows.
Audit the three modules with the highest test-LOC-to-source ratio once per
release cycle and after any change that adds more than 200 items.

#### Resource and behavior guards

Heavy Make and Nox test commands acquire advisory slots through
`scripts/test_slots.py`. `TEST_SLOTS` defaults to `3`; `TEST_SLOT_DIR` defaults
to `${XDG_CACHE_HOME:-$HOME/.cache}/test-slots`. The wrapper waits when all
slots are occupied and prints `waiting for a heavy test slot; held by:` plus
the holders' pid, working directory, command and start time. Slot coordination
is bypassed when `CI` or `TEST_SLOT_HELD` is present or `fcntl` is unavailable.
Focused `make test TESTS=...` uses a slot and xdist for file/directory selections; one node ID remains unwrapped and single-process.

On local slot-held runs the wrapper raises the command's nice value by 10
before `exec`, leaving more CPU priority for interactive work. CI and nested
wrappers that inherit `TEST_SLOT_HELD` do not change priority; nested commands
inherit their parent's nice value.

Ibis DuckDB test connections default to one thread, a memory limit from
`INCREMENT_TEST_DUCKDB_MEMORY` (default `1GB`) and a unique spill directory
per connection; direct `duckdb.connect` calls are not wrapped and retain
DuckDB's own defaults. Explicit Ibis connection settings take precedence;
explicit `None` omits that setting so DuckDB uses its own default.
Hypothesis uses the `dev` profile locally (`deadline=None`, suppressing only
`HealthCheck.too_slow`) and the `ci` profile when `CI` is present (200 ms
deadline and default health checks).
Affected selections of up to six direct test files run without xdist workers.

Unit-selected tests run without DuckDB connections, subprocesses, processes or
Internet sockets. Collection raises `pytest.UsageError` if a unit test requests
a fixture declared with `backend_fixture`; runtime refusals identify the
nodeid and action. Mark a true component boundary `@pytest.mark.integration`
(still included in fast and PR CI), or exercise the pure function on
in-memory data. `tests/_unit_boundary.py` defines the boundary.

The test-hygiene checks in `tests/test_test_hygiene.py` enforce H3 (patches
targeting `increment` internals) and H4 (imports of private `increment`
internals). New occurrences fail with source locations and recommend testing
through the public API or injecting a dependency; lower counts also fail as a
stale `tests/test_hygiene_allowlist.txt` baseline and show observed/baseline
counts.

H5 limits literal `parametrize` fan-out: over twelve expanded cases require
boundary cases plus a Hypothesis property or a reasoned `@pytest.mark.grid`.
Existing literal-grid exceptions are recorded under `# H5` in
`tests/test_hygiene_allowlist.txt`; that baseline can only decrease.
`grid(reason)` is registered in `pyproject.toml`. No item count is committed.
The `item-increase-reason` check collects the pull request's merge base and
head and compares unit, fast, slow and all item counts; a tier that rises
requires a PR-description line `tests: +N items — <reason>` with `N` at least
that rise (always spelled `items`). Lower counts need nothing.
`make check-items BASE=<ref>` prints the same comparison for the working tree.
Computed grids are reported separately from the literal fan-out check.

Together, the weakening check reports fewer test functions or assertions,
requiring owner approval. Local checks accept a `Test-Reduction-Approved:`
trailer on each reducing commit; CI ignores commit trailers and requires the
PR owner to apply the `test-reduction-approved` label. The item ratchet reports
more collected items or a literal grid over 12 without `grid` (requiring a
stated reason); collapsing a grid to boundary cases plus a property lowers
items and keeps functions, passing both.

The pre-push `test-weakening` hook runs
`scripts/check_test_weakening.py`. It checks whole-tree test-function and
assertion counts and reports added skip/xfail behavior, loosened tolerances,
reduced Hypothesis `max_examples` and raised timeouts. Assertion counts include
`assert` statements and recognized assertion calls such as `pytest.raises`,
`pytest.warns`, `pytest.approx` and testing-library `assert_*` functions.
The check command returns `1` for findings and `2` for invalid arguments, base
or repository. The pre-push hook turns findings into a warning and exits
successfully; the required CI job remains the gate for the PR.
Locally, only a `Test-Reduction-Approved:` trailer on the reducing commit
approves its findings; it cannot approve working-tree findings. CI ignores
trailers and approves committed findings only when the PR has the
`test-reduction-approved` label. Because counts are whole
tree totals, a deletion can be offset by an unrelated addition in the same
commit; review the diff as well as the check output.

`make mutate CAMPAIGN=<name> [KILL_MATRIX=1]` runs an on-demand mutation
campaign through the slot wrapper; mutation evidence is not a per-push step.
In kill-matrix output, `cover` is the greedy test subset covering killed
mutants, while `candidates_for_removal`, `killed_nothing` and
`string_literal_only` are review inputs, not automatic deletion instructions.
String-literal-only kills can overstate behavioral coverage.

Before pushing, the pre-push checks run lint, type-checking and the affected
fast tier. CI is the authoritative full-suite gate; `make check` is an optional
broader local check. Use `make fmt` to apply formatting and safe lint fixes.

Additional commands:

- `make test-unit` runs the local unit tier; `make test` aliases it.
- `make test-fast` runs unit plus fast integration tests for PR CI.
- `make test-slow` runs functional slow tests, excluding Monte Carlo recovery,
  deterministic certification and notebook examples.
- `make test-versions` runs the supported-Python Nox sessions.
- `make ci-local` adds documentation, table-extra, build, and installed-wheel
  checks to the normal local gate.
- `make test-all` is reserved for final integration and release candidates
  that change estimation or a wire format. It is not a routine contributor
  check.

Pytest applies a 300-second per-test hang guard. There are no tier-level
wall-clock budgets. The Nox sessions invoke pytest directly; CI uploads JUnit
reports and prints the duration summary for test jobs.

Monte Carlo-only shards use
`make test-parameter-recovery SPLITS=4 GROUP=1`; deterministic certification
uses `make test-numerical-certification`. Nightly scientific-union shards use
`make test-scientific SPLITS=4 GROUP=1`; without local timing data,
pytest-split distributes tests evenly.

Make targets default to `max(1, CPU count // TEST_SLOTS, CPU count // held slots)` workers after acquiring their slot; override with
`WORKERS=auto` or a specific count. CI passes `WORKERS=auto`; ordinary
commands accept `PYTEST_ARGS` for pytest options. The pre-push checks run
lint, type-checking and the unit tests the change executed (testmon selection, skipped for an already-passed tree); the CI matrix is the full-suite
gate.

The test bootstrap defaults numerical-library thread pools to one per process,
matching Nox's controls, while preserving explicit environment overrides.
Notebook checks share one batched Marimo invocation while retaining a separate
guard for each notebook; export and warehouse-regeneration scenarios still
execute independently.

Unit tests do not use the network or real warehouses. Live PostgreSQL,
Snowflake, and BigQuery checks live under `integration/warehouse_execution/`
and require dedicated scratch resources and credentials. Contributors do not
need those services for ordinary changes.

The checked-in CI candidate requires lint, fast tests, functional slow tests,
package builds, and live PostgreSQL 16 probes through `ci-ok`; hosted
enforcement and required-check activation remain owner gates. PR fast tests run
on Python 3.12 and 3.14; 3.13 remains in main pushes and nightly full/weekly floor
matrices. `ci-ok` validates every selected fast shard and required conditional
gate. It combines Python 3.12 fast-shard coverage advisory; cross-platform wheel
smoke runs for packaging-related PR changes and on pushes.
PostgreSQL runs on a PR when a change touches the transitive `tach.toml`
dependency closure of `increment.query`, `integration/warehouse_execution/`,
a test helper the PostgreSQL suite imports, build or workflow files, or an
unclassified path; deletions, renames and an unavailable diff also require it.
Main pushes never use this filter.

The PR PostgreSQL gate runs one duration-balanced shard. The reusable
warehouse workflow defaults to three shards for weekly and manual runs.

Each selected fast Python-version cell has three duration-balanced
`pytest-split` shards, with `xdist` workers inside each shard. `ci-ok` requires
all six PR shards or all nine main-push shards (Python 3.12–3.14). The build
job also runs the documentation snippet and strict-docs checks when changed
paths require them. The machine-generated `.github/fast-test-durations.json`
weights come from a complete, passing local fast run on 2026-10-31 (Python 3.13,
six `xdist` workers); only their proportions matter, and absent weights do not
omit tests. Refresh them from a complete passing fast run; `--clean-durations`
drops tests that no longer exist:

```bash
make test-fast WORKERS=auto PYTEST_ARGS="--store-durations --clean-durations --durations-path .github/fast-test-durations.json"
```

The sharded run weighs a test in an `xdist_group` under `<node ID>@<group>`,
while the single-process selection prediction looks up its bare node ID, so
give each grouped weight both keys and round to milliseconds:

```bash
uv run --no-sync python - <<'PY'
import json
from pathlib import Path

path = Path(".github/fast-test-durations.json")
durations = json.loads(path.read_text())
for nodeid, seconds in list(durations.items()):
    bare, _, group = nodeid.rpartition("@")
    if bare and "]" not in group and "::" not in group:
        durations[bare] = seconds
rounded = {nodeid: round(seconds, 3) for nodeid, seconds in durations.items()}
path.write_text(json.dumps(rounded, indent=4, sort_keys=True) + "\n")
PY
```

The generated `tests/selection_map.json` is an advisory fast-tier map from
executed `increment` modules to test files. Refresh it with `make selection-map`;
the target runs the full fast tier with per-test coverage contexts, snapshots
imports during each test module's collection, records fixture dependencies, and
rebuilds the map. Nightly
CI proposes a refresh in a `selection-map` pull request. Maps older than 14 days,
from another base or tier, or invalidated by a lazy-dispatch change disable
narrowing: the affected-test handler keeps the entire requested tier rather
than trusting the import graph alone. PR CI continues to run the full fast
tier; the separate selection-miss report is evidence, not a test gate.

The map is exact for observed execution and import statements, but advisory for
dependencies with no import statement (for example, a cached module or a
runtime redirect). Do not use this fast-tier map as a gating selector for
another tier. Keep `tests/selection_required.txt` updated for dependencies
which static imports and observed coverage cannot represent.

Functional-slow CI uses four isolated shards and the certification job two,
with workers inside each shard; `ci-ok` requires their aggregate success. Every
case remains selected exactly once. Both split by the recorded weights in
`.github/slow-test-durations.json` and `.github/certification-test-durations.json`;
a test without a weight still runs. Proportions matter, not absolute seconds, so
refresh the weights from a complete local run of the tier when the tier changes
materially:

```bash
make test-slow WORKERS=auto PYTEST_ARGS="--store-durations --durations-path .github/slow-test-durations.json"
make test-numerical-certification WORKERS=auto PYTEST_ARGS="--store-durations --durations-path .github/certification-test-durations.json"
```

The root pytest hook uses the ignored `.test_durations` cache for stable,
longest-first local scheduling across fast and slow tests. Each run refreshes
the call durations it executes; to populate the whole map, run the full suite:

```bash
uv run pytest --durations=0 -m "" tests
```

Without `.test_durations`, ordinary tests retain collection order. Markdown
snippets are still scheduled after ordinary tests and retain their source order
within each file.

Dashboard, simulation and artifact tests reuse expensive prepared fixtures under the
ignored `.cache/increment-fixtures/` directory. Cache entries are keyed by
source and the installed Python environment. Set
`INCREMENT_DISABLE_FIXTURE_CACHE=1` to bypass the cache and rebuild fixtures;
remove `.cache/increment-fixtures/` to clear all cached fixtures.
Artifact seed caches contain only detached input rows; each consumer retains a
fresh mutable database. Schema-invalid matrix inputs refuse before seeding
unused fixture databases.

The complete PostgreSQL suite runs across three isolated PostgreSQL services,
balanced with `pytest-split` and `.github/postgres-test-durations.json`.
The weights come from hosted run 36941053401; new cases still run using the
average weight. To refresh weights, run the complete suite with
`--store-durations --durations-path .github/postgres-test-durations.json`.
Do not run workers against a shared warehouse namespace.

The
[nightly workflow](.github/workflows/nightly.yml) runs daily at 06:00 UTC or
on dispatch on any ref. It runs the full fast and slow tiers with the full
parity matrix, numerical certification, parameter recovery, examples, and a
randomized-order fast run with its seed logged. A failure creates or updates a
GitHub issue labelled `nightly-failure`; tagging also requires a successful
nightly run on the exact release SHA.

The
[weekly workflow](.github/workflows/weekly.yml) runs Sundays at 08:00 UTC or
on manual dispatch. It retains full test runs at dependency floors on all
three Python versions, which nightly does not cover, plus all three live
warehouse backends. Weekly failures create or update a GitHub issue labelled
`weekly-failure`.


#### Refusal registry gate

Register each refusal lookup key in the registry bound to its caller's
`raiser(...)`; registration in another module does not satisfy that dispatcher.
Keep intentional aliases pointing to the canonical spec. The gate checks local
registration and statically known template keys:

```bash
make test TESTS='tests/test_refusal_drift.py tests/test_refusal_uniqueness.py tests/test_docs_refusal_guidance.py tests/test_refusal_gate.py'
```

Dynamic keys are reported as statically unknown and require public
producer/dispatcher coverage; this gate is not a global refusal catalogue.

### Backend verification

DuckDB runs in the default test suite. Dedicated PostgreSQL, Snowflake, and
BigQuery probes check live query execution, native-warehouse/artifact parity,
materialization, and cleanup isolation—not just SQL compilation. PostgreSQL
additionally checks parity against dataframe oracles. Their CI schedule is
described above.

Cloud Nox sessions also execute credential-free namespace regression cases
through the installed backend's real DDL compiler. The generated SQL runs in
DuckDB; these cases do not substitute for the live probes.

Snowflake and BigQuery run weekly or on manual dispatch, not on every PR or
main push. Before merging warehouse changes, dispatch the existing workflow
on the reviewed, trusted PR branch and wait for both cloud jobs:

```bash
gh workflow run warehouse-backends.yml --ref <trusted-pr-branch> -f backend_mode=cloud
```

These manual jobs are not required by `ci-ok`; do not use an untrusted branch
with repository credentials. Use `backend_mode=postgres` for credential-free
PostgreSQL probes or `all` to run all three backends.

BigQuery prints each live probe's result and has a 45-minute workflow hang
guard; PostgreSQL and Snowflake retain 30-minute guards. These are job
deadlines, not performance targets or reduced test coverage.
If BigQuery returns `QueryUsagePerDay`, its project-level daily query quota
is exhausted. Keep the run failed and wait for the
[midnight Pacific quota reset](https://docs.cloud.google.com/bigquery/docs/custom-quotas)
before repeating the full cloud workflow; do not convert the failure to a skip.

Probe coverage is not blanket support for every metric, design, or artifact
extension. Use the [compatibility matrix](docs/guides/compatibility.md) and
[statistical limitations](docs/limitations.md) to check the combination you need.

### Validation evidence and external reports

[`docs/validation.md`](docs/validation.md) is the single inventory of independent-reference
evidence, including methods that have none. Frozen fixtures live under `tests/oracles/` with
their generator scripts; ordinary tests read fixtures only, never R or the network. Run an
oracle with `make test TESTS=tests/oracles/test_welch_oracle.py` (pytest accepts test files
directly; [Reproduce](docs/validation.md#reproduce) lists each one). To report a
discrepancy against a reference, open a GitHub issue with the
increment version or commit, the entry point, the reference tool and version, the two numbers and
the tolerance, and attach aggregate moments or the reference output, never identifiers,
credentials, or raw private data.

### Releases

Dispatch `bump` on the reviewed commit. It derives the baseline from the highest
public PEP 440 `v*` tag reachable from that commit, so a backport branch does not
inherit an unrelated newer release. Preview the calculation and CI gate first:

```bash
gh workflow run bump.yml --ref main -f dry_run=true
```

The workflow summary shows the baseline and selected version. Preview creates
no tag and dispatches no release. After reviewing the candidate, run without
`dry_run` to tag and build it:

```bash
gh workflow run bump.yml --ref main
```

| Input | Default | Behavior |
| --- | --- | --- |
| `bump` | `auto` | Choose `auto`, `patch`, `minor`, or `major` |
| `stage` | `keep-current` | Keep the stage or explicitly choose `alpha`, `beta`, `rc`, or `stable` |
| `version` | blank | Exact canonical PEP 440 override, without `v` or a local (`+`) segment |
| `dry_run` | `false` | Resolve the version and check CI without tagging or dispatching |
| `force` | `false` | Explicitly bypass only the CI gate |

`auto` increments the current alpha/beta/RC counter, or patch when stable.
Explicit patch/minor/major bumps reset lower base components; a retained
prerelease stage restarts at counter 1 on the new base. Stage promotion is
manual: passing CI or reaching a counter never makes a release beta or stable.

| Baseline | Bump / stage | Selected version |
| --- | --- | --- |
| `0.1.0a2` | `auto` / `keep-current` | `0.1.0a3` |
| `0.1.0a2` | `auto` / `beta` | `0.1.0b1` |
| `0.1.0a2` | `auto` / `rc` | `0.1.0rc1` |
| `0.1.0a2` | `auto` / `stable` | `0.1.0` |
| `0.1.0a2` | `patch` / `keep-current` | `0.1.1a1` |
| `0.1.0a2` | `minor` / `keep-current` | `0.2.0a1` |
| `0.1.0a2` | `major` / `keep-current` | `1.0.0a1` |
| `0.1.0` | `auto` / `keep-current` | `0.1.1` |
| `0.1.0` | `auto` / `alpha` | `0.1.1a1` |

Repeating the same stage increments its counter. Moving to an earlier stage on
the same base is refused; select a base bump to start a new prerelease line.
Beta and RC are optional. For example, explicitly promote to beta with:

```bash
gh workflow run bump.yml --ref main -f stage=beta
```

Set `version` for exceptional targets, including first releases or backports.
It is exclusive with nondefault `bump`/`stage`; it does not bypass validation,
duplicate-version protection, CI, or PyPI approval:

```bash
gh workflow run bump.yml --ref main -f version=0.1.0a3 -f dry_run=true
```

Automatic calculation refuses baselines with epochs, post/dev segments, or
other than three base components; use the exact override for those targets.
Versions already tagged on any branch, including equivalent PEP 440
spellings, remain reserved even if publication failed. Bump does not update
or delete tags.

For a local calculation without GitHub or any tag mutation:

```bash
uv run python scripts/next_version.py --bump auto --stage keep-current
```

The default `bump` gate requires successful GitHub Actions `ci-ok` on that exact
commit, no other unfinished or failed checks, and successful legacy commit
statuses when present. It excludes only the running bump workflow itself.
`force=true` skips only this upstream CI gate; it does not bypass any release
workflow checks.

Bump creates the version tag and explicitly dispatches `release.yml` on it.
The release workflow also runs on pushed `v*` tags; a direct manual dispatch is
accepted only when its ref is a version tag. A failed bump dispatch can be
reissued against the existing tag, without creating another tag:

```bash
gh workflow run release.yml --ref v0.1.0a3
```

Every route resolves the tag to the exact checked-out commit, builds and hashes
the wheel and sdist, then runs strict documentation validation and the README
mean-analysis example using the installed built wheel outside the checkout.
Both checks must succeed for that same commit before publication is eligible;
there is no force/bypass input for this gate. The workflow run summary and
`release-receipt` artifact record the candidate SHA, tag/version, artifact
hashes, tested scope, run link, unresolved evidence, and deviations. Missing,
failed, cancelled, or wrong-SHA evidence blocks upload.

After those checks, publication still waits for the `pypi` environment. Inspect
the exact candidate receipt and build artifacts before approving: approval
uploads to real PyPI, not a test registry. A separate post-publication job
installs `increment==<version>` from PyPI into an isolated environment, waits
up to five minutes for that exact version to appear, checks installed metadata
and import location, and runs the README mean-analysis example. A failure here
is after publication; it never retries upload, deletes a tag, or rolls back.
The post-index result is not a prerequisite for the already completed upload.

The hosted-settings audit found the `pypi` environment with a branch-policy
rule and a required reviewer (`kylejcaron`); `can_admins_bypass` is false and
self-review is not prevented. Active `main-review` rules require `ci-ok` and
`cla`; `main-safety` prevents deletion and non-fast-forward updates. The
branch-protection endpoint reports the branch is not protected. These settings
are not changed here; environment and ruleset configuration is distinct from
the workflow's exact-candidate gate.

#### Website documentation sync

After a stable release is on PyPI and GitHub, `release.yml` tells the
[documentation website](https://incrementdocs.pages.dev/) the release tag and
commit. The website then opens a draft PR with that release's docs for someone
to review and merge. Prereleases never notify the website. If the notification
fails, the package release is unaffected.

It is off by default. To turn it on, after the website side is set up (see its
`CONTRIBUTING.md`):

- Install the docs GitHub App on `kylejcaron/getincrement.io` only.
- Here, set variables `DOCS_APP_ID` and `DOCS_SYNC_ENABLED=true`, and secret
  `DOCS_APP_PRIVATE_KEY`.

If a notification was missed, run the sync from the website instead of
re-running `release.yml`:

```bash
gh workflow run sync-release-docs.yml --repo kylejcaron/getincrement.io \
  --ref main -f tag=v1.2.3 -f sha="$(git rev-parse 'v1.2.3^{commit}')"
```

### Calibration commands

Scientific campaigns and probability probes live in `calibration/`.
Repository maintenance commands—test execution, documentation checks,
compatibility rendering, and wheel smoke checks—remain in `scripts/`.

Run calibration commands from the checkout with `uv run python -m`:

| Module | Purpose |
| --- | --- |
| `calibration.cluster_diagnostics` | Cluster-estimator diagnostics |
| `calibration.inference_diagnostics` | Inference and winsorization diagnostics |
| `calibration.sequential` | Sequential-inference research campaign |
| `calibration.unit_cycle` | Unit-cycle switchback research campaign |
| `calibration.binomial_grid` | Deterministic rare-event probability integration |
| `calibration.bernoulli_prior` | Bernoulli prior-weight study |
| `calibration.ratio_precision` | Ratio-denominator precision study |

Use `--help` for each command's options. Campaign output directories must be
new; incomplete runs retain their evidence and do not certify the full design.

### Documentation and comments
Policy: see AGENTS.md › Documentation.


Run the audit:

```bash
make audit-verbosity
```

Standalone Python comment or docstring blocks longer than the configured limit
need a first-line `# prose: allow-long <specific justification>`. Do not split
blocks or add blank lines merely to evade the check. Public API docstrings are
not length-capped; their detail should remain useful to callers.


Exercise documentation snippets and build the complete site with:

```bash
make test-doc-snippets
uv run --with nox nox -s docs
```

Use the Nox documentation session rather than invoking MkDocs directly. It
exports notebook-backed pages before performing the strict site build.

The PR workflow runs snippets for `README.md`, `docs/`, Python package source,
and documentation/build inputs. It runs the strict site build for
documentation/navigation/build/dependency changes, the root public exports and
results modules, or PRs labelled `breaking` or `docs-required`. This intentionally
does not build the strict site for every source-only public API change; an
unlabelled API change can first be caught by the mandatory exact-candidate
strict-docs check before release publication. `docs-required` is an orthogonal
review label for public API documentation, not a release-note category; the
repository owner must create and activate that label. The existing
`documentation` label remains for documentation-only release notes.

### Capability composition
Policy: see AGENTS.md › Making a change.

Every public capability declares its behavior for every metric type in
`tests/compatibility_catalog.py`, with enforcement in
`tests/test_composition_matrix.py`. When adding a capability or metric type:

1. Add every required matrix cell.
2. Classify it as supported, refused, structurally inapplicable, or a known
   gap.
3. Back the classification with an executable probe.
4. Update the authoritative bindings in `tests/capability_entry_point_catalog.py`
   and their consumer/parity evidence; regenerate the path-by-capability page
   with `uv run python scripts/render_capability_table.py`, then run it with
   `--check`. Use `scripts/probe_capability_table.py --case <id>` to exercise
   the named scenario, not to transcribe a second capability table.
5. Decide where the refusal lives. A rule that does not depend on the data
   source goes in the shared readout gate (`validate_request` and its owners).
   Declare it in `GATE_POLICY` in `tests/test_source_capabilities.py` when it
   varies over that table's axes (metric type, option, view). Rules a source
   fixes at construction (sequential, observational, encouragement) are outside
   those axes; the arm-contract sweeps in `tests/test_refusal_uniqueness.py` and
   the pair cells in `tests/compatibility_catalog.py` own them. Add a
   same-code-from-every-source test to `tests/test_refusal_uniqueness.py`.
   A rule that every source expresses through its declared `capabilities` and
   `breakouts` (grain, breakout dimension) is also shared:
   `validate_readout_source` raises it under a `readout.source.*` code. Only a
   limitation specific to one adapter is raised by that adapter under its own
   `source.*` code and recorded in the enumerated parity harness
   (`tests/parity_harness/`).


### Adding a method or design
Policy: see AGENTS.md › Making a change.

Have an approved tracking issue before implementing a new estimator, inference
route, or experiment design: a Kata issue for maintainers and automated
contributors (search for an existing one first), a GitHub issue for external
contributors. Whether to take on a method is a product decision for the
code owners; the conditions below decide whether an approved addition is
complete. They gate new methods only and do not hold up fixes to behavior that
already ships.

For worked source-to-result examples, see the
[estimator and evidence extension guide](docs/development/adding-an-estimator.md).

A new method needs all of the following:

1. **A documented need.** A concrete user workflow or maintenance burden,
   not breadth for its own sake. A published source supports the method's
   provenance; it does not substitute for that need.
2. **An approved design.** The estimand, the identification argument, the
   assumptions, and the guarantee class it is derived to have (`exact`,
   `asymptotic`, `planning approximation`, `experimental`, or `none stated`,
   as defined under [evidence status](docs/validation.md#evidence-status)).
   State assumptions and guarantee strength publicly; `none stated` does
   not imply coverage or calibration.
3. **Interactions classified.** For CUPED, ratio metrics, clustering,
   sequential inference, winsorization, breakouts, and multiplicity roles, give
   every applicable combination one status: `supported`, `source_limited`,
   `construction_limited`, `unfinished`, `unsound`, or `not_expressible`. The
   statuses are defined in the [evidence status](docs/validation.md#evidence-status)
   table. A mathematically invalid combination is `unsound` and carries its
   derivation; it is not missing engineering. A missing implementation is
   `unfinished` and is tracked in an issue, not in the limitations page.
4. **Planning matches runtime.** Where the planner and the runtime derive the
   same quantity, an equivalence test at matched inputs.
5. **Ingress parity.** For a capability reachable through `Analysis`, a row or
   axis value in the enumerated [parity matrix](tests/parity_harness/COVERAGE.md)
   with a stated disposition for each of the six `Analysis` entry points. A
   standalone family (for example `estimate_policy_contrast` over a
   `LoggedTrace`, or a power solver) does not belong in that matrix; it
   enumerates parity across its own public ingresses instead. Path presence
   alone is not evidence of parity.
   PRs run the deterministic strength-3 covering scope plus disposition
   sentinels; the nightly gate sets `INCREMENT_PARITY_MATRIX=full` to execute
   the exhaustive grid. Both scopes always run the asymptotic-retention variants
   and non-parametrized parity tests; see the coverage page for counts.
6. **An independent reference.** Either a frozen comparison against a
   third-party tool or independent primitive under `tests/oracles/`, or a
   justified independent construction check (an enumeration or an exact
   derivation that shares no code with the implementation). Record it in the
   [reference inventory](docs/validation.md#reference-inventory) with its
   tolerance and the regimes it does not cover. If no matched reference
   exists, add a row to
   [methods without an independent reference](docs/validation.md#methods-without-an-independent-reference)
   with its status, guarantee, basis, and ingress coverage, and add an
   [unvalidated regimes](docs/validation.md#unvalidated-regimes) entry where a
   specific regime remains unstated, rather than leaving the gap unstated.
7. **Stable refusals and bounds.** A coded refusal for each expressible
   unsupported variant, raised before data is loaded. A hazard that does not
   depend on the source gets the same code on every ingress path; a limitation
   specific to one adapter (`source_limited`) keeps its own `source.*` code.
   Enforce deterministic operational ceilings where complexity, numerical
   validity, or resource safety justifies them. Report measured latency and
   memory with their environment and input scope; measurements alone do not
   justify arbitrary wall-clock or memory refusals for every method.
8. **A complete change.** Examples, guides, the
   [compatibility reference](docs/guides/compatibility.md), the path-by-capability
   table, and fixtures updated together, following
   [Capability composition](#capability-composition) above.

These conditions do not require every existing method to be validated before a
new one is added, and a missing external case study is recorded as absent, not
manufactured.

### Advisory complexity reports

Use `make complexity` for a full cognitive-complexity report or:

```bash
make complexity-diff BASE=<commit-or-branch>
```

These reports are advisory. Use them to guide review; do not suppress
findings, fragment functions, or alter numerical algorithms merely to reduce
a score.

### Packaging changes

For packaging or public-export changes, build and inspect the release
artifacts:

```bash
uv build
uvx twine check --strict dist/*
uv run --with nox nox -s wheel_smoke wheel_smoke_demo
```

The wheel smoke sessions install the built artifact outside the source
checkout, ensuring imports do not accidentally resolve against local files.

## Project coordination
Policy: see AGENTS.md › Working with others.


### Review and handoffs
Policy: see AGENTS.md › Working with others.

Pin the integration branch, absolute worktree and task base before dispatch.
Assign disjoint files and one writer for shared result schemas, serialization
boundaries and integration configuration. Consumers need an exercised
implementation handoff, not an independently invented carrier from a design.
Keep modules importable after each edit. Preserve partial work and stop the old
writer before transferring ownership.

Coordinate the host's aggregate CPU/RAM allowance. Scoped correctness checks may
overlap; one coordinator runs broad integration checks. Reserve an uncontended
window only for timing measurements, and label measurements under load as
contended. Integrate a lane after its focused checks and review pass, then run
the combined focused selection. Include relevant slow tests of each changed test
file before its integration commit; docs-only clarifications do not justify
repeated broad suites.

Every PR requires a human diff read before merging, including whether an
extraction left parallel implementations behind. Design reviews belong at their
agreed contract gate; subsequent verification reviews the implementation.
Neither a passing primitive test nor an accepted design establishes a complete
analyst journey.

When using Roborev, freeze the head and review the coherent base-to-tip range:

```bash
roborev review --quiet --repo <absolute-worktree> --branch=<branch> --base <base-sha> --wait
```

`--branch` has an optional value: use `=` for a named branch; a separate
word is parsed as a positional commit, not the branch name.

Use `--quiet` to skip Roborev's automatic shared-hook maintenance; `--local`
alone does not skip it. Quiet mode suppresses enqueue headers, so inspect
`roborev list --json` and `roborev show --job <id>` for the resulting job's
pins and verdict. After a Roborev upgrade, reverify that this invocation leaves
the common hooks unchanged before using it without activation authority.

Confirm the job's actual base/head and that the branch did not move.
`--sha` does not replace a branch range's base. Review changed integration
resolutions and substantive fixes; do not skip all merge commits or suppress
finding classes. Run `roborev fix --list --branch <branch>` from the assigned
worktree before closing tracked work and after integration commits. Close each
review only with code/test evidence for every finding; another clean review
does not resolve an older finding.

The repository's installed post-commit hook currently dispatches automatic
reviews. Changing automatic dispatch or hook installation requires an explicit
owner checkpoint, an inventory of live worktree consumers, a tested rollback
and verified supported controls.
Keep current hooks and the before-push `make check` obligation until the
replacement hosted gate is actually required and its failure behavior is observed.

For an owner-authorized manual-ready transition, back up the common Git
`post-commit`, `post-rewrite` and `pre-push` files with their permissions and
record their custom/legacy ownership before running `roborev uninstall-hook`
from the prepared worktree. This command removes Roborev from all three hooks,
not just commit dispatch; verify each remaining hook before relying on it.
Use the pinned range command above for reviews. Do not remove mandatory
pre-push checks. Rollback restores the recorded files and permissions;
`roborev install-hook --binary <stable-roborev-executable>` is the supported
reinstallation command, without `--force`. Reverify actual custom/legacy
behavior and prepared worktree consumers before activation.

Handoffs name exact commits, dirty/untracked files, executed commands and evidence,
unresolved decisions, remaining acceptance and next action. Distinguish
implemented, verified, integrated and issue-closed state. Update the ledger at
each state change. Ignored plans and temporary probes stay local; do not force-add
them to obtain commit-based reviews or replace product documentation.
