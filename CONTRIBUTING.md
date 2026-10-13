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

The candidate-safe pre-push wrapper validates every non-deletion ref against
the clean checked-out HEAD, then replays Git's complete input to the existing
native and legacy hooks. Its Python interpreter and `prek` executable are
resolved during explicit installation, not bootstrapped during pushes.
Every active worktree must contain `scripts/check_push_targets.py` before
installing this shared wrapper; unprepared branches are not eligible.

Test launchers clear inherited Git repository-local environment before starting
their supervised commands. Temporary fixture repositories must not reuse a
calling hook's Git directory or index; this isolation does not disable hooks.

The functional-slow CI job is part of `ci-ok`, with run-attempt-specific test
evidence retained even on failure. The numerical-certification decision and
result are also enforced by `ci-ok`; this checked-in logic does not establish
hosted branch-protection activation. Do not relax `make check` or the full
local fast/slow hooks until hosted success, failure/cancellation blocking,
retained evidence, required-check configuration, and owner-approved activation
have been verified. Workflow preparation alone does not establish those conditions.

The two Python 3.12 fast CI shards also publish an optional combined executed-code
coverage artifact when both shards succeed for the same commit and run attempt.
It counts `increment/` source statements executed during the fast tier (including
pytest-cov's measured test workers); it does not measure the slow or scientific
tiers, establish whole-suite coverage, or rate test quality. Missing, failed or
mismatched shard data is reported as incomplete and is not published as an
aggregate. Collection runs inside the required shards to avoid a second fast-tier
run, so a coverage save or combine error after the tests pass fails that shard. If
that happens, remove the coverage arguments from the 3.12 shards rather than
relaxing the test result.

Commit `uv.lock` whenever a dependency change updates the resolution.

Notebook examples use Marimo; the development environment does not include
Jupyter Notebook, JupyterLab, or ipywidgets.

### Tests and quality checks

#### Test categories and runtime expectations

Unmarked categories default to cheap unit or small deterministic regression
checks. Most unit bodies should be sub-second; the existing local timeout is
a hang guard, not a target. Legacy unmarked tests are not automatically units:
inspect their assertions, fixtures and helpers before classifying them.
Scheduling/metadata marks such as `xdist_group`, `skip` and `compatibility`
do not define a test's purpose. Do not add redundant `unit` or `fast` marks.

| Purpose | Marks | Execution |
| --- | --- | --- |
| Unit/small regression | No category mark | Focused/affected feedback and ordinary fast CI |
| Cooperating components, local query engines, UI or subprocess boundaries | `integration`; add `slow` if expensive | Fast unless slow; relevant task checks and required CI |
| Expensive functional regression | `slow`, plus its purpose where useful | Relevant affected slow and current functional-slow/pre-push gates |
| Large-scale/exhaustive deterministic validation | `stress` and `slow` | Currently functional-slow; relevant changes and final integration/release |
| Monte Carlo coverage, bias or parameter recovery | `parameter_recovery` and `slow` | Monte Carlo-only target remains available; included in scientific union shards and relevant inference validation |
| Deterministic inferential certification | `numerical_certification` and `slow` | Dedicated relevant-change CI decision/job; also included in weekly scientific union shards |
| Runnable notebook exports | `examples` and `slow` | Notebook acceptance/documentation and weekly full-functional jobs |

`slow` describes cost, not purpose. A solver regression at ordinary inputs
does not become stress merely because it is slow: investigate production
performance or redundant test work. Category checks refuse recovery,
numerical certification, examples or stress tests lacking an effective `slow` mark before execution,
including CI; module, class and parameter inheritance count.

Integration does not change selection. Stress alone remains in the
functional-slow tier; an inferential capacity campaign also carries numerical
certification. Moving it outside routine CI requires an executable,
required gate for relevant changes and a scheduled/release route, with
failure evidence retained; a label or proposed workflow is insufficient.
Monte Carlo and deterministic numerical certification are excluded from fast
and functional-slow. The dedicated certification job requires relevant changes;
weekly scientific-union shards execute both scientific purposes. Full and
dependency-floor matrices retain their previously covered numerical cases:
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

CI publishes advisory runtime warnings and bounded job summaries after fast,
functional-slow and numerical-certification runs; the evidence artifact retains
the complete JSON report. To inspect a local receipt:

```bash
uv run python -m scripts.report_test_runtime .test-evidence/run-<id> \
  --base <task-base-sha> --json .test-evidence/runtime.json \
  --markdown .test-evidence/runtime.md
```

A parent evidence directory selects its latest receipt; use an exact run path
to avoid ambiguity. Optional `--baseline <run-path>` compares matching Python,
platform, worker count, distribution mode and thread limits (including Polars).
Timing increases are candidates, not demonstrated regressions. Selected
baseline receipts do not prove that an unobserved test is newly authored, and
AST family membership cannot establish parameter-level novelty. Partial,
malformed or legacy evidence remains explicitly uncertain. Warnings never
retag source or fail a run because a test exceeded five seconds; inspect shared
fixture/setup costs and repeat under representative conditions before retagging.

On the fast tier, Python 3.13 compares complete test nodeids with
`.github/fast-test-durations.json`; the shard figure is summed case-work seconds,
not wall time. Provisional regression thresholds are 1.5x and 2s per file,
1.25x and 30s per shard. The guard only warns and never fails a run;
`inconclusive` means the run or baseline coverage is insufficient for comparison.
Read the file and test rows; investigate shared setup and contention first, then
optimize genuine regressions. Mark a test `slow` only when its workload is
intentionally expensive and its required execution remains covered. For an
intentional increase, refresh the weights through the complete hosted procedure
below in the same PR and state the reason. The slow tier has no complete hosted
baseline, so it remains covered by the 5s/10s advisories.

Record each family's contract, current source anchor, expensive workload,
disposition (keep, optimize, split, reclassify or remove genuinely redundant
coverage), required execution gate and exact parameter identities. Group
shared causes without overlooking expensive boundary variants. Keep small
deterministic witnesses in the routine tier; preserve full campaign
assertions, seeds, repetitions and tolerances separately. A small smoke does
not establish empirical coverage. Update selectors, campaign runners,
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


- Start with the exact regression:

  ```bash
  make test TESTS=tests/file.py::test_name
  ```

  For a scoped follow-up, run the affected fast tier against the task's recorded
  base commit:

  ```bash
  make test-affected BASE=<task-base-sha>
  make test-affected BASE=<task-base-sha> TIER=slow
  ```

  `BASE` is required and resolved to an immutable commit. `HEAD` is valid for
  a working-tree-only comparison; long-running tasks MUST record their dispatch
  base SHA and reuse it across later commits and resumes so committed task work
  is not dropped.

  `PYTEST_ARGS` may contain only typed controls: `-k`/`--keyword`,
  `-m`/`--markexpr`, `-x`/`--maxfail`, `-q`, `-v`, `--durations`, numeric
  `-n 1..8`, `--dist loadgroup`, `--evidence-root`, and
  `--runtime-diagnostics`. Runs are serial by default. `-n auto`/`logical`,
  maximum-process caps, custom xdist transports, arbitrary pytest arguments,
  `-c`/`-o`, response files, non-empty inherited `PYTEST_ADDOPTS` or
  `PYTEST_PLUGINS`, and project pytest addopts/plugins are not accepted in
  affected mode. The launcher uses a fixed config, sanitized environment,
  explicit plugin set, and a parser-backed effective-option allowlist.

  The selector uses Testmon only with a complete, matching map for the requested
  tier and changed function bodies; otherwise it falls back to the conservative
  local AST import graph. The graph follows transitive source/test-helper
  imports, resolves public lazy exports, and retains all tests if any graph file
  is unparseable. Module/class-level or import-executed changes use the graph.
  Changed tests, dynamic-import helpers and their transitive test consumers,
  subprocess tests and filesystem guards remain selected. Changes to config,
  dependencies/plugins, conftest/shared fixtures, data, public lazy exports, or
  source that dynamically loads arbitrary paths refuse with an explicit broader route.

  Ordinary full-tier validation does not record or update Testmon maps.
  Record one explicitly with `make test-map TIER=fast` or `TIER=slow`; the
  underlying `--record-testmon-map` option goes before the tier. Recording
  requires a complete, unfiltered fast/slow/all run and acquires the database
  lock. Unsafe or partial requests refuse, and a successful pytest run without
  a usable published map is not a successful recording. Map generation remains
  expensive and uses the existing tier budgets; it is optional for ordinary
  validation and affected feedback, which can use the conservative graph.
  Completeness markers bind
  the database, exact Python/dependency environment, source inventory, collected
  node IDs, import-time coverage from a fresh interpreter, module/class AST
  fingerprints, resolved selection controls and terminal reports. Fast/slow
  certify only their own tier; all certifies both. A tier marker remains valid
  only while its node IDs still exist, including scheduler group suffixes.

  Affected startup pins and validates the worktree map before Testmon mutates
  it. Missing, unreadable, corrupt, stale or incomplete maps use the graph.
  Affected runs never certify a tier. On the Testmon route, changed test-helper
  consumers remain protected from deselection. Collection restores scheduling
  after plugin hooks without changing final node IDs.
  Import-graph fallback collects only the conservative retained files; it does
  not import unrelated tests just to deselect them. The certified Testmon route
  retains full collection. Collection failures in retained files still fail.


  Supervised pytest children carry an explicit owner-child role. Nested normal
  runners bypass Testmon reads/writes and certification; without a valid owner
  token they do not acquire its database lock. Reusing a live same-worktree lock
  requires a verified owner token.
  Selector collection and affected subprocesses load conftests only from their
  declared repository root and descendants, including retained temporary repos.
  Controlled pytest children and certification collection use a private empty
  bytecode namespace with writes disabled, so rapid same-size edits cannot
  reuse source-tree or pytest-rewritten bytecode. Shared caches are not deleted.
  This costs cold Python compilation, not a rebuild of native extensions.
  It does not refresh already-loaded parent modules, in-memory imports, explicit
  descendant overrides, or source edited during execution; keep inputs stable.


  A successful affected run requires finished owner-matched evidence whose exact
  terminal node IDs equal its nonempty selected node IDs. Evidence records
  `testmon` or `import_graph` as the selection route. Zero selection is rejected.
  Affected-fast is feedback, not a replacement for slow, parity, fixtures, public
  smokes, documentation, scientific or full-suite acceptance.

  Dispatch/resume notes record the immutable base, exact commands, serial/numeric
  worker allowance, remaining acceptance and evidence location. Evidence is
  source/worktree/environment/dirty-input specific; changed inputs require a
  new affected run, not reuse of an earlier green receipt.
  A read-only review does not invalidate matching execution evidence. Reuse
  receipts only while their recorded source, worktree, environment and dirty
  inputs still match; run broad gates at the required integration point, not
  simply because another reviewer looked at unchanged code.



Before pushing, run:

```bash
make check
```

This runs formatting and lint checks, architecture validation, dead-code and
docstring-baseline checks, type checking, fast tests, and functional-slow
tests. Use `make fmt` to apply formatting and safe lint fixes.

Additional commands:

- `make test-fast` runs the ordinary test tier.
- `make test-slow` runs functional slow tests, excluding Monte Carlo recovery,
  deterministic certification and notebook examples.
- `make test-versions` runs the supported-Python Nox sessions.
- `make ci-local` adds documentation, table-extra, build, and installed-wheel
  checks to the normal local gate.
- `make test-all` is reserved for final integration and release candidates
  that change estimation or a wire format. It is not a routine contributor
  check.

Automatic per-test and aggregate test-tier budgets apply locally only.
GitHub Actions (`GITHUB_ACTIONS=true`) skips those performance budgets;
workflow job timeouts still stop hangs. Explicit timeouts and scientific
campaign deadlines remain enforced, and runners still clean up subprocess
groups on interruption. No private authorization or task environment is
required. For Monte Carlo-only shards, use
`make test-parameter-recovery SPLITS=4 GROUP=1`; the certification-only
target is `make test-numerical-certification`. The weekly scientific union
uses `make test-scientific SPLITS=4 GROUP=1`; without local timing data,
pytest-split distributes tests evenly.

`make test-all` retains failure outputs and JUnit reports under
`.test-evidence/`. Add `--runtime-diagnostics` through `PYTEST_ARGS` when
investigating slow tests; retained artifacts are not committed.
Full Nox sessions retain separate receipts for the functional run and the
exact-mass checks. Explicit `tests_floor_full` arguments replace the default
selection and run once.

Ordinary tier commands disable Tach's unused impact-analysis plugin at the shared
runner boundary; direct `uv run pytest --tach ...` remains available for
impact-selected work.
Ordinary Make tiers accept standard `PYTEST_ARGS`, including capped auto workers
with `-n auto --maxprocesses=<allowance>`. Affected mode remains serial by default
and accepts only the typed controls listed above.

The test bootstrap defaults numerical-library thread pools to one per process,
matching Nox's controls, while preserving explicit environment overrides.
Subprocess-heavy runner, selection and evidence guard modules share one native
xdist loadgroup so their child launches do not multiply across outer workers.
Their nested parallel scenarios still execute; other functional tests remain
parallel. Notebook checks share one batched Marimo invocation while retaining a
separate guard for each notebook; export and warehouse-regeneration scenarios
still execute independently.

Unit tests do not use the network or real warehouses. Live PostgreSQL,
Snowflake, and BigQuery checks live under `integration/warehouse_execution/`
and require dedicated scratch resources and credentials. Contributors do not
need those services for ordinary changes.

The checked-in CI candidate requires lint, fast tests on Python 3.12–3.14,
functional slow tests, package builds, and live PostgreSQL 16 probes through
`ci-ok`; hosted enforcement and required-check activation remain owner gates.
PostgreSQL may skip a PR only when all changes are known prose or static
documentation assets. Library, test, fixture, dependency, CI, unknown,
deletion, and rename changes require it; an unavailable diff also requires it.
Main pushes never use this filter.

The PR PostgreSQL gate runs one duration-balanced shard. The reusable
warehouse workflow defaults to three shards for weekly and manual runs.

Each fast Python-version cell has two duration-balanced `pytest-split` shards,
with `xdist` workers inside each shard. `ci-ok` requires all six jobs. The
machine-generated `.github/fast-test-durations.json` weights come from the
complete hosted Python 3.13 run 36941053401; absent weights do not omit tests.
Refresh them with the complete fast suite, preserving worker group suffixes:

```bash
make test-fast PYTEST_ARGS="-n auto --dist loadgroup --store-durations --durations-path .github/fast-test-durations.json"
```

Functional-slow CI uses four isolated shards with workers inside each shard;
`ci-ok` requires their aggregate success. Every case remains selected exactly
once. Until a complete hosted timing baseline is available, the splitter uses
equal default weights. Each shard retains its own test evidence and
`slow-test-durations.json`; partial shard timings are not a complete baseline.

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
[weekly workflow](.github/workflows/weekly.yml) runs Sundays at 08:00 UTC or
on manual dispatch: full slow/examples suites at locked dependencies and
floors on all three Python versions, four scientific-union shards on Python
3.12, and all three warehouse backends. Fast, full and scientific jobs retain
test evidence, including per-test timings, for 30 days. The full-suite jobs
allow 90 minutes: prior completed jobs needed up to 55 minutes of pytest time,
with roughly one minute for session startup and dependency setup, so the limit
leaves about 34 minutes of margin without adding shard/configuration overhead.
Live warehouse timings stay in the job logs, where GitHub masks secrets; raw
cloud tracebacks are not uploaded as artifacts. Weekly failures create or
update a GitHub issue labelled `weekly-failure`.

Tests run in parallel and must not depend on execution order or state created
by another test. Mark functional slow tests with `slow`; simulation-based
coverage and parameter-recovery checks also use `parameter_recovery`, while
deterministic inferential certification uses `numerical_certification`. See
[AGENTS.md](AGENTS.md) for the repository's architecture, parity, numerical,
and test-contract requirements.

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
their generator scripts; ordinary tests read the fixtures only, never R or the network. Run an
oracle with `make test TESTS=tests/oracles/test_welch_oracle.py` (the focused runner takes test
files, not directories; [Reproduce](docs/validation.md#reproduce) lists each one). To report a
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

Write concrete, current explanations. Lead guides with the task and required
inputs. Keep assumptions, statistical guarantees, limitations, and citations
explicit. Do not include internal work-item labels, agent instructions, or
competitor comparisons in user documentation.

Comments should explain an invariant or a non-obvious choice and normally stay
brief. Preserve derivations and assumptions when their detail is necessary.
Run:

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

Compatibility probes establish composition behavior. Statistical calibration
belongs in the capability's dedicated tests.

### Adding a method or design

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

Repository architecture and engineering requirements live in
[AGENTS.md](AGENTS.md).

Maintainers and automated contributors use
[kata](https://github.com/kenn-io/kata) for internal work tracking. External
contributors may use GitHub issues and pull requests and do not need Kata.
Maintainers using Kata should run `kata quickstart`, search before creating an
issue, and prefer adding evidence to an existing issue over opening a
duplicate.

Roborev is optional. The checked-in `.roborev.toml` contains only
repository-specific review criteria; agent, model, reasoning, timeout, and
credential settings come from each contributor's global configuration.

### Review and handoffs

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
