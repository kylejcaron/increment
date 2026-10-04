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

Commit `uv.lock` whenever a dependency change updates the resolution.

Notebook examples use Marimo; the development environment does not include
Jupyter Notebook, JupyterLab, or ipywidgets.

### Tests and quality checks

Start with the smallest test selection that exercises your change:

```bash
make test TESTS=tests/file.py::test_name
```

Before pushing, run:

```bash
make check
```

This runs formatting and lint checks, architecture validation, dead-code and
docstring-baseline checks, type checking, fast tests, and functional-slow
tests. Use `make fmt` to apply formatting and safe lint fixes.

Additional commands:

- `make test-fast` runs the ordinary test tier.
- `make test-slow` runs functional slow tests, excluding statistical
  parameter-recovery campaigns and notebook examples.
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
required. For statistical shards, use
`make test-parameter-recovery SPLITS=4 GROUP=1`; without local timing data,
pytest-split distributes tests evenly.

`make test-all` retains failure outputs and JUnit reports under
`.test-evidence/`. Add `--runtime-diagnostics` through `PYTEST_ARGS` when
investigating slow tests; retained artifacts are not committed.

Ordinary Make/Nox suites disable Tach's unused impact-analysis plugin; direct
`uv run pytest --tach ...` remains available for impact-selected work. Override
Make's `PYTEST_ARGS` when you need different pytest options. Notebook checks share one
batched Marimo invocation while retaining a separate guard for each notebook;
export and warehouse-regeneration scenarios still execute independently.

Unit tests do not use the network or real warehouses. Live PostgreSQL,
Snowflake, and BigQuery checks live under `integration/warehouse_execution/`
and require dedicated scratch resources and credentials. Contributors do not
need those services for ordinary changes.

CI requires lint, fast tests on Python 3.12–3.14, package builds, and live
PostgreSQL 16 probes through `ci-ok`. PostgreSQL may skip a PR only when all
changes are known prose or static documentation assets. Library, test,
fixture, dependency, CI, unknown, deletion, and rename changes require it;
an unavailable diff also requires it. Main pushes never use this filter.

Each fast Python-version cell has two duration-balanced `pytest-split` shards,
with `xdist` workers inside each shard. `ci-ok` requires all six jobs. The
machine-generated `.github/fast-test-durations.json` weights come from the
complete hosted Python 3.13 run 36941053401; absent weights do not omit tests.
Refresh them with the complete fast suite, preserving worker group suffixes:

```bash
make test-fast PYTEST_ARGS="-n auto --dist loadgroup -p no:tach --store-durations --durations-path .github/fast-test-durations.json"
```

The complete PostgreSQL suite runs across three isolated PostgreSQL services,
balanced with `pytest-split` and `.github/postgres-test-durations.json`.
The weights come from hosted run 36941053401; new cases still run using the
average weight. To refresh weights, run the complete suite with
`--store-durations --durations-path .github/postgres-test-durations.json`.
Do not run workers against a shared warehouse namespace.

The
[weekly workflow](.github/workflows/weekly.yml) runs Sundays at 08:00 UTC or
on manual dispatch: full slow/examples suites at locked dependencies and
floors on all three Python versions, four Monte-Carlo shards on Python 3.12,
and all three warehouse backends. Fast, full and Monte-Carlo jobs retain test
evidence, including per-test timings, for 30 days. Live warehouse timings stay
in the job logs, where GitHub masks secrets; raw cloud tracebacks are not
uploaded as artifacts. Weekly failures create or update a GitHub issue
labelled `weekly-failure`.

Tests run in parallel and must not depend on execution order or state created
by another test. Mark functional slow tests with `slow`; simulation-based
coverage, bias, and parameter-recovery checks also use
`parameter_recovery` and retain a small fast smoke case. See
[AGENTS.md](AGENTS.md) for the repository's architecture, parity, numerical,
and test-contract requirements.

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
their generator scripts; ordinary tests read the fixtures only (`make test TESTS=tests/oracles`),
never R or the network. To report a discrepancy against a reference, open a GitHub issue with the
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

The default gate requires successful GitHub Actions `ci-ok` on that exact
commit, no other unfinished or failed checks, and successful legacy commit
statuses when present. It excludes only the running bump workflow itself.
`force=true` explicitly bypasses this CI gate; do not use it to release an
unverified candidate.

Bump creates the version tag and explicitly dispatches `release.yml` on it.
The job's `GITHUB_TOKEN` cannot trigger a second workflow through a tag push.
If the dispatch fails after the tag was pushed, dispatch the release on the
existing tag rather than creating another one:

```bash
gh workflow run release.yml --ref v0.1.0a3
```

Release accepts only `v*` tag refs, verifies the version and package artifacts,
then waits for the protected `pypi` environment approval. Inspect the artifacts
before approving: approval uploads to real PyPI, not a test registry.

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

### Capability composition

Every public capability declares its behavior for every metric type in
`tests/compatibility_catalog.py`, with enforcement in
`tests/test_composition_matrix.py`. When adding a capability or metric type:

1. Add every required matrix cell.
2. Classify it as supported, refused, structurally inapplicable, or a known
   gap.
3. Back the classification with an executable probe.
4. Update the path-by-capability documentation in `docs/limitations.md`.
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
