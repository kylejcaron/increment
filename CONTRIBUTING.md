# Contributing to increment

Thank you for your interest in contributing.

## Submitting changes

If you do not have write access, fork the repository, create a branch in your
fork, and open a pull request into `main`.

A pull request can merge only when CI (`ci-ok`) and the CLA check pass and a
code owner has approved it. Commits are squash-merged.

Release notes are generated from merged PR titles. Label your PR so it lands in
the right section: `breaking`, `feature`, `fix`, `stats`/`inference`, `docs`,
or `internal`. Use `skip-changelog` to omit it.

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

Test tiers enforce wall-clock limits and stop their subprocess groups on
timeout. No private authorization or task environment is required. For
statistical shards, use `make test-parameter-recovery SPLITS=4 GROUP=1`;
without local timing data, pytest-split distributes tests evenly.

`make test-all` retains failure outputs and JUnit reports under
`.test-evidence/`. Add `--runtime-diagnostics` through `PYTEST_ARGS` when
investigating slow tests; retained artifacts are not committed.

Unit tests do not use the network or real warehouses. Live PostgreSQL,
Snowflake, and BigQuery checks live under `integration/warehouse_execution/`
and require dedicated scratch resources and credentials. Contributors do not
need those services for ordinary changes.

Tests run in parallel and must not depend on execution order or state created
by another test. Mark functional slow tests with `slow`; simulation-based
coverage, bias, and parameter-recovery checks also use
`parameter_recovery` and retain a small fast smoke case. See
[AGENTS.md](AGENTS.md) for the repository's architecture, parity, numerical,
and test-contract requirements.

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
