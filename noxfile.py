"""Test, lint, and typecheck sessions."""

from pathlib import Path
from tempfile import TemporaryDirectory

import nox

nox.options.default_venv_backend = "uv"
nox.options.reuse_existing_virtualenvs = True

PYTHON_VERSIONS = ["3.12", "3.13", "3.14"]

# Direct dependency floors declared in pyproject.toml. Keyed companions keep
# each supported interpreter on the oldest usable combination: older NumPy,
# SciPy, Pandas, and DuckDB releases do not publish wheels for every Python.
COMMON_FLOORS = (
    "pydantic==2.12.3",
    "packaging==24.0",
    "sqlglot==27.24.2",
    "narwhals==2.24.0",
)
IBIS_FLOORS: dict[str, str] = {
    "3.12": "ibis-framework==10.1.0",
    "3.13": "ibis-framework==10.1.0",
    # Ibis 10.1 imports duckdb.functional, but no DuckDB release that both
    # exposes that module and ships CPython 3.14 Linux wheels exists.
    "3.14": "ibis-framework==11.0.0",
}
FLOOR_PINS: dict[str, tuple[str, ...]] = {
    # Floors need both interpreter wheels and compatible dataframe libraries.
    # Arrow 23.0.1 is the security floor and supports the NumPy 2 ABI; 3.12
    # retains NumPy 1.26 as its oldest supported floor independently of the published range.
    "3.12": (
        *COMMON_FLOORS,
        "pyyaml==6.0.1",
        "numpy==1.26.*",
        "scipy==1.14.*",
        "pyarrow==23.0.1",
        "pandas==2.2.*",
    ),
    "3.13": (
        *COMMON_FLOORS,
        "pyyaml==6.0.2",
        "numpy==2.1.*",
        "scipy==1.14.*",
        "pyarrow==23.0.1",
        "pandas==2.2.*",
    ),
    "3.14": (
        *COMMON_FLOORS,
        "pyyaml==6.0.3",
        "numpy==2.3.*",
        "scipy==1.16.*",
        "pyarrow==23.0.1",
        "pandas==2.3.3",
    ),
}
TABLE_FLOORS: dict[str, tuple[str, ...]] = {
    "3.12": ("coeftable==0.13.1",),
    "3.13": ("coeftable==0.13.1",),
    "3.14": ("coeftable==0.13.1",),
}
DASHBOARD_FLOORS: dict[str, tuple[str, ...]] = {
    "3.12": ("marimo==0.25.0",),
    "3.13": ("marimo==0.25.0",),
    "3.14": ("marimo==0.25.0",),
}
DUCKDB_FLOORS: dict[str, str] = {
    "3.12": "duckdb==1.3.2",
    "3.13": "duckdb==1.3.2",
    "3.14": "duckdb==1.4.3",
}

# `--dist loadgroup` keeps an `xdist_group` module on one worker: a few
# suites mutate shared paths on disk and would race otherwise. Sessions
# whose whole selection is one such group stay serial -- splitting them
# buys nothing and costs worker startup.
PYTEST_ARGS = ("-n", "auto", "--dist", "loadgroup", "-p", "no:tach")
TEST_RUNNER = ("python", "-m", "scripts.run_test_tier")


def _sync(session: nox.Session, *groups: str, extras: tuple[str, ...] = ()) -> None:
    """Sync the project's uv-managed dependency groups (and extras) into the session's venv."""
    args = ["uv", "sync", "--locked"]
    for group in groups:
        args += ["--group", group]
    for extra in extras:
        args += ["--extra", extra]
    session.run_install(*args, env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location})


def _run_test_tier(session: nox.Session, tier: str, *pytest_args: str) -> None:
    session.run(*TEST_RUNNER, tier, *pytest_args)


def _run_pytest_at_floor(
    session: nox.Session,
    tier: str,
    *pytest_args: str,
    extras: tuple[str, ...] = ("demo",),
) -> None:
    """Sync the given extras, then run pytest against direct dependency floors.

    ``uv run --with`` overlays the floor pins on top of the synced environment
    for this invocation only, without touching the lockfile. Explicit session
    arguments replace the default pytest selection.
    """
    python = session.python
    if not isinstance(python, str) or python not in PYTHON_VERSIONS:
        session.error("Floor tests require one supported Python version.")
    _sync(session, "dev", extras=extras)
    pins = [IBIS_FLOORS[python], *FLOOR_PINS[python], DUCKDB_FLOORS[python]]
    if "tables" in extras or "dashboard" in extras:
        pins.extend(TABLE_FLOORS[python])
    if "dashboard" in extras:
        pins.extend(DASHBOARD_FLOORS[python])
    with_args = [arg for pin in pins for arg in ("--with", pin)]
    if session.posargs and Path(session.posargs[0].split("::", 1)[0]).is_file():
        tier = "focused"
    session.run(
        "uv",
        "run",
        "--no-sync",
        *with_args,
        *TEST_RUNNER,
        tier,
        *(session.posargs or pytest_args),
        # xdist already parallelizes processes; BLAS and polars must not multiply their threads.
        env={
            "UV_PROJECT_ENVIRONMENT": session.virtualenv.location,
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "POLARS_MAX_THREADS": "1",
        },
    )


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the fast test suite against every supported Python version.

    Syncs the ``demo`` extra: most of the fast suite (query builders,
    ``Analysis``, the calibration fixtures) exercises a live
    ``ibis.duckdb.connect()`` backend directly, with no
    ``importorskip`` guard, so duckdb has to actually be installed.
    """
    _sync(session, "dev", extras=("demo", "dashboard"))
    _run_test_tier(session, "fast", "-q", *PYTEST_ARGS)


@nox.session(python=PYTHON_VERSIONS)
def tests_full(session: nox.Session) -> None:
    """Weekly: slow + examples against locked dependencies, across every
    supported Python version. Mirrors ``make test-all`` (demo, tables and
    dashboard extras, marker filter cleared) minus the Monte-Carlo tier --
    ``tests_parameter_recovery`` runs that once, sharded, instead of once per
    matrix cell. Pins the interpreter per matrix cell -- ``make test-all``
    alone resolves whatever ``.python-version``/the caller's default is, so
    the weekly workflow's 3.13/3.14 matrix legs would otherwise silently
    re-run 3.12 three times instead of actually exercising each version.
    """
    _sync(session, "dev", extras=("demo", "tables", "dashboard"))
    _run_test_tier(
        session,
        "all-except-parameter-recovery",
        "-q",
        *PYTEST_ARGS,
        "--evidence-root",
        ".test-evidence",
    )


@nox.session(python="3.12")
def tests_warehouse_postgres(session: nox.Session) -> None:
    """Execute real PostgreSQL probes; missing connection settings fail."""
    _sync(session, "dev", "warehouse-backends", extras=("demo",))
    session.run(
        "pytest",
        "integration/warehouse_execution/test_postgres_execution.py",
        "-q",
        "-m",
        "warehouse_postgres",
        "-p",
        "no:tach",
        "--durations=0",
        *session.posargs,
    )


@nox.session(python="3.12")
def tests_warehouse_snowflake(session: nox.Session) -> None:
    """Execute real Snowflake probes; missing credentials fail."""
    _sync(session, "dev", "warehouse-backends", extras=("demo",))
    session.run(
        "pytest",
        "tests/query/test_session.py::test_hosted_metadata_creation_targets_the_explicit_catalog",
        "-q",
        "-k",
        "snowflake",
        "-m",
        "",
        "-p",
        "no:tach",
    )
    session.run(
        "pytest",
        "integration/warehouse_execution/test_snowflake_execution.py",
        "-q",
        "-m",
        "warehouse_snowflake",
        "-p",
        "no:tach",
        "--durations=0",
        *session.posargs,
    )


@nox.session(python="3.12")
def tests_warehouse_bigquery(session: nox.Session) -> None:
    """Execute real BigQuery probes; missing credentials fail."""
    _sync(session, "dev", "warehouse-backends", extras=("demo",))
    session.run(
        "pytest",
        "tests/query/test_session.py::test_hosted_metadata_creation_targets_the_explicit_catalog",
        "-q",
        "-k",
        "bigquery",
        "-m",
        "",
        "-p",
        "no:tach",
    )
    session.run(
        "pytest",
        "integration/warehouse_execution/test_bigquery_execution.py",
        "-vv",
        "-m",
        "warehouse_bigquery",
        "-p",
        "no:tach",
        "--durations=0",
        *session.posargs,
    )


@nox.session(python=PYTHON_VERSIONS)
def tests_floor(session: nox.Session) -> None:
    """Run the fast suite at dependency floors on every supported interpreter."""
    _run_pytest_at_floor(session, "fast", "-q", *PYTEST_ARGS, extras=("demo", "dashboard"))


@nox.session(python=PYTHON_VERSIONS)
def tests_slow_floor(session: nox.Session) -> None:
    """Run functional-slow tests at dependency floors on every supported interpreter."""
    _run_pytest_at_floor(
        session,
        "slow",
        "-q",
        *PYTEST_ARGS,
        extras=("demo", "tables", "dashboard"),
    )


@nox.session(python=PYTHON_VERSIONS)
def tests_floor_full(session: nox.Session) -> None:
    """Weekly: slow + examples against direct dependency floors, across
    every supported Python version, minus the Monte-Carlo tier (see
    ``tests_parameter_recovery``).

    ``tests_floor`` above is the narrower fast-only floor session.
    This full floor session also exercises coverage checks against core,
    table and dashboard dependency floors.
    """
    _run_pytest_at_floor(
        session,
        "all-except-parameter-recovery",
        "-q",
        *PYTEST_ARGS,
        "--evidence-root",
        ".test-evidence",
        extras=("demo", "tables", "dashboard"),
    )


@nox.session(python="3.12")
def tests_parameter_recovery(session: nox.Session) -> None:
    """Run one optional shard of the Monte-Carlo tier on Python 3.12.

    Pass ``--splits N --group k`` via posargs, for example
    ``nox -s tests_parameter_recovery -- --splits 4 --group 1``. Without a
    machine-specific durations file, pytest-split distributes tests evenly.
    """
    _sync(session, "dev", extras=("demo", "tables", "dashboard"))
    _run_test_tier(
        session,
        "parameter-recovery",
        "-q",
        *PYTEST_ARGS,
        "--evidence-root",
        ".test-evidence",
        *session.posargs,
    )


@nox.session(python="3.12")
def examples(session: nox.Session) -> None:
    """Run example lint, formatting, and notebook acceptance.

    Syncs ``demo``, ``tables`` and ``dashboard`` so optional notebook
    dependencies are exercised rather than silently skipped.
    Single Python version -- these exercise the notebooks' behavior, not
    interpreter compatibility, so the 3-way matrix the ``tests`` session
    runs would just triple the cost for no extra coverage.
    Documentation snippets run separately in ``docs_snippets`` so this
    complete examples tier has one five-minute local aggregate budget.
    """
    _sync(session, "dev", extras=("demo", "tables", "dashboard"))
    session.run("ruff", "check", "examples")
    session.run("ruff", "format", "--check", "examples")
    _run_test_tier(session, "examples", "-q", *PYTEST_ARGS)


@nox.session(python="3.12")
def docs_snippets(session: nox.Session) -> None:
    """Execute shipped README/docs Python snippets under one local fast-tier budget."""
    _sync(session, "dev", extras=("demo", "tables", "dashboard"))
    _run_test_tier(
        session,
        "fast",
        "README.md",
        "docs",
        "-q",
    )


@nox.session(python="3.12")
def tests_tables(session: nox.Session) -> None:
    """Run `tests/test_tables.py` with the `tables` extra installed.

    Every readout_table test in that file is individually fast
    (hand-built frames, no warehouse -- well inside AGENTS.md's <100ms
    budget) and belongs in the fast suite by that budget alone, but it
    is guarded by ``pytest.importorskip("coeftable")`` because the plain
    ``tests`` session above only syncs the ``demo`` extra -- most of
    the fast suite exercises ibis/duckdb directly and never needs
    coeftable, so bundling ``tables`` into that session would pay its sync
    cost on every one of the fast suite's other tests for zero benefit
    to them. Without a session that installs ``tables``, every guarded
    test here -- including the `trend=` sparkline's ValueError guards and
    blank-cell/NaN degradation -- silently skips in every CI job: the
    ``examples`` and ``docs_snippets`` sessions install ``tables``, but
    collect only examples-marked tests and documentation snippets
    respectively; this file is in neither.
    """
    _sync(session, "dev", extras=("demo", "tables"))
    _run_test_tier(session, "fast", "tests/test_tables.py", "-q")


@nox.session(python="3.12")
def tests_slow(session: nox.Session) -> None:
    """Run the functional slow tier: `slow` minus Monte-Carlo and notebooks.

    These are the cross-path tests -- warehouse/frame/moments parity, export
    round trips, breakout dispatch -- that are too slow for the fast suite
    but prove nothing statistical, so they do not belong in the nightly
    Monte-Carlo tier either. Cheap enough to run on every push, and the
    only tier that sees a disagreement between two code paths over the
    same data. Without a job here they run once a night, where a red
    result notifies nobody.
    """
    _sync(session, "dev", extras=("demo", "tables", "dashboard"))
    _run_test_tier(session, "slow", "-q", *PYTEST_ARGS)


@nox.session(python="3.12")
def lint(session: nox.Session) -> None:
    """Check formatting and lint rules."""
    _sync(session, "dev")
    session.run("ruff", "check", ".")
    session.run(
        "ruff",
        "check",
        "--select",
        "PLR0915",
        "--ignore-noqa",
        "--config",
        "lint.pylint.max-statements=100",
        "increment/",
    )
    session.run("ruff", "format", "--check", ".")
    session.run("tach", "check")
    session.run("vulture", "increment", "vulture_whitelist.py", "--min-confidence", "80")
    session.run("python", "-m", "scripts.check_docstring_baseline")
    session.run("python", "scripts/audit_verbosity.py", "--gate")


EXAMPLE_NOTEBOOKS = (
    "analysis_from_a_dataframe",
    "analysis_from_a_warehouse",
    "ab_testing_dashboard",
    "observational",
    "cuped",
    "breakout",
    "power",
    "hte",
    "encouragement",
    "late_over_time",
)


@nox.session(python="3.12")
def docs(session: nox.Session) -> None:
    """Build the documentation site; --strict fails on any warning.

    Catches unresolved pages, broken cross-references, and unknown
    ``:::`` identifiers. It does NOT catch renamed-parameter drift in
    numpy docstrings: ``warn_unknown_params`` is disabled in mkdocs.yml
    because pydantic models document field "Parameters" that never
    appear in a function signature.

    Exports every example notebook to static HTML under ``docs/examples/``
    first (the wrapper pages iframe those files, so a strict build needs
    them present). The exports execute the notebooks, hence the same
    ``dev`` + ``demo`` + ``tables`` + ``dashboard`` dependencies the
    ``examples`` session uses.

    Re-runs the compatibility catalog's own probe/evidence tests and
    checks ``docs/guides/compatibility.md`` is the current output of
    ``scripts/render_compatibility.py`` before the strict build, so a
    stale committed page (or a catalog change nobody regenerated from)
    fails ``nox -s docs`` instead of silently shipping.
    """
    _sync(session, "docs", "dev", extras=("demo", "tables", "dashboard"))
    for name in EXAMPLE_NOTEBOOKS:
        session.run(
            "marimo",
            "export",
            "html",
            f"examples/{name}.py",
            "-o",
            f"docs/examples/{name}.html",
            "-f",
        )
    _run_test_tier(
        session,
        "fast",
        "tests/test_composition_matrix.py",
        "tests/test_compatibility_catalog.py",
        "-q",
    )
    session.run("python", "scripts/render_compatibility.py", "--check")
    session.run("mkdocs", "build", "--strict")


@nox.session(python="3.12")
def typecheck(session: nox.Session) -> None:
    """Run the static type checker.

    Install ``dev``, ``demo``, ``tables`` and ``dashboard`` so ty checks
    optional integration call sites rather than skipping their imports.
    """
    _sync(session, "dev", extras=("demo", "tables", "dashboard"))
    session.run("ty", "check")


@nox.session(python="3.12", reuse_venv=False)
def wheel_smoke(session: nox.Session) -> None:
    """Install the built wheel with ONLY core dependencies (no demo/tables
    extras, no editable source install) and run scripts/wheel_smoke.py
    from a directory outside the checkout, proving increment.__all__ --
    including power, which needs no extra at all -- works from the
    actual installed wheel, not the source tree."""
    script = Path("scripts/wheel_smoke.py").resolve()
    session.install(str(_single_wheel(session)))
    with TemporaryDirectory(prefix="increment-wheel-smoke-") as tmp, session.chdir(tmp):
        session.run("python", str(script))


@nox.session(python="3.12", reuse_venv=False)
def wheel_smoke_demo(session: nox.Session) -> None:
    """Install the built wheel with the demo extra and run the DuckDB memtable
    probe, which fails when a dependency breaks Ibis create_table from PyArrow."""
    script = Path("scripts/wheel_smoke.py").resolve()
    session.install(f"{_single_wheel(session)}[demo]")
    with TemporaryDirectory(prefix="increment-wheel-smoke-demo-") as tmp, session.chdir(tmp):
        session.run("python", str(script), "--demo-extras")


def _single_wheel(session: nox.Session) -> Path:
    wheels = sorted(Path("dist").glob("increment-*.whl"))
    if not wheels:
        session.error("no wheel in dist/ -- run `uv build` (or the `build` CI job) first")
    if len(wheels) > 1:
        session.error(
            "ambiguous wheel selection in dist/ -- expected exactly one "
            f"increment-*.whl, found {[str(w) for w in wheels]}; clear dist/ "
            "and rebuild so a stale artifact from a prior version is never "
            "silently picked"
        )
    return wheels[0]
